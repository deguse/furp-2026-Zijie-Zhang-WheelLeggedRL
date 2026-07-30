import json
from pathlib import Path
import unittest

import numpy as np
import torch

from hoppertrex_mjlab.hybrid.innovation_detector import parse_innovation_predictor

from hoppertrex_mjlab.scripts import probe_hybrid_c2_transition_floor as probe


class ProbeHybridC2TransitionFloorTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    predictor_path = (
      Path(__file__).resolve().parents[1]
      / "docs/experiments/artifacts/c2_innovation_predictor_2cccb36_seed1/c2_innovation_predictor.json"
    )
    cls.predictor = parse_innovation_predictor(
      json.loads(predictor_path.read_text(encoding="utf-8"))
    )

  def test_cells_and_schedule_are_frozen(self):
    cells = probe.transition_cells()
    self.assertEqual(len(cells), 10)
    self.assertEqual(cells[0]["name"], "pitch_zero")
    self.assertEqual(cells[1]["name"], "fast_lean_0p032")
    corner = cells[-1]
    self.assertEqual(probe.raw_command(corner, 79), tuple(corner["target"]))
    self.assertEqual(probe.raw_command(corner, 80), tuple(corner["target"]))
    self.assertEqual(probe.raw_command(corner, 419), tuple(corner["target"]))
    self.assertEqual(probe.raw_command(corner, 499), probe.CENTER)

  def test_official_protocol_is_seed2_non_evidence(self):
    protocol = probe.protocol(False, "cuda:0")
    self.assertEqual(protocol["seed"], 2)
    self.assertEqual(protocol["envs_per_cell"], 16)
    self.assertEqual(protocol["settle_steps"], 200)
    self.assertEqual(protocol["drive_steps"], 500)
    self.assertEqual(protocol["threshold_factors"], [1.05, 1.25, 1.5, 2.0, 3.0])
    self.assertFalse(protocol["evidence_eligible"])

  def test_registered_ramp_is_reachable_under_deployed_slew(self):
    for cell in probe.transition_cells()[2:]:
      previous = probe.CENTER
      for tick in range(500):
        current = probe.raw_command(cell, tick)
        self.assertLessEqual(
          abs(current[0] - previous[0]),
          probe.POSTURE_HEIGHT_SLEW_RATE * probe.CONTROL_DT_S + 1e-15,
        )
        self.assertLessEqual(
          abs(current[1] - previous[1]),
          probe.POSTURE_PITCH_SLEW_RATE * probe.CONTROL_DT_S + 1e-15,
        )
        previous = current

  def test_restore_prevents_post_step_resample_without_second_slew(self):
    class Twist:
      def __init__(self):
        self.vel_command_b = torch.full((2, 3), 9.0)
        self.vel_command_w = torch.full((2, 3), 8.0)

    class Posture:
      def __init__(self):
        self._command = torch.full((2, 2), 7.0)
        self._target = torch.full((2, 2), 6.0)

    twist = Twist()
    posture = Posture()

    class Manager:
      def get_term(self, name):
        return twist if name == "twist" else posture

    class Env:
      device = "cpu"
      command_manager = Manager()

    shaped = np.asarray([[0.31, -0.01], [0.32, 0.02]], dtype=np.float64)
    probe._restore_deployed_commands(Env(), vx=0.07, shaped_posture=shaped)
    np.testing.assert_allclose(twist.vel_command_b[:, 0], 0.07)
    np.testing.assert_allclose(twist.vel_command_w[:, 0], 0.07)
    np.testing.assert_allclose(twist.vel_command_b[:, 1:], 0.0)
    np.testing.assert_allclose(posture._command, shaped)
    np.testing.assert_allclose(posture._target, shaped)

  def test_runtime_stack_assertion_rejects_binding_or_slew_drift(self):
    class Value:
      pass

    action = Value()
    action.cfg = Value()
    action.cfg.controller_gain_hash = probe.SCHEDULE_HASH
    action.cfg.calibration_hash = probe.CALIBRATION_HASH
    action.cfg.posture_artifact_hash = probe.POSTURE_HASH
    action.cfg.station_calibration_hash = probe.STATION_HASH
    action.cfg.yaw_calibration_hash = None
    action.cfg.wheel_slew_limit = probe.WHEEL_SLEW_RADPS_PER_TICK
    action.cfg.controller_schedule = Value()
    action.cfg.controller_schedule.bindings = {
      "identification_controller_gain_hash": probe.IDENTIFICATION_GAIN_HASH,
      "identification_calibration_hash": probe.CALIBRATION_HASH,
      "posture_artifact_hash": probe.POSTURE_HASH,
    }
    posture = Value()
    posture.cfg = Value()
    posture.cfg.height_slew_rate = probe.POSTURE_HEIGHT_SLEW_RATE
    posture.cfg.pitch_slew_rate = probe.POSTURE_PITCH_SLEW_RATE

    class Manager:
      def __init__(self, value):
        self.value = value

      def get_term(self, _name):
        return self.value

    class Env:
      action_manager = Manager(action)
      command_manager = Manager(posture)

    probe._assert_runtime_stack(Env())
    action.cfg.wheel_slew_limit = 5.0
    with self.assertRaisesRegex(RuntimeError, "slew"):
      probe._assert_runtime_stack(Env())
    action.cfg.wheel_slew_limit = probe.WHEEL_SLEW_RADPS_PER_TICK
    action.cfg.calibration_hash = "wrong"
    with self.assertRaisesRegex(RuntimeError, "bindings"):
      probe._assert_runtime_stack(Env())

  def test_domain_counter_distinguishes_nonfinite_from_finite_u_violation(self):
    state = np.asarray([0.0, 0.0])
    center = probe.CENTER
    self.assertFalse(
      probe._is_finite_domain_violation(
        self.predictor, state, 0.0, float("nan"), center[1]
      )
    )
    self.assertFalse(
      probe._is_finite_domain_violation(
        self.predictor, state, float("nan"), center[0], center[1]
      )
    )
    self.assertTrue(
      probe._is_finite_domain_violation(
        self.predictor, state, 100.0, center[0], center[1]
      )
    )


if __name__ == "__main__":
  unittest.main()
