import math
import unittest

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from hoppertrex_mjlab.scripts.probe_hybrid_posture_transition import (
  _directional_overshoot,
  kick_postures,
  parse_args,
  qualification_payload,
  run_kick_cell,
  run_transition,
  transition_legs,
)


class TransitionProtocolTest(unittest.TestCase):
  def test_legs_cover_corners_and_single_axes(self):
    legs = transition_legs((0.30, 0.33), (-0.08, 0.08))

    self.assertEqual(len(legs), 12)
    center = (0.315, 0.0)
    corner_targets = [target for start, target in legs if start == center]
    self.assertEqual(len(corner_targets), 4)
    self.assertIn((0.30, -0.08), corner_targets)
    self.assertIn((0.33, 0.08), corner_targets)
    # Return legs come back to center from every corner.
    self.assertEqual(
      sum(1 for _start, target in legs if target == center), 4
    )
    # Height-only and pitch-only legs hold the other axis at its center.
    self.assertIn(((0.30, 0.0), (0.33, 0.0)), legs)
    self.assertIn(((0.315, 0.08), (0.315, -0.08)), legs)

  def test_degenerate_envelope_produces_null_legs(self):
    legs = transition_legs((0.42, 0.42), (0.0, 0.0))
    self.assertEqual(len(legs), 12)
    for start, target in legs:
      self.assertEqual(start, target)

  def test_kick_postures_are_center_plus_corners(self):
    postures = kick_postures((0.30, 0.33), (-0.08, 0.08))
    self.assertEqual(len(postures), 5)
    self.assertEqual(postures[0], (0.315, 0.0))
    self.assertIn((0.33, -0.08), postures)

  def test_directional_overshoot_follows_travel_direction(self):
    values = torch.tensor([[0.30], [0.335], [0.332]])
    self.assertAlmostEqual(
      _directional_overshoot(values, 0.30, 0.33), 0.005, places=6
    )
    # Traveling down: only excursion BELOW the target counts.
    downward = torch.tensor([[0.340], [0.331], [0.333]])
    self.assertAlmostEqual(
      _directional_overshoot(downward, 0.34, 0.332), 0.001, places=6
    )
    # No travel on this axis -> no overshoot by definition.
    self.assertEqual(_directional_overshoot(values, 0.33, 0.33), 0.0)


class QualificationPayloadTest(unittest.TestCase):
  def test_payload_summary_and_bindings(self):
    transition = {
      "start_height": 0.315,
      "start_pitch": 0.0,
      "target_height": 0.30,
      "target_pitch": -0.08,
      "settling_time_s": 0.42,
      "height_overshoot": 0.003,
      "pitch_overshoot": 0.01,
      "pitch_rate_abs_max": 0.6,
      "pitch_rate_abs_p99": 0.4,
      "lin_x_abs_max": 0.09,
      "lin_x_abs_mean": 0.02,
      "non_wheel_contact_rate": 0.0,
      "terminated_events": 0.0,
    }
    kick = {
      "target_height": 0.315,
      "target_pitch": 0.0,
      "recovery_time_s": 1.3,
      "kick_event_count": 64.0,
      "post_kick_lin_x_abs_max": 0.08,
      "non_wheel_contact_rate": 0.0,
      "terminated_events": 0.0,
    }
    payload = qualification_payload(
      transitions=[transition],
      kick_cells=[kick],
      controller_gain_hash="gain" * 16,
      controller_qualified=True,
      posture_map_hash="map" * 16,
      posture_map_qualified=True,
      station_calibration_hash="station-hash",
      station_calibration_qualified=True,
      source_probe={"git_sha": "test"},
    )

    self.assertEqual(payload["kind"], "posture_transition_qualification")
    self.assertTrue(payload["station_calibration_qualified"])
    summary = payload["summary"]
    self.assertEqual(summary["transition_count"], 1)
    self.assertEqual(summary["kick_event_count_total"], 64.0)
    self.assertEqual(summary["worst_settling_time_s"], 0.42)
    self.assertEqual(summary["worst_recovery_time_s"], 1.3)
    self.assertEqual(summary["best_recovery_time_s"], 1.3)

    with self.assertRaisesRegex(ValueError, "transitions and kick"):
      qualification_payload(
        transitions=[],
        kick_cells=[kick],
        controller_gain_hash=None,
        controller_qualified=False,
        posture_map_hash=None,
        posture_map_qualified=False,
        station_calibration_hash=None,
        station_calibration_qualified=False,
        source_probe={},
      )

  def test_default_arguments_target_stage3(self):
    args = parse_args([])
    self.assertEqual(args.task, "HopperTrex-Hybrid-v2-Stage3")
    self.assertEqual(args.kicks_per_posture, 4)
    self.assertEqual(args.kick_interval, 200)
    self.assertIsNone(args.fit_output)


class ProbeSmokeTest(unittest.TestCase):
  def test_transition_and_kick_run_on_cpu_with_default_artifacts(self):
    """Mechanical correctness only: the default (unqualified) posture
    artifact collapses the envelope, so the legs degenerate to null steps;
    official data requires the machine-room artifact chain."""

    cfg = load_env_cfg("HopperTrex-Hybrid-v2-Stage3", play=True)
    cfg.scene.num_envs = 1
    if cfg.scene.terrain is not None:
      cfg.scene.terrain.num_envs = 1
    legs = transition_legs(
      tuple(cfg.commands["posture"].height_range),
      tuple(cfg.commands["posture"].pitch_range),
    )

    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
      transition = run_transition(
        env,
        start=legs[0][0],
        target=legs[0][1],
        settle_steps=5,
        measure_steps=30,
        height_band=0.005,
        pitch_band=0.015,
      )
      kick = run_kick_cell(
        env,
        height=legs[0][0][0],
        pitch=legs[0][0][1],
        kicks=1,
        kick_interval=25,
        settle_steps=5,
      )
    finally:
      env.close()

    for key in (
      "settling_time_s",
      "height_overshoot",
      "pitch_overshoot",
      "pitch_rate_abs_max",
      "lin_x_abs_max",
      "non_wheel_contact_rate",
      "terminated_events",
    ):
      self.assertTrue(math.isfinite(transition[key]), key)
    self.assertTrue(math.isfinite(kick["recovery_time_s"]))
    self.assertEqual(kick["kick_event_count"], 1.0)
    # The Stage1 kick displaces the plant, so the post-kick velocity trace
    # must actually show the impulse (guards against a silent no-op kick).
    self.assertGreater(kick["post_kick_lin_x_abs_max"], 0.005)


if __name__ == "__main__":
  unittest.main()
