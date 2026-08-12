import math
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from hoppertrex_mjlab.scripts import probe_roll_boundary as roll


def _cells(flags, unsafe_height=None):
  result = []
  for card in roll.POSTURE_CARDS:
    for height in roll.formal_heights(10_000):
      passed = bool(flags[card["name"]][height])
      unsafe = unsafe_height is not None and math.isclose(height, unsafe_height)
      result.append({
        "posture_card": card["name"],
        "stair_height_m": height,
        "trials": 48,
        "successes": 48 if passed else 0,
        "success_rate": 1.0 if passed else 0.0,
        "terminated_trials": 1 if unsafe else 0,
        "termination_rate": 1 / 48 if unsafe else 0.0,
        "non_wheel_contact_trials": 0,
        "non_wheel_contact_rate": 0.0,
        "bilateral_airborne_trials": 0,
        "bilateral_airborne_rate": 0.0,
        "passed": passed and not unsafe,
      })
  return result


def _flags(first_failure):
  heights = roll.formal_heights(10_000)
  return {
    card["name"]: {height: first_failure is None or height < first_failure for height in heights}
    for card in roll.POSTURE_CARDS
  }


class TerrainContractTest(unittest.TestCase):
  def test_five_subcentimetre_heights_are_unique_ordered_and_exact(self):
    heights = (0.0, 0.0025, 0.005, 0.0075, 0.01)
    sub = roll.roll_boundary_sub_terrains(heights)
    self.assertEqual(tuple(sub), (
      "stair_000000um", "stair_002500um", "stair_005000um",
      "stair_007500um", "stair_010000um",
    ))
    self.assertEqual(
      tuple(cfg.step_height_range for cfg in sub.values()),
      tuple((height, height) for height in heights),
    )

  def test_invalid_height_requests_fail_closed(self):
    for heights in (
      (0.0, 0.005, 0.0025),
      (0.0, 0.0025, 0.0025),
      (0.0, 0.0025004),
      (0.0, float("nan")),
      (0.0, -0.0025),
    ):
      with self.subTest(heights=heights), self.assertRaises(ValueError):
        roll.roll_boundary_sub_terrains(heights)

  def test_environment_binds_final_stack_and_zeroes_all_residual_heads(self):
    cfg = roll.make_roll_boundary_env_cfg((0.0, 0.0025, 0.005), 1)
    action = cfg.actions["hybrid_wheel_leg"]
    self.assertEqual(action.action_mask, roll.ZERO_ACTION_MASK)
    self.assertEqual(action.controller_gain_hash, roll.EXPECTED_SCHEDULE_HASH)
    self.assertTrue(action.yaw_calibration_qualified)
    self.assertTrue(action.posture_map_qualified)
    self.assertTrue(action.station_calibration_qualified)
    self.assertIsNone(action.dynamic_stair_maneuver)
    self.assertIsNone(action.stair_trigger_sensor_name)
    self.assertFalse(cfg.auto_reset)
    generator = cfg.scene.terrain.terrain_generator
    self.assertEqual(generator.num_cols, 3)
    self.assertEqual(tuple(generator.sub_terrains), (
      "stair_000000um", "stair_002500um", "stair_005000um",
    ))

  def test_artifact_sha_drift_is_rejected(self):
    with (
      patch.object(roll, "_sha256", return_value="0" * 64),
      self.assertRaises(ValueError),
    ):
      roll.frozen_artifact_paths()


class SafetyTest(unittest.TestCase):
  def test_bilateral_airborne_requires_both_wheels_unloaded(self):
    left = torch.tensor([True, False, False, True])
    right = torch.tensor([True, True, False, False])
    self.assertEqual(roll.bilateral_airborne(left, right).tolist(), [False, False, True, False])

  def test_event_is_latched_before_reset(self):
    history = torch.tensor([False, False])
    event = torch.tensor([True, True])
    was_active = torch.tensor([True, False])
    self.assertEqual(roll.latch_before_reset(history, event, was_active).tolist(), [True, False])

  def test_model_torque_is_clipped_and_reports_saturation(self):
    torque, saturated = roll.model_wheel_torque(
      torch.tensor([[1.0, 0.001]]), torch.zeros(1, 2)
    )
    self.assertAlmostEqual(float(torque[0, 0]), roll.RMD_L_9025_35T_PEAK_TORQUE, places=6)
    self.assertEqual(saturated.tolist(), [[True, False]])

  def test_progress_metric_starts_only_when_drive_starts(self):
    source = Path(roll.__file__).read_text(encoding="utf-8")
    start = source.index("def run_card_repeat")
    drive_guard = source.index("if drive_index is not None:", start)
    progress_update = source.index("max_progress.copy_", drive_guard)
    self.assertLess(drive_guard, progress_update)

  def test_policy_observes_the_command_forced_on_the_same_step(self):
    source = Path(roll.__file__).read_text(encoding="utf-8")
    force = source.index("_force_commands(env, active=was_active")
    refresh = source.index("observation = env.get_observations()", force)
    policy = source.index("candidate = policy(observation)", refresh)
    step = source.index("env.step(actions)", policy)
    self.assertLess(force, refresh)
    self.assertLess(refresh, policy)
    self.assertLess(policy, step)



class AggregationTest(unittest.TestCase):
  def test_unexpected_trial_cell_fails_closed(self):
    row = {
      "posture_card": "unexpected", "stair_height_m": 0.0,
      "repeat": 1, "env_id": 0, "success": False,
      "termination": False, "non_wheel_contact": False,
      "bilateral_airborne_ever": False,
    }
    with self.assertRaises(ValueError):
      roll.aggregate_trials(
        [row], heights=(0.0,), expected_repeats=1,
        expected_envs_per_height=1, cards=({"name": "expected"},),
      )

class VerdictTest(unittest.TestCase):
  def test_positive_safe_failure_produces_training_eligible_bracket(self):
    verdict = roll.classify_results(_cells(_flags(0.0075)), heights=roll.formal_heights(10_000))
    self.assertEqual(verdict["classification"], "CLASSICAL_CROLL_BRACKETED")
    self.assertEqual(verdict["croll_bracket_m"], [0.005, 0.0075])
    self.assertTrue(verdict["training_eligible"])

  def test_no_positive_height_stops_ppo(self):
    verdict = roll.classify_results(_cells(_flags(0.0025)), heights=roll.formal_heights(10_000))
    self.assertEqual(verdict["classification"], "NO_POSITIVE_CLASSICAL_CROLL")
    self.assertFalse(verdict["training_eligible"])

  def test_unsafe_next_height_stops_training(self):
    verdict = roll.classify_results(
      _cells(_flags(0.0075), unsafe_height=0.0075), heights=roll.formal_heights(10_000)
    )
    self.assertEqual(verdict["classification"], "NEXT_HEIGHT_UNSAFE_STOP")
    self.assertFalse(verdict["training_eligible"])

  def test_all_pass_at_10mm_requests_extension(self):
    verdict = roll.classify_results(_cells(_flags(None)), heights=roll.formal_heights(10_000))
    self.assertEqual(verdict["classification"], "EXTEND_ROLL_BOUNDARY_SWEEP")
    self.assertEqual(verdict["croll_bracket_m"], [0.01, None])

  def test_nonmonotonic_result_fails_closed(self):
    flags = _flags(0.005)
    for card in roll.POSTURE_CARDS:
      flags[card["name"]][0.0075] = True
    verdict = roll.classify_results(_cells(flags), heights=roll.formal_heights(10_000))
    self.assertEqual(verdict["classification"], "NON_MONOTONIC_STOP")


class CliTest(unittest.TestCase):
  def test_smoke_is_not_evidence(self):
    args = roll.parse_args(["--output", "x.json", "--device", "cpu", "--smoke"])
    self.assertTrue(args.smoke)
    protocol = roll.protocol_for_mode(True)
    self.assertFalse(protocol["evidence_eligible"])
    self.assertEqual(protocol["heights_m"], (0.0025, 0.005, 0.0075))

  def test_existing_output_is_rejected_before_runtime(self):
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / "existing.json"
      output.write_text("{}", encoding="utf-8")
      with self.assertRaises(SystemExit):
        roll.parse_args(["--output", str(output), "--device", "cpu", "--smoke"])

  def test_formal_cpu_is_rejected(self):
    with self.assertRaises(SystemExit):
      roll.parse_args(["--output", "x.json", "--device", "cpu"])


if __name__ == "__main__":
  unittest.main()
