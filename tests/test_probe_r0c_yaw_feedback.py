import unittest

from hoppertrex_mjlab.scripts import diagnose_roll_boundary as diag
from hoppertrex_mjlab.scripts import probe_r0c_yaw_feedback as probe


def _reset(value):
  return {"marker": value}


def _source_payload():
  arms = []
  for candidate_index, candidate in enumerate(diag.r0c_sync_candidates()):
    rows = []
    for env_id in range(16):
      rows.append({
        "env_id": env_id,
        "stair_height_m": 0.0 if env_id < 8 else 0.0025,
        "root_reset": _reset(env_id),
      })
    arms.append({
      "candidate_definition": diag._schedule_candidate_definition(candidate),
      "trials": rows,
    })
  return {
    "kind": "r0c_synchronized_reference_rejection_screen",
    "matched_reset_perturbations_across_candidates": True,
    "candidates": arms,
  }


class R0cYawFeedbackProbeTest(unittest.TestCase):
  def test_source_reset_map_binds_each_reviewed_arm(self):
    expected = probe._source_reset_map(_source_payload(), "c1")
    self.assertEqual(len(expected), 16)
    self.assertEqual(expected[(14, 0.0025)], _reset(14))
    rows = [
      {"env_id": env_id, "stair_height_m": 0.0 if env_id < 8 else 0.0025,
       "root_reset": _reset(env_id)}
      for env_id in range(16)
    ]
    self.assertEqual(probe._validate_resets(rows, expected), expected)
    rows[-1]["root_reset"] = _reset("drift")
    with self.assertRaisesRegex(ValueError, "exact resets drifted"):
      probe._validate_resets(rows, expected)

  def test_trace_summary_separates_deployable_and_simulation_truth(self):
    sample_a = {
      "measured_roll_rad": 0.006,
      "measured_roll_rate_radps": -0.02,
      "roll_feedback_amplitude_rad": 0.0009,
      "yaw_rad": -0.02,
      "measured_yaw_rate_radps": 0.1,
      "yaw_heading_error_rad": -0.01,
      "yaw_feedback_radps": -0.2,
      "yaw_differential_radps": -0.2,
      "forward_wheel_speed_difference_radps": -1.2,
      "wheel_center_x_difference_m": 0.004,
    }
    sample_b = {name: -2.0 * value for name, value in sample_a.items()}
    summary = probe._trace_summary([{
      "stair_height_m": 0.0025,
      "control_trace": [sample_a, sample_b],
    }])
    self.assertEqual(summary["step_max_abs_yaw_rad"], 0.04)
    self.assertEqual(summary["step_max_abs_roll_rad"], 0.012)
    self.assertEqual(
      summary["step_max_abs_roll_feedback_amplitude_rad"], 0.0018
    )
    self.assertEqual(
      summary["step_max_abs_forward_wheel_speed_difference_radps"], 2.4,
    )
    self.assertEqual(
      summary["step_max_abs_wheel_center_x_difference_m"], 0.008,
    )

  def test_parse_args_rejects_duplicate_and_negative_gains(self):
    args = probe.parse_args([
      "--source-result", "source.json", "--output", "out.json",
      "--candidate", "c0", "--kp", "0", "2.5",
    ])
    self.assertEqual(args.candidate, ["c0"])
    self.assertEqual(args.kp, [0.0, 2.5])
    self.assertEqual(args.heading_kp, [0.0])
    self.assertEqual(args.heading_error_limit, 0.12)
    self.assertEqual(args.roll_kp, 0.0)
    self.assertEqual(args.roll_kd, 0.0)
    self.assertEqual(args.roll_limit_mrad, 0.0)
    with self.assertRaises(SystemExit):
      probe.parse_args([
        "--source-result", "source.json", "--output", "out.json",
        "--kp", "1", "1",
      ])
    with self.assertRaises(SystemExit):
      probe.parse_args([
        "--source-result", "source.json", "--output", "out.json",
        "--kp", "-1",
      ])
    with self.assertRaises(SystemExit):
      probe.parse_args([
        "--source-result", "source.json", "--output", "out.json",
        "--roll-kp", "0.15", "--roll-limit-mrad", "4.1",
      ])


if __name__ == "__main__":
  unittest.main()
