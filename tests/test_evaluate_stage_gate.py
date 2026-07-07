import unittest

from hoppertrex_mjlab.scripts.rsl_rl.evaluate_stage_gate import (
  _stage2_fixed_command_checks,
  _stage3_fixed_yaw_checks,
)


class Stage2PromotionGateTest(unittest.TestCase):
  def test_stage2_fixed_command_checks_reject_late_wrong_direction(self):
    summaries = [
      {
        "lin_x": -0.07,
        "command_match_frac": 0.95,
        "late_slow_env_frac": 1.0,
        "late_wrong_direction_env_frac": 1.0,
        "mean_abs_error": 0.04,
        "p95_pitch": 0.03,
        "p99_pitch_rate": 0.7,
        "terminated_event_rate": 0.0,
      },
      {
        "lin_x": 0.07,
        "command_match_frac": 0.95,
        "late_slow_env_frac": 0.0,
        "late_wrong_direction_env_frac": 0.0,
        "mean_abs_error": 0.04,
        "p95_pitch": 0.03,
        "p99_pitch_rate": 0.7,
        "terminated_event_rate": 0.0,
      },
    ]

    checks = _stage2_fixed_command_checks(summaries)

    self.assertFalse(all(passed for passed, _detail in checks))
    self.assertTrue(
      any("fixed_-0.070_late_wrong_direction_env_frac" in detail for _passed, detail in checks)
    )

  def test_stage2_fixed_command_checks_accept_clean_late_tracking(self):
    summaries = [
      {
        "lin_x": -0.07,
        "command_match_frac": 0.96,
        "late_slow_env_frac": 0.0,
        "late_wrong_direction_env_frac": 0.0,
        "mean_abs_error": 0.03,
        "p95_pitch": 0.03,
        "p99_pitch_rate": 0.7,
        "terminated_event_rate": 0.0,
      },
      {
        "lin_x": 0.07,
        "command_match_frac": 0.96,
        "late_slow_env_frac": 0.0,
        "late_wrong_direction_env_frac": 0.0,
        "mean_abs_error": 0.03,
        "p95_pitch": 0.03,
        "p99_pitch_rate": 0.7,
        "terminated_event_rate": 0.0,
      },
    ]

    checks = _stage2_fixed_command_checks(summaries)

    self.assertTrue(all(passed for passed, _detail in checks))


class Stage3PromotionGateTest(unittest.TestCase):
  def test_stage3_fixed_yaw_checks_reject_late_linear_drift(self):
    summaries = [
      {
        "yaw": -0.07,
        "mean_actual_yaw": -0.07,
        "command_match_frac": 0.95,
        "late_slow_env_frac": 0.0,
        "late_wrong_direction_env_frac": 0.0,
        "late_lin_drift_env_frac": 1.0,
        "yaw_abs_error_mean": 0.03,
        "lin_drift_abs_mean": 0.08,
        "p95_pitch": 0.03,
        "p99_pitch_rate": 0.7,
        "wheel_saturation_ratio": 0.0,
        "terminated_event_rate": 0.0,
      },
      {
        "yaw": 0.07,
        "mean_actual_yaw": 0.07,
        "command_match_frac": 0.95,
        "late_slow_env_frac": 0.0,
        "late_wrong_direction_env_frac": 0.0,
        "late_lin_drift_env_frac": 0.0,
        "yaw_abs_error_mean": 0.03,
        "lin_drift_abs_mean": 0.02,
        "p95_pitch": 0.03,
        "p99_pitch_rate": 0.7,
        "wheel_saturation_ratio": 0.0,
        "terminated_event_rate": 0.0,
      },
    ]

    checks = _stage3_fixed_yaw_checks(summaries)

    self.assertFalse(all(passed for passed, _detail in checks))
    self.assertTrue(
      any("fixed_yaw_-0.070_late_lin_drift_env_frac" in detail for _passed, detail in checks)
    )

  def test_stage3_fixed_yaw_checks_accept_clean_late_turning(self):
    summaries = [
      {
        "yaw": -0.07,
        "mean_actual_yaw": -0.07,
        "command_match_frac": 0.96,
        "late_slow_env_frac": 0.0,
        "late_wrong_direction_env_frac": 0.0,
        "late_lin_drift_env_frac": 0.0,
        "yaw_abs_error_mean": 0.03,
        "lin_drift_abs_mean": 0.02,
        "p95_pitch": 0.03,
        "p99_pitch_rate": 0.7,
        "wheel_saturation_ratio": 0.0,
        "terminated_event_rate": 0.0,
      },
      {
        "yaw": 0.07,
        "mean_actual_yaw": 0.07,
        "command_match_frac": 0.96,
        "late_slow_env_frac": 0.0,
        "late_wrong_direction_env_frac": 0.0,
        "late_lin_drift_env_frac": 0.0,
        "yaw_abs_error_mean": 0.03,
        "lin_drift_abs_mean": 0.02,
        "p95_pitch": 0.03,
        "p99_pitch_rate": 0.7,
        "wheel_saturation_ratio": 0.0,
        "terminated_event_rate": 0.0,
      },
    ]

    checks = _stage3_fixed_yaw_checks(summaries)

    self.assertTrue(all(passed for passed, _detail in checks))

  def test_stage3_fixed_yaw_checks_reject_no_turn_policy(self):
    summaries = [
      {
        "yaw": -0.07,
        "mean_actual_yaw": -0.01,
        "command_match_frac": 0.95,
        "late_slow_env_frac": 0.0,
        "late_wrong_direction_env_frac": 0.0,
        "late_lin_drift_env_frac": 0.0,
        "yaw_abs_error_mean": 0.065,
        "lin_drift_abs_mean": 0.02,
        "p95_pitch": 0.03,
        "p99_pitch_rate": 0.7,
        "wheel_saturation_ratio": 0.0,
        "terminated_event_rate": 0.0,
      },
      {
        "yaw": 0.07,
        "mean_actual_yaw": 0.01,
        "command_match_frac": 0.95,
        "late_slow_env_frac": 0.0,
        "late_wrong_direction_env_frac": 0.0,
        "late_lin_drift_env_frac": 0.0,
        "yaw_abs_error_mean": 0.065,
        "lin_drift_abs_mean": 0.02,
        "p95_pitch": 0.03,
        "p99_pitch_rate": 0.7,
        "wheel_saturation_ratio": 0.0,
        "terminated_event_rate": 0.0,
      },
    ]

    checks = _stage3_fixed_yaw_checks(summaries)

    self.assertFalse(all(passed for passed, _detail in checks))
    self.assertTrue(
      any("signed_mean_actual_yaw" in detail for _passed, detail in checks)
    )


if __name__ == "__main__":
  unittest.main()
