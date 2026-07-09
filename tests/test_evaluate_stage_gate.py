import contextlib
import io
import unittest

from hoppertrex_mjlab.scripts.rsl_rl.evaluate_stage_gate import (
  _promotion_output_context,
  _stage2_fixed_command_checks,
  _stage3_fixed_yaw_checks,
  _stage45_fixed_combo_checks,
)


def _stage2_summary(lin_x: float, **overrides: float) -> dict[str, float]:
  summary = {
    "lin_x": lin_x,
    "command_match_frac": 0.96,
    "late_slow_env_frac": 0.0,
    "late_wrong_direction_env_frac": 0.0,
    "in_band_frac": 0.95,
    "fast_frac": 0.02,
    "late_in_band_frac": 0.95,
    "lin_x_delta_rms": 0.02,
    "lin_x_delta_abs_p95": 0.04,
    "late_lin_x_delta_rms": 0.02,
    "late_lin_x_delta_abs_p95": 0.04,
    "mean_abs_error": 0.03,
    "p95_pitch": 0.03,
    "p99_pitch_rate": 0.7,
    "terminated_event_rate": 0.0,
  }
  summary.update(overrides)
  return summary


def _stage3_summary(yaw: float, **overrides: float) -> dict[str, float]:
  summary = {
    "yaw": yaw,
    "mean_actual_yaw": yaw,
    "command_match_frac": 0.96,
    "wrong_direction_frac": 0.0,
    "late_slow_env_frac": 0.0,
    "late_wrong_direction_env_frac": 0.0,
    "late_lin_drift_env_frac": 0.0,
    "in_band_frac": 0.95,
    "fast_frac": 0.02,
    "late_in_band_frac": 0.95,
    "yaw_delta_rms": 0.02,
    "yaw_delta_abs_p95": 0.04,
    "late_yaw_delta_rms": 0.02,
    "late_yaw_delta_abs_p95": 0.04,
    "yaw_abs_error_mean": 0.03,
    "yaw_abs_error_p90": 0.06,
    "lin_drift_abs_mean": 0.02,
    "p95_pitch": 0.03,
    "p99_pitch_rate": 0.7,
    "wheel_saturation_ratio": 0.0,
    "terminated_event_rate": 0.0,
  }
  summary.update(overrides)
  return summary


def _stage45_combo_summary(
  lin_x: float,
  yaw: float,
  **overrides: float,
) -> dict[str, float]:
  summary = _stage3_summary(yaw)
  summary.update(
    {
      "lin_x": lin_x,
      "lin_command_match_frac": 0.94,
      "lin_wrong_direction_frac": 0.0,
      "lin_in_band_frac": 0.86,
      "lin_fast_frac": 0.08,
      "late_lin_in_band_frac": 0.86,
      "lin_abs_error_mean": 0.035,
      "lin_abs_error_p90": 0.070,
      "lin_x_delta_rms": 0.030,
      "lin_x_delta_abs_p95": 0.060,
      "late_lin_x_delta_rms": 0.030,
      "late_lin_x_delta_abs_p95": 0.060,
    }
  )
  summary.update(overrides)
  return summary


class Stage2PromotionGateTest(unittest.TestCase):
  def test_stage2_fixed_command_checks_reject_late_wrong_direction(self):
    summaries = [
      _stage2_summary(
        -0.07,
        late_slow_env_frac=1.0,
        late_wrong_direction_env_frac=1.0,
      ),
      _stage2_summary(0.07),
    ]

    checks = _stage2_fixed_command_checks(summaries)

    self.assertFalse(all(passed for passed, _detail in checks))
    self.assertTrue(
      any(
        "fixed_-0.070_late_wrong_direction_env_frac" in detail
        for _passed, detail in checks
      )
    )

  def test_stage2_fixed_command_checks_accept_clean_late_tracking(self):
    summaries = [_stage2_summary(-0.07), _stage2_summary(0.07)]

    checks = _stage2_fixed_command_checks(summaries)

    self.assertTrue(all(passed for passed, _detail in checks))

  def test_stage2_fixed_command_checks_reject_pulsed_tracking(self):
    summaries = [
      _stage2_summary(
        -0.07,
        in_band_frac=0.50,
        fast_frac=0.35,
        late_in_band_frac=0.40,
        lin_x_delta_rms=0.05,
        lin_x_delta_abs_p95=0.09,
      ),
      _stage2_summary(0.07),
    ]

    checks = _stage2_fixed_command_checks(summaries)

    self.assertFalse(all(passed for passed, _detail in checks))
    self.assertTrue(
      any("fixed_-0.070_in_band_frac" in detail for _passed, detail in checks)
    )
    self.assertTrue(
      any("fixed_-0.070_fast_frac" in detail for _passed, detail in checks)
    )
    self.assertTrue(
      any("fixed_-0.070_late_in_band_frac" in detail for _passed, detail in checks)
    )
    self.assertTrue(
      any("fixed_-0.070_lin_x_delta_rms" in detail for _passed, detail in checks)
    )
    self.assertTrue(
      any("fixed_-0.070_lin_x_delta_abs_p95" in detail for _passed, detail in checks)
    )

  def test_stage2_fixed_command_checks_reject_late_delta_pulsing(self):
    summaries = [
      _stage2_summary(
        -0.07,
        late_lin_x_delta_rms=0.05,
        late_lin_x_delta_abs_p95=0.09,
      ),
      _stage2_summary(0.07),
    ]

    checks = _stage2_fixed_command_checks(summaries)

    self.assertFalse(all(passed for passed, _detail in checks))
    self.assertTrue(
      any(
        "fixed_-0.070_late_lin_x_delta_rms" in detail
        for _passed, detail in checks
      )
    )
    self.assertTrue(
      any(
        "fixed_-0.070_late_lin_x_delta_abs_p95" in detail
        for _passed, detail in checks
      )
    )


class PromotionOutputContextTest(unittest.TestCase):
  def test_json_mode_redirects_promotion_prints_to_stderr(self):
    stdout = io.StringIO()
    stderr = io.StringIO()

    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
      with _promotion_output_context(json_output=True):
        print("promotion progress")

    self.assertEqual(stdout.getvalue(), "")
    self.assertEqual(stderr.getvalue(), "promotion progress\n")

  def test_human_mode_keeps_promotion_prints_on_stdout(self):
    stdout = io.StringIO()
    stderr = io.StringIO()

    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
      with _promotion_output_context(json_output=False):
        print("promotion progress")

    self.assertEqual(stdout.getvalue(), "promotion progress\n")
    self.assertEqual(stderr.getvalue(), "")


class Stage3PromotionGateTest(unittest.TestCase):
  def test_stage3_fixed_yaw_checks_reject_late_linear_drift(self):
    summaries = [
      _stage3_summary(-0.07, late_lin_drift_env_frac=1.0, lin_drift_abs_mean=0.08),
      _stage3_summary(0.07),
    ]

    checks = _stage3_fixed_yaw_checks(summaries)

    self.assertFalse(all(passed for passed, _detail in checks))
    self.assertTrue(
      any(
        "fixed_yaw_-0.070_late_lin_drift_env_frac" in detail
        for _passed, detail in checks
      )
    )

  def test_stage3_fixed_yaw_checks_accept_clean_late_turning(self):
    summaries = [_stage3_summary(-0.07), _stage3_summary(0.07)]

    checks = _stage3_fixed_yaw_checks(summaries)

    self.assertTrue(all(passed for passed, _detail in checks))

  def test_stage3_fixed_yaw_checks_reject_no_turn_policy(self):
    summaries = [
      _stage3_summary(-0.07, mean_actual_yaw=-0.01, yaw_abs_error_mean=0.065),
      _stage3_summary(0.07, mean_actual_yaw=0.01, yaw_abs_error_mean=0.065),
    ]

    checks = _stage3_fixed_yaw_checks(summaries)

    self.assertFalse(all(passed for passed, _detail in checks))
    self.assertTrue(
      any("signed_mean_actual_yaw" in detail for _passed, detail in checks)
    )

  def test_stage3_fixed_yaw_checks_reject_high_p90_yaw_error(self):
    summaries = [
      _stage3_summary(-0.07, yaw_abs_error_mean=0.04, yaw_abs_error_p90=0.12),
      _stage3_summary(0.07, yaw_abs_error_mean=0.04, yaw_abs_error_p90=0.12),
    ]

    checks = _stage3_fixed_yaw_checks(summaries)

    self.assertFalse(all(passed for passed, _detail in checks))
    self.assertTrue(
      any("yaw_abs_error_p90" in detail for _passed, detail in checks)
    )

  def test_stage3_fixed_yaw_checks_reject_pulsed_or_overshoot_tracking(self):
    summaries = [
      _stage3_summary(
        -0.07,
        in_band_frac=0.50,
        fast_frac=0.35,
        yaw_abs_error_mean=0.04,
        yaw_abs_error_p90=0.08,
      ),
      _stage3_summary(0.07),
    ]

    checks = _stage3_fixed_yaw_checks(summaries)

    self.assertFalse(all(passed for passed, _detail in checks))
    self.assertTrue(
      any("fixed_yaw_-0.070_in_band_frac" in detail for _passed, detail in checks)
    )
    self.assertTrue(
      any("fixed_yaw_-0.070_fast_frac" in detail for _passed, detail in checks)
    )

  def test_stage3_fixed_yaw_checks_reject_pulsed_delta_tracking(self):
    summaries = [
      _stage3_summary(
        -0.07,
        late_in_band_frac=0.40,
        yaw_delta_rms=0.05,
        yaw_delta_abs_p95=0.09,
      ),
      _stage3_summary(0.07),
    ]

    checks = _stage3_fixed_yaw_checks(summaries)

    self.assertFalse(all(passed for passed, _detail in checks))
    self.assertTrue(
      any(
        "fixed_yaw_-0.070_late_in_band_frac" in detail
        for _passed, detail in checks
      )
    )
    self.assertTrue(
      any("fixed_yaw_-0.070_yaw_delta_rms" in detail for _passed, detail in checks)
    )
    self.assertTrue(
      any(
        "fixed_yaw_-0.070_yaw_delta_abs_p95" in detail
        for _passed, detail in checks
      )
    )

  def test_stage3_fixed_yaw_checks_reject_late_delta_pulsing(self):
    summaries = [
      _stage3_summary(
        -0.07,
        late_yaw_delta_rms=0.05,
        late_yaw_delta_abs_p95=0.09,
      ),
      _stage3_summary(0.07),
    ]

    checks = _stage3_fixed_yaw_checks(summaries)

    self.assertFalse(all(passed for passed, _detail in checks))
    self.assertTrue(
      any(
        "fixed_yaw_-0.070_late_yaw_delta_rms" in detail
        for _passed, detail in checks
      )
    )
    self.assertTrue(
      any(
        "fixed_yaw_-0.070_late_yaw_delta_abs_p95" in detail
        for _passed, detail in checks
      )
    )


class Stage45PromotionGateTest(unittest.TestCase):
  def test_stage45_fixed_combo_checks_accept_clean_combined_tracking(self):
    summaries = [
      _stage45_combo_summary(-0.05, -0.05),
      _stage45_combo_summary(-0.05, 0.05),
      _stage45_combo_summary(0.05, -0.05),
      _stage45_combo_summary(0.05, 0.05),
    ]

    checks = _stage45_fixed_combo_checks(summaries)

    self.assertTrue(all(passed for passed, _detail in checks))

  def test_stage45_fixed_combo_checks_reject_linear_pulsing(self):
    summaries = [
      _stage45_combo_summary(
        -0.05,
        0.05,
        lin_in_band_frac=0.40,
        lin_fast_frac=0.38,
        late_lin_in_band_frac=0.35,
        lin_x_delta_rms=0.06,
        lin_x_delta_abs_p95=0.11,
      ),
      _stage45_combo_summary(0.05, -0.05),
    ]

    checks = _stage45_fixed_combo_checks(summaries)

    self.assertFalse(all(passed for passed, _detail in checks))
    self.assertTrue(
      any("combo_-0.050_+0.050_lin_in_band_frac" in detail for _passed, detail in checks)
    )
    self.assertTrue(
      any("combo_-0.050_+0.050_lin_x_delta_rms" in detail for _passed, detail in checks)
    )

  def test_stage45_fixed_combo_checks_reject_yaw_pulsing(self):
    summaries = [
      _stage45_combo_summary(
        0.05,
        -0.05,
        in_band_frac=0.45,
        fast_frac=0.36,
        late_in_band_frac=0.40,
        yaw_delta_rms=0.06,
        yaw_delta_abs_p95=0.12,
      ),
      _stage45_combo_summary(-0.05, 0.05),
    ]

    checks = _stage45_fixed_combo_checks(summaries)

    self.assertFalse(all(passed for passed, _detail in checks))
    self.assertTrue(
      any("combo_+0.050_-0.050_yaw_in_band_frac" in detail for _passed, detail in checks)
    )
    self.assertTrue(
      any("combo_+0.050_-0.050_yaw_delta_rms" in detail for _passed, detail in checks)
    )


if __name__ == "__main__":
  unittest.main()
