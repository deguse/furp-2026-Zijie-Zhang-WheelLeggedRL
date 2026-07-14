import json
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hoppertrex_mjlab.scripts.rsl_rl.hybrid_gate import (
  aggregate_seed_results,
  boolean_mask_on_device,
  evaluate_capability_suite,
  make_result_envelope,
  resolve_wheel_action,
  wheel_target_saturation_threshold,
  zero_where_masked,
  to_deterministic_json,
)
from hoppertrex_mjlab.scripts.rsl_rl.evaluate_hybrid_gate import (
  HYBRID_STAGE_SUITES,
  HYBRID_STAGE_TASKS,
  _controller_hash,
  _extract_stage4_reference,
  _fixed_rows_to_scenarios,
  _linear_row_to_scenario,
  _mean_event_error_integral,
  _settling_time_s,
  _survival_rate,
  _validate_scenario_file_profile,
  _validate_live_scenario_coverage,
  _validate_rollout_args,
  parse_args,
  validate_hybrid_evaluation_checkpoint,
)


def _linear_metrics(**overrides: float) -> dict[str, float]:
  metrics = {
    "command_match_frac": 0.90,
    "late_slow_env_frac": 0.10,
    "late_wrong_direction_env_frac": 0.10,
    "in_band_frac": 0.70,
    "fast_frac": 0.25,
    "late_in_band_frac": 0.80,
    "target_band_frac": 0.70,
    "late_target_band_frac": 0.80,
    "signed_speed_ratio_mean": 0.75,
    "lin_x_delta_rms": 0.035,
    "lin_x_delta_abs_p95": 0.070,
    "late_lin_x_delta_rms": 0.035,
    "late_lin_x_delta_abs_p95": 0.070,
    "mean_abs_error": 0.06,
    "p95_pitch": 0.08,
    "p99_pitch_rate": 0.90,
    "balance_residual_abs_mean": 0.30,
    "balance_residual_abs_p95": 0.45,
    "terminated_event_rate": 0.01,
  }
  metrics.update(overrides)
  return metrics


def _yaw_metrics(yaw: float = 0.10, **overrides: float) -> dict[str, float]:
  metrics = {
    "mean_actual_yaw": 0.5 * yaw,
    "command_match_frac": 0.90,
    "late_slow_env_frac": 0.10,
    "late_wrong_direction_env_frac": 0.10,
    "late_lin_drift_env_frac": 0.10,
    "in_band_frac": 0.70,
    "fast_frac": 0.25,
    "late_in_band_frac": 0.70,
    "yaw_delta_rms": 0.035,
    "yaw_delta_abs_p95": 0.080,
    "late_yaw_delta_rms": 0.035,
    "late_yaw_delta_abs_p95": 0.080,
    "yaw_abs_error_mean": 0.07,
    "yaw_abs_error_p90": 0.10,
    "lin_drift_abs_mean": 0.05,
    "p95_pitch": 0.10,
    "p99_pitch_rate": 0.90,
    "wheel_saturation_ratio": 0.20,
    "terminated_event_rate": 0.01,
  }
  metrics.update(overrides)
  return metrics


def _combo_metrics(**overrides: float) -> dict[str, float]:
  metrics = {
    "lin_command_match_frac": 0.85,
    "lin_wrong_direction_frac": 0.10,
    "lin_in_band_frac": 0.70,
    "lin_fast_frac": 0.30,
    "late_lin_in_band_frac": 0.70,
    "lin_abs_error_mean": 0.07,
    "lin_abs_error_p90": 0.12,
    "lin_x_delta_rms": 0.045,
    "lin_x_delta_abs_p95": 0.090,
    "late_lin_x_delta_rms": 0.045,
    "late_lin_x_delta_abs_p95": 0.090,
    "command_match_frac": 0.85,
    "wrong_direction_frac": 0.10,
    "in_band_frac": 0.65,
    "fast_frac": 0.30,
    "late_in_band_frac": 0.65,
    "yaw_abs_error_mean": 0.08,
    "yaw_abs_error_p90": 0.12,
    "yaw_delta_rms": 0.045,
    "yaw_delta_abs_p95": 0.090,
    "late_yaw_delta_rms": 0.045,
    "late_yaw_delta_abs_p95": 0.090,
    "p95_pitch": 0.12,
    "p99_pitch_rate": 0.95,
    "wheel_saturation_ratio": 0.20,
    "terminated_event_rate": 0.01,
  }
  metrics.update(overrides)
  return metrics


def _scenario(
  name: str,
  kind: str,
  metrics: dict[str, float],
  **commands: float,
) -> dict[str, object]:
  return {"name": name, "kind": kind, "metrics": metrics, **commands}


def _stage1_metrics(**overrides: float) -> dict[str, float]:
  metrics = {
    "candidate_terminated_event_rate": 0.0,
    "candidate_p95_pitch": 0.02,
    "candidate_p99_pitch_rate": 0.20,
    "candidate_balance_residual_abs_mean": 0.02,
    "candidate_balance_residual_abs_p95": 0.05,
    "candidate_mean_abs_error": 0.009,
    "baseline_mean_abs_error": 0.010,
    "baseline_p95_pitch": 0.02,
    "candidate_lin_x_delta_rms": 0.009,
    "baseline_lin_x_delta_rms": 0.010,
  }
  metrics.update(overrides)
  return metrics


def _integrated_metrics(**overrides: float) -> dict[str, float]:
  metrics = {
    "tracking_error": 0.08,
    "terminated_event_rate": 0.0,
    "survival_rate": 1.0,
    "recovery_time_s": 1.0,
    "non_wheel_contact_rate": 0.0,
    "wheel_saturation_ratio": 0.0,
  }
  metrics.update(overrides)
  return metrics


class CapabilitySuiteTest(unittest.TestCase):
  def assert_gate_passes(self, suite: str, scenarios, **kwargs) -> None:
    checks = evaluate_capability_suite(suite, scenarios, **kwargs)
    self.assertTrue(all(check.passed for check in checks), checks)

  def test_controller_accepts_exact_safety_and_speed_boundaries(self):
    scenarios = [
      _scenario(
        "stand",
        "controller",
        {
          "duration_s": 60.0,
          "terminated_event_rate": 0.01,
          "p95_pitch": 0.08,
          "p99_pitch_rate": 0.90,
          "mean_actual_lin_x": 0.01,
        },
        lin_x=0.0,
      ),
      _scenario(
        "reverse",
        "controller",
        {
          "duration_s": 60.0,
          "terminated_event_rate": 0.01,
          "p95_pitch": 0.08,
          "p99_pitch_rate": 0.90,
          "signed_speed_ratio_mean": 0.75,
        },
        lin_x=-0.07,
      ),
      _scenario(
        "forward",
        "controller",
        {
          "duration_s": 60.0,
          "terminated_event_rate": 0.01,
          "p95_pitch": 0.08,
          "p99_pitch_rate": 0.90,
          "signed_speed_ratio_mean": 1.25,
        },
        lin_x=0.07,
      ),
    ]

    self.assert_gate_passes("controller", scenarios)

  def test_controller_rejects_zero_command_drift(self):
    scenarios = [_scenario(
      "stand", "controller",
      {"duration_s": 60.0, "terminated_event_rate": 0.0,
       "p95_pitch": 0.02, "p99_pitch_rate": 0.2,
       "mean_actual_lin_x": 0.01001},
      lin_x=0.0,
    )]
    checks = evaluate_capability_suite("controller", scenarios)
    self.assertTrue(any(
      check.name == "mean_abs_stand_velocity" and not check.passed
      for check in checks
    ))

  def test_controller_rejects_nan_metrics(self):
    scenarios = [
      _scenario(
        "stand",
        "controller",
        {
          "duration_s": 60.0,
          "terminated_event_rate": 0.0,
          "p95_pitch": math.nan,
          "p99_pitch_rate": 0.2,
        },
        lin_x=0.0,
      )
    ]

    checks = evaluate_capability_suite("controller", scenarios)

    self.assertFalse(all(check.passed for check in checks))
    self.assertTrue(any(check.name == "p95_pitch" for check in checks))

  def test_linear_reuses_stage2_thresholds_inclusively(self):
    scenarios = [
      _scenario("reverse", "linear", _linear_metrics(), lin_x=-0.07),
      _scenario(
        "forward",
        "linear",
        _linear_metrics(signed_speed_ratio_mean=1.25),
        lin_x=0.07,
      ),
    ]

    self.assert_gate_passes("linear", scenarios)

  def test_stage1_residual_gate_requires_no_regression_and_hard_improvement(self):
    scenarios = [
      _scenario(
        f"nominal_{command:+.2f}",
        "nominal",
        _stage1_metrics(),
        lin_x=command,
      )
      for command in (-0.07, 0.0, 0.07)
    ]
    scenarios.extend([
      _scenario(
        "extension_reverse",
        "extension",
        _stage1_metrics(
          candidate_mean_abs_error=0.018,
          baseline_mean_abs_error=0.020,
        ),
        lin_x=-0.10,
      ),
      _scenario(
        "extension_forward",
        "extension",
        _stage1_metrics(
          candidate_mean_abs_error=0.018,
          baseline_mean_abs_error=0.020,
        ),
        lin_x=0.10,
      ),
      _scenario(
        "model_mismatch",
        "mismatch",
        _stage1_metrics(
          candidate_mean_abs_error=0.018,
          baseline_mean_abs_error=0.020,
        ),
        lin_x=0.07,
      ),
      _scenario(
        "pushes",
        "disturbance",
        _stage1_metrics(
          candidate_recovery_time_s=0.80,
          baseline_recovery_time_s=1.00,
          candidate_post_kick_error_integral=0.018,
          baseline_post_kick_error_integral=0.020,
        ),
      ),
      _scenario(
        "transitions",
        "transition",
        _stage1_metrics(
          candidate_settling_time_s=0.90,
          baseline_settling_time_s=1.00,
          candidate_tracking_error_integral=0.09,
          baseline_tracking_error_integral=0.10,
          candidate_overshoot_abs_mean=0.018,
          baseline_overshoot_abs_mean=0.020,
        ),
      ),
    ])

    self.assert_gate_passes("residual", scenarios)

  def test_stage1_residual_gate_rejects_nominal_damage(self):
    scenarios = [
      _scenario(
        "nominal",
        "nominal",
        _stage1_metrics(
          candidate_mean_abs_error=0.020,
          baseline_mean_abs_error=0.010,
        ),
      ),
      _scenario(
        "extension",
        "extension",
        _stage1_metrics(
          candidate_mean_abs_error=0.018,
          baseline_mean_abs_error=0.020,
        ),
      ),
      _scenario("mismatch", "mismatch", _stage1_metrics(
        candidate_mean_abs_error=0.010,
        baseline_mean_abs_error=0.010,
      )),
      _scenario(
        "pushes",
        "disturbance",
        _stage1_metrics(
          candidate_recovery_time_s=0.80,
          baseline_recovery_time_s=1.00,
          candidate_post_kick_error_integral=0.018,
          baseline_post_kick_error_integral=0.020,
        ),
      ),
      _scenario(
        "transitions",
        "transition",
        _stage1_metrics(
          candidate_settling_time_s=0.90,
          baseline_settling_time_s=1.00,
          candidate_tracking_error_integral=0.09,
          baseline_tracking_error_integral=0.10,
          candidate_overshoot_abs_mean=0.018,
          baseline_overshoot_abs_mean=0.020,
        ),
      ),
    ]

    checks = evaluate_capability_suite("residual", scenarios)

    self.assertTrue(any(
      check.name == "nominal_tracking_no_regression" and not check.passed
      for check in checks
    ))

  def test_stage1_unsafe_extension_cannot_supply_improvement_evidence(self):
    scenarios = [
      _scenario(
        f"nominal_{command:+.2f}",
        "nominal",
        _stage1_metrics(
          candidate_mean_abs_error=0.010,
          baseline_mean_abs_error=0.010,
        ),
        lin_x=command,
      )
      for command in (-0.07, 0.0, 0.07)
    ]
    scenarios.extend([
      _scenario(
        "unsafe_extension",
        "extension",
        _stage1_metrics(
          candidate_terminated_event_rate=1.0,
          candidate_mean_abs_error=0.010,
          baseline_mean_abs_error=0.020,
        ),
        lin_x=0.10,
      ),
      _scenario("mismatch", "mismatch", _stage1_metrics(
        candidate_mean_abs_error=0.010,
        baseline_mean_abs_error=0.010,
      )),
      _scenario(
        "pushes",
        "disturbance",
        _stage1_metrics(
          candidate_recovery_time_s=1.0,
          baseline_recovery_time_s=1.0,
          candidate_post_kick_error_integral=0.02,
          baseline_post_kick_error_integral=0.02,
        ),
      ),
      _scenario(
        "transitions",
        "transition",
        _stage1_metrics(
          candidate_settling_time_s=1.0,
          baseline_settling_time_s=1.0,
          candidate_tracking_error_integral=0.10,
          baseline_tracking_error_integral=0.10,
          candidate_overshoot_abs_mean=0.02,
          baseline_overshoot_abs_mean=0.02,
        ),
      ),
    ])

    checks = evaluate_capability_suite("residual", scenarios)

    improvement = next(
      check for check in checks
      if check.name.startswith("hard_regime_fractional_improvement:")
    )
    self.assertFalse(improvement.passed)

  def test_stage1_safe_extension_can_supply_improvement_evidence(self):
    scenarios = [
      _scenario(
        f"nominal_{command:+.2f}",
        "nominal",
        _stage1_metrics(),
        lin_x=command,
      )
      for command in (-0.07, 0.0, 0.07)
    ]
    scenarios.extend([
      _scenario(
        "safe_extension",
        "extension",
        _stage1_metrics(
          candidate_mean_abs_error=0.005,
          baseline_mean_abs_error=0.020,
        ),
        lin_x=0.10,
      ),
      _scenario("mismatch", "mismatch", _stage1_metrics()),
      _scenario("pushes", "disturbance", _stage1_metrics()),
      _scenario("transitions", "transition", _stage1_metrics()),
    ])

    checks = evaluate_capability_suite("residual", scenarios)
    improvement = next(
      check for check in checks
      if check.name.startswith("hard_regime_fractional_improvement:")
    )

    self.assertTrue(improvement.passed)
    self.assertEqual(
      improvement.name,
      "hard_regime_fractional_improvement:extension:mean_abs_error",
    )

  def test_stage1_mismatch_rejects_tracking_regression(self):
    scenarios = [
      _scenario("nominal", "nominal", _stage1_metrics()),
      _scenario("extension", "extension", _stage1_metrics(), lin_x=0.10),
      _scenario(
        "mismatch",
        "mismatch",
        _stage1_metrics(
          candidate_mean_abs_error=0.020,
          baseline_mean_abs_error=0.010,
        ),
        lin_x=0.07,
      ),
      _scenario("pushes", "disturbance", _stage1_metrics(
        candidate_recovery_time_s=0.8,
        baseline_recovery_time_s=1.0,
      )),
      _scenario("transitions", "transition", _stage1_metrics(
        candidate_settling_time_s=0.8,
        baseline_settling_time_s=1.0,
      )),
    ]

    checks = evaluate_capability_suite("residual", scenarios)

    self.assertTrue(any(
      check.name == "mismatch_tracking_no_severe_regression"
      and not check.passed
      for check in checks
    ))

  def test_linear_zero_command_skips_undefined_speed_ratio(self):
    scenario = _scenario(
      "stand",
      "linear",
      _linear_metrics(signed_speed_ratio_mean=math.nan),
      lin_x=0.0,
    )

    checks = evaluate_capability_suite("linear", [scenario])

    self.assertTrue(all(check.passed for check in checks), checks)
    self.assertFalse(any(
      "signed_speed_ratio_mean" in check.name for check in checks
    ))

  def test_linear_rejects_policy_that_overrides_controller(self):
    scenario = _scenario(
      "forward",
      "linear",
      _linear_metrics(balance_residual_abs_mean=0.30001),
      lin_x=0.07,
    )

    checks = evaluate_capability_suite("linear", [scenario])

    self.assertTrue(any(
      check.name == "fixed_+0.070_balance_residual_abs_mean"
      and not check.passed
      for check in checks
    ))

  def test_planar_checks_linear_yaw_and_combination_scenarios(self):
    scenarios = [
      _scenario("linear", "linear", _linear_metrics(), lin_x=0.07),
      _scenario("yaw", "yaw", _yaw_metrics(), lin_x=0.0, yaw=0.10),
      _scenario("combo", "combo", _combo_metrics(), lin_x=0.07, yaw=0.10),
    ]

    self.assert_gate_passes("planar", scenarios)

  def test_posture_accepts_exact_boundaries(self):
    scenarios = [
      _scenario(
        "posture_center",
        "posture",
        {
          "height_rmse": 0.015,
          "pitch_rmse": 0.04,
          "non_wheel_contact_rate": 0.01,
          "terminated_event_rate": 0.01,
        },
        target_height=0.32,
        target_pitch=0.0,
      )
    ]

    self.assert_gate_passes("posture", scenarios)

  def test_integrated_requires_every_component_kind(self):
    scenarios = [
      _scenario("linear", "linear", _linear_metrics(), lin_x=0.07),
      _scenario("yaw", "yaw", _yaw_metrics(), lin_x=0.0, yaw=0.10),
      _scenario(
        "posture",
        "posture",
        {
          "height_rmse": 0.01,
          "pitch_rmse": 0.03,
          "non_wheel_contact_rate": 0.0,
          "terminated_event_rate": 0.0,
        },
      ),
      _scenario(
        "random",
        "random",
        _integrated_metrics(),
      ),
    ]

    checks = evaluate_capability_suite("integrated", scenarios)

    missing = [check for check in checks if check.name == "scenario_kind:combo"]
    self.assertEqual(len(missing), 1)
    self.assertFalse(missing[0].passed)

  def test_integrated_allows_a_stage4_tracking_reference_scenario(self):
    scenarios = [
      _scenario("linear", "linear", _linear_metrics(), lin_x=0.07),
      _scenario("yaw", "yaw", _yaw_metrics(), lin_x=0.0, yaw=0.10),
      _scenario("combo", "combo", _combo_metrics(), lin_x=0.07, yaw=0.10),
      _scenario(
        "posture",
        "posture",
        {
          "height_rmse": 0.01,
          "pitch_rmse": 0.03,
          "non_wheel_contact_rate": 0.0,
          "terminated_event_rate": 0.0,
        },
      ),
      _scenario(
        "integrated_reference",
        "reference",
        _integrated_metrics(terminated_event_rate=0.01),
      ),
      _scenario(
        "random",
        "random",
        _integrated_metrics(),
      ),
    ]

    self.assert_gate_passes("integrated", scenarios)

  def test_integrated_rejects_termination_in_fixed_joint_command(self):
    scenarios = [
      _scenario("linear", "linear", _linear_metrics(), lin_x=0.07),
      _scenario("yaw", "yaw", _yaw_metrics(), lin_x=0.0, yaw=0.10),
      _scenario("combo", "combo", _combo_metrics(), lin_x=0.07, yaw=0.10),
      _scenario(
        "posture",
        "posture",
        {
          "height_rmse": 0.01,
          "pitch_rmse": 0.03,
          "non_wheel_contact_rate": 0.0,
          "terminated_event_rate": 0.0,
        },
      ),
      _scenario(
        "integrated_reference",
        "reference",
        _integrated_metrics(terminated_event_rate=0.011),
      ),
      _scenario("random", "random", _integrated_metrics()),
    ]

    checks = evaluate_capability_suite("integrated", scenarios)

    reference_checks = [
      check for check in checks if check.scenario == "integrated_reference"
    ]
    self.assertTrue(any(
      check.name == "terminated_event_rate" and not check.passed
      for check in reference_checks
    ))

  def test_integrated_rejects_bad_tracking_without_termination(self):
    scenario = _scenario(
      "random_integrated",
      "random",
      _integrated_metrics(tracking_error=0.121),
    )

    checks = evaluate_capability_suite("integrated", [scenario])

    self.assertTrue(any(
      check.name == "tracking_error" and not check.passed
      for check in checks
    ))

  def test_robust_checks_survival_recovery_and_stage4_degradation(self):
    scenarios = [
      _scenario("linear", "linear", _linear_metrics(), lin_x=0.07),
      _scenario("yaw", "yaw", _yaw_metrics(), lin_x=0.0, yaw=0.10),
      _scenario("combo", "combo", _combo_metrics(), lin_x=0.07, yaw=0.10),
      _scenario(
        "posture",
        "posture",
        {
          "height_rmse": 0.015,
          "pitch_rmse": 0.04,
          "non_wheel_contact_rate": 0.01,
          "terminated_event_rate": 0.01,
        },
      ),
      _scenario(
        "pushes",
        "robust",
        {
          "survival_rate": 0.95,
          "recovery_time_s": 2.0,
          "tracking_error": 0.13,
          "terminated_event_rate": 0.05,
          "non_wheel_contact_rate": 0.02,
          "wheel_saturation_ratio": 0.20,
        },
      ),
      _scenario(
        "random",
        "random",
        _integrated_metrics(
          tracking_error=0.16,
          terminated_event_rate=0.05,
          survival_rate=0.95,
          recovery_time_s=2.0,
          non_wheel_contact_rate=0.02,
          wheel_saturation_ratio=0.20,
        ),
      ),
    ]

    self.assert_gate_passes(
      "robust",
      scenarios,
      stage4_reference={"tracking_error": 0.10},
    )

  def test_unknown_suite_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "Unknown capability suite"):
      evaluate_capability_suite("stage6", [])


class ResultEnvelopeTest(unittest.TestCase):
  def _result(self, seed: int, value: float, passed: bool = True) -> dict:
    scenarios = [
      _scenario(
        "stand",
        "controller",
        {
          "duration_s": 60.0,
          "terminated_event_rate": 0.0,
          "p95_pitch": value,
          "p99_pitch_rate": 0.4,
          "mean_actual_lin_x": 0.0,
        },
        lin_x=0.0,
      )
    ]
    checks = evaluate_capability_suite("controller", scenarios)
    if not passed:
      checks[0] = type(checks[0])(
        name=checks[0].name,
        value=checks[0].value,
        operator=checks[0].operator,
        limit=checks[0].limit,
        passed=False,
        scenario=checks[0].scenario,
      )
    return make_result_envelope(
      suite="controller",
      task="HopperTrex-Hybrid-v2-Stage0",
      git_sha="abc123",
      controller_gain_hash="gain456",
      calibration_hash="calibration789",
      seed=seed,
      checkpoint=None,
      scenarios=scenarios,
      checks=checks,
    )

  def test_envelope_contains_required_audit_fields(self):
    result = self._result(seed=3, value=0.03)

    self.assertEqual(result["task"], "HopperTrex-Hybrid-v2-Stage0")
    self.assertEqual(result["git_sha"], "abc123")
    self.assertEqual(result["controller_gain_hash"], "gain456")
    self.assertEqual(result["calibration_hash"], "calibration789")
    self.assertEqual(result["seed"], 3)
    self.assertIsNone(result["checkpoint"])
    self.assertIsNone(result["checkpoint_file_sha256"])
    self.assertEqual(result["evaluation_profile"], "formal")
    self.assertEqual(result["evaluation_source"], "live")
    self.assertIn("metrics", result)
    self.assertIn("checks", result)
    self.assertTrue(result["gate_pass"])

  def test_json_is_deterministic_and_newline_terminated(self):
    result = self._result(seed=3, value=0.03)

    first = to_deterministic_json(result)
    second = to_deterministic_json(result)

    self.assertEqual(first, second)
    self.assertTrue(first.endswith("\n"))
    self.assertEqual(json.loads(first)["seed"], 3)

  def test_three_seed_aggregation_reports_mean_std_and_all_seed_pass(self):
    results = [
      self._result(seed=1, value=0.02),
      self._result(seed=2, value=0.03),
      self._result(seed=3, value=0.04),
    ]

    aggregate = aggregate_seed_results(results)

    self.assertTrue(aggregate["gate_pass"])
    self.assertEqual(aggregate["pass_rate"], 1.0)
    self.assertEqual(aggregate["seeds"], [1, 2, 3])
    pitch = aggregate["metrics"]["stand"]["p95_pitch"]
    self.assertAlmostEqual(pitch["mean"], 0.03)
    self.assertAlmostEqual(pitch["std"], math.sqrt(2.0 / 3.0) * 0.01)
    self.assertEqual(pitch["min"], 0.02)
    self.assertEqual(pitch["max"], 0.04)

  def test_stage1_aggregation_requires_consistent_improvement_evidence(self):
    results = [
      self._result(seed=seed, value=0.02 + seed * 0.001)
      for seed in (1, 2, 3)
    ]
    evidence = (
      "disturbance:recovery_time_s",
      "transition:settling_time_s",
      "disturbance:recovery_time_s",
    )
    for result, name in zip(results, evidence):
      result["suite"] = "residual"
      result["task"] = "HopperTrex-Hybrid-v2-Stage1"
      result["checks"] = [{
        "name": f"hard_regime_fractional_improvement:{name}",
        "pass": True,
      }]
      result["gate_pass"] = True

    aggregate = aggregate_seed_results(results)

    self.assertFalse(aggregate["consistent_hard_improvement_evidence"])
    self.assertFalse(aggregate["gate_pass"])

    for result in results:
      result["checks"][0]["name"] = (
        "hard_regime_fractional_improvement:disturbance:recovery_time_s"
      )
    aggregate = aggregate_seed_results(results)
    self.assertTrue(aggregate["consistent_hard_improvement_evidence"])
    self.assertTrue(aggregate["gate_pass"])

  def test_stage1_aggregation_rejects_mismatch_profile_drift(self):
    results = [
      self._result(seed=seed, value=0.02 + seed * 0.001)
      for seed in (1, 2, 3)
    ]
    for result in results:
      result["suite"] = "residual"
      result["task"] = "HopperTrex-Hybrid-v2-Stage1"
      result["stage1_profile_version"] = "stage1b_speed010_mild_v1"
      result["mismatch_profile"] = {"wheel_radius_scale_range": [0.98, 1.02]}
      result["checks"] = [{
        "name": "hard_regime_fractional_improvement:disturbance:recovery_time_s",
        "pass": True,
      }]
      result["gate_pass"] = True
    results[2]["mismatch_profile"] = {"wheel_radius_scale_range": [0.97, 1.02]}

    with self.assertRaisesRegex(ValueError, "mismatch_profile"):
      aggregate_seed_results(results)

  def test_three_seed_aggregation_requires_exactly_three_unique_seeds(self):
    with self.assertRaisesRegex(ValueError, "exactly three unique seeds"):
      aggregate_seed_results(
        [
          self._result(seed=1, value=0.02),
          self._result(seed=1, value=0.03),
          self._result(seed=2, value=0.04),
        ]
      )

  def test_three_seed_aggregation_rejects_screen_results(self):
    results = [
      self._result(seed=seed, value=0.02)
      for seed in (1, 2, 3)
    ]
    results[0]["evaluation_profile"] = "screen"

    with self.assertRaisesRegex(ValueError, "rejects screen"):
      aggregate_seed_results(results)

  def test_legacy_unlabelled_results_are_grandfathered_only_for_stage0(self):
    results = [
      self._result(seed=seed, value=0.02)
      for seed in (1, 2, 3)
    ]
    for result in results:
      result["schema_version"] = 1
      result.pop("evaluation_profile")
    aggregate_seed_results(results)

    for result in results:
      result["suite"] = "residual"
    with self.assertRaisesRegex(ValueError, "frozen Stage0"):
      aggregate_seed_results(results)


class WheelActionAdapterTest(unittest.TestCase):
  def test_zero_where_masked_colocates_mask_with_value(self):
    mask = torch.tensor([True, False], device='cpu')
    value = torch.ones((2, 3), device='meta')

    result = zero_where_masked(mask, value)

    self.assertEqual(result.device.type, 'meta')
    self.assertEqual(result.shape, value.shape)

  def test_boolean_mask_moves_to_reference_device(self):
    mask = torch.tensor([True, False], device='cpu')
    reference = torch.empty(2, device='meta')

    moved = boolean_mask_on_device(mask, reference)

    self.assertEqual(moved.dtype, torch.bool)
    self.assertEqual(moved.device.type, 'meta')

  def test_saturation_threshold_uses_hybrid_wheel_limit(self):
    term = SimpleNamespace(
      cfg=SimpleNamespace(wheel_velocity_limit=12.0),
      _balance_scale=24.0,
    )

    self.assertAlmostEqual(wheel_target_saturation_threshold(term), 11.94)

  def test_saturation_threshold_falls_back_to_legacy_balance_scale(self):
    term = SimpleNamespace(cfg=SimpleNamespace(), _balance_scale=24.0)

    self.assertAlmostEqual(wheel_target_saturation_threshold(term), 23.88)

  def test_resolves_legacy_wheel_targets(self):
    target = object()
    raw = object()
    legacy = SimpleNamespace(_processed_actions=target, _raw_actions=raw)
    manager = SimpleNamespace(
      get_term=lambda name: {"wheel_balance": legacy}[name],
    )

    view = resolve_wheel_action(manager)

    self.assertEqual(view.term_name, "wheel_balance")
    self.assertIs(view.wheel_targets, target)
    self.assertIs(view.raw_actions, raw)
    self.assertIsNone(view.applied_residual)

  def test_resolves_hybrid_wheel_targets(self):
    target = object()
    raw = object()
    residual = object()
    hybrid = SimpleNamespace(
      wheel_targets=target,
      raw_action=raw,
      applied_residual=residual,
    )
    manager = SimpleNamespace(
      get_term=lambda name: {"hybrid_wheel_leg": hybrid}[name],
    )

    view = resolve_wheel_action(manager)

    self.assertEqual(view.term_name, "hybrid_wheel_leg")
    self.assertIs(view.wheel_targets, target)
    self.assertIs(view.raw_actions, raw)
    self.assertIs(view.applied_residual, residual)

  def test_skips_candidate_that_does_not_expose_wheel_targets(self):
    target = object()
    legacy = SimpleNamespace(_processed_actions=target, _raw_actions=None)
    unrelated = SimpleNamespace(_yaw_scale=2.5)
    manager = SimpleNamespace(
      get_term=lambda name: {
        "hybrid_wheel_leg": unrelated,
        "wheel_balance": legacy,
      }[name],
    )

    view = resolve_wheel_action(manager)

    self.assertEqual(view.term_name, "wheel_balance")
    self.assertIs(view.wheel_targets, target)


class HybridEvaluatorContractTest(unittest.TestCase):
  def test_live_checkpoint_requires_matching_training_revision(self):
    env_cfg = SimpleNamespace(
      actions={
        "hybrid_wheel_leg": SimpleNamespace(
          controller_gain_hash="controller123",
          calibration_hash="calibration123",
        )
      }
    )
    checkpoint = {
      "infos": {
        "hybrid_stage1_bootstrap": {
          "task": "HopperTrex-Hybrid-v2-Stage1",
          "stage": 1,
          "controller_gain_hash": "controller123",
          "calibration_hash": "calibration123",
          "action_order": [
            "wheel_balance_residual",
            "wheel_yaw_residual",
            "left_thigh_residual",
            "right_thigh_residual",
            "left_knee_residual",
            "right_knee_residual",
          ],
        },
        "hybrid_stage1_extension": {
          "target_profile_version": "stage1b_speed010_mild_v1",
          "source_action_std": [0.1] * 6,
          "collapsed_active_actions": [],
          "reset_collapsed_active_std": False,
        },
        "hybrid_training": {"git_sha": "abc123"},
      }
    }

    validate_hybrid_evaluation_checkpoint(
      "HopperTrex-Hybrid-v2-Stage1",
      env_cfg,
      checkpoint,
      evaluation_git_sha="abc123",
    )
    with self.assertRaisesRegex(ValueError, "training git SHA"):
      validate_hybrid_evaluation_checkpoint(
        "HopperTrex-Hybrid-v2-Stage1",
        env_cfg,
        checkpoint,
        evaluation_git_sha="different",
      )

  def test_stage_mapping_uses_six_capability_suites_and_registered_tasks(self):
    self.assertEqual(
      HYBRID_STAGE_SUITES,
      {
        0: "controller",
        1: "residual",
        2: "planar",
        3: "posture",
        4: "integrated",
        5: "robust",
      },
    )
    self.assertEqual(
      HYBRID_STAGE_TASKS[5],
      "HopperTrex-Hybrid-v2-Stage5",
    )

  def test_fixed_rows_are_wrapped_without_dropping_metrics(self):
    rows = [
      {"lin_x": -0.07, "mean_abs_error": 0.03},
      {"lin_x": 0.07, "mean_abs_error": 0.04},
    ]

    scenarios = _fixed_rows_to_scenarios("linear", rows)

    self.assertEqual(scenarios[0]["name"], "linear_vx_-0.070")
    self.assertEqual(scenarios[1]["name"], "linear_vx_+0.070")
    self.assertIs(scenarios[0]["metrics"], rows[0])

  def test_stage4_reference_is_extracted_from_auditable_envelope(self):
    envelope = {
      "metrics": {
        "integrated_reference": {
          "tracking_error": 0.08,
          "terminated_event_rate": 0.0,
        }
      }
    }

    reference = _extract_stage4_reference(envelope)

    self.assertEqual(reference, {"tracking_error": 0.08})

  def test_cli_keeps_remote_gate_defaults_explicit(self):
    args = parse_args(
      [
        "--stage",
        "2",
        "--checkpoint-file",
        "model.pt",
        "--seed",
        "3",
      ]
    )

    self.assertEqual(args.num_envs, 16)
    self.assertEqual(args.steps, 3000)
    self.assertEqual(args.warmup_steps, 300)
    self.assertEqual(args.window_steps, 800)
    self.assertEqual(args.seed, 3)

  def test_controller_hash_rejects_blank_explicit_value(self):
    with self.assertRaisesRegex(ValueError, "controller artifact"):
      _controller_hash("unused-task", "  ")

  def test_controller_hash_preserves_explicit_scenario_value(self):
    self.assertEqual(
      _controller_hash("unused-task", "gain456"),
      "gain456",
    )

  def test_controller_hash_rejects_unqualified_task_fallback(self):
    env_cfg = SimpleNamespace(
      actions={
        "hybrid_wheel_leg": SimpleNamespace(controller_gain_hash=None),
      }
    )
    module = (
      "hoppertrex_mjlab.scripts.rsl_rl.evaluate_hybrid_gate.load_env_cfg"
    )

    with patch(module, return_value=env_cfg):
      with self.assertRaisesRegex(ValueError, "controller artifact"):
        _controller_hash("HopperTrex-Hybrid-v2-Stage0", None)

  def test_live_controller_hash_must_match_qualified_environment(self):
    env_cfg = SimpleNamespace(
      actions={
        'hybrid_wheel_leg': SimpleNamespace(
          controller_qualified=True,
          controller_gain_hash='actual-hash',
        ),
      }
    )
    module = (
      'hoppertrex_mjlab.scripts.rsl_rl.evaluate_hybrid_gate.load_env_cfg'
    )

    with patch(module, return_value=env_cfg):
      with self.assertRaisesRegex(ValueError, 'does not match'):
        _controller_hash(
          'qualified-task',
          'claimed-hash',
          require_loaded_match=True,
        )

  def test_zero_velocity_is_linear_outside_controller_suite(self):
    row = {'lin_x': 0.0, 'mean_abs_error': 0.01}

    controller = _linear_row_to_scenario('controller', row)
    stage1 = _linear_row_to_scenario('linear', row)

    self.assertEqual(controller['kind'], 'controller')
    self.assertEqual(stage1['kind'], 'linear')
    self.assertEqual(stage1['name'], 'linear_vx_+0.000')

  def test_rollout_arguments_require_samples_after_warmup(self):
    args = parse_args(
      ["--stage", "1", "--steps", "300", "--warmup-steps", "300"]
    )

    with self.assertRaisesRegex(ValueError, "greater than --warmup-steps"):
      _validate_rollout_args(args)

  def test_formal_profile_rejects_short_rollout(self):
    args = parse_args(
      ["--stage", "2", "--steps", "1000", "--warmup-steps", "200"]
    )

    with self.assertRaisesRegex(ValueError, "Formal Hybrid gate"):
      _validate_rollout_args(args)

  def test_screen_profile_allows_short_non_stage1_rollout(self):
    args = parse_args([
      "--stage", "2",
      "--profile", "screen",
      "--steps", "1000",
      "--warmup-steps", "200",
    ])

    _validate_rollout_args(args)

  def test_formal_scenario_file_requires_declared_rollout_length(self):
    with self.assertRaisesRegex(ValueError, "rollout.steps"):
      _validate_scenario_file_profile("formal", None, {})

    _validate_scenario_file_profile(
      "formal",
      "formal",
      {"steps": 3000},
    )
    _validate_scenario_file_profile("screen", None, {})

  def test_stage1_rollout_requires_every_transition_segment(self):
    args = parse_args(
      [
        "--stage", "1",
        "--profile", "screen",
        "--steps", "999",
        "--warmup-steps", "300",
      ]
    )

    with self.assertRaisesRegex(ValueError, "at least 700 measured steps"):
      _validate_rollout_args(args)

  def test_formal_stage1_requires_32_environments(self):
    args = parse_args([
      "--stage", "1", "--num-envs", "16", "--steps", "3000",
    ])

    with self.assertRaisesRegex(ValueError, "--num-envs >= 32"):
      _validate_rollout_args(args)

  def test_event_integral_is_averaged_across_windows(self):
    error = torch.tensor([
      [1.0, 3.0],
      [1.0, 3.0],
      [2.0, 4.0],
      [2.0, 4.0],
    ])

    value = _mean_event_error_integral(error, [0, 2])

    self.assertAlmostEqual(value, 0.10)

  def test_settling_time_requires_consecutive_healthy_samples(self):
    healthy = torch.zeros((50, 1), dtype=torch.bool)
    healthy[5:15] = True
    healthy[15] = False
    healthy[16:] = True

    value = _settling_time_s(healthy, [0])

    self.assertAlmostEqual(value, 16 / 50)

  def test_live_coverage_rejects_missing_required_linear_command(self):
    scenarios = [
      {'name': f'linear_vx_{value:+.3f}', 'kind': 'linear'}
      for value in (-0.07, -0.04, 0.0, 0.07)
    ]

    with self.assertRaisesRegex(ValueError, 'linear_vx_\\+0.040'):
      _validate_live_scenario_coverage('linear', scenarios)

  def test_stage1_coverage_rejects_partial_ablation_profile(self):
    scenarios = [
      {"name": f"stage1_nominal_vx_{value:+.3f}", "kind": "nominal"}
      for value in (-0.07, 0.0, 0.07)
    ]
    scenarios.extend([
      {"name": "stage1_extension_vx_-0.100", "kind": "extension"},
      {"name": "stage1_disturbance_recovery", "kind": "disturbance"},
      {"name": "stage1_command_transition", "kind": "transition"},
    ])

    with self.assertRaisesRegex(
      ValueError,
      "stage1_extension_vx_\\+0.100.*stage1_model_mismatch",
    ):
      _validate_live_scenario_coverage("residual", scenarios)

  def test_posture_coverage_rejects_five_duplicate_targets(self):
    scenarios = [
      {
        "name": f"posture_{index}",
        "kind": "posture",
        "target_height": 0.30,
        "target_pitch": 0.02,
      }
      for index in range(5)
    ]

    with self.assertRaisesRegex(ValueError, "5 unique finite targets"):
      _validate_live_scenario_coverage("posture", scenarios)

  def test_posture_coverage_accepts_center_and_four_corners(self):
    points = [
      (0.30, 0.02),
      (0.29, 0.00),
      (0.29, 0.04),
      (0.31, 0.00),
      (0.31, 0.04),
    ]
    scenarios = [
      {
        "name": f"posture_{index}",
        "kind": "posture",
        "target_height": height,
        "target_pitch": pitch,
      }
      for index, (height, pitch) in enumerate(points)
    ]

    _validate_live_scenario_coverage("posture", scenarios)

  def test_survival_rate_counts_unique_terminated_environments(self):
    self.assertEqual(_survival_rate([True, False, True, False]), 0.5)


if __name__ == "__main__":
  unittest.main()
