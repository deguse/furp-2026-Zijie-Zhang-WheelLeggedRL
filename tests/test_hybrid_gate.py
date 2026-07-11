import json
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hoppertrex_mjlab.scripts.rsl_rl.hybrid_gate import (
  aggregate_seed_results,
  evaluate_capability_suite,
  make_result_envelope,
  resolve_wheel_action,
  to_deterministic_json,
)
from hoppertrex_mjlab.scripts.rsl_rl.evaluate_hybrid_gate import (
  HYBRID_STAGE_SUITES,
  HYBRID_STAGE_TASKS,
  _controller_hash,
  _extract_stage4_reference,
  _fixed_rows_to_scenarios,
  _linear_row_to_scenario,
  _survival_rate,
  _validate_live_scenario_coverage,
  _validate_rollout_args,
  parse_args,
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
        {"terminated_event_rate": 0.0},
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
        {
          "tracking_error": 0.08,
          "terminated_event_rate": 0.01,
        },
      ),
      _scenario(
        "random",
        "random",
        {"terminated_event_rate": 0.0},
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
        {
          "tracking_error": 0.08,
          "terminated_event_rate": 0.011,
        },
      ),
      _scenario("random", "random", {"terminated_event_rate": 0.0}),
    ]

    checks = evaluate_capability_suite("integrated", scenarios)

    reference_checks = [
      check for check in checks if check.scenario == "integrated_reference"
    ]
    self.assertEqual(len(reference_checks), 1)
    self.assertFalse(reference_checks[0].passed)

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
        },
      ),
      _scenario(
        "random",
        "random",
        {"terminated_event_rate": 0.01},
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
    self.assertEqual(result["seed"], 3)
    self.assertIsNone(result["checkpoint"])
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
    self.assertEqual(aggregate["seeds"], [1, 2, 3])
    pitch = aggregate["metrics"]["stand"]["p95_pitch"]
    self.assertAlmostEqual(pitch["mean"], 0.03)
    self.assertAlmostEqual(pitch["std"], math.sqrt(2.0 / 3.0) * 0.01)

  def test_three_seed_aggregation_requires_exactly_three_unique_seeds(self):
    with self.assertRaisesRegex(ValueError, "exactly three unique seeds"):
      aggregate_seed_results(
        [
          self._result(seed=1, value=0.02),
          self._result(seed=1, value=0.03),
          self._result(seed=2, value=0.04),
        ]
      )


class WheelActionAdapterTest(unittest.TestCase):
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

  def test_resolves_hybrid_wheel_targets(self):
    target = object()
    raw = object()
    hybrid = SimpleNamespace(wheel_targets=target, raw_action=raw)
    manager = SimpleNamespace(
      get_term=lambda name: {"hybrid_wheel_leg": hybrid}[name],
    )

    view = resolve_wheel_action(manager)

    self.assertEqual(view.term_name, "hybrid_wheel_leg")
    self.assertIs(view.wheel_targets, target)
    self.assertIs(view.raw_actions, raw)

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
  def test_stage_mapping_uses_six_capability_suites_and_registered_tasks(self):
    self.assertEqual(
      HYBRID_STAGE_SUITES,
      {
        0: "controller",
        1: "linear",
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

  def test_live_coverage_rejects_missing_required_linear_command(self):
    scenarios = [
      {'name': f'linear_vx_{value:+.3f}', 'kind': 'linear'}
      for value in (-0.07, -0.04, 0.0, 0.07)
    ]

    with self.assertRaisesRegex(ValueError, 'linear_vx_\\+0.040'):
      _validate_live_scenario_coverage('linear', scenarios)

  def test_survival_rate_counts_unique_terminated_environments(self):
    self.assertEqual(_survival_rate([True, False, True, False]), 0.5)


if __name__ == "__main__":
  unittest.main()
