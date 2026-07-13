"""Pure capability-gate decisions shared by legacy and Hybrid evaluators."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from statistics import fmean, pstdev
from typing import Any, Iterable, Mapping, Sequence

import torch


@dataclass(frozen=True)
class GateCheck:
  """One auditable threshold decision."""

  name: str
  value: float | bool
  operator: str
  limit: float | bool
  passed: bool
  scenario: str

  @property
  def detail(self) -> str:
    if isinstance(self.value, bool) or isinstance(self.limit, bool):
      return f"{self.name}={self.value!r} {self.operator} {self.limit!r}"
    return (
      f"{self.name}={float(self.value):.5f} "
      f"{self.operator} {float(self.limit):.5f}"
    )


@dataclass(frozen=True)
class WheelActionView:
  """Normalized access to legacy and Hybrid wheel action terms."""

  term_name: str
  term: Any
  wheel_targets: Any
  raw_actions: Any | None
  applied_residual: Any | None


LINEAR_RULES = (
  ("command_match_frac", ">=", 0.90),
  ("late_slow_env_frac", "<=", 0.10),
  ("late_wrong_direction_env_frac", "<=", 0.10),
  ("in_band_frac", ">=", 0.70),
  ("fast_frac", "<=", 0.25),
  ("late_in_band_frac", ">=", 0.80),
  ("target_band_frac", ">=", 0.70),
  ("late_target_band_frac", ">=", 0.80),
  ("signed_speed_ratio_mean", ">=", 0.75),
  ("signed_speed_ratio_mean", "<=", 1.25),
  ("lin_x_delta_rms", "<=", 0.035),
  ("lin_x_delta_abs_p95", "<=", 0.070),
  ("late_lin_x_delta_rms", "<=", 0.035),
  ("late_lin_x_delta_abs_p95", "<=", 0.070),
  ("mean_abs_error", "<=", 0.06),
  ("p95_pitch", "<=", 0.08),
  ("p99_pitch_rate", "<=", 0.90),
  ("terminated_event_rate", "<=", 0.01),
)

LINEAR_RESIDUAL_RULES = (
  ("balance_residual_abs_mean", "<=", 0.30),
  ("balance_residual_abs_p95", "<=", 0.45),
)

YAW_RULES = (
  ("command_match_frac", ">=", 0.90),
  ("late_slow_env_frac", "<=", 0.10),
  ("late_wrong_direction_env_frac", "<=", 0.10),
  ("late_lin_drift_env_frac", "<=", 0.10),
  ("in_band_frac", ">=", 0.70),
  ("fast_frac", "<=", 0.25),
  ("late_in_band_frac", ">=", 0.70),
  ("yaw_delta_rms", "<=", 0.035),
  ("yaw_delta_abs_p95", "<=", 0.080),
  ("late_yaw_delta_rms", "<=", 0.035),
  ("late_yaw_delta_abs_p95", "<=", 0.080),
  ("yaw_abs_error_mean", "<=", 0.07),
  ("yaw_abs_error_p90", "<=", 0.10),
  ("lin_drift_abs_mean", "<=", 0.05),
  ("p95_pitch", "<=", 0.10),
  ("p99_pitch_rate", "<=", 0.90),
  ("wheel_saturation_ratio", "<=", 0.20),
  ("terminated_event_rate", "<=", 0.01),
)

COMBO_RULES = (
  ("lin_command_match_frac", "lin_command_match_frac", ">=", 0.85),
  ("lin_wrong_direction_frac", "lin_wrong_direction_frac", "<=", 0.10),
  ("lin_in_band_frac", "lin_in_band_frac", ">=", 0.70),
  ("lin_fast_frac", "lin_fast_frac", "<=", 0.30),
  ("late_lin_in_band_frac", "late_lin_in_band_frac", ">=", 0.70),
  ("lin_abs_error_mean", "lin_abs_error_mean", "<=", 0.07),
  ("lin_abs_error_p90", "lin_abs_error_p90", "<=", 0.12),
  ("lin_x_delta_rms", "lin_x_delta_rms", "<=", 0.045),
  ("lin_x_delta_abs_p95", "lin_x_delta_abs_p95", "<=", 0.090),
  ("late_lin_x_delta_rms", "late_lin_x_delta_rms", "<=", 0.045),
  ("late_lin_x_delta_abs_p95", "late_lin_x_delta_abs_p95", "<=", 0.090),
  ("yaw_command_match_frac", "command_match_frac", ">=", 0.85),
  ("yaw_wrong_direction_frac", "wrong_direction_frac", "<=", 0.10),
  ("yaw_in_band_frac", "in_band_frac", ">=", 0.65),
  ("yaw_fast_frac", "fast_frac", "<=", 0.30),
  ("yaw_late_in_band_frac", "late_in_band_frac", ">=", 0.65),
  ("yaw_abs_error_mean", "yaw_abs_error_mean", "<=", 0.08),
  ("yaw_abs_error_p90", "yaw_abs_error_p90", "<=", 0.12),
  ("yaw_delta_rms", "yaw_delta_rms", "<=", 0.045),
  ("yaw_delta_abs_p95", "yaw_delta_abs_p95", "<=", 0.090),
  ("late_yaw_delta_rms", "late_yaw_delta_rms", "<=", 0.045),
  ("late_yaw_delta_abs_p95", "late_yaw_delta_abs_p95", "<=", 0.090),
  ("p95_pitch", "p95_pitch", "<=", 0.12),
  ("p99_pitch_rate", "p99_pitch_rate", "<=", 0.95),
  ("wheel_saturation_ratio", "wheel_saturation_ratio", "<=", 0.20),
  ("terminated_event_rate", "terminated_event_rate", "<=", 0.01),
)

POSTURE_RULES = (
  ("height_rmse", "<=", 0.015),
  ("pitch_rmse", "<=", 0.04),
  ("non_wheel_contact_rate", "<=", 0.01),
  ("terminated_event_rate", "<=", 0.01),
)

REQUIRED_SCENARIO_KINDS = {
  "controller": frozenset(("controller",)),
  "linear": frozenset(("linear",)),
  "residual": frozenset(("nominal", "boundary", "disturbance", "transition")),
  "planar": frozenset(("linear", "yaw", "combo")),
  "posture": frozenset(("posture",)),
  "integrated": frozenset(("linear", "yaw", "combo", "posture", "random")),
  "robust": frozenset(
    ("linear", "yaw", "combo", "posture", "random", "robust")
  ),
}

STAGE1_NOMINAL_RESIDUAL_RULES = (
  ("candidate_balance_residual_abs_mean", "<=", 0.10),
  ("candidate_balance_residual_abs_p95", "<=", 0.25),
)

INTEGRATED_RULES = (
  ("tracking_error", "<=", 0.12),
  ("terminated_event_rate", "<=", 0.01),
  ("survival_rate", ">=", 0.99),
  ("recovery_time_s", "<=", 2.0),
  ("non_wheel_contact_rate", "<=", 0.01),
  ("wheel_saturation_ratio", "<=", 0.20),
)

ROBUST_INTEGRATED_RULES = (
  ("tracking_error", "<=", 0.16),
  ("terminated_event_rate", "<=", 0.05),
  ("survival_rate", ">=", 0.95),
  ("recovery_time_s", "<=", 2.0),
  ("non_wheel_contact_rate", "<=", 0.02),
  ("wheel_saturation_ratio", "<=", 0.20),
)
STAGE1_SAFETY_RULES = (
  ("candidate_terminated_event_rate", "<=", 0.01),
  ("candidate_p95_pitch", "<=", 0.10),
  ("candidate_p99_pitch_rate", "<=", 0.90),
)


def _is_finite_number(value: object) -> bool:
  return (
    isinstance(value, (int, float))
    and not isinstance(value, bool)
    and math.isfinite(float(value))
  )


def boolean_mask_on_device(
  mask: torch.Tensor,
  reference: torch.Tensor,
) -> torch.Tensor:
  """Return a boolean mask colocated with the tensor it will index."""

  return mask.to(device=reference.device, dtype=torch.bool)


def wheel_target_saturation_threshold(action_term: Any) -> float:
  """Derive a near-limit threshold from the active wheel action config."""

  cfg = getattr(action_term, "cfg", None)
  configured_limit = getattr(cfg, "wheel_velocity_limit", None)
  if _is_finite_number(configured_limit) and float(configured_limit) > 0.0:
    return 0.995 * abs(float(configured_limit))
  balance_scale = getattr(action_term, "_balance_scale", None)
  if _is_finite_number(balance_scale) and float(balance_scale) > 0.0:
    return 0.995 * abs(float(balance_scale))
  return 23.9


def zero_where_masked(
  mask: torch.Tensor,
  value: torch.Tensor,
) -> torch.Tensor:
  """Zero rows of a tensor using a mask colocated with that tensor."""

  condition = boolean_mask_on_device(mask, value)
  while condition.ndim < value.ndim:
    condition = condition.unsqueeze(-1)
  return torch.where(condition, torch.zeros_like(value), value)


def metric_check(
  metrics: Mapping[str, object],
  metric_name: str,
  operator: str,
  limit: float,
  *,
  scenario: str,
  check_name: str | None = None,
) -> GateCheck:
  """Evaluate one inclusive or strict numeric threshold."""

  raw_value = metrics.get(metric_name, math.nan)
  value = float(raw_value) if _is_finite_number(raw_value) else math.nan
  passed = False
  if math.isfinite(value):
    if operator == "<=":
      passed = value <= limit
    elif operator == ">=":
      passed = value >= limit
    elif operator == "<":
      passed = value < limit
    else:
      raise ValueError(f"Unsupported gate operator: {operator}")
  return GateCheck(
    name=check_name or metric_name,
    value=value,
    operator=operator,
    limit=float(limit),
    passed=passed,
    scenario=scenario,
  )


def _presence_check(kind: str, present: bool) -> GateCheck:
  return GateCheck(
    name=f"scenario_kind:{kind}",
    value=present,
    operator="==",
    limit=True,
    passed=present,
    scenario="suite",
  )


def _scenario_metrics(scenario: Mapping[str, object]) -> Mapping[str, object]:
  metrics = scenario.get("metrics")
  if not isinstance(metrics, Mapping):
    raise ValueError("Every capability scenario must contain a metrics object.")
  return metrics


def _scenario_name(scenario: Mapping[str, object]) -> str:
  name = scenario.get("name")
  if not isinstance(name, str) or not name:
    raise ValueError("Every capability scenario must have a non-empty name.")
  return name


def _linear_prefix(scenario: Mapping[str, object]) -> str:
  return f"fixed_{float(scenario.get('lin_x', 0.0)):+.3f}_"


def _yaw_prefix(scenario: Mapping[str, object]) -> str:
  return f"fixed_yaw_{float(scenario.get('yaw', 0.0)):+.3f}_"


def _combo_prefix(scenario: Mapping[str, object]) -> str:
  return (
    f"combo_{float(scenario.get('lin_x', 0.0)):+.3f}_"
    f"{float(scenario.get('yaw', 0.0)):+.3f}_"
  )


def linear_scenario_checks(
  scenario: Mapping[str, object],
) -> list[GateCheck]:
  metrics = _scenario_metrics(scenario)
  name = _scenario_name(scenario)
  prefix = _linear_prefix(scenario)
  lin_x = float(scenario.get("lin_x", 0.0))
  rules = (
    rule
    for rule in LINEAR_RULES
    if not (
      abs(lin_x) <= 1.0e-12
      and rule[0] == "signed_speed_ratio_mean"
    )
  )
  checks = [
    metric_check(
      metrics,
      metric,
      operator,
      limit,
      scenario=name,
      check_name=f"{prefix}{metric}",
    )
    for metric, operator, limit in rules
  ]
  if all(metric in metrics for metric, _operator, _limit in LINEAR_RESIDUAL_RULES):
    checks.extend(
      metric_check(
        metrics,
        metric,
        operator,
        limit,
        scenario=name,
        check_name=f"{prefix}{metric}",
      )
      for metric, operator, limit in LINEAR_RESIDUAL_RULES
    )
  return checks


def _relative_upper_check(
  metrics: Mapping[str, object],
  candidate_metric: str,
  baseline_metric: str,
  *,
  ratio: float,
  absolute_tolerance: float,
  scenario: str,
  check_name: str,
) -> GateCheck:
  candidate_raw = metrics.get(candidate_metric, math.nan)
  baseline_raw = metrics.get(baseline_metric, math.nan)
  candidate = (
    float(candidate_raw) if _is_finite_number(candidate_raw) else math.nan
  )
  baseline = (
    float(baseline_raw) if _is_finite_number(baseline_raw) else math.nan
  )
  limit = (
    ratio * baseline + absolute_tolerance
    if math.isfinite(baseline)
    else math.nan
  )
  return GateCheck(
    name=check_name,
    value=candidate,
    operator="<=",
    limit=limit,
    passed=(
      math.isfinite(candidate)
      and math.isfinite(limit)
      and candidate <= limit
    ),
    scenario=scenario,
  )


def _fractional_improvement(
  metrics: Mapping[str, object],
  candidate_metric: str,
  baseline_metric: str,
  *,
  floor: float,
) -> float:
  candidate_raw = metrics.get(candidate_metric, math.nan)
  baseline_raw = metrics.get(baseline_metric, math.nan)
  if not _is_finite_number(candidate_raw) or not _is_finite_number(baseline_raw):
    return math.nan
  candidate = float(candidate_raw)
  baseline = float(baseline_raw)
  return (baseline - candidate) / max(abs(baseline), floor)


def residual_scenario_checks(
  scenario: Mapping[str, object],
) -> list[GateCheck]:
  """Evaluate Stage1 as an ablation against the zero-residual LQR."""

  metrics = _scenario_metrics(scenario)
  name = _scenario_name(scenario)
  kind = str(scenario.get("kind"))
  checks: list[GateCheck] = []

  if kind != "boundary":
    checks.extend(
      metric_check(metrics, metric, operator, limit, scenario=name)
      for metric, operator, limit in STAGE1_SAFETY_RULES
    )

  if kind == "nominal":
    checks.extend(
      metric_check(metrics, metric, operator, limit, scenario=name)
      for metric, operator, limit in STAGE1_NOMINAL_RESIDUAL_RULES
    )
    for candidate_metric, baseline_metric, tolerance, label in (
      (
        "candidate_mean_abs_error",
        "baseline_mean_abs_error",
        0.001,
        "nominal_tracking_no_regression",
      ),
      (
        "candidate_p95_pitch",
        "baseline_p95_pitch",
        0.005,
        "nominal_pitch_no_regression",
      ),
      (
        "candidate_lin_x_delta_rms",
        "baseline_lin_x_delta_rms",
        0.002,
        "nominal_oscillation_no_regression",
      ),
    ):
      checks.append(_relative_upper_check(
        metrics,
        candidate_metric,
        baseline_metric,
        ratio=1.10,
        absolute_tolerance=tolerance,
        scenario=name,
        check_name=label,
      ))
  elif kind == "disturbance":
    for candidate_metric, baseline_metric, tolerance, label in (
      (
        "candidate_recovery_time_s",
        "baseline_recovery_time_s",
        0.10,
        "disturbance_recovery_no_severe_regression",
      ),
      (
        "candidate_post_kick_error_integral",
        "baseline_post_kick_error_integral",
        0.002,
        "disturbance_error_no_severe_regression",
      ),
    ):
      checks.append(_relative_upper_check(
        metrics,
        candidate_metric,
        baseline_metric,
        ratio=1.15,
        absolute_tolerance=tolerance,
        scenario=name,
        check_name=label,
      ))
  elif kind == "transition":
    for candidate_metric, baseline_metric, tolerance, label in (
      (
        "candidate_settling_time_s",
        "baseline_settling_time_s",
        0.10,
        "transition_settling_no_severe_regression",
      ),
      (
        "candidate_tracking_error_integral",
        "baseline_tracking_error_integral",
        0.002,
        "transition_error_no_severe_regression",
      ),
      (
        "candidate_overshoot_abs_mean",
        "baseline_overshoot_abs_mean",
        0.002,
        "transition_overshoot_no_severe_regression",
      ),
    ):
      checks.append(_relative_upper_check(
        metrics,
        candidate_metric,
        baseline_metric,
        ratio=1.15,
        absolute_tolerance=tolerance,
        scenario=name,
        check_name=label,
      ))
  elif kind != "boundary":
    checks.append(_presence_check(f"stage1_supported:{kind}", False))
  return checks


def _stage1_improvement_check(
  scenarios: Sequence[Mapping[str, object]],
) -> GateCheck:
  improvements: list[tuple[float, str]] = []
  for scenario in scenarios:
    metrics = _scenario_metrics(scenario)
    kind = str(scenario.get("kind"))
    metric_pairs: tuple[tuple[str, str, float], ...] = ()
    if kind == "boundary":
      candidate_terminated = metrics.get(
        "candidate_terminated_event_rate", math.nan
      )
      candidate_pitch = metrics.get("candidate_p95_pitch", math.nan)
      boundary_safe_enough_for_evidence = (
        _is_finite_number(candidate_terminated)
        and float(candidate_terminated) <= 0.01
        and _is_finite_number(candidate_pitch)
        and float(candidate_pitch) <= 0.10
      )
      if boundary_safe_enough_for_evidence:
        metric_pairs = (
          ("candidate_mean_abs_error", "baseline_mean_abs_error", 0.005),
        )
    elif kind == "disturbance":
      metric_pairs = (
        ("candidate_recovery_time_s", "baseline_recovery_time_s", 0.10),
        (
          "candidate_post_kick_error_integral",
          "baseline_post_kick_error_integral",
          0.005,
        ),
      )
    elif kind == "transition":
      metric_pairs = (
        ("candidate_settling_time_s", "baseline_settling_time_s", 0.10),
        (
          "candidate_tracking_error_integral",
          "baseline_tracking_error_integral",
          0.005,
        ),
        (
          "candidate_overshoot_abs_mean",
          "baseline_overshoot_abs_mean",
          0.005,
        ),
      )
    improvements.extend(
      (
        _fractional_improvement(
          metrics,
          candidate_metric,
          baseline_metric,
          floor=floor,
        ),
        f"{kind}:{candidate_metric.removeprefix('candidate_')}",
      )
      for candidate_metric, baseline_metric, floor in metric_pairs
    )
  finite = [item for item in improvements if math.isfinite(item[0])]
  best, evidence = max(finite, default=(math.nan, "none"), key=lambda item: item[0])
  return GateCheck(
    name=f"hard_regime_fractional_improvement:{evidence}",
    value=best,
    operator=">=",
    limit=0.10,
    passed=math.isfinite(best) and best >= 0.10,
    scenario="stage1_ablation",
  )


def yaw_scenario_checks(
  scenario: Mapping[str, object],
) -> list[GateCheck]:
  metrics = _scenario_metrics(scenario)
  name = _scenario_name(scenario)
  prefix = _yaw_prefix(scenario)
  checks = [
    metric_check(
      metrics,
      metric,
      operator,
      limit,
      scenario=name,
      check_name=f"{prefix}{metric}",
    )
    for metric, operator, limit in YAW_RULES[:1]
  ]
  yaw = float(scenario.get("yaw", 0.0))
  actual = metrics.get("mean_actual_yaw", math.nan)
  signed_actual = (
    (1.0 if yaw >= 0.0 else -1.0) * float(actual)
    if _is_finite_number(actual)
    else math.nan
  )
  checks.append(
    metric_check(
      {"value": signed_actual},
      "value",
      ">=",
      0.5 * abs(yaw),
      scenario=name,
      check_name=f"{prefix}signed_mean_actual_yaw",
    )
  )
  checks.extend(
    metric_check(
      metrics,
      metric,
      operator,
      limit,
      scenario=name,
      check_name=f"{prefix}{metric}",
    )
    for metric, operator, limit in YAW_RULES[1:]
  )
  return checks


def combo_scenario_checks(
  scenario: Mapping[str, object],
) -> list[GateCheck]:
  metrics = _scenario_metrics(scenario)
  name = _scenario_name(scenario)
  prefix = _combo_prefix(scenario)
  return [
    metric_check(
      metrics,
      source_metric,
      operator,
      limit,
      scenario=name,
      check_name=f"{prefix}{check_name}",
    )
    for check_name, source_metric, operator, limit in COMBO_RULES
  ]


def _controller_scenario_checks(
  scenario: Mapping[str, object],
) -> list[GateCheck]:
  metrics = _scenario_metrics(scenario)
  name = _scenario_name(scenario)
  checks = [
    metric_check(metrics, metric, operator, limit, scenario=name)
    for metric, operator, limit in (
      ("duration_s", ">=", 60.0),
      ("terminated_event_rate", "<=", 0.01),
      ("p95_pitch", "<=", 0.08),
      ("p99_pitch_rate", "<=", 0.90),
    )
  ]
  if abs(float(scenario.get("lin_x", 0.0))) > 1.0e-12:
    checks.extend(
      (
        metric_check(
          metrics,
          "signed_speed_ratio_mean",
          ">=",
          0.75,
          scenario=name,
        ),
        metric_check(
          metrics,
          "signed_speed_ratio_mean",
          "<=",
          1.25,
          scenario=name,
        ),
      )
      )
  else:
    raw_velocity = metrics.get("mean_actual_lin_x", math.nan)
    stand_velocity = (
      abs(float(raw_velocity)) if _is_finite_number(raw_velocity) else math.nan
    )
    checks.append(GateCheck(
      name="mean_abs_stand_velocity",
      value=stand_velocity,
      operator="<=",
      limit=0.01,
      passed=math.isfinite(stand_velocity) and stand_velocity <= 0.01,
      scenario=name,
    ))
  return checks


def _posture_scenario_checks(
  scenario: Mapping[str, object],
) -> list[GateCheck]:
  metrics = _scenario_metrics(scenario)
  name = _scenario_name(scenario)
  return [
    metric_check(metrics, metric, operator, limit, scenario=name)
    for metric, operator, limit in POSTURE_RULES
  ]


def _integrated_scenario_checks(
  scenario: Mapping[str, object],
  *,
  robust: bool,
) -> list[GateCheck]:
  metrics = _scenario_metrics(scenario)
  name = _scenario_name(scenario)
  rules = ROBUST_INTEGRATED_RULES if robust else INTEGRATED_RULES
  return [
    metric_check(metrics, metric, operator, limit, scenario=name)
    for metric, operator, limit in rules
  ]


def _robust_scenario_checks(
  scenario: Mapping[str, object],
  stage4_reference: Mapping[str, float] | None,
) -> list[GateCheck]:
  metrics = _scenario_metrics(scenario)
  name = _scenario_name(scenario)
  checks = [
    metric_check(metrics, "survival_rate", ">=", 0.95, scenario=name),
    metric_check(metrics, "recovery_time_s", "<=", 2.0, scenario=name),
    metric_check(metrics, "terminated_event_rate", "<=", 0.05, scenario=name),
    metric_check(metrics, "non_wheel_contact_rate", "<=", 0.02, scenario=name),
    metric_check(metrics, "wheel_saturation_ratio", "<=", 0.20, scenario=name),
  ]
  reference = None if stage4_reference is None else stage4_reference.get("tracking_error")
  limit = (
    1.30 * float(reference)
    if _is_finite_number(reference)
    else math.nan
  )
  checks.append(
    metric_check(
      metrics,
      "tracking_error",
      "<=",
      limit,
      scenario=name,
      check_name="tracking_error_vs_stage4",
    )
  )
  return checks


def evaluate_capability_suite(
  suite: str,
  scenarios: Sequence[Mapping[str, object]],
  *,
  stage4_reference: Mapping[str, float] | None = None,
) -> list[GateCheck]:
  """Evaluate one capability suite from structured scenario metrics."""

  if suite not in REQUIRED_SCENARIO_KINDS:
    raise ValueError(f"Unknown capability suite: {suite}")
  kinds = {
    str(scenario.get("kind"))
    for scenario in scenarios
  }
  checks = [
    _presence_check(kind, kind in kinds)
    for kind in sorted(REQUIRED_SCENARIO_KINDS[suite])
  ]
  for scenario in scenarios:
    kind = str(scenario.get("kind"))
    if kind == "controller":
      checks.extend(_controller_scenario_checks(scenario))
    elif kind == "linear":
      checks.extend(linear_scenario_checks(scenario))
    elif kind in ("nominal", "boundary", "disturbance", "transition"):
      if suite == "residual":
        checks.extend(residual_scenario_checks(scenario))
      else:
        checks.append(_presence_check(f"supported:{kind}", False))
    elif kind == "yaw":
      checks.extend(yaw_scenario_checks(scenario))
    elif kind == "combo":
      checks.extend(combo_scenario_checks(scenario))
    elif kind == "posture":
      checks.extend(_posture_scenario_checks(scenario))
    elif kind == "robust":
      checks.extend(_robust_scenario_checks(scenario, stage4_reference))
    elif kind == "random":
      checks.extend(
        _integrated_scenario_checks(
          scenario,
          robust=suite == "robust",
        )
      )
    elif kind == "reference":
      checks.extend(_integrated_scenario_checks(scenario, robust=False))
    else:
      checks.append(_presence_check(f"supported:{kind}", False))
  if suite == "residual":
    checks.append(_stage1_improvement_check(scenarios))
  return checks


def resolve_wheel_action(action_manager: Any) -> WheelActionView:
  """Resolve wheel targets without coupling collectors to one action-term name."""

  errors: list[Exception] = []
  for term_name in ("hybrid_wheel_leg", "wheel_balance"):
    try:
      term = action_manager.get_term(term_name)
    except (KeyError, ValueError, AttributeError) as exc:
      errors.append(exc)
      continue
    wheel_targets = getattr(term, "wheel_targets", None)
    if wheel_targets is None:
      wheel_targets = getattr(term, "_processed_actions", None)
    if wheel_targets is None:
      errors.append(
        AttributeError(f"Action term '{term_name}' does not expose wheel targets.")
      )
      continue
    raw_actions = getattr(term, "raw_action", None)
    if raw_actions is None:
      raw_actions = getattr(term, "_raw_actions", None)
    applied_residual = getattr(term, "applied_residual", None)
    return WheelActionView(
      term_name=term_name,
      term=term,
      wheel_targets=wheel_targets,
      raw_actions=raw_actions,
      applied_residual=applied_residual,
    )
  raise KeyError("No 'hybrid_wheel_leg' or 'wheel_balance' action term found.") from (
    errors[-1] if errors else None
  )


def _checks_payload(checks: Iterable[GateCheck]) -> list[dict[str, object]]:
  return [
    {
      "name": check.name,
      "value": check.value,
      "operator": check.operator,
      "limit": check.limit,
      "pass": check.passed,
      "scenario": check.scenario,
      "detail": check.detail,
    }
    for check in checks
  ]


def _metrics_payload(
  scenarios: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, float]]:
  payload: dict[str, dict[str, float]] = {}
  for scenario in scenarios:
    name = _scenario_name(scenario)
    if name in payload:
      raise ValueError(f"Duplicate scenario name: {name}")
    metrics = _scenario_metrics(scenario)
    payload[name] = {
      str(key): float(value)
      for key, value in metrics.items()
      if _is_finite_number(value)
    }
  return payload


def make_result_envelope(
  *,
  suite: str,
  task: str,
  git_sha: str,
  controller_gain_hash: str | None,
  calibration_hash: str | None = None,
  seed: int,
  checkpoint: str | None,
  scenarios: Sequence[Mapping[str, object]],
  checks: Sequence[GateCheck],
  evaluation_profile: str = "formal",
  evaluation_source: str = "live",
  rollout: Mapping[str, object] | None = None,
) -> dict[str, object]:
  """Build one versioned, auditable gate result."""

  return {
    "schema_version": 2,
    "suite": suite,
    "task": task,
    "git_sha": git_sha,
    "controller_gain_hash": controller_gain_hash,
    "calibration_hash": calibration_hash,
    "seed": int(seed),
    "checkpoint": checkpoint,
    "evaluation_profile": evaluation_profile,
    "evaluation_source": evaluation_source,
    "rollout": dict(rollout or {}),
    "gate_pass": all(check.passed for check in checks),
    "metrics": _metrics_payload(scenarios),
    "checks": _checks_payload(checks),
  }


def _json_safe(value: object) -> object:
  if isinstance(value, float) and not math.isfinite(value):
    return None
  if isinstance(value, dict):
    return {str(key): _json_safe(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_safe(item) for item in value]
  return value


def to_deterministic_json(result: Mapping[str, object]) -> str:
  """Serialize standard JSON with stable ordering and one trailing newline."""

  return json.dumps(
    _json_safe(dict(result)),
    indent=2,
    sort_keys=True,
    allow_nan=False,
  ) + "\n"


def aggregate_seed_results(
  results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
  """Aggregate exactly three comparable seed envelopes."""

  seeds = [int(result["seed"]) for result in results]
  if len(results) != 3 or len(set(seeds)) != 3:
    raise ValueError("Seed aggregation requires exactly three unique seeds.")
  ordered = sorted(results, key=lambda result: int(result["seed"]))
  if any(result.get("evaluation_profile") == "screen" for result in ordered):
    raise ValueError("Formal seed aggregation rejects screen gate envelopes.")
  for result in ordered:
    schema_version = result.get("schema_version")
    profile = result.get("evaluation_profile")
    suite = result.get("suite")
    if schema_version == 2 and profile != "formal":
      raise ValueError("Schema-v2 seed envelopes must declare profile formal.")
    if schema_version == 1 and suite != "controller":
      raise ValueError(
        "Unlabelled schema-v1 envelopes are accepted only for the frozen "
        "Stage0 controller gate."
      )
  seeds = [int(result["seed"]) for result in ordered]

  for key in ("suite", "task", "git_sha", "controller_gain_hash", "calibration_hash"):
    values = {result.get(key) for result in ordered}
    if len(values) != 1:
      raise ValueError(f"Seed results disagree on {key}.")

  metric_maps = [result.get("metrics") for result in ordered]
  if not all(isinstance(metrics, Mapping) for metrics in metric_maps):
    raise ValueError("Every seed result must contain a metrics object.")
  typed_maps = [metrics for metrics in metric_maps if isinstance(metrics, Mapping)]
  common_scenarios = set(typed_maps[0])
  for metrics in typed_maps[1:]:
    common_scenarios.intersection_update(metrics)

  aggregate_metrics: dict[str, dict[str, dict[str, float]]] = {}
  for scenario in sorted(common_scenarios):
    scenario_metrics = [metrics[scenario] for metrics in typed_maps]
    if not all(isinstance(metrics, Mapping) for metrics in scenario_metrics):
      continue
    typed_scenario_metrics = [
      metrics for metrics in scenario_metrics if isinstance(metrics, Mapping)
    ]
    common_names = set(typed_scenario_metrics[0])
    for metrics in typed_scenario_metrics[1:]:
      common_names.intersection_update(metrics)
    aggregate_metrics[scenario] = {}
    for name in sorted(common_names):
      values = [metrics[name] for metrics in typed_scenario_metrics]
      if not all(_is_finite_number(value) for value in values):
        continue
      numeric = [float(value) for value in values]
      aggregate_metrics[scenario][name] = {
        "mean": fmean(numeric),
        "std": pstdev(numeric),
        "min": min(numeric),
        "max": max(numeric),
      }

  return {
    "schema_version": 2,
    "suite": ordered[0]["suite"],
    "task": ordered[0]["task"],
    "git_sha": ordered[0]["git_sha"],
    "controller_gain_hash": ordered[0].get("controller_gain_hash"),
    "calibration_hash": ordered[0].get("calibration_hash"),
    "evaluation_profile": "formal",
    "seeds": seeds,
    "checkpoints": [result.get("checkpoint") for result in ordered],
    "gate_pass": all(bool(result.get("gate_pass")) for result in ordered),
    "pass_rate": (
      sum(bool(result.get("gate_pass")) for result in ordered) / len(ordered)
    ),
    "metrics": aggregate_metrics,
    "seed_results": [
      {"seed": int(result["seed"]), "gate_pass": bool(result.get("gate_pass"))}
      for result in ordered
    ],
  }


__all__ = [
  "COMBO_RULES",
  "GateCheck",
  "INTEGRATED_RULES",
  "LINEAR_RULES",
  "STAGE1_NOMINAL_RESIDUAL_RULES",
  "STAGE1_SAFETY_RULES",
  "ROBUST_INTEGRATED_RULES",
  "POSTURE_RULES",
  "WheelActionView",
  "YAW_RULES",
  "aggregate_seed_results",
  "boolean_mask_on_device",
  "combo_scenario_checks",
  "evaluate_capability_suite",
  "linear_scenario_checks",
  "make_result_envelope",
  "metric_check",
  "resolve_wheel_action",
  "to_deterministic_json",
  "wheel_target_saturation_threshold",
  "zero_where_masked",
  "yaw_scenario_checks",
]
