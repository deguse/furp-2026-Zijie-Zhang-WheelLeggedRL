"""Pure capability-gate decisions shared by legacy and Hybrid evaluators."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from statistics import fmean, pstdev
from typing import Any, Iterable, Mapping, NamedTuple, Sequence

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
  source: str = field(default="", compare=False)

  @property
  def detail(self) -> str:
    if isinstance(self.value, bool) or isinstance(self.limit, bool):
      return f"{self.name}={self.value!r} {self.operator} {self.limit!r}"
    return (
      f"{self.name}={float(self.value):.5f} "
      f"{self.operator} {float(self.limit):.5f}"
    )


class Rule(NamedTuple):
  """One threshold with its evidential provenance (GateManifest v1)."""

  metric: str
  op: str
  limit: float
  source: str


class ComboRule(NamedTuple):
  """A combo-scenario threshold whose check name differs from its metric."""

  name: str
  metric: str
  op: str
  limit: float
  source: str


@dataclass(frozen=True)
class WheelActionView:
  """Normalized access to legacy and Hybrid wheel action terms."""

  term_name: str
  term: Any
  wheel_targets: Any
  raw_actions: Any | None
  applied_residual: Any | None


# GateManifest v1 sources. Every threshold names where its number came from;
# "uncalibrated" limits predate the probe discipline and are recalibrated
# only when a measured noise floor exists for them.
LEGACY_SOURCE = "018efe1 authored, uncalibrated"
YAW_PROBE_SOURCE = (
  "local CPU probe 2026-07-15 (qualified LQR, zero falls), preliminary"
)
COMBO_SCALED_SOURCE = (
  "scaled 2026-07-15 from the yaw-suite recalibration, preliminary"
)
GPU_BASELINE_SOURCE = (
  "GPU zero-residual controller baseline 2026-07-15 "
  "(controller_stand command_match 0.793 at e8e2f06)"
)
# Pre-registered residual-value protocol (Hybrid v3): screen profiles run
# rejection checks only; the improvement claim is made once, on the formal
# profile, against this single primary metric with a minimum event count.
IMPROVEMENT_PROTOCOL_SOURCE = (
  "pre-registered 2026-07-15: primary metric disturbance:recovery_time_s, "
  "formal profile only, >=16 kick events"
)
MIN_IMPROVEMENT_KICK_EVENTS = 16

LINEAR_RULES = (
  Rule("command_match_frac", ">=", 0.90, LEGACY_SOURCE),
  Rule("late_slow_env_frac", "<=", 0.10, LEGACY_SOURCE),
  Rule("late_wrong_direction_env_frac", "<=", 0.10, LEGACY_SOURCE),
  Rule("in_band_frac", ">=", 0.70, LEGACY_SOURCE),
  Rule("fast_frac", "<=", 0.25, LEGACY_SOURCE),
  Rule("late_in_band_frac", ">=", 0.80, LEGACY_SOURCE),
  Rule("target_band_frac", ">=", 0.70, LEGACY_SOURCE),
  Rule("late_target_band_frac", ">=", 0.80, LEGACY_SOURCE),
  Rule("signed_speed_ratio_mean", ">=", 0.75, LEGACY_SOURCE),
  Rule("signed_speed_ratio_mean", "<=", 1.25, LEGACY_SOURCE),
  Rule("lin_x_delta_rms", "<=", 0.035, LEGACY_SOURCE),
  Rule("lin_x_delta_abs_p95", "<=", 0.070, LEGACY_SOURCE),
  Rule("late_lin_x_delta_rms", "<=", 0.035, LEGACY_SOURCE),
  Rule("late_lin_x_delta_abs_p95", "<=", 0.070, LEGACY_SOURCE),
  Rule("mean_abs_error", "<=", 0.06, LEGACY_SOURCE),
  Rule("p95_pitch", "<=", 0.08, LEGACY_SOURCE),
  Rule("p99_pitch_rate", "<=", 0.90, LEGACY_SOURCE),
  Rule("terminated_event_rate", "<=", 0.01, LEGACY_SOURCE),
)

# The standing scenario measures yaw-channel idle noise against thresholds
# authored for moving tracking; the probe put the standing floor above them.
# command_match demands per-sample |v_x| <= 0.01 m/s; the GPU zero-residual
# LQR itself only holds 0.793, so the CPU-preliminary 0.85 was above the
# plant floor (pre-registered Q3 correction, 2026-07-15).
STANDING_LINEAR_OVERRIDES = {
  "command_match_frac": Rule(
    "command_match_frac", ">=", 0.75, GPU_BASELINE_SOURCE
  ),
  "late_in_band_frac": Rule("late_in_band_frac", ">=", 0.75, YAW_PROBE_SOURCE),
  "late_target_band_frac": Rule(
    "late_target_band_frac", ">=", 0.75, YAW_PROBE_SOURCE
  ),
}

LINEAR_RESIDUAL_RULES = (
  Rule("balance_residual_abs_mean", "<=", 0.30, LEGACY_SOURCE),
  Rule("balance_residual_abs_p95", "<=", 0.45, LEGACY_SOURCE),
)

PLANAR_BALANCE_RESIDUAL_RULES = (
  Rule("balance_residual_abs_mean", "<=", 0.10, LEGACY_SOURCE),
  Rule("balance_residual_abs_p95", "<=", 0.25, LEGACY_SOURCE),
)

# Yaw thresholds recalibrated against the measured plant (probe, qualified
# LQR): delta_rms floor 0.036 standing / 0.064 at the wz=0.10 operating
# point; a perfectly centered response holds ~0.67 per-sample in-band.
YAW_RULES = (
  Rule("command_match_frac", ">=", 0.90, LEGACY_SOURCE),
  Rule("late_slow_env_frac", "<=", 0.10, LEGACY_SOURCE),
  Rule("late_wrong_direction_env_frac", "<=", 0.10, LEGACY_SOURCE),
  Rule("late_lin_drift_env_frac", "<=", 0.10, LEGACY_SOURCE),
  Rule("in_band_frac", ">=", 0.60, YAW_PROBE_SOURCE),
  Rule("fast_frac", "<=", 0.25, LEGACY_SOURCE),
  Rule("late_in_band_frac", ">=", 0.60, YAW_PROBE_SOURCE),
  Rule("yaw_delta_rms", "<=", 0.075, YAW_PROBE_SOURCE),
  Rule("yaw_delta_abs_p95", "<=", 0.15, YAW_PROBE_SOURCE),
  Rule("late_yaw_delta_rms", "<=", 0.075, YAW_PROBE_SOURCE),
  Rule("late_yaw_delta_abs_p95", "<=", 0.15, YAW_PROBE_SOURCE),
  Rule("yaw_abs_error_mean", "<=", 0.07, LEGACY_SOURCE),
  Rule("yaw_abs_error_p90", "<=", 0.12, YAW_PROBE_SOURCE),
  Rule("lin_drift_abs_mean", "<=", 0.05, LEGACY_SOURCE),
  Rule("p95_pitch", "<=", 0.10, LEGACY_SOURCE),
  Rule("p99_pitch_rate", "<=", 0.90, LEGACY_SOURCE),
  Rule("wheel_saturation_ratio", "<=", 0.20, LEGACY_SOURCE),
  Rule("terminated_event_rate", "<=", 0.01, LEGACY_SOURCE),
)

COMBO_RULES = (
  ComboRule(
    "lin_command_match_frac", "lin_command_match_frac", ">=", 0.85,
    LEGACY_SOURCE,
  ),
  ComboRule(
    "lin_wrong_direction_frac", "lin_wrong_direction_frac", "<=", 0.10,
    LEGACY_SOURCE,
  ),
  ComboRule("lin_in_band_frac", "lin_in_band_frac", ">=", 0.70, LEGACY_SOURCE),
  ComboRule("lin_fast_frac", "lin_fast_frac", "<=", 0.30, LEGACY_SOURCE),
  ComboRule(
    "late_lin_in_band_frac", "late_lin_in_band_frac", ">=", 0.70,
    LEGACY_SOURCE,
  ),
  ComboRule(
    "lin_abs_error_mean", "lin_abs_error_mean", "<=", 0.07, LEGACY_SOURCE
  ),
  ComboRule(
    "lin_abs_error_p90", "lin_abs_error_p90", "<=", 0.12, LEGACY_SOURCE
  ),
  ComboRule("lin_x_delta_rms", "lin_x_delta_rms", "<=", 0.045, LEGACY_SOURCE),
  ComboRule(
    "lin_x_delta_abs_p95", "lin_x_delta_abs_p95", "<=", 0.090, LEGACY_SOURCE
  ),
  ComboRule(
    "late_lin_x_delta_rms", "late_lin_x_delta_rms", "<=", 0.045, LEGACY_SOURCE
  ),
  ComboRule(
    "late_lin_x_delta_abs_p95", "late_lin_x_delta_abs_p95", "<=", 0.090,
    LEGACY_SOURCE,
  ),
  ComboRule(
    "yaw_command_match_frac", "command_match_frac", ">=", 0.85, LEGACY_SOURCE
  ),
  ComboRule(
    "yaw_wrong_direction_frac", "wrong_direction_frac", "<=", 0.10,
    LEGACY_SOURCE,
  ),
  ComboRule("yaw_in_band_frac", "in_band_frac", ">=", 0.60, COMBO_SCALED_SOURCE),
  ComboRule("yaw_fast_frac", "fast_frac", "<=", 0.30, LEGACY_SOURCE),
  ComboRule(
    "yaw_late_in_band_frac", "late_in_band_frac", ">=", 0.60,
    COMBO_SCALED_SOURCE,
  ),
  ComboRule(
    "yaw_abs_error_mean", "yaw_abs_error_mean", "<=", 0.08, LEGACY_SOURCE
  ),
  ComboRule(
    "yaw_abs_error_p90", "yaw_abs_error_p90", "<=", 0.145, COMBO_SCALED_SOURCE
  ),
  ComboRule("yaw_delta_rms", "yaw_delta_rms", "<=", 0.095, COMBO_SCALED_SOURCE),
  ComboRule(
    "yaw_delta_abs_p95", "yaw_delta_abs_p95", "<=", 0.17, COMBO_SCALED_SOURCE
  ),
  ComboRule(
    "late_yaw_delta_rms", "late_yaw_delta_rms", "<=", 0.095,
    COMBO_SCALED_SOURCE,
  ),
  ComboRule(
    "late_yaw_delta_abs_p95", "late_yaw_delta_abs_p95", "<=", 0.17,
    COMBO_SCALED_SOURCE,
  ),
  ComboRule("p95_pitch", "p95_pitch", "<=", 0.12, LEGACY_SOURCE),
  ComboRule("p99_pitch_rate", "p99_pitch_rate", "<=", 0.95, LEGACY_SOURCE),
  ComboRule(
    "wheel_saturation_ratio", "wheel_saturation_ratio", "<=", 0.20,
    LEGACY_SOURCE,
  ),
  ComboRule(
    "terminated_event_rate", "terminated_event_rate", "<=", 0.01,
    LEGACY_SOURCE,
  ),
)

POSTURE_RULES = (
  Rule("height_rmse", "<=", 0.015, "018efe1 authored; pending Stage 3.0 probe"),
  Rule("pitch_rmse", "<=", 0.04, "018efe1 authored; pending Stage 3.0 probe"),
  Rule(
    "non_wheel_contact_rate", "<=", 0.01,
    "018efe1 authored; pending Stage 3.0 probe",
  ),
  Rule(
    "terminated_event_rate", "<=", 0.01,
    "018efe1 authored; pending Stage 3.0 probe",
  ),
)

REQUIRED_SCENARIO_KINDS = {
  "controller": frozenset(("controller",)),
  "linear": frozenset(("linear",)),
  "residual": frozenset(
    ("nominal", "extension", "disturbance", "transition", "mismatch")
  ),
  "planar": frozenset(("linear", "yaw", "combo")),
  "posture": frozenset(("posture",)),
  "integrated": frozenset(("linear", "yaw", "combo", "posture", "random")),
  "robust": frozenset(
    ("linear", "yaw", "combo", "posture", "random", "robust")
  ),
}

STAGE1_NOMINAL_RESIDUAL_RULES = (
  Rule("candidate_balance_residual_abs_mean", "<=", 0.10, LEGACY_SOURCE),
  Rule("candidate_balance_residual_abs_p95", "<=", 0.25, LEGACY_SOURCE),
)

STAGE1_EXTENSION_MISMATCH_RULES = (
  Rule("candidate_mean_abs_error", "<=", 0.02, LEGACY_SOURCE),
  Rule("candidate_balance_residual_abs_mean", "<=", 0.10, LEGACY_SOURCE),
  Rule("candidate_balance_residual_abs_p95", "<=", 0.25, LEGACY_SOURCE),
)

INTEGRATED_RULES = (
  Rule("tracking_error", "<=", 0.12, LEGACY_SOURCE),
  Rule("terminated_event_rate", "<=", 0.01, LEGACY_SOURCE),
  Rule("survival_rate", ">=", 0.99, LEGACY_SOURCE),
  Rule("recovery_time_s", "<=", 2.0, LEGACY_SOURCE),
  Rule("non_wheel_contact_rate", "<=", 0.01, LEGACY_SOURCE),
  Rule("wheel_saturation_ratio", "<=", 0.20, LEGACY_SOURCE),
)

ROBUST_INTEGRATED_RULES = (
  Rule("tracking_error", "<=", 0.16, LEGACY_SOURCE),
  Rule("terminated_event_rate", "<=", 0.05, LEGACY_SOURCE),
  Rule("survival_rate", ">=", 0.95, LEGACY_SOURCE),
  Rule("recovery_time_s", "<=", 2.0, LEGACY_SOURCE),
  Rule("non_wheel_contact_rate", "<=", 0.02, LEGACY_SOURCE),
  Rule("wheel_saturation_ratio", "<=", 0.20, LEGACY_SOURCE),
)
STAGE1_SAFETY_RULES = (
  Rule("candidate_terminated_event_rate", "<=", 0.01, LEGACY_SOURCE),
  Rule("candidate_p95_pitch", "<=", 0.10, LEGACY_SOURCE),
  Rule("candidate_p99_pitch_rate", "<=", 0.90, LEGACY_SOURCE),
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
  source: str = "",
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
    source=source,
  )


def rule_check(
  metrics: Mapping[str, object],
  rule: Rule,
  *,
  scenario: str,
  check_name: str | None = None,
) -> GateCheck:
  """Evaluate one manifest rule, carrying its provenance into the check."""

  return metric_check(
    metrics,
    rule.metric,
    rule.op,
    rule.limit,
    scenario=scenario,
    check_name=check_name,
    source=rule.source,
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
  standing = abs(lin_x) <= 1.0e-12
  rules = (
    (
      STANDING_LINEAR_OVERRIDES.get(rule.metric, rule)
      if standing
      else rule
    )
    for rule in LINEAR_RULES
    if not (standing and rule.metric == "signed_speed_ratio_mean")
  )
  checks = [
    rule_check(
      metrics,
      rule,
      scenario=name,
      check_name=f"{prefix}{rule.metric}",
    )
    for rule in rules
  ]
  if all(rule.metric in metrics for rule in LINEAR_RESIDUAL_RULES):
    checks.extend(
      rule_check(
        metrics,
        rule,
        scenario=name,
        check_name=f"{prefix}{rule.metric}",
      )
      for rule in LINEAR_RESIDUAL_RULES
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

  checks.extend(
    rule_check(metrics, rule, scenario=name)
    for rule in STAGE1_SAFETY_RULES
  )

  if kind == "nominal":
    checks.extend(
      rule_check(metrics, rule, scenario=name)
      for rule in STAGE1_NOMINAL_RESIDUAL_RULES
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
  elif kind in ("extension", "mismatch"):
    checks.extend(
      rule_check(metrics, rule, scenario=name)
      for rule in STAGE1_EXTENSION_MISMATCH_RULES
    )
    prefix = "extension" if kind == "extension" else "mismatch"
    for candidate_metric, baseline_metric, tolerance, label in (
      (
        "candidate_mean_abs_error",
        "baseline_mean_abs_error",
        0.001,
        f"{prefix}_tracking_no_severe_regression",
      ),
      (
        "candidate_p95_pitch",
        "baseline_p95_pitch",
        0.005,
        f"{prefix}_pitch_no_severe_regression",
      ),
      (
        "candidate_lin_x_delta_rms",
        "baseline_lin_x_delta_rms",
        0.002,
        f"{prefix}_oscillation_no_severe_regression",
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
  else:
    checks.append(_presence_check(f"stage1_supported:{kind}", False))
  return checks


def _scenario_is_safe(scenario: Mapping[str, object]) -> bool:
  """A regime can only supply evidence if the candidate was safe in it."""

  metrics = _scenario_metrics(scenario)
  return all(
    rule_check(
      metrics,
      rule,
      scenario=_scenario_name(scenario),
    ).passed
    for rule in STAGE1_SAFETY_RULES
  )


_OBSERVATIONAL_IMPROVEMENT_PAIRS: dict[str, tuple[tuple[str, str, float], ...]] = {
  "extension": (
    ("candidate_mean_abs_error", "baseline_mean_abs_error", 0.005),
  ),
  "mismatch": (
    ("candidate_mean_abs_error", "baseline_mean_abs_error", 0.005),
  ),
  "disturbance": (
    ("candidate_recovery_time_s", "baseline_recovery_time_s", 0.10),
    (
      "candidate_post_kick_error_integral",
      "baseline_post_kick_error_integral",
      0.005,
    ),
  ),
  "transition": (
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
  ),
}


def stage1_observational_improvements(
  scenarios: Sequence[Mapping[str, object]],
) -> dict[str, float]:
  """Compute every hard-regime improvement as observational data.

  Only the pre-registered primary metric can pass or fail the formal gate;
  these values go into the result envelope so trends stay visible without
  reintroducing the max-of-eight multiple-comparison problem that 44a44b1
  demonstrated empirically (26% same-seed baseline spread on screen runs).
  """

  observations: dict[str, float] = {}
  for scenario in scenarios:
    kind = str(scenario.get("kind"))
    pairs = _OBSERVATIONAL_IMPROVEMENT_PAIRS.get(kind, ())
    if not pairs or not _scenario_is_safe(scenario):
      continue
    metrics = _scenario_metrics(scenario)
    for candidate_metric, baseline_metric, floor in pairs:
      value = _fractional_improvement(
        metrics,
        candidate_metric,
        baseline_metric,
        floor=floor,
      )
      if math.isfinite(value):
        observations[
          f"{kind}:{candidate_metric.removeprefix('candidate_')}"
        ] = value
  return observations


def _stage1_improvement_check(
  scenarios: Sequence[Mapping[str, object]],
) -> list[GateCheck]:
  """Judge residual value on the pre-registered primary metric only.

  Formal-profile protocol (Hybrid v3): disturbance recovery time is the one
  metric that can certify the residual adds value, and it may only testify
  when the run contains at least MIN_IMPROVEMENT_KICK_EVENTS kick events —
  the screen-profile ~4-event estimates moved 26% between same-seed
  invocations, which is wider than the 10% improvement threshold itself.
  """

  checks: list[GateCheck] = []
  improvements: list[float] = []
  event_counts: list[float] = []
  for scenario in scenarios:
    if str(scenario.get("kind")) != "disturbance":
      continue
    metrics = _scenario_metrics(scenario)
    events_raw = metrics.get("candidate_kick_event_count", math.nan)
    events = float(events_raw) if _is_finite_number(events_raw) else math.nan
    event_counts.append(events)
    if not _scenario_is_safe(scenario):
      continue
    if not math.isfinite(events) or events < MIN_IMPROVEMENT_KICK_EVENTS:
      continue
    improvements.append(_fractional_improvement(
      metrics,
      "candidate_recovery_time_s",
      "baseline_recovery_time_s",
      floor=0.10,
    ))
  total_events = (
    sum(events for events in event_counts if math.isfinite(events))
    if event_counts
    else math.nan
  )
  checks.append(GateCheck(
    name="disturbance_kick_event_count",
    value=total_events,
    operator=">=",
    limit=float(MIN_IMPROVEMENT_KICK_EVENTS),
    passed=(
      math.isfinite(total_events)
      and total_events >= MIN_IMPROVEMENT_KICK_EVENTS
    ),
    scenario="stage1_ablation",
    source=IMPROVEMENT_PROTOCOL_SOURCE,
  ))
  finite = [value for value in improvements if math.isfinite(value)]
  # If several disturbance scenarios qualify, the claim must hold on the
  # weakest of them; there is no metric shopping left to do.
  best = min(finite) if finite else math.nan
  checks.append(GateCheck(
    name="hard_regime_fractional_improvement:disturbance:recovery_time_s",
    value=best,
    operator=">=",
    limit=0.10,
    passed=math.isfinite(best) and best >= 0.10,
    scenario="stage1_ablation",
    source=IMPROVEMENT_PROTOCOL_SOURCE,
  ))
  return checks


def yaw_scenario_checks(
  scenario: Mapping[str, object],
  *,
  require_balance_residual: bool = False,
) -> list[GateCheck]:
  metrics = _scenario_metrics(scenario)
  name = _scenario_name(scenario)
  prefix = _yaw_prefix(scenario)
  checks = [
    rule_check(
      metrics,
      rule,
      scenario=name,
      check_name=f"{prefix}{rule.metric}",
    )
    for rule in YAW_RULES[:1]
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
      source=LEGACY_SOURCE,
    )
  )
  checks.extend(
    rule_check(
      metrics,
      rule,
      scenario=name,
      check_name=f"{prefix}{rule.metric}",
    )
    for rule in YAW_RULES[1:]
  )
  if require_balance_residual or all(
    _is_finite_number(metrics.get(rule.metric))
    for rule in PLANAR_BALANCE_RESIDUAL_RULES
  ):
    checks.extend(
      rule_check(
        metrics,
        rule,
        scenario=name,
        check_name=f"{prefix}{rule.metric}",
      )
      for rule in PLANAR_BALANCE_RESIDUAL_RULES
    )
  return checks


def combo_scenario_checks(
  scenario: Mapping[str, object],
  *,
  require_balance_residual: bool = False,
) -> list[GateCheck]:
  metrics = _scenario_metrics(scenario)
  name = _scenario_name(scenario)
  prefix = _combo_prefix(scenario)
  checks = [
    metric_check(
      metrics,
      rule.metric,
      rule.op,
      rule.limit,
      scenario=name,
      check_name=f"{prefix}{rule.name}",
      source=rule.source,
    )
    for rule in COMBO_RULES
  ]
  if require_balance_residual or all(
    _is_finite_number(metrics.get(rule.metric))
    for rule in PLANAR_BALANCE_RESIDUAL_RULES
  ):
    checks.extend(
      rule_check(
        metrics,
        rule,
        scenario=name,
        check_name=f"{prefix}{rule.metric}",
      )
      for rule in PLANAR_BALANCE_RESIDUAL_RULES
    )
  return checks


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
    rule_check(metrics, rule, scenario=name)
    for rule in POSTURE_RULES
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
    rule_check(metrics, rule, scenario=name)
    for rule in rules
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
  profile: str = "formal",
  stage4_reference: Mapping[str, float] | None = None,
) -> list[GateCheck]:
  """Evaluate one capability suite from structured scenario metrics.

  Screen profiles are rejection checks only (safety, no-regression,
  tracking floors). The residual-value improvement claim runs exclusively
  on the formal profile, where the event count supports it.
  """

  if suite not in REQUIRED_SCENARIO_KINDS:
    raise ValueError(f"Unknown capability suite: {suite}")
  if profile not in ("screen", "formal"):
    raise ValueError(f"Unknown evaluation profile: {profile}")
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
    elif kind in ("nominal", "extension", "mismatch", "disturbance", "transition"):
      if suite == "residual":
        checks.extend(residual_scenario_checks(scenario))
      else:
        checks.append(_presence_check(f"supported:{kind}", False))
    elif kind == "yaw":
      checks.extend(yaw_scenario_checks(
        scenario,
        require_balance_residual=suite in ("planar", "integrated", "robust"),
      ))
    elif kind == "combo":
      checks.extend(combo_scenario_checks(
        scenario,
        require_balance_residual=suite in ("planar", "integrated", "robust"),
      ))
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
  if suite == "residual" and profile == "formal":
    checks.extend(_stage1_improvement_check(scenarios))
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
      "source": check.source,
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
  checkpoint_file_sha256: str | None = None,
  scenarios: Sequence[Mapping[str, object]],
  checks: Sequence[GateCheck],
  stage1_profile_version: str | None = None,
  mismatch_profile: Mapping[str, object] | None = None,
  retention_gate: Mapping[str, object] | None = None,
  evaluation_profile: str = "formal",
  evaluation_source: str = "live",
  rollout: Mapping[str, object] | None = None,
  observations: Mapping[str, float] | None = None,
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
    "checkpoint_file_sha256": checkpoint_file_sha256,
    "stage1_profile_version": stage1_profile_version,
    "mismatch_profile": dict(mismatch_profile or {}),
    "retention_gate": dict(retention_gate or {}),
    "evaluation_profile": evaluation_profile,
    "evaluation_source": evaluation_source,
    "rollout": dict(rollout or {}),
    "gate_pass": all(check.passed for check in checks),
    "metrics": _metrics_payload(scenarios),
    "checks": _checks_payload(checks),
    "observed_improvements": dict(observations or {}),
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

  for key in (
    "suite",
    "task",
    "git_sha",
    "controller_gain_hash",
    "calibration_hash",
    "stage1_profile_version",
  ):
    values = {result.get(key) for result in ordered}
    if len(values) != 1:
      raise ValueError(f"Seed results disagree on {key}.")
  mismatch_profiles = [result.get("mismatch_profile", {}) for result in ordered]
  if any(profile != mismatch_profiles[0] for profile in mismatch_profiles[1:]):
    raise ValueError("Seed results disagree on mismatch_profile.")

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

  hard_evidence_prefix = "hard_regime_fractional_improvement:"
  passing_hard_evidence: list[set[str]] = []
  if ordered[0]["suite"] == "residual":
    for result in ordered:
      checks = result.get("checks")
      if not isinstance(checks, Sequence):
        passing_hard_evidence.append(set())
        continue
      passing_hard_evidence.append({
        str(check.get("name"))
        for check in checks
        if isinstance(check, Mapping)
        and check.get("pass") is True
        and str(check.get("name", "")).startswith(hard_evidence_prefix)
      })
  consistent_hard_evidence = (
    True
    if not passing_hard_evidence
    else bool(set.intersection(*passing_hard_evidence))
  )
  all_seed_pass = all(bool(result.get("gate_pass")) for result in ordered)

  return {
    "schema_version": 2,
    "suite": ordered[0]["suite"],
    "task": ordered[0]["task"],
    "git_sha": ordered[0]["git_sha"],
    "controller_gain_hash": ordered[0].get("controller_gain_hash"),
    "calibration_hash": ordered[0].get("calibration_hash"),
    "stage1_profile_version": ordered[0].get("stage1_profile_version"),
    "mismatch_profile": ordered[0].get("mismatch_profile", {}),
    "evaluation_profile": "formal",
    "seeds": seeds,
    "checkpoints": [result.get("checkpoint") for result in ordered],
    "checkpoint_file_sha256s": [
      result.get("checkpoint_file_sha256") for result in ordered
    ],
    "consistent_hard_improvement_evidence": consistent_hard_evidence,
    "gate_pass": all_seed_pass and consistent_hard_evidence,
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
  "ComboRule",
  "GateCheck",
  "INTEGRATED_RULES",
  "LINEAR_RULES",
  "MIN_IMPROVEMENT_KICK_EVENTS",
  "Rule",
  "STAGE1_NOMINAL_RESIDUAL_RULES",
  "STAGE1_SAFETY_RULES",
  "STANDING_LINEAR_OVERRIDES",
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
  "rule_check",
  "stage1_observational_improvements",
  "to_deterministic_json",
  "wheel_target_saturation_threshold",
  "zero_where_masked",
  "yaw_scenario_checks",
]
