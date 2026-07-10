"""Pure capability-gate decisions shared by legacy and Hybrid evaluators."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from statistics import fmean, pstdev
from typing import Any, Iterable, Mapping, Sequence


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
  "planar": frozenset(("linear", "yaw", "combo")),
  "posture": frozenset(("posture",)),
  "integrated": frozenset(("linear", "yaw", "combo", "posture", "random")),
  "robust": frozenset(
    ("linear", "yaw", "combo", "posture", "random", "robust")
  ),
}


def _is_finite_number(value: object) -> bool:
  return (
    isinstance(value, (int, float))
    and not isinstance(value, bool)
    and math.isfinite(float(value))
  )


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
  return [
    metric_check(
      metrics,
      metric,
      operator,
      limit,
      scenario=name,
      check_name=f"{prefix}{metric}",
    )
    for metric, operator, limit in LINEAR_RULES
  ]


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


def _robust_scenario_checks(
  scenario: Mapping[str, object],
  stage4_reference: Mapping[str, float] | None,
) -> list[GateCheck]:
  metrics = _scenario_metrics(scenario)
  name = _scenario_name(scenario)
  checks = [
    metric_check(metrics, "survival_rate", ">=", 0.95, scenario=name),
    metric_check(metrics, "recovery_time_s", "<=", 2.0, scenario=name),
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
    elif kind == "yaw":
      checks.extend(yaw_scenario_checks(scenario))
    elif kind == "combo":
      checks.extend(combo_scenario_checks(scenario))
    elif kind == "posture":
      checks.extend(_posture_scenario_checks(scenario))
    elif kind == "robust":
      checks.extend(_robust_scenario_checks(scenario, stage4_reference))
    elif kind == "random":
      checks.append(
        metric_check(
          _scenario_metrics(scenario),
          "terminated_event_rate",
          "<=",
          0.01,
          scenario=_scenario_name(scenario),
        )
      )
    elif kind == "reference":
      checks.append(
        metric_check(
          _scenario_metrics(scenario),
          "terminated_event_rate",
          "<=",
          0.01,
          scenario=_scenario_name(scenario),
        )
      )
    else:
      checks.append(_presence_check(f"supported:{kind}", False))
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
    return WheelActionView(
      term_name=term_name,
      term=term,
      wheel_targets=wheel_targets,
      raw_actions=raw_actions,
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
  seed: int,
  checkpoint: str | None,
  scenarios: Sequence[Mapping[str, object]],
  checks: Sequence[GateCheck],
) -> dict[str, object]:
  """Build one versioned, auditable gate result."""

  return {
    "schema_version": 1,
    "suite": suite,
    "task": task,
    "git_sha": git_sha,
    "controller_gain_hash": controller_gain_hash,
    "seed": int(seed),
    "checkpoint": checkpoint,
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
  seeds = [int(result["seed"]) for result in ordered]

  for key in ("suite", "task", "git_sha", "controller_gain_hash"):
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
      }

  return {
    "schema_version": 1,
    "suite": ordered[0]["suite"],
    "task": ordered[0]["task"],
    "git_sha": ordered[0]["git_sha"],
    "controller_gain_hash": ordered[0].get("controller_gain_hash"),
    "seeds": seeds,
    "checkpoints": [result.get("checkpoint") for result in ordered],
    "gate_pass": all(bool(result.get("gate_pass")) for result in ordered),
    "metrics": aggregate_metrics,
    "seed_results": [
      {"seed": int(result["seed"]), "gate_pass": bool(result.get("gate_pass"))}
      for result in ordered
    ],
  }


__all__ = [
  "COMBO_RULES",
  "GateCheck",
  "LINEAR_RULES",
  "POSTURE_RULES",
  "WheelActionView",
  "YAW_RULES",
  "aggregate_seed_results",
  "combo_scenario_checks",
  "evaluate_capability_suite",
  "linear_scenario_checks",
  "make_result_envelope",
  "metric_check",
  "resolve_wheel_action",
  "to_deterministic_json",
  "yaw_scenario_checks",
]
