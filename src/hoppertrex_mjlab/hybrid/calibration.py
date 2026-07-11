"""Velocity-command calibration and deterministic candidate scoring."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CalibrationCandidate:
  scale: float
  bias: float
  scenarios: tuple[Mapping[str, float], ...]


@dataclass(frozen=True)
class CandidateScore:
  scale: float
  bias: float
  accepted: bool
  score: float
  rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class VelocityCalibration:
  scale: float
  bias: float
  calibration_hash: str
  controller_gain_hash: str


def apply_velocity_calibration(
  requested_vx: float,
  *,
  scale: float,
  bias: float,
) -> float:
  if not all(math.isfinite(value) for value in (requested_vx, scale, bias)):
    raise ValueError("Velocity calibration inputs must be finite.")
  if scale <= 0.0:
    raise ValueError("Velocity command scale must be positive.")
  return scale * requested_vx + bias


def _hash_payload(payload: Mapping[str, object]) -> dict[str, object]:
  return {
    "schema_version": payload.get("schema_version"),
    "controller_gain_hash": payload.get("controller_gain_hash"),
    "velocity_command_scale": payload.get("velocity_command_scale"),
    "velocity_command_bias": payload.get("velocity_command_bias"),
    "seed": payload.get("seed"),
    "candidates": payload.get("candidates"),
  }


def calibration_hash(payload: Mapping[str, object]) -> str:
  encoded = json.dumps(
    _hash_payload(payload), sort_keys=True, separators=(",", ":"),
  ).encode("ascii")
  return hashlib.sha256(encoded).hexdigest()


def calibration_artifact(
  *,
  controller_gain_hash: str,
  scale: float,
  bias: float,
  seed: int,
  candidates: Sequence[Mapping[str, object]],
) -> dict[str, object]:
  if len(controller_gain_hash) != 64:
    raise ValueError("Controller gain hash must contain 64 characters.")
  apply_velocity_calibration(0.0, scale=scale, bias=bias)
  payload: dict[str, object] = {
    "schema_version": 1,
    "controller_gain_hash": controller_gain_hash,
    "velocity_command_scale": scale,
    "velocity_command_bias": bias,
    "seed": int(seed),
    "candidates": list(candidates),
  }
  payload["calibration_hash"] = calibration_hash(payload)
  return payload


def parse_calibration_artifact(
  payload: Mapping[str, object],
  *,
  controller_gain_hash: str,
) -> VelocityCalibration:
  if payload.get("schema_version") != 1:
    raise ValueError("Calibration schema_version must be 1.")
  if payload.get("controller_gain_hash") != controller_gain_hash:
    raise ValueError("Calibration artifact was created for a different controller.")
  if payload.get("calibration_hash") != calibration_hash(payload):
    raise ValueError("Calibration hash does not match its artifact data.")
  scale = float(payload.get("velocity_command_scale", math.nan))
  bias = float(payload.get("velocity_command_bias", math.nan))
  apply_velocity_calibration(0.0, scale=scale, bias=bias)
  return VelocityCalibration(
    scale=scale,
    bias=bias,
    calibration_hash=str(payload["calibration_hash"]),
    controller_gain_hash=controller_gain_hash,
  )


def score_candidate(candidate: CalibrationCandidate) -> CandidateScore:
  reasons: list[str] = []
  errors: list[float] = []
  zero_actual = math.inf
  requested_values: set[float] = set()
  for scenario in candidate.scenarios:
    values = tuple(float(scenario[name]) for name in (
      "requested_vx", "actual_vx", "terminated_event_rate",
      "p95_pitch", "p99_pitch_rate",
    ))
    if not all(math.isfinite(value) for value in values):
      reasons.append("non_finite_metric")
      continue
    requested, actual, terminated, pitch, pitch_rate = values
    requested_values.add(round(requested, 6))
    if terminated > 0.01:
      reasons.append("termination_rate")
    if pitch > 0.08:
      reasons.append("pitch")
    if pitch_rate > 0.90:
      reasons.append("pitch_rate")
    errors.append((actual - requested) ** 2)
    if abs(requested) <= 1.0e-12:
      zero_actual = abs(actual)
  if requested_values != {-0.07, 0.0, 0.07}:
    reasons.append("missing_scenario")
  if not errors or not math.isfinite(zero_actual):
    reasons.append("missing_scenario")
  score = (
    math.sqrt(sum(errors) / len(errors)) + 2.0 * zero_actual
    if errors and math.isfinite(zero_actual) else math.inf
  )
  return CandidateScore(
    scale=candidate.scale,
    bias=candidate.bias,
    accepted=not reasons,
    score=score if not reasons else math.inf,
    rejection_reasons=tuple(sorted(set(reasons))),
  )


def candidate_from_envelope(
  envelope: Mapping[str, object],
  *,
  scale: float,
  bias: float,
) -> CalibrationCandidate:
  rows: list[Mapping[str, float]] = []
  scenarios = envelope.get("scenarios")
  if not isinstance(scenarios, list):
    metrics_map = envelope.get("metrics")
    if not isinstance(metrics_map, Mapping):
      raise ValueError("Gate envelope must contain scenarios or metrics.")
    scenarios = [
      {"lin_x": metrics.get("lin_x", math.nan), "metrics": metrics}
      for metrics in metrics_map.values()
      if isinstance(metrics, Mapping)
    ]
  for scenario in scenarios:
    if not isinstance(scenario, Mapping) or not isinstance(scenario.get("metrics"), Mapping):
      raise ValueError("Gate scenario must contain a metrics object.")
    metrics = scenario["metrics"]
    rows.append({
      "requested_vx": float(scenario.get("lin_x", math.nan)),
      "actual_vx": float(metrics.get("mean_actual_lin_x", math.nan)),
      "terminated_event_rate": float(metrics.get("terminated_event_rate", math.nan)),
      "p95_pitch": float(metrics.get("p95_pitch", math.nan)),
      "p99_pitch_rate": float(metrics.get("p99_pitch_rate", math.nan)),
    })
  rows.sort(key=lambda row: row["requested_vx"])
  return CalibrationCandidate(scale=scale, bias=bias, scenarios=tuple(rows))


def fine_grid(center_scale: float, center_bias: float) -> tuple[tuple[float, float], ...]:
  scales = tuple(round(center_scale + offset, 10) for offset in (-0.04, -0.02, 0.0, 0.02, 0.04))
  biases = tuple(round(center_bias + offset, 10) for offset in (-0.004, -0.002, 0.0, 0.002, 0.004))
  return tuple((scale, bias) for scale in scales for bias in biases)


__all__ = [
  "CalibrationCandidate", "CandidateScore", "VelocityCalibration",
  "apply_velocity_calibration",
  "calibration_artifact", "calibration_hash", "fine_grid", "score_candidate",
  "candidate_from_envelope",
  "parse_calibration_artifact",
]
