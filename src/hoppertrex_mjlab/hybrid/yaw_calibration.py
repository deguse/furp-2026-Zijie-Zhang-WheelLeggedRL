"""Yaw feedforward calibration and deterministic artifact handling.

Stage 2.0 of the Hybrid v3 schema: the classical layer owns nominal yaw
tracking through a probe-fitted monotone feedforward map from the commanded
yaw rate to the same-sign wheel differential. The PPO yaw head is a residual
around this map. The artifact mirrors the velocity-calibration pattern:
canonical-JSON SHA-256 self-hash plus a binding to the controller gain hash.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


Breakpoints = tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class YawCalibration:
  breakpoints: Breakpoints
  kp: float
  yaw_calibration_hash: str
  controller_gain_hash: str


def validate_yaw_breakpoints(value: object) -> Breakpoints:
  """Validate a monotone (wz, differential) map that pins (0, 0)."""

  if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
    raise ValueError("Yaw breakpoints must be a sequence of (wz, diff) pairs.")
  pairs: list[tuple[float, float]] = []
  for item in value:
    if not isinstance(item, Sequence) or len(item) != 2:
      raise ValueError("Each yaw breakpoint must contain exactly two values.")
    wz, differential = (float(item[0]), float(item[1]))
    if not math.isfinite(wz) or not math.isfinite(differential):
      raise ValueError("Yaw breakpoints must contain only finite values.")
    pairs.append((wz, differential))
  if len(pairs) < 2:
    raise ValueError("Yaw breakpoints must contain at least two points.")
  for (wz_a, diff_a), (wz_b, diff_b) in zip(pairs, pairs[1:]):
    if wz_b <= wz_a:
      raise ValueError("Yaw breakpoint wz values must strictly increase.")
    if diff_b < diff_a:
      raise ValueError("Yaw breakpoint differentials must be non-decreasing.")
  if not any(abs(wz) <= 1.0e-12 and abs(diff) <= 1.0e-12 for wz, diff in pairs):
    raise ValueError("Yaw breakpoints must pin (0.0, 0.0) so standing has no bias.")
  return tuple(pairs)


def yaw_feedforward(
  wz: ArrayLike,
  breakpoints: Breakpoints,
) -> NDArray[np.float64]:
  """Interpolate the commanded yaw rate to a wheel differential.

  Linear between breakpoints; clamped to the end differentials outside the
  calibrated command domain.
  """

  points = validate_yaw_breakpoints(breakpoints)
  wz_values = np.asarray([point[0] for point in points], dtype=np.float64)
  diff_values = np.asarray([point[1] for point in points], dtype=np.float64)
  command = np.asarray(wz, dtype=np.float64)
  if not np.all(np.isfinite(command)):
    raise ValueError("Yaw feedforward commands must be finite.")
  return np.interp(command, wz_values, diff_values)


def _hash_payload(payload: Mapping[str, object]) -> dict[str, object]:
  return {
    "schema_version": payload.get("schema_version"),
    "controller_gain_hash": payload.get("controller_gain_hash"),
    "kp": payload.get("kp"),
    "breakpoints": payload.get("breakpoints"),
    "source_probe": payload.get("source_probe"),
  }


def yaw_calibration_hash(payload: Mapping[str, object]) -> str:
  encoded = json.dumps(
    _hash_payload(payload), sort_keys=True, separators=(",", ":"),
  ).encode("ascii")
  return hashlib.sha256(encoded).hexdigest()


def yaw_calibration_artifact(
  *,
  controller_gain_hash: str,
  breakpoints: Sequence[Sequence[float]],
  kp: float = 0.0,
  source_probe: Mapping[str, object],
) -> dict[str, object]:
  if len(controller_gain_hash) != 64:
    raise ValueError("Controller gain hash must contain 64 characters.")
  validated = validate_yaw_breakpoints(breakpoints)
  if not math.isfinite(kp) or kp < 0.0:
    raise ValueError("Yaw feedback gain kp must be finite and non-negative.")
  payload: dict[str, object] = {
    "schema_version": 1,
    "controller_gain_hash": controller_gain_hash,
    "kp": float(kp),
    "breakpoints": [[wz, diff] for wz, diff in validated],
    "source_probe": dict(source_probe),
  }
  payload["yaw_calibration_hash"] = yaw_calibration_hash(payload)
  return payload


def parse_yaw_calibration_artifact(
  payload: Mapping[str, object],
  *,
  controller_gain_hash: str,
) -> YawCalibration:
  if payload.get("schema_version") != 1:
    raise ValueError("Yaw calibration schema_version must be 1.")
  if payload.get("controller_gain_hash") != controller_gain_hash:
    raise ValueError(
      "Yaw calibration artifact was created for a different controller."
    )
  if payload.get("yaw_calibration_hash") != yaw_calibration_hash(payload):
    raise ValueError("Yaw calibration hash does not match its artifact data.")
  breakpoints = validate_yaw_breakpoints(payload.get("breakpoints"))
  kp = float(payload.get("kp", math.nan))
  if not math.isfinite(kp) or kp < 0.0:
    raise ValueError("Yaw feedback gain kp must be finite and non-negative.")
  return YawCalibration(
    breakpoints=breakpoints,
    kp=kp,
    yaw_calibration_hash=str(payload["yaw_calibration_hash"]),
    controller_gain_hash=controller_gain_hash,
  )


__all__ = [
  "Breakpoints",
  "YawCalibration",
  "parse_yaw_calibration_artifact",
  "validate_yaw_breakpoints",
  "yaw_calibration_artifact",
  "yaw_calibration_hash",
  "yaw_feedforward",
]
