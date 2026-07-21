"""Station-keeping calibration and deterministic artifact handling.

Stage 3.0 of the Hybrid v3 schema: the qualified posture x balance probe
measured a systematic steady-state forward drift that is affine in the
commanded body pitch (the LQR pitch reference is fixed at the identified
nominal equilibrium, so holding any other commanded pitch settles into a
constant wheel velocity instead of station keeping). The classical layer
owns that channel through a probe-fitted monotone map from the commanded
pitch to the measured drift; the wheel velocity reference subtracts the
interpolated drift so every commanded posture station-keeps. PPO remains a
residual around this compensated baseline.

The artifact mirrors the yaw-calibration pattern: canonical-JSON SHA-256
self-hash plus bindings to BOTH the controller gain hash and the posture
map hash, because the measured drift is a joint property of the identified
LQR reference and the leg geometry realized by the posture map.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


StationBreakpoints = tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class StationCalibration:
  breakpoints: StationBreakpoints
  station_calibration_hash: str
  controller_gain_hash: str
  posture_map_hash: str
  posture_artifact_hash: str | None


def validate_station_breakpoints(value: object) -> StationBreakpoints:
  """Validate a (pitch, drift) map: pitch increases, drift never increases.

  Unlike the yaw feedforward there is no (0, 0) pin: the measured drift at
  zero commanded pitch is nonzero whenever the identified LQR reference
  pitch is nonzero, and that offset is exactly what the compensation must
  carry.
  """

  if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
    raise ValueError(
      "Station breakpoints must be a sequence of (pitch, drift) pairs."
    )
  pairs: list[tuple[float, float]] = []
  for item in value:
    if not isinstance(item, Sequence) or len(item) != 2:
      raise ValueError(
        "Each station breakpoint must contain exactly two values."
      )
    pitch, drift = (float(item[0]), float(item[1]))
    if not math.isfinite(pitch) or not math.isfinite(drift):
      raise ValueError("Station breakpoints must contain only finite values.")
    pairs.append((pitch, drift))
  if len(pairs) < 2:
    raise ValueError("Station breakpoints must contain at least two points.")
  for (pitch_a, drift_a), (pitch_b, drift_b) in zip(pairs, pairs[1:]):
    if pitch_b <= pitch_a:
      raise ValueError(
        "Station breakpoint pitch values must strictly increase."
      )
    if drift_b > drift_a:
      raise ValueError(
        "Station breakpoint drifts must be non-increasing: pitching further "
        "up must never drive the plant further forward."
      )
  return tuple(pairs)


def station_drift(
  pitch: ArrayLike,
  breakpoints: StationBreakpoints,
) -> NDArray[np.float64]:
  """Interpolate the commanded pitch to the measured steady-state drift.

  Linear between breakpoints; clamped to the end drifts outside the
  calibrated pitch domain. The compensation applied by the classical layer
  is the NEGATIVE of this value, added to the wheel velocity reference.
  """

  points = validate_station_breakpoints(breakpoints)
  pitch_values = np.asarray([point[0] for point in points], dtype=np.float64)
  drift_values = np.asarray([point[1] for point in points], dtype=np.float64)
  command = np.asarray(pitch, dtype=np.float64)
  if not np.all(np.isfinite(command)):
    raise ValueError("Station drift commands must be finite.")
  return np.interp(command, pitch_values, drift_values)


def _hash_payload(payload: Mapping[str, object]) -> dict[str, object]:
  identity = {
    "schema_version": payload.get("schema_version"),
    "controller_gain_hash": payload.get("controller_gain_hash"),
    "posture_map_hash": payload.get("posture_map_hash"),
    "breakpoints": payload.get("breakpoints"),
    "source_probe": payload.get("source_probe"),
  }
  if payload.get("posture_artifact_hash") is not None:
    identity["posture_artifact_hash"] = payload["posture_artifact_hash"]
  return identity


def station_calibration_hash(payload: Mapping[str, object]) -> str:
  encoded = json.dumps(
    _hash_payload(payload), sort_keys=True, separators=(",", ":"),
  ).encode("ascii")
  return hashlib.sha256(encoded).hexdigest()


def station_calibration_artifact(
  *,
  controller_gain_hash: str,
  posture_map_hash: str,
  posture_artifact_hash: str | None = None,
  breakpoints: Sequence[Sequence[float]],
  source_probe: Mapping[str, object],
) -> dict[str, object]:
  if len(controller_gain_hash) != 64:
    raise ValueError("Controller gain hash must contain 64 characters.")
  if len(posture_map_hash) != 64:
    raise ValueError("Posture map hash must contain 64 characters.")
  if posture_artifact_hash is not None and len(posture_artifact_hash) != 64:
    raise ValueError("Posture artifact hash must contain 64 characters.")
  validated = validate_station_breakpoints(breakpoints)
  payload: dict[str, object] = {
    "schema_version": 1,
    "controller_gain_hash": controller_gain_hash,
    "posture_map_hash": posture_map_hash,
    "breakpoints": [[pitch, drift] for pitch, drift in validated],
    "source_probe": dict(source_probe),
  }
  if posture_artifact_hash is not None:
    payload["posture_artifact_hash"] = posture_artifact_hash
  payload["station_calibration_hash"] = station_calibration_hash(payload)
  return payload


def parse_station_calibration_artifact(
  payload: Mapping[str, object],
  *,
  controller_gain_hash: str,
  posture_map_hash: str,
  posture_artifact_hash: str | None = None,
) -> StationCalibration:
  if payload.get("schema_version") != 1:
    raise ValueError("Station calibration schema_version must be 1.")
  if payload.get("controller_gain_hash") != controller_gain_hash:
    raise ValueError(
      "Station calibration artifact was created for a different controller."
    )
  if payload.get("posture_map_hash") != posture_map_hash:
    raise ValueError(
      "Station calibration artifact was created for a different posture map."
    )
  payload_artifact_hash = payload.get("posture_artifact_hash")
  if posture_artifact_hash is not None:
    if payload_artifact_hash != posture_artifact_hash:
      raise ValueError(
        "Station calibration artifact was created for a different posture "
        "command envelope."
      )
  elif payload_artifact_hash is not None:
    raise ValueError(
      "Station calibration binds a full posture artifact, but the runtime "
      "posture map does not provide that identity."
    )
  if payload.get("station_calibration_hash") != station_calibration_hash(
    payload
  ):
    raise ValueError(
      "Station calibration hash does not match its artifact data."
    )
  breakpoints = validate_station_breakpoints(payload.get("breakpoints"))
  return StationCalibration(
    breakpoints=breakpoints,
    station_calibration_hash=str(payload["station_calibration_hash"]),
    controller_gain_hash=controller_gain_hash,
    posture_map_hash=posture_map_hash,
    posture_artifact_hash=(
      str(payload_artifact_hash) if payload_artifact_hash is not None else None
    ),
  )


__all__ = [
  "StationBreakpoints",
  "StationCalibration",
  "parse_station_calibration_artifact",
  "station_calibration_artifact",
  "station_calibration_hash",
  "station_drift",
  "validate_station_breakpoints",
]
