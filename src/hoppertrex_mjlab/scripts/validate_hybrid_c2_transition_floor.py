"""Independently validate a C2-j2 transition-floor capture and its raw arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from hoppertrex_mjlab.hybrid.innovation_detector import (
  FEATURE_NAMES,
  OFFICIAL_TRANSITION_FLOOR_PROTOCOL,
  TRANSITION_CENTER,
  parse_innovation_predictor,
  parse_transition_floor,
  transition_floor_cells,
)
from hoppertrex_mjlab.scripts.probe_hybrid_c2_transition_floor import (
  CONTROL_DT_S,
  POSTURE_HEIGHT_SLEW_RATE,
  POSTURE_PITCH_SLEW_RATE,
  _is_finite_domain_violation,
  raw_command,
)

RAW_KEYS = (
  "z",
  "u",
  "next_z",
  "shaped_posture",
  "innovation",
  "accelerometer_specific_force_x",
  "projected_gravity_x",
  "forward_deceleration",
  "active",
  "raw_command",
)
WHEEL_RADIUS_M = 0.1


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise ValueError(message)


def _matches_or_nonfinite(actual: np.ndarray, expected: np.ndarray, atol: float) -> bool:
  finite = np.isfinite(actual)
  broadcast = np.broadcast_to(expected, actual.shape)
  return bool(np.allclose(actual[finite], broadcast[finite], atol=atol, rtol=0.0))


def _voting_maxima(features: np.ndarray, voting: np.ndarray) -> np.ndarray:
  if not np.any(voting):
    return np.full(features.shape[2], np.nan, dtype=np.float64)
  return np.asarray(
    [np.max(features[:, :, feature][voting]) for feature in range(features.shape[2])],
    dtype=np.float64,
  )


def _expected_shaped_posture(cell: dict[str, Any]) -> np.ndarray:
  current = np.asarray(TRANSITION_CENTER[:2], dtype=np.float64)
  limit = np.asarray(
    [POSTURE_HEIGHT_SLEW_RATE, POSTURE_PITCH_SLEW_RATE], dtype=np.float64
  ) * CONTROL_DT_S
  values = []
  for tick in range(500):
    target = np.asarray(raw_command(cell, tick)[:2], dtype=np.float64)
    current = current + np.clip(target - current, -limit, limit)
    values.append(current.copy())
  return np.asarray(values, dtype=np.float64)


def validate_capture(
  output_dir: Path,
  predictor_path: Path,
  *,
  expected_git_sha: str | None = None,
  expected_mjlab_git_sha: str | None = None,
) -> dict[str, Any]:
  payload_path = output_dir / "c2_innovation_floor.json"
  payload = json.loads(payload_path.read_text(encoding="utf-8"))
  predictor = parse_innovation_predictor(
    json.loads(predictor_path.read_text(encoding="utf-8"))
  )
  _require(payload.get("schema_version") == 1, "Floor schema drifted.")
  _require(
    payload.get("artifact_type") == "c2_innovation_transition_floor",
    "Floor artifact type drifted.",
  )
  _require(
    payload.get("probe") == "hybrid_c2_transition_floor_v1",
    "Floor probe identity drifted.",
  )
  _require(
    payload.get("protocol") == OFFICIAL_TRANSITION_FLOOR_PROTOCOL,
    "Floor protocol drifted.",
  )
  _require(
    payload.get("predictor_hash") == predictor.predictor_hash,
    "Floor predictor binding drifted.",
  )
  _require(
    payload.get("bindings") == predictor.bindings,
    "Floor deployment bindings drifted.",
  )
  if expected_git_sha is not None:
    _require(payload.get("git_sha") == expected_git_sha, "Floor Git SHA drifted.")
  if expected_mjlab_git_sha is not None:
    _require(
      payload.get("mjlab_git_sha") == expected_mjlab_git_sha,
      "Floor MjLab SHA drifted.",
    )
  for key in (
    "evidence_eligible",
    "detector_fit_eligible",
    "promotion_eligible",
    "training_eligible",
  ):
    _require(payload.get(key) is False, f"Floor {key} flag drifted.")
  _require(payload.get("checkpoint") is None, "Floor checkpoint must be null.")

  cells = payload.get("cells")
  registered_cells = transition_floor_cells()
  _require(isinstance(cells, list) and len(cells) == 10, "Floor must have ten cells.")
  observed_maxima = np.zeros(3, dtype=np.float64)
  any_domain_violation = False
  any_invalid = False
  for index, registered in enumerate(registered_cells):
    cell = cells[index]
    _require(isinstance(cell, dict), f"Cell {index} metadata is invalid.")
    _require(cell.get("cell_index") == index, f"Cell {index} index drifted.")
    for key in ("name", "kind", "target"):
      _require(cell.get(key) == registered[key], f"Cell {index} {key} drifted.")
    raw_file = f"cell_{index:02d}.npz"
    raw_path = output_dir / raw_file
    _require(cell.get("raw_file") == raw_file, f"Cell {index} raw filename drifted.")
    _require(raw_path.is_file(), f"Cell {index} raw file is missing.")
    _require(cell.get("raw_sha256") == _sha256(raw_path), f"Cell {index} raw hash drifted.")
    _require(cell.get("raw_shape") == [500, 16], f"Cell {index} raw shape drifted.")

    with np.load(raw_path, allow_pickle=False) as raw:
      _require(tuple(raw.files) == RAW_KEYS, f"Cell {index} raw keys drifted.")
      expected_shapes = {
        "z": (500, 16, 2),
        "u": (500, 16, 1),
        "next_z": (500, 16, 2),
        "shaped_posture": (500, 16, 2),
        "innovation": (500, 16, 2),
        "accelerometer_specific_force_x": (500, 16, 1),
        "projected_gravity_x": (500, 16, 1),
        "forward_deceleration": (500, 16, 1),
        "active": (500, 16, 1),
        "raw_command": (500, 16, 3),
      }
      for key, shape in expected_shapes.items():
        _require(raw[key].shape == shape, f"Cell {index} {key} shape drifted.")
      raw_finite = all(np.isfinite(raw[key]).all() for key in RAW_KEYS)
      any_invalid |= not raw_finite
      finite_deceleration = raw["forward_deceleration"][
        np.isfinite(raw["forward_deceleration"])
      ]
      _require(
        np.all(finite_deceleration >= 0.0),
        f"Cell {index} finite deceleration is negative.",
      )
      expected_deceleration = np.maximum(
        -(
          raw["accelerometer_specific_force_x"].astype(np.float32)
          + np.float32(9.81) * raw["projected_gravity_x"].astype(np.float32)
        ),
        np.float32(0.0),
      ).astype(np.float64)
      _require(
        np.array_equal(
          raw["forward_deceleration"], expected_deceleration, equal_nan=True
        ),
        f"Cell {index} direct IMU deceleration reconstruction drifted.",
      )
      _require(
        np.array_equal(raw["next_z"][:-1], raw["z"][1:], equal_nan=True),
        f"Cell {index} transition alignment drifted.",
      )

      expected_raw = np.asarray(
        [raw_command(registered, tick) for tick in range(500)], dtype=np.float64
      )
      _require(
        _matches_or_nonfinite(raw["raw_command"], expected_raw[:, None, :], 1.0e-12),
        f"Cell {index} raw command schedule drifted.",
      )
      expected_shaped = _expected_shaped_posture(registered)
      _require(
        _matches_or_nonfinite(
          raw["shaped_posture"], expected_shaped[:, None, :], 2.0e-7
        ),
        f"Cell {index} deployed posture shaping drifted.",
      )

      progress = np.cumsum(
        raw["next_z"][:, :, 1] * WHEEL_RADIUS_M * CONTROL_DT_S, axis=0
      )
      expected_active = np.logical_and.accumulate(progress < 0.35, axis=0)
      _require(raw["active"].dtype == np.bool_, f"Cell {index} active mask is not Boolean.")
      _require(
        np.array_equal(raw["active"][:, :, 0], expected_active),
        f"Cell {index} active mask drifted.",
      )
      voting = expected_active.copy()
      voting[0] = False
      voting_per_env = np.count_nonzero(voting, axis=0).astype(int).tolist()
      _require(
        cell.get("active_voting_ticks_per_env") == voting_per_env,
        f"Cell {index} per-attempt voting counts drifted.",
      )
      _require(
        cell.get("active_voting_ticks") == int(np.count_nonzero(voting)),
        f"Cell {index} pooled voting count drifted.",
      )

      expected_innovation = np.empty((500, 16, 2), dtype=np.float64)
      domain_violations = 0
      for tick in range(500):
        for env_index in range(16):
          height, pitch = raw["shaped_posture"][tick, env_index]
          try:
            prediction = predictor.predict(
              raw["z"][tick, env_index],
              float(raw["u"][tick, env_index, 0]),
              float(height),
              float(pitch),
            )
            expected_innovation[tick, env_index] = np.abs(
              raw["next_z"][tick, env_index] - prediction
            )
          except ValueError as error:
            if _is_finite_domain_violation(
              predictor,
              raw["z"][tick, env_index],
              float(raw["u"][tick, env_index, 0]),
              float(height),
              float(pitch),
            ):
              domain_violations += int(expected_active[tick, env_index])
            elif not (
              "outside the fitted domain" in str(error)
              or "must be finite" in str(error)
            ):
              raise
            expected_innovation[tick, env_index] = np.nan
      _require(
        np.allclose(
          raw["innovation"], expected_innovation, atol=1.0e-12, rtol=0.0, equal_nan=True
        ),
        f"Cell {index} innovation reconstruction drifted.",
      )
      _require(
        cell.get("domain_violation_count") == domain_violations,
        f"Cell {index} domain count drifted.",
      )
      any_domain_violation |= domain_violations > 0
      health_invalid = any(
        cell.get(key) != 0
        for key in (
          "termination_count",
          "timeout_count",
          "non_wheel_contact_count",
        )
      ) or any(value <= 0 for value in voting_per_env)

      features = np.concatenate(
        (expected_innovation, raw["forward_deceleration"]), axis=2
      )
      maxima = _voting_maxima(features, voting)
      maxima_valid = bool(np.all(np.isfinite(maxima)) and np.all(maxima > 0.0))
      any_invalid |= health_invalid or not maxima_valid
      metadata_maxima = cell.get("feature_maxima")
      _require(isinstance(metadata_maxima, dict), f"Cell {index} maxima are missing.")
      for feature, name in enumerate(FEATURE_NAMES):
        expected_value = float(maxima[feature]) if np.isfinite(maxima[feature]) else None
        _require(
          metadata_maxima.get(name) == expected_value,
          f"Cell {index} {name} maximum drifted.",
        )
      if maxima_valid:
        observed_maxima = np.maximum(observed_maxima, maxima)

  expected_classification = (
    "PREDICTOR_DOMAIN_UNCOVERED_STOP"
    if any_domain_violation
    else ("INVALID_INNOVATION_FLOOR" if any_invalid else "INNOVATION_FLOOR_QUALIFIED")
  )
  _require(
    payload.get("classification") == expected_classification,
    "Floor classification does not match raw evidence.",
  )
  if expected_classification == "INNOVATION_FLOOR_QUALIFIED":
    maxima_mapping = payload.get("pooled_feature_maxima")
    _require(isinstance(maxima_mapping, dict), "Pooled maxima are missing.")
    _require(
      np.array_equal(
        observed_maxima,
        np.asarray([maxima_mapping[name] for name in FEATURE_NAMES], dtype=np.float64),
      ),
      "Pooled maxima do not match raw evidence.",
    )
    parse_transition_floor(payload, predictor_hash=predictor.predictor_hash)
  else:
    _require("threshold_table" not in payload, "Stopped floor must not freeze thresholds.")
    _require("threshold_table_hash" not in payload, "Stopped floor must not freeze a table hash.")
    _require("floor_hash" not in payload, "Stopped floor must not freeze a floor hash.")
  return payload


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--predictor", type=Path, required=True)
  parser.add_argument("--expected-git-sha")
  parser.add_argument("--expected-mjlab-git-sha")
  args = parser.parse_args(argv)
  payload = validate_capture(
    args.output_dir,
    args.predictor,
    expected_git_sha=args.expected_git_sha,
    expected_mjlab_git_sha=args.expected_mjlab_git_sha,
  )
  print(f"VALIDATED_CLASSIFICATION={payload['classification']}")


if __name__ == "__main__":
  main()
