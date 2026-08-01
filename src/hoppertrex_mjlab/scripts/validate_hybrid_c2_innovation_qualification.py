"""Validate C2-j3 raw arrays and independently replay all 125 candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hoppertrex_mjlab.hybrid.controller_schedule import canonical_hash
from hoppertrex_mjlab.hybrid.innovation_detector import (
  EXPECTED_BINDINGS,
  OFFICIAL_QUALIFICATION_PROTOCOL,
  QUALIFICATION_ARTIFACT_TYPE,
  QUALIFICATION_DRIVE_STEPS,
  QUALIFICATION_PAIRS_PER_CELL,
  QUALIFICATION_PORTABLE_EQUIVALENCE_ATOL,
  QUALIFICATION_POST_IMPACT_STEPS,
  QUALIFICATION_POSTURE_CAPTURE_ATOL,
  QUALIFICATION_PRE_IMPACT_STEPS,
  QUALIFICATION_RESET_WRITE_ATOL,
  QUALIFICATION_SCHEMA_VERSION,
  RESET_PERTURBATION_BOUNDS,
  parse_innovation_detector_qualification,
  parse_innovation_predictor,
  parse_transition_floor,
  qualification_cells,
)

PREDICTOR_HASH = "d1374e4c0c071777bdb3e964e644cad3ba854df4f9976dab016bf9a8d861232d"
FLOOR_HASH = "1692f8e6a3ff9d82b22ee5ac579b48d832a852b8bcfccb88fb02d85b360e4e58"
THRESHOLD_TABLE_HASH = "098888c153e60d5539e98e85c7e523a5a27c0848f6628d191c79f0613d3566fc"
GRAVITY_MPS2 = 9.81
FORMAL_CLASSIFICATIONS = {
  "INNOVATION_DETECTOR_QUALIFIED",
  "C2_INNOVATION_DETECTOR_UNQUALIFIED_STOP",
  "INVALID_INNOVATION_CAPTURE",
}
RAW_KEYS = {
  "flat_z", "stair_z", "flat_u", "stair_u", "flat_next_z", "stair_next_z",
  "flat_shaped_posture", "stair_shaped_posture", "flat_features", "stair_features",
  "flat_active", "stair_active", "stair_riser_contact", "impact_steps",
  "stair_contact_found", "stair_contact_force_contact_frame",
  "stair_contact_pos_global", "stair_contact_normal_global",
  "stair_outer_face_x", "stair_terrain_origin_x",
  "reset_perturbations", "flat_reset_relative", "stair_reset_relative",
  "flat_written_reset_relative", "stair_written_reset_relative",
  "flat_wheel_targets", "stair_wheel_targets", "flat_portable_targets",
  "stair_portable_targets", "flat_specific_force_x", "stair_specific_force_x",
  "flat_projected_gravity_x", "stair_projected_gravity_x", "flat_terminated",
  "stair_terminated", "flat_timeout", "stair_timeout", "flat_non_wheel_contact",
  "stair_non_wheel_contact", "flat_settle_riser_contact",
  "stair_settle_riser_contact", "flat_drive_start_past_face",
  "stair_drive_start_past_face",
}
REPLAY_FEATURE_NAMES = (
  "pitch_rate_innovation_radps",
  "wheel_speed_innovation_radps",
  "forward_deceleration_mps2",
)
REPLAY_HEIGHT_NODES = (0.2907321708, 0.3092089487, 0.3276857266)
REPLAY_PITCH_NODES = (-0.032, 0.0, 0.032)
REPLAY_POSTURE_ATOL = 1.0e-7
REPLAY_INPUT_ATOL_RADPS = 1.0e-6
REPLAY_PAIRS_PER_CELL = 16
REPLAY_DRIVE_STEPS = 500
REPLAY_PRE_IMPACT_STEPS = 25
REPLAY_POST_IMPACT_STEPS = 75
REPLAY_MAX_DELAY_TICKS = 3
REPLAY_OVERALL_TIMELY_MIN = 274
REPLAY_PER_CELL_TIMELY_MIN = 15
REPLAY_RISER_MIN_ABS_NORMAL_X = 0.25
REPLAY_RISER_FACE_X_TOLERANCE_M = 0.02
REPLAY_RISER_MIN_NORMAL_FORCE_N = 1.0
REPLAY_OUTER_FACE_OFFSET_FROM_TERRAIN_ORIGIN_M = -3.0
REPLAY_GEOMETRY_WRITE_ATOL_M = 2.0e-5


class _ReplayPostureDomainError(ValueError):
  pass


class _ReplayInputDomainError(ValueError):
  pass


def _replay_cells() -> list[dict[str, Any]]:
  cells: list[dict[str, Any]] = []
  for height_index, height in enumerate(REPLAY_HEIGHT_NODES):
    for pitch_index, pitch in enumerate(REPLAY_PITCH_NODES):
      node_index = 3 * height_index + pitch_index
      for vx_index, vx in enumerate((0.07, 0.10)):
        cells.append({
          "cell_index": 2 * node_index + vx_index,
          "node_index": node_index,
          "height_index": height_index,
          "pitch_index": pitch_index,
          "vx_index": vx_index,
          "height_m": height,
          "pitch_rad": pitch,
          "vx_mps": vx,
        })
  return cells


def _axis_interval(
  nodes: tuple[float, ...], value: float
) -> tuple[int, int, float]:
  clipped = min(max(float(value), nodes[0]), nodes[-1])
  upper = int(np.searchsorted(np.asarray(nodes), clipped, side="right"))
  upper = min(max(upper, 1), len(nodes) - 1)
  lower = upper - 1
  weight = (clipped - nodes[lower]) / (nodes[upper] - nodes[lower])
  return lower, upper, float(weight)


def _bilinear(values: np.ndarray, height: float, pitch: float) -> np.ndarray:
  h0, h1, hw = _axis_interval(REPLAY_HEIGHT_NODES, height)
  p0, p1, pw = _axis_interval(REPLAY_PITCH_NODES, pitch)
  low = (1.0 - pw) * values[h0, p0] + pw * values[h0, p1]
  high = (1.0 - pw) * values[h1, p0] + pw * values[h1, p1]
  return np.asarray((1.0 - hw) * low + hw * high, dtype=np.float64)


class _ReplayPredictor:
  def __init__(self, payload: dict[str, Any]):
    _require(
      tuple(float(value) for value in payload.get("height_nodes", ()))
      == REPLAY_HEIGHT_NODES,
      "Independent predictor height nodes drifted.",
    )
    _require(
      tuple(float(value) for value in payload.get("pitch_nodes", ()))
      == REPLAY_PITCH_NODES,
      "Independent predictor pitch nodes drifted.",
    )
    nodes = payload.get("nodes")
    _require(
      isinstance(nodes, list) and len(nodes) == 9,
      "Independent predictor node count drifted.",
    )
    self.a = np.empty((3, 3, 2, 2), dtype=np.float64)
    self.b = np.empty((3, 3, 2, 1), dtype=np.float64)
    self.c = np.empty((3, 3, 2), dtype=np.float64)
    self.u_min = np.empty((3, 3), dtype=np.float64)
    self.u_max = np.empty((3, 3), dtype=np.float64)
    for index, node in enumerate(nodes):
      _require(
        isinstance(node, dict) and node.get("node_index") == index,
        "Independent predictor node order drifted.",
      )
      height_index, pitch_index = divmod(index, 3)
      a = np.asarray(node.get("a"), dtype=np.float64)
      b = np.asarray(node.get("b"), dtype=np.float64)
      c = np.asarray(node.get("c"), dtype=np.float64)
      _require(
        a.shape == (2, 2) and b.shape == (2, 1) and c.shape == (2,),
        "Independent predictor coefficient shape drifted.",
      )
      self.a[height_index, pitch_index] = a
      self.b[height_index, pitch_index] = b
      self.c[height_index, pitch_index] = c
      self.u_min[height_index, pitch_index] = float(
        node.get("fit_u_min_radps", math.nan)
      )
      self.u_max[height_index, pitch_index] = float(
        node.get("fit_u_max_radps", math.nan)
      )
    _require(
      all(
        np.all(np.isfinite(value))
        for value in (self.a, self.b, self.c, self.u_min, self.u_max)
      )
      and np.all(self.u_min < self.u_max),
      "Independent predictor contains invalid coefficients or domains.",
    )

  @staticmethod
  def _posture(height: float, pitch: float) -> tuple[float, float]:
    h = float(height)
    p = float(pitch)
    if not math.isfinite(h) or not math.isfinite(p):
      raise ValueError("Independent predictor posture is nonfinite.")
    if not (
      REPLAY_HEIGHT_NODES[0] - REPLAY_POSTURE_ATOL
      <= h
      <= REPLAY_HEIGHT_NODES[-1] + REPLAY_POSTURE_ATOL
    ):
      raise _ReplayPostureDomainError("Independent predictor height is outside.")
    if not (
      REPLAY_PITCH_NODES[0] - REPLAY_POSTURE_ATOL
      <= p
      <= REPLAY_PITCH_NODES[-1] + REPLAY_POSTURE_ATOL
    ):
      raise _ReplayPostureDomainError("Independent predictor pitch is outside.")
    return (
      min(max(h, REPLAY_HEIGHT_NODES[0]), REPLAY_HEIGHT_NODES[-1]),
      min(max(p, REPLAY_PITCH_NODES[0]), REPLAY_PITCH_NODES[-1]),
    )

  def predict(
    self,
    state: np.ndarray,
    control: float,
    height: float,
    pitch: float,
  ) -> np.ndarray:
    z = np.asarray(state, dtype=np.float64)
    u = float(control)
    if z.shape != (2,) or not np.all(np.isfinite(z)) or not math.isfinite(u):
      raise ValueError("Independent predictor state/input is invalid.")
    height, pitch = self._posture(height, pitch)
    minimum = float(_bilinear(self.u_min, height, pitch))
    maximum = float(_bilinear(self.u_max, height, pitch))
    if not minimum - REPLAY_INPUT_ATOL_RADPS <= u <= maximum + REPLAY_INPUT_ATOL_RADPS:
      raise _ReplayInputDomainError("Independent predictor input is outside.")
    a = _bilinear(self.a, height, pitch)
    b = _bilinear(self.b, height, pitch)
    c = _bilinear(self.c, height, pitch)
    return a @ z + b[:, 0] * u + c


def _thresholds(row: dict[str, Any]) -> np.ndarray:
  _require(
    isinstance(row, dict) and set(row) == {"index", *REPLAY_FEATURE_NAMES},
    "Independent threshold row drifted.",
  )
  _require(
    isinstance(row["index"], int) and not isinstance(row["index"], bool),
    "Independent threshold index drifted.",
  )
  values = np.asarray([row[name] for name in REPLAY_FEATURE_NAMES], dtype=np.float64)
  _require(
    np.all(np.isfinite(values)) and np.all(values > 0.0),
    "Independent thresholds are invalid.",
  )
  return values


def _first_trigger(
  features: np.ndarray, active: np.ndarray, thresholds: np.ndarray
) -> int | None:
  _require(
    features.shape == (REPLAY_DRIVE_STEPS, 3)
    and active.shape == (REPLAY_DRIVE_STEPS,)
    and active.dtype == np.bool_
    and np.all(np.isfinite(features))
    and np.all(features >= 0.0),
    "Independent detector series is invalid.",
  )
  consecutive = 0
  for tick in range(REPLAY_DRIVE_STEPS):
    if not bool(active[tick]):
      consecutive = 0
      continue
    if tick == 0:
      continue
    consecutive = consecutive + 1 if np.count_nonzero(
      features[tick] >= thresholds
    ) >= 2 else 0
    if consecutive >= 2:
      return tick
  return None


def _evaluate_candidate(
  row: dict[str, Any], cells: list[dict[str, Any]]
) -> dict[str, Any]:
  thresholds = _thresholds(row)
  registered = _replay_cells()
  _require(len(cells) == len(registered), "Independent replay requires 18 cells.")
  totals = {"flat": 0, "pre": 0, "timely": 0, "late": 0, "missing": 0}
  delays: list[int] = []
  cell_results: list[dict[str, Any]] = []
  for index, (cell, expected) in enumerate(zip(cells, registered, strict=True)):
    _require(cell.get("cell") == expected, "Independent cell identity drifted.")
    flat = np.asarray(cell.get("flat_features"), dtype=np.float64)
    stair = np.asarray(cell.get("stair_features"), dtype=np.float64)
    flat_active = cell.get("flat_active")
    stair_active = cell.get("stair_active")
    impacts = np.asarray(cell.get("impact_steps"))
    _require(
      flat.shape == (REPLAY_DRIVE_STEPS, REPLAY_PAIRS_PER_CELL, 3)
      and stair.shape == flat.shape
      and isinstance(flat_active, np.ndarray)
      and isinstance(stair_active, np.ndarray)
      and flat_active.shape == flat.shape[:2]
      and stair_active.shape == stair.shape[:2]
      and flat_active.dtype == np.bool_
      and stair_active.dtype == np.bool_
      and np.all(flat_active)
      and np.all(stair_active)
      and impacts.shape == (REPLAY_PAIRS_PER_CELL,)
      and np.issubdtype(impacts.dtype, np.integer),
      "Independent cell arrays drifted.",
    )
    _require(
      np.all(impacts >= REPLAY_PRE_IMPACT_STEPS)
      and np.all(
        impacts + REPLAY_POST_IMPACT_STEPS < REPLAY_DRIVE_STEPS
      ),
      "Independent impact window is incomplete.",
    )
    counts = {"flat": 0, "pre": 0, "timely": 0, "late": 0, "missing": 0}
    cell_delays: list[int] = []
    for slot in range(REPLAY_PAIRS_PER_CELL):
      if _first_trigger(flat[:, slot], flat_active[:, slot], thresholds) is not None:
        counts["flat"] += 1
      trigger = _first_trigger(stair[:, slot], stair_active[:, slot], thresholds)
      if trigger is None:
        counts["missing"] += 1
        continue
      impact = int(impacts[slot])
      if trigger < impact:
        counts["pre"] += 1
        continue
      delay = trigger - impact
      if delay <= REPLAY_MAX_DELAY_TICKS:
        counts["timely"] += 1
        cell_delays.append(delay)
      else:
        counts["late"] += 1
    for name in totals:
      totals[name] += counts[name]
    delays.extend(cell_delays)
    cell_results.append({
      "cell_index": index,
      "flat_trigger_count": counts["flat"],
      "stair_pre_impact_trigger_count": counts["pre"],
      "timely_detection_count": counts["timely"],
      "late_detection_count": counts["late"],
      "missing_detection_count": counts["missing"],
      "timely_detection_rate": counts["timely"] / REPLAY_PAIRS_PER_CELL,
      "timely_delays_ticks": cell_delays,
    })
  qualified = (
    totals["flat"] == 0
    and totals["pre"] == 0
    and totals["timely"] >= REPLAY_OVERALL_TIMELY_MIN
    and all(
      cell["timely_detection_count"] >= REPLAY_PER_CELL_TIMELY_MIN
      for cell in cell_results
    )
  )
  return {
    "threshold_table_index": int(row["index"]),
    "thresholds": {
      name: float(value)
      for name, value in zip(REPLAY_FEATURE_NAMES, thresholds, strict=True)
    },
    "qualified": qualified,
    "flat_trigger_count": totals["flat"],
    "stair_pre_impact_trigger_count": totals["pre"],
    "timely_detection_count": totals["timely"],
    "timely_detection_rate": totals["timely"] / (
      len(registered) * REPLAY_PAIRS_PER_CELL
    ),
    "late_detection_count": totals["late"],
    "missing_detection_count": totals["missing"],
    "mean_timely_delay_ticks": (
      float(np.mean(np.asarray(delays, dtype=np.float64))) if delays else None
    ),
    "timely_delays_ticks": delays,
    "cells": cell_results,
  }


def _candidate_key(candidate: dict[str, Any]) -> tuple[float | int, ...]:
  thresholds = np.asarray(
    [candidate["thresholds"][name] for name in REPLAY_FEATURE_NAMES],
    dtype=np.float64,
  )
  return (
    -int(candidate["timely_detection_count"]),
    float(candidate["mean_timely_delay_ticks"]),
    float(thresholds[0]),
    float(thresholds[1]),
    float(thresholds[2]),
    int(candidate["threshold_table_index"]),
  )


def _select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
  qualified = [candidate for candidate in candidates if candidate["qualified"] is True]
  return min(qualified, key=_candidate_key) if qualified else None


def _selection(candidate: dict[str, Any]) -> dict[str, Any]:
  return {
    "threshold_table_index": int(candidate["threshold_table_index"]),
    "thresholds": dict(candidate["thresholds"]),
    "timely_detection_count": int(candidate["timely_detection_count"]),
    "timely_detection_rate": float(candidate["timely_detection_rate"]),
    "mean_timely_delay_ticks": float(candidate["mean_timely_delay_ticks"]),
  }


def _recompute_riser_truth(
  raw: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
  found = raw["stair_contact_found"]
  force = raw["stair_contact_force_contact_frame"]
  position = raw["stair_contact_pos_global"]
  normal = raw["stair_contact_normal_global"]
  outer_face_x = raw["stair_outer_face_x"]
  terrain_origin_x = raw["stair_terrain_origin_x"]
  _require(
    found.dtype == np.float32
    and found.ndim == 3
    and found.shape[:2] == (REPLAY_DRIVE_STEPS, REPLAY_PAIRS_PER_CELL)
    and found.shape[2] > 0,
    "Independent contact-found shape/dtype drifted.",
  )
  expected_vector_shape = (*found.shape, 3)
  for name, value in (
    ("force", force),
    ("position", position),
    ("normal", normal),
  ):
    _require(
      value.dtype == np.float32 and value.shape == expected_vector_shape,
      f"Independent contact-{name} shape/dtype drifted.",
    )
  for name, value in (
    ("outer-face", outer_face_x),
    ("terrain-origin", terrain_origin_x),
  ):
    _require(
      value.dtype == np.float32
      and value.shape == (REPLAY_PAIRS_PER_CELL,),
      f"Independent {name} shape/dtype drifted.",
    )
  finite_found = found[np.isfinite(found)]
  _require(
    np.all(finite_found >= 0.0)
    and np.all(finite_found == np.floor(finite_found)),
    "Independent raw contact values are invalid.",
  )
  face_error = np.abs(
    position[..., 0] - outer_face_x[np.newaxis, :, np.newaxis]
  )
  contact_mask = (
    found.astype(np.bool_)
    & (np.abs(normal[..., 0]) >= REPLAY_RISER_MIN_ABS_NORMAL_X)
    & (face_error <= REPLAY_RISER_FACE_X_TOLERANCE_M)
    & (np.abs(force[..., 0]) >= REPLAY_RISER_MIN_NORMAL_FORCE_N)
  )
  riser = np.any(contact_mask, axis=-1)
  impacts = np.full(REPLAY_PAIRS_PER_CELL, -1, dtype=np.int64)
  has_impact = np.any(riser, axis=0)
  impacts[has_impact] = np.argmax(riser[:, has_impact], axis=0)
  finite_geometry = np.isfinite(outer_face_x) & np.isfinite(terrain_origin_x)
  expected_outer_face_x = (
    terrain_origin_x + REPLAY_OUTER_FACE_OFFSET_FROM_TERRAIN_ORIGIN_M
  )
  contact_health = {
    "nonfinite_sample_count": sum(
      int(np.count_nonzero(~np.isfinite(value)))
      for value in (
        found, force, position, normal, outer_face_x, terrain_origin_x
      )
    ),
    "outer_face_binding_violation_count": int(np.count_nonzero(
      finite_geometry
      & (
        np.abs(outer_face_x - expected_outer_face_x)
        > REPLAY_GEOMETRY_WRITE_ATOL_M
      )
    )),
  }
  return riser, impacts, contact_health


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", type=Path)
  parser.add_argument("--predictor", type=Path, required=True)
  parser.add_argument("--transition-floor", type=Path, required=True)
  parser.add_argument("--expected-git-sha")
  parser.add_argument("--expected-mjlab-git-sha")
  parser.add_argument("--inputs-only", action="store_true")
  args = parser.parse_args(argv)
  if not args.inputs_only and (
    args.output_dir is None
    or args.expected_git_sha is None
    or args.expected_mjlab_git_sha is None
  ):
    parser.error("Full validation requires output-dir and both expected SHAs.")
  return args


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise ValueError(message)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _load_inputs(predictor_path: Path, floor_path: Path):
  predictor_payload = json.loads(predictor_path.read_text(encoding="utf-8-sig"))
  floor_payload = json.loads(floor_path.read_text(encoding="utf-8-sig"))
  parsed_predictor = parse_innovation_predictor(predictor_payload)
  predictor = _ReplayPredictor(predictor_payload)
  floor = parse_transition_floor(
    floor_payload, predictor_hash=parsed_predictor.predictor_hash
  )
  _require(parsed_predictor.predictor_hash == PREDICTOR_HASH, "Predictor hash drifted.")
  _require(floor["floor_hash"] == FLOOR_HASH, "Transition floor hash drifted.")
  _require(
    floor["threshold_table_hash"] == THRESHOLD_TABLE_HASH,
    "Threshold table hash drifted.",
  )
  return predictor, floor_payload, floor


def _expected_perturbations(cell_index: int) -> np.ndarray:
  generator = torch.Generator(device="cpu")
  generator.manual_seed(30_000 + cell_index)
  unit = 2.0 * torch.rand((QUALIFICATION_PAIRS_PER_CELL, 4), generator=generator) - 1.0
  bounds = torch.tensor(RESET_PERTURBATION_BOUNDS, dtype=torch.float32)
  return (unit * bounds).numpy()


def _validate_reset(
  raw: dict[str, np.ndarray], cell: dict[str, Any]
) -> dict[str, float]:
  perturbations = raw["reset_perturbations"]
  flat = raw["flat_reset_relative"]
  stair = raw["stair_reset_relative"]
  flat_written = raw["flat_written_reset_relative"]
  stair_written = raw["stair_written_reset_relative"]
  _require(perturbations.dtype == np.float32, "Reset perturbation dtype drifted.")
  _require(
    np.array_equal(perturbations, _expected_perturbations(int(cell["cell_index"]))),
    "Reset perturbations do not match the frozen CPU generator.",
  )
  _require(flat.shape == (16, 13) and stair.shape == (16, 13), "Reset shape drifted.")
  _require(np.array_equal(flat, stair), "Flat/stair paired resets are not identical.")
  _require(
    flat_written.shape == (16, 13) and stair_written.shape == (16, 13),
    "Written reset shape drifted.",
  )
  _require(
    all(
      np.all(np.isfinite(value))
      for value in (flat, stair, flat_written, stair_written)
    ),
    "Reset arrays must be finite.",
  )
  _require(
    np.allclose(flat[:, 0], -0.25 + perturbations[:, 0], rtol=0.0, atol=1.0e-7)
    and np.allclose(flat[:, 1], perturbations[:, 1], rtol=0.0, atol=1.0e-7)
    and np.allclose(flat[:, 2], cell["height_m"], rtol=0.0, atol=1.0e-7),
    "Reset translation drifted.",
  )
  half = 0.5 * float(cell["pitch_rad"])
  expected_quat = np.asarray([math.cos(half), 0.0, math.sin(half), 0.0])
  _require(
    np.allclose(flat[:, 3:7], expected_quat, rtol=0.0, atol=1.0e-7),
    "Reset pitch quaternion drifted.",
  )
  _require(
    np.allclose(flat[:, 7], perturbations[:, 2], rtol=0.0, atol=1.0e-7)
    and np.allclose(flat[:, 11], perturbations[:, 3], rtol=0.0, atol=1.0e-7)
    and np.array_equal(flat[:, [8, 9, 10, 12]], np.zeros((16, 4))),
    "Reset velocity semantics drifted.",
  )
  written_reset_error = float(max(
    np.max(np.abs(flat_written - flat)),
    np.max(np.abs(stair_written - stair)),
  ))
  written_paired_error = float(
    np.max(np.abs(flat_written - stair_written))
  )
  _require(
    written_reset_error <= QUALIFICATION_RESET_WRITE_ATOL
    and written_paired_error <= QUALIFICATION_RESET_WRITE_ATOL,
    "Written reset exceeds the float32 representation tolerance.",
  )
  quaternion = flat[:, 3:7].astype(np.float64)
  w, x, y, z = (quaternion[:, index] for index in range(4))
  roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
  pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
  yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  return {
    "paired_reset_max_abs_error": float(np.max(np.abs(flat - stair))),
    "written_reset_max_abs_error": written_reset_error,
    "written_paired_reset_max_abs_error": written_paired_error,
    "root_pitch_max_abs_error_rad": float(
      np.max(np.abs(pitch - float(cell["pitch_rad"])))
    ),
    "root_roll_yaw_max_abs_rad": float(
      np.max(np.abs(np.column_stack((roll, yaw))))
    ),
    "other_root_velocity_max_abs": float(
      np.max(np.abs(flat[:, [8, 9, 10, 12]]))
    ),
  }


def _validate_side(
  raw: dict[str, np.ndarray],
  *,
  prefix: str,
  predictor: Any,
  cell: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
  z = raw[f"{prefix}_z"]
  u = raw[f"{prefix}_u"]
  next_z = raw[f"{prefix}_next_z"]
  posture = raw[f"{prefix}_shaped_posture"]
  features = raw[f"{prefix}_features"]
  active = raw[f"{prefix}_active"]
  targets = raw[f"{prefix}_wheel_targets"]
  portable = raw[f"{prefix}_portable_targets"]
  specific_force = raw[f"{prefix}_specific_force_x"]
  gravity = raw[f"{prefix}_projected_gravity_x"]
  _require(z.shape == (500, 16, 2), f"{prefix} z shape drifted.")
  _require(u.shape == (500, 16, 1), f"{prefix} u shape drifted.")
  _require(next_z.shape == z.shape, f"{prefix} next-z shape drifted.")
  _require(posture.shape == (500, 16, 2), f"{prefix} posture shape drifted.")
  _require(features.shape == (500, 16, 3), f"{prefix} feature shape drifted.")
  _require(active.shape == (500, 16), f"{prefix} active shape drifted.")
  _require(targets.shape == (500, 16, 2), f"{prefix} target shape drifted.")
  _require(portable.shape == targets.shape, f"{prefix} portable-target shape drifted.")
  _require(specific_force.shape == (500, 16, 1), f"{prefix} force shape drifted.")
  _require(gravity.shape == (500, 16, 1), f"{prefix} gravity shape drifted.")
  _require(active.dtype == np.bool_ and np.all(active), f"{prefix} mask is not full true.")
  expected_posture = np.asarray([cell["height_m"], cell["pitch_rad"]])
  posture_violations = int(np.count_nonzero(
    np.max(np.abs(posture - expected_posture), axis=2)
    > QUALIFICATION_POSTURE_CAPTURE_ATOL
  ))
  portable_delta = np.abs(targets - portable)
  portable_finite = np.all(np.isfinite(portable_delta), axis=2)
  portable_max_error = (
    float(np.max(portable_delta)) if np.all(portable_finite) else None
  )
  portable_target_violations = int(np.count_nonzero(
    portable_finite
    & (np.max(np.where(np.isfinite(portable_delta), portable_delta, 0.0), axis=2)
       > QUALIFICATION_PORTABLE_EQUIVALENCE_ATOL)
  ))
  projected_u = 0.5 * (targets[:, :, 1] - targets[:, :, 0])
  _require(
    np.allclose(
      u[:, :, 0], projected_u, rtol=0.0, atol=0.0, equal_nan=True
    ),
    f"{prefix} u is not the applied wheel-target projection.",
  )
  predicted = np.empty_like(next_z)
  predicted.fill(np.nan)
  domain_violations = 0
  predictor_evaluation_errors = 0
  for tick in range(500):
    for slot in range(16):
      try:
        predicted[tick, slot] = predictor.predict(
          z[tick, slot],
          float(u[tick, slot, 0]),
          float(posture[tick, slot, 0]),
          float(posture[tick, slot, 1]),
        )
      except _ReplayInputDomainError:
        domain_violations += 1
      except _ReplayPostureDomainError:
        pass
      except ValueError:
        predictor_evaluation_errors += 1
  innovation = np.abs(next_z - predicted)
  # The deployment path computes this feature from float32 torch tensors and
  # only converts the archived inputs/result to float64 afterwards.  Replay
  # that arithmetic exactly; evaluating the archived inputs directly in
  # float64 creates ~1e-7 differences on untampered captures.
  specific_force_f32 = specific_force[:, :, 0].astype(np.float32)
  gravity_f32 = gravity[:, :, 0].astype(np.float32)
  deceleration = np.maximum(
    np.float32(0.0),
    -(specific_force_f32 + np.float32(GRAVITY_MPS2) * gravity_f32),
  ).astype(np.float64)
  _require(
    np.allclose(
      features[:, :, :2],
      innovation,
      rtol=0.0,
      atol=1.0e-12,
      equal_nan=True,
    )
    and np.allclose(
      features[:, :, 2],
      deceleration,
      rtol=0.0,
      atol=1.0e-12,
      equal_nan=True,
    ),
    f"{prefix} features do not reproduce from raw deployment signals.",
  )
  raw_numeric = (
    z, u, next_z, posture, features, targets, portable, specific_force, gravity
  )
  health = {
    "predictor_domain_violation_count": domain_violations,
    "posture_violation_count": posture_violations,
    "predictor_evaluation_error_count": predictor_evaluation_errors,
    "nonfinite_sample_count": sum(
      int(np.count_nonzero(~np.isfinite(value))) for value in raw_numeric
    ),
    "negative_feature_sample_count": int(np.count_nonzero(features < 0.0)),
    "portable_target_violation_count": portable_target_violations,
    "portable_max_abs_target_error_radps": portable_max_error,
  }
  return health, features


def _boolean_health(raw: dict[str, np.ndarray], name: str) -> int:
  value = raw[name]
  _require(value.shape == (16,) and value.dtype == np.bool_, f"{name} shape/dtype drifted.")
  return int(np.count_nonzero(value))


def validate_output(
  output_dir: Path,
  *,
  predictor: Any,
  floor_payload: dict[str, Any],
  floor: dict[str, Any],
  expected_git_sha: str,
  expected_mjlab_git_sha: str,
) -> dict[str, Any]:
  result_path = output_dir / "c2_innovation_detector_qualification.json"
  payload = json.loads(result_path.read_text(encoding="utf-8-sig"))
  classification = payload.get("classification")
  _require(classification in FORMAL_CLASSIFICATIONS, "Classification is invalid.")
  _require(payload.get("schema_version") == QUALIFICATION_SCHEMA_VERSION, "Schema drifted.")
  _require(payload.get("artifact_type") == QUALIFICATION_ARTIFACT_TYPE, "Artifact type drifted.")
  _require(
    payload.get("probe") == OFFICIAL_QUALIFICATION_PROTOCOL["probe"],
    "Probe identity drifted.",
  )
  _require(payload.get("protocol") == OFFICIAL_QUALIFICATION_PROTOCOL, "Protocol drifted.")
  registered_cells = _replay_cells()
  _require(
    qualification_cells() == registered_cells,
    "Core and independent cell registrations disagree.",
  )
  _require(
    QUALIFICATION_PAIRS_PER_CELL == REPLAY_PAIRS_PER_CELL
    and QUALIFICATION_DRIVE_STEPS == REPLAY_DRIVE_STEPS
    and QUALIFICATION_PRE_IMPACT_STEPS == REPLAY_PRE_IMPACT_STEPS
    and QUALIFICATION_POST_IMPACT_STEPS == REPLAY_POST_IMPACT_STEPS,
    "Core and independent replay dimensions disagree.",
  )
  impact_truth = payload["protocol"].get("impact_truth", {})
  _require(
    impact_truth.get("abs_normal_x_min") == REPLAY_RISER_MIN_ABS_NORMAL_X
    and impact_truth.get("face_distance_max_m")
    == REPLAY_RISER_FACE_X_TOLERANCE_M
    and impact_truth.get("abs_contact_frame_normal_force_min_n")
    == REPLAY_RISER_MIN_NORMAL_FORCE_N,
    "Core and independent first-riser criteria disagree.",
  )
  _require(
    impact_truth.get("archived_raw_replay") is True
    and impact_truth.get("outer_face_offset_from_terrain_origin_m")
    == REPLAY_OUTER_FACE_OFFSET_FROM_TERRAIN_ORIGIN_M
    and impact_truth.get("outer_face_binding_atol_m")
    == REPLAY_GEOMETRY_WRITE_ATOL_M,
    "Core and independent first-riser geometry binding disagree.",
  )
  _require(payload.get("git_sha") == expected_git_sha, "Git SHA drifted.")
  _require(payload.get("mjlab_git_sha") == expected_mjlab_git_sha, "MjLab SHA drifted.")
  _require(payload.get("predictor_hash") == PREDICTOR_HASH, "Predictor binding drifted.")
  _require(payload.get("floor_hash") == FLOOR_HASH, "Floor binding drifted.")
  _require(payload.get("threshold_table_hash") == THRESHOLD_TABLE_HASH, "Table binding drifted.")
  _require(payload.get("bindings") == EXPECTED_BINDINGS, "Deployment bindings drifted.")
  _require(
    payload.get("detector_hash") == canonical_hash(payload, hash_field="detector_hash"),
    "Detector result self-hash drifted.",
  )
  cells = payload.get("cells")
  _require(isinstance(cells, list) and len(cells) == 18, "Result requires 18 cells.")
  expected_raw_files = [f"cell_{index:02d}.npz" for index in range(18)]
  _require(
    sorted(path.name for path in output_dir.glob("cell_*.npz"))
    == expected_raw_files,
    "Raw cell file set drifted.",
  )
  evaluation_cells: list[dict[str, Any]] = []
  any_invalid = False
  for index, (summary, registered) in enumerate(
    zip(cells, registered_cells, strict=True)
  ):
    expected_summary_fields = {
      "cell",
      "raw_file",
      "raw_sha256",
      "raw_shape",
      "impact_steps",
      "diagnostic_windows",
      "paired_reset_max_abs_error",
      "written_reset_max_abs_error",
      "written_paired_reset_max_abs_error",
      "root_pitch_max_abs_error_rad",
      "root_roll_yaw_max_abs_rad",
      "other_root_velocity_max_abs",
      "portable_max_abs_target_error_radps",
      "health",
    }
    _require(
      isinstance(summary, dict) and set(summary) == expected_summary_fields,
      "Cell summary field set drifted.",
    )
    _require(summary.get("cell") == registered, "Cell identity/order drifted.")
    raw_path = output_dir / f"cell_{index:02d}.npz"
    _require(summary.get("raw_file") == raw_path.name, "Cell raw filename drifted.")
    _require(summary.get("raw_sha256") == _sha256(raw_path), "Cell raw hash drifted.")
    with np.load(raw_path, allow_pickle=False) as loaded:
      _require(set(loaded.files) == RAW_KEYS, "Cell raw field set drifted.")
      raw = {name: loaded[name] for name in loaded.files}
    reset_metrics = _validate_reset(raw, registered)
    flat_health, flat_features = _validate_side(
      raw, prefix="flat", predictor=predictor, cell=registered
    )
    stair_health, stair_features = _validate_side(
      raw, prefix="stair", predictor=predictor, cell=registered
    )
    impacts = raw["impact_steps"]
    riser = raw["stair_riser_contact"]
    _require(
      impacts.shape == (16,) and np.issubdtype(impacts.dtype, np.integer),
      "Impact table shape/dtype drifted.",
    )
    _require(riser.shape == (500, 16) and riser.dtype == np.bool_, "Riser mask drifted.")
    recomputed_riser, recomputed_impact, contact_health = (
      _recompute_riser_truth(raw)
    )
    _require(
      np.array_equal(riser, recomputed_riser),
      "Stored first-riser mask does not match raw contact fields.",
    )
    _require(np.array_equal(impacts, recomputed_impact), "First-riser impact drifted.")
    expected_windows = [
      {
        "slot": slot,
        "start_tick": int(impact - QUALIFICATION_PRE_IMPACT_STEPS),
        "impact_tick": int(impact),
        "end_tick": int(impact + QUALIFICATION_POST_IMPACT_STEPS),
      }
      for slot, impact in enumerate(impacts)
      if impact >= 0
    ]
    _require(summary.get("raw_shape") == [500, 16], "Cell raw shape summary drifted.")
    _require(summary.get("impact_steps") == impacts.tolist(), "Impact summary drifted.")
    _require(
      summary.get("diagnostic_windows") == expected_windows,
      "Diagnostic-window summary drifted.",
    )
    window_valid = (
      (impacts >= QUALIFICATION_PRE_IMPACT_STEPS)
      & (impacts + QUALIFICATION_POST_IMPACT_STEPS < QUALIFICATION_DRIVE_STEPS)
    )
    health_counts = {
      "flat_termination_count": _boolean_health(raw, "flat_terminated"),
      "stair_termination_count": _boolean_health(raw, "stair_terminated"),
      "flat_timeout_count": _boolean_health(raw, "flat_timeout"),
      "stair_timeout_count": _boolean_health(raw, "stair_timeout"),
      "flat_non_wheel_contact_count": _boolean_health(raw, "flat_non_wheel_contact"),
      "stair_non_wheel_contact_count": _boolean_health(raw, "stair_non_wheel_contact"),
      "settle_riser_contact_count": (
        _boolean_health(raw, "flat_settle_riser_contact")
        + _boolean_health(raw, "stair_settle_riser_contact")
      ),
      "drive_start_past_face_count": (
        _boolean_health(raw, "flat_drive_start_past_face")
        + _boolean_health(raw, "stair_drive_start_past_face")
      ),
      "missing_impact_count": int(np.count_nonzero(impacts < 0)),
      "invalid_window_count": int(np.count_nonzero(~window_valid)),
      "predictor_domain_violation_count": (
        flat_health["predictor_domain_violation_count"]
        + stair_health["predictor_domain_violation_count"]
      ),
      "posture_violation_count": (
        flat_health["posture_violation_count"]
        + stair_health["posture_violation_count"]
      ),
      "predictor_evaluation_error_count": (
        flat_health["predictor_evaluation_error_count"]
        + stair_health["predictor_evaluation_error_count"]
      ),
      "nonfinite_sample_count": (
        flat_health["nonfinite_sample_count"]
        + stair_health["nonfinite_sample_count"]
        + contact_health["nonfinite_sample_count"]
      ),
      "negative_feature_sample_count": (
        flat_health["negative_feature_sample_count"]
        + stair_health["negative_feature_sample_count"]
      ),
      "portable_target_violation_count": (
        flat_health["portable_target_violation_count"]
        + stair_health["portable_target_violation_count"]
      ),
      "outer_face_binding_violation_count": contact_health[
        "outer_face_binding_violation_count"
      ],
    }
    _require(summary.get("health") == health_counts, "Cell health summary drifted.")
    for name, value in reset_metrics.items():
      observed = summary.get(name)
      replay_atol = (
        1.0e-9
        if name in {
          "root_pitch_max_abs_error_rad",
          "root_roll_yaw_max_abs_rad",
        }
        else 1.0e-12
      )
      _require(
        isinstance(observed, (int, float))
        and not isinstance(observed, bool)
        and math.isclose(
          float(observed), float(value), rel_tol=0.0, abs_tol=replay_atol
        ),
        f"{name} summary drifted.",
      )
    portable_maxima = (
      flat_health["portable_max_abs_target_error_radps"],
      stair_health["portable_max_abs_target_error_radps"],
    )
    portable_maximum = (
      None if any(value is None for value in portable_maxima)
      else max(float(value) for value in portable_maxima)
    )
    observed_portable_maximum = summary.get(
      "portable_max_abs_target_error_radps"
    )
    _require(
      observed_portable_maximum is None
      if portable_maximum is None
      else (
        isinstance(observed_portable_maximum, (int, float))
        and not isinstance(observed_portable_maximum, bool)
        and math.isclose(
          float(observed_portable_maximum),
          portable_maximum,
          rel_tol=0.0,
          abs_tol=1.0e-12,
        )
      ),
      "Portable-target maximum summary drifted.",
    )
    raw_invalid = (
      any(health_counts.values())
      or reset_metrics["paired_reset_max_abs_error"] != 0.0
      or reset_metrics["written_reset_max_abs_error"]
      > QUALIFICATION_RESET_WRITE_ATOL
      or reset_metrics["written_paired_reset_max_abs_error"]
      > QUALIFICATION_RESET_WRITE_ATOL
      or reset_metrics["root_pitch_max_abs_error_rad"]
      > QUALIFICATION_POSTURE_CAPTURE_ATOL
      or reset_metrics["root_roll_yaw_max_abs_rad"]
      > QUALIFICATION_POSTURE_CAPTURE_ATOL
      or reset_metrics["other_root_velocity_max_abs"] != 0.0
    )
    any_invalid = any_invalid or raw_invalid
    evaluation_cells.append({
      "cell": registered,
      "flat_features": flat_features,
      "stair_features": stair_features,
      "flat_active": raw["flat_active"],
      "stair_active": raw["stair_active"],
      "impact_steps": impacts,
    })

  candidates: list[dict[str, Any]] = []
  selected = None
  if not any_invalid:
    table = floor["threshold_table"]
    for index, row in enumerate(table):
      if index % 10 == 0 or index == len(table) - 1:
        print(f"[C2-j3 validator] replay candidate {index}/{len(table) - 1}")
      candidates.append(_evaluate_candidate(row, evaluation_cells))
    selected = _select_candidate(candidates)
  expected_classification = (
    "INVALID_INNOVATION_CAPTURE"
    if any_invalid
    else "INNOVATION_DETECTOR_QUALIFIED"
    if selected is not None
    else "C2_INNOVATION_DETECTOR_UNQUALIFIED_STOP"
  )
  _require(classification == expected_classification, "Classification does not match raw data.")
  _require(payload.get("candidates") == candidates, "Candidate replay drifted.")
  _require(
    payload.get("selected_candidate")
    == (None if selected is None else _selection(selected)),
    "Selected candidate drifted.",
  )
  _require(payload.get("completed_cell_count") == 18, "Completed-cell count drifted.")
  _require(payload.get("completed_pair_count") == 288, "Completed-pair count drifted.")
  _require(
    payload.get("completed_candidate_count") == len(candidates),
    "Completed-candidate count drifted.",
  )
  _require(
    payload.get("qualified_candidate_count")
    == sum(candidate["qualified"] is True for candidate in candidates),
    "Qualified-candidate count drifted.",
  )
  expected_next_step = {
    "INNOVATION_DETECTOR_QUALIFIED": "FREEZE_AND_INDEPENDENT_AUDIT_BEFORE_C3",
    "C2_INNOVATION_DETECTOR_UNQUALIFIED_STOP": "STOP_FOR_USER_ROUTE_DECISION",
    "INVALID_INNOVATION_CAPTURE": "INDEPENDENT_IMPLEMENTATION_DIAGNOSIS_ONLY",
  }[classification]
  _require(payload.get("next_step") == expected_next_step, "Next-step field drifted.")
  _require(
    payload.get("evidence_eligible") is (classification != "INVALID_INNOVATION_CAPTURE")
    and payload.get("promotion_eligible") is False
    and payload.get("training_eligible") is False
    and payload.get("checkpoint") is None,
    "Eligibility flags drifted.",
  )
  if classification == "INNOVATION_DETECTOR_QUALIFIED":
    parse_innovation_detector_qualification(
      payload, predictor_hash=PREDICTOR_HASH, floor_payload=floor_payload
    )
  return payload


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  predictor, floor_payload, floor = _load_inputs(
    args.predictor, args.transition_floor
  )
  if args.inputs_only:
    print("INPUT_VALIDATION=PASS")
    return
  payload = validate_output(
    args.output_dir,
    predictor=predictor,
    floor_payload=floor_payload,
    floor=floor,
    expected_git_sha=args.expected_git_sha,
    expected_mjlab_git_sha=args.expected_mjlab_git_sha,
  )
  print("VALIDATION=PASS")
  print(f"CLASSIFICATION={payload['classification']}")
  print(f"DETECTOR_HASH={payload['detector_hash']}")


if __name__ == "__main__":
  main()
