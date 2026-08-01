"""Deployment-visible affine predictor used by the C2 innovation detector."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .controller_schedule import bilinear_interpolate, canonical_hash

PREDICTOR_ARTIFACT_TYPE = "c2_innovation_predictor"
PREDICTOR_SCHEMA_VERSION = 1
FLOOR_ARTIFACT_TYPE = "c2_innovation_transition_floor"
FLOOR_SCHEMA_VERSION = 1
QUALIFICATION_ARTIFACT_TYPE = "c2_innovation_detector_qualification"
QUALIFICATION_SCHEMA_VERSION = 1
PREDICTOR_STATE_NAMES = ("imu_pitch_rate_radps", "signed_wheel_speed_radps")
REGISTERED_HEIGHT_NODES = (0.2907321708, 0.3092089487, 0.3276857266)
REGISTERED_PITCH_NODES = (-0.032, 0.0, 0.032)
GRID_SHA256 = "3ba8c0f13667c430c02f4ffdeedcffd97da0e779758b0cb05a86c5fcc09ef628"
RANK_ABSOLUTE_TOLERANCE = 1.0e-12
DOMAIN_TOLERANCE_RADPS = 1.0e-6
EXPECTED_BINDINGS: dict[str, str | None] = {
  "controller_schedule_hash": "8fe8548bca85978c164bbd7de39d2d6463cdfd8d7ab91796cf57696b0f64e203",
  "identification_controller_gain_hash": "8fee25a0339dd1e99127cbed912941dc3ad8ef2030ce49a0d310d1563cb87d98",
  "velocity_calibration_hash": "f62648b57bd17a3503bcbdbf58f349f91fcd8de8ef0cf04551c200401233ed01",
  "posture_artifact_hash": "3b96fd3dae66ad781b5b875c74184db101c42da02c53dfcc40a5137a6b5de11a",
  "station_calibration_hash": "c00e859b3093b4812d54799253accdaeb99171a2cf4028b08bc39e68eaaa7d8a",
  "yaw_calibration_hash": None,
}
OFFICIAL_IDENTIFICATION_PROTOCOL: dict[str, Any] = {
  "probe": "hybrid_c2_predictor_identification_v1",
  "seed": 1,
  "device": "cuda:0",
  "height_nodes": list(REGISTERED_HEIGHT_NODES),
  "pitch_nodes": list(REGISTERED_PITCH_NODES),
  "grid_sha256": GRID_SHA256,
  "num_envs": 32,
  "fit_envs": list(range(24)),
  "heldout_envs": list(range(24, 32)),
  "warmup_steps": 250,
  "collection_steps": 2500,
  "control_dt_s": 0.02,
  "prbs": {
    "bit_generator": "numpy.PCG64",
    "seed_formula": "1+1000*node_index+env_index",
    "draw_count": 550,
    "dtype": "uint8",
    "levels_vx_mps": [0.0, 0.10],
    "hold_ticks": 5,
    "collection_stream_ticks": [250, 2749],
  },
  "residual_action": [0.0] * 6,
  "yaw_command": 0.0,
  "evidence_eligible": False,
  "detector_fit_eligible": False,
  "promotion_eligible": False,
  "training_eligible": False,
}
THRESHOLD_FACTORS = (1.05, 1.25, 1.5, 2.0, 3.0)
FEATURE_NAMES = (
  "pitch_rate_innovation_radps",
  "wheel_speed_innovation_radps",
  "forward_deceleration_mps2",
)
TRANSITION_CENTER = (0.3092089487, 0.0, 0.07)
QUALIFICATION_VELOCITIES = (0.07, 0.10)
QUALIFICATION_PAIRS_PER_CELL = 16
QUALIFICATION_SETTLE_STEPS = 200
QUALIFICATION_DRIVE_STEPS = 500
QUALIFICATION_PRE_IMPACT_STEPS = 25
QUALIFICATION_POST_IMPACT_STEPS = 75
QUALIFICATION_MAX_DELAY_TICKS = 3
QUALIFICATION_OVERALL_TIMELY_MIN = 274
QUALIFICATION_PER_CELL_TIMELY_MIN = 15
RESET_PERTURBATION_BOUNDS = (0.02, 0.03, 0.01, 0.02)
PREDICTOR_POSTURE_DOMAIN_ATOL = 1.0e-7
QUALIFICATION_PORTABLE_EQUIVALENCE_ATOL = 2.0e-5
QUALIFICATION_POSTURE_CAPTURE_ATOL = PREDICTOR_POSTURE_DOMAIN_ATOL
QUALIFICATION_RESET_WRITE_ATOL = 2.0e-5
QUALIFICATION_GEOMETRY_WRITE_ATOL = QUALIFICATION_RESET_WRITE_ATOL
QUALIFICATION_OUTER_FACE_OFFSET_FROM_TERRAIN_ORIGIN_M = -3.0
QUALIFICATION_WHEEL_VELOCITY_LIMIT_RADPS = 12.0
QUALIFICATION_WHEEL_SLEW_RADPS_PER_TICK = 6.0
QUALIFICATION_POSTURE_HEIGHT_SLEW_RATE_MPS = 0.01215
QUALIFICATION_POSTURE_PITCH_SLEW_RATE_RADPS = 0.07755


def transition_floor_cells() -> list[dict[str, Any]]:
  cells: list[dict[str, Any]] = [
    {"name": "pitch_zero", "kind": "constant", "target": list(TRANSITION_CENTER)},
    {
      "name": "fast_lean_0p032",
      "kind": "constant",
      "target": [TRANSITION_CENTER[0], -0.032, 0.10],
    },
  ]
  for height in (REGISTERED_HEIGHT_NODES[0], REGISTERED_HEIGHT_NODES[-1]):
    for pitch in (REGISTERED_PITCH_NODES[0], REGISTERED_PITCH_NODES[-1]):
      for vx in (0.0, 0.10):
        cells.append({
          "name": f"corner_h{height:.10f}_p{pitch:+.3f}_v{vx:.2f}",
          "kind": "transition",
          "target": [height, pitch, vx],
        })
  return cells


OFFICIAL_TRANSITION_FLOOR_PROTOCOL: dict[str, Any] = {
  "probe": "hybrid_c2_transition_floor_v1",
  "seed": 2,
  "device": "cuda:0",
  "cells": transition_floor_cells(),
  "envs_per_cell": 16,
  "settle_steps": 200,
  "drive_steps": 500,
  "settle_raw_command": list(TRANSITION_CENTER),
  "transition_schedule": {
    "outward_ticks": [0, 79],
    "hold_ticks": [80, 419],
    "return_ticks": [420, 499],
  },
  "height_slew_rate_mps": 0.01215,
  "pitch_slew_rate_radps": 0.07755,
  "wheel_slew_radps_per_tick": 6.0,
  "activation": "integrated_signed_wheel_odometry_lt_0p35m",
  "first_tick_no_vote": True,
  "threshold_factors": list(THRESHOLD_FACTORS),
  "evidence_eligible": False,
  "detector_fit_eligible": False,
  "promotion_eligible": False,
  "training_eligible": False,
}


def qualification_cells() -> list[dict[str, Any]]:
  cells: list[dict[str, Any]] = []
  for height_index, height in enumerate(REGISTERED_HEIGHT_NODES):
    for pitch_index, pitch in enumerate(REGISTERED_PITCH_NODES):
      node_index = 3 * height_index + pitch_index
      for vx_index, vx in enumerate(QUALIFICATION_VELOCITIES):
        cell_index = 2 * node_index + vx_index
        cells.append({
          "cell_index": cell_index,
          "node_index": node_index,
          "height_index": height_index,
          "pitch_index": pitch_index,
          "vx_index": vx_index,
          "height_m": height,
          "pitch_rad": pitch,
          "vx_mps": vx,
        })
  return cells


OFFICIAL_QUALIFICATION_PROTOCOL: dict[str, Any] = {
  "probe": "hybrid_c2_innovation_qualification_v1",
  "task": "HopperTrex-Hybrid-v2-Stage5",
  "seed": 3,
  "device": "cuda:0",
  "control_dt_s": 0.02,
  "cells": qualification_cells(),
  "pairs_per_cell": QUALIFICATION_PAIRS_PER_CELL,
  "terrain": {
    "flat_height_m": 0.0,
    "stair_height_m": 0.01,
    "stair_geometry": "single_first_riser_shared_c0",
  },
  "reset": {
    "start_offset_m": 0.25,
    "paired_identical": True,
    "generator": "torch.Generator(device='cpu')",
    "seed_formula": "30000+cell_index",
    "draw": "2*torch.rand((16,4))-1",
    "dtype": "torch.float32",
    "fields": [
      "x_offset_m",
      "y_offset_m",
      "root_vx_mps",
      "root_pitch_rate_radps",
    ],
    "bounds": list(RESET_PERTURBATION_BOUNDS),
    "other_root_velocities_zero": True,
    "root_height_source": "cell.height_m",
    "root_pitch_source": "cell.pitch_rad",
    "canonical_relative_reset": "bit_exact",
    "written_relative_atol": QUALIFICATION_RESET_WRITE_ATOL,
  },
  "settle_steps": QUALIFICATION_SETTLE_STEPS,
  "drive_steps": QUALIFICATION_DRIVE_STEPS,
  "settle_vx_mps": 0.0,
  "drive_command": "constant_cell_vx_height_pitch",
  "residual_action": [0.0] * 6,
  "yaw_command": 0.0,
  "runtime_assertions": {
    "action_delay_steps": 0,
    "sensor_noise_std": 0.0,
    "wheel_velocity_limit_radps": QUALIFICATION_WHEEL_VELOCITY_LIMIT_RADPS,
    "wheel_slew_radps_per_tick": QUALIFICATION_WHEEL_SLEW_RADPS_PER_TICK,
    "posture_height_slew_rate_mps": QUALIFICATION_POSTURE_HEIGHT_SLEW_RATE_MPS,
    "posture_pitch_slew_rate_radps": QUALIFICATION_POSTURE_PITCH_SLEW_RATE_RADPS,
    "portable_target_atol_radps": QUALIFICATION_PORTABLE_EQUIVALENCE_ATOL,
    "posture_capture_atol": QUALIFICATION_POSTURE_CAPTURE_ATOL,
    "posture_boundary_snap_atol": PREDICTOR_POSTURE_DOMAIN_ATOL,
  },
  "attempt_mask": "full_true",
  "first_tick_no_vote": True,
  "pre_impact_steps": QUALIFICATION_PRE_IMPACT_STEPS,
  "post_impact_steps": QUALIFICATION_POST_IMPACT_STEPS,
  "impact_truth": {
    "implementation": "shared_c0_first_riser_contact",
    "archived_raw_replay": True,
    "outer_face_offset_from_terrain_origin_m": (
      QUALIFICATION_OUTER_FACE_OFFSET_FROM_TERRAIN_ORIGIN_M
    ),
    "outer_face_binding_atol_m": QUALIFICATION_GEOMETRY_WRITE_ATOL,
    "abs_normal_x_min": 0.25,
    "face_distance_max_m": 0.02,
    "abs_contact_frame_normal_force_min_n": 1.0,
  },
  "voting": {
    "rule": "at_least_2_of_3",
    "consecutive_ticks": 2,
    "max_delay_ticks": QUALIFICATION_MAX_DELAY_TICKS,
  },
  "qualification": {
    "flat_trigger_count": 0,
    "stair_pre_impact_trigger_count": 0,
    "overall_timely_min": QUALIFICATION_OVERALL_TIMELY_MIN,
    "overall_stair_attempts": 288,
    "per_cell_timely_min": QUALIFICATION_PER_CELL_TIMELY_MIN,
    "per_cell_stair_attempts": QUALIFICATION_PAIRS_PER_CELL,
  },
  "evidence_eligible": True,
  "promotion_eligible": False,
  "training_eligible": False,
}


@dataclass(frozen=True)
class InnovationVoteState:
  consecutive_hits: int = 0
  triggered: bool = False


def innovation_vote_step(
  state: InnovationVoteState,
  features: ArrayLike,
  thresholds: ArrayLike,
  *,
  active: bool = True,
  vote_allowed: bool = True,
  consecutive_ticks: int = 2,
) -> tuple[bool, InnovationVoteState, tuple[bool, bool, bool]]:
  values = np.asarray(features, dtype=np.float64)
  limits = np.asarray(thresholds, dtype=np.float64)
  if (
    values.shape != (3,)
    or limits.shape != (3,)
    or not np.all(np.isfinite(values))
    or not np.all(np.isfinite(limits))
    or np.any(values < 0.0)
    or np.any(limits <= 0.0)
  ):
    raise ValueError("Innovation vote inputs must be finite and nonnegative.")
  if consecutive_ticks < 1:
    raise ValueError("Innovation detector consecutive_ticks must be positive.")
  votes = tuple(bool(value >= limit) for value, limit in zip(values, limits, strict=True))
  if not active:
    return False, InnovationVoteState(), votes
  if not vote_allowed:
    return state.triggered, InnovationVoteState(0, state.triggered), votes
  hits = state.consecutive_hits + 1 if sum(votes) >= 2 else 0
  triggered = state.triggered or hits >= consecutive_ticks
  return triggered, InnovationVoteState(hits, triggered), votes


def signed_balance_channel(values: ArrayLike) -> NDArray[np.float64]:
  array = np.asarray(values, dtype=np.float64)
  if array.ndim < 1 or array.shape[-1] != 2:
    raise ValueError("Wheel values must end in [left, right].")
  return 0.5 * (array[..., 1] - array[..., 0])


def velocity_prbs(node_index: int, env_index: int) -> NDArray[np.float64]:
  """Return the frozen 2750-tick raw-vx stream for one node/environment."""

  if not 0 <= node_index < 9 or not 0 <= env_index < 32:
    raise ValueError("PRBS node/env index is outside the registered grid.")
  rng = np.random.Generator(np.random.PCG64(1 + 1000 * node_index + env_index))
  bits = rng.integers(0, 2, size=550, dtype=np.uint8)
  return np.repeat(bits.astype(np.float64) * 0.10, 5)


def regression_rank(design: ArrayLike) -> int:
  x = np.asarray(design, dtype=np.float64)
  if x.ndim != 2 or not np.all(np.isfinite(x)):
    raise ValueError("Regression design must be a finite matrix.")
  singular = np.linalg.svd(x, compute_uv=False)
  relative = max(x.shape) * np.finfo(np.float64).eps * float(singular[0])
  tolerance = max(RANK_ABSOLUTE_TOLERANCE, relative)
  return int(np.count_nonzero(singular > tolerance))


def output_nrmse(actual: ArrayLike, predicted: ArrayLike) -> NDArray[np.float64]:
  y = np.asarray(actual, dtype=np.float64)
  y_hat = np.asarray(predicted, dtype=np.float64)
  if y.ndim != 2 or y.shape != y_hat.shape or y.shape[1] != 2:
    raise ValueError("NRMSE arrays must have matching shape (samples, 2).")
  rmse = np.sqrt(np.mean(np.square(y_hat - y), axis=0))
  span = np.ptp(y, axis=0)
  epsilon = np.finfo(np.float64).eps
  result = np.empty(2, dtype=np.float64)
  varying = span > epsilon
  result[varying] = rmse[varying] / span[varying]
  result[~varying] = np.where(rmse[~varying] <= epsilon, 0.0, np.inf)
  return result


def fit_predictor_node(
  fit_z: ArrayLike,
  fit_u: ArrayLike,
  fit_next_z: ArrayLike,
  heldout_z: ArrayLike,
  heldout_u: ArrayLike,
  heldout_next_z: ArrayLike,
) -> dict[str, Any]:
  arrays = [
    np.asarray(value, dtype=np.float64)
    for value in (fit_z, fit_u, fit_next_z, heldout_z, heldout_u, heldout_next_z)
  ]
  x, u, y, x_hold, u_hold, y_hold = arrays
  if x.ndim != 2 or x.shape[1] != 2 or y.shape != x.shape:
    raise ValueError("Fit state arrays must have shape (samples, 2).")
  if u.shape != (x.shape[0], 1):
    raise ValueError("Fit input must have shape (samples, 1).")
  if x_hold.ndim != 2 or x_hold.shape[1] != 2 or y_hold.shape != x_hold.shape:
    raise ValueError("Heldout state arrays must have shape (samples, 2).")
  if u_hold.shape != (x_hold.shape[0], 1):
    raise ValueError("Heldout input must have shape (samples, 1).")
  if not all(np.all(np.isfinite(value)) for value in arrays):
    raise ValueError("Predictor transitions must be finite.")
  design = np.column_stack((x, u, np.ones(x.shape[0], dtype=np.float64)))
  rank = regression_rank(design)
  coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
  prediction = np.column_stack(
    (x_hold, u_hold, np.ones(x_hold.shape[0], dtype=np.float64))
  ) @ coefficients
  nrmse = output_nrmse(y_hold, prediction)
  return {
    "a": coefficients[0:2, :].T.tolist(),
    "b": coefficients[2:3, :].T.tolist(),
    "c": coefficients[3, :].tolist(),
    "regression_rank": rank,
    "heldout_nrmse": nrmse.tolist(),
    "fit_u_min_radps": float(np.min(u)),
    "fit_u_max_radps": float(np.max(u)),
  }


def threshold_table(maxima: ArrayLike) -> list[dict[str, float | int]]:
  values = np.asarray(maxima, dtype=np.float64)
  if values.shape != (3,) or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
    raise ValueError("Innovation floor maxima must contain three positive values.")
  rows: list[dict[str, float | int]] = []
  index = 0
  for pitch_factor in THRESHOLD_FACTORS:
    for wheel_factor in THRESHOLD_FACTORS:
      for deceleration_factor in THRESHOLD_FACTORS:
        rows.append({
          "index": index,
          FEATURE_NAMES[0]: float(values[0] * pitch_factor),
          FEATURE_NAMES[1]: float(values[1] * wheel_factor),
          FEATURE_NAMES[2]: float(values[2] * deceleration_factor),
        })
        index += 1
  return rows


def threshold_table_hash(rows: list[dict[str, float | int]]) -> str:
  import hashlib
  import json

  encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("ascii")
  return hashlib.sha256(encoded).hexdigest()


def _candidate_thresholds(row: dict[str, Any]) -> NDArray[np.float64]:
  if set(row) != {"index", *FEATURE_NAMES}:
    raise ValueError("Innovation threshold row has an invalid field set.")
  if isinstance(row["index"], bool) or not isinstance(row["index"], int):
    raise TypeError("Innovation threshold index must be an integer.")
  values = np.asarray([row[name] for name in FEATURE_NAMES], dtype=np.float64)
  if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
    raise ValueError("Innovation thresholds must be finite and positive.")
  return values


def _first_trigger_tick(
  features: NDArray[np.float64],
  active: NDArray[np.bool_],
  thresholds: NDArray[np.float64],
  *,
  consecutive_ticks: int = 2,
) -> int | None:
  if features.ndim != 2 or features.shape[1] != 3:
    raise ValueError("Innovation feature sequence must have shape (ticks, 3).")
  if active.shape != (features.shape[0],):
    raise ValueError("Innovation activation mask must have shape (ticks,).")
  if not np.all(np.isfinite(features)) or np.any(features < 0.0):
    raise ValueError("Innovation feature sequence must be finite and nonnegative.")
  if consecutive_ticks < 1:
    raise ValueError("Innovation detector consecutive_ticks must be positive.")
  state = InnovationVoteState()
  for tick in range(features.shape[0]):
    triggered, state, _votes = innovation_vote_step(
      state,
      features[tick],
      thresholds,
      active=bool(active[tick]),
      vote_allowed=tick != 0,
      consecutive_ticks=consecutive_ticks,
    )
    if triggered:
      return tick
  return None


def evaluate_qualification_candidate(
  row: dict[str, Any],
  cells: list[dict[str, Any]],
) -> dict[str, Any]:
  """Evaluate one frozen threshold row over all 18 seed-3 cells."""

  thresholds = _candidate_thresholds(row)
  registered = qualification_cells()
  if len(cells) != len(registered):
    raise ValueError("Innovation qualification requires exactly 18 cells.")
  flat_trigger_count = 0
  pre_impact_trigger_count = 0
  timely_count = 0
  late_count = 0
  missing_count = 0
  timely_delays: list[int] = []
  cell_results: list[dict[str, Any]] = []
  for cell_index, (cell, expected) in enumerate(zip(cells, registered, strict=True)):
    if cell.get("cell") != expected:
      raise ValueError("Innovation qualification cell order or identity drifted.")
    flat = np.asarray(cell.get("flat_features"), dtype=np.float64)
    stair = np.asarray(cell.get("stair_features"), dtype=np.float64)
    flat_active_raw = cell.get("flat_active")
    stair_active_raw = cell.get("stair_active")
    if (
      not isinstance(flat_active_raw, np.ndarray)
      or not isinstance(stair_active_raw, np.ndarray)
      or flat_active_raw.dtype != np.bool_
      or stair_active_raw.dtype != np.bool_
    ):
      raise ValueError("Innovation qualification masks must have Boolean dtype.")
    flat_active = flat_active_raw
    stair_active = stair_active_raw
    impacts = np.asarray(cell.get("impact_steps"))
    expected_shape = (
      QUALIFICATION_DRIVE_STEPS,
      QUALIFICATION_PAIRS_PER_CELL,
      3,
    )
    expected_mask_shape = expected_shape[:2]
    if flat.shape != expected_shape or stair.shape != expected_shape:
      raise ValueError("Innovation qualification features have invalid shape.")
    if flat_active.shape != expected_mask_shape or stair_active.shape != expected_mask_shape:
      raise ValueError("Innovation qualification masks have invalid shape.")
    if not np.all(flat_active) or not np.all(stair_active):
      raise ValueError("Formal innovation qualification requires full-true masks.")
    if impacts.shape != (QUALIFICATION_PAIRS_PER_CELL,):
      raise ValueError("Innovation qualification impact table has invalid shape.")
    if not np.issubdtype(impacts.dtype, np.integer):
      raise ValueError("Innovation qualification impact ticks must be integers.")
    if np.any(impacts < QUALIFICATION_PRE_IMPACT_STEPS) or np.any(
      impacts + QUALIFICATION_POST_IMPACT_STEPS >= QUALIFICATION_DRIVE_STEPS
    ):
      raise ValueError("Innovation qualification impact window is incomplete.")

    cell_flat = 0
    cell_pre = 0
    cell_timely = 0
    cell_late = 0
    cell_missing = 0
    cell_delays: list[int] = []
    for slot in range(QUALIFICATION_PAIRS_PER_CELL):
      flat_trigger = _first_trigger_tick(
        flat[:, slot], flat_active[:, slot], thresholds
      )
      if flat_trigger is not None:
        cell_flat += 1
      stair_trigger = _first_trigger_tick(
        stair[:, slot], stair_active[:, slot], thresholds
      )
      if stair_trigger is None:
        cell_missing += 1
        continue
      impact = int(impacts[slot])
      if stair_trigger < impact:
        cell_pre += 1
        continue
      delay = stair_trigger - impact
      if delay <= QUALIFICATION_MAX_DELAY_TICKS:
        cell_timely += 1
        cell_delays.append(delay)
      else:
        cell_late += 1
    flat_trigger_count += cell_flat
    pre_impact_trigger_count += cell_pre
    timely_count += cell_timely
    late_count += cell_late
    missing_count += cell_missing
    timely_delays.extend(cell_delays)
    cell_results.append({
      "cell_index": cell_index,
      "flat_trigger_count": cell_flat,
      "stair_pre_impact_trigger_count": cell_pre,
      "timely_detection_count": cell_timely,
      "late_detection_count": cell_late,
      "missing_detection_count": cell_missing,
      "timely_detection_rate": cell_timely / QUALIFICATION_PAIRS_PER_CELL,
      "timely_delays_ticks": cell_delays,
    })

  qualified = (
    flat_trigger_count == 0
    and pre_impact_trigger_count == 0
    and timely_count >= QUALIFICATION_OVERALL_TIMELY_MIN
    and all(
      cell["timely_detection_count"] >= QUALIFICATION_PER_CELL_TIMELY_MIN
      for cell in cell_results
    )
  )
  mean_delay = (
    float(np.mean(np.asarray(timely_delays, dtype=np.float64)))
    if timely_delays
    else None
  )
  return {
    "threshold_table_index": int(row["index"]),
    "thresholds": {
      name: float(value) for name, value in zip(FEATURE_NAMES, thresholds, strict=True)
    },
    "qualified": qualified,
    "flat_trigger_count": flat_trigger_count,
    "stair_pre_impact_trigger_count": pre_impact_trigger_count,
    "timely_detection_count": timely_count,
    "timely_detection_rate": timely_count / (
      len(registered) * QUALIFICATION_PAIRS_PER_CELL
    ),
    "late_detection_count": late_count,
    "missing_detection_count": missing_count,
    "mean_timely_delay_ticks": mean_delay,
    "timely_delays_ticks": timely_delays,
    "cells": cell_results,
  }


def candidate_sort_key(candidate: dict[str, Any]) -> tuple[float | int, ...]:
  if candidate.get("qualified") is not True:
    raise ValueError("Only qualified innovation candidates may be ranked.")
  mean_delay = candidate.get("mean_timely_delay_ticks")
  if not isinstance(mean_delay, (int, float)) or not math.isfinite(float(mean_delay)):
    raise ValueError("Qualified innovation candidate requires a finite mean delay.")
  thresholds = candidate.get("thresholds")
  if not isinstance(thresholds, dict):
    raise TypeError("Qualified innovation candidate thresholds are missing.")
  values = _candidate_thresholds({
    "index": candidate.get("threshold_table_index"),
    **thresholds,
  })
  return (
    -int(candidate["timely_detection_count"]),
    float(mean_delay),
    float(values[0]),
    float(values[1]),
    float(values[2]),
    int(candidate["threshold_table_index"]),
  )


def select_qualification_candidate(
  candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
  qualified = [candidate for candidate in candidates if candidate.get("qualified") is True]
  return min(qualified, key=candidate_sort_key) if qualified else None


def qualification_selection(candidate: dict[str, Any]) -> dict[str, Any]:
  return {
    "threshold_table_index": int(candidate["threshold_table_index"]),
    "thresholds": dict(candidate["thresholds"]),
    "timely_detection_count": int(candidate["timely_detection_count"]),
    "timely_detection_rate": float(candidate["timely_detection_rate"]),
    "mean_timely_delay_ticks": float(candidate["mean_timely_delay_ticks"]),
  }


def _qualification_count(
  value: Any,
  *,
  name: str,
  maximum: int,
) -> int:
  if (
    isinstance(value, bool)
    or not isinstance(value, int)
    or not 0 <= value <= maximum
  ):
    raise ValueError(f"Innovation qualification {name} is invalid.")
  return value


def _lower_hex_digest(value: Any, *, length: int) -> bool:
  return (
    isinstance(value, str)
    and len(value) == length
    and all(character in "0123456789abcdef" for character in value)
  )


def _qualification_rate(value: Any, *, count: int, total: int, name: str) -> None:
  if (
    isinstance(value, bool)
    or not isinstance(value, (int, float))
    or not math.isfinite(float(value))
    or not math.isclose(
      float(value), count / total, rel_tol=0.0, abs_tol=1.0e-15
    )
  ):
    raise ValueError(f"Innovation qualification {name} is invalid.")


def _qualification_delays(
  value: Any,
  *,
  count: int,
  name: str,
) -> list[int]:
  if not isinstance(value, list) or len(value) != count:
    raise ValueError(f"Innovation qualification {name} is invalid.")
  if any(
    isinstance(delay, bool)
    or not isinstance(delay, int)
    or not 0 <= delay <= QUALIFICATION_MAX_DELAY_TICKS
    for delay in value
  ):
    raise ValueError(f"Innovation qualification {name} is invalid.")
  return value


def validate_qualification_candidate_summary(
  candidate: dict[str, Any],
  row: dict[str, Any],
) -> None:
  """Validate every aggregate in a serialized C2-j3 candidate summary."""

  expected_fields = {
    "threshold_table_index",
    "thresholds",
    "qualified",
    "flat_trigger_count",
    "stair_pre_impact_trigger_count",
    "timely_detection_count",
    "timely_detection_rate",
    "late_detection_count",
    "missing_detection_count",
    "mean_timely_delay_ticks",
    "timely_delays_ticks",
    "cells",
  }
  if not isinstance(candidate, dict) or set(candidate) != expected_fields:
    raise ValueError("Innovation qualification candidate field set is invalid.")
  thresholds = {name: row[name] for name in FEATURE_NAMES}
  if (
    candidate.get("threshold_table_index") != row["index"]
    or candidate.get("thresholds") != thresholds
  ):
    raise ValueError("Innovation qualification candidate table drifted.")

  cell_summaries = candidate.get("cells")
  registered = qualification_cells()
  if not isinstance(cell_summaries, list) or len(cell_summaries) != len(registered):
    raise ValueError("Innovation qualification candidate cell summary is invalid.")
  expected_cell_fields = {
    "cell_index",
    "flat_trigger_count",
    "stair_pre_impact_trigger_count",
    "timely_detection_count",
    "late_detection_count",
    "missing_detection_count",
    "timely_detection_rate",
    "timely_delays_ticks",
  }
  aggregate = {
    "flat": 0,
    "pre": 0,
    "timely": 0,
    "late": 0,
    "missing": 0,
  }
  aggregate_delays: list[int] = []
  for index, cell in enumerate(cell_summaries):
    if (
      not isinstance(cell, dict)
      or set(cell) != expected_cell_fields
      or cell.get("cell_index") != index
    ):
      raise ValueError("Innovation qualification candidate cell summary is invalid.")
    flat = _qualification_count(
      cell.get("flat_trigger_count"),
      name="cell flat-trigger count",
      maximum=QUALIFICATION_PAIRS_PER_CELL,
    )
    pre = _qualification_count(
      cell.get("stair_pre_impact_trigger_count"),
      name="cell pre-impact count",
      maximum=QUALIFICATION_PAIRS_PER_CELL,
    )
    timely = _qualification_count(
      cell.get("timely_detection_count"),
      name="cell timely count",
      maximum=QUALIFICATION_PAIRS_PER_CELL,
    )
    late = _qualification_count(
      cell.get("late_detection_count"),
      name="cell late count",
      maximum=QUALIFICATION_PAIRS_PER_CELL,
    )
    missing = _qualification_count(
      cell.get("missing_detection_count"),
      name="cell missing count",
      maximum=QUALIFICATION_PAIRS_PER_CELL,
    )
    if pre + timely + late + missing != QUALIFICATION_PAIRS_PER_CELL:
      raise ValueError("Innovation qualification cell counts do not conserve attempts.")
    _qualification_rate(
      cell.get("timely_detection_rate"),
      count=timely,
      total=QUALIFICATION_PAIRS_PER_CELL,
      name="cell timely rate",
    )
    delays = _qualification_delays(
      cell.get("timely_delays_ticks"),
      count=timely,
      name="cell timely-delay list",
    )
    aggregate["flat"] += flat
    aggregate["pre"] += pre
    aggregate["timely"] += timely
    aggregate["late"] += late
    aggregate["missing"] += missing
    aggregate_delays.extend(delays)

  total = len(registered) * QUALIFICATION_PAIRS_PER_CELL
  if (
    aggregate["pre"]
    + aggregate["timely"]
    + aggregate["late"]
    + aggregate["missing"]
    != total
  ):
    raise ValueError("Innovation qualification overall counts do not conserve attempts.")
  for field, key in (
    ("flat_trigger_count", "flat"),
    ("stair_pre_impact_trigger_count", "pre"),
    ("timely_detection_count", "timely"),
    ("late_detection_count", "late"),
    ("missing_detection_count", "missing"),
  ):
    count = _qualification_count(
      candidate.get(field), name=field, maximum=total
    )
    if count != aggregate[key]:
      raise ValueError("Innovation qualification candidate aggregate drifted.")
  _qualification_rate(
    candidate.get("timely_detection_rate"),
    count=aggregate["timely"],
    total=total,
    name="overall timely rate",
  )
  delays = _qualification_delays(
    candidate.get("timely_delays_ticks"),
    count=aggregate["timely"],
    name="overall timely-delay list",
  )
  if delays != aggregate_delays:
    raise ValueError("Innovation qualification timely-delay aggregate drifted.")
  mean_delay = candidate.get("mean_timely_delay_ticks")
  if delays:
    expected_mean = float(np.mean(np.asarray(delays, dtype=np.float64)))
    if (
      isinstance(mean_delay, bool)
      or not isinstance(mean_delay, (int, float))
      or not math.isfinite(float(mean_delay))
      or not math.isclose(
        float(mean_delay), expected_mean, rel_tol=0.0, abs_tol=1.0e-15
      )
    ):
      raise ValueError("Innovation qualification mean timely delay is invalid.")
  elif mean_delay is not None:
    raise ValueError("Innovation qualification empty delay set requires null mean.")

  expected_qualified = (
    aggregate["flat"] == 0
    and aggregate["pre"] == 0
    and aggregate["timely"] >= QUALIFICATION_OVERALL_TIMELY_MIN
    and all(
      cell["timely_detection_count"] >= QUALIFICATION_PER_CELL_TIMELY_MIN
      for cell in cell_summaries
    )
  )
  if candidate.get("qualified") is not expected_qualified:
    raise ValueError("Innovation qualification candidate verdict is invalid.")


def validate_qualification_cell_summaries(payload: dict[str, Any]) -> None:
  """Validate the deployable artifact's raw-capture provenance summaries."""

  cells = payload.get("cells")
  registered = qualification_cells()
  if not isinstance(cells, list) or len(cells) != len(registered):
    raise ValueError("Innovation qualification requires exactly 18 raw cells.")
  expected_fields = {
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
  health_fields = {
    "flat_termination_count",
    "stair_termination_count",
    "flat_timeout_count",
    "stair_timeout_count",
    "flat_non_wheel_contact_count",
    "stair_non_wheel_contact_count",
    "settle_riser_contact_count",
    "drive_start_past_face_count",
    "missing_impact_count",
    "invalid_window_count",
    "predictor_domain_violation_count",
    "posture_violation_count",
    "predictor_evaluation_error_count",
    "nonfinite_sample_count",
    "negative_feature_sample_count",
    "portable_target_violation_count",
    "outer_face_binding_violation_count",
  }
  for index, (cell, expected) in enumerate(zip(cells, registered, strict=True)):
    if (
      not isinstance(cell, dict)
      or set(cell) != expected_fields
      or cell.get("cell") != expected
      or cell.get("raw_file") != f"cell_{index:02d}.npz"
      or cell.get("raw_shape")
      != [QUALIFICATION_DRIVE_STEPS, QUALIFICATION_PAIRS_PER_CELL]
    ):
      raise ValueError("Innovation qualification raw-cell identity is invalid.")
    raw_hash = cell.get("raw_sha256")
    if (
      not isinstance(raw_hash, str)
      or len(raw_hash) != 64
      or any(character not in "0123456789abcdef" for character in raw_hash)
    ):
      raise ValueError("Innovation qualification raw-cell hash is invalid.")
    health = cell.get("health")
    if (
      not isinstance(health, dict)
      or set(health) != health_fields
      or any(
        isinstance(value, bool) or not isinstance(value, int) or value != 0
        for value in health.values()
      )
    ):
      raise ValueError("Innovation qualification raw-cell health is invalid.")
    numeric_limits = {
      "paired_reset_max_abs_error": 0.0,
      "written_reset_max_abs_error": QUALIFICATION_RESET_WRITE_ATOL,
      "written_paired_reset_max_abs_error": QUALIFICATION_RESET_WRITE_ATOL,
      "root_pitch_max_abs_error_rad": QUALIFICATION_POSTURE_CAPTURE_ATOL,
      "root_roll_yaw_max_abs_rad": QUALIFICATION_POSTURE_CAPTURE_ATOL,
      "other_root_velocity_max_abs": 0.0,
      "portable_max_abs_target_error_radps": QUALIFICATION_PORTABLE_EQUIVALENCE_ATOL,
    }
    for name, limit in numeric_limits.items():
      value = cell.get(name)
      if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        or float(value) > limit
      ):
        raise ValueError(f"Innovation qualification {name} is invalid.")
    impacts = cell.get("impact_steps")
    windows = cell.get("diagnostic_windows")
    if (
      not isinstance(impacts, list)
      or len(impacts) != QUALIFICATION_PAIRS_PER_CELL
      or not isinstance(windows, list)
      or len(windows) != QUALIFICATION_PAIRS_PER_CELL
    ):
      raise ValueError("Innovation qualification impact-window count is invalid.")
    for slot, (impact, window) in enumerate(zip(impacts, windows, strict=True)):
      if (
        isinstance(impact, bool)
        or not isinstance(impact, int)
        or impact < QUALIFICATION_PRE_IMPACT_STEPS
        or impact + QUALIFICATION_POST_IMPACT_STEPS
        >= QUALIFICATION_DRIVE_STEPS
        or window
        != {
          "slot": slot,
          "start_tick": impact - QUALIFICATION_PRE_IMPACT_STEPS,
          "impact_tick": impact,
          "end_tick": impact + QUALIFICATION_POST_IMPACT_STEPS,
        }
      ):
        raise ValueError("Innovation qualification impact window is invalid.")


def parse_transition_floor(
  payload: dict[str, Any], *, predictor_hash: str
) -> dict[str, Any]:
  if payload.get("schema_version") != FLOOR_SCHEMA_VERSION:
    raise ValueError("Innovation floor schema_version must be 1.")
  if payload.get("artifact_type") != FLOOR_ARTIFACT_TYPE:
    raise ValueError("Innovation floor artifact_type is invalid.")
  if payload.get("probe") != "hybrid_c2_transition_floor_v1":
    raise ValueError("Innovation floor probe identity is invalid.")
  if payload.get("classification") != "INNOVATION_FLOOR_QUALIFIED":
    raise ValueError("Innovation floor is not qualified.")
  if payload.get("predictor_hash") != predictor_hash:
    raise ValueError("Innovation floor predictor binding is invalid.")
  if payload.get("bindings") != EXPECTED_BINDINGS:
    raise ValueError("Innovation floor deployment bindings are invalid.")
  if payload.get("protocol") != OFFICIAL_TRANSITION_FLOOR_PROTOCOL:
    raise ValueError("Innovation floor protocol is invalid.")
  if (
    payload.get("evidence_eligible") is not False
    or payload.get("detector_fit_eligible") is not False
    or payload.get("promotion_eligible") is not False
    or payload.get("training_eligible") is not False
    or payload.get("checkpoint") is not None
  ):
    raise ValueError("Innovation floor eligibility flags are invalid.")
  maxima_mapping = payload.get("pooled_feature_maxima")
  if (
    not isinstance(maxima_mapping, dict)
    or set(maxima_mapping) != set(FEATURE_NAMES)
  ):
    raise ValueError("Innovation floor feature maxima are invalid.")
  maxima = np.asarray([maxima_mapping[name] for name in FEATURE_NAMES], dtype=np.float64)
  expected_table = threshold_table(maxima)
  if payload.get("threshold_table") != expected_table:
    raise ValueError("Innovation floor threshold table is invalid.")
  if payload.get("threshold_table_hash") != threshold_table_hash(expected_table):
    raise ValueError("Innovation floor threshold-table hash is invalid.")
  if payload.get("floor_hash") != canonical_hash(payload, hash_field="floor_hash"):
    raise ValueError("Innovation floor self-hash is invalid.")
  cells = payload.get("cells")
  if not isinstance(cells, list) or len(cells) != 10:
    raise ValueError("Innovation floor requires ten cells.")
  observed_maxima = np.zeros(3, dtype=np.float64)
  for index, (cell, registered) in enumerate(zip(cells, transition_floor_cells(), strict=True)):
    if not isinstance(cell, dict) or (
      cell.get("cell_index") != index
      or cell.get("name") != registered["name"]
      or cell.get("kind") != registered["kind"]
      or cell.get("target") != registered["target"]
      or cell.get("raw_file") != f"cell_{index:02d}.npz"
      or not isinstance(cell.get("raw_sha256"), str)
      or len(cell["raw_sha256"]) != 64
      or any(character not in "0123456789abcdef" for character in cell["raw_sha256"])
      or cell.get("raw_shape") != [500, 16]
      or cell.get("domain_violation_count") != 0
      or cell.get("termination_count") != 0
      or cell.get("timeout_count") != 0
      or cell.get("non_wheel_contact_count") != 0
      or not isinstance(cell.get("active_voting_ticks"), int)
      or cell["active_voting_ticks"] <= 0
      or not isinstance(cell.get("active_voting_ticks_per_env"), list)
      or len(cell["active_voting_ticks_per_env"]) != 16
      or any(
        not isinstance(value, int) or value <= 0
        for value in cell["active_voting_ticks_per_env"]
      )
    ):
      raise ValueError("Innovation floor cell is invalid.")
    feature_maxima = cell.get("feature_maxima")
    if (
      not isinstance(feature_maxima, dict)
      or set(feature_maxima) != set(FEATURE_NAMES)
      or any(
        not math.isfinite(float(feature_maxima[name]))
        or float(feature_maxima[name]) <= 0.0
        for name in FEATURE_NAMES
      )
    ):
      raise ValueError("Innovation floor cell maxima are invalid.")
    observed_maxima = np.maximum(
      observed_maxima,
      np.asarray([feature_maxima[name] for name in FEATURE_NAMES], dtype=np.float64),
    )
  if not np.array_equal(observed_maxima, maxima):
    raise ValueError("Innovation floor pooled maxima do not match its cells.")
  return payload


@dataclass(frozen=True)
class InnovationDetectorQualification:
  pitch_rate_innovation_radps: float
  wheel_speed_innovation_radps: float
  forward_deceleration_mps2: float
  consecutive_ticks: int
  threshold_table_index: int
  detector_hash: str
  predictor_hash: str
  floor_hash: str
  threshold_table_hash: str
  bindings: dict[str, str | None]


def parse_innovation_detector_qualification(
  payload: dict[str, Any],
  *,
  predictor_hash: str,
  floor_payload: dict[str, Any],
) -> InnovationDetectorQualification:
  floor = parse_transition_floor(floor_payload, predictor_hash=predictor_hash)
  if payload.get("schema_version") != QUALIFICATION_SCHEMA_VERSION:
    raise ValueError("Innovation qualification schema_version must be 1.")
  if payload.get("artifact_type") != QUALIFICATION_ARTIFACT_TYPE:
    raise ValueError("Innovation qualification artifact_type is invalid.")
  if payload.get("probe") != OFFICIAL_QUALIFICATION_PROTOCOL["probe"]:
    raise ValueError("Innovation qualification probe identity is invalid.")
  if payload.get("classification") != "INNOVATION_DETECTOR_QUALIFIED":
    raise ValueError("Innovation detector qualification did not pass.")
  if payload.get("protocol") != OFFICIAL_QUALIFICATION_PROTOCOL:
    raise ValueError("Innovation qualification protocol is invalid.")
  if payload.get("predictor_hash") != predictor_hash:
    raise ValueError("Innovation qualification predictor binding is invalid.")
  if payload.get("floor_hash") != floor["floor_hash"]:
    raise ValueError("Innovation qualification floor binding is invalid.")
  if payload.get("threshold_table_hash") != floor["threshold_table_hash"]:
    raise ValueError("Innovation qualification threshold-table binding is invalid.")
  if payload.get("bindings") != EXPECTED_BINDINGS:
    raise ValueError("Innovation qualification deployment bindings are invalid.")
  if (
    payload.get("evidence_eligible") is not True
    or payload.get("promotion_eligible") is not False
    or payload.get("training_eligible") is not False
    or payload.get("checkpoint") is not None
  ):
    raise ValueError("Innovation qualification eligibility flags are invalid.")
  if payload.get("detector_hash") != canonical_hash(
    payload, hash_field="detector_hash"
  ):
    raise ValueError("Innovation qualification self-hash is invalid.")
  if (
    not _lower_hex_digest(payload.get("git_sha"), length=40)
    or not _lower_hex_digest(payload.get("mjlab_git_sha"), length=40)
  ):
    raise ValueError("Innovation qualification Git provenance is invalid.")
  expected_total = len(qualification_cells()) * QUALIFICATION_PAIRS_PER_CELL
  if (
    payload.get("completed_cell_count") != len(qualification_cells())
    or payload.get("completed_pair_count") != expected_total
    or payload.get("completed_candidate_count") != len(floor["threshold_table"])
    or payload.get("next_step") != "FREEZE_AND_INDEPENDENT_AUDIT_BEFORE_C3"
  ):
    raise ValueError("Innovation qualification completion metadata is invalid.")
  validate_qualification_cell_summaries(payload)
  candidates = payload.get("candidates")
  table = floor["threshold_table"]
  if not isinstance(candidates, list) or len(candidates) != len(table):
    raise ValueError("Innovation qualification requires all threshold candidates.")
  for candidate, row in zip(candidates, table, strict=True):
    validate_qualification_candidate_summary(candidate, row)
  qualified_count = sum(candidate["qualified"] is True for candidate in candidates)
  if (
    isinstance(payload.get("qualified_candidate_count"), bool)
    or payload.get("qualified_candidate_count") != qualified_count
    or qualified_count < 1
  ):
    raise ValueError("Innovation qualification qualified-candidate count is invalid.")
  selected = select_qualification_candidate(candidates)
  if selected is None or payload.get("selected_candidate") != qualification_selection(selected):
    raise ValueError("Innovation qualification selected candidate is invalid.")
  thresholds = _candidate_thresholds({
    "index": selected["threshold_table_index"],
    **selected["thresholds"],
  })
  return InnovationDetectorQualification(
    pitch_rate_innovation_radps=float(thresholds[0]),
    wheel_speed_innovation_radps=float(thresholds[1]),
    forward_deceleration_mps2=float(thresholds[2]),
    consecutive_ticks=2,
    threshold_table_index=int(selected["threshold_table_index"]),
    detector_hash=str(payload["detector_hash"]),
    predictor_hash=predictor_hash,
    floor_hash=str(floor["floor_hash"]),
    threshold_table_hash=str(floor["threshold_table_hash"]),
    bindings=dict(EXPECTED_BINDINGS),
  )


@dataclass(frozen=True)
class InnovationPredictor:
  height_nodes: tuple[float, ...]
  pitch_nodes: tuple[float, ...]
  a: NDArray[np.float64]
  b: NDArray[np.float64]
  c: NDArray[np.float64]
  u_min: NDArray[np.float64]
  u_max: NDArray[np.float64]
  predictor_hash: str
  bindings: dict[str, str | None]

  def _validate_posture(self, height: float, pitch: float) -> tuple[float, float]:
    h = float(height)
    p = float(pitch)
    if not math.isfinite(h) or not math.isfinite(p):
      raise ValueError("Predictor posture must be finite.")
    if not (
      self.height_nodes[0] - PREDICTOR_POSTURE_DOMAIN_ATOL
      <= h
      <= self.height_nodes[-1] + PREDICTOR_POSTURE_DOMAIN_ATOL
    ):
      raise ValueError("Predictor height is outside the registered rectangle.")
    if not (
      self.pitch_nodes[0] - PREDICTOR_POSTURE_DOMAIN_ATOL
      <= p
      <= self.pitch_nodes[-1] + PREDICTOR_POSTURE_DOMAIN_ATOL
    ):
      raise ValueError("Predictor pitch is outside the registered rectangle.")
    return (
      min(max(h, self.height_nodes[0]), self.height_nodes[-1]),
      min(max(p, self.pitch_nodes[0]), self.pitch_nodes[-1]),
    )

  def interpolate(
    self, height: float, pitch: float
  ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    height, pitch = self._validate_posture(height, pitch)
    return (
      np.asarray(bilinear_interpolate(self.height_nodes, self.pitch_nodes, self.a, height, pitch)),
      np.asarray(bilinear_interpolate(self.height_nodes, self.pitch_nodes, self.b, height, pitch)),
      np.asarray(bilinear_interpolate(self.height_nodes, self.pitch_nodes, self.c, height, pitch)),
    )

  def input_domain(self, height: float, pitch: float) -> tuple[float, float]:
    height, pitch = self._validate_posture(height, pitch)
    minimum = float(bilinear_interpolate(
      self.height_nodes, self.pitch_nodes, self.u_min, height, pitch
    ))
    maximum = float(bilinear_interpolate(
      self.height_nodes, self.pitch_nodes, self.u_max, height, pitch
    ))
    return minimum, maximum

  def predict(self, z: ArrayLike, u: float, height: float, pitch: float) -> NDArray[np.float64]:
    state = np.asarray(z, dtype=np.float64)
    if state.shape != (2,) or not np.all(np.isfinite(state)) or not math.isfinite(u):
      raise ValueError("Predictor state/input must be finite.")
    minimum, maximum = self.input_domain(height, pitch)
    if not minimum - DOMAIN_TOLERANCE_RADPS <= u <= maximum + DOMAIN_TOLERANCE_RADPS:
      raise ValueError("Predictor input is outside the fitted domain.")
    a, b, c = self.interpolate(height, pitch)
    return a @ state + b[:, 0] * u + c


def parse_innovation_predictor(payload: dict[str, Any]) -> InnovationPredictor:
  if payload.get("schema_version") != PREDICTOR_SCHEMA_VERSION:
    raise ValueError("Innovation predictor schema_version must be 1.")
  if payload.get("artifact_type") != PREDICTOR_ARTIFACT_TYPE:
    raise ValueError("Innovation predictor artifact_type is invalid.")
  if payload.get("grid_sha256") != GRID_SHA256:
    raise ValueError("Innovation predictor grid hash is invalid.")
  if tuple(payload.get("state_names", ())) != PREDICTOR_STATE_NAMES:
    raise ValueError("Innovation predictor state names are invalid.")
  if payload.get("probe") != "hybrid_c2_predictor_identification_v1":
    raise ValueError("Innovation predictor probe identity is invalid.")
  if payload.get("classification") != "PREDICTOR_IDENTIFICATION_QUALIFIED":
    raise ValueError("Innovation predictor is not qualified.")
  if payload.get("protocol") != OFFICIAL_IDENTIFICATION_PROTOCOL:
    raise ValueError("Innovation predictor protocol is invalid.")
  if (
    payload.get("evidence_eligible") is not False
    or payload.get("detector_fit_eligible") is not False
    or payload.get("promotion_eligible") is not False
    or payload.get("training_eligible") is not False
    or payload.get("checkpoint") is not None
  ):
    raise ValueError("Innovation predictor eligibility flags are invalid.")
  if payload.get("predictor_hash") != canonical_hash(payload, hash_field="predictor_hash"):
    raise ValueError("Innovation predictor self-hash is invalid.")
  heights = tuple(float(value) for value in payload.get("height_nodes", ()))
  pitches = tuple(float(value) for value in payload.get("pitch_nodes", ()))
  if heights != REGISTERED_HEIGHT_NODES or pitches != REGISTERED_PITCH_NODES:
    raise ValueError("Innovation predictor nodes do not match registration.")
  nodes = payload.get("nodes")
  if not isinstance(nodes, list) or len(nodes) != 9:
    raise ValueError("Innovation predictor requires exactly nine nodes.")
  a = np.empty((3, 3, 2, 2), dtype=np.float64)
  b = np.empty((3, 3, 2, 1), dtype=np.float64)
  c = np.empty((3, 3, 2), dtype=np.float64)
  u_min = np.empty((3, 3), dtype=np.float64)
  u_max = np.empty((3, 3), dtype=np.float64)
  for index, node in enumerate(nodes):
    if not isinstance(node, dict) or node.get("node_index") != index:
      raise ValueError("Innovation predictor node order is invalid.")
    h, p = divmod(index, 3)
    node_a = np.asarray(node.get("a"), dtype=np.float64)
    node_b = np.asarray(node.get("b"), dtype=np.float64)
    node_c = np.asarray(node.get("c"), dtype=np.float64)
    if node_a.shape != (2, 2) or node_b.shape != (2, 1) or node_c.shape != (2,):
      raise ValueError("Innovation predictor node coefficient shape is invalid.")
    a[h, p] = node_a
    b[h, p] = node_b
    c[h, p] = node_c
    u_min[h, p] = float(node.get("fit_u_min_radps", math.nan))
    u_max[h, p] = float(node.get("fit_u_max_radps", math.nan))
    nrmse = np.asarray(node.get("heldout_nrmse"), dtype=np.float64)
    if (
      node.get("regression_rank") != 4
      or nrmse.shape != (2,)
      or not np.all(np.isfinite(nrmse))
      or np.any(nrmse < 0.0)
      or np.any(nrmse > 0.15)
    ):
      raise ValueError("Innovation predictor node failed qualification.")
  if not all(np.all(np.isfinite(value)) for value in (a, b, c, u_min, u_max)):
    raise ValueError("Innovation predictor contains nonfinite data.")
  if np.any(u_min >= u_max):
    raise ValueError("Innovation predictor input domains must have positive width.")
  bindings = payload.get("bindings")
  if bindings != EXPECTED_BINDINGS:
    raise ValueError("Innovation predictor bindings are invalid.")
  return InnovationPredictor(
    height_nodes=heights, pitch_nodes=pitches, a=a, b=b, c=c,
    u_min=u_min, u_max=u_max, predictor_hash=str(payload["predictor_hash"]),
    bindings={str(key): (None if value is None else str(value)) for key, value in bindings.items()},
  )
