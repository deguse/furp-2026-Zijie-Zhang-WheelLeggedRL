"""Deployment-visible affine predictor used by the C2 innovation detector."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .controller_schedule import bilinear_interpolate, canonical_hash

PREDICTOR_ARTIFACT_TYPE = "c2_innovation_predictor"
PREDICTOR_SCHEMA_VERSION = 1
FLOOR_ARTIFACT_TYPE = "c2_innovation_transition_floor"
FLOOR_SCHEMA_VERSION = 1
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

  def interpolate(
    self, height: float, pitch: float
  ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    return (
      np.asarray(bilinear_interpolate(self.height_nodes, self.pitch_nodes, self.a, height, pitch)),
      np.asarray(bilinear_interpolate(self.height_nodes, self.pitch_nodes, self.b, height, pitch)),
      np.asarray(bilinear_interpolate(self.height_nodes, self.pitch_nodes, self.c, height, pitch)),
    )

  def input_domain(self, height: float, pitch: float) -> tuple[float, float]:
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
