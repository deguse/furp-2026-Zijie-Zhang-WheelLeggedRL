"""Posture sweep qualification and local two-leg joint target mapping."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import ConvexHull, QhullError


LEG_JOINT_NAMES = (
  "thigh_left_01",
  "thigh_right_01",
  "knee_left",
  "knee_right",
)
POSTURE_FEATURE_NAMES = ("bias", "height", "pitch")


@dataclass(frozen=True)
class PostureEnvelope:
  height_range: tuple[float, float]
  pitch_range: tuple[float, float]
  verified_grid_shape: tuple[int, int]
  verification_method: str = "all_feasible_grid_rectangle"


@dataclass(frozen=True)
class PostureMap:
  coefficients: NDArray[np.float64]

  @property
  def map_hash(self) -> str:
    payload = {
      "feature_names": POSTURE_FEATURE_NAMES,
      "joint_names": LEG_JOINT_NAMES,
      "coefficients": self.coefficients.tolist(),
    }
    encoded = json.dumps(
      payload,
      sort_keys=True,
      separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _vector(name: str, value: ArrayLike) -> NDArray[np.float64]:
  array = np.asarray(value, dtype=np.float64)
  if array.ndim != 1 or array.size < 1:
    raise ValueError(f"{name} must be a non-empty one-dimensional array.")
  if not np.all(np.isfinite(array)):
    raise ValueError(f"{name} must contain only finite values.")
  return array


def _matrix(
  name: str,
  value: ArrayLike,
  *,
  columns: int,
) -> NDArray[np.float64]:
  array = np.asarray(value, dtype=np.float64)
  if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != columns:
    raise ValueError(f"{name} must have shape (num_samples, {columns}).")
  if not np.all(np.isfinite(array)):
    raise ValueError(f"{name} must contain only finite values.")
  return array


def select_feasible_samples(
  *,
  non_wheel_contact: ArrayLike,
  joint_positions: ArrayLike,
  joint_lower: ArrayLike,
  joint_upper: ArrayLike,
  actuator_load_fraction: ArrayLike,
  joint_margin_fraction: float = 0.10,
  actuator_load_limit: float = 0.80,
) -> NDArray[np.bool_]:
  """Return samples with wheel-only contact and sufficient joint/load margin."""

  positions = _matrix("joint_positions", joint_positions, columns=4)
  loads = _matrix("actuator_load_fraction", actuator_load_fraction, columns=4)
  lower = _vector("joint_lower", joint_lower)
  upper = _vector("joint_upper", joint_upper)
  contacts = np.asarray(non_wheel_contact)
  sample_count = positions.shape[0]
  if contacts.shape != (sample_count,):
    raise ValueError("non_wheel_contact must contain one value per sample.")
  if loads.shape[0] != sample_count:
    raise ValueError("actuator_load_fraction must contain one row per sample.")
  if lower.shape != (4,) or upper.shape != (4,):
    raise ValueError("joint limits must each contain four values.")
  if np.any(lower >= upper):
    raise ValueError("Each joint lower limit must be below its upper limit.")
  if not 0.0 <= joint_margin_fraction < 0.5:
    raise ValueError("joint_margin_fraction must be in [0, 0.5).")
  if not 0.0 < actuator_load_limit <= 1.0:
    raise ValueError("actuator_load_limit must be in (0, 1].")

  margin = joint_margin_fraction * (upper - lower)
  within_margin = np.all(
    (positions >= lower + margin) & (positions <= upper - margin),
    axis=1,
  )
  below_load_limit = np.all(loads < actuator_load_limit, axis=1)
  return (~contacts.astype(bool)) & within_margin & below_load_limit


def _largest_verified_grid_rectangle(
  first_coordinates: NDArray[np.float64],
  second_coordinates: NDArray[np.float64],
  feasible: NDArray[np.bool_],
) -> tuple[NDArray[np.bool_], tuple[int, int]]:
  unique_first = np.unique(first_coordinates)
  unique_second = np.unique(second_coordinates)
  verified = np.ones(
    (unique_first.size, unique_second.size),
    dtype=bool,
  )
  observed = np.zeros_like(verified)
  first_indices = np.searchsorted(unique_first, first_coordinates)
  second_indices = np.searchsorted(unique_second, second_coordinates)
  for index, is_feasible in enumerate(feasible):
    cell = (first_indices[index], second_indices[index])
    observed[cell] = True
    verified[cell] &= bool(is_feasible)
  verified &= observed

  best: tuple[tuple[float, int, float, float], int, int, int, int] | None = None
  for first_start in range(max(0, unique_first.size - 1)):
    valid_columns = verified[first_start].copy()
    for first_end in range(first_start + 1, unique_first.size):
      valid_columns &= verified[first_end]
      second_start = 0
      while second_start < unique_second.size:
        if not valid_columns[second_start]:
          second_start += 1
          continue
        second_end = second_start
        while (
          second_end + 1 < unique_second.size
          and valid_columns[second_end + 1]
        ):
          second_end += 1
        if second_end > second_start:
          first_span = float(
            unique_first[first_end] - unique_first[first_start]
          )
          second_span = float(
            unique_second[second_end] - unique_second[second_start]
          )
          shape = (
            first_end - first_start + 1,
            second_end - second_start + 1,
          )
          score = (
            first_span * second_span,
            shape[0] * shape[1],
            first_span,
            second_span,
          )
          candidate = (
            score,
            first_start,
            first_end,
            second_start,
            second_end,
          )
          if best is None or candidate > best:
            best = candidate
        second_start = second_end + 1

  if best is None:
    raise ValueError(
      "Feasible samples do not contain a verified 2D rectangle."
    )
  _, first_start, first_end, second_start, second_end = best
  selected = (
    (first_indices >= first_start)
    & (first_indices <= first_end)
    & (second_indices >= second_start)
    & (second_indices <= second_end)
  )
  return selected, (
    first_end - first_start + 1,
    second_end - second_start + 1,
  )


def _shrink_range(
  low: float,
  high: float,
  inward_fraction: float,
) -> tuple[float, float]:
  inset = inward_fraction * (high - low)
  return round(low + inset, 15), round(high - inset, 15)


def training_envelope(
  *,
  heights: ArrayLike,
  pitches: ArrayLike,
  feasible: ArrayLike,
  inward_fraction: float = 0.10,
  pitch_limit: float = 0.08,
) -> PostureEnvelope:
  """Build a conservative command envelope from feasible sweep samples."""

  height_values = _vector("heights", heights)
  pitch_values = _vector("pitches", pitches)
  feasible_mask = np.asarray(feasible, dtype=bool)
  if height_values.shape != pitch_values.shape:
    raise ValueError("heights and pitches must have identical shapes.")
  if feasible_mask.shape != height_values.shape:
    raise ValueError("feasible must contain one value per posture sample.")
  if not np.any(feasible_mask):
    raise ValueError("At least one feasible posture sample is required.")
  if not 0.0 <= inward_fraction < 0.5:
    raise ValueError("inward_fraction must be in [0, 0.5).")
  if pitch_limit <= 0.0:
    raise ValueError("pitch_limit must be positive.")

  selected, grid_shape = _largest_verified_grid_rectangle(
    height_values,
    pitch_values,
    feasible_mask,
  )
  height_range = _shrink_range(
    float(np.min(height_values[selected])),
    float(np.max(height_values[selected])),
    inward_fraction,
  )
  pitch_range = _shrink_range(
    float(np.min(pitch_values[selected])),
    float(np.max(pitch_values[selected])),
    inward_fraction,
  )
  pitch_range = (
    max(pitch_range[0], -pitch_limit),
    min(pitch_range[1], pitch_limit),
  )
  if height_range[0] > height_range[1] or pitch_range[0] > pitch_range[1]:
    raise ValueError("Feasible samples do not overlap the requested posture limits.")
  return PostureEnvelope(
    height_range=height_range,
    pitch_range=pitch_range,
    verified_grid_shape=grid_shape,
  )


def training_envelope_from_sweep_grid(
  *,
  heights: ArrayLike,
  pitches: ArrayLike,
  feasible: ArrayLike,
  first_coordinates: ArrayLike,
  second_coordinates: ArrayLike,
  inward_fraction: float = 0.10,
  pitch_limit: float = 0.08,
) -> PostureEnvelope:
  """Build a command rectangle inside an all-feasible sweep-grid hull."""

  height_values = _vector("heights", heights)
  pitch_values = _vector("pitches", pitches)
  first_values = _vector("first_coordinates", first_coordinates)
  second_values = _vector("second_coordinates", second_coordinates)
  feasible_mask = np.asarray(feasible, dtype=bool)
  expected_shape = height_values.shape
  if any(
    value.shape != expected_shape
    for value in (
      pitch_values,
      first_values,
      second_values,
      feasible_mask,
    )
  ):
    raise ValueError("Sweep coordinates and measurements must have identical shapes.")
  if not np.any(feasible_mask):
    raise ValueError("At least one feasible posture sample is required.")
  if not 0.0 <= inward_fraction < 0.5:
    raise ValueError("inward_fraction must be in [0, 0.5).")
  if pitch_limit <= 0.0:
    raise ValueError("pitch_limit must be positive.")

  selected, grid_shape = _largest_verified_grid_rectangle(
    first_values,
    second_values,
    feasible_mask,
  )
  measured_points = np.column_stack(
    (height_values[selected], pitch_values[selected])
  )
  if np.linalg.matrix_rank(measured_points - measured_points.mean(axis=0)) < 2:
    raise ValueError(
      "Verified sweep rectangle does not span both height and pitch."
    )
  try:
    hull = ConvexHull(measured_points)
  except QhullError as error:
    raise ValueError(
      "Verified sweep measurements do not form a two-dimensional hull."
    ) from error

  center = measured_points.mean(axis=0)
  full_half_span = 0.5 * np.ptp(measured_points, axis=0)
  if np.any(full_half_span <= 0.0):
    raise ValueError(
      "Verified sweep rectangle must vary both height and pitch."
    )
  normals = hull.equations[:, :2]
  offsets = hull.equations[:, 2]
  slack = -(normals @ center + offsets)
  corner_growth = np.abs(normals) @ full_half_span
  active = corner_growth > np.finfo(np.float64).eps
  scale = float(np.min(slack[active] / corner_growth[active]))
  if not np.isfinite(scale) or scale <= 0.0:
    raise ValueError("Could not inscribe a command rectangle in the sweep hull.")
  half_span = min(scale, 1.0) * full_half_span
  lower = center - half_span
  upper = center + half_span
  height_range = _shrink_range(
    float(lower[0]),
    float(upper[0]),
    inward_fraction,
  )
  pitch_range = _shrink_range(
    float(lower[1]),
    float(upper[1]),
    inward_fraction,
  )
  pitch_range = (
    max(pitch_range[0], -pitch_limit),
    min(pitch_range[1], pitch_limit),
  )
  if height_range[0] >= height_range[1] or pitch_range[0] >= pitch_range[1]:
    raise ValueError("Sweep hull does not contain a non-degenerate command range.")
  return PostureEnvelope(
    height_range=height_range,
    pitch_range=pitch_range,
    verified_grid_shape=grid_shape,
    verification_method="all_feasible_sweep_grid_hull_rectangle",
  )


def fit_posture_map(
  heights: ArrayLike,
  pitches: ArrayLike,
  joint_positions: ArrayLike,
) -> PostureMap:
  """Fit ``[1, height, pitch]`` to the four joints of the two legs."""

  height_values = _vector("heights", heights)
  pitch_values = _vector("pitches", pitches)
  positions = _matrix("joint_positions", joint_positions, columns=4)
  if height_values.shape != pitch_values.shape:
    raise ValueError("heights and pitches must have identical shapes.")
  if positions.shape[0] != height_values.size:
    raise ValueError("joint_positions must contain one row per posture sample.")
  features = np.column_stack(
    (np.ones(height_values.size), height_values, pitch_values)
  )
  if np.linalg.matrix_rank(features) != len(POSTURE_FEATURE_NAMES):
    raise ValueError("Posture samples must independently excite height and pitch.")
  coefficients, _, _, _ = np.linalg.lstsq(features, positions, rcond=None)
  return PostureMap(coefficients=coefficients)


def predict_leg_targets(
  posture_map: PostureMap,
  *,
  heights: ArrayLike,
  pitches: ArrayLike,
) -> NDArray[np.float64]:
  height_values = _vector("heights", heights)
  pitch_values = _vector("pitches", pitches)
  if height_values.shape != pitch_values.shape:
    raise ValueError("heights and pitches must have identical shapes.")
  if posture_map.coefficients.shape != (3, 4):
    raise ValueError("Posture map coefficients must have shape (3, 4).")
  features = np.column_stack(
    (np.ones(height_values.size), height_values, pitch_values)
  )
  return features @ posture_map.coefficients


def posture_map_to_dict(
  posture_map: PostureMap,
  envelope: PostureEnvelope,
  *,
  feasible_sample_count: int,
  total_sample_count: int,
) -> dict[str, object]:
  if not 0 < feasible_sample_count <= total_sample_count:
    raise ValueError("Sample counts must be positive and internally consistent.")
  return {
    "schema_version": 1,
    "feature_names": list(POSTURE_FEATURE_NAMES),
    "joint_names": list(LEG_JOINT_NAMES),
    "coefficients": posture_map.coefficients.tolist(),
    "training_envelope": {
      "height": list(envelope.height_range),
      "pitch": list(envelope.pitch_range),
    },
    "envelope_verification": {
      "method": envelope.verification_method,
      "grid_shape": list(envelope.verified_grid_shape),
    },
    "feasible_sample_count": feasible_sample_count,
    "total_sample_count": total_sample_count,
    "map_hash": posture_map.map_hash,
  }
