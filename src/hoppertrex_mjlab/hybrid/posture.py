"""Posture sweep qualification and local two-leg joint target mapping."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

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
POSTURE_ENVELOPE_VERIFICATION_METHODS = frozenset(
  (
    "all_feasible_grid_rectangle",
    "all_feasible_sweep_grid_hull_rectangle",
    "registered_fixed_symmetric_hull_rectangle",
  )
)


def posture_artifact_hash(payload: dict[str, object]) -> str:
  """Hash the fitted mapping together with its qualified command envelope.

  ``PostureMap.map_hash`` identifies only the fitted joint mapping. Multiple
  command envelopes can reuse those coefficients, so safety-sensitive users
  need a second identity that includes the envelope and fit criteria. Absolute
  source paths are excluded so the hash is portable across machines.
  """

  identity = {
    "schema_version": payload.get("schema_version"),
    "feature_names": payload.get("feature_names"),
    "joint_names": payload.get("joint_names"),
    "coefficients": payload.get("coefficients"),
    "training_envelope": payload.get("training_envelope"),
    "envelope_verification": payload.get("envelope_verification"),
    "feasible_sample_count": payload.get("feasible_sample_count"),
    "total_sample_count": payload.get("total_sample_count"),
    "map_hash": payload.get("map_hash"),
    "fit_criteria": payload.get("fit_criteria"),
    "source_sweep": payload.get("source_sweep"),
  }
  if "fit_provenance" in payload:
    identity["fit_provenance"] = payload["fit_provenance"]
  encoded = json.dumps(
    identity, sort_keys=True, separators=(",", ":"),
  ).encode("ascii")
  return hashlib.sha256(encoded).hexdigest()


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
  joint_margin_rad: float | None = None,
) -> NDArray[np.bool_]:
  """Return samples with wheel-only contact and sufficient joint/load margin.

  ``joint_margin_rad`` replaces the fraction-of-range margin with an
  absolute one. The fractional default is range-relative, so a wide joint
  demands a wide margin regardless of how the robot actually moves: on the
  2026-07-15 sweep it demanded 0.279 rad on the knees and thereby declared
  the nominal standing posture (0.245 rad from its knee limit, measured
  actuator load <= 0.33) infeasible. The headroom a posture really needs is
  the dynamic excursion around the held target - leg residual scale plus
  tracking jitter - which is an absolute quantity.
  """

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

  if joint_margin_rad is None:
    margin = joint_margin_fraction * (upper - lower)
  else:
    if not np.isfinite(joint_margin_rad) or joint_margin_rad < 0.0:
      raise ValueError("joint_margin_rad must be finite and non-negative.")
    if np.any(2.0 * joint_margin_rad >= upper - lower):
      raise ValueError(
        "joint_margin_rad must leave usable room inside every joint range."
      )
    margin = np.full_like(upper, joint_margin_rad)
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


def _height_priority_rectangle(
  measured_points: NDArray[np.float64],
  normals: NDArray[np.float64],
  offsets: NDArray[np.float64],
  *,
  pitch_half_span: float,
  pitch_center_bound: float | None = None,
  center_resolution: int = 400,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  """Maximize the height half-span at a fixed pitch half-width.

  Deterministic grid search over (height center, pitch center) with the
  pitch center bounded so the eventual command range contains zero; for
  each candidate center the maximal height half-span is exact from the
  hull's facet inequalities (all four rectangle corners must satisfy
  n.x + b <= 0).
  """

  if pitch_center_bound is None:
    pitch_center_bound = pitch_half_span
  height_low = float(measured_points[:, 0].min())
  height_high = float(measured_points[:, 0].max())
  pitch_low = float(measured_points[:, 1].min())
  pitch_high = float(measured_points[:, 1].max())
  pitch_centers = np.linspace(
    max(pitch_low + pitch_half_span, -pitch_center_bound),
    min(pitch_high - pitch_half_span, pitch_center_bound),
    center_resolution,
  )
  height_centers = np.linspace(height_low, height_high, center_resolution)
  epsilon = np.finfo(np.float64).eps
  best: tuple[float, float, float] | None = None
  for pitch_center in pitch_centers:
    for height_center in height_centers:
      half_span = np.inf
      feasible = True
      for pitch_sign in (-1.0, 1.0):
        pitch_corner = pitch_center + pitch_sign * pitch_half_span
        residual = -(
          offsets
          + normals[:, 0] * height_center
          + normals[:, 1] * pitch_corner
        )
        if np.any(residual[np.abs(normals[:, 0]) <= epsilon] < 0.0):
          feasible = False
          break
        growing = np.abs(normals[:, 0]) > epsilon
        half_span = min(
          half_span,
          float(
            np.min(residual[growing] / np.abs(normals[:, 0][growing]))
          ),
        )
      if (
        feasible
        and half_span > 0.0
        and (best is None or half_span > best[0])
      ):
        best = (half_span, float(height_center), float(pitch_center))
  if best is None:
    raise ValueError(
      "No feasible height-priority rectangle contains a zero pitch."
    )
  half_span, height_center, pitch_center = best
  lower = np.asarray(
    [height_center - half_span, pitch_center - pitch_half_span]
  )
  upper = np.asarray(
    [height_center + half_span, pitch_center + pitch_half_span]
  )
  return lower, upper


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
  pitch_half_span: float | None = None,
  height_floor: float | None = None,
  fixed_height_nodes: ArrayLike | None = None,
  symmetric_pitch_half_span: float | None = None,
) -> PostureEnvelope:
  """Build a command rectangle inside an all-feasible sweep-grid hull.

  The default inscription scales the full (height, pitch) half-span
  uniformly until the rectangle fits the verified hull. In a diagonal
  feasible band (tall postures only at positive pitch, low at negative -
  the measured HopperTrex geometry) that crushes the height span, so
  ``pitch_half_span`` switches to a height-priority inscription: fix the
  pitch half-width, then maximize the height half-span (exact linear
  program over the same verified hull), requiring the pitch range to
  contain zero so the neutral command stays reachable.
  """

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
  fixed_mode = (
    fixed_height_nodes is not None or symmetric_pitch_half_span is not None
  )
  if fixed_mode and (
    fixed_height_nodes is None or symmetric_pitch_half_span is None
  ):
    raise ValueError(
      "fixed_height_nodes and symmetric_pitch_half_span must be set together."
    )
  if fixed_mode and pitch_half_span is not None:
    raise ValueError(
      "Fixed symmetric envelope mode cannot also use pitch_half_span."
    )
  if height_floor is not None:
    # Balance-probe authority over static feasibility: postures below the
    # measured dynamic floor terminated under the qualified LQR even
    # though the static sweep accepted them (2026-07-19 expanded sweep:
    # height 0.265 fell in 4/5 pitch cells while 0.28+ was fully clean).
    feasible_mask = feasible_mask & (height_values >= float(height_floor))
    if not np.any(feasible_mask):
      raise ValueError(
        "height_floor excludes every feasible posture sample."
      )

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
  if fixed_mode:
    nodes = _vector("fixed_height_nodes", fixed_height_nodes)
    if nodes.size != 3 or not np.all(np.diff(nodes) > 0.0):
      raise ValueError(
        "fixed_height_nodes must contain three strictly increasing values."
      )
    bound = float(symmetric_pitch_half_span)
    if not np.isfinite(bound) or bound <= 0.0:
      raise ValueError("symmetric_pitch_half_span must be positive and finite.")
    if bound > pitch_limit:
      raise ValueError("symmetric_pitch_half_span exceeds pitch_limit.")
    corners = np.asarray(
      [
        [nodes[0], -bound],
        [nodes[0], bound],
        [nodes[-1], -bound],
        [nodes[-1], bound],
      ],
      dtype=np.float64,
    )
    violation = normals @ corners.T + offsets[:, None]
    if np.any(violation > 1.0e-12):
      raise ValueError(
        "Registered fixed symmetric posture rectangle is outside the "
        "qualified sweep hull."
      )
    return PostureEnvelope(
      height_range=(float(nodes[0]), float(nodes[-1])),
      pitch_range=(-bound, bound),
      verified_grid_shape=grid_shape,
      verification_method="registered_fixed_symmetric_hull_rectangle",
    )
  if pitch_half_span is not None:
    if pitch_half_span <= 0.0:
      raise ValueError("pitch_half_span must be positive.")
    lower, upper = _height_priority_rectangle(
      measured_points,
      normals,
      offsets,
      pitch_half_span=float(pitch_half_span),
      # The final command range is shrunk inward afterwards; bound the
      # center so zero pitch stays inside the SHRUNK range, keeping the
      # nominal standing command reachable.
      pitch_center_bound=(1.0 - 2.0 * inward_fraction)
      * float(pitch_half_span),
    )
  else:
    slack = -(normals @ center + offsets)
    corner_growth = np.abs(normals) @ full_half_span
    active = corner_growth > np.finfo(np.float64).eps
    scale = float(np.min(slack[active] / corner_growth[active]))
    if not np.isfinite(scale) or scale <= 0.0:
      raise ValueError(
        "Could not inscribe a command rectangle in the sweep hull."
      )
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
