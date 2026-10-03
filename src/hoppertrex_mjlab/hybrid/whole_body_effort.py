"""Model-based bilateral support-load and whole-body effort controller.

The kernel is simulator independent: callers provide one causal state sample,
the generalized mass matrix/bias, and vertical Jacobians at the two wheel axle
centres.  It never consumes contact force or a future state.  A 50 Hz outer
loop may hold posture/wheel targets while this kernel refreshes torque at the
5 ms physics cadence.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .support_load_allocator import (
  SupportLoadAllocation,
  SupportLoadAllocatorConfig,
  allocate_support_loads,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class WholeBodyEffortConfig:
  total_support_force_n: float = 164.5048209
  minimum_side_load_n: float = 5.0
  roll_natural_frequency_rad_s: float = 10.0
  roll_damping_ratio: float = 1.0
  joint_natural_frequency_rad_s: float = 20.0
  leg_kp_max: float = 500.0
  leg_kd_max: float = 5.0
  leg_torque_limit_nm: float = 30.0
  wheel_torque_limit_nm: float = 5.8
  wheel_rated_torque_nm: float = 2.79
  wheel_velocity_damping_nm_per_rad_s: float = 200.0

  def __post_init__(self) -> None:
    positive = (
      self.total_support_force_n,
      self.minimum_side_load_n,
      self.roll_natural_frequency_rad_s,
      self.roll_damping_ratio,
      self.joint_natural_frequency_rad_s,
      self.leg_kp_max,
      self.leg_kd_max,
      self.leg_torque_limit_nm,
      self.wheel_torque_limit_nm,
      self.wheel_rated_torque_nm,
      self.wheel_velocity_damping_nm_per_rad_s,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in positive):
      raise ValueError("Whole-body effort parameters must be finite and positive.")
    if 2.0 * self.minimum_side_load_n >= self.total_support_force_n:
      raise ValueError("Bilateral minimum loads must fit under total support.")
    if self.wheel_rated_torque_nm > self.wheel_torque_limit_nm:
      raise ValueError("Rated wheel torque cannot exceed the development limit.")


@dataclass(frozen=True)
class WholeBodyEffortResult:
  allocation: SupportLoadAllocation
  roll_inertia_kg_m2: float
  desired_roll_moment_unclamped_nm: float
  desired_roll_moment_nm: float
  roll_moment_limit_nm: float
  leg_kp: FloatArray
  leg_kd: FloatArray
  inverse_effort_nm: FloatArray
  leg_pd_effort_nm: FloatArray
  wheel_drive_effort_nm: FloatArray
  raw_leg_effort_nm: FloatArray
  raw_wheel_effort_nm: FloatArray
  leg_effort_nm: FloatArray
  wheel_effort_nm: FloatArray
  leg_saturated: NDArray[np.bool_]
  wheel_saturated: NDArray[np.bool_]
  wheel_above_rated: NDArray[np.bool_]

  @property
  def finite(self) -> bool:
    arrays = (
      self.leg_kp,
      self.leg_kd,
      self.inverse_effort_nm,
      self.leg_pd_effort_nm,
      self.wheel_drive_effort_nm,
      self.raw_leg_effort_nm,
      self.raw_wheel_effort_nm,
      self.leg_effort_nm,
      self.wheel_effort_nm,
    )
    scalars = (
      self.roll_inertia_kg_m2,
      self.desired_roll_moment_unclamped_nm,
      self.desired_roll_moment_nm,
      self.roll_moment_limit_nm,
    )
    return all(math.isfinite(value) for value in scalars) and all(
      bool(np.isfinite(values).all()) for values in arrays
    )


def _vector(values: Sequence[float] | FloatArray, size: int, name: str) -> FloatArray:
  result = np.asarray(values, dtype=np.float64)
  if result.shape != (size,) or not np.isfinite(result).all():
    raise ValueError(f"{name} must be a finite vector with shape ({size},).")
  return result


def _dof_ids(values: Sequence[int], size: int, nv: int, name: str) -> tuple[int, ...]:
  result = tuple(int(value) for value in values)
  if len(result) != size or len(set(result)) != size:
    raise ValueError(f"{name} must contain {size} unique DOF ids.")
  if any(value < 0 or value >= nv for value in result):
    raise ValueError(f"{name} contains a DOF outside [0, {nv}).")
  return result


def _effective_inertias(
  inverse_mass_matrix: FloatArray,
  dof_ids: Sequence[int],
) -> FloatArray:
  diagonal = np.asarray(
    [inverse_mass_matrix[dof_id, dof_id] for dof_id in dof_ids],
    dtype=np.float64,
  )
  if np.any(diagonal <= 0.0) or not np.isfinite(diagonal).all():
    raise ValueError("Inverse-mass diagonal must be finite and positive.")
  return 1.0 / diagonal


def compute_whole_body_effort(
  config: WholeBodyEffortConfig,
  *,
  mass_matrix: FloatArray,
  bias_effort: Sequence[float] | FloatArray,
  left_vertical_jacobian: Sequence[float] | FloatArray,
  right_vertical_jacobian: Sequence[float] | FloatArray,
  leg_dof_ids: Sequence[int],
  wheel_dof_ids: Sequence[int],
  roll_dof_id: int,
  track_width_m: float,
  roll_rad: float,
  roll_rate_rad_s: float,
  leg_positions_rad: Sequence[float],
  leg_velocities_rad_s: Sequence[float],
  leg_position_targets_rad: Sequence[float],
  wheel_velocities_rad_s: Sequence[float],
  wheel_velocity_targets_rad_s: Sequence[float],
) -> WholeBodyEffortResult:
  """Compute one 5 ms bilateral-load / effort command."""

  matrix = np.asarray(mass_matrix, dtype=np.float64)
  if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
    raise ValueError("Mass matrix must be square.")
  nv = matrix.shape[0]
  if not np.isfinite(matrix).all():
    raise ValueError("Mass matrix must be finite.")
  bias = _vector(bias_effort, nv, "bias_effort")
  left_jacobian = _vector(
    left_vertical_jacobian, nv, "left_vertical_jacobian"
  )
  right_jacobian = _vector(
    right_vertical_jacobian, nv, "right_vertical_jacobian"
  )
  leg_ids = _dof_ids(leg_dof_ids, 4, nv, "leg_dof_ids")
  wheel_ids = _dof_ids(wheel_dof_ids, 2, nv, "wheel_dof_ids")
  if set(leg_ids) & set(wheel_ids):
    raise ValueError("Leg and wheel DOF ids must be disjoint.")
  if roll_dof_id < 0 or roll_dof_id >= nv:
    raise ValueError("roll_dof_id is outside the mass matrix.")

  scalars = (track_width_m, roll_rad, roll_rate_rad_s)
  if any(not math.isfinite(value) for value in scalars) or track_width_m <= 0.0:
    raise ValueError("Track width must be positive and roll state finite.")
  leg_positions = _vector(leg_positions_rad, 4, "leg_positions_rad")
  leg_velocities = _vector(leg_velocities_rad_s, 4, "leg_velocities_rad_s")
  leg_targets = _vector(
    leg_position_targets_rad, 4, "leg_position_targets_rad"
  )
  wheel_velocities = _vector(
    wheel_velocities_rad_s, 2, "wheel_velocities_rad_s"
  )
  wheel_targets = _vector(
    wheel_velocity_targets_rad_s, 2, "wheel_velocity_targets_rad_s"
  )

  try:
    inverse_mass = np.linalg.inv(matrix)
  except np.linalg.LinAlgError as exc:
    raise ValueError("Mass matrix is singular.") from exc
  if not np.isfinite(inverse_mass).all():
    raise ValueError("Inverse mass matrix is non-finite.")

  roll_inertia = _effective_inertias(inverse_mass, (roll_dof_id,))[0]
  desired_roll_unclamped = -roll_inertia * (
    config.roll_natural_frequency_rad_s**2 * roll_rad
    + 2.0
    * config.roll_damping_ratio
    * config.roll_natural_frequency_rad_s
    * roll_rate_rad_s
  )
  half_track = 0.5 * track_width_m
  moment_limit = half_track * (
    config.total_support_force_n - 2.0 * config.minimum_side_load_n
  )
  desired_roll = float(
    np.clip(desired_roll_unclamped, -moment_limit, moment_limit)
  )

  allocator_config = SupportLoadAllocatorConfig(
    track_width_m=track_width_m,
    minimum_contact_load_n=config.minimum_side_load_n,
    maximum_contact_load_n=(
      config.total_support_force_n - config.minimum_side_load_n
    ),
    total_load_weight=1.0 / config.total_support_force_n**2,
    roll_moment_weight=1.0 / (
      half_track * config.total_support_force_n
    ) ** 2,
  )
  allocation = allocate_support_loads(
    allocator_config,
    desired_total_load_n=config.total_support_force_n,
    desired_roll_moment_nm=desired_roll,
    left_in_contact=True,
    right_in_contact=True,
  )

  generalized_inverse = (
    bias
    - left_jacobian * allocation.left_load_n
    - right_jacobian * allocation.right_load_n
  )
  actuator_ids = (*leg_ids, *wheel_ids)
  inverse_effort = generalized_inverse[np.asarray(actuator_ids, dtype=np.int64)]

  joint_inertias = _effective_inertias(inverse_mass, leg_ids)
  leg_kp = np.minimum(
    config.leg_kp_max,
    joint_inertias * config.joint_natural_frequency_rad_s**2,
  )
  leg_kd = np.minimum(
    config.leg_kd_max,
    2.0 * joint_inertias * config.joint_natural_frequency_rad_s,
  )
  leg_pd = leg_kp * (leg_targets - leg_positions) - leg_kd * leg_velocities
  raw_leg = inverse_effort[:4] + leg_pd
  leg_effort = np.clip(
    raw_leg,
    -config.leg_torque_limit_nm,
    config.leg_torque_limit_nm,
  )

  wheel_drive = np.clip(
    config.wheel_velocity_damping_nm_per_rad_s
    * (wheel_targets - wheel_velocities),
    -config.wheel_torque_limit_nm,
    config.wheel_torque_limit_nm,
  )
  raw_wheel = inverse_effort[4:] + wheel_drive
  wheel_effort = np.clip(
    raw_wheel,
    -config.wheel_torque_limit_nm,
    config.wheel_torque_limit_nm,
  )

  result = WholeBodyEffortResult(
    allocation=allocation,
    roll_inertia_kg_m2=float(roll_inertia),
    desired_roll_moment_unclamped_nm=float(desired_roll_unclamped),
    desired_roll_moment_nm=desired_roll,
    roll_moment_limit_nm=float(moment_limit),
    leg_kp=leg_kp,
    leg_kd=leg_kd,
    inverse_effort_nm=inverse_effort,
    leg_pd_effort_nm=leg_pd,
    wheel_drive_effort_nm=wheel_drive,
    raw_leg_effort_nm=raw_leg,
    raw_wheel_effort_nm=raw_wheel,
    leg_effort_nm=leg_effort,
    wheel_effort_nm=wheel_effort,
    leg_saturated=np.abs(raw_leg) > config.leg_torque_limit_nm,
    wheel_saturated=np.abs(raw_wheel) > config.wheel_torque_limit_nm,
    wheel_above_rated=np.abs(wheel_effort) > config.wheel_rated_torque_nm,
  )
  if not result.finite:
    raise RuntimeError("Whole-body effort output is non-finite.")
  return result
