"""Controller/residual composition independent of the simulator runtime."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class HybridActionOutput:
  """Final actuator targets and the masked residual exposed to the policy."""

  wheel_targets: NDArray[np.float64]
  leg_targets: NDArray[np.float64]
  applied_residual: NDArray[np.float64]


def _matrix(name: str, value: ArrayLike, columns: int) -> NDArray[np.float64]:
  array = np.asarray(value, dtype=np.float64)
  if array.ndim != 2 or array.shape[1] != columns:
    raise ValueError(f"{name} must have shape (num_envs, {columns}), got {array.shape}.")
  return array


def _vector(name: str, value: ArrayLike, length: int) -> NDArray[np.float64]:
  array = np.asarray(value, dtype=np.float64)
  if array.shape != (length,):
    raise ValueError(f"{name} must have shape ({length},), got {array.shape}.")
  return array


def compose_hybrid_targets(
  *,
  policy_actions: ArrayLike,
  action_mask: tuple[bool, ...],
  action_scales: tuple[float, ...],
  baseline_wheel_targets: ArrayLike,
  nominal_leg_targets: ArrayLike,
  previous_wheel_targets: ArrayLike,
  wheel_slew_limit: float,
  wheel_velocity_limit: float,
  leg_position_lower: ArrayLike,
  leg_position_upper: ArrayLike,
) -> HybridActionOutput:
  """Compose controller targets and six policy residuals into actuator targets."""

  policy = _matrix("policy_actions", policy_actions, 6)
  baseline = _matrix("baseline_wheel_targets", baseline_wheel_targets, 2)
  nominal_legs = _matrix("nominal_leg_targets", nominal_leg_targets, 4)
  previous_wheels = _matrix("previous_wheel_targets", previous_wheel_targets, 2)
  num_envs = policy.shape[0]
  for name, value in (
    ("baseline_wheel_targets", baseline),
    ("nominal_leg_targets", nominal_legs),
    ("previous_wheel_targets", previous_wheels),
  ):
    if value.shape[0] != num_envs:
      raise ValueError(f"{name} must use the same num_envs as policy_actions.")
  if len(action_mask) != 6 or len(action_scales) != 6:
    raise ValueError("action_mask and action_scales must each contain six values.")
  if wheel_slew_limit <= 0.0 or wheel_velocity_limit <= 0.0:
    raise ValueError("wheel limits must be positive.")

  mask = np.asarray(action_mask, dtype=np.float64)
  scales = np.asarray(action_scales, dtype=np.float64)
  residual = np.clip(policy, -1.0, 1.0) * mask * scales

  balance = residual[:, 0]
  yaw = residual[:, 1]
  wheel_residual = np.stack((-balance + yaw, balance + yaw), axis=1)
  desired_wheels = baseline + wheel_residual
  wheel_delta = np.clip(
    desired_wheels - previous_wheels,
    -wheel_slew_limit,
    wheel_slew_limit,
  )
  wheel_targets = np.clip(
    previous_wheels + wheel_delta,
    -wheel_velocity_limit,
    wheel_velocity_limit,
  )

  leg_lower = _vector("leg_position_lower", leg_position_lower, 4)
  leg_upper = _vector("leg_position_upper", leg_position_upper, 4)
  if np.any(leg_lower > leg_upper):
    raise ValueError("leg position lower limits must not exceed upper limits.")
  leg_targets = np.clip(nominal_legs + residual[:, 2:], leg_lower, leg_upper)

  return HybridActionOutput(
    wheel_targets=wheel_targets,
    leg_targets=leg_targets,
    applied_residual=residual,
  )
