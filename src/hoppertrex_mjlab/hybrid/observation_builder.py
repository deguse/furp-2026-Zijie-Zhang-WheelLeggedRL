"""Portable 34-dim observation constructor for the trained hybrid policy.

Mirrors the actor observation group of the hybrid task (nine terms, fixed
order, no normalizer) in pure numpy so the deployed R3 stack can feed the
exported policy without mjlab. The equivalence test pins this builder
against a live CPU env's actor observations element for element.

Term layout (34 = 3+3+3+3+2+6+6+2+6):

  base_lin_vel | base_ang_vel | projected_gravity | velocity_command |
  posture_command | joint_pos_rel (wheel entries zeroed) | joint_vel_rel |
  controller_baseline | applied_residual

Joint order is the sorted-actuator order of the robot asset:
(thigh_left_01, knee_left, wheel_left, thigh_right_01, knee_right,
wheel_right). ``joint_pos_rel`` subtracts the default (initial) joint
positions and zeroes the wheel entries; ``joint_vel_rel`` subtracts the
default joint velocities (zero for every joint on this robot).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

OBSERVATION_DIM = 34
OBSERVATION_TERMS = (
  ("base_lin_vel", 3),
  ("base_ang_vel", 3),
  ("projected_gravity", 3),
  ("velocity_command", 3),
  ("posture_command", 2),
  ("joint_pos", 6),
  ("joint_vel", 6),
  ("controller_baseline", 2),
  ("applied_residual", 6),
)
JOINT_ORDER = (
  "thigh_left_01",
  "knee_left",
  "wheel_left",
  "thigh_right_01",
  "knee_right",
  "wheel_right",
)
WHEEL_JOINT_INDICES = (2, 5)


@dataclass(frozen=True)
class ObservationInputs:
  """One tick of sensor/command/pipeline inputs, SI units, body frame.

  On hardware ``base_lin_vel`` comes from the odometry estimator (only the
  x component feeds the classical layer; the policy consumed the full
  privileged vector in training, so deployment supplies the estimate with
  zeros for unobservable components — a modeled gap, listed in the
  runbook), ``base_ang_vel``/``projected_gravity`` from the IMU, joints
  from encoders, and the last two terms from the classical stack step.
  """

  base_lin_vel: tuple[float, float, float]
  base_ang_vel: tuple[float, float, float]
  projected_gravity: tuple[float, float, float]
  velocity_command: tuple[float, float, float]
  posture_command: tuple[float, float]
  joint_pos: tuple[float, ...]
  joint_vel: tuple[float, ...]
  controller_baseline: tuple[float, float]
  applied_residual: tuple[float, ...]


def build_observation(
  inputs: ObservationInputs,
  *,
  default_joint_pos: tuple[float, ...],
  default_joint_vel: tuple[float, ...] = (0.0,) * 6,
) -> NDArray[np.float32]:
  """Assemble the 34-dim actor observation vector (float32)."""

  joint_pos = np.asarray(inputs.joint_pos, dtype=np.float32)
  joint_vel = np.asarray(inputs.joint_vel, dtype=np.float32)
  defaults_pos = np.asarray(default_joint_pos, dtype=np.float32)
  defaults_vel = np.asarray(default_joint_vel, dtype=np.float32)
  if joint_pos.shape != (6,) or joint_vel.shape != (6,):
    raise ValueError("joint_pos and joint_vel must each contain six values.")
  if defaults_pos.shape != (6,) or defaults_vel.shape != (6,):
    raise ValueError("joint defaults must each contain six values.")
  joint_pos_rel = joint_pos - defaults_pos
  for index in WHEEL_JOINT_INDICES:
    joint_pos_rel[index] = 0.0
  joint_vel_rel = joint_vel - defaults_vel

  parts = (
    np.asarray(inputs.base_lin_vel, dtype=np.float32),
    np.asarray(inputs.base_ang_vel, dtype=np.float32),
    np.asarray(inputs.projected_gravity, dtype=np.float32),
    np.asarray(inputs.velocity_command, dtype=np.float32),
    np.asarray(inputs.posture_command, dtype=np.float32),
    joint_pos_rel,
    joint_vel_rel,
    np.asarray(inputs.controller_baseline, dtype=np.float32),
    np.asarray(inputs.applied_residual, dtype=np.float32),
  )
  for (name, dim), part in zip(OBSERVATION_TERMS, parts):
    if part.shape != (dim,):
      raise ValueError(f"{name} must contain {dim} values, got {part.shape}.")
  observation = np.concatenate(parts).astype(np.float32)
  assert observation.shape == (OBSERVATION_DIM,)
  return observation
