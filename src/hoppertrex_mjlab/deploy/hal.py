"""Hardware abstraction layer for HopperTrex deployment.

Protocol interfaces plus fully programmable mocks. The real adapters
(CAN bus for the DM-J6248P leg motors and RMD L-9025 wheel motors, the
IMU driver) implement these Protocols once the hardware parameters are
known; everything above this layer — safety supervisor, control loop,
classical stack, policy — is hardware-independent and tested against the
mocks.

Conventions (SI, body frame, the same as the simulation contract):

- Wheel joints take velocity targets (rad/s), leg joints position targets
  (rad). Joint order everywhere is the sorted-actuator order
  (thigh_left_01, knee_left, wheel_left, thigh_right_01, knee_right,
  wheel_right); the HAL adapters own any motor-index remapping.
- The IMU yields pitch (rad, +nose-down convention matching the sim's
  projected-gravity construction) and body angular velocity (rad/s).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

LEG_JOINT_COUNT = 4
WHEEL_JOINT_COUNT = 2


@dataclass(frozen=True)
class JointStates:
  """One synchronized read of all six joints."""

  leg_positions: tuple[float, float, float, float]
  leg_velocities: tuple[float, float, float, float]
  wheel_velocities: tuple[float, float]
  timestamp_s: float


@dataclass(frozen=True)
class ImuSample:
  pitch: float
  pitch_rate: float
  yaw_rate: float
  timestamp_s: float


class MotorBus(Protocol):
  """Actuator bus. Implementations must be non-blocking or fast (<2 ms)."""

  def read_joint_states(self) -> JointStates: ...

  def send_targets(
    self,
    *,
    wheel_velocity_targets: tuple[float, float],
    leg_position_targets: tuple[float, float, float, float],
  ) -> None: ...

  def disable_torque(self) -> None:
    """Hard stop: zero torque on every motor. Must always succeed fast."""
    ...


class Imu(Protocol):
  def read(self) -> ImuSample: ...


@dataclass
class MockMotorBus:
  """Programmable first-order motor model for closed-loop dry runs.

  Wheels approach their velocity targets and legs their position targets
  with simple first-order lags, which is enough to exercise the control
  loop, the safety supervisor, and the data recorder without hardware.
  """

  wheel_time_constant_s: float = 0.05
  leg_time_constant_s: float = 0.10
  dt: float = 0.02
  clock: float = 0.0
  torque_enabled: bool = True
  wheel_velocities: list[float] = field(
    default_factory=lambda: [0.0] * WHEEL_JOINT_COUNT
  )
  leg_positions: list[float] = field(
    default_factory=lambda: [0.0] * LEG_JOINT_COUNT
  )
  leg_velocities: list[float] = field(
    default_factory=lambda: [0.0] * LEG_JOINT_COUNT
  )
  sent_targets: list[tuple[tuple[float, float], tuple[float, ...]]] = field(
    default_factory=list
  )

  def read_joint_states(self) -> JointStates:
    return JointStates(
      leg_positions=tuple(self.leg_positions),  # type: ignore[arg-type]
      leg_velocities=tuple(self.leg_velocities),  # type: ignore[arg-type]
      wheel_velocities=tuple(self.wheel_velocities),  # type: ignore[arg-type]
      timestamp_s=self.clock,
    )

  def send_targets(
    self,
    *,
    wheel_velocity_targets: tuple[float, float],
    leg_position_targets: tuple[float, float, float, float],
  ) -> None:
    if not self.torque_enabled:
      return
    self.sent_targets.append(
      (tuple(wheel_velocity_targets), tuple(leg_position_targets))
    )
    self.clock += self.dt
    wheel_alpha = 1.0 - math.exp(-self.dt / self.wheel_time_constant_s)
    leg_alpha = 1.0 - math.exp(-self.dt / self.leg_time_constant_s)
    for index in range(WHEEL_JOINT_COUNT):
      self.wheel_velocities[index] += wheel_alpha * (
        wheel_velocity_targets[index] - self.wheel_velocities[index]
      )
    for index in range(LEG_JOINT_COUNT):
      before = self.leg_positions[index]
      self.leg_positions[index] += leg_alpha * (
        leg_position_targets[index] - self.leg_positions[index]
      )
      self.leg_velocities[index] = (
        (self.leg_positions[index] - before) / self.dt
      )

  def disable_torque(self) -> None:
    self.torque_enabled = False
    self.wheel_velocities = [0.0] * WHEEL_JOINT_COUNT


@dataclass
class MockImu:
  """Scriptable IMU: feed it a pitch trajectory, it replays with rates."""

  dt: float = 0.02
  clock: float = 0.0
  pitch: float = 0.0
  pitch_rate: float = 0.0
  yaw_rate: float = 0.0
  schedule: list[float] = field(default_factory=list)
  _cursor: int = 0

  def read(self) -> ImuSample:
    if self._cursor < len(self.schedule):
      next_pitch = self.schedule[self._cursor]
      self.pitch_rate = (next_pitch - self.pitch) / self.dt
      self.pitch = next_pitch
      self._cursor += 1
    else:
      self.pitch_rate = 0.0
    self.clock += self.dt
    return ImuSample(
      pitch=self.pitch,
      pitch_rate=self.pitch_rate,
      yaw_rate=self.yaw_rate,
      timestamp_s=self.clock,
    )
