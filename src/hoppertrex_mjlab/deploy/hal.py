"""Hardware abstraction layer for HopperTrex deployment.

Protocol interfaces plus fully programmable mocks. The real adapters
(CAN bus for the DM-J6248P leg motors and RMD L-9025 wheel motors, the
IMU driver) implement these Protocols once the hardware parameters are
known; everything above this layer — safety supervisor, control loop,
classical stack, policy — is hardware-independent and tested against the
mocks.

Conventions (SI, body frame, the same as the simulation contract):

- Wheel joints take velocity targets (rad/s), leg joints position targets
  (rad). Public leg tuples use (thigh_left_01, thigh_right_01, knee_left,
  knee_right); wheel tuples use (wheel_left, wheel_right). HAL adapters own
  every motor-index and sign remapping.
- The IMU yields pitch/roll (rad; pitch uses the +nose-down convention
  matching the sim's projected-gravity construction), body angular velocity
  (rad/s), and gravity-compensated forward body deceleration (m/s^2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

LEG_JOINT_COUNT = 4
WHEEL_JOINT_COUNT = 2

# Public tuple order. This is the same order used by the posture map and the
# hybrid action term; hardware adapters own motor-index/sign remapping.
LEG_JOINT_NAMES = (
  "thigh_left_01",
  "thigh_right_01",
  "knee_left",
  "knee_right",
)
WHEEL_JOINT_NAMES = ("wheel_left", "wheel_right")


@dataclass(frozen=True)
class JointEffortTelemetry:
  """Synchronized effort telemetry without conflating torque and current.

  DaMiao MIT feedback reports an estimated output torque. RMD 0xA1/0x9C
  reports q-axis current; a wheel torque estimate is only valid when the real
  adapter has an independently calibrated output torque constant.
  """

  timestamp_s: float
  leg_torque_estimates_nm: tuple[float, float, float, float] | None = None
  wheel_iq_currents_a: tuple[float, float] | None = None
  wheel_torque_estimates_nm: tuple[float, float] | None = None


@dataclass(frozen=True)
class JointStates:
  """One synchronized read of all six joints."""

  leg_positions: tuple[float, float, float, float]
  leg_velocities: tuple[float, float, float, float]
  wheel_velocities: tuple[float, float]
  timestamp_s: float
  # Appended default keeps all v1 adapters and call sites source-compatible.
  effort: JointEffortTelemetry | None = None


@dataclass(frozen=True)
class EffortControlCapabilities:
  """Capabilities required by the explicit support-load controller."""

  leg_mit_impedance: bool = False
  leg_torque_feedback: bool = False
  wheel_current_control: bool = False
  wheel_current_feedback: bool = False
  calibrated_wheel_torque_mapping: bool = False

  @property
  def supports_bilateral_load_control(self) -> bool:
    return all(
      (
        self.leg_mit_impedance,
        self.leg_torque_feedback,
        self.wheel_current_control,
        self.wheel_current_feedback,
        self.calibrated_wheel_torque_mapping,
      )
    )


@dataclass(frozen=True)
class EffortTargets:
  """One synchronized wheel-torque / leg-MIT command in SI units."""

  wheel_torque_targets_nm: tuple[float, float]
  leg_position_targets: tuple[float, float, float, float]
  leg_velocity_targets: tuple[float, float, float, float]
  leg_kp: tuple[float, float, float, float]
  leg_kd: tuple[float, float, float, float]
  leg_feedforward_torques_nm: tuple[float, float, float, float]


@dataclass(frozen=True)
class ImuSample:
  pitch: float
  pitch_rate: float
  yaw_rate: float
  timestamp_s: float
  forward_deceleration: float
  # Appended defaults preserve compatibility with existing IMU adapters while
  # exposing the signals needed by the default-inert roll controller.
  roll: float = 0.0
  roll_rate: float = 0.0


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


@runtime_checkable
class EffortMotorBus(MotorBus, Protocol):
  """Optional v2 bus; v1 buses deliberately do not satisfy this protocol."""

  def get_effort_capabilities(self) -> EffortControlCapabilities: ...

  def send_effort_targets(self, targets: EffortTargets) -> None: ...


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
class MockEffortMotorBus(MockMotorBus):
  """Programmable v2 mock kept separate from the legacy bus by design."""

  capabilities: EffortControlCapabilities = field(
    default_factory=lambda: EffortControlCapabilities(
      leg_mit_impedance=True,
      leg_torque_feedback=True,
      wheel_current_control=True,
      wheel_current_feedback=True,
      calibrated_wheel_torque_mapping=True,
    )
  )
  leg_torque_estimates_nm: list[float] = field(
    default_factory=lambda: [0.0] * LEG_JOINT_COUNT
  )
  wheel_iq_currents_a: list[float] = field(
    default_factory=lambda: [0.0] * WHEEL_JOINT_COUNT
  )
  wheel_torque_estimates_nm: list[float] = field(
    default_factory=lambda: [0.0] * WHEEL_JOINT_COUNT
  )
  sent_effort_targets: list[EffortTargets] = field(default_factory=list)

  def get_effort_capabilities(self) -> EffortControlCapabilities:
    return self.capabilities

  def read_joint_states(self) -> JointStates:
    return JointStates(
      leg_positions=tuple(self.leg_positions),  # type: ignore[arg-type]
      leg_velocities=tuple(self.leg_velocities),  # type: ignore[arg-type]
      wheel_velocities=tuple(self.wheel_velocities),  # type: ignore[arg-type]
      timestamp_s=self.clock,
      effort=JointEffortTelemetry(
        timestamp_s=self.clock,
        leg_torque_estimates_nm=tuple(  # type: ignore[arg-type]
          self.leg_torque_estimates_nm
        ),
        wheel_iq_currents_a=tuple(  # type: ignore[arg-type]
          self.wheel_iq_currents_a
        ),
        wheel_torque_estimates_nm=tuple(  # type: ignore[arg-type]
          self.wheel_torque_estimates_nm
        ),
      ),
    )

  def send_effort_targets(self, targets: EffortTargets) -> None:
    if not self.torque_enabled:
      return
    self.sent_effort_targets.append(targets)
    leg_before = tuple(self.leg_positions)
    self.clock += self.dt
    for index in range(WHEEL_JOINT_COUNT):
      self.wheel_velocities[index] += (
        targets.wheel_torque_targets_nm[index] * self.dt
      )
      self.wheel_torque_estimates_nm[index] = (
        targets.wheel_torque_targets_nm[index]
      )
      # The mock intentionally does not invent a real motor Kt. A unit test
      # can script iq independently when current feedback matters.
    for index in range(LEG_JOINT_COUNT):
      position_error = (
        targets.leg_position_targets[index] - leg_before[index]
      )
      velocity_error = (
        targets.leg_velocity_targets[index] - self.leg_velocities[index]
      )
      torque = (
        targets.leg_kp[index] * position_error
        + targets.leg_kd[index] * velocity_error
        + targets.leg_feedforward_torques_nm[index]
      )
      self.leg_torque_estimates_nm[index] = torque
      leg_alpha = 1.0 - math.exp(-self.dt / self.leg_time_constant_s)
      self.leg_positions[index] += leg_alpha * position_error
      self.leg_velocities[index] = (
        self.leg_positions[index] - leg_before[index]
      ) / self.dt


@dataclass
class MockImu:
  """Scriptable IMU: feed it a pitch trajectory, it replays with rates."""

  dt: float = 0.02
  clock: float = 0.0
  pitch: float = 0.0
  pitch_rate: float = 0.0
  yaw_rate: float = 0.0
  roll: float = 0.0
  roll_rate: float = 0.0
  forward_deceleration: float = 0.0
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
      forward_deceleration=self.forward_deceleration,
      roll=self.roll,
      roll_rate=self.roll_rate,
    )
