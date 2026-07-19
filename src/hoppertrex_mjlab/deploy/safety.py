"""Safety supervisor for HopperTrex deployment.

A small explicit state machine that wraps every actuator command. It is
the ONLY component allowed to forward targets to the motor bus, so every
code path — classical stack, policy residual, operator scripts — inherits
the same envelope:

  IDLE -> ARMED -> ACTIVE -> FAULT

- IDLE: torque off, nothing forwarded.
- ARMED: torque on, targets clamped, but the vehicle is expected to be on
  a stand (R0/R1); the tilt guard is active from here on.
- ACTIVE: normal ground operation (R2+).
- FAULT: latched; torque disabled once, all further commands dropped.
  Only an explicit reset() returns to IDLE.

Guards (defaults; the tilt limit is intentionally far outside the sim
qualification envelope |pitch| <= 0.08 rad so it only fires on real loss
of control):

- Tilt: |pitch| > 0.35 rad -> FAULT.
- Watchdog: more than watchdog_s between consecutive commands -> FAULT
  (a stalled control loop must not leave torque on).
- Sensor staleness: joint/IMU timestamps older than watchdog_s -> FAULT.
- Clamps: wheel velocity targets to +-wheel_velocity_limit, leg position
  targets to the artifact leg limits (same numbers the simulation
  pipeline enforces).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from .hal import ImuSample, JointStates, MotorBus

DEFAULT_TILT_LIMIT_RAD = 0.35
DEFAULT_WATCHDOG_S = 0.040
DEFAULT_WHEEL_VELOCITY_LIMIT = 12.0


class SafetyState(enum.Enum):
  IDLE = "idle"
  ARMED = "armed"
  ACTIVE = "active"
  FAULT = "fault"


@dataclass
class SafetySupervisor:
  bus: MotorBus
  leg_position_lower: tuple[float, float, float, float]
  leg_position_upper: tuple[float, float, float, float]
  tilt_limit_rad: float = DEFAULT_TILT_LIMIT_RAD
  watchdog_s: float = DEFAULT_WATCHDOG_S
  wheel_velocity_limit: float = DEFAULT_WHEEL_VELOCITY_LIMIT
  state: SafetyState = SafetyState.IDLE
  fault_reason: str | None = None
  _last_command_time_s: float | None = field(default=None, repr=False)

  def arm(self) -> None:
    if self.state is SafetyState.FAULT:
      raise RuntimeError(
        f"Cannot arm from FAULT (reason: {self.fault_reason}); reset first."
      )
    self.state = SafetyState.ARMED

  def activate(self) -> None:
    if self.state is not SafetyState.ARMED:
      raise RuntimeError("ACTIVE is only reachable from ARMED.")
    self.state = SafetyState.ACTIVE

  def reset(self) -> None:
    """Operator acknowledgement: clear the latch back to IDLE."""

    self.state = SafetyState.IDLE
    self.fault_reason = None
    self._last_command_time_s = None

  def fault(self, reason: str) -> None:
    self.state = SafetyState.FAULT
    self.fault_reason = reason
    self.bus.disable_torque()

  def command(
    self,
    *,
    now_s: float,
    imu: ImuSample,
    joints: JointStates,
    wheel_velocity_targets: tuple[float, float],
    leg_position_targets: tuple[float, float, float, float],
  ) -> bool:
    """Validate and forward one command. Returns True if forwarded."""

    if self.state in (SafetyState.IDLE, SafetyState.FAULT):
      return False
    if (
      self._last_command_time_s is not None
      and now_s - self._last_command_time_s > self.watchdog_s
    ):
      self.fault(
        f"watchdog: {now_s - self._last_command_time_s:.3f} s between "
        f"commands exceeds {self.watchdog_s:.3f} s"
      )
      return False
    if (
      now_s - imu.timestamp_s > self.watchdog_s
      or now_s - joints.timestamp_s > self.watchdog_s
    ):
      self.fault("stale sensors: IMU or joint states older than the watchdog")
      return False
    if abs(imu.pitch) > self.tilt_limit_rad:
      self.fault(
        f"tilt: |pitch| {abs(imu.pitch):.3f} rad exceeds "
        f"{self.tilt_limit_rad:.3f} rad"
      )
      return False
    self._last_command_time_s = now_s
    wheels = tuple(
      min(max(target, -self.wheel_velocity_limit), self.wheel_velocity_limit)
      for target in wheel_velocity_targets
    )
    legs = tuple(
      min(max(target, lower), upper)
      for target, lower, upper in zip(
        leg_position_targets,
        self.leg_position_lower,
        self.leg_position_upper,
      )
    )
    self.bus.send_targets(
      wheel_velocity_targets=wheels,  # type: ignore[arg-type]
      leg_position_targets=legs,  # type: ignore[arg-type]
    )
    return True
