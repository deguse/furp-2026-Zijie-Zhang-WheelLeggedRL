"""Safety supervisor for HopperTrex deployment.

The legacy position/velocity path remains the default.  The optional effort
path is separately capability-gated and fails closed: it never silently falls
back to legacy targets when torque/current telemetry or calibration is absent.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from .hal import (
  EffortMotorBus,
  EffortTargets,
  ImuSample,
  JointStates,
  MotorBus,
)

DEFAULT_TILT_LIMIT_RAD = 0.35
DEFAULT_ROLL_LIMIT_RAD = 0.35
DEFAULT_WATCHDOG_S = 0.040
DEFAULT_WHEEL_VELOCITY_LIMIT = 12.0
DEFAULT_LEG_TORQUE_LIMIT_NM = 30.0
DEFAULT_WHEEL_TORQUE_LIMIT_NM = 2.79
DEFAULT_LEG_KP_LIMIT = 500.0
DEFAULT_LEG_KD_LIMIT = 5.0
DEFAULT_LEG_VELOCITY_LIMIT_RAD_S = 2.0 * math.pi
DEFAULT_FUTURE_TIMESTAMP_TOLERANCE_S = DEFAULT_WATCHDOG_S
DEFAULT_LEG_FEEDBACK_TRIP_NM = 97.0
DEFAULT_WHEEL_IQ_FEEDBACK_TRIP_A = 7.6
DEFAULT_WHEEL_TORQUE_FEEDBACK_TRIP_NM = 5.8


class SafetyState(enum.Enum):
  IDLE = "idle"
  ARMED = "armed"
  ACTIVE = "active"
  FAULT = "fault"


@dataclass(frozen=True)
class EffortSafetyLimits:
  """Explicit hardware envelope required before effort mode can be used."""

  leg_torque_limits_nm: tuple[float, float, float, float] = (
    DEFAULT_LEG_TORQUE_LIMIT_NM,
  ) * 4
  wheel_torque_limits_nm: tuple[float, float] = (
    DEFAULT_WHEEL_TORQUE_LIMIT_NM,
  ) * 2
  leg_kp_max: float = DEFAULT_LEG_KP_LIMIT
  leg_kd_max: float = DEFAULT_LEG_KD_LIMIT
  leg_velocity_limit_rad_s: float = DEFAULT_LEG_VELOCITY_LIMIT_RAD_S
  leg_feedback_trip_limits_nm: tuple[float, float, float, float] = (
    DEFAULT_LEG_FEEDBACK_TRIP_NM,
  ) * 4
  wheel_iq_feedback_trip_limits_a: tuple[float, float] = (
    DEFAULT_WHEEL_IQ_FEEDBACK_TRIP_A,
  ) * 2
  wheel_torque_feedback_trip_limits_nm: tuple[float, float] = (
    DEFAULT_WHEEL_TORQUE_FEEDBACK_TRIP_NM,
  ) * 2

  def __post_init__(self) -> None:
    if len(self.leg_torque_limits_nm) != 4:
      raise ValueError("Effort safety requires four leg torque limits.")
    if len(self.wheel_torque_limits_nm) != 2:
      raise ValueError("Effort safety requires two wheel torque limits.")
    if len(self.leg_feedback_trip_limits_nm) != 4:
      raise ValueError("Effort safety requires four leg feedback trip limits.")
    if len(self.wheel_iq_feedback_trip_limits_a) != 2:
      raise ValueError("Effort safety requires two wheel-current trip limits.")
    if len(self.wheel_torque_feedback_trip_limits_nm) != 2:
      raise ValueError("Effort safety requires two wheel-torque trip limits.")
    values = (
      *self.leg_torque_limits_nm,
      *self.wheel_torque_limits_nm,
      self.leg_kp_max,
      self.leg_kd_max,
      self.leg_velocity_limit_rad_s,
      *self.leg_feedback_trip_limits_nm,
      *self.wheel_iq_feedback_trip_limits_a,
      *self.wheel_torque_feedback_trip_limits_nm,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
      raise ValueError("Effort safety limits must be finite and positive.")


def _all_finite(values: Sequence[float], expected: int) -> bool:
  return len(values) == expected and all(math.isfinite(value) for value in values)


@dataclass
class SafetySupervisor:
  bus: MotorBus
  leg_position_lower: tuple[float, float, float, float]
  leg_position_upper: tuple[float, float, float, float]
  tilt_limit_rad: float = DEFAULT_TILT_LIMIT_RAD
  watchdog_s: float = DEFAULT_WATCHDOG_S
  # MockImu advances its synthetic sample clock by one 20 ms tick before the
  # loop supplies now_s. A bounded tolerance no larger than the watchdog
  # accepts that contract while still rejecting arbitrary future timestamps.
  future_timestamp_tolerance_s: float = DEFAULT_FUTURE_TIMESTAMP_TOLERANCE_S
  roll_limit_rad: float = DEFAULT_ROLL_LIMIT_RAD
  wheel_velocity_limit: float = DEFAULT_WHEEL_VELOCITY_LIMIT
  # None is deliberate: legacy callers cannot accidentally arm effort mode.
  effort_limits: EffortSafetyLimits | None = None
  state: SafetyState = SafetyState.IDLE
  fault_reason: str | None = None
  _last_command_time_s: float | None = field(default=None, repr=False)
  _last_imu_timestamp_s: float | None = field(default=None, repr=False)
  _last_joint_timestamp_s: float | None = field(default=None, repr=False)
  _last_effort_timestamp_s: float | None = field(default=None, repr=False)

  def __post_init__(self) -> None:
    if not math.isfinite(self.watchdog_s) or self.watchdog_s <= 0.0:
      raise ValueError("watchdog_s must be finite and positive")
    if not math.isfinite(self.tilt_limit_rad) or self.tilt_limit_rad < 0.0:
      raise ValueError("tilt_limit_rad must be finite and non-negative")
    if not math.isfinite(self.roll_limit_rad) or self.roll_limit_rad < 0.0:
      raise ValueError("roll_limit_rad must be finite and non-negative")
    if (
      not math.isfinite(self.future_timestamp_tolerance_s)
      or self.future_timestamp_tolerance_s < 0.0
      or self.future_timestamp_tolerance_s > self.watchdog_s
    ):
      raise ValueError(
        "future_timestamp_tolerance_s must be finite and within the watchdog"
      )

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
    self._last_imu_timestamp_s = None
    self._last_joint_timestamp_s = None
    self._last_effort_timestamp_s = None

  def fault(self, reason: str) -> None:
    self.state = SafetyState.FAULT
    self.fault_reason = reason
    self.bus.disable_torque()

  def _common_guard(
    self,
    *,
    now_s: float,
    imu: ImuSample,
    joints: JointStates,
  ) -> bool:
    if self.state in (SafetyState.IDLE, SafetyState.FAULT):
      return False
    if not math.isfinite(now_s):
      self.fault("non-finite control timestamp")
      return False
    timestamps = (
      ("IMU", imu.timestamp_s),
      ("joint", joints.timestamp_s),
    )
    if any(not math.isfinite(timestamp) for _, timestamp in timestamps):
      self.fault("non-finite sensor timestamp")
      return False
    if (
      self._last_command_time_s is not None
      and not math.isfinite(self._last_command_time_s)
    ):
      self.fault("non-finite previous command timestamp")
      return False
    if self._last_command_time_s is not None:
      command_age_s = now_s - self._last_command_time_s
      if command_age_s < 0.0:
        self.fault(
          f"clock rollback: now {now_s:.6f} s precedes the previous "
          f"command {self._last_command_time_s:.6f} s"
        )
        return False
      if command_age_s > self.watchdog_s:
        self.fault(
          f"watchdog: {command_age_s:.3f} s between commands exceeds "
          f"{self.watchdog_s:.3f} s"
        )
        return False
    timestamp_history = (
      ("IMU", imu.timestamp_s, self._last_imu_timestamp_s),
      ("joint", joints.timestamp_s, self._last_joint_timestamp_s),
    )
    for label, timestamp, previous in timestamp_history:
      if previous is not None and (
        not math.isfinite(previous) or timestamp < previous
      ):
        self.fault(f"{label} timestamp rollback detected")
        return False
    for label, timestamp in timestamps:
      age_s = now_s - timestamp
      if age_s > self.watchdog_s:
        self.fault(
          f"stale sensors: {label} sample is {age_s:.3f} s old, beyond "
          f"the {self.watchdog_s:.3f} s watchdog"
        )
        return False
      if age_s < -self.future_timestamp_tolerance_s:
        self.fault(
          f"future sensor timestamp: {label} sample leads control time by "
          f"{-age_s:.3f} s"
        )
        return False
    sensor_vectors = (
      (
        (
          imu.pitch,
          imu.pitch_rate,
          imu.yaw_rate,
          imu.forward_deceleration,
          imu.roll,
          imu.roll_rate,
        ),
        6,
      ),
      (joints.leg_positions, 4),
      (joints.leg_velocities, 4),
      (joints.wheel_velocities, 2),
    )
    if any(
      not _all_finite(values, expected)
      for values, expected in sensor_vectors
    ):
      self.fault("IMU or joint feedback contains non-finite values")
      return False
    if abs(imu.pitch) > self.tilt_limit_rad:
      self.fault(
        f"tilt: |pitch| {abs(imu.pitch):.3f} rad exceeds "
        f"{self.tilt_limit_rad:.3f} rad"
      )
      return False
    if abs(imu.roll) > self.roll_limit_rad:
      self.fault(
        f"tilt: |roll| {abs(imu.roll):.3f} rad exceeds "
        f"{self.roll_limit_rad:.3f} rad"
      )
      return False
    return True

  def command(
    self,
    *,
    now_s: float,
    imu: ImuSample,
    joints: JointStates,
    wheel_velocity_targets: tuple[float, float],
    leg_position_targets: tuple[float, float, float, float],
  ) -> bool:
    """Validate and forward one legacy command. Returns True if forwarded."""

    if not self._common_guard(now_s=now_s, imu=imu, joints=joints):
      return False
    legacy_targets = (
      (wheel_velocity_targets, 2),
      (leg_position_targets, 4),
    )
    if any(
      not _all_finite(values, expected)
      for values, expected in legacy_targets
    ):
      self.fault("legacy command contains non-finite or malformed targets")
      return False
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
        strict=True,
      )
    )
    self.bus.send_targets(
      wheel_velocity_targets=wheels,  # type: ignore[arg-type]
      leg_position_targets=legs,  # type: ignore[arg-type]
    )
    self._last_command_time_s = now_s
    self._last_imu_timestamp_s = imu.timestamp_s
    self._last_joint_timestamp_s = joints.timestamp_s
    return True

  def command_effort(
    self,
    *,
    now_s: float,
    imu: ImuSample,
    joints: JointStates,
    targets: EffortTargets,
  ) -> bool:
    """Capability-gate, constrain, and forward one v2 effort command."""

    if not self._common_guard(now_s=now_s, imu=imu, joints=joints):
      return False
    if self.effort_limits is None:
      self.fault("effort mode has no explicit safety limits")
      return False
    if not isinstance(self.bus, EffortMotorBus):
      self.fault("motor bus does not implement the effort protocol")
      return False
    capabilities = self.bus.get_effort_capabilities()
    if not capabilities.supports_bilateral_load_control:
      self.fault("motor bus lacks a required effort capability or calibration")
      return False

    telemetry = joints.effort
    if telemetry is None:
      self.fault("effort telemetry is required in effort mode")
      return False
    if not math.isfinite(telemetry.timestamp_s):
      self.fault("non-finite effort telemetry timestamp")
      return False
    telemetry_age_s = now_s - telemetry.timestamp_s
    if telemetry_age_s > self.watchdog_s:
      self.fault("stale effort telemetry older than the watchdog")
      return False
    if telemetry_age_s < -self.future_timestamp_tolerance_s:
      self.fault("future effort telemetry timestamp beyond tolerance")
      return False
    if self._last_effort_timestamp_s is not None and (
      not math.isfinite(self._last_effort_timestamp_s)
      or telemetry.timestamp_s < self._last_effort_timestamp_s
    ):
      self.fault("effort telemetry timestamp rollback detected")
      return False
    telemetry_vectors = (
      (telemetry.leg_torque_estimates_nm, 4),
      (telemetry.wheel_iq_currents_a, 2),
      (telemetry.wheel_torque_estimates_nm, 2),
    )
    if any(
      values is None or not _all_finite(values, expected)
      for values, expected in telemetry_vectors
    ):
      self.fault("effort telemetry is incomplete or non-finite")
      return False

    limits = self.effort_limits
    feedback_trip_vectors = (
      (
        telemetry.leg_torque_estimates_nm,
        limits.leg_feedback_trip_limits_nm,
        "leg torque feedback",
      ),
      (
        telemetry.wheel_iq_currents_a,
        limits.wheel_iq_feedback_trip_limits_a,
        "wheel iq feedback",
      ),
      (
        telemetry.wheel_torque_estimates_nm,
        limits.wheel_torque_feedback_trip_limits_nm,
        "wheel torque feedback",
      ),
    )
    for values, trip_limits, label in feedback_trip_vectors:
      assert values is not None
      if any(
        abs(value) > trip_limit
        for value, trip_limit in zip(values, trip_limits, strict=True)
      ):
        self.fault(f"{label} exceeded its immediate trip limit")
        return False

    target_vectors = (
      (targets.wheel_torque_targets_nm, 2),
      (targets.leg_position_targets, 4),
      (targets.leg_velocity_targets, 4),
      (targets.leg_kp, 4),
      (targets.leg_kd, 4),
      (targets.leg_feedforward_torques_nm, 4),
      (joints.leg_positions, 4),
      (joints.leg_velocities, 4),
      ((imu.pitch, imu.pitch_rate, imu.roll, imu.roll_rate), 4),
    )
    if any(
      not _all_finite(values, expected)
      for values, expected in target_vectors
    ):
      self.fault("effort command or feedback contains non-finite values")
      return False

    positions = tuple(
      min(max(target, lower), upper)
      for target, lower, upper in zip(
        targets.leg_position_targets,
        self.leg_position_lower,
        self.leg_position_upper,
        strict=True,
      )
    )
    velocities = tuple(
      min(
        max(target, -limits.leg_velocity_limit_rad_s),
        limits.leg_velocity_limit_rad_s,
      )
      for target in targets.leg_velocity_targets
    )
    kp = tuple(min(max(value, 0.0), limits.leg_kp_max) for value in targets.leg_kp)
    kd = tuple(min(max(value, 0.0), limits.leg_kd_max) for value in targets.leg_kd)
    wheels = tuple(
      min(max(target, -limit), limit)
      for target, limit in zip(
        targets.wheel_torque_targets_nm,
        limits.wheel_torque_limits_nm,
        strict=True,
      )
    )

    feedforward: list[float] = []
    for index, torque_limit in enumerate(limits.leg_torque_limits_nm):
      pd_torque = (
        kp[index] * (positions[index] - joints.leg_positions[index])
        + kd[index] * (velocities[index] - joints.leg_velocities[index])
      )
      lower = max(-torque_limit, -torque_limit - pd_torque)
      upper = min(torque_limit, torque_limit - pd_torque)
      if lower > upper:
        self.fault(
          f"leg {index} impedance term cannot fit the effort envelope"
        )
        return False
      feedforward.append(
        min(max(targets.leg_feedforward_torques_nm[index], lower), upper)
      )

    safe_targets = EffortTargets(
      wheel_torque_targets_nm=wheels,  # type: ignore[arg-type]
      leg_position_targets=positions,  # type: ignore[arg-type]
      leg_velocity_targets=velocities,  # type: ignore[arg-type]
      leg_kp=kp,  # type: ignore[arg-type]
      leg_kd=kd,  # type: ignore[arg-type]
      leg_feedforward_torques_nm=tuple(feedforward),  # type: ignore[arg-type]
    )
    self.bus.send_effort_targets(safe_targets)
    self._last_command_time_s = now_s
    self._last_imu_timestamp_s = imu.timestamp_s
    self._last_joint_timestamp_s = joints.timestamp_s
    self._last_effort_timestamp_s = telemetry.timestamp_s
    return True
