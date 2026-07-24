"""50 Hz deployment control loop with jitter statistics and data logging.

Wires HAL sensors -> vx estimator -> classical stack -> safety supervisor
-> motor bus at CONTROL_DT_S, recording one JSONL row per tick. The
recorded ``state``/``input`` columns follow the identification NPZ
contract (pitch, pitch_rate, vx_error, signed_wheel_speed_error / wheel
input), so an R1/R2 session log converts directly into re-identification
data for fitting the hardware LQR.

The loop is deliberately synchronous and single-threaded: at 50 Hz with a
measured classical-stack cost of ~0.2 ms there is no need for concurrency,
and a single thread keeps the watchdog semantics honest.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..hybrid.classical_stack import (
  CONTROL_DT_S,
  ClassicalCommands,
  ClassicalSensors,
  ClassicalStackConfig,
  ClassicalStackState,
  classical_step,
  shape_posture_command,
)
from .hal import Imu, MotorBus
from .safety import SafetySupervisor


@dataclass
class WheelOdometryEstimator:
  """vx estimate from wheel speeds: v = r * signed wheel speed.

  The R2 baseline estimator (decision point in the runbook): exact while
  wheels do not slip, degrades under slip/kicks. The signed wheel speed
  convention matches the LQR state construction
  (0.5 * (right - left)).
  """

  wheel_radius: float

  def estimate(
    self, wheel_vel_left: float, wheel_vel_right: float
  ) -> float:
    return self.wheel_radius * 0.5 * (wheel_vel_right - wheel_vel_left)


@dataclass
class LoopStatistics:
  ticks: int = 0
  overruns: int = 0
  jitter_abs_max_s: float = 0.0
  jitter_abs_sum_s: float = 0.0

  @property
  def jitter_abs_mean_s(self) -> float:
    return self.jitter_abs_sum_s / max(self.ticks, 1)


@dataclass
class ControlLoop:
  """One classical (R2) control session; residual policy hooks in later."""

  bus: MotorBus
  imu: Imu
  supervisor: SafetySupervisor
  config: ClassicalStackConfig
  commands: ClassicalCommands = field(default_factory=ClassicalCommands)
  dt: float = CONTROL_DT_S
  log_path: Path | None = None
  statistics: LoopStatistics = field(default_factory=LoopStatistics)
  state: ClassicalStackState = field(default_factory=ClassicalStackState)
  _log_rows: list[dict[str, object]] = field(default_factory=list)

  def tick(self, now_s: float) -> bool:
    """One control tick at time now_s. Returns supervisor forwarding."""

    imu_sample = self.imu.read()
    joints = self.bus.read_joint_states()
    estimator = WheelOdometryEstimator(wheel_radius=self.config.wheel_radius)
    vx_estimate = estimator.estimate(*joints.wheel_velocities)
    sensors = ClassicalSensors(
      pitch=imu_sample.pitch,
      pitch_rate=imu_sample.pitch_rate,
      vx=vx_estimate,
      wheel_vel_left=joints.wheel_velocities[0],
      wheel_vel_right=joints.wheel_velocities[1],
    )
    if self.config.stair_maneuver is None:
      self.state = shape_posture_command(self.state, dt=self.dt)
    commanded_height = (
      self.state.posture_command[0]
      if self.config.stair_maneuver is None
      else self.commands.height
    )
    commanded_pitch = (
      self.state.posture_command[1]
      if self.config.stair_maneuver is None
      else self.commands.pitch
    )
    shaped = ClassicalCommands(
      vx=self.commands.vx,
      wz=self.commands.wz,
      height=commanded_height,
      pitch=commanded_pitch,
      stair_mode=self.commands.stair_mode,
    )
    wheel_targets, leg_targets, self.state = classical_step(
      self.config, self.state, sensors, shaped
    )
    forwarded = self.supervisor.command(
      now_s=now_s,
      imu=imu_sample,
      joints=joints,
      wheel_velocity_targets=(
        float(wheel_targets[0]),
        float(wheel_targets[1]),
      ),
      leg_position_targets=(
        float(leg_targets[0]),
        float(leg_targets[1]),
        float(leg_targets[2]),
        float(leg_targets[3]),
      ),
    )
    if self.log_path is not None:
      signed_wheel_speed = 0.5 * (
        joints.wheel_velocities[1] - joints.wheel_velocities[0]
      )
      self._log_rows.append(
        {
          "t": now_s,
          "state": [
            imu_sample.pitch,
            imu_sample.pitch_rate,
            vx_estimate,
            signed_wheel_speed,
          ],
          "input": [float(wheel_targets[0]), float(wheel_targets[1])],
          "leg_targets": [float(value) for value in leg_targets],
          "leg_positions": list(joints.leg_positions),
          "commands": [shaped.vx, shaped.wz, shaped.height, shaped.pitch],
          "forwarded": forwarded,
          "supervisor_state": self.supervisor.state.value,
        }
      )
    return forwarded

  def run(self, *, ticks: int, realtime: bool = False) -> LoopStatistics:
    """Run a bounded number of ticks; realtime=False for tests/dry runs.

    Clock contract: ``tick`` receives seconds on the same monotonic
    clock the HAL adapters use for their sample timestamps. Real
    adapters stamp with time.perf_counter(); the mocks use a synthetic
    zero-based clock, so the dry-run path feeds zero-based time.
    """

    start = time.perf_counter()
    for index in range(ticks):
      deadline = start + (index + 1) * self.dt
      now = (
        time.perf_counter() if realtime else index * self.dt
      )
      self.tick(now)
      self.statistics.ticks += 1
      if realtime:
        remaining = deadline - time.perf_counter()
        if remaining > 0.0:
          time.sleep(remaining)
        else:
          self.statistics.overruns += 1
          jitter = -remaining
          self.statistics.jitter_abs_max_s = max(
            self.statistics.jitter_abs_max_s, jitter
          )
          self.statistics.jitter_abs_sum_s += jitter
    self.flush_log()
    return self.statistics

  def flush_log(self) -> None:
    if self.log_path is None or not self._log_rows:
      return
    self.log_path.parent.mkdir(parents=True, exist_ok=True)
    with self.log_path.open("a", encoding="utf-8") as stream:
      for row in self._log_rows:
        stream.write(json.dumps(row) + "\n")
    self._log_rows.clear()


def session_log_to_identification_arrays(
  log_path: Path,
) -> dict[str, np.ndarray]:
  """Convert a session JSONL into identification-shaped arrays.

  Returns states/inputs/next_states aligned one tick apart — the exact
  input contract of identify_hybrid_controller.py (the caller still
  splits train/heldout).
  """

  rows = [
    json.loads(line)
    for line in log_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
  ]
  states = np.asarray([row["state"] for row in rows], dtype=np.float64)
  wheel_inputs = np.asarray(
    [row["input"] for row in rows], dtype=np.float64
  )
  if len(rows) < 2:
    raise ValueError("Need at least two ticks to form transitions.")
  return {
    "states": states[:-1],
    "inputs": wheel_inputs[:-1],
    "next_states": states[1:],
  }
