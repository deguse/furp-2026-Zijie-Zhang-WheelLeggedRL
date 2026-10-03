import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from hoppertrex_mjlab.deploy.hal import (
  ImuSample,
  JointStates,
  MockImu,
  MockMotorBus,
)
from hoppertrex_mjlab.deploy.loop import (
  ControlLoop,
  WheelOdometryEstimator,
  session_log_to_identification_arrays,
)
from hoppertrex_mjlab.deploy.safety import (
  SafetyState,
  SafetySupervisor,
)
from hoppertrex_mjlab.hybrid.classical_stack import (
  ClassicalCommands,
  ClassicalStackConfig,
)
from hoppertrex_mjlab.hybrid.stair_classical import (
  ContactDetectorCfg,
  StairManeuver,
  StairPhase,
)

LEG_LOWER = (-1.0, -1.0, -1.5, -1.5)
LEG_UPPER = (1.0, 1.0, 1.5, 1.5)


def _config() -> ClassicalStackConfig:
  return ClassicalStackConfig(
    controller_gain=(8.0, 1.0, 3.0, 0.2),
    velocity_command_scale=1.0,
    velocity_command_bias=0.0,
    yaw_feedforward_breakpoints=((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)),
    station_drift_breakpoints=((-1.0, 0.0), (1.0, 0.0)),
    posture_coefficients=(
      (0.4, -0.4, 0.9, -0.9),
      (0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0),
    ),
    action_mask=(True,) * 6,
    action_scales=(0.5, 0.3, 0.035, 0.035, 0.035, 0.035),
    leg_position_lower=LEG_LOWER,
    leg_position_upper=LEG_UPPER,
  )


def _supervisor(bus: MockMotorBus) -> SafetySupervisor:
  return SafetySupervisor(
    bus=bus,
    leg_position_lower=LEG_LOWER,
    leg_position_upper=LEG_UPPER,
  )


def _imu(pitch: float = 0.0, t: float = 0.0) -> ImuSample:
  return ImuSample(
    pitch=pitch,
    pitch_rate=0.0,
    yaw_rate=0.0,
    timestamp_s=t,
    forward_deceleration=0.0,
  )


def _joints(t: float = 0.0) -> JointStates:
  return JointStates(
    leg_positions=(0.4, -0.4, 0.9, -0.9),
    leg_velocities=(0.0, 0.0, 0.0, 0.0),
    wheel_velocities=(0.0, 0.0),
    timestamp_s=t,
  )


class SafetySupervisorTest(unittest.TestCase):
  def test_state_machine_transitions(self):
    bus = MockMotorBus()
    supervisor = _supervisor(bus)
    self.assertIs(supervisor.state, SafetyState.IDLE)
    with self.assertRaises(RuntimeError):
      supervisor.activate()
    supervisor.arm()
    self.assertIs(supervisor.state, SafetyState.ARMED)
    supervisor.activate()
    self.assertIs(supervisor.state, SafetyState.ACTIVE)
    supervisor.fault("test")
    self.assertIs(supervisor.state, SafetyState.FAULT)
    self.assertFalse(bus.torque_enabled)
    with self.assertRaisesRegex(RuntimeError, "reset first"):
      supervisor.arm()
    supervisor.reset()
    self.assertIs(supervisor.state, SafetyState.IDLE)
    self.assertIsNone(supervisor.fault_reason)

  def test_idle_forwards_nothing(self):
    bus = MockMotorBus()
    supervisor = _supervisor(bus)
    forwarded = supervisor.command(
      now_s=0.0,
      imu=_imu(),
      joints=_joints(),
      wheel_velocity_targets=(1.0, 1.0),
      leg_position_targets=(0.0, 0.0, 0.0, 0.0),
    )
    self.assertFalse(forwarded)
    self.assertEqual(bus.sent_targets, [])

  def test_tilt_guard_latches_fault_and_kills_torque(self):
    bus = MockMotorBus()
    supervisor = _supervisor(bus)
    supervisor.arm()
    forwarded = supervisor.command(
      now_s=0.0,
      imu=_imu(pitch=0.40),
      joints=_joints(),
      wheel_velocity_targets=(0.0, 0.0),
      leg_position_targets=(0.0, 0.0, 0.0, 0.0),
    )
    self.assertFalse(forwarded)
    self.assertIs(supervisor.state, SafetyState.FAULT)
    self.assertIn("tilt", supervisor.fault_reason)
    self.assertFalse(bus.torque_enabled)

  def test_roll_guard_checks_both_signs_and_boundary(self):
    for roll in (-0.35, 0.35):
      with self.subTest(roll=roll, disposition="boundary accepted"):
        bus = MockMotorBus()
        supervisor = _supervisor(bus)
        supervisor.arm()
        self.assertTrue(
          supervisor.command(
            now_s=0.0,
            imu=replace(_imu(), roll=roll),
            joints=_joints(),
            wheel_velocity_targets=(0.0, 0.0),
            leg_position_targets=(0.0, 0.0, 0.0, 0.0),
          )
        )
        self.assertEqual(len(bus.sent_targets), 1)

    for roll in (-0.351, 0.351):
      with self.subTest(roll=roll, disposition="over-limit rejected"):
        bus = MockMotorBus()
        supervisor = _supervisor(bus)
        supervisor.arm()
        self.assertFalse(
          supervisor.command(
            now_s=0.0,
            imu=replace(_imu(), roll=roll),
            joints=_joints(),
            wheel_velocity_targets=(0.0, 0.0),
            leg_position_targets=(0.0, 0.0, 0.0, 0.0),
          )
        )
        self.assertIs(supervisor.state, SafetyState.FAULT)
        self.assertIn("roll", supervisor.fault_reason or "")
        self.assertFalse(bus.torque_enabled)
        self.assertEqual(bus.sent_targets, [])

  def test_watchdog_fires_on_command_gap(self):
    bus = MockMotorBus()
    supervisor = _supervisor(bus)
    supervisor.arm()
    self.assertTrue(
      supervisor.command(
        now_s=0.0,
        imu=_imu(t=0.0),
        joints=_joints(t=0.0),
        wheel_velocity_targets=(0.0, 0.0),
        leg_position_targets=(0.0, 0.0, 0.0, 0.0),
      )
    )
    forwarded = supervisor.command(
      now_s=0.10,
      imu=_imu(t=0.10),
      joints=_joints(t=0.10),
      wheel_velocity_targets=(0.0, 0.0),
      leg_position_targets=(0.0, 0.0, 0.0, 0.0),
    )
    self.assertFalse(forwarded)
    self.assertIn("watchdog", supervisor.fault_reason)

  def test_stale_sensors_fault(self):
    bus = MockMotorBus()
    supervisor = _supervisor(bus)
    supervisor.arm()
    forwarded = supervisor.command(
      now_s=1.0,
      imu=_imu(t=0.90),
      joints=_joints(t=1.0),
      wheel_velocity_targets=(0.0, 0.0),
      leg_position_targets=(0.0, 0.0, 0.0, 0.0),
    )
    self.assertFalse(forwarded)
    self.assertIn("stale", supervisor.fault_reason)

  def test_nonfinite_time_or_sensor_feedback_fails_closed(self):
    cases = (
      ("nan now", math.nan, _imu(), _joints()),
      ("infinite now", math.inf, _imu(), _joints()),
      ("nan IMU timestamp", 0.0, replace(_imu(), timestamp_s=math.nan), _joints()),
      ("nan pitch", 0.0, replace(_imu(), pitch=math.nan), _joints()),
      (
        "infinite wheel velocity",
        0.0,
        _imu(),
        replace(_joints(), wheel_velocities=(math.inf, 0.0)),
      ),
    )
    for label, now_s, imu, joints in cases:
      with self.subTest(label=label):
        bus = MockMotorBus()
        supervisor = _supervisor(bus)
        supervisor.arm()
        forwarded = supervisor.command(
          now_s=now_s,
          imu=imu,
          joints=joints,
          wheel_velocity_targets=(0.0, 0.0),
          leg_position_targets=(0.0, 0.0, 0.0, 0.0),
        )
        self.assertFalse(forwarded)
        self.assertIs(supervisor.state, SafetyState.FAULT)
        self.assertFalse(bus.torque_enabled)
        self.assertEqual(bus.sent_targets, [])

  def test_sensor_future_skew_is_bounded(self):
    accepted_bus = MockMotorBus()
    accepted = _supervisor(accepted_bus)
    accepted.arm()
    self.assertTrue(
      accepted.command(
        now_s=0.0,
        imu=_imu(t=0.02),
        joints=_joints(t=0.0),
        wheel_velocity_targets=(0.0, 0.0),
        leg_position_targets=(0.0, 0.0, 0.0, 0.0),
      )
    )

    rejected_bus = MockMotorBus()
    rejected = _supervisor(rejected_bus)
    rejected.arm()
    self.assertFalse(
      rejected.command(
        now_s=0.0,
        imu=_imu(t=0.041),
        joints=_joints(t=0.0),
        wheel_velocity_targets=(0.0, 0.0),
        leg_position_targets=(0.0, 0.0, 0.0, 0.0),
      )
    )
    self.assertIs(rejected.state, SafetyState.FAULT)
    self.assertIn("future", rejected.fault_reason or "")
    self.assertFalse(rejected_bus.torque_enabled)

  def test_control_clock_rollback_fails_closed(self):
    bus = MockMotorBus()
    supervisor = _supervisor(bus)
    supervisor.arm()
    self.assertTrue(
      supervisor.command(
        now_s=1.0,
        imu=_imu(t=1.0),
        joints=_joints(t=1.0),
        wheel_velocity_targets=(0.0, 0.0),
        leg_position_targets=(0.0, 0.0, 0.0, 0.0),
      )
    )
    self.assertFalse(
      supervisor.command(
        now_s=0.99,
        imu=_imu(t=0.99),
        joints=_joints(t=0.99),
        wheel_velocity_targets=(0.0, 0.0),
        leg_position_targets=(0.0, 0.0, 0.0, 0.0),
      )
    )
    self.assertIs(supervisor.state, SafetyState.FAULT)
    self.assertIn("rollback", supervisor.fault_reason or "")
    self.assertFalse(bus.torque_enabled)
    self.assertEqual(len(bus.sent_targets), 1)

  def test_sensor_timestamp_rollback_fails_closed(self):
    for source in ("IMU", "joint"):
      with self.subTest(source=source):
        bus = MockMotorBus()
        supervisor = _supervisor(bus)
        supervisor.arm()
        self.assertTrue(
          supervisor.command(
            now_s=1.0,
            imu=_imu(t=1.0),
            joints=_joints(t=1.0),
            wheel_velocity_targets=(0.0, 0.0),
            leg_position_targets=(0.0, 0.0, 0.0, 0.0),
          )
        )
        imu = _imu(t=0.99 if source == "IMU" else 1.01)
        joints = _joints(t=0.99 if source == "joint" else 1.01)
        self.assertFalse(
          supervisor.command(
            now_s=1.01,
            imu=imu,
            joints=joints,
            wheel_velocity_targets=(0.0, 0.0),
            leg_position_targets=(0.0, 0.0, 0.0, 0.0),
          )
        )
        self.assertIs(supervisor.state, SafetyState.FAULT)
        self.assertIn("timestamp rollback", supervisor.fault_reason or "")
        self.assertFalse(bus.torque_enabled)
        self.assertEqual(len(bus.sent_targets), 1)

  def test_reset_clears_sensor_timestamp_history(self):
    bus = MockMotorBus()
    supervisor = _supervisor(bus)
    supervisor.arm()
    self.assertTrue(
      supervisor.command(
        now_s=1.0,
        imu=_imu(t=1.0),
        joints=_joints(t=1.0),
        wheel_velocity_targets=(0.0, 0.0),
        leg_position_targets=(0.0, 0.0, 0.0, 0.0),
      )
    )
    supervisor.reset()
    supervisor.arm()
    self.assertTrue(
      supervisor.command(
        now_s=0.0,
        imu=_imu(t=0.0),
        joints=_joints(t=0.0),
        wheel_velocity_targets=(0.0, 0.0),
        leg_position_targets=(0.0, 0.0, 0.0, 0.0),
      )
    )

  def test_clamps_wheel_and_leg_targets(self):
    bus = MockMotorBus()
    supervisor = _supervisor(bus)
    supervisor.arm()
    supervisor.command(
      now_s=0.0,
      imu=_imu(),
      joints=_joints(),
      wheel_velocity_targets=(50.0, -50.0),
      leg_position_targets=(9.0, -9.0, 9.0, -9.0),
    )
    wheels, legs = bus.sent_targets[-1]
    self.assertEqual(wheels, (12.0, -12.0))
    self.assertEqual(legs, (1.0, -1.0, 1.5, -1.5))


class ControlLoopTest(unittest.TestCase):
  def test_stair_mode_reaches_deployment_classical_stack(self):
    maneuver = StairManeuver(
      approach_vx=0.08,
      preload_trigger_m=0.10,
      preload_duration_s=0.5,
      preload_height_m=0.29,
      preload_pitch_rad=-0.02,
      contact_vx=0.06,
      climb_vx=0.08,
      drive_feedforward_radps=1.0,
      climb_height_m=0.32,
      climb_pitch_rad=-0.01,
      climb_timeout_s=1.0,
      crest_progress_m=0.40,
      recover_duration_s=0.5,
      detector=ContactDetectorCfg(0.1, 0.2, 1.0),
      maneuver_hash="a" * 64,
      bindings={"controller_schedule_hash": "b" * 64},
    )
    bus = MockMotorBus()
    supervisor = _supervisor(bus)
    supervisor.arm()
    supervisor.activate()
    loop = ControlLoop(
      bus=bus,
      imu=MockImu(),
      supervisor=supervisor,
      config=replace(_config(), stair_maneuver=maneuver),
      commands=ClassicalCommands(height=0.31, pitch=0.01, stair_mode=True),
    )
    loop.tick(0.0)
    self.assertIs(loop.state.stair_state.phase, StairPhase.APPROACH)
    self.assertGreater(loop.state.posture_command[0], 0.0)

  def test_stair_detector_consumes_imu_deceleration(self):
    maneuver = StairManeuver(
      approach_vx=0.08,
      preload_trigger_m=0.10,
      preload_duration_s=0.5,
      preload_height_m=0.29,
      preload_pitch_rad=-0.02,
      contact_vx=0.06,
      climb_vx=0.08,
      drive_feedforward_radps=1.0,
      climb_height_m=0.32,
      climb_pitch_rad=-0.01,
      climb_timeout_s=1.0,
      crest_progress_m=0.40,
      recover_duration_s=0.5,
      detector=ContactDetectorCfg(0.1, 0.2, 1.0, consecutive_ticks=1),
      maneuver_hash="a" * 64,
      bindings={"controller_schedule_hash": "b" * 64},
    )
    bus = MockMotorBus()
    imu = MockImu(forward_deceleration=2.0)
    supervisor = _supervisor(bus)
    supervisor.arm()
    supervisor.activate()
    loop = ControlLoop(
      bus=bus,
      imu=imu,
      supervisor=supervisor,
      config=replace(_config(), stair_maneuver=maneuver),
      commands=ClassicalCommands(height=0.31, stair_mode=True),
    )
    loop.tick(0.0)
    bus.wheel_velocities = [-3.0, 3.0]
    loop.tick(0.02)
    self.assertTrue(loop.state.stair_state.contact_latched)

  def test_roll_signals_reach_bounded_portable_leg_feedback(self):
    bus = MockMotorBus()
    imu = MockImu(roll=0.006, roll_rate=0.004)
    supervisor = _supervisor(bus)
    supervisor.arm()
    supervisor.activate()
    config = replace(
      _config(),
      roll_feedback_kp=0.15,
      roll_feedback_kd=0.02,
      roll_feedback_max_amplitude_rad=0.001,
    )
    loop = ControlLoop(
      bus=bus, imu=imu, supervisor=supervisor, config=config,
    )
    loop.tick(0.0)

    expected_amplitude = 0.15 * imu.roll + 0.02 * imu.roll_rate
    self.assertAlmostEqual(loop.state.roll_feedback_amplitude, expected_amplitude)
    _wheels, legs = bus.sent_targets[-1]
    self.assertNotEqual(legs, (0.4, -0.4, 0.9, -0.9))
    self.assertLessEqual(abs(loop.state.roll_feedback_amplitude), 0.001)

  def test_wheel_odometry_sign_convention(self):
    estimator = WheelOdometryEstimator(wheel_radius=0.1)
    # right - left over two, scaled by radius.
    self.assertAlmostEqual(estimator.estimate(-2.0, 2.0), 0.2)
    self.assertAlmostEqual(estimator.estimate(2.0, -2.0), -0.2)

  def test_closed_loop_runs_and_logs_identification_shape(self):
    bus = MockMotorBus()
    imu = MockImu(schedule=[0.01 * index for index in range(10)])
    supervisor = _supervisor(bus)
    supervisor.arm()
    supervisor.activate()
    with tempfile.TemporaryDirectory() as temp:
      log_path = Path(temp) / "session.jsonl"
      loop = ControlLoop(
        bus=bus,
        imu=imu,
        supervisor=supervisor,
        config=_config(),
        log_path=log_path,
      )
      statistics = loop.run(ticks=100, realtime=False)
      self.assertEqual(statistics.ticks, 100)
      self.assertGreater(len(bus.sent_targets), 90)

      rows = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
      ]
      self.assertEqual(len(rows), 100)
      first = rows[0]
      self.assertEqual(len(first["state"]), 4)
      self.assertEqual(len(first["input"]), 2)
      self.assertEqual(first["supervisor_state"], "active")
      self.assertIn("roll", first)
      self.assertIn("roll_rate", first)
      self.assertIn("roll_feedback_amplitude", first)

      arrays = session_log_to_identification_arrays(log_path)
      self.assertEqual(arrays["states"].shape, (99, 4))
      self.assertEqual(arrays["inputs"].shape, (99, 2))
      self.assertEqual(arrays["next_states"].shape, (99, 4))
      np.testing.assert_array_equal(
        arrays["next_states"][:-1], arrays["states"][1:]
      )

  def test_fault_mid_session_stops_forwarding_but_loop_survives(self):
    bus = MockMotorBus()
    schedule = [0.0] * 5 + [0.5] * 5  # tilt fault at tick 5
    imu = MockImu(schedule=schedule)
    supervisor = _supervisor(bus)
    supervisor.arm()
    supervisor.activate()
    loop = ControlLoop(
      bus=bus,
      imu=imu,
      supervisor=supervisor,
      config=_config(),
    )
    loop.run(ticks=10, realtime=False)
    self.assertIs(supervisor.state, SafetyState.FAULT)
    self.assertIn("tilt", supervisor.fault_reason)
    self.assertFalse(bus.torque_enabled)
    self.assertLessEqual(len(bus.sent_targets), 6)


if __name__ == "__main__":
  unittest.main()
