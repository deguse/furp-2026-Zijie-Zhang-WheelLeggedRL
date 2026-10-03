import math
import unittest
from dataclasses import replace

from hoppertrex_mjlab.deploy.hal import (
  EffortControlCapabilities,
  EffortMotorBus,
  EffortTargets,
  ImuSample,
  JointStates,
  MockEffortMotorBus,
  MockMotorBus,
)
from hoppertrex_mjlab.deploy.safety import (
  EffortSafetyLimits,
  SafetyState,
  SafetySupervisor,
)

LEG_LOWER = (-1.0, -1.0, -1.5, -1.5)
LEG_UPPER = (1.0, 1.0, 1.5, 1.5)


def _imu(t: float = 0.0) -> ImuSample:
  return ImuSample(
    pitch=0.0,
    pitch_rate=0.0,
    yaw_rate=0.0,
    timestamp_s=t,
    forward_deceleration=0.0,
  )


def _targets(**overrides: object) -> EffortTargets:
  values: dict[str, object] = {
    "wheel_torque_targets_nm": (0.0, 0.0),
    "leg_position_targets": (0.0, 0.0, 0.0, 0.0),
    "leg_velocity_targets": (0.0, 0.0, 0.0, 0.0),
    "leg_kp": (0.0, 0.0, 0.0, 0.0),
    "leg_kd": (0.0, 0.0, 0.0, 0.0),
    "leg_feedforward_torques_nm": (0.0, 0.0, 0.0, 0.0),
  }
  values.update(overrides)
  return EffortTargets(**values)  # type: ignore[arg-type]


def _supervisor(
  bus: MockMotorBus,
  *,
  with_limits: bool = True,
) -> SafetySupervisor:
  return SafetySupervisor(
    bus=bus,
    leg_position_lower=LEG_LOWER,
    leg_position_upper=LEG_UPPER,
    effort_limits=EffortSafetyLimits() if with_limits else None,
  )


class EffortHalTest(unittest.TestCase):
  def test_joint_states_v1_constructor_remains_compatible(self) -> None:
    states = JointStates(
      leg_positions=(0.0, 0.0, 0.0, 0.0),
      leg_velocities=(0.0, 0.0, 0.0, 0.0),
      wheel_velocities=(0.0, 0.0),
      timestamp_s=0.0,
    )
    self.assertIsNone(states.effort)

  def test_legacy_mock_does_not_claim_effort_protocol(self) -> None:
    self.assertFalse(isinstance(MockMotorBus(), EffortMotorBus))
    self.assertTrue(isinstance(MockEffortMotorBus(), EffortMotorBus))

  def test_effort_command_clamps_without_reordering(self) -> None:
    bus = MockEffortMotorBus()
    supervisor = _supervisor(bus)
    supervisor.arm()
    joints = bus.read_joint_states()
    forwarded = supervisor.command_effort(
      now_s=0.0,
      imu=_imu(),
      joints=joints,
      targets=_targets(
        wheel_torque_targets_nm=(9.0, -9.0),
        leg_position_targets=(9.0, -9.0, 9.0, -9.0),
        leg_kp=(0.0, 0.0, 0.0, 0.0),
        leg_kd=(0.0, 0.0, 0.0, 0.0),
        leg_feedforward_torques_nm=(99.0, -99.0, 99.0, -99.0),
      ),
    )
    self.assertTrue(forwarded)
    sent = bus.sent_effort_targets[-1]
    self.assertEqual(sent.wheel_torque_targets_nm, (2.79, -2.79))
    self.assertEqual(sent.leg_position_targets, (1.0, -1.0, 1.5, -1.5))
    self.assertEqual(
      sent.leg_feedforward_torques_nm,
      (30.0, -30.0, 30.0, -30.0),
    )

  def test_effort_command_clips_feedforward_against_predicted_pd(self) -> None:
    bus = MockEffortMotorBus()
    supervisor = _supervisor(bus)
    supervisor.arm()
    forwarded = supervisor.command_effort(
      now_s=0.0,
      imu=_imu(),
      joints=bus.read_joint_states(),
      targets=_targets(
        leg_position_targets=(0.1, 0.0, 0.0, 0.0),
        leg_kp=(100.0, 0.0, 0.0, 0.0),
        leg_feedforward_torques_nm=(30.0, 0.0, 0.0, 0.0),
      ),
    )
    self.assertTrue(forwarded)
    sent = bus.sent_effort_targets[-1]
    self.assertAlmostEqual(sent.leg_feedforward_torques_nm[0], 20.0)

  def test_effort_mode_requires_explicit_limits(self) -> None:
    bus = MockEffortMotorBus()
    supervisor = _supervisor(bus, with_limits=False)
    supervisor.arm()
    self.assertFalse(
      supervisor.command_effort(
        now_s=0.0,
        imu=_imu(),
        joints=bus.read_joint_states(),
        targets=_targets(),
      )
    )
    self.assertIs(supervisor.state, SafetyState.FAULT)
    self.assertIn("safety limits", supervisor.fault_reason or "")

  def test_missing_calibration_fails_closed(self) -> None:
    bus = MockEffortMotorBus(
      capabilities=EffortControlCapabilities(
        leg_mit_impedance=True,
        leg_torque_feedback=True,
        wheel_current_control=True,
        wheel_current_feedback=True,
        calibrated_wheel_torque_mapping=False,
      )
    )
    supervisor = _supervisor(bus)
    supervisor.arm()
    self.assertFalse(
      supervisor.command_effort(
        now_s=0.0,
        imu=_imu(),
        joints=bus.read_joint_states(),
        targets=_targets(),
      )
    )
    self.assertFalse(bus.torque_enabled)
    self.assertIn("capability", supervisor.fault_reason or "")

  def test_stale_effort_telemetry_fails_closed(self) -> None:
    bus = MockEffortMotorBus()
    supervisor = _supervisor(bus)
    supervisor.arm()
    joints = bus.read_joint_states()
    stale = JointStates(
      leg_positions=joints.leg_positions,
      leg_velocities=joints.leg_velocities,
      wheel_velocities=joints.wheel_velocities,
      timestamp_s=1.0,
      effort=joints.effort,
    )
    self.assertFalse(
      supervisor.command_effort(
        now_s=1.0,
        imu=_imu(1.0),
        joints=stale,
        targets=_targets(),
      )
    )
    self.assertIn("stale effort", supervisor.fault_reason or "")

  def test_future_or_nonfinite_effort_timestamp_fails_closed(self) -> None:
    for label, timestamp, reason in (
      ("future", 0.041, "future effort"),
      ("nan", math.nan, "non-finite effort"),
      ("infinite", math.inf, "non-finite effort"),
    ):
      with self.subTest(label=label):
        bus = MockEffortMotorBus()
        supervisor = _supervisor(bus)
        supervisor.arm()
        joints = bus.read_joint_states()
        assert joints.effort is not None
        scripted = replace(
          joints,
          effort=replace(joints.effort, timestamp_s=timestamp),
        )
        self.assertFalse(
          supervisor.command_effort(
            now_s=0.0,
            imu=_imu(),
            joints=scripted,
            targets=_targets(),
          )
        )
        self.assertIs(supervisor.state, SafetyState.FAULT)
        self.assertIn(reason, supervisor.fault_reason or "")
        self.assertFalse(bus.torque_enabled)
        self.assertEqual(bus.sent_effort_targets, [])

  def test_effort_telemetry_timestamp_rollback_fails_closed(self) -> None:
    bus = MockEffortMotorBus()
    supervisor = _supervisor(bus)
    supervisor.arm()
    initial = bus.read_joint_states()
    assert initial.effort is not None
    initial = replace(
      initial,
      timestamp_s=1.0,
      effort=replace(initial.effort, timestamp_s=1.0),
    )
    self.assertTrue(
      supervisor.command_effort(
        now_s=1.0,
        imu=_imu(1.0),
        joints=initial,
        targets=_targets(),
      )
    )

    regressed = bus.read_joint_states()
    assert regressed.effort is not None
    regressed = replace(
      regressed,
      timestamp_s=1.01,
      effort=replace(regressed.effort, timestamp_s=0.99),
    )
    self.assertFalse(
      supervisor.command_effort(
        now_s=1.01,
        imu=_imu(1.01),
        joints=regressed,
        targets=_targets(),
      )
    )
    self.assertIs(supervisor.state, SafetyState.FAULT)
    self.assertIn("timestamp rollback", supervisor.fault_reason or "")
    self.assertFalse(bus.torque_enabled)
    self.assertEqual(len(bus.sent_effort_targets), 1)

  def test_effort_feedback_trip_limits_fail_closed(self) -> None:
    cases = (
      ("leg_torque_estimates_nm", [97.01, 0.0, 0.0, 0.0], "leg torque"),
      ("wheel_iq_currents_a", [7.61, 0.0], "wheel iq"),
      ("wheel_torque_estimates_nm", [5.81, 0.0], "wheel torque"),
    )
    for attribute, values, reason in cases:
      with self.subTest(attribute=attribute):
        bus = MockEffortMotorBus()
        setattr(bus, attribute, values)
        supervisor = _supervisor(bus)
        supervisor.arm()
        self.assertFalse(
          supervisor.command_effort(
            now_s=0.0,
            imu=_imu(),
            joints=bus.read_joint_states(),
            targets=_targets(),
          )
        )
        self.assertIs(supervisor.state, SafetyState.FAULT)
        self.assertIn(reason, supervisor.fault_reason or "")
        self.assertFalse(bus.torque_enabled)
        self.assertEqual(bus.sent_effort_targets, [])

  def test_nonfinite_effort_target_fails_closed(self) -> None:
    bus = MockEffortMotorBus()
    supervisor = _supervisor(bus)
    supervisor.arm()
    self.assertFalse(
      supervisor.command_effort(
        now_s=0.0,
        imu=_imu(),
        joints=bus.read_joint_states(),
        targets=_targets(wheel_torque_targets_nm=(math.nan, 0.0)),
      )
    )
    self.assertIs(supervisor.state, SafetyState.FAULT)
    self.assertFalse(bus.torque_enabled)


if __name__ == "__main__":
  unittest.main()
