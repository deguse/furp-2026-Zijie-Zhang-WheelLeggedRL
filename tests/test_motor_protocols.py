import math
import unittest

from hoppertrex_mjlab.deploy.motor_protocols import (
  DmMitMapping,
  decode_dm_mit_feedback,
  decode_rmd_torque_feedback,
  encode_dm_mit_command,
  encode_rmd_torque_current_command,
  iq_current_to_wheel_torque,
  wheel_torque_to_iq_current,
)


class MotorProtocolTest(unittest.TestCase):
  def setUp(self) -> None:
    self.mapping = DmMitMapping(
      position_max_rad=12.566,
      velocity_max_rad_s=20.0,
      torque_max_nm=120.0,
    )

  def test_dm_mit_command_is_exactly_eight_bytes(self) -> None:
    payload = encode_dm_mit_command(
      position_rad=0.0,
      velocity_rad_s=0.0,
      kp=100.0,
      kd=1.0,
      feedforward_torque_nm=0.0,
      mapping=self.mapping,
    )
    self.assertEqual(len(payload), 8)

  def test_dm_feedback_decodes_status_and_signed_mapped_values(self) -> None:
    payload = bytes((0x11, 0x7F, 0xFF, 0x7F, 0xF7, 0xFF, 42, 43))
    feedback = decode_dm_mit_feedback(payload, mapping=self.mapping)
    self.assertEqual(feedback.status_code, 1)
    self.assertEqual(feedback.motor_id, 1)
    self.assertAlmostEqual(feedback.position_rad, 0.0, delta=4.0e-4)
    self.assertAlmostEqual(feedback.velocity_rad_s, 0.0, delta=0.011)
    self.assertAlmostEqual(feedback.torque_estimate_nm, 0.0, delta=0.06)
    self.assertEqual(feedback.mos_temperature_c, 42.0)
    self.assertEqual(feedback.rotor_temperature_c, 43.0)

  def test_dm_command_rejects_mapping_overflow(self) -> None:
    with self.assertRaisesRegex(ValueError, "outside"):
      encode_dm_mit_command(
        position_rad=13.0,
        velocity_rad_s=0.0,
        kp=0.0,
        kd=0.0,
        feedforward_torque_nm=0.0,
        mapping=self.mapping,
      )

  def test_rmd_positive_and_negative_one_amp_golden_frames(self) -> None:
    positive = encode_rmd_torque_current_command(1.0)
    negative = encode_rmd_torque_current_command(-1.0)
    self.assertEqual(positive, bytes((0xA1, 0, 0, 0, 0x64, 0, 0, 0)))
    self.assertEqual(negative, bytes((0xA1, 0, 0, 0, 0x9C, 0xFF, 0, 0)))

  def test_rmd_reply_decodes_current_speed_and_angle(self) -> None:
    payload = bytes((0xA1, 50, 0x64, 0, 0xF4, 0x01, 0x2D, 0))
    feedback = decode_rmd_torque_feedback(payload)
    self.assertEqual(feedback.temperature_c, 50.0)
    self.assertEqual(feedback.iq_current_a, 1.0)
    self.assertAlmostEqual(feedback.output_speed_rad_s, math.radians(500.0))
    self.assertAlmostEqual(feedback.output_angle_rad, math.radians(45.0))

  def test_wheel_torque_conversion_requires_explicit_valid_constant(self) -> None:
    current = wheel_torque_to_iq_current(
      5.776,
      torque_constant_nm_per_a=0.76,
    )
    self.assertAlmostEqual(current, 7.6)
    self.assertAlmostEqual(
      iq_current_to_wheel_torque(
        current,
        torque_constant_nm_per_a=0.76,
      ),
      5.776,
    )
    with self.assertRaisesRegex(ValueError, "positive"):
      wheel_torque_to_iq_current(1.0, torque_constant_nm_per_a=0.0)


if __name__ == "__main__":
  unittest.main()
