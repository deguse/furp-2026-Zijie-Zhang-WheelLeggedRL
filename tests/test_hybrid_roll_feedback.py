import unittest

import numpy as np

from hoppertrex_mjlab.hybrid.roll_feedback import (
  LEFT_LIFT_BASIS,
  RIGHT_LIFT_BASIS,
  ROLL_FEEDBACK_MAX_IDENTIFIED_AMPLITUDE_RAD,
  roll_feedback_amplitude,
  roll_feedback_leg_offsets,
  validate_roll_feedback_parameters,
)


class RollFeedbackTest(unittest.TestCase):
  def test_amplitude_uses_measured_sign_and_clamps_to_authority(self):
    actual = roll_feedback_amplitude(
      np.array([-0.010, 0.002, 0.010]),
      np.array([0.0, 0.004, 0.0]),
      kp=0.15,
      kd=0.05,
      max_amplitude_rad=0.001,
    )
    np.testing.assert_allclose(
      actual,
      np.array([-0.001, 0.0005, 0.001], dtype=np.float32),
      rtol=0.0,
      atol=1.0e-9,
    )

  def test_zero_authority_is_exactly_inert(self):
    actual = roll_feedback_amplitude(
      np.array([-1.0, 1.0]),
      np.array([5.0, -5.0]),
      kp=10.0,
      kd=10.0,
      max_amplitude_rad=0.0,
    )
    np.testing.assert_array_equal(actual, np.zeros(2, dtype=np.float32))

  def test_leg_offsets_follow_identified_differential_bases(self):
    amplitude = np.float32(0.001)
    offsets = roll_feedback_leg_offsets(amplitude)
    np.testing.assert_allclose(
      offsets,
      np.array([
        amplitude * LEFT_LIFT_BASIS[0],
        -amplitude * RIGHT_LIFT_BASIS[0],
        amplitude * LEFT_LIFT_BASIS[1],
        -amplitude * RIGHT_LIFT_BASIS[1],
      ], dtype=np.float32),
      rtol=0.0,
      atol=0.0,
    )

  def test_parameters_cannot_extrapolate_beyond_identified_bracket(self):
    self.assertEqual(
      validate_roll_feedback_parameters(kp=0.0, kd=0.0, max_amplitude_rad=0.0),
      (0.0, 0.0, 0.0),
    )
    with self.assertRaisesRegex(ValueError, "finite and non-negative"):
      validate_roll_feedback_parameters(kp=-0.1, kd=0.0, max_amplitude_rad=0.0)
    with self.assertRaisesRegex(ValueError, "identified"):
      validate_roll_feedback_parameters(
        kp=0.1,
        kd=0.0,
        max_amplitude_rad=ROLL_FEEDBACK_MAX_IDENTIFIED_AMPLITUDE_RAD + 1e-6,
      )
    with self.assertRaisesRegex(ValueError, "finite"):
      roll_feedback_amplitude(
        float("nan"), 0.0, kp=0.1, kd=0.0, max_amplitude_rad=0.001,
      )


if __name__ == "__main__":
  unittest.main()
