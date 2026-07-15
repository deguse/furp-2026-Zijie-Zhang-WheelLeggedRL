import unittest

import numpy as np

from hoppertrex_mjlab.hybrid.yaw_calibration import (
  parse_yaw_calibration_artifact,
  validate_yaw_breakpoints,
  yaw_calibration_artifact,
  yaw_calibration_hash,
  yaw_feedforward,
)


CONTROLLER_HASH = "a" * 64
BREAKPOINTS = (
  (-0.10, -0.55),
  (-0.05, -0.28),
  (0.0, 0.0),
  (0.05, 0.28),
  (0.10, 0.55),
)
SOURCE_PROBE = {
  "git_sha": "44a44b1",
  "device": "cpu",
  "num_envs": 8,
  "steps": 200,
}


class YawCalibrationContractTest(unittest.TestCase):
  def test_artifact_round_trip_preserves_breakpoints_and_kp(self):
    payload = yaw_calibration_artifact(
      controller_gain_hash=CONTROLLER_HASH,
      breakpoints=BREAKPOINTS,
      source_probe=SOURCE_PROBE,
    )
    parsed = parse_yaw_calibration_artifact(
      payload,
      controller_gain_hash=CONTROLLER_HASH,
    )

    self.assertEqual(parsed.breakpoints, BREAKPOINTS)
    self.assertEqual(parsed.kp, 0.0)
    self.assertEqual(parsed.controller_gain_hash, CONTROLLER_HASH)
    self.assertEqual(len(parsed.yaw_calibration_hash), 64)
    self.assertEqual(payload["schema_version"], 1)

  def test_hash_is_deterministic_and_excludes_itself(self):
    payload = yaw_calibration_artifact(
      controller_gain_hash=CONTROLLER_HASH,
      breakpoints=BREAKPOINTS,
      source_probe=SOURCE_PROBE,
    )
    self.assertEqual(
      payload["yaw_calibration_hash"],
      yaw_calibration_hash(payload),
    )

  def test_tampered_breakpoints_are_rejected(self):
    payload = yaw_calibration_artifact(
      controller_gain_hash=CONTROLLER_HASH,
      breakpoints=BREAKPOINTS,
      source_probe=SOURCE_PROBE,
    )
    payload["breakpoints"][-1][1] = 0.99

    with self.assertRaisesRegex(ValueError, "hash"):
      parse_yaw_calibration_artifact(
        payload,
        controller_gain_hash=CONTROLLER_HASH,
      )

  def test_controller_binding_is_enforced(self):
    payload = yaw_calibration_artifact(
      controller_gain_hash=CONTROLLER_HASH,
      breakpoints=BREAKPOINTS,
      source_probe=SOURCE_PROBE,
    )

    with self.assertRaisesRegex(ValueError, "different controller"):
      parse_yaw_calibration_artifact(
        payload,
        controller_gain_hash="b" * 64,
      )

  def test_breakpoints_must_pin_zero_and_be_monotone(self):
    with self.assertRaisesRegex(ValueError, "pin"):
      validate_yaw_breakpoints(((-0.10, -0.55), (0.10, 0.55)))
    with self.assertRaisesRegex(ValueError, "strictly increase"):
      validate_yaw_breakpoints(((0.0, 0.0), (0.0, 0.1)))
    with self.assertRaisesRegex(ValueError, "non-decreasing"):
      validate_yaw_breakpoints(((0.0, 0.0), (0.05, 0.3), (0.10, 0.2)))
    with self.assertRaisesRegex(ValueError, "at least two"):
      validate_yaw_breakpoints(((0.0, 0.0),))
    with self.assertRaisesRegex(ValueError, "finite"):
      validate_yaw_breakpoints(((0.0, 0.0), (0.1, float("nan"))))

  def test_negative_kp_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "kp"):
      yaw_calibration_artifact(
        controller_gain_hash=CONTROLLER_HASH,
        breakpoints=BREAKPOINTS,
        kp=-0.1,
        source_probe=SOURCE_PROBE,
      )

  def test_feedforward_interpolates_and_clamps(self):
    np.testing.assert_allclose(
      yaw_feedforward(0.0, BREAKPOINTS),
      0.0,
    )
    np.testing.assert_allclose(
      yaw_feedforward(0.05, BREAKPOINTS),
      0.28,
    )
    np.testing.assert_allclose(
      yaw_feedforward(0.075, BREAKPOINTS),
      0.5 * (0.28 + 0.55),
    )
    np.testing.assert_allclose(
      yaw_feedforward(-0.075, BREAKPOINTS),
      -0.5 * (0.28 + 0.55),
    )
    np.testing.assert_allclose(
      yaw_feedforward(0.5, BREAKPOINTS),
      0.55,
    )
    np.testing.assert_allclose(
      yaw_feedforward(-0.5, BREAKPOINTS),
      -0.55,
    )
    np.testing.assert_allclose(
      yaw_feedforward(np.array([-0.10, 0.0, 0.10]), BREAKPOINTS),
      np.array([-0.55, 0.0, 0.55]),
    )


if __name__ == "__main__":
  unittest.main()
