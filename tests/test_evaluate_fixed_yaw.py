import unittest

import torch

from hoppertrex_mjlab.scripts.rsl_rl.evaluate_fixed_yaw import _yaw_tracking_health


class FixedYawHealthTest(unittest.TestCase):
  def test_positive_yaw_detects_clean_late_turning(self):
    yaw_by_step = torch.full((100, 4), 0.08)
    lin_x_by_step = torch.zeros((100, 4))

    metrics = _yaw_tracking_health(
      yaw_by_step=yaw_by_step,
      lin_x_by_step=lin_x_by_step,
      target_yaw=0.07,
      yaw_deadband=0.01,
      lin_drift_speed=0.05,
      window_steps=40,
    )

    self.assertEqual(metrics["command_match_frac"], 1.0)
    self.assertEqual(metrics["wrong_direction_frac"], 0.0)
    self.assertEqual(metrics["late_slow_env_frac"], 0.0)
    self.assertEqual(metrics["late_wrong_direction_env_frac"], 0.0)
    self.assertEqual(metrics["late_lin_drift_env_frac"], 0.0)

  def test_negative_yaw_detects_late_wrong_direction(self):
    yaw_by_step = torch.full((100, 4), -0.07)
    yaw_by_step[-40:, :] = 0.04
    lin_x_by_step = torch.zeros((100, 4))

    metrics = _yaw_tracking_health(
      yaw_by_step=yaw_by_step,
      lin_x_by_step=lin_x_by_step,
      target_yaw=-0.07,
      yaw_deadband=0.01,
      lin_drift_speed=0.05,
      window_steps=40,
    )

    self.assertEqual(metrics["late_wrong_direction_env_frac"], 1.0)

  def test_transient_late_wrong_samples_do_not_fail_env_level_direction(self):
    yaw_by_step = torch.full((100, 4), -0.08)
    yaw_by_step[-40:, :] = -0.08
    yaw_by_step[-40, :] = 0.02
    lin_x_by_step = torch.zeros((100, 4))

    metrics = _yaw_tracking_health(
      yaw_by_step=yaw_by_step,
      lin_x_by_step=lin_x_by_step,
      target_yaw=-0.07,
      yaw_deadband=0.01,
      lin_drift_speed=0.05,
      window_steps=40,
    )

    self.assertGreater(metrics["late_wrong_direction_sample_frac"], 0.0)
    self.assertEqual(metrics["late_wrong_direction_env_frac"], 0.0)

  def test_yaw_health_detects_late_linear_drift(self):
    yaw_by_step = torch.full((100, 4), 0.08)
    lin_x_by_step = torch.zeros((100, 4))
    lin_x_by_step[-40:, :] = 0.08

    metrics = _yaw_tracking_health(
      yaw_by_step=yaw_by_step,
      lin_x_by_step=lin_x_by_step,
      target_yaw=0.07,
      yaw_deadband=0.01,
      lin_drift_speed=0.05,
      window_steps=40,
    )

    self.assertEqual(metrics["late_lin_drift_env_frac"], 1.0)


if __name__ == "__main__":
  unittest.main()
