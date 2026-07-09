import unittest
from types import SimpleNamespace

import torch

from hoppertrex_mjlab.scripts.rsl_rl.evaluate_fixed_yaw import (
  _apply_yaw_scale_override,
  _yaw_tracking_health,
)


class FixedYawHealthTest(unittest.TestCase):
  def test_apply_yaw_scale_override_updates_wheel_action_scale(self):
    wheel_action = SimpleNamespace(_yaw_scale=2.5)
    wrapped = SimpleNamespace(
      unwrapped=SimpleNamespace(
        action_manager=SimpleNamespace(get_term=lambda name: wheel_action)
      )
    )

    _apply_yaw_scale_override(wrapped, 2.1)

    self.assertEqual(wheel_action._yaw_scale, 2.1)

  def test_apply_yaw_scale_override_leaves_scale_when_none(self):
    wheel_action = SimpleNamespace(_yaw_scale=2.5)
    wrapped = SimpleNamespace(
      unwrapped=SimpleNamespace(
        action_manager=SimpleNamespace(get_term=lambda name: wheel_action)
      )
    )

    _apply_yaw_scale_override(wrapped, None)

    self.assertEqual(wheel_action._yaw_scale, 2.5)

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

  def test_yaw_health_reports_in_band_and_fast_fractions(self):
    yaw_by_step = torch.tensor(
      [
        [0.00, 0.04, 0.08, 0.13],
        [0.00, 0.04, 0.08, 0.13],
      ],
      dtype=torch.float32,
    )
    lin_x_by_step = torch.zeros_like(yaw_by_step)

    metrics = _yaw_tracking_health(
      yaw_by_step=yaw_by_step,
      lin_x_by_step=lin_x_by_step,
      target_yaw=0.08,
      yaw_deadband=0.01,
      lin_drift_speed=0.05,
      window_steps=2,
    )

    self.assertEqual(metrics["in_band_frac"], 0.5)
    self.assertEqual(metrics["fast_frac"], 0.25)
    self.assertEqual(metrics["slow_sample_frac"], 0.25)
    self.assertEqual(metrics["late_in_band_frac"], 0.5)
    self.assertEqual(metrics["late_fast_sample_frac"], 0.25)

  def test_yaw_health_reports_velocity_delta_metrics(self):
    yaw_by_step = torch.tensor(
      [
        [0.02, 0.02],
        [0.05, 0.01],
        [0.11, 0.03],
      ],
      dtype=torch.float32,
    )
    lin_x_by_step = torch.zeros_like(yaw_by_step)

    metrics = _yaw_tracking_health(
      yaw_by_step=yaw_by_step,
      lin_x_by_step=lin_x_by_step,
      target_yaw=0.06,
      yaw_deadband=0.01,
      lin_drift_speed=0.05,
      window_steps=3,
    )

    delta = torch.tensor([0.03, -0.01, 0.06, 0.02])
    self.assertAlmostEqual(metrics["yaw_delta_rms"], torch.sqrt(torch.mean(delta.square())).item())
    self.assertAlmostEqual(
      metrics["yaw_delta_abs_p95"],
      torch.quantile(delta.abs(), 0.95).item(),
    )
    self.assertAlmostEqual(metrics["late_yaw_delta_rms"], metrics["yaw_delta_rms"])
    self.assertAlmostEqual(metrics["late_yaw_delta_abs_p95"], metrics["yaw_delta_abs_p95"])

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
