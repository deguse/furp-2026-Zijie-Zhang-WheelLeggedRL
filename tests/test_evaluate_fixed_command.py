import unittest

import torch

from hoppertrex_mjlab.scripts.rsl_rl.evaluate_fixed_command import (
  _command_tracking_health,
  _late_command_health,
  parse_args,
)


class LateCommandHealthTest(unittest.TestCase):
  def test_backward_command_with_negative_velocity_is_not_slow_or_wrong_direction(self):
    late_lin_x = torch.tensor(
      [
        [-0.09, -0.08],
        [-0.10, -0.07],
      ]
    )

    health = _late_command_health(
      late_lin_x=late_lin_x,
      target_lin_x=-0.08,
      stuck_speed=0.01,
    )

    self.assertFalse(torch.any(health["slow_env"]))
    self.assertFalse(torch.any(health["wrong_direction_env"]))

  def test_forward_command_with_negative_velocity_is_wrong_direction(self):
    late_lin_x = torch.tensor(
      [
        [-0.01, 0.08],
        [-0.02, 0.07],
      ]
    )

    health = _late_command_health(
      late_lin_x=late_lin_x,
      target_lin_x=0.08,
      stuck_speed=0.01,
    )

    self.assertTrue(health["wrong_direction_env"][0])
    self.assertFalse(health["wrong_direction_env"][1])

  def test_command_tracking_reports_sample_and_late_direction_rates(self):
    lin_x_by_step = torch.tensor(
      [
        [0.08, 0.08],
        [-0.02, 0.07],
        [0.09, 0.06],
      ]
    )

    metrics = _command_tracking_health(
      lin_x_by_step=lin_x_by_step,
      target_lin_x=0.06,
      stuck_speed=0.01,
      window_steps=2,
    )

    self.assertAlmostEqual(metrics["command_match_frac"], 5.0 / 6.0)
    self.assertAlmostEqual(metrics["wrong_direction_frac"], 1.0 / 6.0)
    self.assertAlmostEqual(metrics["late_wrong_direction_sample_frac"], 0.25)
    self.assertAlmostEqual(metrics["late_wrong_direction_env_frac"], 0.5)

  def test_argparse_accepts_multiple_lin_x_values(self):
    args = parse_args(
      [
        "--task",
        "task",
        "--checkpoint-file",
        "model.pt",
        "--lin-x",
        "-0.06",
        "0.06",
        "0.08",
      ]
    )

    self.assertEqual(args.lin_x, [-0.06, 0.06, 0.08])


if __name__ == "__main__":
  unittest.main()
