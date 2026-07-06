import unittest

import torch

from hoppertrex_mjlab.scripts.rsl_rl.evaluate_fixed_command import (
  _late_command_health,
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


if __name__ == "__main__":
  unittest.main()
