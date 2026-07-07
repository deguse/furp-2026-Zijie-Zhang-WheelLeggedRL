import unittest

import torch

from hoppertrex_mjlab.scripts.rsl_rl.migrate_balance_1d_to_turn_2d import (
  _reset_action_std,
)


class MigrateBalance1dToTurn2dTest(unittest.TestCase):
  def test_reset_action_std_sets_all_target_actions(self):
    actor_state_dict = {
      "distribution.std_param": torch.tensor([0.01, 0.60]),
    }

    report = _reset_action_std(actor_state_dict, 0.15)

    torch.testing.assert_close(
      actor_state_dict["distribution.std_param"],
      torch.tensor([0.15, 0.15]),
    )
    self.assertEqual(report, ["set distribution.std_param to 0.15"])

  def test_reset_action_std_rejects_non_positive_value(self):
    actor_state_dict = {
      "distribution.std_param": torch.tensor([0.01, 0.60]),
    }

    with self.assertRaisesRegex(ValueError, "--action-std must be positive"):
      _reset_action_std(actor_state_dict, 0.0)


if __name__ == "__main__":
  unittest.main()
