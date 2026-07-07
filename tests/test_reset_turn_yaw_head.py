import unittest

import torch

from hoppertrex_mjlab.scripts.rsl_rl.reset_turn_yaw_head import (
  _copy_actor_with_reset_yaw_head,
)


def _actor_state() -> dict[str, torch.Tensor]:
  return {
    "mlp.0.weight": torch.full((128, 26), 1.0),
    "mlp.0.bias": torch.full((128,), 2.0),
    "mlp.2.weight": torch.full((128, 128), 3.0),
    "mlp.2.bias": torch.full((128,), 4.0),
    "mlp.4.weight": torch.stack(
      [torch.full((128,), 5.0), torch.full((128,), 6.0)]
    ),
    "mlp.4.bias": torch.tensor([7.0, 8.0]),
    "distribution.std_param": torch.tensor([0.11, 0.22]),
  }


class ResetTurnYawHeadTest(unittest.TestCase):
  def test_default_resets_yaw_output_row_to_zero(self):
    source = _actor_state()
    target = _actor_state()
    target["mlp.4.weight"] = torch.zeros((2, 128))
    target["mlp.4.bias"] = torch.zeros(2)
    target["distribution.std_param"] = torch.tensor([0.30, 0.40])

    actor, report = _copy_actor_with_reset_yaw_head(
      source,
      target,
      yaw_std=0.15,
    )

    torch.testing.assert_close(actor["mlp.4.weight"][0], source["mlp.4.weight"][0])
    torch.testing.assert_close(actor["mlp.4.bias"][0], source["mlp.4.bias"][0])
    torch.testing.assert_close(actor["mlp.4.weight"][1], torch.zeros(128))
    self.assertEqual(actor["mlp.4.bias"][1].item(), 0.0)
    torch.testing.assert_close(actor["distribution.std_param"], torch.tensor([0.11, 0.15]))
    self.assertIn("zeroed action[1] yaw output row", "\n".join(report))

  def test_preserve_yaw_head_copies_yaw_output_row(self):
    source = _actor_state()
    target = _actor_state()
    target["mlp.4.weight"] = torch.zeros((2, 128))
    target["mlp.4.bias"] = torch.zeros(2)
    target["distribution.std_param"] = torch.tensor([0.30, 0.40])

    actor, report = _copy_actor_with_reset_yaw_head(
      source,
      target,
      yaw_std=0.45,
      preserve_yaw_head=True,
    )

    torch.testing.assert_close(actor["mlp.4.weight"][1], source["mlp.4.weight"][1])
    torch.testing.assert_close(actor["mlp.4.bias"][1], source["mlp.4.bias"][1])
    torch.testing.assert_close(actor["distribution.std_param"], torch.tensor([0.11, 0.45]))
    self.assertIn("preserved action[1] yaw output row", "\n".join(report))


if __name__ == "__main__":
  unittest.main()
