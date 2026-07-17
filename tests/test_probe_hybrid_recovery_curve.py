import sys
import unittest
from unittest import mock

from hoppertrex_mjlab.scripts.probe_hybrid_recovery_curve import (
  DEFAULT_SCALES,
  POLICY_MODES,
  parse_args,
)


class RecoveryCurveProtocolTest(unittest.TestCase):
  def test_default_arguments_match_the_formal_rollout(self):
    argv = [
      "probe",
      "--checkpoint-file",
      "model.pt",
      "--output",
      "curve.json",
    ]
    with mock.patch.object(sys, "argv", argv):
      args = parse_args()

    self.assertEqual(args.num_envs, 32)
    self.assertEqual(args.warmup_steps, 300)
    self.assertEqual(args.seed, 1)
    self.assertEqual(tuple(args.kick_scales), DEFAULT_SCALES)

  def test_protocol_covers_three_policies_and_the_preregistered_point(self):
    self.assertEqual(
      POLICY_MODES, ("candidate", "legs_ablated", "zero_residual")
    )
    self.assertIn(8.0, DEFAULT_SCALES)
    self.assertNotIn(1.0, DEFAULT_SCALES)


if __name__ == "__main__":
  unittest.main()
