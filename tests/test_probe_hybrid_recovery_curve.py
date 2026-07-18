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

  def test_provenance_call_composes_with_a_play_env_cfg(self):
    # Regression: main() once called hybrid_provenance_lines() without the
    # required env_cfg and only failed at machine-room runtime.
    from mjlab.tasks.registry import load_env_cfg

    from hoppertrex_mjlab.scripts.probe_hybrid_recovery_curve import (
      HYBRID_STAGE_TASKS,
      hybrid_provenance_lines,
    )

    lines = hybrid_provenance_lines(
      load_env_cfg(HYBRID_STAGE_TASKS[5], play=True)
    )

    self.assertTrue(lines)
    self.assertTrue(all(isinstance(line, str) for line in lines))


if __name__ == "__main__":
  unittest.main()
