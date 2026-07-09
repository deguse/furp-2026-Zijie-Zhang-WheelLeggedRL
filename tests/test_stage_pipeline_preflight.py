import unittest

from hoppertrex_mjlab.scripts.rsl_rl.stage_pipeline_preflight import run_preflight


class StagePipelinePreflightTest(unittest.TestCase):
  def test_stage2_to_stage5_pipeline_preflight_passes(self):
    checks = run_preflight()

    failed = [check for check in checks if not check.passed]

    self.assertEqual(failed, [])

  def test_preflight_covers_current_stage_defaults_and_sustained_controls(self):
    check_names = {check.name for check in run_preflight()}

    for name in (
      "stage2_gate_default_task",
      "stage3_gate_default_task",
      "stage4_gate_default_task",
      "stage5_gate_default_task",
      "stage2_episode_length_s",
      "stage3_episode_length_s",
      "stage4_episode_length_s",
      "stage5_episode_length_s",
      "stage2_balance_smoothing_alpha",
      "stage2_action_dim_kind",
      "stage3_yaw_smoothing_alpha",
      "stage4_balance_smoothing_alpha",
      "stage5_balance_smoothing_alpha",
    ):
      with self.subTest(name=name):
        self.assertIn(name, check_names)


if __name__ == "__main__":
  unittest.main()
