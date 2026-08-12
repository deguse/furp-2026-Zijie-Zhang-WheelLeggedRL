import re
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
R0 = ROOT / "scripts" / "run_roll_boundary.ps1"
R1 = ROOT / "scripts" / "run_stair_roll_assist.ps1"


class WrapperTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.r0 = R0.read_text(encoding="utf-8-sig")
    cls.r1 = R1.read_text(encoding="utf-8-sig")

  def test_r0_is_hash_pinned_non_overwriting_and_has_no_training(self):
    for fragment in (
      "codex/p2-classical-upper-bound",
      "43e0f3ea9c92ddbb4de9f3bb1ac772d604e3ebf6",
      "8fe8548bca85978c164bbd7de39d2d6463cdfd8d7ab91796cf57696b0f64e203",
      "git status --porcelain", "origin/$Branch", ".incomplete.",
      "Refusing to overwrite", "probe_roll_boundary", "--max-height-mm",
      "controller_schedule_hash", "action_mask", "SHA256SUMS.txt",
      "Probe Git provenance drifted", "$env:PYTHONPATH",
      "run Probe20 next", "run Probe30 next", "stair PPO is forbidden",
    ):
      self.assertIn(fragment, self.r0)
    self.assertNotIn("scripts.rsl_rl.train", self.r0)
    self.assertNotIn("StairDynamic", self.r0)

  def test_r1_has_only_approved_phases_and_exact_training_contract(self):
    for phase in (
      "Validate", "MeasureReward", "CalibrateReward", "Train100",
      "Envelope", "Screen", "SelectK3", "Evaluate", "ExtendBlock", "Package",
    ):
      self.assertIn(f"'{phase}'", self.r1)
    for fragment in (
      "HopperTrex-Hybrid-v2-StairRollAssist",
      "probe_roll_assist_reward_stall", "calibrate_roll_assist_reward",
      "evaluate_roll_assist", "adjudicate_roll_assist",
      "--agent.seed 1", "--agent.max-iterations 100",
      "--agent.save-interval 25", "--agent.num-steps-per-env 24",
      "--agent.resume False", "--gpu-ids '[0]'", "--env.scene.num-envs 256",
      "HOPPERTREX_ROLL_ASSIST_R0_PATH", "RollBoundary does not authorize RollAssist",
      "HOPPERTREX_ROLL_ASSIST_REWARD_CALIBRATION_PATH",
      "HOPPERTREX_ROLL_ASSIST_EXTENSION_AUTHORIZATION_PATH",
      "--profile screen", "--profile formal", "checkpoint-envelope",
      "Evidence collection and training phases are pinned to cuda:0",
      "--agent.resume True", "TargetTotalUpdates", "SHA256SUMS.txt",
      "[string]$Selection", "select_k3_u${LatestUpdate}.json",
      "Selected checkpoint differs from formal evidence",
      "Evaluate requires the selected K=3 checkpoint",
      "ExtendBlock requires the selected K=3 checkpoint",
      ".package.incomplete.", "Refusing duplicate Train100 run name",
      "simulation_only=$true", "provisional=$true",
      "Package checkpoint differs from formal evidence",
      "Package requires selected K=3 checkpoint evidence",
      "RollBoundary Git SHA differs from the current checkout",
      "RollBoundary controller schedule differs from the frozen C1 schedule",
      "ROLL_ASSIST_K3_NO_PASSER",
      "No-passer package requires exactly three screen candidates",
      "validate-k3 --selection $Selection --verify-screen-files",
      "--reward-calibration $Reward", "$env:PYTHONPATH",
      "[IO.Path]::GetFileNameWithoutExtension($Checkpoint)",
      "FORMAL_GATE_REJECTED_CONTINUATION",
      "8fe8548bca85978c164bbd7de39d2d6463cdfd8d7ab91796cf57696b0f64e203",
    ):
      self.assertIn(fragment, self.r1)
    for forbidden in (
      "-LeafBase", "StairDynamic", "migrate_stage5_to_stair_dynamic", "3000",
      "ROLL_ASSIST_NO_EXPANSION may be packaged only after the 500-update cap",
    ):
      self.assertNotIn(forbidden, self.r1)


  def test_roll_assist_training_argv_parses_against_real_tyro_schema(self):
    import mjlab
    import tyro
    from mjlab.scripts.train import TrainConfig

    import hoppertrex_mjlab.tasks  # noqa: F401

    calls = re.findall(
      r"& \$Python -m hoppertrex_mjlab\.scripts\.rsl_rl\.train ([^\r\n]+)",
      self.r1,
    )
    self.assertEqual(len(calls), 2)
    substitutions = {
      "$Task": "HopperTrex-Hybrid-v2-StairRollAssist",
      "$TargetTotalUpdates": "200",
      "$RunName": "rollassist_test",
      "$ResumeRun": "source_run",
      "(Split-Path $Checkpoint -Leaf)": "model_99.pt",
    }
    for index, call in enumerate(calls):
      normalized = call
      for source, target in substitutions.items():
        normalized = normalized.replace(source, target)
      # PowerShell passes '[0]' as one literal token; shlex on POSIX syntax
      # accurately models this quote removal without invoking the wrapper.
      import shlex
      args = shlex.split(normalized, posix=True)
      self.assertEqual(args[:2], ["--task", "HopperTrex-Hybrid-v2-StairRollAssist"])
      cli_args = args[2:]
      parsed = tyro.cli(
        TrainConfig,
        args=cli_args,
        default=TrainConfig.from_task("HopperTrex-Hybrid-v2-StairRollAssist"),
        config=mjlab.TYRO_FLAGS,
      )
      self.assertEqual(parsed.gpu_ids, [0])
      self.assertEqual(parsed.agent.seed, 1)
      self.assertEqual(parsed.agent.save_interval, 25)
      self.assertEqual(parsed.agent.num_steps_per_env, 24)
      self.assertEqual(parsed.env.scene.num_envs, 256)
      self.assertEqual(parsed.agent.max_iterations, 100 if index == 0 else 200)
      self.assertIs(parsed.agent.resume, index == 1)



if __name__ == "__main__":
  unittest.main()
