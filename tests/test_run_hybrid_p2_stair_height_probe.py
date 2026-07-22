from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "run_hybrid_p2_stair_height_probe.ps1"


class HybridP2StairWrapperTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.source = SCRIPT.read_text(encoding="utf-8")

  def test_pins_branch_mjlab_and_artifact_provenance(self):
    for fragment in (
      '"codex/p2-stair-probe"',
      '"4411057ecdcb2fd89314fcd4350dc9d66c493c54"',
      '"codex/hybrid-v2-runtime-r1"',
      '"43e0f3ea9c92ddbb4de9f3bb1ac772d604e3ebf6"',
      '"sync", "--frozen", "--python", "3.11"',
      '"-Phase", "Preflight", "-Python", $Python',
      'Remove-Item Env:HOPPERTREX_HYBRID_YAW_CALIBRATION_PATH',
    ):
      self.assertIn(fragment, self.source)

  def test_pins_stair_protocol_and_atomic_output(self):
    for fragment in (
      '"--device", "cuda:0", "--output", $WorkingOutput',
      '"stair_height_probe.json"',
      '".incomplete." + $runToken',
      'Move-Item -LiteralPath $WorkingDirectory -Destination $OutputDirectory',
      '"CLASSICAL_DEATH_HEIGHT_BRACKETED", "EXTEND_SWEEP_BEFORE_P3", "STOP_FOR_VARIANCE_ANALYSIS", "INVALID_FLAT_CONTROL_STOP"',
      '"$hash  stair_height_probe.json"',
    ):
      self.assertIn(fragment, self.source)

  def test_rejects_training_checkpoint_and_migration_entrypoints(self):
    for forbidden in (
      "hoppertrex_mjlab.scripts.rsl_rl.train",
      "migrate_hybrid_stage",
      "--checkpoint-file",
      "run_hybrid_leg_authority_seed1.ps1",
    ):
      self.assertNotIn(forbidden, self.source)

  def test_validates_non_promotable_zero_yaw_result(self):
    for fragment in (
      "$result.evidence_eligible -ne $true",
      "$result.promotion_eligible -ne $false",
      "$result.training_eligible -ne $false",
      "$null -ne $result.checkpoint",
      "$null -ne $result.yaw_calibration_hash",
      "$result.protocol.step_width_m -ne 0.30",
      "$result.protocol.envs_per_height -ne 16",
      "$result.protocol.repeats -ne 3",
      "$result.controller_gain_hash -ne $ControllerGainHash",
      "$result.posture_artifact_hash -ne $PostureArtifactHash",
      "$result.protocol.root_reset.start_offset_outside_m -ne 0.25",
    ):
      self.assertIn(fragment, self.source)

  def test_required_paths_are_not_corrupted(self):
    self.assertIn(".venv/Scripts/python.exe", self.source)
    self.assertIn("docs/experiments/artifacts/hybrid_p1_1", self.source)
    self.assertNotIn("docsexperimentsartifacts", self.source)


if __name__ == "__main__":
  unittest.main()
