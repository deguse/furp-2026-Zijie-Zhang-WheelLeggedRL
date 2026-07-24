import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "run_hybrid_p2_stall_diagnostic.ps1"


class HybridP2StallWrapperTest(unittest.TestCase):
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
      '"fetch", "--quiet", "origin", $RequiredBranch',
      '$fullSha -ne $remoteHead',
    ):
      self.assertIn(fragment, self.source)

  def test_pins_diagnostic_protocol_and_atomic_output(self):
    for fragment in (
      'probe_hybrid_stall_diagnostic',
      '"--device", "cuda:0", "--output", $WorkingOutput',
      '"stall_diagnostic.json"',
      '".incomplete." + $runToken',
      'Move-Item -LiteralPath $WorkingDirectory -Destination $OutputDirectory',
      '"CLASSICAL_CARD_CANDIDATE_FOUND"',
      '"WHEEL_SPIN_FRICTION_LIMITED"',
      '"TORQUE_SATURATED_STALL"',
      '"DRIVE_TARGET_COLLAPSED"',
      '"MIXED_STALL_MECHANISM"',
      '"INVALID_FLAT_CONTROL_STOP"',
      '$ExpectedCellCount = 16',
      '$ExpectedTrialCount = 256',
      '"$hash  stall_diagnostic.json"',
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
      '$result.probe -ne "hybrid_p2_stall_mechanism_diagnostic"',
      "$result.evidence_eligible -ne $true",
      "$result.promotion_eligible -ne $false",
      "$result.training_eligible -ne $false",
      "$null -ne $result.checkpoint",
      "$null -ne $result.yaw_calibration_hash",
      "$result.git_sha -ne $fullSha",
      "$result.mjlab_git_sha -ne $MjlabCommit",
      "$result.controller_gain_hash -ne $ControllerGainHash",
      "$result.posture_artifact_hash -ne $PostureArtifactHash",
      "$AllowedClassifications -notcontains $result.classification",
      '$result.runtime.cuda_available -ne $true',
      '$result.protocol.wheel_model.forward_channel -ne "0.5 * (right - left)"',
      '@($result.trials).Count -ne $ExpectedTrialCount',
    ):
      self.assertIn(fragment, self.source)

  def test_no_authored_numeric_reset_bounds(self):
    # The 2026-07-24 root-height incident: authored float64 tolerances in
    # the wrapper failed against float32 read-back on every trial. The
    # diagnostic wrapper must not re-author numeric reset-state bounds;
    # protocol identity tolerances for hashes/scales/tables are allowed.
    for forbidden in (
      "1.0e-9",
      "root_reset.root_height_m",
      "root_reset.root_linear_velocity_mps",
      "root_reset.root_angular_velocity_radps",
    ):
      self.assertNotIn(forbidden, self.source)


if __name__ == "__main__":
  unittest.main()
