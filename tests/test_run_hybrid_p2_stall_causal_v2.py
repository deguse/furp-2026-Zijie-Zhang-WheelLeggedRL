import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "run_hybrid_p2_stall_causal_v2.ps1"


class HybridP2StallCausalWrapperTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.source = SCRIPT.read_text(encoding="utf-8")

  def test_pins_branch_runtime_and_remote_head(self):
    for fragment in (
      '"codex/p2-stair-probe"',
      '"fc80fd5d58687ebf6f00908d7ce6fc5c1e61038c"',
      '"codex/hybrid-v2-runtime-r1"',
      '"43e0f3ea9c92ddbb4de9f3bb1ac772d604e3ebf6"',
      '"sync", "--frozen", "--python", "3.11"',
      '"fetch", "--quiet", "origin", $RequiredBranch',
      '$fullSha -ne $remoteHead',
      '"-Phase", "Preflight", "-Python", $Python',
    ):
      self.assertIn(fragment, self.source)

  def test_pins_capture_protocol_and_atomic_output(self):
    for fragment in (
      'probe_hybrid_stall_causal_v2',
      '"--device", "cuda:0", "--output", $WorkingOutput',
      '"stall_causal_v2.json"',
      '"protocol_note.json"',
      '"SHA256SUMS.txt"',
      '".incomplete." + $runToken',
      'Move-Item -LiteralPath $WorkingDirectory -Destination $OutputDirectory',
      '$ExpectedCellCount = 2',
      '$ExpectedTrialCount = 64',
      '$ExpectedPairCount = 32',
      '$ExpectedAlignedSamples = 101',
      '"ANALYSIS_READY"',
      '"INVALID_CAPTURE"',
    ):
      self.assertIn(fragment, self.source)

  def test_validates_zero_action_no_checkpoint_and_no_cause_label(self):
    for fragment in (
      'Remove-Item Env:HOPPERTREX_HYBRID_YAW_CALIBRATION_PATH',
      '$null -ne $result.checkpoint',
      '$null -ne $result.yaw_calibration_hash',
      '$null -ne $result.single_cause_label',
      '$result.runtime.cuda_available -ne $true',
      '$result.runtime.gpu_name',
      '$result.protocol.paired_flat_stair_by_terrain_slot -ne $true',
      '[int]$result.protocol.pre_impact_steps -ne 25',
      '[int]$result.protocol.post_impact_steps -ne 75',
      '$capture.aligned_series.relative_steps',
      '$capture.impact_contact_slots',
    ):
      self.assertIn(fragment, self.source)

  def test_never_runs_training_migration_or_checkpoint_entrypoints(self):
    for forbidden in (
      'hoppertrex_mjlab.scripts.rsl_rl.train',
      'migrate_hybrid_stage',
      '--checkpoint-file',
      'run_hybrid_leg_authority_seed1.ps1',
      'probe_hybrid_yaw_transfer',
    ):
      self.assertNotIn(forbidden, self.source)

  def test_final_messages_preserve_stop_boundary(self):
    for fragment in (
      'P2_STALL_CAUSAL_CAPTURE_ANALYSIS_READY_STOP_NO_TRAINING',
      'INVALID_CAPTURE_STOP_RERUN_NO_TRAINING',
      'Do not force a single physical cause from this capture.',
      'p3_eligible = $false',
    ):
      self.assertIn(fragment, self.source)


if __name__ == "__main__":
  unittest.main()
