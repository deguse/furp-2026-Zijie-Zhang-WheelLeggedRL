from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / 'scripts' / 'run_hybrid_stage5_seed1.ps1'


class HybridStage5ScriptTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.source = SCRIPT.read_text(encoding='utf-8')

  def test_exports_all_five_classical_artifacts_before_migration(self):
    for env_var in (
      'HOPPERTREX_HYBRID_CONTROLLER_PATH',
      'HOPPERTREX_HYBRID_CALIBRATION_PATH',
      'HOPPERTREX_HYBRID_YAW_CALIBRATION_PATH',
      'HOPPERTREX_HYBRID_POSTURE_MAP_PATH',
      'HOPPERTREX_HYBRID_STATION_CALIBRATION_PATH',
    ):
      self.assertIn(env_var, self.source)
      self.assertLess(
        self.source.index(env_var),
        self.source.index('migrate_hybrid_stage'),
      )

  def test_triple_migration_chain_carries_the_no_harm_gate(self):
    self.assertIn('--source-gate-json $sourceGate', self.source)
    self.assertIn('--source-stage 2', self.source)
    self.assertIn('--target-stage 3', self.source)
    self.assertIn('--source-stage 3', self.source)
    self.assertIn('--target-stage 4', self.source)
    self.assertIn('--source-stage 4', self.source)
    self.assertIn('--target-stage 5', self.source)
    self.assertEqual(self.source.count('--posture-map $postureMap'), 3)
    self.assertEqual(
      self.source.count('--station-calibration $stationCalibration'), 3
    )

  def test_screens_last_three_checkpoints_newest_first(self):
    self.assertIn('Select-Object -Last 3', self.source)
    self.assertIn('[array]::Reverse($candidates)', self.source)
    self.assertIn(
      'No Stage5 checkpoint passed the Stage1 retention screen (K=3)',
      self.source,
    )

  def test_formals_run_after_a_manual_viser_verdict(self):
    self.assertIn('Read-Host', self.source)
    self.assertIn('-cne "PASS"', self.source)
    self.assertLess(
      self.source.index('Read-Host'),
      self.source.index('--profile formal'),
    )

  def test_ablated_attribution_run_never_blocks(self):
    self.assertIn('--ablate-leg-residuals', self.source)
    self.assertIn('$ablatedFormalExit = $LASTEXITCODE', self.source)
    ablated_index = self.source.index('--ablate-leg-residuals')
    self.assertNotIn(
      'throw', self.source[ablated_index:self.source.index('STOP FOR ANALYSIS')]
    )

  def test_seed_is_parametrized_everywhere_except_the_frozen_carrier(self):
    self.assertIn('[int] $Seed = 1,', self.source)
    self.assertIn('--agent.seed $Seed', self.source)
    self.assertEqual(self.source.count('--seed $Seed'), 5)
    self.assertNotIn('--seed 1', self.source)
    self.assertNotIn('--agent.seed 1', self.source)
    # Run/handoff names and every gate JSON carry the seed; the only literal
    # seed1 paths left are the frozen seed-1 Stage2 carrier defaults.
    self.assertIn('_seed$Seed"', self.source)
    self.assertIn('seed${Seed}_stage5_robust_formal.json', self.source)
    self.assertIn(
      'seed${Seed}_stage5_robust_formal_legs_ablated.json', self.source
    )
    self.assertIn('seed${Seed}_stage1_retention_screen_', self.source)


if __name__ == '__main__':
  unittest.main()
