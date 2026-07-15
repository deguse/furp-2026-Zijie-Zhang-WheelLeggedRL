from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / 'scripts' / 'run_hybrid_stage2_seed1.ps1'


class HybridStage2ScriptTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.source = SCRIPT.read_text(encoding='utf-8')

  def test_fits_and_exports_the_yaw_calibration_before_migration(self):
    self.assertIn('probe_hybrid_yaw_transfer', self.source)
    self.assertIn('--fit-output', self.source)
    self.assertIn('HOPPERTREX_HYBRID_YAW_CALIBRATION_PATH', self.source)
    self.assertLess(
      self.source.index('HOPPERTREX_HYBRID_YAW_CALIBRATION_PATH'),
      self.source.index('migrate_hybrid_stage'),
    )

  def test_migration_records_the_yaw_calibration(self):
    self.assertIn('--yaw-calibration $yawCalibration', self.source)

  def test_screens_last_three_checkpoints_newest_first(self):
    self.assertIn('Select-Object -Last 3', self.source)
    self.assertIn('[array]::Reverse($candidates)', self.source)
    self.assertIn(
      'No Stage2 checkpoint passed the Stage1 retention screen (K=3)',
      self.source,
    )

  def test_planar_screen_binds_the_promoted_retention_json(self):
    self.assertIn('--stage1-retention-file $retentionScreen', self.source)


if __name__ == '__main__':
  unittest.main()
