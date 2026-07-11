from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / 'scripts' / 'run_hybrid_v2_machine_room.ps1'


class HybridMachineRoomScriptTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.source = SCRIPT.read_text(encoding='utf-8')

  def test_exposes_calibration_probe_and_formal_phases(self):
    self.assertIn("'Calibrate'", self.source)
    self.assertIn("'Stage0Probe'", self.source)
    self.assertIn("[int[]]$Seeds", self.source)

  def test_defaults_formal_stage0_to_three_seeds(self):
    self.assertIn("@(1, 2, 3)", self.source)
    self.assertIn("Stage0 requires a passing Stage0Probe", self.source)

  def test_streams_and_persists_subprocess_output(self):
    self.assertIn('Tee-Object -FilePath $LogPath', self.source)

  def test_reuses_identification_without_switching_sha_artifact_root(self):
    self.assertNotIn('$artifactRoot = $reusableIdentification.Directory.FullName', self.source)
    self.assertIn('Copy-Item -LiteralPath $reusableIdentification.FullName', self.source)


if __name__ == '__main__':
  unittest.main()
