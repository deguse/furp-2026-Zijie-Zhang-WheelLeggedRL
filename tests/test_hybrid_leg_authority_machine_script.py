from pathlib import Path
import unittest


REPOSITORY_PATH = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_PATH / "scripts" / "run_hybrid_leg_authority_seed1.ps1"


class HybridLegAuthorityMachineScriptTest(unittest.TestCase):
  def test_pythonpath_contains_package_and_legacy_top_level_modules(self):
    script = SCRIPT_PATH.read_text(encoding="utf-8")
    self.assertIn(
      '$packagePath = (Resolve-Path -LiteralPath "src\\hoppertrex_mjlab").Path',
      script,
    )
    self.assertIn('$env:PYTHONPATH = "$sourcePath;$packagePath"', script)

  def test_empty_failed_probe_directory_is_resumable(self):
    script = SCRIPT_PATH.read_text(encoding="utf-8")
    self.assertIn("$existingOutputs.Count -ne 0", script)
    self.assertIn("output already exists and is non-empty", script)

  def test_script_preflights_all_registered_stages_with_override_active(self):
    script = SCRIPT_PATH.read_text(encoding="utf-8")
    self.assertIn("Stage0-5 registration and authority preflight", script)
    self.assertIn("--preflight-only", script)
    self.assertNotIn("$preflightCode", script)


if __name__ == "__main__":
  unittest.main()
