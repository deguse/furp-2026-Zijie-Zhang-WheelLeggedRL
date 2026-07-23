import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from hoppertrex_mjlab.hybrid.calibration import parse_calibration_artifact


ROOT = Path(__file__).parents[1]
ARTIFACT_ROOT = (
  ROOT / "docs" / "experiments" / "artifacts" / "hybrid_runtime_seed1"
)
CONTROLLER = ARTIFACT_ROOT / "controller_seed1.json"
CALIBRATION = ARTIFACT_ROOT / "velocity_calibration_seed1.json"
MANIFEST = ARTIFACT_ROOT / "manifest.json"
SCRIPT = ROOT / "scripts" / "bootstrap_hybrid_zero_residual_standing.ps1"
EVALUATOR = (
  ROOT / "src" / "hoppertrex_mjlab" / "scripts" / "rsl_rl"
  / "evaluate_hybrid_gate.py"
)

CONTROLLER_FILE_SHA256 = (
  "663ab77f77521581cde77ea2bd8c72c7f395f33b05b62348ef6d82a752aad7fc"
)
CALIBRATION_FILE_SHA256 = (
  "ef002d0d622725509b47c8ff40d8af658fd42f705bdeac67ac35bae4458f889d"
)
CONTROLLER_GAIN_HASH = (
  "8fee25a0339dd1e99127cbed912941dc3ad8ef2030ce49a0d310d1563cb87d98"
)
CALIBRATION_HASH = (
  "f62648b57bd17a3503bcbdbf58f349f91fcd8de8ef0cf04551c200401233ed01"
)


class HybridRuntimeArtifactTest(unittest.TestCase):
  def test_frozen_file_hashes_and_newlines_are_preserved(self):
    controller_bytes = CONTROLLER.read_bytes()
    calibration_bytes = CALIBRATION.read_bytes()
    line_feed = bytes((10,))
    carriage_return_line_feed = bytes((13, 10))

    self.assertEqual(
      hashlib.sha256(controller_bytes).hexdigest(), CONTROLLER_FILE_SHA256
    )
    self.assertEqual(
      hashlib.sha256(calibration_bytes).hexdigest(), CALIBRATION_FILE_SHA256
    )
    self.assertEqual(
      controller_bytes.count(line_feed),
      controller_bytes.count(carriage_return_line_feed),
    )
    self.assertEqual(
      calibration_bytes.count(line_feed),
      calibration_bytes.count(carriage_return_line_feed),
    )

  def test_controller_and_calibration_provenance_are_bound(self):
    controller = json.loads(CONTROLLER.read_text(encoding="utf-8"))
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))

    self.assertEqual(controller["controller_type"], "lqr")
    self.assertEqual(controller["controllability_rank"], 4)
    self.assertEqual(controller["fallback_reasons"], [])
    self.assertEqual(controller["gain_hash"], CONTROLLER_GAIN_HASH)
    parsed = parse_calibration_artifact(
      calibration,
      controller_gain_hash=CONTROLLER_GAIN_HASH,
    )
    self.assertEqual(parsed.calibration_hash, CALIBRATION_HASH)

  def test_manifest_matches_frozen_artifacts(self):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    self.assertEqual(
      manifest["controller"]["file_sha256"], CONTROLLER_FILE_SHA256
    )
    self.assertEqual(
      manifest["velocity_calibration"]["file_sha256"],
      CALIBRATION_FILE_SHA256,
    )
    self.assertEqual(
      manifest["velocity_calibration"]["controller_gain_hash"],
      CONTROLLER_GAIN_HASH,
    )
    self.assertEqual(
      manifest["velocity_calibration"]["calibration_hash"], CALIBRATION_HASH
    )

  def test_gitattributes_disable_text_conversion(self):
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    self.assertIn(
      "hybrid_runtime_seed1/controller_seed1.json -text -diff", attributes
    )
    self.assertIn(
      "hybrid_runtime_seed1/velocity_calibration_seed1.json -text -diff",
      attributes,
    )


class HybridZeroResidualBootstrapScriptTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.source = SCRIPT.read_text(encoding="utf-8")
    cls.source_bytes = SCRIPT.read_bytes()
    cls.evaluator_source = EVALUATOR.read_text(encoding="utf-8")

  def test_pins_framework_environment_and_preflight(self):
    self.assertIn("43e0f3ea9c92ddbb4de9f3bb1ac772d604e3ebf6", self.source)
    self.assertIn("codex/hybrid-v2-runtime-r1", self.source)
    self.assertIn("uv sync --frozen --python 3.11", self.source)
    self.assertIn('"-File", $Preflight', self.source)
    self.assertIn('"-Phase", "Preflight", "-Python", $Python', self.source)
    self.assertIn("$PowerShellExecutable", self.source)
    self.assertNotIn("& $Preflight -Phase Preflight", self.source)
    self.assertIn('("git", "uv", "nvidia-smi")', self.source)
    self.assertIn('"fetch", "--quiet", "origin", $RequiredBranch', self.source)
    self.assertIn('$fullSha -ne $remoteHead', self.source)

  def test_runtime_paths_survive_source_generation(self):
    line_feed = bytes((10,))
    carriage_return_line_feed = bytes((13, 10))
    without_line_endings = self.source_bytes.replace(carriage_return_line_feed, b"")
    without_line_endings = without_line_endings.replace(line_feed, b"")
    self.assertNotIn(bytes((13,)), without_line_endings)
    for fragment in (
      '.venv/Scripts/python.exe',
      'docs/experiments/artifacts/hybrid_runtime_seed1',
      'docs/experiments/artifacts/hybrid_p1_1/posture_map_seed1_floor028_fullhash.json',
      'scripts/run_hybrid_v2_machine_room.ps1',
      "src/hoppertrex_mjlab",
      'experiments/hybrid_zero_residual_standing_',
    ):
      self.assertIn(fragment, self.source)
    for corrupted in (
      ".venvScriptspython.exe",
      "docsexperimentsartifacts",
      "srchoppertrex_mjlab",
      "experimentshybrid_zero_residual_standing_",
    ):
      self.assertNotIn(corrupted, self.source)

  def test_pins_the_two_repeat_standing_protocol(self):
    for fragment in (
      '"--stage", "5"',
      '"--profile", "screen"',
      '"--seed", "1"',
      '"--device", "cuda:0"',
      '"--num-envs", "16"',
      '"--steps", "1000"',
      '"--warmup-steps", "300"',
      '"--window-steps", "300"',
      '"--zero-residual-standing-diagnostic"',
      '"--diagnostic-repeats", "2"',
    ):
      self.assertIn(fragment, self.source)

  def test_isolates_yaw_and_records_non_promotable_scope(self):
    self.assertIn(
      "Remove-Item Env:HOPPERTREX_HYBRID_YAW_CALIBRATION_PATH", self.source
    )
    self.assertIn("yaw calibration hash must be null", self.source.lower())
    self.assertIn(
      'evidence_scope = "standing_channel_counterfactual_only"', self.source
    )
    self.assertIn("nonzero_yaw_gate_eligible = $false", self.source)
    self.assertIn("promotion_eligible = $false", self.source)

  def test_refuses_overwrite_and_reports_all_classifications(self):
    self.assertIn(
      "Refusing to overwrite existing diagnostic directory", self.source
    )
    self.assertIn(
      "REAL_POLICY_REGRESSION_NEXT_EVALUATION_TIME_LEG_ABLATION", self.source
    )
    self.assertIn(
      "PROTOCOL_OR_THRESHOLD_DRIFT_STOP_NO_RETRAINING", self.source
    )
    self.assertIn("MIXED_REPEAT_STOP_FOR_VARIANCE_ANALYSIS", self.source)

  def test_publishes_atomically_without_blocking_retry_after_runtime_failure(self):
    self.assertIn('".incomplete." + $runToken', self.source)
    self.assertIn(
      "Incomplete diagnostic retained for inspection: $WorkingDirectory",
      self.source,
    )
    self.assertIn(
      "Move-Item -LiteralPath $WorkingDirectory -Destination $OutputDirectory",
      self.source,
    )
    self.assertIn(
      '[PASS] Zero-residual standing diagnostic complete.', self.source
    )

  def test_validates_full_result_provenance_and_action_scales(self):
    for fragment in (
      "$result.controller_gain_hash -ne $ControllerGainHash",
      "$result.calibration_hash -ne $CalibrationHash",
      "$result.posture_map_hash -ne $PostureMapHash",
      "$result.posture_artifact_hash -ne $PostureArtifactHash",
      "$result.station_calibration_hash -ne $StationCalibrationHash",
      "$ExpectedActionScales = @(0.5, 0.3, 0.035, 0.035, 0.035, 0.035)",
      "wrong action scale at index",
      "lacks complete GPU runtime provenance",
      '$protocol["runtime"] = $result.runtime',
    ):
      self.assertIn(fragment, self.source)

  def test_records_gpu_runtime_metadata_in_result_and_protocol_note(self):
    for fragment in (
      '"runtime": _runtime_metadata(args.device)',
      '"gpu_name": gpu_name',
      '"driver_version": driver_version',
      '"torch_version": str(torch.__version__)',
      '"cuda_version": torch.version.cuda',
    ):
      self.assertIn(fragment, self.evaluator_source)
    self.assertIn('$protocol["runtime"] = $result.runtime', self.source)

  def test_never_invokes_training_migration_or_checkpoint_workflows(self):
    for forbidden in (
      "hoppertrex_mjlab.scripts.rsl_rl.train",
      "migrate_hybrid_stage",
      "--checkpoint-file",
      "run_hybrid_leg_authority_seed1.ps1",
    ):
      self.assertNotIn(forbidden, self.source)

  @unittest.skipUnless(shutil.which("powershell"), "PowerShell is unavailable")
  def test_powershell_syntax_is_valid(self):
    escaped = str(SCRIPT).replace("'", "''")
    command = (
      "$tokens=$null; $errors=$null; "
      "[System.Management.Automation.Language.Parser]::ParseFile("
      f"'{escaped}', [ref]$tokens, [ref]$errors) | Out-Null; "
      "if ($errors.Count -ne 0) { $errors | Out-String | Write-Error; exit 1 }"
    )
    completed = subprocess.run(
      ["powershell", "-NoProfile", "-Command", command],
      check=False,
      capture_output=True,
      text=True,
    )
    self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
  unittest.main()
