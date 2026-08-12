from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_stair_dynamic_v3.ps1"
SIDECAR = SCRIPT.with_suffix(SCRIPT.suffix + ".sha256")


class StairDynamicWrapperTest(unittest.TestCase):
  def test_powershell_51_ast_is_clean(self) -> None:
    command = (
      "$e=$null;$t=$null;"
      "[void][System.Management.Automation.Language.Parser]::ParseFile("
      f"'{SCRIPT}',[ref]$t,[ref]$e);"
      "if($e.Count){$e|%{$_.ToString()};exit 1}"
    )
    completed = subprocess.run(
      ["powershell.exe", "-NoProfile", "-Command", command],
      check=False,
      capture_output=True,
      text=True,
    )
    self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

  def test_canonical_self_hash_and_no_placeholders(self) -> None:
    raw = SCRIPT.read_text(encoding="utf-8")
    canonical = raw.replace("\r\n", "\n").replace("\r", "\n").encode()
    expected = SIDECAR.read_text(encoding="ascii").strip()
    self.assertRegex(expected, r"^[0-9a-f]{64}$")
    self.assertEqual(hashlib.sha256(canonical).hexdigest(), expected)
    self.assertNotRegex(raw, r"(?i)<[^>]+>|TODO|PLACEHOLDER")
    self.assertNotIn(" -c ", raw)
    self.assertNotIn("'-c'", raw)

  def test_phases_protocol_and_direct_cli_flags_are_registered(self) -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    for phase in (
      "Validate",
      "QualifyTrigger",
      "Search",
      "Migrate",
      "ZeroEval",
      "Train100",
      "SelectK3",
      "Extend500",
      "Evaluate",
      "Package",
    ):
      self.assertIn(f"'{phase}'", text)
    for token in (
      "HopperTrex-Hybrid-v3-StairDynamic",
      "HOPPERTREX_DYNAMIC_STAIR_STAGE5_CHECKPOINT_PATH",
      "HOPPERTREX_DYNAMIC_STAIR_TRIGGER_QUALIFICATION_PATH",
      'Set-Item "Env:$triggerEnv" $q',
      "--reset-collapsed-active-std",
      "--completed-updates','0'",
      "--agent.max-iterations",
      "--agent.save-interval','25'",
      "--agent.num-steps-per-env','24'",
      "--budget-updates",
      "authorize-extension",
      "bundle-ablations",
      "single_seed_status='provisional'",
      "STOP_DYNAMIC_STAIR_UNQUALIFIED",
    ):
      self.assertIn(token, text)
    self.assertRegex(text, r"if\(\$Budget-eq100\).*@\(50,75,99\)")
    self.assertRegex(text, r"else\s*\{@\(450,475,499\)\}")

  def test_atomic_no_clobber_and_classified_exit_codes_are_present(self) -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    self.assertIn(".incomplete.", text)
    self.assertIn("$script:PhaseFinal", text)
    self.assertIn("Move-Item -LiteralPath $Dir -Destination $final", text)
    self.assertIn("Refusing to overwrite", text)
    self.assertIn("Phase exists", text)
    self.assertIn("{20}", text)
    self.assertIn("{30}", text)
    self.assertIn("else{40}", text)
    self.assertIn("CanonicalSelfHash", text)
    self.assertIn("SHA256SUMS.txt", text)
    self.assertIn("Compress-Archive", text)

  def test_run_failure_kinds_are_explicit_and_correct(self) -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    calls_only = text.replace("function Run(", "function RunDefinition(")
    self.assertNotRegex(calls_only, r"\bRun\s+(?!-FailureKind)")
    self.assertIn(
      "[ValidateSet('PROVENANCE','PROTOCOL','OPERATIONAL')]"
      "[string]$FailureKind",
      text,
    )
    for token in (
      "Run -FailureKind 'PROVENANCE' -Module $Preflight "
      "-ModuleArgs @('search-bindings'",
      "Run -FailureKind 'PROTOCOL' -Module $Preflight "
      "-ModuleArgs @('runtime-expectation'",
      "Run -FailureKind 'PROVENANCE' -Module $Evaluator "
      "-ModuleArgs @('checkpoint-envelope'",
      "Run -FailureKind 'PROVENANCE' -Module "
      "'hoppertrex_mjlab.scripts.rsl_rl.migrate_stage5_to_stair_dynamic'",
      "Run -FailureKind 'PROTOCOL' -Module $Evaluator "
      "-ModuleArgs @('make-request'",
      "Run -FailureKind 'PROTOCOL' -Module $Evaluator "
      "-ModuleArgs @('finalize'",
      "Run -FailureKind 'PROTOCOL' -Module $Evaluator "
      "-ModuleArgs @('select-k3'",
      "Run -FailureKind 'PROTOCOL' -Module $Evaluator "
      "-ModuleArgs @('authorize-extension'",
      "Run -FailureKind 'PROTOCOL' -Module $Evaluator "
      "-ModuleArgs @('bundle-ablations'",
      "Run -FailureKind 'PROVENANCE' -Module $Evaluator "
      "-ModuleArgs @('migration-checkpoint-envelope'",
      "Run -FailureKind 'OPERATIONAL' -Module $Live",
      "Run -FailureKind 'OPERATIONAL' -Module $Search",
      "Run -FailureKind 'OPERATIONAL' -Module $Train",
    ):
      self.assertIn(token, text)
    self.assertIn("Status $d 'STOP_DYNAMIC_STAIR_UNQUALIFIED'", text)
    self.assertNotRegex(text, r"Fail\s+'[^']+'\s+[^\n]*STOP_DYNAMIC")
    self.assertIn("exit 0", text)

  def test_package_is_self_describing_atomic_and_non_overwriting(self) -> None:
    head = subprocess.run(
      ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
      check=True,
      capture_output=True,
      text=True,
    ).stdout.strip()
    with tempfile.TemporaryDirectory() as directory:
      base = Path(directory)
      campaign = base / "campaign"
      campaign.mkdir()
      (campaign / "evidence.json").write_text(
        '{"qualified": true}\n', encoding="utf-8"
      )
      command = [
        "powershell.exe",
        "-NoProfile",
        "-File",
        str(SCRIPT),
        "-Phase",
        "Package",
        "-ExpectedGitSha",
        head,
        "-CampaignRoot",
        str(campaign),
        "-Python",
        sys.executable,
      ]
      completed = subprocess.run(
        command, check=False, capture_output=True, text=True
      )
      self.assertEqual(
        completed.returncode, 0, completed.stdout + completed.stderr
      )

      status_path = campaign / "10_package" / "status.json"
      sums_path = campaign / "10_package" / "SHA256SUMS.txt"
      archive = base / "campaign.zip"
      archive_sidecar = base / "campaign.zip.sha256"
      for output in (status_path, sums_path, archive, archive_sidecar):
        self.assertTrue(output.is_file(), str(output))

      status = json.loads(status_path.read_text(encoding="utf-8"))
      self.assertEqual(
        status["classification"], "STAIR_DYNAMIC_PACKAGE_COMPLETE"
      )
      self.assertEqual(status["archive"], archive.name)
      self.assertEqual(status["archive_sha256_sidecar"], archive_sidecar.name)
      checksums = sums_path.read_text(encoding="utf-8")
      status_hash = hashlib.sha256(status_path.read_bytes()).hexdigest()
      self.assertIn(f"{status_hash}  10_package/status.json", checksums)
      self.assertNotIn("10_package/SHA256SUMS.txt", checksums)
      with zipfile.ZipFile(archive) as package:
        names = set(package.namelist())
      self.assertIn("10_package/status.json", names)
      self.assertIn("10_package/SHA256SUMS.txt", names)

      archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
      self.assertEqual(
        archive_sidecar.read_text(encoding="utf-8").split()[0],
        archive_hash,
      )
      status_before = status_path.read_bytes()
      archive_before = archive.read_bytes()
      retry = subprocess.run(
        command, check=False, capture_output=True, text=True
      )
      self.assertEqual(retry.returncode, 40)
      self.assertIn("STAIR_DYNAMIC_OPERATIONAL", retry.stderr)
      self.assertEqual(status_path.read_bytes(), status_before)
      self.assertEqual(archive.read_bytes(), archive_before)

  def test_provenance_failure_uses_classified_exit_20(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      completed = subprocess.run(
        [
          "powershell.exe",
          "-NoProfile",
          "-File",
          str(SCRIPT),
          "-ExpectedGitSha",
          "0" * 40,
          "-CampaignRoot",
          directory,
        ],
        check=False,
        capture_output=True,
        text=True,
      )
    self.assertEqual(completed.returncode, 20)
    self.assertIn("STAIR_DYNAMIC_PROVENANCE", completed.stderr)

  def test_protocol_failure_uses_classified_exit_30(self) -> None:
    head = subprocess.run(
      ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
      check=True,
      capture_output=True,
      text=True,
    ).stdout.strip()
    with tempfile.TemporaryDirectory() as directory:
      completed = subprocess.run(
        [
          "powershell.exe",
          "-NoProfile",
          "-File",
          str(SCRIPT),
          "-ExpectedGitSha",
          head,
          "-CampaignRoot",
          directory,
          "-Python",
          sys.executable,
        ],
        check=False,
        capture_output=True,
        text=True,
      )
    self.assertEqual(completed.returncode, 30)
    self.assertIn("STAIR_DYNAMIC_PROTOCOL", completed.stderr)

  def test_missing_stage5_input_is_provenance_exit_20(self) -> None:
    head = subprocess.run(
      ["git", "rev-parse", "HEAD"],
      cwd=ROOT,
      check=True,
      capture_output=True,
      text=True,
    ).stdout.strip()
    with tempfile.TemporaryDirectory() as directory:
      missing = str(Path(directory) / "missing.pt")
      completed = subprocess.run(
        [
          "powershell.exe",
          "-NoProfile",
          "-File",
          str(SCRIPT),
          "-ExpectedGitSha",
          head,
          "-CampaignRoot",
          str(Path(directory) / "campaign"),
          "-Stage5Checkpoint",
          missing,
          "-Stage5Gate",
          missing + ".json",
        ],
        check=False,
        capture_output=True,
        text=True,
      )
    self.assertEqual(completed.returncode, 20)
    self.assertIn("STAIR_DYNAMIC_PROVENANCE", completed.stderr)
    self.assertIn("Missing Stage5 checkpoint", completed.stderr)

  def test_simple_wrapper_does_not_copy_the_old_monolith(self) -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    self.assertLess(len(text.splitlines()), 180)
    self.assertLess(len(text.encode("utf-8")), 24_000)


if __name__ == "__main__":
  unittest.main()
