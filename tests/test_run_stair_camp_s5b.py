"""Static and AST contracts for the formal StairCamp S5B wrapper."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_stair_camp_s5b.ps1"
SIDECAR = SCRIPT.with_suffix(SCRIPT.suffix + ".sha256")
ARTIFACTS = ROOT / "docs" / "experiments" / "artifacts"


def canonical_script_hash(path: Path) -> str:
  text = path.read_text(encoding="utf-8-sig")
  normalized = text.replace("\r\n", "\n").replace("\r", "\n")
  return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def function_body(text: str, name: str, next_name: str) -> str:
  start = text.index(f"function {name} {{")
  end = text.index(f"function {next_name} {{", start)
  return text[start:end]


class RunStairCampS5BWrapperTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    cls.raw = SCRIPT.read_bytes()
    cls.text = cls.raw.decode("utf-8")

  def test_wrapper_is_canonical_lf_utf8_and_self_hash_matches(self) -> None:
    self.assertFalse(self.raw.startswith(b"\xef\xbb\xbf"))
    self.assertNotIn(b"\r", self.raw)
    self.assertTrue(self.raw.endswith(b"\n"))
    self.assertTrue(SIDECAR.is_file())
    sidecar = SIDECAR.read_bytes()
    self.assertRegex(sidecar.decode("ascii").strip(), r"^[0-9a-f]{64}$")
    self.assertEqual(sidecar, sidecar.strip() + b"\n")
    self.assertEqual(sidecar.decode("ascii").strip(), canonical_script_hash(SCRIPT))

  def test_powershell_ast_parses_when_parser_is_available(self) -> None:
    executable = (
      shutil.which("powershell.exe")
      or shutil.which("powershell")
      or shutil.which("pwsh")
    )
    if executable is None:
      self.skipTest("PowerShell parser is not installed")
    quoted = str(SCRIPT).replace("'", "''")
    command = (
      "$tokens=$null; $errors=$null; "
      "[System.Management.Automation.Language.Parser]::ParseFile("
      f"'{quoted}',[ref]$tokens,[ref]$errors) | Out-Null; "
      "if($errors.Count){$errors | ForEach-Object { $_.Message }; exit 1}"
    )
    completed = subprocess.run(
      [executable, "-NoProfile", "-Command", command],
      capture_output=True,
      text=True,
      check=False,
    )
    self.assertEqual(
      completed.returncode,
      0,
      msg=f"stdout={completed.stdout}\nstderr={completed.stderr}",
    )

  def test_embedded_python_helpers_compile(self) -> None:
    payloads = re.findall(
      r"\$(?:extractor|progressExtractor|collector) = @'\n(.*?)\n'@",
      self.text,
      flags=re.DOTALL,
    )
    self.assertEqual(len(payloads), 3)
    for index, payload in enumerate(payloads):
      with self.subTest(index=index):
        compile(payload, f"embedded_wrapper_helper_{index}.py", "exec")

  def test_phase_surface_and_mandatory_identity_are_locked(self) -> None:
    for phase in (
      "'Validate'",
      "'Fresh1000'",
      "'Extend3000'",
      "'SelectK3'",
      "'Evaluate'",
      "'Adjudicate'",
      "'Package'",
    ):
      self.assertIn(phase, self.text)
    self.assertIn("[ValidatePattern('^[0-9a-fA-F]{40}$')]", self.text)
    self.assertIn("[string]$ExpectedGitSha", self.text)
    self.assertIn("[string]$CampaignRoot", self.text)
    self.assertIn("[string]$ClassicalRowsPath", self.text)
    self.assertNotIn("[string]$FlatFalsePositivePath", self.text)
    self.assertNotIn("[string]$Stage5KickFalsePositivePath", self.text)
    self.assertIn("[ValidateSet(1, 2, 3)]", self.text)
    self.assertIn("[ValidateSet(1000, 3000)]", self.text)
    self.assertNotIn("'All'", self.text)

  def test_repository_mjlab_and_runtime_provenance_are_fail_closed(self) -> None:
    required = (
      "codex/p2-classical-upper-bound",
      "43e0f3ea9c92ddbb4de9f3bb1ac772d604e3ebf6",
      "HopperTrex-Hybrid-v2-StairCamp",
      "1d4b18db32e48b3ae8803e385a032203bdddc7f8198da9679f519bc8947190cb",
      "git status --porcelain",
      "git -C $script:MjLabRoot status --porcelain",
      "$mjlabStatusExitCode = $LASTEXITCODE",
      "$headExitCode = $LASTEXITCODE",
      "'fetch', '--quiet', 'origin', $RequiredBranch",
      "Local HEAD does not equal mandatory -ExpectedGitSha",
      "Local HEAD does not equal origin/codex/p2-classical-upper-bound",
      'mjlab = { path = "../mjlab-main", editable = true }',
      "Python imports MjLab from a checkout other than the pinned editable source",
      "torch.cuda.is_available()",
      "nvidia-smi --query-gpu",
      "$env:PYTHONPATH",
    )
    for value in required:
      self.assertIn(value, self.text)
    self.assertLess(
      self.text.index("Wrapper canonical self-hash mismatch"),
      self.text.index("git branch --show-current"),
    )

  def test_frozen_c0_probe_and_classical_rows_ship_in_repo_unnormalized(
    self,
  ) -> None:
    """The camp's classical evidence must survive a fresh Windows checkout.

    `stair_height_probe.json` is stored with CRLF line endings and the wrapper
    pins its RAW byte hash, so Git must not normalize it. Without the
    `-text` gitattribute a clone would rewrite the bytes and the frozen
    `e85ee64f...` pin would never match again. Shipping both files in the
    repository also removes a single-point-of-failure: the classical arm's
    only evidence previously lived on one developer machine, and this
    project has lost its machine-room storage twice.
    """

    probe = (
      ARTIFACTS
      / "hybrid_p2_stair_height_9edb8b7_seed1"
      / "stair_height_probe.json"
    )
    rows = probe.with_name("classical_rows.json")
    self.assertTrue(probe.is_file())
    self.assertTrue(rows.is_file())

    probe_bytes = probe.read_bytes()
    self.assertIn(b"\r\n", probe_bytes)
    self.assertEqual(
      hashlib.sha256(probe_bytes).hexdigest(),
      "e85ee64ff60337fc60c894558af193c5a82f00811772d22fcb00fc5d10830da5",
    )
    self.assertIn(
      "e85ee64ff60337fc60c894558af193c5a82f00811772d22fcb00fc5d10830da5",
      self.text,
    )

    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    self.assertIn(
      "docs/experiments/artifacts/hybrid_p2_stair_height_9edb8b7_seed1/"
      "* -text -diff",
      attributes,
    )

    payload = json.loads(rows.read_text(encoding="utf-8"))
    self.assertEqual(list(payload), ["rows"])
    self.assertEqual(
      [row["height_m"] for row in payload["rows"]],
      [0.01, 0.02, 0.03, 0.05, 0.07, 0.10],
    )
    centers = {
      round(float(cell["stair_height_m"]), 10): cell
      for cell in json.loads(probe.read_text(encoding="utf-8"))["cells"]
      if cell["posture_card"] == "envelope_center"
    }
    for row in payload["rows"]:
      cell = centers[round(float(row["height_m"]), 10)]
      self.assertEqual(row["success_rate"], cell["success_rate"])
      self.assertEqual(row["terminations"], cell["terminated_trials"])
      self.assertEqual(row["non_wheel_contacts"], cell["non_wheel_contact_trials"])
      self.assertEqual(row["trials"], cell["trials"])

  def test_five_repo_artifacts_have_current_bytes_and_are_pinned(self) -> None:
    expected = {
      ARTIFACTS
      / "c1_schedule_candidate24_1f54968_seed1"
      / "c1_schedule.json": (
        "9b21125e7cc48be3ea61e12a67171a855892ad3ced1f54b3176ed979e76224ec"
      ),
      ARTIFACTS
      / "hybrid_runtime_seed1"
      / "velocity_calibration_seed1.json": (
        "ef002d0d622725509b47c8ff40d8af658fd42f705bdeac67ac35bae4458f889d"
      ),
      ARTIFACTS
      / "yaw_gpu_3f8a9330b88fa6129d05ce42ac3a8cc835295a6f_seed1"
      / "yaw_calibration.json": (
        "123122e75955468dfc475d86ac3f9160b428720fd8e1b90ab614bc1bc0749765"
      ),
      ARTIFACTS
      / "c1_posture_requalification_seed1"
      / "posture_map_seed1_registered_p032.json": (
        "b8e627f85b53d21dd8d9c26edbe2943151d9bcf9e5864ff998ede5f909118e23"
      ),
      ARTIFACTS
      / "c1_posture_requalification_seed1"
      / "station_calibration_seed1.json": (
        "f22a9b66f734004ff14b6586a22a991d527f360806bbbdefe096e9f0474db72a"
      ),
    }
    self.assertEqual(len(expected), 5)
    for path, digest in expected.items():
      with self.subTest(path=path):
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        self.assertIn(digest, self.text)
        self.assertIn(path.name, self.text)
    for variable in (
      "HOPPERTREX_HYBRID_CONTROLLER_PATH",
      "HOPPERTREX_HYBRID_CALIBRATION_PATH",
      "HOPPERTREX_HYBRID_YAW_CALIBRATION_PATH",
      "HOPPERTREX_HYBRID_POSTURE_MAP_PATH",
      "HOPPERTREX_HYBRID_STATION_CALIBRATION_PATH",
    ):
      self.assertIn(variable, self.text)

  def test_preflight_pins_read_only_trigger_and_defaults_contract(self) -> None:
    body = function_body(self.text, "Invoke-ValidatePhase", "Invoke-Fresh1000Phase")
    required = (
      "hoppertrex_mjlab.scripts.rsl_rl.preflight_stair_camp",
      "'replay-c2'",
      "'--input-dir'",
      "'finalize'",
      "'--c2-replay'",
      "'--flat-fp'",
      "'--stage5-kick-fp'",
      "trigger_replay.completed_pairs -ne 288",
      "trigger_replay.detections -ne 288",
      "trigger_replay.pre_impact_triggers -ne 0",
      "trigger_replay.files_unchanged -ne $true",
      "camp_flat_rolling,stage5_kick",
      "STAIR_CAMP_PREFLIGHT_PASS",
      "STOP_NO_PROMOTION",
      "$RegisteredNumEnvs = 256",
      "$RegisteredFreshUpdates = 1000",
      "$RegisteredSaveInterval = 100",
      "$RegisteredStepsPerIteration = 24",
      "classical_rows.json",
      "@{ name = 'ClassicalRowsPath'; value = $ClassicalRowsPath }",
      "Read-ClassicalRowsSource -Path $classicalRowsSource",
      "Assert-ClassicalRowsMatchFrozenProbe",
      "Copy-Item -LiteralPath $classicalRowsSource",
      "classical_rows_source_sha256",
      "classical_rows_file_sha256",
      "hoppertrex_mjlab.scripts.rsl_rl.stair_camp_live_adapter",
      "stair_camp_trigger_pretraining_request",
      "pretraining_trigger_request.json",
      "Write-AtomicJsonNoClobber -Path $pretrainingRequestPath",
      "'trigger-fp'",
      "'--domain', 'camp_flat_rolling'",
      "'--domain', 'stage5_kick'",
      "'--request', $pretrainingRequestPath",
      "Assert-LiveTriggerFalsePositivePayload",
      "-ExpectedEvents 96000",
      "-ExpectedEvents 128",
      "pretraining_trigger_request_sha256",
      "pretraining_policy = 'deterministic_zero_residual'",
      "camp_flat_rolling_fp_sha256",
      "stage5_kick_fp_sha256",
    )
    for value in required:
      self.assertIn(value, body + self.text)
    self.assertNotIn("'--expected-mjlab-git-sha'", body)
    for forbidden in (
      "FlatFalsePositivePath",
      "Stage5KickFalsePositivePath",
      "$flatFpSource",
      "$kickFpSource",
      "checkpoint =",
    ):
      self.assertNotIn(forbidden, body)

  def test_classical_rows_source_is_strict_and_never_synthesized(self) -> None:
    schema = function_body(
      self.text, "Assert-ClassicalRows", "New-CheckpointEnvelope"
    )
    probe_match = function_body(
      self.text, "Assert-ClassicalRowsMatchFrozenProbe", "New-CheckpointEnvelope"
    )
    for value in (
      "$rootFields.Count -ne 1",
      "$rootFields[0] -cne 'rows'",
      "$payload.rows -isnot [System.Array]",
      "$actualFields.Count -ne $expectedFields.Count",
      "'height_m'",
      "'success_rate'",
      "'terminations'",
      "'non_wheel_contacts'",
      "'trials'",
      "$probeBackedHeights = @(0.01, 0.02, 0.03, 0.05, 0.07, 0.10)",
      "$expectedHeights = @(0.01, 0.02, 0.03, 0.05, 0.07, 0.10)",
      "Every accepted classical row is now probe-backed",
    ):
      self.assertIn(value, schema)
    for forbidden in (
      "Get-ClassicalRowsFromProbe",
      "ClassicalSentinel",
      "classical_nonpassing_sentinel_heights",
      "classical_sentinel_cannot_change_closed_prefix",
    ):
      self.assertNotIn(forbidden, self.text)
    self.assertNotIn("$rows.Add", probe_match)
    self.assertNotIn("success_rate =", probe_match)

  def test_fresh_training_has_exact_defaults_and_no_resume_surface(self) -> None:
    body = function_body(self.text, "Invoke-Fresh1000Phase", "Invoke-Extend3000Phase")
    for value in (
      "$FreshTrainingArguments",
      "$Budget -ne $RegisteredFreshUpdates",
      "'--gpu-ids', '0'",
      "'--log-root', $script:TrainingBaseLogRoot",
      "'--env.seed', [string]$Seed",
      "'--env.scene.num-envs', [string]$RegisteredNumEnvs",
      "'--agent.seed', [string]$Seed",
      "'--agent.max-iterations', [string]$RegisteredFreshUpdates",
      "'--agent.save-interval', [string]$RegisteredSaveInterval",
      "'--agent.num-steps-per-env', [string]$RegisteredStepsPerIteration",
      "'--agent.run-name', $runName",
      "'model_999.pt'",
      "model_700.progress.json",
      "model_800.progress.json",
      "model_999.progress.json",
      "-CompletedUpdates 701 -ExpectedEvaluations 14",
      "-CompletedUpdates 801 -ExpectedEvaluations 16",
      "-ExpectedEvaluations 20",
      "stall_predicate.json",
      "evaluation_delta = 6",
      "monotone_curriculum_bound = $true",
      "first_stalled_evaluation = 15",
      "last_stalled_evaluation = 20",
      "predicate_satisfied = $upperHeightUnchanged",
      "resume = $false",
    ):
      self.assertIn(value, body)
    for forbidden in (
      "'--agent.resume'",
      "'--agent.load-run'",
      "'--agent.load-checkpoint'",
    ):
      self.assertNotIn(forbidden, body)

  def test_extension_uses_seed1_stall_and_one_campaign_budget_decision(self) -> None:
    body = function_body(self.text, "Invoke-Extend3000Phase", "Invoke-SelectK3Phase")
    for value in (
      "$Budget -ne $RegisteredExtensionTotalUpdates",
      "$Seed -eq 1",
      "$AuthorizeExtension.IsPresent",
      "Get-VerifiedStallPredicate -TrainingSeed 1",
      "six-evaluation stall predicate",
      "Get-VerifiedBudgetDecision",
      "Seeds 2 and 3 do not evaluate their own stall",
      "budget_decision.json",
      "stair_camp_campaign_budget_decision",
      "EXTEND_ALL_SEEDS_TO_TOTAL_3000",
      "decision_maker_training_seed = 1",
      "seed1_stall_predicate_sha256",
      "seed1_extension_checkpoint_sha256",
      "seed1_evaluation_delta = 6",
      "Write-AtomicJsonNoClobber -Path $budgetDecisionPath",
      "seed{0}\\fresh-1000",
      "model_999.envelope.json",
      "$freshEnvelope.checkpoint_file_sha256 -ne $freshManifest.final_checkpoint_sha256",
      "extension_source.validation.json",
      "extension_source_completed_updates = $RegisteredFreshUpdates",
      "'--log-root', $script:TrainingBaseLogRoot",
      "'--env.seed', [string]$Seed",
      "'--agent.seed', [string]$Seed",
      "'--agent.max-iterations', [string]$RegisteredExtensionTotalUpdates",
      "'--agent.resume', 'True'",
      "'--agent.load-run', $loadRunRegex",
      "'--agent.load-checkpoint', $loadCheckpointRegex",
      "'model_2999.pt'",
      "user_authorized = [bool]$budgetDecision.user_authorized",
      "authorization_source = 'seed1_campaign_budget_decision'",
    ):
      self.assertIn(value, body)
    self.assertIn("completed seed-1 extension evidence", self.text)
    seed_two_branch = body[body.index("} else {") : body.index("$freshRoot")]
    self.assertNotIn("Get-VerifiedStallPredicate -TrainingSeed $Seed", seed_two_branch)
    self.assertLess(
      body.index("$seedOnePredicate.predicate_satisfied -ne $true"),
      body.index("Write-AtomicJsonNoClobber -Path $budgetDecisionPath"),
    )
    self.assertLess(
      body.index("New-CheckpointEnvelope -CheckpointPath $finalCheckpoint"),
      body.index("Write-AtomicJsonNoClobber -Path $budgetDecisionPath"),
    )

  def test_training_log_root_and_unique_run_attribution_match_train_cli(self) -> None:
    for value in (
      "$script:TrainingBaseLogRoot = Join-Path $script:RepoRoot 'logs\\rsl_rl'",
      "$script:TrainingLogRoot = Join-Path $script:TrainingBaseLogRoot 'hoppertrex_stair_camp_s5b'",
      "Get-ChildItem -LiteralPath $script:TrainingLogRoot -Directory",
      "Fresh training did not create exactly one attributable run directory",
      "Extension did not create exactly one attributable run directory",
      "$runs.Count -ne 1",
    ):
      self.assertIn(value, self.text)
    self.assertNotIn("src\\hoppertrex_mjlab\\logs\\rsl_rl", self.text)

  def test_progress_extractor_and_stall_reports_are_strictly_bound(self) -> None:
    progress = function_body(
      self.text, "New-StairCampProgressReport", "Assert-ExactPropertyNames"
    )
    for value in (
      'infos.get("stair_camp_progress")',
      'infos.get("stair_camp_curriculum")',
      'infos.get("env_state")',
      'common_step != completed_updates * 24',
      'evaluations != expected_evaluations',
      'evaluation_interval != 1200',
      '"checkpoint_file_sha256"',
      '"upper_height_m"',
      '"trigger_rate"',
      '"residual_abs_mean"',
      '"residual_rms"',
      '"residual_abs_max"',
      '"curriculum_sha256"',
    ):
      self.assertIn(value, progress)
    stall = function_body(
      self.text, "Get-VerifiedStallPredicate", "Get-VerifiedBudgetDecision"
    )
    for value in (
      "-CompletedUpdates 701 -ExpectedEvaluations 14",
      "-CompletedUpdates 1000 -ExpectedEvaluations 20",
      "evaluation_delta -ne 6",
      "predicate_satisfied -ne $upperUnchanged",
    ):
      self.assertIn(value, stall)

  def test_k3_uses_real_zero_based_periodic_and_final_cadence(self) -> None:
    selection = function_body(
      self.text, "Invoke-SelectK3Phase", "Invoke-EvaluatePhase"
    )
    screen = function_body(self.text, "Invoke-K3Screen", "Invoke-FormalEvaluation")
    for value in (
      "model_800/model_900/model_999 with 801/901/1000 updates",
      "model_2800/model_2900/model_2999 with 2801/2901/3000 updates",
      "$Budget - (2 * $RegisteredSaveInterval)",
      "$Budget - $RegisteredSaveInterval",
      "$Budget - 1",
      "$Budget - (2 * $RegisteredSaveInterval) + 1",
      "$Budget - $RegisteredSaveInterval + 1",
      "'select-k3'",
      "--candidate",
      "newest passing K=3 checkpoint selected",
    ):
      self.assertIn(value, selection)
    for value in (
      "K3_SCREEN_PROTOCOL",
      "profile=\"smoke\"",
      "profile=\"screen\"",
      "num_envs_per_cell=int(protocol.num_envs_per_cell)",
      "repeats=int(protocol.repeats)",
      "make_k3_screen_candidate",
      "flat_collection = adapter(flat_config)",
    ):
      self.assertIn(value, screen)
    for forbidden in ("model_799", "model_899", "model_2799", "model_2899"):
      self.assertNotIn(forbidden, self.text)

  def test_formal_evidence_uses_live_adapter_and_strict_compose_seed(self) -> None:
    body = function_body(self.text, "Invoke-EvaluatePhase", "Invoke-AdjudicatePhase")
    for value in (
      "hoppertrex_mjlab.scripts.rsl_rl.stair_camp_live_adapter:collect",
      "Invoke-FormalEvaluation -Domain 'flat' -Ablation 'baseline'",
      "Invoke-FormalEvaluation -Domain 'stairs' -Ablation 'baseline'",
      "Invoke-FormalEvaluation -Domain 'slope' -Ablation 'baseline'",
      "'leg-off'",
      "'zero-shot-scale-0.035'",
      "'zero-shot-scale-0.070'",
      "'zero-shot-scale-0.100'",
      "'mode-always-on'",
      "'compose-seed'",
      "'--stairs-result'",
      "'--flat-result'",
      "'--classical-rows'",
      "'--ablation-result'",
      "'--k3-selection'",
      "'--budget-iterations'",
      "Assert-ComposedSeedEnvelope",
    ):
      self.assertIn(value, body if value != "hoppertrex_mjlab.scripts.rsl_rl.stair_camp_live_adapter:collect" else self.text)
    self.assertNotIn("residual_rows =", body)
    self.assertNotIn("gate_stair_mode_false_positives =", body)

  def test_composed_and_adjudicated_schema_is_strict(self) -> None:
    for value in (
      "evidence_eligible -ne $true",
      "completed_ablations",
      "gate_stair_mode_false_positives",
      "checkpoint_file_sha256",
      "artifact_bindings.PSObject.Properties.Name).Count -ne 6",
      "$actualBindingNames.Count -ne $bindingNames.Count",
      "$falsePositiveNames.Count -ne $gateNames.Count",
      "gateValue -isnot [bool]",
      "checkpoint bytes do not match its declared hash",
      "controller_gain_hash",
      "calibration_hash",
      "yaw_calibration_hash",
      "posture_map_hash",
      "posture_artifact_hash",
      "station_calibration_hash",
      "RESIDUAL_PPO_EXTENDS_CLASSICAL_BOUNDARY",
      "STOP_NO_PROMOTION",
    ):
      self.assertIn(value, self.text)

  def test_atomic_no_clobber_and_immutable_packaging_are_locked(self) -> None:
    for value in (
      ".incomplete.",
      "[System.Guid]::NewGuid().ToString('N')",
      "Refusing to overwrite",
      "[System.IO.File]::Move($temporary, $Path)",
      "Move-Item -LiteralPath $WorkingPath -Destination $FinalPath",
      "Compress-Archive",
      "SHA256SUMS.txt",
      "Copy-Item -LiteralPath $script:CampaignRootPath",
      "selected_checkpoints",
      "extension_source_checkpoint_sha256",
      "Move-Item -LiteralPath $working -Destination $outputDirectory",
      "Move-Item -LiteralPath $temporaryZip -Destination $outputZip",
      "$temporaryZipHashPath",
      "pathsStayInPackageParent",
      "Move-Item -LiteralPath $zipHashPath -Destination $temporaryZipHashPath",
      "immutable campaign package",
    ):
      self.assertIn(value, self.text)

  def test_cli_flags_are_checked_against_current_help_before_use(self) -> None:
    self.assertIn("Get-NativeHelpText", self.text)
    self.assertIn("Assert-HelpContains", self.text)
    self.assertIn("'--domain', '--request', '--output'", self.text)
    self.assertIn("'--log-root'", self.text)
    self.assertIn("'--agent.seed'", self.text)
    for command in (
      "'replay-c2', '--help'",
      "'finalize', '--help'",
      "'manifest', '--help'",
      "'trigger-fp', '--help'",
      "$Task, '--help'",
      "'validate-checkpoint', '--help'",
      "'select-k3', '--help'",
      "'live', '--help'",
      "'compose-seed', '--help'",
      "$AdjudicatorModule, '--help'",
    ):
      self.assertIn(command, self.text)
    self.assertLess(
      self.text.rindex("# Each flag used by the selected phase"),
      self.text.rindex("switch ($Phase)"),
    )

  def test_exit_classes_keep_legal_scientific_stop_at_zero(self) -> None:
    for value in (
      "$ExitCodeSuccess = 0",
      "$ExitCodeScientificStop = 0",
      "$ExitCodeProvenance = 20",
      "$ExitCodeProtocol = 30",
      "$ExitCodeOperational = 40",
      "$exception.Data['StairCampExitCode'] = $code",
      "exit $exitCode",
      "legal scientific STOP archived",
      "Scientific STOP archived",
    ):
      self.assertIn(value, self.text)
    self.assertNotIn("STOP_NO_PROMOTION archived at", self.text)

  def test_no_placeholders_unsafe_dispatch_or_environment_mutation(self) -> None:
    for forbidden in (
      "Invoke-Expression",
      "uv sync",
      "TODO",
      "TBD",
      "FIXME",
      "PLACEHOLDER",
    ):
      self.assertNotIn(forbidden.lower(), self.text.lower())
    self.assertIsNone(re.search(r"<[A-Za-z][^>\r\n]{0,80}>", self.text))


if __name__ == "__main__":
  unittest.main()
