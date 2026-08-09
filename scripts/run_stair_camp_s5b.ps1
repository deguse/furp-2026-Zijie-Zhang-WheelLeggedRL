[CmdletBinding()]
param(
  [ValidateSet('Validate', 'Fresh1000', 'Extend3000', 'SelectK3', 'Evaluate', 'Adjudicate', 'Package')]
  [string]$Phase = 'Validate',
  [Parameter(Mandatory = $true)]
  [ValidatePattern('^[0-9a-fA-F]{40}$')]
  [string]$ExpectedGitSha,
  [Parameter(Mandatory = $true)]
  [ValidateNotNullOrEmpty()]
  [string]$CampaignRoot,
  [ValidateSet(1, 2, 3)]
  [int]$Seed,
  [ValidateSet(1000, 3000)]
  [int]$Budget = 1000,
  [switch]$AuthorizeExtension,
  [string]$C2ReplayDirectory,
  [string]$ClassicalProbePath,
  [string]$ClassicalRowsPath,
  [string]$Python
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$RequiredBranch = 'codex/p2-classical-upper-bound'
$RequiredMjLabSha = '43e0f3ea9c92ddbb4de9f3bb1ac772d604e3ebf6'
$Task = 'HopperTrex-Hybrid-v2-StairCamp'
$RequiredContractSha256 = '1d4b18db32e48b3ae8803e385a032203bdddc7f8198da9679f519bc8947190cb'
$LiveAdapter = 'hoppertrex_mjlab.scripts.rsl_rl.stair_camp_live_adapter:collect'
$LiveAdapterModule = 'hoppertrex_mjlab.scripts.rsl_rl.stair_camp_live_adapter'
$TrainingModule = 'hoppertrex_mjlab.scripts.rsl_rl.train'
$PreflightModule = 'hoppertrex_mjlab.scripts.rsl_rl.preflight_stair_camp'
$EvaluatorModule = 'hoppertrex_mjlab.scripts.rsl_rl.evaluate_stair_camp'
$AdjudicatorModule = 'hoppertrex_mjlab.scripts.rsl_rl.adjudicate_stair_camp'
$RequiredDevice = 'cuda:0'
$RegisteredTrainingSeeds = @(1, 2, 3)
$RegisteredEvaluationSeed = 1
$RegisteredNumEnvs = 256
$RegisteredFreshUpdates = 1000
$RegisteredExtensionTotalUpdates = 3000
$RegisteredSaveInterval = 100
$RegisteredStepsPerIteration = 24
$RequiredClassicalProbeSha256 = 'e85ee64ff60337fc60c894558af193c5a82f00811772d22fcb00fc5d10830da5'

# Scientific STOP is a complete result and exits zero. Input identity,
# protocol, and operational failures are distinct nonzero classes.
$ExitCodeSuccess = 0
$ExitCodeScientificStop = 0
$ExitCodeProvenance = 20
$ExitCodeProtocol = 30
$ExitCodeOperational = 40

$ScheduleFileSha256 = '9b21125e7cc48be3ea61e12a67171a855892ad3ced1f54b3176ed979e76224ec'
$CalibrationFileSha256 = 'ef002d0d622725509b47c8ff40d8af658fd42f705bdeac67ac35bae4458f889d'
$YawFileSha256 = '123122e75955468dfc475d86ac3f9160b428720fd8e1b90ab614bc1bc0749765'
$PostureFileSha256 = 'b8e627f85b53d21dd8d9c26edbe2943151d9bcf9e5864ff998ede5f909118e23'
$StationFileSha256 = 'f22a9b66f734004ff14b6586a22a991d527f360806bbbdefe096e9f0474db72a'
$ControllerScheduleHash = '8fe8548bca85978c164bbd7de39d2d6463cdfd8d7ab91796cf57696b0f64e203'
$IdentificationControllerGainHash = '8fee25a0339dd1e99127cbed912941dc3ad8ef2030ce49a0d310d1563cb87d98'
$VelocityCalibrationHash = 'f62648b57bd17a3503bcbdbf58f349f91fcd8de8ef0cf04551c200401233ed01'
$YawCalibrationHash = 'b2fd044fd355cc1f57558c76bde8a6fd2ab4435dbdb6c1dc6209caa2dd91a641'
$PostureMapHash = '4289fb286c6a76a2aca2652d6bcc40acb1bf9c1f70b779a47ceff65c4dca3513'
$PostureArtifactHash = '3b96fd3dae66ad781b5b875c74184db101c42da02c53dfcc40a5137a6b5de11a'
$StationCalibrationHash = 'c00e859b3093b4812d54799253accdaeb99171a2cf4028b08bc39e68eaaa7d8a'

function Stop-Campaign {
  param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Provenance', 'Protocol', 'Operational')]
    [string]$Kind,
    [Parameter(Mandatory = $true)]
    [string]$Message
  )
  $code = switch ($Kind) {
    'Provenance' { $ExitCodeProvenance }
    'Protocol' { $ExitCodeProtocol }
    default { $ExitCodeOperational }
  }
  $exception = New-Object System.InvalidOperationException(
    ('STAIR_CAMP_{0}: {1}' -f $Kind.ToUpperInvariant(), $Message)
  )
  $exception.Data['StairCampExitCode'] = $code
  throw $exception
}

function Assert-CommandAvailable {
  param([Parameter(Mandatory = $true)][string]$Name)
  if ($null -eq (Get-Command $Name -ErrorAction SilentlyContinue)) {
    Stop-Campaign -Kind 'Operational' -Message ('Missing required command: {0}' -f $Name)
  }
}

function Get-FileSha256 {
  param([Parameter(Mandatory = $true)][string]$Path)
  return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-CanonicalScriptSha256 {
  param([Parameter(Mandatory = $true)][string]$Path)
  $content = [System.IO.File]::ReadAllText($Path)
  $normalized = $content.Replace([string][char]13 + [char]10, [string][char]10).Replace([string][char]13, [string][char]10)
  $bytes = [System.Text.UTF8Encoding]::new($false).GetBytes($normalized)
  $sha256 = [System.Security.Cryptography.SHA256]::Create()
  try {
    $digest = $sha256.ComputeHash($bytes)
    return ([System.BitConverter]::ToString($digest)).Replace('-', '').ToLowerInvariant()
  } finally {
    $sha256.Dispose()
  }
}

function Write-AtomicJsonNoClobber {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][object]$Value,
    [int]$Depth = 30
  )
  if (Test-Path -LiteralPath $Path) {
    Stop-Campaign -Kind 'Operational' -Message ('Refusing to overwrite JSON output: {0}' -f $Path)
  }
  $parent = Split-Path -Parent $Path
  if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
    New-Item -ItemType Directory -Path $parent | Out-Null
  }
  $temporary = Join-Path $parent ('.{0}.incomplete.{1}' -f (Split-Path -Leaf $Path), [System.Guid]::NewGuid().ToString('N'))
  try {
    $json = ($Value | ConvertTo-Json -Depth $Depth)
    $json = $json.Replace([string][char]13 + [char]10, [string][char]10).Replace([string][char]13, [string][char]10) + [char]10
    [System.IO.File]::WriteAllText($temporary, $json, [System.Text.UTF8Encoding]::new($false))
    [System.IO.File]::Move($temporary, $Path)
  } finally {
    if (Test-Path -LiteralPath $temporary -PathType Leaf) {
      [System.IO.File]::Delete($temporary)
    }
  }
}

function Write-AtomicTextNoClobber {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string[]]$Lines
  )
  if (Test-Path -LiteralPath $Path) {
    Stop-Campaign -Kind 'Operational' -Message ('Refusing to overwrite text output: {0}' -f $Path)
  }
  $parent = Split-Path -Parent $Path
  if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
    New-Item -ItemType Directory -Path $parent | Out-Null
  }
  $temporary = Join-Path $parent ('.{0}.incomplete.{1}' -f (Split-Path -Leaf $Path), [System.Guid]::NewGuid().ToString('N'))
  try {
    $text = (($Lines -join [char]10) + [char]10)
    [System.IO.File]::WriteAllText($temporary, $text, [System.Text.UTF8Encoding]::new($false))
    [System.IO.File]::Move($temporary, $Path)
  } finally {
    if (Test-Path -LiteralPath $temporary -PathType Leaf) {
      [System.IO.File]::Delete($temporary)
    }
  }
}

function Invoke-NativeChecked {
  param(
    [Parameter(Mandatory = $true)][string]$Executable,
    [Parameter(Mandatory = $true)][string[]]$Arguments,
    [Parameter(Mandatory = $true)][string]$FailureMessage,
    [ValidateSet('Provenance', 'Protocol', 'Operational')]
    [string]$FailureKind = 'Operational'
  )
  $previousPreference = $ErrorActionPreference
  $exitCode = -1
  try {
    $ErrorActionPreference = 'Continue'
    & $Executable @Arguments
    $exitCode = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $previousPreference
  }
  if ($exitCode -ne 0) {
    Stop-Campaign -Kind $FailureKind -Message ('{0} Exit code: {1}' -f $FailureMessage, $exitCode)
  }
}

function Invoke-NativeLogged {
  param(
    [Parameter(Mandatory = $true)][string]$Executable,
    [Parameter(Mandatory = $true)][string[]]$Arguments,
    [Parameter(Mandatory = $true)][string]$LogPath,
    [Parameter(Mandatory = $true)][string]$FailureMessage,
    [ValidateSet('Provenance', 'Protocol', 'Operational')]
    [string]$FailureKind = 'Operational',
    [switch]$Append
  )
  $previousPreference = $ErrorActionPreference
  $exitCode = -1
  try {
    $ErrorActionPreference = 'Continue'
    if ($Append.IsPresent) {
      & $Executable @Arguments 2>&1 | Tee-Object -FilePath $LogPath -Append
    } else {
      & $Executable @Arguments 2>&1 | Tee-Object -LiteralPath $LogPath
    }
    $exitCode = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $previousPreference
  }
  if ($exitCode -ne 0) {
    Stop-Campaign -Kind $FailureKind -Message ('{0} Exit code: {1}. Log: {2}' -f $FailureMessage, $exitCode, $LogPath)
  }
}

function Get-NativeHelpText {
  param(
    [Parameter(Mandatory = $true)][string]$Executable,
    [Parameter(Mandatory = $true)][string[]]$Arguments,
    [Parameter(Mandatory = $true)][string]$Name
  )
  $previousPreference = $ErrorActionPreference
  $exitCode = -1
  try {
    $ErrorActionPreference = 'Continue'
    $lines = @(& $Executable @Arguments 2>&1)
    $exitCode = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $previousPreference
  }
  if ($exitCode -ne 0) {
    Stop-Campaign -Kind 'Protocol' -Message ('{0} --help failed with exit code {1}.' -f $Name, $exitCode)
  }
  return ($lines -join [char]10)
}

function Assert-HelpContains {
  param(
    [Parameter(Mandatory = $true)][string]$HelpText,
    [Parameter(Mandatory = $true)][string[]]$Flags,
    [Parameter(Mandatory = $true)][string]$Name
  )
  foreach ($flag in $Flags) {
    if (-not $HelpText.Contains($flag)) {
      Stop-Campaign -Kind 'Protocol' -Message ('{0} --help no longer exposes required flag {1}.' -f $Name, $flag)
    }
  }
}

function Assert-SeedPhase {
  if ($RegisteredTrainingSeeds -notcontains $Seed) {
    Stop-Campaign -Kind 'Protocol' -Message ('Phase {0} requires one training seed exactly in 1, 2, 3.' -f $Phase)
  }
}

function New-PhaseWorkspace {
  param([Parameter(Mandatory = $true)][string]$FinalPath)
  if (Test-Path -LiteralPath $FinalPath) {
    Stop-Campaign -Kind 'Operational' -Message ('Refusing to overwrite completed phase output: {0}' -f $FinalPath)
  }
  $parent = Split-Path -Parent $FinalPath
  if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
    New-Item -ItemType Directory -Path $parent | Out-Null
  }
  $working = $FinalPath + '.incomplete.' + [System.Guid]::NewGuid().ToString('N')
  New-Item -ItemType Directory -Path $working | Out-Null
  return $working
}

function Publish-PhaseWorkspace {
  param(
    [Parameter(Mandatory = $true)][string]$WorkingPath,
    [Parameter(Mandatory = $true)][string]$FinalPath
  )
  if (Test-Path -LiteralPath $FinalPath) {
    Stop-Campaign -Kind 'Operational' -Message ('Phase destination appeared before atomic publish: {0}' -f $FinalPath)
  }
  Move-Item -LiteralPath $WorkingPath -Destination $FinalPath
}

function Test-JsonNumber {
  param([AllowNull()][object]$Value)
  return (
    $Value -is [byte] -or
    $Value -is [sbyte] -or
    $Value -is [int16] -or
    $Value -is [uint16] -or
    $Value -is [int32] -or
    $Value -is [uint32] -or
    $Value -is [int64] -or
    $Value -is [uint64] -or
    $Value -is [decimal] -or
    $Value -is [single] -or
    $Value -is [double]
  )
}

function Test-JsonInteger {
  param([AllowNull()][object]$Value)
  return (
    $Value -is [byte] -or
    $Value -is [sbyte] -or
    $Value -is [int16] -or
    $Value -is [uint16] -or
    $Value -is [int32] -or
    $Value -is [uint32] -or
    $Value -is [int64] -or
    $Value -is [uint64]
  )
}

function Assert-ClassicalRows {
  param([Parameter(Mandatory = $true)][object[]]$Rows)
  # Frozen C0 evidence grid, NOT the seven-height residual scan. The frozen
  # stair probe swept 0.00-0.10 m in 0.01 m steps, so it cannot supply a
  # measured 0.15 m classical row, and that cell cannot change the verdict:
  # the classical contiguous passing prefix already terminates at 0.01 m
  # (measured 0/48 at every tier from 0.01 m up). Demanding it here would
  # force either an authored number or a re-sweep of a frozen script.
  $expectedHeights = @(0.01, 0.02, 0.03, 0.05, 0.07, 0.10)
  $expectedFields = @(
    'height_m',
    'success_rate',
    'terminations',
    'non_wheel_contacts',
    'trials'
  )
  if ($Rows.Count -ne $expectedHeights.Count) {
    Stop-Campaign -Kind 'Protocol' -Message 'Classical rows must contain the six frozen-C0 positive heights.'
  }
  for ($index = 0; $index -lt $expectedHeights.Count; $index += 1) {
    $row = $Rows[$index]
    $actualFields = @($row.PSObject.Properties.Name)
    $unexpectedFields = @($actualFields | Where-Object { $expectedFields -cnotcontains $_ })
    $missingFields = @($expectedFields | Where-Object { $actualFields -cnotcontains $_ })
    if (
      $actualFields.Count -ne $expectedFields.Count -or
      $unexpectedFields.Count -ne 0 -or
      $missingFields.Count -ne 0
    ) {
      Stop-Campaign -Kind 'Protocol' -Message ('Classical row {0} must contain exactly the five registered fields.' -f $index)
    }
    if (
      -not (Test-JsonNumber -Value $row.height_m) -or
      -not (Test-JsonNumber -Value $row.success_rate) -or
      -not (Test-JsonInteger -Value $row.terminations) -or
      -not (Test-JsonInteger -Value $row.non_wheel_contacts) -or
      -not (Test-JsonInteger -Value $row.trials)
    ) {
      Stop-Campaign -Kind 'Protocol' -Message ('Classical row {0} contains a non-numeric or non-integral field.' -f $index)
    }
    $height = [double]$row.height_m
    $successRate = [double]$row.success_rate
    $terminations = [long]$row.terminations
    $nonWheelContacts = [long]$row.non_wheel_contacts
    $trials = [long]$row.trials
    if (
      [double]::IsNaN($height) -or
      [double]::IsInfinity($height) -or
      [double]::IsNaN($successRate) -or
      [double]::IsInfinity($successRate) -or
      [Math]::Abs($height - [double]$expectedHeights[$index]) -gt 1.0e-12 -or
      $successRate -lt 0.0 -or
      $successRate -gt 1.0 -or
      $terminations -lt 0 -or
      $nonWheelContacts -lt 0 -or
      $trials -ne 48 -or
      $terminations -gt $trials -or
      $nonWheelContacts -gt $trials
    ) {
      Stop-Campaign -Kind 'Protocol' -Message ('Classical row {0} violates the registered height, trial, or safety-count schema.' -f $index)
    }
  }
}

function Read-ClassicalRowsSource {
  param([Parameter(Mandatory = $true)][string]$Path)
  try {
    $payload = Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
  } catch {
    Stop-Campaign -Kind 'Protocol' -Message 'Classical rows source is not valid UTF-8 JSON.'
  }
  $rootFields = @($payload.PSObject.Properties.Name)
  if (
    $rootFields.Count -ne 1 -or
    $rootFields[0] -cne 'rows' -or
    $payload.rows -isnot [System.Array]
  ) {
    Stop-Campaign -Kind 'Protocol' -Message 'Classical rows source must contain only one array-valued key named "rows".'
  }
  $rows = @($payload.rows)
  Assert-ClassicalRows -Rows $rows
  return $rows
}

function Assert-ClassicalRowsEquivalent {
  param(
    [Parameter(Mandatory = $true)][object[]]$Expected,
    [Parameter(Mandatory = $true)][object[]]$Actual
  )
  Assert-ClassicalRows -Rows $Expected
  Assert-ClassicalRows -Rows $Actual
  for ($index = 0; $index -lt $Expected.Count; $index += 1) {
    if (
      [Math]::Abs([double]$Actual[$index].height_m - [double]$Expected[$index].height_m) -gt 1.0e-12 -or
      [Math]::Abs([double]$Actual[$index].success_rate - [double]$Expected[$index].success_rate) -gt 1.0e-12 -or
      [long]$Actual[$index].terminations -ne [long]$Expected[$index].terminations -or
      [long]$Actual[$index].non_wheel_contacts -ne [long]$Expected[$index].non_wheel_contacts -or
      [long]$Actual[$index].trials -ne [long]$Expected[$index].trials
    ) {
      Stop-Campaign -Kind 'Provenance' -Message ('Archived classical row {0} differs from the campaign manifest.' -f $index)
    }
  }
}

function Assert-ClassicalRowsMatchFrozenProbe {
  param(
    [Parameter(Mandatory = $true)][object[]]$Rows,
    [Parameter(Mandatory = $true)][string]$ProbePath
  )
  try {
    $probe = Get-Content -LiteralPath $ProbePath -Raw -Encoding UTF8 | ConvertFrom-Json
  } catch {
    Stop-Campaign -Kind 'Protocol' -Message 'Frozen classical probe is not valid UTF-8 JSON.'
  }
  if (@($probe.PSObject.Properties.Name) -cnotcontains 'cells' -or $probe.cells -isnot [System.Array]) {
    Stop-Campaign -Kind 'Protocol' -Message 'Frozen classical probe omitted its cells array.'
  }
  $centerCells = @($probe.cells | Where-Object { $_.posture_card -ceq 'envelope_center' })
  $probeBackedHeights = @(0.01, 0.02, 0.03, 0.05, 0.07, 0.10)
  for ($index = 0; $index -lt $probeBackedHeights.Count; $index += 1) {
    $height = [double]$probeBackedHeights[$index]
    $matches = @($centerCells | Where-Object {
      [Math]::Abs([double]$_.stair_height_m - $height) -le 1.0e-12
    })
    if ($matches.Count -ne 1) {
      Stop-Campaign -Kind 'Protocol' -Message ('Frozen C0 probe has invalid center-card multiplicity at {0} m.' -f $height)
    }
    $cell = $matches[0]
    $row = $Rows[$index]
    if (
      [Math]::Abs([double]$row.success_rate - [double]$cell.success_rate) -gt 1.0e-12 -or
      [long]$row.terminations -ne [long]$cell.terminated_trials -or
      [long]$row.non_wheel_contacts -ne [long]$cell.non_wheel_contact_trials -or
      [long]$row.trials -ne [long]$cell.trials
    ) {
      Stop-Campaign -Kind 'Protocol' -Message ('Canonical classical row at {0} m does not match the frozen C0 center card.' -f $height)
    }
  }
  $probeFifteenRows = @($centerCells | Where-Object {
    [Math]::Abs([double]$_.stair_height_m - 0.15) -le 1.0e-12
  })
  if ($probeFifteenRows.Count -ne 0) {
    Stop-Campaign -Kind 'Protocol' -Message 'Frozen C0 probe unexpectedly contains a 0.15 m center card.'
  }
  # Every accepted classical row is now probe-backed: the required grid equals
  # the six heights cross-checked above, so no classical cell can enter the
  # campaign without a frozen measurement behind it.
}

function New-CheckpointEnvelope {
  param(
    [Parameter(Mandatory = $true)][string]$CheckpointPath,
    [Parameter(Mandatory = $true)][string]$EnvelopePath,
    [Parameter(Mandatory = $true)][int]$TrainingSeed,
    [Parameter(Mandatory = $true)][int]$CompletedUpdates,
    [Parameter(Mandatory = $true)][string]$LogPath
  )
  if (-not (Test-Path -LiteralPath $CheckpointPath -PathType Leaf)) {
    Stop-Campaign -Kind 'Protocol' -Message ('Required checkpoint is missing: {0}' -f $CheckpointPath)
  }
  $extractor = @'
import pathlib
import sys
import torch
from hoppertrex_mjlab.scripts.rsl_rl.evaluate_stair_camp import (
    CheckpointExpectation,
    checkpoint_envelope_from_loaded_checkpoint,
    write_machine_output,
)
checkpoint_path = pathlib.Path(sys.argv[1]).resolve()
output_path = pathlib.Path(sys.argv[2]).resolve()
expectation = CheckpointExpectation(
    git_sha=sys.argv[3],
    contract_sha256=sys.argv[4],
    training_seed=int(sys.argv[5]),
    completed_updates=int(sys.argv[6]),
)
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
if not isinstance(checkpoint, dict):
    raise TypeError("runner checkpoint must be a mapping")
envelope = checkpoint_envelope_from_loaded_checkpoint(
    checkpoint_path,
    checkpoint,
    expectation=expectation,
)
write_machine_output(envelope, output_path)
'@
  Invoke-NativeLogged -Executable $script:PythonExe -Arguments @(
    '-c', $extractor,
    $CheckpointPath,
    $EnvelopePath,
    $script:FullGitSha,
    $RequiredContractSha256,
    [string]$TrainingSeed,
    [string]$CompletedUpdates
  ) -LogPath $LogPath -Append -FailureMessage 'Checkpoint envelope extraction failed.' -FailureKind 'Protocol'
  Invoke-NativeLogged -Executable $script:PythonExe -Arguments @(
    '-m', $EvaluatorModule,
    'validate-checkpoint',
    '--envelope', $EnvelopePath,
    '--expected-git-sha', $script:FullGitSha,
    '--expected-contract-sha256', $RequiredContractSha256,
    '--expected-training-seed', [string]$TrainingSeed,
    '--verify-checkpoint-file',
    '--output', ($EnvelopePath + '.validation.json')
  ) -LogPath $LogPath -Append -FailureMessage 'Evaluator rejected checkpoint envelope.' -FailureKind 'Protocol'
}

function New-StairCampProgressReport {
  param(
    [Parameter(Mandatory = $true)][string]$CheckpointPath,
    [Parameter(Mandatory = $true)][string]$ReportPath,
    [Parameter(Mandatory = $true)][int]$TrainingSeed,
    [Parameter(Mandatory = $true)][int]$CompletedUpdates,
    [Parameter(Mandatory = $true)][int]$ExpectedEvaluations,
    [Parameter(Mandatory = $true)][string]$LogPath
  )
  if (-not (Test-Path -LiteralPath $CheckpointPath -PathType Leaf)) {
    Stop-Campaign -Kind 'Protocol' -Message ('Progress checkpoint is missing: {0}' -f $CheckpointPath)
  }
  $progressExtractor = @'
import hashlib
import json
import math
import pathlib
import sys

import torch

from hoppertrex_mjlab.scripts.rsl_rl.evaluate_stair_camp import (
    CheckpointExpectation,
    checkpoint_envelope_from_loaded_checkpoint,
    write_machine_output,
)

PROGRESS_KEYS = {
    "upper_height_m",
    "trigger_rate",
    "residual_abs_mean",
    "residual_rms",
    "residual_abs_max",
    "evaluations",
}
CURRICULUM_KEYS = {
    "schema_version",
    "lower_height_m",
    "upper_height_m",
    "consecutive_ready_evaluations",
    "evaluation_interval_steps",
    "next_evaluation_step",
    "episodes_at_upper",
    "successes_at_upper",
    "evaluations",
    "last_processed_step",
    "started",
    "triggered_episodes",
    "completed_episodes",
    "residual_abs_sum",
    "residual_sq_sum",
    "residual_sample_count",
    "residual_abs_max",
}
ARTIFACT_KEYS = {
    "controller_gain_hash",
    "calibration_hash",
    "yaw_calibration_hash",
    "posture_map_hash",
    "posture_artifact_hash",
    "station_calibration_hash",
}


def mapping(value, name):
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    return value


def exact_int(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def close(actual, expected, name):
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"{name} is inconsistent")


checkpoint_path = pathlib.Path(sys.argv[1]).resolve()
report_path = pathlib.Path(sys.argv[2]).resolve()
git_sha = sys.argv[3]
contract_sha256 = sys.argv[4]
training_seed = int(sys.argv[5])
completed_updates = int(sys.argv[6])
expected_evaluations = int(sys.argv[7])
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
if not isinstance(checkpoint, dict):
    raise TypeError("runner checkpoint must be a mapping")
expectation = CheckpointExpectation(
    git_sha=git_sha,
    contract_sha256=contract_sha256,
    training_seed=training_seed,
    completed_updates=completed_updates,
)
envelope = checkpoint_envelope_from_loaded_checkpoint(
    checkpoint_path,
    checkpoint,
    expectation=expectation,
)
infos = mapping(checkpoint.get("infos"), "checkpoint infos")
training = mapping(infos.get("stair_camp_training"), "stair_camp_training")
progress = mapping(infos.get("stair_camp_progress"), "stair_camp_progress")
curriculum = mapping(infos.get("stair_camp_curriculum"), "stair_camp_curriculum")
env_state = mapping(infos.get("env_state"), "env_state")
if set(progress) != PROGRESS_KEYS:
    raise ValueError("stair_camp_progress schema drifted")
if set(curriculum) != CURRICULUM_KEYS:
    raise ValueError("stair_camp_curriculum schema drifted")
if set(env_state) != {"common_step_counter"}:
    raise ValueError("env_state schema drifted")
artifacts = mapping(training.get("artifact_bindings"), "artifact_bindings")
if set(artifacts) != ARTIFACT_KEYS:
    raise ValueError("progress checkpoint artifact bindings drifted")
iteration = exact_int(checkpoint.get("iter"), "checkpoint iteration")
if iteration + 1 != completed_updates:
    raise ValueError("checkpoint iteration/completed update cadence drifted")
common_step = exact_int(env_state.get("common_step_counter"), "common_step_counter")
if common_step != completed_updates * 24:
    raise ValueError("common step does not equal completed_updates * 24")
evaluation_interval = exact_int(
    curriculum.get("evaluation_interval_steps"), "evaluation_interval_steps", 1
)
if evaluation_interval != 1200:
    raise ValueError("curriculum evaluation interval drifted")
evaluations = exact_int(curriculum.get("evaluations"), "curriculum evaluations")
progress_evaluations = finite(progress.get("evaluations"), "progress evaluations")
if not progress_evaluations.is_integer() or int(progress_evaluations) != evaluations:
    raise ValueError("progress and curriculum evaluations differ")
if evaluations != expected_evaluations or evaluations != common_step // 1200:
    raise ValueError("checkpoint evaluation cadence drifted")
next_evaluation_step = exact_int(
    curriculum.get("next_evaluation_step"), "next_evaluation_step", 1
)
last_processed_step = exact_int(
    curriculum.get("last_processed_step"), "last_processed_step"
)
if next_evaluation_step != (evaluations + 1) * 1200:
    raise ValueError("next evaluation step drifted")
if last_processed_step != common_step or common_step >= next_evaluation_step:
    raise ValueError("curriculum/common step state drifted")
upper_height = finite(progress.get("upper_height_m"), "progress upper height")
close(
    upper_height,
    finite(curriculum.get("upper_height_m"), "curriculum upper height"),
    "upper height",
)
if upper_height < 0.01 or upper_height > 0.15:
    raise ValueError("upper height is outside the registered grid")
if not math.isclose(upper_height / 0.01, round(upper_height / 0.01), abs_tol=1.0e-9):
    raise ValueError("upper height is not aligned to the registered grid")
triggered = exact_int(curriculum.get("triggered_episodes"), "triggered_episodes")
completed = exact_int(curriculum.get("completed_episodes"), "completed_episodes")
if triggered > completed:
    raise ValueError("triggered episodes exceed completed episodes")
trigger_rate = finite(progress.get("trigger_rate"), "trigger_rate")
close(trigger_rate, triggered / max(completed, 1), "trigger rate")
residual_count = exact_int(
    curriculum.get("residual_sample_count"), "residual_sample_count"
)
residual_abs_sum = finite(curriculum.get("residual_abs_sum"), "residual_abs_sum")
residual_sq_sum = finite(curriculum.get("residual_sq_sum"), "residual_sq_sum")
residual_abs_max = finite(curriculum.get("residual_abs_max"), "residual_abs_max")
if min(residual_abs_sum, residual_sq_sum, residual_abs_max) < 0.0:
    raise ValueError("residual accumulators must be nonnegative")
normalizer = max(residual_count, 1)
residual_abs_mean = finite(progress.get("residual_abs_mean"), "residual_abs_mean")
residual_rms = finite(progress.get("residual_rms"), "residual_rms")
progress_residual_max = finite(progress.get("residual_abs_max"), "progress residual_abs_max")
close(residual_abs_mean, residual_abs_sum / normalizer, "residual absolute mean")
close(residual_rms, math.sqrt(residual_sq_sum / normalizer), "residual RMS")
close(progress_residual_max, residual_abs_max, "residual maximum")
curriculum_json = json.dumps(
    curriculum, sort_keys=True, separators=(",", ":"), allow_nan=False
).encode("utf-8")
report = {
    "schema_version": 1,
    "kind": "stair_camp_checkpoint_progress",
    "task": training["task"],
    "training_seed": training_seed,
    "git_sha": git_sha,
    "contract_sha256": contract_sha256,
    "artifact_bindings": dict(artifacts),
    "checkpoint": str(checkpoint_path),
    "checkpoint_file_sha256": envelope["checkpoint_file_sha256"],
    "checkpoint_iteration": iteration,
    "completed_updates": completed_updates,
    "upper_height_m": upper_height,
    "trigger_rate": trigger_rate,
    "residual_abs_mean": residual_abs_mean,
    "residual_rms": residual_rms,
    "residual_abs_max": progress_residual_max,
    "evaluations": evaluations,
    "evaluation_interval_steps": evaluation_interval,
    "next_evaluation_step": next_evaluation_step,
    "common_step_counter": common_step,
    "curriculum_sha256": hashlib.sha256(curriculum_json).hexdigest(),
}
write_machine_output(report, report_path)
'@
  Invoke-NativeLogged -Executable $script:PythonExe -Arguments @(
    '-c', $progressExtractor,
    $CheckpointPath,
    $ReportPath,
    $script:FullGitSha,
    $RequiredContractSha256,
    [string]$TrainingSeed,
    [string]$CompletedUpdates,
    [string]$ExpectedEvaluations
  ) -LogPath $LogPath -Append -FailureMessage 'Checkpoint progress extraction failed.' -FailureKind 'Protocol'
}

function Assert-ExactPropertyNames {
  param(
    [Parameter(Mandatory = $true)][object]$Value,
    [Parameter(Mandatory = $true)][string[]]$ExpectedNames,
    [Parameter(Mandatory = $true)][string]$Context
  )
  $actualNames = @($Value.PSObject.Properties.Name)
  if (
    $actualNames.Count -ne $ExpectedNames.Count -or
    @($actualNames | Where-Object { $ExpectedNames -cnotcontains $_ }).Count -ne 0 -or
    @($ExpectedNames | Where-Object { $actualNames -cnotcontains $_ }).Count -ne 0
  ) {
    Stop-Campaign -Kind 'Protocol' -Message ('{0} schema drifted.' -f $Context)
  }
}

function Assert-ExactArtifactBindings {
  param(
    [Parameter(Mandatory = $true)][object]$Bindings,
    [Parameter(Mandatory = $true)][string]$Context
  )
  $bindingNames = @(
    'controller_gain_hash',
    'calibration_hash',
    'yaw_calibration_hash',
    'posture_map_hash',
    'posture_artifact_hash',
    'station_calibration_hash'
  )
  Assert-ExactPropertyNames -Value $Bindings -ExpectedNames $bindingNames -Context ($Context + ' artifact bindings')
  foreach ($name in $bindingNames) {
    if (
      [string]$Bindings.$name -notmatch '^[0-9a-f]{64}$' -or
      [string]$Bindings.$name -ne [string]$script:ExpectedArtifactBindings.$name
    ) {
      Stop-Campaign -Kind 'Protocol' -Message ('{0} drifted artifact binding {1}.' -f $Context, $name)
    }
  }
}

function Assert-StairCampProgressReport {
  param(
    [Parameter(Mandatory = $true)][object]$Report,
    [Parameter(Mandatory = $true)][int]$TrainingSeed,
    [Parameter(Mandatory = $true)][int]$CompletedUpdates,
    [Parameter(Mandatory = $true)][int]$ExpectedEvaluations
  )
  $reportFields = @(
    'schema_version',
    'kind',
    'task',
    'training_seed',
    'git_sha',
    'contract_sha256',
    'artifact_bindings',
    'checkpoint',
    'checkpoint_file_sha256',
    'checkpoint_iteration',
    'completed_updates',
    'upper_height_m',
    'trigger_rate',
    'residual_abs_mean',
    'residual_rms',
    'residual_abs_max',
    'evaluations',
    'evaluation_interval_steps',
    'next_evaluation_step',
    'common_step_counter',
    'curriculum_sha256'
  )
  Assert-ExactPropertyNames -Value $Report -ExpectedNames $reportFields -Context 'StairCamp progress report'
  if (
    [int]$Report.schema_version -ne 1 -or
    $Report.kind -ne 'stair_camp_checkpoint_progress' -or
    $Report.task -ne $Task -or
    [int]$Report.training_seed -ne $TrainingSeed -or
    $Report.git_sha -ne $script:FullGitSha -or
    $Report.contract_sha256 -ne $RequiredContractSha256 -or
    [string]$Report.checkpoint_file_sha256 -notmatch '^[0-9a-f]{64}$' -or
    [string]$Report.curriculum_sha256 -notmatch '^[0-9a-f]{64}$' -or
    [int]$Report.checkpoint_iteration + 1 -ne $CompletedUpdates -or
    [int]$Report.completed_updates -ne $CompletedUpdates -or
    [int]$Report.evaluations -ne $ExpectedEvaluations -or
    [int]$Report.evaluation_interval_steps -ne 1200 -or
    [int]$Report.next_evaluation_step -ne (($ExpectedEvaluations + 1) * 1200) -or
    [int]$Report.common_step_counter -ne ($CompletedUpdates * $RegisteredStepsPerIteration)
  ) {
    Stop-Campaign -Kind 'Protocol' -Message 'StairCamp progress report identity or cadence drifted.'
  }
  foreach ($name in @('upper_height_m', 'trigger_rate', 'residual_abs_mean', 'residual_rms', 'residual_abs_max')) {
    if (-not (Test-JsonNumber -Value $Report.$name)) {
      Stop-Campaign -Kind 'Protocol' -Message ('StairCamp progress report field {0} is not numeric.' -f $name)
    }
    $value = [double]$Report.$name
    if ([double]::IsNaN($value) -or [double]::IsInfinity($value)) {
      Stop-Campaign -Kind 'Protocol' -Message ('StairCamp progress report field {0} is non-finite.' -f $name)
    }
  }
  if (
    [double]$Report.upper_height_m -lt 0.01 -or
    [double]$Report.upper_height_m -gt 0.15 -or
    [double]$Report.trigger_rate -lt 0.0 -or
    [double]$Report.trigger_rate -gt 1.0 -or
    [double]$Report.residual_abs_mean -lt 0.0 -or
    [double]$Report.residual_rms -lt 0.0 -or
    [double]$Report.residual_abs_max -lt 0.0
  ) {
    Stop-Campaign -Kind 'Protocol' -Message 'StairCamp progress report metric ranges drifted.'
  }
  Assert-ExactArtifactBindings -Bindings $Report.artifact_bindings -Context 'StairCamp progress report'
  $checkpointPath = [System.IO.Path]::GetFullPath([string]$Report.checkpoint)
  if (
    -not (Test-Path -LiteralPath $checkpointPath -PathType Leaf) -or
    (Get-FileSha256 -Path $checkpointPath) -ne [string]$Report.checkpoint_file_sha256
  ) {
    Stop-Campaign -Kind 'Provenance' -Message 'StairCamp progress report checkpoint bytes drifted.'
  }
}

function Get-VerifiedStallPredicate {
  param([Parameter(Mandatory = $true)][int]$TrainingSeed)
  $freshRoot = Join-Path $script:CampaignRootPath ('seed{0}\fresh-1000' -f $TrainingSeed)
  $predicatePath = Join-Path $freshRoot 'stall_predicate.json'
  $model700Path = Join-Path $freshRoot 'model_700.progress.json'
  $model999Path = Join-Path $freshRoot 'model_999.progress.json'
  foreach ($path in @($predicatePath, $model700Path, $model999Path)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
      Stop-Campaign -Kind 'Protocol' -Message ('Fresh stall evidence is missing: {0}' -f $path)
    }
  }
  $model700 = Get-Content -LiteralPath $model700Path -Raw -Encoding UTF8 | ConvertFrom-Json
  $model999 = Get-Content -LiteralPath $model999Path -Raw -Encoding UTF8 | ConvertFrom-Json
  Assert-StairCampProgressReport -Report $model700 -TrainingSeed $TrainingSeed -CompletedUpdates 701 -ExpectedEvaluations 14
  Assert-StairCampProgressReport -Report $model999 -TrainingSeed $TrainingSeed -CompletedUpdates 1000 -ExpectedEvaluations 20
  $predicate = Get-Content -LiteralPath $predicatePath -Raw -Encoding UTF8 | ConvertFrom-Json
  $predicateFields = @(
    'schema_version',
    'kind',
    'task',
    'training_seed',
    'git_sha',
    'contract_sha256',
    'artifact_bindings',
    'model_700_progress_file',
    'model_700_progress_sha256',
    'model_999_progress_file',
    'model_999_progress_sha256',
    'start_completed_updates',
    'end_completed_updates',
    'start_evaluations',
    'end_evaluations',
    'evaluation_delta',
    'monotone_curriculum_bound',
    'first_stalled_evaluation',
    'last_stalled_evaluation',
    'start_upper_height_m',
    'end_upper_height_m',
    'upper_height_unchanged',
    'predicate_satisfied'
  )
  Assert-ExactPropertyNames -Value $predicate -ExpectedNames $predicateFields -Context 'StairCamp stall predicate'
  $upperUnchanged = [Math]::Abs([double]$model700.upper_height_m - [double]$model999.upper_height_m) -le 1.0e-12
  if (
    [int]$predicate.schema_version -ne 1 -or
    $predicate.kind -ne 'stair_camp_extension_stall_predicate' -or
    $predicate.task -ne $Task -or
    [int]$predicate.training_seed -ne $TrainingSeed -or
    $predicate.git_sha -ne $script:FullGitSha -or
    $predicate.contract_sha256 -ne $RequiredContractSha256 -or
    $predicate.model_700_progress_file -ne 'model_700.progress.json' -or
    $predicate.model_999_progress_file -ne 'model_999.progress.json' -or
    $predicate.model_700_progress_sha256 -ne (Get-FileSha256 -Path $model700Path) -or
    $predicate.model_999_progress_sha256 -ne (Get-FileSha256 -Path $model999Path) -or
    [int]$predicate.start_completed_updates -ne 701 -or
    [int]$predicate.end_completed_updates -ne 1000 -or
    [int]$predicate.start_evaluations -ne 14 -or
    [int]$predicate.end_evaluations -ne 20 -or
    [int]$predicate.evaluation_delta -ne 6 -or
    $predicate.monotone_curriculum_bound -ne $true -or
    [int]$predicate.first_stalled_evaluation -ne 15 -or
    [int]$predicate.last_stalled_evaluation -ne 20 -or
    [Math]::Abs([double]$predicate.start_upper_height_m - [double]$model700.upper_height_m) -gt 1.0e-12 -or
    [Math]::Abs([double]$predicate.end_upper_height_m - [double]$model999.upper_height_m) -gt 1.0e-12 -or
    $predicate.upper_height_unchanged -isnot [bool] -or
    $predicate.predicate_satisfied -isnot [bool] -or
    [bool]$predicate.upper_height_unchanged -ne $upperUnchanged -or
    [bool]$predicate.predicate_satisfied -ne $upperUnchanged
  ) {
    Stop-Campaign -Kind 'Protocol' -Message 'StairCamp six-evaluation stall predicate drifted.'
  }
  Assert-ExactArtifactBindings -Bindings $predicate.artifact_bindings -Context 'StairCamp stall predicate'
  return $predicate
}

function Get-VerifiedBudgetDecision {
  $decisionPath = Join-Path $script:CampaignRootPath 'budget_decision.json'
  if (-not (Test-Path -LiteralPath $decisionPath -PathType Leaf)) {
    Stop-Campaign -Kind 'Protocol' -Message 'Seeds 2 and 3 require the seed-1 campaign budget decision.'
  }
  $decision = Get-Content -LiteralPath $decisionPath -Raw -Encoding UTF8 | ConvertFrom-Json
  $decisionFields = @(
    'schema_version',
    'kind',
    'task',
    'campaign_id',
    'git_sha',
    'contract_sha256',
    'wrapper_canonical_sha256',
    'artifact_bindings',
    'decision',
    'source_budget_updates',
    'selected_total_budget_updates',
    'training_seeds',
    'decision_maker_training_seed',
    'evaluation_seed',
    'user_authorized',
    'seed1_stall_predicate_file',
    'seed1_stall_predicate_sha256',
    'seed1_final_checkpoint_sha256',
    'seed1_extension_checkpoint_sha256',
    'seed1_evaluation_delta'
  )
  Assert-ExactPropertyNames -Value $decision -ExpectedNames $decisionFields -Context 'StairCamp campaign budget decision'
  $manifest = Get-CampaignManifest
  $seedOnePredicatePath = Join-Path $script:CampaignRootPath 'seed1\fresh-1000\stall_predicate.json'
  $seedOneProgressPath = Join-Path $script:CampaignRootPath 'seed1\fresh-1000\model_999.progress.json'
  $seedOneExtensionManifestPath = Join-Path $script:CampaignRootPath 'seed1\extension-3000\training_manifest.json'
  if (-not (Test-Path -LiteralPath $seedOneExtensionManifestPath -PathType Leaf)) {
    Stop-Campaign -Kind 'Protocol' -Message 'Campaign budget decision requires completed seed-1 extension evidence.'
  }
  $seedOnePredicate = Get-VerifiedStallPredicate -TrainingSeed 1
  $seedOneProgress = Get-Content -LiteralPath $seedOneProgressPath -Raw -Encoding UTF8 | ConvertFrom-Json
  $seedOneExtensionManifest = Get-Content -LiteralPath $seedOneExtensionManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
  $seedOneExtensionCheckpoint = [string]$seedOneExtensionManifest.final_checkpoint
  if (
    [int]$decision.schema_version -ne 1 -or
    $decision.kind -ne 'stair_camp_campaign_budget_decision' -or
    $decision.task -ne $Task -or
    $decision.campaign_id -ne $manifest.campaign_id -or
    $decision.git_sha -ne $script:FullGitSha -or
    $decision.contract_sha256 -ne $RequiredContractSha256 -or
    $decision.wrapper_canonical_sha256 -ne $script:ActualSelfHash -or
    $decision.decision -ne 'EXTEND_ALL_SEEDS_TO_TOTAL_3000' -or
    [int]$decision.source_budget_updates -ne 1000 -or
    [int]$decision.selected_total_budget_updates -ne 3000 -or
    (@($decision.training_seeds) -join ',') -ne '1,2,3' -or
    [int]$decision.decision_maker_training_seed -ne 1 -or
    [int]$decision.evaluation_seed -ne $RegisteredEvaluationSeed -or
    $decision.user_authorized -ne $true -or
    $decision.seed1_stall_predicate_file -ne 'seed1/fresh-1000/stall_predicate.json' -or
    $decision.seed1_stall_predicate_sha256 -ne (Get-FileSha256 -Path $seedOnePredicatePath) -or
    $decision.seed1_final_checkpoint_sha256 -ne [string]$seedOneProgress.checkpoint_file_sha256 -or
    [int]$seedOneExtensionManifest.training_seed -ne 1 -or
    [int]$seedOneExtensionManifest.completed_updates -ne 3000 -or
    -not (Test-Path -LiteralPath $seedOneExtensionCheckpoint -PathType Leaf) -or
    $decision.seed1_extension_checkpoint_sha256 -ne [string]$seedOneExtensionManifest.final_checkpoint_sha256 -or
    $decision.seed1_extension_checkpoint_sha256 -ne (Get-FileSha256 -Path $seedOneExtensionCheckpoint) -or
    $seedOneExtensionManifest.budget_decision_sha256 -ne (Get-FileSha256 -Path $decisionPath) -or
    [int]$decision.seed1_evaluation_delta -ne 6 -or
    $seedOnePredicate.predicate_satisfied -ne $true
  ) {
    Stop-Campaign -Kind 'Protocol' -Message 'Campaign-level 3000-update budget decision drifted.'
  }
  Assert-ExactArtifactBindings -Bindings $decision.artifact_bindings -Context 'StairCamp campaign budget decision'
  return $decision
}

function Invoke-K3Screen {
  param(
    [Parameter(Mandatory = $true)][string]$CheckpointEnvelopePath,
    [Parameter(Mandatory = $true)][string]$CandidatePath,
    [Parameter(Mandatory = $true)][string]$RawCollectionPath,
    [Parameter(Mandatory = $true)][int]$PoolBudget,
    [Parameter(Mandatory = $true)][string]$LogPath
  )
  # Four registered smoke-profile gate screens are rejection-only. The height
  # screen separately executes the exact K3 protocol: 0.01 m, 16 envs, once.
  $collector = @'
import importlib
import json
import pathlib
import sys
from hoppertrex_mjlab.scripts.rsl_rl.evaluate_stair_camp import (
    K3_SCREEN_PROTOCOL,
    finalize_adapter_output,
    load_live_adapter,
    make_adapter_config,
    make_k3_screen_candidate,
    validate_stair_camp_checkpoint_envelope,
    write_machine_output,
)
envelope_path = pathlib.Path(sys.argv[1]).resolve()
candidate_path = pathlib.Path(sys.argv[2]).resolve()
raw_path = pathlib.Path(sys.argv[3]).resolve()
budget = int(sys.argv[4])
device = sys.argv[5]
adapter_spec = sys.argv[6]
checkpoint = json.loads(envelope_path.read_text(encoding="utf-8"))
checkpoint = validate_stair_camp_checkpoint_envelope(checkpoint, verify_file=True)
adapter = load_live_adapter(adapter_spec)
flat_config = make_adapter_config(
    domain="flat",
    checkpoint_envelope=checkpoint,
    profile="smoke",
    ablation="baseline",
    device=device,
    verify_checkpoint_file=True,
)
flat_collection = adapter(flat_config)
flat_result = finalize_adapter_output(flat_config, flat_collection)
live_module = importlib.import_module(adapter.__module__)
dependencies_loader = getattr(live_module, "_load_live_dependencies", None)
backend_type = getattr(live_module, "_MjLabBackend", None)
scan_type = getattr(live_module, "ScanRequest", None)
aggregate = getattr(live_module, "aggregate_scan_trials", None)
if not all(callable(value) for value in (
    dependencies_loader,
    backend_type,
    scan_type,
    aggregate,
)):
    raise RuntimeError("Default live adapter lacks exact K=3 scan integration")
stairs_config = make_adapter_config(
    domain="stairs",
    checkpoint_envelope=checkpoint,
    profile="formal",
    ablation="baseline",
    device=device,
    verify_checkpoint_file=True,
)
protocol = K3_SCREEN_PROTOCOL
request = scan_type(
    domain="stairs",
    profile="screen",
    terrain=protocol.terrain,
    cell_key=str(protocol.cell_key),
    cells=tuple(protocol.cells),
    num_envs_per_cell=int(protocol.num_envs_per_cell),
    repeats=int(protocol.repeats),
    settle_steps=int(protocol.settle_steps),
    drive_steps=int(protocol.drive_steps),
    stable_steps=int(protocol.stable_steps),
    travel_distance_m=float(protocol.travel_distance_m),
)
backend = backend_type(stairs_config, dependencies_loader())
height_rows = aggregate(request, backend.run_scan(request))
if len(height_rows) != 1:
    raise RuntimeError("Exact K=3 height screen did not return one row")
gate_false_positives = {
    str(row["name"]): int(row["stair_mode_false_positives"])
    for row in flat_result["gates"]
}
raw = {
    "schema_version": 1,
    "kind": "stair_camp_k3_live_collection",
    "gate_profile": "smoke",
    "height_profile": "screen",
    "gate_result": flat_result,
    "height_row": height_rows[0],
}
write_machine_output(raw, raw_path)
candidate = make_k3_screen_candidate(
    checkpoint_envelope=checkpoint,
    budget_updates=budget,
    gate_passes=flat_result["gate_booleans"],
    gate_stair_mode_false_positives=gate_false_positives,
    height_row=height_rows[0],
)
write_machine_output(candidate, candidate_path)
'@
  Invoke-NativeLogged -Executable $script:PythonExe -Arguments @(
    '-c', $collector,
    $CheckpointEnvelopePath,
    $CandidatePath,
    $RawCollectionPath,
    [string]$PoolBudget,
    $RequiredDevice,
    $LiveAdapter
  ) -LogPath $LogPath -Append -FailureMessage 'Exact K=3 live screen failed.' -FailureKind 'Protocol'
}

function Invoke-FormalEvaluation {
  param(
    [Parameter(Mandatory = $true)][string]$Domain,
    [Parameter(Mandatory = $true)][string]$Ablation,
    [Parameter(Mandatory = $true)][string]$CheckpointEnvelopePath,
    [Parameter(Mandatory = $true)][string]$OutputPath,
    [Parameter(Mandatory = $true)][int]$TrainingSeed,
    [Parameter(Mandatory = $true)][string]$LogPath
  )
  Invoke-NativeLogged -Executable $script:PythonExe -Arguments @(
    '-m', $EvaluatorModule,
    'live',
    '--domain', $Domain,
    '--profile', 'formal',
    '--checkpoint-envelope', $CheckpointEnvelopePath,
    '--ablation', $Ablation,
    '--device', $RequiredDevice,
    '--expected-git-sha', $script:FullGitSha,
    '--expected-contract-sha256', $RequiredContractSha256,
    '--expected-training-seed', [string]$TrainingSeed,
    '--verify-checkpoint-file',
    '--adapter', $LiveAdapter,
    '--output', $OutputPath
  ) -LogPath $LogPath -Append -FailureMessage ('Formal evaluator failed for {0}/{1}.' -f $Domain, $Ablation) -FailureKind 'Protocol'
}

function Assert-EvaluationEnvelope {
  param(
    [Parameter(Mandatory = $true)][object]$Payload,
    [Parameter(Mandatory = $true)][string]$Domain,
    [Parameter(Mandatory = $true)][string]$Ablation
  )
  if (
    [int]$Payload.schema_version -ne 1 -or
    $Payload.kind -ne 'stair_camp_evaluation' -or
    $Payload.status -ne 'complete' -or
    $Payload.task -ne $Task -or
    $Payload.domain -ne $Domain -or
    $Payload.profile -ne 'formal' -or
    [int]$Payload.evaluation_seed -ne $RegisteredEvaluationSeed -or
    $Payload.evidence_eligible -ne $true -or
    $Payload.ablation.name -ne $Ablation -or
    $Payload.checkpoint.training.git_sha -ne $script:FullGitSha -or
    $Payload.checkpoint.training.contract_sha256 -ne $RequiredContractSha256
  ) {
    Stop-Campaign -Kind 'Protocol' -Message ('Formal evaluator envelope drifted for {0}/{1}.' -f $Domain, $Ablation)
  }
  if ($Domain -eq 'stairs') {
    if (@($Payload.rows).Count -ne 7) {
      Stop-Campaign -Kind 'Protocol' -Message 'Formal stairs evaluation omitted registered height rows.'
    }
    $expectedPromotion = $Ablation -eq 'baseline'
    if ([bool]$Payload.promotion_evidence_eligible -ne $expectedPromotion) {
      Stop-Campaign -Kind 'Protocol' -Message 'Stairs promotion-evidence eligibility drifted.'
    }
  } elseif ($Domain -eq 'flat') {
    if (@($Payload.gates).Count -ne 4) {
      Stop-Campaign -Kind 'Protocol' -Message 'Formal flat evaluation omitted registered gates.'
    }
  } elseif (
    $Payload.secondary_metric_only -ne $true -or
    $null -ne $Payload.registered_pass_threshold
  ) {
    Stop-Campaign -Kind 'Protocol' -Message 'Slope result is not marked secondary-only.'
  }
}

function Get-CampaignManifest {
  $manifestPath = Join-Path $script:CampaignRootPath 'campaign_manifest.json'
  if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    Stop-Campaign -Kind 'Provenance' -Message 'Campaign preflight manifest is missing.'
  }
  $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
  $manifestFields = @($manifest.PSObject.Properties.Name)
  foreach ($requiredField in @(
    'classical_probe',
    'classical_probe_sha256',
    'classical_rows_source',
    'classical_rows_source_sha256',
    'classical_rows',
    'classical_rows_file',
    'classical_rows_file_sha256',
    'pretraining_trigger_request_file',
    'pretraining_trigger_request_sha256',
    'pretraining_policy',
    'camp_flat_rolling_fp_file',
    'camp_flat_rolling_fp_sha256',
    'stage5_kick_fp_file',
    'stage5_kick_fp_sha256'
  )) {
    if ($manifestFields -cnotcontains $requiredField) {
      Stop-Campaign -Kind 'Provenance' -Message ('Campaign manifest omitted {0}.' -f $requiredField)
    }
  }
  if (
    [int]$manifest.schema_version -ne 1 -or
    $manifest.kind -ne 'stair_camp_s5b_campaign' -or
    $manifest.task -ne $Task -or
    $manifest.git_sha -ne $script:FullGitSha -or
    $manifest.mjlab_git_sha -ne $RequiredMjLabSha -or
    $manifest.contract_sha256 -ne $RequiredContractSha256 -or
    $manifest.wrapper_canonical_sha256 -ne $script:ActualSelfHash -or
    $manifest.classical_probe_sha256 -ne $RequiredClassicalProbeSha256 -or
    [string]$manifest.classical_rows_source_sha256 -notmatch '^[0-9a-f]{64}$' -or
    $manifest.classical_rows_file_sha256 -ne $manifest.classical_rows_source_sha256 -or
    $manifest.classical_rows_file -ne 'classical_rows.json' -or
    $manifest.pretraining_policy -ne 'deterministic_zero_residual'
  ) {
    Stop-Campaign -Kind 'Provenance' -Message 'Campaign manifest no longer matches this audited checkout.'
  }
  $archivedClassicalRowsPath = Join-Path $script:CampaignRootPath ([string]$manifest.classical_rows_file)
  if (
    -not (Test-Path -LiteralPath $archivedClassicalRowsPath -PathType Leaf) -or
    (Get-FileSha256 -Path $archivedClassicalRowsPath) -ne $manifest.classical_rows_file_sha256
  ) {
    Stop-Campaign -Kind 'Provenance' -Message 'Archived classical rows source hash drifted.'
  }
  $probePath = [string]$manifest.classical_probe
  if (
    -not (Test-Path -LiteralPath $probePath -PathType Leaf) -or
    (Get-FileSha256 -Path $probePath) -ne $RequiredClassicalProbeSha256
  ) {
    Stop-Campaign -Kind 'Provenance' -Message 'Frozen classical probe is missing or changed.'
  }
  $archivedClassicalRows = @(Read-ClassicalRowsSource -Path $archivedClassicalRowsPath)
  $manifestClassicalRows = @($manifest.classical_rows)
  Assert-ClassicalRowsEquivalent -Expected $manifestClassicalRows -Actual $archivedClassicalRows
  Assert-ClassicalRowsMatchFrozenProbe -Rows $archivedClassicalRows -ProbePath $probePath
  $pretrainingRequestPath = Join-Path $script:CampaignRootPath ([string]$manifest.pretraining_trigger_request_file)
  $flatFpPath = Join-Path $script:CampaignRootPath ([string]$manifest.camp_flat_rolling_fp_file)
  $kickFpPath = Join-Path $script:CampaignRootPath ([string]$manifest.stage5_kick_fp_file)
  foreach ($entry in @(
    @{ path = $pretrainingRequestPath; sha256 = [string]$manifest.pretraining_trigger_request_sha256 },
    @{ path = $flatFpPath; sha256 = [string]$manifest.camp_flat_rolling_fp_sha256 },
    @{ path = $kickFpPath; sha256 = [string]$manifest.stage5_kick_fp_sha256 }
  )) {
    if (
      [string]$entry.sha256 -notmatch '^[0-9a-f]{64}$' -or
      -not (Test-Path -LiteralPath $entry.path -PathType Leaf) -or
      (Get-FileSha256 -Path $entry.path) -ne [string]$entry.sha256
    ) {
      Stop-Campaign -Kind 'Provenance' -Message 'Archived live trigger preflight evidence hash drifted.'
    }
  }
  $pretrainingRequest = Get-Content -LiteralPath $pretrainingRequestPath -Raw -Encoding UTF8 | ConvertFrom-Json
  $pretrainingRequestFields = @(
    'schema_version',
    'kind',
    'task',
    'evaluation_seed',
    'device',
    'git_sha',
    'contract_sha256',
    'artifact_bindings'
  )
  Assert-ExactPropertyNames -Value $pretrainingRequest -ExpectedNames $pretrainingRequestFields -Context 'Pretraining trigger request'
  if (
    [int]$pretrainingRequest.schema_version -ne 1 -or
    $pretrainingRequest.kind -ne 'stair_camp_trigger_pretraining_request' -or
    $pretrainingRequest.task -ne $Task -or
    [int]$pretrainingRequest.evaluation_seed -ne $RegisteredEvaluationSeed -or
    $pretrainingRequest.device -ne $RequiredDevice -or
    $pretrainingRequest.git_sha -ne $script:FullGitSha -or
    $pretrainingRequest.contract_sha256 -ne $RequiredContractSha256
  ) {
    Stop-Campaign -Kind 'Provenance' -Message 'Pretraining trigger request identity drifted.'
  }
  Assert-ExactArtifactBindings -Bindings $pretrainingRequest.artifact_bindings -Context 'Pretraining trigger request'
  $flatFp = Get-Content -LiteralPath $flatFpPath -Raw -Encoding UTF8 | ConvertFrom-Json
  $kickFp = Get-Content -LiteralPath $kickFpPath -Raw -Encoding UTF8 | ConvertFrom-Json
  Assert-LiveTriggerFalsePositivePayload -Payload $flatFp -Domain 'camp_flat_rolling' -ExpectedEvents 96000
  Assert-LiveTriggerFalsePositivePayload -Payload $kickFp -Domain 'stage5_kick' -ExpectedEvents 128
  if ($manifest.preflight_classification -ne 'STAIR_CAMP_PREFLIGHT_PASS' -or $manifest.training_authorized -ne $true) {
    Stop-Campaign -Kind 'Protocol' -Message 'Training phases are forbidden after a scientific preflight STOP.'
  }
  return $manifest
}

function Assert-PreflightPayload {
  param([Parameter(Mandatory = $true)][object]$Payload)
  if (
    [int]$Payload.schema_version -ne 1 -or
    $Payload.kind -ne 'stair_camp_training_preflight' -or
    @('STAIR_CAMP_PREFLIGHT_PASS', 'STOP_NO_PROMOTION') -notcontains $Payload.classification -or
    [int]$Payload.trigger_replay.completed_pairs -ne 288 -or
    [int]$Payload.trigger_replay.detections -ne 288 -or
    [int]$Payload.trigger_replay.pre_impact_triggers -ne 0 -or
    $Payload.trigger_replay.files_unchanged -ne $true -or
    [string]$Payload.trigger_replay.sha256s_file_sha256 -notmatch '^[0-9a-f]{64}$' -or
    @($Payload.false_positive_checks).Count -ne 2
  ) {
    Stop-Campaign -Kind 'Protocol' -Message 'StairCamp preflight replay identity or completeness drifted.'
  }
  $domains = @($Payload.false_positive_checks | ForEach-Object { [string]$_.domain })
  if (($domains -join ',') -ne 'camp_flat_rolling,stage5_kick') {
    Stop-Campaign -Kind 'Protocol' -Message 'StairCamp preflight FP domains or order drifted.'
  }
  foreach ($check in @($Payload.false_positive_checks)) {
    if (
      [int]$check.events -lt 1 -or
      [int]$check.stair_mode_false_positives -lt 0 -or
      [int]$check.stair_mode_false_positives -gt [int]$check.events
    ) {
      Stop-Campaign -Kind 'Protocol' -Message 'StairCamp preflight FP evidence has invalid counts.'
    }
  }
  $allFpZero = @($Payload.false_positive_checks | Where-Object {
    [int]$_.stair_mode_false_positives -ne 0
  }).Count -eq 0
  if ($Payload.classification -eq 'STAIR_CAMP_PREFLIGHT_PASS') {
    if ($Payload.training_authorized -ne $true -or -not $allFpZero) {
      Stop-Campaign -Kind 'Protocol' -Message 'Passing preflight does not authorize exactly zero-FP training.'
    }
  } elseif ($Payload.training_authorized -ne $false -or $allFpZero) {
    Stop-Campaign -Kind 'Protocol' -Message 'Scientific preflight STOP has inconsistent authorization.'
  }
}

function Assert-LiveTriggerFalsePositivePayload {
  param(
    [Parameter(Mandatory = $true)][object]$Payload,
    [Parameter(Mandatory = $true)][string]$Domain,
    [Parameter(Mandatory = $true)][int]$ExpectedEvents
  )
  $fields = @(
    'schema_version',
    'kind',
    'domain',
    'threshold_n',
    'window_steps',
    'events',
    'stair_mode_false_positives',
    'completed'
  )
  Assert-ExactPropertyNames -Value $Payload -ExpectedNames $fields -Context ('Live trigger FP ' + $Domain)
  if (
    [int]$Payload.schema_version -ne 1 -or
    $Payload.kind -ne 'stair_camp_trigger_false_positive_check' -or
    $Payload.domain -ne $Domain -or
    [double]$Payload.threshold_n -ne 18.0 -or
    [int]$Payload.window_steps -ne 3 -or
    [int]$Payload.events -ne $ExpectedEvents -or
    [int]$Payload.stair_mode_false_positives -lt 0 -or
    [int]$Payload.stair_mode_false_positives -gt $ExpectedEvents -or
    $Payload.completed -ne $true
  ) {
    Stop-Campaign -Kind 'Protocol' -Message ('Live trigger FP payload drifted for {0}.' -f $Domain)
  }
}

function Assert-ComposedSeedEnvelope {
  param(
    [Parameter(Mandatory = $true)][object]$Envelope,
    [Parameter(Mandatory = $true)][int]$TrainingSeed,
    [Parameter(Mandatory = $true)][int]$PoolBudget
  )
  $requiredAblations = @(
    'leg-off',
    'zero-shot-scale-0.035',
    'zero-shot-scale-0.070',
    'zero-shot-scale-0.100',
    'mode-always-on'
  )
  $bindingNames = @(
    'controller_gain_hash',
    'calibration_hash',
    'yaw_calibration_hash',
    'posture_map_hash',
    'posture_artifact_hash',
    'station_calibration_hash'
  )
  if (
    [int]$Envelope.training_seed -ne $TrainingSeed -or
    [int]$Envelope.evaluation_seed -ne $RegisteredEvaluationSeed -or
    [int]$Envelope.budget_iterations -ne $PoolBudget -or
    $Envelope.git_sha -ne $script:FullGitSha -or
    $Envelope.contract_hash -ne $RequiredContractSha256 -or
    $Envelope.evidence_eligible -ne $true -or
    $Envelope.ablations_complete -ne $true -or
    @($Envelope.completed_ablations).Count -ne 5 -or
    @($Envelope.artifact_bindings.PSObject.Properties.Name).Count -ne 6 -or
    [string]$Envelope.checkpoint_file_sha256 -notmatch '^[0-9a-f]{64}$' -or
    [string]::IsNullOrWhiteSpace([string]$Envelope.checkpoint)
  ) {
    Stop-Campaign -Kind 'Protocol' -Message 'Evaluator compose-seed emitted an incomplete adjudicator envelope.'
  }
  for ($index = 0; $index -lt $requiredAblations.Count; $index += 1) {
    if ([string]$Envelope.completed_ablations[$index] -ne $requiredAblations[$index]) {
      Stop-Campaign -Kind 'Protocol' -Message 'Evaluator compose-seed ablation order drifted.'
    }
  }
  $actualBindingNames = @($Envelope.artifact_bindings.PSObject.Properties.Name)
  if (
    $actualBindingNames.Count -ne $bindingNames.Count -or
    @($actualBindingNames | Where-Object { $bindingNames -cnotcontains $_ }).Count -ne 0 -or
    @($bindingNames | Where-Object { $actualBindingNames -cnotcontains $_ }).Count -ne 0
  ) {
    Stop-Campaign -Kind 'Protocol' -Message 'Evaluator compose-seed artifact bindings are not exactly the six frozen names.'
  }
  foreach ($name in $bindingNames) {
    $actualBinding = [string]$Envelope.artifact_bindings.$name
    $expectedBinding = [string]$script:ExpectedArtifactBindings.$name
    if ($actualBinding -notmatch '^[0-9a-f]{64}$' -or $actualBinding -ne $expectedBinding) {
      Stop-Campaign -Kind 'Protocol' -Message ('Evaluator compose-seed drifted binding {0}.' -f $name)
    }
  }
  $gateNames = @('flat_gate_passed', 'standing_gate_passed', 'velocity_gate_passed', 'stage5_gate_passed')
  $falsePositiveNames = @($Envelope.gate_stair_mode_false_positives.PSObject.Properties.Name)
  if (
    $falsePositiveNames.Count -ne $gateNames.Count -or
    @($falsePositiveNames | Where-Object { $gateNames -cnotcontains $_ }).Count -ne 0 -or
    @($gateNames | Where-Object { $falsePositiveNames -cnotcontains $_ }).Count -ne 0
  ) {
    Stop-Campaign -Kind 'Protocol' -Message 'Evaluator compose-seed FP counts are not exactly the four registered gate names.'
  }
  foreach ($gate in $gateNames) {
    $gateValue = $Envelope.$gate
    $falsePositiveCount = $Envelope.gate_stair_mode_false_positives.$gate
    if (
      $gateValue -isnot [bool] -or
      $falsePositiveCount -isnot [int] -and $falsePositiveCount -isnot [long] -or
      [long]$falsePositiveCount -lt 0 -or
      ([bool]$gateValue -and [long]$falsePositiveCount -ne 0)
    ) {
      Stop-Campaign -Kind 'Protocol' -Message ('Evaluator compose-seed emitted invalid FP evidence for {0}.' -f $gate)
    }
  }
  $checkpointPath = [System.IO.Path]::GetFullPath([string]$Envelope.checkpoint)
  if (
    -not (Test-Path -LiteralPath $checkpointPath -PathType Leaf) -or
    (Get-FileSha256 -Path $checkpointPath) -ne [string]$Envelope.checkpoint_file_sha256
  ) {
    Stop-Campaign -Kind 'Protocol' -Message 'Evaluator compose-seed checkpoint bytes do not match its declared hash.'
  }
}

function Invoke-ValidatePhase {
  foreach ($required in @(
    @{ name = 'C2ReplayDirectory'; value = $C2ReplayDirectory },
    @{ name = 'ClassicalProbePath'; value = $ClassicalProbePath },
    @{ name = 'ClassicalRowsPath'; value = $ClassicalRowsPath }
  )) {
    if ([string]::IsNullOrWhiteSpace([string]$required.value)) {
      Stop-Campaign -Kind 'Protocol' -Message ('Validate requires -{0}.' -f $required.name)
    }
  }
  if (Test-Path -LiteralPath $script:CampaignRootPath) {
    Stop-Campaign -Kind 'Operational' -Message ('Refusing to overwrite campaign root: {0}' -f $script:CampaignRootPath)
  }
  if (-not (Test-Path -LiteralPath $C2ReplayDirectory -PathType Container)) {
    Stop-Campaign -Kind 'Provenance' -Message 'C2 replay directory is missing.'
  }
  foreach ($inputFile in @(
    $ClassicalProbePath,
    $ClassicalRowsPath
  )) {
    if (-not (Test-Path -LiteralPath $inputFile -PathType Leaf)) {
      Stop-Campaign -Kind 'Provenance' -Message ('Required preflight input is missing: {0}' -f $inputFile)
    }
  }
  $c2Root = (Resolve-Path -LiteralPath $C2ReplayDirectory).Path
  $classicalProbe = (Resolve-Path -LiteralPath $ClassicalProbePath).Path
  $classicalRowsSource = (Resolve-Path -LiteralPath $ClassicalRowsPath).Path
  if ((Get-FileSha256 -Path $classicalProbe) -ne $RequiredClassicalProbeSha256) {
    Stop-Campaign -Kind 'Provenance' -Message 'Frozen classical probe byte hash drifted.'
  }
  $classicalRowsSourceSha256 = Get-FileSha256 -Path $classicalRowsSource
  $classicalRows = @(Read-ClassicalRowsSource -Path $classicalRowsSource)
  Assert-ClassicalRowsMatchFrozenProbe -Rows $classicalRows -ProbePath $classicalProbe

  $working = $script:CampaignRootPath + '.incomplete.' + [System.Guid]::NewGuid().ToString('N')
  New-Item -ItemType Directory -Path $working | Out-Null
  $logPath = Join-Path $working 'preflight.log'
  try {
    Invoke-NativeLogged -Executable $script:PythonExe -Arguments @(
      '-m', $EvaluatorModule,
      'manifest',
      '--profile', 'formal',
      '--output', (Join-Path $working 'evaluator_manifest.json')
    ) -LogPath $logPath -FailureMessage 'Evaluator manifest generation failed.' -FailureKind 'Protocol'

    $c2ReplayPath = Join-Path $working 'c2_trigger_replay.json'
    Invoke-NativeLogged -Executable $script:PythonExe -Arguments @(
      '-m', $PreflightModule,
      'replay-c2',
      '--input-dir', $c2Root,
      '--output', $c2ReplayPath
    ) -LogPath $logPath -Append -FailureMessage 'Read-only C2 trigger replay failed.' -FailureKind 'Protocol'

    $pretrainingRequest = [ordered]@{
      schema_version = 1
      kind = 'stair_camp_trigger_pretraining_request'
      task = $Task
      evaluation_seed = $RegisteredEvaluationSeed
      device = $RequiredDevice
      git_sha = $script:FullGitSha
      contract_sha256 = $RequiredContractSha256
      artifact_bindings = $script:ExpectedArtifactBindings
    }
    $pretrainingRequestPath = Join-Path $working 'pretraining_trigger_request.json'
    Write-AtomicJsonNoClobber -Path $pretrainingRequestPath -Value $pretrainingRequest
    $flatFpPath = Join-Path $working 'camp_flat_rolling_fp.json'
    $kickFpPath = Join-Path $working 'stage5_kick_fp.json'
    Invoke-NativeLogged -Executable $script:PythonExe -Arguments @(
      '-m', $LiveAdapterModule,
      'trigger-fp',
      '--domain', 'camp_flat_rolling',
      '--request', $pretrainingRequestPath,
      '--output', $flatFpPath
    ) -LogPath $logPath -Append -FailureMessage 'Live camp-flat trigger FP collection failed.' -FailureKind 'Protocol'
    Invoke-NativeLogged -Executable $script:PythonExe -Arguments @(
      '-m', $LiveAdapterModule,
      'trigger-fp',
      '--domain', 'stage5_kick',
      '--request', $pretrainingRequestPath,
      '--output', $kickFpPath
    ) -LogPath $logPath -Append -FailureMessage 'Live Stage5-kick trigger FP collection failed.' -FailureKind 'Protocol'
    $flatFp = Get-Content -LiteralPath $flatFpPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $kickFp = Get-Content -LiteralPath $kickFpPath -Raw -Encoding UTF8 | ConvertFrom-Json
    Assert-LiveTriggerFalsePositivePayload -Payload $flatFp -Domain 'camp_flat_rolling' -ExpectedEvents 96000
    Assert-LiveTriggerFalsePositivePayload -Payload $kickFp -Domain 'stage5_kick' -ExpectedEvents 128
    $preflightPath = Join-Path $working 'preflight.json'
    Invoke-NativeLogged -Executable $script:PythonExe -Arguments @(
      '-m', $PreflightModule,
      'finalize',
      '--c2-replay', $c2ReplayPath,
      '--flat-fp', $flatFpPath,
      '--stage5-kick-fp', $kickFpPath,
      '--output', $preflightPath
    ) -LogPath $logPath -Append -FailureMessage 'StairCamp preflight finalization failed.' -FailureKind 'Protocol'
    if (-not (Test-Path -LiteralPath $preflightPath -PathType Leaf)) {
      Stop-Campaign -Kind 'Protocol' -Message 'Preflight entrypoint omitted preflight.json.'
    }
    $preflight = Get-Content -LiteralPath $preflightPath -Raw -Encoding UTF8 | ConvertFrom-Json
    Assert-PreflightPayload -Payload $preflight
    $archivedClassicalRowsPath = Join-Path $working 'classical_rows.json'
    Copy-Item -LiteralPath $classicalRowsSource -Destination $archivedClassicalRowsPath
    if ((Get-FileSha256 -Path $archivedClassicalRowsPath) -ne $classicalRowsSourceSha256) {
      Stop-Campaign -Kind 'Provenance' -Message 'Archived classical rows source changed during validation.'
    }
    $campaignId = 's5b_' + $script:FullGitSha.Substring(0, 12) + '_' + [System.Guid]::NewGuid().ToString('N')
    $manifest = [ordered]@{
      schema_version = 1
      kind = 'stair_camp_s5b_campaign'
      campaign_id = $campaignId
      task = $Task
      git_sha = $script:FullGitSha
      branch = $RequiredBranch
      mjlab_git_sha = $RequiredMjLabSha
      contract_sha256 = $RequiredContractSha256
      wrapper_canonical_sha256 = $script:ActualSelfHash
      evaluation_seed = $RegisteredEvaluationSeed
      training_seeds = $RegisteredTrainingSeeds
      registered_defaults = [ordered]@{
        num_envs = $RegisteredNumEnvs
        max_iterations = $RegisteredFreshUpdates
        save_interval = $RegisteredSaveInterval
        num_steps_per_env = $RegisteredStepsPerIteration
      }
      preflight_classification = [string]$preflight.classification
      training_authorized = [bool]$preflight.training_authorized
      c2_replay_directory = $c2Root
      classical_probe = $classicalProbe
      classical_probe_sha256 = $RequiredClassicalProbeSha256
      classical_rows_source = $classicalRowsSource
      classical_rows_source_sha256 = $classicalRowsSourceSha256
      classical_rows = $classicalRows
      classical_rows_file = 'classical_rows.json'
      classical_rows_file_sha256 = $classicalRowsSourceSha256
      pretraining_trigger_request_file = 'pretraining_trigger_request.json'
      pretraining_trigger_request_sha256 = Get-FileSha256 -Path $pretrainingRequestPath
      pretraining_policy = 'deterministic_zero_residual'
      camp_flat_rolling_fp_file = 'camp_flat_rolling_fp.json'
      camp_flat_rolling_fp_sha256 = Get-FileSha256 -Path $flatFpPath
      stage5_kick_fp_file = 'stage5_kick_fp.json'
      stage5_kick_fp_sha256 = Get-FileSha256 -Path $kickFpPath
      frozen_artifact_files = $script:FrozenArtifactManifest
      artifact_bindings = $script:ExpectedArtifactBindings
      preflight_sha256 = Get-FileSha256 -Path $preflightPath
      live_adapter = $LiveAdapter
      exit_semantics = [ordered]@{
        scientific_stop = $ExitCodeScientificStop
        provenance_error = $ExitCodeProvenance
        protocol_error = $ExitCodeProtocol
        operational_error = $ExitCodeOperational
      }
    }
    Write-AtomicJsonNoClobber -Path (Join-Path $working 'campaign_manifest.json') -Value $manifest
    Publish-PhaseWorkspace -WorkingPath $working -FinalPath $script:CampaignRootPath
  } catch {
    Write-Warning ('Incomplete preflight retained: {0}' -f $working)
    throw
  }
  if ($preflight.classification -eq 'STOP_NO_PROMOTION') {
    Write-Host '[COMPLETE] Trigger preflight produced a legal scientific STOP.'
  } else {
    Write-Host '[PASS] StairCamp validation/preflight completed.'
  }
  Write-Host ('CLASSIFICATION={0}' -f $preflight.classification)
  Write-Host ('RESULT={0}' -f $script:CampaignRootPath)
}

function Invoke-Fresh1000Phase {
  Assert-SeedPhase
  if ($Budget -ne $RegisteredFreshUpdates) {
    Stop-Campaign -Kind 'Protocol' -Message 'Fresh1000 requires -Budget 1000.'
  }
  $manifest = Get-CampaignManifest
  $final = Join-Path $script:CampaignRootPath ('seed{0}\fresh-1000' -f $Seed)
  $working = New-PhaseWorkspace -FinalPath $final
  $logPath = Join-Path $working 'training.log'
  $runName = '{0}_seed{1}_fresh_total1000' -f $manifest.campaign_id, $Seed
  $existingRuns = @(Get-ChildItem -LiteralPath $script:TrainingLogRoot -Directory -ErrorAction SilentlyContinue | Where-Object {
    $_.Name -like ('*_{0}' -f $runName)
  })
  if ($existingRuns.Count -ne 0) {
    Stop-Campaign -Kind 'Operational' -Message 'Fresh run name already exists in the training log root.'
  }

  # Fresh camps deliberately contain no resume or load flags. All four
  # registered defaults and GPU 0 are explicit and checked against --help.
  $FreshTrainingArguments = @(
    '-m', $TrainingModule,
    $Task,
    '--gpu-ids', '0',
    '--log-root', $script:TrainingBaseLogRoot,
    '--env.seed', [string]$Seed,
    '--env.scene.num-envs', [string]$RegisteredNumEnvs,
    '--agent.seed', [string]$Seed,
    '--agent.max-iterations', [string]$RegisteredFreshUpdates,
    '--agent.save-interval', [string]$RegisteredSaveInterval,
    '--agent.num-steps-per-env', [string]$RegisteredStepsPerIteration,
    '--agent.run-name', $runName
  )
  try {
    Invoke-NativeLogged -Executable $script:PythonExe -Arguments $FreshTrainingArguments -LogPath $logPath -FailureMessage ('Fresh seed {0} training failed.' -f $Seed)
    $runs = @(Get-ChildItem -LiteralPath $script:TrainingLogRoot -Directory | Where-Object {
      $_.Name -like ('*_{0}' -f $runName)
    })
    if ($runs.Count -ne 1) {
      Stop-Campaign -Kind 'Operational' -Message 'Fresh training did not create exactly one attributable run directory.'
    }
    $runDirectory = $runs[0].FullName
    $model700Checkpoint = Join-Path $runDirectory 'model_700.pt'
    $model800Checkpoint = Join-Path $runDirectory 'model_800.pt'
    $finalCheckpoint = Join-Path $runDirectory 'model_999.pt'
    $model700ProgressPath = Join-Path $working 'model_700.progress.json'
    $model800ProgressPath = Join-Path $working 'model_800.progress.json'
    $model999ProgressPath = Join-Path $working 'model_999.progress.json'
    New-StairCampProgressReport -CheckpointPath $model700Checkpoint -ReportPath $model700ProgressPath -TrainingSeed $Seed -CompletedUpdates 701 -ExpectedEvaluations 14 -LogPath $logPath
    New-StairCampProgressReport -CheckpointPath $model800Checkpoint -ReportPath $model800ProgressPath -TrainingSeed $Seed -CompletedUpdates 801 -ExpectedEvaluations 16 -LogPath $logPath
    New-StairCampProgressReport -CheckpointPath $finalCheckpoint -ReportPath $model999ProgressPath -TrainingSeed $Seed -CompletedUpdates $RegisteredFreshUpdates -ExpectedEvaluations 20 -LogPath $logPath
    $model700Progress = Get-Content -LiteralPath $model700ProgressPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $model800Progress = Get-Content -LiteralPath $model800ProgressPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $model999Progress = Get-Content -LiteralPath $model999ProgressPath -Raw -Encoding UTF8 | ConvertFrom-Json
    Assert-StairCampProgressReport -Report $model700Progress -TrainingSeed $Seed -CompletedUpdates 701 -ExpectedEvaluations 14
    Assert-StairCampProgressReport -Report $model800Progress -TrainingSeed $Seed -CompletedUpdates 801 -ExpectedEvaluations 16
    Assert-StairCampProgressReport -Report $model999Progress -TrainingSeed $Seed -CompletedUpdates $RegisteredFreshUpdates -ExpectedEvaluations 20
    $upperHeightUnchanged = [Math]::Abs(
      [double]$model700Progress.upper_height_m - [double]$model999Progress.upper_height_m
    ) -le 1.0e-12
    $stallPredicate = [ordered]@{
      schema_version = 1
      kind = 'stair_camp_extension_stall_predicate'
      task = $Task
      training_seed = $Seed
      git_sha = $script:FullGitSha
      contract_sha256 = $RequiredContractSha256
      artifact_bindings = $script:ExpectedArtifactBindings
      model_700_progress_file = 'model_700.progress.json'
      model_700_progress_sha256 = Get-FileSha256 -Path $model700ProgressPath
      model_999_progress_file = 'model_999.progress.json'
      model_999_progress_sha256 = Get-FileSha256 -Path $model999ProgressPath
      start_completed_updates = 701
      end_completed_updates = $RegisteredFreshUpdates
      start_evaluations = 14
      end_evaluations = 20
      evaluation_delta = 6
      monotone_curriculum_bound = $true
      first_stalled_evaluation = 15
      last_stalled_evaluation = 20
      start_upper_height_m = [double]$model700Progress.upper_height_m
      end_upper_height_m = [double]$model999Progress.upper_height_m
      upper_height_unchanged = $upperHeightUnchanged
      predicate_satisfied = $upperHeightUnchanged
    }
    $stallPredicatePath = Join-Path $working 'stall_predicate.json'
    Write-AtomicJsonNoClobber -Path $stallPredicatePath -Value $stallPredicate
    $envelopePath = Join-Path $working 'model_999.envelope.json'
    New-CheckpointEnvelope -CheckpointPath $finalCheckpoint -EnvelopePath $envelopePath -TrainingSeed $Seed -CompletedUpdates $RegisteredFreshUpdates -LogPath $logPath
    $envelope = Get-Content -LiteralPath $envelopePath -Raw -Encoding UTF8 | ConvertFrom-Json
    $runManifest = [ordered]@{
      schema_version = 1
      kind = 'stair_camp_training_run'
      mode = 'fresh'
      task = $Task
      training_seed = $Seed
      total_budget_updates = $RegisteredFreshUpdates
      resume = $false
      run_directory = $runDirectory
      final_checkpoint = $finalCheckpoint
      final_checkpoint_sha256 = Get-FileSha256 -Path $finalCheckpoint
      completed_updates = [int]$envelope.training.completed_updates
      model_700_progress_sha256 = Get-FileSha256 -Path $model700ProgressPath
      model_800_progress_sha256 = Get-FileSha256 -Path $model800ProgressPath
      model_999_progress_sha256 = Get-FileSha256 -Path $model999ProgressPath
      stall_predicate_sha256 = Get-FileSha256 -Path $stallPredicatePath
      stall_predicate_satisfied = $upperHeightUnchanged
      git_sha = $script:FullGitSha
      contract_sha256 = $RequiredContractSha256
    }
    Write-AtomicJsonNoClobber -Path (Join-Path $working 'training_manifest.json') -Value $runManifest
    Publish-PhaseWorkspace -WorkingPath $working -FinalPath $final
  } catch {
    Write-Warning ('Incomplete fresh training retained: {0}' -f $working)
    throw
  }
  $publishedPredicate = Get-Content -LiteralPath (Join-Path $final 'stall_predicate.json') -Raw -Encoding UTF8 | ConvertFrom-Json
  Write-Host ('[PASS] Fresh StairCamp seed {0} completed 1000 updates; six-evaluation stall={1}.' -f $Seed, $publishedPredicate.predicate_satisfied)
  Write-Host ('STALL_PREDICATE={0}' -f $publishedPredicate.predicate_satisfied)
  Write-Host ('RESULT={0}' -f $final)
}

function Invoke-Extend3000Phase {
  Assert-SeedPhase
  if ($Budget -ne $RegisteredExtensionTotalUpdates) {
    Stop-Campaign -Kind 'Protocol' -Message 'Extend3000 requires -Budget 3000.'
  }
  $manifest = Get-CampaignManifest
  $budgetDecision = $null
  if ($Seed -eq 1) {
    if (-not $AuthorizeExtension.IsPresent) {
      Stop-Campaign -Kind 'Protocol' -Message 'Seed 1 extension requires the one campaign-level -AuthorizeExtension decision.'
    }
    $seedOnePredicate = Get-VerifiedStallPredicate -TrainingSeed 1
    if ($seedOnePredicate.predicate_satisfied -ne $true) {
      Stop-Campaign -Kind 'Protocol' -Message 'Seed 1 did not satisfy the registered six-evaluation stall predicate.'
    }
  } else {
    # Budget homogeneity is decided exactly once by seed 1. Seeds 2 and 3 do not evaluate their own stall or require a second authorization.
    $budgetDecision = Get-VerifiedBudgetDecision
  }
  $freshRoot = Join-Path $script:CampaignRootPath ('seed{0}\fresh-1000' -f $Seed)
  $freshManifestPath = Join-Path $freshRoot 'training_manifest.json'
  $freshEnvelopePath = Join-Path $freshRoot 'model_999.envelope.json'
  if (-not (Test-Path -LiteralPath $freshManifestPath -PathType Leaf) -or -not (Test-Path -LiteralPath $freshEnvelopePath -PathType Leaf)) {
    Stop-Campaign -Kind 'Protocol' -Message 'Extension requires that seed own final fresh-1000 checkpoint.'
  }
  $freshManifest = Get-Content -LiteralPath $freshManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
  $freshEnvelope = Get-Content -LiteralPath $freshEnvelopePath -Raw -Encoding UTF8 | ConvertFrom-Json
  if (
    $freshManifest.kind -ne 'stair_camp_training_run' -or
    $freshManifest.mode -ne 'fresh' -or
    $freshManifest.task -ne $Task -or
    [int]$freshManifest.training_seed -ne $Seed -or
    [int]$freshManifest.total_budget_updates -ne $RegisteredFreshUpdates -or
    [int]$freshManifest.completed_updates -ne $RegisteredFreshUpdates -or
    $freshManifest.resume -ne $false -or
    $freshManifest.git_sha -ne $script:FullGitSha -or
    $freshManifest.contract_sha256 -ne $RequiredContractSha256 -or
    [int]$freshEnvelope.training.training_seed -ne $Seed -or
    [int]$freshEnvelope.training.completed_updates -ne $RegisteredFreshUpdates -or
    $freshEnvelope.training.git_sha -ne $script:FullGitSha -or
    $freshEnvelope.training.contract_sha256 -ne $RequiredContractSha256 -or
    $freshEnvelope.checkpoint_file_sha256 -ne $freshManifest.final_checkpoint_sha256
  ) {
    Stop-Campaign -Kind 'Provenance' -Message 'Extension source is not this seed final 1000-update checkpoint.'
  }
  $freshRunDirectory = [System.IO.Path]::GetFullPath([string]$freshManifest.run_directory)
  $freshCheckpoint = [System.IO.Path]::GetFullPath([string]$freshManifest.final_checkpoint)
  $envelopeCheckpoint = [System.IO.Path]::GetFullPath([string]$freshEnvelope.checkpoint_file)
  $trainingLogPrefix = $script:TrainingLogRoot.TrimEnd('\') + '\'
  if (
    -not (Test-Path -LiteralPath $freshRunDirectory -PathType Container) -or
    -not (Test-Path -LiteralPath $freshCheckpoint -PathType Leaf) -or
    -not $freshRunDirectory.StartsWith($trainingLogPrefix, [System.StringComparison]::OrdinalIgnoreCase) -or
    -not $freshCheckpoint.Equals($envelopeCheckpoint, [System.StringComparison]::OrdinalIgnoreCase) -or
    -not (Split-Path -Parent $freshCheckpoint).Equals($freshRunDirectory, [System.StringComparison]::OrdinalIgnoreCase) -or
    (Split-Path -Leaf $freshCheckpoint) -cne 'model_999.pt' -or
    (Get-FileSha256 -Path $freshCheckpoint) -ne $freshManifest.final_checkpoint_sha256
  ) {
    Stop-Campaign -Kind 'Provenance' -Message 'Extension source checkpoint path or bytes drifted.'
  }

  $runName = '{0}_seed{1}_extension_total3000' -f $manifest.campaign_id, $Seed
  $existingRuns = @(Get-ChildItem -LiteralPath $script:TrainingLogRoot -Directory -ErrorAction SilentlyContinue | Where-Object {
    $_.Name -like ('*_{0}' -f $runName)
  })
  if ($existingRuns.Count -ne 0) {
    Stop-Campaign -Kind 'Operational' -Message 'Extension run name already exists in the training log root.'
  }
  $final = Join-Path $script:CampaignRootPath ('seed{0}\extension-3000' -f $Seed)
  if (Test-Path -LiteralPath $final) {
    Stop-Campaign -Kind 'Operational' -Message ('Refusing to overwrite completed phase output: {0}' -f $final)
  }
  $budgetDecisionPath = Join-Path $script:CampaignRootPath 'budget_decision.json'
  if ($Seed -eq 1 -and (Test-Path -LiteralPath $budgetDecisionPath)) {
    Stop-Campaign -Kind 'Operational' -Message 'Refusing to overwrite the campaign-level budget decision.'
  }
  $working = New-PhaseWorkspace -FinalPath $final
  $logPath = Join-Path $working 'training.log'
  $sourceRunName = Split-Path -Leaf ([string]$freshManifest.run_directory)
  $sourceCheckpointName = Split-Path -Leaf ([string]$freshManifest.final_checkpoint)
  $loadRunRegex = '^{0}$' -f [System.Text.RegularExpressions.Regex]::Escape($sourceRunName)
  $loadCheckpointRegex = '^{0}$' -f [System.Text.RegularExpressions.Regex]::Escape($sourceCheckpointName)
  $ExtensionTrainingArguments = @(
    '-m', $TrainingModule,
    $Task,
    '--gpu-ids', '0',
    '--log-root', $script:TrainingBaseLogRoot,
    '--env.seed', [string]$Seed,
    '--env.scene.num-envs', [string]$RegisteredNumEnvs,
    '--agent.seed', [string]$Seed,
    '--agent.max-iterations', [string]$RegisteredExtensionTotalUpdates,
    '--agent.save-interval', [string]$RegisteredSaveInterval,
    '--agent.num-steps-per-env', [string]$RegisteredStepsPerIteration,
    '--agent.run-name', $runName,
    '--agent.resume', 'True',
    '--agent.load-run', $loadRunRegex,
    '--agent.load-checkpoint', $loadCheckpointRegex
  )
  try {
    Invoke-NativeLogged -Executable $script:PythonExe -Arguments @(
      '-m', $EvaluatorModule,
      'validate-checkpoint',
      '--envelope', $freshEnvelopePath,
      '--expected-git-sha', $script:FullGitSha,
      '--expected-contract-sha256', $RequiredContractSha256,
      '--expected-training-seed', [string]$Seed,
      '--verify-checkpoint-file',
      '--output', (Join-Path $working 'extension_source.validation.json')
    ) -LogPath $logPath -FailureMessage 'Evaluator rejected the extension source checkpoint.' -FailureKind 'Provenance'
    Invoke-NativeLogged -Executable $script:PythonExe -Arguments $ExtensionTrainingArguments -LogPath $logPath -Append -FailureMessage ('Authorized seed {0} extension failed.' -f $Seed)
    $runs = @(Get-ChildItem -LiteralPath $script:TrainingLogRoot -Directory | Where-Object {
      $_.Name -like ('*_{0}' -f $runName)
    })
    if ($runs.Count -ne 1) {
      Stop-Campaign -Kind 'Operational' -Message 'Extension did not create exactly one attributable run directory.'
    }
    $runDirectory = $runs[0].FullName
    $finalCheckpoint = Join-Path $runDirectory 'model_2999.pt'
    $envelopePath = Join-Path $working 'model_2999.envelope.json'
    New-CheckpointEnvelope -CheckpointPath $finalCheckpoint -EnvelopePath $envelopePath -TrainingSeed $Seed -CompletedUpdates $RegisteredExtensionTotalUpdates -LogPath $logPath
    $envelope = Get-Content -LiteralPath $envelopePath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($Seed -eq 1) {
      $seedOnePredicatePath = Join-Path $freshRoot 'stall_predicate.json'
      $seedOneProgressPath = Join-Path $freshRoot 'model_999.progress.json'
      $seedOneProgress = Get-Content -LiteralPath $seedOneProgressPath -Raw -Encoding UTF8 | ConvertFrom-Json
      $newBudgetDecision = [ordered]@{
        schema_version = 1
        kind = 'stair_camp_campaign_budget_decision'
        task = $Task
        campaign_id = [string]$manifest.campaign_id
        git_sha = $script:FullGitSha
        contract_sha256 = $RequiredContractSha256
        wrapper_canonical_sha256 = $script:ActualSelfHash
        artifact_bindings = $script:ExpectedArtifactBindings
        decision = 'EXTEND_ALL_SEEDS_TO_TOTAL_3000'
        source_budget_updates = $RegisteredFreshUpdates
        selected_total_budget_updates = $RegisteredExtensionTotalUpdates
        training_seeds = $RegisteredTrainingSeeds
        decision_maker_training_seed = 1
        evaluation_seed = $RegisteredEvaluationSeed
        user_authorized = $true
        seed1_stall_predicate_file = 'seed1/fresh-1000/stall_predicate.json'
        seed1_stall_predicate_sha256 = Get-FileSha256 -Path $seedOnePredicatePath
        seed1_final_checkpoint_sha256 = [string]$seedOneProgress.checkpoint_file_sha256
        seed1_extension_checkpoint_sha256 = [string]$envelope.checkpoint_file_sha256
        seed1_evaluation_delta = 6
      }
      Write-AtomicJsonNoClobber -Path $budgetDecisionPath -Value $newBudgetDecision
      $budgetDecision = $newBudgetDecision
    }
    $runManifest = [ordered]@{
      schema_version = 1
      kind = 'stair_camp_training_run'
      mode = 'authorized_extension'
      task = $Task
      training_seed = $Seed
      total_budget_updates = $RegisteredExtensionTotalUpdates
      resume = $true
      extension_source_checkpoint = [string]$freshManifest.final_checkpoint
      extension_source_checkpoint_sha256 = [string]$freshManifest.final_checkpoint_sha256
      extension_source_completed_updates = $RegisteredFreshUpdates
      run_directory = $runDirectory
      final_checkpoint = $finalCheckpoint
      final_checkpoint_sha256 = Get-FileSha256 -Path $finalCheckpoint
      completed_updates = [int]$envelope.training.completed_updates
      git_sha = $script:FullGitSha
      contract_sha256 = $RequiredContractSha256
      user_authorized = [bool]$budgetDecision.user_authorized
      authorization_source = 'seed1_campaign_budget_decision'
      budget_decision_sha256 = Get-FileSha256 -Path $budgetDecisionPath
      budget_decision_maker_training_seed = 1
    }
    Write-AtomicJsonNoClobber -Path (Join-Path $working 'training_manifest.json') -Value $runManifest
    Publish-PhaseWorkspace -WorkingPath $working -FinalPath $final
    if ($Seed -eq 1) {
      [void](Get-VerifiedBudgetDecision)
    }
  } catch {
    Write-Warning ('Incomplete authorized extension retained: {0}' -f $working)
    throw
  }
  Write-Host ('[PASS] Authorized StairCamp seed {0} reached total 3000 updates.' -f $Seed)
  Write-Host ('RESULT={0}' -f $final)
}

function Invoke-SelectK3Phase {
  Assert-SeedPhase
  [void](Get-CampaignManifest)
  if ($Budget -eq $RegisteredExtensionTotalUpdates) {
    [void](Get-VerifiedBudgetDecision)
  }
  $runPhase = if ($Budget -eq $RegisteredFreshUpdates) { 'fresh-1000' } else { 'extension-3000' }
  $runRoot = Join-Path $script:CampaignRootPath ('seed{0}\{1}' -f $Seed, $runPhase)
  $runManifestPath = Join-Path $runRoot 'training_manifest.json'
  if (-not (Test-Path -LiteralPath $runManifestPath -PathType Leaf)) {
    Stop-Campaign -Kind 'Protocol' -Message ('K=3 requires completed {0} training for seed {1}.' -f $Budget, $Seed)
  }
  $runManifest = Get-Content -LiteralPath $runManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
  if (
    [int]$runManifest.training_seed -ne $Seed -or
    [int]$runManifest.total_budget_updates -ne $Budget -or
    [int]$runManifest.completed_updates -ne $Budget
  ) {
    Stop-Campaign -Kind 'Provenance' -Message 'K=3 training manifest seed or budget drifted.'
  }

  $final = Join-Path $script:CampaignRootPath ('seed{0}\k3-{1}' -f $Seed, $Budget)
  $working = New-PhaseWorkspace -FinalPath $final
  $logPath = Join-Path $working 'k3.log'
  $candidatePaths = [System.Collections.Generic.List[string]]::new()
  # Base RSL-RL periodic saves use the just-completed zero-based iteration,
  # while the final save is budget minus one. Provenance is iter plus one.
  # Budget 1000 is model_800/model_900/model_999 with 801/901/1000 updates.
  # Budget 3000 is model_2800/model_2900/model_2999 with 2801/2901/3000 updates.
  $K3CheckpointIterations = @(
    $Budget - (2 * $RegisteredSaveInterval),
    $Budget - $RegisteredSaveInterval,
    $Budget - 1
  )
  $K3CompletedUpdates = @(
    $Budget - (2 * $RegisteredSaveInterval) + 1,
    $Budget - $RegisteredSaveInterval + 1,
    $Budget
  )
  try {
    for ($index = 0; $index -lt 3; $index += 1) {
      $iteration = [int]$K3CheckpointIterations[$index]
      $completedUpdates = [int]$K3CompletedUpdates[$index]
      $checkpointPath = Join-Path ([string]$runManifest.run_directory) ('model_{0}.pt' -f $iteration)
      $envelopePath = Join-Path $working ('model_{0}.envelope.json' -f $iteration)
      New-CheckpointEnvelope -CheckpointPath $checkpointPath -EnvelopePath $envelopePath -TrainingSeed $Seed -CompletedUpdates $completedUpdates -LogPath $logPath
      $candidatePath = Join-Path $working ('candidate_{0}.json' -f $completedUpdates)
      $rawPath = Join-Path $working ('candidate_{0}.collection.json' -f $completedUpdates)
      Invoke-K3Screen -CheckpointEnvelopePath $envelopePath -CandidatePath $candidatePath -RawCollectionPath $rawPath -PoolBudget $Budget -LogPath $logPath
      $candidatePaths.Add($candidatePath)
    }
    if ($candidatePaths.Count -ne 3) {
      Stop-Campaign -Kind 'Protocol' -Message 'K=3 did not produce exactly three candidate envelopes.'
    }
    $selectionPath = Join-Path $working 'selection.json'
    Invoke-NativeLogged -Executable $script:PythonExe -Arguments @(
      '-m', $EvaluatorModule,
      'select-k3',
      '--candidate', $candidatePaths[0], $candidatePaths[1], $candidatePaths[2],
      '--output', $selectionPath
    ) -LogPath $logPath -Append -FailureMessage 'Evaluator K=3 selection failed.' -FailureKind 'Protocol'
    $selection = Get-Content -LiteralPath $selectionPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $actualCompleted = @($selection.ordered_candidates | ForEach-Object { [int]$_.completed_updates })
    $expectedNewestFirst = @($Budget, $Budget - $RegisteredSaveInterval + 1, $Budget - (2 * $RegisteredSaveInterval) + 1)
    if (
      [int]$selection.schema_version -ne 1 -or
      $selection.kind -ne 'stair_camp_k3_selection' -or
      $selection.task -ne $Task -or
      [int]$selection.training_seed -ne $Seed -or
      [int]$selection.budget_updates -ne $Budget -or
      @($selection.ordered_candidates).Count -ne 3 -or
      ($actualCompleted -join ',') -ne ($expectedNewestFirst -join ',') -or
      @('STAIR_CAMP_CHECKPOINT_SELECTED', 'STOP_NO_PROMOTION') -notcontains $selection.classification
    ) {
      Stop-Campaign -Kind 'Protocol' -Message 'K=3 selection result drifted from real RSL-RL cadence.'
    }
    Publish-PhaseWorkspace -WorkingPath $working -FinalPath $final
  } catch {
    Write-Warning ('Incomplete K=3 phase retained: {0}' -f $working)
    throw
  }
  $publishedSelection = Get-Content -LiteralPath (Join-Path $final 'selection.json') -Raw -Encoding UTF8 | ConvertFrom-Json
  if ($publishedSelection.classification -eq 'STOP_NO_PROMOTION') {
    Write-Host ('[COMPLETE] Seed {0}, budget {1}: no K=3 checkpoint passed. Scientific STOP archived.' -f $Seed, $Budget)
  } else {
    Write-Host ('[PASS] Seed {0}, budget {1}: newest passing K=3 checkpoint selected.' -f $Seed, $Budget)
  }
  Write-Host ('CLASSIFICATION={0}' -f $publishedSelection.classification)
  Write-Host ('RESULT={0}' -f $final)
}

function Invoke-EvaluatePhase {
  Assert-SeedPhase
  $campaignManifest = Get-CampaignManifest
  if ($Budget -eq $RegisteredExtensionTotalUpdates) {
    [void](Get-VerifiedBudgetDecision)
  }
  $selectionPath = Join-Path $script:CampaignRootPath ('seed{0}\k3-{1}\selection.json' -f $Seed, $Budget)
  if (-not (Test-Path -LiteralPath $selectionPath -PathType Leaf)) {
    Stop-Campaign -Kind 'Protocol' -Message 'Formal evaluation requires a completed K=3 selection.'
  }
  $selection = Get-Content -LiteralPath $selectionPath -Raw -Encoding UTF8 | ConvertFrom-Json
  if ($selection.classification -eq 'STOP_NO_PROMOTION') {
    Write-Host ('[COMPLETE] Seed {0}, budget {1}: K=3 scientific STOP; formal evaluation not started.' -f $Seed, $Budget)
    Write-Host 'CLASSIFICATION=STOP_NO_PROMOTION'
    return
  }
  if ($selection.classification -ne 'STAIR_CAMP_CHECKPOINT_SELECTED' -or $null -eq $selection.selected_checkpoint) {
    Stop-Campaign -Kind 'Protocol' -Message 'K=3 selection has an unsupported classification.'
  }

  $final = Join-Path $script:CampaignRootPath ('seed{0}\formal-{1}' -f $Seed, $Budget)
  $working = New-PhaseWorkspace -FinalPath $final
  $logPath = Join-Path $working 'evaluation.log'
  $checkpointEnvelopePath = Join-Path $working 'selected_checkpoint.envelope.json'
  try {
    Write-AtomicJsonNoClobber -Path $checkpointEnvelopePath -Value $selection.selected_checkpoint
    $flatPath = Join-Path $working 'flat_gates.json'
    $stairsPath = Join-Path $working 'stairs_baseline.json'
    $slopePath = Join-Path $working 'slope_secondary.json'
    Invoke-FormalEvaluation -Domain 'flat' -Ablation 'baseline' -CheckpointEnvelopePath $checkpointEnvelopePath -OutputPath $flatPath -TrainingSeed $Seed -LogPath $logPath
    Invoke-FormalEvaluation -Domain 'stairs' -Ablation 'baseline' -CheckpointEnvelopePath $checkpointEnvelopePath -OutputPath $stairsPath -TrainingSeed $Seed -LogPath $logPath

    $ablationNames = @(
      'leg-off',
      'zero-shot-scale-0.035',
      'zero-shot-scale-0.070',
      'zero-shot-scale-0.100',
      'mode-always-on'
    )
    $ablationPaths = [System.Collections.Generic.List[string]]::new()
    foreach ($ablationName in $ablationNames) {
      $safeName = $ablationName.Replace('.', 'p')
      $path = Join-Path $working ('stairs_ablation_{0}.json' -f $safeName)
      Invoke-FormalEvaluation -Domain 'stairs' -Ablation $ablationName -CheckpointEnvelopePath $checkpointEnvelopePath -OutputPath $path -TrainingSeed $Seed -LogPath $logPath
      $ablationPaths.Add($path)
    }
    Invoke-FormalEvaluation -Domain 'slope' -Ablation 'baseline' -CheckpointEnvelopePath $checkpointEnvelopePath -OutputPath $slopePath -TrainingSeed $Seed -LogPath $logPath

    $flat = Get-Content -LiteralPath $flatPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $stairs = Get-Content -LiteralPath $stairsPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $slope = Get-Content -LiteralPath $slopePath -Raw -Encoding UTF8 | ConvertFrom-Json
    Assert-EvaluationEnvelope -Payload $flat -Domain 'flat' -Ablation 'baseline'
    Assert-EvaluationEnvelope -Payload $stairs -Domain 'stairs' -Ablation 'baseline'
    Assert-EvaluationEnvelope -Payload $slope -Domain 'slope' -Ablation 'baseline'
    for ($index = 0; $index -lt $ablationNames.Count; $index += 1) {
      $payload = Get-Content -LiteralPath $ablationPaths[$index] -Raw -Encoding UTF8 | ConvertFrom-Json
      Assert-EvaluationEnvelope -Payload $payload -Domain 'stairs' -Ablation $ablationNames[$index]
    }

    $classicalRowsPath = Join-Path $script:CampaignRootPath ([string]$campaignManifest.classical_rows_file)
    $composedPath = Join-Path $working 'adjudication_envelope.json'
    Invoke-NativeLogged -Executable $script:PythonExe -Arguments @(
      '-m', $EvaluatorModule,
      'compose-seed',
      '--stairs-result', $stairsPath,
      '--flat-result', $flatPath,
      '--classical-rows', $classicalRowsPath,
      '--ablation-result', $ablationPaths[0], $ablationPaths[1], $ablationPaths[2], $ablationPaths[3], $ablationPaths[4],
      '--k3-selection', $selectionPath,
      '--budget-iterations', [string]$Budget,
      '--output', $composedPath
    ) -LogPath $logPath -Append -FailureMessage 'Evaluator compose-seed rejected formal evidence.' -FailureKind 'Protocol'
    $composed = Get-Content -LiteralPath $composedPath -Raw -Encoding UTF8 | ConvertFrom-Json
    Assert-ComposedSeedEnvelope -Envelope $composed -TrainingSeed $Seed -PoolBudget $Budget
    $evidenceIndex = [ordered]@{
      schema_version = 1
      kind = 'stair_camp_seed_formal_evidence_index'
      training_seed = $Seed
      evaluation_seed = $RegisteredEvaluationSeed
      budget_iterations = $Budget
      selected_checkpoint_sha256 = [string]$composed.checkpoint_file_sha256
      formal_gates = 'flat_gates.json'
      formal_stairs = 'stairs_baseline.json'
      completed_ablations = @($composed.completed_ablations)
      slope_secondary = 'slope_secondary.json'
      adjudication_envelope = 'adjudication_envelope.json'
      evidence_eligible = [bool]$composed.evidence_eligible
    }
    Write-AtomicJsonNoClobber -Path (Join-Path $working 'evidence_index.json') -Value $evidenceIndex
    Publish-PhaseWorkspace -WorkingPath $working -FinalPath $final
  } catch {
    Write-Warning ('Incomplete formal evaluation retained: {0}' -f $working)
    throw
  }
  Write-Host ('[COMPLETE] Formal gates, stairs, five ablations, and slope archived for seed {0}, budget {1}.' -f $Seed, $Budget)
  Write-Host ('RESULT={0}' -f $final)
}

function Invoke-AdjudicatePhase {
  [void](Get-CampaignManifest)
  if ($Budget -eq $RegisteredExtensionTotalUpdates) {
    [void](Get-VerifiedBudgetDecision)
  }
  $final = Join-Path $script:CampaignRootPath ('adjudication-{0}' -f $Budget)
  $working = New-PhaseWorkspace -FinalPath $final
  $logPath = Join-Path $working 'adjudication.log'
  $envelopes = [System.Collections.Generic.List[object]]::new()
  try {
    foreach ($trainingSeed in $RegisteredTrainingSeeds) {
      $path = Join-Path $script:CampaignRootPath ('seed{0}\formal-{1}\adjudication_envelope.json' -f $trainingSeed, $Budget)
      if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        Stop-Campaign -Kind 'Protocol' -Message ('Three-seed adjudication is missing seed {0}, budget {1} composed evidence.' -f $trainingSeed, $Budget)
      }
      $envelope = Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
      Assert-ComposedSeedEnvelope -Envelope $envelope -TrainingSeed $trainingSeed -PoolBudget $Budget
      $envelopes.Add($envelope)
    }
    if ($envelopes.Count -ne 3) {
      Stop-Campaign -Kind 'Protocol' -Message 'Adjudication input must contain exactly seeds 1, 2, and 3.'
    }
    $inputPath = Join-Path $working 'three_seed_input.json'
    Write-AtomicJsonNoClobber -Path $inputPath -Value ([ordered]@{ envelopes = @($envelopes) })
    $resultPath = Join-Path $working 'adjudication.json'
    Invoke-NativeLogged -Executable $script:PythonExe -Arguments @(
      '-m', $AdjudicatorModule,
      '--input', $inputPath,
      '--output', $resultPath
    ) -LogPath $logPath -FailureMessage 'Three-seed StairCamp adjudication failed.' -FailureKind 'Protocol'
    $result = Get-Content -LiteralPath $resultPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (
      [int]$result.schema_version -ne 1 -or
      $result.kind -ne 'stair_camp_three_seed_adjudication' -or
      @('RESIDUAL_PPO_EXTENDS_CLASSICAL_BOUNDARY', 'STOP_NO_PROMOTION') -notcontains $result.classification -or
      @($result.training_seeds).Count -ne 3 -or
      [int]$result.evaluation_seed -ne $RegisteredEvaluationSeed -or
      [int]$result.budget_iterations -ne $Budget -or
      $result.git_sha -ne $script:FullGitSha -or
      $result.contract_hash -ne $RequiredContractSha256
    ) {
      Stop-Campaign -Kind 'Protocol' -Message 'Three-seed adjudication result identity drifted.'
    }
    Publish-PhaseWorkspace -WorkingPath $working -FinalPath $final
  } catch {
    Write-Warning ('Incomplete adjudication retained: {0}' -f $working)
    throw
  }
  $published = Get-Content -LiteralPath (Join-Path $final 'adjudication.json') -Raw -Encoding UTF8 | ConvertFrom-Json
  if ($published.classification -eq 'STOP_NO_PROMOTION') {
    Write-Host ('[COMPLETE] Budget {0}: legal scientific STOP archived.' -f $Budget)
  } else {
    Write-Host ('[PASS] Budget {0}: three-seed boundary extension qualified.' -f $Budget)
  }
  Write-Host ('CLASSIFICATION={0}' -f $published.classification)
  Write-Host ('RESULT={0}' -f $final)
}

function Invoke-PackagePhase {
  $campaignManifest = Get-CampaignManifest
  if ($Budget -eq $RegisteredExtensionTotalUpdates) {
    [void](Get-VerifiedBudgetDecision)
  }
  $adjudicationPath = Join-Path $script:CampaignRootPath ('adjudication-{0}\adjudication.json' -f $Budget)
  if (-not (Test-Path -LiteralPath $adjudicationPath -PathType Leaf)) {
    Stop-Campaign -Kind 'Protocol' -Message 'Packaging requires completed three-seed adjudication.'
  }
  $adjudication = Get-Content -LiteralPath $adjudicationPath -Raw -Encoding UTF8 | ConvertFrom-Json
  if (@('RESIDUAL_PPO_EXTENDS_CLASSICAL_BOUNDARY', 'STOP_NO_PROMOTION') -notcontains $adjudication.classification) {
    Stop-Campaign -Kind 'Protocol' -Message 'Packaging refuses unsupported adjudication classification.'
  }
  $incompleteChildren = @(Get-ChildItem -LiteralPath $script:CampaignRootPath -Recurse -Force | Where-Object {
    $_.Name -like '*.incomplete.*'
  })
  if ($incompleteChildren.Count -ne 0) {
    Stop-Campaign -Kind 'Operational' -Message 'Packaging refuses a campaign containing incomplete phase outputs.'
  }

  $outputDirectory = $script:CampaignRootPath + ('.budget{0}.evidence' -f $Budget)
  $outputZip = $outputDirectory + '.zip'
  $zipHashPath = $outputZip + '.sha256'
  if ((Test-Path -LiteralPath $outputDirectory) -or (Test-Path -LiteralPath $outputZip) -or (Test-Path -LiteralPath $zipHashPath)) {
    Stop-Campaign -Kind 'Operational' -Message 'Refusing to overwrite immutable campaign package.'
  }
  $runToken = [System.Guid]::NewGuid().ToString('N')
  $working = $outputDirectory + '.incomplete.' + $runToken
  $temporaryZip = $outputZip + '.incomplete.' + $runToken + '.zip'
  $temporaryZipHashPath = $temporaryZip + '.sha256'
  New-Item -ItemType Directory -Path $working | Out-Null
  try {
    $snapshotRoot = Join-Path $working 'campaign'
    Copy-Item -LiteralPath $script:CampaignRootPath -Destination $snapshotRoot -Recurse
    $checkpointArchive = Join-Path $working 'selected_checkpoints'
    New-Item -ItemType Directory -Path $checkpointArchive | Out-Null
    foreach ($trainingSeed in $RegisteredTrainingSeeds) {
      $seedEnvelopePath = Join-Path $script:CampaignRootPath ('seed{0}\formal-{1}\adjudication_envelope.json' -f $trainingSeed, $Budget)
      $seedEnvelope = Get-Content -LiteralPath $seedEnvelopePath -Raw -Encoding UTF8 | ConvertFrom-Json
      $selectedCheckpoint = [string]$seedEnvelope.checkpoint
      $selectedSha256 = [string]$seedEnvelope.checkpoint_file_sha256
      if (-not (Test-Path -LiteralPath $selectedCheckpoint -PathType Leaf) -or (Get-FileSha256 -Path $selectedCheckpoint) -ne $selectedSha256) {
        Stop-Campaign -Kind 'Provenance' -Message ('Selected checkpoint bytes drifted for seed {0}.' -f $trainingSeed)
      }
      $selectedName = 'seed{0}_selected_{1}' -f $trainingSeed, (Split-Path -Leaf $selectedCheckpoint)
      Copy-Item -LiteralPath $selectedCheckpoint -Destination (Join-Path $checkpointArchive $selectedName)
      if ($Budget -eq $RegisteredExtensionTotalUpdates) {
        $extensionManifestPath = Join-Path $script:CampaignRootPath ('seed{0}\extension-3000\training_manifest.json' -f $trainingSeed)
        $extensionManifest = Get-Content -LiteralPath $extensionManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $sourceCheckpoint = [string]$extensionManifest.extension_source_checkpoint
        $sourceSha256 = [string]$extensionManifest.extension_source_checkpoint_sha256
        $budgetDecisionPath = Join-Path $script:CampaignRootPath 'budget_decision.json'
        if (
          $extensionManifest.authorization_source -ne 'seed1_campaign_budget_decision' -or
          [int]$extensionManifest.budget_decision_maker_training_seed -ne 1 -or
          $extensionManifest.budget_decision_sha256 -ne (Get-FileSha256 -Path $budgetDecisionPath)
        ) {
          Stop-Campaign -Kind 'Provenance' -Message ('Extension budget decision binding drifted for seed {0}.' -f $trainingSeed)
        }
        if (-not (Test-Path -LiteralPath $sourceCheckpoint -PathType Leaf) -or (Get-FileSha256 -Path $sourceCheckpoint) -ne $sourceSha256) {
          Stop-Campaign -Kind 'Provenance' -Message ('Extension source checkpoint bytes drifted for seed {0}.' -f $trainingSeed)
        }
        Copy-Item -LiteralPath $sourceCheckpoint -Destination (Join-Path $checkpointArchive ('seed{0}_extension_source_model_999.pt' -f $trainingSeed))
      }
    }
    $protocolNote = [ordered]@{
      schema_version = 1
      kind = 'stair_camp_s5b_immutable_package'
      task = $Task
      git_sha = $script:FullGitSha
      mjlab_git_sha = $RequiredMjLabSha
      contract_sha256 = $RequiredContractSha256
      wrapper_canonical_sha256 = $script:ActualSelfHash
      budget_iterations = $Budget
      classification = [string]$adjudication.classification
      scientific_stop_is_successful_exit = $true
      campaign_id = [string]$campaignManifest.campaign_id
    }
    Write-AtomicJsonNoClobber -Path (Join-Path $working 'protocol_note.json') -Value $protocolNote
    $checksumLines = [System.Collections.Generic.List[string]]::new()
    $files = @(Get-ChildItem -LiteralPath $working -Recurse -File | Where-Object {
      $_.Name -ne 'SHA256SUMS.txt'
    } | Sort-Object FullName)
    foreach ($file in $files) {
      $relative = $file.FullName.Substring($working.Length + 1).Replace('\', '/')
      $checksumLines.Add(('{0}  {1}' -f (Get-FileSha256 -Path $file.FullName), $relative))
    }
    Write-AtomicTextNoClobber -Path (Join-Path $working 'SHA256SUMS.txt') -Lines @($checksumLines)
    Compress-Archive -Path (Join-Path $working '*') -DestinationPath $temporaryZip -CompressionLevel Optimal
    Write-AtomicTextNoClobber -Path $temporaryZipHashPath -Lines @((Get-FileSha256 -Path $temporaryZip))
    Move-Item -LiteralPath $working -Destination $outputDirectory
    Move-Item -LiteralPath $temporaryZip -Destination $outputZip
    Move-Item -LiteralPath $temporaryZipHashPath -Destination $zipHashPath
  } catch {
    $resolvedParent = [System.IO.Path]::GetFullPath((Split-Path -Parent $outputDirectory)).TrimEnd('\') + '\'
    $resolvedPaths = @($outputDirectory, $working, $outputZip, $temporaryZip, $zipHashPath, $temporaryZipHashPath) | ForEach-Object {
      [System.IO.Path]::GetFullPath([string]$_)
    }
    $pathsStayInPackageParent = @($resolvedPaths | Where-Object {
      -not $_.StartsWith($resolvedParent, [System.StringComparison]::OrdinalIgnoreCase)
    }).Count -eq 0
    if ($pathsStayInPackageParent) {
      if ((Test-Path -LiteralPath $zipHashPath -PathType Leaf) -and -not (Test-Path -LiteralPath $temporaryZipHashPath)) {
        Move-Item -LiteralPath $zipHashPath -Destination $temporaryZipHashPath
      }
      if ((Test-Path -LiteralPath $outputZip -PathType Leaf) -and -not (Test-Path -LiteralPath $temporaryZip)) {
        Move-Item -LiteralPath $outputZip -Destination $temporaryZip
      }
      if ((Test-Path -LiteralPath $outputDirectory -PathType Container) -and -not (Test-Path -LiteralPath $working)) {
        Move-Item -LiteralPath $outputDirectory -Destination $working
      }
    }
    Write-Warning ('Incomplete immutable package retained: {0}' -f $working)
    throw
  }
  Write-Host '[COMPLETE] Immutable StairCamp ZIP and SHA256SUMS package published.'
  Write-Host ('CLASSIFICATION={0}' -f $adjudication.classification)
  Write-Host ('RESULT={0}' -f $outputDirectory)
  Write-Host ('ZIP={0}' -f $outputZip)
  Write-Host ('ZIP_SHA256={0}' -f (Get-FileSha256 -Path $outputZip))
}

function Invoke-StairCampCampaign {
  if ($PSVersionTable.PSVersion.Major -ne 5 -or $PSVersionTable.PSVersion.Minor -ne 1) {
    Stop-Campaign -Kind 'Operational' -Message ('Formal StairCamp wrapper requires Windows PowerShell 5.1; got {0}.' -f $PSVersionTable.PSVersion)
  }

  # Canonical UTF-8/LF self-hash verification precedes repository inspection,
  # preflight, training, evaluation, adjudication, and packaging.
  $selfHashPath = Join-Path $PSScriptRoot 'run_stair_camp_s5b.ps1.sha256'
  if (-not (Test-Path -LiteralPath $selfHashPath -PathType Leaf)) {
    Stop-Campaign -Kind 'Provenance' -Message 'Wrapper canonical self-hash sidecar is missing.'
  }
  $expectedSelfHash = (Get-Content -LiteralPath $selfHashPath -Raw -Encoding ASCII).Trim()
  if ($expectedSelfHash -notmatch '^[0-9a-f]{64}$') {
    Stop-Campaign -Kind 'Provenance' -Message 'Wrapper canonical self-hash sidecar is malformed.'
  }
  $script:ActualSelfHash = Get-CanonicalScriptSha256 -Path $PSCommandPath
  if ($script:ActualSelfHash -ne $expectedSelfHash) {
    Stop-Campaign -Kind 'Provenance' -Message 'Wrapper canonical self-hash mismatch.'
  }

  foreach ($command in @('git', 'nvidia-smi')) {
    Assert-CommandAvailable -Name $command
  }
  $script:RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
  Set-Location -LiteralPath $script:RepoRoot
  $script:CampaignRootPath = [System.IO.Path]::GetFullPath($CampaignRoot)
  $repoPrefix = $script:RepoRoot.TrimEnd('\') + '\'
  if (
    $script:CampaignRootPath.Equals($script:RepoRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
    $script:CampaignRootPath.StartsWith($repoPrefix, [System.StringComparison]::OrdinalIgnoreCase)
  ) {
    Stop-Campaign -Kind 'Operational' -Message 'CampaignRoot must be outside the Git checkout.'
  }
  $campaignParent = Split-Path -Parent $script:CampaignRootPath
  if (-not (Test-Path -LiteralPath $campaignParent -PathType Container)) {
    New-Item -ItemType Directory -Path $campaignParent | Out-Null
  }

  $branchLines = @(& git branch --show-current)
  if ($LASTEXITCODE -ne 0 -or $branchLines.Count -ne 1) {
    Stop-Campaign -Kind 'Provenance' -Message 'Unable to identify current Git branch.'
  }
  if (([string]$branchLines[0]).Trim() -ne $RequiredBranch) {
    Stop-Campaign -Kind 'Provenance' -Message ('Expected branch {0}.' -f $RequiredBranch)
  }
  $statusLines = @(& git status --porcelain)
  if ($LASTEXITCODE -ne 0 -or $statusLines.Count -ne 0) {
    Stop-Campaign -Kind 'Provenance' -Message 'Local StairCamp worktree must be clean.'
  }
  Invoke-NativeChecked -Executable 'git' -Arguments @(
    'fetch', '--quiet', 'origin', $RequiredBranch
  ) -FailureMessage 'Unable to refresh the registered remote branch.' -FailureKind 'Operational'
  $headLines = @(& git rev-parse HEAD)
  $headExitCode = $LASTEXITCODE
  $remoteLines = @(& git rev-parse ('origin/{0}' -f $RequiredBranch))
  $remoteExitCode = $LASTEXITCODE
  if (
    $headExitCode -ne 0 -or
    $remoteExitCode -ne 0 -or
    $headLines.Count -ne 1 -or
    $remoteLines.Count -ne 1
  ) {
    Stop-Campaign -Kind 'Provenance' -Message 'Unable to resolve local or remote StairCamp SHA.'
  }
  $script:FullGitSha = ([string]$headLines[0]).Trim().ToLowerInvariant()
  $remoteSha = ([string]$remoteLines[0]).Trim().ToLowerInvariant()
  $expectedSha = $ExpectedGitSha.ToLowerInvariant()
  if ($script:FullGitSha -ne $expectedSha) {
    Stop-Campaign -Kind 'Provenance' -Message 'Local HEAD does not equal mandatory -ExpectedGitSha.'
  }
  if ($script:FullGitSha -ne $remoteSha) {
    Stop-Campaign -Kind 'Provenance' -Message 'Local HEAD does not equal origin/codex/p2-classical-upper-bound.'
  }

  $pyProject = Join-Path $script:RepoRoot 'pyproject.toml'
  $mjlabSourceDeclaration = 'mjlab = { path = "../mjlab-main", editable = true }'
  $pyProjectLines = @(Get-Content -LiteralPath $pyProject -Encoding UTF8)
  if ($pyProjectLines -notcontains $mjlabSourceDeclaration) {
    Stop-Campaign -Kind 'Provenance' -Message 'pyproject.toml MjLab editable source declaration drifted.'
  }
  $mjlabDeclaredRoot = (Resolve-Path -LiteralPath (Join-Path $script:RepoRoot '..\mjlab-main')).Path
  $mjlabTop = @(& git -C $mjlabDeclaredRoot rev-parse --show-toplevel)
  if ($LASTEXITCODE -ne 0 -or $mjlabTop.Count -ne 1) {
    Stop-Campaign -Kind 'Provenance' -Message 'Pinned MjLab source is not a Git checkout.'
  }
  $script:MjLabRoot = (Resolve-Path -LiteralPath ([string]$mjlabTop[0]).Trim()).Path
  $mjlabHead = @(& git -C $script:MjLabRoot rev-parse HEAD)
  $mjlabHeadExitCode = $LASTEXITCODE
  $mjlabStatus = @(& git -C $script:MjLabRoot status --porcelain)
  $mjlabStatusExitCode = $LASTEXITCODE
  if (
    $mjlabHeadExitCode -ne 0 -or
    $mjlabStatusExitCode -ne 0 -or
    $mjlabHead.Count -ne 1 -or
    ([string]$mjlabHead[0]).Trim().ToLowerInvariant() -ne $RequiredMjLabSha -or
    $mjlabStatus.Count -ne 0
  ) {
    Stop-Campaign -Kind 'Provenance' -Message 'MjLab must be clean and pinned to the registered SHA.'
  }

  $artifactRoot = Join-Path $script:RepoRoot 'docs\experiments\artifacts'
  $schedule = Join-Path $artifactRoot 'c1_schedule_candidate24_1f54968_seed1\c1_schedule.json'
  $calibration = Join-Path $artifactRoot 'hybrid_runtime_seed1\velocity_calibration_seed1.json'
  $yaw = Join-Path $artifactRoot 'yaw_gpu_3f8a9330b88fa6129d05ce42ac3a8cc835295a6f_seed1\yaw_calibration.json'
  $posture = Join-Path $artifactRoot 'c1_posture_requalification_seed1\posture_map_seed1_registered_p032.json'
  $station = Join-Path $artifactRoot 'c1_posture_requalification_seed1\station_calibration_seed1.json'
  $script:FrozenArtifactManifest = [ordered]@{
    controller_schedule = [ordered]@{ path = $schedule; sha256 = $ScheduleFileSha256 }
    velocity_calibration = [ordered]@{ path = $calibration; sha256 = $CalibrationFileSha256 }
    yaw_calibration = [ordered]@{ path = $yaw; sha256 = $YawFileSha256 }
    posture_map = [ordered]@{ path = $posture; sha256 = $PostureFileSha256 }
    station_calibration = [ordered]@{ path = $station; sha256 = $StationFileSha256 }
  }
  foreach ($entry in $script:FrozenArtifactManifest.GetEnumerator()) {
    $path = [string]$entry.Value.path
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
      Stop-Campaign -Kind 'Provenance' -Message ('Missing frozen artifact: {0}' -f $entry.Key)
    }
    if ((Get-FileSha256 -Path $path) -ne [string]$entry.Value.sha256) {
      Stop-Campaign -Kind 'Provenance' -Message ('Frozen artifact byte hash drifted: {0}' -f $entry.Key)
    }
  }
  $schedulePayload = Get-Content -LiteralPath $schedule -Raw -Encoding UTF8 | ConvertFrom-Json
  $calibrationPayload = Get-Content -LiteralPath $calibration -Raw -Encoding UTF8 | ConvertFrom-Json
  $yawPayload = Get-Content -LiteralPath $yaw -Raw -Encoding UTF8 | ConvertFrom-Json
  $posturePayload = Get-Content -LiteralPath $posture -Raw -Encoding UTF8 | ConvertFrom-Json
  $stationPayload = Get-Content -LiteralPath $station -Raw -Encoding UTF8 | ConvertFrom-Json
  if (
    $schedulePayload.schedule_hash -ne $ControllerScheduleHash -or
    $schedulePayload.bindings.identification_controller_gain_hash -ne $IdentificationControllerGainHash -or
    $schedulePayload.bindings.identification_calibration_hash -ne $VelocityCalibrationHash -or
    $schedulePayload.bindings.posture_artifact_hash -ne $PostureArtifactHash -or
    $calibrationPayload.calibration_hash -ne $VelocityCalibrationHash -or
    $calibrationPayload.controller_gain_hash -ne $IdentificationControllerGainHash -or
    $yawPayload.yaw_calibration_hash -ne $YawCalibrationHash -or
    $yawPayload.controller_gain_hash -ne $IdentificationControllerGainHash -or
    $posturePayload.map_hash -ne $PostureMapHash -or
    $posturePayload.posture_artifact_hash -ne $PostureArtifactHash -or
    $stationPayload.station_calibration_hash -ne $StationCalibrationHash -or
    $stationPayload.controller_gain_hash -ne $IdentificationControllerGainHash -or
    $stationPayload.posture_map_hash -ne $PostureMapHash -or
    $stationPayload.posture_artifact_hash -ne $PostureArtifactHash
  ) {
    Stop-Campaign -Kind 'Provenance' -Message 'Frozen five-artifact semantic bindings drifted.'
  }
  $script:ExpectedArtifactBindings = [ordered]@{
    controller_gain_hash = $ControllerScheduleHash
    calibration_hash = $VelocityCalibrationHash
    yaw_calibration_hash = $YawCalibrationHash
    posture_map_hash = $PostureMapHash
    posture_artifact_hash = $PostureArtifactHash
    station_calibration_hash = $StationCalibrationHash
  }

  if ([string]::IsNullOrWhiteSpace($Python)) {
    $script:PythonExe = Join-Path $script:RepoRoot '.venv\Scripts\python.exe'
  } else {
    $script:PythonExe = (Resolve-Path -LiteralPath $Python).Path
  }
  if (-not (Test-Path -LiteralPath $script:PythonExe -PathType Leaf)) {
    Stop-Campaign -Kind 'Operational' -Message 'Configured Python executable is missing.'
  }
  $env:PYTHONPATH = ('{0};{1}' -f (Join-Path $script:RepoRoot 'src'), (Join-Path $script:RepoRoot 'src\hoppertrex_mjlab'))
  $env:HOPPERTREX_HYBRID_CONTROLLER_PATH = $schedule
  $env:HOPPERTREX_HYBRID_CALIBRATION_PATH = $calibration
  $env:HOPPERTREX_HYBRID_YAW_CALIBRATION_PATH = $yaw
  $env:HOPPERTREX_HYBRID_POSTURE_MAP_PATH = $posture
  $env:HOPPERTREX_HYBRID_STATION_CALIBRATION_PATH = $station

  $runtimeJson = & $script:PythonExe -c 'import json,pathlib,mjlab,torch; print(json.dumps(dict(cuda_available=torch.cuda.is_available(),cuda_device_count=torch.cuda.device_count(),mjlab_root=str(pathlib.Path(mjlab.__file__).resolve().parents[2]))))'
  if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace([string]$runtimeJson)) {
    Stop-Campaign -Kind 'Operational' -Message 'Unable to query Python, CUDA, and MjLab runtime provenance.'
  }
  $runtime = $runtimeJson | ConvertFrom-Json
  if ($runtime.cuda_available -ne $true -or [int]$runtime.cuda_device_count -lt 1) {
    Stop-Campaign -Kind 'Operational' -Message 'Formal StairCamp campaign requires CUDA device 0.'
  }
  $importedMjLabRoot = (Resolve-Path -LiteralPath ([string]$runtime.mjlab_root)).Path
  if ($importedMjLabRoot -ne $script:MjLabRoot) {
    Stop-Campaign -Kind 'Provenance' -Message 'Python imports MjLab from a checkout other than the pinned editable source.'
  }
  Invoke-NativeChecked -Executable $script:PythonExe -Arguments @(
    '-c', 'import torch; x=torch.ones(1,device="cuda:0"); assert float(x.item()) == 1.0'
  ) -FailureMessage 'CUDA device 0 smoke failed.' -FailureKind 'Operational'
  $gpuLines = @(& nvidia-smi --query-gpu=name,driver_version,memory.total,pci.bus_id --format=csv,noheader)
  if ($LASTEXITCODE -ne 0 -or $gpuLines.Count -lt 1) {
    Stop-Campaign -Kind 'Operational' -Message 'Unable to query GPU provenance with nvidia-smi.'
  }

  # train.py resolves log_root from the repository project root, then appends
  # the registered agent experiment_name. Keep lookup and the explicit CLI root
  # on that same canonical path so run attribution cannot silently miss output.
  $script:TrainingBaseLogRoot = Join-Path $script:RepoRoot 'logs\rsl_rl'
  $script:TrainingLogRoot = Join-Path $script:TrainingBaseLogRoot 'hoppertrex_stair_camp_s5b'
  if (-not (Test-Path -LiteralPath $script:TrainingLogRoot -PathType Container)) {
    New-Item -ItemType Directory -Path $script:TrainingLogRoot | Out-Null
  }

  # Each flag used by the selected phase is verified against current --help
  # before the phase creates any output.
  if ($Phase -eq 'Validate') {
    $replayHelp = Get-NativeHelpText -Executable $script:PythonExe -Arguments @('-m', $PreflightModule, 'replay-c2', '--help') -Name ($PreflightModule + ' replay-c2')
    Assert-HelpContains -HelpText $replayHelp -Name ($PreflightModule + ' replay-c2') -Flags @('--input-dir', '--output')
    $finalizeHelp = Get-NativeHelpText -Executable $script:PythonExe -Arguments @('-m', $PreflightModule, 'finalize', '--help') -Name ($PreflightModule + ' finalize')
    Assert-HelpContains -HelpText $finalizeHelp -Name ($PreflightModule + ' finalize') -Flags @('--c2-replay', '--flat-fp', '--stage5-kick-fp', '--output')
    $manifestHelp = Get-NativeHelpText -Executable $script:PythonExe -Arguments @('-m', $EvaluatorModule, 'manifest', '--help') -Name ($EvaluatorModule + ' manifest')
    Assert-HelpContains -HelpText $manifestHelp -Name ($EvaluatorModule + ' manifest') -Flags @('--profile', '--output')
    $triggerFpHelp = Get-NativeHelpText -Executable $script:PythonExe -Arguments @('-m', $LiveAdapterModule, 'trigger-fp', '--help') -Name ($LiveAdapterModule + ' trigger-fp')
    Assert-HelpContains -HelpText $triggerFpHelp -Name ($LiveAdapterModule + ' trigger-fp') -Flags @('--domain', '--request', '--output')
  } elseif ($Phase -eq 'Fresh1000' -or $Phase -eq 'Extend3000') {
    $trainHelp = Get-NativeHelpText -Executable $script:PythonExe -Arguments @('-m', $TrainingModule, $Task, '--help') -Name $TrainingModule
    $trainFlags = @('--gpu-ids', '--log-root', '--env.seed', '--env.scene.num-envs', '--agent.seed', '--agent.max-iterations', '--agent.save-interval', '--agent.num-steps-per-env', '--agent.run-name')
    if ($Phase -eq 'Extend3000') {
      $trainFlags += @('--agent.resume', '--agent.load-run', '--agent.load-checkpoint')
    }
    Assert-HelpContains -HelpText $trainHelp -Name $TrainingModule -Flags $trainFlags
    $validateHelp = Get-NativeHelpText -Executable $script:PythonExe -Arguments @('-m', $EvaluatorModule, 'validate-checkpoint', '--help') -Name ($EvaluatorModule + ' validate-checkpoint')
    Assert-HelpContains -HelpText $validateHelp -Name ($EvaluatorModule + ' validate-checkpoint') -Flags @('--envelope', '--expected-git-sha', '--expected-contract-sha256', '--expected-training-seed', '--verify-checkpoint-file', '--output')
  } elseif ($Phase -eq 'SelectK3') {
    $selectHelp = Get-NativeHelpText -Executable $script:PythonExe -Arguments @('-m', $EvaluatorModule, 'select-k3', '--help') -Name ($EvaluatorModule + ' select-k3')
    Assert-HelpContains -HelpText $selectHelp -Name ($EvaluatorModule + ' select-k3') -Flags @('--candidate', '--output')
    $validateHelp = Get-NativeHelpText -Executable $script:PythonExe -Arguments @('-m', $EvaluatorModule, 'validate-checkpoint', '--help') -Name ($EvaluatorModule + ' validate-checkpoint')
    Assert-HelpContains -HelpText $validateHelp -Name ($EvaluatorModule + ' validate-checkpoint') -Flags @('--envelope', '--expected-git-sha', '--expected-contract-sha256', '--expected-training-seed', '--verify-checkpoint-file', '--output')
  } elseif ($Phase -eq 'Evaluate') {
    $liveHelp = Get-NativeHelpText -Executable $script:PythonExe -Arguments @('-m', $EvaluatorModule, 'live', '--help') -Name ($EvaluatorModule + ' live')
    Assert-HelpContains -HelpText $liveHelp -Name ($EvaluatorModule + ' live') -Flags @('--domain', '--profile', '--checkpoint-envelope', '--ablation', '--device', '--expected-git-sha', '--expected-contract-sha256', '--expected-training-seed', '--verify-checkpoint-file', '--adapter', '--output')
    $composeHelp = Get-NativeHelpText -Executable $script:PythonExe -Arguments @('-m', $EvaluatorModule, 'compose-seed', '--help') -Name ($EvaluatorModule + ' compose-seed')
    Assert-HelpContains -HelpText $composeHelp -Name ($EvaluatorModule + ' compose-seed') -Flags @('--stairs-result', '--flat-result', '--classical-rows', '--ablation-result', '--k3-selection', '--budget-iterations', '--output')
  } elseif ($Phase -eq 'Adjudicate') {
    $adjudicatorHelp = Get-NativeHelpText -Executable $script:PythonExe -Arguments @('-m', $AdjudicatorModule, '--help') -Name $AdjudicatorModule
    Assert-HelpContains -HelpText $adjudicatorHelp -Name $AdjudicatorModule -Flags @('--input', '--output')
  }

  switch ($Phase) {
    'Validate' { Invoke-ValidatePhase }
    'Fresh1000' { Invoke-Fresh1000Phase }
    'Extend3000' { Invoke-Extend3000Phase }
    'SelectK3' { Invoke-SelectK3Phase }
    'Evaluate' { Invoke-EvaluatePhase }
    'Adjudicate' { Invoke-AdjudicatePhase }
    'Package' { Invoke-PackagePhase }
    default { Stop-Campaign -Kind 'Protocol' -Message 'Unsupported campaign phase.' }
  }
}

try {
  Invoke-StairCampCampaign
  exit $ExitCodeSuccess
} catch {
  $exitCode = $ExitCodeOperational
  if ($null -ne $_.Exception.Data['StairCampExitCode']) {
    $exitCode = [int]$_.Exception.Data['StairCampExitCode']
  }
  [Console]::Error.WriteLine([string]$_.Exception.Message)
  exit $exitCode
}
