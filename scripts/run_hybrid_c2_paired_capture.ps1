[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RequiredBranch = "codex/p2-classical-upper-bound"
$RequiredBase = "716a9b30eeb234e171f1606495581e7744e34a7c"
$MjlabCommit = "43e0f3ea9c92ddbb4de9f3bb1ac772d604e3ebf6"
$C1ScheduleHash = "8fe8548bca85978c164bbd7de39d2d6463cdfd8d7ab91796cf57696b0f64e203"
$CalibrationHash = "f62648b57bd17a3503bcbdbf58f349f91fcd8de8ef0cf04551c200401233ed01"
$PostureArtifactHash = "3b96fd3dae66ad781b5b875c74184db101c42da02c53dfcc40a5137a6b5de11a"
$StationCalibrationHash = "c00e859b3093b4812d54799253accdaeb99171a2cf4028b08bc39e68eaaa7d8a"
$AllowedClassifications = @("ANALYSIS_READY", "INVALID_CAPTURE")
$DetectorSignalSchema = "deployment_direct_v1"
$DetectorSeriesFields = @(
  "pitch_rate_radps", "wheel_speed_error_radps", "body_vx_mps"
)
$ExpectedCellCount = 2
$ExpectedCaptureCount = 32
$ExpectedCapturesPerCell = 16
$ExpectedDriveSteps = 500
$ExpectedAlignedSamples = 101
$FlatControlSuccessRate = 0.90

function Assert-CommandAvailable {
  param([Parameter(Mandatory)][string]$Name)
  if ($null -eq (Get-Command $Name -ErrorAction SilentlyContinue)) {
    throw "Required command is unavailable: $Name"
  }
}

function Invoke-NativeChecked {
  param(
    [Parameter(Mandatory)][string]$Executable,
    [Parameter(Mandatory)][string[]]$Arguments,
    [Parameter(Mandatory)][string]$FailureMessage
  )
  $previousPreference = $ErrorActionPreference
  try {
    $ErrorActionPreference = "Continue"
    & $Executable @Arguments
    $exitCode = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $previousPreference
  }
  if ($exitCode -ne 0) {
    throw "$FailureMessage Exit code: $exitCode"
  }
}

function Get-FileSha256 {
  param([Parameter(Mandatory)][string]$Path)
  $result = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash
  return $result.ToLowerInvariant()
}

foreach ($command in ("git", "uv", "nvidia-smi")) {
  Assert-CommandAvailable -Name $command
}

$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $RepoRoot
$branch = (git branch --show-current).Trim()
if ($branch -ne $RequiredBranch) {
  throw "Expected branch $RequiredBranch, got $branch."
}
$fullSha = (git rev-parse HEAD).Trim()
Invoke-NativeChecked -Executable "git" -Arguments @(
  "merge-base", "--is-ancestor", $RequiredBase, "HEAD"
) -FailureMessage "C2 branch does not contain required C1 closure base."
if (@(git status --porcelain).Count -ne 0) {
  throw "C2 checkout must be clean before paired capture."
}
Invoke-NativeChecked -Executable "git" -Arguments @(
  "fetch", "--quiet", "origin", $RequiredBranch
) -FailureMessage "Failed to refresh origin/$RequiredBranch."
$remoteHead = (git rev-parse "origin/$RequiredBranch").Trim()
if ($fullSha -ne $remoteHead) {
  throw "Checkout HEAD $fullSha does not match origin/$RequiredBranch $remoteHead."
}

$MjLabCandidates = @(
  (Join-Path $RepoRoot "..\mjlab-main"),
  (Join-Path $RepoRoot "..\..\mjlab-main"),
  (Join-Path $RepoRoot "..\..\..\mjlab-main")
)
$MjLab = $null
foreach ($Candidate in $MjLabCandidates) {
  if (Test-Path -LiteralPath $Candidate -PathType Container) {
    $MjLab = (Resolve-Path -LiteralPath $Candidate).Path
    break
  }
}
if ($null -eq $MjLab) {
  throw "Could not locate sibling mjlab-main."
}
if ((git -C $MjLab rev-parse HEAD).Trim() -ne $MjlabCommit) {
  throw "MjLab must be pinned to $MjlabCommit."
}
if (@(git -C $MjLab status --porcelain).Count -ne 0) {
  throw "MjLab checkout must be clean."
}

$RequiredEnvVars = @{
  HOPPERTREX_HYBRID_CONTROLLER_PATH = "C1 schedule artifact"
  HOPPERTREX_HYBRID_CALIBRATION_PATH = "velocity calibration"
  HOPPERTREX_HYBRID_POSTURE_MAP_PATH = "posture map"
  HOPPERTREX_HYBRID_STATION_CALIBRATION_PATH = "station calibration"
}
foreach ($Entry in $RequiredEnvVars.GetEnumerator()) {
  $value = [Environment]::GetEnvironmentVariable($Entry.Key)
  if (-not $value -or -not (Test-Path -LiteralPath $value -PathType Leaf)) {
    throw "Missing or invalid $($Entry.Value) env var: $($Entry.Key)"
  }
}

$SchedulePath = $env:HOPPERTREX_HYBRID_CONTROLLER_PATH
$ScheduleContent = Get-Content -LiteralPath $SchedulePath -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not $ScheduleContent.schedule_hash) {
  throw "C1 schedule artifact does not contain schedule_hash field."
}
if ([string]$ScheduleContent.schedule_hash -ne $C1ScheduleHash) {
  throw "C1 schedule_hash mismatch: expected $C1ScheduleHash."
}

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
  uv sync --frozen --python 3.11
}
$env:PYTHONPATH = ('{0};{1}' -f (Join-Path $RepoRoot 'src'), (Join-Path $RepoRoot 'src\hoppertrex_mjlab'))
Invoke-NativeChecked -Executable $Python -Arguments @(
  "-m", "hoppertrex_mjlab.scripts.probe_hybrid_c2_paired_capture_v1", "--help"
) -FailureMessage "C2 paired capture probe --help failed."

nvidia-smi | Out-Null

$OutputDir = Join-Path $RepoRoot "experiments\c2_paired_capture_${fullSha}_seed1"
if (Test-Path -LiteralPath $OutputDir) {
  throw "C2 capture output directory already exists: $OutputDir"
}
$runToken = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssfffZ") +
  "_" + [guid]::NewGuid().ToString("N")
$WorkingDirectory = "$OutputDir.incomplete.$runToken"
New-Item -ItemType Directory -Path $WorkingDirectory | Out-Null
$WorkingOutput = Join-Path $WorkingDirectory "c2_paired_capture.json"
$WorkingNote = Join-Path $WorkingDirectory "protocol_note.json"
$WorkingChecksumFile = Join-Path $WorkingDirectory "SHA256SUMS.txt"

try {
  Invoke-NativeChecked -Executable $Python -Arguments @(
    "-u", "-m", "hoppertrex_mjlab.scripts.probe_hybrid_c2_paired_capture_v1",
    "--device", "cuda:0", "--output", $WorkingOutput
  ) -FailureMessage "C2 paired capture failed."
} catch {
  Write-Warning "Incomplete C2 capture retained: $WorkingDirectory"
  throw
}

$result = Get-Content -LiteralPath $WorkingOutput -Raw -Encoding UTF8 | ConvertFrom-Json
if ($result.probe -ne "hybrid_c2_paired_capture_v1") {
  throw "Unexpected C2 capture result type."
}
if ($result.evidence_eligible -ne $true) {
  throw "Official C2 capture is not marked evidence eligible."
}
if ($result.promotion_eligible -ne $false -or $result.training_eligible -ne $false) {
  throw "C2 capture must not authorize promotion or training."
}
if ($null -ne $result.checkpoint) {
  throw "C2 capture must not carry a checkpoint."
}
if ($null -ne $result.yaw_calibration_hash) {
  throw "C2 capture must not load a yaw artifact."
}
if ($result.git_sha -ne $fullSha -or $result.mjlab_git_sha -ne $MjlabCommit) {
  throw "C2 capture Git provenance is wrong."
}
if ($result.controller_schedule_hash -ne $C1ScheduleHash) {
  throw "C2 capture schedule_hash does not match C1 artifact."
}
if ($result.calibration_hash -ne $CalibrationHash -or
    $result.posture_artifact_hash -ne $PostureArtifactHash -or
    $result.station_calibration_hash -ne $StationCalibrationHash) {
  throw "C2 capture artifact provenance drifted."
}
if ($AllowedClassifications -notcontains $result.classification) {
  throw "C2 capture classification is outside the registered set."
}
if ($result.task -ne "HopperTrex-Hybrid-v2-Stage5" -or
    [int]$result.seed -ne 1 -or $result.device -ne "cuda:0") {
  throw "C2 capture identity does not match the registered run."
}
if ($result.protocol.detector_signal_schema -ne $DetectorSignalSchema -or
    [double]$result.protocol.control_dt_s -ne 0.02 -or
    [int]$result.protocol.envs_per_height -ne $ExpectedCapturesPerCell -or
    [int]$result.protocol.settle_steps -ne 200 -or
    [int]$result.protocol.drive_steps -ne $ExpectedDriveSteps -or
    [int]$result.protocol.pre_impact_steps -ne 25 -or
    [int]$result.protocol.post_impact_steps -ne 75 -or
    [int]$result.protocol.stable_steps -ne 25 -or
    [int]$result.protocol.detector_series_samples -ne $ExpectedAlignedSamples -or
    [int]$result.protocol.expected_capture_count -ne $ExpectedCaptureCount) {
  throw "C2 detector signal schema does not match deployment replay."
}
$ActualDetectorFields = @($result.protocol.detector_series_fields)
if ($ActualDetectorFields.Count -ne $DetectorSeriesFields.Count -or
    [string]::Join("|", $ActualDetectorFields) -ne
      [string]::Join("|", $DetectorSeriesFields)) {
  throw "C2 detector series fields do not match deployment replay."
}
if (@($result.protocol.heights_m).Count -ne 2 -or
    [double]$result.protocol.heights_m[0] -ne 0.0 -or
    [double]$result.protocol.heights_m[1] -ne 0.01 -or
    @($result.protocol.command_cells).Count -ne 2 -or
    $result.protocol.command_cells[0].name -ne "pitch_zero" -or
    [double]$result.protocol.command_cells[0].pitch_rad -ne 0.0 -or
    [double]$result.protocol.command_cells[0].vx_mps -ne 0.07 -or
    $result.protocol.command_cells[1].name -ne "fast_lean_0p032" -or
    [double]$result.protocol.command_cells[1].pitch_rad -ne -0.032 -or
    [double]$result.protocol.command_cells[1].vx_mps -ne 0.10) {
  throw "C2 height or command-cell table drifted from registration."
}
$Captures = @($result.paired_captures)
$Trials = @($result.trials)
if ($Trials.Count -ne $ExpectedCellCount) {
  throw "C2 capture trial rows are incomplete."
}
if ($result.classification -eq "ANALYSIS_READY") {
  if ($Captures.Count -ne $ExpectedCaptureCount -or
      [int]$result.valid_capture_count -ne $ExpectedCaptureCount -or
      [int]$result.invalid_capture_count -ne 0 -or
      $result.flat_control_passed -ne $true) {
    throw "ANALYSIS_READY does not contain 32/32 valid flat-qualified captures."
  }
  foreach ($trial in $Trials) {
    if ([int]$trial.recorded_drive_steps -ne $ExpectedDriveSteps -or
        [int]$trial.stair_terminated -ne 0 -or
        [int]$trial.stair_envs_without_impact -ne 0 -or
        [int]$trial.flat_terminated -ne 0 -or
        [int]$trial.flat_non_wheel_contact -ne 0 -or
        [int]$trial.paired_captures -ne $ExpectedCapturesPerCell -or
        [int]$trial.valid_paired_captures -ne $ExpectedCapturesPerCell -or
        -not ([double]$trial.flat_success_rate -ge $FlatControlSuccessRate)) {
      throw "ANALYSIS_READY contains an incomplete or unhealthy trial."
    }
  }
  foreach ($capture in $Captures) {
    if ($capture.valid -ne $true -or $null -eq $capture.aligned_series) {
      throw "ANALYSIS_READY contains an invalid pair."
    }
    foreach ($side in @("flat", "stair")) {
      $series = $capture.aligned_series.$side
      foreach ($field in $DetectorSeriesFields) {
        if ($null -eq $series.$field -or
            @($series.$field).Count -ne $ExpectedAlignedSamples) {
          throw "C2 $side series field $field is missing or not 101 samples."
        }
      }
    }
  }
}

$ProtocolNote = @{
  git_sha = $fullSha
  mjlab_git_sha = $MjlabCommit
  controller_schedule_hash = $C1ScheduleHash
  calibration_hash = $CalibrationHash
  posture_artifact_hash = $PostureArtifactHash
  station_calibration_hash = $StationCalibrationHash
  classification = $result.classification
  valid_capture_count = $result.valid_capture_count
  flat_control_passed = $result.flat_control_passed
  detector_signal_schema = $DetectorSignalSchema
  next_step = if ($result.classification -eq "ANALYSIS_READY") {
    "DOWNLOAD_FOR_DETECTOR_FITTING"
  } else {
    "INVALID_CAPTURE_STOP"
  }
}
$WorkingNote | Out-Null
[System.IO.File]::WriteAllText(
  $WorkingNote,
  ($ProtocolNote | ConvertTo-Json -Depth 10),
  [System.Text.UTF8Encoding]::new($false)
)

$ChecksumEntries = @()
foreach ($file in @($WorkingOutput, $WorkingNote)) {
  $basename = Split-Path -Leaf $file
  $hash = Get-FileSha256 -Path $file
  $ChecksumEntries += "$hash  $basename"
}
[System.IO.File]::WriteAllLines(
  $WorkingChecksumFile,
  $ChecksumEntries,
  [System.Text.UTF8Encoding]::new($false)
)

Move-Item -LiteralPath $WorkingDirectory -Destination $OutputDir
$ZipPath = "$OutputDir.zip"
Compress-Archive -LiteralPath $OutputDir -DestinationPath $ZipPath -CompressionLevel Optimal
$ZipSha256 = Get-FileSha256 -Path $ZipPath

Write-Host "CLASSIFICATION=$($result.classification)"
Write-Host "VALID_CAPTURES=$($result.valid_capture_count)"
Write-Host "NEXT=$($ProtocolNote.next_step)"
Write-Host "RESULT=$OutputDir"
Write-Host "ZIP=$ZipPath"
Write-Host "ZIP_SHA256=$ZipSha256"
Write-Host ""
Write-Host "C2 paired capture complete. Download ZIP for detector fitting."
