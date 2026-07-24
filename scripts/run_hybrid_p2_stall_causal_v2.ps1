[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RequiredBranch = "codex/p2-stair-probe"
$RequiredBase = "fc80fd5d58687ebf6f00908d7ce6fc5c1e61038c"
$MjlabBranch = "codex/hybrid-v2-runtime-r1"
$MjlabCommit = "43e0f3ea9c92ddbb4de9f3bb1ac772d604e3ebf6"
$MjlabRemote = "https://github.com/deguse/mjlab.git"
$ControllerFileSha256 = "663ab77f77521581cde77ea2bd8c72c7f395f33b05b62348ef6d82a752aad7fc"
$CalibrationFileSha256 = "ef002d0d622725509b47c8ff40d8af658fd42f705bdeac67ac35bae4458f889d"
$ControllerGainHash = "8fee25a0339dd1e99127cbed912941dc3ad8ef2030ce49a0d310d1563cb87d98"
$CalibrationHash = "f62648b57bd17a3503bcbdbf58f349f91fcd8de8ef0cf04551c200401233ed01"
$PostureMapHash = "8849ce39ff24b3342376dbae9c62d658c01288ad8c2b71dcd2ec20741b19a2f1"
$PostureArtifactHash = "0d54fca78b38a880678d0ee69964ac86cb18e1a1f62a0ee716a4715071687ad3"
$StationCalibrationHash = "a4d805ce87fff2ef786c740ff366d24833e4c1162a9f70740cc1941dbeaf004a"
$ExpectedActionScales = @(0.5, 0.3, 0.035, 0.035, 0.035, 0.035)
$AllowedClassifications = @("ANALYSIS_READY", "INVALID_CAPTURE")
$ExpectedCellCount = 2
$ExpectedTrialCount = 64
$ExpectedPairCount = 32
$ExpectedAlignedSamples = 101

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
  return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
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
) -FailureMessage "P2 branch does not contain the frozen formal-result base."
if (@(git status --porcelain).Count -ne 0) {
  throw "P2 checkout must be clean before the causal capture."
}
Invoke-NativeChecked -Executable "git" -Arguments @(
  "fetch", "--quiet", "origin", $RequiredBranch
) -FailureMessage "Failed to refresh origin/$RequiredBranch."
$remoteHead = (git rev-parse "origin/$RequiredBranch").Trim()
if ($fullSha -ne $remoteHead) {
  throw "Checkout HEAD $fullSha does not match origin/$RequiredBranch $remoteHead."
}

$shortSha = (git rev-parse --short=7 HEAD).Trim()
$OutputDirectory = Join-Path $RepoRoot (
  "experiments/hybrid_p2_stall_causal_v2_" + $shortSha + "_seed1"
)
if (Test-Path -LiteralPath $OutputDirectory) {
  throw "Refusing to overwrite existing causal capture: $OutputDirectory"
}

$WorkspaceRoot = Split-Path $RepoRoot -Parent
$MjlabRoot = Join-Path $WorkspaceRoot "mjlab-main"
if (-not (Test-Path -LiteralPath $MjlabRoot)) {
  Invoke-NativeChecked -Executable "git" -Arguments @(
    "clone", "--branch", $MjlabBranch, "--single-branch",
    $MjlabRemote, $MjlabRoot
  ) -FailureMessage "MjLab clone failed."
} else {
  if (-not (Test-Path -LiteralPath (Join-Path $MjlabRoot ".git"))) {
    throw "Existing mjlab-main is not a Git checkout: $MjlabRoot"
  }
  if (@(git -C $MjlabRoot status --porcelain).Count -ne 0) {
    throw "Existing mjlab-main checkout is dirty: $MjlabRoot"
  }
  Invoke-NativeChecked -Executable "git" -Arguments @(
    "-C", $MjlabRoot, "fetch", "origin", $MjlabBranch
  ) -FailureMessage "MjLab fetch failed."
}
Invoke-NativeChecked -Executable "git" -Arguments @(
  "-C", $MjlabRoot, "checkout", "--detach", $MjlabCommit
) -FailureMessage "MjLab pinned checkout failed."
if ((git -C $MjlabRoot rev-parse HEAD).Trim() -ne $MjlabCommit) {
  throw "Unexpected MjLab HEAD."
}

Invoke-NativeChecked -Executable "uv" -Arguments @(
  "sync", "--frozen", "--python", "3.11"
) -FailureMessage "uv sync failed."
$Python = Join-Path $RepoRoot ".venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
  throw "uv did not create the expected Python: $Python"
}

$RuntimeArtifacts = Join-Path $RepoRoot "docs/experiments/artifacts/hybrid_runtime_seed1"
$P1Artifacts = Join-Path $RepoRoot "docs/experiments/artifacts/hybrid_p1_1"
$Controller = Join-Path $RuntimeArtifacts "controller_seed1.json"
$Calibration = Join-Path $RuntimeArtifacts "velocity_calibration_seed1.json"
$PostureMap = Join-Path $P1Artifacts "posture_map_seed1_floor028_fullhash.json"
$StationCalibration = Join-Path $P1Artifacts "station_calibration_floor028_fullhash_seed1.json"
foreach ($path in ($Controller, $Calibration, $PostureMap, $StationCalibration)) {
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    throw "Required Git artifact is missing: $path"
  }
}
if ((Get-FileSha256 $Controller) -ne $ControllerFileSha256) {
  throw "Controller file SHA256 mismatch."
}
if ((Get-FileSha256 $Calibration) -ne $CalibrationFileSha256) {
  throw "Velocity calibration file SHA256 mismatch."
}

$env:PYTHONPATH = "$(Join-Path $RepoRoot 'src');$(Join-Path $RepoRoot 'src/hoppertrex_mjlab')"
$env:HOPPERTREX_HYBRID_CONTROLLER_PATH = $Controller
$env:HOPPERTREX_HYBRID_CALIBRATION_PATH = $Calibration
$env:HOPPERTREX_HYBRID_POSTURE_MAP_PATH = $PostureMap
$env:HOPPERTREX_HYBRID_STATION_CALIBRATION_PATH = $StationCalibration
$env:HOPPERTREX_HYBRID_LEG_RESIDUAL_SCALE = "0.035"
Remove-Item Env:HOPPERTREX_HYBRID_YAW_CALIBRATION_PATH -ErrorAction SilentlyContinue

$PowerShellExecutable = if ($PSVersionTable.PSEdition -eq "Core") {
  Join-Path $PSHOME "pwsh.exe"
} else {
  Join-Path $PSHOME "powershell.exe"
}
Invoke-NativeChecked -Executable $PowerShellExecutable -Arguments @(
  "-NoProfile", "-ExecutionPolicy", "Bypass",
  "-File", (Join-Path $RepoRoot "scripts/run_hybrid_v2_machine_room.ps1"),
  "-Phase", "Preflight", "-Python", $Python
) -FailureMessage "Machine-room preflight failed."
Invoke-NativeChecked -Executable $Python -Arguments @(
  "-m", "hoppertrex_mjlab.scripts.probe_hybrid_stall_causal_v2", "--help"
) -FailureMessage "Causal capture --help failed."

$runToken = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssfffZ") +
  "_" + [guid]::NewGuid().ToString("N")
$WorkingDirectory = $OutputDirectory + ".incomplete." + $runToken
New-Item -ItemType Directory -Path $WorkingDirectory | Out-Null
$WorkingOutput = Join-Path $WorkingDirectory "stall_causal_v2.json"
$WorkingNote = Join-Path $WorkingDirectory "protocol_note.json"
$WorkingChecksumFile = Join-Path $WorkingDirectory "SHA256SUMS.txt"
try {
  Invoke-NativeChecked -Executable $Python -Arguments @(
    "-u", "-m", "hoppertrex_mjlab.scripts.probe_hybrid_stall_causal_v2",
    "--device", "cuda:0", "--output", $WorkingOutput
  ) -FailureMessage "P2 stall causal capture failed."
} catch {
  Write-Warning "Incomplete causal capture retained: $WorkingDirectory"
  throw
}

$result = Get-Content -LiteralPath $WorkingOutput -Raw | ConvertFrom-Json
if ($result.probe -ne "hybrid_p2_stall_causal_capture_v2") {
  throw "Unexpected causal capture result type."
}
if ($result.evidence_eligible -ne $true) {
  throw "Official causal capture is not marked evidence eligible."
}
if ($result.promotion_eligible -ne $false -or $result.training_eligible -ne $false) {
  throw "Causal capture must not authorize promotion or training."
}
if ($null -ne $result.checkpoint -or $null -ne $result.checkpoint_file_sha256) {
  throw "Causal capture must not carry a checkpoint."
}
if ($null -ne $result.yaw_calibration_hash) {
  throw "Official zero-yaw causal capture must not load a yaw artifact."
}
if ($null -ne $result.single_cause_label) {
  throw "Causal capture must not assign a single-cause label."
}
if ($result.git_sha -ne $fullSha -or $result.mjlab_git_sha -ne $MjlabCommit) {
  throw "Causal capture Git provenance is wrong."
}
if ($result.task -ne "HopperTrex-Hybrid-v2-Stage5" -or
    [int]$result.seed -ne 1 -or $result.device -ne "cuda:0") {
  throw "Causal capture identity does not match the registered run."
}
if ($null -eq $result.runtime -or
    $result.runtime.device -ne "cuda:0" -or
    $result.runtime.cuda_available -ne $true -or
    [string]::IsNullOrWhiteSpace([string]$result.runtime.gpu_name) -or
    [string]::IsNullOrWhiteSpace([string]$result.runtime.driver_version) -or
    [string]::IsNullOrWhiteSpace([string]$result.runtime.torch_version) -or
    [string]::IsNullOrWhiteSpace([string]$result.runtime.cuda_version)) {
  throw "Causal capture lacks complete GPU runtime provenance."
}
if ($result.controller_gain_hash -ne $ControllerGainHash -or
    $result.calibration_hash -ne $CalibrationHash -or
    $result.posture_map_hash -ne $PostureMapHash -or
    $result.posture_artifact_hash -ne $PostureArtifactHash -or
    $result.station_calibration_hash -ne $StationCalibrationHash) {
  throw "Causal capture artifact provenance drifted."
}
if ($AllowedClassifications -notcontains $result.classification) {
  throw "Causal capture classification is outside the registered set."
}
if (@($result.action_scales).Count -ne $ExpectedActionScales.Count) {
  throw "Causal capture has the wrong action scale count."
}
for ($index = 0; $index -lt $ExpectedActionScales.Count; $index++) {
  if ([Math]::Abs(
      [double]$result.action_scales[$index] - $ExpectedActionScales[$index]
    ) -gt 1.0e-12) {
    throw "Causal capture action scales drifted."
  }
}
if ($result.protocol.terrain -ne "pyramid_stairs" -or
    [double]$result.protocol.step_width_m -ne 0.30 -or
    [int]$result.protocol.environment_seed -ne 1 -or
    $result.protocol.paired_resets_across_cells -ne $true -or
    $result.protocol.paired_flat_stair_by_terrain_slot -ne $true -or
    [int]$result.protocol.envs_per_height -ne 16 -or
    [int]$result.protocol.settle_steps -ne 200 -or
    [int]$result.protocol.drive_steps -ne 500 -or
    [int]$result.protocol.pre_impact_steps -ne 25 -or
    [int]$result.protocol.post_impact_steps -ne 75 -or
    [int]$result.protocol.aligned_sample_count -ne $ExpectedAlignedSamples -or
    [double]$result.protocol.commanded_yaw_rate -ne 0.0) {
  throw "Causal capture protocol drifted from the registered contract."
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
  throw "Causal capture height or command-cell table drifted."
}
if (@($result.protocol.policy_action).Count -ne 6) {
  throw "Causal capture policy action must contain six zeros."
}
foreach ($value in $result.protocol.policy_action) {
  if ([double]$value -ne 0.0) {
    throw "Causal capture policy action must remain zero."
  }
}
if ($result.protocol.contact_sensor.name -ne "wheel_terrain_causal_capture" -or
    $result.protocol.contact_sensor.reduce -ne "none" -or
    [int]$result.protocol.contact_sensor.num_slots_per_wheel -ne 8 -or
    $result.protocol.riser_contact_selector.purpose -ne
      "first-impact time anchor only") {
  throw "Causal contact capture configuration drifted."
}
if (@($result.cells).Count -ne $ExpectedCellCount -or
    @($result.trials).Count -ne $ExpectedTrialCount -or
    @($result.paired_captures).Count -ne $ExpectedPairCount) {
  throw "Causal capture output counts are incomplete."
}
if ($result.classification -eq "ANALYSIS_READY") {
  foreach ($capture in $result.paired_captures) {
    if ($capture.valid -ne $true -or $null -eq $capture.aligned_series) {
      throw "ANALYSIS_READY contains an invalid pair."
    }
    if (@($capture.aligned_series.relative_steps).Count -ne
        $ExpectedAlignedSamples) {
      throw "Aligned causal series has the wrong sample count."
    }
    if (@($capture.impact_contact_slots).Count -lt 1) {
      throw "Aligned causal series lacks its impact contact snapshot."
    }
  }
} elseif (@($result.invalid_reasons).Count -lt 1) {
  throw "INVALID_CAPTURE must record at least one invalid reason."
}

$note = [ordered]@{
  schema_version = 1
  diagnostic = "hybrid_p2_stall_causal_capture_v2"
  classification = $result.classification
  prior_registered_label = "WHEEL_SPIN_FRICTION_LIMITED"
  prior_label_boundary = @(
    "The v1 absolute saturation/slip thresholds were also exceeded by paired flat controls.",
    "The v1 label is retained as registered output, not accepted as friction-only causality."
  )
  current_interpretation = @(
    "Use paired stair-minus-flat deltas and first-impact-aligned contact evidence.",
    "Do not force a single physical cause from this capture."
  )
  checkpoint = $null
  yaw_artifact = $null
  policy_action = @(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  promotion_eligible = $false
  training_eligible = $false
  p3_eligible = $false
  next_action = "Analyze capture; do not train or launch P3 automatically."
}
$noteJson = $note | ConvertTo-Json -Depth 8
[System.IO.File]::WriteAllText(
  $WorkingNote,
  $noteJson + [Environment]::NewLine,
  [System.Text.UTF8Encoding]::new($false)
)
$outputHash = Get-FileSha256 $WorkingOutput
$noteHash = Get-FileSha256 $WorkingNote
$checksumText = "$outputHash  stall_causal_v2.json" +
  [Environment]::NewLine + "$noteHash  protocol_note.json" +
  [Environment]::NewLine
[System.IO.File]::WriteAllText(
  $WorkingChecksumFile,
  $checksumText,
  [System.Text.Encoding]::ASCII
)
Move-Item -LiteralPath $WorkingDirectory -Destination $OutputDirectory
$FinalOutput = Join-Path $OutputDirectory "stall_causal_v2.json"
Write-Host "[PASS] P2 stall causal capture complete."
Write-Host "RESULT=$FinalOutput"
Write-Host "PROTOCOL_NOTE=$(Join-Path $OutputDirectory 'protocol_note.json')"
Write-Host "SHA256SUMS=$(Join-Path $OutputDirectory 'SHA256SUMS.txt')"
Write-Host "CLASSIFICATION=$($result.classification)"
if ($result.classification -eq "ANALYSIS_READY") {
  Write-Host "P2_STALL_CAUSAL_CAPTURE_ANALYSIS_READY_STOP_NO_TRAINING"
} else {
  Write-Host "INVALID_CAPTURE_STOP_RERUN_NO_TRAINING"
}
