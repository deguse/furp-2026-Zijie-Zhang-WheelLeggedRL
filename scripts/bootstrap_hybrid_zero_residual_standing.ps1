[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RequiredBranch = "codex/hybrid-v2"
$RequiredDiagnosticBase = "7d57fd591f2da5936d409c90708d2afc3479f3bc"
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
  "merge-base", "--is-ancestor", $RequiredDiagnosticBase, "HEAD"
) -FailureMessage "HEAD $fullSha does not contain diagnostic base $RequiredDiagnosticBase."
if (@(git status --porcelain).Count -ne 0) {
  throw "HopperTrex checkout must be clean before bootstrap."
}

$shortSha = (git rev-parse --short=7 HEAD).Trim()
$OutputDirectory = Join-Path $RepoRoot (
  "experiments/hybrid_zero_residual_standing_" + $shortSha + "_seed1_new_machine"
)
if (Test-Path -LiteralPath $OutputDirectory) {
  throw "Refusing to overwrite existing diagnostic directory: $OutputDirectory"
}

$WorkspaceRoot = Split-Path $RepoRoot -Parent
$MjlabRoot = Join-Path $WorkspaceRoot "mjlab-main"
if (-not (Test-Path -LiteralPath $MjlabRoot)) {
  Invoke-NativeChecked -Executable "git" -Arguments @(
    "clone", "--branch", $MjlabBranch, "--single-branch", $MjlabRemote,
    $MjlabRoot
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
$mjlabHead = (git -C $MjlabRoot rev-parse HEAD).Trim()
if ($mjlabHead -ne $MjlabCommit) {
  throw "Unexpected MjLab HEAD: $mjlabHead"
}

# Required environment command: uv sync --frozen --python 3.11
Invoke-NativeChecked -Executable "uv" -Arguments @(
  "sync", "--frozen", "--python", "3.11"
) -FailureMessage "uv sync failed."
$Python = Join-Path $RepoRoot ".venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
  throw "uv did not create the expected Python: $Python"
}

$RuntimeArtifacts = Join-Path $RepoRoot "docs/experiments/artifacts/hybrid_runtime_seed1"
$Controller = Join-Path $RuntimeArtifacts "controller_seed1.json"
$Calibration = Join-Path $RuntimeArtifacts "velocity_calibration_seed1.json"
$PostureMap = Join-Path $RepoRoot "docs/experiments/artifacts/hybrid_p1_1/posture_map_seed1_floor028_fullhash.json"
$StationCalibration = Join-Path $RepoRoot "docs/experiments/artifacts/hybrid_p1_1/station_calibration_floor028_fullhash_seed1.json"
$Manifest = Join-Path $RuntimeArtifacts "manifest.json"
foreach ($path in ($Controller, $Calibration, $PostureMap, $StationCalibration, $Manifest)) {
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    throw "Required Git artifact is missing: $path"
  }
}

$controllerFileHash = Get-FileSha256 -Path $Controller
$calibrationFileHash = Get-FileSha256 -Path $Calibration
if ($controllerFileHash -ne $ControllerFileSha256) {
  throw "Controller file SHA256 mismatch: $controllerFileHash"
}
if ($calibrationFileHash -ne $CalibrationFileSha256) {
  throw "Velocity calibration file SHA256 mismatch: $calibrationFileHash"
}
$controllerPayload = Get-Content -LiteralPath $Controller -Raw | ConvertFrom-Json
$calibrationPayload = Get-Content -LiteralPath $Calibration -Raw | ConvertFrom-Json
if ($controllerPayload.controller_type -ne "lqr") {
  throw "Frozen controller is not LQR."
}
if ($controllerPayload.gain_hash -ne $ControllerGainHash) {
  throw "Controller gain hash mismatch."
}
if ($calibrationPayload.calibration_hash -ne $CalibrationHash) {
  throw "Velocity calibration logical hash mismatch."
}
if ($calibrationPayload.controller_gain_hash -ne $ControllerGainHash) {
  throw "Velocity calibration is bound to a different controller."
}

$Preflight = Join-Path $RepoRoot "scripts/run_hybrid_v2_machine_room.ps1"
$PowerShellExecutable = if ($PSVersionTable.PSEdition -eq "Core") {
  Join-Path $PSHOME "pwsh.exe"
} else {
  Join-Path $PSHOME "powershell.exe"
}
if (-not (Test-Path -LiteralPath $PowerShellExecutable -PathType Leaf)) {
  throw "Current PowerShell executable was not found: $PowerShellExecutable"
}
Invoke-NativeChecked -Executable $PowerShellExecutable -Arguments @(
  "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $Preflight,
  "-Phase", "Preflight", "-Python", $Python
) -FailureMessage "Machine-room preflight failed."

$env:PYTHONPATH = "$(Join-Path $RepoRoot 'src');$(Join-Path $RepoRoot 'src/hoppertrex_mjlab')"
$env:HOPPERTREX_HYBRID_CONTROLLER_PATH = $Controller
$env:HOPPERTREX_HYBRID_CALIBRATION_PATH = $Calibration
$env:HOPPERTREX_HYBRID_POSTURE_MAP_PATH = $PostureMap
$env:HOPPERTREX_HYBRID_STATION_CALIBRATION_PATH = $StationCalibration
$env:HOPPERTREX_HYBRID_LEG_RESIDUAL_SCALE = "0.035"
Remove-Item Env:HOPPERTREX_HYBRID_YAW_CALIBRATION_PATH -ErrorAction SilentlyContinue

Invoke-NativeChecked -Executable $Python -Arguments @(
  "-m", "hoppertrex_mjlab.scripts.rsl_rl.evaluate_hybrid_gate", "--help"
) -FailureMessage "evaluate_hybrid_gate --help failed."

$runToken = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssfffZ") + "_" + [guid]::NewGuid().ToString("N")
$WorkingDirectory = $OutputDirectory + ".incomplete." + $runToken
if (Test-Path -LiteralPath $WorkingDirectory) {
  throw "Unexpected incomplete diagnostic collision: $WorkingDirectory"
}
New-Item -ItemType Directory -Path $WorkingDirectory | Out-Null
$WorkingOutput = Join-Path $WorkingDirectory "zero_residual_standing.json"
$WorkingProtocolNote = Join-Path $WorkingDirectory "protocol_note.json"
$WorkingChecksumFile = Join-Path $WorkingDirectory "SHA256SUMS.txt"

$protocol = [ordered]@{
  schema_version = 1
  diagnostic = "hybrid_stage5_zero_residual_standing"
  git_sha = $fullSha
  mjlab_git_sha = $MjlabCommit
  seed = 1
  machine_context = "new_machine_room"
  missing_historical_artifact = "hybrid_yaw_calibration_e8e2f06/yaw_calibration_seed1.json"
  yaw_channel_equivalence = "commanded_yaw_zero_native_policy_yaw_residual_zero_and_valid_yaw_maps_pin_zero_zero"
  evidence_scope = "standing_channel_counterfactual_only"
  nonzero_yaw_gate_eligible = $false
  promotion_eligible = $false
  fixed_protocol = [ordered]@{
    stage = 5
    profile = "screen"
    device = "cuda:0"
    num_envs = 16
    steps = 1000
    warmup_steps = 300
    window_steps = 300
    repeats = 2
    leg_residual_scale = 0.035
  }
  controller_file_sha256 = $controllerFileHash
  calibration_file_sha256 = $calibrationFileHash
}
$protocol | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $WorkingProtocolNote -Encoding UTF8

try {
  Invoke-NativeChecked -Executable $Python -Arguments @(
    "-u", "-m", "hoppertrex_mjlab.scripts.rsl_rl.evaluate_hybrid_gate",
    "--stage", "5",
    "--profile", "screen",
    "--seed", "1",
    "--device", "cuda:0",
    "--num-envs", "16",
    "--steps", "1000",
    "--warmup-steps", "300",
    "--window-steps", "300",
    "--progress-interval", "250",
    "--episode-length-s", "1000000000",
    "--zero-residual-standing-diagnostic",
    "--diagnostic-repeats", "2",
    "--output", $WorkingOutput
  ) -FailureMessage "Zero-residual standing diagnostic runtime failed."
} catch {
  Write-Warning "Incomplete diagnostic retained for inspection: $WorkingDirectory"
  throw
}

$result = Get-Content -LiteralPath $WorkingOutput -Raw | ConvertFrom-Json
if ($result.diagnostic -ne "hybrid_stage5_zero_residual_standing") {
  throw "Unexpected diagnostic result type."
}
if ($result.promotion_eligible -ne $false) {
  throw "Standing diagnostic must not be promotion eligible."
}
if ($null -ne $result.yaw_calibration_hash) {
  throw "Yaw calibration hash must be null for the isolated fallback run."
}
if ($result.task -ne "HopperTrex-Hybrid-v2-Stage5") {
  throw "Standing diagnostic result has the wrong task."
}
if ($result.git_sha -ne $fullSha) {
  throw "Standing diagnostic result has the wrong Git SHA."
}
if ([int]$result.seed -ne 1) {
  throw "Standing diagnostic result has the wrong seed."
}
if ($result.controller_gain_hash -ne $ControllerGainHash) {
  throw "Standing diagnostic result has the wrong controller gain hash."
}
if ($result.calibration_hash -ne $CalibrationHash) {
  throw "Standing diagnostic result has the wrong velocity calibration hash."
}
if ($result.posture_map_hash -ne $PostureMapHash) {
  throw "Standing diagnostic result has the wrong posture map hash."
}
if ($result.posture_artifact_hash -ne $PostureArtifactHash) {
  throw "Standing diagnostic result has the wrong posture artifact hash."
}
if ($result.station_calibration_hash -ne $StationCalibrationHash) {
  throw "Standing diagnostic result has the wrong station calibration hash."
}
if (@($result.action_scales).Count -ne $ExpectedActionScales.Count) {
  throw "Standing diagnostic result has the wrong action-scale count."
}
for ($index = 0; $index -lt $ExpectedActionScales.Count; $index++) {
  if ([Math]::Abs([double]$result.action_scales[$index] - $ExpectedActionScales[$index]) -gt 1.0e-12) {
    throw "Standing diagnostic result has the wrong action scale at index $index."
  }
}
if (@($result.repeats).Count -ne 2) {
  throw "Standing diagnostic did not produce exactly two repeats."
}
if (
  $result.rollout.profile -ne "screen" -or
  [int]$result.rollout.num_envs -ne 16 -or
  [int]$result.rollout.steps -ne 1000 -or
  [int]$result.rollout.warmup_steps -ne 300 -or
  [int]$result.rollout.window_steps -ne 300 -or
  [int]$result.rollout.repeats -ne 2
) {
  throw "Standing diagnostic result does not match the frozen protocol."
}

$result.repeats | ForEach-Object {
  [pscustomobject]@{
    repeat = $_.repeat
    current_rules_pass = $_.current_rules_pass
    command_match_frac = $_.metrics.command_match_frac
    fast_frac = $_.metrics.fast_frac
    late_in_band_frac = $_.metrics.late_in_band_frac
    late_target_band_frac = $_.metrics.late_target_band_frac
    terminated_event_rate = $_.metrics.terminated_event_rate
    non_wheel_contact_rate = $_.metrics.non_wheel_contact_rate
  }
} | Format-Table -AutoSize

$passCount = @($result.repeats | Where-Object { $_.current_rules_pass }).Count
if ($passCount -eq 2) {
  $classification = "REAL_POLICY_REGRESSION_NEXT_EVALUATION_TIME_LEG_ABLATION"
} elseif ($passCount -eq 0) {
  $classification = "PROTOCOL_OR_THRESHOLD_DRIFT_STOP_NO_RETRAINING"
} else {
  $classification = "MIXED_REPEAT_STOP_FOR_VARIANCE_ANALYSIS"
}
$protocol["classification"] = $classification
$protocol["completed_at_utc"] = (Get-Date).ToUniversalTime().ToString("o")
$protocol | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $WorkingProtocolNote -Encoding UTF8

$checksumLines = foreach ($path in ($WorkingOutput, $WorkingProtocolNote)) {
  $hash = Get-FileSha256 -Path $path
  "$hash  $(Split-Path $path -Leaf)"
}
$checksumLines | Set-Content -LiteralPath $WorkingChecksumFile -Encoding ASCII

Move-Item -LiteralPath $WorkingDirectory -Destination $OutputDirectory
$Output = Join-Path $OutputDirectory "zero_residual_standing.json"
$ChecksumFile = Join-Path $OutputDirectory "SHA256SUMS.txt"

Write-Host "[PASS] Zero-residual standing diagnostic complete."
Write-Host "RESULT=$Output"
Write-Host "SHA256SUMS=$ChecksumFile"
Write-Host "CLASSIFICATION=$classification"
