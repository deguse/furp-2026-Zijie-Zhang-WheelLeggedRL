[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RequiredBranch = "codex/p2-stair-probe"
$RequiredBase = "4411057ecdcb2fd89314fcd4350dc9d66c493c54"
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
$AllowedClassifications = @(
  "CLASSICAL_CARD_CANDIDATE_FOUND",
  "WHEEL_SPIN_FRICTION_LIMITED",
  "TORQUE_SATURATED_STALL",
  "DRIVE_TARGET_COLLAPSED",
  "MIXED_STALL_MECHANISM",
  "INVALID_FLAT_CONTROL_STOP"
)
$ExpectedCellCount = 16
$ExpectedTrialCount = 256

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
) -FailureMessage "P2 branch does not contain the frozen Hybrid v2 base."
if (@(git status --porcelain).Count -ne 0) {
  throw "P2 checkout must be clean before the stall diagnostic."
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
  "experiments/hybrid_p2_stall_diagnostic_" + $shortSha + "_seed1"
)
if (Test-Path -LiteralPath $OutputDirectory) {
  throw "Refusing to overwrite existing stall diagnostic directory: $OutputDirectory"
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
  "-m", "hoppertrex_mjlab.scripts.probe_hybrid_stall_diagnostic", "--help"
) -FailureMessage "Stall diagnostic --help failed."

$runToken = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssfffZ") + "_" + [guid]::NewGuid().ToString("N")
$WorkingDirectory = $OutputDirectory + ".incomplete." + $runToken
New-Item -ItemType Directory -Path $WorkingDirectory | Out-Null
$WorkingOutput = Join-Path $WorkingDirectory "stall_diagnostic.json"
$WorkingChecksumFile = Join-Path $WorkingDirectory "SHA256SUMS.txt"
try {
  Invoke-NativeChecked -Executable $Python -Arguments @(
    "-u", "-m", "hoppertrex_mjlab.scripts.probe_hybrid_stall_diagnostic",
    "--device", "cuda:0", "--output", $WorkingOutput
  ) -FailureMessage "P2 stall diagnostic failed."
} catch {
  Write-Warning "Incomplete stall diagnostic retained for inspection: $WorkingDirectory"
  throw
}

# Post-run validation is deliberately light: provenance identity and the
# non-promotable contract only. Numeric tolerances live with the probe's
# own recorded channels, not in authored wrapper bounds (2026-07-24
# root-height incident).
$result = Get-Content -LiteralPath $WorkingOutput -Raw | ConvertFrom-Json
if ($result.probe -ne "hybrid_p2_stall_mechanism_diagnostic") {
  throw "Unexpected stall diagnostic result type."
}
if ($result.evidence_eligible -ne $true) {
  throw "Official stall diagnostic is not marked evidence eligible."
}
if ($result.promotion_eligible -ne $false -or $result.training_eligible -ne $false) {
  throw "Stall diagnostic must not authorize promotion or training."
}
if ($null -ne $result.checkpoint -or $null -ne $result.checkpoint_file_sha256) {
  throw "Stall diagnostic must not carry a checkpoint."
}
if ($null -ne $result.yaw_calibration_hash) {
  throw "Official zero-yaw diagnostic must not load a yaw artifact."
}
if ($result.git_sha -ne $fullSha) {
  throw "Stall diagnostic result has the wrong Git SHA."
}
if ($result.task -ne "HopperTrex-Hybrid-v2-Stage5" -or
    [int]$result.seed -ne 1 -or
    $result.device -ne "cuda:0") {
  throw "Stall diagnostic identity does not match the registered run."
}
if ($null -eq $result.runtime -or
    $result.runtime.device -ne "cuda:0" -or
    $result.runtime.cuda_available -ne $true -or
    [string]::IsNullOrWhiteSpace([string]$result.runtime.gpu_name) -or
    [string]::IsNullOrWhiteSpace([string]$result.runtime.driver_version) -or
    [string]::IsNullOrWhiteSpace([string]$result.runtime.torch_version) -or
    [string]::IsNullOrWhiteSpace([string]$result.runtime.cuda_version)) {
  throw "Stall diagnostic lacks complete GPU runtime provenance."
}
if ($result.mjlab_git_sha -ne $MjlabCommit) {
  throw "Stall diagnostic result has the wrong MjLab SHA."
}
if ($result.controller_gain_hash -ne $ControllerGainHash) {
  throw "Stall diagnostic result has the wrong controller gain hash."
}
if ($result.calibration_hash -ne $CalibrationHash) {
  throw "Stall diagnostic result has the wrong velocity calibration hash."
}
if ($result.posture_map_hash -ne $PostureMapHash) {
  throw "Stall diagnostic result has the wrong posture map hash."
}
if ($result.posture_artifact_hash -ne $PostureArtifactHash) {
  throw "Stall diagnostic result has the wrong posture artifact hash."
}
if ($result.station_calibration_hash -ne $StationCalibrationHash) {
  throw "Stall diagnostic result has the wrong station calibration hash."
}
if ($AllowedClassifications -notcontains $result.classification) {
  throw "Stall diagnostic classification is not in the registered set."
}
if (@($result.action_scales).Count -ne $ExpectedActionScales.Count) {
  throw "Stall diagnostic has the wrong action scale count."
}
for ($index = 0; $index -lt $ExpectedActionScales.Count; $index++) {
  if ([Math]::Abs([double]$result.action_scales[$index] - $ExpectedActionScales[$index]) -gt 1.0e-12) {
    throw "Stall diagnostic action scales drifted from the registered values."
  }
}
if ($result.protocol.terrain -ne "pyramid_stairs" -or
    [double]$result.protocol.step_width_m -ne 0.30 -or
    [int]$result.protocol.environment_seed -ne 1 -or
    $result.protocol.paired_resets_across_cells -ne $true -or
    [int]$result.protocol.envs_per_height -ne 16 -or
    [int]$result.protocol.settle_steps -ne 200 -or
    [int]$result.protocol.drive_steps -ne 500 -or
    [int]$result.protocol.stable_steps -ne 25 -or
    [int]$result.protocol.stall_window_steps -ne 150 -or
    [double]$result.protocol.commanded_yaw_rate -ne 0.0 -or
    $result.protocol.wheel_model.forward_channel -ne "0.5 * (right - left)") {
  throw "Stall diagnostic protocol drifted from the registered contract."
}
if (@($result.protocol.heights_m).Count -ne 2 -or
    [Math]::Abs([double]$result.protocol.heights_m[0]) -gt 1.0e-12 -or
    [Math]::Abs([double]$result.protocol.heights_m[1] - 0.01) -gt 1.0e-12 -or
    @($result.protocol.command_cells).Count -ne 8) {
  throw "Stall diagnostic height or command-cell table drifted."
}
if (@($result.protocol.policy_action).Count -ne 6) {
  throw "Stall diagnostic policy action must contain six zeros."
}
foreach ($value in $result.protocol.policy_action) {
  if ([Math]::Abs([double]$value) -gt 1.0e-12) {
    throw "Stall diagnostic policy action must remain zero."
  }
}
if (@($result.cells).Count -ne $ExpectedCellCount) {
  throw "Stall diagnostic did not produce the registered cell count."
}
if (@($result.trials).Count -ne $ExpectedTrialCount) {
  throw "Stall diagnostic did not produce the registered trial count."
}
foreach ($cell in $result.cells) {
  if ([int]$cell.trials -ne 16) {
    throw "Stall diagnostic cell does not contain 16 trials."
  }
}

$hash = Get-FileSha256 $WorkingOutput
"$hash  stall_diagnostic.json" | Set-Content -LiteralPath $WorkingChecksumFile -Encoding ASCII
Move-Item -LiteralPath $WorkingDirectory -Destination $OutputDirectory
$FinalOutput = Join-Path $OutputDirectory "stall_diagnostic.json"
Write-Host "[PASS] P2 stall diagnostic complete."
Write-Host "RESULT=$FinalOutput"
Write-Host "SHA256SUMS=$(Join-Path $OutputDirectory 'SHA256SUMS.txt')"
Write-Host "CLASSIFICATION=$($result.classification)"
