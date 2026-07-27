$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-NativeLogged {
  param(
    [Parameter(Mandatory)][string]$Executable,
    [Parameter(Mandatory)][string[]]$Arguments,
    [Parameter(Mandatory)][string]$LogPath,
    [Parameter(Mandatory)][string]$FailureMessage
  )
  $previousPreference = $ErrorActionPreference
  try {
    $ErrorActionPreference = 'Continue'
    & $Executable @Arguments 2>&1 | Tee-Object -FilePath $LogPath
    $exitCode = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $previousPreference
  }
  if ($exitCode -ne 0) {
    throw ('{0} Exit code: {1}. Log: {2}' -f $FailureMessage, $exitCode, $LogPath)
  }
}

$Repository = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Repository

$ExpectedBranch = "codex/p2-classical-upper-bound"
$RequiredBase = "59ff3cd4d86c569d7d0ea8e207640a6d11c178ab"
$RequiredMjLab = "43e0f3ea9c92ddbb4de9f3bb1ac772d604e3ebf6"
$RequiredControllerGainHash = "8fee25a0339dd1e99127cbed912941dc3ad8ef2030ce49a0d310d1563cb87d98"

if ((git branch --show-current).Trim() -ne $ExpectedBranch) {
  throw "Expected branch $ExpectedBranch."
}
$CurrentHead = (git rev-parse HEAD).Trim()
$RemoteHead = (git rev-parse "origin/$ExpectedBranch").Trim()
if ($CurrentHead -ne $RemoteHead) {
  throw "Local HEAD diverges from origin/$ExpectedBranch. Run git pull --ff-only first."
}
git merge-base --is-ancestor $RequiredBase HEAD
if ($LASTEXITCODE -ne 0) { throw "Branch does not contain required C0 archive." }
if (git status --porcelain) { throw "Repository must be clean." }

foreach ($Command in @("git", "uv", "nvidia-smi")) {
  if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
    throw "Missing required command: $Command"
  }
}

$MjLabCandidates = @(
  (Join-Path $Repository "..\\mjlab-main"),
  (Join-Path $Repository "..\\..\\..\\mjlab-main")
)
$MjLab = $null
foreach ($Candidate in $MjLabCandidates) {
  if (Test-Path -LiteralPath $Candidate -PathType Container) {
    $MjLab = (Resolve-Path -LiteralPath $Candidate).Path
    break
  }
}
if ($null -eq $MjLab) {
  throw "Could not locate sibling mjlab-main from normal clone or worktree layout."
}
if ((git -C $MjLab rev-parse HEAD).Trim() -ne $RequiredMjLab) {
  throw "MjLab must be pinned to $RequiredMjLab."
}
if (git -C $MjLab status --porcelain) { throw "MjLab checkout must be clean." }

$RequiredController = $env:HOPPERTREX_HYBRID_CONTROLLER_PATH
if (-not $RequiredController -or -not (Test-Path -LiteralPath $RequiredController -PathType Leaf)) {
  throw "Missing controller artifact path."
}

$Python = Join-Path $Repository ".venv\\Scripts\\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
  uv sync --frozen --python 3.11
}
$env:PYTHONPATH = ('{0};{1}' -f (Join-Path $Repository 'src'), (Join-Path $Repository 'src\hoppertrex_mjlab'))

$ControllerContent = Get-Content -LiteralPath $RequiredController -Raw -Encoding UTF8 | ConvertFrom-Json
$ActualGainHash = [string]$ControllerContent.gain_hash
if ($ActualGainHash -ne $RequiredControllerGainHash) {
  throw "Controller gain hash mismatch: expected $RequiredControllerGainHash, got $ActualGainHash."
}

nvidia-smi | Out-Null

$Timestamp = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
$OutputDir = Join-Path $Repository "experiments\\yaw_gpu_${CurrentHead}_seed1"
if (Test-Path -LiteralPath $OutputDir) {
  throw "Output directory already exists: $OutputDir"
}
New-Item -ItemType Directory -Path $OutputDir | Out-Null

$OutputArtifact = Join-Path $OutputDir "yaw_calibration.json"
$ProtocolNote = Join-Path $OutputDir "protocol_note.json"
$ConsoleLog = Join-Path $OutputDir "console.log"
$Sha256Sums = Join-Path $OutputDir "SHA256SUMS.txt"

$ProbeArgs = @(
  "-u", "-m", "hoppertrex_mjlab.scripts.probe_hybrid_yaw_transfer",
  "--task", "HopperTrex-Hybrid-v2-Stage2",
  "--device", "cuda:0",
  "--num-envs", "16",
  "--settle-steps", "50",
  "--measure-steps", "150",
  "--yaw-actions", "-1.0", "-0.85", "-0.7", "-0.55", "-0.4", "-0.25", "-0.15",
                   "0.15", "0.25", "0.4", "0.55", "0.7", "0.85", "1.0",
  "--probe-yaw-scale", "1.0",
  "--fit-output", $OutputArtifact
)

Write-Host "[yaw-gpu] Starting GPU yaw transfer sweep and fit..."
Invoke-NativeLogged -Executable $Python -Arguments $ProbeArgs -LogPath $ConsoleLog -FailureMessage "Yaw GPU probe failed."

if (-not (Test-Path -LiteralPath $OutputArtifact -PathType Leaf)) {
  throw "Probe did not produce yaw calibration artifact: $OutputArtifact"
}

$YawContent = Get-Content -LiteralPath $OutputArtifact -Raw -Encoding UTF8 | ConvertFrom-Json
$YawGainHash = [string]$YawContent.controller_gain_hash
if ($YawGainHash -ne $RequiredControllerGainHash) {
  throw "Yaw artifact gain hash mismatch: expected $RequiredControllerGainHash, got $YawGainHash."
}
$YawCalibrationHash = [string]$YawContent.yaw_calibration_hash
if (-not $YawCalibrationHash -or $YawCalibrationHash.Length -ne 64) {
  throw "Invalid yaw_calibration_hash in artifact."
}
$BreakpointRates = @($YawContent.breakpoints | ForEach-Object { [double]$_[0] })
$BreakpointDiffs = @($YawContent.breakpoints | ForEach-Object { [double]$_[1] })
if ($BreakpointRates.Count -lt 3) {
  throw "Yaw calibration has fewer than 3 breakpoints; fit is degenerate."
}
$MinRate = ($BreakpointRates | Measure-Object -Minimum).Minimum
$MaxRate = ($BreakpointRates | Measure-Object -Maximum).Maximum
if ($MinRate -gt -0.10 -or $MaxRate -lt 0.10) {
  throw "Yaw breakpoint body-yaw coverage [$MinRate, $MaxRate] rad/s does not span the Stage2 yaw command domain [-0.10, 0.10]."
}
$MinDiff = ($BreakpointDiffs | Measure-Object -Minimum).Minimum
$MaxDiff = ($BreakpointDiffs | Measure-Object -Maximum).Maximum
if ($MinDiff -gt -0.8 -or $MaxDiff -lt 0.8) {
  throw "Yaw breakpoint differential coverage [$MinDiff, $MaxDiff] rad/s does not reach the registered [-0.8, 0.8] sweep authority."
}

$ProtocolPayload = @{
  kind = "yaw_gpu_requalification"
  git_sha = $CurrentHead
  mjlab_git_sha = $RequiredMjLab
  controller_gain_hash = $RequiredControllerGainHash
  yaw_calibration_hash = $YawCalibrationHash
  timestamp_utc = $Timestamp
  task = "HopperTrex-Hybrid-v2-Stage2"
  device = "cuda:0"
  num_envs = 16
  settle_steps = 50
  measure_steps = 150
  yaw_actions = @(-1.0, -0.85, -0.7, -0.55, -0.4, -0.25, -0.15, 0.15, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0)
  probe_yaw_scale = 1.0
  seed = 1
  classification = "YAW_GPU_QUALIFIED"
  next_step = "FREEZE_ARTIFACT_UNBLOCK_STAGE1B"
}
$ProtocolPayload | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $ProtocolNote -Encoding UTF8

$Hashes = @()
foreach ($File in @("yaw_calibration.json", "protocol_note.json", "console.log")) {
  $FilePath = Join-Path $OutputDir $File
  if (Test-Path -LiteralPath $FilePath -PathType Leaf) {
    $Hash = (Get-FileHash -LiteralPath $FilePath -Algorithm SHA256).Hash.ToLowerInvariant()
    $Hashes += "$Hash  $File"
  }
}
$Hashes | Set-Content -LiteralPath $Sha256Sums -Encoding ASCII

$ZipName = "yaw_gpu_${CurrentHead}_seed1.zip"
$ZipPath = Join-Path $Repository "experiments\\$ZipName"
Compress-Archive -LiteralPath $OutputDir -DestinationPath $ZipPath -CompressionLevel Optimal
$ZipHash = (Get-FileHash -LiteralPath $ZipPath -Algorithm SHA256).Hash.ToLowerInvariant()

Write-Host "[yaw-gpu] CLASSIFICATION=YAW_GPU_QUALIFIED"
Write-Host "[yaw-gpu] YAW_CALIBRATION_HASH=$YawCalibrationHash"
Write-Host "[yaw-gpu] ZIP_SHA256=$ZipHash"
Write-Host "[yaw-gpu] Wrapper completed successfully."
