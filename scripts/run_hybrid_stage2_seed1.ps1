param(
  [string] $Controller = "C:\mjlab_workspace\hoppertrex_archive\20260712_222216\hybrid_v2\furp-2026-Zijie-Zhang-WheelLeggedRL-hybrid-v2\experiments\hybrid_v2\artifacts\de4ba075ff8b\controller_seed1.json",
  [string] $Calibration = "C:\mjlab_workspace\hoppertrex_archive\20260712_222216\hybrid_v2\furp-2026-Zijie-Zhang-WheelLeggedRL-hybrid-v2\experiments\hybrid_v2\artifacts\de4ba075ff8b\velocity_calibration_seed1.json",
  [string] $SourceCheckpoint = "src\hoppertrex_mjlab\logs\rsl_rl\hoppertrex_balance\2026-07-14_21-01-36_hybrid_v2_stage1b_probe_seed1\model_99.pt",
  [string] $SourceGate = "experiments\hybrid_stage1b_probe_gate\seed1_formal_stage1b.json",
  # Empty means: run the official yaw transfer probe on this machine and fit
  # the Stage 2.0 feedforward artifact before anything else touches Stage2.
  [string] $YawCalibration = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repository = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Set-Location $repository
$python = ".\.venv\Scripts\python.exe"
$logRoot = "src\hoppertrex_mjlab\logs\rsl_rl\hoppertrex_balance"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
  throw "Python not found: $python"
}
if (@(git status --short).Count -ne 0) {
  throw "Machine-room checkout is dirty before pull."
}
git pull --ff-only origin codex/hybrid-v2
if ($LASTEXITCODE -ne 0) {
  throw "git pull failed."
}
$head = (git rev-parse HEAD).Trim()
$remoteHead = (git rev-parse origin/codex/hybrid-v2).Trim()
if ($head -ne $remoteHead) {
  throw "HEAD does not match origin/codex/hybrid-v2 after pull."
}
if (@(git status --short).Count -ne 0) {
  throw "Machine-room checkout is dirty after pull."
}

$controller = $Controller
$calibration = $Calibration
$source = $SourceCheckpoint
$sourceGate = $SourceGate
$shortSha = $head.Substring(0, 7)

foreach ($path in @($controller, $calibration, $source, $sourceGate)) {
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    throw "Required file not found: $path"
  }
}
$env:PYTHONPATH = (Resolve-Path -LiteralPath "src").Path
$env:HOPPERTREX_HYBRID_CONTROLLER_PATH = (
  Resolve-Path -LiteralPath $controller
).Path
$env:HOPPERTREX_HYBRID_CALIBRATION_PATH = (
  Resolve-Path -LiteralPath $calibration
).Path

# Stage 2.0: the classical layer must own nominal yaw before any Stage2 PPO
# step runs. Fit the feedforward on this machine (GPU + qualified LQR) unless
# an already-fitted artifact was passed in.
if ($YawCalibration -eq "") {
  $yawRoot = "experiments\hybrid_yaw_calibration_$shortSha"
  New-Item -ItemType Directory -Path $yawRoot -Force | Out-Null
  $YawCalibration = Join-Path $yawRoot "yaw_calibration_seed1.json"
  if (Test-Path -LiteralPath $YawCalibration) {
    throw "Yaw calibration already exists: $YawCalibration. Pass -YawCalibration to reuse it."
  }
  & $python -m hoppertrex_mjlab.scripts.probe_hybrid_yaw_transfer `
    --device cuda:0 `
    --num-envs 16 `
    --fit-output $YawCalibration
  if ($LASTEXITCODE -ne 0) {
    throw "Yaw transfer probe / calibration fit failed."
  }
}
if (-not (Test-Path -LiteralPath $YawCalibration -PathType Leaf)) {
  throw "Yaw calibration artifact not found: $YawCalibration"
}
$yawCalibration = (Resolve-Path -LiteralPath $YawCalibration).Path
$env:HOPPERTREX_HYBRID_YAW_CALIBRATION_PATH = $yawCalibration

$sourceGatePayload = Get-Content -LiteralPath $sourceGate -Raw |
  ConvertFrom-Json
$sourceHash = (
  Get-FileHash -LiteralPath $source -Algorithm SHA256
).Hash.ToLowerInvariant()
if ($sourceGatePayload.gate_pass -ne $true) {
  throw "Stage1-B formal gate did not pass."
}
if ($sourceGatePayload.evaluation_profile -ne "formal") {
  throw "Stage1-B gate is not formal."
}
if ($sourceGatePayload.evaluation_source -ne "live") {
  throw "Stage1-B gate is not live."
}
if ($sourceGatePayload.checkpoint_file_sha256 -ne $sourceHash) {
  throw "Stage1-B gate/checkpoint SHA256 mismatch."
}

$handoffRun = "hybrid_v2_stage2_handoff_${shortSha}_seed1"
$handoffRoot = Join-Path $logRoot $handoffRun
$migrated = Join-Path $handoffRoot "model_0.pt"
if (Test-Path -LiteralPath $handoffRoot) {
  throw "Stage2 handoff already exists: $handoffRoot"
}

& $python -m hoppertrex_mjlab.scripts.rsl_rl.migrate_hybrid_stage `
  --source-checkpoint $source `
  --source-gate-json $sourceGate `
  --yaw-calibration $yawCalibration `
  --output-checkpoint $migrated `
  --source-stage 1 `
  --target-stage 2
if ($LASTEXITCODE -ne 0) {
  throw "Stage1->Stage2 migration failed. Do not add the std-reset flag automatically."
}

$stage2RunName = "hybrid_v2_stage2_probe_${shortSha}_seed1"
$existingRuns = @(
  Get-ChildItem -LiteralPath $logRoot -Directory |
    Where-Object { $_.Name -like "*_$stage2RunName" }
)
if ($existingRuns.Count -ne 0) {
  throw "Stage2 probe run name already exists."
}

& $python -m hoppertrex_mjlab.scripts.rsl_rl.train `
  HopperTrex-Hybrid-v2-Stage2 `
  --env.scene.num-envs 256 `
  --agent.max-iterations 100 `
  --agent.save-interval 25 `
  --agent.seed 1 `
  --agent.resume True `
  --agent.load-run ".*$handoffRun.*" `
  --agent.load-checkpoint "model_0.pt" `
  --agent.run-name $stage2RunName
if ($LASTEXITCODE -ne 0) {
  throw "Stage2 probe training failed."
}

$stage2Runs = @(
  Get-ChildItem -LiteralPath $logRoot -Directory |
    Where-Object { $_.Name -like "*_$stage2RunName" }
)
if ($stage2Runs.Count -ne 1) {
  throw "Expected exactly one Stage2 probe run, got $($stage2Runs.Count)."
}
$stage2Run = $stage2Runs[0]
$checkpoints = @(
  Get-ChildItem -LiteralPath $stage2Run.FullName -File -Filter "model_*.pt" |
    Where-Object { $_.BaseName -match "^model_[0-9]+$" } |
    Sort-Object { [int]($_.BaseName.Substring(6)) }
)
if ($checkpoints.Count -eq 0) {
  throw "No Stage2 checkpoint was produced."
}

$gateRoot = "experiments\hybrid_stage2_probe_gate_$shortSha"
New-Item -ItemType Directory -Path $gateRoot -Force | Out-Null
$planarScreen = Join-Path $gateRoot "seed1_stage2_planar_screen.json"
$retentionFormal = Join-Path $gateRoot "seed1_stage1_retention_formal.json"
$planarFormal = Join-Path $gateRoot "seed1_stage2_planar_formal.json"

# K=3 checkpoint selection, fixed in advance: screen the last three saved
# checkpoints for Stage1 retention, newest first, and promote the newest
# passer. Selecting by recency alone was falsified on 44a44b1, where
# model_75 passed retention while the newer model_99 failed.
$candidates = @($checkpoints | Select-Object -Last 3)
[array]::Reverse($candidates)
$checkpoint = $null
$retentionScreen = $null
foreach ($candidate in $candidates) {
  $candidatePath = $candidate.FullName
  $candidateScreen = Join-Path $gateRoot (
    "seed1_stage1_retention_screen_$($candidate.BaseName).json"
  )
  Write-Host "Stage2 candidate: $candidatePath"
  Get-FileHash -LiteralPath $candidatePath -Algorithm SHA256 | Format-List
  & $python -m hoppertrex_mjlab.scripts.rsl_rl.evaluate_hybrid_gate `
    --stage 1 `
    --profile screen `
    --checkpoint-file $candidatePath `
    --seed 1 `
    --device cuda:0 `
    --num-envs 16 `
    --steps 1000 `
    --warmup-steps 300 `
    --window-steps 300 `
    --progress-interval 250 `
    --episode-length-s 1000000000 `
    --output $candidateScreen
  # The gate writes its JSON before exiting non-zero on a failed screen, so
  # a failure here advances to the next candidate instead of aborting.
  if ($LASTEXITCODE -eq 0) {
    $checkpoint = $candidatePath
    $retentionScreen = $candidateScreen
    break
  }
  Write-Host "Retention screen failed for $($candidate.BaseName); trying the next candidate."
}
if ($null -eq $checkpoint) {
  throw "No Stage2 checkpoint passed the Stage1 retention screen (K=3). Stop; Stage1-B remains frozen."
}

& $python -m hoppertrex_mjlab.scripts.rsl_rl.evaluate_hybrid_gate `
  --stage 2 `
  --profile screen `
  --checkpoint-file $checkpoint `
  --stage1-retention-file $retentionScreen `
  --seed 1 `
  --device cuda:0 `
  --num-envs 16 `
  --steps 1000 `
  --warmup-steps 300 `
  --window-steps 300 `
  --progress-interval 250 `
  --episode-length-s 1000000000 `
  --output $planarScreen
if ($LASTEXITCODE -ne 0) {
  throw "Stage2 planar screen failed. Stop and analyze Stage2."
}

$retentionScreenPayload = Get-Content -LiteralPath $retentionScreen -Raw |
  ConvertFrom-Json
$planarScreenPayload = Get-Content -LiteralPath $planarScreen -Raw |
  ConvertFrom-Json
Write-Host "Retention screen pass: $($retentionScreenPayload.gate_pass)"
Write-Host "Planar screen pass: $($planarScreenPayload.gate_pass)"

& $python -m hoppertrex_mjlab.scripts.rsl_rl.play `
  HopperTrex-Hybrid-v2-Stage2 `
  --agent trained `
  --checkpoint-file $checkpoint `
  --viewer viser `
  --num-envs 1 `
  --device cuda:0
if ($LASTEXITCODE -ne 0) {
  throw "Trained Stage2 Viser failed."
}

& $python -m hoppertrex_mjlab.scripts.rsl_rl.play `
  HopperTrex-Hybrid-v2-Stage2 `
  --agent zero `
  --viewer viser `
  --num-envs 1 `
  --device cuda:0
if ($LASTEXITCODE -ne 0) {
  throw "Zero-residual LQR Viser failed."
}

$viserVerdict = Read-Host (
  "Type PASS only if opposite yaw commands turn oppositely without " +
  "sustained drift or balance takeover"
)
if ($viserVerdict -cne "PASS") {
  throw "Viser not accepted; formal gate intentionally not started."
}

& $python -m hoppertrex_mjlab.scripts.rsl_rl.evaluate_hybrid_gate `
  --stage 1 `
  --profile formal `
  --checkpoint-file $checkpoint `
  --seed 1 `
  --device cuda:0 `
  --num-envs 32 `
  --steps 3000 `
  --warmup-steps 300 `
  --window-steps 800 `
  --progress-interval 500 `
  --episode-length-s 1000000000 `
  --output $retentionFormal
if ($LASTEXITCODE -ne 0) {
  throw "Stage1 retention formal gate failed. Stop; do not retrain Stage1."
}

& $python -m hoppertrex_mjlab.scripts.rsl_rl.evaluate_hybrid_gate `
  --stage 2 `
  --profile formal `
  --checkpoint-file $checkpoint `
  --stage1-retention-file $retentionFormal `
  --seed 1 `
  --device cuda:0 `
  --num-envs 32 `
  --steps 3000 `
  --warmup-steps 300 `
  --window-steps 800 `
  --progress-interval 500 `
  --episode-length-s 1000000000 `
  --output $planarFormal
if ($LASTEXITCODE -ne 0) {
  throw "Stage2 planar formal gate failed. Stop and analyze Stage2."
}

$retentionFormalPayload = Get-Content -LiteralPath $retentionFormal -Raw |
  ConvertFrom-Json
$planarFormalPayload = Get-Content -LiteralPath $planarFormal -Raw |
  ConvertFrom-Json
Write-Host "Retention formal pass: $($retentionFormalPayload.gate_pass)"
Write-Host "Planar formal pass: $($planarFormalPayload.gate_pass)"
Write-Host "STOP FOR ANALYSIS. Do not start Stage3."
