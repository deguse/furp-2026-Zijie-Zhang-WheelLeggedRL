param(
  # Stage5 training seed. The carrier checkpoint/gate stay the frozen seed-1
  # Stage2 candidate for every seed: multi-seed reproduction varies the
  # Stage5 training seed only, so runs share one handoff lineage.
  [int] $Seed = 1,
  # Training length. 100 = the standard probe; 500 = the bounded
  # ceiling-probe run (log section 3.12 hardening item 2). K=3 screens the
  # last three saves either way, so the save cadence scales with length.
  [int] $MaxIterations = 100,
  [int] $SaveInterval = 25,
  [string] $Controller = "C:\mjlab_workspace\hoppertrex_archive\20260712_222216\hybrid_v2\furp-2026-Zijie-Zhang-WheelLeggedRL-hybrid-v2\experiments\hybrid_v2\artifacts\de4ba075ff8b\controller_seed1.json",
  [string] $Calibration = "C:\mjlab_workspace\hoppertrex_archive\20260712_222216\hybrid_v2\furp-2026-Zijie-Zhang-WheelLeggedRL-hybrid-v2\experiments\hybrid_v2\artifacts\de4ba075ff8b\velocity_calibration_seed1.json",
  [string] $YawCalibration = "experiments\hybrid_yaw_calibration_e8e2f06\yaw_calibration_seed1.json",
  # Margin-0.12 refit map (map_hash d041a1c3...) and the station-keeping
  # artifact double-bound to it (accc04d5...), both produced at 2b07e31.
  # Defaults match the machine-room checkout paths verified 2026-07-17.
  [string] $PostureMap = "experiments\hybrid_posture_map_9201194\posture_map_seed1_margin012.json",
  [string] $StationCalibration = "experiments\hybrid_station_calibration_2b07e31\station_calibration_seed1.json",
  # Stage2 no-harm carrier: retention formal gate_pass=false solely on the
  # pre-registered improvement check (codified exemption in the migrator).
  [string] $SourceCheckpoint = "src\hoppertrex_mjlab\logs\rsl_rl\hoppertrex_balance\2026-07-15_19-21-26_hybrid_v2_stage2_probe_9201194_seed1\model_99.pt",
  [string] $SourceGate = "experiments\hybrid_stage2_probe_gate_9201194\seed1_stage1_retention_formal.json"
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
$shortSha = $head.Substring(0, 7)

foreach ($path in @(
  $Controller, $Calibration, $YawCalibration,
  $PostureMap, $StationCalibration,
  $SourceCheckpoint, $SourceGate
)) {
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    throw "Required file not found: $path"
  }
}
$controller = (Resolve-Path -LiteralPath $Controller).Path
$calibration = (Resolve-Path -LiteralPath $Calibration).Path
$yawCalibration = (Resolve-Path -LiteralPath $YawCalibration).Path
$postureMap = (Resolve-Path -LiteralPath $PostureMap).Path
$stationCalibration = (Resolve-Path -LiteralPath $StationCalibration).Path
$source = (Resolve-Path -LiteralPath $SourceCheckpoint).Path
$sourceGate = (Resolve-Path -LiteralPath $SourceGate).Path

$env:PYTHONPATH = (Resolve-Path -LiteralPath "src").Path
$env:HOPPERTREX_HYBRID_CONTROLLER_PATH = $controller
$env:HOPPERTREX_HYBRID_CALIBRATION_PATH = $calibration
$env:HOPPERTREX_HYBRID_YAW_CALIBRATION_PATH = $yawCalibration
$env:HOPPERTREX_HYBRID_POSTURE_MAP_PATH = $postureMap
$env:HOPPERTREX_HYBRID_STATION_CALIBRATION_PATH = $stationCalibration

# Triple migration 2->3->4->5. Stage2->3 consumes the no-harm-carrier gate;
# 3->4 and 4->5 are mechanical pass-through hops (no training happened at
# stages 3/4 per the Route A decision; their scopes are classically closed).
$handoffRun = "hybrid_v2_stage5_handoff_${shortSha}_it${MaxIterations}_seed$Seed"
$handoffRoot = Join-Path $logRoot $handoffRun
if (Test-Path -LiteralPath $handoffRoot) {
  throw "Stage5 handoff already exists: $handoffRoot"
}
$stage3Handoff = Join-Path $handoffRoot "model_stage3.pt"
$stage4Handoff = Join-Path $handoffRoot "model_stage4.pt"
$migrated = Join-Path $handoffRoot "model_0.pt"

& $python -m hoppertrex_mjlab.scripts.rsl_rl.migrate_hybrid_stage `
  --source-checkpoint $source `
  --source-gate-json $sourceGate `
  --yaw-calibration $yawCalibration `
  --posture-map $postureMap `
  --station-calibration $stationCalibration `
  --output-checkpoint $stage3Handoff `
  --source-stage 2 `
  --target-stage 3
if ($LASTEXITCODE -ne 0) {
  throw "Stage2->Stage3 migration failed. Do not add the std-reset flag automatically."
}

& $python -m hoppertrex_mjlab.scripts.rsl_rl.migrate_hybrid_stage `
  --source-checkpoint $stage3Handoff `
  --yaw-calibration $yawCalibration `
  --posture-map $postureMap `
  --station-calibration $stationCalibration `
  --output-checkpoint $stage4Handoff `
  --source-stage 3 `
  --target-stage 4
if ($LASTEXITCODE -ne 0) {
  throw "Stage3->Stage4 migration failed."
}

& $python -m hoppertrex_mjlab.scripts.rsl_rl.migrate_hybrid_stage `
  --source-checkpoint $stage4Handoff `
  --yaw-calibration $yawCalibration `
  --posture-map $postureMap `
  --station-calibration $stationCalibration `
  --output-checkpoint $migrated `
  --source-stage 4 `
  --target-stage 5
if ($LASTEXITCODE -ne 0) {
  throw "Stage4->Stage5 migration failed."
}

$stage5RunName = "hybrid_v2_stage5_probe_${shortSha}_it${MaxIterations}_seed$Seed"
$existingRuns = @(
  Get-ChildItem -LiteralPath $logRoot -Directory |
    Where-Object { $_.Name -like "*_$stage5RunName" }
)
if ($existingRuns.Count -ne 0) {
  throw "Stage5 probe run name already exists."
}

& $python -m hoppertrex_mjlab.scripts.rsl_rl.train `
  HopperTrex-Hybrid-v2-Stage5 `
  --env.scene.num-envs 256 `
  --agent.max-iterations $MaxIterations `
  --agent.save-interval $SaveInterval `
  --agent.seed $Seed `
  --agent.resume True `
  --agent.load-run ".*$handoffRun.*" `
  --agent.load-checkpoint "model_0.pt" `
  --agent.run-name $stage5RunName
if ($LASTEXITCODE -ne 0) {
  throw "Stage5 probe training failed."
}

$stage5Runs = @(
  Get-ChildItem -LiteralPath $logRoot -Directory |
    Where-Object { $_.Name -like "*_$stage5RunName" }
)
if ($stage5Runs.Count -ne 1) {
  throw "Expected exactly one Stage5 probe run, got $($stage5Runs.Count)."
}
$stage5Run = $stage5Runs[0]
$checkpoints = @(
  Get-ChildItem -LiteralPath $stage5Run.FullName -File -Filter "model_*.pt" |
    Where-Object { $_.BaseName -match "^model_[0-9]+$" } |
    Sort-Object { [int]($_.BaseName.Substring(6)) }
)
if ($checkpoints.Count -eq 0) {
  throw "No Stage5 checkpoint was produced."
}

$gateRoot = "experiments\hybrid_stage5_probe_gate_$shortSha"
New-Item -ItemType Directory -Path $gateRoot -Force | Out-Null
$robustScreen = Join-Path $gateRoot "seed${Seed}_it${MaxIterations}_stage5_robust_screen.json"
$retentionFormal = Join-Path $gateRoot "seed${Seed}_it${MaxIterations}_stage1_retention_formal.json"
$robustFormal = Join-Path $gateRoot "seed${Seed}_it${MaxIterations}_stage5_robust_formal.json"
$ablatedFormal = Join-Path $gateRoot "seed${Seed}_it${MaxIterations}_stage5_robust_formal_legs_ablated.json"

# K=3 checkpoint selection, fixed in advance: screen the last three saved
# checkpoints for Stage1 retention, newest first, and promote the newest
# passer (recency-only selection was falsified on 44a44b1).
$candidates = @($checkpoints | Select-Object -Last 3)
[array]::Reverse($candidates)
$checkpoint = $null
$retentionScreen = $null
foreach ($candidate in $candidates) {
  $candidatePath = $candidate.FullName
  $candidateScreen = Join-Path $gateRoot (
    "seed${Seed}_it${MaxIterations}_stage1_retention_screen_$($candidate.BaseName).json"
  )
  Write-Host "Stage5 candidate: $candidatePath"
  Get-FileHash -LiteralPath $candidatePath -Algorithm SHA256 | Format-List
  & $python -m hoppertrex_mjlab.scripts.rsl_rl.evaluate_hybrid_gate `
    --stage 1 `
    --profile screen `
    --checkpoint-file $candidatePath `
    --seed $Seed `
    --device cuda:0 `
    --num-envs 16 `
    --steps 1000 `
    --warmup-steps 300 `
    --window-steps 300 `
    --progress-interval 250 `
    --episode-length-s 1000000000 `
    --output $candidateScreen
  if ($LASTEXITCODE -eq 0) {
    $checkpoint = $candidatePath
    $retentionScreen = $candidateScreen
    break
  }
  Write-Host "Retention screen failed for $($candidate.BaseName); trying the next candidate."
}
if ($null -eq $checkpoint) {
  throw "No Stage5 checkpoint passed the Stage1 retention screen (K=3). Stop; analyze the Stage5 training."
}

# Robust gates derive the stage4 tracking reference from the zero-residual
# classical stack in-run (Route A: no Stage4 training run exists), so no
# --stage4-reference-file is passed here.
& $python -m hoppertrex_mjlab.scripts.rsl_rl.evaluate_hybrid_gate `
  --stage 5 `
  --profile screen `
  --checkpoint-file $checkpoint `
  --seed $Seed `
  --device cuda:0 `
  --num-envs 16 `
  --steps 1000 `
  --warmup-steps 300 `
  --window-steps 300 `
  --progress-interval 250 `
  --episode-length-s 1000000000 `
  --output $robustScreen
if ($LASTEXITCODE -ne 0) {
  throw "Stage5 robust screen failed. Stop and analyze Stage5."
}

$retentionScreenPayload = Get-Content -LiteralPath $retentionScreen -Raw |
  ConvertFrom-Json
$robustScreenPayload = Get-Content -LiteralPath $robustScreen -Raw |
  ConvertFrom-Json
Write-Host "Retention screen pass: $($retentionScreenPayload.gate_pass)"
Write-Host "Robust screen pass: $($robustScreenPayload.gate_pass)"

& $python -m hoppertrex_mjlab.scripts.rsl_rl.play `
  HopperTrex-Hybrid-v2-Stage5 `
  --agent trained `
  --checkpoint-file $checkpoint `
  --viewer viser `
  --num-envs 1 `
  --device cuda:0
if ($LASTEXITCODE -ne 0) {
  throw "Trained Stage5 Viser failed."
}

& $python -m hoppertrex_mjlab.scripts.rsl_rl.play `
  HopperTrex-Hybrid-v2-Stage5 `
  --agent zero `
  --viewer viser `
  --num-envs 1 `
  --device cuda:0
if ($LASTEXITCODE -ne 0) {
  throw "Zero-residual classical-stack Viser failed."
}

$viserVerdict = Read-Host (
  "Type PASS only if posture changes stay smooth, pushes are recovered " +
  "without falls, and turning works under both agents"
)
if ($viserVerdict -cne "PASS") {
  throw "Viser not accepted; formal gates intentionally not started."
}

& $python -m hoppertrex_mjlab.scripts.rsl_rl.evaluate_hybrid_gate `
  --stage 1 `
  --profile formal `
  --checkpoint-file $checkpoint `
  --seed $Seed `
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

# Primary pre-registered adjudication: robust formal carries the large-kick
# recovery improvement check (center@8x, baseline 0.970 s, bar 0.097 s,
# >=128 kick events). A non-zero exit here is a legitimate adjudication
# outcome, not an operational error, so the JSON is kept either way.
& $python -m hoppertrex_mjlab.scripts.rsl_rl.evaluate_hybrid_gate `
  --stage 5 `
  --profile formal `
  --checkpoint-file $checkpoint `
  --seed $Seed `
  --device cuda:0 `
  --num-envs 32 `
  --steps 3000 `
  --warmup-steps 300 `
  --window-steps 800 `
  --progress-interval 500 `
  --episode-length-s 1000000000 `
  --output $robustFormal
$robustFormalExit = $LASTEXITCODE

# Attribution run: same checkpoint with leg residual heads zeroed at eval
# time. Observational evidence only; its gate verdict never blocks.
& $python -m hoppertrex_mjlab.scripts.rsl_rl.evaluate_hybrid_gate `
  --stage 5 `
  --profile formal `
  --checkpoint-file $checkpoint `
  --ablate-leg-residuals `
  --seed $Seed `
  --device cuda:0 `
  --num-envs 32 `
  --steps 3000 `
  --warmup-steps 300 `
  --window-steps 800 `
  --progress-interval 500 `
  --episode-length-s 1000000000 `
  --output $ablatedFormal
$ablatedFormalExit = $LASTEXITCODE

$retentionFormalPayload = Get-Content -LiteralPath $retentionFormal -Raw |
  ConvertFrom-Json
Write-Host "Retention formal pass: $($retentionFormalPayload.gate_pass)"
if (Test-Path -LiteralPath $robustFormal) {
  $robustFormalPayload = Get-Content -LiteralPath $robustFormal -Raw |
    ConvertFrom-Json
  Write-Host "Robust formal pass: $($robustFormalPayload.gate_pass) (exit $robustFormalExit)"
} else {
  Write-Host "Robust formal JSON missing (exit $robustFormalExit)."
}
if (Test-Path -LiteralPath $ablatedFormal) {
  $ablatedFormalPayload = Get-Content -LiteralPath $ablatedFormal -Raw |
    ConvertFrom-Json
  Write-Host "Ablated formal pass: $($ablatedFormalPayload.gate_pass) (exit $ablatedFormalExit; observational)"
} else {
  Write-Host "Ablated formal JSON missing (exit $ablatedFormalExit)."
}
Write-Host "STOP FOR ANALYSIS. The pre-registered adjudication lives in the robust formal JSON."
