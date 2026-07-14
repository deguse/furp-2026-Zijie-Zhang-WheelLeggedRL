param(
  [string] $Controller = "C:\mjlab_workspace\hoppertrex_archive\20260712_222216\hybrid_v2\furp-2026-Zijie-Zhang-WheelLeggedRL-hybrid-v2\experiments\hybrid_v2\artifacts\de4ba075ff8b\controller_seed1.json",
  [string] $Calibration = "C:\mjlab_workspace\hoppertrex_archive\20260712_222216\hybrid_v2\furp-2026-Zijie-Zhang-WheelLeggedRL-hybrid-v2\experiments\hybrid_v2\artifacts\de4ba075ff8b\velocity_calibration_seed1.json",
  [string] $SourceCheckpoint = "src\hoppertrex_mjlab\logs\rsl_rl\hoppertrex_balance\2026-07-14_21-01-36_hybrid_v2_stage1b_probe_seed1\model_99.pt",
  [string] $SourceGate = "experiments\hybrid_stage1b_probe_gate\seed1_formal_stage1b.json"
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
$checkpoint = $checkpoints[-1].FullName
Write-Host "Stage2 candidate: $checkpoint"
Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256 | Format-List

$gateRoot = "experiments\hybrid_stage2_probe_gate_$shortSha"
New-Item -ItemType Directory -Path $gateRoot -Force | Out-Null
$retentionScreen = Join-Path $gateRoot "seed1_stage1_retention_screen.json"
$planarScreen = Join-Path $gateRoot "seed1_stage2_planar_screen.json"
$retentionFormal = Join-Path $gateRoot "seed1_stage1_retention_formal.json"
$planarFormal = Join-Path $gateRoot "seed1_stage2_planar_formal.json"

& $python -m hoppertrex_mjlab.scripts.rsl_rl.evaluate_hybrid_gate `
  --stage 1 `
  --profile screen `
  --checkpoint-file $checkpoint `
  --seed 1 `
  --device cuda:0 `
  --num-envs 16 `
  --steps 1000 `
  --warmup-steps 300 `
  --window-steps 300 `
  --progress-interval 250 `
  --episode-length-s 1000000000 `
  --output $retentionScreen
if ($LASTEXITCODE -ne 0) {
  throw "Stage1 retention screen failed. Stop; Stage1-B remains frozen."
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
