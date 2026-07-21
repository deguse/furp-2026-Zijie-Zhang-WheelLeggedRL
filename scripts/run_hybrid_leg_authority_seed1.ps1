param(
  [int] $Seed = 1,
  [int] $MaxIterations = 100,
  [int] $SaveInterval = 25,
  [double[]] $LegScales = @(0.035, 0.070, 0.100),
  [string] $Controller = "C:\mjlab_workspace\hoppertrex_archive\20260712_222216\hybrid_v2\furp-2026-Zijie-Zhang-WheelLeggedRL-hybrid-v2\experiments\hybrid_v2\artifacts\de4ba075ff8b\controller_seed1.json",
  [string] $Calibration = "C:\mjlab_workspace\hoppertrex_archive\20260712_222216\hybrid_v2\furp-2026-Zijie-Zhang-WheelLeggedRL-hybrid-v2\experiments\hybrid_v2\artifacts\de4ba075ff8b\velocity_calibration_seed1.json",
  [string] $YawCalibration = "experiments\hybrid_yaw_calibration_e8e2f06\yaw_calibration_seed1.json",
  [string] $PostureMap = "docs\experiments\artifacts\hybrid_p1_1\posture_map_seed1_floor028_fullhash.json",
  [string] $StationCalibration = "docs\experiments\artifacts\hybrid_p1_1\station_calibration_floor028_fullhash_seed1.json",
  [string] $SourceCheckpoint = "src\hoppertrex_mjlab\logs\rsl_rl\hoppertrex_balance\2026-07-15_19-21-26_hybrid_v2_stage2_probe_9201194_seed1\model_99.pt",
  [string] $SourceGate = "experiments\hybrid_stage2_probe_gate_9201194\seed1_stage1_retention_formal.json"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($Seed -ne 1) {
  throw "P1.2 is pre-registered as a seed-1 development experiment."
}
if ($MaxIterations -ne 100 -or $SaveInterval -ne 25) {
  throw "P1.2 requires 100 iterations and save interval 25."
}
$expectedScales = @(0.035, 0.070, 0.100)
if ($LegScales.Count -ne 3) {
  throw "P1.2 requires exactly three leg scales."
}
for ($index = 0; $index -lt 3; $index++) {
  if ([Math]::Abs($LegScales[$index] - $expectedScales[$index]) -gt 1e-9) {
    throw "P1.2 leg scales must be 0.035, 0.070, 0.100 in that order."
  }
}

$repository = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Set-Location $repository
$python = Join-Path $repository ".venv\Scripts\python.exe"
$logRoot = Join-Path $repository "src\hoppertrex_mjlab\logs\rsl_rl\hoppertrex_balance"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
  throw "Python not found: $python"
}
if (@(git status --short).Count -ne 0) {
  throw "Machine-room checkout is dirty before pull."
}
git pull --ff-only origin codex/hybrid-v2
if ($LASTEXITCODE -ne 0) { throw "git pull failed." }
$head = (git rev-parse HEAD).Trim()
$remoteHead = (git rev-parse origin/codex/hybrid-v2).Trim()
if ($head -ne $remoteHead) { throw "HEAD does not match the remote branch." }
if (@(git status --short).Count -ne 0) {
  throw "Machine-room checkout is dirty after pull."
}
$shortSha = $head.Substring(0, 7)

$required = @(
  $Controller, $Calibration, $YawCalibration, $PostureMap,
  $StationCalibration, $SourceCheckpoint, $SourceGate
)
foreach ($path in $required) {
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

$sourcePath = (Resolve-Path -LiteralPath "src").Path
$packagePath = (Resolve-Path -LiteralPath "src\hoppertrex_mjlab").Path
$env:PYTHONPATH = "$sourcePath;$packagePath"
$env:HOPPERTREX_HYBRID_CONTROLLER_PATH = $controller
$env:HOPPERTREX_HYBRID_CALIBRATION_PATH = $calibration
$env:HOPPERTREX_HYBRID_YAW_CALIBRATION_PATH = $yawCalibration
$env:HOPPERTREX_HYBRID_POSTURE_MAP_PATH = $postureMap
$env:HOPPERTREX_HYBRID_STATION_CALIBRATION_PATH = $stationCalibration

$experimentRoot = Join-Path $repository (
  "experiments\hybrid_leg_authority_${shortSha}_seed${Seed}"
)
if (Test-Path -LiteralPath $experimentRoot) {
  $existingOutputs = @(Get-ChildItem -LiteralPath $experimentRoot -Force)
  if ($existingOutputs.Count -ne 0) {
    throw "Authority output already exists and is non-empty: $experimentRoot"
  }
}
New-Item -ItemType Directory -Path $experimentRoot -Force | Out-Null

# Measure the zero-residual run noise before looking at any trained variant.
$env:HOPPERTREX_HYBRID_LEG_RESIDUAL_SCALE = "0.035"
$preflightCode = @'
import hoppertrex_mjlab.tasks
from mjlab.tasks.registry import load_env_cfg

expected = (0.5, 0.3, 0.035, 0.035, 0.035, 0.035)
for stage in range(6):
    cfg = load_env_cfg(f"HopperTrex-Hybrid-v2-Stage{stage}", play=True)
    actual = tuple(cfg.actions["hybrid_wheel_leg"].action_scales)
    if actual != expected:
        raise RuntimeError(f"Stage{stage} action scales {actual} != {expected}")
print("[PASS] Stage0-5 registration and authority preflight")
'@
& $python -c $preflightCode
if ($LASTEXITCODE -ne 0) {
  throw "Stage0-5 registration and authority preflight failed."
}
$baseline = Join-Path $experimentRoot "zero_residual_repeat.json"
& $python -u -m hoppertrex_mjlab.scripts.probe_hybrid_leg_authority `
  --seed $Seed `
  --device cuda:0 `
  --num-envs 32 `
  --warmup-steps 300 `
  --output $baseline
if ($LASTEXITCODE -ne 0) { throw "Zero-residual repeat probe failed." }
$baselinePayload = Get-Content -LiteralPath $baseline -Raw | ConvertFrom-Json
if (-not [bool]$baselinePayload.baseline_safety_pass) {
  throw "Zero-residual baseline failed safety. Do not train P1.2."
}

$matrixRows = @()
foreach ($scale in $LegScales) {
  $scaleTag = $scale.ToString("0.000", [Globalization.CultureInfo]::InvariantCulture).Replace(".", "p")
  $scaleText = $scale.ToString("0.000", [Globalization.CultureInfo]::InvariantCulture)
  $env:HOPPERTREX_HYBRID_LEG_RESIDUAL_SCALE = $scaleText
  $handoffRun = "hybrid_p12_handoff_${shortSha}_leg${scaleTag}_seed${Seed}"
  $handoffRoot = Join-Path $logRoot $handoffRun
  if (Test-Path -LiteralPath $handoffRoot) {
    throw "Handoff already exists: $handoffRoot"
  }
  $stage3 = Join-Path $handoffRoot "model_stage3.pt"
  $stage4 = Join-Path $handoffRoot "model_stage4.pt"
  $model0 = Join-Path $handoffRoot "model_0.pt"

  & $python -m hoppertrex_mjlab.scripts.rsl_rl.migrate_hybrid_stage `
    --source-checkpoint $source `
    --source-gate-json $sourceGate `
    --yaw-calibration $yawCalibration `
    --posture-map $postureMap `
    --station-calibration $stationCalibration `
    --output-checkpoint $stage3 `
    --source-stage 2 `
    --target-stage 3
  if ($LASTEXITCODE -ne 0) { throw "Stage2->3 migration failed." }
  & $python -m hoppertrex_mjlab.scripts.rsl_rl.migrate_hybrid_stage `
    --source-checkpoint $stage3 `
    --yaw-calibration $yawCalibration `
    --posture-map $postureMap `
    --station-calibration $stationCalibration `
    --output-checkpoint $stage4 `
    --source-stage 3 `
    --target-stage 4
  if ($LASTEXITCODE -ne 0) { throw "Stage3->4 migration failed." }
  & $python -m hoppertrex_mjlab.scripts.rsl_rl.migrate_hybrid_stage `
    --source-checkpoint $stage4 `
    --yaw-calibration $yawCalibration `
    --posture-map $postureMap `
    --station-calibration $stationCalibration `
    --output-checkpoint $model0 `
    --source-stage 4 `
    --target-stage 5 `
    --leg-residual-scale $scaleText
  if ($LASTEXITCODE -ne 0) { throw "Stage4->5 migration failed." }

  $runName = "hybrid_p12_${shortSha}_leg${scaleTag}_seed${Seed}"
  & $python -u -m hoppertrex_mjlab.scripts.rsl_rl.train `
    HopperTrex-Hybrid-v2-Stage5 `
    --env.scene.num-envs 256 `
    --agent.max-iterations 100 `
    --agent.save-interval 25 `
    --agent.seed $Seed `
    --agent.resume True `
    --agent.load-run ".*$handoffRun.*" `
    --agent.load-checkpoint "model_0.pt" `
    --agent.run-name $runName
  if ($LASTEXITCODE -ne 0) { throw "Training failed for leg scale $scaleText." }

  $runs = @(
    Get-ChildItem -LiteralPath $logRoot -Directory |
      Where-Object { $_.Name -like "*_$runName" }
  )
  if ($runs.Count -ne 1) { throw "Expected one run for leg scale $scaleText." }
  $checkpoints = @(
    Get-ChildItem -LiteralPath $runs[0].FullName -Filter "model_*.pt" -File |
      Where-Object { $_.BaseName -match "^model_[0-9]+$" } |
      Sort-Object { [int]($_.BaseName.Substring(6)) }
  )
  $candidates = @($checkpoints | Select-Object -Last 3)
  if ($candidates.Count -ne 3) {
    throw "Scale $scaleText did not produce the pre-registered K=3 saves."
  }
  [array]::Reverse($candidates)
  $selected = $null
  foreach ($candidate in $candidates) {
    $retention = Join-Path $experimentRoot (
      "leg${scaleTag}_$($candidate.BaseName)_retention.json"
    )
    Remove-Item Env:HOPPERTREX_HYBRID_LEG_RESIDUAL_SCALE -ErrorAction SilentlyContinue
    & $python -u -m hoppertrex_mjlab.scripts.rsl_rl.evaluate_hybrid_gate `
      --stage 1 `
      --profile screen `
      --checkpoint-file $candidate.FullName `
      --seed $Seed `
      --device cuda:0 `
      --num-envs 16 `
      --steps 1000 `
      --warmup-steps 300 `
      --window-steps 300 `
      --progress-interval 250 `
      --episode-length-s 1000000000 `
      --output $retention
    $retentionExit = $LASTEXITCODE
    $env:HOPPERTREX_HYBRID_LEG_RESIDUAL_SCALE = $scaleText
    if ($retentionExit -ne 0) { continue }
    $safety = Join-Path $experimentRoot (
      "leg${scaleTag}_$($candidate.BaseName)_robust_screen.json"
    )
    & $python -u -m hoppertrex_mjlab.scripts.rsl_rl.evaluate_hybrid_gate `
      --stage 5 `
      --profile screen `
      --checkpoint-file $candidate.FullName `
      --seed $Seed `
      --device cuda:0 `
      --num-envs 16 `
      --steps 1000 `
      --warmup-steps 300 `
      --window-steps 300 `
      --progress-interval 250 `
      --episode-length-s 1000000000 `
      --output $safety
    if ($LASTEXITCODE -eq 0) {
      $selected = $candidate.FullName
      break
    }
  }
  if ($null -eq $selected) {
    $matrixRows += [ordered]@{
      leg_residual_scale = $scale
      status = "rejected_no_k3_checkpoint_passed_retention_and_safety"
      selected_checkpoint = $null
      authority_json = $null
    }
    Write-Host "[REJECTED] No K=3 checkpoint passed for $scaleText; continuing matrix."
    continue
  }

  $authority = Join-Path $experimentRoot "leg${scaleTag}_authority.json"
  & $python -u -m hoppertrex_mjlab.scripts.probe_hybrid_leg_authority `
    --checkpoint-file $selected `
    --baseline-json $baseline `
    --seed $Seed `
    --device cuda:0 `
    --num-envs 32 `
    --warmup-steps 300 `
    --output $authority
  if ($LASTEXITCODE -ne 0) { throw "Authority probe failed for $scaleText." }
  $authorityPayload = Get-Content -LiteralPath $authority -Raw | ConvertFrom-Json
  $matrixRows += [ordered]@{
    leg_residual_scale = $scale
    status = "measured"
    selected_checkpoint = $selected
    checkpoint_sha256 = (Get-FileHash -LiteralPath $selected -Algorithm SHA256).Hash.ToLowerInvariant()
    authority_json = $authority
    safety_pass = [bool]$authorityPayload.safety_pass
    fractional_improvement = [double]$authorityPayload.candidate_fractional_improvement
    legs_ablated_fractional_improvement = [double]$authorityPayload.legs_ablated_fractional_improvement
    leg_residual_abs_mean = [double]$authorityPayload.candidate.leg_residual_abs_mean
    leg_residual_saturation_rate = [double]$authorityPayload.candidate.leg_residual_saturation_rate
  }
}

$summary = [ordered]@{
  schema_version = 1
  experiment = "hybrid_p1_2_leg_residual_authority"
  git_sha = $head
  seed = $Seed
  baseline_json = $baseline
  baseline_run_noise_s = [double](
    (Get-Content -LiteralPath $baseline -Raw | ConvertFrom-Json).baseline_run_noise_s
  )
  scales = $matrixRows
  decision = "STOP_FOR_ANALYSIS"
}
$summaryPath = Join-Path $experimentRoot "authority_matrix_summary.json"
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $summaryPath -Encoding utf8
Write-Host "[DONE] P1.2 matrix complete: $summaryPath"
Write-Host "Stop for analysis. Do not launch another seed or longer training."
