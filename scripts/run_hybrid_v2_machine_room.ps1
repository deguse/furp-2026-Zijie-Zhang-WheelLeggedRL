[CmdletBinding()]
param(
  [ValidateSet('Preflight', 'Smoke', 'Identify', 'Calibrate', 'Stage0Probe', 'Stage0', 'All')]
  [string]$Phase = 'All',
  [string]$Python,
  [string]$Device = 'cuda:0',
  [switch]$SkipSmoke,
  [int[]]$Seeds
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $RepoRoot
$RequiredBaseSha = 'c22f2149f9911cbfd31ce3fa88d2d6ff9c1e4c4f'

function Invoke-NativeLogged {
  param(
    [Parameter(Mandatory)] [string]$Executable,
    [Parameter(Mandatory)] [string[]]$Arguments,
    [Parameter(Mandatory)] [string]$LogPath
  )
  $previousPreference = $ErrorActionPreference
  try {
    $ErrorActionPreference = 'Continue'
    & $Executable @Arguments 2>&1 |
      Tee-Object -FilePath $LogPath
    $exitCode = $LASTEXITCODE
  } catch {
    $_ | Out-File -LiteralPath $LogPath -Encoding utf8 -Append
    Write-Host $_
    $exitCode = -1
  } finally {
    $ErrorActionPreference = $previousPreference
  }
  if ($exitCode -ne 0) {
    throw "Command failed with exit code $exitCode. Log: $LogPath"
  }
}

function Test-PythonCandidate {
  param([string]$Candidate)
  if (-not (Test-Path -LiteralPath $Candidate -PathType Leaf)) {
    return $false
  }
  $previousPreference = $ErrorActionPreference
  try {
    $ErrorActionPreference = 'Continue'
    & $Candidate -c 'import mjlab, numpy, scipy, torch; assert torch.cuda.is_available()' *> $null
    $candidateExitCode = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $previousPreference
  }
  return $candidateExitCode -eq 0
}

function Find-HybridPython {
  if ($Python) {
    $resolved = (Resolve-Path -LiteralPath $Python).Path
    if (-not (Test-PythonCandidate $resolved)) {
      throw "Explicit Python cannot import MjLab with CUDA: $resolved"
    }
    return $resolved
  }

  $repoParent = Split-Path $RepoRoot -Parent
  if ((Split-Path $repoParent -Leaf) -eq '.worktrees') {
    $mainRepo = Split-Path $repoParent -Parent
    $workspace = Split-Path $mainRepo -Parent
  } else {
    $mainRepo = $RepoRoot
    $workspace = $repoParent
  }
  $candidates = [System.Collections.Generic.List[string]]::new()
  if ($env:VIRTUAL_ENV) {
    $candidates.Add((Join-Path $env:VIRTUAL_ENV 'Scripts/python.exe'))
  }
  foreach ($path in @(
    (Join-Path $RepoRoot '.venv/Scripts/python.exe'),
    (Join-Path $mainRepo '.venv/Scripts/python.exe'),
    (Join-Path $workspace '.venv/Scripts/python.exe'),
    (Join-Path $workspace 'mjlab-main/.venv/Scripts/python.exe')
  )) {
    $candidates.Add($path)
  }
  Get-ChildItem -LiteralPath $workspace -Directory -Force -ErrorAction SilentlyContinue |
    ForEach-Object {
      $candidates.Add((Join-Path $_.FullName '.venv/Scripts/python.exe'))
    }

  foreach ($candidate in $candidates | Select-Object -Unique) {
    Write-Host "Checking Python: $candidate"
    if (Test-PythonCandidate $candidate) {
      return (Resolve-Path -LiteralPath $candidate).Path
    }
  }
  throw 'No workspace virtual environment can import MjLab with CUDA. Pass -Python explicitly.'
}

function Assert-ControllerQualified {
  param([string]$ControllerPath)
  $payload = Get-Content -LiteralPath $ControllerPath -Raw | ConvertFrom-Json
  $nrmse = [double]$payload.heldout_one_step_nrmse.maximum
  if ($payload.controller_type -ne 'lqr') {
    throw "Controller is $($payload.controller_type), not LQR. Stop."
  }
  if ([int]$payload.controllability_rank -ne 4) {
    throw "Controllability rank is $($payload.controllability_rank), not 4. Stop."
  }
  if ([double]::IsNaN($nrmse) -or [double]::IsInfinity($nrmse) -or $nrmse -gt 0.15) {
    throw "Held-out NRMSE is $nrmse, above 0.15. Stop."
  }
  if (@($payload.fallback_reasons).Count -ne 0) {
    throw "Controller has fallback reasons: $($payload.fallback_reasons -join '; ')"
  }
  if (-not $payload.gain_hash -or $payload.gain_hash.Length -ne 64) {
    throw 'Controller gain hash is invalid.'
  }
  return $payload
}

$fullSha = (git rev-parse HEAD).Trim()
git merge-base --is-ancestor $RequiredBaseSha HEAD
if ($LASTEXITCODE -ne 0) {
  throw "HEAD $fullSha does not contain required Hybrid v2 base $RequiredBaseSha."
}
if (git status --porcelain) {
  throw 'Git worktree is not clean.'
}

$pythonExe = Find-HybridPython
$env:PYTHONPATH = (Resolve-Path (Join-Path $RepoRoot 'src')).Path
$shortSha = (git rev-parse --short=12 HEAD).Trim()
$artifactRoot = Join-Path $RepoRoot 'experiments'
$artifactRoot = Join-Path $artifactRoot 'hybrid_v2'
$artifactRoot = Join-Path $artifactRoot 'artifacts'
$artifactBase = $artifactRoot
New-Item -ItemType Directory -Force $artifactBase | Out-Null
$legacyArtifactRoot = Get-ChildItem -LiteralPath $RepoRoot -Directory |
  Where-Object { $_.Name -like 'experimentshybrid_v2artifacts*' } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
if ($legacyArtifactRoot) {
  $legacyPrefix = 'experimentshybrid_v2artifacts'
  $legacySha = $legacyArtifactRoot.Name.Substring($legacyPrefix.Length)
  $legacyDestination = Join-Path $artifactBase $legacySha
  if (-not (Test-Path $legacyDestination)) {
    Move-Item -LiteralPath $legacyArtifactRoot.FullName -Destination $legacyDestination
    Write-Host "Migrated artifacts from malformed legacy path to: $legacyDestination"
  }
}
$artifactRoot = Join-Path $artifactBase $shortSha
$currentIdentification = Join-Path $artifactRoot 'identification_seed1.npz'
$reusableIdentification = Get-ChildItem -LiteralPath $artifactBase -Recurse -Filter 'identification_seed1.npz' -File -ErrorAction SilentlyContinue |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
if (-not (Test-Path $currentIdentification) -and $reusableIdentification) {
  New-Item -ItemType Directory -Force $artifactRoot | Out-Null
  Copy-Item -LiteralPath $reusableIdentification.FullName -Destination $currentIdentification
  $sourceMetadata = $reusableIdentification.FullName.Replace('.npz', '.json')
  if (-not (Test-Path -LiteralPath $sourceMetadata -PathType Leaf)) {
    throw "Reusable identification has no JSON sidecar: $sourceMetadata"
  }
  Copy-Item -LiteralPath $sourceMetadata -Destination $currentIdentification.Replace('.npz', '.json')
  Write-Host "Copied reusable identification into current SHA artifacts: $artifactRoot"
}
New-Item -ItemType Directory -Force $artifactRoot | Out-Null

Write-Host "Selected Python: $pythonExe"
& $pythonExe -c 'import inspect,mjlab,torch; print(inspect.getfile(mjlab)); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0))'
if ($LASTEXITCODE -ne 0) { throw 'Runtime preflight failed.' }
nvidia-smi | Out-File (Join-Path $artifactRoot 'nvidia_smi.txt') -Encoding utf8

if ($Phase -eq 'Preflight') {
  Write-Host '[PASS] Preflight complete.'
  exit 0
}

if (-not $SkipSmoke) {
  $smoke = Join-Path $artifactRoot 'identification_smoke.npz'
  $smokeLog = Join-Path $artifactRoot 'identification_smoke.log'
  Invoke-NativeLogged $pythonExe @(
    '-u', '-m', 'hoppertrex_mjlab.scripts.collect_hybrid_identification',
    '--output', $smoke, '--device', $Device, '--num-envs', '2',
    '--steps', '12', '--warmup-steps', '2', '--hold-steps', '2',
    '--balance-amplitude', '0.20', '--heldout-fraction', '0.20',
    '--progress-interval', '12', '--seed', '1'
  ) $smokeLog
  if (-not (Test-Path $smoke) -or -not (Test-Path $smoke.Replace('.npz', '.json'))) {
    throw 'Smoke completed without both NPZ and JSON artifacts.'
  }
  Write-Host '[PASS] GPU identification smoke complete.'
}

if ($Phase -eq 'Smoke') { exit 0 }

$identification = Join-Path $artifactRoot 'identification_seed1.npz'
$identificationLog = Join-Path $artifactRoot 'identification_seed1.log'
if (-not (Test-Path $identification)) {
  Invoke-NativeLogged $pythonExe @(
    '-u', '-m', 'hoppertrex_mjlab.scripts.collect_hybrid_identification',
    '--output', $identification, '--device', $Device, '--num-envs', '32',
    '--steps', '2500', '--warmup-steps', '250', '--hold-steps', '5',
    '--balance-amplitude', '0.35', '--heldout-fraction', '0.20',
    '--progress-interval', '250', '--seed', '1'
  ) $identificationLog
}

$controller = Join-Path $artifactRoot 'controller_seed1.json'
$controllerLog = Join-Path $artifactRoot 'controller_seed1.log'
Invoke-NativeLogged $pythonExe @(
  '-u', '-m', 'hoppertrex_mjlab.scripts.identify_hybrid_controller',
  '--input', $identification, '--output', $controller,
  '--q-diag', '20.0', '2.0', '4.0', '0.5', '--r', '1.0',
  '--pd-gain', '8.0', '1.0', '3.0', '0.2', '--nrmse-limit', '0.15'
) $controllerLog
$controllerPayload = Assert-ControllerQualified $controller
Write-Host "[PASS] Qualified LQR: $($controllerPayload.gain_hash)"

if ($Phase -eq 'Identify') { exit 0 }

$env:HOPPERTREX_HYBRID_CONTROLLER_PATH = (Resolve-Path $controller).Path
$calibration = Join-Path $artifactRoot 'velocity_calibration_seed1.json'
$calibrationRoot = Join-Path $artifactRoot 'velocity_calibration_sweep'
if ($Phase -eq 'Calibrate' -or ($Phase -eq 'All' -and -not (Test-Path -LiteralPath $calibration))) {
  Invoke-NativeLogged $pythonExe @(
    '-u', '-m', 'hoppertrex_mjlab.scripts.calibrate_hybrid_velocity',
    '--controller', $controller, '--output', $calibration,
    '--work-dir', $calibrationRoot, '--device', $Device,
    '--seed', '1', '--num-envs', '16', '--steps', '600',
    '--warmup-steps', '150', '--window-steps', '300'
  ) (Join-Path $artifactRoot 'velocity_calibration.log')
  Write-Host "[PASS] Calibration sweep completed: $calibration"
}
if ($Phase -eq 'Calibrate') { exit 0 }
if (-not (Test-Path -LiteralPath $calibration -PathType Leaf)) {
  throw 'gate_failed: no velocity calibration artifact. Run -Phase Calibrate first.'
}
$env:HOPPERTREX_HYBRID_CALIBRATION_PATH = (Resolve-Path $calibration).Path
$calibrationPayload = Get-Content -LiteralPath $calibration -Raw | ConvertFrom-Json

if ($Phase -eq 'Stage0Probe' -or $Phase -eq 'All') {
  $probeRoot = Join-Path $artifactRoot 'stage0_probe'
  New-Item -ItemType Directory -Force $probeRoot | Out-Null
  $probeJson = Join-Path $probeRoot 'seed1.json'
  Invoke-NativeLogged $pythonExe @(
    '-u', '-m', 'hoppertrex_mjlab.scripts.rsl_rl.evaluate_hybrid_gate',
    '--stage', '0', '--seed', '1', '--device', $Device,
    '--num-envs', '16', '--steps', '3000', '--warmup-steps', '300',
    '--window-steps', '800', '--progress-interval', '500',
    '--episode-length-s', '1.0e9',
    '--controller-gain-hash', $controllerPayload.gain_hash,
    '--output', $probeJson
  ) (Join-Path $probeRoot 'seed1.log')
  $probePayload = Get-Content -LiteralPath $probeJson -Raw | ConvertFrom-Json
  if ($probePayload.gate_pass -ne $true) {
    throw 'gate_failed: calibrated Stage0Probe seed 1 did not pass.'
  }
  if ($probePayload.calibration_hash -ne $calibrationPayload.calibration_hash) {
    throw 'gate_failed: Stage0Probe calibration hash does not match the selected calibration.'
  }
  Write-Host "[PASS] Calibrated Stage0Probe seed 1: $probeJson"
}
if ($Phase -eq 'Stage0Probe') { exit 0 }

$probeRequired = Join-Path $artifactRoot 'stage0_probe/seed1.json'
if (-not (Test-Path -LiteralPath $probeRequired -PathType Leaf)) {
  throw 'gate_failed: Stage0 requires a passing Stage0Probe artifact.'
}
$probeRequiredPayload = Get-Content -LiteralPath $probeRequired -Raw | ConvertFrom-Json
if ($probeRequiredPayload.gate_pass -ne $true) {
  throw 'gate_failed: Stage0Probe artifact is not passing.'
}
if ($probeRequiredPayload.calibration_hash -ne $calibrationPayload.calibration_hash) {
  throw 'gate_failed: Stage0Probe was produced by a different calibration artifact.'
}

$stage0Root = Join-Path $artifactRoot 'stage0_gate_calibrated'
New-Item -ItemType Directory -Force $stage0Root | Out-Null
$failedSeeds = [System.Collections.Generic.List[int]]::new()
$stage0Seeds = if ($Seeds -and $Seeds.Count -gt 0) { $Seeds } else { @(1, 2, 3) }
foreach ($seed in $stage0Seeds) {
  $seedJson = Join-Path $stage0Root "seed$seed.json"
  $seedLog = Join-Path $stage0Root "seed$seed.log"
  try {
    Invoke-NativeLogged $pythonExe @(
      '-u', '-m', 'hoppertrex_mjlab.scripts.rsl_rl.evaluate_hybrid_gate',
      '--stage', '0', '--seed', "$seed", '--device', $Device,
      '--num-envs', '16', '--steps', '3000', '--warmup-steps', '300',
      '--window-steps', '800', '--progress-interval', '500',
      '--episode-length-s', '1.0e9',
      '--controller-gain-hash', $controllerPayload.gain_hash,
      '--output', $seedJson
    ) $seedLog
  } catch {
    Write-Warning "Stage0 seed $seed failed: $_"
    $failedSeeds.Add($seed)
  }
}
if ($failedSeeds.Count -ne 0) {
  throw "gate_failed: Stage0 failed seeds: $($failedSeeds -join ', '). Artifacts were retained."
}

$stage0SeedKey = (@($stage0Seeds | Sort-Object) -join ',')
if ($stage0Seeds.Count -ne 3 -or $stage0SeedKey -ne '1,2,3') {
  Write-Host "[PASS] Requested Stage0 seeds: $($stage0Seeds -join ', '). No formal aggregate was produced."
  exit 0
}

$aggregate = Join-Path $stage0Root 'aggregate.json'
Invoke-NativeLogged $pythonExe @(
  '-u', '-m', 'hoppertrex_mjlab.scripts.rsl_rl.evaluate_hybrid_gate',
  '--aggregate-input',
  (Join-Path $stage0Root 'seed1.json'),
  (Join-Path $stage0Root 'seed2.json'),
  (Join-Path $stage0Root 'seed3.json'),
  '--output', $aggregate
) (Join-Path $stage0Root 'aggregate.log')
$aggregatePayload = Get-Content -LiteralPath $aggregate -Raw | ConvertFrom-Json
if ($aggregatePayload.gate_pass -ne $true) {
  throw 'Stage0 aggregate did not pass.'
}
Write-Host "[PASS] Stage0 three-seed gate: $aggregate"
Write-Host 'STOP: inspect Stage0 in Viser before posture sweep or PPO.'
