[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$RequiredBranch = 'codex/p2-classical-upper-bound'
$RequiredImplementation = 'fc940b9f0116608bfdfc2e08f996ecd5e9e76e5e'
$RequiredMjLab = '43e0f3ea9c92ddbb4de9f3bb1ac772d604e3ebf6'
$ControllerFileSha256 = '663ab77f77521581cde77ea2bd8c72c7f395f33b05b62348ef6d82a752aad7fc'
$CalibrationFileSha256 = 'ef002d0d622725509b47c8ff40d8af658fd42f705bdeac67ac35bae4458f889d'
$PostureFileSha256 = 'b8e627f85b53d21dd8d9c26edbe2943151d9bcf9e5864ff998ede5f909118e23'
$UncompensatedFileSha256 = '4ae258eaf73121fd1cffc1186c5611b20a3c95b1ef684060fafa39383b55ca06'
$StationFileSha256 = 'f22a9b66f734004ff14b6586a22a991d527f360806bbbdefe096e9f0474db72a'
$CompensatedFileSha256 = 'c003192963b257c8d497ffd347be2cd60695c5ce8653932403709d8193c88e55'
$ControllerGainHash = '8fee25a0339dd1e99127cbed912941dc3ad8ef2030ce49a0d310d1563cb87d98'
$CalibrationHash = 'f62648b57bd17a3503bcbdbf58f349f91fcd8de8ef0cf04551c200401233ed01'
$PostureMapHash = '4289fb286c6a76a2aca2652d6bcc40acb1bf9c1f70b779a47ceff65c4dca3513'
$PostureArtifactHash = '3b96fd3dae66ad781b5b875c74184db101c42da02c53dfcc40a5137a6b5de11a'
$StationCalibrationHash = 'c00e859b3093b4812d54799253accdaeb99171a2cf4028b08bc39e68eaaa7d8a'
$HeightNodes = @(0.2907321708, 0.3092089487, 0.3276857266)
$PitchNodes = @(-0.032, 0.0, 0.032)
$AffineStateDefinition = 'hybrid_lqr_affine_equilibrium_v3'
$EquilibriumWindowSteps = 100

function Get-FileSha256 {
  param([Parameter(Mandatory)][string]$Path)
  return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Invoke-NativeChecked {
  param(
    [Parameter(Mandatory)][string]$Executable,
    [Parameter(Mandatory)][string[]]$Arguments,
    [Parameter(Mandatory)][string]$FailureMessage
  )
  $previousPreference = $ErrorActionPreference
  try {
    $ErrorActionPreference = 'Continue'
    & $Executable @Arguments
    $exitCode = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $previousPreference
  }
  if ($exitCode -ne 0) {
    throw "$FailureMessage Exit code: $exitCode"
  }
}

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
    throw "$FailureMessage Exit code: $exitCode. Log: $LogPath"
  }
}

function ConvertTo-InvariantString {
  param([Parameter(Mandatory)][double]$Value)
  return $Value.ToString('R', [Globalization.CultureInfo]::InvariantCulture)
}

$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $RepoRoot

foreach ($command in @('git', 'uv', 'nvidia-smi')) {
  if ($null -eq (Get-Command $command -ErrorAction SilentlyContinue)) {
    throw "Missing required command: $command"
  }
}

$SelfHashFile = Join-Path $PSScriptRoot 'run_hybrid_c1_affine_identification_collection.ps1.sha256'
if (-not (Test-Path -LiteralPath $SelfHashFile -PathType Leaf)) {
  throw "Missing wrapper self-hash: $SelfHashFile"
}
$expectedSelfHash = (Get-Content -LiteralPath $SelfHashFile -Raw -Encoding ASCII).Trim()
$actualSelfHash = Get-FileSha256 -Path $PSCommandPath
if ($actualSelfHash -ne $expectedSelfHash) {
  throw 'Affine collection wrapper self-hash mismatch.'
}

$branch = (git branch --show-current).Trim()
if ($branch -ne $RequiredBranch) {
  throw "Expected branch $RequiredBranch, got $branch."
}
if (@(git status --porcelain).Count -ne 0) {
  throw 'Repository must be clean before formal C1 collection.'
}
Invoke-NativeChecked -Executable 'git' -Arguments @(
  'merge-base', '--is-ancestor', $RequiredImplementation, 'HEAD'
) -FailureMessage 'Checkout predates the C1 affine-equilibrium repair.'
Invoke-NativeChecked -Executable 'git' -Arguments @(
  'fetch', '--quiet', 'origin', $RequiredBranch
) -FailureMessage 'Failed to refresh the remote C1 branch.'
$fullSha = (git rev-parse HEAD).Trim()
$remoteSha = (git rev-parse "origin/$RequiredBranch").Trim()
if ($fullSha -ne $remoteSha) {
  throw "Checkout HEAD $fullSha does not match remote HEAD $remoteSha."
}
$shortSha = (git rev-parse --short=7 HEAD).Trim()

$PyProject = Join-Path $RepoRoot 'pyproject.toml'
$MjLabSourceDeclaration = 'mjlab = { path = "../mjlab-main", editable = true }'
if (@(Get-Content -LiteralPath $PyProject -Encoding UTF8) -notcontains $MjLabSourceDeclaration) {
  throw 'pyproject.toml no longer pins the expected editable MjLab source.'
}
$MjLabDeclaredRoot = (Resolve-Path -LiteralPath (Join-Path $RepoRoot '..\mjlab-main')).Path
$mjlabTopLevel = (& git -C $MjLabDeclaredRoot rev-parse --show-toplevel).Trim()
if ($LASTEXITCODE -ne 0 -or -not $mjlabTopLevel) {
  throw 'Editable MjLab source is not a Git checkout.'
}
$MjLabRoot = (Resolve-Path -LiteralPath $mjlabTopLevel).Path
if (@(git -C $MjLabRoot status --porcelain).Count -ne 0) {
  throw 'MjLab checkout must be clean.'
}
$mjlabSha = (git -C $MjLabRoot rev-parse HEAD).Trim()
if ($mjlabSha -ne $RequiredMjLab) {
  throw "MjLab must be pinned to $RequiredMjLab, got $mjlabSha."
}

$Python = Join-Path $RepoRoot '.venv/Scripts/python.exe'
Invoke-NativeChecked -Executable 'uv' -Arguments @(
  'sync', '--frozen', '--python', '3.11'
) -FailureMessage 'uv sync failed.'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
  throw "Missing project Python: $Python"
}
$importedMjLabRoot = (& $Python -c 'import pathlib, mjlab; print(pathlib.Path(mjlab.__file__).resolve().parents[2])').Trim()
if ($LASTEXITCODE -ne 0 -or (Resolve-Path -LiteralPath $importedMjLabRoot).Path -ne $MjLabRoot) {
  throw 'Python does not import the verified editable MjLab checkout.'
}

$RuntimeArtifacts = Join-Path $RepoRoot 'docs/experiments/artifacts/hybrid_runtime_seed1'
$C1Artifacts = Join-Path $RepoRoot 'docs/experiments/artifacts/c1_posture_requalification_seed1'
$Controller = Join-Path $RuntimeArtifacts 'controller_seed1.json'
$Calibration = Join-Path $RuntimeArtifacts 'velocity_calibration_seed1.json'
$Posture = Join-Path $C1Artifacts 'posture_map_seed1_registered_p032.json'
$Uncompensated = Join-Path $C1Artifacts 'balance_uncompensated_seed1.json'
$Station = Join-Path $C1Artifacts 'station_calibration_seed1.json'
$Compensated = Join-Path $C1Artifacts 'balance_compensated_seed1.json'

$ExpectedFiles = [ordered]@{
  $Controller = $ControllerFileSha256
  $Calibration = $CalibrationFileSha256
  $Posture = $PostureFileSha256
  $Uncompensated = $UncompensatedFileSha256
  $Station = $StationFileSha256
  $Compensated = $CompensatedFileSha256
}
foreach ($entry in $ExpectedFiles.GetEnumerator()) {
  if (-not (Test-Path -LiteralPath $entry.Key -PathType Leaf)) {
    throw "Missing frozen artifact: $($entry.Key)"
  }
  $actualHash = Get-FileSha256 -Path $entry.Key
  if ($actualHash -ne $entry.Value) {
    throw "Frozen artifact SHA256 mismatch: $($entry.Key)"
  }
}

$posturePayload = Get-Content -LiteralPath $Posture -Raw -Encoding UTF8 | ConvertFrom-Json
$stationPayload = Get-Content -LiteralPath $Station -Raw -Encoding UTF8 | ConvertFrom-Json
$compensatedPayload = Get-Content -LiteralPath $Compensated -Raw -Encoding UTF8 | ConvertFrom-Json
if (
  $posturePayload.map_hash -ne $PostureMapHash -or
  $posturePayload.posture_artifact_hash -ne $PostureArtifactHash
) {
  throw 'Posture artifact binding mismatch.'
}
if (
  $stationPayload.controller_gain_hash -ne $ControllerGainHash -or
  $stationPayload.posture_map_hash -ne $PostureMapHash -or
  $stationPayload.posture_artifact_hash -ne $PostureArtifactHash -or
  $stationPayload.station_calibration_hash -ne $StationCalibrationHash
) {
  throw 'Station artifact binding mismatch.'
}
if (
  $compensatedPayload.controller_qualified -ne $true -or
  $compensatedPayload.posture_map_qualified -ne $true -or
  $compensatedPayload.station_calibration_qualified -ne $true -or
  $compensatedPayload.calibration_hash -ne $CalibrationHash -or
  $compensatedPayload.station_calibration_hash -ne $StationCalibrationHash -or
  [double]$compensatedPayload.summary.worst_abs_station_drift -gt 0.015 -or
  [double]$compensatedPayload.summary.worst_height_rmse -gt 0.002 -or
  [double]$compensatedPayload.summary.worst_pitch_rmse -gt 0.015 -or
  [double]$compensatedPayload.summary.terminated_events -ne 0.0
) {
  throw 'Frozen compensated posture qualification does not pass C1.'
}
foreach ($cell in @($compensatedPayload.grid_cells) + @($compensatedPayload.vx_checks)) {
  if (
    [double]$cell.terminated_events -ne 0.0 -or
    [double]$cell.non_wheel_contact_rate -ne 0.0
  ) {
    throw 'Frozen compensated qualification contains an unsafe cell.'
  }
}

$env:HOPPERTREX_HYBRID_CONTROLLER_PATH = $Controller
$env:HOPPERTREX_HYBRID_CALIBRATION_PATH = $Calibration
$env:HOPPERTREX_HYBRID_POSTURE_MAP_PATH = $Posture
$env:HOPPERTREX_HYBRID_STATION_CALIBRATION_PATH = $Station
$env:PYTHONPATH = "$(Join-Path $RepoRoot 'src');$(Join-Path $RepoRoot 'src/hoppertrex_mjlab')"
Remove-Item Env:HOPPERTREX_HYBRID_YAW_CALIBRATION_PATH -ErrorAction SilentlyContinue

& (Join-Path $PSScriptRoot 'run_hybrid_c1_schedule_preflight.ps1')
if ($LASTEXITCODE -ne 0) {
  throw 'C1 schedule preflight failed.'
}

$OutputDirectory = Join-Path $RepoRoot (
  'experiments/c1_affine_identification_nodes_' + $shortSha + '_seed1'
)
$OutputZip = $OutputDirectory + '.zip'
if (
  (Test-Path -LiteralPath $OutputDirectory) -or
  (Test-Path -LiteralPath $OutputZip)
) {
  throw "Refusing to overwrite existing C1 output: $OutputDirectory"
}
$runToken = [Guid]::NewGuid().ToString('N')
$WorkingDirectory = $OutputDirectory + '.incomplete.' + $runToken
$WorkingZip = $WorkingDirectory + '.zip'
New-Item -ItemType Directory -Path $WorkingDirectory | Out-Null

# Capture all lines first: piping a native command into Select-Object
# -First stops the pipeline early and leaves a nonzero $LASTEXITCODE.
$gpuLines = @(& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)
$gpuLine = $gpuLines | Select-Object -First 1
if ($LASTEXITCODE -ne 0 -or -not $gpuLine) {
  throw 'Unable to query GPU provenance with nvidia-smi.'
}
# No double quotes in the -c payload: PowerShell 5.1 does not escape
# inner double quotes for native commands (the P1.2 preflight lesson).
$runtimeJson = & $Python -c 'import json, sys, mujoco, torch, warp; print(json.dumps(dict(python=sys.version.split()[0], torch=torch.__version__, torch_cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(), cuda_device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None), mujoco=getattr(mujoco, ''__version__'', None), warp=getattr(warp, ''__version__'', None))))'
if ($LASTEXITCODE -ne 0) {
  throw 'Unable to query Python runtime provenance.'
}
$runtime = $runtimeJson | ConvertFrom-Json
if ($runtime.cuda_available -ne $true) {
  throw 'PyTorch does not expose CUDA on this machine.'
}

$nodeRecords = @()
for ($heightIndex = 0; $heightIndex -lt $HeightNodes.Count; $heightIndex++) {
  for ($pitchIndex = 0; $pitchIndex -lt $PitchNodes.Count; $pitchIndex++) {
    $height = [double]$HeightNodes[$heightIndex]
    $pitch = [double]$PitchNodes[$pitchIndex]
    $stem = "node_h${heightIndex}_p${pitchIndex}"
    $npz = Join-Path $WorkingDirectory ($stem + '.npz')
    $metadataPath = Join-Path $WorkingDirectory ($stem + '.json')
    $log = Join-Path $WorkingDirectory ($stem + '.log')
    Write-Host "[C1] Collecting $stem height=$height pitch=$pitch"
    $collectArgs = @(
      '-u', '-m', 'hoppertrex_mjlab.scripts.collect_hybrid_identification',
      '--output', $npz,
      '--controller-path', $Controller,
      '--calibration-path', $Calibration,
      '--posture-map-path', $Posture,
      '--station-calibration-path', $Station,
      '--height-command', (ConvertTo-InvariantString -Value $height),
      '--pitch-command', (ConvertTo-InvariantString -Value $pitch),
      '--device', 'cuda:0',
      '--num-envs', '32',
      '--steps', '2500',
      '--warmup-steps', '250',
      '--hold-steps', '5',
      '--balance-amplitude', '0.35',
      '--heldout-fraction', '0.20',
      '--seed', '1',
      '--progress-interval', '250'
    )
    Invoke-NativeLogged -Executable $Python -Arguments $collectArgs -LogPath $log -FailureMessage "C1 collection failed for $stem."
    if (
      -not (Test-Path -LiteralPath $npz -PathType Leaf) -or
      -not (Test-Path -LiteralPath $metadataPath -PathType Leaf)
    ) {
      throw "C1 collection did not write both outputs for $stem."
    }
    $metadata = Get-Content -LiteralPath $metadataPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (
      $metadata.git_sha -ne $fullSha -or
      $metadata.device -ne 'cuda:0' -or
      [int]$metadata.num_envs -ne 32 -or
      [int]$metadata.steps -ne 2500 -or
      [int]$metadata.warmup_steps -ne 250 -or
      [int]$metadata.hold_steps -ne 5 -or
      [double]$metadata.balance_amplitude -ne 0.35 -or
      [double]$metadata.heldout_fraction -ne 0.20 -or
      [int]$metadata.seed -ne 1 -or
      [Math]::Abs([double]$metadata.height_command - $height) -gt 1.0e-12 -or
      [Math]::Abs([double]$metadata.pitch_command - $pitch) -gt 1.0e-12 -or
      $metadata.state_definition_version -ne $AffineStateDefinition -or
      $metadata.input_name -ne 'delta_actual_signed_balance_wheel_velocity_target' -or
      [int]$metadata.equilibrium_window_steps -ne $EquilibriumWindowSteps -or
      @($metadata.equilibrium_state).Count -ne 4 -or
      @($metadata.equilibrium_input).Count -ne 1 -or
      @($metadata.delta_state_mean).Count -ne 4 -or
      @($metadata.delta_input_mean).Count -ne 1 -or
      $metadata.controller.gain_hash -ne $ControllerGainHash -or
      $metadata.calibration_hash -ne $CalibrationHash -or
      $metadata.posture_artifact_hash -ne $PostureArtifactHash -or
      $metadata.station_calibration_hash -ne $StationCalibrationHash
    ) {
      throw "C1 metadata protocol or provenance mismatch for $stem."
    }
    $nodeRecords += [ordered]@{
      stem = $stem
      height_m = $height
      pitch_rad = $pitch
      equilibrium_pitch_rad = [double]$metadata.equilibrium_pitch
      equilibrium_state = @($metadata.equilibrium_state)
      equilibrium_input = [double]$metadata.equilibrium_input[0]
      delta_state_mean = @($metadata.delta_state_mean)
      delta_input_mean = [double]$metadata.delta_input_mean[0]
      valid_sample_count = [int]$metadata.valid_sample_count
      discarded_sample_count = [int]$metadata.discarded_sample_count
      npz_sha256 = Get-FileSha256 -Path $npz
      metadata_sha256 = Get-FileSha256 -Path $metadataPath
      log_sha256 = Get-FileSha256 -Path $log
    }
  }
}

$CenterSmokePath = Join-Path $WorkingDirectory 'affine_center_smoke.json'
$CenterSmokeLog = Join-Path $WorkingDirectory 'affine_center_smoke.log'
$centerSmokeArgs = @(
  '-u', '-m', 'hoppertrex_mjlab.scripts.evaluate_hybrid_c1_affine_center_smoke',
  '--nodes-dir', $WorkingDirectory,
  '--output', $CenterSmokePath,
  '--compensated-qualification', $Compensated,
  '--git-sha', $fullSha,
  '--mjlab-git-sha', $mjlabSha,
  '--task', 'HopperTrex-Hybrid-v2-Stage3',
  '--device', 'cuda:0',
  '--num-envs', '16',
  '--settle-steps', '100',
  '--measure-steps', '200',
  '--vx-check', '0.05'
)
Invoke-NativeLogged -Executable $Python -Arguments $centerSmokeArgs -LogPath $CenterSmokeLog -FailureMessage 'C1 affine center smoke failed.'
if (-not (Test-Path -LiteralPath $CenterSmokePath -PathType Leaf)) {
  throw 'Affine center smoke did not write its adjudication.'
}
$centerSmoke = Get-Content -LiteralPath $CenterSmokePath -Raw -Encoding UTF8 | ConvertFrom-Json
$validSmokeClassifications = @(
  'AFFINE_CENTER_SMOKE_HAS_CANDIDATES',
  'AFFINE_CENTER_SMOKE_NO_CANDIDATE_STOP'
)
if (
  $validSmokeClassifications -notcontains $centerSmoke.classification -or
  $centerSmoke.incumbent.flat_gate_passed -ne $true -or
  $centerSmoke.affine_incumbent.flat_gate_passed -ne $true -or
  [double]$centerSmoke.affine_incumbent.anchor_alpha -ne 0.0 -or
  $centerSmoke.git_sha -ne $fullSha -or
  $centerSmoke.collection_git_sha -ne $fullSha -or
  $centerSmoke.mjlab_git_sha -ne $mjlabSha -or
  [int]$centerSmoke.completed_candidate_count -ne 27 -or
  [int]$centerSmoke.completed_node_fit_count -ne 243 -or
  [int]$centerSmoke.fit_qualification.candidate_count -ne 27 -or
  [int]$centerSmoke.fit_qualification.node_fit_count -ne 243 -or
  [int]$centerSmoke.fit_qualification.minimum_controllability_rank -ne 4 -or
  [double]$centerSmoke.fit_qualification.maximum_heldout_nrmse -gt 0.15 -or
  [int]$centerSmoke.fit_qualification.fallback_count -ne 0 -or
  @($centerSmoke.candidates).Count -ne 27 -or
  [int]$centerSmoke.passed_candidate_count -lt 0 -or
  [int]$centerSmoke.passed_candidate_count -gt 27 -or
  $centerSmoke.bindings.controller_gain_hash -ne $ControllerGainHash -or
  $centerSmoke.bindings.velocity_calibration_hash -ne $CalibrationHash -or
  $centerSmoke.bindings.posture_artifact_hash -ne $PostureArtifactHash -or
  $centerSmoke.bindings.station_calibration_hash -ne $StationCalibrationHash -or
  $centerSmoke.evidence_eligible -ne $true -or
  $centerSmoke.promotion_eligible -ne $false -or
  $centerSmoke.training_eligible -ne $false -or
  $null -ne $centerSmoke.checkpoint
) {
  throw 'Affine center smoke adjudication is incomplete or invalid.'
}
$passedCandidateCount = [int]$centerSmoke.passed_candidate_count
$nextStep = [string]$centerSmoke.next_step
if (
  ($centerSmoke.classification -eq 'AFFINE_CENTER_SMOKE_HAS_CANDIDATES' -and
    ($passedCandidateCount -le 0 -or $nextStep -ne 'DOWNLOAD_FOR_REVIEW')) -or
  ($centerSmoke.classification -eq 'AFFINE_CENTER_SMOKE_NO_CANDIDATE_STOP' -and
    ($passedCandidateCount -ne 0 -or $nextStep -ne 'STOP'))
) {
  throw 'Affine center smoke classification, pass count, and next step disagree.'
}

$protocol = [ordered]@{
  schema_version = 1
  kind = 'c1_affine_identification_collection'
  git_sha = $fullSha
  mjlab_git_sha = $mjlabSha
  seed = 1
  device = 'cuda:0'
  gpu = [string]$gpuLine
  runtime = $runtime
  protocol = [ordered]@{
    height_nodes_m = $HeightNodes
    pitch_nodes_rad = $PitchNodes
    num_envs = 32
    steps = 2500
    warmup_steps = 250
    equilibrium_window_steps = $EquilibriumWindowSteps
    hold_steps = 5
    balance_amplitude = 0.35
    heldout_fraction = 0.20
  }
  bindings = [ordered]@{
    controller_gain_hash = $ControllerGainHash
    velocity_calibration_hash = $CalibrationHash
    posture_artifact_hash = $PostureArtifactHash
    station_calibration_hash = $StationCalibrationHash
  }
  nodes = $nodeRecords
  center_smoke = [ordered]@{
    classification = [string]$centerSmoke.classification
    incumbent_passed = [bool]$centerSmoke.incumbent.flat_gate_passed
    affine_incumbent_passed = [bool]$centerSmoke.affine_incumbent.flat_gate_passed
    passed_candidate_count = $passedCandidateCount
  }
  evidence_eligible = $true
  promotion_eligible = $false
  training_eligible = $false
  checkpoint = $null
  yaw_calibration_hash = $null
  state_definition_version = $AffineStateDefinition
  next_step = $nextStep
}
$protocolPath = Join-Path $WorkingDirectory 'protocol_note.json'
$protocol | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $protocolPath -Encoding UTF8

$checksumPath = Join-Path $WorkingDirectory 'SHA256SUMS.txt'
$hashLines = Get-ChildItem -LiteralPath $WorkingDirectory -File |
  Where-Object { $_.Name -ne 'SHA256SUMS.txt' } |
  Sort-Object Name |
  ForEach-Object {
    $hash = Get-FileSha256 -Path $_.FullName
    "$hash  $($_.Name)"
  }
$hashLines | Set-Content -LiteralPath $checksumPath -Encoding ASCII

Compress-Archive -Path (Join-Path $WorkingDirectory '*') -DestinationPath $WorkingZip -CompressionLevel Optimal
Move-Item -LiteralPath $WorkingZip -Destination $OutputZip
try {
  Move-Item -LiteralPath $WorkingDirectory -Destination $OutputDirectory
} catch {
  Remove-Item -LiteralPath $OutputZip -Force -ErrorAction SilentlyContinue
  throw
}
$zipSha = Get-FileSha256 -Path $OutputZip

Write-Host '[PASS] C1 affine nine-node identification collection complete.'
Write-Host "RESULT=$OutputDirectory"
Write-Host "ZIP=$OutputZip"
Write-Host "ZIP_SHA256=$zipSha"
Write-Host "CLASSIFICATION=$($centerSmoke.classification)"
Write-Host "PASSED_CANDIDATES=$passedCandidateCount/27"
Write-Host "NEXT=$nextStep"
