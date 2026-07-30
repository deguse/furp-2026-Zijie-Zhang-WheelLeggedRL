[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$RequiredBranch = 'codex/p2-classical-upper-bound'
$RequiredBase = '4b0210420d3dd35f5c8b74561b49bcb4e8b49034'
$RequiredMjLab = '43e0f3ea9c92ddbb4de9f3bb1ac772d604e3ebf6'
$ScheduleHash = '8fe8548bca85978c164bbd7de39d2d6463cdfd8d7ab91796cf57696b0f64e203'
$IdentificationGainHash = '8fee25a0339dd1e99127cbed912941dc3ad8ef2030ce49a0d310d1563cb87d98'
$CalibrationHash = 'f62648b57bd17a3503bcbdbf58f349f91fcd8de8ef0cf04551c200401233ed01'
$PostureArtifactHash = '3b96fd3dae66ad781b5b875c74184db101c42da02c53dfcc40a5137a6b5de11a'
$StationHash = 'c00e859b3093b4812d54799253accdaeb99171a2cf4028b08bc39e68eaaa7d8a'
$GridHash = '3ba8c0f13667c430c02f4ffdeedcffd97da0e779758b0cb05a86c5fcc09ef628'
$HeightNodes = @(0.2907321708, 0.3092089487, 0.3276857266)
$PitchNodes = @(-0.032, 0.0, 0.032)

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

function Get-FileSha256 {
  param([Parameter(Mandatory)][string]$Path)
  return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-CanonicalScriptSha256 {
  param([Parameter(Mandatory)][string]$Path)
  $content = [IO.File]::ReadAllText($Path).Replace("`r`n", "`n")
  $bytes = [Text.Encoding]::UTF8.GetBytes($content)
  $sha = [Security.Cryptography.SHA256]::Create()
  try {
    return (($sha.ComputeHash($bytes) | ForEach-Object ToString x2) -join '')
  } finally {
    $sha.Dispose()
  }
}

$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $RepoRoot
foreach ($command in @('git', 'nvidia-smi')) {
  if ($null -eq (Get-Command $command -ErrorAction SilentlyContinue)) {
    throw "Missing required command: $command"
  }
}
$SelfHashPath = Join-Path $PSScriptRoot 'run_hybrid_c2_predictor_identification.ps1.sha256'
if (-not (Test-Path -LiteralPath $SelfHashPath -PathType Leaf)) {
  throw "Missing wrapper self-hash: $SelfHashPath"
}
$expectedSelfHash = (Get-Content -Raw -Encoding ASCII -LiteralPath $SelfHashPath).Trim()
$actualSelfHash = Get-CanonicalScriptSha256 -Path $PSCommandPath
if ($expectedSelfHash -ne $actualSelfHash) {
  throw 'C2-j1 wrapper canonical self-hash mismatch.'
}

if ((git branch --show-current).Trim() -ne $RequiredBranch) {
  throw "Expected branch $RequiredBranch."
}
if (@(git status --porcelain).Count -ne 0) {
  throw 'Repository must be clean before C2-j1.'
}
Invoke-NativeChecked -Executable 'git' -Arguments @(
  'cat-file', '-e', ($RequiredBase + '^{commit}')
) -FailureMessage 'Registered C2-j base is not a commit.'
Invoke-NativeChecked -Executable 'git' -Arguments @(
  'merge-base', '--is-ancestor', $RequiredBase, 'HEAD'
) -FailureMessage 'Checkout predates the C2-j preregistration.'
Invoke-NativeChecked -Executable 'git' -Arguments @(
  'fetch', '--quiet', 'origin', $RequiredBranch
) -FailureMessage 'Failed to refresh the remote branch.'
$fullSha = (git rev-parse HEAD).Trim()
if ($fullSha -ne (git rev-parse "origin/$RequiredBranch").Trim()) {
  throw 'Checkout HEAD does not match the remote branch.'
}

$MjLab = (Resolve-Path -LiteralPath (Join-Path $RepoRoot '..\mjlab-main')).Path
if ((git -C $MjLab rev-parse HEAD).Trim() -ne $RequiredMjLab) {
  throw "MjLab must be pinned to $RequiredMjLab."
}
if (@(git -C $MjLab status --porcelain).Count -ne 0) {
  throw 'MjLab checkout must be clean.'
}

$ArtifactRoot = Join-Path $RepoRoot 'docs\experiments\artifacts'
$Schedule = Join-Path $ArtifactRoot 'c1_schedule_candidate24_1f54968_seed1\c1_schedule.json'
$Calibration = Join-Path $ArtifactRoot 'hybrid_runtime_seed1\velocity_calibration_seed1.json'
$Posture = Join-Path $ArtifactRoot 'c1_posture_requalification_seed1\posture_map_seed1_registered_p032.json'
$Station = Join-Path $ArtifactRoot 'c1_posture_requalification_seed1\station_calibration_seed1.json'
foreach ($path in @($Schedule, $Calibration, $Posture, $Station)) {
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    throw "Missing frozen input artifact: $path"
  }
}
$schedulePayload = Get-Content -Raw -Encoding UTF8 -LiteralPath $Schedule | ConvertFrom-Json
$calibrationPayload = Get-Content -Raw -Encoding UTF8 -LiteralPath $Calibration | ConvertFrom-Json
$posturePayload = Get-Content -Raw -Encoding UTF8 -LiteralPath $Posture | ConvertFrom-Json
$stationPayload = Get-Content -Raw -Encoding UTF8 -LiteralPath $Station | ConvertFrom-Json
if (
  $schedulePayload.schedule_hash -ne $ScheduleHash -or
  $schedulePayload.bindings.identification_controller_gain_hash -ne $IdentificationGainHash -or
  $schedulePayload.bindings.identification_calibration_hash -ne $CalibrationHash -or
  $schedulePayload.bindings.posture_artifact_hash -ne $PostureArtifactHash -or
  $calibrationPayload.calibration_hash -ne $CalibrationHash -or
  $posturePayload.posture_artifact_hash -ne $PostureArtifactHash -or
  $stationPayload.station_calibration_hash -ne $StationHash
) {
  throw 'Frozen C2-j1 input binding mismatch.'
}

$Python = Join-Path $RepoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
  throw "Missing project Python: $Python"
}
$env:PYTHONPATH = ('{0};{1}' -f (Join-Path $RepoRoot 'src'), (Join-Path $RepoRoot 'src\hoppertrex_mjlab'))
$env:HOPPERTREX_HYBRID_CONTROLLER_PATH = $Schedule
$env:HOPPERTREX_HYBRID_CALIBRATION_PATH = $Calibration
$env:HOPPERTREX_HYBRID_POSTURE_MAP_PATH = $Posture
$env:HOPPERTREX_HYBRID_STATION_CALIBRATION_PATH = $Station
Remove-Item Env:HOPPERTREX_HYBRID_YAW_CALIBRATION_PATH -ErrorAction SilentlyContinue
Invoke-NativeChecked -Executable $Python -Arguments @(
  '-m', 'hoppertrex_mjlab.scripts.probe_hybrid_c2_predictor_identification', '--help'
) -FailureMessage 'C2-j1 probe --help failed.'

$gpuLines = @(& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)
$gpuLine = $gpuLines | Select-Object -First 1
if ($LASTEXITCODE -ne 0 -or -not $gpuLine) {
  throw 'Unable to query GPU provenance.'
}

$OutputDirectory = Join-Path $RepoRoot ("experiments\c2_predictor_identification_${fullSha}_seed1")
$OutputZip = $OutputDirectory + '.zip'
if ((Test-Path -LiteralPath $OutputDirectory) -or (Test-Path -LiteralPath $OutputZip)) {
  throw "Refusing to overwrite C2-j1 output: $OutputDirectory"
}
$WorkingDirectory = $OutputDirectory + '.incomplete.' + [Guid]::NewGuid().ToString('N')
New-Item -ItemType Directory -Path $WorkingDirectory | Out-Null

try {
  Invoke-NativeChecked -Executable $Python -Arguments @(
    '-u', '-m', 'hoppertrex_mjlab.scripts.probe_hybrid_c2_predictor_identification',
    '--output-dir', $WorkingDirectory,
    '--controller-path', $Schedule,
    '--calibration-path', $Calibration,
    '--posture-map-path', $Posture,
    '--station-calibration-path', $Station,
    '--device', 'cuda:0'
  ) -FailureMessage 'C2-j1 predictor identification failed.'
} catch {
  Write-Warning "Incomplete C2-j1 output retained: $WorkingDirectory"
  throw
}

$ResultPath = Join-Path $WorkingDirectory 'c2_innovation_predictor.json'
$result = Get-Content -Raw -Encoding UTF8 -LiteralPath $ResultPath | ConvertFrom-Json
if (
  $result.schema_version -ne 1 -or
  $result.artifact_type -ne 'c2_innovation_predictor' -or
  $result.probe -ne 'hybrid_c2_predictor_identification_v1' -or
  $result.classification -ne 'PREDICTOR_IDENTIFICATION_QUALIFIED' -or
  [int]$result.protocol.seed -ne 1 -or
  $result.protocol.device -ne 'cuda:0' -or
  [int]$result.protocol.num_envs -ne 32 -or
  [int]$result.protocol.warmup_steps -ne 250 -or
  [int]$result.protocol.collection_steps -ne 2500 -or
  $result.grid_sha256 -ne $GridHash -or
  $result.evidence_eligible -ne $false -or
  $result.detector_fit_eligible -ne $false -or
  $result.promotion_eligible -ne $false -or
  $result.training_eligible -ne $false -or
  $null -ne $result.checkpoint
) {
  throw 'C2-j1 result protocol or classification drifted.'
}
if (
  $result.git_sha -ne $fullSha -or
  $result.mjlab_git_sha -ne $RequiredMjLab -or
  $result.bindings.controller_schedule_hash -ne $ScheduleHash -or
  $result.bindings.identification_controller_gain_hash -ne $IdentificationGainHash -or
  $result.bindings.velocity_calibration_hash -ne $CalibrationHash -or
  $result.bindings.posture_artifact_hash -ne $PostureArtifactHash -or
  $result.bindings.station_calibration_hash -ne $StationHash -or
  $null -ne $result.bindings.yaw_calibration_hash
) {
  throw 'C2-j1 result provenance or binding drifted.'
}
if (@($result.nodes).Count -ne 9 -or @($result.height_nodes).Count -ne 3 -or @($result.pitch_nodes).Count -ne 3) {
  throw 'C2-j1 predictor grid is incomplete.'
}
for ($index = 0; $index -lt 9; $index++) {
  $node = $result.nodes[$index]
  $heightIndex = [Math]::Floor($index / 3)
  $pitchIndex = $index % 3
  $RawPath = Join-Path $WorkingDirectory ([string]$node.raw_file)
  if (
    [int]$node.node_index -ne $index -or
    [double]$node.height_m -ne [double]$HeightNodes[$heightIndex] -or
    [double]$node.pitch_rad -ne [double]$PitchNodes[$pitchIndex] -or
    [int]$node.regression_rank -ne 4 -or
    @($node.heldout_nrmse).Count -ne 2 -or
    [double]$node.heldout_nrmse[0] -gt 0.15 -or
    [double]$node.heldout_nrmse[1] -gt 0.15 -or
    [int]$node.termination_count -ne 0 -or
    [int]$node.timeout_count -ne 0 -or
    [int]$node.non_wheel_contact_count -ne 0 -or
    @($node.raw_shape).Count -ne 2 -or
    [int]$node.raw_shape[0] -ne 2500 -or
    [int]$node.raw_shape[1] -ne 32 -or
    [double]$node.portable_max_abs_target_error_radps -gt 0.00002 -or
    [double]$node.fit_u_min_radps -ge [double]$node.fit_u_max_radps -or
    @($node.a).Count -ne 2 -or
    @($node.b).Count -ne 2 -or
    @($node.c).Count -ne 2 -or
    -not (Test-Path -LiteralPath $RawPath -PathType Leaf) -or
    (Get-FileSha256 -Path $RawPath) -ne $node.raw_sha256
  ) {
    throw "C2-j1 node $index is incomplete, unsafe, or unqualified."
  }
  Invoke-NativeChecked -Executable $Python -Arguments @(
    '-c', 'import sys,numpy as np; p=np.load(sys.argv[1],allow_pickle=False); assert tuple(p.files)==(''z'',''u'',''next_z'',''shaped_posture''); assert p[''z''].shape==(2500,32,2); assert p[''u''].shape==(2500,32,1); assert p[''next_z''].shape==(2500,32,2); assert p[''shaped_posture''].shape==(2500,32,2); assert all(np.isfinite(p[k]).all() for k in p.files)',
    $RawPath
  ) -FailureMessage "C2-j1 raw NPZ schema validation failed for node $index."
}
Invoke-NativeChecked -Executable $Python -Arguments @(
  '-c', 'import json,sys; from pathlib import Path; from hoppertrex_mjlab.hybrid.innovation_detector import parse_innovation_predictor; parse_innovation_predictor(json.loads(Path(sys.argv[1]).read_text(encoding=''utf-8'')))',
  $ResultPath
) -FailureMessage 'Independent predictor parser rejected C2-j1 output.'

$ProtocolNote = [ordered]@{
  git_sha = $fullSha
  mjlab_git_sha = $RequiredMjLab
  gpu = $gpuLine
  classification = $result.classification
  predictor_hash = $result.predictor_hash
  bindings = $result.bindings
  next_step = 'FREEZE_AND_INDEPENDENT_AUDIT_BEFORE_C2_J2'
}
$ProtocolPath = Join-Path $WorkingDirectory 'protocol_note.json'
[IO.File]::WriteAllText(
  $ProtocolPath,
  ($ProtocolNote | ConvertTo-Json -Depth 10),
  [Text.UTF8Encoding]::new($false)
)
$ChecksumPath = Join-Path $WorkingDirectory 'SHA256SUMS.txt'
$checksumLines = @()
foreach ($file in @(Get-ChildItem -LiteralPath $WorkingDirectory -File | Sort-Object Name)) {
  if ($file.Name -eq 'SHA256SUMS.txt') { continue }
  $checksumLines += "$(Get-FileSha256 -Path $file.FullName)  $($file.Name)"
}
[IO.File]::WriteAllLines($ChecksumPath, $checksumLines, [Text.UTF8Encoding]::new($false))

Move-Item -LiteralPath $WorkingDirectory -Destination $OutputDirectory
Compress-Archive -LiteralPath $OutputDirectory -DestinationPath $OutputZip -CompressionLevel Optimal
$ZipSha = Get-FileSha256 -Path $OutputZip
Write-Host "CLASSIFICATION=$($result.classification)"
Write-Host "PREDICTOR_HASH=$($result.predictor_hash)"
Write-Host "RESULT=$OutputDirectory"
Write-Host "ZIP=$OutputZip"
Write-Host "ZIP_SHA256=$ZipSha"
Write-Host 'NEXT=FREEZE_AND_INDEPENDENT_AUDIT_BEFORE_C2_J2'
