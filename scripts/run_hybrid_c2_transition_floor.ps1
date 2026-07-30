[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$RequiredBranch = 'codex/p2-classical-upper-bound'
$RequiredBase = '16da4416c80a3a6bbbe53c39c21a15ad45bdc69f'
$RequiredMjLab = '43e0f3ea9c92ddbb4de9f3bb1ac772d604e3ebf6'
$PredictorHash = 'd1374e4c0c071777bdb3e964e644cad3ba854df4f9976dab016bf9a8d861232d'
$PredictorFileHash = 'fe43855f6c34b440b007c0628e0bf4aacf39d3e8c0a4b209501398c99e4ee877'
$ScheduleHash = '8fe8548bca85978c164bbd7de39d2d6463cdfd8d7ab91796cf57696b0f64e203'
$IdentificationGainHash = '8fee25a0339dd1e99127cbed912941dc3ad8ef2030ce49a0d310d1563cb87d98'
$CalibrationHash = 'f62648b57bd17a3503bcbdbf58f349f91fcd8de8ef0cf04551c200401233ed01'
$PostureArtifactHash = '3b96fd3dae66ad781b5b875c74184db101c42da02c53dfcc40a5137a6b5de11a'
$StationHash = 'c00e859b3093b4812d54799253accdaeb99171a2cf4028b08bc39e68eaaa7d8a'

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
$SelfHashPath = Join-Path $PSScriptRoot 'run_hybrid_c2_transition_floor.ps1.sha256'
if (-not (Test-Path -LiteralPath $SelfHashPath -PathType Leaf)) {
  throw "Missing wrapper self-hash: $SelfHashPath"
}
$expectedSelfHash = (Get-Content -Raw -Encoding ASCII -LiteralPath $SelfHashPath).Trim()
$actualSelfHash = Get-CanonicalScriptSha256 -Path $PSCommandPath
if ($expectedSelfHash -ne $actualSelfHash) {
  throw 'C2-j2 wrapper canonical self-hash mismatch.'
}

if ((git branch --show-current).Trim() -ne $RequiredBranch) {
  throw "Expected branch $RequiredBranch."
}
if (@(git status --porcelain).Count -ne 0) {
  throw 'Repository must be clean before C2-j2.'
}
Invoke-NativeChecked -Executable 'git' -Arguments @(
  'cat-file', '-e', ($RequiredBase + '^{commit}')
) -FailureMessage 'Registered C2-j2 base is not a commit.'
Invoke-NativeChecked -Executable 'git' -Arguments @(
  'merge-base', '--is-ancestor', $RequiredBase, 'HEAD'
) -FailureMessage 'Checkout predates the frozen C2-j1 predictor.'
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
$Predictor = Join-Path $ArtifactRoot 'c2_innovation_predictor_2cccb36_seed1\c2_innovation_predictor.json'
foreach ($path in @($Schedule, $Calibration, $Posture, $Station, $Predictor)) {
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    throw "Missing frozen input artifact: $path"
  }
}
if ((Get-FileSha256 -Path $Predictor) -ne $PredictorFileHash) {
  throw 'Frozen C2-j1 predictor file hash mismatch.'
}
$schedulePayload = Get-Content -Raw -Encoding UTF8 -LiteralPath $Schedule | ConvertFrom-Json
$calibrationPayload = Get-Content -Raw -Encoding UTF8 -LiteralPath $Calibration | ConvertFrom-Json
$posturePayload = Get-Content -Raw -Encoding UTF8 -LiteralPath $Posture | ConvertFrom-Json
$stationPayload = Get-Content -Raw -Encoding UTF8 -LiteralPath $Station | ConvertFrom-Json
$predictorPayload = Get-Content -Raw -Encoding UTF8 -LiteralPath $Predictor | ConvertFrom-Json
if (
  $schedulePayload.schedule_hash -ne $ScheduleHash -or
  $schedulePayload.bindings.identification_controller_gain_hash -ne $IdentificationGainHash -or
  $schedulePayload.bindings.identification_calibration_hash -ne $CalibrationHash -or
  $schedulePayload.bindings.posture_artifact_hash -ne $PostureArtifactHash -or
  $calibrationPayload.calibration_hash -ne $CalibrationHash -or
  $posturePayload.posture_artifact_hash -ne $PostureArtifactHash -or
  $stationPayload.station_calibration_hash -ne $StationHash -or
  $predictorPayload.predictor_hash -ne $PredictorHash
) {
  throw 'Frozen C2-j2 input binding mismatch.'
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
  '-c', 'import json,sys; from pathlib import Path; from hoppertrex_mjlab.hybrid.innovation_detector import parse_innovation_predictor; parse_innovation_predictor(json.loads(Path(sys.argv[1]).read_text(encoding=''utf-8'')))',
  $Predictor
) -FailureMessage 'Frozen predictor parser validation failed.'
Invoke-NativeChecked -Executable $Python -Arguments @(
  '-m', 'hoppertrex_mjlab.scripts.probe_hybrid_c2_transition_floor', '--help'
) -FailureMessage 'C2-j2 probe --help failed.'
Invoke-NativeChecked -Executable $Python -Arguments @(
  '-m', 'hoppertrex_mjlab.scripts.validate_hybrid_c2_transition_floor', '--help'
) -FailureMessage 'C2-j2 validator --help failed.'
Invoke-NativeChecked -Executable $Python -Arguments @(
  '-c', 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() >= 1; x=torch.ones(1,device=''cuda:0''); assert float(x.item())==1.0'
) -FailureMessage 'C2-j2 CUDA preflight failed.'

$gpuLines = @(& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)
$gpuLine = $gpuLines | Select-Object -First 1
if ($LASTEXITCODE -ne 0 -or -not $gpuLine) {
  throw 'Unable to query GPU provenance.'
}

$OutputDirectory = Join-Path $RepoRoot ("experiments\c2_transition_floor_${fullSha}_seed2")
$OutputZip = $OutputDirectory + '.zip'
if ((Test-Path -LiteralPath $OutputDirectory) -or (Test-Path -LiteralPath $OutputZip)) {
  throw "Refusing to overwrite C2-j2 output: $OutputDirectory"
}
$WorkingDirectory = $OutputDirectory + '.incomplete.' + [Guid]::NewGuid().ToString('N')
New-Item -ItemType Directory -Path $WorkingDirectory | Out-Null

try {
  Invoke-NativeChecked -Executable $Python -Arguments @(
    '-u', '-m', 'hoppertrex_mjlab.scripts.probe_hybrid_c2_transition_floor',
    '--output-dir', $WorkingDirectory,
    '--predictor', $Predictor,
    '--device', 'cuda:0'
  ) -FailureMessage 'C2-j2 transition floor failed.'

  Invoke-NativeChecked -Executable $Python -Arguments @(
    '-u', '-m', 'hoppertrex_mjlab.scripts.validate_hybrid_c2_transition_floor',
    '--output-dir', $WorkingDirectory,
    '--predictor', $Predictor,
    '--expected-git-sha', $fullSha,
    '--expected-mjlab-git-sha', $RequiredMjLab
  ) -FailureMessage 'C2-j2 independent raw validation failed.'
} catch {
  Write-Warning "Incomplete C2-j2 output retained: $WorkingDirectory"
  throw
}

$ResultPath = Join-Path $WorkingDirectory 'c2_innovation_floor.json'
$result = Get-Content -Raw -Encoding UTF8 -LiteralPath $ResultPath | ConvertFrom-Json
$allowedClassifications = @(
  'INNOVATION_FLOOR_QUALIFIED',
  'PREDICTOR_DOMAIN_UNCOVERED_STOP',
  'INVALID_INNOVATION_FLOOR'
)
$protocolCellCount = @($result.protocol.cells).Count
$protocolFactorCount = @($result.protocol.threshold_factors).Count
if (
  $allowedClassifications -notcontains $result.classification -or
  $result.git_sha -ne $fullSha -or
  $result.mjlab_git_sha -ne $RequiredMjLab -or
  $result.predictor_hash -ne $PredictorHash -or
  $result.bindings.controller_schedule_hash -ne $ScheduleHash -or
  $result.bindings.identification_controller_gain_hash -ne $IdentificationGainHash -or
  $result.bindings.velocity_calibration_hash -ne $CalibrationHash -or
  $result.bindings.posture_artifact_hash -ne $PostureArtifactHash -or
  $result.bindings.station_calibration_hash -ne $StationHash -or
  $null -ne $result.bindings.yaw_calibration_hash -or
  @($result.cells).Count -ne 10 -or
  $result.protocol.probe -ne 'hybrid_c2_transition_floor_v1' -or
  [int]$result.protocol.seed -ne 2 -or
  $result.protocol.device -ne 'cuda:0' -or
  $protocolCellCount -ne 10 -or
  [int]$result.protocol.envs_per_cell -ne 16 -or
  [int]$result.protocol.settle_steps -ne 200 -or
  [int]$result.protocol.drive_steps -ne 500 -or
  $protocolFactorCount -ne 5 -or
  [double]$result.protocol.height_slew_rate_mps -ne 0.01215 -or
  [double]$result.protocol.pitch_slew_rate_radps -ne 0.07755 -or
  [double]$result.protocol.wheel_slew_radps_per_tick -ne 6.0 -or
  $result.protocol.activation -ne 'integrated_signed_wheel_odometry_lt_0p35m' -or
  $result.protocol.first_tick_no_vote -ne $true -or
  $result.protocol.evidence_eligible -ne $false -or
  $result.protocol.detector_fit_eligible -ne $false -or
  $result.protocol.promotion_eligible -ne $false -or
  $result.protocol.training_eligible -ne $false -or
  $result.evidence_eligible -ne $false -or
  $result.detector_fit_eligible -ne $false -or
  $result.promotion_eligible -ne $false -or
  $result.training_eligible -ne $false -or
  $null -ne $result.checkpoint
) {
  throw 'C2-j2 result protocol, provenance, or classification drifted.'
}
$thresholdTableProperty = $result.PSObject.Properties['threshold_table']
$thresholdTableHashProperty = $result.PSObject.Properties['threshold_table_hash']
$floorHashProperty = $result.PSObject.Properties['floor_hash']
$thresholdTable = if ($null -eq $thresholdTableProperty) { $null } else { $thresholdTableProperty.Value }
$thresholdTableHash = if ($null -eq $thresholdTableHashProperty) { $null } else { $thresholdTableHashProperty.Value }
$floorHash = if ($null -eq $floorHashProperty) { $null } else { $floorHashProperty.Value }
if ($result.classification -eq 'INNOVATION_FLOOR_QUALIFIED') {
  if (
    @($thresholdTable).Count -ne 125 -or
    [string]$thresholdTableHash -notmatch '^[0-9a-f]{64}$' -or
    [string]$floorHash -notmatch '^[0-9a-f]{64}$'
  ) {
    throw 'Qualified C2-j2 result lacks its frozen table or hashes.'
  }
} elseif (
  $null -ne $thresholdTable -or
  $null -ne $thresholdTableHash -or
  $null -ne $floorHash
) {
  throw 'Stopped C2-j2 result must not freeze thresholds.'
}

$nextStep = if ($result.classification -eq 'INNOVATION_FLOOR_QUALIFIED') {
  'FREEZE_AND_INDEPENDENT_AUDIT_BEFORE_C2_J3'
} elseif ($result.classification -eq 'PREDICTOR_DOMAIN_UNCOVERED_STOP') {
  'ARCHIVE_EVIDENCE_AND_STOP_AT_USER_ROUTE_DECISION'
} else {
  'INDEPENDENT_IMPLEMENTATION_DIAGNOSIS_ONLY'
}
$ProtocolNote = [ordered]@{
  git_sha = $fullSha
  mjlab_git_sha = $RequiredMjLab
  gpu = $gpuLine
  classification = $result.classification
  predictor_hash = $result.predictor_hash
  floor_hash = $floorHash
  threshold_table_hash = $thresholdTableHash
  next_step = $nextStep
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

$TemporaryZip = $OutputZip + '.incomplete.' + [Guid]::NewGuid().ToString('N') + '.zip'
try {
  Move-Item -LiteralPath $WorkingDirectory -Destination $OutputDirectory
  Compress-Archive -LiteralPath $OutputDirectory -DestinationPath $TemporaryZip -CompressionLevel Optimal
  Move-Item -LiteralPath $TemporaryZip -Destination $OutputZip
} catch {
  if (Test-Path -LiteralPath $TemporaryZip -PathType Leaf) {
    Remove-Item -LiteralPath $TemporaryZip -Force
  }
  if (Test-Path -LiteralPath $OutputZip -PathType Leaf) {
    Remove-Item -LiteralPath $OutputZip -Force
  }
  if (
    (Test-Path -LiteralPath $OutputDirectory -PathType Container) -and
    -not (Test-Path -LiteralPath $WorkingDirectory)
  ) {
    Move-Item -LiteralPath $OutputDirectory -Destination $WorkingDirectory
  }
  Write-Warning "Incomplete C2-j2 output retained: $WorkingDirectory"
  throw
}
$ZipSha = Get-FileSha256 -Path $OutputZip
Write-Host "CLASSIFICATION=$($result.classification)"
if ($result.classification -eq 'INNOVATION_FLOOR_QUALIFIED') {
  Write-Host "FLOOR_HASH=$floorHash"
  Write-Host "THRESHOLD_TABLE_HASH=$thresholdTableHash"
}
Write-Host "RESULT=$OutputDirectory"
Write-Host "ZIP=$OutputZip"
Write-Host "ZIP_SHA256=$ZipSha"
Write-Host "NEXT=$nextStep"
