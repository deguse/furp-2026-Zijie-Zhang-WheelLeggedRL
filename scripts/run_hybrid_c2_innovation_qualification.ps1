[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$RequiredBranch = 'codex/p2-classical-upper-bound'
# This is the frozen C2-j2 prerequisite base. Exact C2-j3 implementation
# identity is pinned independently by the three canonical source hashes below.
$RequiredBase = '43c379b919d36465ef4e666254e708e26b1a2c6e'
$RequiredMjLab = '43e0f3ea9c92ddbb4de9f3bb1ac772d604e3ebf6'
$CoreCanonicalHash = '8ff70de0ae6bb47827509860f85337e85095acebddcaa0ecc1b4b996332751fe'
$ProducerCanonicalHash = '045c21a1a779cfba38672f7c589c049ab83086953ffe3ac132f852965b415cf6'
$ValidatorCanonicalHash = '76aaf3d9e5781e0d3c7ba35aeaa99c640f8aaee1f51e4449cf06cd6f0836bd4d'
$Task = 'HopperTrex-Hybrid-v2-Stage5'
$ScheduleFileHash = '9b21125e7cc48be3ea61e12a67171a855892ad3ced1f54b3176ed979e76224ec'
$CalibrationFileHash = 'ef002d0d622725509b47c8ff40d8af658fd42f705bdeac67ac35bae4458f889d'
$PostureFileHash = 'b8e627f85b53d21dd8d9c26edbe2943151d9bcf9e5864ff998ede5f909118e23'
$StationFileHash = 'f22a9b66f734004ff14b6586a22a991d527f360806bbbdefe096e9f0474db72a'
$PredictorFileHash = 'fe43855f6c34b440b007c0628e0bf4aacf39d3e8c0a4b209501398c99e4ee877'
$FloorFileHash = 'cffc0e0877025af2cfc2e7292cf7e7b0ef79e6f3c743f3ec23589f38a296b4fd'
$ScheduleHash = '8fe8548bca85978c164bbd7de39d2d6463cdfd8d7ab91796cf57696b0f64e203'
$IdentificationGainHash = '8fee25a0339dd1e99127cbed912941dc3ad8ef2030ce49a0d310d1563cb87d98'
$CalibrationHash = 'f62648b57bd17a3503bcbdbf58f349f91fcd8de8ef0cf04551c200401233ed01'
$PostureArtifactHash = '3b96fd3dae66ad781b5b875c74184db101c42da02c53dfcc40a5137a6b5de11a'
$StationHash = 'c00e859b3093b4812d54799253accdaeb99171a2cf4028b08bc39e68eaaa7d8a'
$PredictorHash = 'd1374e4c0c071777bdb3e964e644cad3ba854df4f9976dab016bf9a8d861232d'
$FloorHash = '1692f8e6a3ff9d82b22ee5ac579b48d832a852b8bcfccb88fb02d85b360e4e58'
$ThresholdTableHash = '098888c153e60d5539e98e85c7e523a5a27c0848f6628d191c79f0613d3566fc'
$PredictorSourceGit = '2cccb361d977489d05e29633a633ded12a7d98b0'
$FloorSourceGit = 'b52776668470c02e90d2d1b741c037fbd02a5d0a'
$FrozenSourcePathspecs = @(
  'src/hoppertrex_mjlab',
  ':(exclude)src/hoppertrex_mjlab/hybrid/innovation_detector.py',
  ':(exclude)src/hoppertrex_mjlab/scripts/probe_hybrid_c2_innovation_qualification.py',
  ':(exclude)src/hoppertrex_mjlab/scripts/validate_hybrid_c2_innovation_qualification.py',
  'pyproject.toml',
  'uv.lock',
  'vendor'
)

function Assert-CommandAvailable {
  param([Parameter(Mandatory)][string]$Name)
  if ($null -eq (Get-Command $Name -ErrorAction SilentlyContinue)) {
    throw ('Missing required command: {0}' -f $Name)
  }
}

function Get-FileSha256 {
  param([Parameter(Mandatory)][string]$Path)
  return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-CanonicalScriptSha256 {
  param([Parameter(Mandatory)][string]$Path)
  $content = [System.IO.File]::ReadAllText($Path)
  $normalized = $content.Replace("`r`n", "`n").Replace("`r", "`n")
  $bytes = [System.Text.UTF8Encoding]::new($false).GetBytes($normalized)
  $sha256 = [System.Security.Cryptography.SHA256]::Create()
  try {
    $digest = $sha256.ComputeHash($bytes)
    return ([System.BitConverter]::ToString($digest)).Replace('-', '').ToLowerInvariant()
  } finally {
    $sha256.Dispose()
  }
}

function Invoke-NativeChecked {
  param(
    [Parameter(Mandatory)][string]$Executable,
    [Parameter(Mandatory)][string[]]$Arguments,
    [Parameter(Mandatory)][string]$FailureMessage
  )
  $previousPreference = $ErrorActionPreference
  $exitCode = -1
  try {
    $ErrorActionPreference = 'Continue'
    & $Executable @Arguments
    $exitCode = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $previousPreference
  }
  if ($exitCode -ne 0) {
    throw ('{0} Exit code: {1}' -f $FailureMessage, $exitCode)
  }
}

function Invoke-NativeLogged {
  param(
    [Parameter(Mandatory)][string]$Executable,
    [Parameter(Mandatory)][string[]]$Arguments,
    [Parameter(Mandatory)][string]$LogPath,
    [Parameter(Mandatory)][string]$FailureMessage,
    [switch]$Append
  )
  $previousPreference = $ErrorActionPreference
  $exitCode = -1
  try {
    $ErrorActionPreference = 'Continue'
    if ($Append.IsPresent) {
      & $Executable @Arguments 2>&1 |
        Tee-Object -FilePath $LogPath -Append
    } else {
      & $Executable @Arguments 2>&1 |
        Tee-Object -LiteralPath $LogPath
    }
    $exitCode = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $previousPreference
  }
  if ($exitCode -ne 0) {
    throw ('{0} Exit code: {1}. Log: {2}' -f $FailureMessage, $exitCode, $LogPath)
  }
}

if (
  $PSVersionTable.PSVersion.Major -ne 5 -or
  $PSVersionTable.PSVersion.Minor -ne 1
) {
  throw ('C2-j3 wrapper requires Windows PowerShell 5.1; got {0}.' -f $PSVersionTable.PSVersion)
}
foreach ($command in @('git', 'nvidia-smi')) {
  Assert-CommandAvailable -Name $command
}

$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $RepoRoot
$SelfHashPath = Join-Path $PSScriptRoot 'run_hybrid_c2_innovation_qualification.ps1.sha256'
if (-not (Test-Path -LiteralPath $SelfHashPath -PathType Leaf)) {
  throw ('Missing wrapper self-hash: {0}' -f $SelfHashPath)
}
$expectedSelfHash = (Get-Content -LiteralPath $SelfHashPath -Raw -Encoding ASCII).Trim()
if ($expectedSelfHash -notmatch '^[0-9a-f]{64}$') {
  throw 'C2-j3 wrapper self-hash sidecar is malformed.'
}
$actualSelfHash = Get-CanonicalScriptSha256 -Path $PSCommandPath
if ($actualSelfHash -ne $expectedSelfHash) {
  throw 'C2-j3 wrapper canonical self-hash mismatch.'
}
$PinnedSources = [ordered]@{
  core = @{
    path = Join-Path $RepoRoot 'src\hoppertrex_mjlab\hybrid\innovation_detector.py'
    sha256 = $CoreCanonicalHash
  }
  producer = @{
    path = Join-Path $RepoRoot 'src\hoppertrex_mjlab\scripts\probe_hybrid_c2_innovation_qualification.py'
    sha256 = $ProducerCanonicalHash
  }
  validator = @{
    path = Join-Path $RepoRoot 'src\hoppertrex_mjlab\scripts\validate_hybrid_c2_innovation_qualification.py'
    sha256 = $ValidatorCanonicalHash
  }
}
foreach ($entry in $PinnedSources.GetEnumerator()) {
  $path = [string]$entry.Value.path
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    throw ('Missing pinned C2-j3 {0} source: {1}' -f $entry.Key, $path)
  }
  if ((Get-CanonicalScriptSha256 -Path $path) -ne [string]$entry.Value.sha256) {
    throw ('Pinned C2-j3 {0} source hash mismatch.' -f $entry.Key)
  }
}

$branchLines = @(& git branch --show-current)
$branchExitCode = $LASTEXITCODE
if ($branchExitCode -ne 0 -or $branchLines.Count -ne 1) {
  throw 'Unable to identify the current Git branch.'
}
$branch = ([string]$branchLines[0]).Trim()
if ($branch -ne $RequiredBranch) {
  throw ('Expected branch {0}, got {1}.' -f $RequiredBranch, $branch)
}
$statusLines = @(& git status --porcelain)
$statusExitCode = $LASTEXITCODE
if ($statusExitCode -ne 0) {
  throw 'Unable to inspect the C2-j3 checkout.'
}
if ($statusLines.Count -ne 0) {
  throw 'Repository must be clean before the formal C2-j3 run.'
}
Invoke-NativeChecked -Executable 'git' -Arguments @(
  'cat-file', '-e', ($RequiredBase + '^{commit}')
) -FailureMessage 'Registered C2-j3 base is not a commit.'
Invoke-NativeChecked -Executable 'git' -Arguments @(
  'merge-base', '--is-ancestor', $RequiredBase, 'HEAD'
) -FailureMessage 'Checkout predates the frozen C2-j3 prerequisites.'
$dependencyDiffArguments = @(
  'diff', '--quiet', $RequiredBase, 'HEAD', '--'
) + $FrozenSourcePathspecs
Invoke-NativeChecked -Executable 'git' -Arguments $dependencyDiffArguments `
  -FailureMessage 'C2-j3 transitive source dependencies drifted from the audited base.'
Invoke-NativeChecked -Executable 'git' -Arguments @(
  'fetch', '--quiet', 'origin', $RequiredBranch
) -FailureMessage ('Failed to refresh origin/{0}.' -f $RequiredBranch)
$headLines = @(& git rev-parse HEAD)
$headExitCode = $LASTEXITCODE
if ($headExitCode -ne 0 -or $headLines.Count -ne 1) {
  throw 'Unable to resolve the local C2-j3 HEAD.'
}
$fullSha = ([string]$headLines[0]).Trim()
$shortSha = $fullSha.Substring(0, 7)
$remoteLines = @(& git rev-parse ('origin/{0}' -f $RequiredBranch))
$remoteExitCode = $LASTEXITCODE
if ($remoteExitCode -ne 0 -or $remoteLines.Count -ne 1) {
  throw 'Unable to resolve the remote C2-j3 HEAD.'
}
$remoteSha = ([string]$remoteLines[0]).Trim()
if ($fullSha -ne $remoteSha) {
  throw ('Checkout HEAD {0} does not match origin/{1} {2}.' -f $fullSha, $RequiredBranch, $remoteSha)
}

$PyProject = Join-Path $RepoRoot 'pyproject.toml'
$MjLabSourceDeclaration = 'mjlab = { path = "../mjlab-main", editable = true }'
$PyProjectLines = @(Get-Content -LiteralPath $PyProject -Encoding UTF8)
if ($PyProjectLines -notcontains $MjLabSourceDeclaration) {
  throw 'pyproject.toml no longer pins the expected editable MjLab source.'
}
$MjLabDeclaredRoot = (Resolve-Path -LiteralPath (
  Join-Path $RepoRoot '..\mjlab-main'
)).Path
$mjlabTopLevelLines = @(& git -C $MjLabDeclaredRoot rev-parse --show-toplevel)
$mjlabTopLevelExitCode = $LASTEXITCODE
if ($mjlabTopLevelExitCode -ne 0 -or $mjlabTopLevelLines.Count -ne 1) {
  throw 'Editable MjLab source is not a Git checkout.'
}
$MjLab = (Resolve-Path -LiteralPath ([string]$mjlabTopLevelLines[0])).Path
$mjlabHeadLines = @(& git -C $MjLab rev-parse HEAD)
$mjlabHeadExitCode = $LASTEXITCODE
if ($mjlabHeadExitCode -ne 0 -or $mjlabHeadLines.Count -ne 1) {
  throw 'Unable to resolve the MjLab HEAD.'
}
$mjlabSha = ([string]$mjlabHeadLines[0]).Trim()
if ($mjlabSha -ne $RequiredMjLab) {
  throw ('MjLab must be pinned to {0}.' -f $RequiredMjLab)
}
$mjlabStatusLines = @(& git -C $MjLab status --porcelain)
$mjlabStatusExitCode = $LASTEXITCODE
if ($mjlabStatusExitCode -ne 0) {
  throw 'Unable to inspect the MjLab checkout.'
}
if ($mjlabStatusLines.Count -ne 0) {
  throw 'MjLab checkout must be clean.'
}

$ArtifactRoot = Join-Path $RepoRoot 'docs\experiments\artifacts'
$Schedule = Join-Path $ArtifactRoot 'c1_schedule_candidate24_1f54968_seed1\c1_schedule.json'
$Calibration = Join-Path $ArtifactRoot 'hybrid_runtime_seed1\velocity_calibration_seed1.json'
$Posture = Join-Path $ArtifactRoot 'c1_posture_requalification_seed1\posture_map_seed1_registered_p032.json'
$Station = Join-Path $ArtifactRoot 'c1_posture_requalification_seed1\station_calibration_seed1.json'
$Predictor = Join-Path $ArtifactRoot 'c2_innovation_predictor_2cccb36_seed1\c2_innovation_predictor.json'
$Floor = Join-Path $ArtifactRoot 'c2_innovation_floor_b527766_seed2\c2_innovation_floor.json'
$FrozenFiles = [ordered]@{
  schedule = @{ path = $Schedule; sha256 = $ScheduleFileHash }
  calibration = @{ path = $Calibration; sha256 = $CalibrationFileHash }
  posture = @{ path = $Posture; sha256 = $PostureFileHash }
  station = @{ path = $Station; sha256 = $StationFileHash }
  predictor = @{ path = $Predictor; sha256 = $PredictorFileHash }
  transition_floor = @{ path = $Floor; sha256 = $FloorFileHash }
}
foreach ($entry in $FrozenFiles.GetEnumerator()) {
  $path = [string]$entry.Value.path
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    throw ('Missing frozen {0} artifact: {1}' -f $entry.Key, $path)
  }
  if ((Get-FileSha256 -Path $path) -ne [string]$entry.Value.sha256) {
    throw ('Frozen {0} artifact byte hash mismatch.' -f $entry.Key)
  }
}

$schedulePayload = Get-Content -LiteralPath $Schedule -Raw -Encoding UTF8 | ConvertFrom-Json
$calibrationPayload = Get-Content -LiteralPath $Calibration -Raw -Encoding UTF8 | ConvertFrom-Json
$posturePayload = Get-Content -LiteralPath $Posture -Raw -Encoding UTF8 | ConvertFrom-Json
$stationPayload = Get-Content -LiteralPath $Station -Raw -Encoding UTF8 | ConvertFrom-Json
$predictorPayload = Get-Content -LiteralPath $Predictor -Raw -Encoding UTF8 | ConvertFrom-Json
$floorPayload = Get-Content -LiteralPath $Floor -Raw -Encoding UTF8 | ConvertFrom-Json
if (
  $schedulePayload.artifact_type -ne 'gain_scheduled_lqr' -or
  [int]$schedulePayload.schema_version -ne 2 -or
  $schedulePayload.schedule_hash -ne $ScheduleHash -or
  $schedulePayload.bindings.identification_controller_gain_hash -ne $IdentificationGainHash -or
  $schedulePayload.bindings.identification_calibration_hash -ne $CalibrationHash -or
  $schedulePayload.bindings.posture_artifact_hash -ne $PostureArtifactHash -or
  $calibrationPayload.calibration_hash -ne $CalibrationHash -or
  $calibrationPayload.controller_gain_hash -ne $IdentificationGainHash -or
  $posturePayload.posture_artifact_hash -ne $PostureArtifactHash -or
  $stationPayload.station_calibration_hash -ne $StationHash -or
  $stationPayload.controller_gain_hash -ne $IdentificationGainHash -or
  $stationPayload.posture_artifact_hash -ne $PostureArtifactHash
) {
  throw 'Frozen C1 runtime artifact binding mismatch.'
}
if (
  [int]$predictorPayload.schema_version -ne 1 -or
  $predictorPayload.artifact_type -ne 'c2_innovation_predictor' -or
  $predictorPayload.probe -ne 'hybrid_c2_predictor_identification_v1' -or
  $predictorPayload.classification -ne 'PREDICTOR_IDENTIFICATION_QUALIFIED' -or
  $predictorPayload.git_sha -ne $PredictorSourceGit -or
  $predictorPayload.mjlab_git_sha -ne $RequiredMjLab -or
  $predictorPayload.predictor_hash -ne $PredictorHash -or
  $predictorPayload.bindings.controller_schedule_hash -ne $ScheduleHash -or
  $predictorPayload.bindings.identification_controller_gain_hash -ne $IdentificationGainHash -or
  $predictorPayload.bindings.velocity_calibration_hash -ne $CalibrationHash -or
  $predictorPayload.bindings.posture_artifact_hash -ne $PostureArtifactHash -or
  $predictorPayload.bindings.station_calibration_hash -ne $StationHash -or
  $null -ne $predictorPayload.bindings.yaw_calibration_hash
) {
  throw 'Frozen C2-j1 predictor binding mismatch.'
}
if (
  [int]$floorPayload.schema_version -ne 1 -or
  $floorPayload.artifact_type -ne 'c2_innovation_transition_floor' -or
  $floorPayload.probe -ne 'hybrid_c2_transition_floor_v1' -or
  $floorPayload.classification -ne 'INNOVATION_FLOOR_QUALIFIED' -or
  $floorPayload.git_sha -ne $FloorSourceGit -or
  $floorPayload.mjlab_git_sha -ne $RequiredMjLab -or
  $floorPayload.predictor_hash -ne $PredictorHash -or
  $floorPayload.floor_hash -ne $FloorHash -or
  $floorPayload.threshold_table_hash -ne $ThresholdTableHash -or
  @($floorPayload.threshold_table).Count -ne 125 -or
  $floorPayload.bindings.controller_schedule_hash -ne $ScheduleHash -or
  $floorPayload.bindings.identification_controller_gain_hash -ne $IdentificationGainHash -or
  $floorPayload.bindings.velocity_calibration_hash -ne $CalibrationHash -or
  $floorPayload.bindings.posture_artifact_hash -ne $PostureArtifactHash -or
  $floorPayload.bindings.station_calibration_hash -ne $StationHash -or
  $null -ne $floorPayload.bindings.yaw_calibration_hash
) {
  throw 'Frozen C2-j2 floor or threshold-table binding mismatch.'
}

$Python = Join-Path $RepoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
  throw ('Missing configured project Python: {0}' -f $Python)
}
$env:PYTHONPATH = ('{0};{1}' -f (Join-Path $RepoRoot 'src'), (Join-Path $RepoRoot 'src\hoppertrex_mjlab'))
$env:HOPPERTREX_HYBRID_CONTROLLER_PATH = $Schedule
$env:HOPPERTREX_HYBRID_CALIBRATION_PATH = $Calibration
$env:HOPPERTREX_HYBRID_POSTURE_MAP_PATH = $Posture
$env:HOPPERTREX_HYBRID_STATION_CALIBRATION_PATH = $Station
Remove-Item Env:HOPPERTREX_HYBRID_YAW_CALIBRATION_PATH -ErrorAction SilentlyContinue
Invoke-NativeChecked -Executable $Python -Arguments @(
  '-m', 'hoppertrex_mjlab.scripts.probe_hybrid_c2_innovation_qualification', '--help'
) -FailureMessage 'C2-j3 producer --help failed.'
Invoke-NativeChecked -Executable $Python -Arguments @(
  '-m', 'hoppertrex_mjlab.scripts.validate_hybrid_c2_innovation_qualification', '--help'
) -FailureMessage 'C2-j3 validator --help failed.'
Invoke-NativeChecked -Executable $Python -Arguments @(
  '-m', 'hoppertrex_mjlab.scripts.validate_hybrid_c2_innovation_qualification',
  '--predictor', $Predictor,
  '--transition-floor', $Floor,
  '--inputs-only'
) -FailureMessage 'Frozen C2-j3 input parser validation failed.'
$runtimeJson = & $Python -c 'import json, pathlib, sys, mjlab, mujoco, torch, warp; print(json.dumps(dict(python=sys.version.split()[0], torch=torch.__version__, torch_cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(), cuda_device_count=torch.cuda.device_count(), cuda_device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None), mjlab_root=str(pathlib.Path(mjlab.__file__).resolve().parents[2]), mujoco=getattr(mujoco, ''__version__'', None), warp=getattr(warp, ''__version__'', None))))'
$runtimeExitCode = $LASTEXITCODE
if ($runtimeExitCode -ne 0 -or [string]::IsNullOrWhiteSpace([string]$runtimeJson)) {
  throw 'Unable to query Python runtime provenance.'
}
$runtime = $runtimeJson | ConvertFrom-Json
if ($runtime.cuda_available -ne $true -or [int]$runtime.cuda_device_count -lt 1) {
  throw 'PyTorch does not expose CUDA device 0.'
}
$importedMjLabRoot = (Resolve-Path -LiteralPath ([string]$runtime.mjlab_root)).Path
if ($importedMjLabRoot -ne $MjLab) {
  throw ('Python imports MjLab from {0}, expected {1}.' -f $importedMjLabRoot, $MjLab)
}
Invoke-NativeChecked -Executable $Python -Arguments @(
  '-c', 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() >= 1; x=torch.ones(1,device=''cuda:0''); assert float(x.item())==1.0'
) -FailureMessage 'C2-j3 CUDA 0 preflight failed.'

$gpuLines = @(& nvidia-smi --query-gpu=name,driver_version,memory.total,pci.bus_id --format=csv,noheader)
$gpuExitCode = $LASTEXITCODE
if ($gpuExitCode -ne 0 -or $gpuLines.Count -lt 1) {
  throw 'Unable to query GPU provenance with nvidia-smi.'
}
$gpuLine = [string]$gpuLines[0]

$OutputDirectory = Join-Path $RepoRoot ('experiments\c2_innovation_qualification_{0}_seed3' -f $shortSha)
$OutputZip = $OutputDirectory + '.zip'
if (
  (Test-Path -LiteralPath $OutputDirectory) -or
  (Test-Path -LiteralPath $OutputZip)
) {
  throw ('Refusing to overwrite C2-j3 output: {0}' -f $OutputDirectory)
}
$runToken = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffZ') +
  '_' + [System.Guid]::NewGuid().ToString('N')
$WorkingDirectory = $OutputDirectory + '.incomplete.' + $runToken
$ConsoleTemp = $WorkingDirectory + '.console.log'
$TemporaryZip = $OutputZip + '.incomplete.' + $runToken + '.zip'
New-Item -ItemType Directory -Path $WorkingDirectory | Out-Null

$producerArguments = @(
  '-u', '-m', 'hoppertrex_mjlab.scripts.probe_hybrid_c2_innovation_qualification',
  '--output-dir', $WorkingDirectory,
  '--predictor', $Predictor,
  '--transition-floor', $Floor,
  '--controller-path', $Schedule,
  '--calibration-path', $Calibration,
  '--posture-map-path', $Posture,
  '--station-calibration-path', $Station,
  '--device', 'cuda:0'
)
try {
  Invoke-NativeLogged -Executable $Python -Arguments $producerArguments -LogPath $ConsoleTemp -FailureMessage 'C2-j3 formal producer failed.'
  Invoke-NativeLogged -Executable $Python -Arguments @(
    '-u', '-m', 'hoppertrex_mjlab.scripts.validate_hybrid_c2_innovation_qualification',
    '--output-dir', $WorkingDirectory,
    '--predictor', $Predictor,
    '--transition-floor', $Floor,
    '--expected-git-sha', $fullSha,
    '--expected-mjlab-git-sha', $RequiredMjLab
  ) -LogPath $ConsoleTemp -Append -FailureMessage 'C2-j3 independent raw validation failed.'
} catch {
  if (Test-Path -LiteralPath $ConsoleTemp -PathType Leaf) {
    Move-Item -LiteralPath $ConsoleTemp -Destination (
      Join-Path $WorkingDirectory 'console.log'
    )
  }
  Write-Warning ('Incomplete C2-j3 output retained: {0}' -f $WorkingDirectory)
  throw
}
Move-Item -LiteralPath $ConsoleTemp -Destination (
  Join-Path $WorkingDirectory 'console.log'
)

$ResultPath = Join-Path $WorkingDirectory 'c2_innovation_detector_qualification.json'
if (-not (Test-Path -LiteralPath $ResultPath -PathType Leaf)) {
  throw 'C2-j3 producer omitted its qualification result.'
}
$result = Get-Content -LiteralPath $ResultPath -Raw -Encoding UTF8 | ConvertFrom-Json
$AllowedClassifications = @(
  'INNOVATION_DETECTOR_QUALIFIED',
  'C2_INNOVATION_DETECTOR_UNQUALIFIED_STOP',
  'INVALID_INNOVATION_CAPTURE'
)
if (
  $AllowedClassifications -notcontains $result.classification -or
  [int]$result.schema_version -ne 1 -or
  $result.artifact_type -ne 'c2_innovation_detector_qualification' -or
  $result.probe -ne 'hybrid_c2_innovation_qualification_v1' -or
  $result.git_sha -ne $fullSha -or
  $result.mjlab_git_sha -ne $RequiredMjLab -or
  $result.predictor_hash -ne $PredictorHash -or
  $result.floor_hash -ne $FloorHash -or
  $result.threshold_table_hash -ne $ThresholdTableHash -or
  $result.bindings.controller_schedule_hash -ne $ScheduleHash -or
  $result.bindings.identification_controller_gain_hash -ne $IdentificationGainHash -or
  $result.bindings.velocity_calibration_hash -ne $CalibrationHash -or
  $result.bindings.posture_artifact_hash -ne $PostureArtifactHash -or
  $result.bindings.station_calibration_hash -ne $StationHash -or
  $null -ne $result.bindings.yaw_calibration_hash -or
  [string]$result.detector_hash -notmatch '^[0-9a-f]{64}$' -or
  $result.promotion_eligible -ne $false -or
  $result.training_eligible -ne $false -or
  $null -ne $result.checkpoint
) {
  throw 'C2-j3 result provenance, binding, or eligibility drifted.'
}
if (
  [int]$result.protocol.seed -ne 3 -or
  $result.protocol.device -ne 'cuda:0' -or
  @($result.protocol.cells).Count -ne 18 -or
  [int]$result.protocol.pairs_per_cell -ne 16 -or
  [int]$result.protocol.settle_steps -ne 200 -or
  [int]$result.protocol.drive_steps -ne 500 -or
  [double]$result.protocol.settle_vx_mps -ne 0.0 -or
  $result.protocol.attempt_mask -ne 'full_true' -or
  $result.protocol.first_tick_no_vote -ne $true -or
  [int]$result.protocol.pre_impact_steps -ne 25 -or
  [int]$result.protocol.post_impact_steps -ne 75 -or
  [double]$result.protocol.runtime_assertions.posture_boundary_snap_atol -ne 1.0e-7 -or
  $result.protocol.impact_truth.archived_raw_replay -ne $true -or
  [double]$result.protocol.impact_truth.outer_face_offset_from_terrain_origin_m -ne -3.0 -or
  [double]$result.protocol.impact_truth.outer_face_binding_atol_m -ne 2.0e-5 -or
  [double]$result.protocol.yaw_command -ne 0.0 -or
  @($result.protocol.residual_action).Count -ne 6 -or
  @($result.protocol.residual_action | Where-Object { [double]$_ -ne 0.0 }).Count -ne 0 -or
  [int]$result.protocol.voting.consecutive_ticks -ne 2 -or
  [int]$result.protocol.voting.max_delay_ticks -ne 3 -or
  [int]$result.protocol.qualification.flat_trigger_count -ne 0 -or
  [int]$result.protocol.qualification.stair_pre_impact_trigger_count -ne 0 -or
  [int]$result.protocol.qualification.overall_timely_min -ne 274 -or
  [int]$result.protocol.qualification.overall_stair_attempts -ne 288 -or
  [int]$result.protocol.qualification.per_cell_timely_min -ne 15 -or
  [int]$result.protocol.qualification.per_cell_stair_attempts -ne 16 -or
  $result.protocol.evidence_eligible -ne $true -or
  $result.protocol.promotion_eligible -ne $false -or
  $result.protocol.training_eligible -ne $false
) {
  throw 'C2-j3 result protocol drifted from the seed-3 preregistration.'
}
if (
  [int]$result.completed_cell_count -ne 18 -or
  [int]$result.completed_pair_count -ne 288 -or
  @($result.cells).Count -ne 18
) {
  throw 'C2-j3 result did not complete all 18 cells and 288 pairs.'
}

$selectedIndex = $null
if ($null -ne $result.selected_candidate) {
  $selectedIndex = [int]$result.selected_candidate.threshold_table_index
}
$nextStep = [string]$result.next_step
if ($result.classification -eq 'INNOVATION_DETECTOR_QUALIFIED') {
  if (
    $result.evidence_eligible -ne $true -or
    [int]$result.completed_candidate_count -ne 125 -or
    [int]$result.qualified_candidate_count -lt 1 -or
    $null -eq $result.selected_candidate -or
    $nextStep -ne 'FREEZE_AND_INDEPENDENT_AUDIT_BEFORE_C3'
  ) {
    throw 'Qualified C2-j3 result is incomplete.'
  }
} elseif ($result.classification -eq 'C2_INNOVATION_DETECTOR_UNQUALIFIED_STOP') {
  if (
    $result.evidence_eligible -ne $true -or
    [int]$result.completed_candidate_count -ne 125 -or
    [int]$result.qualified_candidate_count -ne 0 -or
    $null -ne $result.selected_candidate -or
    $nextStep -ne 'STOP_FOR_USER_ROUTE_DECISION'
  ) {
    throw 'All-failed C2-j3 result is incomplete or falsely deployable.'
  }
} else {
  if (
    $result.evidence_eligible -ne $false -or
    [int]$result.completed_candidate_count -ne 0 -or
    [int]$result.qualified_candidate_count -ne 0 -or
    $null -ne $result.selected_candidate -or
    $nextStep -ne 'INDEPENDENT_IMPLEMENTATION_DIAGNOSIS_ONLY'
  ) {
    throw 'Invalid C2-j3 capture has inconsistent stop semantics.'
  }
}

$ProtocolNote = [ordered]@{
  schema_version = 1
  kind = 'c2_innovation_qualification_machine_room_run'
  git_sha = $fullSha
  mjlab_git_sha = $RequiredMjLab
  task = $Task
  seed = 3
  device = 'cuda:0'
  gpu = $gpuLine
  runtime = [ordered]@{
    python_version = [string]$runtime.python
    torch_version = [string]$runtime.torch
    torch_cuda_version = [string]$runtime.torch_cuda
    cuda_available = [bool]$runtime.cuda_available
    cuda_device_count = [int]$runtime.cuda_device_count
    cuda_device_name = [string]$runtime.cuda_device
    mjlab_import_root = $importedMjLabRoot
    mujoco_version = [string]$runtime.mujoco
    warp_version = [string]$runtime.warp
    powershell_version = $PSVersionTable.PSVersion.ToString()
    powershell_edition = [string]$PSVersionTable.PSEdition
    wrapper_canonical_sha256 = $actualSelfHash
  }
  input_file_sha256 = [ordered]@{
    schedule = $ScheduleFileHash
    velocity_calibration = $CalibrationFileHash
    posture_map = $PostureFileHash
    station_calibration = $StationFileHash
    predictor = $PredictorFileHash
    transition_floor = $FloorFileHash
  }
  source_canonical_sha256 = [ordered]@{
    innovation_detector = $CoreCanonicalHash
    producer = $ProducerCanonicalHash
    validator = $ValidatorCanonicalHash
  }
  predictor_hash = $PredictorHash
  floor_hash = $FloorHash
  threshold_table_hash = $ThresholdTableHash
  detector_hash = [string]$result.detector_hash
  bindings = $result.bindings
  classification = [string]$result.classification
  completed_cell_count = [int]$result.completed_cell_count
  completed_pair_count = [int]$result.completed_pair_count
  completed_candidate_count = [int]$result.completed_candidate_count
  qualified_candidate_count = [int]$result.qualified_candidate_count
  selected_threshold_table_index = $selectedIndex
  evidence_eligible = [bool]$result.evidence_eligible
  promotion_eligible = $false
  training_eligible = $false
  checkpoint = $null
  yaw_calibration_hash = $null
  next_step = $nextStep
  exit_semantics = if ($result.classification -eq 'INVALID_INNOVATION_CAPTURE') {
    'ARCHIVED_THEN_NONZERO'
  } else {
    'COMPLETE_ZERO'
  }
}
$ProtocolPath = Join-Path $WorkingDirectory 'protocol_note.json'
[System.IO.File]::WriteAllText(
  $ProtocolPath,
  ($ProtocolNote | ConvertTo-Json -Depth 20),
  [System.Text.UTF8Encoding]::new($false)
)
$ChecksumPath = Join-Path $WorkingDirectory 'SHA256SUMS.txt'
$checksumLines = @()
foreach ($file in @(Get-ChildItem -LiteralPath $WorkingDirectory -File | Sort-Object Name)) {
  if ($file.Name -eq 'SHA256SUMS.txt') {
    continue
  }
  $checksumLines += ('{0}  {1}' -f (Get-FileSha256 -Path $file.FullName), $file.Name)
}
[System.IO.File]::WriteAllLines(
  $ChecksumPath,
  $checksumLines,
  [System.Text.UTF8Encoding]::new($false)
)

try {
  Compress-Archive -Path (Join-Path $WorkingDirectory '*') -DestinationPath $TemporaryZip -CompressionLevel Optimal
  Move-Item -LiteralPath $WorkingDirectory -Destination $OutputDirectory
  Move-Item -LiteralPath $TemporaryZip -Destination $OutputZip
} catch {
  if (
    (Test-Path -LiteralPath $OutputDirectory -PathType Container) -and
    -not (Test-Path -LiteralPath $WorkingDirectory)
  ) {
    Move-Item -LiteralPath $OutputDirectory -Destination $WorkingDirectory
  }
  Write-Warning ('Incomplete C2-j3 output retained: {0}' -f $WorkingDirectory)
  throw
}

$ZipSha256 = Get-FileSha256 -Path $OutputZip
if ($result.classification -eq 'INNOVATION_DETECTOR_QUALIFIED') {
  Write-Host '[PASS] C2-j3 innovation detector qualified.'
} elseif ($result.classification -eq 'C2_INNOVATION_DETECTOR_UNQUALIFIED_STOP') {
  Write-Host '[COMPLETE] C2-j3 ran all 125 candidates; none qualified.'
} else {
  Write-Warning 'C2-j3 produced an invalid capture; evidence was archived for diagnosis.'
}
Write-Host ('CLASSIFICATION={0}' -f $result.classification)
Write-Host ('DETECTOR_HASH={0}' -f $result.detector_hash)
Write-Host ('QUALIFIED_CANDIDATES={0}' -f $result.qualified_candidate_count)
Write-Host ('RESULT={0}' -f $OutputDirectory)
Write-Host ('ZIP={0}' -f $OutputZip)
Write-Host ('ZIP_SHA256={0}' -f $ZipSha256)
Write-Host ('NEXT={0}' -f $nextStep)

if ($result.classification -eq 'INVALID_INNOVATION_CAPTURE') {
  throw ('INVALID_INNOVATION_CAPTURE archived at {0}' -f $OutputDirectory)
}
