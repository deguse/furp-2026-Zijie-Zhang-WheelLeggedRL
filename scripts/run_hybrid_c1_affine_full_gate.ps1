[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$RequiredBranch = 'codex/p2-classical-upper-bound'
$RequiredImplementation = '9fe48c31a5cc1c3cbea8b163d3fafe860e3aba53'
$RequiredMjLab = '43e0f3ea9c92ddbb4de9f3bb1ac772d604e3ebf6'
$SourceZipSha256 = '10e0f8f498107406e969e9f7d8390f8ac8c22f5838b60d5254e65196453eb4f9'
$SourceManifestSha256 = '6609c3086a88b07a9a903c15897a4aab838c80ab9bc23ddf862618a16793d341'
$SourceProtocolSha256 = 'e5d692831b2c676ecbe37d3124527e72abf146b2708919fdce8cde9a68fec1ee'
$SourceSmokeSha256 = 'a2d65437f094a604d4f47145d63c7342e81d59cef096d7153fca27ba64fcd1b8'
$RetryResultSha256 = '18cea95353b227b47370af25265f16c2450ba25e224069e08c52f92d6d472f07'
$RetryProtocolSha256 = 'c336eb937a12252412bb2a8837504eaf13635fb5a0f3a4da2d33a1a1443b5c98'
$RetryZipSha256 = '86521c7e5762b669a2c179c590f5c08fbd6454d165087ee8a02b86ae293f14dd'
$RetryManifestSha256 = '44c51a18affe8e96dc5292aa31f78090161380828d6c4e3e835d1dc332319378'
$ControllerFileSha256 = '663ab77f77521581cde77ea2bd8c72c7f395f33b05b62348ef6d82a752aad7fc'
$CalibrationFileSha256 = 'ef002d0d622725509b47c8ff40d8af658fd42f705bdeac67ac35bae4458f889d'
$PostureFileSha256 = 'b8e627f85b53d21dd8d9c26edbe2943151d9bcf9e5864ff998ede5f909118e23'
$StationFileSha256 = 'f22a9b66f734004ff14b6586a22a991d527f360806bbbdefe096e9f0474db72a'
$CompensatedFileSha256 = 'c003192963b257c8d497ffd347be2cd60695c5ce8653932403709d8193c88e55'
$ControllerGainHash = '8fee25a0339dd1e99127cbed912941dc3ad8ef2030ce49a0d310d1563cb87d98'
$CalibrationHash = 'f62648b57bd17a3503bcbdbf58f349f91fcd8de8ef0cf04551c200401233ed01'
$PostureArtifactHash = '3b96fd3dae66ad781b5b875c74184db101c42da02c53dfcc40a5137a6b5de11a'
$StationCalibrationHash = 'c00e859b3093b4812d54799253accdaeb99171a2cf4028b08bc39e68eaaa7d8a'

function Get-FileSha256 {
  param([Parameter(Mandatory)][string]$Path)
  return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-CanonicalTextSha256 {
  param([Parameter(Mandatory)][string]$Path)
  $text = [System.IO.File]::ReadAllText($Path)
  $normalized = $text.Replace("`r`n", "`n").Replace("`r", "`n")
  $utf8 = [System.Text.UTF8Encoding]::new($false)
  $sha256 = [System.Security.Cryptography.SHA256]::Create()
  try {
    $digest = $sha256.ComputeHash($utf8.GetBytes($normalized))
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

$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $RepoRoot
foreach ($command in @('git', 'uv', 'nvidia-smi')) {
  if ($null -eq (Get-Command $command -ErrorAction SilentlyContinue)) {
    throw ('Missing required command: {0}' -f $command)
  }
}

$SelfHashFile = Join-Path $PSScriptRoot 'run_hybrid_c1_affine_full_gate.ps1.sha256'
if (-not (Test-Path -LiteralPath $SelfHashFile -PathType Leaf)) {
  throw ('Missing wrapper self-hash: {0}' -f $SelfHashFile)
}
$expectedSelfHash = (Get-Content -LiteralPath $SelfHashFile -Raw -Encoding ASCII).Trim()
$actualSelfHash = Get-CanonicalTextSha256 -Path $PSCommandPath
if ($actualSelfHash -ne $expectedSelfHash) {
  throw 'C1 affine full-gate wrapper self-hash mismatch.'
}

$branch = (git branch --show-current).Trim()
if ($branch -ne $RequiredBranch) {
  throw ('Expected branch {0}, got {1}.' -f $RequiredBranch, $branch)
}
if (@(git status --porcelain).Count -ne 0) {
  throw 'Repository must be clean before the formal C1 full gate.'
}
Invoke-NativeChecked -Executable 'git' -Arguments @(
  'merge-base', '--is-ancestor', $RequiredImplementation, 'HEAD'
) -FailureMessage 'Checkout predates the selected candidate-24 evidence.'
Invoke-NativeChecked -Executable 'git' -Arguments @(
  'fetch', '--quiet', 'origin', $RequiredBranch
) -FailureMessage 'Failed to refresh the remote C1 branch.'
$fullSha = (git rev-parse HEAD).Trim()
$remoteSha = (git rev-parse ('origin/{0}' -f $RequiredBranch)).Trim()
if ($fullSha -ne $remoteSha) {
  throw ('Checkout HEAD {0} does not match remote HEAD {1}.' -f $fullSha, $remoteSha)
}
$shortSha = (git rev-parse --short=7 HEAD).Trim()

$MjLabRoot = (Resolve-Path -LiteralPath (Join-Path $RepoRoot '..\mjlab-main')).Path
if (@(git -C $MjLabRoot status --porcelain).Count -ne 0) {
  throw 'MjLab checkout must be clean.'
}
$mjlabSha = (git -C $MjLabRoot rev-parse HEAD).Trim()
if ($mjlabSha -ne $RequiredMjLab) {
  throw ('MjLab must be pinned to {0}, got {1}.' -f $RequiredMjLab, $mjlabSha)
}

$Python = Join-Path $RepoRoot '.venv\Scripts\python.exe'
Invoke-NativeChecked -Executable 'uv' -Arguments @(
  'sync', '--frozen', '--python', '3.11'
) -FailureMessage 'uv sync failed.'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
  throw ('Missing project Python: {0}' -f $Python)
}
$env:PYTHONPATH = ('{0};{1}' -f (Join-Path $RepoRoot 'src'), (Join-Path $RepoRoot 'src\hoppertrex_mjlab'))

$RuntimeArtifacts = Join-Path $RepoRoot 'docs\experiments\artifacts\hybrid_runtime_seed1'
$C1Artifacts = Join-Path $RepoRoot 'docs\experiments\artifacts\c1_posture_requalification_seed1'
$Controller = Join-Path $RuntimeArtifacts 'controller_seed1.json'
$Calibration = Join-Path $RuntimeArtifacts 'velocity_calibration_seed1.json'
$Posture = Join-Path $C1Artifacts 'posture_map_seed1_registered_p032.json'
$Station = Join-Path $C1Artifacts 'station_calibration_seed1.json'
$Compensated = Join-Path $C1Artifacts 'balance_compensated_seed1.json'
$ExpectedArtifacts = [ordered]@{
  $Controller = $ControllerFileSha256
  $Calibration = $CalibrationFileSha256
  $Posture = $PostureFileSha256
  $Station = $StationFileSha256
  $Compensated = $CompensatedFileSha256
}
foreach ($entry in $ExpectedArtifacts.GetEnumerator()) {
  if (-not (Test-Path -LiteralPath $entry.Key -PathType Leaf)) {
    throw ('Missing frozen artifact: {0}' -f $entry.Key)
  }
  if ((Get-FileSha256 -Path $entry.Key) -ne $entry.Value) {
    throw ('Frozen artifact SHA256 mismatch: {0}' -f $entry.Key)
  }
}
$env:HOPPERTREX_HYBRID_CONTROLLER_PATH = $Controller
$env:HOPPERTREX_HYBRID_CALIBRATION_PATH = $Calibration
$env:HOPPERTREX_HYBRID_POSTURE_MAP_PATH = $Posture
$env:HOPPERTREX_HYBRID_STATION_CALIBRATION_PATH = $Station
Remove-Item Env:HOPPERTREX_HYBRID_YAW_CALIBRATION_PATH -ErrorAction SilentlyContinue

& (Join-Path $PSScriptRoot 'run_hybrid_c1_schedule_preflight.ps1')
if ($LASTEXITCODE -ne 0) {
  throw 'C1 schedule preflight failed.'
}

$SourceDirectory = Join-Path $RepoRoot 'experiments\c1_affine_identification_nodes_0c7bd78_seed1'
$SourceZip = $SourceDirectory + '.zip'
$RetryDirectory = Join-Path $RepoRoot 'experiments\c1_affine_center_smoke_retry_9fe48c3_seed1'
$RetryZip = $RetryDirectory + '.zip'
$RetryResult = Join-Path $RetryDirectory 'affine_center_smoke_retry.json'
$RetryProtocol = Join-Path $RetryDirectory 'protocol_note.json'
$SourceManifest = Join-Path $SourceDirectory 'SHA256SUMS.txt'
$SourceProtocol = Join-Path $SourceDirectory 'protocol_note.json'
$SourceSmoke = Join-Path $SourceDirectory 'affine_center_smoke.json'
$RetryManifest = Join-Path $RetryDirectory 'SHA256SUMS.txt'
$FrozenInputs = [ordered]@{
  $SourceZip = $SourceZipSha256
  $SourceManifest = $SourceManifestSha256
  $SourceProtocol = $SourceProtocolSha256
  $SourceSmoke = $SourceSmokeSha256
  $RetryZip = $RetryZipSha256
  $RetryResult = $RetryResultSha256
  $RetryProtocol = $RetryProtocolSha256
  $RetryManifest = $RetryManifestSha256
}
if (-not (Test-Path -LiteralPath $SourceDirectory -PathType Container)) {
  throw ('Missing frozen nine-node directory: {0}' -f $SourceDirectory)
}
if (-not (Test-Path -LiteralPath $RetryDirectory -PathType Container)) {
  throw ('Missing frozen retry directory: {0}' -f $RetryDirectory)
}
foreach ($entry in $FrozenInputs.GetEnumerator()) {
  if (-not (Test-Path -LiteralPath $entry.Key -PathType Leaf)) {
    throw ('Missing frozen C1 input: {0}' -f $entry.Key)
  }
  if ((Get-FileSha256 -Path $entry.Key) -ne $entry.Value) {
    throw ('Frozen C1 input SHA256 mismatch: {0}' -f $entry.Key)
  }
}
foreach ($manifestSpec in @(
  @($SourceDirectory, $SourceManifest, 30),
  @($RetryDirectory, $RetryManifest, 3)
)) {
  $manifestRoot = [string]$manifestSpec[0]
  $manifestPath = [string]$manifestSpec[1]
  $expectedCount = [int]$manifestSpec[2]
  $hashLines = @(Get-Content -LiteralPath $manifestPath -Encoding ASCII)
  if ($hashLines.Count -ne $expectedCount) {
    throw ('Unexpected manifest entry count: {0}' -f $manifestPath)
  }
  foreach ($line in $hashLines) {
    if ($line -notmatch '^([0-9a-f]{64})  ([^/\\]+)$') {
      throw ('Malformed checksum line in {0}: {1}' -f $manifestPath, $line)
    }
    $boundPath = Join-Path $manifestRoot $Matches[2]
    if (-not (Test-Path -LiteralPath $boundPath -PathType Leaf)) {
      throw ('Manifest file is missing: {0}' -f $boundPath)
    }
    if ((Get-FileSha256 -Path $boundPath) -ne $Matches[1]) {
      throw ('Manifest SHA256 mismatch: {0}' -f $boundPath)
    }
  }
}

$OutputDirectory = Join-Path $RepoRoot ('experiments\c1_affine_full_gate_{0}_seed1' -f $shortSha)
$OutputZip = $OutputDirectory + '.zip'
if ((Test-Path -LiteralPath $OutputDirectory) -or (Test-Path -LiteralPath $OutputZip)) {
  throw ('Refusing to overwrite existing C1 full-gate output: {0}' -f $OutputDirectory)
}
$runToken = [Guid]::NewGuid().ToString('N')
$WorkingDirectory = $OutputDirectory + '.incomplete.' + $runToken
$WorkingZip = $WorkingDirectory + '.zip'
New-Item -ItemType Directory -Path $WorkingDirectory | Out-Null

$gpuLines = @(& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)
$gpuLine = $gpuLines | Select-Object -First 1
if ($LASTEXITCODE -ne 0 -or -not $gpuLine) {
  throw 'Unable to query GPU provenance with nvidia-smi.'
}
$runtimeJson = & $Python -c 'import json,sys,mujoco,torch,warp; print(json.dumps(dict(python=sys.version.split()[0],torch=torch.__version__,torch_cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),cuda_device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),mujoco=getattr(mujoco,"__version__",None),warp=getattr(warp,"__version__",None))))'
if ($LASTEXITCODE -ne 0) {
  throw 'Unable to query Python runtime provenance.'
}
$runtime = $runtimeJson | ConvertFrom-Json
if ($runtime.cuda_available -ne $true) {
  throw 'PyTorch does not expose CUDA on this machine.'
}

Invoke-NativeChecked -Executable $Python -Arguments @(
  '-m', 'hoppertrex_mjlab.scripts.evaluate_hybrid_c1_affine_full_gate', '--help'
) -FailureMessage 'C1 affine full-gate CLI validation failed.'
$ResultPath = Join-Path $WorkingDirectory 'c1_affine_full_gate.json'
$SelectionPath = Join-Path $WorkingDirectory 'c1_affine_full_gate_selection.json'
$ConsolePath = Join-Path $WorkingDirectory 'console.log'
$evaluateArgs = @(
  '-u', '-m', 'hoppertrex_mjlab.scripts.evaluate_hybrid_c1_affine_full_gate',
  '--nodes-dir', $SourceDirectory,
  '--source-zip', $SourceZip,
  '--retry-result', $RetryResult,
  '--retry-protocol', $RetryProtocol,
  '--retry-zip', $RetryZip,
  '--output', $ResultPath,
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
Invoke-NativeLogged -Executable $Python -Arguments $evaluateArgs -LogPath $ConsolePath -FailureMessage 'C1 affine full gate failed.'
if (-not (Test-Path -LiteralPath $ResultPath -PathType Leaf)) {
  throw 'C1 affine full gate did not write its adjudication.'
}
$result = Get-Content -LiteralPath $ResultPath -Raw -Encoding UTF8 | ConvertFrom-Json
$validClassifications = @('C1_AFFINE_FULL_GATE_SELECTED', 'C1_AFFINE_FULL_GATE_FAILED_STOP')
if (
  $validClassifications -notcontains $result.classification -or
  $result.git_sha -ne $fullSha -or
  $result.mjlab_git_sha -ne $RequiredMjLab -or
  [int]$result.candidate.index -ne 24 -or
  [double]$result.candidate.anchor_alpha -ne 0.25 -or
  [int]$result.completed_candidate_count -ne 1 -or
  [int]$result.completed_cell_count -ne 15 -or
  $result.bindings.controller_gain_hash -ne $ControllerGainHash -or
  $result.bindings.velocity_calibration_hash -ne $CalibrationHash -or
  $result.bindings.posture_artifact_hash -ne $PostureArtifactHash -or
  $result.bindings.station_calibration_hash -ne $StationCalibrationHash -or
  $result.evidence_eligible -ne $true -or
  $result.promotion_eligible -ne $false -or
  $result.training_eligible -ne $false -or
  $null -ne $result.checkpoint -or
  $null -ne $result.yaw_calibration_hash
) {
  throw 'C1 affine full-gate adjudication is incomplete or invalid.'
}
if (
  ($result.classification -eq 'C1_AFFINE_FULL_GATE_SELECTED' -and $result.next_step -ne 'DOWNLOAD_FOR_OFFLINE_SCHEDULE_BUILD') -or
  ($result.classification -eq 'C1_AFFINE_FULL_GATE_FAILED_STOP' -and $result.next_step -ne 'STOP')
) {
  throw 'C1 affine full-gate classification and next step disagree.'
}
$selectionSha = $null
if ($result.classification -eq 'C1_AFFINE_FULL_GATE_SELECTED') {
  if (-not (Test-Path -LiteralPath $SelectionPath -PathType Leaf)) {
    throw 'Passing C1 full gate did not write selection evidence.'
  }
  $selection = Get-Content -LiteralPath $SelectionPath -Raw -Encoding UTF8 | ConvertFrom-Json
  $resultSha = Get-FileSha256 -Path $ResultPath
  if (
    $selection.status -ne 'affine_full_gate_selected' -or
    $selection.classification -ne 'C1_AFFINE_FULL_GATE_SELECTED' -or
    [int]$selection.selected_candidate_index -ne 24 -or
    $selection.full_gate_artifact_path -ne 'c1_affine_full_gate.json' -or
    $selection.full_gate_artifact_sha256 -ne $resultSha -or
    @($selection.screened_candidates).Count -ne 27 -or
    $selection.final_gate_candidate.flat_gate_passed -ne $true -or
    $selection.final_gate_candidate.safety_clean -ne $true
  ) {
    throw 'C1 affine full-gate selection evidence is invalid.'
  }
  $selectionSha = Get-FileSha256 -Path $SelectionPath
} elseif (Test-Path -LiteralPath $SelectionPath) {
  throw 'Failed C1 full gate must not write selection evidence.'
}

$protocol = [ordered]@{
  schema_version = 1
  kind = 'c1_affine_full_gate'
  git_sha = $fullSha
  mjlab_git_sha = $RequiredMjLab
  seed = 1
  device = 'cuda:0'
  gpu = [string]$gpuLine
  runtime = $runtime
  source_collection_zip_sha256 = $SourceZipSha256
  retry_result_sha256 = $RetryResultSha256
  retry_protocol_sha256 = $RetryProtocolSha256
  retry_zip_sha256 = $RetryZipSha256
  candidate_index = 24
  q_diag = @(40.0, 4.0, 8.0, 1.0)
  r_diag = @(0.5)
  anchor_alpha = 0.25
  cells = 15
  selection_sha256 = $selectionSha
  classification = [string]$result.classification
  evidence_eligible = $true
  promotion_eligible = $false
  training_eligible = $false
  checkpoint = $null
  yaw_calibration_hash = $null
  next_step = [string]$result.next_step
}
$protocolPath = Join-Path $WorkingDirectory 'protocol_note.json'
$protocol | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $protocolPath -Encoding UTF8
$checksumPath = Join-Path $WorkingDirectory 'SHA256SUMS.txt'
$hashLines = Get-ChildItem -LiteralPath $WorkingDirectory -File |
  Where-Object { $_.Name -ne 'SHA256SUMS.txt' } |
  Sort-Object Name |
  ForEach-Object { ('{0}  {1}' -f (Get-FileSha256 -Path $_.FullName), $_.Name) }
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
Write-Host '[PASS] C1 affine candidate-24 full gate complete.'
Write-Host ('RESULT={0}' -f $OutputDirectory)
Write-Host ('ZIP={0}' -f $OutputZip)
Write-Host ('ZIP_SHA256={0}' -f $zipSha)
Write-Host ('CLASSIFICATION={0}' -f $result.classification)
Write-Host ('NEXT={0}' -f $result.next_step)
