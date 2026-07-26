[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$RequiredBranch = 'codex/p2-classical-upper-bound'
$RequiredImplementation = '0c7bd78893998f0a1c6d58615fb3ea7fd97f0bdd'
$RequiredMjLab = '43e0f3ea9c92ddbb4de9f3bb1ac772d604e3ebf6'
$SourceCollectionGit = '0c7bd78893998f0a1c6d58615fb3ea7fd97f0bdd'
$SourceZipSha256 = '10e0f8f498107406e969e9f7d8390f8ac8c22f5838b60d5254e65196453eb4f9'
$SourceManifestSha256 = '6609c3086a88b07a9a903c15897a4aab838c80ab9bc23ddf862618a16793d341'
$SourceProtocolSha256 = 'e5d692831b2c676ecbe37d3124527e72abf146b2708919fdce8cde9a68fec1ee'
$SourceSmokeSha256 = 'a2d65437f094a604d4f47145d63c7342e81d59cef096d7153fca27ba64fcd1b8'
$ControllerFileSha256 = '663ab77f77521581cde77ea2bd8c72c7f395f33b05b62348ef6d82a752aad7fc'
$CalibrationFileSha256 = 'ef002d0d622725509b47c8ff40d8af658fd42f705bdeac67ac35bae4458f889d'
$PostureFileSha256 = 'b8e627f85b53d21dd8d9c26edbe2943151d9bcf9e5864ff998ede5f909118e23'
$StationFileSha256 = 'f22a9b66f734004ff14b6586a22a991d527f360806bbbdefe096e9f0474db72a'
$CompensatedFileSha256 = 'c003192963b257c8d497ffd347be2cd60695c5ce8653932403709d8193c88e55'
$ControllerGainHash = '8fee25a0339dd1e99127cbed912941dc3ad8ef2030ce49a0d310d1563cb87d98'
$CalibrationHash = 'f62648b57bd17a3503bcbdbf58f349f91fcd8de8ef0cf04551c200401233ed01'
$PostureArtifactHash = '3b96fd3dae66ad781b5b875c74184db101c42da02c53dfcc40a5137a6b5de11a'
$StationCalibrationHash = 'c00e859b3093b4812d54799253accdaeb99171a2cf4028b08bc39e68eaaa7d8a'
$MinimumCommandGainRatio = 0.70

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

$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $RepoRoot

foreach ($command in @('git', 'uv', 'nvidia-smi')) {
  if ($null -eq (Get-Command $command -ErrorAction SilentlyContinue)) {
    throw "Missing required command: $command"
  }
}

$SelfHashFile = Join-Path $PSScriptRoot 'run_hybrid_c1_affine_center_smoke_retry.ps1.sha256'
if (-not (Test-Path -LiteralPath $SelfHashFile -PathType Leaf)) {
  throw "Missing wrapper self-hash: $SelfHashFile"
}
$expectedSelfHash = (Get-Content -LiteralPath $SelfHashFile -Raw -Encoding ASCII).Trim()
$actualSelfHash = Get-FileSha256 -Path $PSCommandPath
if ($actualSelfHash -ne $expectedSelfHash) {
  throw 'Affine center-smoke retry wrapper self-hash mismatch.'
}

$branch = (git branch --show-current).Trim()
if ($branch -ne $RequiredBranch) {
  throw "Expected branch $RequiredBranch, got $branch."
}
if (@(git status --porcelain).Count -ne 0) {
  throw 'Repository must be clean before formal C1 retry.'
}
Invoke-NativeChecked -Executable 'git' -Arguments @(
  'merge-base', '--is-ancestor', $RequiredImplementation, 'HEAD'
) -FailureMessage 'Checkout predates the frozen affine C1 source run.'
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
$MjLabRoot = (Resolve-Path -LiteralPath (Join-Path $RepoRoot '..\mjlab-main')).Path
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
$env:PYTHONPATH = "$(Join-Path $RepoRoot 'src');$(Join-Path $RepoRoot 'src/hoppertrex_mjlab')"

$RuntimeArtifacts = Join-Path $RepoRoot 'docs/experiments/artifacts/hybrid_runtime_seed1'
$C1Artifacts = Join-Path $RepoRoot 'docs/experiments/artifacts/c1_posture_requalification_seed1'
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
    throw "Missing frozen artifact: $($entry.Key)"
  }
  if ((Get-FileSha256 -Path $entry.Key) -ne $entry.Value) {
    throw "Frozen artifact SHA256 mismatch: $($entry.Key)"
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

$SourceDirectory = Join-Path $RepoRoot 'experiments/c1_affine_identification_nodes_0c7bd78_seed1'
$SourceZip = $SourceDirectory + '.zip'
if (-not (Test-Path -LiteralPath $SourceDirectory -PathType Container)) {
  throw "Missing frozen affine source directory: $SourceDirectory"
}
if (-not (Test-Path -LiteralPath $SourceZip -PathType Leaf)) {
  throw "Missing frozen affine source ZIP: $SourceZip"
}
if ((Get-FileSha256 -Path $SourceZip) -ne $SourceZipSha256) {
  throw 'Frozen affine source ZIP SHA256 mismatch.'
}
$SourceManifest = Join-Path $SourceDirectory 'SHA256SUMS.txt'
$SourceProtocol = Join-Path $SourceDirectory 'protocol_note.json'
$SourceSmoke = Join-Path $SourceDirectory 'affine_center_smoke.json'
if ((Get-FileSha256 -Path $SourceManifest) -ne $SourceManifestSha256) {
  throw 'Frozen affine source SHA256SUMS.txt mismatch.'
}
if ((Get-FileSha256 -Path $SourceProtocol) -ne $SourceProtocolSha256) {
  throw 'Frozen affine source protocol_note.json mismatch.'
}
if ((Get-FileSha256 -Path $SourceSmoke) -ne $SourceSmokeSha256) {
  throw 'Frozen affine source smoke JSON mismatch.'
}
$sourceHashLines = Get-Content -LiteralPath $SourceManifest -Encoding ASCII
if ($sourceHashLines.Count -ne 30) {
  throw 'Frozen affine source must contain exactly 30 manifest entries.'
}
foreach ($line in $sourceHashLines) {
  if ($line -notmatch '^([0-9a-f]{64})  (.+)$') {
    throw "Malformed frozen source checksum line: $line"
  }
  $sourcePath = Join-Path $SourceDirectory $Matches[2]
  if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
    throw "Frozen source manifest file is missing: $sourcePath"
  }
  if ((Get-FileSha256 -Path $sourcePath) -ne $Matches[1]) {
    throw "Frozen source manifest mismatch: $sourcePath"
  }
}
$sourceProtocolPayload = Get-Content -LiteralPath $SourceProtocol -Raw -Encoding UTF8 | ConvertFrom-Json
$sourceSmokePayload = Get-Content -LiteralPath $SourceSmoke -Raw -Encoding UTF8 | ConvertFrom-Json
if (
  $sourceProtocolPayload.kind -ne 'c1_affine_identification_collection' -or
  $sourceProtocolPayload.git_sha -ne $SourceCollectionGit -or
  $sourceProtocolPayload.mjlab_git_sha -ne $RequiredMjLab -or
  @($sourceProtocolPayload.nodes).Count -ne 9 -or
  $sourceProtocolPayload.center_smoke.classification -ne 'AFFINE_CENTER_SMOKE_NO_CANDIDATE_STOP' -or
  $sourceProtocolPayload.center_smoke.incumbent_passed -ne $true -or
  [int]$sourceProtocolPayload.center_smoke.passed_candidate_count -ne 0 -or
  $sourceProtocolPayload.next_step -ne 'STOP' -or
  $sourceSmokePayload.git_sha -ne $SourceCollectionGit -or
  $sourceSmokePayload.classification -ne 'AFFINE_CENTER_SMOKE_NO_CANDIDATE_STOP' -or
  $sourceSmokePayload.incumbent.flat_gate_passed -ne $true -or
  [int]$sourceSmokePayload.passed_candidate_count -ne 0
) {
  throw 'Frozen affine source provenance or adjudication mismatch.'
}

$OutputDirectory = Join-Path $RepoRoot (
  'experiments/c1_affine_center_smoke_retry_' + $shortSha + '_seed1'
)
$OutputZip = $OutputDirectory + '.zip'
if ((Test-Path -LiteralPath $OutputDirectory) -or (Test-Path -LiteralPath $OutputZip)) {
  throw "Refusing to overwrite existing C1 retry output: $OutputDirectory"
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
$runtimeJson = & $Python -c 'import json, sys, mujoco, torch, warp; print(json.dumps(dict(python=sys.version.split()[0], torch=torch.__version__, torch_cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(), cuda_device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None), mujoco=getattr(mujoco, ''__version__'', None), warp=getattr(warp, ''__version__'', None))))'
if ($LASTEXITCODE -ne 0) {
  throw 'Unable to query Python runtime provenance.'
}
$runtime = $runtimeJson | ConvertFrom-Json
if ($runtime.cuda_available -ne $true) {
  throw 'PyTorch does not expose CUDA on this machine.'
}

$ResultPath = Join-Path $WorkingDirectory 'affine_center_smoke_retry.json'
$ConsolePath = Join-Path $WorkingDirectory 'console.log'
$evaluateArgs = @(
  '-u', '-m', 'hoppertrex_mjlab.scripts.evaluate_hybrid_c1_affine_center_smoke',
  '--nodes-dir', $SourceDirectory,
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
Invoke-NativeLogged -Executable $Python -Arguments $evaluateArgs -LogPath $ConsolePath -FailureMessage 'C1 affine center-smoke retry failed.'
if (-not (Test-Path -LiteralPath $ResultPath -PathType Leaf)) {
  throw 'Affine center-smoke retry did not write its adjudication.'
}
$result = Get-Content -LiteralPath $ResultPath -Raw -Encoding UTF8 | ConvertFrom-Json
$validClassifications = @(
  'AFFINE_CENTER_SMOKE_HAS_CANDIDATES',
  'AFFINE_CENTER_SMOKE_NO_CANDIDATE_STOP'
)
if (
  $validClassifications -notcontains $result.classification -or
  $result.git_sha -ne $fullSha -or
  $result.collection_git_sha -ne $SourceCollectionGit -or
  $result.mjlab_git_sha -ne $RequiredMjLab -or
  $result.incumbent.flat_gate_passed -ne $true -or
  $result.affine_incumbent.flat_gate_passed -ne $true -or
  [double]$result.affine_incumbent.anchor_alpha -ne 0.0 -or
  [int]$result.completed_candidate_count -ne 27 -or
  [int]$result.completed_node_fit_count -ne 243 -or
  [int]$result.fit_qualification.minimum_controllability_rank -ne 4 -or
  [double]$result.fit_qualification.maximum_heldout_nrmse -gt 0.15 -or
  [int]$result.fit_qualification.fallback_count -ne 0 -or
  @($result.candidates).Count -ne 27 -or
  $result.bindings.controller_gain_hash -ne $ControllerGainHash -or
  $result.bindings.velocity_calibration_hash -ne $CalibrationHash -or
  $result.bindings.posture_artifact_hash -ne $PostureArtifactHash -or
  $result.bindings.station_calibration_hash -ne $StationCalibrationHash -or
  $result.evidence_eligible -ne $true -or
  $result.promotion_eligible -ne $false -or
  $result.training_eligible -ne $false -or
  $null -ne $result.checkpoint
) {
  throw 'Affine center-smoke retry adjudication is incomplete or invalid.'
}
foreach ($candidate in @($result.candidates)) {
  if ([Math]::Abs([double]$candidate.anchor_alpha - 0.25) -gt 1.0e-15) {
    throw 'Retry candidate did not use the preregistered alpha=0.25 blend.'
  }
  foreach ($fact in $candidate.node_facts.PSObject.Properties.Value) {
    if ([double]$fact.command_gain_ratio -lt $MinimumCommandGainRatio) {
      throw 'Retry candidate violated the command-gain retention constraint.'
    }
  }
}
$passedCandidateCount = [int]$result.passed_candidate_count
$nextStep = [string]$result.next_step
if (
  ($result.classification -eq 'AFFINE_CENTER_SMOKE_HAS_CANDIDATES' -and
    ($passedCandidateCount -le 0 -or $nextStep -ne 'DOWNLOAD_FOR_REVIEW')) -or
  ($result.classification -eq 'AFFINE_CENTER_SMOKE_NO_CANDIDATE_STOP' -and
    ($passedCandidateCount -ne 0 -or $nextStep -ne 'STOP'))
) {
  throw 'Retry classification, pass count, and next step disagree.'
}

$protocol = [ordered]@{
  schema_version = 1
  kind = 'c1_affine_center_smoke_retry'
  git_sha = $fullSha
  mjlab_git_sha = $RequiredMjLab
  source_collection_git_sha = $SourceCollectionGit
  source_zip_sha256 = $SourceZipSha256
  source_manifest_sha256 = $SourceManifestSha256
  source_protocol_sha256 = $SourceProtocolSha256
  source_smoke_sha256 = $SourceSmokeSha256
  seed = 1
  device = 'cuda:0'
  gpu = [string]$gpuLine
  runtime = $runtime
  correction = [ordered]@{
    affine_incumbent_equivalence_required = $true
    command_gain_direction = @(0.0, 0.0, 1.0, 10.0)
    minimum_incumbent_command_gain_ratio = $MinimumCommandGainRatio
    expected_anchor_alpha = 0.25
    rejected_source_anchor_alpha = 0.50
  }
  classification = [string]$result.classification
  passed_candidate_count = $passedCandidateCount
  evidence_eligible = $true
  promotion_eligible = $false
  training_eligible = $false
  checkpoint = $null
  yaw_calibration_hash = $null
  next_step = $nextStep
}
$protocolPath = Join-Path $WorkingDirectory 'protocol_note.json'
$protocol | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $protocolPath -Encoding UTF8

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

Write-Host '[PASS] C1 affine center-smoke retry complete.'
Write-Host "RESULT=$OutputDirectory"
Write-Host "ZIP=$OutputZip"
Write-Host "ZIP_SHA256=$zipSha"
Write-Host "CLASSIFICATION=$($result.classification)"
Write-Host "PASSED_CANDIDATES=$passedCandidateCount/27"
Write-Host "NEXT=$nextStep"
