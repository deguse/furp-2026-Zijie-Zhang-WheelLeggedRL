[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$RequiredBranch = 'codex/p2-classical-upper-bound'
$RequiredImplementation = 'ffbb01850787ceead53ba407a0a7bf9c6f6a9b11'
$RequiredMjLab = '43e0f3ea9c92ddbb4de9f3bb1ac772d604e3ebf6'
$NodesDirectoryName = 'c1_identification_nodes_e54bd1a_seed1'
$NodesZipSha256 = '364590b8d9f2f5c66fdaac2b3fa124ee914236e33f6fc47e31e75f64d53c72e2'
$ControllerFileSha256 = '663ab77f77521581cde77ea2bd8c72c7f395f33b05b62348ef6d82a752aad7fc'
$CalibrationFileSha256 = 'ef002d0d622725509b47c8ff40d8af658fd42f705bdeac67ac35bae4458f889d'
$PostureFileSha256 = 'b8e627f85b53d21dd8d9c26edbe2943151d9bcf9e5864ff998ede5f909118e23'
$StationFileSha256 = 'f22a9b66f734004ff14b6586a22a991d527f360806bbbdefe096e9f0474db72a'
$CompensatedFileSha256 = 'c003192963b257c8d497ffd347be2cd60695c5ce8653932403709d8193c88e55'
$ControllerGainHash = '8fee25a0339dd1e99127cbed912941dc3ad8ef2030ce49a0d310d1563cb87d98'
$CalibrationHash = 'f62648b57bd17a3503bcbdbf58f349f91fcd8de8ef0cf04551c200401233ed01'
$PostureMapHash = '4289fb286c6a76a2aca2652d6bcc40acb1bf9c1f70b779a47ceff65c4dca3513'
$PostureArtifactHash = '3b96fd3dae66ad781b5b875c74184db101c42da02c53dfcc40a5137a6b5de11a'
$StationCalibrationHash = 'c00e859b3093b4812d54799253accdaeb99171a2cf4028b08bc39e68eaaa7d8a'

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

function Find-NodesDirectory {
  param([Parameter(Mandatory)][string]$Repository)
  $searchRoots = @(
    (Join-Path $Repository 'experiments'),
    $Repository
  )
  $candidate = Split-Path $Repository -Parent
  for ($depth = 0; $depth -lt 5; $depth++) {
    $searchRoots += $candidate
    $searchRoots += (Join-Path $candidate 'experiments')
    $parent = Split-Path $candidate -Parent
    if (-not $parent -or $parent -eq $candidate) {
      break
    }
    $candidate = $parent
  }
  foreach ($searchRoot in @($searchRoots | Select-Object -Unique)) {
    $nodes = Join-Path $searchRoot $NodesDirectoryName
    $nodesZip = $nodes + '.zip'
    if (
      (Test-Path -LiteralPath $nodes -PathType Container) -and
      (Test-Path -LiteralPath $nodesZip -PathType Leaf)
    ) {
      return (Resolve-Path -LiteralPath $nodes).Path
    }
  }
  throw 'Could not locate the frozen C1 nine-node directory and ZIP.'
}

foreach ($command in @('git', 'uv', 'nvidia-smi')) {
  if ($null -eq (Get-Command $command -ErrorAction SilentlyContinue)) {
    throw "Missing required command: $command"
  }
}
if (
  $PSVersionTable.PSEdition -ne 'Desktop' -or
  $PSVersionTable.PSVersion.Major -ne 5 -or
  $PSVersionTable.PSVersion.Minor -ne 1
) {
  throw 'Formal C1 flat-gate evidence requires Windows PowerShell 5.1.'
}

$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $RepoRoot
$SelfHashFile = Join-Path $PSScriptRoot 'run_hybrid_c1_flat_gate.ps1.sha256'
if (-not (Test-Path -LiteralPath $SelfHashFile -PathType Leaf)) {
  throw "Missing wrapper self-hash manifest: $SelfHashFile"
}
$expectedSelfHash = (Get-Content -LiteralPath $SelfHashFile -Raw -Encoding ASCII).Trim()
if ($expectedSelfHash -notmatch '^[0-9a-f]{64}$') {
  throw 'Wrapper self-hash manifest is malformed.'
}
$actualSelfHash = Get-FileSha256 -Path $PSCommandPath
if ($actualSelfHash -ne $expectedSelfHash) {
  throw 'Wrapper self-hash mismatch; fetch the published branch again.'
}

$branch = (git branch --show-current).Trim()
if ($branch -ne $RequiredBranch) {
  throw "Expected branch $RequiredBranch, got $branch."
}
if (@(git status --porcelain).Count -ne 0) {
  throw 'Repository must be clean before formal C1 flat-gate evaluation.'
}
Invoke-NativeChecked -Executable 'git' -Arguments @(
  'merge-base', '--is-ancestor', $RequiredImplementation, 'HEAD'
) -FailureMessage 'Checkout predates the qualified flat-gate implementation.'
Invoke-NativeChecked -Executable 'git' -Arguments @(
  'fetch', '--quiet', 'origin', $RequiredBranch
) -FailureMessage 'Failed to refresh the remote C1 branch.'
$fullSha = (git rev-parse HEAD).Trim()
$remoteSha = (git rev-parse "origin/$RequiredBranch").Trim()
if ($fullSha -ne $remoteSha) {
  throw "Checkout HEAD $fullSha does not match remote HEAD $remoteSha."
}
$shortSha = (git rev-parse --short=7 HEAD).Trim()

$NodesDirectory = Find-NodesDirectory -Repository $RepoRoot
$NodesZip = $NodesDirectory + '.zip'
$PyProject = Join-Path $RepoRoot 'pyproject.toml'
$MjLabSourceDeclaration = 'mjlab = { path = "../mjlab-main", editable = true }'
$PyProjectLines = @(Get-Content -LiteralPath $PyProject -Encoding UTF8)
if ($PyProjectLines -notcontains $MjLabSourceDeclaration) {
  throw 'pyproject.toml no longer pins the expected editable MjLab source.'
}
$MjLabDeclaredRoot = (Resolve-Path -LiteralPath (
  Join-Path $RepoRoot '..\mjlab-main'
)).Path
$mjlabTopLevel = (& git -C $MjLabDeclaredRoot rev-parse --show-toplevel).Trim()
if ($LASTEXITCODE -ne 0 -or -not $mjlabTopLevel) {
  throw 'Editable MjLab source is not a Git checkout.'
}
$MjLabRoot = (Resolve-Path -LiteralPath $mjlabTopLevel).Path
if (@(git -C $MjLabRoot status --porcelain).Count -ne 0) {
  throw 'Editable MjLab checkout must be clean.'
}
$mjlabSha = (git -C $MjLabRoot rev-parse HEAD).Trim()
if ($mjlabSha -ne $RequiredMjLab) {
  throw "MjLab must be pinned to $RequiredMjLab, got $mjlabSha."
}
if ((Get-FileSha256 -Path $NodesZip) -ne $NodesZipSha256) {
  throw 'Frozen C1 nine-node ZIP SHA256 mismatch.'
}

Invoke-NativeChecked -Executable 'uv' -Arguments @(
  'sync', '--frozen', '--python', '3.11'
) -FailureMessage 'uv sync failed.'
$Python = Join-Path $RepoRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
  throw "Missing project Python after uv sync: $Python"
}
$importedMjLabRoot = (& $Python -c 'import pathlib, mjlab; print(pathlib.Path(mjlab.__file__).resolve().parents[2])').Trim()
if ($LASTEXITCODE -ne 0) {
  throw 'Unable to resolve the MjLab package imported by the project environment.'
}
if ((Resolve-Path -LiteralPath $importedMjLabRoot).Path -ne $MjLabRoot) {
  throw "Python imports MjLab from $importedMjLabRoot, expected $MjLabRoot."
}

$RuntimeArtifacts = Join-Path $RepoRoot 'docs/experiments/artifacts/hybrid_runtime_seed1'
$C1Artifacts = Join-Path $RepoRoot 'docs/experiments/artifacts/c1_posture_requalification_seed1'
$Controller = Join-Path $RuntimeArtifacts 'controller_seed1.json'
$Calibration = Join-Path $RuntimeArtifacts 'velocity_calibration_seed1.json'
$Posture = Join-Path $C1Artifacts 'posture_map_seed1_registered_p032.json'
$Station = Join-Path $C1Artifacts 'station_calibration_seed1.json'
$Compensated = Join-Path $C1Artifacts 'balance_compensated_seed1.json'
$ExpectedFiles = [ordered]@{
  $Controller = $ControllerFileSha256
  $Calibration = $CalibrationFileSha256
  $Posture = $PostureFileSha256
  $Station = $StationFileSha256
  $Compensated = $CompensatedFileSha256
}
foreach ($entry in $ExpectedFiles.GetEnumerator()) {
  if (-not (Test-Path -LiteralPath $entry.Key -PathType Leaf)) {
    throw "Missing frozen artifact: $($entry.Key)"
  }
  if ((Get-FileSha256 -Path $entry.Key) -ne $entry.Value) {
    throw "Frozen artifact SHA256 mismatch: $($entry.Key)"
  }
}

$controllerPayload = Get-Content -LiteralPath $Controller -Raw -Encoding UTF8 |
  ConvertFrom-Json
$calibrationPayload = Get-Content -LiteralPath $Calibration -Raw -Encoding UTF8 |
  ConvertFrom-Json
$posturePayload = Get-Content -LiteralPath $Posture -Raw -Encoding UTF8 |
  ConvertFrom-Json
$stationPayload = Get-Content -LiteralPath $Station -Raw -Encoding UTF8 |
  ConvertFrom-Json
$compensatedPayload = Get-Content -LiteralPath $Compensated -Raw -Encoding UTF8 |
  ConvertFrom-Json
if (
  $controllerPayload.controller_type -ne 'lqr' -or
  $controllerPayload.gain_hash -ne $ControllerGainHash
) {
  throw 'Controller artifact binding mismatch.'
}
if (
  $calibrationPayload.calibration_hash -ne $CalibrationHash -or
  $calibrationPayload.controller_gain_hash -ne $ControllerGainHash
) {
  throw 'Velocity calibration binding mismatch.'
}
if (
  $posturePayload.map_hash -ne $PostureMapHash -or
  $posturePayload.posture_artifact_hash -ne $PostureArtifactHash -or
  $posturePayload.envelope_verification.method -ne
    'registered_fixed_symmetric_hull_rectangle'
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
  $compensatedPayload.controller_gain_hash -ne $ControllerGainHash -or
  $compensatedPayload.calibration_hash -ne $CalibrationHash -or
  $compensatedPayload.posture_map_hash -ne $PostureMapHash -or
  $compensatedPayload.posture_artifact_hash -ne $PostureArtifactHash -or
  $compensatedPayload.station_calibration_hash -ne $StationCalibrationHash
) {
  throw 'Compensated qualification binding mismatch.'
}

$env:PYTHONPATH = "$(Join-Path $RepoRoot 'src');$(Join-Path $RepoRoot 'src/hoppertrex_mjlab')"
$env:HOPPERTREX_HYBRID_CONTROLLER_PATH = $Controller
$env:HOPPERTREX_HYBRID_CALIBRATION_PATH = $Calibration
$env:HOPPERTREX_HYBRID_POSTURE_MAP_PATH = $Posture
$env:HOPPERTREX_HYBRID_STATION_CALIBRATION_PATH = $Station
Remove-Item Env:HOPPERTREX_HYBRID_YAW_CALIBRATION_PATH -ErrorAction SilentlyContinue

$gpuLines = @(& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)
$gpuLine = $gpuLines | Select-Object -First 1
if ($LASTEXITCODE -ne 0 -or -not $gpuLine) {
  throw 'Unable to query GPU provenance with nvidia-smi.'
}
$runtimeJson = & $Python -c 'import json, sys, mujoco, torch, warp; print(json.dumps(dict(python=sys.version.split()[0], torch=torch.__version__, torch_cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(), cuda_device_count=torch.cuda.device_count(), cuda_device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None), mujoco=getattr(mujoco, ''__version__'', None), warp=getattr(warp, ''__version__'', None))))'
if ($LASTEXITCODE -ne 0) {
  throw 'Unable to query Python runtime provenance.'
}
$runtime = $runtimeJson | ConvertFrom-Json
if ($runtime.cuda_available -ne $true -or [int]$runtime.cuda_device_count -lt 1) {
  throw 'PyTorch does not expose CUDA device 0.'
}
Invoke-NativeChecked -Executable $Python -Arguments @(
  '-m', 'hoppertrex_mjlab.scripts.evaluate_hybrid_c1_flat_gate', '--help'
) -FailureMessage 'C1 flat-gate evaluator --help failed.'

$OutputDirectory = Join-Path $RepoRoot (
  'experiments/c1_flat_gate_' + $shortSha + '_seed1'
)
$OutputZip = $OutputDirectory + '.zip'
if (
  (Test-Path -LiteralPath $OutputDirectory) -or
  (Test-Path -LiteralPath $OutputZip)
) {
  throw "Refusing to overwrite existing C1 flat-gate output: $OutputDirectory"
}
$runToken = [Guid]::NewGuid().ToString('N')
$WorkingDirectory = $OutputDirectory + '.incomplete.' + $runToken
$WorkingZip = $WorkingDirectory + '.zip'
$ConsoleTemp = $WorkingDirectory + '.console.log'
New-Item -ItemType Directory -Path $WorkingDirectory | Out-Null

$evaluateArgs = @(
  '-u', '-m', 'hoppertrex_mjlab.scripts.evaluate_hybrid_c1_flat_gate',
  '--nodes-dir', $NodesDirectory,
  '--output-dir', $WorkingDirectory,
  '--compensated-qualification', $Compensated,
  '--mjlab-git-sha', $RequiredMjLab,
  '--task', 'HopperTrex-Hybrid-v2-Stage3',
  '--device', 'cuda:0',
  '--num-envs', '16',
  '--settle-steps', '100',
  '--measure-steps', '200',
  '--vx-check', '0.05'
)
try {
  Invoke-NativeLogged -Executable $Python -Arguments $evaluateArgs -LogPath $ConsoleTemp -FailureMessage 'C1 flat-gate evaluator failed.'
} catch {
  if (Test-Path -LiteralPath $ConsoleTemp -PathType Leaf) {
    Move-Item -LiteralPath $ConsoleTemp -Destination (
      Join-Path $WorkingDirectory 'console.log'
    )
  }
  Write-Warning "Incomplete C1 flat-gate output retained: $WorkingDirectory"
  throw
}

$Detail = Join-Path $WorkingDirectory 'flat_gate_evaluation_detail.json'
$Adjudication = Join-Path $WorkingDirectory 'flat_gate_adjudication.json'
$Selection = Join-Path $WorkingDirectory 'flat_gate_selection.json'
foreach ($path in @($Detail, $Adjudication)) {
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    throw "Evaluator omitted required output: $path"
  }
}
$adjudicationPayload = Get-Content -LiteralPath $Adjudication -Raw -Encoding UTF8 |
  ConvertFrom-Json
$validClassifications = @(
  'C1_FLAT_GATE_SELECTED',
  'NO_QR_CANDIDATE_PASSED_FLAT_GATE'
)
if (
  $validClassifications -notcontains $adjudicationPayload.classification -or
  $adjudicationPayload.git_sha -ne $fullSha -or
  $adjudicationPayload.mjlab_git_sha -ne $RequiredMjLab -or
  [int]$adjudicationPayload.completed_candidate_count -ne 27 -or
  [int]$adjudicationPayload.completed_node_fit_count -ne 243 -or
  $adjudicationPayload.evidence_eligible -ne $true -or
  $adjudicationPayload.promotion_eligible -ne $false -or
  $adjudicationPayload.training_eligible -ne $false -or
  $null -ne $adjudicationPayload.checkpoint -or
  $null -ne $adjudicationPayload.yaw_calibration_hash
) {
  throw 'Evaluator adjudication provenance or eligibility mismatch.'
}
$detailHash = Get-FileSha256 -Path $Detail
if ($adjudicationPayload.evaluation_detail_sha256 -ne $detailHash) {
  throw 'Evaluator detail SHA256 does not match adjudication.'
}
if (
  [Math]::Abs(
    [double]$adjudicationPayload.caps.worst_velocity_error -
    0.01040513883344829
  ) -gt 1.0e-15 -or
  [Math]::Abs(
    [double]$adjudicationPayload.caps.p95_pitch -
    0.023371753748506308
  ) -gt 1.0e-15 -or
  [Math]::Abs(
    [double]$adjudicationPayload.caps.p99_pitch_rate -
    0.3286285623908043
  ) -gt 1.0e-15
) {
  throw 'Evaluator caps do not match the preregistered 1.5x floors.'
}
if (
  $adjudicationPayload.classification -eq 'C1_FLAT_GATE_SELECTED' -and
  -not (Test-Path -LiteralPath $Selection -PathType Leaf)
) {
  throw 'Selected adjudication omitted flat_gate_selection.json.'
}
if (
  $adjudicationPayload.classification -eq 'C1_FLAT_GATE_SELECTED' -and
  (
    $adjudicationPayload.next_step -ne
      'DOWNLOAD_FOR_OFFLINE_SCHEDULE_BUILD' -or
    $adjudicationPayload.selection_sha256 -ne
      (Get-FileSha256 -Path $Selection)
  )
) {
  throw 'Selected adjudication has an invalid next step or selection SHA256.'
}
if (
  $adjudicationPayload.classification -eq
    'NO_QR_CANDIDATE_PASSED_FLAT_GATE' -and
  (Test-Path -LiteralPath $Selection)
) {
  throw 'All-failed adjudication must not write flat_gate_selection.json.'
}
if (
  $adjudicationPayload.classification -eq
    'NO_QR_CANDIDATE_PASSED_FLAT_GATE' -and
  (
    $adjudicationPayload.next_step -ne 'STOP' -or
    $null -ne $adjudicationPayload.selection_sha256
  )
) {
  throw 'All-failed adjudication has an invalid next step or selection SHA256.'
}

Move-Item -LiteralPath $ConsoleTemp -Destination (
  Join-Path $WorkingDirectory 'console.log'
)
$protocol = [ordered]@{
  schema_version = 1
  kind = 'c1_flat_gate_machine_room_run'
  git_sha = $fullSha
  mjlab_git_sha = $RequiredMjLab
  collection_git_sha = 'e54bd1a604b08b634821d88ce3a53a0f2fe66724'
  nodes_zip_sha256 = $NodesZipSha256
  seed = 1
  device = 'cuda:0'
  gpu = [string]$gpuLine
  powershell = [ordered]@{
    edition = [string]$PSVersionTable.PSEdition
    version = [string]$PSVersionTable.PSVersion
  }
  runtime = $runtime
  fixed_protocol = [ordered]@{
    task = 'HopperTrex-Hybrid-v2-Stage3'
    num_envs = 16
    settle_steps = 100
    measure_steps = 200
    vx_check_m_s = 0.05
    candidate_count = 27
    nodes_per_candidate = 9
    cells_per_candidate = 15
    environment_process_count = 1
    reset_before_each_cell = $true
    protocol_scope = 'qr_selection_screen_not_cstar_formal_validation'
  }
  classification = [string]$adjudicationPayload.classification
  passed_candidate_count = [int]$adjudicationPayload.passed_candidate_count
  evidence_eligible = $true
  promotion_eligible = $false
  training_eligible = $false
  checkpoint = $null
  yaw_calibration_hash = $null
  next_step = [string]$adjudicationPayload.next_step
}
$ProtocolPath = Join-Path $WorkingDirectory 'protocol_note.json'
$protocol | ConvertTo-Json -Depth 12 |
  Set-Content -LiteralPath $ProtocolPath -Encoding UTF8

$ChecksumPath = Join-Path $WorkingDirectory 'SHA256SUMS.txt'
$hashLines = Get-ChildItem -LiteralPath $WorkingDirectory -File |
  Where-Object { $_.Name -ne 'SHA256SUMS.txt' } |
  Sort-Object Name |
  ForEach-Object {
    "$(Get-FileSha256 -Path $_.FullName)  $($_.Name)"
  }
$hashLines | Set-Content -LiteralPath $ChecksumPath -Encoding ASCII

Compress-Archive -Path (Join-Path $WorkingDirectory '*') -DestinationPath $WorkingZip -CompressionLevel Optimal
Move-Item -LiteralPath $WorkingZip -Destination $OutputZip
try {
  Move-Item -LiteralPath $WorkingDirectory -Destination $OutputDirectory
} catch {
  Remove-Item -LiteralPath $OutputZip -Force -ErrorAction SilentlyContinue
  throw
}
$zipSha = Get-FileSha256 -Path $OutputZip

Write-Host '[PASS] C1 27-candidate flat gate complete.'
Write-Host "RESULT=$OutputDirectory"
Write-Host "ZIP=$OutputZip"
Write-Host "ZIP_SHA256=$zipSha"
Write-Host "CLASSIFICATION=$($adjudicationPayload.classification)"
Write-Host "NEXT=$($adjudicationPayload.next_step)"
