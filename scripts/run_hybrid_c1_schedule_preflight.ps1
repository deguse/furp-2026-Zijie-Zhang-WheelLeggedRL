$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Repository = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Repository

$ExpectedBranch = "codex/p2-classical-upper-bound"
$RequiredBase = "59ff3cd4d86c569d7d0ea8e207640a6d11c178ab"
$RequiredMjLab = "43e0f3ea9c92ddbb4de9f3bb1ac772d604e3ebf6"
$HeightNodes = @(0.2907321708, 0.3092089487, 0.3276857266)
$PitchCandidates = @(0.032, 0.024, 0.016)

if ((git branch --show-current).Trim() -ne $ExpectedBranch) {
  throw "Expected branch $ExpectedBranch."
}
git merge-base --is-ancestor $RequiredBase HEAD
if ($LASTEXITCODE -ne 0) { throw "Branch does not contain required C0 archive." }
if (git status --porcelain) { throw "Repository must be clean." }

foreach ($Command in @("git", "uv", "nvidia-smi")) {
  if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
    throw "Missing required command: $Command"
  }
}

$MjLab = (Resolve-Path (Join-Path $Repository "..\\..\\..\\mjlab-main")).Path
if ((git -C $MjLab rev-parse HEAD).Trim() -ne $RequiredMjLab) {
  throw "MjLab must be pinned to $RequiredMjLab."
}
if (git -C $MjLab status --porcelain) { throw "MjLab checkout must be clean." }

$RequiredArtifacts = @{
  controller = $env:HOPPERTREX_HYBRID_CONTROLLER_PATH
  calibration = $env:HOPPERTREX_HYBRID_CALIBRATION_PATH
  posture = $env:HOPPERTREX_HYBRID_POSTURE_MAP_PATH
  station = $env:HOPPERTREX_HYBRID_STATION_CALIBRATION_PATH
}
foreach ($Entry in $RequiredArtifacts.GetEnumerator()) {
  if (-not $Entry.Value -or -not (Test-Path -LiteralPath $Entry.Value -PathType Leaf)) {
    throw "Missing $($Entry.Key) artifact path."
  }
}

$Posture = Get-Content -LiteralPath $RequiredArtifacts.posture -Raw -Encoding UTF8 | ConvertFrom-Json
$HeightRange = @($Posture.training_envelope.height)
$PitchRange = @($Posture.training_envelope.pitch)
foreach ($Height in $HeightNodes) {
  if ($Height -lt $HeightRange[0] -or $Height -gt $HeightRange[1]) {
    throw "Posture artifact does not cover height node $Height."
  }
}
$SelectedPitch = $null
foreach ($Bound in $PitchCandidates) {
  if ($PitchRange[0] -le -$Bound -and $PitchRange[1] -ge $Bound) {
    $SelectedPitch = $Bound
    break
  }
}
if ($null -eq $SelectedPitch) {
  throw "No qualified symmetric pitch range covers +/-0.016 rad. Re-run posture qualification."
}

$Python = Join-Path $Repository ".venv\\Scripts\\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
  uv sync --frozen --python 3.11
}
& $Python -m hoppertrex_mjlab.scripts.collect_hybrid_identification --help | Out-Null
& $Python -m hoppertrex_mjlab.scripts.build_hybrid_controller_schedule --help | Out-Null
& $Python -m hoppertrex_mjlab.scripts.fit_hybrid_stair_contact_detector --help | Out-Null

nvidia-smi | Out-Null
Write-Host "[PASS] C1 schedule preflight complete."
Write-Host "HEIGHT_NODES=$($HeightNodes -join ',')"
Write-Host "PITCH_NODES=$(-$SelectedPitch),0,$SelectedPitch"
Write-Host "C1_GPU_NODE_COLLECTION_READY_NO_TRAINING"
