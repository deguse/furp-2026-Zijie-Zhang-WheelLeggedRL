[CmdletBinding()]
param(
  [ValidateSet('Validate','Probe10','Probe20','Probe30')][string]$Phase='Validate',
  [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-fA-F]{40}$')][string]$ExpectedGitSha,
  [Parameter(Mandatory=$true)][string]$CampaignRoot,
  [string]$Python,
  [string]$Device='cuda:0'
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$Branch='codex/p2-classical-upper-bound'
$MjLabSha='43e0f3ea9c92ddbb4de9f3bb1ac772d604e3ebf6'
$ScheduleHash='8fe8548bca85978c164bbd7de39d2d6463cdfd8d7ab91796cf57696b0f64e203'
$Artifacts=[ordered]@{
  'docs\experiments\artifacts\c1_schedule_candidate24_1f54968_seed1\c1_schedule.json'='9b21125e7cc48be3ea61e12a67171a855892ad3ced1f54b3176ed979e76224ec'
  'docs\experiments\artifacts\hybrid_runtime_seed1\velocity_calibration_seed1.json'='ef002d0d622725509b47c8ff40d8af658fd42f705bdeac67ac35bae4458f889d'
  'docs\experiments\artifacts\yaw_gpu_3f8a9330b88fa6129d05ce42ac3a8cc835295a6f_seed1\yaw_calibration.json'='123122e75955468dfc475d86ac3f9160b428720fd8e1b90ab614bc1bc0749765'
  'docs\experiments\artifacts\c1_posture_requalification_seed1\posture_map_seed1_registered_p032.json'='b8e627f85b53d21dd8d9c26edbe2943151d9bcf9e5864ff998ede5f909118e23'
  'docs\experiments\artifacts\c1_posture_requalification_seed1\station_calibration_seed1.json'='f22a9b66f734004ff14b6586a22a991d527f360806bbbdefe096e9f0474db72a'
}
function Fail([string]$Message){throw "ROLL_BOUNDARY: $Message"}
function Need([string]$Path){if(-not(Test-Path -LiteralPath $Path -PathType Leaf)){Fail "Missing file: $Path"}}
function Sha([string]$Path){(Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()}
$Repo=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $Repo
if((git branch --show-current).Trim() -ne $Branch){Fail "Expected branch $Branch"}
$Head=(git rev-parse HEAD).Trim()
if($Head -ne $ExpectedGitSha.ToLowerInvariant()){Fail "HEAD $Head != expected SHA"}
if(@(git status --porcelain).Count -ne 0){Fail 'Formal probe requires a clean worktree'}
git fetch --quiet origin $Branch
if($LASTEXITCODE -ne 0 -or (git rev-parse "origin/$Branch").Trim() -ne $Head){Fail 'HEAD does not match origin branch'}
$MjLab=(Resolve-Path -LiteralPath (Join-Path $Repo '..\mjlab-main')).Path
if((git -C $MjLab rev-parse HEAD).Trim() -ne $MjLabSha){Fail 'MjLab SHA drifted'}
foreach($pair in $Artifacts.GetEnumerator()){
  $path=Join-Path $Repo $pair.Key; Need $path
  if((Sha $path) -ne $pair.Value){Fail "Artifact SHA drifted: $($pair.Key)"}
}
if([string]::IsNullOrWhiteSpace($Python)){$Python=Join-Path $Repo '.venv\Scripts\python.exe'}
Need $Python
$sourcePath=(Resolve-Path 'src').Path
$packagePath=(Resolve-Path 'src\hoppertrex_mjlab').Path
$env:PYTHONPATH="$sourcePath;$packagePath"
$Root=[IO.Path]::GetFullPath($CampaignRoot)
if(-not $Root.StartsWith([IO.Path]::GetPathRoot($Root),[StringComparison]::OrdinalIgnoreCase)){Fail 'CampaignRoot is invalid'}
if($Phase-ne'Validate'-and$Device-ne'cuda:0'){Fail 'Formal RollBoundary phases are pinned to cuda:0'}
if($Phase -eq 'Validate'){
  & $Python -m hoppertrex_mjlab.scripts.probe_roll_boundary --help | Out-Null
  if($LASTEXITCODE -ne 0){Fail 'Probe --help failed'}
  Write-Host '[PASS] RollBoundary wrapper validated.'
  exit 0
}
$Max=[int]$Phase.Substring(5)
$Name="r0_roll_boundary_${Max}mm_seed1"
$Final=Join-Path $Root $Name
if(Test-Path -LiteralPath $Final){Fail "Refusing to overwrite $Final"}
$Work=Join-Path $Root (".$Name.incomplete."+[Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $Work -Force | Out-Null
try {
  $Output=Join-Path $Work 'roll_boundary.json'
  & $Python -m hoppertrex_mjlab.scripts.probe_roll_boundary --output $Output --device $Device --max-height-mm $Max 2>&1 | Tee-Object -FilePath (Join-Path $Work 'console.log')
  if($LASTEXITCODE -ne 0){Fail 'RollBoundary probe failed'}
  $Result=Get-Content -LiteralPath $Output -Raw -Encoding UTF8 | ConvertFrom-Json
  if($Result.evidence_eligible -ne $true -or $Result.seed -ne 1 -or $Result.device -ne 'cuda:0'){Fail 'Evidence eligibility drifted'}
  if($Result.git_sha-ne$Head-or$Result.mjlab_git_sha-ne$MjLabSha){Fail 'Probe Git provenance drifted'}
  if($Result.controller_schedule_hash -ne $ScheduleHash -or @($Result.action_mask|Where-Object{$_ -ne $false}).Count -ne 0){Fail 'Classical stack/residual contract drifted'}
  if(@($Result.protocol.heights_m).Count -ne ($Max/2.5+1)){Fail 'Height grid count drifted'}
  for($i=0;$i -lt @($Result.protocol.heights_m).Count;$i++){
    if([Math]::Abs([double]$Result.protocol.heights_m[$i]-0.0025*$i)-gt 1e-12){Fail 'Height grid drifted'}
  }
  if(@($Result.trials).Count -ne (2*3*16*@($Result.protocol.heights_m).Count)){Fail 'Trial count drifted'}
  if($Result.protocol.terrain -ne 'flat_box_at_zero_else_pyramid_stairs'){Fail 'Zero-height terrain fix drifted'}
  if($Result.protocol.strict_physics_substep_support_required -ne $true){Fail 'Strict 5 ms support latch is disabled'}
  if([Math]::Abs([double]$Result.protocol.wheel_contact_solref[0]-0.020)-gt 1e-12 -or
     [Math]::Abs([double]$Result.protocol.wheel_contact_solref[1]-1.0)-gt 1e-12){Fail 'Wheel contact solref drifted'}
  $ExpectedSolimp=@(0.90,0.95,0.001)
  for($i=0;$i -lt 3;$i++){
    if([Math]::Abs([double]$Result.protocol.wheel_contact_solimp[$i]-$ExpectedSolimp[$i])-gt 1e-12){Fail 'Wheel contact solimp drifted'}
  }
  $SubstepTrials=@($Result.trials|Where-Object{[int]$_.bilateral_unsupported_physics_substeps -gt 0})
  if(@($SubstepTrials|Where-Object{$_.bilateral_airborne_ever -ne $true -or $_.success -eq $true}).Count -gt 0){
    Fail 'Substep support event was not fail-closed latched'
  }
  if($Max-eq10-and$Result.classification-eq'EXTEND_ROLL_BOUNDARY_SWEEP'){
    Write-Warning '10 mm all passed: run Probe20 next; do not start RollAssist.'
  }elseif($Max-eq20-and$Result.classification-eq'EXTEND_ROLL_BOUNDARY_SWEEP'){
    Write-Warning '20 mm all passed: run Probe30 next; do not start RollAssist.'
  }elseif($Result.classification-eq'CLASSICAL_CROLL_BRACKETED'){
    Write-Host '[PASS] Safe positive bracket: reward calibration and RollAssist are eligible.'
  }elseif($Result.classification-eq'CLASSICAL_CROLL_AT_LEAST_CAP'){
    Write-Warning 'Croll,classical >= 30 mm; current paper stops without RollAssist.'
  }elseif($Result.classification-in@('NO_POSITIVE_CLASSICAL_CROLL','NEXT_HEIGHT_UNSAFE_STOP','INVALID_FLAT_CONTROL_STOP','NON_MONOTONIC_STOP')){
    Write-Warning "R0 classification $($Result.classification): stair PPO is forbidden."
  }else{
    Fail "Unexpected R0 classification: $($Result.classification)"
  }
  "$(Sha $Output)  roll_boundary.json" | Set-Content -LiteralPath (Join-Path $Work 'SHA256SUMS.txt') -Encoding ASCII
  Move-Item -LiteralPath $Work -Destination $Final
} finally {
  if(Test-Path -LiteralPath $Work){Remove-Item -LiteralPath $Work -Recurse -Force}
}
Write-Host "[PASS] RESULT=$(Join-Path $Final 'roll_boundary.json')"
