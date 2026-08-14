[CmdletBinding()]
param(
  [ValidateSet('Validate','MeasureReward','CalibrateReward','Train100','Envelope','Screen','SelectK3','Evaluate','ExtendBlock','Package')]
  [string]$Phase='Validate',
  [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-fA-F]{40}$')][string]$ExpectedGitSha,
  [Parameter(Mandatory=$true)][string]$CampaignRoot,
  [Parameter(Mandatory=$true)][string]$RollBoundary,
  [string]$StallArtifact,
  [string]$RewardCalibration,
  [string]$Checkpoint,
  [string[]]$CheckpointEnvelope,
  [string[]]$ScreenEvidence,
  [string]$Evidence,
  [string]$Selection,
  [int]$SelectedCompletedUpdates=0,
  [int]$TargetTotalUpdates=0,
  [string]$ResumeRun,
  [string]$Python,
  [string]$Device='cuda:0'
)
$ErrorActionPreference='Stop'; Set-StrictMode -Version Latest
$Task='HopperTrex-Hybrid-v2-StairRollAssist'
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
function Fail([string]$Message){throw "ROLL_ASSIST: $Message"}
function Need([string]$Path){if(-not(Test-Path -LiteralPath $Path -PathType Leaf)){Fail "Missing file: $Path"}}
function Sha([string]$Path){(Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()}
$Repo=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path; Set-Location $Repo
if((git branch --show-current).Trim() -ne $Branch){Fail 'Wrong branch'}
$Head=(git rev-parse HEAD).Trim(); if($Head -ne $ExpectedGitSha.ToLowerInvariant()){Fail 'Git SHA mismatch'}
$RepoStatus=@(git status --porcelain)
if($LASTEXITCODE -ne 0){Fail 'Unable to inspect repository worktree'}
if($RepoStatus.Count -ne 0){Fail 'Formal RollAssist requires a clean worktree'}
git fetch --quiet origin $Branch
if($LASTEXITCODE -ne 0 -or (git rev-parse "origin/$Branch").Trim() -ne $Head){Fail 'Origin mismatch'}
$MjLab=(Resolve-Path -LiteralPath (Join-Path $Repo '..\mjlab-main')).Path
$MjLabHead=(git -C $MjLab rev-parse HEAD).Trim()
if($LASTEXITCODE -ne 0 -or $MjLabHead -ne $MjLabSha){Fail 'MjLab SHA mismatch'}
$MjLabStatus=@(git -C $MjLab status --porcelain)
if($LASTEXITCODE -ne 0){Fail 'Unable to inspect MjLab worktree'}
if($MjLabStatus.Count -ne 0){Fail 'Formal RollAssist requires a clean MjLab worktree'}
foreach($pair in $Artifacts.GetEnumerator()){
  $path=Join-Path $Repo $pair.Key;Need $path
  if((Sha $path)-ne $pair.Value){Fail "Artifact SHA drifted: $($pair.Key)"}
}
Need $RollBoundary
$R0=Get-Content -LiteralPath $RollBoundary -Raw -Encoding UTF8|ConvertFrom-Json
if($R0.probe-ne'hoppertrex_roll_boundary_r0'){Fail 'RollBoundary artifact identity drifted'}
if($R0.git_sha-ne$Head){Fail 'RollBoundary Git SHA differs from the current checkout'}
if($R0.controller_schedule_hash-ne$ScheduleHash){Fail 'RollBoundary controller schedule differs from the frozen C1 schedule'}
if($R0.classification-ne'CLASSICAL_CROLL_BRACKETED'-or$R0.training_eligible-ne$true-or$R0.verdict.next_height_unsafe-ne$false){Fail 'RollBoundary does not authorize RollAssist'}

if([string]::IsNullOrWhiteSpace($Python)){$Python=Join-Path $Repo '.venv\Scripts\python.exe'}; Need $Python
$sourcePath=(Resolve-Path 'src').Path
$packagePath=(Resolve-Path 'src\hoppertrex_mjlab').Path
$env:PYTHONPATH="$sourcePath;$packagePath"
$env:HOPPERTREX_ROLL_ASSIST_R0_PATH=(Resolve-Path $RollBoundary).Path
$env:HOPPERTREX_ROLL_ASSIST_EXPECTED_GIT_SHA=$Head
& $Python -c "import os; from pathlib import Path; from hoppertrex_mjlab.hybrid.roll_assist import load_roll_boundary_verdict; load_roll_boundary_verdict(Path(os.environ['HOPPERTREX_ROLL_ASSIST_R0_PATH']), expected_git_sha=os.environ['HOPPERTREX_ROLL_ASSIST_EXPECTED_GIT_SHA'])"
if($LASTEXITCODE -ne 0){Fail 'RollBoundary evidence contract validation failed'}
$Root=[IO.Path]::GetFullPath($CampaignRoot); New-Item -ItemType Directory -Path $Root -Force|Out-Null
$Reward=Join-Path $Root 'reward_calibration.json'
if($Phase -eq 'Validate'){
  foreach($module in @('hoppertrex_mjlab.scripts.probe_roll_assist_reward_stall','hoppertrex_mjlab.scripts.calibrate_roll_assist_reward','hoppertrex_mjlab.scripts.evaluate_roll_assist','hoppertrex_mjlab.scripts.adjudicate_roll_assist','hoppertrex_mjlab.scripts.rsl_rl.train')){
    & $Python -m $module --help | Out-Null; if($LASTEXITCODE-ne0){Fail "$module --help failed"}
  }
  Write-Host '[PASS] RollAssist wrapper validated.'; exit 0
}
if($Phase-ne'Validate'-and$Phase-ne'Envelope'-and$Phase-ne'SelectK3'-and$Phase-ne'Package'-and$Device-ne'cuda:0'){Fail 'Evidence collection and training phases are pinned to cuda:0'}

if($Phase -eq 'MeasureReward'){
  $StallArtifact=Join-Path $Root 'reward_stall.json'
  if(Test-Path $StallArtifact){Fail "Refusing to overwrite $StallArtifact"}
  & $Python -m hoppertrex_mjlab.scripts.probe_roll_assist_reward_stall --roll-boundary $RollBoundary --output $StallArtifact --device $Device
  if($LASTEXITCODE-ne0){Fail 'Reward stall measurement failed'}
  Write-Host "[PASS] STALL=$StallArtifact"; exit 0
}
if($Phase -eq 'CalibrateReward'){
  Need $StallArtifact
  if(Test-Path $Reward){Fail "Refusing to overwrite $Reward"}
  & $Python -m hoppertrex_mjlab.scripts.calibrate_roll_assist_reward --roll-boundary $RollBoundary --stall $StallArtifact --output $Reward
  if($LASTEXITCODE-ne0){Fail 'Reward calibration failed'}
  Write-Host "[PASS] REWARD=$Reward"; exit 0
}
if([string]::IsNullOrWhiteSpace($RewardCalibration)){$RewardCalibration=Join-Path $Root 'reward_calibration.json'}
Need $RewardCalibration
$Reward=(Resolve-Path $RewardCalibration).Path
$env:HOPPERTREX_ROLL_ASSIST_R0_PATH=(Resolve-Path $RollBoundary).Path
$env:HOPPERTREX_ROLL_ASSIST_REWARD_CALIBRATION_PATH=$Reward
$env:HOPPERTREX_HYBRID_CONTROLLER_PATH=(Resolve-Path ($Artifacts.Keys|Select-Object -Index 0)).Path
$env:HOPPERTREX_HYBRID_CALIBRATION_PATH=(Resolve-Path ($Artifacts.Keys|Select-Object -Index 1)).Path
$env:HOPPERTREX_HYBRID_YAW_CALIBRATION_PATH=(Resolve-Path ($Artifacts.Keys|Select-Object -Index 2)).Path
$env:HOPPERTREX_HYBRID_POSTURE_MAP_PATH=(Resolve-Path ($Artifacts.Keys|Select-Object -Index 3)).Path
$env:HOPPERTREX_HYBRID_STATION_CALIBRATION_PATH=(Resolve-Path ($Artifacts.Keys|Select-Object -Index 4)).Path
if($Phase-eq'Train100'){
  $RunName="rollassist_$($Head.Substring(0,7))_seed1_u100"
  $ExperimentRoot=Join-Path $Repo 'src\hoppertrex_mjlab\logs\rsl_rl\hoppertrex_stair_roll_assist'
  if(Test-Path $ExperimentRoot){
    $Existing=@(Get-ChildItem -LiteralPath $ExperimentRoot -Directory -Filter "*_$RunName")
    if($Existing.Count-ne0){Fail "Refusing duplicate Train100 run name: $RunName"}
  }
  & $Python -m hoppertrex_mjlab.scripts.rsl_rl.train --task $Task --agent.seed 1 --agent.max-iterations 100 --agent.save-interval 25 --agent.num-steps-per-env 24 --agent.resume False --gpu-ids '[0]' --agent.run-name $RunName --env.scene.num-envs 256
  if($LASTEXITCODE-ne0){Fail 'Train100 failed'};exit 0
}
if($Phase-eq'Envelope'){
  Need $Checkpoint;$CheckpointTag=(Sha $Checkpoint).Substring(0,12)
  $CheckpointBase=[IO.Path]::GetFileNameWithoutExtension($Checkpoint)
  $Out=Join-Path $Root ($CheckpointBase+".$CheckpointTag.envelope.json")
  if(Test-Path $Out){Fail "Refusing to overwrite $Out"}
  & $Python -m hoppertrex_mjlab.scripts.evaluate_roll_assist checkpoint-envelope --checkpoint-file $Checkpoint --output $Out
  if($LASTEXITCODE-ne0){Fail 'Envelope failed'};Write-Host "[PASS] ENVELOPE=$Out";exit 0
}
if($Phase-eq'Screen'){
  if(@($CheckpointEnvelope).Count-ne1){Fail 'Screen requires one checkpoint envelope'};Need $CheckpointEnvelope[0]
  $EnvelopeMeta=Get-Content -LiteralPath $CheckpointEnvelope[0] -Raw -Encoding UTF8|ConvertFrom-Json
  $Out=Join-Path $Root "u$($EnvelopeMeta.completed_updates).$($EnvelopeMeta.checkpoint_file_sha256.Substring(0,12)).screen.json"
  if(Test-Path $Out){Fail "Refusing to overwrite $Out"}
  & $Python -m hoppertrex_mjlab.scripts.evaluate_roll_assist live --checkpoint-envelope $CheckpointEnvelope[0] --roll-boundary $RollBoundary --reward-calibration $Reward --device $Device --profile screen --output $Out
  if($LASTEXITCODE-ne0){Fail 'Screen evaluation failed'};Write-Host "[PASS] SCREEN=$Out";exit 0
}
if($Phase-eq'Evaluate'){
  if(@($CheckpointEnvelope).Count-ne1){Fail 'Evaluate requires one checkpoint envelope'};Need $CheckpointEnvelope[0]
  $EnvelopeMeta=Get-Content -LiteralPath $CheckpointEnvelope[0] -Raw -Encoding UTF8|ConvertFrom-Json
  Need $Selection
  $Selected=Get-Content -LiteralPath $Selection -Raw -Encoding UTF8|ConvertFrom-Json
  if($Selected.classification-ne'ROLL_ASSIST_K3_PASSER_SELECTED'-or$Selected.selected.checkpoint_file_sha256-ne$EnvelopeMeta.checkpoint_file_sha256){Fail 'Evaluate requires the selected K=3 checkpoint'}

  $Out=Join-Path $Root "u$($EnvelopeMeta.completed_updates).$($EnvelopeMeta.checkpoint_file_sha256.Substring(0,12)).formal.json"
  if(Test-Path $Out){Fail "Refusing to overwrite $Out"}
  & $Python -m hoppertrex_mjlab.scripts.evaluate_roll_assist live --checkpoint-envelope $CheckpointEnvelope[0] --roll-boundary $RollBoundary --reward-calibration $Reward --device $Device --profile formal --output $Out
  if($LASTEXITCODE-ne0){Fail 'Formal evaluation failed'};Write-Host "[PASS] EVIDENCE=$Out";exit 0
}
if($Phase-eq'SelectK3'){
  if(@($CheckpointEnvelope).Count-ne3-or@($ScreenEvidence).Count-ne3){Fail 'SelectK3 needs three envelopes and screens'}
  $InputEnvelopeHashes=@($CheckpointEnvelope|ForEach-Object{$Meta=Get-Content -LiteralPath $_ -Raw -Encoding UTF8|ConvertFrom-Json;[string]$Meta.checkpoint_file_sha256})
  if(@($InputEnvelopeHashes|Select-Object -Unique).Count-ne3){Fail 'SelectK3 checkpoint envelopes must reference distinct bytes'}

  $screenEnvelopes=@()
  for($i=0;$i-lt3;$i++){
    Need $CheckpointEnvelope[$i];Need $ScreenEvidence[$i]
    $ItemMeta=Get-Content -LiteralPath $CheckpointEnvelope[$i] -Raw -Encoding UTF8|ConvertFrom-Json
    $itemOut=Join-Path $Root "k3_u$($ItemMeta.completed_updates).json";if(Test-Path $itemOut){Fail "Refusing to overwrite $itemOut"}
    & $Python -m hoppertrex_mjlab.scripts.evaluate_roll_assist screen-envelope --checkpoint-envelope $CheckpointEnvelope[$i] --screen-json $ScreenEvidence[$i] --output $itemOut
    if($LASTEXITCODE-ne0){Fail 'Screen envelope failed'};$screenEnvelopes+=$itemOut
  }
  $LatestUpdate=(@($CheckpointEnvelope|ForEach-Object{(Get-Content -LiteralPath $_ -Raw -Encoding UTF8|ConvertFrom-Json).completed_updates})|Measure-Object -Maximum).Maximum
  $Out=Join-Path $Root "select_k3_u${LatestUpdate}.json";if(Test-Path $Out){Fail "Refusing to overwrite $Out"}
  if(@($screenEnvelopes|Select-Object -Unique).Count-ne3){Fail 'K3 screen envelopes must be distinct'}
  $arguments=@('select-k3','--output',$Out);foreach($item in $screenEnvelopes){$arguments+=@('--checkpoint-envelope',$item)}
  & $Python -m hoppertrex_mjlab.scripts.adjudicate_roll_assist @arguments
  if($LASTEXITCODE-ne0){Fail 'K3 selection failed'};Write-Host "[PASS] SELECTION=$Out";exit 0
}
if($Phase-eq'ExtendBlock'){
  Need $Evidence;Need $Checkpoint
  Need $Selection
  $Selected=Get-Content -LiteralPath $Selection -Raw -Encoding UTF8|ConvertFrom-Json
  if($Selected.classification-ne'ROLL_ASSIST_K3_PASSER_SELECTED'-or$Selected.selected.checkpoint_file_sha256-ne(Sha $Checkpoint)){Fail 'ExtendBlock requires the selected K=3 checkpoint'}

  $Formal=Get-Content -LiteralPath $Evidence -Raw -Encoding UTF8|ConvertFrom-Json
  if($Formal.kind-ne'roll_assist_evaluation'-or$Formal.profile-ne'formal'-or$Formal.evidence_eligible-ne$true){Fail 'ExtendBlock requires formal evaluator evidence'}
  if([bool]$Formal.final.passed){Fail 'Hnext already passed; stop training immediately'}

  if($SelectedCompletedUpdates-lt51-or$TargetTotalUpdates-ne($SelectedCompletedUpdates+100)-or$TargetTotalUpdates-gt500){Fail 'Extension must add exactly 100 to a selected K=3 checkpoint up to 500'}
  if($Formal.checkpoint.checkpoint_file_sha256-ne(Sha $Checkpoint)-or[int]$Formal.checkpoint.completed_updates-ne$SelectedCompletedUpdates){Fail 'Selected checkpoint differs from formal evidence'}
  $Authorization=Join-Path $Root "extension_${SelectedCompletedUpdates}_to_${TargetTotalUpdates}.json";if(Test-Path $Authorization){Fail "Refusing to overwrite $Authorization"}
  & $Python -m hoppertrex_mjlab.scripts.adjudicate_roll_assist continuation --evidence $Evidence --selected-checkpoint $Checkpoint --selected-completed-updates $SelectedCompletedUpdates --target-total-updates $TargetTotalUpdates --output $Authorization
  if($LASTEXITCODE-ne0){Fail 'Authorization failed'}
  if([string]::IsNullOrWhiteSpace($ResumeRun)){Fail 'ExtendBlock requires ResumeRun'}
  $env:HOPPERTREX_ROLL_ASSIST_EXTENSION_AUTHORIZATION_PATH=(Resolve-Path $Authorization).Path
  $RunName="rollassist_$($Head.Substring(0,7))_seed1_u${TargetTotalUpdates}"
  $ExperimentRoot=Join-Path $Repo 'src\hoppertrex_mjlab\logs\rsl_rl\hoppertrex_stair_roll_assist'
  if(Test-Path $ExperimentRoot){
    $Existing=@(Get-ChildItem -LiteralPath $ExperimentRoot -Directory -Filter "*_$RunName")
    if($Existing.Count-ne0){Fail "Refusing duplicate extension run name: $RunName"}
  }
  & $Python -m hoppertrex_mjlab.scripts.rsl_rl.train --task $Task --agent.seed 1 --agent.max-iterations $TargetTotalUpdates --agent.save-interval 25 --agent.num-steps-per-env 24 --agent.resume True --gpu-ids '[0]' --agent.load-run $ResumeRun --agent.load-checkpoint (Split-Path $Checkpoint -Leaf) --agent.run-name $RunName --env.scene.num-envs 256
  if($LASTEXITCODE-ne0){Fail 'Extension training failed'};exit 0
}
if($Phase-eq'Package'){
  Need $Selection
  $Selected=Get-Content -LiteralPath $Selection -Raw -Encoding UTF8|ConvertFrom-Json
  & $Python -m hoppertrex_mjlab.scripts.adjudicate_roll_assist validate-k3 --selection $Selection --verify-screen-files
  if($LASTEXITCODE-ne0){Fail 'Package K=3 canonical selection validation failed'}
  $Selected=Get-Content -LiteralPath $Selection -Raw -Encoding UTF8|ConvertFrom-Json

  $Package=Join-Path $Root 'package';if(Test-Path $Package){Fail "Refusing to overwrite $Package"}
  $WorkPackage=Join-Path $Root ('.package.incomplete.'+[Guid]::NewGuid().ToString('N'))
  New-Item -ItemType Directory -Path $WorkPackage|Out-Null
  try{
    Copy-Item -LiteralPath $RollBoundary -Destination (Join-Path $WorkPackage 'roll_boundary.json')
    Copy-Item -LiteralPath $Reward -Destination (Join-Path $WorkPackage 'reward_calibration.json')
    Copy-Item -LiteralPath $Selection -Destination (Join-Path $WorkPackage 'selection.json')
    $TerminationReason=$null
    if($Selected.classification-eq'ROLL_ASSIST_K3_NO_PASSER'){
      if($null-ne$Selected.selected){Fail 'No-passer selection unexpectedly contains a selected checkpoint'}
      $Candidates=@($Selected.candidates)
      if($Candidates.Count-ne3){Fail 'No-passer package requires exactly three screen candidates'}
      $Updates=@($Candidates|ForEach-Object{[int]$_.completed_updates})
      $SortedUpdates=@($Updates|Sort-Object)
      for($i=0;$i-lt$Updates.Count;$i++){
        if($Updates[$i]-ne$SortedUpdates[$i]){Fail 'No-passer candidates are not ordered by completed update'}
      }
      if(@($Candidates|ForEach-Object{[string]$_.checkpoint_file_sha256}|Select-Object -Unique).Count-ne3){Fail 'No-passer candidates do not bind three distinct checkpoints'}
      foreach($Candidate in $Candidates){
        if([bool]$Candidate.passed){Fail 'No-passer package contains a passing screen'}
        $ScreenPath=[string]$Candidate.screen_envelope_file;Need $ScreenPath
        if((Sha $ScreenPath)-ne[string]$Candidate.screen_envelope_sha256){Fail 'No-passer screen envelope SHA drifted'}
        $Screen=Get-Content -LiteralPath $ScreenPath -Raw -Encoding UTF8|ConvertFrom-Json
        if($Screen.schema_version-ne1-or$Screen.kind-ne'roll_assist_k3_screen'-or[bool]$Screen.passed){Fail 'No-passer screen envelope is not a canonical rejection'}
        if([string]$Screen.checkpoint_file_sha256-ne[string]$Candidate.checkpoint_file_sha256-or[int]$Screen.completed_updates-ne[int]$Candidate.completed_updates){Fail 'No-passer screen/selection checkpoint binding drifted'}
        foreach($Check in @('flat_retention_passed','hpass_retained','hnext_safe','wheel_residual_exact_zero')){
          if([bool]$Screen.checks.$Check-ne[bool]$Candidate.checks.$Check){Fail "No-passer screen check drifted: $Check"}
        }
        $ScreenName="u$([int]$Candidate.completed_updates).$(([string]$Candidate.screen_envelope_sha256).Substring(0,12)).screen-envelope.json"
        Copy-Item -LiteralPath $ScreenPath -Destination (Join-Path $WorkPackage $ScreenName)
      }
      $Classification='ROLL_ASSIST_NO_EXPANSION'
      $TerminationReason='ROLL_ASSIST_K3_NO_PASSER'
    }elseif($Selected.classification-eq'ROLL_ASSIST_K3_PASSER_SELECTED'){
      Need $Evidence;Need $Checkpoint
      $Result=Get-Content -LiteralPath $Evidence -Raw -Encoding UTF8|ConvertFrom-Json
      if($Result.kind-ne'roll_assist_evaluation'-or$Result.profile-ne'formal'-or$Result.evidence_eligible-ne$true){Fail 'Package requires formal evidence'}
      if($Result.checkpoint.checkpoint_file_sha256-ne(Sha $Checkpoint)){Fail 'Package checkpoint differs from formal evidence'}
      if($Selected.selected.checkpoint_file_sha256-ne$Result.checkpoint.checkpoint_file_sha256){Fail 'Package requires selected K=3 checkpoint evidence'}
      if(-not[bool]$Result.final.passed-and[bool]$Result.continuation.authorized-and[int]$Result.checkpoint.completed_updates-lt500){Fail 'Passing continuation evidence requires another authorized block before packaging'}
      if([bool]$Result.recovery_claim.eligible){Fail 'Recovery claims are frozen off until paired recovery bootstrap exists'}
      Copy-Item -LiteralPath $Evidence -Destination (Join-Path $WorkPackage 'formal_evidence.json')
      Copy-Item -LiteralPath $Checkpoint -Destination (Join-Path $WorkPackage (Split-Path $Checkpoint -Leaf))
      $Classification=if([bool]$Result.final.passed){'ROLL_ASSIST_BOUNDARY_EXPANDED'}else{'ROLL_ASSIST_NO_EXPANSION'}
      if(-not[bool]$Result.final.passed){$TerminationReason='FORMAL_GATE_REJECTED_CONTINUATION'}
    }else{
      Fail 'Package requires a canonical passer or no-passer K=3 selection'
    }
    $Manifest=[ordered]@{schema_version=1;classification=$Classification;termination_reason=$TerminationReason;recovery_claim_eligible=$false;single_seed=$true;simulation_only=$true;provisional=$true;git_sha=$Head;mjlab_git_sha=$MjLabSha;controller_schedule_hash=$ScheduleHash}
    $Manifest|ConvertTo-Json -Depth 10|Set-Content -LiteralPath (Join-Path $WorkPackage 'manifest.json') -Encoding UTF8
    Get-ChildItem -LiteralPath $WorkPackage -File|Sort-Object Name|ForEach-Object{"$(Sha $_.FullName)  $($_.Name)"}|Set-Content -LiteralPath (Join-Path $WorkPackage 'SHA256SUMS.txt') -Encoding ASCII
    Move-Item -LiteralPath $WorkPackage -Destination $Package
  }finally{
    if(Test-Path -LiteralPath $WorkPackage){Remove-Item -LiteralPath $WorkPackage -Recurse -Force}
  }
  Write-Host "[PASS] PACKAGE=$Package";exit 0
}
Fail "Unhandled phase $Phase"
