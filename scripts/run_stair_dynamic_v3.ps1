[CmdletBinding()]
param(
  [ValidateSet('Validate','QualifyTrigger','Search','Migrate','ZeroEval','Train100','SelectK3','Extend500','Evaluate','Package')]
  [string]$Phase='Validate',
  [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-fA-F]{40}$')][string]$ExpectedGitSha,
  [Parameter(Mandatory=$true)][string]$CampaignRoot,
  [ValidateSet(100,500)][int]$Budget=100,
  [string]$Stage5Checkpoint,
  [string]$Stage5Gate,
  [string]$Python,
  [string]$Device='cuda:0'
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$Task='HopperTrex-Hybrid-v3-StairDynamic'
$Branch='codex/p2-classical-upper-bound'
$MjLabSha='43e0f3ea9c92ddbb4de9f3bb1ac772d604e3ebf6'
$Experiment='hoppertrex_stair_dynamic_v3'
$Train='hoppertrex_mjlab.scripts.rsl_rl.train'
$Qualify='hoppertrex_mjlab.scripts.rsl_rl.qualify_stair_dynamic_trigger'
$Preflight='hoppertrex_mjlab.scripts.rsl_rl.preflight_stair_dynamic'
$Search='hoppertrex_mjlab.scripts.rsl_rl.search_stair_dynamic'
$Evaluator='hoppertrex_mjlab.scripts.rsl_rl.evaluate_stair_dynamic'
$Live='hoppertrex_mjlab.scripts.rsl_rl.stair_dynamic_live_adapter'
$SearchAdapter='hoppertrex_mjlab.scripts.rsl_rl.stair_dynamic_search_live_adapter:collect'
$script:Saved=@{};$script:PhaseFinal=@{}
$Artifacts=[ordered]@{
 HOPPERTREX_HYBRID_CONTROLLER_PATH=@('docs\experiments\artifacts\c1_schedule_candidate24_1f54968_seed1\c1_schedule.json','9b21125e7cc48be3ea61e12a67171a855892ad3ced1f54b3176ed979e76224ec')
 HOPPERTREX_HYBRID_CALIBRATION_PATH=@('docs\experiments\artifacts\hybrid_runtime_seed1\velocity_calibration_seed1.json','ef002d0d622725509b47c8ff40d8af658fd42f705bdeac67ac35bae4458f889d')
 HOPPERTREX_HYBRID_YAW_CALIBRATION_PATH=@('docs\experiments\artifacts\yaw_gpu_3f8a9330b88fa6129d05ce42ac3a8cc835295a6f_seed1\yaw_calibration.json','123122e75955468dfc475d86ac3f9160b428720fd8e1b90ab614bc1bc0749765')
 HOPPERTREX_HYBRID_POSTURE_MAP_PATH=@('docs\experiments\artifacts\c1_posture_requalification_seed1\posture_map_seed1_registered_p032.json','b8e627f85b53d21dd8d9c26edbe2943151d9bcf9e5864ff998ede5f909118e23')
 HOPPERTREX_HYBRID_STATION_CALIBRATION_PATH=@('docs\experiments\artifacts\c1_posture_requalification_seed1\station_calibration_seed1.json','f22a9b66f734004ff14b6586a22a991d527f360806bbbdefe096e9f0474db72a')
}
function Fail([string]$Kind,[string]$Message){$e=New-Object InvalidOperationException("STAIR_DYNAMIC_${Kind}: $Message");$e.Data['ExitCode']=if($Kind-eq'PROVENANCE'){20}elseif($Kind-eq'PROTOCOL'){30}else{40};throw $e}
function Sha([string]$Path){(Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()}
function Need([string]$Path,[string]$Name){if(-not(Test-Path -LiteralPath $Path -PathType Leaf)){Fail 'PROVENANCE' "Missing $Name`: $Path"}}
function CanonicalSelfHash([string]$Path){$s=[IO.File]::ReadAllText($Path).Replace("`r`n","`n").Replace("`r","`n");$h=[Security.Cryptography.SHA256]::Create();try{([BitConverter]::ToString($h.ComputeHash([Text.UTF8Encoding]::new($false).GetBytes($s)))).Replace('-','').ToLowerInvariant()}finally{$h.Dispose()}}
function AtomicText([string]$Path,[string]$Value){if(Test-Path -LiteralPath $Path){Fail 'OPERATIONAL' "Refusing to overwrite $Path"};$tmp="$Path.incomplete.$([Guid]::NewGuid().ToString('N'))";try{[IO.File]::WriteAllText($tmp,$Value,[Text.UTF8Encoding]::new($false));[IO.File]::Move($tmp,$Path)}finally{if(Test-Path -LiteralPath $tmp){Remove-Item -LiteralPath $tmp -Force}}}
function AtomicJson([string]$Path,[object]$Value){AtomicText $Path (($Value|ConvertTo-Json -Depth 100).Replace("`r`n","`n")+"`n")}
function PhaseDir([string]$Name){$final=Join-Path $script:Root $Name;if(Test-Path -LiteralPath $final){Fail 'OPERATIONAL' "Phase exists: $final"};$work=Join-Path $script:Root (".$Name.incomplete."+[Guid]::NewGuid().ToString('N'));New-Item -ItemType Directory -Path $work|Out-Null;$script:PhaseFinal[$work]=$final;$work}
function Run([ValidateSet('PROVENANCE','PROTOCOL','OPERATIONAL')][string]$FailureKind,[string]$Module,[string[]]$ModuleArgs,[string]$Log){$old=$ErrorActionPreference;try{$ErrorActionPreference='Continue';& $script:Py '-m' $Module @ModuleArgs 2>&1|Tee-Object -FilePath $Log -Append;$code=$LASTEXITCODE}finally{$ErrorActionPreference=$old};if($code-ne0){Fail $FailureKind "$Module failed (exit $code)"}}
function Help([string]$Module,[string[]]$Prefix,[string[]]$Flags){$text=@(& $script:Py '-m' $Module @Prefix '--help' 2>&1)-join"`n";if($LASTEXITCODE-ne0){Fail 'PROTOCOL' "$Module --help failed"};foreach($f in $Flags){if($text-notmatch[regex]::Escape($f)){Fail 'PROTOCOL' "$Module --help lacks $f"}}}
function Manifest(){$p=Join-Path $script:Root 'campaign_manifest.json';Need $p 'campaign manifest';Get-Content -LiteralPath $p -Raw -Encoding UTF8|ConvertFrom-Json}
function Status([string]$Dir,[string]$Class,[hashtable]$Extra=@{}){$final=[string]$script:PhaseFinal[$Dir];$x=[ordered]@{task=$Task;phase=$Phase;budget=$Budget;classification=$Class;git_sha=$script:Git;created_at=[DateTime]::UtcNow.ToString('o')};foreach($k in $Extra.Keys){$v=$Extra[$k];if($v-is[string]-and$v.StartsWith($Dir,[StringComparison]::OrdinalIgnoreCase)){$v=$final+$v.Substring($Dir.Length)};$x[$k]=$v};AtomicJson (Join-Path $Dir 'status.json') $x;Move-Item -LiteralPath $Dir -Destination $final}
function Envelope([string]$Checkpoint,[string]$Output,[string]$Log){Run -FailureKind 'PROVENANCE' -Module $Evaluator -ModuleArgs @('checkpoint-envelope','--checkpoint-file',$Checkpoint,'--output',$Output) -Log $Log}
function Formal([string]$Suite,[string]$Ablation,[string]$Envelope,[string]$Expectation,[string]$Dir,[string]$Stem){$req=Join-Path $Dir "$Stem.request.json";$raw=Join-Path $Dir "$Stem.collection.json";$out=Join-Path $Dir "$Stem.result.json";$log=Join-Path $Dir "$Stem.log";Run -FailureKind 'PROTOCOL' -Module $Evaluator -ModuleArgs @('make-request','--suite',$Suite,'--checkpoint-envelope',$Envelope,'--expectation',$Expectation,'--ablation',$Ablation,'--device',$Device,'--output',$req) -Log $log;Run -FailureKind 'OPERATIONAL' -Module $Live -ModuleArgs @('collect','--request',$req,'--output',$raw) -Log $log;Run -FailureKind 'PROTOCOL' -Module $Evaluator -ModuleArgs @('finalize','--request',$req,'--collection',$raw,'--output',$out) -Log $log;Get-Content -LiteralPath $out -Raw -Encoding UTF8|ConvertFrom-Json}
function ValidatePhase{
 if([string]::IsNullOrWhiteSpace($Stage5Checkpoint)-or[string]::IsNullOrWhiteSpace($Stage5Gate)){Fail 'PROTOCOL' 'Validate requires Stage5 checkpoint and gate'}
 Need $Stage5Checkpoint 'Stage5 checkpoint';Need $Stage5Gate 'Stage5 formal gate';$ck=(Resolve-Path -LiteralPath $Stage5Checkpoint).Path;$gate=(Resolve-Path -LiteralPath $Stage5Gate).Path
 if(@(& git -C $script:Repo status --porcelain).Count-ne0){Fail 'PROVENANCE' 'Repository is not clean'}
 $remote=(& git -C $script:Repo rev-parse "origin/$Branch").Trim().ToLowerInvariant();if($remote-ne$script:Git){Fail 'PROVENANCE' 'Remote branch SHA differs'}
 $mj=(Resolve-Path (Join-Path $script:Repo '..\mjlab-main')).Path;if((& git -C $mj rev-parse HEAD).Trim().ToLowerInvariant()-ne$MjLabSha-or @(& git -C $mj status --porcelain).Count-ne0){Fail 'PROVENANCE' 'MjLab is not clean/pinned'}
 $a=[ordered]@{};foreach($n in $Artifacts.Keys){$p=(Resolve-Path (Join-Path $script:Repo $Artifacts[$n][0])).Path;if((Sha $p)-ne$Artifacts[$n][1]){Fail 'PROVENANCE' "Artifact drift: $n"};$a[$n]=@{path=$p;sha256=$Artifacts[$n][1]}}
 if($null-eq(Get-Command nvidia-smi -ErrorAction SilentlyContinue)){Fail 'OPERATIONAL' 'nvidia-smi unavailable'};& nvidia-smi --query-gpu=index --format=csv,noheader|Out-Null;if($LASTEXITCODE-ne0){Fail 'OPERATIONAL' 'GPU query failed'}
 Help $Qualify @('collect') @('--device','--output');Help $Preflight @('search-bindings') @('--stage5-checkpoint','--trigger-qualification');Help $Search @() @('--adapter','--device');Help $Evaluator @('make-request') @('--checkpoint-envelope','--expectation');Help $Live @('collect-k3') @('--budget-updates');Help $Train @($Task) @('--agent.max-iterations','--agent.load-run')
 AtomicJson (Join-Path $script:Root 'campaign_manifest.json') ([ordered]@{campaign_id='stair_dynamic_'+[DateTime]::UtcNow.ToString('yyyyMMdd_HHmmss');task=$Task;git_sha=$script:Git;mjlab_sha=$MjLabSha;stage5_checkpoint=$ck;stage5_checkpoint_sha256=Sha $ck;stage5_gate=$gate;stage5_gate_sha256=Sha $gate;artifacts=$a;single_seed_status='provisional'})
 $d=PhaseDir '00_validate';Status $d 'STAIR_DYNAMIC_STATIC_VALIDATE_PASS' @{simulation_started=$false};Write-Host '[PASS] static Validate; no rollout'
}function QualifyPhase{$d=PhaseDir '01_trigger_qualification';$o=Join-Path $d 'qualification.json';Run -FailureKind 'OPERATIONAL' -Module $Qualify -ModuleArgs @('collect','--device',$Device,'--output',$o) -Log (Join-Path $d 'collect.log');Run -FailureKind 'PROTOCOL' -Module $Qualify -ModuleArgs @('verify','--input',$o) -Log (Join-Path $d 'verify.log');Status $d 'DYNAMIC_STAIR_TRIGGER_QUALIFIED';Write-Host '[PASS] trigger qualified'}
function SearchPhase{
 $m=Manifest;$q=Join-Path $script:Root '01_trigger_qualification\qualification.json';Need $q 'trigger qualification';$d=PhaseDir '02_search';$b=Join-Path $d 'bindings.json';$r=Join-Path $d 'report.json';$mvr=Join-Path $d 'maneuver.json';$log=Join-Path $d 'search.log'
 $policyEnv='HOPPERTREX_DYNAMIC_STAIR_STAGE5_CHECKPOINT_PATH';$script:Saved[$policyEnv]=[Environment]::GetEnvironmentVariable($policyEnv,'Process');Set-Item "Env:$policyEnv" ([string]$m.stage5_checkpoint)
 $triggerEnv='HOPPERTREX_DYNAMIC_STAIR_TRIGGER_QUALIFICATION_PATH';$script:Saved[$triggerEnv]=[Environment]::GetEnvironmentVariable($triggerEnv,'Process');Set-Item "Env:$triggerEnv" $q
 Run -FailureKind 'PROVENANCE' -Module $Preflight -ModuleArgs @('search-bindings','--stage5-checkpoint',[string]$m.stage5_checkpoint,'--stage5-gate',[string]$m.stage5_gate,'--trigger-qualification',$q,'--output',$b) -Log $log
 Run -FailureKind 'OPERATIONAL' -Module $Search -ModuleArgs @('--adapter',$SearchAdapter,'--bindings-json',$b,'--device',$Device,'--report',$r,'--maneuver-output',$mvr) -Log $log
 $x=Get-Content $r -Raw -Encoding UTF8|ConvertFrom-Json;if($x.classification-ne'DYNAMIC_STAIR_MANEUVER_QUALIFIED'){Status $d 'STOP_DYNAMIC_STAIR_UNQUALIFIED';Write-Host '[STOP] no safe feedforward; PPO blocked';return};Need $mvr 'maneuver';Status $d 'DYNAMIC_STAIR_MANEUVER_QUALIFIED' @{maneuver=$mvr};Write-Host '[PASS] maneuver qualified'
}
function MigratePhase{
 $m=Manifest;$mvr=Join-Path $script:Root '02_search\maneuver.json';Need $mvr 'maneuver';$d=PhaseDir '03_migration';$boot=Join-Path $script:ExpRoot 'bootstrap_stage5_migration';if(Test-Path $boot){Fail 'OPERATIONAL' 'bootstrap exists'};New-Item -ItemType Directory $boot|Out-Null;$o=Join-Path $boot 'model_0.pt'
 Run -FailureKind 'PROVENANCE' -Module 'hoppertrex_mjlab.scripts.rsl_rl.migrate_stage5_to_stair_dynamic' -ModuleArgs @('--source-checkpoint',[string]$m.stage5_checkpoint,'--source-gate-json',[string]$m.stage5_gate,'--output-checkpoint',$o,'--reset-collapsed-active-std') -Log (Join-Path $d 'migration.log')
 Status $d 'STAIR_DYNAMIC_ZERO_UPDATE_MIGRATION_READY' @{checkpoint=$o;checkpoint_sha256=Sha $o};Write-Host '[PASS] zero-update migration ready'
}
function ZeroEvalPhase{
 $ck=Join-Path $script:ExpRoot 'bootstrap_stage5_migration\model_0.pt';Need $ck 'migration';$d=PhaseDir '04_zero_eval';$exp=Join-Path $d 'expectation.json';$env=Join-Path $d 'envelope.json';$log=Join-Path $d 'zero.log'
 Run -FailureKind 'PROTOCOL' -Module $Preflight -ModuleArgs @('runtime-expectation','--completed-updates','0','--output',$exp) -Log $log;Run -FailureKind 'PROVENANCE' -Module $Evaluator -ModuleArgs @('migration-checkpoint-envelope','--checkpoint-file',$ck,'--expectation',$exp,'--output',$env) -Log $log
 $s=Formal 'single-riser' 'full' $env $exp $d 'single_full';if(-not$s.result_passed){Status $d 'STAIR_DYNAMIC_TRAIN100_REQUIRED' @{reason='single';checkpoint_envelope=$env};Write-Host '[CONTINUE] Train100 required';return}
 $c=Formal 'continuous-stairs' 'full' $env $exp $d 'continuous_full';if(-not$c.result_passed){Status $d 'STAIR_DYNAMIC_TRAIN100_REQUIRED' @{reason='continuous';checkpoint_envelope=$env};Write-Host '[CONTINUE] Train100 required';return}
 $r=Formal 'retention-gates' 'full' $env $exp $d 'retention_full';if(-not$r.result_passed){Status $d 'STAIR_DYNAMIC_TRAIN100_REQUIRED' @{reason='retention';checkpoint_envelope=$env};Write-Host '[CONTINUE] Train100 required';return}
 Status $d 'STAIR_DYNAMIC_TARGET_MET_ZERO_UPDATE' @{checkpoint_envelope=$env};Write-Host '[PASS] target met at zero update; do not train'
}
function TrainTo([int]$Target,[string]$SourceRun,[string]$SourceModel,[string]$DirName,[string]$Suffix,[string]$Auth=''){
 $m=Manifest;$d=PhaseDir $DirName;$runName="$($m.campaign_id)_$Suffix";if($Auth){$n='HOPPERTREX_DYNAMIC_STAIR_EXTENSION_AUTHORIZATION_PATH';if(-not$script:Saved.ContainsKey($n)){$script:Saved[$n]=[Environment]::GetEnvironmentVariable($n,'Process')};Set-Item "Env:$n" $Auth}
 Run -FailureKind 'OPERATIONAL' -Module $Train -ModuleArgs @($Task,'--log-root',$script:LogRoot,'--env.seed','1','--env.scene.num-envs','256','--agent.seed','1','--agent.max-iterations',[string]$Target,'--agent.save-interval','25','--agent.num-steps-per-env','24','--agent.run-name',$runName,'--agent.resume','--agent.load-run',$SourceRun,'--agent.load-checkpoint',$SourceModel) -Log (Join-Path $d 'training.log')
 $runs=@(Get-ChildItem $script:ExpRoot -Directory|Where-Object{$_.Name-like"*_$runName"});if($runs.Count-ne1){Fail 'OPERATIONAL' 'training run attribution failed'};AtomicJson (Join-Path $d 'run.json') @{target=$Target;run_directory=$runs[0].FullName;run_name=$runName};Status $d "STAIR_DYNAMIC_TRAIN_${Target}_COMPLETE" @{run_directory=$runs[0].FullName};Write-Host "[PASS] trained to total $Target"
}
function Train100Phase{$p=Join-Path $script:Root '04_zero_eval\status.json';Need $p 'zero status';$s=Get-Content $p -Raw|ConvertFrom-Json;if($s.classification-eq'STAIR_DYNAMIC_TARGET_MET_ZERO_UPDATE'){Write-Host '[STOP] zero-update already passed';return};TrainTo 100 'bootstrap_stage5_migration' 'model_0.pt' '05_train100' 'seed1_probe100'}function K3Phase{
 $name=if($Budget-eq100){'06_k3_100'}else{'08_k3_500'};$meta=if($Budget-eq100){'05_train100\run.json'}else{'07_extend500\run.json'};$rp=Join-Path $script:Root $meta;Need $rp 'run metadata';$run=Get-Content $rp -Raw|ConvertFrom-Json;$d=PhaseDir $name;$indices=if($Budget-eq100){@(50,75,99)}else{@(450,475,499)};$cand=@()
 foreach($i in $indices){$ck=Join-Path ([string]$run.run_directory) "model_$i.pt";Need $ck "model_$i";$e=Join-Path $d "model_$i.envelope.json";$c=Join-Path $d "model_$i.candidate.json";$log=Join-Path $d "model_$i.log";Envelope $ck $e $log;Run -FailureKind 'OPERATIONAL' -Module $Live -ModuleArgs @('collect-k3','--checkpoint-envelope',$e,'--budget-updates',[string]$Budget,'--device',$Device,'--output',$c) -Log $log;$cand+=$c}
 $sel=Join-Path $d 'selection.json';Run -FailureKind 'PROTOCOL' -Module $Evaluator -ModuleArgs @('select-k3','--candidate',$cand[0],$cand[1],$cand[2],'--output',$sel) -Log (Join-Path $d 'selection.log');$x=Get-Content $sel -Raw|ConvertFrom-Json;if($x.status-ne'selected'){Status $d 'STOP_DYNAMIC_STAIR_UNQUALIFIED' @{reason='k3'};Write-Host '[STOP] K3 no passer';return}
 $env=Join-Path $d 'selected.envelope.json';Envelope ([string]$x.selected_checkpoint.checkpoint_file) $env (Join-Path $d 'formal.log');$exp=Join-Path $d 'expectation.json';Run -FailureKind 'PROTOCOL' -Module $Preflight -ModuleArgs @('runtime-expectation','--output',$exp) -Log (Join-Path $d 'formal.log')
 $single=Formal 'single-riser' 'full' $env $exp $d 'single_full';if(-not$single.result_passed){Status $d 'STOP_DYNAMIC_STAIR_UNQUALIFIED' @{reason='single';selected_envelope=$env};Write-Host '[STOP] formal 44/48 failed';return}
 $ret=Formal 'retention-gates' 'full' $env $exp $d 'retention_full';if(-not$ret.result_passed){Status $d 'STOP_DYNAMIC_STAIR_UNQUALIFIED' @{reason='retention';selected_envelope=$env};Write-Host '[STOP] retention failed';return}
 $cont=Formal 'continuous-stairs' 'full' $env $exp $d 'continuous_full'
 if($Budget-eq100-and-not$cont.result_passed){$auth=Join-Path $d 'extension_authorization.json';Run -FailureKind 'PROTOCOL' -Module $Evaluator -ModuleArgs @('authorize-extension','--selection',$sel,'--retention-result',(Join-Path $d 'retention_full.result.json'),'--single-riser-result',(Join-Path $d 'single_full.result.json'),'--output',$auth) -Log (Join-Path $d 'auth.log');Status $d 'STAIR_DYNAMIC_EXTENSION_500_AUTHORIZED' @{authorization=$auth;selected_envelope=$env};Write-Host '[CONTINUE] extension authorized';return}
 $class=if($cont.result_passed){'STAIR_DYNAMIC_TARGET_MET_3X1CM'}else{'STOP_DYNAMIC_STAIR_UNQUALIFIED'};Status $d $class @{selected_envelope=$env};Write-Host "[$class]"
}
function ExtendPhase{
 $p=Join-Path $script:Root '06_k3_100\status.json';Need $p 'K3-100 status';$s=Get-Content $p -Raw|ConvertFrom-Json;if($s.classification-eq'STAIR_DYNAMIC_TARGET_MET_3X1CM'){Write-Host '[STOP] 100 already passed';return};if($s.classification-ne'STAIR_DYNAMIC_EXTENSION_500_AUTHORIZED'){Fail 'PROTOCOL' 'extension not authorized'};$a=Get-Content ([string]$s.authorization) -Raw|ConvertFrom-Json;$ck=[string]$a.selected_checkpoint_file;TrainTo 500 (Split-Path -Leaf (Split-Path -Parent $ck)) (Split-Path -Leaf $ck) '07_extend500' 'seed1_total500' ([string]$s.authorization)
}
function EvaluatePhase{
 $source=$null;foreach($n in @('08_k3_500','06_k3_100','04_zero_eval')){$sp=Join-Path $script:Root "$n\status.json";if(Test-Path $sp){$st=Get-Content $sp -Raw|ConvertFrom-Json;if($st.PSObject.Properties.Name-contains'selected_envelope'-or ($n-eq'04_zero_eval'-and $st.classification-eq'STAIR_DYNAMIC_TARGET_MET_ZERO_UPDATE')){$source=@($n,$st);break}}};if($null-eq$source){Fail 'PROTOCOL' 'no final selected source'};if($source[0]-eq'06_k3_100'-and$source[1].classification-eq'STAIR_DYNAMIC_EXTENSION_500_AUTHORIZED'){Fail 'PROTOCOL' 'run authorized Extend500 before final Evaluate'};$src=Join-Path $script:Root $source[0];$st=$source[1];$env=if($st.PSObject.Properties.Name-contains'selected_envelope'){[string]$st.selected_envelope}else{[string]$st.checkpoint_envelope};$exp=Join-Path $src 'expectation.json';$full=Join-Path $src 'continuous_full.result.json';Need $env 'envelope';Need $exp 'expectation';Need $full 'full result';$d=PhaseDir '09_evaluate';$results=@()
 foreach($a in @('roll-only','feedforward-only','policy-only','full','leg-PPO-off','wheel-PPO-off')){if($a-eq'full'){$results+=$full}else{$stem='continuous_'+$a.Replace('-','_');$null=Formal 'continuous-stairs' $a $env $exp $d $stem;$results+=Join-Path $d "$stem.result.json"}}
 Run -FailureKind 'PROTOCOL' -Module $Evaluator -ModuleArgs @('bundle-ablations','--result',$results[0],$results[1],$results[2],$results[3],$results[4],$results[5],'--output',(Join-Path $d 'ablation_bundle.json')) -Log (Join-Path $d 'bundle.log');Status $d ([string]$st.classification) @{source_phase=$source[0];single_seed_status='provisional'};Write-Host '[PASS] ablations archived'
}
function PackagePhase{
 $d=PhaseDir '10_package';$final=[string]$script:PhaseFinal[$d];$parent=Split-Path -Parent $script:Root;$leaf=(Split-Path -Leaf $script:Root)+'.zip';$zip=Join-Path $parent $leaf;$zs="$zip.sha256";if((Test-Path $zip)-or(Test-Path $zs)){Fail 'OPERATIONAL' 'package output exists'}
 AtomicJson (Join-Path $d 'status.json') ([ordered]@{task=$Task;phase=$Phase;budget=$Budget;classification='STAIR_DYNAMIC_PACKAGE_COMPLETE';git_sha=$script:Git;created_at=[DateTime]::UtcNow.ToString('o');archive=$leaf;archive_sha256_sidecar="$leaf.sha256";checksum_manifest='10_package/SHA256SUMS.txt'})
 $sum=Join-Path $d 'SHA256SUMS.txt';$rootPrefix=$script:Root.TrimEnd('\')+'\';$workPrefix=$d.TrimEnd('\')+'\';$lines=@();foreach($f in Get-ChildItem $script:Root -File -Recurse|Where-Object{$_.FullName-ne$sum-and $_.Extension-ne'.zip'}|Sort-Object FullName){$rel=if($f.FullName.StartsWith($workPrefix,[StringComparison]::OrdinalIgnoreCase)){"10_package/"+$f.FullName.Substring($workPrefix.Length).Replace('\','/')}else{$f.FullName.Substring($rootPrefix.Length).Replace('\','/')};$lines+="$(Sha $f.FullName)  $rel"};AtomicText $sum (($lines-join"`n")+"`n");Move-Item -LiteralPath $d -Destination $final
 $zt=Join-Path $parent (".$leaf.incomplete.$([Guid]::NewGuid().ToString('N')).zip");$zst="$zs.incomplete.$([Guid]::NewGuid().ToString('N'))";try{Compress-Archive -Path (Join-Path $script:Root '*') -DestinationPath $zt -CompressionLevel Optimal;AtomicText $zst "$(Sha $zt)  $leaf`n";Move-Item -LiteralPath $zt -Destination $zip;Move-Item -LiteralPath $zst -Destination $zs}finally{if(Test-Path $zt){Remove-Item -LiteralPath $zt -Force};if(Test-Path $zst){Remove-Item -LiteralPath $zst -Force}};Write-Host "[PASS] $zip"
}
try{
 $script:Repo=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path;$script:Root=[IO.Path]::GetFullPath($CampaignRoot);if(-not(Test-Path $script:Root)){New-Item -ItemType Directory $script:Root|Out-Null};$script:Git=(& git -C $script:Repo rev-parse HEAD).Trim().ToLowerInvariant();if($script:Git-ne$ExpectedGitSha.ToLowerInvariant()){Fail 'PROVENANCE' 'HEAD differs from ExpectedGitSha'}
 $side="$PSCommandPath.sha256";Need $side 'self-hash sidecar';if((CanonicalSelfHash $PSCommandPath)-ne([IO.File]::ReadAllText($side).Trim().ToLowerInvariant())){Fail 'PROVENANCE' 'wrapper self-hash failed'}
 $script:Py=if($Python){(Resolve-Path $Python).Path}else{Join-Path $script:Repo '.venv\Scripts\python.exe'};Need $script:Py 'Python';$script:LogRoot=Join-Path $script:Root 'training_logs';$script:ExpRoot=Join-Path $script:LogRoot $Experiment;if(-not(Test-Path $script:ExpRoot)){New-Item -ItemType Directory $script:ExpRoot -Force|Out-Null}
 $vars=@{PYTHONPATH="$(Join-Path $script:Repo 'src');$(Join-Path $script:Repo 'src\hoppertrex_mjlab')"};foreach($n in $Artifacts.Keys){$vars[$n]=Join-Path $script:Repo $Artifacts[$n][0]};$mvr=Join-Path $script:Root '02_search\maneuver.json';$mn='HOPPERTREX_DYNAMIC_STAIR_MANEUVER_PATH';$script:Saved[$mn]=[Environment]::GetEnvironmentVariable($mn,'Process');if(Test-Path $mvr){Set-Item "Env:$mn" $mvr}else{Remove-Item "Env:$mn" -ErrorAction SilentlyContinue};foreach($n in $vars.Keys){$script:Saved[$n]=[Environment]::GetEnvironmentVariable($n,'Process');Set-Item "Env:$n" $vars[$n]}
 switch($Phase){'Validate'{ValidatePhase};'QualifyTrigger'{QualifyPhase};'Search'{SearchPhase};'Migrate'{MigratePhase};'ZeroEval'{ZeroEvalPhase};'Train100'{Train100Phase};'SelectK3'{K3Phase};'Extend500'{ExtendPhase};'Evaluate'{EvaluatePhase};'Package'{PackagePhase}}
 exit 0
}catch{$code=40;if($_.Exception.Data.Contains('ExitCode')){$code=[int]$_.Exception.Data['ExitCode']};[Console]::Error.WriteLine($_.Exception.Message);exit $code}finally{foreach($n in $script:Saved.Keys){if($null-eq$script:Saved[$n]){Remove-Item "Env:$n" -ErrorAction SilentlyContinue}else{Set-Item "Env:$n" $script:Saved[$n]}}}
