[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][ValidateSet('Validate','Baseline','Train','Evaluate','Package')][string]$Phase,
  [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{40}$')][string]$ExpectedGitSha,
  [Parameter(Mandatory=$true)][string]$CampaignRoot,
  [ValidateSet(11,12,13)][int]$Seed,
  [string]$RunDirectory,
  [string]$Python,
  [ValidateSet('cpu','cuda:0')][string]$Device='cuda:0',
  [switch]$LocalCheck,
  [switch]$Viewer,
  [switch]$CpuSmoke
)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
$Repo=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $Repo
if([string]::IsNullOrWhiteSpace($Python)){$Python=Join-Path $Repo '.venv/Scripts/python.exe'}
if(-not(Test-Path -LiteralPath $Python -PathType Leaf)){throw "Missing Python: $Python"}
$env:PYTHONPATH=(Join-Path $Repo 'src')+';'+(Join-Path $Repo 'src/hoppertrex_mjlab')
$env:PYTHONDONTWRITEBYTECODE='1'
$Arguments=@('-B','-m','hoppertrex_mjlab.scripts.recovery_c1_pilot','--phase',$Phase,'--expected-git-sha',$ExpectedGitSha,'--campaign-root',$CampaignRoot,'--device',$Device)
if($PSBoundParameters.ContainsKey('Seed')){$Arguments+=@('--seed',[string]$Seed)}
if($RunDirectory){$Arguments+=@('--run-directory',$RunDirectory)}
if($LocalCheck){$Arguments+='--local-check'}
if($CpuSmoke){$Arguments+='--cpu-smoke'}
if($Viewer){$Arguments+='--viewer'}
# Native warnings remain in logs; only exit codes decide process success.
$Saved=$ErrorActionPreference
try{
  $ErrorActionPreference='Continue'
  $global:LASTEXITCODE=$null
  & $Python @Arguments
  $Code=$LASTEXITCODE
}finally{$ErrorActionPreference=$Saved}
if($null -eq $Code){throw "Native Python did not report an exit code. Check Windows PATH/PATHEXT."}
if($Code -ne 0){throw "C1 pilot failed ($Code). Preserve the incomplete folder; do not retune or rerun."}
