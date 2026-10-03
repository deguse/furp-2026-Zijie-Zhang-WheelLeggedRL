param([string]$Python = (Get-Command python -ErrorAction Stop).Source)
$ErrorActionPreference = 'Stop'
$Repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$Tokens = $null; $Errors = $null
$Ast = [System.Management.Automation.Language.Parser]::ParseFile((Join-Path $Repo 'scripts/bootstrap_research.ps1'), [ref]$Tokens, [ref]$Errors)
if ($Errors.Count) { throw 'Bootstrap parse error' }
$Function = $Ast.Find({param($Node) $Node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $Node.Name -eq 'Invoke-Checked'}, $true)
. ([scriptblock]::Create($Function.Extent.Text))
$Output = Invoke-Checked $Python @('-c', 'import sys; sys.stderr.write(str(12345)); sys.exit(0)')
if ("$Output" -notmatch '12345') { throw 'Native stderr was lost' }
$FailedAsExpected = $false
try { Invoke-Checked $Python @('-c', 'import sys; sys.exit(3)') } catch { $FailedAsExpected = $true }
if (-not $FailedAsExpected) { throw 'Nonzero native exit was ignored' }
Write-Output 'PASS: redirected native stderr does not abort success; nonzero exit is rejected.'
