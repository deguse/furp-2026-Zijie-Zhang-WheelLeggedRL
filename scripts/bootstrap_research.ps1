param([switch]$SkipSync)
$ErrorActionPreference = 'Stop'
$Repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$Mjlab = Join-Path (Split-Path $Repo -Parent) 'mjlab-main'
$Expected = '43e0f3ea9c92ddbb4de9f3bb1ac772d604e3ebf6'
function Invoke-Checked { param([string]$Program, [string[]]$Arguments)
    # Windows PowerShell 5.1 treats native stderr as ErrorRecords when redirected.
    # Progress text is not failure; the native process exit code is authoritative.
    $PreviousErrorAction = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Program @Arguments 2>&1 | ForEach-Object { Write-Output "$_" }
        $Code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $PreviousErrorAction }
    if ($Code -ne 0) { throw "$Program failed with exit code $Code" }
}
if (-not (Test-Path -LiteralPath $Mjlab)) {
    Invoke-Checked 'git' @('clone', 'https://github.com/deguse/mjlab.git', $Mjlab)
    Invoke-Checked 'git' @('-C', $Mjlab, 'checkout', '--detach', $Expected)
}
$Actual = & git -C $Mjlab rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $Actual.Trim() -ne $Expected) {
    throw "Sibling mjlab-main must be at $Expected. Existing checkout was NOT changed."
}
$Dirty = & git -C $Mjlab status --porcelain
if ($LASTEXITCODE -ne 0 -or $Dirty) { throw 'Sibling mjlab-main must be clean; existing changes were NOT touched.' }
$Manifest = Get-Content -LiteralPath (Join-Path $Repo 'vendor/wheels/manifest.json') -Raw | ConvertFrom-Json
$Wheel = Join-Path (Join-Path $Repo 'vendor/wheels') $Manifest.artifact
if ((Get-FileHash -LiteralPath $Wheel -Algorithm SHA256).Hash.ToLower() -ne $Manifest.file_sha256) { throw 'MuJoCo wheel checksum mismatch' }
Push-Location $Repo
try {
    if (-not $SkipSync) { Invoke-Checked 'uv' @('sync', '--frozen', '--python', '3.11') }
    $env:PYTHONPATH = "$(Join-Path $Repo 'src');$(Join-Path $Repo 'src/hoppertrex_mjlab')"
    Invoke-Checked 'uv' @('run', '--no-sync', 'python', '-c', 'import sys, mujoco, torch; import hoppertrex_mjlab.hybrid.control as c; print(sys.version); print(mujoco.__version__); print(torch.__version__); print(c.__file__)')
} finally { Pop-Location }
Write-Host 'Environment ready. No training or simulation has been launched.'
