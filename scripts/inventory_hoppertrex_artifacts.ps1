[CmdletBinding()]
param(
  [string]$WorkspaceRoot = 'C:\mjlab_workspace',
  [string]$OutputDirectory,
  [switch]$SkipHashes
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$WorkspaceRoot = (Resolve-Path -LiteralPath $WorkspaceRoot).Path
if (-not $OutputDirectory) {
  $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
  $OutputDirectory = Join-Path $WorkspaceRoot "artifact_inventory_$stamp"
}
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$OutputDirectory = (Resolve-Path -LiteralPath $OutputDirectory).Path

function Get-Sha256 {
  param([System.IO.FileInfo]$File)
  if ($SkipHashes) { return $null }
  return (Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-Category {
  param([string]$Path, [string]$Name)
  $text = "$Path $Name".ToLowerInvariant()
  if ($text -match 'hybrid[_-]?v2|controller|calibration|identification|posture|stage0') {
    return 'hybrid_stage0_or_runtime'
  }
  if ($Name -like 'model_*.pt') {
    if ($text -match 'clean.wheel|clean.balance') { return 'legacy_clean_balance' }
    if ($text -match 'robust|push_l3') { return 'legacy_robust_push' }
    if ($text -match 'slow.speed.turn|turn_l4') { return 'legacy_slow_turn' }
    if ($text -match 'bidir|stage2|precision') { return 'legacy_stage2_bidir' }
    return 'legacy_other_checkpoint'
  }
  return 'supporting_artifact'
}

function Get-Retention {
  param([string]$Category)
  switch ($Category) {
    'hybrid_stage0_or_runtime' { return 'must_keep' }
    'legacy_clean_balance' { return 'select_representative' }
    'legacy_robust_push' { return 'select_representative' }
    'legacy_slow_turn' { return 'select_representative' }
    'legacy_stage2_bidir' { return 'select_representative_failure' }
    'legacy_other_checkpoint' { return 'cleanup_candidate_after_review' }
    default { return 'keep_with_selected_run' }
  }
}

$extensions = @('.pt', '.json', '.npz', '.yaml', '.yml', '.csv', '.log', '.txt')
$files = Get-ChildItem -LiteralPath $WorkspaceRoot -Recurse -File -ErrorAction SilentlyContinue |
  Where-Object {
    $_.Extension.ToLowerInvariant() -in $extensions -and
    ($_.Name -like 'model_*.pt' -or $_.FullName -match 'trained_models|hoppertrex|hybrid[_-]?v2')
  }

$records = foreach ($file in $files) {
  $category = Get-Category -Path $file.DirectoryName -Name $file.Name
  [pscustomobject]@{
    category = $category
    retention = Get-Retention -Category $category
    name = $file.Name
    extension = $file.Extension.ToLowerInvariant()
    size_bytes = $file.Length
    modified_utc = $file.LastWriteTimeUtc.ToString('o')
    sha256 = Get-Sha256 -File $file
    path = $file.FullName
  }
}

$records = @($records | Sort-Object category, path)
$summary = @($records | Group-Object category, retention | ForEach-Object {
  [pscustomobject]@{
    category = $_.Group[0].category
    retention = $_.Group[0].retention
    file_count = $_.Count
    size_bytes = ($_.Group | Measure-Object size_bytes -Sum).Sum
  }
})

$manifest = [ordered]@{
  schema_version = 1
  generated_at = (Get-Date).ToUniversalTime().ToString('o')
  workspace_root = $WorkspaceRoot
  hashes_computed = -not $SkipHashes
  summary = $summary
  files = $records
}

$jsonPath = Join-Path $OutputDirectory 'inventory.json'
$csvPath = Join-Path $OutputDirectory 'inventory.csv'
$summaryPath = Join-Path $OutputDirectory 'summary.csv'
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding UTF8
$records | Export-Csv -LiteralPath $csvPath -NoTypeInformation -Encoding UTF8
$summary | Export-Csv -LiteralPath $summaryPath -NoTypeInformation -Encoding UTF8

Write-Host "[OK] Inventory JSON: $jsonPath"
Write-Host "[OK] Inventory CSV:  $csvPath"
Write-Host "[OK] Summary CSV:    $summaryPath"
Write-Host '[SAFE] No source artifact was moved or deleted.'
