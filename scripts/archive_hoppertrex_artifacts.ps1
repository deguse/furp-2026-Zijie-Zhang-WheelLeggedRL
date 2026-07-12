[CmdletBinding()]
param(
  [Parameter(Mandatory)]
  [string]$InventoryPath,
  [string]$ArchiveRoot = 'C:\mjlab_workspace\hoppertrex_archive',
  [switch]$AllowMissing
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$InventoryPath = (Resolve-Path -LiteralPath $InventoryPath).Path
$inventory = Get-Content -LiteralPath $InventoryPath -Raw | ConvertFrom-Json
$workspaceRoot = [string]$inventory.workspace_root
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$archive = Join-Path $ArchiveRoot $stamp
New-Item -ItemType Directory -Force -Path $archive | Out-Null

function Get-RelativeWorkspacePath {
  param([string]$Path)
  $root = $workspaceRoot.TrimEnd('\') + '\'
  if (-not $Path.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Path is outside inventory workspace: $Path"
  }
  return $Path.Substring($root.Length)
}

function Copy-RecordedFile {
  param([object]$Record, [string]$Group)
  $source = [string]$Record.path
  if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
    $script:missing.Add($source)
    return
  }
  $relative = Get-RelativeWorkspacePath -Path $source
  $destination = Join-Path (Join-Path $archive $Group) $relative
  $parent = Split-Path $destination -Parent
  New-Item -ItemType Directory -Force -Path $parent | Out-Null
  Copy-Item -LiteralPath $source -Destination $destination -Force
  $sourceHash = [string]$Record.sha256
  $archiveHash = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($sourceHash -and $sourceHash -ne $archiveHash) {
    throw "Archive hash mismatch: $source"
  }
  $script:copied.Add([pscustomobject]@{
    group = $Group
    source = $source
    archived_path = $destination
    size_bytes = (Get-Item -LiteralPath $destination).Length
    sha256 = $archiveHash
  })
}

$copied = [System.Collections.Generic.List[object]]::new()
$missing = [System.Collections.Generic.List[string]]::new()
$records = @($inventory.files)

# Preserve the complete hand-curated legacy model directory.
$records | Where-Object {
  ([string]$_.path).StartsWith(
    (Join-Path $workspaceRoot 'trained_models') + '\',
    [System.StringComparison]::OrdinalIgnoreCase
  )
} | ForEach-Object { Copy-RecordedFile -Record $_ -Group 'legacy_ppo_baselines' }

# Preserve all Hybrid v2 experiment artifacts from the dedicated Hybrid checkout.
$hybridRoot = Join-Path $workspaceRoot 'furp-2026-Zijie-Zhang-WheelLeggedRL-hybrid-v2\experiments\hybrid_v2'
$records | Where-Object {
  ([string]$_.path).StartsWith(
    $hybridRoot.TrimEnd('\') + '\',
    [System.StringComparison]::OrdinalIgnoreCase
  )
} | ForEach-Object { Copy-RecordedFile -Record $_ -Group 'hybrid_v2' }

# Preserve named research baselines that may exist only in the legacy run tree.
$representatives = @(
  @{ pattern = '*clean_wheel_seed1*\model_499.pt'; group = 'legacy_representatives\clean_balance' },
  @{ pattern = '*robust_l2_seed1*\model_1997.pt'; group = 'legacy_representatives\robust_l2' },
  @{ pattern = '*push_l3_seed1*\model_2996.pt'; group = 'legacy_representatives\push_l3' },
  @{ pattern = '*yawscale2p5*slew6_seed1*\model_892.pt'; group = 'legacy_representatives\slow_turn' },
  @{ pattern = '*preserved_checkpoints*\*.pt'; group = 'legacy_representatives\stage2_bidir' }
)
foreach ($selection in $representatives) {
  $matches = @($records | Where-Object { ([string]$_.path) -like $selection.pattern })
  if ($matches.Count -eq 0) {
    $missing.Add("SELECTION:$($selection.pattern)")
    continue
  }
  $matches | ForEach-Object {
    Copy-RecordedFile -Record $_ -Group ([string]$selection.group)
    $runDirectory = Split-Path ([string]$_.path) -Parent
    $records | Where-Object {
      ([string]$_.path).StartsWith(
        $runDirectory.TrimEnd('\') + '\',
        [System.StringComparison]::OrdinalIgnoreCase
      ) -and ([string]$_.name -in @('env.yaml', 'agent.yaml'))
    } | ForEach-Object { Copy-RecordedFile -Record $_ -Group ([string]$selection.group) }
  }
}

$copiedRecords = @($copied | Sort-Object group, source)
$manifest = [ordered]@{
  schema_version = 1
  created_at = (Get-Date).ToUniversalTime().ToString('o')
  source_inventory = $InventoryPath
  source_workspace = $workspaceRoot
  archive_root = $archive
  copied_file_count = $copiedRecords.Count
  copied_size_bytes = ($copiedRecords | Measure-Object size_bytes -Sum).Sum
  missing = @($missing)
  files = $copiedRecords
}
$manifestPath = Join-Path $archive 'archive_manifest.json'
$csvPath = Join-Path $archive 'archive_manifest.csv'
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
$copiedRecords | Export-Csv -LiteralPath $csvPath -NoTypeInformation -Encoding UTF8

if ($missing.Count -gt 0 -and -not $AllowMissing) {
  Write-Warning "Archive created, but $($missing.Count) required item(s) were missing."
  Write-Host "[REVIEW] $manifestPath"
  exit 2
}

Write-Host "[OK] Archive:  $archive"
Write-Host "[OK] Manifest: $manifestPath"
Write-Host "[OK] Files:    $($copiedRecords.Count)"
Write-Host '[SAFE] No source artifact was moved or deleted.'
