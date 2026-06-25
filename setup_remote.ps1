$ErrorActionPreference = "Stop"

$Workspace = "C:\mjlab_workspace"
$FurpRepo = "https://github.com/deguse/furp-2026-Zijie-Zhang-WheelLeggedRL.git"
$FurpDirName = "furp-2026-Zijie-Zhang-WheelLeggedRL"
$MjlabRepo = "https://github.com/deguse/mjlab.git"
$MjlabDirName = "mjlab-main"

$FurpDir = Join-Path $Workspace $FurpDirName
$MjlabDir = Join-Path $Workspace $MjlabDirName

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "[Remote Lab PC Setup - PowerShell HTTPS mode]" -ForegroundColor Cyan
Write-Host "Workspace: $Workspace"
Write-Host "===================================================" -ForegroundColor Cyan

# 0. Create workspace
Write-Host "`n[0/7] Creating workspace..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $Workspace | Out-Null
Write-Host "[OK] Workspace ready: $Workspace" -ForegroundColor Green

# 1. Check Git
Write-Host "`n[1/7] Checking Git..." -ForegroundColor Yellow
$git = Get-Command git -ErrorAction SilentlyContinue
if (-not $git) {
    throw "Git not found. Install Git for Windows: https://git-scm.com/download/win"
}
git --version
Write-Host "[OK] Git available." -ForegroundColor Green

# 2. Clone or update FURP repo
Write-Host "`n[2/7] Syncing FURP repo..." -ForegroundColor Yellow
if (Test-Path (Join-Path $FurpDir ".git")) {
    Write-Host "Found existing FURP repo: $FurpDir"
    git -C $FurpDir pull --ff-only
} elseif (Test-Path $FurpDir) {
    throw "Folder exists but is not a Git repo: $FurpDir. Rename/remove it first."
} else {
    git clone $FurpRepo $FurpDir
}
Write-Host "[OK] FURP repo ready." -ForegroundColor Green

# 3. Clone or update mjlab-main
Write-Host "`n[3/7] Syncing mjlab framework..." -ForegroundColor Yellow
if (Test-Path (Join-Path $MjlabDir ".git")) {
    Write-Host "Found existing mjlab repo: $MjlabDir"
    git -C $MjlabDir pull --ff-only
} elseif (Test-Path $MjlabDir) {
    throw "Folder exists but is not a Git repo: $MjlabDir. Rename/remove it first."
} else {
    git clone $MjlabRepo $MjlabDir
}
Write-Host "[OK] mjlab repo ready." -ForegroundColor Green

# 4. NVIDIA GPU check
Write-Host "`n[4/7] Checking NVIDIA GPU..." -ForegroundColor Yellow
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $nvidiaSmi) {
    $candidate = "C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
    if (Test-Path $candidate) {
        $nvidiaSmi = $candidate
    }
}

if ($nvidiaSmi) {
    $gpuInfo = & $nvidiaSmi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits
    Write-Host $gpuInfo

    $firstGpu = $gpuInfo | Select-Object -First 1
    $parts = $firstGpu -split ","
    $gpuName = $parts[0].Trim()
    $gpuMem = [int]($parts[2].Trim())

    Write-Host "Detected GPU: $gpuName"
    Write-Host "VRAM MiB: $gpuMem"

    if ($gpuMem -lt 6000) {
        Write-Host "[WARN] GPU VRAM is less than 6GB. Use only small tests on this machine." -ForegroundColor Red
    } else {
        Write-Host "[OK] GPU VRAM looks suitable for training tests." -ForegroundColor Green
    }
} else {
    Write-Host "[WARN] nvidia-smi not found. This machine may not have a usable NVIDIA GPU." -ForegroundColor Red
}

# 5. Install or locate uv
Write-Host "`n[5/7] Checking uv..." -ForegroundColor Yellow
$uv = Get-Command uv -ErrorAction SilentlyContinue

if (-not $uv) {
    $uvPath = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
    if (Test-Path $uvPath) {
        $uv = $uvPath
    }
}

if (-not $uv) {
    Write-Host "uv not found. Installing uv..."
    powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $uvPath = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
    if (Test-Path $uvPath) {
        $uv = $uvPath
    }
}

if (-not $uv) {
    throw "uv installation failed or uv.exe not found."
}

# Add uv to current PATH
$uvBin = Join-Path $env:USERPROFILE ".local\bin"
$env:Path = "$uvBin;$env:Path"

# Persist uv to user PATH
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (($userPath -split ';') -notcontains $uvBin) {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$uvBin", "User")
}

Write-Host "Using uv:"
& $uv --version
Write-Host "[OK] uv ready." -ForegroundColor Green

# 6. uv sync
Write-Host "`n[6/7] Synchronizing Python environment..." -ForegroundColor Yellow
Set-Location $FurpDir
& $uv sync
Write-Host "[OK] uv sync complete." -ForegroundColor Green

# 7. Smoke test
Write-Host "`n[7/7] Running environment checks..." -ForegroundColor Yellow

Write-Host "`nPython / Torch / CUDA check:"
& $uv run python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

Write-Host "`nRunning zero-agent smoke test..."
& $uv run python src\hoppertrex_mjlab\scripts\zero_agent.py

Write-Host "`n===================================================" -ForegroundColor Cyan
Write-Host "[SUCCESS] Setup complete." -ForegroundColor Green
Write-Host "Project directory: $FurpDir"
Write-Host "mjlab directory:   $MjlabDir"
Write-Host "===================================================" -ForegroundColor Cyan

Write-Host "`nRecommended commands:"
Write-Host "cd $FurpDir"
Write-Host "uv run python src\hoppertrex_mjlab\scripts\rsl_rl\train.py --env.scene.num-envs 16 --agent.max-iterations 10 --agent.run-name smoke_test"
Write-Host "uv run python src\hoppertrex_mjlab\scripts\rsl_rl\train.py --env.scene.num-envs 256 --agent.max-iterations 500 --agent.save-interval 50 --agent.run-name standing_test"
Write-Host "nvidia-smi -l 1"