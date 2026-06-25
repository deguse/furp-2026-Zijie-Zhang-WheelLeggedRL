#Requires -Version 5.1
$ErrorActionPreference = "Stop"

# ============================================================
#  Remote Lab PC Environment Setup for mjlab Training (SSH, PowerShell)
#
#  Usage in VS Code remote terminal (PowerShell):
#    powershell -ExecutionPolicy Bypass -File .\setup_remote.ps1
#  Or paste the whole file into the terminal.
#
#  Self-bootstrapping: configures SSH key, clones furp + mjlab,
#  installs uv, uv sync, GPU/torch/CUDA checks, zero-agent smoke test.
#  All Git operations use SSH (git@github.com).
# ============================================================

$Workspace       = "C:\mjlab_workspace"
$FurpRepoSSH     = "git@github.com:deguse/furp-2026-Zijie-Zhang-WheelLeggedRL.git"
$FurpDirName     = "furp-2026-Zijie-Zhang-WheelLeggedRL"
$MjlabRepoSSH    = "git@github.com:deguse/mjlab.git"
$MjlabDirName    = "mjlab-main"

$FurpDir  = Join-Path $Workspace $FurpDirName
$MjlabDir = Join-Path $Workspace $MjlabDirName
$SshDir   = Join-Path $env:USERPROFILE ".ssh"

function Write-Step($i, $n, $msg) { Write-Host "`n[$i/$n] $msg" -ForegroundColor Yellow }
function Write-Ok($msg)           { Write-Host "[OK] $msg" -ForegroundColor Green }

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "[Remote Lab PC Setup - SSH / PowerShell]" -ForegroundColor Cyan
Write-Host "Workspace: $Workspace"
Write-Host "===================================================" -ForegroundColor Cyan

# --- 0. Preflight: git + OpenSSH ---
Write-Step 0 8 "Checking git and OpenSSH..."
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git not found. Install Git for Windows: https://git-scm.com/download/win"
}
if (-not (Get-Command ssh-keygen -ErrorAction SilentlyContinue)) {
    throw "ssh-keygen not found. Enable Windows OpenSSH Client: Settings > Apps > Optional features > Add 'OpenSSH Client'"
}
git --version
Write-Ok "git and OpenSSH available."

# --- 1. SSH key for GitHub (idempotent + registration guide) ---
Write-Step 1 8 "Configuring SSH key for GitHub..."
if (-not (Test-Path $SshDir)) { New-Item -ItemType Directory -Force -Path $SshDir | Out-Null }

# Pre-seed github.com host key to avoid interactive host-key prompt
$knownHosts = Join-Path $SshDir "known_hosts"
if (-not (Test-Path $knownHosts) -or -not (Select-String -Path $knownHosts -Pattern "github.com" -Quiet)) {
    Write-Host "Resolving github.com host key..."
    ssh-keyscan -t rsa,ed25519 github.com >> $knownHosts 2>$null
}

# Generate ed25519 key if none exists
$keyPath = Join-Path $SshDir "id_ed25519"
if (-not (Test-Path $keyPath)) {
    Write-Host "Generating ed25519 SSH key (no passphrase)..."
    # NB: -N '""' is the verified way to pass an empty passphrase to
    # ssh-keygen when invoked from PowerShell.
    ssh-keygen -t ed25519 -C "mjlab-remote" -f $keyPath -N '""'
    if ($LASTEXITCODE -ne 0) { throw "ssh-keygen failed." }
} else {
    Write-Host "Existing SSH key found."
}

# Test GitHub SSH auth. `ssh -T git@github.com` exits 1 even on success
# (GitHub refuses shell), so check the message text instead.
Write-Host "Testing GitHub SSH authentication..."
$test = ssh -T -o BatchMode=yes -o ConnectTimeout=10 git@github.com 2>&1
if ($test -notmatch "successfully authenticated") {
    Write-Host "-------------------------------------------------------" -ForegroundColor Red
    Write-Host "SSH key is NOT registered on your GitHub account yet." -ForegroundColor Red
    Write-Host "Copy the public key below and add it at:" -ForegroundColor Red
    Write-Host "  https://github.com/settings/ssh/new" -ForegroundColor Red
    Write-Host "(log in as the deguse account)" -ForegroundColor Red
    Write-Host ""
    Get-Content "$keyPath.pub"
    Write-Host "-------------------------------------------------------" -ForegroundColor Red
    Read-Host "Press Enter after adding the key to GitHub"
    Write-Host "Re-testing GitHub SSH..."
    $test = ssh -T -o BatchMode=yes -o ConnectTimeout=10 git@github.com 2>&1
    if ($test -notmatch "successfully authenticated") {
        Write-Host $test
        throw "GitHub SSH still failing. Make sure the key was added to the deguse account."
    }
}
Write-Ok "GitHub SSH authenticated."

# --- 2. Create workspace ---
Write-Step 2 8 "Creating workspace..."
New-Item -ItemType Directory -Force -Path $Workspace | Out-Null
Write-Ok "Workspace ready: $Workspace"

# --- 3. Clone or update furp repo (SSH) ---
Write-Step 3 8 "Syncing furp repo..."
if (Test-Path (Join-Path $FurpDir ".git")) {
    Write-Host "Found existing furp repo, pulling latest..."
    git -C $FurpDir pull --ff-only
} elseif (Test-Path $FurpDir) {
    throw "Folder exists but is not a git repo: $FurpDir. Rename/remove it first."
} else {
    git clone $FurpRepoSSH $FurpDir
}
Write-Ok "furp repo ready."

# --- 4. Clone or update mjlab framework (SSH, sibling of furp) ---
Write-Step 4 8 "Syncing mjlab framework..."
if (Test-Path (Join-Path $MjlabDir ".git")) {
    Write-Host "Found existing mjlab repo, pulling latest..."
    git -C $MjlabDir pull --ff-only
} elseif (Test-Path $MjlabDir) {
    throw "Folder exists but is not a git repo: $MjlabDir. Rename/remove it first."
} else {
    git clone $MjlabRepoSSH $MjlabDir
}
Write-Ok "mjlab repo ready."

# --- 5. NVIDIA GPU + VRAM check ---
Write-Step 5 8 "Checking NVIDIA GPU and CUDA driver..."
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $nvidiaSmi) {
    $candidate = "C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
    if (Test-Path $candidate) { $nvidiaSmi = $candidate }
}
if ($nvidiaSmi) {
    $gpuInfo = & $nvidiaSmi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits
    Write-Host $gpuInfo
    $firstGpu = ($gpuInfo | Select-Object -First 1) -split ","
    $gpuName = $firstGpu[0].Trim()
    $gpuMem  = [int]$firstGpu[2].Trim()
    Write-Host "Detected GPU: $gpuName  (VRAM: $gpuMem MiB)"
    if ($gpuMem -lt 6000) {
        Write-Host "[WARN] GPU VRAM < 6GB. Use only small smoke tests on this machine." -ForegroundColor Red
    } else {
        Write-Ok "GPU VRAM looks suitable for training."
    }
} else {
    Write-Host "[WARN] nvidia-smi not found. No usable NVIDIA GPU?" -ForegroundColor Red
}

# --- 6. Install or locate uv + persist to user PATH ---
Write-Step 6 8 "Checking uv..."
$uv = Get-Command uv -ErrorAction SilentlyContinue
$uvBin = Join-Path $env:USERPROFILE ".local\bin"
if (-not $uv) {
    $uvExe = Join-Path $uvBin "uv.exe"
    if (Test-Path $uvExe) { $uv = $uvExe }
}
if (-not $uv) {
    Write-Host "uv not found, installing..."
    powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $uvExe = Join-Path $uvBin "uv.exe"
    if (Test-Path $uvExe) { $uv = $uvExe }
}
if (-not $uv) { throw "uv installation failed or uv.exe not found." }
$env:Path = "$uvBin;$env:Path"
# Persist uv to user PATH for future terminals
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (($userPath -split ';') -notcontains $uvBin) {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$uvBin", "User")
    Write-Host "Persisted $uvBin to user PATH."
}
& $uv --version
Write-Ok "uv ready."

# --- 7. uv sync ---
Write-Step 7 8 "Synchronizing Python environment (uv sync)..."
Set-Location $FurpDir
& $uv sync
if ($LASTEXITCODE -ne 0) { throw "uv sync failed. See messages above." }
Write-Ok "uv sync complete."

# --- 8. torch/CUDA check + zero-agent smoke test ---
Write-Step 8 8 "Running environment checks..."
Write-Host "`nPython / torch / CUDA check:"
& $uv run python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

Write-Host "`nRunning zero-agent smoke test..."
& $uv run python src\hoppertrex_mjlab\scripts\zero_agent.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] smoke test returned nonzero. If GPU-related, verify nvidia-smi and CUDA driver >= 12.8." -ForegroundColor Red
} else {
    Write-Ok "Environment verified."
}

Write-Host "`n===================================================" -ForegroundColor Cyan
Write-Host "[SUCCESS] Setup complete." -ForegroundColor Green
Write-Host "Project dir: $FurpDir"
Write-Host "mjlab dir:   $MjlabDir"
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "`nRecommended commands:"
Write-Host "  cd $FurpDir"
Write-Host "  # tiny smoke run (16 envs, 10 iters):"
Write-Host "  uv run python src\hoppertrex_mjlab\scripts\rsl_rl\train.py --env.scene.num-envs 16 --agent.max-iterations 10 --agent.run-name smoke_test"
Write-Host "  # standing-balance run (256 envs, 500 iters):"
Write-Host "  uv run python src\hoppertrex_mjlab\scripts\rsl_rl\train.py --env.scene.num-envs 256 --agent.max-iterations 500 --agent.save-interval 50 --agent.run-name standing_test"
Write-Host "  # full-scale training (4096 envs):"
Write-Host "  uv run python src\hoppertrex_mjlab\scripts\rsl_rl\train.py --env.scene.num-envs 4096"
Write-Host "  nvidia-smi -l 1   # watch GPU utilisation"
