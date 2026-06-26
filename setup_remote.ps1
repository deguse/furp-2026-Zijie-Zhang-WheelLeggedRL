param(
    [int]$Seed = 1,
    [int]$NumEnvs = 256,
    [int]$MaxIterations = 500,
    [int]$SaveInterval = 50,
    [string]$RunName = "",
    [bool]$StashLocalChanges = $true,
    [switch]$SkipSmoke
)

$ErrorActionPreference = "Stop"

$Workspace = "C:\mjlab_workspace"
$FurpRepo = "https://github.com/deguse/furp-2026-Zijie-Zhang-WheelLeggedRL.git"
$FurpDirName = "furp-2026-Zijie-Zhang-WheelLeggedRL"
$MjlabRepo = "https://github.com/deguse/mjlab.git"
$MjlabDirName = "mjlab-main"

$FurpDir = Join-Path $Workspace $FurpDirName
$MjlabDir = Join-Path $Workspace $MjlabDirName
$MachineTag = ($env:COMPUTERNAME -replace "[^A-Za-z0-9_-]", "_").ToLower()
if ([string]::IsNullOrWhiteSpace($RunName)) {
    $RunName = "clean_wheel_${MachineTag}_seed${Seed}"
}

function Sync-GitRepo {
    param(
        [string]$Name,
        [string]$RepoUrl,
        [string]$Directory
    )

    if (Test-Path (Join-Path $Directory ".git")) {
        Write-Host "Found existing $Name repo: $Directory"

        $status = git -C $Directory status --porcelain
        if ($status -and $StashLocalChanges) {
            $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
            Write-Host "[WARN] Local changes detected in $Name. Stashing before pull..." -ForegroundColor Yellow
            git -C $Directory stash push -u -m "setup_remote autostash $stamp"
        } elseif ($status) {
            throw "Local changes detected in $Directory. Re-run with -StashLocalChanges `$true or clean the repo first."
        }

        git -C $Directory pull --ff-only
    } elseif (Test-Path $Directory) {
        throw "Folder exists but is not a Git repo: $Directory. Rename/remove it first."
    } else {
        git clone $RepoUrl $Directory
    }

    Write-Host "[OK] $Name repo ready." -ForegroundColor Green
    Write-Host "$Name commit:"
    git -C $Directory log -1 --oneline
}

function Ensure-MjlabViewerPatch {
    param([string]$Directory)

    $velocityCommand = Join-Path $Directory "src\mjlab\tasks\velocity\mdp\velocity_command.py"
    if (-not (Test-Path $velocityCommand)) {
        Write-Host "[WARN] velocity_command.py not found, skipping viewer patch." -ForegroundColor Yellow
        return
    }

    $text = Get-Content -Path $velocityCommand -Raw
    $updated = $text
    $updated = $updated.Replace('("lin_vel_x", ranges.lin_vel_x[1])', '("lin_vel_x", max(abs(v) for v in ranges.lin_vel_x))')
    $updated = $updated.Replace('("lin_vel_y", ranges.lin_vel_y[1])', '("lin_vel_y", max(abs(v) for v in ranges.lin_vel_y))')
    $updated = $updated.Replace('("ang_vel_z", ranges.ang_vel_z[1])', '("ang_vel_z", max(abs(v) for v in ranges.ang_vel_z))')
    $updated = $updated.Replace("min=0.1,", "min=0.0,")

    if ($updated -ne $text) {
        Set-Content -Path $velocityCommand -Value $updated -NoNewline -Encoding UTF8
        Write-Host "[OK] Applied mjlab viewer zero-range slider patch." -ForegroundColor Green
    } else {
        Write-Host "[OK] mjlab viewer zero-range slider patch already present." -ForegroundColor Green
    }
}

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "[Remote Lab PC Setup - PowerShell HTTPS mode]" -ForegroundColor Cyan
Write-Host "Workspace: $Workspace"
Write-Host "Run name:  $RunName"
Write-Host "Seed:      $Seed"
Write-Host "===================================================" -ForegroundColor Cyan

# 0. Create workspace
Write-Host "`n[0/8] Creating workspace..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $Workspace | Out-Null
Write-Host "[OK] Workspace ready: $Workspace" -ForegroundColor Green

# 1. Check Git
Write-Host "`n[1/8] Checking Git..." -ForegroundColor Yellow
$git = Get-Command git -ErrorAction SilentlyContinue
if (-not $git) {
    throw "Git not found. Install Git for Windows: https://git-scm.com/download/win"
}
git --version
Write-Host "[OK] Git available." -ForegroundColor Green

# 2. Clone or update FURP repo
Write-Host "`n[2/8] Syncing FURP repo..." -ForegroundColor Yellow
Sync-GitRepo -Name "FURP" -RepoUrl $FurpRepo -Directory $FurpDir

# 3. Clone or update mjlab-main
Write-Host "`n[3/8] Syncing mjlab framework..." -ForegroundColor Yellow
Sync-GitRepo -Name "mjlab" -RepoUrl $MjlabRepo -Directory $MjlabDir
Ensure-MjlabViewerPatch -Directory $MjlabDir

# 4. NVIDIA GPU check
Write-Host "`n[4/8] Checking NVIDIA GPU..." -ForegroundColor Yellow
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
Write-Host "`n[5/8] Checking uv..." -ForegroundColor Yellow
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
Write-Host "`n[6/8] Synchronizing Python environment..." -ForegroundColor Yellow
Set-Location $FurpDir
& $uv sync --frozen
Write-Host "[OK] uv sync complete." -ForegroundColor Green

# 7. Smoke test
Write-Host "`n[7/8] Running environment checks..." -ForegroundColor Yellow

Write-Host "`nPython / Torch / CUDA check:"
& $uv run python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
$SmokeDevice = (& $uv run python -c "import torch; print('cuda:0' if torch.cuda.is_available() else 'cpu')").Trim()

if (-not $SkipSmoke) {
    Write-Host "`nRunning fixed-wheel sweep..."
    & $uv run python src\hoppertrex_mjlab\scripts\fixed_wheel_sweep.py --steps 150

    Write-Host "`nRunning zero-agent smoke test on $SmokeDevice..."
    & $uv run python src\hoppertrex_mjlab\scripts\zero_agent.py --device $SmokeDevice --num_envs 1 --max_steps 100
} else {
    Write-Host "[SKIP] Smoke tests skipped by -SkipSmoke." -ForegroundColor Yellow
}

# 8. Final commands
Write-Host "`n[8/8] Training command for this machine..." -ForegroundColor Yellow
$TrainCommand = "uv run python src\hoppertrex_mjlab\scripts\rsl_rl\train.py --env.scene.num-envs $NumEnvs --agent.max-iterations $MaxIterations --agent.save-interval $SaveInterval --agent.seed $Seed --agent.run-name $RunName"

Write-Host "`n===================================================" -ForegroundColor Cyan
Write-Host "[SUCCESS] Setup complete." -ForegroundColor Green
Write-Host "Project directory: $FurpDir"
Write-Host "mjlab directory:   $MjlabDir"
Write-Host "Run name:          $RunName"
Write-Host "===================================================" -ForegroundColor Cyan

Write-Host "`nRecommended commands:"
Write-Host "cd $FurpDir"
Write-Host $TrainCommand
Write-Host "nvidia-smi -l 1"
