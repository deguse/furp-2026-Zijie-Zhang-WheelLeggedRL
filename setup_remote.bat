@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM  Remote Lab PC Environment Setup for mjlab Training (SSH)
REM
REM  This script is self-bootstrapping. Place it in any empty folder
REM  on the remote machine and double-click. It will:
REM    1. Configure an SSH key for GitHub (generate + registration guide)
REM    2. Clone the furp repo (if not already inside it)
REM    3. Clone mjlab to ..\mjlab-main (keeps editable path intact)
REM    4. Install uv, uv sync (auto-installs Python 3.11 + deps)
REM    5. Run a zero-agent smoke test
REM
REM  All Git operations use SSH (git@github.com).
REM ============================================================

set "FURP_REPO_SSH=git@github.com:deguse/furp-2026-Zijie-Zhang-WheelLeggedRL.git"
set "FURP_DIRNAME=furp-2026-Zijie-Zhang-WheelLeggedRL"
set "MJLAB_REPO_SSH=git@github.com:deguse/mjlab.git"
set "MJLAB_DIRNAME=mjlab-main"

set "SCRIPT_DIR=%~dp0"
set "SSH_DIR=%USERPROFILE%\.ssh"

echo ===================================================
echo [Remote Lab PC Setup - SSH mode]
echo ===================================================
echo.

REM --- 0. Preflight: git ---
where git >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] git not found.
    echo Install Git for Windows: https://git-scm.com/download/win
    pause
    exit /b 1
)

REM --- 0b. Preflight: OpenSSH client ---
where ssh-keygen >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] ssh-keygen not found. Enable Windows OpenSSH Client via:
    echo   Settings ^> Apps ^> Optional features ^> Add "OpenSSH Client"
    pause
    exit /b 1
)
echo [OK] git and OpenSSH available.
echo.

REM --- 1. SSH key for GitHub (idempotent + registration guide) ---
echo [1/6] Configuring SSH key for GitHub...
if not exist "%SSH_DIR%" mkdir "%SSH_DIR%"

REM Pre-seed github.com host key to avoid interactive host-key prompt
findstr /C:"github.com" "%SSH_DIR%\known_hosts" >nul 2>&1
if errorlevel 1 (
    echo Resolving github.com host key...
    ssh-keyscan -t rsa,ed25519 github.com >> "%SSH_DIR%\known_hosts" 2>nul
)

REM Generate ed25519 key if none exists
if not exist "%SSH_DIR%\id_ed25519" (
    echo Generating ed25519 SSH key ^(no passphrase^)...
    ssh-keygen -t ed25519 -C "mjlab-remote" -f "%SSH_DIR%\id_ed25519" -N ""
    if errorlevel 1 (
        echo [ERROR] ssh-keygen failed.
        pause
        exit /b 1
    )
) else (
    echo Existing SSH key found.
)

REM Test GitHub SSH auth. Note: `ssh -T git@github.com` exits with
REM code 1 even on success (GitHub refuses shell), so check the message.
echo Testing GitHub SSH authentication...
ssh -T -o BatchMode=yes -o ConnectTimeout=10 git@github.com > "%TEMP%\gh_ssh_test.txt" 2>&1
findstr /C:"successfully authenticated" "%TEMP%\gh_ssh_test.txt" >nul
if errorlevel 1 (
    echo.
    echo -------------------------------------------------------
    echo SSH key is NOT registered on your GitHub account yet.
    echo Copy the public key below and add it at:
    echo   https://github.com/settings/ssh/new
    echo ^(log in as the deguse account^)
    echo.
    type "%SSH_DIR%\id_ed25519.pub"
    echo.
    echo After adding the key, press any key to continue...
    pause >nul
    echo Re-testing GitHub SSH...
    ssh -T -o BatchMode=yes -o ConnectTimeout=10 git@github.com > "%TEMP%\gh_ssh_test2.txt" 2>&1
    findstr /C:"successfully authenticated" "%TEMP%\gh_ssh_test2.txt" >nul
    if errorlevel 1 (
        echo [ERROR] GitHub SSH still failing. Make sure the key was added
        echo        to the deguse account, then re-run this script.
        type "%TEMP%\gh_ssh_test2.txt"
        pause
        exit /b 1
    )
)
echo [OK] GitHub SSH authenticated.
echo.

REM --- 2. Ensure furp repo is present (self-bootstrap) ---
echo [2/6] Ensuring furp repo is present...
if exist "%SCRIPT_DIR%pyproject.toml" (
    REM Script is already inside the furp repo
    set "FURP_DIR=%SCRIPT_DIR%"
    echo Running inside furp repo: !FURP_DIR!
) else if exist "%SCRIPT_DIR%%FURP_DIRNAME%\pyproject.toml" (
    set "FURP_DIR=%SCRIPT_DIR%%FURP_DIRNAME%"
    echo Found existing furp repo at: !FURP_DIR!
    echo Pulling latest...
    git -C "!FURP_DIR!" pull --ff-only
) else (
    set "FURP_DIR=%SCRIPT_DIR%%FURP_DIRNAME%"
    echo Cloning furp repo ^(SSH^) to: !FURP_DIR!
    git clone "%FURP_REPO_SSH%" "!FURP_DIR!"
    if errorlevel 1 (
        echo [ERROR] Failed to clone furp. Check SSH access to GitHub.
        pause
        exit /b 1
    )
)
echo.

REM --- 3. Clone or update mjlab to sibling ..\mjlab-main ---
REM   furp's pyproject.toml references mjlab as ../mjlab-main (editable),
REM   so mjlab must live as a sibling of the furp directory.
for %%I in ("!FURP_DIR!") do set "WORKSPACE_DIR=%%~dpI"
set "WORKSPACE_DIR=!WORKSPACE_DIR:~0,-1!"
set "MJLAB_DIR=!WORKSPACE_DIR!\%MJLAB_DIRNAME%"

echo [3/6] Syncing mjlab framework to !MJLAB_DIR! ...
if exist "!MJLAB_DIR!\.git" (
    echo mjlab already exists, pulling latest...
    git -C "!MJLAB_DIR!" pull --ff-only
    if errorlevel 1 echo [WARN] mjlab pull failed, continuing with existing copy.
) else if exist "!MJLAB_DIR!" (
    echo [WARN] !MJLAB_DIR! exists but is not a git repo. Skipping clone.
    echo        Remove it manually if you want a fresh clone.
) else (
    git clone "%MJLAB_REPO_SSH%" "!MJLAB_DIR!"
    if errorlevel 1 (
        echo [ERROR] Failed to clone mjlab from %MJLAB_REPO_SSH%
        echo        Check SSH access and that the repo exists on GitHub.
        pause
        exit /b 1
    )
)
echo.

REM --- 4. NVIDIA GPU + CUDA driver check ---
echo [4/6] Checking NVIDIA GPU and CUDA driver...
where nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN] nvidia-smi not found. Training requires an NVIDIA GPU with
    echo        a CUDA 12.8+ driver. CPU-only smoke test may still work.
) else (
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
)
echo.

REM --- 5. Install uv (idempotent) ---
echo [5/6] Checking uv...
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo uv not found, installing...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
) else (
    echo uv is already installed.
    uv --version
)
echo.

REM --- 6. uv sync + smoke test ---
echo [6/6] Synchronizing environment and running smoke test...
cd /d "!FURP_DIR!"
uv sync
if errorlevel 1 (
    echo [ERROR] uv sync failed. See messages above.
    pause
    exit /b 1
)

echo.
echo Running zero_agent smoke test...
uv run python src/hoppertrex_mjlab/scripts/zero_agent.py
if errorlevel 1 (
    echo [WARN] smoke test returned nonzero. If GPU-related, verify
    echo        nvidia-smi works and CUDA driver is ^>=12.8.
) else (
    echo [OK] Environment verified.
)
echo.

echo ===================================================
echo [SUCCESS] Setup complete! Environment is ready.
echo.
echo Train (4096 envs):  uv run python src/hoppertrex_mjlab/scripts/rsl_rl/train.py --env.scene.num-envs 4096
echo Play a policy:      uv run python src/hoppertrex_mjlab/scripts/rsl_rl/play.py --agent trained
echo List environments:  uv run python -m mjlab.scripts.list_envs
echo ===================================================
pause
