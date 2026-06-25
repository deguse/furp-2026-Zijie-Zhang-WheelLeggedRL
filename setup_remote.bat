@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM  Remote Lab PC Environment Setup for mjlab Training
REM  Run this script from the furp repo root (next to pyproject.toml).
REM  It will: clone mjlab to ..\mjlab-main, install uv, uv sync,
REM  and run a zero-agent smoke test.
REM ============================================================

set "FURP_DIR=%~dp0"
set "WORKSPACE_DIR=%FURP_DIR%.."
set "MJLAB_DIR=%WORKSPACE_DIR%\mjlab-main"
set "MJLAB_REPO=https://github.com/deguse/mjlab.git"

echo ===================================================
echo [Remote Lab PC Environment Setup for mjlab Training]
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

REM --- 0b. Preflight: confirm we are inside the furp repo ---
if not exist "%FURP_DIR%pyproject.toml" (
    echo [ERROR] pyproject.toml not found next to this script.
    echo Run setup_remote.bat from the furp repo root directory.
    pause
    exit /b 1
)

REM --- 0c. Preflight: NVIDIA GPU + CUDA driver ---
echo [0/4] Checking NVIDIA GPU and CUDA driver...
where nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN] nvidia-smi not found. Training requires an NVIDIA GPU with
    echo        a CUDA 12.8+ driver. CPU-only smoke test may still work.
) else (
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
)
echo.

REM --- 1. Install uv (idempotent) ---
echo [1/4] Checking uv...
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

REM --- 2. Clone or update mjlab framework to ..\mjlab-main ---
echo [2/4] Syncing mjlab framework to %MJLAB_DIR% ...
if exist "%MJLAB_DIR%\.git" (
    echo mjlab already exists, pulling latest...
    git -C "%MJLAB_DIR%" pull --ff-only
    if errorlevel 1 echo [WARN] mjlab pull failed, continuing with existing copy.
) else if exist "%MJLAB_DIR%" (
    echo [WARN] %MJLAB_DIR% exists but is not a git repo. Skipping clone.
    echo        Remove it manually if you want a fresh clone.
) else (
    git clone "%MJLAB_REPO%" "%MJLAB_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to clone mjlab from %MJLAB_REPO%
        echo        Check network and GitHub access.
        pause
        exit /b 1
    )
)
echo.

REM --- 3. uv sync (auto-installs Python 3.11, resolves ../mjlab-main editable) ---
echo [3/4] Synchronizing virtual environment (uv sync)...
cd /d "%FURP_DIR%"
uv sync
if errorlevel 1 (
    echo [ERROR] uv sync failed. See messages above.
    pause
    exit /b 1
)
echo.

REM --- 4. Smoke test: step env with zero actions ---
echo [4/4] Smoke test: zero_agent ...
uv run python src/hoppertrex_mjlab/scripts/zero_agent.py
if errorlevel 1 (
    echo [WARN] smoke test returned nonzero. If the error is GPU-related,
    echo        verify nvidia-smi works and the CUDA driver is ^>=12.8.
) else (
    echo [OK] Environment verified.
)
echo.

echo ===================================================
echo [SUCCESS] Setup complete! Virtual environment is ready.
echo.
echo Train (4096 envs):  uv run python src/hoppertrex_mjlab/scripts/rsl_rl/train.py --env.scene.num-envs 4096
echo Play a policy:      uv run python src/hoppertrex_mjlab/scripts/rsl_rl/play.py --agent trained
echo List environments:  uv run python -m mjlab.scripts.list_envs
echo ===================================================
pause
