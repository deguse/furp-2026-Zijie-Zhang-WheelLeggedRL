@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM  Remote Lab PC Environment Setup for mjlab Training
REM  Public GitHub HTTPS version
REM
REM  What this script does:
REM    1. Create workspace at C:\mjlab_workspace
REM    2. Clone or update furp project
REM    3. Clone or update mjlab-main as sibling folder
REM    4. Check NVIDIA GPU and warn if VRAM is too small
REM    5. Install uv if missing
REM    6. Run uv sync
REM    7. Run basic Python/CUDA check and zero-agent smoke test
REM
REM  No SSH key is required for public repositories.
REM ============================================================

set "WORKSPACE=C:\mjlab_workspace"

set "FURP_REPO_URL=https://github.com/deguse/furp-2026-Zijie-Zhang-WheelLeggedRL.git"
set "FURP_DIRNAME=furp-2026-Zijie-Zhang-WheelLeggedRL"

set "MJLAB_REPO_URL=https://github.com/deguse/mjlab.git"
set "MJLAB_DIRNAME=mjlab-main"

set "FURP_DIR=%WORKSPACE%\%FURP_DIRNAME%"
set "MJLAB_DIR=%WORKSPACE%\%MJLAB_DIRNAME%"

echo ===================================================
echo [Remote Lab PC Setup - Public HTTPS mode]
echo ===================================================
echo Workspace: %WORKSPACE%
echo.

REM ------------------------------------------------------------
REM 0. Create workspace
REM ------------------------------------------------------------
echo [0/7] Creating workspace...
if not exist "%WORKSPACE%" (
    mkdir "%WORKSPACE%"
    if errorlevel 1 (
        echo [ERROR] Failed to create workspace: %WORKSPACE%
        pause
        exit /b 1
    )
)
echo [OK] Workspace ready.
echo.

REM ------------------------------------------------------------
REM 1. Preflight: git
REM ------------------------------------------------------------
echo [1/7] Checking Git...
where git >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git not found.
    echo Please install Git for Windows:
    echo https://git-scm.com/download/win
    pause
    exit /b 1
)
git --version
echo [OK] Git available.
echo.

REM ------------------------------------------------------------
REM 2. Clone or update furp project
REM ------------------------------------------------------------
echo [2/7] Syncing FURP repo...
if exist "%FURP_DIR%\.git" (
    echo Found existing FURP repo:
    echo %FURP_DIR%
    echo Pulling latest...
    git -C "%FURP_DIR%" pull --ff-only
    if errorlevel 1 (
        echo [WARN] git pull failed. Continuing with existing copy.
    )
) else if exist "%FURP_DIR%" (
    echo [WARN] Folder exists but is not a Git repo:
    echo %FURP_DIR%
    echo Please rename or remove it if you want a fresh clone.
    pause
    exit /b 1
) else (
    echo Cloning FURP repo to:
    echo %FURP_DIR%
    git clone "%FURP_REPO_URL%" "%FURP_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to clone FURP repo.
        pause
        exit /b 1
    )
)
echo [OK] FURP repo ready.
echo.

REM ------------------------------------------------------------
REM 3. Clone or update mjlab-main
REM ------------------------------------------------------------
echo [3/7] Syncing mjlab framework...
if exist "%MJLAB_DIR%\.git" (
    echo Found existing mjlab repo:
    echo %MJLAB_DIR%
    echo Pulling latest...
    git -C "%MJLAB_DIR%" pull --ff-only
    if errorlevel 1 (
        echo [WARN] mjlab pull failed. Continuing with existing copy.
    )
) else if exist "%MJLAB_DIR%" (
    echo [WARN] Folder exists but is not a Git repo:
    echo %MJLAB_DIR%
    echo Please rename or remove it if you want a fresh clone.
    pause
    exit /b 1
) else (
    echo Cloning mjlab repo to:
    echo %MJLAB_DIR%
    git clone "%MJLAB_REPO_URL%" "%MJLAB_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to clone mjlab repo.
        pause
        exit /b 1
    )
)
echo [OK] mjlab repo ready.
echo.

REM ------------------------------------------------------------
REM 4. NVIDIA GPU check
REM ------------------------------------------------------------
echo [4/7] Checking NVIDIA GPU...

set "NVSMI=nvidia-smi"
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    if exist "C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe" (
        set "NVSMI=C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
    ) else (
        set "NVSMI="
    )
)

if not defined NVSMI (
    echo [WARN] nvidia-smi not found.
    echo        This machine may not have a usable NVIDIA GPU.
    echo        Training is likely not suitable here.
) else (
    "%NVSMI%" --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits > "%TEMP%\gpu_info.txt" 2>nul

    if errorlevel 1 (
        echo [WARN] nvidia-smi exists but query failed.
    ) else (
        type "%TEMP%\gpu_info.txt"
        for /f "tokens=1,2,3 delims=," %%A in ("%TEMP%\gpu_info.txt") do (
            set "GPU_NAME=%%A"
            set "GPU_DRIVER=%%B"
            set "GPU_MEM=%%C"
        )

        set "GPU_MEM=!GPU_MEM: =!"

        echo.
        echo Detected GPU: !GPU_NAME!
        echo VRAM MiB: !GPU_MEM!

        if not "!GPU_MEM!"=="" (
            if !GPU_MEM! LSS 6000 (
                echo [WARN] GPU VRAM is less than 6GB.
                echo        This machine is only suitable for small tests.
                echo        For formal training, use RTX 2080 / 8GB or better.
            ) else (
                echo [OK] GPU VRAM looks suitable for training tests.
            )
        )
    )
)
echo.

REM ------------------------------------------------------------
REM 5. Install or locate uv
REM ------------------------------------------------------------
echo [5/7] Checking uv...

set "UV_CMD="
where uv >nul 2>&1
if not errorlevel 1 (
    set "UV_CMD=uv"
)

if not defined UV_CMD (
    if exist "%USERPROFILE%\.local\bin\uv.exe" (
        set "UV_CMD=%USERPROFILE%\.local\bin\uv.exe"
    )
)

if not defined UV_CMD (
    echo uv not found. Installing uv...
    powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if exist "%USERPROFILE%\.local\bin\uv.exe" (
        set "UV_CMD=%USERPROFILE%\.local\bin\uv.exe"
    )
)

if not defined UV_CMD (
    echo [ERROR] uv installation failed or uv.exe not found.
    pause
    exit /b 1
)

echo Using uv:
"%UV_CMD%" --version

REM Add uv path to current session
set "PATH=%USERPROFILE%\.local\bin;%PATH%"

REM Persist uv path into user PATH if missing
powershell -NoProfile -ExecutionPolicy ByPass -Command "$u='%USERPROFILE%\.local\bin'; $p=[Environment]::GetEnvironmentVariable('Path','User'); if (($p -split ';') -notcontains $u) { [Environment]::SetEnvironmentVariable('Path', ($p + ';' + $u), 'User') }" >nul 2>&1

echo [OK] uv ready.
echo.

REM ------------------------------------------------------------
REM 6. uv sync
REM ------------------------------------------------------------
echo [6/7] Synchronizing Python environment...
cd /d "%FURP_DIR%"
"%UV_CMD%" sync
if errorlevel 1 (
    echo [ERROR] uv sync failed.
    pause
    exit /b 1
)
echo [OK] uv sync complete.
echo.

REM ------------------------------------------------------------
REM 7. Smoke test
REM ------------------------------------------------------------
echo [7/7] Running environment checks...

echo.
echo Python / Torch / CUDA check:
"%UV_CMD%" run python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
if errorlevel 1 (
    echo [WARN] Python CUDA check failed.
)

echo.
echo Running zero-agent smoke test...
"%UV_CMD%" run python src\hoppertrex_mjlab\scripts\zero_agent.py
if errorlevel 1 (
    echo [WARN] zero_agent smoke test returned nonzero.
    echo        If this is GPU-related, check nvidia-smi and CUDA driver.
) else (
    echo [OK] zero_agent smoke test passed.
)

echo.
echo ===================================================
echo [SUCCESS] Setup complete.
echo ===================================================
echo.
echo Project directory:
echo   %FURP_DIR%
echo.
echo mjlab directory:
echo   %MJLAB_DIR%
echo.
echo Recommended commands:
echo.
echo 1. Open project:
echo   cd /d %FURP_DIR%
echo.
echo 2. Small smoke train:
echo   uv run python src\hoppertrex_mjlab\scripts\rsl_rl\train.py --env.scene.num-envs 16 --agent.max-iterations 10 --agent.run-name smoke_test
echo.
echo 3. RTX 2080 formal-ish test:
echo   uv run python src\hoppertrex_mjlab\scripts\rsl_rl\train.py --env.scene.num-envs 256 --agent.max-iterations 500 --agent.save-interval 50 --agent.run-name standing_test
echo.
echo 4. Monitor GPU:
echo   nvidia-smi -l 1
echo.
echo Notes:
echo   - T400 / 2GB GPU: only use very small tests, e.g. num-envs 8 or 16.
echo   - RTX 2080 / 8GB GPU: suitable for num-envs 256 training.
echo ===================================================
pause