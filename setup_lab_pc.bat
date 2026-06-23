@echo off
echo ===================================================
echo [MjLab Lab PC Environment Auto Setup]
echo ===================================================

:: 1. Install uv in user space (no admin required)
echo [1/2] Checking and installing uv...
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo uv not found, installing...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
) else (
    echo uv is already installed.
)

:: Add uv to PATH for the current session
set "PATH=%USERPROFILE%\.local\bin;%PATH%"

:: 2. Sync dependencies
echo [2/2] Synchronizing virtual environment using uv...
uv sync

echo ===================================================
echo [SUCCESS] Setup complete! Virtual environment (.venv) is ready.
echo To run your training code:
echo   uv run python src/hoppertrex_mjlab/scripts/rsl_rl/train.py
echo ===================================================
pause
