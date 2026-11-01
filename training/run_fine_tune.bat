@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0\.."

REM 1. Check for prerequisite merged dataset file
set "DATASET_FILE=training\data\merged_dpo_dataset.jsonl"
if not exist "!DATASET_FILE!" (
    echo [ERROR] Required dataset file '!DATASET_FILE!' is missing.
    echo Please run 'training\run_data_prep.bat' first to download and prepare training data.
    exit /b 1
)

REM 2. Hardware advisory
echo ======================================================================
echo  Helix Unsloth DPO Fine-Tuning Launcher
echo ======================================================================
echo [ADVISORY] Training requires >= 11 GB free VRAM on NVIDIA RTX 5070 (12 GB).
echo Please close Ollama, Blender, Unreal Engine, and browser hardware acceleration
echo to prevent GPU out-of-memory (OOM) errors.
echo ======================================================================

REM 3. Virtual environment setup for training (.venv-train)
if not exist ".venv-train\Scripts\python.exe" (
    echo [*] Creating dedicated training virtual environment .venv-train...
    py -3.12 -m venv .venv-train 2>nul
    if errorlevel 1 (
        python -m venv .venv-train
    )
)

REM 4. Dependency caching
set "STAMP=.venv-train\.pip_train_stamp"
set "REQ=training\requirements-train.txt"
set "NEED_INSTALL=0"

if not exist "!STAMP!" (
    set "NEED_INSTALL=1"
) else (
    REM reinstall only when the requirements file is NEWER than the stamp written after the last install
    powershell -NoProfile -Command "if ((Get-Item -LiteralPath '!REQ!').LastWriteTimeUtc -gt (Get-Item -LiteralPath '!STAMP!').LastWriteTimeUtc) { exit 1 } else { exit 0 }"
    if errorlevel 1 set "NEED_INSTALL=1"
)

if "!NEED_INSTALL!"=="1" (
    echo [*] Installing training dependencies from !REQ!...
    ".venv-train\Scripts\python.exe" -m pip install -r "!REQ!"
    if errorlevel 1 (
        echo [ERROR] Failed to install training dependencies.
        exit /b !errorlevel!
    )
    type nul > "!STAMP!"
)

REM 5. Run trainer
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONPATH=%~dp0;%~dp0\training"

".venv-train\Scripts\python.exe" training\helix_unsloth_trainer.py %*
set "EXITCODE=%errorlevel%"

exit /b %EXITCODE%
