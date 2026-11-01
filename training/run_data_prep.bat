@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0\.."

REM Determine Python interpreter: use root .venv if present, otherwise system py/python
set "PY_EXE="
if exist ".venv\Scripts\python.exe" (
    set "PY_EXE=.venv\Scripts\python.exe"
) else (
    py -3.12 --version >nul 2>&1
    if not errorlevel 1 (
        set "PY_EXE=py -3.12"
    ) else (
        set "PY_EXE=python"
    )
)

echo [*] Running Helix DPO Dataset Preparation Pipeline...
%PY_EXE% training\prepare_training_data.py %*
set "EXITCODE=%errorlevel%"

if "%EXITCODE%"=="0" (
    echo.
    echo [+] Dataset preparation complete.
    echo [+] Merged dataset: training\data\merged_dpo_dataset.jsonl
    echo [+] Statistics:     training\data\merged_dpo_dataset.stats.json
) else (
    echo.
    echo [!] Dataset preparation exited with error code %EXITCODE%.
)

exit /b %EXITCODE%
