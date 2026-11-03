@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

REM 1. Virtual Environment Bootstrap
if not exist ".venv\Scripts\python.exe" (
    echo [*] Creating virtual environment .venv...
    py -3.12 -m venv .venv 2>nul
    if errorlevel 1 (
        python -m venv .venv
    )
)

REM 2. Pip Requirement Timestamp Cache
set "STAMP=.venv\.pip_stamp"
set "REQ=requirements.txt"
set "NEED_INSTALL=0"

if not exist "!STAMP!" (
    set "NEED_INSTALL=1"
) else (
    REM reinstall only when the requirements file is NEWER than the stamp written after the last install
    powershell -NoProfile -Command "if ((Get-Item -LiteralPath '!REQ!').LastWriteTimeUtc -gt (Get-Item -LiteralPath '!STAMP!').LastWriteTimeUtc) { exit 1 } else { exit 0 }"
    if errorlevel 1 set "NEED_INSTALL=1"
)

if "!NEED_INSTALL!"=="1" (
    echo [*] Installing dependencies from !REQ!...
    ".venv\Scripts\python.exe" -m pip install -r "!REQ!"
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies.
        exit /b !errorlevel!
    )
    type nul > "!STAMP!"
)

REM 3. Execution Environment
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONPATH=%~dp0"

".venv\Scripts\python.exe" main.py %*
set "EXITCODE=%errorlevel%"

exit /b %EXITCODE%
