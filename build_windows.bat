@echo off
REM ============================================================
REM  AstroPipeline - Windows Build Script
REM  Output: dist\AstroPipeline.exe  (single executable)
REM
REM  Prerequisites:
REM    1. Python 3.10+ installed (python.org) with "Add to PATH" checked
REM    2. Install dependencies: pip install -r requirements.txt
REM    3. Run this script from the project root
REM ============================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

set APP_NAME=AstroPipeline
set DIST_DIR=dist
set BUILD_DIR=build

echo ======================================================
echo   AstroPipeline Windows Build
echo ======================================================

REM -- 1. Check Python -----------------------------------------------
echo.
echo [1/4] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH.
    echo Install Python 3.10+ from python.org and check "Add to PATH".
    pause
    exit /b 1
)
python --version

REM -- 2. Check build dependencies -----------------------------------
echo.
echo [2/4] Checking build dependencies...

python -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo   Installing PyInstaller...
    pip install pyinstaller pyinstaller-hooks-contrib
) else (
    echo   PyInstaller OK
)

python -c "import _pyinstaller_hooks_contrib" >nul 2>&1
if errorlevel 1 (
    echo   Installing pyinstaller-hooks-contrib...
    pip install pyinstaller-hooks-contrib
)

REM -- 3. Clean previous build ----------------------------------------
echo.
echo [3/4] Cleaning previous build...
if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
if exist "%DIST_DIR%"  rmdir /s /q "%DIST_DIR%"
echo   Done

REM -- 4. Build -------------------------------------------------------
echo.
echo [4/4] Starting PyInstaller build...
echo   (First build may take several minutes for PySide6 collection)
echo.

python -m PyInstaller --clean astro_pipeline.spec
if errorlevel 1 (
    echo.
    echo ERROR: Build failed. See messages above.
    pause
    exit /b 1
)

REM -- Result check ---------------------------------------------------
set EXE_PATH=%DIST_DIR%\%APP_NAME%.exe
if exist "%EXE_PATH%" (
    echo.
    echo ======================================================
    echo   Build successful!
    echo   Output: %EXE_PATH%
    echo ======================================================
    echo.
    echo Run: double-click %EXE_PATH%
    echo.
    echo Note: First launch extracts files to %%TEMP%% and takes
    echo       5-15 seconds. Subsequent launches are fast.
    echo.
) else (
    echo.
    echo ERROR: %EXE_PATH% was not created.
    echo Check the PyInstaller output above for details.
    pause
    exit /b 1
)

pause
