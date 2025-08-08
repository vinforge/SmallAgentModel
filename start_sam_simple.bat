@echo off
REM SAM Main Launcher for Windows
REM This batch file starts SAM with first-time setup detection

echo ============================================================
echo 🚀 SAM - Starting Up
echo ============================================================
echo Welcome to SAM - The world's most advanced AI system
echo with human-like introspection and self-improvement!
echo ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Start SAM using the main launcher
echo Starting SAM with setup detection...
echo.

python start_sam.py

echo.
echo SAM has stopped
pause
