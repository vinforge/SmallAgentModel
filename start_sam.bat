@echo off
REM SAM Community Edition Beta - Launcher Script for Windows
REM This script starts SAM with proper error handling and logging

setlocal enabledelayedexpansion

REM Colors (limited in Windows CMD)
set "INFO=[SAM]"
set "SUCCESS=[SAM] SUCCESS:"
set "WARNING=[SAM] WARNING:"
set "ERROR=[SAM] ERROR:"

echo %INFO% Starting SAM Community Edition Beta...
echo ==================================

REM Check if we're in the right directory
if not exist "start_sam.py" (
    echo %ERROR% start_sam.py not found. Please run this script from the SAM directory.
    pause
    exit /b 1
)

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    python3 --version >nul 2>&1
    if errorlevel 1 (
        echo %ERROR% Python not found. Please install Python 3.8+ and try again.
        echo %ERROR% Download from: https://www.python.org/downloads/
        pause
        exit /b 1
    ) else (
        set PYTHON_CMD=python3
    )
) else (
    set PYTHON_CMD=python
)

REM Get Python version
for /f "tokens=2" %%i in ('%PYTHON_CMD% --version 2^>^&1') do set PYTHON_VERSION=%%i
echo %INFO% Using Python %PYTHON_VERSION%

REM Check if Ollama is installed
ollama --version >nul 2>&1
if errorlevel 1 (
    echo %WARNING% Ollama not found. SAM requires Ollama for AI functionality.
    echo %WARNING% Please install Ollama from: https://ollama.ai/download
    set /p "continue=Continue anyway? (y/N): "
    if /i not "!continue!"=="y" (
        exit /b 1
    )
) else (
    echo %SUCCESS% Ollama found
    
    REM Check if Ollama service is running
    ollama list >nul 2>&1
    if errorlevel 1 (
        echo %WARNING% Ollama service doesn't seem to be running.
        echo %INFO% Please start Ollama manually in another command prompt:
        echo %INFO%   ollama serve
        pause
    )
    
    REM Check if the required model is available
    ollama list | findstr "DeepSeek-R1" >nul 2>&1
    if errorlevel 1 (
        echo %WARNING% Required AI model not found.
        echo %INFO% Downloading model (this may take several minutes)...
        ollama pull hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M
        if errorlevel 1 (
            echo %ERROR% Failed to download model. Please run manually:
            echo %ERROR%   ollama pull hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M
            pause
            exit /b 1
        )
    )
)

REM Check if ports are available (basic check)
netstat -an | findstr ":5001" >nul 2>&1
if not errorlevel 1 (
    echo %WARNING% Port 5001 is already in use. SAM's chat interface may not start.
)

netstat -an | findstr ":8501" >nul 2>&1
if not errorlevel 1 (
    echo %WARNING% Port 8501 is already in use. SAM's memory interface may not start.
)

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

REM Check if configuration exists
if not exist "config\sam_config.json" (
    echo %WARNING% Configuration not found. Running installer first...
    %PYTHON_CMD% install.py
    if errorlevel 1 (
        echo %ERROR% Installation failed. Please check the error messages above.
        pause
        exit /b 1
    )
)

REM Start SAM
echo %SUCCESS% Starting SAM...
echo %INFO% Chat Interface will be available at: http://localhost:5001
echo %INFO% Memory Control Center will be available at: http://localhost:8501
echo %INFO% Press Ctrl+C to stop SAM
echo.

REM Run SAM with error handling
%PYTHON_CMD% start_sam.py
if errorlevel 1 (
    echo %ERROR% SAM failed to start. Check logs\sam.log for details.
    pause
    exit /b 1
)

echo %INFO% SAM has stopped.
pause
