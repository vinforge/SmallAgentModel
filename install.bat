@echo off
REM SAM Secure AI Assistant - Windows Installer
REM Usage: Run this batch file or download and run from GitHub

setlocal enabledelayedexpansion

REM Print banner
echo.
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                    🧠 SAM - SECURE AI ASSISTANT 🔒                          ║
echo ║                         Windows Installer Script                            ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.

REM Check Python installation
echo 🐍 Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found!
    echo 📥 Please install Python 3.8+ from https://python.org/downloads/
    echo    Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✅ Found Python %PYTHON_VERSION%

REM Check if version is compatible (basic check)
python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python 3.8+ required, found %PYTHON_VERSION%
    pause
    exit /b 1
)
echo ✅ Python version compatible

REM Check if git is available
echo 📥 Checking Git installation...
git --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Git found
    set USE_GIT=true
) else (
    echo ⚠️  Git not found, will download ZIP instead
    set USE_GIT=false
)

REM Set installation directory
set INSTALL_DIR=%USERPROFILE%\SAM
echo 📁 Installation directory: %INSTALL_DIR%

REM Backup existing installation
if exist "%INSTALL_DIR%" (
    echo ⚠️  Directory exists. Creating backup...
    for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (
        for /f "tokens=1-2 delims=: " %%d in ('time /t') do (
            set BACKUP_DIR=%INSTALL_DIR%.backup.%%c%%a%%b_%%d%%e
        )
    )
    move "%INSTALL_DIR%" "!BACKUP_DIR!" >nul 2>&1
)

REM Download SAM
echo 📥 Downloading SAM...
if "%USE_GIT%"=="true" (
    git clone https://github.com/your-repo/SAM.git "%INSTALL_DIR%"
    if %errorlevel% neq 0 (
        echo ❌ Git clone failed
        pause
        exit /b 1
    )
) else (
    REM Download ZIP using PowerShell
    powershell -Command "& {Invoke-WebRequest -Uri 'https://github.com/your-repo/SAM/archive/main.zip' -OutFile 'sam.zip'}"
    if %errorlevel% neq 0 (
        echo ❌ Download failed
        pause
        exit /b 1
    )
    
    REM Extract ZIP using PowerShell
    powershell -Command "& {Expand-Archive -Path 'sam.zip' -DestinationPath '.' -Force}"
    move SAM-main "%INSTALL_DIR%" >nul 2>&1
    del sam.zip >nul 2>&1
)

echo ✅ SAM downloaded successfully

REM Change to installation directory
cd /d "%INSTALL_DIR%"

REM Install dependencies
echo 📦 Installing dependencies...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo ⚠️  Pip upgrade failed, continuing...
)

python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ Dependency installation failed
    echo 🔧 Try running: pip install -r requirements.txt
    pause
    exit /b 1
)

echo ✅ Dependencies installed

REM Run setup
echo ⚙️  Running setup...
python install_sam.py
if %errorlevel% neq 0 (
    echo ⚠️  Setup script had issues, but installation may still work
)

REM Create desktop shortcut (optional)
echo 🖥️  Creating shortcuts...
set SHORTCUT_PATH=%USERPROFILE%\Desktop\SAM.bat
echo @echo off > "%SHORTCUT_PATH%"
echo cd /d "%INSTALL_DIR%" >> "%SHORTCUT_PATH%"
echo python start_sam_secure.py --mode full >> "%SHORTCUT_PATH%"
echo Created desktop shortcut: SAM.bat

REM Final message
echo.
echo 🎉 SAM installation completed successfully!
echo.
echo 📍 Installation location: %INSTALL_DIR%
echo 🚀 To launch SAM:
echo    Double-click SAM.bat on your desktop
echo    OR
echo    cd "%INSTALL_DIR%"
echo    python start_sam_secure.py --mode full
echo.
echo 📖 Documentation: README_SECURE_INSTALLATION.md
echo 🔒 Security: Enterprise-grade encryption enabled
echo 🏠 Privacy: 100%% local processing
echo.
echo Press any key to launch SAM now...
pause >nul

REM Launch SAM
python start_sam_secure.py --mode full

echo.
echo 👋 Thanks for using SAM!
pause
