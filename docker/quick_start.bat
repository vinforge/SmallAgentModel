@echo off
REM SAM Docker Quick Start for Windows
REM This script sets up and runs SAM in Docker containers

echo ========================================
echo SAM Docker Quick Start for Windows
echo ========================================
echo.

REM Check if Docker is installed and running
echo [1/6] Checking Docker installation...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not installed or not in PATH
    echo Please install Docker Desktop from: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running
    echo Please start Docker Desktop and try again
    pause
    exit /b 1
)

echo ✅ Docker is installed and running

REM Check if Docker Compose is available
echo [2/6] Checking Docker Compose...
docker compose version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker Compose is not available
    echo Please ensure you have Docker Desktop with Compose support
    pause
    exit /b 1
)

echo ✅ Docker Compose is available

REM Navigate to project root
cd /d "%~dp0\.."

REM Validate Docker setup
echo [3/6] Validating Docker setup...
python docker\validate_docker_setup.py
if %errorlevel% neq 0 (
    echo ERROR: Docker setup validation failed
    echo Please check the error messages above
    pause
    exit /b 1
)

echo ✅ Docker setup validated

REM Build Docker images
echo [4/6] Building SAM Docker images...
echo This may take several minutes on first run...
docker compose build
if %errorlevel% neq 0 (
    echo ERROR: Failed to build Docker images
    pause
    exit /b 1
)

echo ✅ Docker images built successfully

REM Start SAM services
echo [5/6] Starting SAM services...
docker compose up -d
if %errorlevel% neq 0 (
    echo ERROR: Failed to start SAM services
    pause
    exit /b 1
)

echo ✅ SAM services started

REM Wait for services to be ready
echo [6/6] Waiting for services to be ready...
timeout /t 10 /nobreak >nul

REM Check service status
echo.
echo ========================================
echo SAM Service Status
echo ========================================
docker compose ps

echo.
echo ========================================
echo SAM is now running!
echo ========================================
echo.
echo 🌐 Main Interface:     http://localhost:8502
echo 🧠 Memory Center:      http://localhost:8501  
echo 🔧 API Endpoint:       http://localhost:5001
echo.
echo To stop SAM:           docker compose down
echo To view logs:          docker compose logs -f
echo To restart:            docker compose restart
echo.
echo Press any key to open SAM in your browser...
pause >nul

REM Open SAM in default browser
start http://localhost:8502

echo.
echo SAM Docker Quick Start completed successfully!
echo Check the browser window that just opened.
echo.
pause
