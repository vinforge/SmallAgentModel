#!/usr/bin/env python3
"""
Docker Setup Validation Script
=============================

Validates that all Docker configuration files are properly set up
and ready for SAM deployment.

Usage:
    python docker/validate_docker_setup.py

Author: SAM Development Team
"""

import os
import json
import sys
from pathlib import Path

def print_status(message, status="info"):
    """Print colored status messages."""
    colors = {
        "success": "\033[92m✅",
        "error": "\033[91m❌", 
        "warning": "\033[93m⚠️",
        "info": "\033[94mℹ️"
    }
    reset = "\033[0m"
    print(f"{colors.get(status, colors['info'])} {message}{reset}")

def validate_file_exists(file_path, description):
    """Validate that a file exists."""
    if Path(file_path).exists():
        print_status(f"{description}: Found", "success")
        return True
    else:
        print_status(f"{description}: Missing - {file_path}", "error")
        return False

def validate_json_file(file_path, description):
    """Validate that a JSON file exists and is valid."""
    if not Path(file_path).exists():
        print_status(f"{description}: Missing - {file_path}", "error")
        return False
    
    try:
        with open(file_path, 'r') as f:
            json.load(f)
        print_status(f"{description}: Valid JSON", "success")
        return True
    except json.JSONDecodeError as e:
        print_status(f"{description}: Invalid JSON - {e}", "error")
        return False

def validate_executable(file_path, description):
    """Validate that a file exists and is executable."""
    if not Path(file_path).exists():
        print_status(f"{description}: Missing - {file_path}", "error")
        return False
    
    if os.access(file_path, os.X_OK):
        print_status(f"{description}: Executable", "success")
        return True
    else:
        print_status(f"{description}: Not executable", "warning")
        return False

def validate_docker_files():
    """Validate all Docker-related files."""
    print_status("🐳 Validating Docker Configuration Files", "info")
    print("=" * 50)
    
    all_valid = True
    
    # Core Docker files
    files_to_check = [
        ("Dockerfile", "Main Dockerfile"),
        ("docker-compose.yml", "Docker Compose configuration"),
        (".dockerignore", "Docker ignore file"),
    ]
    
    for file_path, description in files_to_check:
        if not validate_file_exists(file_path, description):
            all_valid = False
    
    # Docker directory files
    docker_files = [
        ("docker/docker_entrypoint.sh", "Docker entrypoint script"),
        ("docker/manage_sam.sh", "SAM management script"),
        ("docker/nginx.conf", "Nginx configuration"),
    ]
    
    for file_path, description in docker_files:
        if not validate_file_exists(file_path, description):
            all_valid = False
    
    # JSON configuration files
    json_files = [
        ("docker/sam_docker_config.json", "SAM Docker configuration"),
    ]
    
    for file_path, description in json_files:
        if not validate_json_file(file_path, description):
            all_valid = False
    
    # Executable files
    executable_files = [
        ("docker/docker_entrypoint.sh", "Docker entrypoint script"),
        ("docker/manage_sam.sh", "SAM management script"),
    ]
    
    for file_path, description in executable_files:
        validate_executable(file_path, description)
    
    return all_valid

def validate_requirements():
    """Validate requirements.txt exists."""
    print_status("\n📦 Validating Python Requirements", "info")
    print("=" * 50)
    
    if validate_file_exists("requirements.txt", "Python requirements file"):
        # Check if requirements file has content
        try:
            with open("requirements.txt", 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
                if lines:
                    print_status(f"Found {len(lines)} package requirements", "success")
                    return True
                else:
                    print_status("Requirements file is empty", "warning")
                    return False
        except Exception as e:
            print_status(f"Error reading requirements.txt: {e}", "error")
            return False
    return False

def validate_directory_structure():
    """Validate expected directory structure."""
    print_status("\n📁 Validating Directory Structure", "info")
    print("=" * 50)
    
    required_dirs = [
        "ui",
        "core", 
        "skills",
        "memory",
        "security",
        "docker"
    ]
    
    all_valid = True
    for dir_name in required_dirs:
        if Path(dir_name).is_dir():
            print_status(f"Directory '{dir_name}': Found", "success")
        else:
            print_status(f"Directory '{dir_name}': Missing", "warning")
            # Not marking as invalid since some directories might be optional
    
    return all_valid

def validate_main_app_files():
    """Validate main application files exist."""
    print_status("\n🚀 Validating Main Application Files", "info")
    print("=" * 50)
    
    main_files = [
        ("secure_streamlit_app.py", "Main Streamlit application"),
        ("start_sam.py", "SAM startup script"),
        ("setup_sam.py", "SAM setup script"),
    ]
    
    all_valid = True
    for file_path, description in main_files:
        if not validate_file_exists(file_path, description):
            all_valid = False
    
    return all_valid

def check_docker_availability():
    """Check if Docker and Docker Compose are available."""
    print_status("\n🔧 Checking Docker Availability", "info")
    print("=" * 50)

    # Check Docker (Windows-compatible)
    import platform
    if platform.system() == "Windows":
        docker_available = os.system("docker --version >nul 2>&1") == 0
        compose_available = os.system("docker compose version >nul 2>&1") == 0
    else:
        docker_available = os.system("docker --version > /dev/null 2>&1") == 0
        compose_available = os.system("docker compose version > /dev/null 2>&1") == 0

    if docker_available:
        print_status("Docker: Available", "success")
    else:
        print_status("Docker: Not available or not in PATH", "error")

    # Check Docker Compose (modern syntax)
    if compose_available:
        print_status("Docker Compose: Available", "success")
    else:
        print_status("Docker Compose: Not available or not in PATH", "error")

    return docker_available and compose_available

def main():
    """Run all validation checks."""
    print_status("🐳 SAM Docker Setup Validation", "info")
    print("=" * 60)
    
    # Run all validation checks
    checks = [
        ("Docker Files", validate_docker_files),
        ("Requirements", validate_requirements),
        ("Directory Structure", validate_directory_structure),
        ("Main App Files", validate_main_app_files),
        ("Docker Availability", check_docker_availability),
    ]
    
    all_passed = True
    results = {}
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results[check_name] = result
            if not result:
                all_passed = False
        except Exception as e:
            print_status(f"Error during {check_name} validation: {e}", "error")
            results[check_name] = False
            all_passed = False
    
    # Summary
    print_status("\n📊 Validation Summary", "info")
    print("=" * 60)
    
    for check_name, result in results.items():
        status = "success" if result else "error"
        print_status(f"{check_name}: {'PASSED' if result else 'FAILED'}", status)
    
    print("\n" + "=" * 60)
    if all_passed:
        print_status("🎉 All validations passed! SAM is ready for Docker deployment.", "success")
        print_status("\nNext steps:", "info")
        print("   1. Build the Docker image: ./docker/manage_sam.sh build")
        print("   2. Start SAM services: ./docker/manage_sam.sh start")
        print("   3. Access SAM at: http://localhost:8502")
        return True
    else:
        print_status("❌ Some validations failed. Please fix the issues above before deploying.", "error")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
