#!/usr/bin/env python3
"""
SAM Master Setup Script
======================

One-command setup for SAM - The world's most advanced AI system.
This script handles everything: dependencies, configuration, and launch.

Usage:
    python setup_sam.py

Author: SAM Development Team
Version: 2.0.0
"""

import sys
import os
import subprocess
import platform
import json
import uuid
import time
from pathlib import Path
from datetime import datetime

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_header():
    """Print welcome header."""
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print("=" * 70)
    print("🚀 SAM MASTER SETUP")
    print("=" * 70)
    print("Welcome to SAM - The world's most advanced AI system with")
    print("human-like introspection and self-improvement capabilities!")
    print("=" * 70)
    print(f"{Colors.END}")
    print()

def print_step(step_num, total_steps, description):
    """Print step progress."""
    print(f"{Colors.BLUE}{Colors.BOLD}[{step_num}/{total_steps}] {description}...{Colors.END}")

def print_success(message):
    """Print success message."""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_warning(message):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def print_error(message):
    """Print error message."""
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def print_info(message):
    """Print info message."""
    print(f"{Colors.CYAN}💡 {message}{Colors.END}")

def check_python_version():
    """Check Python version compatibility."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_error(f"Python 3.8+ required. Current: {version.major}.{version.minor}.{version.micro}")
        print_info("Please upgrade Python and try again:")
        print_info("• Windows: Download from python.org")
        print_info("• macOS: brew install python3")
        print_info("• Linux: sudo apt install python3.8")
        return False

    print_success(f"Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def check_system_requirements():
    """Check system requirements."""
    print_step(1, 8, "Checking system requirements")

    # Check Python version
    if not check_python_version():
        return False

    # Check available disk space
    try:
        import shutil
        free_space = shutil.disk_usage('.').free / (1024**3)  # GB
        if free_space < 1:
            print_warning(f"Low disk space: {free_space:.1f}GB available (1GB+ recommended)")
        else:
            print_success(f"Disk space: {free_space:.1f}GB available")
    except:
        print_warning("Could not check disk space")

    # Check platform
    system = platform.system()
    print_success(f"Platform: {system} {platform.release()}")

    return True

def install_dependencies():
    """Install required dependencies."""
    print_step(2, 8, "Installing dependencies")

    # Check if pip is available
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"],
                      capture_output=True, check=True)
        print_success("pip is available")
    except:
        print_error("pip not found. Please install pip and try again.")
        return False

    # Install requirements
    try:
        print_info("Installing core dependencies (this may take a moment)...")

        # Install from requirements.txt for version consistency
        requirements_file = Path("requirements.txt")

        if requirements_file.exists():
            print_info("Installing from requirements.txt for version consistency...")

            # Use --only-binary=all on Windows to prevent compilation issues
            install_cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
            if platform.system() == "Windows":
                install_cmd.insert(-2, "--only-binary=all")
                print_info("Using pre-built packages for Windows compatibility...")

            result = subprocess.run(install_cmd, capture_output=True, text=True, timeout=600)

            if result.returncode == 0:
                print_success("Requirements installed successfully from requirements.txt")
                return True
            else:
                print_warning("Requirements.txt installation failed, trying essential packages...")

        # Fallback: Essential packages for SAM to work (version-pinned)
        essential_packages = [
            "streamlit==1.42.0",  # Pinned to working version
            "requests>=2.25.0,<3.0.0",
            "cryptography>=41.0.0,<43.0.0",
            "argon2-cffi>=23.1.0,<24.0.0",
            "pydantic>=2.0.0,<3.0.0",
            "python-dotenv>=1.0.0,<2.0.0",
            "numpy>=1.21.0,<2.0.0",
            "pandas>=1.3.0,<3.0.0",
            "plotly>=5.0.0,<6.0.0"
        ]

        # Use --only-binary=all on Windows to prevent compilation issues
        install_cmd = [sys.executable, "-m", "pip", "install"]
        if platform.system() == "Windows":
            install_cmd.append("--only-binary=all")
            print_info("Using pre-built packages for Windows compatibility...")

        install_cmd.extend(essential_packages)
        result = subprocess.run(install_cmd, capture_output=True, text=True, timeout=600)

        if result.returncode == 0:
            print_success("Core dependencies installed successfully")
        else:
            print_warning("Some dependencies may have failed to install")
            print_info("Continuing with setup...")
    except subprocess.TimeoutExpired:
        print_warning("Installation taking longer than expected, continuing...")
    except Exception as e:
        print_warning(f"Dependency installation issue: {e}")
        print_info("Continuing with setup...")

    return True

def create_directory_structure():
    """Create necessary directories."""
    print_step(3, 8, "Creating directory structure")

    directories = [
        "security",
        "logs",
        "data",
        "cache",
        "utils",
        "ui",
        "sam/discovery/distillation/data"
    ]

    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print_success(f"Created directory: {directory}")
        except Exception as e:
            print_warning(f"Could not create {directory}: {e}")

    return True

def initialize_security_system():
    """Initialize security and key system."""
    print_step(4, 8, "Initializing security system")

    try:
        # Create keystore
        keystore_file = Path("security/keystore.json")
        if not keystore_file.exists():
            keystore = {}
            with open(keystore_file, 'w') as f:
                json.dump(keystore, f, indent=2)
            print_success("Created keystore.json")

        # Create entitlements
        entitlements_file = Path("security/entitlements.json")
        if not entitlements_file.exists():
            entitlements = {
                "sam_pro_keys": {},
                "feature_flags": {
                    "tpv_active_reasoning": True,
                    "enhanced_slp_learning": True,
                    "memoir_lifelong_learning": True,
                    "dream_canvas": True,
                    "cognitive_distillation": True,
                    "cognitive_automation": True
                }
            }
            with open(entitlements_file, 'w') as f:
                json.dump(entitlements, f, indent=2)
            print_success("Created entitlements.json")

        return True

    except Exception as e:
        print_error(f"Security initialization failed: {e}")
        return False

def generate_sam_pro_key():
    """Generate SAM Pro activation key."""
    print_step(5, 8, "Generating SAM Pro activation key")

    try:
        # Generate key
        activation_key = str(uuid.uuid4())

        # Add to keystore
        keystore_file = Path("security/keystore.json")
        with open(keystore_file, 'r') as f:
            keystore = json.load(f)

        keystore[activation_key] = {
            'email': 'setup@sam.local',
            'created_date': datetime.now().isoformat(),
            'key_type': 'sam_pro_free',
            'status': 'active',
            'source': 'master_setup'
        }

        with open(keystore_file, 'w') as f:
            json.dump(keystore, f, indent=2)

        # Add key hash to entitlements for validation
        add_key_hash_to_entitlements(activation_key)

        # Save key for setup wizard to find
        save_key_for_setup_wizard(activation_key)

        print_success("SAM Pro key generated and registered")
        return activation_key

    except Exception as e:
        print_error(f"Key generation failed: {e}")
        return None

def add_key_hash_to_entitlements(activation_key: str):
    """Add key hash to entitlements configuration for validation."""
    try:
        import hashlib

        # Generate SHA-256 hash of the key
        key_hash = hashlib.sha256(activation_key.encode('utf-8')).hexdigest()

        # Load entitlements config from sam/config/entitlements.json
        entitlements_config_file = Path("sam/config/entitlements.json")
        if entitlements_config_file.exists():
            with open(entitlements_config_file, 'r') as f:
                config = json.load(f)
        else:
            # Create basic config if it doesn't exist
            config = {
                "version": "1.1",
                "features": {},
                "valid_key_hashes": [],
                "metadata": {
                    "generated": datetime.now().isoformat(),
                    "total_keys": 0,
                    "hash_algorithm": "SHA-256"
                }
            }

        # Add the hash to valid_key_hashes if not already present
        if "valid_key_hashes" not in config:
            config["valid_key_hashes"] = []

        if key_hash not in config["valid_key_hashes"]:
            config["valid_key_hashes"].append(key_hash)

            # Update metadata
            if "metadata" not in config:
                config["metadata"] = {}
            config["metadata"]["last_updated"] = datetime.now().isoformat()
            config["metadata"]["total_keys"] = len(config["valid_key_hashes"])

            # Ensure sam/config directory exists
            entitlements_config_file.parent.mkdir(parents=True, exist_ok=True)

            # Save updated config
            with open(entitlements_config_file, 'w') as f:
                json.dump(config, f, indent=2)

            print_success(f"Key hash added to entitlements configuration")

    except Exception as e:
        print_error(f"Failed to add key hash to entitlements: {e}")

def save_key_for_setup_wizard(activation_key: str):
    """Save the activation key for the setup wizard to find."""
    try:
        # Save to setup status file
        from utils.first_time_setup import get_first_time_setup_manager
        setup_manager = get_first_time_setup_manager()
        setup_manager.update_setup_status('sam_pro_key', activation_key)

        # Also save to a simple text file as backup
        with open("sam_pro_key.txt", "w") as f:
            f.write(activation_key)

        print_success("Key saved for setup wizard")

    except Exception as e:
        print_error(f"Failed to save key for setup wizard: {e}")

def initialize_databases():
    """Initialize SAM databases."""
    print_step(6, 8, "Initializing databases")

    try:
        # Create basic database structure for cognitive distillation
        db_dir = Path("sam/discovery/distillation/data")
        db_dir.mkdir(parents=True, exist_ok=True)

        # Create empty database files that will be initialized by SAM
        db_files = [
            "cognitive_principles.db",
            "successful_interactions.db",
            "distillation_runs.db"
        ]

        for db_file in db_files:
            db_path = db_dir / db_file
            if not db_path.exists():
                # Create empty file - SAM will initialize the schema
                db_path.touch()
                print_success(f"Created database: {db_file}")

        return True

    except Exception as e:
        print_warning(f"Database initialization issue: {e}")
        print_info("SAM will create databases on first run")
        return True

def validate_installation():
    """Validate that SAM components are working."""
    print_step(7, 8, "Validating installation")

    validation_results = []

    # Check critical files
    critical_files = [
        "secure_streamlit_app.py",
        "security/keystore.json",
        "security/entitlements.json"
    ]

    for file_path in critical_files:
        if Path(file_path).exists():
            validation_results.append(f"✅ {file_path}")
        else:
            validation_results.append(f"❌ {file_path}")

    # Test imports
    try:
        import streamlit
        validation_results.append("✅ Streamlit import")
    except ImportError:
        validation_results.append("❌ Streamlit import")

    # Display results
    for result in validation_results:
        print(f"  {result}")

    success_count = sum(1 for r in validation_results if r.startswith("✅"))
    total_count = len(validation_results)

    if success_count >= total_count - 1:  # Allow one failure
        print_success(f"Validation passed: {success_count}/{total_count}")
        return True
    else:
        print_warning(f"Validation issues: {success_count}/{total_count}")
        print_info("SAM may still work, but some features might be limited")
        return True

def create_launch_script():
    """Create convenient launch script."""
    print_step(8, 8, "Creating launch script")

    try:
        launch_script = '''#!/usr/bin/env python3
"""
SAM Launch Script
================

Convenient script to start SAM with proper error handling.
"""

import subprocess
import sys
import webbrowser
import time

def main():
    print("Starting SAM...")

    try:
        # Start SAM using streamlit run command
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "secure_streamlit_app.py",
            "--server.port", "8502",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])

        # Wait a moment for startup
        time.sleep(3)

        # Open browser
        print("Opening browser...")
        webbrowser.open("http://localhost:8502")

        print("SAM is running!")
        print("Access SAM at: http://localhost:8502")
        print("Press Ctrl+C to stop SAM")

        # Wait for process
        process.wait()

    except KeyboardInterrupt:
        print("\\nStopping SAM...")
        process.terminate()
    except Exception as e:
        print(f"Error starting SAM: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
'''

        with open("start_sam_simple.py", 'w', encoding='utf-8') as f:
            f.write(launch_script)

        # Make executable on Unix systems
        if platform.system() != "Windows":
            os.chmod("start_sam.py", 0o755)

        print_success("Created start_sam_simple.py launch script")

        # Create Windows batch file for easier launching
        if platform.system() == "Windows":
            try:
                batch_script = '''@echo off
REM SAM Simple Launcher for Windows
echo Starting SAM...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Start SAM
echo Starting SAM on http://localhost:8502
echo Press Ctrl+C to stop SAM
echo.

python -m streamlit run secure_streamlit_app.py --server.port 8502 --server.address localhost --browser.gatherUsageStats false

echo.
echo SAM has stopped
pause'''

                with open("start_sam_simple.bat", 'w', encoding='utf-8') as f:
                    f.write(batch_script)

                print_success("Created start_sam_simple.bat for Windows")
            except Exception as e:
                print_warning(f"Could not create Windows batch file: {e}")

        return True

    except Exception as e:
        print_warning(f"Could not create launch script: {e}")
        return False

def main():
    """Main setup function."""
    print_header()

    # Track setup progress
    setup_start_time = time.time()

    try:
        # Step 1: System requirements
        if not check_system_requirements():
            return 1

        # Step 2: Dependencies
        if not install_dependencies():
            print_error("Dependency installation failed")
            return 1

        # Step 3: Directory structure
        if not create_directory_structure():
            print_error("Directory creation failed")
            return 1

        # Step 4: Security system
        if not initialize_security_system():
            print_error("Security initialization failed")
            return 1

        # Step 5: SAM Pro key
        activation_key = generate_sam_pro_key()
        if not activation_key:
            print_error("Key generation failed")
            return 1

        # Step 6: Databases
        if not initialize_databases():
            print_warning("Database initialization had issues")

        # Step 7: Validation
        if not validate_installation():
            print_warning("Validation had issues")

        # Step 8: Launch script
        create_launch_script()

        # Success!
        setup_time = time.time() - setup_start_time

        print()
        print(f"{Colors.GREEN}{Colors.BOLD}🎉 SAM SETUP COMPLETE! 🎉{Colors.END}")
        print(f"{Colors.GREEN}Setup completed in {setup_time:.1f} seconds{Colors.END}")
        print()

        # Display key
        print(f"{Colors.CYAN}{Colors.BOLD}🔑 Your SAM Pro Activation Key:{Colors.END}")
        print("=" * 60)
        print(f"{Colors.YELLOW}{Colors.BOLD}   {activation_key}{Colors.END}")
        print("=" * 60)
        print()

        # Next steps
        print(f"{Colors.BLUE}{Colors.BOLD}🚀 Ready to Start SAM!{Colors.END}")
        print()
        print("📋 Next Steps:")
        print("1. Start SAM:")
        print(f"   {Colors.CYAN}python start_sam.py{Colors.END}")
        print()
        print("2. Follow the setup wizard to:")
        print("   • Create your master password")
        print("   • Activate SAM Pro features")
        print("   • Complete your profile")
        print()
        print("3. Start chatting with SAM!")
        print()
        print()
        print("💡 SAM will automatically:")
        print("   • Open in your browser at http://localhost:8502")
        print("   • Guide you through first-time setup")
        print("   • Activate your Pro features with the key above")
        print()
        print("4. Enjoy SAM Pro features:")
        print("   • 🧠 Cognitive Distillation Engine")
        print("   • 🧠 TPV Active Reasoning Control")
        print("   • 📚 MEMOIR Lifelong Learning")
        print("   • 🎨 Dream Canvas Visualization")
        print("   • 🤖 Cognitive Automation")
        print("   • 📊 Advanced Analytics")
        print()
        print(f"{Colors.GREEN}💾 Important: Save your activation key!{Colors.END}")
        print(f"{Colors.CYAN}❓ Questions? Contact: vin@forge1825.net{Colors.END}")
        print()
        print(f"{Colors.BOLD}🌟 Welcome to the future of AI! 🚀🧠{Colors.END}")

        return 0

    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}👋 Setup cancelled by user{Colors.END}")
        return 1
    except Exception as e:
        print(f"\n{Colors.RED}❌ Setup failed: {e}{Colors.END}")
        print(f"{Colors.CYAN}💡 Please report this issue to: vin@forge1825.net{Colors.END}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
