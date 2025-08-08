#!/usr/bin/env python3
"""
Linux Dependency Installer for SAM
==================================

Specialized installer for Linux systems (Ubuntu/Debian) that handles
common dependency installation issues.

Usage: python3 install_linux_dependencies.py
"""

import sys
import subprocess
import platform
import os

def print_header():
    """Print installer header."""
    print("=" * 60)
    print("🐧 SAM Linux Dependency Installer")
    print("=" * 60)
    print("Installing required Python packages for SAM on Linux...")
    print()

def check_system():
    """Check if running on Linux."""
    if platform.system() != "Linux":
        print("⚠️  This installer is designed for Linux systems")
        print(f"💡 Detected: {platform.system()}")
        print("💡 Please use the appropriate installer for your platform")
        return False
    
    print(f"✅ Linux system detected: {platform.platform()}")
    return True

def check_python():
    """Check Python version."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (Requires 3.8+)")
        print("💡 Please upgrade Python:")
        print("   sudo apt update")
        print("   sudo apt install python3.9 python3.9-pip")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_system_packages():
    """Install system-level packages if needed."""
    print("\n🔧 Checking system packages...")
    
    # Check if pip is available
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      capture_output=True, check=True)
        print("✅ pip is available")
        return True
    except:
        print("❌ pip not found")
        print("💡 Installing pip...")
        
        try:
            # Try to install pip
            subprocess.run(["sudo", "apt", "update"], check=True)
            subprocess.run(["sudo", "apt", "install", "-y", "python3-pip"], check=True)
            print("✅ pip installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install pip")
            print("💡 Please run manually:")
            print("   sudo apt update")
            print("   sudo apt install python3-pip")
            return False
        except FileNotFoundError:
            print("❌ sudo or apt not found")
            print("💡 Please install pip manually for your Linux distribution")
            return False

def install_system_packages_first():
    """Try to install system packages that might help with Python package installation."""
    print("\n🔧 Installing system development packages...")

    system_packages = [
        "python3-dev",
        "python3-pip",
        "build-essential",
        "libffi-dev",
        "libssl-dev",
        "python3-numpy",
        "python3-pandas"
    ]

    try:
        # Update package list
        print("📦 Updating package list...")
        subprocess.run(["sudo", "apt", "update"], check=True, capture_output=True)

        # Install system packages
        print("📦 Installing system packages...")
        subprocess.run(["sudo", "apt", "install", "-y"] + system_packages,
                      check=True, capture_output=True, timeout=300)

        print("✅ System packages installed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"⚠️  System package installation failed: {e}")
        return False
    except subprocess.TimeoutExpired:
        print("⚠️  System package installation timeout")
        return False
    except FileNotFoundError:
        print("⚠️  sudo or apt not found")
        return False
    except Exception as e:
        print(f"⚠️  System package error: {e}")
        return False

def install_python_packages():
    """Install required Python packages."""
    print("\n📦 Installing Python packages...")

    # Essential packages for SAM (version-pinned for consistency)
    essential_packages = ["streamlit==1.42.0", "numpy>=1.21.0,<2.0.0", "pandas>=1.3.0,<3.0.0", "requests>=2.25.0,<3.0.0"]
    additional_packages = ["cryptography>=41.0.0,<43.0.0", "argon2-cffi>=23.1.0,<24.0.0", "pydantic>=2.0.0,<3.0.0", "python-dotenv>=1.0.0,<2.0.0", "plotly>=5.0.0,<6.0.0"]

    # Try installing essential packages first
    print("🔄 Installing essential packages first...")
    essential_success = install_package_set(essential_packages, "Essential")

    if essential_success:
        print("🔄 Installing additional packages...")
        additional_success = install_package_set(additional_packages, "Additional")
        return essential_success and additional_success
    else:
        return False

def install_package_set(packages, package_type):
    """Install a set of packages with multiple methods."""
    print(f"\n📦 Installing {package_type} packages: {', '.join(packages)}")

    # Try different installation methods
    methods = [
        # Method 1: User installation with upgrade
        {
            "name": f"{package_type} - User installation (--user --upgrade)",
            "cmd": [sys.executable, "-m", "pip", "install", "--user", "--upgrade"] + packages
        },
        # Method 2: User installation
        {
            "name": f"{package_type} - User installation (--user)",
            "cmd": [sys.executable, "-m", "pip", "install", "--user"] + packages
        },
        # Method 3: Standard installation
        {
            "name": f"{package_type} - Standard installation",
            "cmd": [sys.executable, "-m", "pip", "install"] + packages
        },
        # Method 4: pip3 user installation
        {
            "name": f"{package_type} - pip3 user installation",
            "cmd": ["pip3", "install", "--user"] + packages
        }
    ]

    for method in methods:
        print(f"\n🔄 Trying: {method['name']}")
        try:
            result = subprocess.run(
                method["cmd"],
                capture_output=True,
                text=True,
                timeout=180
            )

            if result.returncode == 0:
                print(f"✅ {package_type} packages installed successfully!")
                return True
            else:
                error_msg = result.stderr[:150] if result.stderr else "Unknown error"
                print(f"⚠️  Method failed: {error_msg}...")

        except subprocess.TimeoutExpired:
            print("⚠️  Installation timeout")
        except FileNotFoundError:
            print("⚠️  Command not found")
        except Exception as e:
            print(f"⚠️  Error: {e}")

    print(f"\n❌ All {package_type.lower()} package installation methods failed")
    return False

def verify_installation():
    """Verify that packages were installed correctly."""
    print("\n🔍 Verifying installation...")
    
    packages = {
        'streamlit': 'Web interface framework',
        'numpy': 'Numerical computing',
        'pandas': 'Data manipulation', 
        'requests': 'HTTP requests',
        'cryptography': 'Encryption and security'
    }
    
    all_good = True
    
    for package, description in packages.items():
        try:
            __import__(package)
            print(f"✅ {package} - {description}")
        except ImportError:
            print(f"❌ {package} - {description} (STILL MISSING)")
            all_good = False
    
    return all_good

def provide_next_steps(success):
    """Provide next steps based on installation result."""
    print("\n" + "=" * 60)
    
    if success:
        print("🎉 Linux dependency installation SUCCESSFUL!")
        print("\n🚀 Next Steps:")
        print("1. Start SAM: python3 start_sam.py")
        print("2. Or run setup: python3 setup_sam.py")
        print("3. Open browser to: http://localhost:8502")
        
    else:
        print("⚠️  Linux dependency installation INCOMPLETE")
        print("\n🔧 Troubleshooting Options:")
        print()
        print("1. Try manual installation:")
        print("   pip3 install --user streamlit numpy pandas requests cryptography")
        print()
        print("2. Check Python path:")
        print("   which python3")
        print("   python3 -m pip --version")
        print()
        print("3. Install system packages:")
        print("   sudo apt install python3-dev python3-pip build-essential")
        print()
        print("4. Use virtual environment:")
        print("   python3 -m venv sam_env")
        print("   source sam_env/bin/activate")
        print("   pip install streamlit numpy pandas requests cryptography")
        print()
        print("5. Contact support with error details")

def main():
    """Main installation function."""
    print_header()
    
    # System checks
    if not check_system():
        return False
    
    if not check_python():
        return False
    
    # Try to install system packages first (helps with compilation)
    print("\n🔧 Step 1: Installing system development packages...")
    system_success = install_system_packages_first()
    if system_success:
        print("✅ System packages installed - this should help with Python packages")
    else:
        print("⚠️  System packages failed - continuing with Python packages only")

    # Install Python packages
    print("\n🔧 Step 2: Installing Python packages...")
    packages_installed = install_python_packages()
    
    # Verify installation
    if packages_installed:
        verification_success = verify_installation()
    else:
        verification_success = False
    
    # Provide next steps
    provide_next_steps(verification_success)
    
    return verification_success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Please report this error to the SAM team")
        sys.exit(1)
