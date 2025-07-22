#!/usr/bin/env python3
"""
Linux Preparation Script for SAM
================================

Quick preparation script that ensures all dependencies are installed
before running SAM. This script focuses on the essential packages
needed for SAM to start successfully.

Usage: python3 prepare_linux.py
"""

import sys
import subprocess
import platform

def print_header():
    """Print preparation header."""
    print("=" * 60)
    print("🐧 SAM Linux Preparation")
    print("=" * 60)
    print("Preparing your Linux system for SAM...")
    print()

def check_system():
    """Check if running on Linux."""
    if platform.system() != "Linux":
        print("⚠️  This script is for Linux systems only")
        print(f"💡 Detected: {platform.system()}")
        return False
    
    print(f"✅ Linux system: {platform.platform()}")
    return True

def install_essential_packages():
    """Install the essential packages needed for SAM."""
    print("\n📦 Installing essential packages for SAM...")
    
    # Essential packages in order of importance (version-pinned)
    essential_packages = ["streamlit==1.42.0", "numpy>=1.21.0,<2.0.0", "pandas>=1.3.0,<3.0.0", "requests>=2.25.0,<3.0.0"]
    
    success_count = 0
    
    for package in essential_packages:
        print(f"\n🔄 Installing {package}...")
        
        # Try multiple installation methods for each package
        methods = [
            [sys.executable, "-m", "pip", "install", "--user", package],
            ["pip3", "install", "--user", package],
            [sys.executable, "-m", "pip", "install", package]
        ]
        
        package_installed = False
        
        for method in methods:
            try:
                result = subprocess.run(
                    method, 
                    capture_output=True, 
                    text=True, 
                    timeout=120
                )
                
                if result.returncode == 0:
                    print(f"✅ {package} installed successfully")
                    package_installed = True
                    success_count += 1
                    break
                    
            except subprocess.TimeoutExpired:
                print(f"⚠️  {package} installation timeout")
            except FileNotFoundError:
                continue  # Try next method
            except Exception as e:
                print(f"⚠️  {package} installation error: {e}")
        
        if not package_installed:
            print(f"❌ Failed to install {package}")
    
    print(f"\n📊 Installation Results: {success_count}/{len(essential_packages)} packages installed")
    
    if success_count >= 3:  # At least streamlit, numpy, pandas
        print("✅ Essential packages installed - SAM should start successfully")
        return True
    else:
        print("⚠️  Some essential packages missing - SAM may have issues")
        return False

def verify_installation():
    """Verify that packages can be imported."""
    print("\n🔍 Verifying package installation...")
    
    packages_to_test = {
        'streamlit': 'Web interface',
        'numpy': 'Numerical computing',
        'pandas': 'Data manipulation',
        'requests': 'HTTP requests'
    }
    
    working_packages = 0
    
    for package, description in packages_to_test.items():
        try:
            __import__(package)
            print(f"✅ {package} - {description}")
            working_packages += 1
        except ImportError:
            print(f"❌ {package} - {description} (Import failed)")
    
    print(f"\n📊 Verification Results: {working_packages}/{len(packages_to_test)} packages working")
    
    if working_packages >= 3:
        return True
    else:
        return False

def provide_next_steps(success):
    """Provide next steps based on preparation results."""
    print("\n" + "=" * 60)
    
    if success:
        print("🎉 Linux preparation SUCCESSFUL!")
        print("\n🚀 Next Steps:")
        print("1. Run SAM setup: python3 setup_sam.py")
        print("2. Start SAM: python3 start_sam.py")
        print("3. Access SAM in your browser")
        
    else:
        print("⚠️  Linux preparation INCOMPLETE")
        print("\n🔧 Manual Installation Required:")
        print()
        print("Try these commands:")
        print("# System packages (if you have sudo)")
        print("sudo apt update")
        print("sudo apt install python3-pip python3-numpy python3-pandas")
        print()
        print("# Python packages")
        print("python3 -m pip install --user streamlit numpy pandas requests")
        print("# OR")
        print("pip3 install --user streamlit numpy pandas requests")
        print()
        print("# Then run preparation again:")
        print("python3 prepare_linux.py")

def main():
    """Main preparation function."""
    print_header()
    
    # Check system
    if not check_system():
        return False
    
    # Install essential packages
    install_success = install_essential_packages()
    
    # Verify installation
    verify_success = verify_installation()
    
    # Overall success
    overall_success = install_success and verify_success
    
    # Provide next steps
    provide_next_steps(overall_success)
    
    return overall_success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Preparation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Please try manual installation")
        sys.exit(1)
