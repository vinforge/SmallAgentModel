#!/usr/bin/env python3
"""
SAM Main Launcher
================

Streamlined launcher for SAM with first-time setup detection.
Automatically guides new users through setup and launches SAM.

Usage: python start_sam.py
"""

import os
import sys
import time
import subprocess
import webbrowser
import json
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def print_header():
    """Print SAM startup header."""
    print("=" * 60)
    print("🚀 SAM - Starting Up")
    print("=" * 60)
    print("Welcome to SAM - The world's most advanced AI system")
    print("with human-like introspection and self-improvement!")
    print("=" * 60)
    print()

def check_first_time_setup():
    """Check if this is a first-time user and handle setup."""
    try:
        from utils.first_time_setup import get_first_time_setup_manager
        setup_manager = get_first_time_setup_manager()

        if setup_manager.is_first_time_user():
            print("🎯 First-time setup detected!")
            print()

            # Show setup progress
            progress = setup_manager.get_setup_progress()
            next_step = progress['next_step']

            print(f"📋 Setup Progress: {progress['completed_steps']}/{progress['total_steps']} steps complete")
            print()

            if next_step == 'master_password':
                print("🔐 Next: Create your master password for secure encryption")
                print("💡 This password protects all your SAM data and conversations")
            elif next_step == 'sam_pro_activation':
                print("🔑 Next: Activate your SAM Pro features")
                sam_pro_key = setup_manager.get_sam_pro_key()
                if sam_pro_key:
                    print(f"💎 Your SAM Pro Key: {sam_pro_key}")
                    print("💡 Enter this key in SAM to unlock all premium features")
            elif next_step == 'onboarding':
                print("🎓 Next: Complete the quick onboarding tour")
                print("💡 Learn about SAM's powerful features and capabilities")

            print()
            print("🌐 SAM will open in your browser with the setup wizard")
            print("📱 Follow the on-screen instructions to complete setup")
            print()

            return True
        else:
            print("✅ Setup complete - launching SAM...")
            print()
            return False

    except ImportError as e:
        print(f"⚠️  Missing dependencies: {e}")
        print("🔧 Run security diagnostic: python security_diagnostic.py")
        print("🚀 Continuing with SAM launch...")
        print()
        return False
    except Exception as e:
        print(f"⚠️  Could not check setup status: {e}")
        print("🚀 Continuing with SAM launch...")
        print()
        return False

def check_dependencies():
    """Check if required dependencies are available."""
    import platform

    missing_packages = []

    # Check essential packages (version-pinned for consistency)
    essential_packages = {
        'streamlit': 'streamlit==1.42.0',  # Pinned to working version
        'numpy': 'numpy>=1.21.0,<2.0.0',
        'pandas': 'pandas>=1.3.0,<3.0.0',
        'requests': 'requests>=2.25.0,<3.0.0',
        'cryptography': 'cryptography>=41.0.0,<43.0.0'
    }

    print("🔍 Checking dependencies...")
    for package_name, package_spec in essential_packages.items():
        try:
            __import__(package_name)
            print(f"✅ {package_name} available")
        except ImportError:
            print(f"❌ {package_name} not found")
            missing_packages.append(package_spec)

    # Install missing packages if any
    if missing_packages:
        print(f"\n💡 Installing {len(missing_packages)} missing packages...")

        # Determine the correct Python command
        system = platform.system()
        python_cmd = sys.executable

        # Try multiple installation methods
        installation_success = False

        # Method 1: Direct pip install with Windows compatibility
        try:
            print("🔄 Attempting installation...")
            install_cmd = [python_cmd, "-m", "pip", "install", "--user"]

            # Add --only-binary=all on Windows to prevent compilation issues
            if system == "Windows":
                install_cmd.append("--only-binary=all")
                print("💡 Using pre-built packages for Windows compatibility...")

            install_cmd.extend(missing_packages)
            result = subprocess.run(install_cmd, capture_output=True, text=True, timeout=180)

            if result.returncode == 0:
                print("✅ Packages installed successfully!")
                installation_success = True
            else:
                print(f"⚠️  Installation had issues: {result.stderr}")

        except subprocess.TimeoutExpired:
            print("⚠️  Installation timeout")
        except Exception as e:
            print(f"⚠️  Installation error: {e}")

        # Method 2: Try without --user flag
        if not installation_success:
            try:
                print("🔄 Trying alternative installation method...")
                install_cmd = [python_cmd, "-m", "pip", "install"]

                # Add --only-binary=all on Windows to prevent compilation issues
                if system == "Windows":
                    install_cmd.append("--only-binary=all")

                install_cmd.extend(missing_packages)
                result = subprocess.run(install_cmd, capture_output=True, text=True, timeout=180)

                if result.returncode == 0:
                    print("✅ Packages installed successfully!")
                    installation_success = True
                else:
                    print(f"⚠️  Alternative installation failed: {result.stderr}")

            except Exception as e:
                print(f"⚠️  Alternative installation error: {e}")

        # If automatic installation failed, try specialized installers
        if not installation_success:
            print("\n⚠️  Automatic installation failed")

            if system == "Linux":
                print("🐧 Trying specialized Linux installer...")
                try:
                    # Run the specialized Linux installer
                    result = subprocess.run([
                        python_cmd, "install_linux_dependencies.py"
                    ], capture_output=True, text=True, timeout=600)

                    if result.returncode == 0:
                        print("✅ Linux installer succeeded!")
                        installation_success = True
                    else:
                        print(f"⚠️  Linux installer failed: {result.stderr[:200]}...")
                except Exception as e:
                    print(f"⚠️  Could not run Linux installer: {e}")

            # If still failed, provide manual instructions
            if not installation_success:
                print("\n❌ All automatic installation methods failed")
                print("📋 Please install dependencies manually:")
                print()

                if system == "Linux":
                    print("🐧 For Linux (Ubuntu/Debian):")
                    print("   # Option 1: Using apt (system packages)")
                    print("   sudo apt update")
                    print("   sudo apt install python3-pip python3-numpy python3-pandas")
                    print("   python3 -m pip install --user streamlit requests cryptography")
                    print()
                    print("   # Option 2: Using pip only")
                    print(f"   python3 -m pip install --user {' '.join(missing_packages)}")
                    print("   # Or try: pip3 install --user " + ' '.join(missing_packages))
                elif system == "Darwin":  # macOS
                    print("🍎 For macOS:")
                    print(f"   python3 -m pip install --user {' '.join(missing_packages)}")
                    print("   # Or try: pip3 install --user " + ' '.join(missing_packages))
                else:  # Windows
                    print("🪟 For Windows:")
                    print(f"   python -m pip install --only-binary=all {' '.join(missing_packages)}")
                    print("   # Or try: pip install --only-binary=all " + ' '.join(missing_packages))
                    print("   # Note: --only-binary=all prevents compilation issues")

                print()
                print("💡 After manual installation, run this script again:")
                print(f"   {python_cmd} start_sam.py")
                print()
                return False

        # Re-check packages after installation
        print("\n🔍 Verifying installation...")
        still_missing = []
        for package_name in missing_packages:
            try:
                __import__(package_name)
                print(f"✅ {package_name} now available")
            except ImportError:
                print(f"❌ {package_name} still missing")
                still_missing.append(package_name)

        if still_missing:
            print(f"\n⚠️  Some packages still missing: {', '.join(still_missing)}")
            print("💡 Please install them manually and try again")
            return False

    # Check security dependencies
    try:
        from security import is_security_available
        if is_security_available():
            print("✅ Security modules available")
        else:
            print("⚠️  Security modules have missing dependencies")
            print("💡 Run diagnostic: python security_diagnostic.py")
    except ImportError:
        print("⚠️  Security modules not available")
        print("💡 Run diagnostic: python security_diagnostic.py")

    return True

def is_first_time_user():
    """Check if this is a first-time user who needs welcome setup."""
    print("🔍 Checking user setup status...")

    # The key distinction: setup_sam.py creates technical files,
    # but welcome page creates the master password and completes user setup

    setup_indicators = []

    # 1. PRIMARY CHECK: Has user completed welcome page setup?
    setup_file = Path("setup_status.json")
    if setup_file.exists():
        try:
            with open(setup_file, 'r') as f:
                status = json.load(f)

            # Check if master password was created via welcome page
            if status.get('master_password_created', False):
                setup_indicators.append("✅ Master password created via welcome page")
                print("✅ Setup status: Welcome page setup completed")
                return False
            else:
                setup_indicators.append("⚠️  Setup status file exists but no master password")

        except Exception as e:
            setup_indicators.append(f"❌ Setup status file corrupted: {e}")
    else:
        setup_indicators.append("📝 No setup status file (welcome page not completed)")

    # 2. SECONDARY CHECK: Security keystore from welcome page
    keystore_file = Path("security/keystore.json")
    if keystore_file.exists():
        try:
            with open(keystore_file, 'r') as f:
                keystore_data = json.load(f)

            # Check if this was created by welcome page (has password_hash)
            if keystore_data.get('password_hash'):
                setup_indicators.append("✅ Welcome page security setup found")
                print("✅ Setup status: Welcome page security completed")
                return False
            else:
                setup_indicators.append("⚠️  Keystore exists but no password hash (technical setup only)")

        except Exception as e:
            setup_indicators.append(f"❌ Keystore file corrupted: {e}")
    else:
        setup_indicators.append("📝 No keystore file")

    # 3. Check if setup_sam.py was run (technical setup)
    sam_pro_file = Path("sam_pro_key.txt")
    if sam_pro_file.exists():
        setup_indicators.append("🔧 Technical setup completed (setup_sam.py)")
    else:
        setup_indicators.append("📝 No technical setup")

    # 4. Check security module status
    try:
        from security import SecureStateManager
        security_manager = SecureStateManager()

        if security_manager.is_setup_required():
            setup_indicators.append("🔧 Security system needs initialization")
        else:
            setup_indicators.append("🔧 Security system initialized")

    except ImportError:
        setup_indicators.append("⚠️  Security module not available")
    except Exception as e:
        setup_indicators.append(f"❌ Security check failed: {e}")

    # Log all indicators for debugging
    print("🔍 Setup indicators found:")
    for indicator in setup_indicators:
        print(f"   • {indicator}")

    # Decision logic: If no master password from welcome page, show welcome page
    print("🎯 Result: First-time user - needs welcome page setup")
    return True

def start_sam():
    """Start SAM using Streamlit."""
    try:
        # Check if this is a first-time user
        first_time = is_first_time_user()

        if first_time:
            print("🎯 First-time user detected!")
            print("🚀 Starting SAM Welcome & Setup page...")

            # Verify welcome_setup.py exists
            welcome_file = Path("welcome_setup.py")
            if not welcome_file.exists():
                print("❌ Welcome setup file not found!")
                print("💡 Falling back to main interface...")
                print("📋 Please run: python setup_sam.py first")
                # Fall back to main interface
                cmd = [
                    sys.executable, "-m", "streamlit", "run", "secure_streamlit_app.py",
                    "--server.port", "8502",
                    "--server.address", "localhost",
                    "--browser.gatherUsageStats", "false",
                    "--server.headless", "true"
                ]
            else:
                # Start welcome setup page
                cmd = [
                    sys.executable, "-m", "streamlit", "run", "welcome_setup.py",
                    "--server.port", "8503",
                    "--server.address", "localhost",
                    "--browser.gatherUsageStats", "false",
                    "--server.headless", "true"
                ]

                print("🌐 Opening SAM Welcome page in your browser...")
                print("📱 Setup page: http://localhost:8503")
                print("📋 Complete setup to create your master password")
                print("🔑 You'll receive a SAM Pro activation key")
                print("🚀 After setup, access SAM at: http://localhost:8502")

        else:
            print("✅ Existing user detected")
            print("🚀 Starting SAM main interface...")

            # Start main SAM interface
            cmd = [
                sys.executable, "-m", "streamlit", "run", "secure_streamlit_app.py",
                "--server.port", "8502",
                "--server.address", "localhost",
                "--browser.gatherUsageStats", "false",
                "--server.headless", "true"
            ]

            print("🌐 Opening SAM in your browser...")
            print("📱 Access SAM at: http://localhost:8502")

        print()
        print("🛑 Press Ctrl+C to stop SAM")
        print("=" * 60)
        print()

        # Start the process
        process = subprocess.Popen(cmd)
        
        # Wait a moment for startup
        time.sleep(3)

        # Open browser to appropriate URL
        try:
            if first_time:
                webbrowser.open("http://localhost:8503")
                print("🌐 Browser opened to Welcome page: http://localhost:8503")
            else:
                webbrowser.open("http://localhost:8502")
                print("🌐 Browser opened to SAM interface: http://localhost:8502")
        except:
            print("⚠️  Could not open browser automatically")
            if first_time:
                print("🌐 Please open: http://localhost:8503 (Welcome & Setup)")
            else:
                print("🌐 Please open: http://localhost:8502 (SAM Interface)")
        
        # Wait for the process
        process.wait()
        
        return True
        
    except KeyboardInterrupt:
        print("\n👋 SAM stopped by user")
        try:
            process.terminate()
        except:
            pass
        return True
    except Exception as e:
        print(f"❌ Error starting SAM: {e}")
        print()
        print("🔧 Troubleshooting:")
        print("• Make sure you're in the SAM directory")
        print("• Try: pip install streamlit")
        print("• Check that secure_streamlit_app.py and welcome_setup.py exist")
        print("💡 Manual start options:")
        print("  First-time: streamlit run welcome_setup.py --server.port 8503")
        print("  Existing: streamlit run secure_streamlit_app.py --server.port 8502")
        return False

def main():
    """Main launcher function."""
    try:
        # Print header
        print_header()
        
        # Check dependencies
        if not check_dependencies():
            return 1
        
        # Check for first-time setup
        is_first_time = check_first_time_setup()
        
        # Start SAM
        if start_sam():
            return 0
        else:
            return 1
            
    except KeyboardInterrupt:
        print("\n👋 Startup cancelled by user")
        return 0
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        print("💡 Please report this issue to: vin@forge1825.net")
        return 1

if __name__ == "__main__":
    sys.exit(main())
