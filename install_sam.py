#!/usr/bin/env python3
"""
SAM Secure Installation Script

One-command installer for SAM with security features.
Handles dependency installation, setup, and first launch.

Author: SAM Development Team
Version: 1.0.0
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """Print SAM installation banner."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🧠 SAM - SECURE AI ASSISTANT 🔒                          ║
║                         Installation & Setup Script                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("⚠️  SAM requires Python 3.8 or higher")
        print("\n📥 Install Python 3.9+ from:")
        print("   • https://python.org/downloads/")
        print("   • Or use your system package manager")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible!")
    return True

def check_system_requirements():
    """Check system requirements."""
    print("\n💻 Checking system requirements...")
    
    # Check available memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 4:
            print(f"⚠️  Low memory detected: {memory_gb:.1f}GB (4GB+ recommended)")
        else:
            print(f"✅ Memory: {memory_gb:.1f}GB")
    except ImportError:
        print("ℹ️  Memory check skipped (psutil not available)")
    
    # Check disk space
    try:
        disk_free = psutil.disk_usage('.').free / (1024**3)
        if disk_free < 2:
            print(f"⚠️  Low disk space: {disk_free:.1f}GB (2GB+ recommended)")
        else:
            print(f"✅ Disk space: {disk_free:.1f}GB available")
    except:
        print("ℹ️  Disk space check skipped")
    
    # Check platform
    system = platform.system()
    print(f"✅ Platform: {system} {platform.machine()}")
    
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    try:
        # Upgrade pip first
        print("  🔄 Upgrading pip...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ], check=True, capture_output=True)
        
        # Install requirements
        print("  📥 Installing SAM dependencies...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Dependency installation failed:")
            print(result.stderr)
            print("\n🔧 Try manual installation:")
            print("   pip install -r requirements.txt")
            return False
        
        print("✅ All dependencies installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False
    except FileNotFoundError:
        print("❌ requirements.txt not found")
        print("⚠️  Make sure you're in the SAM directory")
        return False

def verify_installation():
    """Verify that SAM components can be imported."""
    print("\n🔍 Verifying installation...")
    
    # Test core imports
    test_imports = [
        ("streamlit", "Streamlit web framework"),
        ("flask", "Flask web server"),
        ("sentence_transformers", "AI embeddings"),
        ("chromadb", "Vector database"),
        ("cryptography", "Encryption library"),
        ("argon2", "Password hashing (argon2-cffi)")
    ]
    
    failed_imports = []
    
    for module, description in test_imports:
        try:
            __import__(module)
            print(f"  ✅ {description}")
        except ImportError:
            print(f"  ❌ {description}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠️  Some components failed to import: {', '.join(failed_imports)}")
        print("🔧 Try reinstalling with: pip install -r requirements.txt")
        return False
    
    # Test SAM security module
    try:
        from security import SecureStateManager
        print("  ✅ SAM Security Module")
    except ImportError as e:
        print(f"  ⚠️  SAM Security Module: {e}")
        print("     (This is normal for first-time setup)")
    
    print("✅ Installation verification completed!")
    return True

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = [
        "logs",
        "data",
        "uploads",
        "security",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✅ {directory}/")
    
    return True

def setup_configuration():
    """Create default configuration files."""
    print("\n⚙️  Setting up configuration...")
    
    # Create default config if it doesn't exist
    config_file = Path("config/settings.json")
    if not config_file.exists():
        default_config = {
            "security": {
                "session_timeout": 3600,
                "max_unlock_attempts": 5,
                "auto_lock_enabled": True
            },
            "ui": {
                "default_interface": "streamlit",
                "theme": "auto",
                "show_security_status": True
            },
            "memory": {
                "max_memories": 10000,
                "embedding_dimension": 384,
                "similarity_threshold": 0.7
            },
            "uploads": {
                "max_file_size_mb": 100,
                "allowed_extensions": [".pdf", ".txt", ".docx", ".md"],
                "auto_process": True
            }
        }
        
        import json
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        print("  ✅ Default configuration created")
    else:
        print("  ✅ Configuration already exists")
    
    return True

def get_setup_method():
    """Get user's preferred setup method."""
    print("\n🚀 Choose Setup Method:")
    print("=" * 50)
    print("  1. 🎯 Interactive Setup (Recommended)")
    print("     • Guided configuration wizard")
    print("     • Automatic encryption setup")
    print("     • System optimization")
    print("     • Beginner-friendly")
    print()
    print("  2. ⚡ Quick Launch")
    print("     • Use current configuration")
    print("     • Launch SAM immediately")
    print("     • For experienced users")
    print()
    print("  3. ⚙️  Setup Only")
    print("     • Complete setup without launching")
    print("     • Manual launch later")

    while True:
        try:
            choice = input("\nEnter your choice (1-3) [1]: ").strip()
            if not choice:
                return 1  # Default to interactive

            choice_num = int(choice)
            if 1 <= choice_num <= 3:
                return choice_num
            else:
                print("❌ Please enter 1, 2, or 3")
        except ValueError:
            print("❌ Please enter a valid number")

def run_interactive_setup():
    """Run the interactive setup process."""
    print("\n🎯 Starting Interactive Setup...")
    print("This will guide you through SAM's complete configuration.")

    try:
        # Check if setup_sam.py exists
        if not Path("setup_sam.py").exists():
            print("❌ Interactive setup script not found")
            print("Using fallback quick setup...")
            return launch_sam_quick()

        # Run interactive setup
        result = subprocess.run([sys.executable, "setup_sam.py"], check=True)
        print("✅ Interactive setup completed!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Interactive setup failed: {e}")
        print("Falling back to quick setup...")
        return launch_sam_quick()
    except KeyboardInterrupt:
        print("\n👋 Setup cancelled by user")
        return False

def launch_sam_quick():
    """Quick launch SAM with current configuration."""
    print("\n⚡ Quick Launch")
    print("Choose your preferred interface:")
    print("  1. 🌐 Full Suite (Web UI + Streamlit + Memory Center)")
    print("  2. 📱 Streamlit App (Modern interface)")
    print("  3. 🌐 Web UI (Traditional interface)")
    print("  4. 🧠 Memory Center (Memory management)")

    while True:
        try:
            choice = input("\nEnter your choice (1-4) [1]: ").strip()
            if not choice:
                choice = "1"  # Default to full suite

            if choice == "1":
                launch_command = [sys.executable, "start_sam_secure.py", "--mode", "full"]
                break
            elif choice == "2":
                launch_command = [sys.executable, "start_sam_secure.py", "--mode", "streamlit"]
                break
            elif choice == "3":
                launch_command = [sys.executable, "start_sam_secure.py", "--mode", "web"]
                break
            elif choice == "4":
                launch_command = [sys.executable, "start_sam_secure.py", "--mode", "memory"]
                break
            else:
                print("❌ Invalid choice. Please enter 1-4.")
                continue

        except KeyboardInterrupt:
            print("\n\n👋 Setup cancelled by user")
            return False

    print(f"\n🚀 Launching SAM...")
    print("📝 First-time setup: You'll create your master password")
    print("🔒 This password encrypts all your data - choose carefully!")
    print("💡 SAM will automatically guide you through encryption setup")
    print("\n⚠️  Press Ctrl+C to stop SAM when you're done")

    try:
        subprocess.run(launch_command)
        return True
    except KeyboardInterrupt:
        print("\n👋 SAM stopped by user")
        return True
    except Exception as e:
        print(f"❌ Failed to launch SAM: {e}")
        return False

def launch_sam():
    """Main launch function with setup method selection."""
    setup_method = get_setup_method()

    if setup_method == 1:
        # Interactive Setup
        return run_interactive_setup()

    elif setup_method == 2:
        # Quick Launch
        return launch_sam_quick()

    elif setup_method == 3:
        # Setup Only
        print("\n✅ Setup completed! Launch SAM anytime with:")
        print("   python start_sam_secure.py --mode full")
        print("\n📖 Documentation:")
        print("   • docs/QUICK_ENCRYPTION_SETUP.md - Quick start guide")
        print("   • docs/ENCRYPTION_SETUP_GUIDE.md - Complete guide")
        return True

def main():
    """Main installation function."""
    print_banner()
    
    print("🎯 This script will:")
    print("   • Check system requirements")
    print("   • Install dependencies")
    print("   • Set up SAM configuration")
    print("   • Offer interactive setup (recommended)")
    print("   • Launch SAM for first-time setup")
    
    # Confirm installation
    try:
        response = input("\n🤔 Continue with installation? (Y/n): ").strip().lower()
        if response and response not in ['y', 'yes']:
            print("👋 Installation cancelled")
            return
    except KeyboardInterrupt:
        print("\n👋 Installation cancelled")
        return
    
    # Run installation steps
    steps = [
        ("Python Version", check_python_version),
        ("System Requirements", check_system_requirements),
        ("Dependencies", install_dependencies),
        ("Installation Verification", verify_installation),
        ("Directory Setup", create_directories),
        ("Configuration", setup_configuration)
    ]
    
    for step_name, step_func in steps:
        print(f"\n{'='*60}")
        print(f"📋 {step_name}")
        print('='*60)
        
        if not step_func():
            print(f"\n❌ {step_name} failed!")
            print("🔧 Please fix the issues above and try again")
            return
    
    # Launch SAM
    print(f"\n{'='*60}")
    print("🎉 Installation Complete!")
    print('='*60)
    
    launch_sam()
    
    print("\n🎉 Welcome to SAM - Your Secure AI Assistant!")
    print("\n📖 Quick Tips:")
    print("   • Your master password encrypts all data")
    print("   • SAM runs entirely on your machine")
    print("   • All conversations and documents are private")
    print("   • Use the security dashboard to monitor encryption")
    print("   • Master password setup is automatic on first launch")
    
    print("\n📚 Documentation:")
    print("   • docs/QUICK_ENCRYPTION_SETUP.md - 5-minute setup guide")
    print("   • docs/ENCRYPTION_SETUP_GUIDE.md - Complete encryption guide")
    print("   • docs/README_SECURE_INSTALLATION.md - Full installation guide")
    print("   • docs/ - Complete documentation")
    print("   • start_sam_secure.py --help - Command options")

    print("\n🚀 Launch SAM anytime with:")
    print("   python start_sam_secure.py --mode full")

    print("\n🎯 Setup Options:")
    print("   • Interactive: python setup_sam.py")
    print("   • Quick: python install_sam.py")
    print("   • Encryption only: python setup_encryption.py")

if __name__ == "__main__":
    main()
