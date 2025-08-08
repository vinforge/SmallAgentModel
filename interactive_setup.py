#!/usr/bin/env python3
"""
SAM Interactive Setup Script

Comprehensive guided setup for new SAM users with step-by-step instructions,
dependency installation, and security configuration.

Author: SAM Development Team
Version: 1.0.0
"""

import os
import sys
import subprocess
import platform
import time
from pathlib import Path

def print_banner():
    """Print interactive setup banner."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🧠 SAM INTERACTIVE SETUP WIZARD 🔧                       ║
║                     Guided Installation & Configuration                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

def print_step(step_num, total_steps, title):
    """Print step header."""
    print(f"\n{'='*80}")
    print(f"📋 Step {step_num}/{total_steps}: {title}")
    print('='*80)

def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("⚠️  SAM requires Python 3.8 or higher")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible!")
    return True

def check_system_requirements():
    """Check system requirements."""
    print("\n💻 Checking system requirements...")

    try:
        import psutil

        # Check memory with error handling
        try:
            memory_gb = psutil.virtual_memory().total / (1024**3)
            print(f"✅ Memory: {memory_gb:.1f}GB")
            if memory_gb < 4:
                print("⚠️  Warning: Less than 4GB RAM detected. SAM may run slowly.")
        except:
            print("ℹ️  Memory check skipped")
            memory_gb = 0

        # Check disk space with robust error handling
        try:
            if platform.system() == "Windows":
                # Use current drive on Windows
                import os
                current_drive = os.path.splitdrive(os.getcwd())[0] + os.sep
                disk_gb = psutil.disk_usage(current_drive).free / (1024**3)
            else:
                # Use current directory on Unix-like systems
                disk_gb = psutil.disk_usage('.').free / (1024**3)

            print(f"✅ Disk space: {disk_gb:.1f}GB available")
            if disk_gb < 2:
                print("⚠️  Warning: Less than 2GB disk space. Consider freeing up space.")
        except:
            # Broad exception handling like install_sam.py for maximum compatibility
            print("ℹ️  Disk space check skipped")
            disk_gb = 0

        print(f"✅ Platform: {platform.system()} {platform.machine()}")
        return True

    except ImportError:
        print("ℹ️  System requirements check skipped (psutil not available)")
        print(f"✅ Platform: {platform.system()} {platform.machine()}")
        return True
    except Exception as e:
        print(f"ℹ️  System requirements check skipped: {e}")
        print(f"✅ Platform: {platform.system()} {platform.machine()}")
        return True

def install_dependencies():
    """Install SAM dependencies."""
    print("\n📦 Installing SAM dependencies...")
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found!")
        print("🔧 Creating basic requirements file...")
        
        basic_requirements = """streamlit>=1.28.0
flask>=2.3.0
sentence-transformers>=2.2.0
requests>=2.31.0
beautifulsoup4>=4.12.0
PyPDF2>=3.0.0
python-docx>=0.8.11
psutil>=5.9.0
cryptography>=41.0.0
argon2-cffi>=23.1.0
chromadb>=0.4.0
pandas>=2.0.0
numpy>=1.24.0
"""
        with open("requirements.txt", "w") as f:
            f.write(basic_requirements)
        print("✅ Basic requirements.txt created")
    
    try:
        print("  🔄 Upgrading pip...")
        pip_result = subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                                   capture_output=True, text=True, timeout=120)
        if pip_result.returncode != 0:
            print(f"⚠️  Pip upgrade warning: {pip_result.stderr}")

        print("  📥 Installing SAM dependencies...")
        install_result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                                       capture_output=True, text=True, timeout=300)

        if install_result.returncode == 0:
            print("✅ Dependencies installed successfully!")
            return True
        else:
            print(f"❌ Dependency installation failed:")
            print(f"Error: {install_result.stderr if install_result.stderr else install_result.stdout}")
            print("\n🔧 Try manual installation:")
            print("   pip install -r requirements.txt")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Dependency installation timed out")
        print("🔧 Try manual installation:")
        print("   pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Dependency installation failed: {e}")
        print("\n🔧 Try manual installation:")
        print("   pip install -r requirements.txt")
        return False

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating SAM directories...")
    
    directories = [
        "logs", "memory_store", "security", "config", 
        "uploads", "quarantine", "approved", "archive",
        "data", "data/uploads", "data/documents"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {directory}/")
    
    print("✅ All directories created!")
    return True

def setup_encryption():
    """Setup encryption and master password."""
    print("\n🔐 Setting up encryption...")
    print("SAM uses enterprise-grade encryption to protect your data.")
    print("Let's create your master password now.")

    print("\n" + "="*60)
    print("🔑 MASTER PASSWORD CREATION")
    print("="*60)
    print("⚠️  IMPORTANT:")
    print("   • Choose a strong password you'll remember")
    print("   • This password cannot be recovered if lost")
    print("   • All your SAM data will be encrypted with this password")
    print("   • Minimum 8 characters (12+ recommended)")

    # Ask if user wants to create password now or later
    print("\nYou can create your master password now or during first launch.")
    response = input("Create master password now? (Y/n): ").strip().lower()

    if response == 'n':
        print("✅ Master password will be created on first launch")
        return True

    # Create master password interactively
    try:
        import subprocess
        import sys

        print("\n🔐 Running encryption setup...")
        result = subprocess.run([sys.executable, "setup_encryption.py"],
                               capture_output=False, text=True)

        if result.returncode == 0:
            print("✅ Master password created successfully!")
            print("✅ Encryption setup completed!")
            return True
        else:
            print("❌ Encryption setup failed!")
            print("💡 You can set up encryption later by running:")
            print("   python setup_encryption.py")
            return True  # Don't fail the entire setup

    except FileNotFoundError:
        print("❌ setup_encryption.py not found")
        print("✅ Encryption will be configured on first run")
        return True
    except Exception as e:
        print(f"❌ Encryption setup failed: {e}")
        print("✅ Encryption will be configured on first run")
        return True

def check_ollama():
    """Check if Ollama is installed."""
    print("\n🤖 Checking Ollama installation...")

    try:
        # Try different ways to find ollama on different platforms
        ollama_commands = ["ollama", "ollama.exe"]
        ollama_found = False

        for cmd in ollama_commands:
            try:
                result = subprocess.run([cmd, "--version"],
                                       capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print("✅ Ollama is installed!")
                    ollama_found = True
                    break
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue

        if ollama_found:
            return True

    except Exception as e:
        print(f"⚠️  Error checking Ollama: {e}")

    print("⚠️  Ollama not found. SAM can work without it, but AI features will be limited.")
    print("\n📥 To install Ollama:")
    print("   • Visit: https://ollama.ai/download")
    print("   • Download for your platform")
    if platform.system() == "Windows":
        print("   • Install Ollama for Windows")
        print("   • Open Command Prompt and run: ollama pull hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M")
    else:
        print("   • Install and run: ollama pull hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M")

    response = input("\nContinue without Ollama? (y/N): ").strip().lower()
    return response == 'y'

def final_setup():
    """Final setup steps."""
    print("\n🎉 Final setup...")

    print("✅ SAM installation completed successfully!")
    print("\n🚀 **Next Steps:**")
    print("   1. Start SAM: python start_sam_secure.py --mode full")

    # Check if encryption was already set up
    try:
        from pathlib import Path
        if Path("security").exists() and any(Path("security").glob("*.key")):
            print("   2. Enter your master password when prompted")
        else:
            print("   2. Create your master password when prompted (if not done already)")
    except:
        print("   2. Create your master password when prompted (if not done already)")

    print("   3. Access SAM at http://localhost:8502")

    print("\n📍 **Access Points:**")
    print("   • Secure Chat: http://localhost:8502")
    print("   • Memory Center: Integrated into secure interface")
    print("   • Standard Chat: http://localhost:5001")

    print("\n🔐 **Important:**")
    print("   • Your master password encrypts all SAM data")
    print("   • Keep your password safe - it cannot be recovered if lost")
    print("   • SAM runs entirely on your machine for maximum privacy")

    return True

def main():
    """Main interactive setup process."""
    print_banner()
    
    print("🎯 This wizard will guide you through SAM installation:")
    print("   • System requirements check")
    print("   • Dependency installation")
    print("   • Directory creation")
    print("   • Master password creation (interactive)")
    print("   • AI model configuration")
    
    response = input("\n🤔 Continue with interactive setup? (Y/n): ").strip().lower()
    if response == 'n':
        print("👋 Setup cancelled")
        return
    
    steps = [
        ("System Requirements", lambda: check_python_version() and check_system_requirements()),
        ("Install Dependencies", install_dependencies),
        ("Create Directories", create_directories),
        ("Setup Encryption", setup_encryption),
        ("Check AI Models", check_ollama),
        ("Final Configuration", final_setup)
    ]
    
    total_steps = len(steps)
    
    for i, (title, func) in enumerate(steps, 1):
        print_step(i, total_steps, title)
        
        if not func():
            print(f"\n❌ Step {i} failed. Please resolve the issues and try again.")
            return
        
        if i < total_steps:
            input("\nPress Enter to continue to next step...")
    
    print("\n" + "="*80)
    print("🎉 SAM Interactive Setup Complete!")
    print("="*80)
    print("SAM is ready to use! Run 'python start_sam_secure.py --mode full' to start.")

if __name__ == "__main__":
    main()
