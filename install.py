#!/usr/bin/env python3
"""
SAM Community Edition - Beta Installation Script
Automated installation and setup for SAM (Smart Assistant Memory) Beta Release.
"""

import os
import sys
import subprocess
import platform
import logging
import json
import urllib.request
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SAMBetaInstaller:
    def __init__(self):
        self.system = platform.system().lower()
        self.python_version = sys.version_info
        self.sam_dir = Path.cwd()
        self.version = "1.0.0-beta"
        
    def check_requirements(self):
        """Check system requirements for SAM Beta."""
        logger.info("🔍 Checking system requirements...")

        # Check Python version
        if self.python_version < (3, 8):
            logger.error("❌ Python 3.8+ is required. Current version: {}.{}".format(
                self.python_version.major, self.python_version.minor))
            return False

        logger.info(f"✅ Python {self.python_version.major}.{self.python_version.minor} detected")

        # Check pip
        try:
            subprocess.run([sys.executable, "-m", "pip", "--version"],
                         check=True, capture_output=True)
            logger.info("✅ pip is available")
        except subprocess.CalledProcessError:
            logger.error("❌ pip is not available")
            return False

        # Check available disk space (minimum 2GB)
        try:
            disk_usage = shutil.disk_usage(self.sam_dir)
            free_gb = disk_usage.free / (1024**3)
            if free_gb < 2:
                logger.warning(f"⚠️  Low disk space: {free_gb:.1f}GB available (2GB+ recommended)")
            else:
                logger.info(f"✅ Sufficient disk space: {free_gb:.1f}GB available")
        except Exception as e:
            logger.warning(f"⚠️  Could not check disk space: {e}")

        return True
    
    def install_dependencies(self):
        """Install Python dependencies for SAM Beta."""
        logger.info("📦 Installing Python dependencies...")

        try:
            # Upgrade pip first
            logger.info("Upgrading pip...")
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                         check=True, capture_output=True)

            # Install requirements
            requirements_file = self.sam_dir / "requirements.txt"
            if requirements_file.exists():
                logger.info("Installing SAM dependencies (this may take a few minutes)...")
                result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                                      check=True, capture_output=True, text=True)
                logger.info("✅ Dependencies installed successfully")

                # Check for any warnings
                if "WARNING" in result.stderr:
                    logger.warning("Some warnings occurred during installation:")
                    for line in result.stderr.split('\n'):
                        if "WARNING" in line:
                            logger.warning(f"  {line}")
            else:
                logger.error("❌ requirements.txt not found")
                return False

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            if hasattr(e, 'stderr') and e.stderr:
                logger.error(f"Error details: {e.stderr}")
            return False

        return True
    
    def setup_ollama(self):
        """Setup Ollama for local LLM."""
        logger.info("🤖 Setting up Ollama for SAM Beta...")

        # Check if Ollama is installed
        try:
            result = subprocess.run(["ollama", "--version"],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✅ Ollama is already installed: {result.stdout.strip()}")
            else:
                raise subprocess.CalledProcessError(1, "ollama")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.info("📥 Ollama not found. Installing...")

            if self.system == "darwin":  # macOS
                logger.info("For macOS, please install Ollama manually:")
                logger.info("  1. Visit: https://ollama.ai/download")
                logger.info("  2. Download and install Ollama for macOS")
                logger.info("  3. Or run: brew install ollama")
                logger.warning("⚠️  Please install Ollama and re-run this installer")
                return False
            elif self.system == "linux":
                try:
                    logger.info("Installing Ollama for Linux...")
                    install_cmd = "curl -fsSL https://ollama.ai/install.sh | sh"
                    subprocess.run(install_cmd, shell=True, check=True)
                    logger.info("✅ Ollama installed successfully")
                except subprocess.CalledProcessError:
                    logger.warning("⚠️  Automatic installation failed. Please install manually:")
                    logger.warning("     curl -fsSL https://ollama.ai/install.sh | sh")
                    return False
            else:
                logger.warning("⚠️  Please install Ollama manually from: https://ollama.ai/download")
                return False

        # Check if Ollama service is running
        try:
            result = subprocess.run(["ollama", "list"],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                logger.info("Starting Ollama service...")
                # Try to start Ollama service
                if self.system == "linux":
                    subprocess.run(["systemctl", "--user", "start", "ollama"],
                                 capture_output=True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            logger.warning("⚠️  Ollama service may not be running. Please start it manually:")
            logger.warning("     ollama serve")

        # Pull the required model
        logger.info("📥 Downloading SAM's language model (this may take several minutes)...")
        try:
            model_name = "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M"
            result = subprocess.run(["ollama", "pull", model_name],
                                  check=True, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            logger.info("✅ Language model downloaded successfully")
        except subprocess.TimeoutExpired:
            logger.warning("⚠️  Model download timed out. You can download it later with:")
            logger.warning("    ollama pull hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M")
        except subprocess.CalledProcessError as e:
            logger.warning("⚠️  Could not download model automatically. You can do this later with:")
            logger.warning("    ollama pull hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M")
            logger.warning(f"Error: {e}")

        return True
    
    def create_config(self):
        """Create initial configuration."""
        logger.info("⚙️  Creating configuration...")
        
        config_dir = self.sam_dir / "config"
        config_dir.mkdir(exist_ok=True)
        
        # Create sam_config.json
        config = {
            "model": {
                "provider": "ollama",
                "model_name": "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M",
                "api_url": "http://localhost:11434"
            },
            "memory": {
                "max_memories": 10000,
                "embedding_model": "all-MiniLM-L6-v2"
            },
            "ui": {
                "port": 5001,
                "host": "0.0.0.0"
            },
            "features": {
                "show_thoughts": True,
                "thoughts_default_hidden": True,
                "enable_thought_toggle": True,
                "web_search": False
            }
        }
        
        config_file = config_dir / "sam_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info("✅ Configuration created")
    
    def setup_directories(self):
        """Setup required directories."""
        logger.info("📁 Setting up directories...")
        
        directories = [
            "logs",
            "data/uploads",
            "data/documents", 
            "data/vector_store",
            "memory_store",
            "multimodal_output",
            "web_ui/uploads",
            "web_ui/data",
            "web_ui/memory_store",
            "web_ui/multimodal_output",
            "models/embeddings",
            "vector_store/enriched_chunks",
        ]
        
        for directory in directories:
            dir_path = self.sam_dir / directory
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ Directories created")
    
    def create_launcher_scripts(self):
        """Create convenient launcher scripts."""
        logger.info("🚀 Creating launcher scripts...")
        
        # Create start script for Unix systems
        if self.system in ["linux", "darwin"]:
            start_script = self.sam_dir / "start_sam.sh"
            with open(start_script, 'w') as f:
                f.write("""#!/bin/bash
# SAM Community Edition Launcher
echo "🚀 Starting SAM Community Edition..."
cd "$(dirname "$0")"
python3 launch_web_ui.py
""")
            os.chmod(start_script, 0o755)
            
        # Create start script for Windows
        start_script_win = self.sam_dir / "start_sam.bat"
        with open(start_script_win, 'w') as f:
            f.write("""@echo off
REM SAM Community Edition Launcher
echo 🚀 Starting SAM Community Edition...
cd /d "%~dp0"
python launch_web_ui.py
pause
""")
        
        logger.info("✅ Launcher scripts created")
    
    def run_tests(self):
        """Run basic system tests."""
        logger.info("🧪 Running system tests...")
        
        try:
            # Test imports
            test_imports = [
                "flask",
                "sentence_transformers", 
                "numpy",
                "requests",
                "streamlit"
            ]
            
            for module in test_imports:
                __import__(module)
                logger.info(f"✅ {module} import successful")
            
            logger.info("✅ All tests passed")
            return True
            
        except ImportError as e:
            logger.error(f"❌ Import test failed: {e}")
            return False
    
    def install(self):
        """Run complete SAM Beta installation."""
        logger.info("🎯 Starting SAM Community Edition Beta installation...")
        logger.info(f"Version: {self.version}")
        logger.info("=" * 60)

        # Step 1: Check requirements
        if not self.check_requirements():
            logger.error("❌ System requirements not met")
            return False

        # Step 2: Install dependencies
        if not self.install_dependencies():
            logger.error("❌ Dependency installation failed")
            return False

        # Step 3: Setup Ollama
        if not self.setup_ollama():
            logger.warning("⚠️  Ollama setup incomplete - you may need to install it manually")

        # Step 4: Create configuration
        self.create_config()

        # Step 5: Setup directories
        self.setup_directories()

        # Step 6: Create launcher scripts
        self.create_launcher_scripts()

        # Step 7: Run tests
        if not self.run_tests():
            logger.warning("⚠️  Some tests failed, but installation may still work")

        # Step 8: Show completion message
        logger.info("=" * 60)
        logger.info("🎉 SAM Community Edition Beta installation complete!")
        logger.info("")
        logger.info("📋 Quick Start:")
        logger.info("1. Ensure Ollama is running: ollama serve")
        logger.info("2. Launch SAM:")
        if self.system in ["linux", "darwin"]:
            logger.info("   ./start_sam.sh")
        logger.info("   OR: python start_sam.py")
        logger.info("   OR: python launch_web_ui.py")
        logger.info("3. Open browser to: http://localhost:5001")
        logger.info("")
        logger.info("📚 Resources:")
        logger.info("   Documentation: README.md")
        logger.info("   Configuration: config/sam_config.json")
        logger.info("   Logs: logs/sam.log")
        logger.info("")
        logger.info("🆘 Support:")
        logger.info("   If you encounter issues, check the logs and README.md")
        logger.info("   Make sure Ollama is installed and running")

        return True

def main():
    """Main installation function."""
    installer = SAMBetaInstaller()

    print("=" * 70)
    print("🤖 SAM Community Edition Beta Installer")
    print("   Smart Assistant Memory - Open Source Beta Release")
    print(f"   Version: {installer.version}")
    print("=" * 70)
    print()

    try:
        success = installer.install()
        if success:
            print("\n🎉 Installation completed successfully!")
            print("🚀 You can now start SAM with: python start_sam.py")
            return 0
        else:
            print("\n❌ Installation failed!")
            print("📋 Please check the error messages above and try again")
            return 1
    except KeyboardInterrupt:
        print("\n⚠️  Installation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
