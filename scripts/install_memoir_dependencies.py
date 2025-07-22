#!/usr/bin/env python3
"""
Install MEMOIR Dependencies

Script to install required dependencies for MEMOIR functionality.

Usage:
    python scripts/install_memoir_dependencies.py

Author: SAM Development Team
Version: 1.0.0
"""

import subprocess
import sys
import logging

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def install_package(package_name, import_name=None):
    """
    Install a package using pip.
    
    Args:
        package_name: Name of the package to install
        import_name: Name to use for import test (defaults to package_name)
    
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    import_name = import_name or package_name
    
    # Check if already installed
    try:
        __import__(import_name)
        logger.info(f"✅ {package_name} is already installed")
        return True
    except ImportError:
        pass
    
    # Install the package
    logger.info(f"📦 Installing {package_name}...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", package_name
        ])
        
        # Verify installation
        __import__(import_name)
        logger.info(f"✅ {package_name} installed successfully")
        return True
        
    except (subprocess.CalledProcessError, ImportError) as e:
        logger.error(f"❌ Failed to install {package_name}: {e}")
        return False

def main():
    """Main installation function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("🚀 Installing MEMOIR Dependencies")
    print("=" * 50)
    
    # Required packages
    packages = [
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("faiss-cpu", "faiss"),  # CPU version of FAISS
        # ("faiss-gpu", "faiss"),  # Uncomment for GPU version
    ]
    
    # Optional packages (will warn if not available but won't fail)
    optional_packages = [
        ("sentence-transformers", "sentence_transformers"),
        ("transformers", "transformers"),
    ]
    
    success_count = 0
    total_required = len(packages)
    
    # Install required packages
    print("\n📦 Installing Required Packages:")
    for package_name, import_name in packages:
        if install_package(package_name, import_name):
            success_count += 1
        else:
            print(f"❌ Failed to install required package: {package_name}")
    
    # Install optional packages
    print("\n📦 Installing Optional Packages:")
    for package_name, import_name in optional_packages:
        if install_package(package_name, import_name):
            print(f"✅ Optional package {package_name} installed")
        else:
            print(f"⚠️  Optional package {package_name} not installed (not critical)")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 INSTALLATION SUMMARY")
    print("=" * 50)
    print(f"Required packages installed: {success_count}/{total_required}")
    
    if success_count == total_required:
        print("\n🎉 All required dependencies installed successfully!")
        print("\n✅ MEMOIR components are ready to use.")
        print("\nNext steps:")
        print("  1. Run: python scripts/verify_memoir_phase_a.py")
        print("  2. Generate permutation matrix: python scripts/generate_permutation.py")
        print("  3. Run unit tests: python tests/test_memoir_components.py")
        return 0
    else:
        print(f"\n❌ Installation incomplete!")
        print(f"Please manually install the missing packages.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
