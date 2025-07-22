#!/usr/bin/env python3
"""
SAM Quick Start Setup
====================

Quick setup script to get SAM running with a SAM Pro key.
This script handles the basic setup and key generation for new users.

Usage:
    python quick_start.py

Author: SAM Development Team
Version: 1.0.0
"""

import os
import sys
import subprocess
import uuid
import json
from pathlib import Path
from datetime import datetime

def print_header():
    """Print welcome header."""
    print("🚀 SAM Quick Start Setup")
    print("=" * 50)
    print("Welcome to SAM - The world's most advanced AI with")
    print("human-like introspection and self-improvement!")
    print("=" * 50)
    print()

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def check_dependencies():
    """Check if basic dependencies are available."""
    print("\n🔍 Checking dependencies...")
    
    required_packages = ['streamlit', 'requests', 'sqlite3']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sqlite3':
                import sqlite3
            elif package == 'streamlit':
                import streamlit
            elif package == 'requests':
                import requests
            print(f"✅ {package} - Available")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n💡 Missing packages: {', '.join(missing_packages)}")
        print("   Install with: pip install streamlit requests")
        return False
    
    return True

def create_security_directory():
    """Create security directory and basic files."""
    print("\n🔐 Setting up security directory...")
    
    security_dir = Path("security")
    security_dir.mkdir(exist_ok=True)
    
    # Create basic keystore if it doesn't exist
    keystore_file = security_dir / "keystore.json"
    if not keystore_file.exists():
        with open(keystore_file, 'w') as f:
            json.dump({}, f, indent=2)
        print("✅ Created keystore.json")
    else:
        print("✅ keystore.json already exists")
    
    # Create basic entitlements if it doesn't exist
    entitlements_file = security_dir / "entitlements.json"
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
        print("✅ Created entitlements.json")
    else:
        print("✅ entitlements.json already exists")
    
    return True

def generate_sam_pro_key():
    """Generate a SAM Pro activation key."""
    print("\n🔑 Generating your SAM Pro activation key...")
    
    # Generate key
    activation_key = str(uuid.uuid4())
    
    # Add to keystore
    try:
        keystore_file = Path("security/keystore.json")
        with open(keystore_file, 'r') as f:
            keystore = json.load(f)
        
        keystore[activation_key] = {
            'email': 'quickstart@sam.local',
            'created_date': datetime.now().isoformat(),
            'key_type': 'sam_pro_free',
            'status': 'active',
            'source': 'quick_start_setup'
        }
        
        with open(keystore_file, 'w') as f:
            json.dump(keystore, f, indent=2)
        
        print("✅ SAM Pro key generated and registered")
        return activation_key
        
    except Exception as e:
        print(f"❌ Failed to generate key: {e}")
        return None

def check_sam_files():
    """Check if main SAM files exist."""
    print("\n📁 Checking SAM files...")
    
    required_files = [
        'secure_streamlit_app.py',
        'sam_pro_registration.py',
        'simple_sam_pro_key.py'
    ]
    
    missing_files = []
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file} - Found")
        else:
            print(f"❌ {file} - Missing")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {', '.join(missing_files)}")
        print("   Please ensure you have the complete SAM installation")
        return False
    
    return True

def main():
    """Main setup function."""
    print_header()
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Check dependencies
    if not check_dependencies():
        print("\n💡 Install missing dependencies and run this script again")
        return 1
    
    # Check SAM files
    if not check_sam_files():
        print("\n💡 Please ensure you have the complete SAM installation")
        return 1
    
    # Create security directory
    if not create_security_directory():
        return 1
    
    # Generate SAM Pro key
    activation_key = generate_sam_pro_key()
    if not activation_key:
        return 1
    
    # Success!
    print("\n🎉 SAM Quick Start Setup Complete!")
    print("=" * 50)
    print()
    print("🔑 Your SAM Pro Activation Key:")
    print("=" * 50)
    print(f"   {activation_key}")
    print("=" * 50)
    print()
    print("🚀 Ready to Start SAM!")
    print()
    print("📋 Next Steps:")
    print("1. Start SAM:")
    print("   python secure_streamlit_app.py")
    print()
    print("2. Open your browser and go to:")
    print("   http://localhost:8502")
    print()
    print("3. Enter your activation key when prompted")
    print()
    print("4. Enjoy SAM Pro features including:")
    print("   • 🧠 Cognitive Distillation Engine")
    print("   • 🧠 TPV Active Reasoning Control")
    print("   • 📚 MEMOIR Lifelong Learning")
    print("   • 🎨 Dream Canvas Visualization")
    print("   • 🤖 Cognitive Automation")
    print("   • 📊 Advanced Analytics")
    print()
    print("💾 Important: Save your activation key!")
    print("💡 You can generate more keys with: python simple_sam_pro_key.py")
    print()
    print("❓ Questions? Contact: vin@forge1825.net")
    print()
    print("🌟 Welcome to the future of AI! 🚀🧠")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)
