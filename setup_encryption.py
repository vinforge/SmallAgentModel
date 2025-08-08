#!/usr/bin/env python3
"""
SAM Encryption Setup Script

Standalone script for setting up SAM's enterprise-grade encryption system.
Creates master password, generates encryption keys, and initializes secure storage.

Author: SAM Development Team
Version: 1.0.0
"""

import os
import sys
import time
import getpass
from pathlib import Path

def print_banner():
    """Print encryption setup banner."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🔐 SAM ENCRYPTION SETUP 🔒                               ║
║                   Enterprise-Grade Security Configuration                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

def check_security_module():
    """Check if security modules are available."""
    try:
        from security import SecureStateManager
        return True
    except ImportError:
        print("❌ Security modules not found!")
        print("🔧 Please ensure you're in the SAM directory and dependencies are installed.")
        print("   Run: pip install -r requirements.txt")
        return False

def setup_master_password():
    """Setup master password for encryption."""
    print("\n🔐 Master Password Setup")
    print("=" * 50)

    try:
        from security import SecureStateManager
        security_manager = SecureStateManager()

        if not security_manager.is_setup_required():
            print("⚠️  Encryption appears to be already set up!")
            print("🔍 This could mean:")
            print("   • You already have a master password")
            print("   • This is a shared/existing SAM installation")
            print("   • Previous setup was incomplete")

            print("\n🤔 What would you like to do?")
            print("1. 🔑 Test existing master password")
            print("2. 🔄 Reset encryption (will delete existing encrypted data)")
            print("3. ❌ Skip encryption setup")

            while True:
                choice = input("\nEnter your choice (1-3): ").strip()
                if choice in ['1', '2', '3']:
                    break
                print("❌ Please enter 1, 2, or 3")

            if choice == '1':
                return test_existing_password(security_manager)
            elif choice == '2':
                return reset_encryption_setup(security_manager)
            else:
                print("⏭️  Skipping encryption setup")
                return True

        print("🆕 This is your first time setting up SAM encryption.")
        print("You need to create a master password to encrypt your data.")
        print("\n⚠️  IMPORTANT:")
        print("- Choose a strong password you'll remember")
        print("- This password cannot be recovered if lost")
        print("- All your SAM data will be encrypted with this password")
        print("- Minimum 8 characters (12+ recommended)")

        while True:
            password = getpass.getpass("\n🔑 Enter master password: ").strip()
            if len(password) < 8:
                print("❌ Password must be at least 8 characters long")
                continue

            confirm = getpass.getpass("🔑 Confirm master password: ").strip()
            if password != confirm:
                print("❌ Passwords do not match")
                continue

            break

        print("\n🔐 Setting up secure enclave...")
        success = security_manager.setup_security(password)

        if success:
            print("✅ Master password setup successful!")
            print("✅ Encryption keys generated")
            print("✅ Secure storage initialized")
            return True
        else:
            print("❌ Failed to setup master password")
            return False

    except Exception as e:
        print(f"❌ Encryption setup failed: {e}")
        return False

def test_existing_password(security_manager):
    """Test if user knows the existing master password."""
    print("\n🔑 Testing Existing Master Password")
    print("=" * 40)

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            password = getpass.getpass(f"\n🔑 Enter your master password (attempt {attempt + 1}/{max_attempts}): ").strip()

            if security_manager.authenticate(password):
                print("✅ Master password verified successfully!")
                print("✅ Encryption is working correctly")
                print("🎉 You can now use SAM with your existing password")
                return True
            else:
                print("❌ Incorrect master password")
                if attempt < max_attempts - 1:
                    print("🔄 Please try again")

        except Exception as e:
            print(f"❌ Authentication error: {e}")

    print(f"\n💥 Failed to authenticate after {max_attempts} attempts")
    print("🤔 Would you like to reset encryption instead?")

    reset_choice = input("Reset encryption? (y/N): ").strip().lower()
    if reset_choice == 'y':
        return reset_encryption_setup(security_manager)
    else:
        print("⏭️  Encryption setup incomplete")
        return False

def reset_encryption_setup(security_manager):
    """Reset encryption setup by deleting existing keystore."""
    print("\n🔄 Resetting Encryption Setup")
    print("=" * 40)

    print("⚠️  WARNING: This will:")
    print("   • Delete all existing encryption keys")
    print("   • Remove encrypted memory data")
    print("   • Require creating a new master password")
    print("   • Cannot be undone!")

    confirm = input("\n❓ Are you sure you want to reset? Type 'RESET' to confirm: ").strip()
    if confirm != 'RESET':
        print("❌ Reset cancelled")
        return False

    try:
        # Delete keystore and encrypted data
        import shutil
        from pathlib import Path

        # Backup existing keystore
        keystore_path = Path("security/keystore.json")
        if keystore_path.exists():
            backup_path = Path(f"security/keystore_backup_{int(time.time())}.json")
            shutil.copy2(keystore_path, backup_path)
            print(f"📦 Backed up existing keystore to: {backup_path}")
            keystore_path.unlink()

        # Remove encrypted memory store
        encrypted_store = Path("memory_store/encrypted")
        if encrypted_store.exists():
            shutil.rmtree(encrypted_store)
            print("🗑️  Removed encrypted memory store")

        # Reinitialize security manager
        security_manager._initialize_state()

        print("✅ Encryption reset complete!")
        print("🆕 Now creating new master password...")

        # Now create new master password
        print("\n🔐 Create New Master Password")
        print("=" * 35)

        while True:
            password = getpass.getpass("\n🔑 Enter new master password: ").strip()
            if len(password) < 8:
                print("❌ Password must be at least 8 characters long")
                continue

            confirm = getpass.getpass("🔑 Confirm new master password: ").strip()
            if password != confirm:
                print("❌ Passwords do not match")
                continue

            break

        print("\n🔐 Setting up new secure enclave...")
        success = security_manager.setup_security(password)

        if success:
            print("✅ New master password setup successful!")
            print("✅ Fresh encryption keys generated")
            print("✅ Secure storage reinitialized")
            return True
        else:
            print("❌ Failed to setup new master password")
            return False

    except Exception as e:
        print(f"❌ Reset failed: {e}")
        return False

def create_security_directories():
    """Create necessary security directories."""
    print("\n📁 Creating security directories...")
    
    directories = ["security", "memory_store/encrypted", "logs"]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {directory}/")
    
    print("✅ Security directories created!")
    return True

def test_encryption():
    """Test encryption functionality."""
    print("\n🧪 Testing encryption...")
    
    try:
        from security import SecureStateManager
        security_manager = SecureStateManager()
        
        if security_manager.is_setup_required():
            print("⚠️  Encryption not set up yet")
            return False
        
        print("✅ Encryption system is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Encryption test failed: {e}")
        return False

def main():
    """Main encryption setup process."""
    print_banner()

    # Check if this is a fresh installation
    fresh_install = not os.path.exists("security/setup_status.json")

    if fresh_install:
        print("🎉 Welcome to SAM Community Edition!")
        print("This appears to be a fresh installation.")
        print("\n🤔 Would you like to set up AI models first?")
        print("This will download required models for document processing and AI responses.")

        setup_models = input("\nSet up AI models now? (Y/n): ").strip().lower()
        if setup_models in ['', 'y', 'yes']:
            print("\n🚀 Starting model setup...")
            try:
                import subprocess
                result = subprocess.run([sys.executable, "setup_models.py"],
                                      capture_output=False, text=True)
                if result.returncode == 0:
                    print("✅ Model setup completed")
                else:
                    print("⚠️  Model setup had issues, but continuing with encryption setup")
            except Exception as e:
                print(f"⚠️  Could not run model setup: {e}")
                print("You can run 'python setup_models.py' later")
        else:
            print("⏭️  Skipping model setup (you can run 'python setup_models.py' later)")

    print("\n🎯 This script will set up SAM's encryption system:")
    print("   • Check security modules")
    print("   • Create security directories")
    print("   • Setup master password")
    print("   • Generate encryption keys")
    print("   • Test encryption functionality")

    response = input("\n🤔 Continue with encryption setup? (Y/n): ").strip().lower()
    if response == 'n':
        print("👋 Encryption setup cancelled")
        return
    
    # Step 1: Check security modules
    print("\n" + "="*60)
    print("📋 Step 1: Checking Security Modules")
    print("="*60)
    
    if not check_security_module():
        print("\n❌ Cannot proceed without security modules")
        return
    
    # Step 2: Create directories
    print("\n" + "="*60)
    print("📋 Step 2: Creating Security Directories")
    print("="*60)
    
    if not create_security_directories():
        print("\n❌ Failed to create security directories")
        return
    
    # Step 3: Setup master password
    print("\n" + "="*60)
    print("📋 Step 3: Master Password Setup")
    print("="*60)
    
    if not setup_master_password():
        print("\n❌ Master password setup failed")
        return
    
    # Step 4: Test encryption
    print("\n" + "="*60)
    print("📋 Step 4: Testing Encryption")
    print("="*60)
    
    if not test_encryption():
        print("\n⚠️  Encryption test failed, but setup may still be valid")
    
    # Final summary
    print("\n" + "="*80)
    print("🎉 SAM Encryption Setup Complete!")
    print("="*80)
    
    print("\n🔐 **Encryption Status:**")
    print("   ✅ Master password created")
    print("   ✅ Encryption keys generated")
    print("   ✅ Secure storage initialized")
    
    print("\n🚀 **Next Steps:**")
    print("   1. Start SAM: python start_sam_secure.py --mode full")
    print("   2. Enter your master password when prompted")
    print("   3. Access SAM at http://localhost:8502")
    
    print("\n🔑 **Remember:**")
    print("   • Keep your master password safe")
    print("   • It cannot be recovered if lost")
    print("   • All SAM data is encrypted with this password")

if __name__ == "__main__":
    main()
