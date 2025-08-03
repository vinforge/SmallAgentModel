#!/usr/bin/env python3
"""
SAM Master Password Reset Script
===============================

Quick script to reset the master password and encryption setup.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def print_banner():
    """Print reset banner."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🔄 SAM MASTER PASSWORD RESET 🔐                          ║
║                        Quick Password Reset Utility                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

def reset_master_password():
    """Reset the master password and encryption setup."""
    
    print_banner()
    
    print("⚠️  WARNING: This will:")
    print("   • Delete your current master password")
    print("   • Remove all encrypted data")
    print("   • Clear setup status")
    print("   • Require you to set up a new password")
    print("   • Backup existing data before deletion")
    
    confirm = input("\n❓ Are you sure you want to reset? Type 'RESET' to confirm: ").strip()
    if confirm != 'RESET':
        print("❌ Reset cancelled")
        return False
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Backup keystore
        keystore_path = Path("security/keystore.json")
        if keystore_path.exists():
            backup_path = Path(f"security/keystore_backup_{timestamp}.json")
            shutil.copy2(keystore_path, backup_path)
            print(f"📦 Backed up keystore to: {backup_path}")
            keystore_path.unlink()
            print("🗑️  Removed keystore.json")
        
        # 2. Remove encrypted memory store
        encrypted_store = Path("memory_store/encrypted")
        if encrypted_store.exists():
            backup_encrypted = Path(f"memory_store/encrypted_backup_{timestamp}")
            shutil.move(encrypted_store, backup_encrypted)
            print(f"📦 Backed up encrypted store to: {backup_encrypted}")
            print("🗑️  Removed encrypted memory store")
        
        # 3. Clear setup status files
        setup_files = [
            "security/setup_status.json",
            "setup_status.json"
        ]
        
        for setup_file in setup_files:
            setup_path = Path(setup_file)
            if setup_path.exists():
                backup_setup = Path(f"{setup_file}_backup_{timestamp}")
                shutil.copy2(setup_path, backup_setup)
                setup_path.unlink()
                print(f"🗑️  Removed {setup_file}")
        
        print("\n✅ Master password reset complete!")
        print("\n📋 Next Steps:")
        print("   1. Run: python setup_encryption.py")
        print("   2. Create a new master password")
        print("   3. Start SAM: python start_sam.py")
        
        print(f"\n💾 Backups created with timestamp: {timestamp}")
        print("   • All your data has been safely backed up")
        print("   • You can restore from backups if needed")
        
        return True
        
    except Exception as e:
        print(f"❌ Reset failed: {e}")
        return False

def main():
    """Main function."""
    try:
        success = reset_master_password()
        if success:
            print("\n🎉 Password reset successful!")
            print("   Run 'python setup_encryption.py' to create a new password")
        else:
            print("\n❌ Password reset failed or cancelled")
        
    except KeyboardInterrupt:
        print("\n\n❌ Reset cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
