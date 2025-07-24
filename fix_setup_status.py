#!/usr/bin/env python3
"""
Fix Setup Status Synchronization
================================

This script fixes the disconnect between the welcome setup and the 
security system by synchronizing the setup status files.
"""

import json
import os
from pathlib import Path
from datetime import datetime

def fix_setup_status():
    """Fix the setup status synchronization issue."""
    
    print("🔧 SAM Setup Status Fix")
    print("=" * 50)
    
    # Check current state
    root_setup_file = Path("setup_status.json")
    security_setup_file = Path("security/setup_status.json")
    keystore_file = Path("security/keystore.json")
    
    print("📋 Current Status:")
    print(f"   • Root setup file: {'✅ Exists' if root_setup_file.exists() else '❌ Missing'}")
    print(f"   • Security setup file: {'✅ Exists' if security_setup_file.exists() else '❌ Missing'}")
    print(f"   • Keystore file: {'✅ Exists' if keystore_file.exists() else '❌ Missing'}")
    
    # Read root setup status
    root_status = {}
    if root_setup_file.exists():
        try:
            with open(root_setup_file, 'r') as f:
                root_status = json.load(f)
            print(f"   • Root setup status: {root_status}")
        except Exception as e:
            print(f"   • Error reading root setup: {e}")
    
    # Check if keystore is properly set up
    keystore_valid = False
    if keystore_file.exists():
        try:
            with open(keystore_file, 'r') as f:
                keystore_data = json.load(f)
            
            # Check if keystore has proper structure
            if ('metadata' in keystore_data and 
                'verifier' in keystore_data and 
                keystore_data['metadata'].get('first_setup_completed')):
                keystore_valid = True
                print("   • Keystore: ✅ Valid and set up")
            else:
                print("   • Keystore: ⚠️ Exists but incomplete")
        except Exception as e:
            print(f"   • Keystore error: {e}")
    
    # Determine what needs to be fixed
    master_password_created = (
        root_status.get('master_password_created', False) and keystore_valid
    )
    
    sam_pro_key = root_status.get('sam_pro_key')
    
    print("\n🔍 Analysis:")
    print(f"   • Master password created: {'✅ Yes' if master_password_created else '❌ No'}")
    print(f"   • SAM Pro key available: {'✅ Yes' if sam_pro_key else '❌ No'}")
    
    if master_password_created:
        print("\n✅ Master password is properly set up!")
        print("🔧 Creating synchronized setup status...")
        
        # Create the security setup status file
        security_setup_file.parent.mkdir(parents=True, exist_ok=True)
        
        security_status = {
            'master_password_created': True,
            'sam_pro_activated': True,  # Auto-activate for community edition
            'onboarding_completed': True,  # Skip onboarding if password is set
            'setup_version': '2.0.0',
            'created_date': datetime.now().isoformat(),
            'completed_date': datetime.now().isoformat(),
            'synchronized_from_root': True,
            'sync_timestamp': datetime.now().isoformat()
        }
        
        if sam_pro_key:
            security_status['sam_pro_key'] = sam_pro_key
        
        try:
            with open(security_setup_file, 'w') as f:
                json.dump(security_status, f, indent=2)
            print(f"✅ Created: {security_setup_file}")
        except Exception as e:
            print(f"❌ Failed to create security setup file: {e}")
            return False
        
        # Update root setup status to be complete
        try:
            updated_root_status = root_status.copy()
            updated_root_status.update({
                'master_password_created': True,
                'sam_pro_activated': True,
                'onboarding_completed': True,
                'setup_synchronized': True,
                'sync_timestamp': datetime.now().isoformat()
            })

            with open(root_setup_file, 'w') as f:
                json.dump(updated_root_status, f, indent=2)
            print(f"✅ Updated: {root_setup_file}")
        except Exception as e:
            print(f"⚠️ Warning: Could not update root setup file: {e}")
            print(f"   Error details: {e}")
            # Try to show the current working directory and file permissions
            import os
            print(f"   Current directory: {os.getcwd()}")
            print(f"   File exists: {root_setup_file.exists()}")
            if root_setup_file.exists():
                print(f"   File permissions: {oct(root_setup_file.stat().st_mode)[-3:]}")
        
        print("\n🎉 Setup status synchronized successfully!")
        print("\n📋 Next Steps:")
        print("   1. Run: python start_sam.py")
        print("   2. Enter your master password when prompted")
        print("   3. Enjoy using SAM!")
        
        return True
    
    else:
        print("\n❌ Master password is not properly set up!")
        print("\n📋 To fix this, you need to:")
        print("   1. Run: python start_sam.py")
        print("   2. Complete the setup wizard to create your master password")
        print("   3. Then run this fix script again")
        
        return False

if __name__ == "__main__":
    try:
        success = fix_setup_status()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        exit(1)
