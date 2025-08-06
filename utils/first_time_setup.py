#!/usr/bin/env python3
"""
SAM First-Time Setup Detection and Management
============================================

Handles detection of first-time users and guides them through
the initial setup process including master password creation
and SAM Pro activation.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

class FirstTimeSetupManager:
    """Manages first-time setup detection and configuration."""
    
    def __init__(self):
        """Initialize the setup manager."""
        self.setup_file = Path("security/setup_status.json")
        self.keystore_file = Path("security/keystore.json")
        self.entitlements_file = Path("security/entitlements.json")
        
    def is_first_time_user(self) -> bool:
        """Check if this is a first-time user who needs setup."""
        try:
            # Check if setup status file exists
            if not self.setup_file.exists():
                return True
            
            # Check setup status
            with open(self.setup_file, 'r') as f:
                setup_status = json.load(f)
            
            # Check if all required setup steps are complete
            required_steps = [
                'master_password_created',
                'sam_pro_activated',
                'onboarding_completed'
            ]
            
            for step in required_steps:
                if not setup_status.get(step, False):
                    return True
            
            return False
            
        except Exception:
            # If we can't read the file, assume first-time user
            return True
    
    def get_setup_status(self) -> Dict[str, Any]:
        """Get the current setup status."""
        try:
            if not self.setup_file.exists():
                return self._create_default_setup_status()
            
            with open(self.setup_file, 'r') as f:
                return json.load(f)
                
        except Exception:
            return self._create_default_setup_status()
    
    def _create_default_setup_status(self) -> Dict[str, Any]:
        """Create default setup status."""
        return {
            'master_password_created': False,
            'sam_pro_activated': False,
            'onboarding_completed': False,
            'setup_version': '2.0.0',
            'created_date': None,
            'completed_date': None
        }
    
    def update_setup_status(self, step: str, completed: bool = True) -> bool:
        """Update a specific setup step status."""
        try:
            # Ensure security directory exists
            self.setup_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Get current status
            status = self.get_setup_status()
            
            # Update the step
            status[step] = completed
            
            # Set completion date if all steps are done
            if self._all_steps_complete(status):
                from datetime import datetime
                status['completed_date'] = datetime.now().isoformat()
            
            # Save updated status
            with open(self.setup_file, 'w') as f:
                json.dump(status, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error updating setup status: {e}")
            return False
    
    def _all_steps_complete(self, status: Dict[str, Any]) -> bool:
        """Check if all setup steps are complete."""
        required_steps = [
            'master_password_created',
            'sam_pro_activated', 
            'onboarding_completed'
        ]
        
        return all(status.get(step, False) for step in required_steps)
    
    def mark_setup_complete(self) -> bool:
        """Mark the entire setup process as complete."""
        try:
            from datetime import datetime
            
            status = self.get_setup_status()
            status.update({
                'master_password_created': True,
                'sam_pro_activated': True,
                'onboarding_completed': True,
                'completed_date': datetime.now().isoformat()
            })
            
            # Ensure security directory exists
            self.setup_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.setup_file, 'w') as f:
                json.dump(status, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error marking setup complete: {e}")
            return False
    
    def has_master_password(self) -> bool:
        """Check if master password has been created."""
        try:
            # Check if there's an encryption setup
            from security.crypto_utils import CryptoManager
            crypto_manager = CryptoManager()
            return crypto_manager.is_initialized()
        except ImportError:
            # Security module not available, check for setup status file
            try:
                status = self.get_setup_status()
                return status.get('master_password_created', False)
            except:
                return False
        except Exception:
            # If we can't check, assume no password
            return False
    
    def has_sam_pro_key(self) -> bool:
        """Check if SAM Pro key has been activated."""
        try:
            if not self.keystore_file.exists():
                return False
            
            with open(self.keystore_file, 'r') as f:
                keystore = json.load(f)
            
            # Check if there are any active keys
            return len(keystore) > 0
            
        except Exception:
            return False
    
    def get_sam_pro_key(self) -> Optional[str]:
        """Get the first available SAM Pro key."""
        try:
            # First, check if we stored the key in setup status
            status = self.get_setup_status()
            if 'sam_pro_key' in status:
                return status['sam_pro_key']

            # Check keystore for activation keys (skip metadata entries)
            if self.keystore_file.exists():
                with open(self.keystore_file, 'r') as f:
                    keystore = json.load(f)

                # Look for UUID-format keys (activation keys)
                import re
                uuid_pattern = r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'

                for key, data in keystore.items():
                    # Skip metadata entries
                    if key in ['metadata', 'kdf_config', 'salt', 'verifier', 'security_settings', 'audit_log']:
                        continue

                    # Check if it's a UUID format key with active status
                    if re.match(uuid_pattern, key) and isinstance(data, dict) and data.get('status') == 'active':
                        return key

            # Fallback: Check if there's a key file from setup
            setup_key_file = Path("sam_pro_key.txt")
            if setup_key_file.exists():
                with open(setup_key_file, 'r') as f:
                    key = f.read().strip()
                    if key:
                        # Store it in setup status for future reference
                        self.update_setup_status('sam_pro_key', key)
                        return key

            return None

        except Exception as e:
            print(f"Error getting SAM Pro key: {e}")
            return None
    
    def get_next_setup_step(self) -> str:
        """Get the next setup step that needs to be completed."""
        status = self.get_setup_status()
        
        if not status.get('master_password_created', False):
            return 'master_password'
        elif not status.get('sam_pro_activated', False):
            return 'sam_pro_activation'
        elif not status.get('onboarding_completed', False):
            return 'onboarding'
        else:
            return 'complete'
    
    def get_setup_progress(self) -> Dict[str, Any]:
        """Get setup progress information."""
        status = self.get_setup_status()
        
        steps = [
            ('master_password_created', 'Create Master Password'),
            ('sam_pro_activated', 'Activate SAM Pro'),
            ('onboarding_completed', 'Complete Onboarding')
        ]
        
        completed_steps = sum(1 for step, _ in steps if status.get(step, False))
        total_steps = len(steps)
        
        return {
            'completed_steps': completed_steps,
            'total_steps': total_steps,
            'progress_percent': int((completed_steps / total_steps) * 100),
            'next_step': self.get_next_setup_step(),
            'is_complete': completed_steps == total_steps
        }

def get_first_time_setup_manager():
    """Get a singleton instance of the FirstTimeSetupManager."""
    if not hasattr(get_first_time_setup_manager, '_instance'):
        get_first_time_setup_manager._instance = FirstTimeSetupManager()
    return get_first_time_setup_manager._instance
