"""
SAM Pro Entitlement Validator
============================
Handles validation and activation of SAM Pro features using secure key validation.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
import time

class EntitlementValidator:
    """Validates and manages SAM Pro feature entitlements"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Configuration file path (contains hashed keys)
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = Path(__file__).parent.parent / "config" / "entitlements.json"
        
        # State file path (user's home directory)
        self.state_dir = Path.home() / ".sam"
        self.state_file = self.state_dir / "sam_state.json"
        
        # Rate limiting for activation attempts
        self.max_attempts = 5
        self.attempt_window = 300  # 5 minutes
        
        # Ensure state directory exists
        self.state_dir.mkdir(exist_ok=True)
        
        self.logger.info("EntitlementValidator initialized")
    
    def _load_entitlements_config(self) -> Dict[str, Any]:
        """Load entitlements configuration"""
        try:
            if not self.config_path.exists():
                self.logger.warning(f"Entitlements config not found: {self.config_path}")
                return {"valid_key_hashes": [], "features": {}}
            
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            self.logger.debug(f"Loaded entitlements config with {len(config.get('valid_key_hashes', []))} valid hashes")
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading entitlements config: {e}")
            return {"valid_key_hashes": [], "features": {}}
    
    def _load_state(self) -> Dict[str, Any]:
        """Load current activation state"""
        try:
            if not self.state_file.exists():
                return self._get_default_state()
            
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            # Ensure all required fields exist
            default_state = self._get_default_state()
            for key, value in default_state.items():
                if key not in state:
                    state[key] = value
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")
            return self._get_default_state()
    
    def _get_default_state(self) -> Dict[str, Any]:
        """Get default state structure"""
        return {
            "pro_features_unlocked": False,
            "activation_date": None,
            "activation_attempts": [],
            "version": "1.0"
        }
    
    def _save_state(self, state: Dict[str, Any]) -> bool:
        """Save activation state"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            self.logger.debug("State saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
            return False
    
    def _check_rate_limit(self, state: Dict[str, Any]) -> bool:
        """Check if rate limit is exceeded"""
        current_time = time.time()
        attempts = state.get("activation_attempts", [])
        
        # Remove old attempts outside the window
        recent_attempts = [
            attempt for attempt in attempts 
            if current_time - attempt < self.attempt_window
        ]
        
        # Update state with cleaned attempts
        state["activation_attempts"] = recent_attempts
        
        return len(recent_attempts) < self.max_attempts
    
    def _record_attempt(self, state: Dict[str, Any], success: bool) -> None:
        """Record activation attempt"""
        current_time = time.time()
        
        if "activation_attempts" not in state:
            state["activation_attempts"] = []
        
        # Only record failed attempts for rate limiting
        if not success:
            state["activation_attempts"].append(current_time)
    
    def validate_and_activate_key(self, user_key: str) -> Dict[str, Any]:
        """
        Validate user key and activate pro features if valid
        
        Args:
            user_key: The activation key entered by the user
            
        Returns:
            Dict with success status and message
        """
        try:
            # Load current state
            state = self._load_state()
            
            # Check if already activated
            if state.get("pro_features_unlocked", False):
                return {
                    "success": True,
                    "message": "✅ SAM Pro features are already activated!",
                    "already_activated": True
                }
            
            # Check rate limiting
            if not self._check_rate_limit(state):
                self._save_state(state)
                return {
                    "success": False,
                    "message": "❌ Too many activation attempts. Please wait 5 minutes.",
                    "rate_limited": True
                }
            
            # Validate key format (enhanced UUID check)
            user_key = user_key.strip()

            # Check basic length and format
            if not user_key or len(user_key) != 36:
                self._record_attempt(state, False)
                self._save_state(state)
                return {
                    "success": False,
                    "message": "❌ Invalid key format. Please check your activation key.",
                    "invalid_format": True
                }

            # Check UUID format pattern (8-4-4-4-12)
            import re
            uuid_pattern = r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
            if not re.match(uuid_pattern, user_key):
                self._record_attempt(state, False)
                self._save_state(state)
                return {
                    "success": False,
                    "message": "❌ Invalid key format. Keys must be in UUID format (e.g., 12345678-1234-1234-1234-123456789abc).",
                    "invalid_format": True
                }
            
            # Hash the user's key
            key_hash = hashlib.sha256(user_key.encode('utf-8')).hexdigest()
            
            # Load valid hashes
            config = self._load_entitlements_config()
            valid_hashes = config.get("valid_key_hashes", [])
            
            if not valid_hashes:
                self.logger.warning("No valid key hashes found in configuration")
                return {
                    "success": False,
                    "message": "❌ Entitlement system not properly configured.",
                    "config_error": True
                }
            
            # Check if hash matches
            if key_hash in valid_hashes:
                # Activate pro features
                state["pro_features_unlocked"] = True
                state["activation_date"] = time.time()
                
                self._record_attempt(state, True)
                
                if self._save_state(state):
                    self.logger.info("SAM Pro features activated successfully")
                    return {
                        "success": True,
                        "message": "🎉 SAM Pro Activated! Premium features are now unlocked.",
                        "newly_activated": True
                    }
                else:
                    return {
                        "success": False,
                        "message": "❌ Activation successful but failed to save state.",
                        "save_error": True
                    }
            else:
                # Invalid key
                self._record_attempt(state, False)
                self._save_state(state)
                
                return {
                    "success": False,
                    "message": "❌ Invalid activation key. Please check your key and try again.",
                    "invalid_key": True
                }
                
        except Exception as e:
            self.logger.error(f"Key validation error: {e}")
            return {
                "success": False,
                "message": "❌ Activation system error. Please try again.",
                "system_error": True
            }
    
    def is_pro_unlocked(self) -> bool:
        """Check if pro features are currently unlocked"""
        try:
            state = self._load_state()
            return state.get("pro_features_unlocked", False)
        except Exception as e:
            self.logger.error(f"Error checking pro status: {e}")
            return False
    
    def get_activation_info(self) -> Dict[str, Any]:
        """Get detailed activation information"""
        try:
            state = self._load_state()
            config = self._load_entitlements_config()
            
            info = {
                "is_activated": state.get("pro_features_unlocked", False),
                "activation_date": state.get("activation_date"),
                "available_features": config.get("features", {}),
                "version": state.get("version", "1.0")
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting activation info: {e}")
            return {
                "is_activated": False,
                "activation_date": None,
                "available_features": {},
                "version": "1.0"
            }
    
    def reset_activation(self) -> bool:
        """Reset activation state (for testing/debugging)"""
        try:
            default_state = self._get_default_state()
            success = self._save_state(default_state)
            
            if success:
                self.logger.info("Activation state reset successfully")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error resetting activation: {e}")
            return False
