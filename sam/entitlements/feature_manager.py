"""
SAM Pro Feature Manager
======================
Centralized feature gating and management for SAM Pro features.
"""

import logging
from typing import Dict, Any, List
from .validator import EntitlementValidator

class FeatureManager:
    """Manages feature availability and gating for SAM Pro"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validator = EntitlementValidator()
        self._feature_cache = None
        self._cache_timestamp = 0
        self.cache_duration = 30  # Cache for 30 seconds
        
        self.logger.info("FeatureManager initialized")
    
    def _get_features_config(self) -> Dict[str, Any]:
        """Get features configuration with caching"""
        import time
        
        current_time = time.time()
        
        # Use cache if valid
        if (self._feature_cache is not None and 
            current_time - self._cache_timestamp < self.cache_duration):
            return self._feature_cache
        
        # Refresh cache
        try:
            config = self.validator._load_entitlements_config()
            self._feature_cache = config.get("features", {})
            self._cache_timestamp = current_time
            
            return self._feature_cache
            
        except Exception as e:
            self.logger.error(f"Error loading features config: {e}")
            return {}
    
    def is_feature_available(self, feature_name: str) -> bool:
        """
        Check if a specific feature is available to the user
        
        Args:
            feature_name: Name of the feature to check
            
        Returns:
            bool: True if feature is available, False otherwise
        """
        try:
            features = self._get_features_config()
            feature_config = features.get(feature_name, {})
            
            # Check if feature exists
            if not feature_config:
                self.logger.warning(f"Unknown feature: {feature_name}")
                return False
            
            # Check if feature requires pro tier
            required_tier = feature_config.get("required_tier", "free")
            
            if required_tier == "pro":
                return self.validator.is_pro_unlocked()
            
            # Free features are always available
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking feature availability for {feature_name}: {e}")
            return False
    
    def get_feature_info(self, feature_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a feature
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Dict with feature information
        """
        try:
            features = self._get_features_config()
            feature_config = features.get(feature_name, {})
            
            if not feature_config:
                return {
                    "exists": False,
                    "available": False,
                    "name": feature_name,
                    "description": "Unknown feature",
                    "required_tier": "unknown"
                }
            
            return {
                "exists": True,
                "available": self.is_feature_available(feature_name),
                "name": feature_name,
                "description": feature_config.get("description", ""),
                "ui_label": feature_config.get("ui_label", feature_name),
                "required_tier": feature_config.get("required_tier", "free"),
                "enabled": feature_config.get("enabled", False)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting feature info for {feature_name}: {e}")
            return {
                "exists": False,
                "available": False,
                "name": feature_name,
                "description": "Error loading feature info",
                "required_tier": "unknown"
            }
    
    def get_all_features(self) -> List[Dict[str, Any]]:
        """Get information about all available features"""
        try:
            features = self._get_features_config()
            feature_list = []
            
            for feature_name in features.keys():
                feature_info = self.get_feature_info(feature_name)
                feature_list.append(feature_info)
            
            return feature_list
            
        except Exception as e:
            self.logger.error(f"Error getting all features: {e}")
            return []
    
    def get_locked_features(self) -> List[Dict[str, Any]]:
        """Get list of features that are locked (require pro but not unlocked)"""
        try:
            all_features = self.get_all_features()
            locked_features = [
                feature for feature in all_features
                if feature["required_tier"] == "pro" and not feature["available"]
            ]
            
            return locked_features
            
        except Exception as e:
            self.logger.error(f"Error getting locked features: {e}")
            return []
    
    def get_available_features(self) -> List[Dict[str, Any]]:
        """Get list of features that are currently available"""
        try:
            all_features = self.get_all_features()
            available_features = [
                feature for feature in all_features
                if feature["available"]
            ]
            
            return available_features
            
        except Exception as e:
            self.logger.error(f"Error getting available features: {e}")
            return []
    
    def is_pro_unlocked(self) -> bool:
        """Check if SAM Pro is unlocked"""
        return self.validator.is_pro_unlocked()
    
    def get_pro_status(self) -> Dict[str, Any]:
        """Get comprehensive pro status information"""
        try:
            activation_info = self.validator.get_activation_info()
            locked_features = self.get_locked_features()
            available_features = self.get_available_features()
            
            return {
                "is_pro_unlocked": activation_info["is_activated"],
                "activation_date": activation_info["activation_date"],
                "total_features": len(self.get_all_features()),
                "locked_features_count": len(locked_features),
                "available_features_count": len(available_features),
                "locked_features": locked_features,
                "available_features": available_features
            }
            
        except Exception as e:
            self.logger.error(f"Error getting pro status: {e}")
            return {
                "is_pro_unlocked": False,
                "activation_date": None,
                "total_features": 0,
                "locked_features_count": 0,
                "available_features_count": 0,
                "locked_features": [],
                "available_features": []
            }
    
    def validate_key(self, activation_key: str) -> Dict[str, Any]:
        """Validate and activate using provided key"""
        return self.validator.validate_and_activate_key(activation_key)
    
    def clear_cache(self):
        """Clear the feature cache to force refresh"""
        self._feature_cache = None
        self._cache_timestamp = 0
        self.logger.debug("Feature cache cleared")

# Global feature manager instance
_feature_manager = None

def get_feature_manager() -> FeatureManager:
    """Get the global feature manager instance"""
    global _feature_manager
    if _feature_manager is None:
        _feature_manager = FeatureManager()
    return _feature_manager

# Convenience functions for common operations
def is_feature_available(feature_name: str) -> bool:
    """Check if a feature is available"""
    return get_feature_manager().is_feature_available(feature_name)

def is_pro_unlocked() -> bool:
    """Check if SAM Pro is unlocked"""
    return get_feature_manager().is_pro_unlocked()

def validate_activation_key(key: str) -> Dict[str, Any]:
    """Validate an activation key"""
    return get_feature_manager().validate_key(key)
