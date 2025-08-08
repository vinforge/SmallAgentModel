"""
SAM Orchestration Framework Configuration
========================================

Configuration management for the SOF system including feature flags,
skill registration, and backward compatibility settings.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field

logger = logging.getLogger(__name__)


@dataclass
class SOFConfig:
    """Configuration for SAM Orchestration Framework."""
    
    # Core framework settings
    use_sof_framework: bool = False  # Main feature flag for SOF
    sof_version: str = "2.0.0"
    
    # Skill execution settings
    max_execution_time: float = 60.0  # Maximum total execution time
    skill_timeout: float = 30.0       # Maximum time per skill
    enable_parallel_execution: bool = False  # Future feature
    
    # Plan validation settings
    enable_plan_validation: bool = True
    max_plan_length: int = 10
    allow_circular_dependencies: bool = False
    
    # Error handling settings
    enable_fallback_plans: bool = True
    max_retry_attempts: int = 2
    continue_on_skill_failure: bool = False
    
    # Security settings
    enable_skill_sandboxing: bool = True
    require_vetting_for_external: bool = True
    max_intermediate_data_size: int = 10 * 1024 * 1024  # 10MB
    
    # Performance settings
    enable_plan_caching: bool = True
    plan_cache_ttl: int = 3600  # 1 hour
    enable_execution_metrics: bool = True
    
    # Integration settings
    integrate_with_tpv: bool = True
    integrate_with_memory: bool = True
    integrate_with_vetting: bool = True
    
    # Logging and debugging
    log_level: str = "INFO"
    enable_execution_tracing: bool = True
    save_execution_logs: bool = True
    
    # Backward compatibility
    fallback_to_legacy: bool = True  # Fallback to old system on SOF failure
    legacy_timeout: float = 30.0

    # SELF-REFLECT Configuration (Phase 5C)
    enable_self_reflect: bool = True  # Enable SELF-REFLECT methodology
    self_reflect_confidence_threshold: float = 0.7  # Trigger threshold for low confidence
    self_reflect_profiles: List[str] = field(default_factory=lambda: ['researcher', 'legal', 'academic'])  # User profiles that trigger self-reflection
    self_reflect_query_keywords: List[str] = field(default_factory=lambda: ['what is', 'who was', 'when did', 'define', 'explain', 'describe'])  # Query keywords that trigger self-reflection
    enable_memoir_auto_correction: bool = True  # Auto-feed corrections to MEMOIR
    self_reflect_max_revisions: int = 1  # Maximum revision attempts per response


class SOFConfigManager:
    """Manages SOF configuration loading, saving, and validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else Path("config/sof_config.json")
        self._config: Optional[SOFConfig] = None
        self._ensure_config_directory()
    
    def _ensure_config_directory(self) -> None:
        """Ensure configuration directory exists."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
    
    def load_config(self) -> SOFConfig:
        """
        Load SOF configuration from file or create default.
        
        Returns:
            SOFConfig instance
        """
        if self._config is not None:
            return self._config
        
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Validate and create config
                self._config = SOFConfig(**config_data)
                logger.info(f"Loaded SOF configuration from {self.config_path}")
            else:
                # Create default configuration
                self._config = SOFConfig()
                self.save_config()
                logger.info(f"Created default SOF configuration at {self.config_path}")
            
            return self._config
            
        except Exception as e:
            logger.error(f"Error loading SOF configuration: {e}")
            logger.info("Using default configuration")
            self._config = SOFConfig()
            return self._config
    
    def save_config(self, config: Optional[SOFConfig] = None) -> None:
        """
        Save SOF configuration to file.
        
        Args:
            config: Configuration to save (uses current if None)
        """
        if config is None:
            config = self._config or SOFConfig()
        
        try:
            config_data = asdict(config)
            
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            self._config = config
            logger.info(f"Saved SOF configuration to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error saving SOF configuration: {e}")
    
    def update_config(self, **kwargs) -> SOFConfig:
        """
        Update configuration with new values.
        
        Args:
            **kwargs: Configuration values to update
            
        Returns:
            Updated configuration
        """
        config = self.load_config()
        
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")
        
        self.save_config(config)
        return config
    
    def is_sof_enabled(self) -> bool:
        """
        Check if SOF framework is enabled.
        
        Returns:
            True if SOF is enabled, False otherwise
        """
        config = self.load_config()
        return config.use_sof_framework
    
    def enable_sof(self) -> None:
        """Enable SOF framework."""
        self.update_config(use_sof_framework=True)
        logger.info("SOF framework enabled")
    
    def disable_sof(self) -> None:
        """Disable SOF framework."""
        self.update_config(use_sof_framework=False)
        logger.info("SOF framework disabled")
    
    def get_skill_config(self, skill_name: str) -> Dict[str, Any]:
        """
        Get configuration specific to a skill.
        
        Args:
            skill_name: Name of the skill
            
        Returns:
            Dictionary with skill-specific configuration
        """
        config = self.load_config()
        
        return {
            "max_execution_time": config.skill_timeout,
            "enable_sandboxing": config.enable_skill_sandboxing,
            "enable_metrics": config.enable_execution_metrics,
            "log_level": config.log_level
        }
    
    def validate_config(self, config: Optional[SOFConfig] = None) -> List[str]:
        """
        Validate configuration and return any issues.
        
        Args:
            config: Configuration to validate (uses current if None)
            
        Returns:
            List of validation issues (empty if valid)
        """
        if config is None:
            config = self.load_config()
        
        issues = []
        
        # Validate timeouts
        if config.max_execution_time <= 0:
            issues.append("max_execution_time must be positive")
        
        if config.skill_timeout <= 0:
            issues.append("skill_timeout must be positive")
        
        if config.skill_timeout > config.max_execution_time:
            issues.append("skill_timeout cannot exceed max_execution_time")
        
        # Validate plan settings
        if config.max_plan_length <= 0:
            issues.append("max_plan_length must be positive")
        
        if config.max_retry_attempts < 0:
            issues.append("max_retry_attempts cannot be negative")
        
        # Validate cache settings
        if config.plan_cache_ttl <= 0:
            issues.append("plan_cache_ttl must be positive")
        
        # Validate data size limits
        if config.max_intermediate_data_size <= 0:
            issues.append("max_intermediate_data_size must be positive")
        
        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if config.log_level not in valid_log_levels:
            issues.append(f"log_level must be one of: {valid_log_levels}")
        
        return issues
    
    def reset_to_defaults(self) -> SOFConfig:
        """
        Reset configuration to defaults.
        
        Returns:
            Default configuration
        """
        self._config = SOFConfig()
        self.save_config()
        logger.info("SOF configuration reset to defaults")
        return self._config
    
    def get_migration_config(self) -> Dict[str, Any]:
        """
        Get configuration for migration from legacy system.
        
        Returns:
            Migration configuration dictionary
        """
        config = self.load_config()
        
        return {
            "use_sof_framework": config.use_sof_framework,
            "fallback_to_legacy": config.fallback_to_legacy,
            "legacy_timeout": config.legacy_timeout,
            "enable_execution_tracing": config.enable_execution_tracing
        }


# Global configuration manager instance
_config_manager: Optional[SOFConfigManager] = None


def get_sof_config_manager(config_path: Optional[str] = None) -> SOFConfigManager:
    """
    Get or create global SOF configuration manager.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        SOFConfigManager instance
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = SOFConfigManager(config_path)
    
    return _config_manager


def get_sof_config() -> SOFConfig:
    """
    Get current SOF configuration.
    
    Returns:
        SOFConfig instance
    """
    return get_sof_config_manager().load_config()


def is_sof_enabled() -> bool:
    """
    Check if SOF framework is enabled.
    
    Returns:
        True if SOF is enabled, False otherwise
    """
    return get_sof_config_manager().is_sof_enabled()


def enable_sof_framework() -> None:
    """Enable SOF framework globally."""
    get_sof_config_manager().enable_sof()


def disable_sof_framework() -> None:
    """Disable SOF framework globally."""
    get_sof_config_manager().disable_sof()
