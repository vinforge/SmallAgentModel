"""
TPV Configuration Module for SAM
Phase 1 - Configuration Management

This module provides configuration management for the TPV system
including dissonance monitoring parameters.
"""

import logging
import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class TPVConfig:
    """
    Configuration manager for TPV system.
    
    Handles loading, validation, and management of TPV configuration
    including dissonance monitoring settings.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize TPV configuration.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config: Dict[str, Any] = {}
        self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        # Get the directory of this file
        current_dir = Path(__file__).parent
        return str(current_dir / "tpv_config.yaml")
    
    def _load_config(self):
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    file_config = yaml.safe_load(f) or {}
                self.config.update(file_config)
                logger.info(f"Loaded TPV config from {self.config_path}")
            else:
                logger.warning(f"Config file not found: {self.config_path}, using defaults")
            
            # Ensure default configuration
            self._apply_defaults()
            
        except Exception as e:
            logger.error(f"Error loading TPV config: {e}")
            self._apply_defaults()
    
    def _apply_defaults(self):
        """Apply default configuration values."""
        defaults = {
            'control_params': {
                'completion_threshold': 0.92,
                'max_tokens': 500,
                'min_steps': 2,
                'plateau_patience': 3,
                'plateau_threshold': 0.005,
                # Dissonance control parameters
                'dissonance_threshold': 0.85,
                'dissonance_patience': 4,
                'enable_dissonance_control': True
            },
            'deployment_params': {
                'allow_user_override': True,
                'collect_feedback': True,
                'deployment_timestamp': None,
                'deployment_version': '5.0.0',  # Updated for Phase 5B
                'enable_optimizations': True,
                'enable_telemetry': True,
                'log_interventions': True,
                'production_deployment': True,
                'show_performance_warning': True,
                'tpv_enabled_by_default': False,
                'use_model_quantization': False,
                'use_onnx_runtime': False
            },
            'model_params': {
                'architecture': 'qwen3-8b',
                'discovered_at': None,
                'hidden_dimension': 4096,
                'model_name': 'hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M',
                'vocab_size': 32000  # Added for dissonance monitoring
            },
            'runtime_params': {
                'batch_size': 1,
                'device': 'auto',
                'dtype': 'float32'
            },
            'tpv_params': {
                'activation': 'gelu',
                'dropout': 0.1,
                'num_heads': 8
            },
            # New dissonance monitoring configuration
            'dissonance_params': {
                'enabled': True,
                'calculation_mode': 'entropy',  # entropy, variance, kl_divergence, composite
                'fallback_mode': True,
                'enable_profiling': True,
                'device': 'auto',
                'config': {
                    'entropy_epsilon': 1e-9,
                    'variance_threshold': 0.1,
                    'composite_weights': {
                        'entropy': 0.6,
                        'variance': 0.4
                    }
                }
            },
            # Trigger system configuration
            'trigger_params': {
                'complexity_threshold': 0.5,  # Lowered from 0.7 for easier activation
                'confidence_threshold': 0.4,  # Lowered from 0.6 for easier activation
                'enable_keyword_triggers': True,
                'enable_complexity_analysis': True,
                'enable_domain_detection': True,
                'default_activation_rate': 0.5  # Increased from 0.3 for more frequent activation
            }
        }
        
        # Merge defaults with existing config
        for section, section_config in defaults.items():
            if section not in self.config:
                self.config[section] = {}
            
            for key, value in section_config.items():
                if key not in self.config[section]:
                    self.config[section][key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to parent
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set value
        config[keys[-1]] = value
        logger.info(f"Updated config {key} = {value}")
    
    def update(self, updates: Dict[str, Any]):
        """
        Update multiple configuration values.
        
        Args:
            updates: Dictionary of updates
        """
        for key, value in updates.items():
            self.set(key, value)
    
    def save(self, path: Optional[str] = None):
        """
        Save configuration to file.
        
        Args:
            path: Path to save to (uses default if None)
        """
        save_path = path or self.config_path
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Saved TPV config to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving TPV config: {e}")
    
    def validate(self) -> bool:
        """
        Validate configuration values.
        
        Returns:
            True if configuration is valid
        """
        try:
            # Validate control parameters
            control_params = self.config.get('control_params', {})
            
            # Check required numeric ranges
            validations = [
                ('completion_threshold', 0.0, 1.0),
                ('max_tokens', 1, 10000),
                ('min_steps', 1, 100),
                ('plateau_patience', 1, 50),
                ('plateau_threshold', 0.0, 1.0),
                ('dissonance_threshold', 0.0, 1.0),
                ('dissonance_patience', 1, 50)
            ]
            
            for param, min_val, max_val in validations:
                if param in control_params:
                    value = control_params[param]
                    if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
                        logger.error(f"Invalid {param}: {value} (must be between {min_val} and {max_val})")
                        return False
            
            # Validate dissonance parameters
            dissonance_params = self.config.get('dissonance_params', {})
            if dissonance_params.get('enabled', True):
                calc_mode = dissonance_params.get('calculation_mode', 'entropy')
                valid_modes = ['entropy', 'variance', 'kl_divergence', 'composite']
                if calc_mode not in valid_modes:
                    logger.error(f"Invalid dissonance calculation_mode: {calc_mode}")
                    return False
            
            # Validate model parameters
            model_params = self.config.get('model_params', {})
            vocab_size = model_params.get('vocab_size', 32000)
            if not isinstance(vocab_size, int) or vocab_size < 1000:
                logger.error(f"Invalid vocab_size: {vocab_size}")
                return False
            
            logger.info("TPV configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False
    
    def get_control_config(self) -> Dict[str, Any]:
        """Get control-specific configuration."""
        return self.config.get('control_params', {})
    
    def get_dissonance_config(self) -> Dict[str, Any]:
        """Get dissonance monitoring configuration."""
        return self.config.get('dissonance_params', {})
    
    def get_trigger_config(self) -> Dict[str, Any]:
        """Get trigger system configuration."""
        return self.config.get('trigger_params', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration."""
        return self.config.get('model_params', {})
    
    def get_deployment_config(self) -> Dict[str, Any]:
        """Get deployment configuration."""
        return self.config.get('deployment_params', {})
    
    def is_dissonance_enabled(self) -> bool:
        """Check if dissonance monitoring is enabled."""
        return self.get('dissonance_params.enabled', True)
    
    def is_production_deployment(self) -> bool:
        """Check if this is a production deployment."""
        return self.get('deployment_params.production_deployment', False)
    
    def get_vocab_size(self) -> int:
        """Get model vocabulary size."""
        return self.get('model_params.vocab_size', 32000)
    
    def get_dissonance_threshold(self) -> float:
        """Get dissonance threshold for control decisions."""
        return self.get('control_params.dissonance_threshold', 0.85)
    
    def get_completion_threshold(self) -> float:
        """Get completion threshold for control decisions."""
        return self.get('control_params.completion_threshold', 0.92)
    
    def export_config(self) -> Dict[str, Any]:
        """
        Export current configuration.
        
        Returns:
            Complete configuration dictionary
        """
        return self.config.copy()
    
    def import_config(self, config: Dict[str, Any], validate: bool = True):
        """
        Import configuration from dictionary.
        
        Args:
            config: Configuration dictionary to import
            validate: Whether to validate after import
        """
        self.config = config.copy()
        
        if validate and not self.validate():
            logger.error("Imported configuration failed validation")
            self._apply_defaults()
        else:
            logger.info("Configuration imported successfully")
    
    def reset_to_defaults(self):
        """Reset configuration to defaults."""
        self.config = {}
        self._apply_defaults()
        logger.info("Configuration reset to defaults")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get configuration status information.
        
        Returns:
            Status dictionary
        """
        return {
            'config_path': self.config_path,
            'config_loaded': bool(self.config),
            'dissonance_enabled': self.is_dissonance_enabled(),
            'production_deployment': self.is_production_deployment(),
            'vocab_size': self.get_vocab_size(),
            'completion_threshold': self.get_completion_threshold(),
            'dissonance_threshold': self.get_dissonance_threshold(),
            'sections': list(self.config.keys())
        }
