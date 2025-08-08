"""
SAM 2.0 Configuration System
============================

Centralized configuration management for SAM, including model selection,
performance settings, and feature toggles.

This enables config-driven model switching between Transformer and Hybrid models.

Author: SAM Development Team
Version: 1.0.0
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class ModelBackend(Enum):
    """Available model backends."""
    TRANSFORMER = "transformer"
    HYBRID = "hybrid"
    AUTO = "auto"  # Automatically choose best available

    # Dynamic model support
    @classmethod
    def get_available_backends(cls) -> List[str]:
        """Get all available model backends including dynamic ones."""
        static_backends = [backend.value for backend in cls]

        # Add dynamically discovered models
        try:
            from sam.models.wrappers import get_available_models
            dynamic_models = get_available_models()
            return static_backends + dynamic_models
        except ImportError:
            return static_backends

    @classmethod
    def is_valid_backend(cls, backend: str) -> bool:
        """Check if a backend name is valid."""
        return backend in cls.get_available_backends()

class ModelSize(Enum):
    """Available model sizes."""
    DEBUG = "debug"      # For testing
    SMALL = "1b"         # 1B parameters
    MEDIUM = "3b"        # 3B parameters  
    LARGE = "8b"         # 8B parameters (production)

@dataclass
class ModelConfig:
    """Model-specific configuration."""
    backend: ModelBackend = ModelBackend.TRANSFORMER
    size: ModelSize = ModelSize.LARGE
    max_context_length: int = 16000
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout_seconds: int = 120

    # Dynamic model support
    dynamic_backend: Optional[str] = None  # For dynamic model wrappers

    # Transformer-specific settings
    transformer_model_name: str = "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M"
    transformer_api_url: str = "http://localhost:11434"

    # Hybrid-specific settings
    hybrid_model_path: Optional[str] = None
    hybrid_linear_ratio: int = 3
    hybrid_feature_map: str = "relu"

    # Performance settings
    use_flash_attention: bool = True
    gradient_checkpointing: bool = False
    enable_thinking_tokens: bool = True

@dataclass
class SAMConfig:
    """Complete SAM configuration."""

    # Model configuration
    model: ModelConfig = None
    
    # Application settings
    app_name: str = "SAM"
    version: str = "2.0.0"
    debug_mode: bool = False
    log_level: str = "INFO"
    
    # UI settings
    streamlit_port: int = 8502
    memory_center_port: int = 8501
    welcome_page_port: int = 8503
    
    # Feature flags
    enable_memory_center: bool = True
    enable_dream_canvas: bool = True
    enable_cognitive_automation: bool = True
    enable_self_reflect: bool = True
    
    # Performance settings
    max_concurrent_requests: int = 10
    request_timeout: int = 300
    enable_caching: bool = True
    cache_size_mb: int = 1024
    
    # Security settings
    require_authentication: bool = True
    session_timeout_minutes: int = 60
    enable_encryption: bool = True
    
    # Paths
    data_directory: str = "data"
    logs_directory: str = "logs"
    models_directory: str = "models"
    cache_directory: str = "cache"

    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.model is None:
            self.model = ModelConfig()

class SAMConfigManager:
    """Manages SAM configuration loading, saving, and validation."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file (default: sam_config.json)
        """
        if config_path is None:
            config_path = Path("sam_config.json")
        
        self.config_path = config_path
        self._config: Optional[SAMConfig] = None
        
    def load_config(self) -> SAMConfig:
        """
        Load configuration from file or create default.
        
        Returns:
            SAM configuration object
        """
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config_dict = json.load(f)
                
                # Convert string enums back to enum objects
                if 'model' in config_dict:
                    model_config = config_dict['model']
                    if 'backend' in model_config:
                        model_config['backend'] = ModelBackend(model_config['backend'])
                    if 'size' in model_config:
                        model_config['size'] = ModelSize(model_config['size'])
                
                # Create config objects
                model_config_obj = ModelConfig(**config_dict.get('model', {}))
                config_dict['model'] = model_config_obj
                
                self._config = SAMConfig(**config_dict)
                logger.info(f"✅ Configuration loaded from {self.config_path}")
                
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_path}: {e}")
                logger.info("Using default configuration")
                self._config = SAMConfig()
        else:
            logger.info("No configuration file found, using defaults")
            self._config = SAMConfig()
        
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        return self._config
    
    def save_config(self, config: Optional[SAMConfig] = None) -> bool:
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save (uses current if None)
            
        Returns:
            True if successful
        """
        if config is None:
            config = self._config
        
        if config is None:
            logger.error("No configuration to save")
            return False
        
        try:
            # Convert to dictionary and handle enums
            config_dict = asdict(config)
            
            # Convert enums to strings
            if 'model' in config_dict:
                model_config = config_dict['model']
                if 'backend' in model_config:
                    model_config['backend'] = model_config['backend'].value
                if 'size' in model_config:
                    model_config['size'] = model_config['size'].value
            
            # Save to file
            with open(self.config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"✅ Configuration saved to {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides to configuration."""
        if self._config is None:
            return
        
        # Model backend override
        backend_env = os.getenv('SAM_MODEL_BACKEND')
        if backend_env:
            try:
                self._config.model.backend = ModelBackend(backend_env.lower())
                logger.info(f"Model backend overridden by environment: {backend_env}")
            except ValueError:
                logger.warning(f"Invalid model backend in environment: {backend_env}")
        
        # Model size override
        size_env = os.getenv('SAM_MODEL_SIZE')
        if size_env:
            try:
                self._config.model.size = ModelSize(size_env.lower())
                logger.info(f"Model size overridden by environment: {size_env}")
            except ValueError:
                logger.warning(f"Invalid model size in environment: {size_env}")
        
        # Debug mode override
        debug_env = os.getenv('SAM_DEBUG')
        if debug_env:
            self._config.debug_mode = debug_env.lower() in ('true', '1', 'yes')
            logger.info(f"Debug mode overridden by environment: {self._config.debug_mode}")
        
        # Log level override
        log_level_env = os.getenv('SAM_LOG_LEVEL')
        if log_level_env:
            self._config.log_level = log_level_env.upper()
            logger.info(f"Log level overridden by environment: {log_level_env}")
    
    def get_config(self) -> SAMConfig:
        """Get current configuration, loading if necessary."""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def update_model_backend(self, backend: ModelBackend) -> bool:
        """
        Update model backend and save configuration.

        Args:
            backend: New model backend

        Returns:
            True if successful
        """
        config = self.get_config()
        config.model.backend = backend

        # Update context length based on backend
        if backend == ModelBackend.HYBRID:
            config.model.max_context_length = 100000  # Hybrid target
        else:
            config.model.max_context_length = 16000   # Transformer limit

        return self.save_config(config)

    def update_dynamic_model_backend(self, backend_name: str) -> bool:
        """
        Update to a dynamic model backend and save configuration.

        Args:
            backend_name: Name of the dynamic model backend

        Returns:
            True if successful
        """
        config = self.get_config()

        # Store the dynamic backend name in a special field
        config.model.dynamic_backend = backend_name
        config.model.backend = ModelBackend.AUTO  # Use AUTO to indicate dynamic

        # Try to get context length from model metadata
        try:
            from sam.models.wrappers import create_model_wrapper, create_wrapper_config_template

            temp_config = create_wrapper_config_template(backend_name)
            wrapper = create_model_wrapper(backend_name, temp_config)
            metadata = wrapper.get_model_metadata()

            config.model.max_context_length = metadata.context_window
            logger.info(f"Set context length to {metadata.context_window} for {backend_name}")

        except Exception as e:
            logger.warning(f"Could not get metadata for {backend_name}: {e}")
            config.model.max_context_length = 32000  # Safe default

        return self.save_config(config)
    
    def get_model_config_for_backend(self, backend: ModelBackend) -> Dict[str, Any]:
        """
        Get model configuration dictionary for a specific backend.
        
        Args:
            backend: Model backend
            
        Returns:
            Configuration dictionary for ModelInterface
        """
        config = self.get_config()
        
        if backend == ModelBackend.TRANSFORMER:
            return {
                "model_type": "transformer",
                "model_name": config.model.transformer_model_name,
                "api_url": config.model.transformer_api_url,
                "max_context_length": min(config.model.max_context_length, 16000),
                "timeout_seconds": config.model.timeout_seconds,
                "temperature": config.model.temperature,
                "max_tokens": config.model.max_tokens
            }
        elif backend == ModelBackend.HYBRID:
            return {
                "model_type": "hybrid_linear",
                "model_name": f"sam-hybrid-{config.model.size.value}",
                "api_url": "http://localhost:11435",  # Different port for hybrid
                "max_context_length": config.model.max_context_length,
                "timeout_seconds": config.model.timeout_seconds,
                "temperature": config.model.temperature,
                "max_tokens": config.model.max_tokens,
                "linear_ratio": config.model.hybrid_linear_ratio,
                "feature_map": config.model.hybrid_feature_map
            }
        else:
            raise ValueError(f"Unknown backend: {backend}")

# Global configuration manager
_config_manager = None

def get_config_manager() -> SAMConfigManager:
    """Get the global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = SAMConfigManager()
    return _config_manager

def get_sam_config() -> SAMConfig:
    """Get the current SAM configuration."""
    return get_config_manager().get_config()

def set_model_backend(backend: Union[str, ModelBackend]) -> bool:
    """
    Set the model backend for SAM.

    Args:
        backend: Model backend (e.g., "transformer", "hybrid", "llama31-8b", etc.)

    Returns:
        True if successful
    """
    if isinstance(backend, str):
        backend_str = backend.lower()

        # Check if it's a standard backend
        try:
            backend_enum = ModelBackend(backend_str)
            return get_config_manager().update_model_backend(backend_enum)
        except ValueError:
            # Check if it's a dynamic model
            if ModelBackend.is_valid_backend(backend_str):
                return get_config_manager().update_dynamic_model_backend(backend_str)
            else:
                available = ModelBackend.get_available_backends()
                logger.error(f"Invalid backend '{backend_str}'. Available: {available}")
                return False
    else:
        return get_config_manager().update_model_backend(backend)

def get_current_model_backend() -> ModelBackend:
    """Get the currently configured model backend."""
    return get_sam_config().model.backend

def is_hybrid_model_enabled() -> bool:
    """Check if hybrid model is currently enabled."""
    return get_current_model_backend() == ModelBackend.HYBRID

def is_debug_mode() -> bool:
    """Check if SAM is in debug mode."""
    return get_sam_config().debug_mode

# Configuration validation
def validate_config(config: SAMConfig) -> bool:
    """
    Validate SAM configuration.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    try:
        # Validate model configuration
        if config.model.max_context_length <= 0:
            raise ValueError("max_context_length must be positive")
        
        if config.model.temperature < 0 or config.model.temperature > 2:
            raise ValueError("temperature must be between 0 and 2")
        
        if config.model.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        # Validate ports
        for port_name in ['streamlit_port', 'memory_center_port', 'welcome_page_port']:
            port = getattr(config, port_name)
            if port < 1024 or port > 65535:
                raise ValueError(f"{port_name} must be between 1024 and 65535")
        
        # Validate directories
        for dir_name in ['data_directory', 'logs_directory', 'models_directory', 'cache_directory']:
            directory = getattr(config, dir_name)
            if not directory or not isinstance(directory, str):
                raise ValueError(f"{dir_name} must be a non-empty string")
        
        return True
        
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}")

# Export main functions and classes
__all__ = [
    'SAMConfig',
    'ModelConfig', 
    'ModelBackend',
    'ModelSize',
    'SAMConfigManager',
    'get_config_manager',
    'get_sam_config',
    'set_model_backend',
    'get_current_model_backend',
    'is_hybrid_model_enabled',
    'is_debug_mode',
    'validate_config'
]
