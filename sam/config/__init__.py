"""
SAM 2.0 Configuration Package
============================

Centralized configuration management for SAM 2.0.

Author: SAM Development Team
Version: 1.0.0
"""

from .sam_config import (
    SAMConfig,
    ModelConfig,
    ModelBackend,
    ModelSize,
    SAMConfigManager,
    get_config_manager,
    get_sam_config,
    set_model_backend,
    get_current_model_backend,
    is_hybrid_model_enabled,
    is_debug_mode,
    validate_config
)

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

__version__ = "1.0.0"
