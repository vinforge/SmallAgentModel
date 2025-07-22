"""
SAM 2.0 Models Package
=====================

This package contains the hybrid model implementations for SAM 2.0,
including the HGRN-2 hybrid linear attention architecture.

Author: SAM Development Team
Version: 1.0.0
"""

from .hybrid_config import (
    HybridModelConfig,
    HybridModelConfigs,
    get_default_config,
    validate_config
)

from .sam_hybrid_model import (
    SAMHybridModel,
    HybridTransformerLayer,
    LinearAttentionLayer,
    FullAttentionLayer,
    create_sam_hybrid_model
)

__all__ = [
    # Configuration
    'HybridModelConfig',
    'HybridModelConfigs', 
    'get_default_config',
    'validate_config',
    
    # Model Components
    'SAMHybridModel',
    'HybridTransformerLayer',
    'LinearAttentionLayer', 
    'FullAttentionLayer',
    'create_sam_hybrid_model'
]

__version__ = "1.0.0"
