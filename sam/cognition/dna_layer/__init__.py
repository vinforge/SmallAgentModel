"""
SAM DNA Layer - Dynamic Neural Architecture
==========================================

This module implements the DNA (Dynamic Neural Architecture) layer for SAM,
featuring data-dependent routing and specialized expert modules.

Core Components:
- DNALayer: Main dynamic layer that replaces standard transformer blocks
- Expert Modules: Specialized computational units (Attention, MLP, Identity, Normalization)
- TokenRouter: Learned routing mechanism for dynamic module selection
- Metrics: Comprehensive tracking and analysis tools

Author: SAM Development Team
Version: 1.0.0 (Proof-of-Concept)
"""

from .dynamic_layer import DNALayer, DNALayerFactory
from .modules import (
    BaseModule,
    AttentionModule, 
    MLPModule,
    IdentityModule,
    NormalizationModule
)
from .router import TokenRouter
from .config import DNAConfig, DNAConfigs
from .metrics import DNAMetrics
from .visualizer import RoutingVisualizer

__all__ = [
    'DNALayer',
    'DNALayerFactory',
    'BaseModule',
    'AttentionModule',
    'MLPModule',
    'IdentityModule',
    'NormalizationModule',
    'TokenRouter',
    'DNAConfig',
    'DNAConfigs',
    'DNAMetrics',
    'RoutingVisualizer'
]

__version__ = "1.0.0"
