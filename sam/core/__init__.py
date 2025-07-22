"""
SAM Core Module

Core components for SAM's model architecture including MEMOIR integration
for lifelong learning and knowledge editing capabilities.

Author: SAM Development Team
Version: 1.0.0
"""

from .model_layers import ResidualMemoryLayer, MEMOIRTransformerBlock
from .fingerprinter import TopHashFingerprinter

__all__ = [
    'ResidualMemoryLayer',
    'MEMOIRTransformerBlock', 
    'TopHashFingerprinter'
]
