"""
SAM Discovery Module
===================

This module contains SAM's self-discovery and introspection capabilities,
including the Cognitive Distillation Engine for principle discovery.

Components:
- distillation: Core cognitive distillation engine
- analysis: Behavioral analysis tools
- validation: Principle validation systems

Author: SAM Development Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "SAM Development Team"

# Import main components
from .distillation import DistillationEngine

__all__ = [
    "DistillationEngine"
]
