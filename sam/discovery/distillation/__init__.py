"""
SAM Cognitive Distillation Module
================================

This module implements the Cognitive Distillation Engine that analyzes
SAM's successful behaviors and distills them into human-readable
"Principles of Reasoning" for improved performance and explainability.

Core Components:
- DistillationEngine: Main orchestrator for principle discovery
- InteractionCollector: Gathers successful interaction data
- PrincipleValidator: Validates discovered principles
- PrincipleRegistry: Manages principle storage and retrieval

Author: SAM Development Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "SAM Development Team"

# Import main components
from .engine import DistillationEngine
from .collector import InteractionCollector
from .validator import PrincipleValidator
from .registry import PrincipleRegistry
from .llm_integration import LLMIntegration

# Phase 2 components
from .prompt_augmentation import PromptAugmentation
from .thought_transparency import ThoughtTransparency
from .automation import AutomatedDistillation
from .sam_integration import SAMCognitiveDistillation
from .optimization import PrincipleOptimizer, PerformanceMonitor

__all__ = [
    # Core components
    "DistillationEngine",
    "InteractionCollector",
    "PrincipleValidator",
    "PrincipleRegistry",
    "LLMIntegration",

    # Phase 2 integration components
    "PromptAugmentation",
    "ThoughtTransparency",
    "AutomatedDistillation",
    "SAMCognitiveDistillation",
    "PrincipleOptimizer",
    "PerformanceMonitor"
]
