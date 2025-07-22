"""
SAM Reasoning Framework

Advanced reasoning capabilities for SAM including self-discovery,
tool-augmented reasoning, and meta-cognitive processes.

Author: SAM Development Team
Version: 1.0.0
"""

from .self_decide_framework import SelfDecideFramework
from .self_discover_critic import SelfDiscoverCriticFramework
from .answer_synthesizer import AnswerSynthesizer
from .confidence_justifier import AdvancedConfidenceJustifier as ConfidenceJustifier
from .tool_selector import ToolSelector
from .tool_executor import ToolExecutor

__all__ = [
    'SelfDecideFramework',
    'SelfDiscoverCriticFramework',
    'AnswerSynthesizer',
    'ConfidenceJustifier',
    'ToolSelector',
    'ToolExecutor'
]
