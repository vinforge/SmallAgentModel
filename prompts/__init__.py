"""
Structured Prompt Library for SAM

This module provides organized, maintainable prompts for different tasks,
inspired by LongBioBench's structured approach while maintaining our
superior reasoning capabilities.
"""

from .citation import CitationPrompts
from .reasoning import ReasoningPrompts
from .idk import IDKPrompts
from .summary import SummaryPrompts
from .base import BasePromptTemplate, PromptType

__all__ = [
    'CitationPrompts',
    'ReasoningPrompts', 
    'IDKPrompts',
    'SummaryPrompts',
    'BasePromptTemplate',
    'PromptType'
]
