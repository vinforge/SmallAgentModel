"""
SAM Benchmarks Package
=====================

Comprehensive benchmark suite for evaluating AI models in the SAM ecosystem.

Author: SAM Development Team
Version: 1.0.0
"""

from .benchmark_config import (
    BenchmarkCategory,
    BenchmarkPrompt,
    BenchmarkConfig,
    BenchmarkLoader,
    SCORING_RUBRIC,
    CATEGORY_SCORING,
    get_scoring_template,
    calculate_weighted_score
)

__all__ = [
    'BenchmarkCategory',
    'BenchmarkPrompt',
    'BenchmarkConfig',
    'BenchmarkLoader',
    'SCORING_RUBRIC',
    'CATEGORY_SCORING',
    'get_scoring_template',
    'calculate_weighted_score'
]

__version__ = "1.0.0"
