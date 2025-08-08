"""
SAM Optimization Module - Task 30 Phase 3
==========================================

Provides response caching and performance optimization capabilities
for the conversational coherence engine.

Part of Task 30: Advanced Conversational Coherence Engine
"""

from .response_cache import (
    ResponseCache,
    PerformanceOptimizer,
    CacheEntry,
    get_response_cache,
    get_performance_optimizer
)

__all__ = [
    'ResponseCache',
    'PerformanceOptimizer',
    'CacheEntry',
    'get_response_cache',
    'get_performance_optimizer'
]
