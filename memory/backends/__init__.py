"""
SAM Memory Backend Abstraction Layer (Task 33, Phase 3)
======================================================

This module provides a standardized, backend-agnostic interface for SAM's memory systems.
It allows seamless switching between different memory implementations while maintaining
full compatibility with existing SAM components.

Supported Backends:
- SAMNativeMemoryBackend: Wraps existing custom SQLite and JSON logic
- Mem0MemoryBackend: Integrates with mem0 library for standardized memory operations

The abstraction layer enables:
- Risk-free testing of new memory backends
- Performance comparison between implementations
- Future migration to standardized memory libraries
- Simplified memory system maintenance

Part of SAM's mem0-inspired Memory Augmentation (Task 33, Phase 3)
"""

from .base_backend import BaseMemoryBackend, MemoryBackendConfig
from .sam_native_backend import SAMNativeMemoryBackend
from .mem0_backend import Mem0MemoryBackend
from .backend_factory import create_memory_backend, get_configured_backend

__all__ = [
    'BaseMemoryBackend',
    'MemoryBackendConfig', 
    'SAMNativeMemoryBackend',
    'Mem0MemoryBackend',
    'create_memory_backend',
    'get_configured_backend'
]
