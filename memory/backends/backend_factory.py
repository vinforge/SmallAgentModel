"""
Memory Backend Factory (Task 33, Phase 3)
=========================================

Factory functions for creating and managing memory backends.
Provides easy switching between different backend implementations.
"""

import logging
import os
from typing import Dict, Any, Optional
from pathlib import Path

from .base_backend import BaseMemoryBackend, MemoryBackendConfig, BackendType
from .sam_native_backend import SAMNativeMemoryBackend
from .mem0_backend import Mem0MemoryBackend

logger = logging.getLogger(__name__)

# Global backend instance for singleton pattern
_global_backend: Optional[BaseMemoryBackend] = None

def create_memory_backend(backend_type: BackendType, 
                         config: Optional[MemoryBackendConfig] = None) -> BaseMemoryBackend:
    """
    Create a memory backend of the specified type.
    
    Args:
        backend_type: Type of backend to create
        config: Optional configuration (uses defaults if None)
        
    Returns:
        Initialized memory backend
        
    Raises:
        ValueError: If backend type is not supported
        RuntimeError: If backend initialization fails
    """
    try:
        # Use default config if none provided
        if config is None:
            config = MemoryBackendConfig(backend_type=backend_type)
        
        # Create backend based on type
        if backend_type == BackendType.SAM_NATIVE:
            backend = SAMNativeMemoryBackend(config)
        elif backend_type == BackendType.MEM0:
            backend = Mem0MemoryBackend(config)
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")
        
        # Validate backend functionality
        is_valid, issues = backend.validate_backend()
        if not is_valid:
            logger.warning(f"Backend validation issues: {issues}")
            # Continue anyway - some issues might be non-critical
        
        logger.info(f"Created {backend_type.value} memory backend successfully")
        return backend
        
    except Exception as e:
        logger.error(f"Failed to create {backend_type.value} backend: {e}")
        raise RuntimeError(f"Backend creation failed: {e}")

def get_configured_backend() -> BaseMemoryBackend:
    """
    Get a memory backend based on configuration.
    
    Checks environment variables and configuration files to determine
    which backend to use. Falls back to SAM Native if no configuration found.
    
    Returns:
        Configured memory backend (singleton)
    """
    global _global_backend
    
    if _global_backend is not None:
        return _global_backend
    
    try:
        # Check environment variable for backend type
        backend_type_str = os.getenv('SAM_MEMORY_BACKEND', 'sam_native').lower()
        
        # Parse backend type
        if backend_type_str == 'mem0':
            backend_type = BackendType.MEM0
        else:
            backend_type = BackendType.SAM_NATIVE
        
        # Create configuration from environment
        config = _create_config_from_environment(backend_type)
        
        # Create and cache backend
        _global_backend = create_memory_backend(backend_type, config)
        
        logger.info(f"Configured global backend: {backend_type.value}")
        return _global_backend
        
    except Exception as e:
        logger.error(f"Failed to get configured backend: {e}")
        # Fallback to SAM Native
        logger.info("Falling back to SAM Native backend")
        fallback_config = MemoryBackendConfig(backend_type=BackendType.SAM_NATIVE)
        _global_backend = SAMNativeMemoryBackend(fallback_config)
        return _global_backend

def _create_config_from_environment(backend_type: BackendType) -> MemoryBackendConfig:
    """Create backend configuration from environment variables."""
    
    # Base configuration from environment
    config = MemoryBackendConfig(
        backend_type=backend_type,
        storage_directory=os.getenv('SAM_MEMORY_STORAGE_DIR', 'memory_store'),
        embedding_dimension=int(os.getenv('SAM_EMBEDDING_DIMENSION', '384')),
        max_memory_chunks=int(os.getenv('SAM_MAX_MEMORY_CHUNKS', '10000')),
        similarity_threshold=float(os.getenv('SAM_SIMILARITY_THRESHOLD', '0.1')),
        auto_cleanup_enabled=os.getenv('SAM_AUTO_CLEANUP', 'true').lower() == 'true',
        cleanup_threshold_days=int(os.getenv('SAM_CLEANUP_THRESHOLD_DAYS', '90'))
    )
    
    # Backend-specific configuration
    if backend_type == BackendType.SAM_NATIVE:
        # SAM Native specific config
        from ..memory_vectorstore import VectorStoreType
        
        store_type_str = os.getenv('SAM_VECTOR_STORE_TYPE', 'simple').lower()
        if store_type_str == 'faiss':
            store_type = VectorStoreType.FAISS
        elif store_type_str == 'chroma':
            store_type = VectorStoreType.CHROMA
        else:
            store_type = VectorStoreType.SIMPLE
        
        config.backend_specific_config = {
            'store_type': store_type
        }
        
    elif backend_type == BackendType.MEM0:
        # Mem0 specific config
        mem0_config = {}
        
        # Vector store configuration
        mem0_vector_provider = os.getenv('MEM0_VECTOR_PROVIDER', 'chroma')
        mem0_collection = os.getenv('MEM0_COLLECTION_NAME', 'sam_memories')
        
        mem0_config['mem0_config'] = {
            "vector_store": {
                "provider": mem0_vector_provider,
                "config": {
                    "collection_name": mem0_collection,
                    "path": str(Path(config.storage_directory) / "mem0_chroma")
                }
            }
        }
        
        # Embedder configuration
        mem0_embedder = os.getenv('MEM0_EMBEDDER_MODEL', 'all-MiniLM-L6-v2')
        mem0_config['mem0_config']["embedder"] = {
            "provider": "sentence_transformers",
            "config": {
                "model": mem0_embedder
            }
        }
        
        config.backend_specific_config = mem0_config
    
    return config

def switch_backend(backend_type: BackendType, 
                  config: Optional[MemoryBackendConfig] = None) -> BaseMemoryBackend:
    """
    Switch to a different memory backend.
    
    Args:
        backend_type: New backend type
        config: Optional configuration
        
    Returns:
        New backend instance
        
    Note:
        This will replace the global backend instance.
        Existing references to the old backend will still work.
    """
    global _global_backend
    
    logger.info(f"Switching to {backend_type.value} backend")
    
    # Create new backend
    new_backend = create_memory_backend(backend_type, config)
    
    # Replace global instance
    _global_backend = new_backend
    
    return new_backend

def get_backend_info() -> Dict[str, Any]:
    """
    Get information about the current backend.
    
    Returns:
        Dictionary with backend information
    """
    backend = get_configured_backend()
    
    return {
        'backend_type': backend.config.backend_type.value,
        'backend_class': backend.__class__.__name__,
        'storage_directory': backend.config.storage_directory,
        'embedding_dimension': backend.config.embedding_dimension,
        'max_memory_chunks': backend.config.max_memory_chunks,
        'similarity_threshold': backend.config.similarity_threshold,
        'auto_cleanup_enabled': backend.config.auto_cleanup_enabled,
        'backend_stats': backend.get_memory_stats()
    }

def reset_backend() -> None:
    """Reset the global backend instance (for testing)."""
    global _global_backend
    _global_backend = None
    logger.info("Global backend instance reset")

def compare_backends(query: str = "test query", 
                    content: str = "test content") -> Dict[str, Any]:
    """
    Compare performance between different backends.
    
    Args:
        query: Test query for search performance
        content: Test content for add performance
        
    Returns:
        Comparison results
    """
    import time
    
    results = {}
    
    for backend_type in [BackendType.SAM_NATIVE, BackendType.MEM0]:
        try:
            # Create backend
            config = MemoryBackendConfig(backend_type=backend_type)
            backend = create_memory_backend(backend_type, config)
            
            # Test add performance
            start_time = time.time()
            chunk_id = backend.add_memory(
                content=content,
                memory_type="TEST",
                source="comparison",
                tags=["test"]
            )
            add_time = time.time() - start_time
            
            # Test search performance
            start_time = time.time()
            search_results = backend.search_memories(query, limit=5)
            search_time = time.time() - start_time
            
            # Test get performance
            start_time = time.time()
            retrieved = backend.get_memory(chunk_id)
            get_time = time.time() - start_time
            
            # Cleanup
            backend.delete_memory(chunk_id)
            
            results[backend_type.value] = {
                'add_time_ms': round(add_time * 1000, 2),
                'search_time_ms': round(search_time * 1000, 2),
                'get_time_ms': round(get_time * 1000, 2),
                'search_results_count': len(search_results),
                'retrieval_success': retrieved is not None,
                'backend_stats': backend.get_memory_stats()
            }
            
        except Exception as e:
            results[backend_type.value] = {
                'error': str(e),
                'available': False
            }
    
    return results
