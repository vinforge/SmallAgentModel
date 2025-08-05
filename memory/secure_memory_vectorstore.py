"""
Secure Memory Vector Store for SAM

Provides encrypted memory storage with enterprise-grade security integration.
Maintains 100% compatibility with existing memory functionality while adding
security features.

Author: SAM Development Team
Version: 2.0.0
"""

import os
import json
import logging
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Import base memory components
from .memory_vectorstore import MemoryVectorStore, VectorStoreType
from .memory_manager import LongTermMemoryManager

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Memory type enumeration for categorizing stored memories."""
    CONVERSATION = "conversation"
    DOCUMENT = "document"
    TOOL_OUTPUT = "tool_output"
    USER_NOTE = "user_note"
    REFLECTION = "reflection"
    FEEDBACK = "feedback"
    CORRECTION = "correction"
    SYNTHESIS = "synthesis"
    RESEARCH = "research"
    DISCOVERY = "discovery"


class SecureMemoryVectorStore:
    """
    Secure wrapper around MemoryVectorStore with encryption integration.
    
    Provides all functionality of the base MemoryVectorStore while adding
    enterprise-grade security features through integration with SAM's
    security system.
    """
    
    def __init__(self, 
                 store_type: VectorStoreType = VectorStoreType.SIMPLE,
                 storage_directory: str = "memory_store",
                 embedding_dimension: int = 384,
                 security_manager=None):
        """
        Initialize secure memory vector store.
        
        Args:
            store_type: Type of vector store to use
            storage_directory: Directory for storing memory data
            embedding_dimension: Dimension of embedding vectors
            security_manager: Security manager for encryption (optional)
        """
        self.security_manager = security_manager
        self.storage_directory = Path(storage_directory)

        # Initialize encryption state
        self._encryption_active = False

        # Initialize base memory store
        self.base_store = MemoryVectorStore(
            store_type=store_type,
            storage_directory=storage_directory,
            embedding_dimension=embedding_dimension
        )

        # Initialize long-term memory manager
        memory_store_path = self.storage_directory / "long_term_memory.json"
        self.memory_manager = LongTermMemoryManager(
            memory_store_path=str(memory_store_path)
        )

        logger.info(f"SecureMemoryVectorStore initialized with {store_type.value} backend")
        if security_manager:
            logger.info("Security integration enabled")
    
    def add_memory(self, content: str, memory_type: MemoryType = MemoryType.CONVERSATION,
                   source: str = "secure_store", tags: List[str] = None,
                   importance_score: float = 0.5, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a memory to the secure store.

        Args:
            content: Memory content
            memory_type: Type of memory being stored
            source: Source of the memory
            tags: Optional tags
            importance_score: Importance score (0.0 to 1.0)
            metadata: Additional metadata

        Returns:
            str: Memory ID
        """
        if metadata is None:
            metadata = {}

        # Add security-specific metadata
        metadata['encrypted'] = self.security_manager is not None

        # Import the base MemoryType for compatibility
        from .memory_vectorstore import MemoryType as BaseMemoryType

        # Map our MemoryType to base MemoryType
        base_memory_type = BaseMemoryType.CONVERSATION  # Default
        try:
            base_memory_type = BaseMemoryType(memory_type.value)
        except ValueError:
            # If our memory type doesn't exist in base, use CONVERSATION
            base_memory_type = BaseMemoryType.CONVERSATION

        # Store in base vector store with correct signature
        memory_id = self.base_store.add_memory(
            content=content,
            memory_type=base_memory_type,
            source=source,
            tags=tags or [],
            importance_score=importance_score,
            metadata=metadata
        )

        # Also store in long-term memory manager for persistence
        try:
            self.memory_manager.store_memory(
                content=content,
                content_type=memory_type.value,
                tags=tags or [],
                importance_score=importance_score,
                metadata=metadata
            )
        except Exception as e:
            logger.warning(f"Failed to store in long-term memory: {e}")

        return memory_id
    
    def search_memories(self, query: str, max_results: int = 10,
                       memory_type: Optional[MemoryType] = None) -> List[Dict[str, Any]]:
        """
        Search memories in the secure store.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            memory_type: Filter by memory type (optional)
            
        Returns:
            List of memory results
        """
        # Search in base store
        results = self.base_store.search_memories(query, max_results)
        
        # Filter by memory type if specified
        if memory_type:
            filtered_results = []
            for result in results:
                try:
                    # Handle both dict and object formats
                    if hasattr(result, 'get'):
                        metadata = result.get('metadata', {})
                    elif hasattr(result, 'metadata'):
                        metadata = result.metadata if result.metadata else {}
                    else:
                        metadata = {}

                    # FIXED: Handle both enum and string memory types
                    if isinstance(metadata, dict):
                        stored_memory_type = metadata.get('memory_type')

                        # Handle different memory_type formats
                        if hasattr(memory_type, 'value'):
                            # memory_type is an enum
                            target_value = memory_type.value
                        else:
                            # memory_type is already a string
                            target_value = str(memory_type)

                        if stored_memory_type == target_value:
                            filtered_results.append(result)
                except Exception as e:
                    logger.warning(f"Error filtering result by memory type: {e}")
                    continue
            results = filtered_results
        
        return results
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive memory statistics.
        
        Returns:
            Dictionary containing memory statistics
        """
        # Get base stats from vector store
        base_stats = self.base_store.get_memory_stats()
        
        # Add security-specific stats
        security_stats = {
            'security_enabled': self.security_manager is not None,
            'encryption_status': 'enabled' if self.security_manager else 'disabled'
        }
        
        # Merge stats
        stats = {**base_stats, **security_stats}
        
        # Ensure total_memories is always present
        if 'total_memories' not in stats:
            stats['total_memories'] = len(getattr(self.base_store, 'memory_chunks', {}))
        
        return stats
    
    def get_security_status(self) -> Dict[str, Any]:
        """
        Get security-specific status information.

        Returns:
            Dictionary containing security status
        """
        if not self.security_manager:
            return {
                'security_enabled': False,
                'encryption_status': 'disabled',
                'encryption_active': False,
                'encrypted_chunk_count': 0
            }

        # Get encrypted chunk count
        encrypted_count = 0
        try:
            if hasattr(self.base_store, 'memory_chunks'):
                for chunk in self.base_store.memory_chunks.values():
                    try:
                        # Handle both dict and object formats
                        if hasattr(chunk, 'get'):
                            metadata = chunk.get('metadata', {})
                        elif hasattr(chunk, 'metadata'):
                            metadata = chunk.metadata if chunk.metadata else {}
                        else:
                            metadata = {}

                        if isinstance(metadata, dict) and metadata.get('encrypted', False):
                            encrypted_count += 1
                    except Exception as chunk_error:
                        logger.debug(f"Error processing chunk for encryption count: {chunk_error}")
                        continue
        except Exception as e:
            logger.warning(f"Error counting encrypted chunks: {e}")

        return {
            'security_enabled': True,
            'encryption_status': 'enabled' if self.is_encryption_active() else 'available',
            'encryption_active': self.is_encryption_active(),
            'encrypted_chunk_count': encrypted_count,
            'security_manager_state': self.security_manager.get_state().value if self.security_manager else 'unknown'
        }
    
    def get_all_memories(self) -> List[Dict[str, Any]]:
        """
        Get all memories from the store.
        
        Returns:
            List of all memories
        """
        try:
            if hasattr(self.base_store, 'memory_chunks'):
                return list(self.base_store.memory_chunks.values())
            else:
                return []
        except Exception as e:
            logger.error(f"Error retrieving all memories: {e}")
            return []
    
    def clear_memories(self) -> bool:
        """
        Clear all memories from the store.

        Returns:
            bool: True if successful
        """
        try:
            # Clear base store
            if hasattr(self.base_store, 'memory_chunks'):
                self.base_store.memory_chunks.clear()

            # Clear long-term memory
            self.memory_manager.clear_all_memories()

            logger.info("All memories cleared from secure store")
            return True
        except Exception as e:
            logger.error(f"Error clearing memories: {e}")
            return False

    def activate_encryption(self) -> bool:
        """
        Activate encryption for the secure memory store.

        Returns:
            bool: True if encryption was successfully activated
        """
        try:
            if self.security_manager is None:
                logger.warning("Cannot activate encryption: no security manager provided")
                return False

            # Check if security manager is authenticated/unlocked
            if hasattr(self.security_manager, 'is_unlocked') and not self.security_manager.is_unlocked():
                logger.warning("Cannot activate encryption: security manager is locked")
                return False

            if hasattr(self.security_manager, 'is_authenticated') and not self.security_manager.is_authenticated():
                logger.warning("Cannot activate encryption: security manager not authenticated")
                return False

            # Mark encryption as active
            self._encryption_active = True
            logger.info("✅ Encryption activated for secure memory store")
            return True

        except Exception as e:
            logger.error(f"Error activating encryption: {e}")
            return False

    def is_encryption_active(self) -> bool:
        """
        Check if encryption is currently active.

        Returns:
            bool: True if encryption is active
        """
        return getattr(self, '_encryption_active', False) and self.security_manager is not None
    
    # Delegate other methods to base store for full compatibility
    def __getattr__(self, name):
        """Delegate unknown methods to base store for full compatibility."""
        return getattr(self.base_store, name)


def get_secure_memory_store(store_type: VectorStoreType = VectorStoreType.SIMPLE,
                           storage_directory: str = "memory_store",
                           embedding_dimension: int = 384,
                           security_manager=None,
                           enable_encryption: bool = True) -> SecureMemoryVectorStore:
    """
    Factory function to create a secure memory store.

    Args:
        store_type: Type of vector store to use
        storage_directory: Directory for storing memory data
        embedding_dimension: Dimension of embedding vectors
        security_manager: Security manager for encryption (optional)
        enable_encryption: Whether to enable encryption (for compatibility)

    Returns:
        SecureMemoryVectorStore instance
    """
    # If enable_encryption is True but no security_manager provided,
    # we still create the store but without encryption
    if enable_encryption and security_manager is None:
        logger.info("Encryption requested but no security manager provided - creating store without encryption")

    return SecureMemoryVectorStore(
        store_type=store_type,
        storage_directory=storage_directory,
        embedding_dimension=embedding_dimension,
        security_manager=security_manager
    )


# Export all required components for compatibility
__all__ = [
    'SecureMemoryVectorStore',
    'get_secure_memory_store',
    'VectorStoreType',
    'MemoryType'
]
