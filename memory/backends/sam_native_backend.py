"""
SAM Native Memory Backend (Task 33, Phase 3)
===========================================

Wraps SAM's existing custom SQLite and JSON memory logic in the standardized
backend interface. This maintains full compatibility with existing functionality
while enabling backend abstraction.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

from .base_backend import BaseMemoryBackend, MemoryBackendConfig, MemorySearchResult, MemoryChunkData
from ..memory_vectorstore import MemoryVectorStore, VectorStoreType, MemoryType, MemoryChunk

logger = logging.getLogger(__name__)

class SAMNativeMemoryBackend(BaseMemoryBackend):
    """
    SAM Native memory backend that wraps the existing MemoryVectorStore.
    
    This backend provides the standardized interface while using SAM's
    proven custom memory implementation underneath.
    """
    
    def __init__(self, config: MemoryBackendConfig):
        """Initialize the SAM Native backend."""
        super().__init__(config)
        
        # Initialize the underlying MemoryVectorStore
        store_type = config.backend_specific_config.get('store_type', VectorStoreType.SIMPLE)
        
        self._native_store = MemoryVectorStore(
            store_type=store_type,
            storage_directory=config.storage_directory,
            embedding_dimension=config.embedding_dimension
        )
        
        # Update native store config with our settings
        self._native_store.config.update({
            'max_memory_chunks': config.max_memory_chunks,
            'similarity_threshold': config.similarity_threshold,
            'auto_cleanup_enabled': config.auto_cleanup_enabled,
            'cleanup_threshold_days': config.cleanup_threshold_days
        })
        
        self.logger.info("SAM Native memory backend initialized")
    
    def add_memory(self, 
                   content: str,
                   memory_type: str,
                   source: str = "unknown",
                   tags: Optional[List[str]] = None,
                   importance_score: float = 0.5,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add memory using native MemoryVectorStore."""
        try:
            # Convert string memory_type to MemoryType enum
            try:
                mem_type = MemoryType[memory_type.upper()]
            except (KeyError, AttributeError):
                # Default to FACT if unknown type
                mem_type = MemoryType.FACT
                self.logger.warning(f"Unknown memory type '{memory_type}', using FACT")
            
            # Add to native store
            chunk_id = self._native_store.add_memory(
                content=content,
                memory_type=mem_type,
                source=source,
                tags=tags or [],
                importance_score=importance_score,
                metadata=metadata or {}
            )
            
            return chunk_id
            
        except Exception as e:
            self.logger.error(f"Failed to add memory: {e}")
            raise
    
    def search_memories(self,
                       query: str,
                       limit: int = 10,
                       similarity_threshold: float = None,
                       memory_types: Optional[List[str]] = None,
                       tags: Optional[List[str]] = None) -> List[MemorySearchResult]:
        """Search memories using native MemoryVectorStore."""
        try:
            # Use provided threshold or backend default
            threshold = similarity_threshold or self.config.similarity_threshold
            
            # Convert memory_types to MemoryType enums if provided
            mem_type_filters = None
            if memory_types:
                mem_type_filters = []
                for mt in memory_types:
                    try:
                        mem_type_filters.append(MemoryType[mt.upper()])
                    except KeyError:
                        self.logger.warning(f"Unknown memory type filter: {mt}")
            
            # Search using native store
            native_results = self._native_store.search_memories(
                query=query,
                limit=limit,
                similarity_threshold=threshold,
                memory_types=mem_type_filters,
                tags=tags
            )
            
            # Convert to standardized format
            results = []
            for result in native_results:
                standardized_result = MemorySearchResult(
                    chunk_id=result.chunk_id,
                    content=result.content,
                    similarity_score=result.similarity_score,
                    memory_type=result.memory_type.value if hasattr(result, 'memory_type') else 'UNKNOWN',
                    source=result.source if hasattr(result, 'source') else 'unknown',
                    timestamp=result.timestamp if hasattr(result, 'timestamp') else datetime.now().isoformat(),
                    tags=result.tags if hasattr(result, 'tags') else [],
                    importance_score=result.importance_score if hasattr(result, 'importance_score') else 0.5,
                    metadata=result.metadata if hasattr(result, 'metadata') else {}
                )
                results.append(standardized_result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search memories: {e}")
            return []
    
    def get_memory(self, chunk_id: str) -> Optional[MemoryChunkData]:
        """Retrieve specific memory using native MemoryVectorStore."""
        try:
            # Get from native store
            chunk = self._native_store.get_memory(chunk_id)
            
            if not chunk:
                return None
            
            # Convert to standardized format
            return MemoryChunkData(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                content_hash=chunk.content_hash,
                memory_type=chunk.memory_type.value,
                source=chunk.source,
                timestamp=chunk.timestamp,
                tags=chunk.tags,
                importance_score=chunk.importance_score,
                access_count=chunk.access_count,
                last_accessed=chunk.last_accessed,
                metadata=chunk.metadata
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get memory {chunk_id}: {e}")
            return None
    
    def update_memory(self,
                     chunk_id: str,
                     content: Optional[str] = None,
                     tags: Optional[List[str]] = None,
                     importance_score: Optional[float] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update memory using native MemoryVectorStore."""
        try:
            return self._native_store.update_memory(
                chunk_id=chunk_id,
                content=content,
                tags=tags,
                importance_score=importance_score,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update memory {chunk_id}: {e}")
            return False
    
    def delete_memory(self, chunk_id: str) -> bool:
        """Delete memory using native MemoryVectorStore."""
        try:
            return self._native_store.delete_memory(chunk_id)
            
        except Exception as e:
            self.logger.error(f"Failed to delete memory {chunk_id}: {e}")
            return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics from native MemoryVectorStore."""
        try:
            native_stats = self._native_store.get_statistics()
            
            # Standardize the stats format
            return {
                'backend_type': 'sam_native',
                'total_memories': len(self._native_store.memory_chunks),
                'storage_directory': str(self._native_store.storage_dir),
                'embedding_dimension': self._native_store.embedding_dimension,
                'store_type': self._native_store.store_type.value,
                'native_stats': native_stats
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get memory stats: {e}")
            return {'backend_type': 'sam_native', 'error': str(e)}
    
    def cleanup_old_memories(self, days_threshold: int = None) -> int:
        """Clean up old memories using native MemoryVectorStore."""
        try:
            threshold = days_threshold or self.config.cleanup_threshold_days
            return self._native_store.cleanup_old_memories(days_threshold=threshold)
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup memories: {e}")
            return 0
    
    def export_memories(self, 
                       output_path: str,
                       memory_types: Optional[List[str]] = None,
                       date_range: Optional[Tuple[str, str]] = None) -> bool:
        """Export memories using native MemoryVectorStore."""
        try:
            # Convert memory_types if provided
            mem_type_filters = None
            if memory_types:
                mem_type_filters = []
                for mt in memory_types:
                    try:
                        mem_type_filters.append(MemoryType[mt.upper()])
                    except KeyError:
                        self.logger.warning(f"Unknown memory type for export: {mt}")
            
            return self._native_store.export_memories(
                output_path=output_path,
                memory_types=mem_type_filters,
                date_range=date_range
            )
            
        except Exception as e:
            self.logger.error(f"Failed to export memories: {e}")
            return False
    
    def import_memories(self, input_path: str) -> int:
        """Import memories using native MemoryVectorStore."""
        try:
            return self._native_store.import_memories(input_path)
            
        except Exception as e:
            self.logger.error(f"Failed to import memories: {e}")
            return 0
    
    def get_native_store(self) -> MemoryVectorStore:
        """Get access to the underlying native store for advanced operations."""
        return self._native_store
