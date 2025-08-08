"""
Mem0 Memory Backend (Task 33, Phase 3)
=====================================

Integrates the mem0 library as a memory backend for SAM, providing
standardized memory operations through the mem0 API while maintaining
compatibility with SAM's memory interface.
"""

import logging
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

from .base_backend import BaseMemoryBackend, MemoryBackendConfig, MemorySearchResult, MemoryChunkData

logger = logging.getLogger(__name__)

class Mem0MemoryBackend(BaseMemoryBackend):
    """
    Mem0 memory backend that integrates the mem0 library.
    
    This backend provides SAM's memory interface while using mem0's
    standardized memory operations underneath.
    """
    
    def __init__(self, config: MemoryBackendConfig):
        """Initialize the Mem0 backend."""
        super().__init__(config)
        
        # Try to import and initialize mem0
        try:
            import mem0
            self._mem0_available = True
            
            # Initialize mem0 client with configuration
            mem0_config = config.backend_specific_config.get('mem0_config', {})
            
            # Default mem0 configuration
            default_config = {
                "vector_store": {
                    "provider": "chroma",
                    "config": {
                        "collection_name": "sam_memories",
                        "path": str(Path(config.storage_directory) / "mem0_chroma")
                    }
                },
                "embedder": {
                    "provider": "sentence_transformers",
                    "config": {
                        "model": "all-MiniLM-L6-v2"
                    }
                }
            }
            
            # Merge with user config
            final_config = {**default_config, **mem0_config}
            
            # Initialize mem0 client
            self._mem0_client = mem0.Memory(config=final_config)
            
            self.logger.info("Mem0 memory backend initialized successfully")
            
        except ImportError:
            self._mem0_available = False
            self._mem0_client = None
            self.logger.warning("mem0 library not available, backend will use fallback mode")
        except Exception as e:
            self._mem0_available = False
            self._mem0_client = None
            self.logger.error(f"Failed to initialize mem0: {e}")
    
    def _is_available(self) -> bool:
        """Check if mem0 backend is available."""
        return self._mem0_available and self._mem0_client is not None
    
    def add_memory(self, 
                   content: str,
                   memory_type: str,
                   source: str = "unknown",
                   tags: Optional[List[str]] = None,
                   importance_score: float = 0.5,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add memory using mem0."""
        if not self._is_available():
            raise RuntimeError("Mem0 backend not available")
        
        try:
            # Create unique chunk ID
            chunk_id = f"sam_{uuid.uuid4().hex[:12]}"
            
            # Prepare metadata for mem0
            mem0_metadata = {
                'chunk_id': chunk_id,
                'memory_type': memory_type,
                'source': source,
                'tags': tags or [],
                'importance_score': importance_score,
                'timestamp': datetime.now().isoformat(),
                'sam_metadata': metadata or {}
            }
            
            # Add to mem0 with user_id as chunk_id for retrieval
            result = self._mem0_client.add(
                messages=[{"role": "user", "content": content}],
                user_id=chunk_id,
                metadata=mem0_metadata
            )
            
            self.logger.debug(f"Added memory to mem0: {chunk_id}")
            return chunk_id
            
        except Exception as e:
            self.logger.error(f"Failed to add memory to mem0: {e}")
            raise
    
    def search_memories(self,
                       query: str,
                       limit: int = 10,
                       similarity_threshold: float = None,
                       memory_types: Optional[List[str]] = None,
                       tags: Optional[List[str]] = None) -> List[MemorySearchResult]:
        """Search memories using mem0."""
        if not self._is_available():
            return []
        
        try:
            # Search using mem0
            search_results = self._mem0_client.search(
                query=query,
                limit=limit,
                user_id=None  # Search across all memories
            )
            
            results = []
            for result in search_results:
                # Extract metadata
                metadata = result.get('metadata', {})
                
                # Apply filters
                if memory_types and metadata.get('memory_type') not in memory_types:
                    continue
                
                if tags:
                    result_tags = metadata.get('tags', [])
                    if not any(tag in result_tags for tag in tags):
                        continue
                
                # Apply similarity threshold
                score = result.get('score', 0.0)
                if similarity_threshold and score < similarity_threshold:
                    continue
                
                # Create standardized result
                standardized_result = MemorySearchResult(
                    chunk_id=metadata.get('chunk_id', f"mem0_{result.get('id', 'unknown')}"),
                    content=result.get('memory', ''),
                    similarity_score=score,
                    memory_type=metadata.get('memory_type', 'UNKNOWN'),
                    source=metadata.get('source', 'mem0'),
                    timestamp=metadata.get('timestamp', datetime.now().isoformat()),
                    tags=metadata.get('tags', []),
                    importance_score=metadata.get('importance_score', 0.5),
                    metadata=metadata.get('sam_metadata', {})
                )
                results.append(standardized_result)
            
            return results[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to search memories in mem0: {e}")
            return []
    
    def get_memory(self, chunk_id: str) -> Optional[MemoryChunkData]:
        """Retrieve specific memory using mem0."""
        if not self._is_available():
            return None
        
        try:
            # Get all memories for this user_id (chunk_id)
            memories = self._mem0_client.get_all(user_id=chunk_id)
            
            if not memories:
                return None
            
            # Find the specific memory (should be only one)
            memory = memories[0] if memories else None
            if not memory:
                return None
            
            metadata = memory.get('metadata', {})
            
            return MemoryChunkData(
                chunk_id=chunk_id,
                content=memory.get('memory', ''),
                content_hash=f"mem0_{memory.get('id', 'unknown')}",
                memory_type=metadata.get('memory_type', 'UNKNOWN'),
                source=metadata.get('source', 'mem0'),
                timestamp=metadata.get('timestamp', datetime.now().isoformat()),
                tags=metadata.get('tags', []),
                importance_score=metadata.get('importance_score', 0.5),
                access_count=0,  # mem0 doesn't track access count
                last_accessed=datetime.now().isoformat(),
                metadata=metadata.get('sam_metadata', {})
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get memory {chunk_id} from mem0: {e}")
            return None
    
    def update_memory(self,
                     chunk_id: str,
                     content: Optional[str] = None,
                     tags: Optional[List[str]] = None,
                     importance_score: Optional[float] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update memory using mem0."""
        if not self._is_available():
            return False
        
        try:
            # Get existing memory
            existing = self.get_memory(chunk_id)
            if not existing:
                return False
            
            # mem0 doesn't have direct update, so we need to delete and re-add
            # This is a limitation of the current mem0 API
            
            # Delete existing
            self._mem0_client.delete_all(user_id=chunk_id)
            
            # Re-add with updates
            updated_content = content or existing.content
            updated_tags = tags or existing.tags
            updated_importance = importance_score or existing.importance_score
            updated_metadata = metadata or existing.metadata
            
            self.add_memory(
                content=updated_content,
                memory_type=existing.memory_type,
                source=existing.source,
                tags=updated_tags,
                importance_score=updated_importance,
                metadata=updated_metadata
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update memory {chunk_id} in mem0: {e}")
            return False
    
    def delete_memory(self, chunk_id: str) -> bool:
        """Delete memory using mem0."""
        if not self._is_available():
            return False
        
        try:
            self._mem0_client.delete_all(user_id=chunk_id)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete memory {chunk_id} from mem0: {e}")
            return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics from mem0 backend."""
        try:
            # mem0 doesn't provide direct stats, so we estimate
            return {
                'backend_type': 'mem0',
                'mem0_available': self._is_available(),
                'storage_directory': self.config.storage_directory,
                'embedding_dimension': self.config.embedding_dimension,
                'note': 'mem0 backend statistics are limited by API capabilities'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get mem0 stats: {e}")
            return {'backend_type': 'mem0', 'error': str(e)}
    
    def cleanup_old_memories(self, days_threshold: int = None) -> int:
        """Clean up old memories (limited by mem0 API)."""
        if not self._is_available():
            return 0
        
        # mem0 doesn't provide bulk cleanup operations
        # This would need to be implemented by iterating through memories
        self.logger.warning("Cleanup not fully supported by mem0 backend")
        return 0
    
    def export_memories(self, 
                       output_path: str,
                       memory_types: Optional[List[str]] = None,
                       date_range: Optional[Tuple[str, str]] = None) -> bool:
        """Export memories (limited by mem0 API)."""
        if not self._is_available():
            return False
        
        try:
            # This would need custom implementation to iterate through all memories
            self.logger.warning("Export not fully implemented for mem0 backend")
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to export from mem0: {e}")
            return False
    
    def import_memories(self, input_path: str) -> int:
        """Import memories (limited by mem0 API)."""
        if not self._is_available():
            return 0
        
        try:
            # This would need custom implementation
            self.logger.warning("Import not fully implemented for mem0 backend")
            return 0
            
        except Exception as e:
            self.logger.error(f"Failed to import to mem0: {e}")
            return 0
