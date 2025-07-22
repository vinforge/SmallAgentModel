"""
Base Memory Backend Interface (Task 33, Phase 3)
===============================================

Defines the abstract interface that all SAM memory backends must implement.
This ensures compatibility and enables seamless backend switching.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class BackendType(Enum):
    """Supported memory backend types."""
    SAM_NATIVE = "sam_native"
    MEM0 = "mem0"

@dataclass
class MemoryBackendConfig:
    """Configuration for memory backends."""
    backend_type: BackendType
    storage_directory: str = "memory_store"
    embedding_dimension: int = 384
    max_memory_chunks: int = 10000
    similarity_threshold: float = 0.1
    auto_cleanup_enabled: bool = True
    cleanup_threshold_days: int = 90
    backend_specific_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.backend_specific_config is None:
            self.backend_specific_config = {}

@dataclass
class MemorySearchResult:
    """Standardized memory search result."""
    chunk_id: str
    content: str
    similarity_score: float
    memory_type: str
    source: str
    timestamp: str
    tags: List[str]
    importance_score: float
    metadata: Dict[str, Any]

@dataclass
class MemoryChunkData:
    """Standardized memory chunk data."""
    chunk_id: str
    content: str
    content_hash: str
    memory_type: str
    source: str
    timestamp: str
    tags: List[str]
    importance_score: float
    access_count: int
    last_accessed: str
    metadata: Dict[str, Any]

class BaseMemoryBackend(ABC):
    """
    Abstract base class for all SAM memory backends.
    
    This interface defines the required methods that any memory backend
    must implement to be compatible with SAM's memory system.
    """
    
    def __init__(self, config: MemoryBackendConfig):
        """Initialize the memory backend with configuration."""
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(f"Initializing {config.backend_type.value} memory backend")
    
    @abstractmethod
    def add_memory(self, 
                   content: str,
                   memory_type: str,
                   source: str = "unknown",
                   tags: Optional[List[str]] = None,
                   importance_score: float = 0.5,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a new memory to the backend.
        
        Args:
            content: The content to store
            memory_type: Type of memory (e.g., 'CONVERSATION', 'RESEARCH')
            source: Source of the memory
            tags: Optional tags for categorization
            importance_score: Importance score (0.0 to 1.0)
            metadata: Additional metadata
            
        Returns:
            Unique chunk ID for the stored memory
        """
        pass
    
    @abstractmethod
    def search_memories(self,
                       query: str,
                       limit: int = 10,
                       similarity_threshold: float = None,
                       memory_types: Optional[List[str]] = None,
                       tags: Optional[List[str]] = None) -> List[MemorySearchResult]:
        """
        Search for memories using semantic similarity.
        
        Args:
            query: Search query
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            memory_types: Filter by memory types
            tags: Filter by tags
            
        Returns:
            List of matching memory search results
        """
        pass
    
    @abstractmethod
    def get_memory(self, chunk_id: str) -> Optional[MemoryChunkData]:
        """
        Retrieve a specific memory by ID.
        
        Args:
            chunk_id: Unique chunk identifier
            
        Returns:
            Memory chunk data or None if not found
        """
        pass
    
    @abstractmethod
    def update_memory(self,
                     chunk_id: str,
                     content: Optional[str] = None,
                     tags: Optional[List[str]] = None,
                     importance_score: Optional[float] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an existing memory.
        
        Args:
            chunk_id: Unique chunk identifier
            content: New content (if updating)
            tags: New tags (if updating)
            importance_score: New importance score (if updating)
            metadata: New metadata (if updating)
            
        Returns:
            True if update successful, False otherwise
        """
        pass
    
    @abstractmethod
    def delete_memory(self, chunk_id: str) -> bool:
        """
        Delete a memory from the backend.
        
        Args:
            chunk_id: Unique chunk identifier
            
        Returns:
            True if deletion successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory backend.
        
        Returns:
            Dictionary with backend statistics
        """
        pass
    
    @abstractmethod
    def cleanup_old_memories(self, days_threshold: int = None) -> int:
        """
        Clean up old or low-importance memories.
        
        Args:
            days_threshold: Age threshold in days
            
        Returns:
            Number of memories cleaned up
        """
        pass
    
    @abstractmethod
    def export_memories(self, 
                       output_path: str,
                       memory_types: Optional[List[str]] = None,
                       date_range: Optional[Tuple[str, str]] = None) -> bool:
        """
        Export memories to external format.
        
        Args:
            output_path: Path for export file
            memory_types: Filter by memory types
            date_range: Date range filter (start, end)
            
        Returns:
            True if export successful, False otherwise
        """
        pass
    
    @abstractmethod
    def import_memories(self, input_path: str) -> int:
        """
        Import memories from external format.
        
        Args:
            input_path: Path to import file
            
        Returns:
            Number of memories imported
        """
        pass
    
    def validate_backend(self) -> Tuple[bool, List[str]]:
        """
        Validate backend functionality.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            # Test basic operations
            test_id = self.add_memory(
                content="Backend validation test",
                memory_type="TEST",
                source="validation",
                tags=["test"],
                importance_score=0.5
            )
            
            # Test retrieval
            retrieved = self.get_memory(test_id)
            if not retrieved:
                issues.append("Failed to retrieve test memory")
            
            # Test search
            search_results = self.search_memories("validation test", limit=1)
            if not search_results:
                issues.append("Failed to search test memory")
            
            # Test update
            if not self.update_memory(test_id, tags=["test", "updated"]):
                issues.append("Failed to update test memory")
            
            # Test deletion
            if not self.delete_memory(test_id):
                issues.append("Failed to delete test memory")
            
            # Test stats
            stats = self.get_memory_stats()
            if not isinstance(stats, dict):
                issues.append("Failed to get backend statistics")
                
        except Exception as e:
            issues.append(f"Backend validation exception: {e}")
        
        return len(issues) == 0, issues
