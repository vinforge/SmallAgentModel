"""
Long-Term Memory Management for SAM
Enables persistent knowledge storage, retrieval, and management across sessions.

Sprint 6 Task 1: Long-Term Memory Management
"""

import logging
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class MemoryEntry:
    """Represents a single memory entry in SAM's long-term memory."""
    memory_id: str
    content: str
    summary: str
    embedding: Optional[np.ndarray]
    metadata: Dict[str, Any]
    created_at: str
    last_accessed: str
    access_count: int
    importance_score: float
    tags: List[str]
    content_type: str  # 'conversation', 'tool_output', 'user_note', 'reflection'
    project: Optional[str]
    ttl_expires: Optional[str]  # Time-to-live expiration

@dataclass
class MemorySearchResult:
    """Result from memory search operations."""
    memory_entry: MemoryEntry
    similarity_score: float
    relevance_reason: str

class LongTermMemoryManager:
    """
    Manages SAM's long-term memory storage, retrieval, and maintenance.
    """
    
    def __init__(self, memory_store_path: str = "memory_store.json", 
                 embedding_manager=None, enable_ttl: bool = True):
        """
        Initialize the long-term memory manager.
        
        Args:
            memory_store_path: Path to persistent memory storage file
            embedding_manager: Embedding manager for vectorizing memories
            enable_ttl: Enable time-to-live memory decay
        """
        self.memory_store_path = Path(memory_store_path)
        self.embedding_manager = embedding_manager
        self.enable_ttl = enable_ttl
        
        # In-memory storage for fast access
        self.memories: Dict[str, MemoryEntry] = {}
        
        # Memory configuration
        self.config = {
            'max_memories': 10000,
            'default_ttl_days': 365,  # 1 year default TTL
            'archive_threshold_days': 90,  # Archive after 90 days of no access
            'importance_decay_rate': 0.95,  # Daily importance decay
            'min_importance_threshold': 0.1
        }
        
        # Load existing memories
        self._load_memories()
        
        logger.info(f"Long-term memory manager initialized with {len(self.memories)} memories")
    
    def store_memory(self, content: str, content_type: str = 'conversation',
                    tags: Optional[List[str]] = None, project: Optional[str] = None,
                    importance_score: float = 0.5, ttl_days: Optional[int] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a new memory entry.
        
        Args:
            content: The content to store
            content_type: Type of content ('conversation', 'tool_output', 'user_note', 'reflection')
            tags: List of tags for categorization
            project: Project name for organization
            importance_score: Initial importance score (0.0 to 1.0)
            ttl_days: Time-to-live in days (None for default)
            metadata: Additional metadata
            
        Returns:
            Memory ID of the stored entry
        """
        try:
            # Generate unique memory ID
            memory_id = f"mem_{uuid.uuid4().hex[:12]}"
            
            # Create summary (first 150 chars or custom summary)
            summary = content[:150] + "..." if len(content) > 150 else content
            
            # Generate embedding if embedding manager available
            embedding = None
            if self.embedding_manager:
                try:
                    embedding = self.embedding_manager.embed(content)
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for memory: {e}")
            
            # Calculate TTL expiration
            ttl_expires = None
            if self.enable_ttl:
                ttl_days = ttl_days or self.config['default_ttl_days']
                ttl_expires = (datetime.now() + timedelta(days=ttl_days)).isoformat()
            
            # Create memory entry
            memory_entry = MemoryEntry(
                memory_id=memory_id,
                content=content,
                summary=summary,
                embedding=embedding,
                metadata=metadata or {},
                created_at=datetime.now().isoformat(),
                last_accessed=datetime.now().isoformat(),
                access_count=1,
                importance_score=importance_score,
                tags=tags or [],
                content_type=content_type,
                project=project,
                ttl_expires=ttl_expires
            )
            
            # Store in memory
            self.memories[memory_id] = memory_entry
            
            # Persist to disk
            self._save_memories()
            
            logger.info(f"Stored memory: {memory_id} ({content_type})")
            return memory_id
            
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            raise
    
    def recall_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """
        Recall a specific memory by ID.
        
        Args:
            memory_id: ID of the memory to recall
            
        Returns:
            MemoryEntry if found, None otherwise
        """
        try:
            memory = self.memories.get(memory_id)
            
            if memory:
                # Update access statistics
                memory.last_accessed = datetime.now().isoformat()
                memory.access_count = int(memory.access_count) + 1 if isinstance(memory.access_count, (str, int)) else 1
                
                # Boost importance slightly on access
                memory.importance_score = min(1.0, memory.importance_score * 1.05)
                
                logger.debug(f"Recalled memory: {memory_id}")
                return memory
            
            return None
            
        except Exception as e:
            logger.error(f"Error recalling memory {memory_id}: {e}")
            return None
    
    def search_memories(self, query: str, content_type: Optional[str] = None,
                       tags: Optional[List[str]] = None, project: Optional[str] = None,
                       top_k: int = 5, similarity_threshold: float = 0.3) -> List[MemorySearchResult]:
        """
        Search memories using semantic similarity and filters.
        
        Args:
            query: Search query
            content_type: Filter by content type
            tags: Filter by tags (any match)
            project: Filter by project
            top_k: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of MemorySearchResult objects
        """
        try:
            results = []
            
            # Generate query embedding if available
            query_embedding = None
            if self.embedding_manager:
                try:
                    query_embedding = self.embedding_manager.embed_query(query)
                except Exception as e:
                    logger.warning(f"Failed to generate query embedding: {e}")
            
            # Search through memories
            for memory in self.memories.values():
                # Apply filters
                if content_type and memory.content_type != content_type:
                    continue
                
                if tags and not any(tag in memory.tags for tag in tags):
                    continue
                
                if project and memory.project != project:
                    continue
                
                # Check TTL expiration
                if self._is_memory_expired(memory):
                    continue
                
                # Calculate similarity
                similarity_score = 0.0
                relevance_reason = "keyword match"
                
                if query_embedding is not None and memory.embedding is not None:
                    # Semantic similarity
                    similarity_score = np.dot(query_embedding, memory.embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(memory.embedding)
                    )
                    relevance_reason = "semantic similarity"
                else:
                    # Fallback to keyword matching
                    query_words = set(query.lower().split())
                    content_words = set(memory.content.lower().split())
                    summary_words = set(memory.summary.lower().split())
                    tag_words = set(' '.join(memory.tags).lower().split())
                    
                    all_memory_words = content_words | summary_words | tag_words
                    matches = len(query_words & all_memory_words)
                    similarity_score = matches / len(query_words) if query_words else 0
                    relevance_reason = f"keyword match ({matches} words)"
                
                # Apply importance boost
                similarity_score *= (0.5 + 0.5 * memory.importance_score)
                
                if similarity_score >= similarity_threshold:
                    results.append(MemorySearchResult(
                        memory_entry=memory,
                        similarity_score=similarity_score,
                        relevance_reason=relevance_reason
                    ))
            
            # Sort by similarity score and return top results
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Update access statistics for returned memories
            for result in results[:top_k]:
                result.memory_entry.last_accessed = datetime.now().isoformat()
                result.memory_entry.access_count += 1
            
            logger.debug(f"Memory search for '{query}': {len(results[:top_k])} results")
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []
    
    def pin_memory(self, memory_id: str, importance_score: float = 1.0) -> bool:
        """
        Pin a memory to prevent decay and increase importance.
        
        Args:
            memory_id: ID of memory to pin
            importance_score: New importance score
            
        Returns:
            True if successful, False otherwise
        """
        try:
            memory = self.memories.get(memory_id)
            if memory:
                memory.importance_score = importance_score
                memory.ttl_expires = None  # Remove TTL for pinned memories
                memory.tags = list(set(memory.tags + ['pinned']))
                
                self._save_memories()
                logger.info(f"Pinned memory: {memory_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error pinning memory {memory_id}: {e}")
            return False
    
    def get_recent_memories(self, days: int = 7, limit: int = 20) -> List[MemoryEntry]:
        """
        Get recent memories from the last N days.
        
        Args:
            days: Number of days to look back
            limit: Maximum number of memories to return
            
        Returns:
            List of recent MemoryEntry objects
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_memories = []
            
            for memory in self.memories.values():
                created_date = datetime.fromisoformat(memory.created_at)
                if created_date >= cutoff_date and not self._is_memory_expired(memory):
                    recent_memories.append(memory)
            
            # Sort by creation date (newest first)
            recent_memories.sort(key=lambda x: x.created_at, reverse=True)
            
            return recent_memories[:limit]
            
        except Exception as e:
            logger.error(f"Error getting recent memories: {e}")
            return []
    
    def cleanup_expired_memories(self) -> int:
        """
        Remove expired memories and archive old ones.
        
        Returns:
            Number of memories cleaned up
        """
        try:
            cleaned_count = 0
            current_time = datetime.now()
            
            expired_ids = []
            
            for memory_id, memory in self.memories.items():
                # Check TTL expiration
                if self._is_memory_expired(memory):
                    expired_ids.append(memory_id)
                    continue
                
                # Check for archival (not accessed in archive_threshold_days)
                last_accessed = datetime.fromisoformat(memory.last_accessed)
                days_since_access = (current_time - last_accessed).days
                
                if (days_since_access > self.config['archive_threshold_days'] and 
                    memory.importance_score < self.config['min_importance_threshold'] and
                    'pinned' not in memory.tags):
                    expired_ids.append(memory_id)
            
            # Remove expired memories
            for memory_id in expired_ids:
                del self.memories[memory_id]
                cleaned_count += 1
            
            if cleaned_count > 0:
                self._save_memories()
                logger.info(f"Cleaned up {cleaned_count} expired memories")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up memories: {e}")
            return 0
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory store."""
        try:
            stats = {
                'total_memories': len(self.memories),
                'content_types': {},
                'projects': set(),
                'tags': {},
                'average_importance': 0.0,
                'total_access_count': 0,
                'expired_count': 0
            }
            
            for memory in self.memories.values():
                # Content types
                content_type = memory.content_type
                stats['content_types'][content_type] = stats['content_types'].get(content_type, 0) + 1
                
                # Projects
                if memory.project:
                    stats['projects'].add(memory.project)
                
                # Tags
                for tag in memory.tags:
                    stats['tags'][tag] = stats['tags'].get(tag, 0) + 1
                
                # Importance and access
                stats['average_importance'] += memory.importance_score
                stats['total_access_count'] += int(memory.access_count) if isinstance(memory.access_count, (str, int)) else 0
                
                # Expired count
                if self._is_memory_expired(memory):
                    stats['expired_count'] += 1
            
            if stats['total_memories'] > 0:
                stats['average_importance'] /= stats['total_memories']
            
            stats['projects'] = list(stats['projects'])
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {}
    
    def _is_memory_expired(self, memory: MemoryEntry) -> bool:
        """Check if a memory has expired based on TTL."""
        if not self.enable_ttl or not memory.ttl_expires:
            return False
        
        try:
            expiry_date = datetime.fromisoformat(memory.ttl_expires)
            return datetime.now() > expiry_date
        except Exception:
            return False
    
    def _load_memories(self):
        """Load memories from persistent storage."""
        try:
            if self.memory_store_path.exists():
                with open(self.memory_store_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for memory_data in data.get('memories', []):
                    # Convert embedding back to numpy array if present
                    embedding = None
                    if memory_data.get('embedding'):
                        embedding = np.array(memory_data['embedding'])
                    
                    memory = MemoryEntry(
                        memory_id=memory_data['memory_id'],
                        content=memory_data['content'],
                        summary=memory_data['summary'],
                        embedding=embedding,
                        metadata=memory_data.get('metadata', {}),
                        created_at=memory_data['created_at'],
                        last_accessed=memory_data['last_accessed'],
                        access_count=memory_data.get('access_count', 1),
                        importance_score=memory_data.get('importance_score', 0.5),
                        tags=memory_data.get('tags', []),
                        content_type=memory_data.get('content_type', 'conversation'),
                        project=memory_data.get('project'),
                        ttl_expires=memory_data.get('ttl_expires')
                    )
                    
                    self.memories[memory.memory_id] = memory
                
                logger.info(f"Loaded {len(self.memories)} memories from storage")
            
        except Exception as e:
            logger.error(f"Error loading memories: {e}")
    
    def _save_memories(self):
        """Save memories to persistent storage."""
        try:
            # Prepare data for JSON serialization
            memories_data = []
            
            for memory in self.memories.values():
                memory_dict = asdict(memory)
                
                # Convert numpy array to list for JSON serialization
                if memory_dict['embedding'] is not None:
                    memory_dict['embedding'] = memory_dict['embedding'].tolist()
                
                memories_data.append(memory_dict)
            
            data = {
                'memories': memories_data,
                'config': self.config,
                'last_updated': datetime.now().isoformat()
            }
            
            # Ensure directory exists
            self.memory_store_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to file
            with open(self.memory_store_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved {len(self.memories)} memories to storage")
            
        except Exception as e:
            logger.error(f"Error saving memories: {e}")

# Global memory manager instance
_memory_manager = None

def get_memory_manager(memory_store_path: str = "memory_store.json", 
                      embedding_manager=None) -> LongTermMemoryManager:
    """Get or create a global memory manager instance."""
    global _memory_manager
    
    if _memory_manager is None:
        _memory_manager = LongTermMemoryManager(
            memory_store_path=memory_store_path,
            embedding_manager=embedding_manager
        )
    
    return _memory_manager
