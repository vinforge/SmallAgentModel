#!/usr/bin/env python3
"""
Phase 6: Episodic Memory Layer for SAM
Long-term storage and retrieval of reasoning chains, documents, and user interactions.

This system enables SAM to:
1. Remember past queries and their contexts
2. Track reasoning evolution over time
3. Enable "I already analyzed this" features
4. Build longitudinal user interaction patterns
5. Support personalized learning and adaptation
"""

import logging
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import sqlite3
import threading

logger = logging.getLogger(__name__)

class InteractionType(Enum):
    """Types of user interactions stored in episodic memory."""
    QUERY = "query"
    DOCUMENT_UPLOAD = "document_upload"
    FEEDBACK = "feedback"
    CORRECTION = "correction"
    PROFILE_SWITCH = "profile_switch"
    MEMORY_SEARCH = "memory_search"

class OutcomeType(Enum):
    """Types of interaction outcomes."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    USER_CORRECTION = "user_correction"
    ABANDONED = "abandoned"

@dataclass
class EpisodicMemory:
    """Represents a single episodic memory entry."""
    # Core identification
    memory_id: str
    session_id: str
    user_id: str  # For multi-user support
    timestamp: str

    # Interaction details
    interaction_type: InteractionType
    query: str
    context: Dict[str, Any]

    # Response and reasoning
    response: str
    reasoning_chain: List[Dict[str, Any]]

    # Profile and configuration
    active_profile: str

    # Optional fields with defaults
    meta_reasoning: Optional[Dict[str, Any]] = None
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    confidence_score: float = 0.0
    
    # Outcome and feedback
    outcome_type: OutcomeType = OutcomeType.SUCCESS
    user_feedback: Optional[str] = None
    correction_applied: bool = False
    
    # Learning metadata
    documents_referenced: List[str] = field(default_factory=list)
    memory_chunks_used: List[str] = field(default_factory=list)
    processing_time_ms: int = 0
    
    # Relationships
    related_memories: List[str] = field(default_factory=list)
    follow_up_queries: List[str] = field(default_factory=list)
    
    # Quality metrics
    user_satisfaction: Optional[float] = None  # 0.0 to 1.0
    accuracy_score: Optional[float] = None
    relevance_score: Optional[float] = None

@dataclass
class MemoryPattern:
    """Represents a detected pattern in user interactions."""
    pattern_id: str
    pattern_type: str  # "preference", "behavior", "error", "success"
    description: str
    frequency: int
    confidence: float
    first_seen: str
    last_seen: str
    examples: List[str] = field(default_factory=list)

class EpisodicMemoryStore:
    """
    Advanced episodic memory system for SAM that stores and retrieves
    long-term interaction history for personalized learning.
    """
    
    def __init__(self, db_path: str = "memory_store/episodic_memory.db"):
        """Initialize the episodic memory store."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize database
        self._init_database()
        
        # Memory management settings
        self.max_memories_per_user = 10000
        self.memory_retention_days = 365
        self.pattern_detection_threshold = 3
        
        logger.info(f"Episodic Memory Store initialized at {self.db_path}")
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS episodic_memories (
                    memory_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    interaction_type TEXT NOT NULL,
                    query TEXT NOT NULL,
                    context TEXT,
                    response TEXT NOT NULL,
                    reasoning_chain TEXT,
                    meta_reasoning TEXT,
                    active_profile TEXT NOT NULL,
                    dimension_scores TEXT,
                    confidence_score REAL,
                    outcome_type TEXT,
                    user_feedback TEXT,
                    correction_applied BOOLEAN,
                    documents_referenced TEXT,
                    memory_chunks_used TEXT,
                    processing_time_ms INTEGER,
                    related_memories TEXT,
                    follow_up_queries TEXT,
                    user_satisfaction REAL,
                    accuracy_score REAL,
                    relevance_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    frequency INTEGER,
                    confidence REAL,
                    first_seen TEXT,
                    last_seen TEXT,
                    examples TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_timestamp ON episodic_memories(user_id, timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON episodic_memories(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_interaction_type ON episodic_memories(interaction_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_query_hash ON episodic_memories(query)")
            
            conn.commit()
    
    def store_memory(self, memory: EpisodicMemory) -> bool:
        """Store an episodic memory entry."""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO episodic_memories (
                            memory_id, session_id, user_id, timestamp, interaction_type,
                            query, context, response, reasoning_chain, meta_reasoning,
                            active_profile, dimension_scores, confidence_score,
                            outcome_type, user_feedback, correction_applied,
                            documents_referenced, memory_chunks_used, processing_time_ms,
                            related_memories, follow_up_queries, user_satisfaction,
                            accuracy_score, relevance_score
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        memory.memory_id, memory.session_id, memory.user_id, memory.timestamp,
                        memory.interaction_type.value, memory.query, json.dumps(memory.context),
                        memory.response, json.dumps(memory.reasoning_chain),
                        json.dumps(memory.meta_reasoning) if memory.meta_reasoning else None,
                        memory.active_profile, json.dumps(memory.dimension_scores),
                        memory.confidence_score, memory.outcome_type.value,
                        memory.user_feedback, memory.correction_applied,
                        json.dumps(memory.documents_referenced),
                        json.dumps(memory.memory_chunks_used), memory.processing_time_ms,
                        json.dumps(memory.related_memories),
                        json.dumps(memory.follow_up_queries), memory.user_satisfaction,
                        memory.accuracy_score, memory.relevance_score
                    ))
                    conn.commit()
            
            logger.debug(f"Stored episodic memory: {memory.memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing episodic memory: {e}")
            return False
    
    def retrieve_memories(self, 
                         user_id: str,
                         limit: int = 50,
                         interaction_type: Optional[InteractionType] = None,
                         since: Optional[datetime] = None,
                         profile: Optional[str] = None) -> List[EpisodicMemory]:
        """Retrieve episodic memories with filtering options."""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    
                    # Build query with filters
                    query = "SELECT * FROM episodic_memories WHERE user_id = ?"
                    params = [user_id]
                    
                    if interaction_type:
                        query += " AND interaction_type = ?"
                        params.append(interaction_type.value)
                    
                    if since:
                        query += " AND timestamp >= ?"
                        params.append(since.isoformat())
                    
                    if profile:
                        query += " AND active_profile = ?"
                        params.append(profile)
                    
                    query += " ORDER BY timestamp DESC LIMIT ?"
                    params.append(limit)
                    
                    cursor = conn.execute(query, params)
                    rows = cursor.fetchall()
                    
                    memories = []
                    for row in rows:
                        memory = self._row_to_memory(row)
                        if memory:
                            memories.append(memory)
                    
                    return memories
                    
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []
    
    def find_similar_queries(self, 
                           user_id: str,
                           query: str,
                           similarity_threshold: float = 0.7,
                           limit: int = 5) -> List[EpisodicMemory]:
        """Find similar past queries using simple text similarity."""
        try:
            # Get recent memories for the user
            recent_memories = self.retrieve_memories(
                user_id=user_id,
                limit=200,
                interaction_type=InteractionType.QUERY
            )
            
            # Simple similarity scoring (can be enhanced with embeddings)
            similar_memories = []
            query_words = set(query.lower().split())
            
            for memory in recent_memories:
                memory_words = set(memory.query.lower().split())
                
                # Jaccard similarity
                intersection = len(query_words.intersection(memory_words))
                union = len(query_words.union(memory_words))
                similarity = intersection / union if union > 0 else 0.0
                
                if similarity >= similarity_threshold:
                    similar_memories.append((similarity, memory))
            
            # Sort by similarity and return top results
            similar_memories.sort(key=lambda x: x[0], reverse=True)
            return [memory for _, memory in similar_memories[:limit]]
            
        except Exception as e:
            logger.error(f"Error finding similar queries: {e}")
            return []
    
    def update_feedback(self, 
                       memory_id: str,
                       user_feedback: str,
                       user_satisfaction: Optional[float] = None,
                       correction_applied: bool = False) -> bool:
        """Update memory with user feedback."""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        UPDATE episodic_memories 
                        SET user_feedback = ?, user_satisfaction = ?, correction_applied = ?
                        WHERE memory_id = ?
                    """, (user_feedback, user_satisfaction, correction_applied, memory_id))
                    conn.commit()
            
            logger.debug(f"Updated feedback for memory: {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating feedback: {e}")
            return False

    def _row_to_memory(self, row: sqlite3.Row) -> Optional[EpisodicMemory]:
        """Convert database row to EpisodicMemory object."""
        try:
            return EpisodicMemory(
                memory_id=row['memory_id'],
                session_id=row['session_id'],
                user_id=row['user_id'],
                timestamp=row['timestamp'],
                interaction_type=InteractionType(row['interaction_type']),
                query=row['query'],
                context=json.loads(row['context']) if row['context'] else {},
                response=row['response'],
                reasoning_chain=json.loads(row['reasoning_chain']) if row['reasoning_chain'] else [],
                meta_reasoning=json.loads(row['meta_reasoning']) if row['meta_reasoning'] else None,
                active_profile=row['active_profile'],
                dimension_scores=json.loads(row['dimension_scores']) if row['dimension_scores'] else {},
                confidence_score=row['confidence_score'] or 0.0,
                outcome_type=OutcomeType(row['outcome_type']) if row['outcome_type'] else OutcomeType.SUCCESS,
                user_feedback=row['user_feedback'],
                correction_applied=bool(row['correction_applied']),
                documents_referenced=json.loads(row['documents_referenced']) if row['documents_referenced'] else [],
                memory_chunks_used=json.loads(row['memory_chunks_used']) if row['memory_chunks_used'] else [],
                processing_time_ms=row['processing_time_ms'] or 0,
                related_memories=json.loads(row['related_memories']) if row['related_memories'] else [],
                follow_up_queries=json.loads(row['follow_up_queries']) if row['follow_up_queries'] else [],
                user_satisfaction=row['user_satisfaction'],
                accuracy_score=row['accuracy_score'],
                relevance_score=row['relevance_score']
            )
        except Exception as e:
            logger.error(f"Error converting row to memory: {e}")
            return None

    def create_memory_from_interaction(self,
                                     user_id: str,
                                     session_id: str,
                                     query: str,
                                     response: str,
                                     context: Dict[str, Any],
                                     active_profile: str,
                                     reasoning_chain: Optional[List[Dict[str, Any]]] = None,
                                     meta_reasoning: Optional[Dict[str, Any]] = None,
                                     processing_time_ms: int = 0) -> EpisodicMemory:
        """Create episodic memory from interaction data."""

        # Generate unique memory ID
        memory_content = f"{user_id}_{query}_{datetime.now().isoformat()}"
        memory_id = hashlib.md5(memory_content.encode()).hexdigest()

        # Extract dimension scores from meta_reasoning if available
        dimension_scores = {}
        confidence_score = 0.0

        if meta_reasoning:
            if 'response_analysis' in meta_reasoning:
                dimension_scores = meta_reasoning['response_analysis'].get('dimension_scores', {})
            if 'confidence_justification' in meta_reasoning:
                confidence_score = meta_reasoning['confidence_justification'].get('confidence_score', 0.0)

        # Create memory object
        memory = EpisodicMemory(
            memory_id=memory_id,
            session_id=session_id,
            user_id=user_id,
            timestamp=datetime.now().isoformat(),
            interaction_type=InteractionType.QUERY,
            query=query,
            context=context,
            response=response,
            reasoning_chain=reasoning_chain or [],
            meta_reasoning=meta_reasoning,
            active_profile=active_profile,
            dimension_scores=dimension_scores,
            confidence_score=confidence_score,
            processing_time_ms=processing_time_ms
        )

        return memory

    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a user."""
        try:
            memories = self.retrieve_memories(user_id, limit=1000)

            if not memories:
                return {"total_interactions": 0}

            # Basic statistics
            stats = {
                "total_interactions": len(memories),
                "first_interaction": memories[-1].timestamp,
                "last_interaction": memories[0].timestamp,
                "average_confidence": sum(m.confidence_score for m in memories) / len(memories),
                "profile_distribution": {},
                "interaction_types": {},
                "satisfaction_scores": []
            }

            # Profile distribution
            for memory in memories:
                profile = memory.active_profile
                stats["profile_distribution"][profile] = stats["profile_distribution"].get(profile, 0) + 1

            # Interaction types
            for memory in memories:
                interaction_type = memory.interaction_type.value
                stats["interaction_types"][interaction_type] = stats["interaction_types"].get(interaction_type, 0) + 1

            # Satisfaction scores
            satisfaction_scores = [m.user_satisfaction for m in memories if m.user_satisfaction is not None]
            if satisfaction_scores:
                stats["average_satisfaction"] = sum(satisfaction_scores) / len(satisfaction_scores)
                stats["satisfaction_trend"] = satisfaction_scores[:10]  # Last 10 scores

            # Recent activity
            recent_memories = [m for m in memories if self._is_recent(m.timestamp, days=7)]
            stats["recent_activity"] = {
                "interactions_last_7_days": len(recent_memories),
                "average_daily_interactions": len(recent_memories) / 7
            }

            return stats

        except Exception as e:
            logger.error(f"Error getting user statistics: {e}")
            return {"error": str(e)}

    def _is_recent(self, timestamp_str: str, days: int = 7) -> bool:
        """Check if timestamp is within recent days."""
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            cutoff = datetime.now() - timedelta(days=days)
            return timestamp >= cutoff
        except Exception:
            return False


# Convenience functions for easy integration
def create_episodic_store(db_path: str = "memory_store/episodic_memory.db") -> EpisodicMemoryStore:
    """Create and return an episodic memory store instance."""
    return EpisodicMemoryStore(db_path)

def store_interaction_memory(store: EpisodicMemoryStore,
                           user_id: str,
                           session_id: str,
                           query: str,
                           response: str,
                           context: Dict[str, Any],
                           active_profile: str,
                           **kwargs) -> bool:
    """Convenience function to store interaction memory."""
    memory = store.create_memory_from_interaction(
        user_id=user_id,
        session_id=session_id,
        query=query,
        response=response,
        context=context,
        active_profile=active_profile,
        **kwargs
    )
    return store.store_memory(memory)
