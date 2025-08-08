#!/usr/bin/env python3
"""
Memory Ranking Framework
Implements intelligent ranking and prioritization of memory blocks for context injection.

Sprint 15 Deliverable #1: Memory Ranking Framework
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RankingFactor(Enum):
    """Factors used in memory ranking."""
    SIMILARITY = "similarity"
    RECENCY = "recency"
    USER_PRIORITY = "user_priority"
    SOURCE_CONFIDENCE = "source_confidence"
    USAGE_FREQUENCY = "usage_frequency"
    CONTENT_QUALITY = "content_quality"

@dataclass
class MemoryRankingScore:
    """Comprehensive ranking score for a memory block."""
    memory_id: str
    overall_score: float
    factor_scores: Dict[RankingFactor, float]
    is_priority: bool
    is_pinned: bool
    ranking_explanation: str
    computed_at: str

class MemoryRankingFramework:
    """
    Advanced memory ranking system that prioritizes memory blocks based on multiple factors.
    
    Ranking Factors:
    1. Similarity Score - How well the memory matches the query
    2. Recency - How recently the memory was created/accessed
    3. User Priority - Manual priority set by user
    4. Source Confidence - Quality score from enrichment process
    5. Usage Frequency - How often the memory has been accessed
    6. Content Quality - Length, structure, and information density
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the memory ranking framework."""
        self.config = config or self._get_default_config()
        self.ranking_weights = self.config.get('ranking_weights', {})
        self.priority_threshold = self.config.get('priority_threshold', 0.7)
        self.recency_decay_days = self.config.get('recency_decay_days', 30)
        
        logger.info("Memory ranking framework initialized")
        logger.info(f"Priority threshold: {self.priority_threshold}")
        logger.info(f"Ranking weights: {self.ranking_weights}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for memory ranking."""
        return {
            'ranking_weights': {
                RankingFactor.SIMILARITY.value: 0.35,      # Primary factor
                RankingFactor.SOURCE_CONFIDENCE.value: 0.25, # Quality matters
                RankingFactor.RECENCY.value: 0.15,         # Recent is relevant
                RankingFactor.USER_PRIORITY.value: 0.15,   # User knows best
                RankingFactor.USAGE_FREQUENCY.value: 0.05, # Popular content
                RankingFactor.CONTENT_QUALITY.value: 0.05  # Well-structured content
            },
            'priority_threshold': 0.4,  # CRITICAL FIX: Lowered from 0.7 to 0.4 to allow relevant content
            'recency_decay_days': 30,
            'max_priority_memories': 10,
            'enable_auto_pinning': True
        }
    
    def rank_memories(self, memories: List[Any], query: str = "", 
                     context: Optional[Dict[str, Any]] = None) -> List[MemoryRankingScore]:
        """
        Rank a list of memory blocks based on multiple factors.
        
        Args:
            memories: List of memory chunks to rank
            query: Current query for similarity scoring
            context: Additional context for ranking decisions
            
        Returns:
            List of MemoryRankingScore objects sorted by overall score
        """
        try:
            logger.info(f"Ranking {len(memories)} memory blocks")
            
            ranking_scores = []
            
            for memory in memories:
                score = self._compute_memory_score(memory, query, context)
                ranking_scores.append(score)
            
            # Sort by overall score (descending)
            ranking_scores.sort(key=lambda x: x.overall_score, reverse=True)
            
            # Mark top memories as priority
            self._mark_priority_memories(ranking_scores)
            
            logger.info(f"Ranking completed. Top score: {ranking_scores[0].overall_score:.3f}")
            return ranking_scores
            
        except Exception as e:
            logger.error(f"Error ranking memories: {e}")
            return []
    
    def _compute_memory_score(self, memory: Any, query: str, 
                            context: Optional[Dict[str, Any]]) -> MemoryRankingScore:
        """Compute comprehensive ranking score for a single memory."""
        try:
            factor_scores = {}
            
            # 1. Similarity Score (from search result or compute)
            similarity_score = getattr(memory, 'similarity_score', 0.0)
            if hasattr(memory, 'chunk') and hasattr(memory.chunk, 'content') and query:
                # If we have a search result with similarity, use it
                pass
            factor_scores[RankingFactor.SIMILARITY] = similarity_score
            
            # 2. Recency Score
            recency_score = self._compute_recency_score(memory)
            factor_scores[RankingFactor.RECENCY] = recency_score
            
            # 3. User Priority Score
            user_priority_score = self._compute_user_priority_score(memory)
            factor_scores[RankingFactor.USER_PRIORITY] = user_priority_score
            
            # 4. Source Confidence Score
            source_confidence_score = self._compute_source_confidence_score(memory)
            factor_scores[RankingFactor.SOURCE_CONFIDENCE] = source_confidence_score
            
            # 5. Usage Frequency Score
            usage_frequency_score = self._compute_usage_frequency_score(memory)
            factor_scores[RankingFactor.USAGE_FREQUENCY] = usage_frequency_score
            
            # 6. Content Quality Score
            content_quality_score = self._compute_content_quality_score(memory)
            factor_scores[RankingFactor.CONTENT_QUALITY] = content_quality_score
            
            # Compute weighted overall score
            overall_score = 0.0
            for factor, score in factor_scores.items():
                weight = self.ranking_weights.get(factor.value, 0.0)
                overall_score += weight * score
            
            # Check if memory is pinned
            is_pinned = self._is_memory_pinned(memory)
            if is_pinned:
                overall_score *= 1.2  # Boost pinned memories
            
            # Generate explanation
            explanation = self._generate_ranking_explanation(factor_scores, overall_score)
            
            memory_id = getattr(memory, 'chunk_id', getattr(memory, 'id', 'unknown'))
            if hasattr(memory, 'chunk'):
                memory_id = memory.chunk.chunk_id
            
            return MemoryRankingScore(
                memory_id=memory_id,
                overall_score=overall_score,
                factor_scores=factor_scores,
                is_priority=overall_score >= self.priority_threshold,
                is_pinned=is_pinned,
                ranking_explanation=explanation,
                computed_at=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error computing memory score: {e}")
            return MemoryRankingScore(
                memory_id="error",
                overall_score=0.0,
                factor_scores={},
                is_priority=False,
                is_pinned=False,
                ranking_explanation=f"Error: {e}",
                computed_at=datetime.now().isoformat()
            )
    
    def _compute_recency_score(self, memory: Any) -> float:
        """Compute recency score based on memory age."""
        try:
            # Get timestamp from memory
            timestamp_str = None
            if hasattr(memory, 'chunk') and hasattr(memory.chunk, 'timestamp'):
                timestamp_str = memory.chunk.timestamp
            elif hasattr(memory, 'timestamp'):
                timestamp_str = memory.timestamp
            
            if not timestamp_str:
                return 0.5  # Default score for unknown age
            
            # Parse timestamp
            memory_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            now = datetime.now()
            
            # Calculate age in days
            age_days = (now - memory_time).days
            
            # Apply exponential decay
            decay_factor = max(0.0, 1.0 - (age_days / self.recency_decay_days))
            return decay_factor
            
        except Exception as e:
            logger.debug(f"Error computing recency score: {e}")
            return 0.5
    
    def _compute_user_priority_score(self, memory: Any) -> float:
        """Compute user-defined priority score."""
        try:
            # Check for user-defined priority in metadata or tags
            priority_score = 0.5  # Default
            
            # Check metadata
            metadata = None
            if hasattr(memory, 'chunk') and hasattr(memory.chunk, 'metadata'):
                metadata = memory.chunk.metadata
            elif hasattr(memory, 'metadata'):
                metadata = memory.metadata
            
            if metadata and isinstance(metadata, dict):
                priority_score = metadata.get('user_priority', 0.5)
            
            # Check tags for priority indicators
            tags = []
            if hasattr(memory, 'chunk') and hasattr(memory.chunk, 'tags'):
                tags = memory.chunk.tags
            elif hasattr(memory, 'tags'):
                tags = memory.tags
            
            if tags:
                if 'high_priority' in tags or 'important' in tags:
                    priority_score = max(priority_score, 0.9)
                elif 'low_priority' in tags:
                    priority_score = min(priority_score, 0.2)
            
            return min(1.0, max(0.0, priority_score))
            
        except Exception as e:
            logger.debug(f"Error computing user priority score: {e}")
            return 0.5
    
    def _compute_source_confidence_score(self, memory: Any) -> float:
        """Compute source confidence score from enrichment data."""
        try:
            # Get enrichment score or importance score
            confidence_score = 0.5  # Default
            
            # Check for importance score
            if hasattr(memory, 'chunk') and hasattr(memory.chunk, 'importance_score'):
                confidence_score = memory.chunk.importance_score
            elif hasattr(memory, 'importance_score'):
                confidence_score = memory.importance_score
            
            # Check metadata for enrichment scores
            metadata = None
            if hasattr(memory, 'chunk') and hasattr(memory.chunk, 'metadata'):
                metadata = memory.chunk.metadata
            elif hasattr(memory, 'metadata'):
                metadata = memory.metadata
            
            if metadata and isinstance(metadata, dict):
                enrichment_score = metadata.get('enrichment_score', 0)
                if enrichment_score > 0:
                    # Normalize enrichment score (typically 0-1 range)
                    confidence_score = max(confidence_score, min(1.0, enrichment_score))
            
            return confidence_score
            
        except Exception as e:
            logger.debug(f"Error computing source confidence score: {e}")
            return 0.5
    
    def _compute_usage_frequency_score(self, memory: Any) -> float:
        """Compute usage frequency score."""
        try:
            # This would require tracking usage in metadata
            # For now, return default score
            usage_count = 0
            
            metadata = None
            if hasattr(memory, 'chunk') and hasattr(memory.chunk, 'metadata'):
                metadata = memory.chunk.metadata
            elif hasattr(memory, 'metadata'):
                metadata = memory.metadata
            
            if metadata and isinstance(metadata, dict):
                usage_count = metadata.get('usage_count', 0)
            
            # Normalize usage count (log scale)
            if usage_count > 0:
                import math
                return min(1.0, math.log(usage_count + 1) / 5.0)
            
            return 0.1  # Low score for unused memories
            
        except Exception as e:
            logger.debug(f"Error computing usage frequency score: {e}")
            return 0.1
    
    def _compute_content_quality_score(self, memory: Any) -> float:
        """Compute content quality score based on structure and information density."""
        try:
            content = ""
            if hasattr(memory, 'chunk') and hasattr(memory.chunk, 'content'):
                content = memory.chunk.content
            elif hasattr(memory, 'content'):
                content = memory.content
            
            if not content:
                return 0.1
            
            # Basic quality metrics
            length_score = min(1.0, len(content) / 500.0)  # Prefer substantial content
            
            # Check for structured content
            structure_score = 0.5
            if any(marker in content for marker in ['##', '**', '- ', '1.', '•']):
                structure_score = 0.8
            
            # Check for technical content
            technical_score = 0.5
            if any(term in content.lower() for term in ['date', 'schedule', 'deadline', 'important']):
                technical_score = 0.7
            
            # Combine scores
            quality_score = (length_score * 0.4 + structure_score * 0.3 + technical_score * 0.3)
            return min(1.0, quality_score)
            
        except Exception as e:
            logger.debug(f"Error computing content quality score: {e}")
            return 0.5
    
    def _is_memory_pinned(self, memory: Any) -> bool:
        """Check if memory is manually pinned by user."""
        try:
            # Check metadata for pinned status
            metadata = None
            if hasattr(memory, 'chunk') and hasattr(memory.chunk, 'metadata'):
                metadata = memory.chunk.metadata
            elif hasattr(memory, 'metadata'):
                metadata = memory.metadata
            
            if metadata and isinstance(metadata, dict):
                return metadata.get('is_pinned', False)
            
            # Check tags
            tags = []
            if hasattr(memory, 'chunk') and hasattr(memory.chunk, 'tags'):
                tags = memory.chunk.tags
            elif hasattr(memory, 'tags'):
                tags = memory.tags
            
            return 'pinned' in tags if tags else False
            
        except Exception as e:
            logger.debug(f"Error checking pinned status: {e}")
            return False
    
    def _mark_priority_memories(self, ranking_scores: List[MemoryRankingScore]) -> None:
        """Mark top-ranked memories as priority."""
        try:
            max_priority = self.config.get('max_priority_memories', 10)
            
            for i, score in enumerate(ranking_scores):
                if i < max_priority and score.overall_score >= self.priority_threshold:
                    score.is_priority = True
                else:
                    score.is_priority = False
                    
        except Exception as e:
            logger.error(f"Error marking priority memories: {e}")
    
    def _generate_ranking_explanation(self, factor_scores: Dict[RankingFactor, float], 
                                    overall_score: float) -> str:
        """Generate human-readable explanation of ranking."""
        try:
            explanations = []
            
            # Find top contributing factors
            sorted_factors = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)
            
            for factor, score in sorted_factors[:3]:  # Top 3 factors
                weight = self.ranking_weights.get(factor.value, 0.0)
                contribution = weight * score
                
                if contribution > 0.1:  # Only mention significant factors
                    factor_name = factor.value.replace('_', ' ').title()
                    explanations.append(f"{factor_name}: {score:.2f}")
            
            explanation = f"Score: {overall_score:.3f} | " + " | ".join(explanations)
            return explanation
            
        except Exception as e:
            logger.debug(f"Error generating explanation: {e}")
            return f"Score: {overall_score:.3f}"

# Global ranking framework instance
_ranking_framework = None

def get_memory_ranking_framework() -> MemoryRankingFramework:
    """Get or create a global memory ranking framework instance."""
    global _ranking_framework
    
    if _ranking_framework is None:
        _ranking_framework = MemoryRankingFramework()
    
    return _ranking_framework
