"""
SAM Phase 3: Memory Ranking Engine
Implements hybrid semantic + metadata ranking for intelligent context selection.
"""

import json
import math
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RankingWeights:
    """Configuration for ranking algorithm weights."""
    semantic: float = 0.6
    recency: float = 0.15
    confidence: float = 0.2
    priority: float = 0.05
    
    def normalize(self) -> 'RankingWeights':
        """Ensure weights sum to 1.0."""
        total = self.semantic + self.recency + self.confidence + self.priority
        if total == 0:
            return RankingWeights()  # Default weights
        
        return RankingWeights(
            semantic=self.semantic / total,
            recency=self.recency / total,
            confidence=self.confidence / total,
            priority=self.priority / total
        )

@dataclass
class RankingConfig:
    """Configuration for ranking behavior."""
    initial_candidates: int = 50
    recency_decay_days: float = 30.0
    min_confidence_threshold: float = 0.1
    enable_hybrid_ranking: bool = True
    cache_rankings: bool = True

@dataclass
class RankedMemoryResult:
    """Enhanced memory result with ranking information."""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    semantic_score: float
    recency_score: float
    confidence_score: float
    priority_score: float
    final_score: float
    original_distance: float
    
    @property
    def similarity_score(self) -> float:
        """Compatibility with existing MemorySearchResult."""
        return self.semantic_score

class MemoryRankingEngine:
    """
    Hybrid Memory Ranking Engine for SAM.
    
    Implements multi-factor scoring combining:
    - Semantic similarity (from ChromaDB)
    - Recency (temporal decay)
    - Confidence (metadata-based)
    - Priority (pinned/important items)
    """
    
    def __init__(self, config_path: str = "config/sam_config.json"):
        """Initialize ranking engine with configuration."""
        self.config_path = Path(config_path)
        self.weights = RankingWeights()
        self.config = RankingConfig()
        self._load_configuration()
        
        logger.info(f"Memory Ranking Engine initialized")
        logger.info(f"  Weights: semantic={self.weights.semantic:.2f}, recency={self.weights.recency:.2f}, "
                   f"confidence={self.weights.confidence:.2f}, priority={self.weights.priority:.2f}")
        logger.info(f"  Config: candidates={self.config.initial_candidates}, "
                   f"decay_days={self.config.recency_decay_days}")
    
    def _load_configuration(self):
        """Load ranking configuration from JSON file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                
                memory_config = config_data.get("memory", {})
                
                # Load ranking weights
                weights_data = memory_config.get("ranking_weights", {})
                self.weights = RankingWeights(
                    semantic=weights_data.get("semantic", 0.6),
                    recency=weights_data.get("recency", 0.15),
                    confidence=weights_data.get("confidence", 0.2),
                    priority=weights_data.get("priority", 0.05)
                ).normalize()
                
                # Load ranking config
                config_data_ranking = memory_config.get("ranking_config", {})
                self.config = RankingConfig(
                    initial_candidates=config_data_ranking.get("initial_candidates", 50),
                    recency_decay_days=config_data_ranking.get("recency_decay_days", 30.0),
                    min_confidence_threshold=config_data_ranking.get("min_confidence_threshold", 0.1),
                    enable_hybrid_ranking=config_data_ranking.get("enable_hybrid_ranking", True),
                    cache_rankings=config_data_ranking.get("cache_rankings", True)
                )
                
                logger.info("Ranking configuration loaded from file")
            else:
                logger.warning(f"Config file not found: {self.config_path}, using defaults")
                
        except Exception as e:
            logger.error(f"Error loading ranking configuration: {e}")
            logger.info("Using default ranking configuration")
    
    def calculate_semantic_score(self, chroma_distance: float) -> float:
        """
        Convert ChromaDB distance to similarity score.
        
        Args:
            chroma_distance: Distance from ChromaDB (0 = identical, >0 = different)
            
        Returns:
            Similarity score (0-1, higher = more similar)
        """
        # Convert cosine distance to similarity: similarity = 1 - distance
        # Clamp to [0, 1] range for safety
        similarity = max(0.0, min(1.0, 1.0 - chroma_distance))
        return similarity
    
    def calculate_recency_score(self, created_at: Any) -> float:
        """
        Calculate recency score using exponential decay.
        
        Args:
            created_at: Timestamp (string, int, or datetime)
            
        Returns:
            Recency score (0-1, higher = more recent)
        """
        try:
            # Parse timestamp
            if isinstance(created_at, str):
                if created_at.isdigit():
                    # Unix timestamp as string
                    timestamp = float(created_at)
                    created_time = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                else:
                    # ISO format string
                    created_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            elif isinstance(created_at, (int, float)):
                # Unix timestamp
                created_time = datetime.fromtimestamp(created_at, tz=timezone.utc)
            else:
                # Default to current time if unparseable
                logger.warning(f"Could not parse timestamp: {created_at}")
                return 0.5

            # Ensure created_time is timezone-aware
            if created_time.tzinfo is None:
                created_time = created_time.replace(tzinfo=timezone.utc)
            
            # Calculate age in days
            now = datetime.now(timezone.utc)
            age_days = (now - created_time).total_seconds() / 86400  # seconds per day
            
            # Exponential decay: score = exp(-age_days * ln(2) / half_life)
            half_life = self.config.recency_decay_days
            decay_rate = math.log(2) / half_life
            recency_score = math.exp(-age_days * decay_rate)
            
            # Clamp to [0, 1] range
            return max(0.0, min(1.0, recency_score))
            
        except Exception as e:
            logger.warning(f"Error calculating recency score for {created_at}: {e}")
            return 0.5  # Neutral score on error
    
    def calculate_confidence_score(self, metadata: Dict[str, Any]) -> float:
        """
        Extract confidence score from metadata.
        
        Args:
            metadata: Memory chunk metadata
            
        Returns:
            Confidence score (0-1)
        """
        try:
            # Try multiple possible confidence fields
            confidence_fields = ["confidence_score", "importance_score", "score"]
            
            for field in confidence_fields:
                if field in metadata:
                    score = float(metadata[field])
                    return max(0.0, min(1.0, score))  # Clamp to [0, 1]
            
            # Default confidence if no field found
            return 0.5
            
        except Exception as e:
            logger.warning(f"Error extracting confidence score: {e}")
            return 0.5
    
    def calculate_priority_score(self, metadata: Dict[str, Any]) -> float:
        """
        Calculate priority score based on metadata flags.
        
        Args:
            metadata: Memory chunk metadata
            
        Returns:
            Priority score (0-1)
        """
        try:
            # Check for priority indicators
            priority_fields = ["pinned", "priority", "important", "starred"]
            
            for field in priority_fields:
                if field in metadata:
                    value = metadata[field]
                    if isinstance(value, bool):
                        return 1.0 if value else 0.0
                    elif isinstance(value, (int, float)):
                        return max(0.0, min(1.0, float(value)))
                    elif isinstance(value, str):
                        return 1.0 if value.lower() in ["true", "yes", "1", "high"] else 0.0
            
            # Check for high confidence as priority indicator
            confidence = self.calculate_confidence_score(metadata)
            if confidence > 0.8:
                return 0.3  # Partial priority for high-confidence items
            
            return 0.0  # No priority indicators found
            
        except Exception as e:
            logger.warning(f"Error calculating priority score: {e}")
            return 0.0

    def calculate_final_score(self, chroma_distance: float, metadata: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Calculate final hybrid score for a memory chunk.

        Args:
            chroma_distance: Distance from ChromaDB vector search
            metadata: Memory chunk metadata

        Returns:
            Tuple of (final_score, score_breakdown)
        """
        try:
            # Calculate individual scores
            semantic_score = self.calculate_semantic_score(chroma_distance)
            recency_score = self.calculate_recency_score(metadata.get("created_at", ""))
            confidence_score = self.calculate_confidence_score(metadata)
            priority_score = self.calculate_priority_score(metadata)

            # Calculate weighted final score
            final_score = (
                self.weights.semantic * semantic_score +
                self.weights.recency * recency_score +
                self.weights.confidence * confidence_score +
                self.weights.priority * priority_score
            )

            # Score breakdown for debugging/transparency
            score_breakdown = {
                "semantic": semantic_score,
                "recency": recency_score,
                "confidence": confidence_score,
                "priority": priority_score,
                "final": final_score
            }

            return final_score, score_breakdown

        except Exception as e:
            logger.error(f"Error calculating final score: {e}")
            # Return semantic score as fallback
            semantic_score = self.calculate_semantic_score(chroma_distance)
            return semantic_score, {"semantic": semantic_score, "final": semantic_score}

    def rank_memory_results(self, chroma_results: List[Dict[str, Any]]) -> List[RankedMemoryResult]:
        """
        Re-rank ChromaDB results using hybrid scoring.

        Args:
            chroma_results: Raw results from ChromaDB query

        Returns:
            List of RankedMemoryResult objects, sorted by final_score (descending)
        """
        if not self.config.enable_hybrid_ranking:
            logger.info("Hybrid ranking disabled, returning semantic-only results")
            return self._convert_to_ranked_results(chroma_results, use_semantic_only=True)

        ranked_results = []

        for result in chroma_results:
            try:
                # Extract data from ChromaDB result
                chunk_id = result.get("id", "")
                content = result.get("document", "")
                metadata = result.get("metadata", {})
                distance = result.get("distance", 1.0)

                # Calculate hybrid score
                final_score, score_breakdown = self.calculate_final_score(distance, metadata)

                # Apply confidence threshold filter
                if score_breakdown["confidence"] < self.config.min_confidence_threshold:
                    logger.debug(f"Filtering out low-confidence result: {chunk_id} "
                               f"(confidence: {score_breakdown['confidence']:.3f})")
                    continue

                # Create ranked result
                ranked_result = RankedMemoryResult(
                    chunk_id=chunk_id,
                    content=content,
                    metadata=metadata,
                    semantic_score=score_breakdown["semantic"],
                    recency_score=score_breakdown["recency"],
                    confidence_score=score_breakdown["confidence"],
                    priority_score=score_breakdown["priority"],
                    final_score=final_score,
                    original_distance=distance
                )

                ranked_results.append(ranked_result)

            except Exception as e:
                logger.error(f"Error ranking result {result.get('id', 'unknown')}: {e}")
                continue

        # Sort by final score (descending)
        ranked_results.sort(key=lambda x: x.final_score, reverse=True)

        logger.info(f"Ranked {len(ranked_results)} memory results using hybrid scoring")
        if ranked_results:
            top_result = ranked_results[0]
            logger.debug(f"Top result: {top_result.chunk_id} (final_score: {top_result.final_score:.3f}, "
                        f"semantic: {top_result.semantic_score:.3f}, recency: {top_result.recency_score:.3f})")

        return ranked_results

    def _convert_to_ranked_results(self, chroma_results: List[Dict[str, Any]],
                                 use_semantic_only: bool = False) -> List[RankedMemoryResult]:
        """Convert ChromaDB results to RankedMemoryResult format."""
        ranked_results = []

        for result in chroma_results:
            try:
                chunk_id = result.get("id", "")
                content = result.get("document", "")
                metadata = result.get("metadata", {})
                distance = result.get("distance", 1.0)

                semantic_score = self.calculate_semantic_score(distance)

                if use_semantic_only:
                    final_score = semantic_score
                    recency_score = confidence_score = priority_score = 0.0
                else:
                    final_score, score_breakdown = self.calculate_final_score(distance, metadata)
                    recency_score = score_breakdown["recency"]
                    confidence_score = score_breakdown["confidence"]
                    priority_score = score_breakdown["priority"]

                ranked_result = RankedMemoryResult(
                    chunk_id=chunk_id,
                    content=content,
                    metadata=metadata,
                    semantic_score=semantic_score,
                    recency_score=recency_score,
                    confidence_score=confidence_score,
                    priority_score=priority_score,
                    final_score=final_score,
                    original_distance=distance
                )

                ranked_results.append(ranked_result)

            except Exception as e:
                logger.error(f"Error converting result {result.get('id', 'unknown')}: {e}")
                continue

        return ranked_results

    def get_adaptive_candidate_count(self, total_memories: int, requested_results: int) -> int:
        """
        Calculate adaptive candidate count based on memory store size.

        Args:
            total_memories: Total number of memories in store
            requested_results: Number of final results requested

        Returns:
            Number of initial candidates to retrieve
        """
        # Can't retrieve more candidates than total memories
        if total_memories <= requested_results:
            return total_memories

        # Ensure we have enough candidates for effective re-ranking
        min_candidates = max(requested_results * 3, 10)  # At least 3x final results
        max_candidates = min(self.config.initial_candidates, total_memories)

        # Adaptive scaling: use 10% of total memories, but within bounds
        adaptive_count = max(min_candidates, min(max_candidates, int(total_memories * 0.1)))

        # Final check: don't exceed total memories
        adaptive_count = min(adaptive_count, total_memories)

        logger.debug(f"Adaptive candidate count: {adaptive_count} "
                    f"(total: {total_memories}, requested: {requested_results})")

        return adaptive_count
