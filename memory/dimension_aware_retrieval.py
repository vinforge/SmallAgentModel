#!/usr/bin/env python3
"""
Dimension-Aware Retrieval Engine for SAM Phase 3
Implements conceptual dimension-weighted retrieval that blends semantic similarity
with human-like conceptual understanding for revolutionary search capabilities.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum

# Import existing components
from .memory_vectorstore import MemoryVectorStore, RankedMemoryResult
from .ranking_engine import MemoryRankingEngine

# Import Phase 2 dimension components
try:
    from multimodal_processing.dimension_prober_v2 import EnhancedDimensionProberV2, ProfileManager
    DIMENSION_PROBING_AVAILABLE = True
except ImportError as e:
    EnhancedDimensionProberV2 = None
    ProfileManager = None
    DIMENSION_PROBING_AVAILABLE = False
    logging.warning(f"Dimension-aware retrieval not available: {e}")
    logging.info("Falling back to standard semantic retrieval")

logger = logging.getLogger(__name__)

class RetrievalStrategy(Enum):
    """Retrieval strategy options."""
    VECTOR_ONLY = "vector_only"
    DIMENSION_ONLY = "dimension_only"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"

@dataclass
class DimensionWeights:
    """Dimension weights for retrieval scoring."""
    semantic_similarity: float = 0.4
    dimension_alignment: float = 0.3
    recency_score: float = 0.2
    confidence_score: float = 0.1
    
    # Profile-specific dimension weights
    profile_dimensions: Dict[str, float] = None
    
    def __post_init__(self):
        if self.profile_dimensions is None:
            self.profile_dimensions = {}

@dataclass
class DimensionAwareResult:
    """Enhanced result with dimension scoring information."""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    
    # Traditional scores
    semantic_score: float
    recency_score: float
    confidence_score: float
    
    # NEW: Dimension scores
    dimension_alignment_score: float
    dimension_confidence_boost: float
    profile_relevance_bonus: float
    
    # Final scoring
    final_score: float
    
    # Explanation and transparency
    score_breakdown: Dict[str, float]
    dimension_explanation: str
    ranking_reason: str

class DimensionAwareRetrieval:
    """
    Revolutionary dimension-aware retrieval engine that combines semantic similarity
    with human-like conceptual understanding for unprecedented search accuracy.
    """
    
    def __init__(self, memory_store: MemoryVectorStore, ranking_engine: Optional[MemoryRankingEngine] = None):
        """Initialize dimension-aware retrieval engine."""
        self.memory_store = memory_store
        self.ranking_engine = ranking_engine
        
        # Initialize dimension components if available
        if DIMENSION_PROBING_AVAILABLE:
            self.dimension_prober = EnhancedDimensionProberV2()
            self.profile_manager = ProfileManager()
            self.dimension_enabled = True
            logger.info("Dimension-aware retrieval initialized with conceptual understanding")
        else:
            self.dimension_prober = None
            self.profile_manager = None
            self.dimension_enabled = False
            logger.warning("Dimension-aware retrieval initialized without conceptual understanding (fallback mode)")
        
        # Default configuration
        self.default_weights = DimensionWeights()
        self.default_strategy = RetrievalStrategy.HYBRID
        
        # Profile-specific default weights
        self._init_profile_weights()
    
    def _init_profile_weights(self):
        """Initialize profile-specific dimension weights."""
        self.profile_weights = {
            "general": DimensionWeights(
                semantic_similarity=0.4,
                dimension_alignment=0.3,
                recency_score=0.2,
                confidence_score=0.1,
                profile_dimensions={
                    "utility": 1.2, "relevance": 1.3, "clarity": 1.1,
                    "complexity": 1.0, "credibility": 1.1
                }
            ),
            "researcher": DimensionWeights(
                semantic_similarity=0.3,
                dimension_alignment=0.4,
                recency_score=0.2,
                confidence_score=0.1,
                profile_dimensions={
                    "novelty": 1.5, "technical_depth": 1.3, "methodology": 1.2,
                    "impact": 1.4, "reproducibility": 1.1
                }
            ),
            "business": DimensionWeights(
                semantic_similarity=0.35,
                dimension_alignment=0.35,
                recency_score=0.2,
                confidence_score=0.1,
                profile_dimensions={
                    "market_impact": 1.4, "feasibility": 1.3, "roi_potential": 1.5,
                    "risk": 1.2, "scalability": 1.1
                }
            ),
            "legal": DimensionWeights(
                semantic_similarity=0.3,
                dimension_alignment=0.4,
                recency_score=0.15,
                confidence_score=0.15,
                profile_dimensions={
                    "compliance_risk": 1.5, "liability": 1.4, "precedent": 1.2,
                    "contractual_impact": 1.3, "ethical_considerations": 1.1
                }
            )
        }
    
    def dimension_aware_search(self, 
                             query: str,
                             max_results: int = 5,
                             profile: str = "general",
                             dimension_weights: Optional[Dict[str, float]] = None,
                             strategy: RetrievalStrategy = None,
                             natural_language_filters: Optional[str] = None) -> List[DimensionAwareResult]:
        """
        Revolutionary dimension-aware search that combines semantic similarity
        with human-like conceptual understanding.
        
        Args:
            query: Search query
            max_results: Number of results to return
            profile: Reasoning profile (general, researcher, business, legal)
            dimension_weights: Custom dimension weights (overrides profile defaults)
            strategy: Retrieval strategy (hybrid, vector_only, dimension_only, adaptive)
            natural_language_filters: Natural language filters like "high-utility, low-risk"
            
        Returns:
            List of DimensionAwareResult objects with comprehensive scoring
        """
        start_time = time.time()
        
        try:
            # Determine strategy
            active_strategy = strategy or self.default_strategy
            
            # Parse natural language filters if provided
            parsed_filters = self._parse_natural_language_filters(natural_language_filters) if natural_language_filters else {}
            
            # Get profile-specific weights
            weights = self._get_effective_weights(profile, dimension_weights, parsed_filters)
            
            # Auto-detect profile if not specified and dimension probing available
            if profile == "general" and self.dimension_enabled:
                detected_profile = self._auto_detect_profile(query)
                if detected_profile != "general":
                    profile = detected_profile
                    weights = self._get_effective_weights(profile, dimension_weights, parsed_filters)
                    logger.info(f"Auto-detected profile: {profile}")
            
            # Execute search based on strategy
            if active_strategy == RetrievalStrategy.VECTOR_ONLY:
                results = self._vector_only_search(query, max_results, weights)
            elif active_strategy == RetrievalStrategy.DIMENSION_ONLY:
                results = self._dimension_only_search(query, max_results, weights, profile, parsed_filters)
            elif active_strategy == RetrievalStrategy.ADAPTIVE:
                results = self._adaptive_search(query, max_results, weights, profile, parsed_filters)
            else:  # HYBRID (default)
                results = self._hybrid_search(query, max_results, weights, profile, parsed_filters)
            
            processing_time = (time.time() - start_time) * 1000
            
            logger.info(f"Dimension-aware search completed: {len(results)} results in {processing_time:.2f}ms "
                       f"(strategy: {active_strategy.value}, profile: {profile})")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in dimension-aware search: {e}")
            # Fallback to traditional search
            return self._fallback_search(query, max_results)
    
    def _hybrid_search(self, query: str, max_results: int, weights: DimensionWeights, 
                      profile: str, filters: Dict[str, Any]) -> List[DimensionAwareResult]:
        """Hybrid search combining semantic similarity with dimension alignment."""
        
        # Stage 1: Get semantic similarity candidates (larger pool)
        initial_candidates = max(max_results * 4, 20)
        semantic_results = self.memory_store.enhanced_search_memories(
            query=query,
            max_results=initial_candidates,
            ranking_weights=None  # We'll do our own ranking
        )
        
        if not semantic_results:
            return []
        
        # Stage 2: Apply dimension-aware scoring
        dimension_results = []
        
        for result in semantic_results:
            try:
                # Calculate dimension alignment if dimension probing available
                if self.dimension_enabled:
                    dimension_score, confidence_boost, profile_bonus = self._calculate_dimension_alignment(
                        result, query, profile, weights.profile_dimensions, filters
                    )
                else:
                    dimension_score = 0.0
                    confidence_boost = 0.0
                    profile_bonus = 0.0
                
                # Calculate final hybrid score
                final_score = (
                    weights.semantic_similarity * result.semantic_score +
                    weights.dimension_alignment * dimension_score +
                    weights.recency_score * result.recency_score +
                    weights.confidence_score * result.confidence_score +
                    confidence_boost +
                    profile_bonus
                )
                
                # Generate explanation
                explanation, breakdown = self._generate_score_explanation(
                    result, dimension_score, confidence_boost, profile_bonus, weights, profile
                )
                
                # Create dimension-aware result
                dim_result = DimensionAwareResult(
                    chunk_id=result.chunk_id,
                    content=result.content,
                    metadata=result.metadata,
                    semantic_score=result.semantic_score,
                    recency_score=result.recency_score,
                    confidence_score=result.confidence_score,
                    dimension_alignment_score=dimension_score,
                    dimension_confidence_boost=confidence_boost,
                    profile_relevance_bonus=profile_bonus,
                    final_score=final_score,
                    score_breakdown=breakdown,
                    dimension_explanation=explanation,
                    ranking_reason=f"Hybrid search with {profile} profile"
                )
                
                dimension_results.append(dim_result)
                
            except Exception as e:
                logger.warning(f"Error processing result {result.chunk_id}: {e}")
                continue
        
        # Stage 3: Sort by final score and return top results
        dimension_results.sort(key=lambda x: x.final_score, reverse=True)
        return dimension_results[:max_results]
    
    def _calculate_dimension_alignment(self, result: RankedMemoryResult, query: str, 
                                     profile: str, profile_dimensions: Dict[str, float],
                                     filters: Dict[str, Any]) -> Tuple[float, float, float]:
        """Calculate dimension alignment score for a search result."""
        try:
            # Get chunk dimension scores from metadata
            chunk_dimensions = result.metadata.get('dimension_scores', {})
            chunk_profile = result.metadata.get('dimension_profile', 'general')
            
            if not chunk_dimensions:
                return 0.0, 0.0, 0.0
            
            # Probe query for dimensions using the same profile
            query_result = self.dimension_prober.probe_chunk(query, profile=profile)
            query_dimensions = query_result.scores.scores
            
            # Calculate alignment score
            alignment_score = 0.0
            total_weight = 0.0
            
            for dimension, query_score in query_dimensions.items():
                chunk_score = chunk_dimensions.get(dimension, 0.0)
                weight = profile_dimensions.get(dimension, 1.0)
                
                # Calculate alignment (higher when both scores are high)
                alignment = min(query_score, chunk_score) * weight
                alignment_score += alignment
                total_weight += weight
            
            # Normalize by total weight
            if total_weight > 0:
                alignment_score = alignment_score / total_weight
            
            # Calculate confidence boost based on chunk dimension confidence
            chunk_confidence = result.metadata.get('dimension_confidence', {})
            avg_confidence = sum(chunk_confidence.values()) / len(chunk_confidence) if chunk_confidence else 0.5
            confidence_boost = (avg_confidence - 0.5) * 0.1  # Small boost for high confidence
            
            # Calculate profile relevance bonus
            profile_bonus = 0.05 if chunk_profile == profile else 0.0
            
            # Apply natural language filters
            filter_penalty = self._apply_dimension_filters(chunk_dimensions, filters)
            alignment_score *= (1.0 - filter_penalty)
            
            return alignment_score, confidence_boost, profile_bonus
            
        except Exception as e:
            logger.warning(f"Error calculating dimension alignment: {e}")
            return 0.0, 0.0, 0.0

    def _parse_natural_language_filters(self, filter_text: str) -> Dict[str, Any]:
        """Parse natural language filters into dimension constraints."""
        import re

        filters = {}

        # Define filter patterns
        filter_patterns = {
            # Utility patterns
            r'\b(?:high[- ]utility|very useful|highly valuable)\b': {'utility': 'high'},
            r'\b(?:low[- ]utility|not useful|little value)\b': {'utility': 'low'},

            # Risk/Danger patterns
            r'\b(?:low[- ]risk|safe|secure|low[- ]danger)\b': {'danger': 'low', 'risk': 'low'},
            r'\b(?:high[- ]risk|dangerous|risky|high[- ]danger)\b': {'danger': 'high', 'risk': 'high'},

            # Complexity patterns
            r'\b(?:simple|easy|low[- ]complexity|straightforward)\b': {'complexity': 'low'},
            r'\b(?:complex|difficult|sophisticated|advanced)\b': {'complexity': 'high'},

            # Innovation patterns
            r'\b(?:innovative|novel|cutting[- ]edge|breakthrough)\b': {'novelty': 'high', 'innovation_potential': 'high'},
            r'\b(?:traditional|conventional|standard|established)\b': {'novelty': 'low'},

            # Business patterns
            r'\b(?:profitable|high[- ]ROI|valuable|lucrative)\b': {'roi_potential': 'high', 'market_impact': 'high'},
            r'\b(?:feasible|practical|implementable|viable)\b': {'feasibility': 'high'},

            # Legal patterns
            r'\b(?:compliant|legal|regulatory|approved)\b': {'compliance_risk': 'low'},
            r'\b(?:non[- ]compliant|illegal|violation|breach)\b': {'compliance_risk': 'high'},

            # Quality patterns
            r'\b(?:high[- ]quality|excellent|superior|top[- ]tier)\b': {'credibility': 'high', 'utility': 'high'},
            r'\b(?:reliable|trustworthy|validated|verified)\b': {'credibility': 'high'},
        }

        # Apply patterns
        for pattern, dimension_filters in filter_patterns.items():
            if re.search(pattern, filter_text.lower()):
                filters.update(dimension_filters)

        logger.debug(f"Parsed filters from '{filter_text}': {filters}")
        return filters

    def _auto_detect_profile(self, query: str) -> str:
        """Auto-detect appropriate profile based on query content."""
        import re

        query_lower = query.lower()

        # Research indicators
        research_patterns = [
            r'\b(?:research|study|analysis|investigation|experiment)\b',
            r'\b(?:novel|innovative|breakthrough|cutting[- ]edge)\b',
            r'\b(?:methodology|algorithm|technical|scientific)\b',
            r'\b(?:peer[- ]reviewed|publication|journal|academic)\b'
        ]

        # Business indicators
        business_patterns = [
            r'\b(?:market|business|commercial|revenue|profit)\b',
            r'\b(?:ROI|investment|financial|cost|budget)\b',
            r'\b(?:strategy|competitive|opportunity|growth)\b',
            r'\b(?:feasible|viable|scalable|sustainable)\b'
        ]

        # Legal indicators
        legal_patterns = [
            r'\b(?:legal|law|regulation|compliance|contract)\b',
            r'\b(?:liability|risk|violation|breach|penalty)\b',
            r'\b(?:court|judge|ruling|precedent|case)\b',
            r'\b(?:ITAR|export|controlled|classified)\b'
        ]

        # Count matches for each profile
        research_score = sum(1 for pattern in research_patterns if re.search(pattern, query_lower))
        business_score = sum(1 for pattern in business_patterns if re.search(pattern, query_lower))
        legal_score = sum(1 for pattern in legal_patterns if re.search(pattern, query_lower))

        # Determine best profile
        if research_score > business_score and research_score > legal_score:
            return "researcher"
        elif business_score > legal_score:
            return "business"
        elif legal_score > 0:
            return "legal"
        else:
            return "general"

    def _get_effective_weights(self, profile: str, custom_weights: Optional[Dict[str, float]],
                             filters: Dict[str, Any]) -> DimensionWeights:
        """Get effective weights combining profile defaults with custom weights and filters."""
        # Start with profile defaults
        base_weights = self.profile_weights.get(profile, self.default_weights)

        # Create a copy to modify
        effective_weights = DimensionWeights(
            semantic_similarity=base_weights.semantic_similarity,
            dimension_alignment=base_weights.dimension_alignment,
            recency_score=base_weights.recency_score,
            confidence_score=base_weights.confidence_score,
            profile_dimensions=base_weights.profile_dimensions.copy()
        )

        # Apply custom weights if provided
        if custom_weights:
            for dimension, weight in custom_weights.items():
                if dimension in ['semantic_similarity', 'dimension_alignment', 'recency_score', 'confidence_score']:
                    setattr(effective_weights, dimension, weight)
                else:
                    effective_weights.profile_dimensions[dimension] = weight

        # Boost weights for filtered dimensions
        for dimension, filter_value in filters.items():
            if dimension in effective_weights.profile_dimensions:
                # Boost weight for filtered dimensions
                current_weight = effective_weights.profile_dimensions[dimension]
                effective_weights.profile_dimensions[dimension] = current_weight * 1.3

        return effective_weights

    def _apply_dimension_filters(self, chunk_dimensions: Dict[str, float],
                               filters: Dict[str, Any]) -> float:
        """Apply dimension filters and return penalty score (0.0 = no penalty, 1.0 = full penalty)."""
        if not filters:
            return 0.0

        penalty = 0.0
        filter_count = 0

        for dimension, filter_value in filters.items():
            if dimension not in chunk_dimensions:
                continue

            chunk_score = chunk_dimensions[dimension]
            filter_count += 1

            if filter_value == 'high':
                # Penalize if chunk score is low when we want high
                if chunk_score < 0.6:
                    penalty += (0.6 - chunk_score) * 0.5
            elif filter_value == 'low':
                # Penalize if chunk score is high when we want low
                if chunk_score > 0.4:
                    penalty += (chunk_score - 0.4) * 0.5

        # Normalize penalty by number of filters
        if filter_count > 0:
            penalty = min(1.0, penalty / filter_count)

        return penalty

    def _generate_score_explanation(self, result: RankedMemoryResult, dimension_score: float,
                                  confidence_boost: float, profile_bonus: float,
                                  weights: DimensionWeights, profile: str) -> Tuple[str, Dict[str, float]]:
        """Generate human-readable explanation for scoring."""

        breakdown = {
            'semantic_similarity': result.semantic_score * weights.semantic_similarity,
            'dimension_alignment': dimension_score * weights.dimension_alignment,
            'recency_score': result.recency_score * weights.recency_score,
            'confidence_score': result.confidence_score * weights.confidence_score,
            'confidence_boost': confidence_boost,
            'profile_bonus': profile_bonus
        }

        # Generate explanation
        explanation_parts = []

        if breakdown['semantic_similarity'] > 0.3:
            explanation_parts.append(f"High semantic relevance ({result.semantic_score:.2f})")

        if breakdown['dimension_alignment'] > 0.2:
            explanation_parts.append(f"Strong conceptual alignment ({dimension_score:.2f})")

        if breakdown['confidence_boost'] > 0.05:
            explanation_parts.append("High dimension confidence")

        if breakdown['profile_bonus'] > 0:
            explanation_parts.append(f"Profile match ({profile})")

        if not explanation_parts:
            explanation_parts.append("Standard relevance scoring")

        explanation = f"Ranked highly due to: {', '.join(explanation_parts)}"

        return explanation, breakdown

    def _vector_only_search(self, query: str, max_results: int, weights: DimensionWeights) -> List[DimensionAwareResult]:
        """Traditional vector-only search for fallback."""
        semantic_results = self.memory_store.enhanced_search_memories(
            query=query,
            max_results=max_results
        )

        # Convert to DimensionAwareResult format
        results = []
        for result in semantic_results:
            dim_result = DimensionAwareResult(
                chunk_id=result.chunk_id,
                content=result.content,
                metadata=result.metadata,
                semantic_score=result.semantic_score,
                recency_score=result.recency_score,
                confidence_score=result.confidence_score,
                dimension_alignment_score=0.0,
                dimension_confidence_boost=0.0,
                profile_relevance_bonus=0.0,
                final_score=result.semantic_score,
                score_breakdown={'semantic_similarity': result.semantic_score},
                dimension_explanation="Vector-only search (no dimension analysis)",
                ranking_reason="Traditional semantic similarity"
            )
            results.append(dim_result)

        return results

    def _dimension_only_search(self, query: str, max_results: int, weights: DimensionWeights,
                             profile: str, filters: Dict[str, Any]) -> List[DimensionAwareResult]:
        """Dimension-only search for specialized filtering."""
        if not self.dimension_enabled:
            return self._fallback_search(query, max_results)

        # Get all chunks and score by dimensions only
        all_results = self.memory_store.enhanced_search_memories(
            query=query,
            max_results=max_results * 10  # Get larger pool
        )

        dimension_results = []
        for result in all_results:
            try:
                dimension_score, confidence_boost, profile_bonus = self._calculate_dimension_alignment(
                    result, query, profile, weights.profile_dimensions, filters
                )

                # Use dimension score as primary ranking
                final_score = dimension_score + confidence_boost + profile_bonus

                if final_score > 0.1:  # Only include results with meaningful dimension scores
                    explanation, breakdown = self._generate_score_explanation(
                        result, dimension_score, confidence_boost, profile_bonus, weights, profile
                    )

                    dim_result = DimensionAwareResult(
                        chunk_id=result.chunk_id,
                        content=result.content,
                        metadata=result.metadata,
                        semantic_score=result.semantic_score,
                        recency_score=result.recency_score,
                        confidence_score=result.confidence_score,
                        dimension_alignment_score=dimension_score,
                        dimension_confidence_boost=confidence_boost,
                        profile_relevance_bonus=profile_bonus,
                        final_score=final_score,
                        score_breakdown=breakdown,
                        dimension_explanation=explanation,
                        ranking_reason=f"Dimension-only search with {profile} profile"
                    )
                    dimension_results.append(dim_result)

            except Exception as e:
                logger.warning(f"Error in dimension-only scoring: {e}")
                continue

        # Sort by dimension score and return top results
        dimension_results.sort(key=lambda x: x.final_score, reverse=True)
        return dimension_results[:max_results]

    def _adaptive_search(self, query: str, max_results: int, weights: DimensionWeights,
                        profile: str, filters: Dict[str, Any]) -> List[DimensionAwareResult]:
        """Adaptive search that chooses strategy based on query and available data."""

        # Analyze query to determine best strategy
        if not self.dimension_enabled:
            return self._vector_only_search(query, max_results, weights)

        # If filters are specified, prefer dimension-weighted approach
        if filters:
            return self._hybrid_search(query, max_results, weights, profile, filters)

        # For research queries, emphasize dimensions
        if profile == "researcher" and any(word in query.lower() for word in ['novel', 'innovative', 'research', 'study']):
            # Boost dimension alignment weight
            adaptive_weights = DimensionWeights(
                semantic_similarity=0.3,
                dimension_alignment=0.5,
                recency_score=weights.recency_score,
                confidence_score=weights.confidence_score,
                profile_dimensions=weights.profile_dimensions
            )
            return self._hybrid_search(query, max_results, adaptive_weights, profile, filters)

        # Default to hybrid search
        return self._hybrid_search(query, max_results, weights, profile, filters)

    def _fallback_search(self, query: str, max_results: int) -> List[DimensionAwareResult]:
        """Fallback search when dimension probing is not available."""
        try:
            semantic_results = self.memory_store.search_memories(query, max_results)

            results = []
            for result in semantic_results:
                dim_result = DimensionAwareResult(
                    chunk_id=getattr(result.chunk, 'chunk_id', 'unknown'),
                    content=result.chunk.content,
                    metadata=result.chunk.metadata,
                    semantic_score=result.similarity_score,
                    recency_score=0.0,
                    confidence_score=0.0,
                    dimension_alignment_score=0.0,
                    dimension_confidence_boost=0.0,
                    profile_relevance_bonus=0.0,
                    final_score=result.similarity_score,
                    score_breakdown={'semantic_similarity': result.similarity_score},
                    dimension_explanation="Fallback search (dimension probing unavailable)",
                    ranking_reason="Basic semantic similarity"
                )
                results.append(dim_result)

            return results

        except Exception as e:
            logger.error(f"Error in fallback search: {e}")
            return []
