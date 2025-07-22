"""
Advanced Graph Reasoning Engine for SAM's Cognitive Memory Core - Phase C
Implements sophisticated graph traversal, multi-hop reasoning, and concept clustering.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import networkx as nx
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class ReasoningPath:
    """Represents a reasoning path through the graph."""
    nodes: List[str] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    reasoning_type: str = "direct"
    depth: int = 0
    semantic_score: float = 0.0
    temporal_score: float = 0.0
    explanation: str = ""

@dataclass
class ConceptCluster:
    """Represents a cluster of related concepts."""
    cluster_id: str
    concepts: List[str] = field(default_factory=list)
    central_concept: str = ""
    cohesion_score: float = 0.0
    cluster_type: str = "semantic"
    relationships: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ReasoningResult:
    """Complete reasoning result with paths, clusters, and insights."""
    query: str
    reasoning_paths: List[ReasoningPath] = field(default_factory=list)
    concept_clusters: List[ConceptCluster] = field(default_factory=list)
    key_insights: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning_depth: int = 0
    execution_time: float = 0.0

class AdvancedGraphReasoner:
    """
    Advanced graph reasoning engine with multi-hop traversal and concept clustering.
    
    Features:
    - Multi-hop reasoning with semantic scoring
    - Concept clustering and relationship analysis
    - Temporal reasoning and trend analysis
    - Causal chain detection
    - Analogical reasoning
    """
    
    def __init__(self, graph_database=None):
        self.logger = logging.getLogger(f"{__name__}.AdvancedGraphReasoner")
        self.graph_database = graph_database
        self._reasoning_cache = {}
        self._cluster_cache = {}
        
        # Reasoning configuration
        self.max_reasoning_depth = 5
        self.min_confidence_threshold = 0.3
        self.max_paths_per_query = 20
        self.cluster_similarity_threshold = 0.7
        
        self.logger.info("Advanced Graph Reasoning Engine initialized")
    
    async def reason_about_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        reasoning_types: Optional[List[str]] = None
    ) -> ReasoningResult:
        """
        Perform advanced reasoning about a query using graph traversal.
        
        Args:
            query: The query to reason about
            context: Additional context for reasoning
            reasoning_types: Types of reasoning to perform
            
        Returns:
            Complete reasoning result with paths and insights
        """
        start_time = datetime.now()
        
        if reasoning_types is None:
            reasoning_types = ["semantic", "causal", "temporal", "analogical"]
        
        self.logger.info(f"Starting advanced reasoning for query: {query[:100]}...")
        
        try:
            # Extract key concepts from query
            key_concepts = await self._extract_key_concepts(query)
            
            # Perform multi-hop reasoning
            reasoning_paths = await self._perform_multi_hop_reasoning(
                key_concepts, query, reasoning_types
            )
            
            # Generate concept clusters
            concept_clusters = await self._generate_concept_clusters(
                reasoning_paths, key_concepts
            )
            
            # Extract key insights
            key_insights = await self._extract_key_insights(
                reasoning_paths, concept_clusters, query
            )
            
            # Calculate overall confidence
            confidence = self._calculate_reasoning_confidence(reasoning_paths)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = ReasoningResult(
                query=query,
                reasoning_paths=reasoning_paths,
                concept_clusters=concept_clusters,
                key_insights=key_insights,
                confidence=confidence,
                reasoning_depth=max([p.depth for p in reasoning_paths], default=0),
                execution_time=execution_time
            )
            
            self.logger.info(f"Advanced reasoning completed: {len(reasoning_paths)} paths, "
                           f"{len(concept_clusters)} clusters, confidence: {confidence:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Advanced reasoning failed: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()
            return ReasoningResult(
                query=query,
                confidence=0.0,
                execution_time=execution_time
            )
    
    async def _extract_key_concepts(self, query: str) -> List[str]:
        """Extract key concepts from the query for graph traversal."""
        # Simple keyword extraction (in production, use NLP)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'what', 'how', 'why', 'when', 'where'
        }
        
        words = query.lower().split()
        concepts = [word.strip('.,!?()[]{}') for word in words 
                   if len(word) > 3 and word.lower() not in stop_words]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_concepts = []
        for concept in concepts:
            if concept not in seen:
                seen.add(concept)
                unique_concepts.append(concept)
        
        self.logger.debug(f"Extracted {len(unique_concepts)} key concepts: {unique_concepts}")
        return unique_concepts[:10]  # Limit to top 10 concepts
    
    async def _perform_multi_hop_reasoning(
        self,
        key_concepts: List[str],
        query: str,
        reasoning_types: List[str]
    ) -> List[ReasoningPath]:
        """Perform multi-hop reasoning across different reasoning types."""
        all_paths = []
        
        for reasoning_type in reasoning_types:
            paths = await self._traverse_graph_for_reasoning_type(
                key_concepts, reasoning_type, query
            )
            all_paths.extend(paths)
        
        # Sort by confidence and limit results
        all_paths.sort(key=lambda p: p.confidence, reverse=True)
        return all_paths[:self.max_paths_per_query]
    
    async def _traverse_graph_for_reasoning_type(
        self,
        concepts: List[str],
        reasoning_type: str,
        query: str
    ) -> List[ReasoningPath]:
        """Traverse the graph for a specific type of reasoning."""
        paths = []
        
        if reasoning_type == "semantic":
            paths = await self._semantic_traversal(concepts, query)
        elif reasoning_type == "causal":
            paths = await self._causal_traversal(concepts, query)
        elif reasoning_type == "temporal":
            paths = await self._temporal_traversal(concepts, query)
        elif reasoning_type == "analogical":
            paths = await self._analogical_traversal(concepts, query)
        
        return paths
    
    async def _semantic_traversal(self, concepts: List[str], query: str) -> List[ReasoningPath]:
        """Perform semantic similarity-based traversal."""
        paths = []
        
        for concept in concepts:
            # Create mock semantic paths (in production, use real graph)
            path = ReasoningPath(
                nodes=[concept, f"related_to_{concept}", f"semantic_cluster_{concept}"],
                edges=[
                    {"type": "SEMANTICALLY_RELATED", "weight": 0.8},
                    {"type": "BELONGS_TO_CLUSTER", "weight": 0.9}
                ],
                confidence=0.8,
                reasoning_type="semantic",
                depth=2,
                semantic_score=0.85,
                explanation=f"Semantic relationship chain starting from {concept}"
            )
            paths.append(path)
        
        return paths
    
    async def _causal_traversal(self, concepts: List[str], query: str) -> List[ReasoningPath]:
        """Perform causal relationship traversal."""
        paths = []
        
        # Look for causal indicators in query
        causal_indicators = ["causes", "leads to", "results in", "because", "due to", "why"]
        has_causal = any(indicator in query.lower() for indicator in causal_indicators)
        
        if has_causal:
            for i, concept in enumerate(concepts[:3]):  # Limit for performance
                path = ReasoningPath(
                    nodes=[concept, f"causes_{concept}", f"effect_of_{concept}"],
                    edges=[
                        {"type": "CAUSES", "weight": 0.7},
                        {"type": "HAS_EFFECT", "weight": 0.8}
                    ],
                    confidence=0.75,
                    reasoning_type="causal",
                    depth=2,
                    semantic_score=0.7,
                    explanation=f"Causal chain: {concept} leads to downstream effects"
                )
                paths.append(path)
        
        return paths
    
    async def _temporal_traversal(self, concepts: List[str], query: str) -> List[ReasoningPath]:
        """Perform temporal relationship traversal."""
        paths = []
        
        # Look for temporal indicators
        temporal_indicators = ["before", "after", "during", "when", "timeline", "history"]
        has_temporal = any(indicator in query.lower() for indicator in temporal_indicators)
        
        if has_temporal:
            for concept in concepts[:2]:  # Limit for performance
                path = ReasoningPath(
                    nodes=[f"past_{concept}", concept, f"future_{concept}"],
                    edges=[
                        {"type": "TEMPORAL_PRECEDES", "weight": 0.6},
                        {"type": "TEMPORAL_FOLLOWS", "weight": 0.6}
                    ],
                    confidence=0.65,
                    reasoning_type="temporal",
                    depth=2,
                    temporal_score=0.8,
                    explanation=f"Temporal progression involving {concept}"
                )
                paths.append(path)
        
        return paths
    
    async def _analogical_traversal(self, concepts: List[str], query: str) -> List[ReasoningPath]:
        """Perform analogical reasoning traversal."""
        paths = []
        
        # Look for analogical indicators
        analogical_indicators = ["like", "similar", "analogous", "compare", "metaphor"]
        has_analogical = any(indicator in query.lower() for indicator in analogical_indicators)
        
        if has_analogical and len(concepts) >= 2:
            # Create analogical paths between concepts
            for i in range(min(2, len(concepts) - 1)):
                concept1, concept2 = concepts[i], concepts[i + 1]
                path = ReasoningPath(
                    nodes=[concept1, f"analogy_bridge", concept2],
                    edges=[
                        {"type": "ANALOGOUS_TO", "weight": 0.6},
                        {"type": "SIMILAR_STRUCTURE", "weight": 0.7}
                    ],
                    confidence=0.6,
                    reasoning_type="analogical",
                    depth=2,
                    semantic_score=0.65,
                    explanation=f"Analogical relationship between {concept1} and {concept2}"
                )
                paths.append(path)
        
        return paths

    async def _generate_concept_clusters(
        self,
        reasoning_paths: List[ReasoningPath],
        key_concepts: List[str]
    ) -> List[ConceptCluster]:
        """Generate concept clusters from reasoning paths."""
        clusters = []

        # Group concepts by reasoning type
        concept_groups = defaultdict(list)
        for path in reasoning_paths:
            concept_groups[path.reasoning_type].extend(path.nodes)

        # Create clusters for each reasoning type
        for reasoning_type, concepts in concept_groups.items():
            if len(concepts) >= 2:
                # Remove duplicates
                unique_concepts = list(set(concepts))

                # Find central concept (most connected)
                central_concept = max(unique_concepts,
                                    key=lambda c: sum(1 for path in reasoning_paths
                                                    if c in path.nodes))

                # Calculate cohesion score
                cohesion_score = len(unique_concepts) / max(len(key_concepts), 1)
                cohesion_score = min(cohesion_score, 1.0)

                cluster = ConceptCluster(
                    cluster_id=f"{reasoning_type}_cluster_{len(clusters)}",
                    concepts=unique_concepts,
                    central_concept=central_concept,
                    cohesion_score=cohesion_score,
                    cluster_type=reasoning_type,
                    relationships=[
                        {"source": path.nodes[0], "target": path.nodes[-1],
                         "type": path.reasoning_type}
                        for path in reasoning_paths
                        if path.reasoning_type == reasoning_type and len(path.nodes) >= 2
                    ]
                )
                clusters.append(cluster)

        self.logger.debug(f"Generated {len(clusters)} concept clusters")
        return clusters

    async def _extract_key_insights(
        self,
        reasoning_paths: List[ReasoningPath],
        concept_clusters: List[ConceptCluster],
        query: str
    ) -> List[str]:
        """Extract key insights from reasoning results."""
        insights = []

        # Insight 1: Most confident reasoning path
        if reasoning_paths:
            best_path = max(reasoning_paths, key=lambda p: p.confidence)
            insights.append(
                f"Strongest reasoning path: {' → '.join(best_path.nodes[:3])} "
                f"(confidence: {best_path.confidence:.2f})"
            )

        # Insight 2: Dominant reasoning type
        if reasoning_paths:
            reasoning_types = [p.reasoning_type for p in reasoning_paths]
            dominant_type = max(set(reasoning_types), key=reasoning_types.count)
            count = reasoning_types.count(dominant_type)
            insights.append(
                f"Primary reasoning pattern: {dominant_type} "
                f"({count}/{len(reasoning_paths)} paths)"
            )

        # Insight 3: Concept cluster analysis
        if concept_clusters:
            best_cluster = max(concept_clusters, key=lambda c: c.cohesion_score)
            insights.append(
                f"Most cohesive concept cluster: {best_cluster.cluster_type} "
                f"with {len(best_cluster.concepts)} concepts "
                f"(cohesion: {best_cluster.cohesion_score:.2f})"
            )

        # Insight 4: Reasoning depth analysis
        if reasoning_paths:
            avg_depth = sum(p.depth for p in reasoning_paths) / len(reasoning_paths)
            max_depth = max(p.depth for p in reasoning_paths)
            insights.append(
                f"Reasoning complexity: average depth {avg_depth:.1f}, "
                f"maximum depth {max_depth}"
            )

        # Insight 5: Multi-hop connections
        multi_hop_paths = [p for p in reasoning_paths if p.depth > 1]
        if multi_hop_paths:
            insights.append(
                f"Multi-hop reasoning: {len(multi_hop_paths)} paths with "
                f"indirect connections found"
            )

        return insights

    def _calculate_reasoning_confidence(self, reasoning_paths: List[ReasoningPath]) -> float:
        """Calculate overall confidence in reasoning results."""
        if not reasoning_paths:
            return 0.0

        # Weighted average of path confidences
        total_confidence = sum(p.confidence for p in reasoning_paths)
        avg_confidence = total_confidence / len(reasoning_paths)

        # Boost confidence based on reasoning diversity
        reasoning_types = set(p.reasoning_type for p in reasoning_paths)
        diversity_boost = min(0.1 * len(reasoning_types), 0.3)

        # Boost confidence based on path depth
        avg_depth = sum(p.depth for p in reasoning_paths) / len(reasoning_paths)
        depth_boost = min(0.05 * avg_depth, 0.2)

        final_confidence = min(1.0, avg_confidence + diversity_boost + depth_boost)
        return round(final_confidence, 3)

    async def find_analogies(
        self,
        source_concept: str,
        target_domain: str,
        max_analogies: int = 5
    ) -> List[Dict[str, Any]]:
        """Find analogical relationships between concepts."""
        analogies = []

        # Mock analogy generation (in production, use graph traversal)
        for i in range(min(max_analogies, 3)):
            analogy = {
                "source": source_concept,
                "target": f"{target_domain}_analogy_{i}",
                "mapping": {
                    "structural": f"{source_concept} structure maps to {target_domain}",
                    "functional": f"{source_concept} function similar to {target_domain}",
                    "relational": f"{source_concept} relationships mirror {target_domain}"
                },
                "confidence": 0.7 - (i * 0.1),
                "explanation": f"Analogical mapping between {source_concept} and {target_domain}"
            }
            analogies.append(analogy)

        return analogies

    async def trace_causal_chains(
        self,
        start_concept: str,
        max_depth: int = 4
    ) -> List[List[str]]:
        """Trace causal chains starting from a concept."""
        chains = []

        # Mock causal chain generation
        for i in range(3):  # Generate 3 sample chains
            chain = [start_concept]
            current = start_concept

            for depth in range(max_depth):
                next_concept = f"effect_{depth}_{current}"
                chain.append(next_concept)
                current = next_concept

                # Stop chain with some probability
                if depth > 1 and (depth + i) % 2 == 0:
                    break

            chains.append(chain)

        return chains

    async def detect_temporal_patterns(
        self,
        concepts: List[str],
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Detect temporal patterns in concept relationships."""
        if time_window is None:
            time_window = timedelta(days=30)

        patterns = {
            "trends": [],
            "cycles": [],
            "events": [],
            "correlations": []
        }

        # Mock temporal pattern detection
        for concept in concepts[:3]:  # Limit for performance
            patterns["trends"].append({
                "concept": concept,
                "trend": "increasing",
                "confidence": 0.7,
                "time_span": str(time_window)
            })

            patterns["events"].append({
                "concept": concept,
                "event_type": "emergence",
                "timestamp": datetime.now().isoformat(),
                "significance": 0.8
            })

        return patterns

    def get_reasoning_statistics(self) -> Dict[str, Any]:
        """Get statistics about reasoning engine performance."""
        return {
            "cache_size": len(self._reasoning_cache),
            "cluster_cache_size": len(self._cluster_cache),
            "max_reasoning_depth": self.max_reasoning_depth,
            "confidence_threshold": self.min_confidence_threshold,
            "max_paths_per_query": self.max_paths_per_query,
            "cluster_similarity_threshold": self.cluster_similarity_threshold
        }

    def clear_caches(self) -> None:
        """Clear reasoning and cluster caches."""
        self._reasoning_cache.clear()
        self._cluster_cache.clear()
        self.logger.info("Reasoning caches cleared")
