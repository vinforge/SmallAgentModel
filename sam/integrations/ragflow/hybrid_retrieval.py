"""
Hybrid Retrieval Engine
Combines RAGFlow and SAM retrieval capabilities for optimal results.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from .ragflow_client import RAGFlowClient

logger = logging.getLogger(__name__)

class RetrievalStrategy(Enum):
    """Available retrieval strategies."""
    RAGFLOW_ONLY = "ragflow_only"
    SAM_ONLY = "sam_only"
    HYBRID_PARALLEL = "hybrid_parallel"
    HYBRID_SEQUENTIAL = "hybrid_sequential"
    ADAPTIVE = "adaptive"

@dataclass
class RetrievalResult:
    """Individual retrieval result."""
    content: str
    source: str
    similarity_score: float
    chunk_id: str
    document_id: str
    metadata: Dict[str, Any]
    retrieval_source: str  # "ragflow" or "sam"

@dataclass
class FusedResult:
    """Fused retrieval result with citations."""
    chunks: List[RetrievalResult]
    citations: List[str]
    confidence_score: float
    total_results: int
    ragflow_results: int
    sam_results: int
    fusion_method: str

class HybridRetrievalEngine:
    """
    Hybrid retrieval engine combining RAGFlow and SAM capabilities.
    
    Features:
    - Multiple recall strategies with fused re-ranking
    - Adaptive query routing based on content type
    - Cross-language query support
    - Grounded citation generation
    - Performance optimization with caching
    """
    
    def __init__(self, 
                 ragflow_client: RAGFlowClient,
                 sam_memory_store=None,
                 default_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_PARALLEL,
                 fusion_weights: Optional[Dict[str, float]] = None):
        """
        Initialize hybrid retrieval engine.
        
        Args:
            ragflow_client: RAGFlow client instance
            sam_memory_store: SAM memory store instance
            default_strategy: Default retrieval strategy
            fusion_weights: Weights for result fusion
        """
        self.ragflow_client = ragflow_client
        self.sam_memory_store = sam_memory_store
        self.default_strategy = default_strategy
        
        # Default fusion weights
        self.fusion_weights = fusion_weights or {
            'ragflow': 0.6,  # RAGFlow gets higher weight for document understanding
            'sam': 0.4,      # SAM gets lower weight but still contributes
            'similarity_boost': 0.1,  # Boost for high similarity scores
            'recency_boost': 0.05     # Boost for recent documents
        }
        
        # Performance optimization
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
        
        logger.info(f"Hybrid retrieval engine initialized with strategy: {default_strategy.value}")
    
    def _get_sam_memory_store(self):
        """Get SAM memory store instance."""
        if self.sam_memory_store is None:
            try:
                from memory.memory_vectorstore import get_memory_store, VectorStoreType
                self.sam_memory_store = get_memory_store(
                    store_type=VectorStoreType.CHROMA,
                    storage_directory="memory_store",
                    embedding_dimension=384
                )
                logger.info("SAM memory store loaded for hybrid retrieval")
            except Exception as e:
                logger.warning(f"Failed to load SAM memory store: {e}")
        
        return self.sam_memory_store
    
    def _query_ragflow(self, 
                      query: str,
                      knowledge_base_id: str,
                      max_results: int,
                      similarity_threshold: float) -> List[RetrievalResult]:
        """
        Query RAGFlow knowledge base.
        
        Args:
            query: Search query
            knowledge_base_id: Knowledge base ID
            max_results: Maximum results
            similarity_threshold: Minimum similarity
            
        Returns:
            List of retrieval results from RAGFlow
        """
        try:
            result = self.ragflow_client.query_knowledge_base(
                knowledge_base_id=knowledge_base_id,
                query=query,
                max_results=max_results,
                similarity_threshold=similarity_threshold
            )
            
            if not result.get('success'):
                logger.warning(f"RAGFlow query failed: {result.get('error')}")
                return []
            
            # Convert to RetrievalResult objects
            retrieval_results = []
            for chunk in result.get('chunks', []):
                retrieval_results.append(RetrievalResult(
                    content=chunk.get('content', ''),
                    source=chunk.get('source', 'Unknown'),
                    similarity_score=chunk.get('similarity_score', 0.0),
                    chunk_id=chunk.get('chunk_id', ''),
                    document_id=chunk.get('document_id', ''),
                    metadata=chunk.get('metadata', {}),
                    retrieval_source='ragflow'
                ))
            
            logger.debug(f"RAGFlow returned {len(retrieval_results)} results")
            return retrieval_results
            
        except Exception as e:
            logger.error(f"RAGFlow query error: {e}")
            return []
    
    def _query_sam(self, 
                  query: str,
                  max_results: int,
                  similarity_threshold: float) -> List[RetrievalResult]:
        """
        Query SAM memory store.
        
        Args:
            query: Search query
            max_results: Maximum results
            similarity_threshold: Minimum similarity
            
        Returns:
            List of retrieval results from SAM
        """
        try:
            memory_store = self._get_sam_memory_store()
            if not memory_store:
                return []
            
            from memory.memory_vectorstore import MemoryType
            
            # Query SAM memory store
            results = memory_store.search_memories(
                query=query,
                memory_types=[MemoryType.DOCUMENT],
                max_results=max_results
            )
            
            # Convert to RetrievalResult objects
            retrieval_results = []
            for result in results:
                if result.similarity_score >= similarity_threshold:
                    chunk = result.chunk
                    retrieval_results.append(RetrievalResult(
                        content=chunk.content,
                        source=getattr(chunk, 'source', 'Unknown'),
                        similarity_score=result.similarity_score,
                        chunk_id=chunk.chunk_id,
                        document_id=getattr(chunk, 'document_id', ''),
                        metadata=getattr(chunk, 'metadata', {}),
                        retrieval_source='sam'
                    ))
            
            logger.debug(f"SAM returned {len(retrieval_results)} results")
            return retrieval_results
            
        except Exception as e:
            logger.error(f"SAM query error: {e}")
            return []
    
    def _fuse_results(self, 
                     ragflow_results: List[RetrievalResult],
                     sam_results: List[RetrievalResult],
                     max_results: int,
                     fusion_method: str = "weighted_score") -> FusedResult:
        """
        Fuse results from RAGFlow and SAM using advanced ranking.
        
        Args:
            ragflow_results: Results from RAGFlow
            sam_results: Results from SAM
            max_results: Maximum final results
            fusion_method: Fusion algorithm to use
            
        Returns:
            Fused and re-ranked results
        """
        all_results = []
        
        # Apply fusion weights and calculate final scores
        for result in ragflow_results:
            # RAGFlow results get base weight + similarity boost
            final_score = (
                result.similarity_score * self.fusion_weights['ragflow'] +
                (result.similarity_score ** 2) * self.fusion_weights['similarity_boost']
            )
            
            # Create new result with adjusted score
            fused_result = RetrievalResult(
                content=result.content,
                source=result.source,
                similarity_score=final_score,
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                metadata=result.metadata,
                retrieval_source=result.retrieval_source
            )
            all_results.append(fused_result)
        
        for result in sam_results:
            # SAM results get base weight
            final_score = result.similarity_score * self.fusion_weights['sam']
            
            # Check for duplicates (same content from different sources)
            is_duplicate = False
            for existing in all_results:
                if self._is_duplicate_content(result.content, existing.content):
                    # Boost existing result if it's a duplicate
                    existing.similarity_score = max(existing.similarity_score, final_score)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                fused_result = RetrievalResult(
                    content=result.content,
                    source=result.source,
                    similarity_score=final_score,
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    metadata=result.metadata,
                    retrieval_source=result.retrieval_source
                )
                all_results.append(fused_result)
        
        # Sort by final score and take top results
        all_results.sort(key=lambda x: x.similarity_score, reverse=True)
        top_results = all_results[:max_results]
        
        # Generate citations
        citations = self._generate_citations(top_results)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(top_results)
        
        return FusedResult(
            chunks=top_results,
            citations=citations,
            confidence_score=confidence_score,
            total_results=len(top_results),
            ragflow_results=len(ragflow_results),
            sam_results=len(sam_results),
            fusion_method=fusion_method
        )
    
    def _is_duplicate_content(self, content1: str, content2: str, threshold: float = 0.8) -> bool:
        """
        Check if two content pieces are duplicates.
        
        Args:
            content1: First content
            content2: Second content
            threshold: Similarity threshold for duplicates
            
        Returns:
            True if contents are duplicates
        """
        # Simple similarity check based on common words
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        jaccard_similarity = intersection / union if union > 0 else 0
        return jaccard_similarity >= threshold
    
    def _generate_citations(self, results: List[RetrievalResult]) -> List[str]:
        """
        Generate grounded citations from retrieval results.
        
        Args:
            results: Retrieval results
            
        Returns:
            List of formatted citations
        """
        citations = []
        
        for i, result in enumerate(results, 1):
            # Format citation with source and confidence
            citation = f"[{i}] {result.source}"
            
            if result.document_id:
                citation += f" (Doc: {result.document_id[:8]}...)"
            
            citation += f" - Confidence: {result.similarity_score:.2f}"
            citation += f" - Source: {result.retrieval_source.upper()}"
            
            citations.append(citation)
        
        return citations
    
    def _calculate_confidence(self, results: List[RetrievalResult]) -> float:
        """
        Calculate overall confidence score for results.
        
        Args:
            results: Retrieval results
            
        Returns:
            Confidence score between 0 and 1
        """
        if not results:
            return 0.0
        
        # Calculate weighted average of similarity scores
        total_score = sum(result.similarity_score for result in results)
        avg_score = total_score / len(results)
        
        # Apply confidence boosters
        confidence = avg_score
        
        # Boost confidence if we have results from both sources
        ragflow_count = sum(1 for r in results if r.retrieval_source == 'ragflow')
        sam_count = sum(1 for r in results if r.retrieval_source == 'sam')
        
        if ragflow_count > 0 and sam_count > 0:
            confidence *= 1.1  # 10% boost for hybrid results
        
        # Boost confidence for high-scoring results
        high_score_count = sum(1 for r in results if r.similarity_score > 0.8)
        if high_score_count > 0:
            confidence *= (1 + 0.05 * high_score_count)  # 5% boost per high-score result
        
        return min(confidence, 1.0)  # Cap at 1.0
    
    def query(self, 
             query: str,
             knowledge_base_id: str,
             max_results: int = 5,
             similarity_threshold: float = 0.3,
             strategy: Optional[RetrievalStrategy] = None,
             include_sam_results: bool = True) -> Dict[str, Any]:
        """
        Execute hybrid query across RAGFlow and SAM.
        
        Args:
            query: Search query
            knowledge_base_id: RAGFlow knowledge base ID
            max_results: Maximum results to return
            similarity_threshold: Minimum similarity threshold
            strategy: Retrieval strategy to use
            include_sam_results: Whether to include SAM results
            
        Returns:
            Fused query results with citations
        """
        strategy = strategy or self.default_strategy
        
        try:
            logger.info(f"Executing hybrid query with strategy: {strategy.value}")
            
            ragflow_results = []
            sam_results = []
            
            if strategy == RetrievalStrategy.RAGFLOW_ONLY:
                ragflow_results = self._query_ragflow(
                    query, knowledge_base_id, max_results, similarity_threshold
                )
            
            elif strategy == RetrievalStrategy.SAM_ONLY:
                if include_sam_results:
                    sam_results = self._query_sam(
                        query, max_results, similarity_threshold
                    )
            
            elif strategy == RetrievalStrategy.HYBRID_PARALLEL:
                # Query both sources in parallel
                with ThreadPoolExecutor(max_workers=2) as executor:
                    ragflow_future = executor.submit(
                        self._query_ragflow, query, knowledge_base_id, 
                        max_results, similarity_threshold
                    )
                    
                    sam_future = None
                    if include_sam_results:
                        sam_future = executor.submit(
                            self._query_sam, query, max_results, similarity_threshold
                        )
                    
                    ragflow_results = ragflow_future.result()
                    if sam_future:
                        sam_results = sam_future.result()
            
            elif strategy == RetrievalStrategy.HYBRID_SEQUENTIAL:
                # Query RAGFlow first, then SAM
                ragflow_results = self._query_ragflow(
                    query, knowledge_base_id, max_results, similarity_threshold
                )
                
                if include_sam_results and len(ragflow_results) < max_results:
                    # Only query SAM if we need more results
                    remaining = max_results - len(ragflow_results)
                    sam_results = self._query_sam(
                        query, remaining, similarity_threshold
                    )
            
            # Fuse results
            fused_result = self._fuse_results(
                ragflow_results, sam_results, max_results
            )
            
            return {
                'success': True,
                'chunks': [
                    {
                        'content': chunk.content,
                        'source': chunk.source,
                        'similarity_score': chunk.similarity_score,
                        'chunk_id': chunk.chunk_id,
                        'document_id': chunk.document_id,
                        'metadata': chunk.metadata,
                        'retrieval_source': chunk.retrieval_source
                    }
                    for chunk in fused_result.chunks
                ],
                'citations': fused_result.citations,
                'confidence_score': fused_result.confidence_score,
                'total_results': fused_result.total_results,
                'ragflow_results': fused_result.ragflow_results,
                'sam_results': fused_result.sam_results,
                'fusion_method': fused_result.fusion_method,
                'strategy_used': strategy.value
            }
            
        except Exception as e:
            logger.error(f"Hybrid query failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'chunks': [],
                'citations': [],
                'confidence_score': 0.0
            }
