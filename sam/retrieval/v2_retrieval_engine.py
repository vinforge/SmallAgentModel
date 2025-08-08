#!/usr/bin/env python3
"""
V2 Retrieval Engine for SAM MUVERA Pipeline
Implements two-stage retrieval: fast FDE search + accurate token-level reranking.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Global retrieval engine instance
_v2_retrieval_engine = None

@dataclass
class V2RetrievalConfig:
    """Configuration for v2 retrieval engine."""
    stage1_top_k: int = 50          # Top-K for fast FDE search
    stage2_top_k: int = 10          # Top-K for final reranking
    similarity_metric: str = "maxsim"  # Similarity metric for reranking
    fde_similarity_threshold: float = 0.1  # Minimum FDE similarity
    rerank_similarity_threshold: float = 0.3  # Minimum rerank similarity
    enable_query_expansion: bool = False  # Query expansion (future feature)
    enable_caching: bool = True     # Cache retrieval results
    max_cache_size: int = 1000      # Maximum cache entries

@dataclass
class V2RetrievalResult:
    """Result from v2 retrieval engine."""
    query: str                      # Original query
    stage1_results: List[Tuple[str, float]]  # FDE search results (doc_id, score)
    stage2_results: List[Tuple[str, float]]  # Reranked results (doc_id, score)
    final_documents: List[str]      # Final document IDs
    stage1_time: float              # Time for stage 1
    stage2_time: float              # Time for stage 2
    total_time: float               # Total retrieval time
    num_candidates: int             # Number of candidates from stage 1
    num_reranked: int               # Number of documents reranked
    metadata: Dict[str, Any]        # Additional metadata

class V2RetrievalEngine:
    """
    Two-stage retrieval engine implementing the MUVERA approach.
    
    Stage 1: Fast FDE-based similarity search in ChromaDB
    Stage 2: Accurate token-level reranking with Chamfer/MaxSim
    """
    
    def __init__(self, config: Optional[V2RetrievalConfig] = None):
        """
        Initialize the v2 retrieval engine.
        
        Args:
            config: Retrieval configuration
        """
        self.config = config or V2RetrievalConfig()
        
        # Components (loaded lazily)
        self.embedder = None
        self.fde_transformer = None
        self.storage_manager = None
        self.reranking_engine = None
        self.query_processor = None
        self.is_initialized = False
        
        # Caching
        self.cache = {} if self.config.enable_caching else None
        
        logger.info(f"🔍 V2RetrievalEngine initialized")
        logger.info(f"📊 Stage1 top-k: {self.config.stage1_top_k}, Stage2 top-k: {self.config.stage2_top_k}")
    
    def _initialize_components(self) -> bool:
        """Initialize all retrieval components."""
        if self.is_initialized:
            return True
        
        try:
            logger.info("🔄 Initializing v2 retrieval components...")
            
            # Initialize embedder
            from sam.embedding import get_multivector_embedder
            self.embedder = get_multivector_embedder()
            
            # Initialize FDE transformer
            from sam.cognition import get_muvera_fde
            self.fde_transformer = get_muvera_fde()
            
            # Initialize storage manager
            from sam.storage import get_v2_storage_manager
            self.storage_manager = get_v2_storage_manager()
            
            # Initialize reranking engine
            from sam.retrieval.reranking_engine import get_v2_reranking_engine
            self.reranking_engine = get_v2_reranking_engine()
            
            # Initialize query processor
            from sam.retrieval.query_processor import get_v2_query_processor
            self.query_processor = get_v2_query_processor()
            
            self.is_initialized = True
            logger.info("✅ V2 retrieval components initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize v2 retrieval components: {e}")
            return False
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query."""
        import hashlib
        return hashlib.md5(f"{query}_{self.config.stage1_top_k}_{self.config.stage2_top_k}".encode()).hexdigest()
    
    def retrieve(self, query: str) -> V2RetrievalResult:
        """
        Perform two-stage retrieval for the given query.
        
        Args:
            query: User query
            
        Returns:
            V2RetrievalResult with retrieval results
        """
        try:
            total_start_time = time.time()
            
            logger.info(f"🔍 V2 retrieval: '{query[:50]}...'")
            
            # Check cache
            if self.config.enable_caching and self.cache is not None:
                cache_key = self._get_cache_key(query)
                if cache_key in self.cache:
                    logger.debug("📋 Returning cached result")
                    return self.cache[cache_key]
            
            # Initialize components
            if not self._initialize_components():
                return self._create_empty_result(query, "Failed to initialize components")
            
            # Stage 1: Fast FDE search
            stage1_start_time = time.time()
            stage1_results = self._stage1_fde_search(query)
            stage1_time = time.time() - stage1_start_time
            
            if not stage1_results:
                logger.warning("⚠️  Stage 1 returned no results")
                return self._create_empty_result(query, "No documents found in FDE search")
            
            logger.info(f"📄 Stage 1: {len(stage1_results)} candidates in {stage1_time:.3f}s")
            
            # Stage 2: Accurate reranking
            stage2_start_time = time.time()
            stage2_results = self._stage2_reranking(query, stage1_results)
            stage2_time = time.time() - stage2_start_time
            
            logger.info(f"🎯 Stage 2: {len(stage2_results)} reranked in {stage2_time:.3f}s")
            
            # Create result
            total_time = time.time() - total_start_time
            
            final_documents = [doc_id for doc_id, _ in stage2_results[:self.config.stage2_top_k]]
            
            result = V2RetrievalResult(
                query=query,
                stage1_results=stage1_results,
                stage2_results=stage2_results,
                final_documents=final_documents,
                stage1_time=stage1_time,
                stage2_time=stage2_time,
                total_time=total_time,
                num_candidates=len(stage1_results),
                num_reranked=len(stage2_results),
                metadata={
                    'config': self.config.__dict__,
                    'retrieval_timestamp': time.time(),
                    'stage1_top_score': stage1_results[0][1] if stage1_results else 0.0,
                    'stage2_top_score': stage2_results[0][1] if stage2_results else 0.0
                }
            )
            
            # Cache result
            if self.config.enable_caching and self.cache is not None:
                cache_key = self._get_cache_key(query)
                if len(self.cache) >= self.config.max_cache_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                self.cache[cache_key] = result
            
            logger.info(f"✅ V2 retrieval completed: {len(final_documents)} documents, {total_time:.3f}s total")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ V2 retrieval failed: {e}")
            return self._create_empty_result(query, str(e))
    
    def _stage1_fde_search(self, query: str) -> List[Tuple[str, float]]:
        """Stage 1: Fast FDE-based similarity search."""
        try:
            # Process query to get FDE vector
            query_result = self.query_processor.process_query(query)
            
            if not query_result or not query_result.fde_vector:
                logger.error("❌ Failed to generate query FDE vector")
                return []
            
            # Search using FDE vector
            search_results = self.storage_manager.search_by_fde(
                query_fde=query_result.fde_vector,
                top_k=self.config.stage1_top_k
            )
            
            # Filter by similarity threshold
            filtered_results = [
                (doc_id, score) for doc_id, score in search_results
                if score >= self.config.fde_similarity_threshold
            ]
            
            logger.debug(f"📊 FDE search: {len(search_results)} → {len(filtered_results)} after filtering")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"❌ Stage 1 FDE search failed: {e}")
            return []
    
    def _stage2_reranking(self, query: str, candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Stage 2: Accurate token-level reranking."""
        try:
            if not candidates:
                return []
            
            # Get candidate document IDs
            candidate_ids = [doc_id for doc_id, _ in candidates]
            
            # Rerank using token-level similarity
            rerank_result = self.reranking_engine.rerank_documents(
                query=query,
                document_ids=candidate_ids,
                similarity_metric=self.config.similarity_metric
            )
            
            if not rerank_result or not rerank_result.reranked_documents:
                logger.warning("⚠️  Reranking returned no results")
                return candidates  # Fallback to FDE results
            
            # Filter by reranking threshold
            filtered_results = [
                (doc_id, score) for doc_id, score in rerank_result.reranked_documents
                if score >= self.config.rerank_similarity_threshold
            ]
            
            logger.debug(f"🎯 Reranking: {len(rerank_result.reranked_documents)} → {len(filtered_results)} after filtering")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"❌ Stage 2 reranking failed: {e}")
            return candidates  # Fallback to FDE results
    
    def _create_empty_result(self, query: str, error_message: str) -> V2RetrievalResult:
        """Create empty result for failed retrieval."""
        return V2RetrievalResult(
            query=query,
            stage1_results=[],
            stage2_results=[],
            final_documents=[],
            stage1_time=0.0,
            stage2_time=0.0,
            total_time=0.0,
            num_candidates=0,
            num_reranked=0,
            metadata={'error': error_message}
        )
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval engine statistics."""
        try:
            # Get storage stats
            storage_stats = self.storage_manager.get_storage_stats() if self.storage_manager else {}
            
            return {
                'config': self.config.__dict__,
                'is_initialized': self.is_initialized,
                'cache_size': len(self.cache) if self.cache else 0,
                'storage_stats': storage_stats,
                'components': {
                    'embedder': self.embedder is not None,
                    'fde_transformer': self.fde_transformer is not None,
                    'storage_manager': self.storage_manager is not None,
                    'reranking_engine': self.reranking_engine is not None,
                    'query_processor': self.query_processor is not None
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get retrieval stats: {e}")
            return {'error': str(e)}
    
    def clear_cache(self):
        """Clear the retrieval cache."""
        if self.cache is not None:
            self.cache.clear()
            logger.info("🗑️  Retrieval cache cleared")

def get_v2_retrieval_engine(config: Optional[V2RetrievalConfig] = None) -> V2RetrievalEngine:
    """
    Get or create a v2 retrieval engine instance.
    
    Args:
        config: Retrieval configuration
        
    Returns:
        V2RetrievalEngine instance
    """
    global _v2_retrieval_engine
    
    if _v2_retrieval_engine is None:
        _v2_retrieval_engine = V2RetrievalEngine(config)
    
    return _v2_retrieval_engine

def retrieve_v2_documents(query: str, 
                         stage1_top_k: int = 50,
                         stage2_top_k: int = 10) -> V2RetrievalResult:
    """
    Convenience function to retrieve documents using v2 pipeline.
    
    Args:
        query: User query
        stage1_top_k: Top-K for FDE search
        stage2_top_k: Top-K for final results
        
    Returns:
        V2RetrievalResult with retrieval results
    """
    config = V2RetrievalConfig(
        stage1_top_k=stage1_top_k,
        stage2_top_k=stage2_top_k
    )
    
    engine = get_v2_retrieval_engine(config)
    return engine.retrieve(query)
