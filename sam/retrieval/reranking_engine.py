#!/usr/bin/env python3
"""
V2 Reranking Engine for SAM MUVERA Pipeline
Performs accurate token-level reranking using Chamfer Distance and MaxSim scoring.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Global reranking engine instance
_v2_reranking_engine = None

@dataclass
class RerankingConfig:
    """Configuration for reranking engine."""
    similarity_metric: str = "maxsim"  # Similarity metric ('chamfer', 'maxsim')
    max_documents: int = 50           # Maximum documents to rerank
    similarity_threshold: float = 0.0  # Minimum similarity threshold
    enable_parallel: bool = True      # Enable parallel processing
    batch_size: int = 10             # Batch size for processing
    enable_caching: bool = True      # Cache similarity computations

@dataclass
class RerankingResult:
    """Result from document reranking."""
    query: str                       # Original query
    input_documents: List[str]       # Input document IDs
    reranked_documents: List[Tuple[str, float]]  # Reranked (doc_id, score)
    similarity_scores: Dict[str, float]  # All similarity scores
    processing_time: float           # Time taken for reranking
    num_processed: int               # Number of documents processed
    similarity_metric: str           # Similarity metric used
    metadata: Dict[str, Any]         # Additional metadata

class V2RerankingEngine:
    """
    Reranking engine for accurate token-level document scoring.
    
    Uses Chamfer Distance or MaxSim similarity to compute accurate
    similarity scores between query and document token embeddings.
    """
    
    def __init__(self, config: Optional[RerankingConfig] = None):
        """
        Initialize the v2 reranking engine.
        
        Args:
            config: Reranking configuration
        """
        self.config = config or RerankingConfig()
        
        # Components (loaded lazily)
        self.storage_manager = None
        self.similarity_calculator = None
        self.is_initialized = False
        
        # Caching
        self.cache = {} if self.config.enable_caching else None
        
        logger.info(f"🎯 V2RerankingEngine initialized")
        logger.info(f"📊 Metric: {self.config.similarity_metric}, max docs: {self.config.max_documents}")
    
    def _initialize_components(self) -> bool:
        """Initialize reranking components."""
        if self.is_initialized:
            return True
        
        try:
            logger.info("🔄 Initializing v2 reranking components...")
            
            # Initialize storage manager
            from sam.storage import get_v2_storage_manager
            self.storage_manager = get_v2_storage_manager()
            
            # Initialize similarity calculator
            from sam.cognition import get_similarity_calculator
            self.similarity_calculator = get_similarity_calculator(
                method=self.config.similarity_metric
            )
            
            self.is_initialized = True
            logger.info("✅ V2 reranking components initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize v2 reranking components: {e}")
            return False
    
    def _get_cache_key(self, query: str, document_ids: List[str]) -> str:
        """Generate cache key for reranking."""
        import hashlib
        doc_hash = hashlib.md5('|'.join(sorted(document_ids)).encode()).hexdigest()
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return f"{query_hash}_{doc_hash}_{self.config.similarity_metric}"
    
    def rerank_documents(self, 
                        query: str,
                        document_ids: List[str],
                        query_embeddings: Optional[Any] = None) -> Optional[RerankingResult]:
        """
        Rerank documents using token-level similarity.
        
        Args:
            query: User query
            document_ids: List of document IDs to rerank
            query_embeddings: Pre-computed query embeddings (optional)
            
        Returns:
            RerankingResult with reranked documents
        """
        try:
            start_time = time.time()
            
            logger.debug(f"🎯 Reranking {len(document_ids)} documents for query: '{query[:50]}...'")
            
            # Check cache
            if self.config.enable_caching and self.cache is not None:
                cache_key = self._get_cache_key(query, document_ids)
                if cache_key in self.cache:
                    logger.debug("📋 Returning cached reranking result")
                    return self.cache[cache_key]
            
            # Initialize components
            if not self._initialize_components():
                logger.error("❌ Failed to initialize reranking components")
                return None
            
            # Limit number of documents
            if len(document_ids) > self.config.max_documents:
                logger.warning(f"⚠️  Limiting reranking to {self.config.max_documents} documents")
                document_ids = document_ids[:self.config.max_documents]
            
            # Get query embeddings if not provided
            if query_embeddings is None:
                from sam.retrieval.query_processor import get_v2_query_processor
                query_processor = get_v2_query_processor()
                query_result = query_processor.process_query(query)
                
                if not query_result:
                    logger.error("❌ Failed to get query embeddings for reranking")
                    return None
                
                query_embeddings = query_result.token_embeddings
            
            # Compute similarity scores
            similarity_scores = {}
            processed_count = 0
            
            # Process documents in batches
            for i in range(0, len(document_ids), self.config.batch_size):
                batch_ids = document_ids[i:i + self.config.batch_size]
                batch_scores = self._compute_batch_similarities(query_embeddings, batch_ids)
                similarity_scores.update(batch_scores)
                processed_count += len(batch_ids)
                
                logger.debug(f"📊 Processed {processed_count}/{len(document_ids)} documents")
            
            # Sort by similarity score
            reranked_documents = sorted(
                similarity_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Filter by threshold
            if self.config.similarity_threshold > 0:
                reranked_documents = [
                    (doc_id, score) for doc_id, score in reranked_documents
                    if score >= self.config.similarity_threshold
                ]
            
            processing_time = time.time() - start_time
            
            # Create result
            result = RerankingResult(
                query=query,
                input_documents=document_ids,
                reranked_documents=reranked_documents,
                similarity_scores=similarity_scores,
                processing_time=processing_time,
                num_processed=processed_count,
                similarity_metric=self.config.similarity_metric,
                metadata={
                    'config': self.config.__dict__,
                    'reranking_timestamp': time.time(),
                    'top_score': reranked_documents[0][1] if reranked_documents else 0.0,
                    'avg_score': sum(similarity_scores.values()) / len(similarity_scores) if similarity_scores else 0.0,
                    'filtered_count': len(reranked_documents)
                }
            )
            
            # Cache result
            if self.config.enable_caching and self.cache is not None:
                cache_key = self._get_cache_key(query, document_ids)
                self.cache[cache_key] = result
            
            logger.debug(f"✅ Reranking completed: {len(reranked_documents)} documents, {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Document reranking failed: {e}")
            return None
    
    def _compute_batch_similarities(self, 
                                   query_embeddings: Any,
                                   document_ids: List[str]) -> Dict[str, float]:
        """Compute similarity scores for a batch of documents."""
        batch_scores = {}
        
        try:
            for doc_id in document_ids:
                try:
                    # Load document embeddings
                    doc_embeddings = self.storage_manager.load_token_embeddings(doc_id)
                    
                    if doc_embeddings is None:
                        logger.warning(f"⚠️  Failed to load embeddings for document: {doc_id}")
                        batch_scores[doc_id] = 0.0
                        continue
                    
                    # Compute similarity
                    similarity_result = self.similarity_calculator.compute_similarity(
                        query_embeddings, doc_embeddings
                    )
                    
                    batch_scores[doc_id] = similarity_result.score
                    
                except Exception as e:
                    logger.warning(f"⚠️  Failed to compute similarity for {doc_id}: {e}")
                    batch_scores[doc_id] = 0.0
            
            return batch_scores
            
        except Exception as e:
            logger.error(f"❌ Batch similarity computation failed: {e}")
            return {doc_id: 0.0 for doc_id in document_ids}
    
    def compute_pairwise_similarity(self, 
                                   query_embeddings: Any,
                                   doc_embeddings: Any) -> float:
        """
        Compute similarity between query and document embeddings.
        
        Args:
            query_embeddings: Query token embeddings
            doc_embeddings: Document token embeddings
            
        Returns:
            Similarity score
        """
        try:
            if not self._initialize_components():
                return 0.0
            
            similarity_result = self.similarity_calculator.compute_similarity(
                query_embeddings, doc_embeddings
            )
            
            return similarity_result.score
            
        except Exception as e:
            logger.error(f"❌ Pairwise similarity computation failed: {e}")
            return 0.0
    
    def get_reranking_stats(self) -> Dict[str, Any]:
        """Get reranking engine statistics."""
        try:
            return {
                'config': self.config.__dict__,
                'is_initialized': self.is_initialized,
                'cache_size': len(self.cache) if self.cache else 0,
                'components': {
                    'storage_manager': self.storage_manager is not None,
                    'similarity_calculator': self.similarity_calculator is not None
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get reranking stats: {e}")
            return {'error': str(e)}
    
    def clear_cache(self):
        """Clear the reranking cache."""
        if self.cache is not None:
            self.cache.clear()
            logger.info("🗑️  Reranking cache cleared")
    
    def benchmark_similarity_metrics(self, 
                                    query_embeddings: Any,
                                    doc_embeddings: Any) -> Dict[str, float]:
        """
        Benchmark different similarity metrics on the same embeddings.
        
        Args:
            query_embeddings: Query token embeddings
            doc_embeddings: Document token embeddings
            
        Returns:
            Dictionary of metric scores
        """
        try:
            from sam.cognition import ChamferSimilarity, MaxSimSimilarity
            
            results = {}
            
            # Test Chamfer similarity
            chamfer_calc = ChamferSimilarity(distance_metric="cosine")
            chamfer_result = chamfer_calc.compute_similarity(query_embeddings, doc_embeddings)
            results['chamfer_cosine'] = chamfer_result.score
            
            # Test MaxSim similarity
            maxsim_calc = MaxSimSimilarity(similarity_metric="cosine")
            maxsim_result = maxsim_calc.compute_similarity(query_embeddings, doc_embeddings)
            results['maxsim_cosine'] = maxsim_result.score
            
            # Test dot product variants
            maxsim_dot_calc = MaxSimSimilarity(similarity_metric="dot")
            maxsim_dot_result = maxsim_dot_calc.compute_similarity(query_embeddings, doc_embeddings)
            results['maxsim_dot'] = maxsim_dot_result.score
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Similarity metric benchmarking failed: {e}")
            return {}

def get_v2_reranking_engine(config: Optional[RerankingConfig] = None) -> V2RerankingEngine:
    """
    Get or create a v2 reranking engine instance.
    
    Args:
        config: Reranking configuration
        
    Returns:
        V2RerankingEngine instance
    """
    global _v2_reranking_engine
    
    if _v2_reranking_engine is None:
        _v2_reranking_engine = V2RerankingEngine(config)
    
    return _v2_reranking_engine

def rerank_documents_v2(query: str,
                       document_ids: List[str],
                       similarity_metric: str = "maxsim") -> Optional[RerankingResult]:
    """
    Convenience function to rerank documents using v2 pipeline.
    
    Args:
        query: User query
        document_ids: List of document IDs to rerank
        similarity_metric: Similarity metric to use
        
    Returns:
        RerankingResult with reranked documents
    """
    config = RerankingConfig(similarity_metric=similarity_metric)
    engine = get_v2_reranking_engine(config)
    return engine.rerank_documents(query, document_ids)
