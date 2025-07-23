#!/usr/bin/env python3
"""
V2 Query Processor for SAM MUVERA Pipeline
Processes user queries into multi-vector embeddings and FDE vectors for retrieval.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Global query processor instance
_v2_query_processor = None

@dataclass
class QueryEmbeddingResult:
    """Result from query embedding processing."""
    query: str                      # Original query
    token_embeddings: Any          # Token-level embeddings
    fde_vector: Any                # Fixed dimensional encoding
    num_tokens: int                # Number of query tokens
    embedding_dim: int             # Embedding dimension
    fde_dim: int                   # FDE dimension
    processing_time: float         # Time taken for processing
    metadata: Dict[str, Any]       # Additional metadata

class V2QueryProcessor:
    """
    Query processor for v2 retrieval pipeline.
    
    Converts user queries into:
    - Multi-vector token embeddings (for accurate reranking)
    - FDE vectors (for fast similarity search)
    """
    
    def __init__(self, 
                 max_query_length: int = 64,
                 enable_query_preprocessing: bool = True,
                 enable_caching: bool = True):
        """
        Initialize the v2 query processor.
        
        Args:
            max_query_length: Maximum query length in tokens
            enable_query_preprocessing: Enable query preprocessing
            enable_caching: Enable query result caching
        """
        self.max_query_length = max_query_length
        self.enable_query_preprocessing = enable_query_preprocessing
        self.enable_caching = enable_caching
        
        # Components (loaded lazily)
        self.embedder = None
        self.fde_transformer = None
        self.is_initialized = False
        
        # Caching
        self.cache = {} if enable_caching else None
        
        logger.info(f"🔍 V2QueryProcessor initialized")
        logger.info(f"📏 Max query length: {max_query_length}")
    
    def _initialize_components(self) -> bool:
        """Initialize query processing components."""
        if self.is_initialized:
            return True
        
        try:
            logger.info("🔄 Initializing v2 query processing components...")
            
            # Initialize embedder
            from sam.embedding import get_multivector_embedder
            self.embedder = get_multivector_embedder()
            
            # Initialize FDE transformer
            from sam.cognition import get_muvera_fde
            self.fde_transformer = get_muvera_fde()
            
            self.is_initialized = True
            logger.info("✅ V2 query processing components initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize v2 query processing components: {e}")
            return False
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess the query for better retrieval."""
        if not self.enable_query_preprocessing:
            return query
        
        try:
            # Basic preprocessing
            processed_query = query.strip()
            
            # Remove excessive whitespace
            processed_query = ' '.join(processed_query.split())
            
            # Convert to lowercase for consistency (optional)
            # processed_query = processed_query.lower()
            
            # Remove special characters that might interfere with embedding
            import re
            processed_query = re.sub(r'[^\w\s\-\.\?\!]', ' ', processed_query)
            processed_query = ' '.join(processed_query.split())
            
            # Truncate if too long
            words = processed_query.split()
            if len(words) > self.max_query_length:
                processed_query = ' '.join(words[:self.max_query_length])
                logger.debug(f"📏 Query truncated to {self.max_query_length} words")
            
            return processed_query
            
        except Exception as e:
            logger.warning(f"⚠️  Query preprocessing failed: {e}, using original")
            return query
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query."""
        import hashlib
        return hashlib.md5(query.encode()).hexdigest()
    
    def process_query(self, query: str) -> Optional[QueryEmbeddingResult]:
        """
        Process a query into embeddings and FDE vector.
        
        Args:
            query: User query
            
        Returns:
            QueryEmbeddingResult with processed embeddings
        """
        try:
            start_time = time.time()
            
            logger.debug(f"🔄 Processing query: '{query[:50]}...'")
            
            # Check cache
            if self.enable_caching and self.cache is not None:
                cache_key = self._get_cache_key(query)
                if cache_key in self.cache:
                    logger.debug("📋 Returning cached query result")
                    return self.cache[cache_key]
            
            # Initialize components
            if not self._initialize_components():
                logger.error("❌ Failed to initialize query processing components")
                return None
            
            # Preprocess query
            processed_query = self._preprocess_query(query)
            
            if not processed_query.strip():
                logger.error("❌ Empty query after preprocessing")
                return None
            
            # Generate multi-vector embeddings
            embedding_result = self.embedder.embed_query(processed_query)
            
            if not embedding_result:
                logger.error("❌ Failed to generate query embeddings")
                return None
            
            logger.debug(f"🧠 Query embeddings: {embedding_result.num_tokens} tokens, {embedding_result.embedding_dim}D")
            
            # Generate FDE vector
            fde_result = self.fde_transformer.generate_fde(
                embedding_result.token_embeddings,
                doc_id=f"query_{int(time.time())}"
            )
            
            if not fde_result:
                logger.error("❌ Failed to generate query FDE vector")
                return None
            
            logger.debug(f"🔄 Query FDE: {fde_result.fde_dim}D vector")
            
            processing_time = time.time() - start_time
            
            # Create result
            result = QueryEmbeddingResult(
                query=query,
                token_embeddings=embedding_result.token_embeddings,
                fde_vector=fde_result.fde_vector,
                num_tokens=embedding_result.num_tokens,
                embedding_dim=embedding_result.embedding_dim,
                fde_dim=fde_result.fde_dim,
                processing_time=processing_time,
                metadata={
                    'original_query': query,
                    'processed_query': processed_query,
                    'preprocessing_enabled': self.enable_query_preprocessing,
                    'max_query_length': self.max_query_length,
                    'embedding_metadata': embedding_result.metadata,
                    'fde_metadata': fde_result.metadata,
                    'processing_timestamp': time.time()
                }
            )
            
            # Cache result
            if self.enable_caching and self.cache is not None:
                cache_key = self._get_cache_key(query)
                self.cache[cache_key] = result
            
            logger.debug(f"✅ Query processed: {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Query processing failed: {e}")
            return None
    
    def process_batch_queries(self, queries: List[str]) -> List[Optional[QueryEmbeddingResult]]:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of user queries
            
        Returns:
            List of QueryEmbeddingResult objects
        """
        try:
            logger.info(f"📦 Processing {len(queries)} queries in batch")
            
            results = []
            for i, query in enumerate(queries):
                logger.debug(f"🔄 Processing query {i+1}/{len(queries)}")
                result = self.process_query(query)
                results.append(result)
            
            successful = sum(1 for r in results if r is not None)
            logger.info(f"✅ Batch processing completed: {successful}/{len(queries)} successful")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Batch query processing failed: {e}")
            return [None] * len(queries)
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Get query processor statistics."""
        try:
            return {
                'max_query_length': self.max_query_length,
                'enable_query_preprocessing': self.enable_query_preprocessing,
                'enable_caching': self.enable_caching,
                'is_initialized': self.is_initialized,
                'cache_size': len(self.cache) if self.cache else 0,
                'components': {
                    'embedder': self.embedder is not None,
                    'fde_transformer': self.fde_transformer is not None
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get query stats: {e}")
            return {'error': str(e)}
    
    def clear_cache(self):
        """Clear the query cache."""
        if self.cache is not None:
            self.cache.clear()
            logger.info("🗑️  Query cache cleared")
    
    def validate_query(self, query: str) -> Tuple[bool, str]:
        """
        Validate a query for processing.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not query or not query.strip():
                return False, "Empty query"
            
            if len(query) > 10000:  # Reasonable upper limit
                return False, "Query too long (>10000 characters)"
            
            # Check for valid characters
            if not any(c.isalnum() for c in query):
                return False, "Query contains no alphanumeric characters"
            
            return True, "Valid query"
            
        except Exception as e:
            return False, f"Query validation error: {str(e)}"

def get_v2_query_processor(max_query_length: int = 64,
                          enable_query_preprocessing: bool = True,
                          enable_caching: bool = True) -> V2QueryProcessor:
    """
    Get or create a v2 query processor instance.
    
    Args:
        max_query_length: Maximum query length in tokens
        enable_query_preprocessing: Enable query preprocessing
        enable_caching: Enable query result caching
        
    Returns:
        V2QueryProcessor instance
    """
    global _v2_query_processor
    
    if _v2_query_processor is None:
        _v2_query_processor = V2QueryProcessor(
            max_query_length=max_query_length,
            enable_query_preprocessing=enable_query_preprocessing,
            enable_caching=enable_caching
        )
    
    return _v2_query_processor

def process_query_v2(query: str) -> Optional[QueryEmbeddingResult]:
    """
    Convenience function to process a query using v2 pipeline.
    
    Args:
        query: User query
        
    Returns:
        QueryEmbeddingResult with processed embeddings
    """
    processor = get_v2_query_processor()
    return processor.process_query(query)
