#!/usr/bin/env python3
"""
V2 RAG Pipeline for SAM MUVERA System
Complete RAG pipeline using two-stage retrieval and multi-vector embeddings.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Global v2 RAG pipeline instance
_v2_rag_pipeline = None

@dataclass
class V2RAGConfig:
    """Configuration for v2 RAG pipeline."""
    stage1_top_k: int = 50              # Top-K for FDE search
    stage2_top_k: int = 10              # Top-K for final results
    similarity_metric: str = "maxsim"   # Similarity metric for reranking
    max_context_length: int = 4000      # Maximum context length
    include_metadata: bool = True       # Include document metadata
    include_scores: bool = True         # Include similarity scores
    enable_caching: bool = True         # Enable result caching
    fallback_to_v1: bool = True         # Fallback to v1 if v2 fails

@dataclass
class V2RAGResult:
    """Result from v2 RAG pipeline."""
    query: str                          # Original query
    success: bool                       # Whether processing succeeded
    formatted_context: Optional[str]   # Formatted context for LLM
    source_documents: List[str]         # Source document IDs
    similarity_scores: Dict[str, float] # Document similarity scores
    retrieval_time: float               # Time for retrieval
    context_assembly_time: float       # Time for context assembly
    total_time: float                   # Total processing time
    document_count: int                 # Number of documents retrieved
    context_length: int                 # Length of formatted context
    pipeline_version: str               # Pipeline version used
    error_message: Optional[str]        # Error message if failed
    metadata: Dict[str, Any]            # Additional metadata

class V2RAGPipeline:
    """
    Complete v2 RAG pipeline integrating two-stage retrieval with context assembly.
    
    Provides a unified interface for document-aware question answering using
    the MUVERA approach with ColBERTv2 embeddings and FDE optimization.
    """
    
    def __init__(self, config: Optional[V2RAGConfig] = None):
        """
        Initialize the v2 RAG pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or V2RAGConfig()
        
        # Components (loaded lazily)
        self.retrieval_engine = None
        self.context_assembler = None
        self.is_initialized = False
        
        # Caching
        self.cache = {} if self.config.enable_caching else None
        
        logger.info(f"🚀 V2RAGPipeline initialized")
        logger.info(f"📊 Config: {self.config.stage1_top_k}→{self.config.stage2_top_k}, {self.config.similarity_metric}")
    
    def _initialize_components(self) -> bool:
        """Initialize RAG pipeline components."""
        if self.is_initialized:
            return True
        
        try:
            logger.info("🔄 Initializing v2 RAG pipeline components...")
            
            # Initialize retrieval engine
            from sam.retrieval import get_v2_retrieval_engine, V2RetrievalConfig
            retrieval_config = V2RetrievalConfig(
                stage1_top_k=self.config.stage1_top_k,
                stage2_top_k=self.config.stage2_top_k,
                similarity_metric=self.config.similarity_metric,
                enable_caching=self.config.enable_caching
            )
            self.retrieval_engine = get_v2_retrieval_engine(retrieval_config)
            
            # Initialize context assembler
            from sam.retrieval import get_v2_context_assembler, ContextConfig
            context_config = ContextConfig(
                max_context_length=self.config.max_context_length,
                include_metadata=self.config.include_metadata,
                include_scores=self.config.include_scores
            )
            self.context_assembler = get_v2_context_assembler(context_config)
            
            self.is_initialized = True
            logger.info("✅ V2 RAG pipeline components initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize v2 RAG pipeline components: {e}")
            return False
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query."""
        import hashlib
        config_str = f"{self.config.stage1_top_k}_{self.config.stage2_top_k}_{self.config.similarity_metric}"
        return hashlib.md5(f"{query}_{config_str}".encode()).hexdigest()
    
    def process_query(self, query: str) -> V2RAGResult:
        """
        Process a query using the v2 RAG pipeline.
        
        Args:
            query: User query
            
        Returns:
            V2RAGResult with processing results
        """
        try:
            total_start_time = time.time()
            
            logger.info(f"🔍 V2 RAG processing: '{query[:50]}...'")
            
            # Check cache
            if self.config.enable_caching and self.cache is not None:
                cache_key = self._get_cache_key(query)
                if cache_key in self.cache:
                    logger.debug("📋 Returning cached RAG result")
                    return self.cache[cache_key]
            
            # Initialize components
            if not self._initialize_components():
                return self._create_error_result(query, "Failed to initialize v2 RAG components")
            
            # Stage 1 & 2: Retrieve documents
            retrieval_start_time = time.time()
            retrieval_result = self.retrieval_engine.retrieve(query)
            retrieval_time = time.time() - retrieval_start_time
            
            if not retrieval_result.final_documents:
                logger.warning("⚠️  No documents retrieved")
                return self._create_empty_result(query, retrieval_time, 0.0)
            
            logger.info(f"📄 Retrieved {len(retrieval_result.final_documents)} documents in {retrieval_time:.3f}s")
            
            # Stage 3: Assemble context
            context_start_time = time.time()
            
            # Extract similarity scores for final documents
            similarity_scores = {}
            for doc_id, score in retrieval_result.stage2_results:
                if doc_id in retrieval_result.final_documents:
                    similarity_scores[doc_id] = score
            
            context_result = self.context_assembler.assemble_context(
                query=query,
                document_ids=retrieval_result.final_documents,
                similarity_scores=similarity_scores
            )
            
            context_assembly_time = time.time() - context_start_time
            
            if not context_result:
                logger.error("❌ Failed to assemble context")
                return self._create_error_result(query, "Failed to assemble document context")
            
            logger.info(f"📝 Assembled context: {context_result.total_length} chars in {context_assembly_time:.3f}s")
            
            # Create result
            total_time = time.time() - total_start_time
            
            result = V2RAGResult(
                query=query,
                success=True,
                formatted_context=context_result.formatted_context,
                source_documents=context_result.source_documents,
                similarity_scores=context_result.similarity_scores,
                retrieval_time=retrieval_time,
                context_assembly_time=context_assembly_time,
                total_time=total_time,
                document_count=context_result.document_count,
                context_length=context_result.total_length,
                pipeline_version="v2_muvera",
                error_message=None,
                metadata={
                    'config': self.config.__dict__,
                    'retrieval_metadata': retrieval_result.metadata,
                    'context_metadata': context_result.metadata,
                    'stage1_candidates': retrieval_result.num_candidates,
                    'stage2_reranked': retrieval_result.num_reranked,
                    'context_truncated': context_result.truncated,
                    'processing_timestamp': time.time()
                }
            )
            
            # Cache result
            if self.config.enable_caching and self.cache is not None:
                cache_key = self._get_cache_key(query)
                self.cache[cache_key] = result
            
            logger.info(f"✅ V2 RAG completed: {result.document_count} docs, {total_time:.3f}s total")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ V2 RAG processing failed: {e}")
            return self._create_error_result(query, str(e))
    
    def _create_error_result(self, query: str, error_message: str) -> V2RAGResult:
        """Create error result for failed processing."""
        return V2RAGResult(
            query=query,
            success=False,
            formatted_context=None,
            source_documents=[],
            similarity_scores={},
            retrieval_time=0.0,
            context_assembly_time=0.0,
            total_time=0.0,
            document_count=0,
            context_length=0,
            pipeline_version="v2_muvera",
            error_message=error_message,
            metadata={'error': error_message}
        )
    
    def _create_empty_result(self, query: str, retrieval_time: float, context_time: float) -> V2RAGResult:
        """Create empty result when no documents found."""
        return V2RAGResult(
            query=query,
            success=True,
            formatted_context=None,
            source_documents=[],
            similarity_scores={},
            retrieval_time=retrieval_time,
            context_assembly_time=context_time,
            total_time=retrieval_time + context_time,
            document_count=0,
            context_length=0,
            pipeline_version="v2_muvera",
            error_message=None,
            metadata={'no_documents_found': True}
        )
    
    def get_context_for_llm(self, result: V2RAGResult) -> Optional[str]:
        """
        Extract formatted context for LLM prompt injection.
        
        Args:
            result: Result from process_query
            
        Returns:
            Formatted context string for LLM or None
        """
        if not result.success or not result.formatted_context:
            return None
        
        return result.formatted_context
    
    def get_source_attribution(self, result: V2RAGResult) -> str:
        """
        Get source attribution text for user display.
        
        Args:
            result: Result from process_query
            
        Returns:
            Source attribution text
        """
        if not result.success:
            return f"Error: {result.error_message}"
        
        if result.document_count == 0:
            return "No relevant documents found in uploaded files."
        
        return f"Information retrieved from {result.document_count} uploaded documents using v2 MUVERA pipeline."
    
    def get_detailed_citations(self, result: V2RAGResult) -> List[str]:
        """
        Get detailed citation list for user display.
        
        Args:
            result: Result from process_query
            
        Returns:
            List of detailed citations
        """
        citations = []
        
        if not result.success or not result.source_documents:
            return citations
        
        # Get document records for citations
        try:
            from sam.storage import get_v2_storage_manager
            storage_manager = get_v2_storage_manager()
            
            for i, doc_id in enumerate(result.source_documents, 1):
                record = storage_manager.retrieve_document(doc_id)
                if record:
                    score = result.similarity_scores.get(doc_id, 0.0)
                    citation = f"[{i}] {record.filename}"
                    if score > 0:
                        citation += f" (relevance: {score:.3f})"
                    citations.append(citation)
                else:
                    citations.append(f"[{i}] Document {doc_id} (details unavailable)")
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to generate detailed citations: {e}")
            citations = [f"Document {i}: {doc_id}" for i, doc_id in enumerate(result.source_documents, 1)]
        
        return citations
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get v2 RAG pipeline statistics."""
        try:
            stats = {
                'config': self.config.__dict__,
                'is_initialized': self.is_initialized,
                'cache_size': len(self.cache) if self.cache else 0,
                'components': {
                    'retrieval_engine': self.retrieval_engine is not None,
                    'context_assembler': self.context_assembler is not None
                }
            }
            
            # Add component stats if available
            if self.retrieval_engine:
                stats['retrieval_stats'] = self.retrieval_engine.get_retrieval_stats()
            
            if self.context_assembler:
                stats['context_stats'] = self.context_assembler.get_context_stats()
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get v2 RAG pipeline stats: {e}")
            return {'error': str(e)}
    
    def clear_cache(self):
        """Clear the RAG pipeline cache."""
        if self.cache is not None:
            self.cache.clear()
            logger.info("🗑️  V2 RAG pipeline cache cleared")
        
        # Clear component caches
        if self.retrieval_engine:
            self.retrieval_engine.clear_cache()
        
        if self.context_assembler:
            # Context assembler doesn't have cache, but we could add it

def get_v2_rag_pipeline(config: Optional[V2RAGConfig] = None) -> V2RAGPipeline:
    """
    Get or create a v2 RAG pipeline instance.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        V2RAGPipeline instance
    """
    global _v2_rag_pipeline
    
    if _v2_rag_pipeline is None:
        _v2_rag_pipeline = V2RAGPipeline(config)
    
    return _v2_rag_pipeline

def query_v2_rag(query: str,
                stage1_top_k: int = 50,
                stage2_top_k: int = 10,
                similarity_metric: str = "maxsim") -> V2RAGResult:
    """
    Convenience function to query using v2 RAG pipeline.
    
    Args:
        query: User query
        stage1_top_k: Top-K for FDE search
        stage2_top_k: Top-K for final results
        similarity_metric: Similarity metric for reranking
        
    Returns:
        V2RAGResult with processing results
    """
    config = V2RAGConfig(
        stage1_top_k=stage1_top_k,
        stage2_top_k=stage2_top_k,
        similarity_metric=similarity_metric
    )
    
    pipeline = get_v2_rag_pipeline(config)
    return pipeline.process_query(query)
