#!/usr/bin/env python3
"""
Document-Aware RAG Pipeline
Complete integration module for SAM's document-aware retrieval-augmented generation.
"""

import logging
from typing import Dict, Any, Optional, List

from .semantic_document_search import SemanticDocumentSearchEngine, DocumentSearchResult
from .document_context_assembler import DocumentContextAssembler, AssembledContext
from .document_aware_query_router import DocumentAwareQueryRouter, RoutingDecision, QueryStrategy

logger = logging.getLogger(__name__)

class DocumentAwareRAGPipeline:
    """
    Complete Document-Aware RAG Pipeline that integrates with SAM's existing infrastructure.
    This is the main interface for document-aware query processing.
    """
    
    def __init__(self, 
                 memory_store=None,
                 encrypted_store=None,
                 max_context_length: int = 4000,
                 high_confidence_threshold: float = 0.85,
                 medium_confidence_threshold: float = 0.65,
                 low_confidence_threshold: float = 0.45):
        """
        Initialize the Document-Aware RAG Pipeline.
        
        Args:
            memory_store: SAM's regular memory store
            encrypted_store: SAM's encrypted memory store
            max_context_length: Maximum context length for LLM
            high_confidence_threshold: High confidence threshold
            medium_confidence_threshold: Medium confidence threshold
            low_confidence_threshold: Low confidence threshold
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self.search_engine = SemanticDocumentSearchEngine(
            memory_store=memory_store,
            encrypted_store=encrypted_store
        )
        
        self.context_assembler = DocumentContextAssembler(
            max_context_length=max_context_length
        )
        
        self.query_router = DocumentAwareQueryRouter(
            search_engine=self.search_engine,
            context_assembler=self.context_assembler,
            high_confidence_threshold=high_confidence_threshold,
            medium_confidence_threshold=medium_confidence_threshold,
            low_confidence_threshold=low_confidence_threshold
        )
        
        self.logger.info("🚀 Document-Aware RAG Pipeline initialized")
    
    def process_query(self, 
                     query: str,
                     max_document_results: int = 5,
                     user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a query using the Document-Aware RAG Pipeline.
        
        Args:
            query: User's query
            max_document_results: Maximum document chunks to retrieve
            user_context: Additional user context
            
        Returns:
            Dictionary with routing decision, document context, and metadata
        """
        try:
            self.logger.info(f"🔄 Processing query with Document-Aware RAG: '{query[:50]}...'")
            
            # Route the query using document-first strategy
            routing_decision = self.query_router.route_query(
                query=query,
                max_document_results=max_document_results,
                user_context=user_context
            )
            
            # Prepare response
            response = {
                'success': True,
                'query': query,
                'routing_decision': {
                    'strategy': routing_decision.strategy.value,
                    'confidence_level': routing_decision.confidence_level,
                    'reasoning': routing_decision.reasoning,
                    'should_search_web': routing_decision.should_search_web,
                    'fallback_strategy': routing_decision.fallback_strategy.value if routing_decision.fallback_strategy else None
                },
                'document_context': None,
                'use_document_context': False,
                'metadata': routing_decision.routing_metadata
            }
            
            # Add document context if available
            if routing_decision.document_context:
                response['document_context'] = {
                    'formatted_context': routing_decision.document_context.formatted_context,
                    'source_count': routing_decision.document_context.source_count,
                    'document_count': routing_decision.document_context.document_count,
                    'confidence_level': routing_decision.document_context.confidence_level,
                    'citation_map': routing_decision.document_context.citation_map,
                    'context_metadata': routing_decision.document_context.context_metadata
                }
                response['use_document_context'] = self.query_router.should_use_document_context(routing_decision)
            
            self.logger.info(f"✅ Query processed: {routing_decision.strategy.value} strategy, {routing_decision.confidence_level} confidence")
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Document-Aware RAG processing failed: {e}")
            return {
                'success': False,
                'query': query,
                'error': str(e),
                'routing_decision': {'strategy': 'general_knowledge', 'confidence_level': 'ERROR'},
                'document_context': None,
                'use_document_context': False,
                'metadata': {'error': str(e)}
            }
    
    def get_document_context_for_llm(self, query_result: Dict[str, Any]) -> Optional[str]:
        """
        Extract formatted document context for LLM prompt injection.
        
        Args:
            query_result: Result from process_query
            
        Returns:
            Formatted context string for LLM or None
        """
        if not query_result.get('use_document_context'):
            return None
        
        document_context = query_result.get('document_context')
        if not document_context:
            return None
        
        return document_context.get('formatted_context')
    
    def get_source_attribution(self, query_result: Dict[str, Any]) -> str:
        """
        Get source attribution text for user display.
        
        Args:
            query_result: Result from process_query
            
        Returns:
            Source attribution text
        """
        if not query_result.get('use_document_context'):
            return "Information retrieved using general knowledge."
        
        document_context = query_result.get('document_context')
        if not document_context:
            return "No relevant documents found in uploaded files."
        
        source_count = document_context.get('source_count', 0)
        document_count = document_context.get('document_count', 0)
        
        if source_count == 0:
            return "No relevant documents found in uploaded files."
        
        return f"Information retrieved from {source_count} sources across {document_count} uploaded documents."
    
    def get_detailed_citations(self, query_result: Dict[str, Any]) -> List[str]:
        """
        Get detailed citation list for user display.
        
        Args:
            query_result: Result from process_query
            
        Returns:
            List of detailed citations
        """
        citations = []
        
        if not query_result.get('use_document_context'):
            return citations
        
        document_context = query_result.get('document_context')
        if not document_context:
            return citations
        
        citation_map = document_context.get('citation_map', {})
        for ref, citation in citation_map.items():
            citations.append(f"{ref}: {citation}")
        
        return citations
    
    def should_fallback_to_web_search(self, query_result: Dict[str, Any]) -> bool:
        """
        Check if query should fallback to web search.
        
        Args:
            query_result: Result from process_query
            
        Returns:
            True if should search web
        """
        routing_decision = query_result.get('routing_decision', {})
        return routing_decision.get('should_search_web', False)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get status information about the pipeline."""
        return {
            'pipeline_active': True,
            'components': {
                'search_engine': self.search_engine is not None,
                'context_assembler': self.context_assembler is not None,
                'query_router': self.query_router is not None
            },
            'thresholds': {
                'high_confidence': self.query_router.high_confidence_threshold,
                'medium_confidence': self.query_router.medium_confidence_threshold,
                'low_confidence': self.query_router.low_confidence_threshold
            },
            'max_context_length': self.context_assembler.max_context_length
        }

# Convenience function for easy integration
def create_document_rag_pipeline(memory_store=None, encrypted_store=None, **kwargs) -> DocumentAwareRAGPipeline:
    """
    Create a Document-Aware RAG Pipeline with default settings.
    
    Args:
        memory_store: SAM's regular memory store
        encrypted_store: SAM's encrypted memory store
        **kwargs: Additional configuration options
        
    Returns:
        Configured DocumentAwareRAGPipeline
    """
    return DocumentAwareRAGPipeline(
        memory_store=memory_store,
        encrypted_store=encrypted_store,
        **kwargs
    )

# v2 RAG Components (new)
try:
    from .v2_rag_pipeline import (
        V2RAGPipeline,
        V2RAGConfig,
        V2RAGResult,
        get_v2_rag_pipeline,
        query_v2_rag
    )

    from .rag_pipeline_router import (
        RAGPipelineRouter,
        PipelineSelection,
        get_rag_pipeline_router,
        route_rag_query
    )

    V2_COMPONENTS_AVAILABLE = True
except ImportError:
    V2_COMPONENTS_AVAILABLE = False

# Export main classes for direct import
__all__ = [
    # v1 RAG Components
    'DocumentAwareRAGPipeline',
    'SemanticDocumentSearchEngine',
    'DocumentContextAssembler',
    'DocumentAwareQueryRouter',
    'QueryStrategy',
    'create_document_rag_pipeline'
]

# Add v2 components if available
if V2_COMPONENTS_AVAILABLE:
    __all__.extend([
        'V2RAGPipeline',
        'V2RAGConfig',
        'V2RAGResult',
        'get_v2_rag_pipeline',
        'query_v2_rag',
        'RAGPipelineRouter',
        'PipelineSelection',
        'get_rag_pipeline_router',
        'route_rag_query'
    ])
