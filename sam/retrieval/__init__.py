#!/usr/bin/env python3
"""
SAM v2 Retrieval Module
Advanced two-stage retrieval system with multi-vector embeddings and FDE optimization.
"""

from .v2_retrieval_engine import (
    V2RetrievalEngine,
    V2RetrievalConfig,
    V2RetrievalResult,
    get_v2_retrieval_engine,
    retrieve_v2_documents
)

from .query_processor import (
    V2QueryProcessor,
    QueryEmbeddingResult,
    get_v2_query_processor,
    process_query_v2
)

from .reranking_engine import (
    V2RerankingEngine,
    RerankingResult,
    RerankingConfig,
    get_v2_reranking_engine,
    rerank_documents_v2
)

from .context_assembler import (
    V2ContextAssembler,
    V2AssembledContext,
    ContextConfig,
    get_v2_context_assembler,
    assemble_context_v2
)

__all__ = [
    # Main Retrieval Engine
    'V2RetrievalEngine',
    'V2RetrievalConfig',
    'V2RetrievalResult',
    'get_v2_retrieval_engine',
    'retrieve_v2_documents',
    
    # Query Processing
    'V2QueryProcessor',
    'QueryEmbeddingResult',
    'get_v2_query_processor',
    'process_query_v2',
    
    # Reranking
    'V2RerankingEngine',
    'RerankingResult',
    'RerankingConfig',
    'get_v2_reranking_engine',
    'rerank_documents_v2',
    
    # Context Assembly
    'V2ContextAssembler',
    'V2AssembledContext',
    'ContextConfig',
    'get_v2_context_assembler',
    'assemble_context_v2'
]
