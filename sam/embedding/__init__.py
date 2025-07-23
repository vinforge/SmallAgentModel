#!/usr/bin/env python3
"""
SAM Multi-Vector Embedding Module
Advanced embedding capabilities for SAM's v2 retrieval pipeline.
"""

from .multivector_embedder import (
    MultiVectorEmbedder,
    get_multivector_embedder,
    EmbeddingResult
)

from .embedding_utils import (
    EmbeddingConfig,
    validate_embeddings,
    normalize_embeddings,
    compute_embedding_stats
)

__all__ = [
    'MultiVectorEmbedder',
    'get_multivector_embedder', 
    'EmbeddingResult',
    'EmbeddingConfig',
    'validate_embeddings',
    'normalize_embeddings',
    'compute_embedding_stats'
]
