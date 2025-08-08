# utils/__init__.py

"""
Utilities module for SAM (Small Agent Model).

This module provides core utilities for vector storage, embeddings, retrieval logging,
and CLI administration tools for Sprint 11.
"""

from .vector_manager import VectorManager
from .embedding_utils import EmbeddingManager, get_embedding_manager, embed, embed_batch, embed_query
from .retrieval_logger import RetrievalLogger, get_retrieval_logger
from .cli_utils import SAMAdminCLI

__all__ = [
    'VectorManager',
    'EmbeddingManager',
    'get_embedding_manager',
    'embed',
    'embed_batch',
    'embed_query',
    'RetrievalLogger',
    'get_retrieval_logger',
    'SAMAdminCLI'
]
