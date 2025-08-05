"""
RAGFlow Integration for SAM
Enhanced document understanding and retrieval capabilities.

This module provides integration with RAGFlow's advanced document understanding
and retrieval capabilities, including:

- Deep document understanding with DeepDoc
- Template-based intelligent chunking
- Multi-modal processing (text, images, tables)
- Hybrid retrieval with fused re-ranking
- Grounded citations with reduced hallucinations
- Visual chunking intervention
- Cross-language query support

Example usage:
    from sam.integrations.ragflow import RAGFlowBridge

    bridge = RAGFlowBridge()
    result = bridge.process_document("document.pdf")
"""

from .ragflow_bridge import RAGFlowBridge
from .ragflow_client import RAGFlowClient
from .document_processor import RAGFlowDocumentProcessor
from .hybrid_retrieval import HybridRetrievalEngine
from .knowledge_sync import KnowledgeBaseSynchronizer

__all__ = [
    'RAGFlowBridge',
    'RAGFlowClient',
    'RAGFlowDocumentProcessor',
    'HybridRetrievalEngine',
    'KnowledgeBaseSynchronizer'
]

__version__ = "1.0.0"
