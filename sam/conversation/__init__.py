"""
SAM Conversation Module - Task 31 Phase 1
==========================================

Provides intelligent conversation threading and contextual relevance
capabilities for automatic conversation management.

Part of Task 31: Conversational Intelligence Engine
"""

from .contextual_relevance import (
    ContextualRelevanceEngine,
    RelevanceResult,
    ConversationThread,
    get_contextual_relevance_engine
)

__all__ = [
    'ContextualRelevanceEngine',
    'RelevanceResult',
    'ConversationThread',
    'get_contextual_relevance_engine'
]
