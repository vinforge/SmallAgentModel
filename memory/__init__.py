"""
SAM Memory Module
Long-Term Memory, Vector Store, and Memory-Driven Reasoning

Sprint 11: Long-Term Memory, Vector Store, and Conditional Swarm Unlock
"""

# Import all memory system components
from .memory_vectorstore import MemoryVectorStore, VectorStoreType, MemoryType, MemoryChunk, MemorySearchResult, get_memory_store
from .memory_reasoning import MemoryDrivenReasoningEngine, MemoryContext, ReasoningSession, get_memory_reasoning_engine

# Import synthesis components (Phase 8A: Dream Catcher)
try:
    from .synthesis import (
        SynthesisEngine, SynthesisConfig, SynthesisResult,
        ClusteringService, ConceptCluster,
        SynthesisPromptGenerator, SynthesisPrompt,
        InsightGenerator, SynthesizedInsight,
        SyntheticChunkFormatter, format_synthesis_output
    )
    _synthesis_available = True
except ImportError:
    _synthesis_available = False

__all__ = [
    # Memory Vector Store
    'MemoryVectorStore',
    'VectorStoreType',
    'MemoryType',
    'MemoryChunk',
    'MemorySearchResult',
    'get_memory_store',

    # Memory-Driven Reasoning
    'MemoryDrivenReasoningEngine',
    'MemoryContext',
    'ReasoningSession',
    'get_memory_reasoning_engine'
]

# Add synthesis components if available
if _synthesis_available:
    __all__.extend([
        # Cognitive Synthesis Engine (Dream Catcher)
        'SynthesisEngine',
        'SynthesisConfig',
        'SynthesisResult',
        'ClusteringService',
        'ConceptCluster',
        'SynthesisPromptGenerator',
        'SynthesisPrompt',
        'InsightGenerator',
        'SynthesizedInsight',
        'SyntheticChunkFormatter',
        'format_synthesis_output'
    ])
