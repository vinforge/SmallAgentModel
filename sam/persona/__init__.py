"""
SAM Persona Management Module - Task 30 Phase 2
===============================================

Provides persona memory retrieval and response refinement functionality
for maintaining long-term consistency and personalization across sessions.

Part of Task 30: Advanced Conversational Coherence Engine
"""

from .persona_memory import (
    PersonaMemoryRetriever,
    PersonaMemory,
    get_persona_memory_retriever
)

from .persona_refinement import (
    PersonaRefinementEngine,
    generate_final_response
)

__all__ = [
    'PersonaMemoryRetriever',
    'PersonaMemory',
    'get_persona_memory_retriever',
    'PersonaRefinementEngine',
    'generate_final_response'
]
