"""
SAM Cognitive Synthesis Engine ("Dream Catcher")

This module implements SAM's revolutionary cognitive synthesis system that performs
offline analysis of the memory store to generate emergent insights from concept clusters.

The Dream Catcher mimics human cognitive consolidation during rest periods, identifying
dense clusters of related concepts and synthesizing them into new, high-level insights.

Components:
- ClusteringService: DBSCAN-based clustering of memory vectors
- SynthesisPromptGenerator: Tailored prompt generation for concept clusters  
- InsightGenerator: LLM-based synthesis of emergent insights
- SynthesisEngine: Main orchestrator for the synthesis process
"""

from .clustering_service import ClusteringService, ConceptCluster
from .prompt_generator import SynthesisPromptGenerator, SynthesisPrompt
from .insight_generator import InsightGenerator, SynthesizedInsight
from .chunk_formatter import SyntheticChunkFormatter, format_synthesis_output
from .synthesis_engine import SynthesisEngine, SynthesisResult, SynthesisConfig

__all__ = [
    # Core Services
    'ClusteringService',
    'SynthesisPromptGenerator',
    'InsightGenerator',
    'SyntheticChunkFormatter',
    'SynthesisEngine',

    # Data Classes
    'ConceptCluster',
    'SynthesisPrompt',
    'SynthesizedInsight',
    'SynthesisResult',
    'SynthesisConfig',

    # Utility Functions
    'format_synthesis_output'
]
