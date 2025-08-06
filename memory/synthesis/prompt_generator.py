"""
Synthesis Prompt Generator for SAM's Cognitive Synthesis Engine

This module generates carefully crafted prompts for LLM-based synthesis of concept clusters
into emergent insights during SAM's "dream state" cognitive consolidation.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

from .clustering_service import ConceptCluster
from ..memory_vectorstore import MemoryChunk

logger = logging.getLogger(__name__)

@dataclass
class SynthesisPrompt:
    """Represents a synthesis prompt for a concept cluster."""
    cluster_id: str
    prompt_text: str
    source_chunks: List[MemoryChunk]
    synthesis_context: Dict[str, Any]
    quality_score: float

class SynthesisPromptGenerator:
    """
    Generates tailored prompts for synthesizing concept clusters into emergent insights.
    
    The prompt generator creates context-aware prompts that guide the LLM to:
    - Generate NEW understanding rather than just summarizing
    - Maintain factual accuracy while finding connections
    - Create concise but profound insights
    - Preserve traceability to source materials
    """
    
    def __init__(self, max_chunks_per_prompt: int = 8, max_content_length: int = 8000):
        """
        Initialize the prompt generator.
        
        Args:
            max_chunks_per_prompt: Maximum chunks to include in a single prompt
            max_content_length: Maximum total content length for prompt context
        """
        self.max_chunks_per_prompt = max_chunks_per_prompt
        self.max_content_length = max_content_length
        
        logger.info(f"SynthesisPromptGenerator initialized with max_chunks={max_chunks_per_prompt}")
    
    def generate_synthesis_prompt(self, cluster: ConceptCluster) -> SynthesisPrompt:
        """
        Generate a synthesis prompt for a concept cluster.
        
        Args:
            cluster: The concept cluster to generate a prompt for
            
        Returns:
            A synthesis prompt ready for LLM processing
        """
        try:
            logger.info(f"Generating synthesis prompt for {cluster.cluster_id}")

            # Safety check for empty cluster
            if not cluster.chunks or len(cluster.chunks) == 0:
                logger.warning(f"Cluster {cluster.cluster_id} has no chunks - cannot generate prompt")
                raise ValueError(f"Cluster {cluster.cluster_id} has no chunks")

            # Select best chunks for synthesis
            selected_chunks = self._select_synthesis_chunks(cluster)

            # Safety check for selected chunks
            if not selected_chunks or len(selected_chunks) == 0:
                logger.warning(f"No chunks selected for synthesis from cluster {cluster.cluster_id}")
                raise ValueError(f"No chunks selected for synthesis from cluster {cluster.cluster_id}")

            # Generate synthesis context
            synthesis_context = self._build_synthesis_context(cluster, selected_chunks)
            
            # Create the prompt text
            prompt_text = self._create_prompt_text(cluster, selected_chunks, synthesis_context)
            
            # Calculate prompt quality score
            quality_score = self._calculate_prompt_quality(cluster, selected_chunks)
            
            synthesis_prompt = SynthesisPrompt(
                cluster_id=cluster.cluster_id,
                prompt_text=prompt_text,
                source_chunks=selected_chunks,
                synthesis_context=synthesis_context,
                quality_score=quality_score
            )
            
            logger.info(f"✅ Generated synthesis prompt for {cluster.cluster_id} (quality: {quality_score:.2f})")
            return synthesis_prompt
            
        except Exception as e:
            logger.error(f"Error generating synthesis prompt for {cluster.cluster_id}: {e}")
            raise
    
    def _select_synthesis_chunks(self, cluster: ConceptCluster) -> List[MemoryChunk]:
        """Select the best chunks from a cluster for synthesis."""
        chunks = cluster.chunks.copy()
        
        # Sort by importance and recency
        def chunk_priority(chunk: MemoryChunk) -> float:
            # Parse timestamp for recency calculation
            try:
                chunk_time = datetime.fromisoformat(chunk.timestamp.replace('Z', '+00:00'))
                days_old = (datetime.now() - chunk_time.replace(tzinfo=None)).days
                recency_score = max(0, 1 - (days_old / 365))  # Decay over a year
            except:
                recency_score = 0.5  # Default if timestamp parsing fails
            
            return chunk.importance_score * 0.7 + recency_score * 0.3
        
        chunks.sort(key=chunk_priority, reverse=True)

        # Debug: Log chunk information
        logger.info(f"Cluster has {len(chunks)} chunks, max_chunks={self.max_chunks_per_prompt}, max_length={self.max_content_length}")
        for i, chunk in enumerate(chunks[:5]):  # Log first 5 chunks
            logger.info(f"Chunk {i}: length={len(chunk.content)}, importance={chunk.importance_score:.2f}")

        # Select top chunks within content length limit
        selected_chunks = []
        total_length = 0

        for i, chunk in enumerate(chunks):
            chunk_length = len(chunk.content)
            logger.debug(f"Evaluating chunk {i}: length={chunk_length}, total_so_far={total_length}")

            if (len(selected_chunks) < self.max_chunks_per_prompt and
                total_length + chunk_length <= self.max_content_length):
                selected_chunks.append(chunk)
                total_length += chunk_length
                logger.debug(f"Selected chunk {i}, new total: {total_length}")
            else:
                logger.debug(f"Skipped chunk {i}: would exceed limits (chunks: {len(selected_chunks)}/{self.max_chunks_per_prompt}, length: {total_length + chunk_length}/{self.max_content_length})")
                # Don't break - continue checking smaller chunks

        # Fallback: if no chunks selected due to length limits, select at least the top chunk (truncated if needed)
        if not selected_chunks and chunks:
            logger.warning(f"No chunks selected due to length limits - using fallback with truncated top chunk")
            top_chunk = chunks[0]
            # Truncate the content if it's too long
            if len(top_chunk.content) > self.max_content_length:
                # Create a truncated copy
                from copy import deepcopy
                truncated_chunk = deepcopy(top_chunk)
                truncated_chunk.content = top_chunk.content[:self.max_content_length-100] + "... [truncated]"
                selected_chunks = [truncated_chunk]
                logger.info(f"Using truncated top chunk (original: {len(top_chunk.content)} chars, truncated: {len(truncated_chunk.content)} chars)")
            else:
                selected_chunks = [top_chunk]

        logger.info(f"Selected {len(selected_chunks)} chunks from {len(chunks)} for synthesis (total length: {total_length})")
        return selected_chunks
    
    def _build_synthesis_context(self, cluster: ConceptCluster, chunks: List[MemoryChunk]) -> Dict[str, Any]:
        """Build context information for synthesis."""
        # Analyze chunk characteristics
        sources = list(set(chunk.source for chunk in chunks))
        memory_types = list(set(chunk.memory_type.value for chunk in chunks))
        all_tags = []
        for chunk in chunks:
            all_tags.extend(chunk.tags)
        
        # Calculate tag frequencies
        tag_counts = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        common_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'cluster_size': cluster.size,
            'selected_chunks': len(chunks),
            'coherence_score': cluster.coherence_score,
            'dominant_themes': cluster.dominant_themes,
            'source_count': len(sources),
            'source_names': sources[:3],  # Top 3 sources
            'memory_types': memory_types,
            'common_tags': [tag for tag, count in common_tags],
            'avg_importance': sum(chunk.importance_score for chunk in chunks) / len(chunks) if chunks else 0.0,
            'synthesis_timestamp': datetime.now().isoformat()
        }
    
    def _create_prompt_text(self, cluster: ConceptCluster, chunks: List[MemoryChunk], 
                           context: Dict[str, Any]) -> str:
        """Create the actual prompt text for synthesis."""
        
        # Build the main prompt
        prompt_parts = []
        
        # System context
        prompt_parts.append("""🧠 **SAM COGNITIVE SYNTHESIS MODE** 🧠

You are SAM performing cognitive synthesis during a "dream state" - analyzing clusters of related concepts from the user's documents to generate novel, emergent insights.

**SYNTHESIS OBJECTIVE:**
Generate ONE profound, emergent insight that reveals hidden patterns, unexpected connections, or strategic implications across these related concepts. This should be NEW understanding that emerges from the intersection of ideas, not just a summary.

**SYNTHESIS REQUIREMENTS:**
• Identify NOVEL patterns or connections that span multiple sources
• Reveal strategic implications, future trends, or actionable opportunities
• Connect seemingly unrelated concepts to generate breakthrough insights
• Focus on what the USER should know that isn't obvious from individual sources
• Be specific and actionable (3-5 sentences with concrete implications)
• Highlight contradictions, gaps, or emerging themes across the literature""")
        
        # Cluster context
        prompt_parts.append(f"""
**CLUSTER ANALYSIS:**
• Cluster ID: {cluster.cluster_id}
• Coherence Score: {cluster.coherence_score:.2f}/1.0
• Dominant Themes: {', '.join(cluster.dominant_themes[:3])}
• Source Diversity: {context['source_count']} different sources
• Memory Types: {', '.join(context['memory_types'])}""")
        
        # Source chunks
        prompt_parts.append("\n**SOURCE CONCEPTS:**")
        
        for i, chunk in enumerate(chunks, 1):
            # Truncate content if too long
            content = chunk.content
            if len(content) > 300:
                content = content[:297] + "..."
            
            prompt_parts.append(f"""
**Source {i}** (from: {chunk.source}, type: {chunk.memory_type.value})
{content}""")
        
        # Synthesis instruction
        prompt_parts.append(f"""
**SYNTHESIS TASK:**
Analyze the {len(chunks)} source concepts above to identify:
1. Hidden patterns or connections that span multiple sources
2. Strategic implications or future opportunities
3. Novel insights that emerge from combining these perspectives
4. Actionable recommendations based on the synthesized understanding

Generate a compelling insight that reveals something valuable and non-obvious about the user's domain of interest.

**EMERGENT INSIGHT:**""")
        
        return "\n".join(prompt_parts)
    
    def _calculate_prompt_quality(self, cluster: ConceptCluster, chunks: List[MemoryChunk]) -> float:
        """Calculate the quality score for a synthesis prompt."""
        # Factors that contribute to prompt quality
        coherence_factor = cluster.coherence_score
        
        # Diversity factor (more diverse sources = better synthesis potential)
        sources = set(chunk.source for chunk in chunks)
        diversity_factor = min(len(sources) / 5.0, 1.0)  # Normalize to max 5 sources
        
        # Content richness factor
        if chunks:
            avg_content_length = sum(len(chunk.content) for chunk in chunks) / len(chunks)
            richness_factor = min(avg_content_length / 500.0, 1.0)  # Normalize to 500 chars

            # Importance factor
            avg_importance = sum(chunk.importance_score for chunk in chunks) / len(chunks)
        else:
            richness_factor = 0.0
            avg_importance = 0.0
        
        # Theme clarity factor
        theme_factor = min(len(cluster.dominant_themes) / 3.0, 1.0) if cluster.dominant_themes else 0.0
        
        # Weighted combination
        quality_score = (
            coherence_factor * 0.3 +
            diversity_factor * 0.25 +
            richness_factor * 0.2 +
            avg_importance * 0.15 +
            theme_factor * 0.1
        )
        
        return min(quality_score, 1.0)
