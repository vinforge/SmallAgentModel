"""
Insight Generator for SAM's Cognitive Synthesis Engine

This module handles LLM-based generation of synthesized insights from concept clusters
during SAM's cognitive consolidation process.
"""

import logging

from sam.core.sam_model_client import create_ollama_compatible_client
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime

from .prompt_generator import SynthesisPrompt
from ..memory_vectorstore import MemoryChunk

logger = logging.getLogger(__name__)

@dataclass
class SynthesizedInsight:
    """Represents a synthesized insight generated from a concept cluster."""
    insight_id: str
    cluster_id: str
    synthesized_text: str
    source_chunk_ids: List[str]
    source_chunks: List[MemoryChunk]
    confidence_score: float
    novelty_score: float
    utility_score: float
    synthesis_metadata: Dict[str, Any]
    generated_at: str

class InsightGenerator:
    """
    Generates synthesized insights using LLM processing of concept clusters.
    
    This component takes synthesis prompts and uses SAM's core LLM to generate
    emergent insights that represent new understanding derived from memory clusters.
    """
    
    def __init__(self, llm_client=None, temperature: float = 0.7, max_tokens: int = 200):
        """
        Initialize the insight generator.
        
        Args:
            llm_client: LLM client for generating insights (will use SAM's default if None)
            temperature: Temperature for LLM generation (higher = more creative)
            max_tokens: Maximum tokens for generated insights
        """
        self.llm_client = llm_client
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        logger.info(f"InsightGenerator initialized with temperature={temperature}")
    
    def generate_insight(self, synthesis_prompt: SynthesisPrompt) -> Optional[SynthesizedInsight]:
        """
        Generate a synthesized insight from a synthesis prompt.
        
        Args:
            synthesis_prompt: The synthesis prompt to process
            
        Returns:
            A synthesized insight or None if generation fails
        """
        try:
            logger.info(f"Generating insight for cluster {synthesis_prompt.cluster_id}")
            
            # Generate insight text using LLM
            insight_text = self._call_llm_for_synthesis(synthesis_prompt.prompt_text)
            
            if not insight_text or len(insight_text.strip()) < 10:
                logger.warning(f"Generated insight too short or empty for {synthesis_prompt.cluster_id}")
                return None
            
            # Clean and validate the insight
            cleaned_insight = self._clean_insight_text(insight_text)
            
            # Calculate insight quality scores
            confidence_score = self._calculate_confidence_score(synthesis_prompt, cleaned_insight)
            novelty_score = self._calculate_novelty_score(synthesis_prompt, cleaned_insight)
            utility_score = self._calculate_utility_score(synthesis_prompt, cleaned_insight)
            
            # Create synthesis metadata
            synthesis_metadata = self._create_synthesis_metadata(synthesis_prompt, cleaned_insight)
            
            # Generate unique insight ID
            insight_id = f"insight_{synthesis_prompt.cluster_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            synthesized_insight = SynthesizedInsight(
                insight_id=insight_id,
                cluster_id=synthesis_prompt.cluster_id,
                synthesized_text=cleaned_insight,
                source_chunk_ids=[chunk.chunk_id for chunk in synthesis_prompt.source_chunks],
                source_chunks=synthesis_prompt.source_chunks,
                confidence_score=confidence_score,
                novelty_score=novelty_score,
                utility_score=utility_score,
                synthesis_metadata=synthesis_metadata,
                generated_at=datetime.now().isoformat()
            )
            
            logger.info(f"✅ Generated insight {insight_id} (confidence: {confidence_score:.2f})")
            return synthesized_insight
            
        except Exception as e:
            logger.error(f"Error generating insight for {synthesis_prompt.cluster_id}: {e}")
            return None
    
    def _call_llm_for_synthesis(self, prompt_text: str) -> Optional[str]:
        """Call the LLM to generate synthesis text."""
        try:
            # If no LLM client provided, try to get SAM's default
            if self.llm_client is None:
                self.llm_client = self._get_sam_llm_client()
            
            if self.llm_client is None:
                logger.error("No LLM client available for synthesis")
                return None
            
            # Call LLM with synthesis prompt
            logger.info(f"Calling LLM for synthesis (prompt length: {len(prompt_text)} chars)")
            response = self.llm_client.generate(
                prompt=prompt_text,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop_sequences=["**", "---", "\n\n\n"]  # Stop at formatting markers
            )
            logger.info(f"LLM response received: {type(response)} (length: {len(str(response)) if response else 0})")

            # Handle different response types
            if isinstance(response, str):
                return response.strip()
            elif hasattr(response, 'text'):
                return response.text.strip()
            elif response is None:
                logger.error("LLM returned None response")
                return None
            else:
                logger.error(f"Unexpected LLM response type: {type(response)}")
                return None
                
        except Exception as e:
            logger.error(f"Error calling LLM for synthesis: {e}")
            return None
    
    def _get_sam_llm_client(self):
        """Get SAM's default LLM client."""
        try:
            # Create Ollama client using SAM's configuration
            return self._create_ollama_client()
        except Exception as e:
            logger.warning(f"Could not create SAM's LLM client: {e}")
            return None

    def _create_ollama_client(self):
        """Create SAM-compatible client."""
        return create_ollama_compatible_client()
    
    def _clean_insight_text(self, raw_text: str) -> str:
        """Clean and format the generated insight text."""
        # Remove common LLM artifacts
        cleaned = raw_text.strip()
        
        # Remove markdown formatting
        cleaned = cleaned.replace("**", "").replace("*", "")
        
        # Remove common prefixes
        prefixes_to_remove = [
            "EMERGENT INSIGHT:",
            "Insight:",
            "Synthesis:",
            "The insight is:",
            "Based on the analysis:"
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        # Ensure proper sentence structure
        if cleaned and not cleaned.endswith(('.', '!', '?')):
            cleaned += '.'
        
        return cleaned
    
    def _calculate_confidence_score(self, prompt: SynthesisPrompt, insight: str) -> float:
        """Calculate confidence score for the generated insight."""
        # Base confidence from prompt quality
        base_confidence = prompt.quality_score
        
        # Length factor (too short or too long reduces confidence)
        length_factor = 1.0
        if len(insight) < 50:
            length_factor = 0.6  # Too short
        elif len(insight) > 500:
            length_factor = 0.8  # Too long
        
        # Coherence factor (simple heuristics)
        coherence_factor = 1.0
        if insight.count('.') == 0:  # No sentences
            coherence_factor = 0.7
        elif insight.count('.') > 5:  # Too many sentences
            coherence_factor = 0.8
        
        # Specificity factor (avoid generic statements)
        generic_phrases = ['it is important', 'this shows', 'we can see', 'it appears']
        specificity_factor = 1.0
        for phrase in generic_phrases:
            if phrase in insight.lower():
                specificity_factor *= 0.9
        
        confidence = base_confidence * length_factor * coherence_factor * specificity_factor
        return min(confidence, 1.0)
    
    def _calculate_novelty_score(self, prompt: SynthesisPrompt, insight: str) -> float:
        """Calculate novelty score for the generated insight."""
        # Check if insight contains novel connections
        connection_words = ['however', 'therefore', 'consequently', 'implies', 'suggests', 
                           'reveals', 'indicates', 'demonstrates', 'connects', 'bridges']
        
        novelty_score = 0.5  # Base novelty
        
        # Boost for connection words
        for word in connection_words:
            if word in insight.lower():
                novelty_score += 0.1
        
        # Boost for cross-domain synthesis (different memory types)
        memory_types = set(chunk.memory_type.value for chunk in prompt.source_chunks)
        if len(memory_types) > 1:
            novelty_score += 0.2
        
        # Boost for source diversity
        sources = set(chunk.source for chunk in prompt.source_chunks)
        if len(sources) > 2:
            novelty_score += 0.1
        
        return min(novelty_score, 1.0)
    
    def _calculate_utility_score(self, prompt: SynthesisPrompt, insight: str) -> float:
        """Calculate utility score for the generated insight."""
        # Base utility from source importance
        avg_importance = sum(chunk.importance_score for chunk in prompt.source_chunks) / len(prompt.source_chunks)
        
        # Boost for actionable language
        actionable_words = ['should', 'must', 'recommend', 'suggest', 'propose', 
                           'strategy', 'approach', 'solution', 'opportunity', 'risk']
        
        utility_score = avg_importance
        
        for word in actionable_words:
            if word in insight.lower():
                utility_score += 0.05
        
        # Boost for strategic implications
        strategic_words = ['strategic', 'critical', 'significant', 'important', 'key']
        for word in strategic_words:
            if word in insight.lower():
                utility_score += 0.05
        
        return min(utility_score, 1.0)
    
    def _create_synthesis_metadata(self, prompt: SynthesisPrompt, insight: str) -> Dict[str, Any]:
        """Create metadata for the synthesized insight."""
        return {
            'prompt_quality': prompt.quality_score,
            'source_count': len(prompt.source_chunks),
            'source_types': list(set(chunk.memory_type.value for chunk in prompt.source_chunks)),
            'source_names': list(set(chunk.source for chunk in prompt.source_chunks)),
            'dominant_themes': prompt.synthesis_context.get('dominant_themes', []),
            'insight_length': len(insight),
            'synthesis_method': 'llm_cognitive_consolidation',
            'llm_temperature': self.temperature,
            'generation_timestamp': datetime.now().isoformat()
        }
