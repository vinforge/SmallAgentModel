#!/usr/bin/env python3
"""
SAM Persona Refinement Engine - Task 30 Phase 2
===============================================

Implements the second stage of the two-stage response pipeline.
Takes a draft response and refines it based on retrieved persona
memories to ensure consistency and personalization.

Part of Task 30: Advanced Conversational Coherence Engine
Author: SAM Development Team
Version: 1.0.0
"""

import logging

from sam.core.sam_model_client import create_legacy_ollama_client
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import re

from .persona_memory import PersonaMemoryRetriever, get_persona_memory_retriever

logger = logging.getLogger(__name__)

class PersonaRefinementEngine:
    """
    Refines draft responses using persona memories for consistency.
    
    Features:
    - Two-stage response generation (draft → refined)
    - Persona memory integration
    - User preference alignment
    - Consistency enforcement
    - Configurable refinement intensity
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the persona refinement engine.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(f"{__name__}.PersonaRefinementEngine")
        
        # Default configuration
        self.config = {
            'enable_persona_refinement': True,
            'refinement_temperature': 0.7,
            'max_refinement_tokens': 600,
            'persona_weight': 0.8,
            'consistency_threshold': 0.6,
            'skip_refinement_confidence': 0.9,
            'max_persona_context_length': 1500
        }
        
        if config:
            self.config.update(config)
        
        # Initialize persona memory retriever
        self.persona_retriever = get_persona_memory_retriever()
        
        self.logger.info("PersonaRefinementEngine initialized")
    
    def refine_with_persona(self, draft_response: str, user_question: str, 
                           user_id: Optional[str] = None, 
                           draft_confidence: float = 0.5) -> Tuple[str, Dict[str, Any]]:
        """
        Refine a draft response using persona memories.
        
        Args:
            draft_response: The initial draft response
            user_question: The original user question
            user_id: User identifier for personalized refinement
            draft_confidence: Confidence score of the draft response
            
        Returns:
            Tuple of (refined_response, refinement_metadata)
        """
        try:
            # Check if refinement should be skipped
            if not self.config['enable_persona_refinement']:
                self.logger.debug("Persona refinement disabled")
                return draft_response, {'skipped': True, 'reason': 'disabled'}
            
            if draft_confidence >= self.config['skip_refinement_confidence']:
                self.logger.debug(f"Skipping refinement due to high confidence: {draft_confidence}")
                return draft_response, {'skipped': True, 'reason': 'high_confidence', 'confidence': draft_confidence}
            
            # Retrieve persona memories
            persona_context = self.persona_retriever.retrieve_persona_memories(
                draft_response, user_id
            )
            
            if persona_context['total_memories'] == 0:
                self.logger.debug("No persona memories found, returning draft")
                return draft_response, {'skipped': True, 'reason': 'no_persona_memories'}
            
            # Generate refinement prompt
            refinement_prompt = self._create_refinement_prompt(
                draft_response, user_question, persona_context
            )
            
            # Perform refinement
            refined_response = self._perform_refinement(refinement_prompt)
            
            # Validate refinement
            refinement_metadata = self._validate_refinement(
                draft_response, refined_response, persona_context
            )
            
            if refinement_metadata['is_valid']:
                self.logger.info(f"Successfully refined response using {persona_context['total_memories']} persona memories")
                return refined_response, refinement_metadata
            else:
                self.logger.warning("Refinement validation failed, returning draft")
                return draft_response, {'skipped': True, 'reason': 'validation_failed', **refinement_metadata}
            
        except Exception as e:
            self.logger.error(f"Error in persona refinement: {e}")
            return draft_response, {'skipped': True, 'reason': 'error', 'error': str(e)}
    
    def _create_refinement_prompt(self, draft_response: str, user_question: str, 
                                 persona_context: Dict[str, Any]) -> str:
        """Create the refinement prompt template."""
        
        # Format persona memories
        persona_sections = []
        
        if persona_context['preferences']:
            prefs = []
            for pref in persona_context['preferences']:
                prefs.append(f"• {pref.content}")
            persona_sections.append(f"**User Preferences:**\n" + "\n".join(prefs))
        
        if persona_context['learned_facts']:
            facts = []
            for fact in persona_context['learned_facts']:
                facts.append(f"• {fact.content}")
            persona_sections.append(f"**Learned Facts:**\n" + "\n".join(facts))
        
        if persona_context['corrections']:
            corrections = []
            for correction in persona_context['corrections']:
                corrections.append(f"• {correction.content}")
            persona_sections.append(f"**Previous Corrections:**\n" + "\n".join(corrections))
        
        if persona_context['conversation_summaries']:
            summaries = []
            for summary in persona_context['conversation_summaries']:
                summaries.append(f"• {summary.content}")
            persona_sections.append(f"**Conversation Context:**\n" + "\n".join(summaries))
        
        persona_context_text = "\n\n".join(persona_sections)
        
        # Truncate if too long
        if len(persona_context_text) > self.config['max_persona_context_length']:
            persona_context_text = persona_context_text[:self.config['max_persona_context_length']] + "... [truncated]"
        
        # Create refinement prompt
        refinement_prompt = f"""System: You are SAM. Your task is to refine a draft response to better align with your established persona and your memory of the user. Use the provided persona context to make the response more personalized, consistent, and helpful.

--- DRAFT RESPONSE ---
{draft_response}
--- END OF DRAFT ---

--- PERSONA & USER CONTEXT ---
{persona_context_text}
--- END OF CONTEXT ---

**Refinement Guidelines:**
1. **Maintain Core Information**: Keep all factual content from the draft
2. **Apply User Preferences**: Adjust style, detail level, and format based on preferences
3. **Use Learned Facts**: Reference relevant facts the user has taught you
4. **Honor Corrections**: Ensure any previous corrections are respected
5. **Personalize Tone**: Match the communication style the user prefers
6. **Be Consistent**: Align with your established persona from past interactions

**Original Question**: {user_question}

Refine the draft into its final form that better reflects your persona and relationship with this user:"""
        
        return refinement_prompt
    
    def _perform_refinement(self, refinement_prompt: str) -> str:
        """Perform the actual refinement using the LLM."""
        try:
            import requests
            
            # Use Ollama for refinement
            response = create_legacy_ollama_client().generate(prompt)
            
            if response.status_code == 200:
                response_data = response.json()
                refined_text = response_data.get('response', '').strip()
                
                if refined_text:
                    return refined_text
                else:
                    self.logger.warning("Empty response from refinement LLM")
                    return ""
            else:
                self.logger.error(f"Refinement LLM call failed with status {response.status_code}")
                return ""
                
        except Exception as e:
            self.logger.error(f"Error performing refinement: {e}")
            return ""
    
    def _validate_refinement(self, draft_response: str, refined_response: str, 
                           persona_context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that the refinement is appropriate."""
        metadata = {
            'is_valid': False,
            'validation_score': 0.0,
            'persona_memories_used': persona_context['total_memories'],
            'refinement_confidence': persona_context['retrieval_confidence'],
            'changes_made': False
        }
        
        try:
            # Check if refinement was actually performed
            if not refined_response or len(refined_response.strip()) < 10:
                metadata['validation_error'] = 'Empty or too short refined response'
                return metadata
            
            # Check if changes were made
            if refined_response.strip() == draft_response.strip():
                metadata['validation_error'] = 'No changes made during refinement'
                return metadata
            
            metadata['changes_made'] = True
            
            # Basic quality checks
            draft_length = len(draft_response.split())
            refined_length = len(refined_response.split())
            
            # Refined response shouldn't be too much shorter (information loss)
            if refined_length < draft_length * 0.5:
                metadata['validation_error'] = 'Refined response too short (possible information loss)'
                return metadata
            
            # Refined response shouldn't be excessively longer
            if refined_length > draft_length * 2.5:
                metadata['validation_error'] = 'Refined response too long (possible hallucination)'
                return metadata
            
            # Check for persona alignment indicators
            persona_score = 0.0
            
            # Check if user preferences are reflected
            for pref in persona_context.get('preferences', []):
                pref_words = [word for word in pref.content.lower().split() if len(word) > 3][:3]
                if pref_words and any(word in refined_response.lower() for word in pref_words):
                    persona_score += 0.3

            # Check if learned facts are preserved/referenced
            for fact in persona_context.get('learned_facts', []):
                fact_words = [word for word in fact.content.lower().split() if len(word) > 3][:3]
                if fact_words and any(word in refined_response.lower() for word in fact_words):
                    persona_score += 0.2

            # Check if corrections are honored
            for correction in persona_context.get('corrections', []):
                correction_words = [word for word in correction.content.lower().split() if len(word) > 3][:3]
                if correction_words and any(word in refined_response.lower() for word in correction_words):
                    persona_score += 0.4

            # Give some base score for having persona memories available
            if persona_context['total_memories'] > 0:
                persona_score += 0.1
            
            metadata['validation_score'] = min(1.0, persona_score)
            
            # Consider valid if score meets threshold
            if metadata['validation_score'] >= self.config['consistency_threshold']:
                metadata['is_valid'] = True
            else:
                metadata['validation_error'] = f"Persona alignment score too low: {metadata['validation_score']}"
            
            return metadata
            
        except Exception as e:
            metadata['validation_error'] = f"Validation error: {e}"
            return metadata

def generate_final_response(user_question: str, draft_response: str, 
                          user_id: Optional[str] = None, 
                          draft_confidence: float = 0.5) -> Tuple[str, Dict[str, Any]]:
    """
    Generate final response using two-stage pipeline (Task 30 Phase 2).
    
    Args:
        user_question: The original user question
        draft_response: The draft response from stage 1
        user_id: User identifier for personalization
        draft_confidence: Confidence score of the draft
        
    Returns:
        Tuple of (final_response, metadata)
    """
    try:
        # Initialize refinement engine
        refinement_engine = PersonaRefinementEngine()
        
        # Stage 2: Refine the draft with persona
        final_response, refinement_metadata = refinement_engine.refine_with_persona(
            draft_response, user_question, user_id, draft_confidence
        )
        
        # Add pipeline metadata
        pipeline_metadata = {
            'pipeline_stage': 'two_stage_complete',
            'draft_length': len(draft_response.split()),
            'final_length': len(final_response.split()),
            'refinement_applied': not refinement_metadata.get('skipped', False),
            'refinement_metadata': refinement_metadata
        }
        
        return final_response, pipeline_metadata
        
    except Exception as e:
        logger.error(f"Error in two-stage response generation: {e}")
        return draft_response, {
            'pipeline_stage': 'fallback_to_draft',
            'error': str(e),
            'refinement_applied': False
        }
