#!/usr/bin/env python3
"""
SAM LLM-as-a-Judge Evaluation System - Task 30 Phase 3
======================================================

Implements LLM-based evaluation for response quality assessment.
Uses an LLM to judge conversational coherence, persona consistency,
and overall response quality for A/B testing validation.

Part of Task 30: Advanced Conversational Coherence Engine
Author: SAM Development Team
Version: 1.0.0
"""

import logging
import json
import requests
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import re

logger = logging.getLogger(__name__)

@dataclass
class EvaluationCriteria:
    """Criteria for LLM-based evaluation."""
    coherence_weight: float = 0.3
    persona_consistency_weight: float = 0.25
    factual_accuracy_weight: float = 0.25
    helpfulness_weight: float = 0.2
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class EvaluationResult:
    """Result of LLM-based evaluation."""
    overall_score: float  # 0.0 to 1.0
    coherence_score: float
    persona_consistency_score: float
    factual_accuracy_score: float
    helpfulness_score: float
    reasoning: str
    confidence: float
    evaluation_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationResult':
        return cls(**data)

class LLMJudge:
    """
    LLM-based evaluation system for response quality assessment.
    
    Features:
    - Multi-criteria evaluation (coherence, persona, accuracy, helpfulness)
    - Configurable evaluation weights
    - Detailed reasoning and confidence scores
    - Support for multiple LLM backends
    - Batch evaluation capabilities
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LLM judge.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(f"{__name__}.LLMJudge")
        
        # Default configuration
        self.config = {
            'llm_endpoint': 'http://localhost:11434/api/generate',
            'judge_model': 'hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M',
            'evaluation_temperature': 0.3,
            'max_evaluation_tokens': 800,
            'timeout_seconds': 60,
            'enable_llm_evaluation': True
        }
        
        if config:
            self.config.update(config)
        
        # Default evaluation criteria
        self.criteria = EvaluationCriteria()
        
        self.logger.info("LLMJudge initialized")
    
    def evaluate_response(self, user_question: str, response: str, 
                         conversation_history: Optional[str] = None,
                         persona_context: Optional[str] = None,
                         criteria: Optional[EvaluationCriteria] = None) -> Optional[EvaluationResult]:
        """
        Evaluate a response using LLM-as-a-Judge.
        
        Args:
            user_question: The user's question
            response: The response to evaluate
            conversation_history: Recent conversation context
            persona_context: User persona and preferences
            criteria: Evaluation criteria (uses default if None)
            
        Returns:
            EvaluationResult or None if evaluation fails
        """
        try:
            if not self.config['enable_llm_evaluation']:
                self.logger.debug("LLM evaluation disabled")
                return None
            
            start_time = datetime.now()
            
            # Use provided criteria or default
            eval_criteria = criteria or self.criteria
            
            # Create evaluation prompt
            evaluation_prompt = self._create_evaluation_prompt(
                user_question, response, conversation_history, persona_context, eval_criteria
            )
            
            # Get LLM evaluation
            evaluation_text = self._call_llm_judge(evaluation_prompt)
            
            if not evaluation_text:
                self.logger.warning("Empty evaluation from LLM judge")
                return None
            
            # Parse evaluation result
            result = self._parse_evaluation_result(evaluation_text, eval_criteria)
            
            # Calculate evaluation time
            end_time = datetime.now()
            result.evaluation_time_ms = (end_time - start_time).total_seconds() * 1000
            
            self.logger.debug(f"Evaluated response with overall score: {result.overall_score:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating response: {e}")
            return None
    
    def compare_responses(self, user_question: str, response_a: str, response_b: str,
                         conversation_history: Optional[str] = None,
                         persona_context: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Compare two responses using LLM-as-a-Judge.
        
        Args:
            user_question: The user's question
            response_a: First response (e.g., control)
            response_b: Second response (e.g., treatment)
            conversation_history: Recent conversation context
            persona_context: User persona and preferences
            
        Returns:
            Comparison result dictionary
        """
        try:
            # Evaluate both responses
            eval_a = self.evaluate_response(user_question, response_a, conversation_history, persona_context)
            eval_b = self.evaluate_response(user_question, response_b, conversation_history, persona_context)
            
            if not eval_a or not eval_b:
                return None
            
            # Create comparison
            comparison = {
                'response_a_score': eval_a.overall_score,
                'response_b_score': eval_b.overall_score,
                'winner': 'response_a' if eval_a.overall_score > eval_b.overall_score else 'response_b',
                'score_difference': abs(eval_a.overall_score - eval_b.overall_score),
                'detailed_comparison': {
                    'coherence': {
                        'response_a': eval_a.coherence_score,
                        'response_b': eval_b.coherence_score,
                        'winner': 'response_a' if eval_a.coherence_score > eval_b.coherence_score else 'response_b'
                    },
                    'persona_consistency': {
                        'response_a': eval_a.persona_consistency_score,
                        'response_b': eval_b.persona_consistency_score,
                        'winner': 'response_a' if eval_a.persona_consistency_score > eval_b.persona_consistency_score else 'response_b'
                    },
                    'factual_accuracy': {
                        'response_a': eval_a.factual_accuracy_score,
                        'response_b': eval_b.factual_accuracy_score,
                        'winner': 'response_a' if eval_a.factual_accuracy_score > eval_b.factual_accuracy_score else 'response_b'
                    },
                    'helpfulness': {
                        'response_a': eval_a.helpfulness_score,
                        'response_b': eval_b.helpfulness_score,
                        'winner': 'response_a' if eval_a.helpfulness_score > eval_b.helpfulness_score else 'response_b'
                    }
                },
                'evaluation_a': eval_a.to_dict(),
                'evaluation_b': eval_b.to_dict()
            }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing responses: {e}")
            return None
    
    def _create_evaluation_prompt(self, user_question: str, response: str,
                                 conversation_history: Optional[str],
                                 persona_context: Optional[str],
                                 criteria: EvaluationCriteria) -> str:
        """Create the evaluation prompt for the LLM judge."""
        
        # Build context sections
        context_sections = []
        
        if conversation_history:
            context_sections.append(f"**Recent Conversation History:**\n{conversation_history}")
        
        if persona_context:
            context_sections.append(f"**User Persona & Preferences:**\n{persona_context}")
        
        context_text = "\n\n".join(context_sections) if context_sections else "No additional context provided."
        
        # Create evaluation prompt
        prompt = f"""You are an expert AI evaluator tasked with assessing the quality of AI assistant responses. Evaluate the following response across multiple criteria and provide detailed scores.

--- CONTEXT ---
{context_text}

--- USER QUESTION ---
{user_question}

--- RESPONSE TO EVALUATE ---
{response}

--- EVALUATION CRITERIA ---
Please evaluate the response on a scale of 0.0 to 1.0 for each criterion:

1. **Coherence** (Weight: {criteria.coherence_weight}): How well does the response flow logically and maintain consistency with the conversation context?

2. **Persona Consistency** (Weight: {criteria.persona_consistency_weight}): How well does the response align with the user's established preferences and the AI's persona?

3. **Factual Accuracy** (Weight: {criteria.factual_accuracy_weight}): How accurate and reliable is the information provided in the response?

4. **Helpfulness** (Weight: {criteria.helpfulness_weight}): How useful and relevant is the response in addressing the user's question?

--- REQUIRED OUTPUT FORMAT ---
Please provide your evaluation in the following JSON format:

{{
  "coherence_score": 0.0-1.0,
  "persona_consistency_score": 0.0-1.0,
  "factual_accuracy_score": 0.0-1.0,
  "helpfulness_score": 0.0-1.0,
  "reasoning": "Detailed explanation of your evaluation",
  "confidence": 0.0-1.0
}}

Provide your evaluation:"""
        
        return prompt
    
    def _call_llm_judge(self, prompt: str) -> Optional[str]:
        """Call the LLM judge with the evaluation prompt."""
        try:
            response = requests.post(
                self.config['llm_endpoint'],
                json={
                    "model": self.config['judge_model'],
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config['evaluation_temperature'],
                        "top_p": 0.9,
                        "max_tokens": self.config['max_evaluation_tokens']
                    }
                },
                timeout=self.config['timeout_seconds']
            )
            
            if response.status_code == 200:
                response_data = response.json()
                return response_data.get('response', '').strip()
            else:
                self.logger.error(f"LLM judge call failed with status {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error calling LLM judge: {e}")
            return None
    
    def _parse_evaluation_result(self, evaluation_text: str, criteria: EvaluationCriteria) -> EvaluationResult:
        """Parse the LLM evaluation result."""
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', evaluation_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
                eval_data = json.loads(json_text)
            else:
                # Fallback: try to parse the entire response as JSON
                eval_data = json.loads(evaluation_text)
            
            # Extract scores
            coherence_score = float(eval_data.get('coherence_score', 0.5))
            persona_score = float(eval_data.get('persona_consistency_score', 0.5))
            accuracy_score = float(eval_data.get('factual_accuracy_score', 0.5))
            helpfulness_score = float(eval_data.get('helpfulness_score', 0.5))
            
            # Calculate weighted overall score
            overall_score = (
                coherence_score * criteria.coherence_weight +
                persona_score * criteria.persona_consistency_weight +
                accuracy_score * criteria.factual_accuracy_weight +
                helpfulness_score * criteria.helpfulness_weight
            )
            
            # Extract reasoning and confidence
            reasoning = eval_data.get('reasoning', 'No reasoning provided')
            confidence = float(eval_data.get('confidence', 0.5))
            
            return EvaluationResult(
                overall_score=overall_score,
                coherence_score=coherence_score,
                persona_consistency_score=persona_score,
                factual_accuracy_score=accuracy_score,
                helpfulness_score=helpfulness_score,
                reasoning=reasoning,
                confidence=confidence,
                evaluation_time_ms=0.0  # Will be set by caller
            )
            
        except Exception as e:
            self.logger.warning(f"Error parsing evaluation result, using defaults: {e}")
            # Return default scores if parsing fails
            return EvaluationResult(
                overall_score=0.5,
                coherence_score=0.5,
                persona_consistency_score=0.5,
                factual_accuracy_score=0.5,
                helpfulness_score=0.5,
                reasoning=f"Evaluation parsing failed: {e}",
                confidence=0.1,
                evaluation_time_ms=0.0
            )

# Global LLM judge instance
_llm_judge: Optional[LLMJudge] = None

def get_llm_judge(config: Optional[Dict[str, Any]] = None) -> LLMJudge:
    """Get the global LLM judge instance."""
    global _llm_judge
    
    if _llm_judge is None:
        _llm_judge = LLMJudge(config)
    return _llm_judge
