#!/usr/bin/env python3
"""
SSRL Self-Search Tool
=====================

Implements the Self-Supervised Reasoning and Learning (SSRL) methodology
as a tool that can recursively call the LLM with specialized prompts for
deep reasoning and self-assessment.

Features:
- Recursive LLM calls with structured reasoning prompts
- Self-confidence assessment
- Infinite loop prevention with multiple safety mechanisms
- Integration with SAM's existing tool framework

Author: SAM Development Team
Version: 1.0.0
"""

import logging
import time
import json
import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SSRLConfidenceLevel(Enum):
    """Confidence levels for SSRL self-assessment."""
    VERY_HIGH = "VERY_HIGH"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    VERY_LOW = "VERY_LOW"


@dataclass
class SSRLResult:
    """Result from SSRL self-search with confidence assessment."""
    content: str
    confidence_score: float  # 0.0 - 1.0
    confidence_level: SSRLConfidenceLevel
    reasoning_steps: List[str]
    search_iterations: int
    execution_time: float
    success: bool
    error: Optional[str] = None


class SelfSearchTool:
    """
    SSRL Self-Search Tool that implements recursive reasoning with safety mechanisms.
    
    This tool uses the LLM to reason about queries in a structured way,
    generating step-by-step thinking, searching internal knowledge,
    and providing self-assessed confidence scores.
    """
    
    def __init__(self, 
                 max_depth: int = 3,
                 confidence_threshold: float = 0.7,
                 timeout_seconds: int = 60,
                 max_iterations: int = 5):
        """
        Initialize the SelfSearchTool with safety parameters.
        
        Args:
            max_depth: Maximum recursion depth to prevent infinite loops
            confidence_threshold: Minimum confidence to accept result
            timeout_seconds: Maximum execution time per search
            max_iterations: Maximum number of reasoning iterations
        """
        self.max_depth = max_depth
        self.confidence_threshold = confidence_threshold
        self.timeout_seconds = timeout_seconds
        self.max_iterations = max_iterations
        
        # Safety mechanisms
        self.call_stack = []  # Track recursion depth
        self.start_time = None
        self.iteration_count = 0
        
        # Circuit breaker for repeated failures
        self.failure_count = 0
        self.max_failures = 3
        self.circuit_breaker_timeout = 300  # 5 minutes
        self.last_failure_time = 0
        
        logger.info(f"SelfSearchTool initialized with max_depth={max_depth}, "
                   f"confidence_threshold={confidence_threshold}")
    
    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> SSRLResult:
        """
        Execute self-search reasoning for the given query.
        
        Args:
            query: The user's question or request
            context: Additional context including conversation history
            
        Returns:
            SSRLResult with reasoning outcome and confidence assessment
        """
        self.start_time = time.time()
        self.iteration_count = 0
        
        try:
            # Check circuit breaker
            if self._is_circuit_breaker_active():
                return self._create_failure_result(
                    "Self-search temporarily disabled due to repeated failures"
                )
            
            # Safety check: prevent nested self-searches
            if context and context.get('is_self_search', False):
                return self._create_failure_result(
                    "Nested self-search detected - preventing infinite recursion"
                )
            
            # Check recursion depth
            if len(self.call_stack) >= self.max_depth:
                return self._create_failure_result(
                    f"Maximum recursion depth ({self.max_depth}) reached"
                )
            
            # Add to call stack
            call_id = f"self_search_{len(self.call_stack)}_{int(time.time())}"
            self.call_stack.append(call_id)
            
            # Prepare context for self-search
            search_context = (context or {}).copy()
            search_context['is_self_search'] = True
            search_context['call_id'] = call_id
            
            try:
                # Perform the self-search reasoning
                result = self._perform_self_search_reasoning(query, search_context)
                
                # Reset failure count on success
                if result.success:
                    self.failure_count = 0
                else:
                    self.failure_count += 1
                    self.last_failure_time = time.time()
                
                return result
                
            finally:
                # Always remove from call stack
                if self.call_stack and self.call_stack[-1] == call_id:
                    self.call_stack.pop()
                    
        except Exception as e:
            logger.error(f"SelfSearchTool execution failed: {e}")
            self.failure_count += 1
            self.last_failure_time = time.time()
            return self._create_failure_result(f"Execution error: {e}")
    
    def _perform_self_search_reasoning(self, query: str, context: Dict[str, Any]) -> SSRLResult:
        """
        Perform the core SSRL reasoning process.
        
        Args:
            query: The user's question
            context: Search context with safety flags
            
        Returns:
            SSRLResult with reasoning outcome
        """
        reasoning_steps = []
        
        try:
            # Generate SSRL prompt
            ssrl_prompt = self._create_ssrl_prompt(query, context)
            
            # Get LLM response with structured reasoning
            llm_response = self._call_llm_with_ssrl_prompt(ssrl_prompt)
            
            # Parse structured response
            parsed_response = self._parse_ssrl_response(llm_response)
            
            # Extract reasoning steps
            reasoning_steps = parsed_response.get('reasoning_steps', [])
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(parsed_response)
            confidence_level = self._score_to_level(confidence_score)
            
            # Determine if we need another iteration
            if (confidence_score < self.confidence_threshold and 
                self.iteration_count < self.max_iterations and
                not self._is_timeout_reached()):
                
                self.iteration_count += 1
                logger.info(f"Low confidence ({confidence_score:.2f}), "
                           f"attempting iteration {self.iteration_count}")
                
                # Recursive call with refined query
                refined_query = self._refine_query_for_iteration(query, parsed_response)
                return self._perform_self_search_reasoning(refined_query, context)
            
            # Create final result
            execution_time = time.time() - self.start_time
            
            return SSRLResult(
                content=parsed_response.get('answer', ''),
                confidence_score=confidence_score,
                confidence_level=confidence_level,
                reasoning_steps=reasoning_steps,
                search_iterations=self.iteration_count + 1,
                execution_time=execution_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"SSRL reasoning failed: {e}")
            return self._create_failure_result(f"Reasoning error: {e}")
    
    def _create_ssrl_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """
        Create the specialized SSRL prompt for structured reasoning.
        
        Args:
            query: The user's question
            context: Additional context
            
        Returns:
            Formatted SSRL prompt
        """
        # Get conversation history if available
        history_context = ""
        if context.get('conversation_history'):
            recent_history = context['conversation_history'][-3:]  # Last 3 exchanges
            history_context = "\n".join([
                f"User: {h.get('user', '')}\nAssistant: {h.get('assistant', '')}"
                for h in recent_history
            ])
        
        # Build the SSRL prompt
        base_prompt = """You are an AI assistant using Self-Supervised Reasoning and Learning (SSRL).
Your task is to answer the user's question through structured reasoning and self-assessment.

IMPORTANT: Structure your response using these exact tags:

<think>
[Your step-by-step reasoning process. Break down the problem, consider different angles,
and work through the logic systematically.]
</think>

<search>
[Search your internal knowledge for relevant information. What do you know about this topic?
What facts, concepts, or examples are relevant?]
</search>

<information>
[Synthesize the information you found. What are the key points? How do they relate to the question?]
</information>

<confidence>
[Assess your confidence in this answer on a scale of 0.0 to 1.0. Consider:
- How certain are you about the facts?
- How complete is your knowledge on this topic?
- Are there any gaps or uncertainties?
Provide just the number, e.g., 0.8]
</confidence>

<answer>
[Your final, clear answer to the user's question based on your reasoning above.]
</answer>

"""

        # Add conversation context if available
        if history_context:
            base_prompt += f"Recent conversation context:\n{history_context}\n\n"

        # Add the user's question
        base_prompt += f"User's question: {query}\n\n"
        base_prompt += "Remember: Be thorough in your reasoning, honest about uncertainties, and provide a realistic confidence assessment."

        ssrl_prompt = base_prompt

        return ssrl_prompt
    
    def _call_llm_with_ssrl_prompt(self, prompt: str) -> str:
        """
        Call the LLM with the SSRL prompt.
        
        Args:
            prompt: The formatted SSRL prompt
            
        Returns:
            LLM response text
        """
        try:
            # Import the LLM interface
            from sam.models.model_interface import get_model_interface
            
            model_interface = get_model_interface()
            
            # Call the LLM with the SSRL prompt
            response = model_interface.generate_response(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.7,
                system_message="You are a helpful AI assistant using structured reasoning."
            )
            
            return response
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise Exception(f"Failed to get LLM response: {e}")
    
    def _parse_ssrl_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the structured SSRL response.
        
        Args:
            response: Raw LLM response with SSRL tags
            
        Returns:
            Parsed response dictionary
        """
        parsed = {
            'thinking': '',
            'search': '',
            'information': '',
            'confidence': 0.5,
            'answer': '',
            'reasoning_steps': []
        }
        
        try:
            # Extract sections using regex
            sections = {
                'thinking': r'<think>(.*?)</think>',
                'search': r'<search>(.*?)</search>',
                'information': r'<information>(.*?)</information>',
                'confidence': r'<confidence>(.*?)</confidence>',
                'answer': r'<answer>(.*?)</answer>'
            }
            
            for key, pattern in sections.items():
                match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                if match:
                    content = match.group(1).strip()
                    if key == 'confidence':
                        try:
                            parsed[key] = float(content)
                        except ValueError:
                            parsed[key] = 0.5  # Default confidence
                    else:
                        parsed[key] = content
            
            # Create reasoning steps from the structured sections
            reasoning_steps = []
            if parsed['thinking']:
                reasoning_steps.append(f"Thinking: {parsed['thinking'][:200]}...")
            if parsed['search']:
                reasoning_steps.append(f"Knowledge Search: {parsed['search'][:200]}...")
            if parsed['information']:
                reasoning_steps.append(f"Information Synthesis: {parsed['information'][:200]}...")
            
            parsed['reasoning_steps'] = reasoning_steps
            
            return parsed
            
        except Exception as e:
            logger.error(f"Failed to parse SSRL response: {e}")
            # Return fallback with original response as answer
            parsed['answer'] = response
            parsed['reasoning_steps'] = ["Failed to parse structured response"]
            return parsed

    def _calculate_confidence_score(self, parsed_response: Dict[str, Any]) -> float:
        """
        Calculate overall confidence score from SSRL response.

        Args:
            parsed_response: Parsed SSRL response

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Start with self-assessed confidence
        base_confidence = parsed_response.get('confidence', 0.5)

        # Adjust based on response quality indicators
        quality_factors = []

        # Factor 1: Completeness of structured response
        sections = ['thinking', 'search', 'information', 'answer']
        completeness = sum(1 for section in sections if parsed_response.get(section, '').strip())
        completeness_factor = completeness / len(sections)
        quality_factors.append(completeness_factor)

        # Factor 2: Length and detail of reasoning
        reasoning_length = len(parsed_response.get('thinking', '') + parsed_response.get('search', ''))
        length_factor = min(reasoning_length / 200, 1.0)  # Normalize to 200 chars
        quality_factors.append(length_factor)

        # Factor 3: Presence of uncertainty indicators
        answer_text = parsed_response.get('answer', '').lower()
        uncertainty_words = ['maybe', 'might', 'possibly', 'uncertain', 'not sure', 'unclear']
        uncertainty_count = sum(1 for word in uncertainty_words if word in answer_text)
        uncertainty_factor = max(0.0, 1.0 - (uncertainty_count * 0.1))
        quality_factors.append(uncertainty_factor)

        # Calculate weighted confidence
        quality_weight = 0.3
        self_assessment_weight = 0.7

        avg_quality = sum(quality_factors) / len(quality_factors)
        final_confidence = (self_assessment_weight * base_confidence +
                          quality_weight * avg_quality)

        # Ensure bounds
        return max(0.0, min(1.0, final_confidence))

    def _score_to_level(self, score: float) -> SSRLConfidenceLevel:
        """Convert confidence score to level enum."""
        if score >= 0.9:
            return SSRLConfidenceLevel.VERY_HIGH
        elif score >= 0.75:
            return SSRLConfidenceLevel.HIGH
        elif score >= 0.6:
            return SSRLConfidenceLevel.MEDIUM
        elif score >= 0.4:
            return SSRLConfidenceLevel.LOW
        else:
            return SSRLConfidenceLevel.VERY_LOW

    def _refine_query_for_iteration(self, original_query: str,
                                   previous_response: Dict[str, Any]) -> str:
        """
        Refine the query for the next iteration based on previous response.

        Args:
            original_query: The original user query
            previous_response: Previous SSRL response

        Returns:
            Refined query for next iteration
        """
        # Identify gaps or uncertainties from previous response
        thinking = previous_response.get('thinking', '')
        answer = previous_response.get('answer', '')

        # Create a refined query that addresses gaps
        refined_query = f"""Original question: {original_query}

Previous reasoning identified some uncertainties. Please provide a more detailed analysis focusing on:
- Any gaps or uncertainties mentioned in: {thinking[:300]}
- More specific information to improve this answer: {answer[:200]}

Please be more thorough and specific in your reasoning."""

        return refined_query

    def _is_timeout_reached(self) -> bool:
        """Check if execution timeout has been reached."""
        if self.start_time is None:
            return False
        return (time.time() - self.start_time) > self.timeout_seconds

    def _is_circuit_breaker_active(self) -> bool:
        """Check if circuit breaker is active due to repeated failures."""
        if self.failure_count < self.max_failures:
            return False

        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure < self.circuit_breaker_timeout

    def _create_failure_result(self, error_message: str) -> SSRLResult:
        """Create a failure result with error information."""
        execution_time = time.time() - self.start_time if self.start_time else 0.0

        return SSRLResult(
            content="",
            confidence_score=0.0,
            confidence_level=SSRLConfidenceLevel.VERY_LOW,
            reasoning_steps=[f"Error: {error_message}"],
            search_iterations=self.iteration_count,
            execution_time=execution_time,
            success=False,
            error=error_message
        )

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the SelfSearchTool."""
        return {
            'max_depth': self.max_depth,
            'confidence_threshold': self.confidence_threshold,
            'current_call_stack_depth': len(self.call_stack),
            'failure_count': self.failure_count,
            'circuit_breaker_active': self._is_circuit_breaker_active(),
            'timeout_seconds': self.timeout_seconds,
            'max_iterations': self.max_iterations
        }


# Global instance for easy access
_self_search_tool = None

def get_self_search_tool() -> SelfSearchTool:
    """Get the global SelfSearchTool instance."""
    global _self_search_tool
    if _self_search_tool is None:
        _self_search_tool = SelfSearchTool()
    return _self_search_tool
