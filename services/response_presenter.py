#!/usr/bin/env python3
"""
Response Presenter Module
Rule-based formatter that converts RichAnswer objects into clean, focused user responses.
"""

import logging
from typing import Dict, Any, Optional
from .response_types import RichAnswer, QueryType, PresentationType, SourceReference

logger = logging.getLogger(__name__)

class ResponsePresenter:
    """
    Rule-based response formatter that presents RichAnswer objects
    according to query type and user preferences.
    """
    
    def __init__(self):
        # Default presentation rules
        self.presentation_rules = {
            QueryType.PURE_MATH: PresentationType.MINIMAL,
            QueryType.MULTI_STEP_MATH: PresentationType.WITH_STEPS,
            QueryType.COMPLEX_CALCULATION: PresentationType.DETAILED,
            QueryType.SIMPLE_FACT: PresentationType.DIRECT,
            QueryType.DOCUMENT_ANALYSIS: PresentationType.SOURCED,
            QueryType.HYBRID_DOC_CALC: PresentationType.SOURCED,
            QueryType.CONVERSATION: PresentationType.NATURAL,
            QueryType.TOOL_REQUEST: PresentationType.DETAILED,
            QueryType.CLARIFICATION: PresentationType.NATURAL,
            QueryType.ERROR: PresentationType.ERROR_FRIENDLY
        }
        
        # Confidence thresholds for presentation adjustments
        self.confidence_thresholds = {
            'high': 0.9,
            'medium': 0.7,
            'low': 0.5
        }
        
        logger.info("ResponsePresenter initialized")
    
    def present(self, rich_answer: RichAnswer, user_preferences: Optional[Dict[str, Any]] = None) -> str:
        """
        Convert a RichAnswer object into a formatted response string.
        
        Args:
            rich_answer: The structured answer object
            user_preferences: Optional user preferences for response formatting
            
        Returns:
            Formatted response string ready for display
        """
        user_preferences = user_preferences or {}
        
        try:
            # Determine presentation type
            presentation_type = self._determine_presentation_type(rich_answer, user_preferences)
            
            # Format based on presentation type
            if presentation_type == PresentationType.MINIMAL:
                return self._format_minimal(rich_answer)
            elif presentation_type == PresentationType.DIRECT:
                return self._format_direct(rich_answer)
            elif presentation_type == PresentationType.WITH_STEPS:
                return self._format_with_steps(rich_answer)
            elif presentation_type == PresentationType.DETAILED:
                return self._format_detailed(rich_answer)
            elif presentation_type == PresentationType.SOURCED:
                return self._format_sourced(rich_answer)
            elif presentation_type == PresentationType.NATURAL:
                return self._format_natural(rich_answer)
            elif presentation_type == PresentationType.ERROR_FRIENDLY:
                return self._format_error_friendly(rich_answer)
            else:
                # Fallback to direct
                return self._format_direct(rich_answer)
                
        except Exception as e:
            logger.error(f"Response presentation failed: {e}")
            return self._format_fallback(rich_answer, str(e))
    
    def _determine_presentation_type(self, rich_answer: RichAnswer, 
                                   user_preferences: Dict[str, Any]) -> PresentationType:
        """Determine the appropriate presentation type based on context."""
        
        # Check user preferences first
        if 'verbosity' in user_preferences:
            verbosity = user_preferences['verbosity']
            if verbosity == 'minimal':
                return PresentationType.MINIMAL
            elif verbosity == 'detailed':
                return PresentationType.DETAILED
        
        # Use the presentation type from RichAnswer if available
        if rich_answer.presentation_type:
            return rich_answer.presentation_type
        
        # Fall back to query type rules
        return self.presentation_rules.get(rich_answer.query_type, PresentationType.DIRECT)
    
    def _format_minimal(self, rich_answer: RichAnswer) -> str:
        """Ultra-minimal formatting: just the answer."""
        return rich_answer.direct_answer.strip()
    
    def _format_direct(self, rich_answer: RichAnswer) -> str:
        """Direct formatting: answer with minimal context if needed."""
        answer = rich_answer.direct_answer.strip()
        
        # Add brief reasoning summary if confidence is low
        if (rich_answer.confidence < self.confidence_thresholds['medium'] and 
            rich_answer.reasoning_summary):
            answer += f" ({rich_answer.reasoning_summary})"
        
        return answer
    
    def _format_with_steps(self, rich_answer: RichAnswer) -> str:
        """Format with calculation steps or brief reasoning."""
        answer = rich_answer.direct_answer.strip()
        
        # Add detailed reasoning if available
        if rich_answer.detailed_reasoning:
            # Clean up the detailed reasoning for presentation
            steps = self._format_reasoning_steps(rich_answer.detailed_reasoning)
            if steps:
                answer += f"\n\n**Calculation:**\n{steps}"
        
        # Fallback to reasoning summary
        elif rich_answer.reasoning_summary:
            answer += f"\n\n**Method:** {rich_answer.reasoning_summary}"
        
        return answer
    
    def _format_detailed(self, rich_answer: RichAnswer) -> str:
        """Detailed formatting with full explanation."""
        answer = rich_answer.direct_answer.strip()
        
        # Add reasoning summary
        if rich_answer.reasoning_summary:
            answer += f"\n\n**Summary:** {rich_answer.reasoning_summary}"
        
        # Add detailed reasoning
        if rich_answer.detailed_reasoning:
            steps = self._format_reasoning_steps(rich_answer.detailed_reasoning)
            if steps:
                answer += f"\n\n**Details:**\n{steps}"
        
        # Add supporting evidence
        if rich_answer.supporting_evidence:
            evidence = ", ".join(rich_answer.supporting_evidence)
            answer += f"\n\n**Tools used:** {evidence}"
        
        return answer
    
    def _format_sourced(self, rich_answer: RichAnswer) -> str:
        """Format with source citations."""
        answer = rich_answer.direct_answer.strip()
        
        # Add sources if available
        if rich_answer.sources:
            sources_text = self._format_sources(rich_answer.sources)
            answer += f"\n\n**Sources:** {sources_text}"
        
        # Add reasoning for complex calculations
        if (rich_answer.query_type == QueryType.HYBRID_DOC_CALC and 
            rich_answer.detailed_reasoning):
            steps = self._format_reasoning_steps(rich_answer.detailed_reasoning)
            if steps:
                answer += f"\n\n**Calculation:**\n{steps}"
        
        return answer
    
    def _format_natural(self, rich_answer: RichAnswer) -> str:
        """Natural conversational formatting."""
        # For conversational responses, use the direct answer as-is
        # since it should already be in natural language
        return rich_answer.direct_answer.strip()
    
    def _format_error_friendly(self, rich_answer: RichAnswer) -> str:
        """User-friendly error formatting."""
        error_msg = rich_answer.direct_answer.strip()
        
        # Make error messages more user-friendly
        if "error" in error_msg.lower() or "failed" in error_msg.lower():
            if rich_answer.error_details:
                # Provide helpful context without technical details
                return f"I couldn't complete that request. {error_msg}"
            else:
                return f"I couldn't complete that request: {error_msg}"
        
        return error_msg
    
    def _format_reasoning_steps(self, detailed_reasoning: str) -> str:
        """Format detailed reasoning into clean steps."""
        if not detailed_reasoning:
            return ""
        
        # Clean up common formatting issues
        steps = detailed_reasoning.strip()
        
        # If it's already formatted as numbered steps, return as-is
        if any(steps.startswith(f"{i}.") for i in range(1, 6)):
            return steps
        
        # If it's a single line, return as-is
        if '\n' not in steps:
            return steps
        
        # Split into lines and format as steps
        lines = [line.strip() for line in steps.split('\n') if line.strip()]
        if len(lines) > 1:
            formatted_steps = []
            for i, line in enumerate(lines, 1):
                if not line.startswith(f"{i}."):
                    formatted_steps.append(f"{i}. {line}")
                else:
                    formatted_steps.append(line)
            return '\n'.join(formatted_steps)
        
        return steps
    
    def _format_sources(self, sources: list) -> str:
        """Format source references for display."""
        if not sources:
            return ""
        
        formatted_sources = []
        for source in sources:
            if isinstance(source, SourceReference):
                if source.source_type == "document":
                    formatted_sources.append(f"{source.source_name}")
                elif source.source_type == "web":
                    formatted_sources.append(f"Web search")
                elif source.source_type == "calculation":
                    formatted_sources.append(f"Calculator")
                else:
                    formatted_sources.append(f"{source.source_name}")
            else:
                # Handle string sources
                formatted_sources.append(str(source))
        
        return ", ".join(formatted_sources)
    
    def _format_fallback(self, rich_answer: RichAnswer, error: str) -> str:
        """Fallback formatting when presentation fails."""
        logger.warning(f"Using fallback formatting due to error: {error}")
        
        # Return the direct answer with a note about formatting issues
        answer = rich_answer.direct_answer.strip()
        if not answer:
            answer = "Response formatting failed"
        
        return answer
    
    def get_debug_info(self, rich_answer: RichAnswer) -> Dict[str, Any]:
        """Get debug information about the response formatting."""
        return {
            'query_type': rich_answer.query_type.value,
            'presentation_type': rich_answer.presentation_type.value,
            'confidence': rich_answer.confidence,
            'has_sources': len(rich_answer.sources) > 0,
            'has_detailed_reasoning': bool(rich_answer.detailed_reasoning),
            'execution_time_ms': rich_answer.execution_metadata.execution_time_ms,
            'tools_used': rich_answer.execution_metadata.tools_used
        }

# Global instance for easy access
_response_presenter = None

def get_response_presenter() -> ResponsePresenter:
    """Get or create the global response presenter instance."""
    global _response_presenter
    if _response_presenter is None:
        _response_presenter = ResponsePresenter()
    return _response_presenter
