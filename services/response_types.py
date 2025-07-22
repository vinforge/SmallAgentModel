#!/usr/bin/env python3
"""
Response Types and Data Structures
Defines the RichAnswer object and related structures for the two-stage response system.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union
from enum import Enum

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of queries for presentation formatting."""
    PURE_MATH = "pure_math"
    MULTI_STEP_MATH = "multi_step_math"
    COMPLEX_CALCULATION = "complex_calculation"
    SIMPLE_FACT = "simple_fact"
    DOCUMENT_ANALYSIS = "document_analysis"
    HYBRID_DOC_CALC = "hybrid_doc_calc"
    CONVERSATION = "conversation"
    TOOL_REQUEST = "tool_request"
    CLARIFICATION = "clarification"
    ERROR = "error"

class PresentationType(Enum):
    """How the response should be presented to the user."""
    MINIMAL = "minimal"           # Just the answer: "12"
    DIRECT = "direct"             # Answer with brief context: "12 (5+7)"
    WITH_STEPS = "with_steps"     # Answer with calculation steps
    DETAILED = "detailed"         # Full explanation with reasoning
    SOURCED = "sourced"          # Answer with source citations
    NATURAL = "natural"          # Conversational response
    ERROR_FRIENDLY = "error_friendly"  # User-friendly error message

@dataclass
class ExecutionMetadata:
    """Metadata about how the query was processed."""
    route_type: str = ""
    program_used: Optional[str] = None
    execution_time_ms: float = 0.0
    steps_completed: int = 0
    steps_failed: int = 0
    tools_used: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)

@dataclass
class SourceReference:
    """Reference to a source used in generating the answer."""
    source_id: str
    source_type: str  # "document", "web", "calculation", "tool"
    source_name: str
    relevance_score: float = 0.0
    excerpt: Optional[str] = None

@dataclass
class RichAnswer:
    """
    Structured answer object that separates the direct answer from reasoning.
    
    This is the core object that bridges SAM's sophisticated reasoning
    with clean, focused user presentation.
    """
    # Core answer components
    direct_answer: str
    query_type: QueryType
    presentation_type: PresentationType
    confidence: float
    
    # Supporting information
    supporting_evidence: List[str] = field(default_factory=list)
    reasoning_summary: str = ""
    detailed_reasoning: str = ""
    
    # Source and metadata
    sources: List[SourceReference] = field(default_factory=list)
    execution_metadata: ExecutionMetadata = field(default_factory=ExecutionMetadata)
    
    # Additional context
    alternative_interpretations: List[str] = field(default_factory=list)
    follow_up_suggestions: List[str] = field(default_factory=list)
    error_details: Optional[str] = None
    
    # User context
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    conversation_context: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> str:
        """Convert RichAnswer to JSON string."""
        try:
            # Convert enums to strings for JSON serialization
            data = asdict(self)
            data['query_type'] = self.query_type.value
            data['presentation_type'] = self.presentation_type.value
            return json.dumps(data, indent=2)
        except Exception as e:
            logger.error(f"Failed to serialize RichAnswer to JSON: {e}")
            return json.dumps({
                "direct_answer": self.direct_answer,
                "error": f"Serialization failed: {str(e)}"
            })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'RichAnswer':
        """Create RichAnswer from JSON string with robust error handling."""
        try:
            data = json.loads(json_str)

            # Ensure required fields are present with defaults
            if 'direct_answer' not in data:
                data['direct_answer'] = "No answer provided"

            if 'query_type' not in data:
                data['query_type'] = 'simple_fact'

            if 'presentation_type' not in data:
                # Auto-determine presentation type based on query type
                query_type_str = data['query_type']
                if query_type_str == 'pure_math':
                    data['presentation_type'] = 'minimal'
                elif query_type_str == 'multi_step_math':
                    data['presentation_type'] = 'with_steps'
                elif query_type_str in ['document_analysis', 'hybrid_doc_calc']:
                    data['presentation_type'] = 'sourced'
                else:
                    data['presentation_type'] = 'direct'

            if 'confidence' not in data:
                data['confidence'] = 0.8  # Default confidence

            # Convert string enums back to enum objects
            try:
                data['query_type'] = QueryType(data['query_type'])
            except ValueError:
                data['query_type'] = QueryType.SIMPLE_FACT

            try:
                data['presentation_type'] = PresentationType(data['presentation_type'])
            except ValueError:
                data['presentation_type'] = PresentationType.DIRECT

            # Handle nested objects with defaults
            if 'execution_metadata' in data and isinstance(data['execution_metadata'], dict):
                data['execution_metadata'] = ExecutionMetadata(**data['execution_metadata'])
            elif 'execution_metadata' not in data:
                data['execution_metadata'] = ExecutionMetadata()

            if 'sources' in data and isinstance(data['sources'], list):
                data['sources'] = [
                    SourceReference(**src) if isinstance(src, dict) else src
                    for src in data['sources']
                ]
            elif 'sources' not in data:
                data['sources'] = []

            # Set defaults for other optional fields
            for field in ['supporting_evidence', 'alternative_interpretations', 'follow_up_suggestions']:
                if field not in data:
                    data[field] = []

            for field in ['reasoning_summary', 'detailed_reasoning', 'error_details']:
                if field not in data:
                    data[field] = ""

            for field in ['user_preferences', 'conversation_context']:
                if field not in data:
                    data[field] = {}

            return cls(**data)

        except Exception as e:
            logger.error(f"Failed to deserialize RichAnswer from JSON: {e}")
            # Return a basic error RichAnswer
            return cls(
                direct_answer="Error parsing response",
                query_type=QueryType.ERROR,
                presentation_type=PresentationType.ERROR_FRIENDLY,
                confidence=0.0,
                error_details=str(e)
            )
    
    @classmethod
    def create_simple(cls, answer: str, query_type: QueryType, confidence: float = 1.0) -> 'RichAnswer':
        """Create a simple RichAnswer with minimal information."""
        presentation_map = {
            QueryType.PURE_MATH: PresentationType.MINIMAL,
            QueryType.MULTI_STEP_MATH: PresentationType.WITH_STEPS,
            QueryType.SIMPLE_FACT: PresentationType.DIRECT,
            QueryType.CONVERSATION: PresentationType.NATURAL,
            QueryType.DOCUMENT_ANALYSIS: PresentationType.SOURCED,
            QueryType.ERROR: PresentationType.ERROR_FRIENDLY
        }
        
        return cls(
            direct_answer=answer,
            query_type=query_type,
            presentation_type=presentation_map.get(query_type, PresentationType.DIRECT),
            confidence=confidence
        )
    
    @classmethod
    def create_error(cls, error_message: str, error_details: str = "") -> 'RichAnswer':
        """Create an error RichAnswer."""
        return cls(
            direct_answer=error_message,
            query_type=QueryType.ERROR,
            presentation_type=PresentationType.ERROR_FRIENDLY,
            confidence=0.0,
            error_details=error_details
        )
    
    def add_source(self, source_id: str, source_type: str, source_name: str, 
                   relevance_score: float = 0.0, excerpt: str = None) -> None:
        """Add a source reference to the answer."""
        source_ref = SourceReference(
            source_id=source_id,
            source_type=source_type,
            source_name=source_name,
            relevance_score=relevance_score,
            excerpt=excerpt
        )
        self.sources.append(source_ref)
    
    def set_execution_metadata(self, route_type: str, execution_time_ms: float,
                              tools_used: List[str] = None, **kwargs) -> None:
        """Set execution metadata for the answer."""
        self.execution_metadata = ExecutionMetadata(
            route_type=route_type,
            execution_time_ms=execution_time_ms,
            tools_used=tools_used or [],
            **kwargs
        )
    
    def is_successful(self) -> bool:
        """Check if this represents a successful answer."""
        return (self.query_type != QueryType.ERROR and 
                self.confidence > 0.0 and 
                self.direct_answer and 
                self.direct_answer.lower() not in ['error', 'failed', 'unknown'])
    
    def get_display_confidence(self) -> str:
        """Get a user-friendly confidence indicator."""
        if self.confidence >= 0.9:
            return "High"
        elif self.confidence >= 0.7:
            return "Medium"
        elif self.confidence >= 0.5:
            return "Low"
        else:
            return "Very Low"

# Utility functions for working with RichAnswer objects

def parse_llm_response_to_rich_answer(llm_response: str, fallback_query_type: QueryType = QueryType.SIMPLE_FACT) -> RichAnswer:
    """
    Parse LLM response that should be in JSON format into a RichAnswer object.
    Includes fallback parsing for malformed JSON.
    """
    try:
        # Try to parse as JSON first
        if llm_response.strip().startswith('{') and llm_response.strip().endswith('}'):
            return RichAnswer.from_json(llm_response)
        
        # If not JSON, try to extract JSON from the response
        json_start = llm_response.find('{')
        json_end = llm_response.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_part = llm_response[json_start:json_end]
            return RichAnswer.from_json(json_part)
        
        # Fallback: treat entire response as direct answer
        logger.warning("LLM response not in expected JSON format, using fallback parsing")
        return RichAnswer.create_simple(
            answer=llm_response.strip(),
            query_type=fallback_query_type,
            confidence=0.5
        )
        
    except Exception as e:
        logger.error(f"Failed to parse LLM response: {e}")
        return RichAnswer.create_error(
            error_message="Failed to parse response",
            error_details=str(e)
        )

def merge_rich_answers(answers: List[RichAnswer], primary_answer: RichAnswer = None) -> RichAnswer:
    """
    Merge multiple RichAnswer objects, useful for hybrid queries.
    """
    if not answers:
        return RichAnswer.create_error("No answers to merge")
    
    if primary_answer is None:
        primary_answer = answers[0]
    
    # Combine sources from all answers
    all_sources = primary_answer.sources.copy()
    for answer in answers[1:]:
        all_sources.extend(answer.sources)
    
    # Combine supporting evidence
    all_evidence = primary_answer.supporting_evidence.copy()
    for answer in answers[1:]:
        all_evidence.extend(answer.supporting_evidence)
    
    # Use the highest confidence
    max_confidence = max(answer.confidence for answer in answers)
    
    return RichAnswer(
        direct_answer=primary_answer.direct_answer,
        query_type=primary_answer.query_type,
        presentation_type=primary_answer.presentation_type,
        confidence=max_confidence,
        supporting_evidence=list(set(all_evidence)),  # Remove duplicates
        reasoning_summary=primary_answer.reasoning_summary,
        detailed_reasoning=primary_answer.detailed_reasoning,
        sources=all_sources,
        execution_metadata=primary_answer.execution_metadata
    )
