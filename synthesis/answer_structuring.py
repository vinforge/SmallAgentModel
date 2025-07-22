"""
Advanced Answer Structuring for SAM
Improves clarity, readability, and actionability of SAM's responses.

Sprint 7 Task 3: Advanced Answer Structuring
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re

logger = logging.getLogger(__name__)

class SectionType(Enum):
    """Types of response sections."""
    SUMMARY = "summary"
    EVIDENCE = "evidence"
    ANALYSIS = "analysis"
    METHODOLOGY = "methodology"
    RESULTS = "results"
    CONCLUSIONS = "conclusions"
    NEXT_STEPS = "next_steps"
    LIMITATIONS = "limitations"
    SOURCES = "sources"
    CONFIDENCE = "confidence"

class ConfidenceLevel(Enum):
    """Confidence levels for content sections."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"

@dataclass
class DecisionNode:
    """Node in a decision tree."""
    node_id: str
    question: str
    answer: str
    confidence: float
    reasoning: str
    children: List[str]  # IDs of child nodes
    alternatives_considered: List[str]
    metadata: Dict[str, Any]

@dataclass
class StructuredSection:
    """A structured section of a response."""
    section_type: SectionType
    title: str
    content: str
    confidence_level: ConfidenceLevel
    confidence_score: float
    sources: List[str]
    metadata: Dict[str, Any]

@dataclass
class StructuredResponse:
    """A fully structured response."""
    response_id: str
    query: str
    sections: List[StructuredSection]
    decision_tree: List[DecisionNode]
    overall_confidence: float
    response_metadata: Dict[str, Any]
    created_at: str

class AdvancedAnswerStructurer:
    """
    Structures responses for improved clarity, readability, and actionability.
    """
    
    def __init__(self):
        """Initialize the answer structurer."""
        self.config = {
            'enable_decision_trees': True,
            'enable_confidence_tagging': True,
            'enable_sectioned_output': True,
            'max_sections': 8,
            'min_section_length': 50
        }
        
        logger.info("Advanced answer structurer initialized")
    
    def structure_response(self, query: str, raw_response: str, 
                         sources: List[Dict[str, Any]] = None,
                         reasoning_trace: List[Dict[str, Any]] = None,
                         tool_outputs: List[Dict[str, Any]] = None) -> StructuredResponse:
        """
        Structure a raw response into a well-organized format.
        
        Args:
            query: Original query
            raw_response: Raw response text
            sources: Source information
            reasoning_trace: Reasoning steps taken
            tool_outputs: Tool execution results
            
        Returns:
            StructuredResponse with organized sections
        """
        try:
            import uuid
            response_id = f"resp_{uuid.uuid4().hex[:12]}"
            
            logger.info(f"Structuring response for: {query[:50]}...")
            
            # Analyze response content
            content_analysis = self._analyze_response_content(raw_response)
            
            # Create structured sections
            sections = self._create_sections(
                raw_response, content_analysis, sources, tool_outputs
            )
            
            # Generate decision tree if reasoning trace available
            decision_tree = []
            if reasoning_trace and self.config['enable_decision_trees']:
                decision_tree = self._create_decision_tree(reasoning_trace)
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(sections)
            
            # Create metadata
            response_metadata = {
                'query_type': self._classify_query_type(query),
                'response_length': len(raw_response),
                'sections_count': len(sections),
                'sources_count': len(sources) if sources else 0,
                'tools_used': [t.get('tool_name') for t in (tool_outputs or [])],
                'structure_timestamp': datetime.now().isoformat()
            }
            
            structured_response = StructuredResponse(
                response_id=response_id,
                query=query,
                sections=sections,
                decision_tree=decision_tree,
                overall_confidence=overall_confidence,
                response_metadata=response_metadata,
                created_at=datetime.now().isoformat()
            )
            
            logger.info(f"Response structured: {len(sections)} sections, "
                       f"confidence: {overall_confidence:.2f}")
            
            return structured_response
            
        except Exception as e:
            logger.error(f"Error structuring response: {e}")
            # Return minimal structure on error
            return self._create_fallback_structure(query, raw_response)
    
    def format_structured_response(self, structured_response: StructuredResponse,
                                 include_confidence: bool = True,
                                 include_decision_tree: bool = False,
                                 include_metadata: bool = False) -> str:
        """
        Format a structured response as markdown.
        
        Args:
            structured_response: StructuredResponse to format
            include_confidence: Whether to include confidence indicators
            include_decision_tree: Whether to include decision tree
            include_metadata: Whether to include metadata
            
        Returns:
            Formatted markdown response
        """
        try:
            output_parts = []
            
            # Add title
            output_parts.append(f"# Response to: {structured_response.query}\n")
            
            # Add sections
            for section in structured_response.sections:
                # Section header with confidence if enabled
                if include_confidence:
                    confidence_emoji = self._get_confidence_emoji(section.confidence_level)
                    header = f"## {confidence_emoji} {section.title}"
                    if section.confidence_score > 0:
                        header += f" (confidence: {section.confidence_score:.1%})"
                else:
                    header = f"## {section.title}"
                
                output_parts.append(f"{header}\n")
                output_parts.append(f"{section.content}\n")
                
                # Add sources if available
                if section.sources and include_metadata:
                    sources_text = ", ".join(section.sources)
                    output_parts.append(f"*Sources: {sources_text}*\n")
                
                output_parts.append("")  # Empty line between sections
            
            # Add decision tree if requested
            if include_decision_tree and structured_response.decision_tree:
                output_parts.append("## 🌳 Decision Tree\n")
                tree_text = self._format_decision_tree(structured_response.decision_tree)
                output_parts.append(f"{tree_text}\n")
            
            # Add overall confidence
            if include_confidence:
                confidence_emoji = "🟢" if structured_response.overall_confidence > 0.7 else "🟡" if structured_response.overall_confidence > 0.4 else "🔴"
                output_parts.append(f"---\n{confidence_emoji} **Overall Confidence:** {structured_response.overall_confidence:.1%}\n")
            
            # Add metadata if requested
            if include_metadata:
                metadata = structured_response.response_metadata
                output_parts.append("---\n**Response Metadata:**\n")
                output_parts.append(f"- Query Type: {metadata.get('query_type', 'unknown')}")
                output_parts.append(f"- Sections: {metadata.get('sections_count', 0)}")
                output_parts.append(f"- Sources: {metadata.get('sources_count', 0)}")
                if metadata.get('tools_used'):
                    output_parts.append(f"- Tools Used: {', '.join(metadata['tools_used'])}")
                output_parts.append("")
            
            return "\n".join(output_parts)
            
        except Exception as e:
            logger.error(f"Error formatting structured response: {e}")
            return f"Error formatting response: {str(e)}"
    
    def _analyze_response_content(self, response: str) -> Dict[str, Any]:
        """Analyze response content to determine structure."""
        analysis = {
            'has_summary': False,
            'has_evidence': False,
            'has_analysis': False,
            'has_conclusions': False,
            'has_next_steps': False,
            'paragraphs': [],
            'key_phrases': [],
            'question_type': 'general'
        }
        
        response_lower = response.lower()
        
        # Check for existing structure indicators
        structure_indicators = {
            'has_summary': ['summary', 'overview', 'in brief', 'tldr'],
            'has_evidence': ['evidence', 'data', 'research', 'studies', 'findings'],
            'has_analysis': ['analysis', 'examination', 'evaluation', 'assessment'],
            'has_conclusions': ['conclusion', 'therefore', 'in summary', 'to conclude'],
            'has_next_steps': ['next steps', 'recommendations', 'action items', 'follow up']
        }
        
        for key, indicators in structure_indicators.items():
            analysis[key] = any(indicator in response_lower for indicator in indicators)
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        analysis['paragraphs'] = paragraphs
        
        # Extract key phrases (simplified)
        sentences = response.split('.')
        key_phrases = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and any(word in sentence.lower() for word in ['important', 'key', 'significant', 'crucial', 'essential']):
                key_phrases.append(sentence)
        
        analysis['key_phrases'] = key_phrases[:5]  # Top 5 key phrases
        
        return analysis
    
    def _create_sections(self, response: str, analysis: Dict[str, Any],
                        sources: List[Dict[str, Any]] = None,
                        tool_outputs: List[Dict[str, Any]] = None) -> List[StructuredSection]:
        """Create structured sections from response content."""
        sections = []
        paragraphs = analysis['paragraphs']
        
        if not paragraphs:
            # Fallback: treat entire response as one section
            sections.append(StructuredSection(
                section_type=SectionType.ANALYSIS,
                title="Analysis",
                content=response,
                confidence_level=ConfidenceLevel.MEDIUM,
                confidence_score=0.5,
                sources=[],
                metadata={}
            ))
            return sections
        
        # Create summary section if response is long enough
        if len(response) > 300:
            summary_content = self._extract_summary(response, analysis)
            if summary_content:
                sections.append(StructuredSection(
                    section_type=SectionType.SUMMARY,
                    title="Summary",
                    content=summary_content,
                    confidence_level=ConfidenceLevel.HIGH,
                    confidence_score=0.8,
                    sources=[],
                    metadata={}
                ))
        
        # Create evidence section if sources available
        if sources:
            evidence_content = self._create_evidence_section(sources)
            sections.append(StructuredSection(
                section_type=SectionType.EVIDENCE,
                title="Evidence",
                content=evidence_content,
                confidence_level=ConfidenceLevel.HIGH,
                confidence_score=0.9,
                sources=[s.get('title', 'Unknown') for s in sources],
                metadata={'source_count': len(sources)}
            ))
        
        # Create main analysis section
        analysis_content = self._extract_main_analysis(paragraphs, analysis)
        sections.append(StructuredSection(
            section_type=SectionType.ANALYSIS,
            title="Analysis",
            content=analysis_content,
            confidence_level=ConfidenceLevel.MEDIUM,
            confidence_score=0.7,
            sources=[],
            metadata={}
        ))
        
        # Create methodology section if tools were used
        if tool_outputs:
            methodology_content = self._create_methodology_section(tool_outputs)
            sections.append(StructuredSection(
                section_type=SectionType.METHODOLOGY,
                title="Methodology",
                content=methodology_content,
                confidence_level=ConfidenceLevel.HIGH,
                confidence_score=0.8,
                sources=[],
                metadata={'tools_used': [t.get('tool_name') for t in tool_outputs]}
            ))
        
        # Create conclusions section
        conclusions_content = self._extract_conclusions(paragraphs, analysis)
        if conclusions_content:
            sections.append(StructuredSection(
                section_type=SectionType.CONCLUSIONS,
                title="Conclusions",
                content=conclusions_content,
                confidence_level=ConfidenceLevel.MEDIUM,
                confidence_score=0.6,
                sources=[],
                metadata={}
            ))
        
        # Create next steps section if applicable
        next_steps_content = self._extract_next_steps(response, analysis)
        if next_steps_content:
            sections.append(StructuredSection(
                section_type=SectionType.NEXT_STEPS,
                title="Suggested Next Steps",
                content=next_steps_content,
                confidence_level=ConfidenceLevel.MEDIUM,
                confidence_score=0.5,
                sources=[],
                metadata={}
            ))
        
        return sections
    
    def _create_decision_tree(self, reasoning_trace: List[Dict[str, Any]]) -> List[DecisionNode]:
        """Create decision tree from reasoning trace."""
        decision_nodes = []
        
        for i, step in enumerate(reasoning_trace):
            node_id = f"node_{i+1}"
            
            # Extract decision information
            question = step.get('step', f"Step {i+1}")
            answer = step.get('reasoning', '')
            confidence = step.get('confidence', 0.5)
            
            # Determine alternatives considered
            alternatives = []
            if 'alternatives' in step:
                alternatives = step['alternatives']
            elif 'output_data' in step and isinstance(step['output_data'], dict):
                alternatives = step['output_data'].get('alternatives', [])
            
            decision_node = DecisionNode(
                node_id=node_id,
                question=question,
                answer=answer,
                confidence=confidence,
                reasoning=answer,
                children=[],  # Would need more complex logic to determine children
                alternatives_considered=alternatives,
                metadata=step.get('metadata', {})
            )
            
            decision_nodes.append(decision_node)
        
        return decision_nodes
    
    def _extract_summary(self, response: str, analysis: Dict[str, Any]) -> str:
        """Extract or create a summary from the response."""
        # Look for existing summary indicators
        paragraphs = analysis['paragraphs']
        
        for paragraph in paragraphs:
            if any(indicator in paragraph.lower() for indicator in ['summary', 'overview', 'in brief']):
                return paragraph
        
        # Create summary from first paragraph if it's substantial
        if paragraphs and len(paragraphs[0]) > 100:
            return paragraphs[0]
        
        # Create summary from key phrases
        if analysis['key_phrases']:
            return ' '.join(analysis['key_phrases'][:2])
        
        return ""
    
    def _create_evidence_section(self, sources: List[Dict[str, Any]]) -> str:
        """Create evidence section from sources."""
        evidence_parts = ["The following sources support this analysis:\n"]
        
        for i, source in enumerate(sources, 1):
            title = source.get('title', f'Source {i}')
            content = source.get('content', '')
            confidence = source.get('confidence', 0.5)
            
            # Extract key excerpt
            excerpt = content[:150] + "..." if len(content) > 150 else content
            
            evidence_parts.append(f"{i}. **{title}** (confidence: {confidence:.2f})")
            evidence_parts.append(f"   {excerpt}\n")
        
        return "\n".join(evidence_parts)
    
    def _extract_main_analysis(self, paragraphs: List[str], analysis: Dict[str, Any]) -> str:
        """Extract main analysis content."""
        # Skip summary paragraph if it exists
        start_index = 1 if len(paragraphs) > 1 and len(paragraphs[0]) < 200 else 0
        
        # Take middle paragraphs as main analysis
        end_index = len(paragraphs) - 1 if len(paragraphs) > 2 else len(paragraphs)
        
        analysis_paragraphs = paragraphs[start_index:end_index]
        
        if analysis_paragraphs:
            return "\n\n".join(analysis_paragraphs)
        else:
            return paragraphs[0] if paragraphs else "No analysis content available."
    
    def _create_methodology_section(self, tool_outputs: List[Dict[str, Any]]) -> str:
        """Create methodology section from tool outputs."""
        methodology_parts = ["The following tools and methods were used:\n"]
        
        for tool_output in tool_outputs:
            tool_name = tool_output.get('tool_name', 'Unknown Tool')
            success = tool_output.get('success', False)
            execution_time = tool_output.get('execution_time_ms', 0)
            
            status = "✅ Success" if success else "❌ Failed"
            methodology_parts.append(f"- **{tool_name}**: {status} ({execution_time}ms)")
        
        return "\n".join(methodology_parts)
    
    def _extract_conclusions(self, paragraphs: List[str], analysis: Dict[str, Any]) -> str:
        """Extract conclusions from the response."""
        # Look for conclusion indicators in last paragraphs
        for paragraph in reversed(paragraphs):
            if any(indicator in paragraph.lower() for indicator in ['conclusion', 'therefore', 'in summary']):
                return paragraph
        
        # Use last paragraph if it's substantial
        if paragraphs and len(paragraphs[-1]) > 50:
            return paragraphs[-1]
        
        return ""
    
    def _extract_next_steps(self, response: str, analysis: Dict[str, Any]) -> str:
        """Extract next steps or recommendations."""
        response_lower = response.lower()
        
        # Look for next steps indicators
        next_steps_indicators = ['next steps', 'recommendations', 'action items', 'follow up', 'consider', 'should']
        
        sentences = response.split('.')
        next_steps_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(indicator in sentence.lower() for indicator in next_steps_indicators):
                next_steps_sentences.append(sentence)
        
        if next_steps_sentences:
            return '. '.join(next_steps_sentences[:3]) + '.'
        
        return ""
    
    def _calculate_overall_confidence(self, sections: List[StructuredSection]) -> float:
        """Calculate overall confidence from section confidences."""
        if not sections:
            return 0.0
        
        total_confidence = sum(section.confidence_score for section in sections)
        return total_confidence / len(sections)
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what', 'define', 'explain']):
            return 'definitional'
        elif any(word in query_lower for word in ['how', 'steps', 'process']):
            return 'procedural'
        elif any(word in query_lower for word in ['why', 'because', 'reason']):
            return 'causal'
        elif any(word in query_lower for word in ['compare', 'difference', 'versus']):
            return 'comparative'
        elif any(word in query_lower for word in ['calculate', 'compute', 'solve']):
            return 'computational'
        else:
            return 'general'
    
    def _get_confidence_emoji(self, confidence_level: ConfidenceLevel) -> str:
        """Get emoji for confidence level."""
        emoji_map = {
            ConfidenceLevel.HIGH: "🟢",
            ConfidenceLevel.MEDIUM: "🟡",
            ConfidenceLevel.LOW: "🟠",
            ConfidenceLevel.UNCERTAIN: "🔴"
        }
        return emoji_map.get(confidence_level, "⚪")
    
    def _format_decision_tree(self, decision_tree: List[DecisionNode]) -> str:
        """Format decision tree as text."""
        tree_parts = []
        
        for i, node in enumerate(decision_tree, 1):
            tree_parts.append(f"{i}. **{node.question}**")
            tree_parts.append(f"   Answer: {node.answer}")
            tree_parts.append(f"   Confidence: {node.confidence:.2f}")
            
            if node.alternatives_considered:
                tree_parts.append(f"   Alternatives considered: {', '.join(node.alternatives_considered)}")
            
            tree_parts.append("")  # Empty line
        
        return "\n".join(tree_parts)
    
    def _create_fallback_structure(self, query: str, response: str) -> StructuredResponse:
        """Create minimal structure when full structuring fails."""
        import uuid
        
        fallback_section = StructuredSection(
            section_type=SectionType.ANALYSIS,
            title="Response",
            content=response,
            confidence_level=ConfidenceLevel.MEDIUM,
            confidence_score=0.5,
            sources=[],
            metadata={}
        )
        
        return StructuredResponse(
            response_id=f"resp_{uuid.uuid4().hex[:12]}",
            query=query,
            sections=[fallback_section],
            decision_tree=[],
            overall_confidence=0.5,
            response_metadata={'fallback': True},
            created_at=datetime.now().isoformat()
        )

# Global structurer instance
_structurer = None

def get_answer_structurer() -> AdvancedAnswerStructurer:
    """Get or create a global answer structurer instance."""
    global _structurer
    
    if _structurer is None:
        _structurer = AdvancedAnswerStructurer()
    
    return _structurer
