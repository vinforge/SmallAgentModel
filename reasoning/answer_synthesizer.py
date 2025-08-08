"""
Enhanced Answer Synthesis for SAM
Constructs final responses using retrieved data, tool outputs, and SELF-DECIDE reasoning logs.

Sprint 5 Task 4: Enhanced Answer Synthesis
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .self_decide_framework import SelfDecideSession, ReasoningStep
from .tool_executor import ToolResponse

logger = logging.getLogger(__name__)

@dataclass
class SourceAttribution:
    """Attribution information for response sources."""
    source_type: str  # 'knowledge_base', 'tool_execution', 'reasoning'
    source_name: str
    content_preview: str
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class SynthesizedResponse:
    """Complete synthesized response with annotations."""
    answer: str
    confidence_score: float
    source_attributions: List[SourceAttribution]
    tools_used: List[str]
    reasoning_trace: str
    metadata: Dict[str, Any]
    synthesis_timestamp: str

class AnswerSynthesizer:
    """
    Enhanced answer synthesizer that constructs final responses using all available information.
    """
    
    def __init__(self, model=None):
        """
        Initialize the answer synthesizer.
        
        Args:
            model: Language model for response generation
        """
        self.model = model
        logger.info("Enhanced answer synthesizer initialized")
    
    def synthesize_response(self, session: SelfDecideSession, 
                          tool_responses: Optional[List[ToolResponse]] = None) -> SynthesizedResponse:
        """
        Synthesize a comprehensive response from SELF-DECIDE session and tool outputs.
        
        Args:
            session: Complete SELF-DECIDE reasoning session
            tool_responses: Additional tool responses (if any)
            
        Returns:
            SynthesizedResponse with complete answer and annotations
        """
        try:
            logger.info(f"Synthesizing response for session: {session.session_id}")
            
            # Gather all information sources
            sources = self._gather_information_sources(session, tool_responses)
            
            # Create source attributions
            attributions = self._create_source_attributions(sources)
            
            # Generate enhanced answer
            answer = self._generate_enhanced_answer(session, sources)
            
            # Extract tools used
            tools_used = self._extract_tools_used(session, tool_responses)
            
            # Create reasoning trace
            reasoning_trace = self._create_reasoning_trace(session)
            
            # Calculate overall confidence
            confidence = self._calculate_synthesis_confidence(session, sources)
            
            # Create metadata
            metadata = self._create_synthesis_metadata(session, sources, tools_used)
            
            synthesized = SynthesizedResponse(
                answer=answer,
                confidence_score=confidence,
                source_attributions=attributions,
                tools_used=tools_used,
                reasoning_trace=reasoning_trace,
                metadata=metadata,
                synthesis_timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"Response synthesized: {len(answer)} chars, {len(tools_used)} tools, {confidence:.2f} confidence")
            return synthesized
            
        except Exception as e:
            logger.error(f"Error synthesizing response: {e}")
            
            # Create fallback response
            return SynthesizedResponse(
                answer=f"I apologize, but I encountered an error while synthesizing the response: {str(e)}",
                confidence_score=0.0,
                source_attributions=[],
                tools_used=[],
                reasoning_trace="Error in synthesis process",
                metadata={"error": str(e)},
                synthesis_timestamp=datetime.now().isoformat()
            )
    
    def format_response_for_chat(self, synthesized: SynthesizedResponse, 
                               show_sources: bool = True, show_reasoning: bool = False) -> str:
        """
        Format synthesized response for chat display.
        
        Args:
            synthesized: SynthesizedResponse object
            show_sources: Whether to show source attributions
            show_reasoning: Whether to show reasoning trace
            
        Returns:
            Formatted markdown response
        """
        response_parts = []
        
        # Main answer
        response_parts.append(synthesized.answer)
        
        # Tools used annotation
        if synthesized.tools_used:
            tools_list = ", ".join(f"🔧 {tool}" for tool in synthesized.tools_used)
            response_parts.append(f"\n**Tools Used:** {tools_list}")
        
        # Source attributions
        if show_sources and synthesized.source_attributions:
            response_parts.append("\n**📚 Sources:**")
            for i, attribution in enumerate(synthesized.source_attributions[:5], 1):
                source_icon = self._get_source_icon(attribution.source_type)
                confidence_bar = self._create_confidence_bar(attribution.confidence)
                response_parts.append(
                    f"{i}. {source_icon} **{attribution.source_name}** {confidence_bar}\n"
                    f"   _{attribution.content_preview[:100]}..._"
                )
        
        # Reasoning trace (expandable)
        if show_reasoning and synthesized.reasoning_trace:
            response_parts.append(f"\n<details>\n<summary>🤔 **Reasoning Trace**</summary>\n\n{synthesized.reasoning_trace}\n</details>")
        
        # Confidence indicator
        confidence_emoji = "🟢" if synthesized.confidence_score > 0.7 else "🟡" if synthesized.confidence_score > 0.4 else "🔴"
        response_parts.append(f"\n{confidence_emoji} **Confidence:** {synthesized.confidence_score:.1%}")
        
        return "\n".join(response_parts)
    
    def _gather_information_sources(self, session: SelfDecideSession, 
                                  tool_responses: Optional[List[ToolResponse]]) -> Dict[str, List[Any]]:
        """Gather all information sources from session and tool responses."""
        sources = {
            'knowledge_base': [],
            'tool_outputs': [],
            'reasoning_steps': []
        }
        
        # Extract knowledge base retrievals
        retrieval_step = next((s for s in session.reasoning_steps if s.step == ReasoningStep.EXPLORE_RETRIEVALS), None)
        if retrieval_step:
            retrievals = retrieval_step.output_data.get('retrievals', [])
            sources['knowledge_base'] = retrievals
        
        # Extract tool outputs from session
        if session.tool_executions:
            sources['tool_outputs'].extend(session.tool_executions)
        
        # Add additional tool responses
        if tool_responses:
            sources['tool_outputs'].extend([{
                'tool_name': tr.tool_name,
                'result': {'output': tr.output, 'success': tr.success},
                'success': tr.success,
                'metadata': tr.metadata
            } for tr in tool_responses])
        
        # Add reasoning steps as sources
        sources['reasoning_steps'] = session.reasoning_steps
        
        return sources
    
    def _create_source_attributions(self, sources: Dict[str, List[Any]]) -> List[SourceAttribution]:
        """Create source attribution objects."""
        attributions = []
        
        # Knowledge base sources
        for retrieval in sources['knowledge_base']:
            attribution = SourceAttribution(
                source_type='knowledge_base',
                source_name=retrieval.get('metadata', {}).get('source_file', 'Knowledge Base'),
                content_preview=retrieval.get('content', '')[:150],
                confidence=retrieval.get('similarity', 0.5),
                metadata=retrieval.get('metadata', {})
            )
            attributions.append(attribution)
        
        # Tool execution sources
        for tool_output in sources['tool_outputs']:
            if tool_output.get('success', False):
                tool_name = tool_output.get('tool_name', 'Unknown Tool')
                result = tool_output.get('result', {})
                output = result.get('output', '')
                
                attribution = SourceAttribution(
                    source_type='tool_execution',
                    source_name=tool_name.replace('_', ' ').title(),
                    content_preview=str(output)[:150] if output else 'Tool execution completed',
                    confidence=0.8,  # High confidence for successful tool execution
                    metadata=tool_output.get('metadata', {})
                )
                attributions.append(attribution)
        
        return attributions
    
    def _generate_enhanced_answer(self, session: SelfDecideSession, sources: Dict[str, List[Any]]) -> str:
        """Generate enhanced answer using all available information."""
        if self.model:
            return self._generate_llm_answer(session, sources)
        else:
            return self._generate_fallback_answer(session, sources)
    
    def _generate_llm_answer(self, session: SelfDecideSession, sources: Dict[str, List[Any]]) -> str:
        """Generate answer using language model."""
        prompt = self._create_synthesis_prompt(session, sources)
        
        try:
            answer = self.model.generate(prompt, temperature=0.4, max_tokens=1000)
            return self._clean_generated_answer(answer)
        except Exception as e:
            logger.error(f"LLM answer generation failed: {e}")
            return self._generate_fallback_answer(session, sources)
    
    def _create_synthesis_prompt(self, session: SelfDecideSession, sources: Dict[str, List[Any]]) -> str:
        """Create comprehensive synthesis prompt."""
        prompt_parts = [
            "You are SAM, an intelligent assistant using the SELF-DECIDE reasoning framework.",
            "Synthesize a comprehensive, well-structured answer using all available information.",
            "",
            f"**Original Query:** {session.original_query}",
            ""
        ]
        
        # Add knowledge base information
        if sources['knowledge_base']:
            prompt_parts.append("**Knowledge Base Information:**")
            for i, retrieval in enumerate(sources['knowledge_base'][:3], 1):
                content = retrieval.get('content', '')[:200]
                similarity = retrieval.get('similarity', 0)
                prompt_parts.append(f"{i}. (Relevance: {similarity:.2f}) {content}...")
            prompt_parts.append("")
        
        # Add tool execution results
        if sources['tool_outputs']:
            prompt_parts.append("**Tool Execution Results:**")
            for i, tool_output in enumerate(sources['tool_outputs'], 1):
                tool_name = tool_output.get('tool_name', 'Unknown')
                if tool_output.get('success', False):
                    result = tool_output.get('result', {})
                    output = result.get('output', '')
                    prompt_parts.append(f"{i}. **{tool_name}**: {str(output)[:300]}")
                else:
                    prompt_parts.append(f"{i}. **{tool_name}**: Execution failed")
            prompt_parts.append("")
        
        # Add reasoning context
        if session.knowledge_gaps:
            prompt_parts.append("**Identified Knowledge Gaps:**")
            for gap in session.knowledge_gaps[:3]:
                if hasattr(gap, 'description'):
                    prompt_parts.append(f"- {gap.description}")
                elif isinstance(gap, dict):
                    prompt_parts.append(f"- {gap.get('description', 'Unknown gap')}")
            prompt_parts.append("")
        
        # Add synthesis instructions
        prompt_parts.extend([
            "**Synthesis Instructions:**",
            "1. Provide a comprehensive answer that directly addresses the query",
            "2. Integrate information from knowledge base and tool results seamlessly",
            "3. Use clear, well-structured markdown formatting",
            "4. Be specific and detailed while remaining concise",
            "5. If information is incomplete or uncertain, acknowledge limitations clearly",
            "6. Highlight key insights and actionable information",
            "",
            "**Synthesized Answer:**"
        ])
        
        return "\n".join(prompt_parts)
    
    def _generate_fallback_answer(self, session: SelfDecideSession, sources: Dict[str, List[Any]]) -> str:
        """Generate fallback answer when LLM is not available."""
        answer_parts = [
            f"Based on my analysis of your query: **{session.original_query}**",
            ""
        ]
        
        # Add knowledge base information
        if sources['knowledge_base']:
            answer_parts.append("**From Knowledge Base:**")
            for retrieval in sources['knowledge_base'][:2]:
                content = retrieval.get('content', '')[:200]
                answer_parts.append(f"- {content}...")
            answer_parts.append("")
        
        # Add tool results
        if sources['tool_outputs']:
            answer_parts.append("**Tool Analysis Results:**")
            for tool_output in sources['tool_outputs']:
                if tool_output.get('success', False):
                    tool_name = tool_output.get('tool_name', 'Unknown')
                    result = tool_output.get('result', {})
                    output = result.get('output', '')
                    answer_parts.append(f"- **{tool_name}**: {str(output)[:150]}...")
            answer_parts.append("")
        
        # Add reasoning summary
        if session.reasoning_plan:
            plan = session.reasoning_plan
            answer_parts.append(f"**Approach Used:** {plan.approach}")
            answer_parts.append("")
        
        # Add limitations if any
        if not sources['knowledge_base'] and not sources['tool_outputs']:
            answer_parts.append("**Note:** Limited information available for this query. Consider providing more context or specific details.")
        
        return "\n".join(answer_parts)
    
    def _clean_generated_answer(self, answer: str) -> str:
        """Clean up generated answer."""
        # Remove thinking tags
        if '<think>' in answer:
            parts = answer.split('</think>')
            if len(parts) > 1:
                answer = parts[-1].strip()
        
        # Remove leading labels
        for label in ["Answer:", "Response:", "Synthesized Answer:"]:
            if answer.startswith(label):
                answer = answer[len(label):].strip()
        
        return answer.strip()
    
    def _extract_tools_used(self, session: SelfDecideSession, 
                          tool_responses: Optional[List[ToolResponse]]) -> List[str]:
        """Extract list of tools used."""
        tools = set()
        
        # From session tool executions
        for tool_exec in session.tool_executions:
            if tool_exec.get('success', False):
                tools.add(tool_exec.get('tool_name', 'Unknown'))
        
        # From additional tool responses
        if tool_responses:
            for tr in tool_responses:
                if tr.success:
                    tools.add(tr.tool_name)
        
        return sorted(list(tools))
    
    def _create_reasoning_trace(self, session: SelfDecideSession) -> str:
        """Create condensed reasoning trace for display."""
        trace_parts = [
            f"**SELF-DECIDE Reasoning Process** (Session: {session.session_id})",
            ""
        ]
        
        # Add key reasoning steps
        key_steps = [
            ReasoningStep.STATE_QUERY,
            ReasoningStep.LABEL_GAPS,
            ReasoningStep.FORMULATE_PLAN,
            ReasoningStep.EXECUTE_TOOLS,
            ReasoningStep.INFER_ANSWER
        ]
        
        for step_enum in key_steps:
            step_result = next((s for s in session.reasoning_steps if s.step == step_enum), None)
            if step_result:
                step_name = step_enum.value.replace('_', ' ').title()
                trace_parts.append(f"**{step_name}:** {step_result.reasoning}")
        
        return "\n".join(trace_parts)
    
    def _calculate_synthesis_confidence(self, session: SelfDecideSession, 
                                      sources: Dict[str, List[Any]]) -> float:
        """Calculate overall confidence for synthesized response."""
        confidence = session.confidence_score  # Base confidence from session
        
        # Boost confidence based on successful tool executions
        successful_tools = sum(1 for tool in sources['tool_outputs'] if tool.get('success', False))
        if successful_tools > 0:
            confidence += min(successful_tools * 0.1, 0.2)
        
        # Boost confidence based on knowledge base quality
        if sources['knowledge_base']:
            avg_similarity = sum(r.get('similarity', 0) for r in sources['knowledge_base']) / len(sources['knowledge_base'])
            confidence += avg_similarity * 0.1
        
        return min(confidence, 1.0)
    
    def _create_synthesis_metadata(self, session: SelfDecideSession, sources: Dict[str, List[Any]], 
                                 tools_used: List[str]) -> Dict[str, Any]:
        """Create metadata for synthesized response."""
        return {
            'session_id': session.session_id,
            'reasoning_steps_count': len(session.reasoning_steps),
            'knowledge_sources_count': len(sources['knowledge_base']),
            'tool_executions_count': len(sources['tool_outputs']),
            'tools_used': tools_used,
            'synthesis_method': 'llm' if self.model else 'fallback',
            'total_session_duration_ms': session.total_duration_ms
        }
    
    def _get_source_icon(self, source_type: str) -> str:
        """Get icon for source type."""
        icons = {
            'knowledge_base': '📚',
            'tool_execution': '🔧',
            'reasoning': '🤔'
        }
        return icons.get(source_type, '📄')
    
    def _create_confidence_bar(self, confidence: float) -> str:
        """Create visual confidence bar."""
        filled = int(confidence * 5)
        empty = 5 - filled
        return '●' * filled + '○' * empty + f' ({confidence:.1%})'

# Global synthesizer instance
_answer_synthesizer = None

def get_answer_synthesizer(model=None) -> AnswerSynthesizer:
    """Get or create a global answer synthesizer instance."""
    global _answer_synthesizer
    
    if _answer_synthesizer is None:
        _answer_synthesizer = AnswerSynthesizer(model=model)
    
    return _answer_synthesizer
