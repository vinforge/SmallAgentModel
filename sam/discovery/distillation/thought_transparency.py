"""
Thought Transparency System
==========================

Displays active cognitive principles and reasoning traces for enhanced explainability.

Author: SAM Development Team
Version: 1.0.0
"""

import json
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from .registry import CognitivePrinciple
from .prompt_augmentation import AugmentedPrompt

logger = logging.getLogger(__name__)

@dataclass
class ReasoningTrace:
    """Represents a reasoning trace with applied principles."""
    trace_id: str
    query: str
    applied_principles: List[CognitivePrinciple]
    reasoning_steps: List[Dict[str, Any]]
    confidence_boost: float
    final_response: str
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class ThoughtTransparencyData:
    """Data structure for thought transparency display."""
    active_principles: List[Dict[str, Any]]
    reasoning_trace: Optional[ReasoningTrace]
    meta_cognition: Dict[str, Any]
    principle_impact: Dict[str, Any]

class ThoughtTransparency:
    """Manages thought transparency and meta-cognition display."""
    
    def __init__(self):
        """Initialize the thought transparency system."""
        self.max_trace_history = 10
        self.reasoning_traces = []
        
        logger.info("Thought transparency system initialized")
    
    def create_transparency_data(self, augmented_prompt: AugmentedPrompt,
                               response: str = None,
                               reasoning_steps: List[Dict[str, Any]] = None) -> ThoughtTransparencyData:
        """
        Create transparency data for display in SAM's interface.
        
        Args:
            augmented_prompt: The augmented prompt with applied principles
            response: The final response (if available)
            reasoning_steps: Detailed reasoning steps (if available)
            
        Returns:
            ThoughtTransparencyData for UI display
        """
        try:
            # Prepare active principles for display
            active_principles = self._format_active_principles(augmented_prompt.applied_principles)
            
            # Create reasoning trace if we have a response
            reasoning_trace = None
            if response:
                reasoning_trace = self._create_reasoning_trace(
                    augmented_prompt, response, reasoning_steps
                )
                self._store_reasoning_trace(reasoning_trace)
            
            # Generate meta-cognition insights
            meta_cognition = self._generate_meta_cognition(augmented_prompt)
            
            # Calculate principle impact
            principle_impact = self._calculate_principle_impact(augmented_prompt)
            
            return ThoughtTransparencyData(
                active_principles=active_principles,
                reasoning_trace=reasoning_trace,
                meta_cognition=meta_cognition,
                principle_impact=principle_impact
            )
            
        except Exception as e:
            logger.error(f"Failed to create transparency data: {e}")
            return ThoughtTransparencyData(
                active_principles=[],
                reasoning_trace=None,
                meta_cognition={'error': str(e)},
                principle_impact={}
            )
    
    def _format_active_principles(self, principles: List[CognitivePrinciple]) -> List[Dict[str, Any]]:
        """Format principles for UI display."""
        formatted_principles = []
        
        for principle in principles:
            formatted_principle = {
                'id': principle.principle_id,
                'text': principle.principle_text,
                'confidence': round(principle.confidence_score, 2),
                'domains': principle.domain_tags,
                'usage_count': principle.usage_count,
                'success_rate': round(principle.success_rate, 2),
                'source_strategy': principle.source_strategy_id,
                'discovered_date': principle.date_discovered.strftime('%Y-%m-%d'),
                'display_priority': self._calculate_display_priority(principle)
            }
            formatted_principles.append(formatted_principle)
        
        # Sort by display priority
        formatted_principles.sort(key=lambda x: x['display_priority'], reverse=True)
        
        return formatted_principles
    
    def _calculate_display_priority(self, principle: CognitivePrinciple) -> float:
        """Calculate display priority for a principle."""
        # Combine confidence, usage success, and recency
        confidence_factor = principle.confidence_score
        usage_factor = min(1.0, principle.usage_count / 10.0)  # Normalize usage count
        success_factor = principle.success_rate
        
        # Recency factor (more recent = higher priority)
        days_since_discovery = (datetime.now() - principle.date_discovered).days
        recency_factor = max(0.1, 1.0 - (days_since_discovery / 365.0))  # Decay over a year
        
        priority = (confidence_factor * 0.4 + 
                   usage_factor * 0.2 + 
                   success_factor * 0.3 + 
                   recency_factor * 0.1)
        
        return round(priority, 3)
    
    def _create_reasoning_trace(self, augmented_prompt: AugmentedPrompt,
                              response: str, reasoning_steps: List[Dict[str, Any]] = None) -> ReasoningTrace:
        """Create a reasoning trace for the current interaction."""
        import uuid
        
        # Extract query from original prompt
        query = self._extract_query_from_prompt(augmented_prompt.original_prompt)
        
        # Create default reasoning steps if none provided
        if not reasoning_steps:
            reasoning_steps = self._generate_default_reasoning_steps(augmented_prompt, response)
        
        trace = ReasoningTrace(
            trace_id=str(uuid.uuid4()),
            query=query,
            applied_principles=augmented_prompt.applied_principles,
            reasoning_steps=reasoning_steps,
            confidence_boost=augmented_prompt.confidence_boost,
            final_response=response,
            timestamp=datetime.now(),
            metadata=augmented_prompt.augmentation_metadata
        )
        
        return trace
    
    def _extract_query_from_prompt(self, prompt: str) -> str:
        """Extract the main query from a prompt."""
        # Simple extraction - look for question patterns
        lines = prompt.split('\n')
        
        # Look for lines that seem like questions
        for line in lines:
            line = line.strip()
            if (line.endswith('?') or 
                line.startswith(('What', 'How', 'Why', 'When', 'Where', 'Who', 'Should', 'Can', 'Is', 'Are'))):
                return line
        
        # Fallback: return first non-empty line or truncated prompt
        for line in lines:
            if line.strip():
                return line.strip()[:100] + ('...' if len(line) > 100 else '')
        
        return prompt[:100] + ('...' if len(prompt) > 100 else '')
    
    def _generate_default_reasoning_steps(self, augmented_prompt: AugmentedPrompt, 
                                        response: str) -> List[Dict[str, Any]]:
        """Generate default reasoning steps when detailed steps aren't available."""
        steps = []
        
        # Step 1: Principle application
        if augmented_prompt.applied_principles:
            principle_texts = [p.principle_text for p in augmented_prompt.applied_principles]
            steps.append({
                'step': 1,
                'type': 'principle_application',
                'description': 'Applied cognitive principles to guide reasoning',
                'details': {
                    'principles_applied': principle_texts,
                    'confidence_boost': augmented_prompt.confidence_boost
                }
            })
        
        # Step 2: Query analysis
        steps.append({
            'step': len(steps) + 1,
            'type': 'query_analysis',
            'description': 'Analyzed query requirements and context',
            'details': {
                'domain_info': augmented_prompt.augmentation_metadata.get('domain_info', {}),
                'query_complexity': 'inferred from prompt structure'
            }
        })
        
        # Step 3: Response generation
        steps.append({
            'step': len(steps) + 1,
            'type': 'response_generation',
            'description': 'Generated response following applied principles',
            'details': {
                'response_length': len(response),
                'principle_influence': 'principles guided response structure and content'
            }
        })
        
        return steps
    
    def _store_reasoning_trace(self, trace: ReasoningTrace):
        """Store reasoning trace in memory for later analysis."""
        self.reasoning_traces.append(trace)
        
        # Keep only recent traces
        if len(self.reasoning_traces) > self.max_trace_history:
            self.reasoning_traces = self.reasoning_traces[-self.max_trace_history:]
        
        logger.info(f"Stored reasoning trace: {trace.trace_id}")
    
    def _generate_meta_cognition(self, augmented_prompt: AugmentedPrompt) -> Dict[str, Any]:
        """Generate meta-cognitive insights about the reasoning process."""
        meta_cognition = {
            'reasoning_approach': self._describe_reasoning_approach(augmented_prompt),
            'principle_selection': self._describe_principle_selection(augmented_prompt),
            'confidence_assessment': self._describe_confidence_assessment(augmented_prompt),
            'potential_improvements': self._suggest_improvements(augmented_prompt)
        }
        
        return meta_cognition
    
    def _describe_reasoning_approach(self, augmented_prompt: AugmentedPrompt) -> str:
        """Describe the reasoning approach taken."""
        if not augmented_prompt.applied_principles:
            return "Standard reasoning without specific cognitive principles"
        
        domain_info = augmented_prompt.augmentation_metadata.get('domain_info', {})
        domains = domain_info.get('domains', [])
        query_type = domain_info.get('query_type', 'general')
        
        if domains:
            domain_text = ', '.join(domains)
            return f"Domain-specific reasoning for {domain_text} using {len(augmented_prompt.applied_principles)} cognitive principles"
        else:
            return f"General reasoning enhanced with {len(augmented_prompt.applied_principles)} cognitive principles"
    
    def _describe_principle_selection(self, augmented_prompt: AugmentedPrompt) -> str:
        """Describe how principles were selected."""
        if not augmented_prompt.applied_principles:
            return "No principles were applicable to this query"
        
        avg_confidence = sum(p.confidence_score for p in augmented_prompt.applied_principles) / len(augmented_prompt.applied_principles)
        
        return f"Selected {len(augmented_prompt.applied_principles)} principles with average confidence {avg_confidence:.2f} based on domain relevance and query type"
    
    def _describe_confidence_assessment(self, augmented_prompt: AugmentedPrompt) -> str:
        """Describe the confidence assessment."""
        boost = augmented_prompt.confidence_boost
        
        if boost > 0.15:
            return f"High confidence boost (+{boost:.1%}) from well-established principles"
        elif boost > 0.05:
            return f"Moderate confidence boost (+{boost:.1%}) from relevant principles"
        elif boost > 0:
            return f"Minor confidence boost (+{boost:.1%}) from applicable principles"
        else:
            return "No confidence boost - reasoning without established principles"
    
    def _suggest_improvements(self, augmented_prompt: AugmentedPrompt) -> List[str]:
        """Suggest potential improvements to the reasoning process."""
        suggestions = []
        
        if not augmented_prompt.applied_principles:
            suggestions.append("Consider developing principles for this type of query")
        
        if augmented_prompt.confidence_boost < 0.05:
            suggestions.append("Low principle relevance - may need more specific principles")
        
        domain_info = augmented_prompt.augmentation_metadata.get('domain_info', {})
        if not domain_info.get('domains'):
            suggestions.append("Query domain unclear - better domain detection could improve principle selection")
        
        if len(augmented_prompt.applied_principles) == 1:
            suggestions.append("Single principle applied - additional complementary principles might enhance reasoning")
        
        return suggestions
    
    def _calculate_principle_impact(self, augmented_prompt: AugmentedPrompt) -> Dict[str, Any]:
        """Calculate the impact of applied principles."""
        if not augmented_prompt.applied_principles:
            return {
                'total_impact': 0.0,
                'principle_contributions': [],
                'impact_summary': 'No principles applied'
            }
        
        principle_contributions = []
        total_confidence = sum(p.confidence_score for p in augmented_prompt.applied_principles)
        
        for principle in augmented_prompt.applied_principles:
            contribution = {
                'principle_id': principle.principle_id,
                'principle_text': principle.principle_text,
                'individual_confidence': principle.confidence_score,
                'relative_contribution': principle.confidence_score / total_confidence if total_confidence > 0 else 0,
                'usage_history': {
                    'usage_count': principle.usage_count,
                    'success_rate': principle.success_rate
                }
            }
            principle_contributions.append(contribution)
        
        impact_summary = f"{len(augmented_prompt.applied_principles)} principles contributing {augmented_prompt.confidence_boost:.1%} confidence boost"
        
        return {
            'total_impact': augmented_prompt.confidence_boost,
            'principle_contributions': principle_contributions,
            'impact_summary': impact_summary
        }
    
    def get_recent_traces(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent reasoning traces for analysis."""
        recent_traces = self.reasoning_traces[-limit:] if self.reasoning_traces else []
        
        # Convert to serializable format
        serializable_traces = []
        for trace in recent_traces:
            trace_dict = asdict(trace)
            trace_dict['timestamp'] = trace.timestamp.isoformat()
            # Convert principles to dict format
            trace_dict['applied_principles'] = [
                {
                    'id': p.principle_id,
                    'text': p.principle_text,
                    'confidence': p.confidence_score
                }
                for p in trace.applied_principles
            ]
            serializable_traces.append(trace_dict)
        
        return serializable_traces
    
    def get_transparency_stats(self) -> Dict[str, Any]:
        """Get statistics about thought transparency usage."""
        if not self.reasoning_traces:
            return {
                'total_traces': 0,
                'avg_principles_per_trace': 0,
                'avg_confidence_boost': 0
            }
        
        total_traces = len(self.reasoning_traces)
        total_principles = sum(len(trace.applied_principles) for trace in self.reasoning_traces)
        total_confidence_boost = sum(trace.confidence_boost for trace in self.reasoning_traces)
        
        return {
            'total_traces': total_traces,
            'avg_principles_per_trace': round(total_principles / total_traces, 1),
            'avg_confidence_boost': round(total_confidence_boost / total_traces, 3),
            'max_trace_history': self.max_trace_history
        }

# Global thought transparency instance
thought_transparency = ThoughtTransparency()
