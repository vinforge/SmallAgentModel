"""
Heuristic Estimator Module

Implements LLM-based heuristic estimation for A* search planning.
Uses SAM's core LLM to estimate the cost-to-go from any planning state
to the goal, enabling optimal path finding in the search space.
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from .state import PlanningState
from .sam_context_manager import SAMContextManager

logger = logging.getLogger(__name__)


@dataclass
class HeuristicEstimate:
    """Result of heuristic estimation."""
    
    estimated_cost: int
    """Estimated cost to reach goal from this state"""
    
    confidence: float
    """Confidence in the estimate (0.0 to 1.0)"""
    
    reasoning: str
    """LLM's reasoning for the estimate"""
    
    context_factors: List[str]
    """Factors that influenced the estimate"""
    
    fallback_used: bool = False
    """Whether fallback estimation was used"""


class HeuristicEstimator:
    """
    LLM-based heuristic estimator for A* search planning.
    
    This class uses SAM's core LLM to estimate the cost-to-go from any
    planning state to the goal state. It includes SAM-specific context
    awareness and optimization strategies.
    """
    
    def __init__(self,
                 llm_interface=None,
                 context_manager: Optional[SAMContextManager] = None,
                 max_cost: int = 100,
                 use_caching: bool = True,
                 enforce_order: bool = False,
                 scaffold_weight: float = 0.5):
        """
        Initialize the heuristic estimator.

        Args:
            llm_interface: SAM's LLM interface for generating estimates
            context_manager: SAM context manager for enhanced estimates
            max_cost: Maximum cost to return (for failed estimates)
            use_caching: Whether to cache estimates for similar states
            enforce_order: If True, penalize states that deviate from next expected procedure step
            scaffold_weight: 0..1 blend toward remaining procedure steps as target cost
        """
        self.llm_interface = llm_interface
        self.context_manager = context_manager
        self.max_cost = max_cost
        self.use_caching = use_caching
        self.enforce_order = enforce_order
        self.scaffold_weight = max(0.0, min(1.0, scaffold_weight))

        # Estimation cache for performance
        self._estimate_cache: Dict[str, HeuristicEstimate] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        logger.info("Initialized HeuristicEstimator")
    
    def estimate_cost_to_go(self, state: PlanningState) -> int:
        """
        Estimate the cost to reach the goal from the given state.
        
        Args:
            state: Planning state to estimate cost for
            
        Returns:
            Estimated cost as integer (lower is better)
        """
        try:
            # Get detailed estimate
            estimate = self.get_detailed_estimate(state)
            return estimate.estimated_cost
            
        except Exception as e:
            logger.error(f"Error in cost estimation: {e}")
            # Return high cost for failed estimates
            return self.max_cost
    
    def get_detailed_estimate(self, state: PlanningState) -> HeuristicEstimate:
        """
        Get detailed heuristic estimate with reasoning and confidence.
        
        Args:
            state: Planning state to estimate cost for
            
        Returns:
            HeuristicEstimate with detailed information
        """
        # Check cache first
        if self.use_caching:
            cached_estimate = self._get_cached_estimate(state)
            if cached_estimate:
                self._cache_hits += 1
                return cached_estimate
            self._cache_misses += 1
        
        # Generate new estimate
        estimate = self._generate_llm_estimate(state)
        
        # Cache the result
        if self.use_caching:
            self._cache_estimate(state, estimate)
        
        return estimate
    
    def _generate_llm_estimate(self, state: PlanningState) -> HeuristicEstimate:
        """Generate heuristic estimate using LLM, blended with procedure scaffold if available."""

        # Baseline estimate
        base = None
        if self.llm_interface:
            try:
                prompt = self._build_heuristic_prompt(state)
                response = self._call_llm(prompt)
                base = self._parse_llm_response(response, state)
            except Exception as e:
                logger.warning(f"LLM estimation failed: {e}")
                base = self._fallback_estimate(state)
        else:
            base = self._fallback_estimate(state)

        # Blend with scaffold if available
        try:
            scaffold_cost_adj, context_factors = self._scaffold_adjustment(state)
            if scaffold_cost_adj is not None:
                blended = int(round((1.0 - self.scaffold_weight) * base.estimated_cost + self.scaffold_weight * scaffold_cost_adj))
                return HeuristicEstimate(
                    estimated_cost=blended,
                    confidence=min(1.0, base.confidence + 0.1),
                    reasoning=base.reasoning + " | scaffold_blend",
                    context_factors=base.context_factors + context_factors,
                    fallback_used=base.fallback_used,
                )
            return base
        except Exception:
            return base
    
    def _build_heuristic_prompt(self, state: PlanningState) -> str:
        """Build context-aware prompt for heuristic estimation."""
        
        # Base prompt structure
        prompt_parts = [
            "You are an expert planning assistant. Estimate the number of steps needed to complete a task.",
            "",
            f"TASK: {state.task_description}",
            "",
            "CURRENT STATE:",
            f"- Actions completed: {len(state.action_history)}",
            f"- Action history: {state.action_history if state.action_history else 'None'}",
            f"- Current observation: {state.current_observation}",
            ""
        ]
        
        # Add SAM-specific context
        if self.context_manager:
            context_info = self._get_context_info(state)
            if context_info:
                prompt_parts.extend([
                    "AVAILABLE CONTEXT:",
                    context_info,
                    ""
                ])
        
        # Add estimation instructions
        prompt_parts.extend([
            "ESTIMATION TASK:",
            "Estimate how many more steps are needed to complete the task.",
            "Consider:",
            "- Complexity of remaining work",
            "- Available resources and context",
            "- Typical patterns for similar tasks",
            "",
            "Respond with ONLY a number between 0 and 50.",
            "If the task appears complete, respond with 0.",
            "If the task seems very complex or unclear, respond with a higher number.",
            "",
            "ESTIMATE:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _get_context_info(self, state: PlanningState) -> str:
        """Get context information for the prompt, including procedure hints."""
        context_parts = []

        # Document context
        if state.has_document_context():
            doc_count = len(state.document_context)
            context_parts.append(f"- Documents available: {doc_count}")

        # Memory context
        if state.has_memory_context():
            memory_count = len(state.memory_context)
            context_parts.append(f"- Relevant memories: {memory_count}")

        # Conversation context
        if state.has_conversation_context():
            context_parts.append("- Conversation context available")

        # SAM context manager info
        if self.context_manager:
            summary = self.context_manager.get_context_summary()
            if summary['documents']['count'] > 0:
                context_parts.append(f"- SAM documents: {summary['documents']['count']}")
            if summary['memory']['relevant_memories_count'] > 0:
                context_parts.append(f"- SAM memories: {summary['memory']['relevant_memories_count']}")
            # Procedure guidance hint
            pc = self.context_manager.get_planning_context()
            if pc.get('procedural_guidance'):
                try:
                    name = pc['procedural_guidance'].get('procedure', {}).get('name', 'procedure')
                    context_parts.append(f"- Procedure guidance available: {name}")
                except Exception:
                    context_parts.append("- Procedure guidance available")

        return "\n".join(context_parts) if context_parts else "- No additional context available"
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM interface with the prompt."""
        # This would integrate with SAM's actual LLM interface
        # For now, we'll simulate the call
        
        if hasattr(self.llm_interface, 'generate'):
            return self.llm_interface.generate(prompt, temperature=0.3, max_tokens=50)
        elif hasattr(self.llm_interface, 'generate_response'):
            return self.llm_interface.generate_response(prompt)
        elif callable(self.llm_interface):
            return self.llm_interface(prompt)
        else:
            raise ValueError("Invalid LLM interface provided")
    
    def _parse_llm_response(self, response: str, state: PlanningState) -> HeuristicEstimate:
        """Parse LLM response to extract cost estimate and blend with scaffold if set at call site."""
        import re
        numbers = re.findall(r'\b\d+\b', response.strip())
        if numbers:
            try:
                estimated_cost = int(numbers[0])
                estimated_cost = max(0, min(estimated_cost, self.max_cost))
                confidence = self._assess_response_confidence(response)
                reasoning = response.strip()
                context_factors = self._identify_context_factors(state)
                return HeuristicEstimate(
                    estimated_cost=estimated_cost,
                    confidence=confidence,
                    reasoning=reasoning,
                    context_factors=context_factors,
                    fallback_used=False
                )
            except ValueError:
                pass
        logger.warning(f"Failed to parse LLM response: {response}")
        return self._fallback_estimate(state)
    
    def _assess_response_confidence(self, response: str) -> float:
        """Assess confidence in the LLM response."""
        response_lower = response.lower()
        
        # High confidence indicators
        if any(word in response_lower for word in ['certain', 'confident', 'clear', 'obvious']):
            return 0.9
        
        # Low confidence indicators
        if any(word in response_lower for word in ['uncertain', 'unclear', 'maybe', 'possibly']):
            return 0.4
        
        # Medium confidence by default
        return 0.7
    
    def _identify_context_factors(self, state: PlanningState) -> List[str]:
        """Identify factors that influenced the estimate."""
        factors = []
        
        if state.action_history:
            factors.append(f"action_history_length_{len(state.action_history)}")
        
        if state.has_document_context():
            factors.append("document_context_available")
        
        if state.has_memory_context():
            factors.append("memory_context_available")
        
        if state.has_conversation_context():
            factors.append("conversation_context_available")
        
        return factors
    
    def _fallback_estimate(self, state: PlanningState) -> HeuristicEstimate:
        """Generate fallback estimate when LLM is unavailable, with scaffold blending."""

        # Simple heuristic based on task complexity and progress
        base_cost = 10
        progress_factor = max(0, base_cost - len(state.action_history))
        task_words = len(state.task_description.split())
        complexity_factor = min(5, task_words // 5)
        context_factor = 0
        if state.has_document_context():
            context_factor -= 2
        if state.has_memory_context():
            context_factor -= 1
        if state.has_conversation_context():
            context_factor -= 1
        est = max(0, progress_factor + complexity_factor + context_factor)

        # Try scaffold adjustment
        try:
            scaffold_cost_adj, context_factors = self._scaffold_adjustment(state)
            if scaffold_cost_adj is not None:
                est = int(round((1.0 - self.scaffold_weight) * est + self.scaffold_weight * scaffold_cost_adj))
                cf = self._identify_context_factors(state) + context_factors
            else:
                cf = self._identify_context_factors(state)
        except Exception:
            cf = self._identify_context_factors(state)

        return HeuristicEstimate(
            estimated_cost=est,
            confidence=0.5,
            reasoning="Fallback estimate with optional scaffold",
            context_factors=cf,
            fallback_used=True
        )
    
    def _get_cached_estimate(self, state: PlanningState) -> Optional[HeuristicEstimate]:
        """Get cached estimate for similar state."""
        # Simple cache key based on state signature
        cache_key = f"{state.task_description}_{len(state.action_history)}_{state.current_observation}"
        return self._estimate_cache.get(cache_key)
    
    def _cache_estimate(self, state: PlanningState, estimate: HeuristicEstimate):
        """Cache estimate for future use."""
        cache_key = f"{state.task_description}_{len(state.action_history)}_{state.current_observation}"
        self._estimate_cache[cache_key] = estimate
        
        # Limit cache size
        if len(self._estimate_cache) > 1000:
            # Remove oldest entries (simple FIFO)
            oldest_keys = list(self._estimate_cache.keys())[:100]
            for key in oldest_keys:
                del self._estimate_cache[key]
    
    def get_estimation_stats(self) -> Dict[str, Any]:
        """Get statistics about estimation performance."""
        total_requests = self._cache_hits + self._cache_misses
        cache_hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self._estimate_cache),
            'max_cost': self.max_cost,
            'caching_enabled': self.use_caching
        }
    
    def clear_cache(self):
        """Clear the estimation cache."""
        self._estimate_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Cleared heuristic estimation cache")

    def _scaffold_adjustment(self, state: PlanningState):
        """
        Compute a scaffold-adjusted target cost and context factors based on procedure guidance.
        If enforce_order is enabled and current action history deviates strongly from expected next step,
        increase the estimated cost; otherwise, use remaining steps as a target cost.
        Returns (adjusted_cost: Optional[int], context_factors: List[str])
        """
        if not self.context_manager:
            return None, []
        pc = self.context_manager.get_planning_context()
        guidance = pc.get('procedural_guidance')
        if not guidance or not isinstance(guidance, dict):
            return None, []
        proc = guidance.get('procedure') or {}
        steps = proc.get('steps') or []
        # Determine how many steps are already represented in action_history (by tool name)
        used_tools = set(state.action_history or [])
        remaining = 0
        next_tool = None
        for s in steps:
            t = s.get('tool')
            if not t:
                continue
            if t not in used_tools:
                remaining += 1
                if not next_tool:
                    next_tool = t
        if remaining == 0:
            return 0, ["procedure_completed"]
        # Base adjusted target: remaining steps
        adjusted = remaining
        factors = [f"procedure_remaining_{remaining}"]
        # Enforce order: if next expected tool differs from last suggested/used tool, add penalty
        if self.enforce_order and next_tool:
            if state.action_history and state.action_history[-1] != next_tool:
                adjusted += 1
                factors.append("order_penalty")
        return adjusted, factors
