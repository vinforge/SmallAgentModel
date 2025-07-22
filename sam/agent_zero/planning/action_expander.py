"""
Action Expander Module

Implements LLM-based action expansion for A* search planning.
Uses SAM's core LLM and tool registry to generate possible next actions
from any planning state, enabling exploration of the search space.
"""

import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from .state import PlanningState
from .sam_tool_registry import SAMToolRegistry, get_sam_tool_registry
from .sam_context_manager import SAMContextManager

logger = logging.getLogger(__name__)


@dataclass
class ActionCandidate:
    """Represents a candidate action that can be taken from a state."""
    
    action_name: str
    """Name of the action/tool to execute"""
    
    parameters: Dict[str, Any]
    """Parameters for the action"""
    
    description: str
    """Human-readable description of what this action does"""
    
    estimated_cost: int
    """Estimated cost to execute this action"""
    
    confidence: float
    """Confidence that this action is relevant (0.0 to 1.0)"""
    
    prerequisites_met: bool
    """Whether prerequisites for this action are satisfied"""
    
    reasoning: str
    """LLM's reasoning for suggesting this action"""


class ActionExpander:
    """
    LLM-based action expander for A* search planning.
    
    This class uses SAM's core LLM and tool registry to generate
    possible next actions from any planning state. It includes
    SAM-specific context awareness and tool integration.
    """
    
    def __init__(self,
                 llm_interface=None,
                 tool_registry: Optional[SAMToolRegistry] = None,
                 context_manager: Optional[SAMContextManager] = None,
                 max_actions: int = 5):
        """
        Initialize the action expander.
        
        Args:
            llm_interface: SAM's LLM interface for generating actions
            tool_registry: SAM tool registry for available actions
            context_manager: SAM context manager for context-aware expansion
            max_actions: Maximum number of actions to generate per state
        """
        self.llm_interface = llm_interface
        self.tool_registry = tool_registry or get_sam_tool_registry()
        self.context_manager = context_manager
        self.max_actions = max_actions
        
        # Action generation cache
        self._action_cache: Dict[str, List[ActionCandidate]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info("Initialized ActionExpander")
    
    def get_next_possible_actions(self, state: PlanningState) -> List[str]:
        """
        Get list of possible next actions from the given state.
        
        Args:
            state: Planning state to expand from
            
        Returns:
            List of action names that can be executed
        """
        candidates = self.get_action_candidates(state)
        return [candidate.action_name for candidate in candidates]
    
    def get_action_candidates(self, state: PlanningState) -> List[ActionCandidate]:
        """
        Get detailed action candidates with metadata.
        
        Args:
            state: Planning state to expand from
            
        Returns:
            List of ActionCandidate objects
        """
        # Check cache first
        cache_key = self._get_cache_key(state)
        if cache_key in self._action_cache:
            self._cache_hits += 1
            return self._action_cache[cache_key]
        
        self._cache_misses += 1
        
        # Generate new action candidates
        candidates = self._generate_action_candidates(state)
        
        # Cache the results
        self._action_cache[cache_key] = candidates
        
        return candidates
    
    def _generate_action_candidates(self, state: PlanningState) -> List[ActionCandidate]:
        """Generate action candidates using LLM and tool registry."""
        
        # Get available tools based on context
        available_tools = self._get_available_tools(state)
        
        # Generate LLM-based action suggestions
        llm_suggestions = self._get_llm_action_suggestions(state, available_tools)
        
        # Combine and rank candidates
        candidates = self._create_action_candidates(state, available_tools, llm_suggestions)
        
        # Filter and rank candidates
        filtered_candidates = self._filter_and_rank_candidates(state, candidates)
        
        return filtered_candidates[:self.max_actions]
    
    def _get_available_tools(self, state: PlanningState) -> List[Dict[str, Any]]:
        """Get tools available for the current state."""
        
        # Get planning context
        planning_context = {}
        if self.context_manager:
            planning_context = self.context_manager.get_planning_context()
        
        # Merge state context with manager context
        combined_context = {
            'documents': state.document_context or planning_context.get('documents'),
            'memory': state.memory_context or planning_context.get('memory'),
            'conversation': state.conversation_context or planning_context.get('conversation')
        }
        
        # Get available tools from registry
        available_tools = self.tool_registry.get_available_tools(combined_context)
        
        # Get task-relevant tools
        task_relevant_tools = self.tool_registry.get_tools_for_task_type(state.task_description)
        
        # Combine and deduplicate
        all_tools = available_tools + task_relevant_tools
        unique_tools = []
        seen_names = set()
        
        for tool in all_tools:
            if tool.name not in seen_names:
                unique_tools.append(tool)
                seen_names.add(tool.name)
        
        # Convert to dict format for easier handling
        return [
            {
                'name': tool.name,
                'description': tool.description,
                'parameters': tool.parameters,
                'cost_estimate': tool.cost_estimate,
                'category': tool.category.value,
                'can_execute': tool.can_execute(combined_context)
            }
            for tool in unique_tools
        ]
    
    def _get_llm_action_suggestions(self, state: PlanningState, available_tools: List[Dict[str, Any]]) -> List[str]:
        """Get action suggestions from LLM."""
        
        if not self.llm_interface:
            # Return tool-based suggestions if no LLM
            return [tool['name'] for tool in available_tools if tool['can_execute']]
        
        try:
            prompt = self._build_action_expansion_prompt(state, available_tools)
            response = self._call_llm(prompt)
            return self._parse_action_response(response)
            
        except Exception as e:
            logger.warning(f"LLM action expansion failed: {e}")
            # Fallback to tool-based suggestions
            return [tool['name'] for tool in available_tools if tool['can_execute']]
    
    def _build_action_expansion_prompt(self, state: PlanningState, available_tools: List[Dict[str, Any]]) -> str:
        """Build prompt for LLM action expansion."""
        
        prompt_parts = [
            "You are an expert planning assistant. Suggest the next best actions to take.",
            "",
            f"TASK: {state.task_description}",
            "",
            "CURRENT STATE:",
            f"- Actions completed: {state.action_history}",
            f"- Current observation: {state.current_observation}",
            ""
        ]
        
        # Add context information
        if self.context_manager:
            context_summary = self.context_manager.get_context_summary()
            prompt_parts.extend([
                "AVAILABLE CONTEXT:",
                f"- Documents: {context_summary['documents']['count']}",
                f"- Memories: {context_summary['memory']['relevant_memories_count']}",
                f"- Conversation: {context_summary['conversation']['message_count']} messages",
                ""
            ])
        
        # Add available tools
        if available_tools:
            prompt_parts.extend([
                "AVAILABLE TOOLS:",
                ""
            ])
            
            for tool in available_tools[:10]:  # Limit to avoid token overflow
                status = "✓" if tool['can_execute'] else "✗"
                prompt_parts.append(f"{status} {tool['name']}: {tool['description']}")
            
            prompt_parts.append("")
        
        # Add instructions
        prompt_parts.extend([
            "INSTRUCTIONS:",
            "1. Suggest 3-5 specific next actions that would help complete the task",
            "2. Focus on actions that build logically on what's already been done",
            "3. Prefer actions that use available context (documents, memory, etc.)",
            "4. Only suggest actions from the available tools list",
            "",
            "Format your response as a simple list:",
            "- action_name_1",
            "- action_name_2", 
            "- action_name_3",
            "",
            "SUGGESTED ACTIONS:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM interface with the prompt."""
        if hasattr(self.llm_interface, 'generate'):
            return self.llm_interface.generate(prompt, temperature=0.7, max_tokens=200)
        elif hasattr(self.llm_interface, 'generate_response'):
            return self.llm_interface.generate_response(prompt)
        elif callable(self.llm_interface):
            return self.llm_interface(prompt)
        else:
            raise ValueError("Invalid LLM interface provided")
    
    def _parse_action_response(self, response: str) -> List[str]:
        """Parse LLM response to extract action names."""
        actions = []
        
        for line in response.strip().split('\n'):
            line = line.strip()
            
            # Look for list items
            if line.startswith('-') or line.startswith('*'):
                action = line[1:].strip()
                if action:
                    actions.append(action)
            
            # Look for numbered items
            elif line and line[0].isdigit() and '.' in line:
                parts = line.split('.', 1)
                if len(parts) > 1:
                    action = parts[1].strip()
                    if action:
                        actions.append(action)
        
        return actions
    
    def _create_action_candidates(self, 
                                state: PlanningState, 
                                available_tools: List[Dict[str, Any]], 
                                llm_suggestions: List[str]) -> List[ActionCandidate]:
        """Create ActionCandidate objects from suggestions and tools."""
        
        candidates = []
        tool_map = {tool['name']: tool for tool in available_tools}
        
        # Process LLM suggestions
        for suggestion in llm_suggestions:
            if suggestion in tool_map:
                tool = tool_map[suggestion]
                candidate = ActionCandidate(
                    action_name=suggestion,
                    parameters=self._infer_parameters(state, tool),
                    description=tool['description'],
                    estimated_cost=tool['cost_estimate'],
                    confidence=0.8,  # High confidence for LLM suggestions
                    prerequisites_met=tool['can_execute'],
                    reasoning=f"LLM suggested this action as relevant for the current state"
                )
                candidates.append(candidate)
        
        # Add high-priority tools not suggested by LLM
        for tool in available_tools:
            if tool['name'] not in llm_suggestions and tool['can_execute']:
                # Only add if it's highly relevant
                relevance_score = self._assess_tool_relevance(state, tool)
                if relevance_score > 0.6:
                    candidate = ActionCandidate(
                        action_name=tool['name'],
                        parameters=self._infer_parameters(state, tool),
                        description=tool['description'],
                        estimated_cost=tool['cost_estimate'],
                        confidence=relevance_score,
                        prerequisites_met=True,
                        reasoning=f"Tool is highly relevant (score: {relevance_score:.2f}) for current state"
                    )
                    candidates.append(candidate)
        
        return candidates
    
    def _infer_parameters(self, state: PlanningState, tool: Dict[str, Any]) -> Dict[str, Any]:
        """Infer parameters for a tool based on current state."""
        parameters = {}
        
        # Simple parameter inference based on tool type and state
        if 'filename' in tool['parameters'] and state.has_document_context():
            # Use first available document
            if state.document_context:
                filenames = list(state.document_context.keys())
                if filenames:
                    parameters['filename'] = filenames[0]
        
        if 'query' in tool['parameters']:
            # Use task description as query
            parameters['query'] = state.task_description
        
        # Add default values for common parameters
        if 'summary_type' in tool['parameters']:
            parameters['summary_type'] = 'comprehensive'
        
        if 'analysis_depth' in tool['parameters']:
            parameters['analysis_depth'] = 'detailed'
        
        return parameters
    
    def _assess_tool_relevance(self, state: PlanningState, tool: Dict[str, Any]) -> float:
        """Assess how relevant a tool is for the current state."""
        relevance_score = 0.0
        
        # Check if tool category matches task type
        task_lower = state.task_description.lower()
        tool_category = tool['category']
        
        if tool_category == 'document_analysis' and any(word in task_lower for word in ['document', 'paper', 'file', 'analyze']):
            relevance_score += 0.4
        
        if tool_category == 'research_tools' and any(word in task_lower for word in ['research', 'search', 'find']):
            relevance_score += 0.4
        
        if tool_category == 'memory_operations' and any(word in task_lower for word in ['remember', 'recall', 'memory']):
            relevance_score += 0.4
        
        # Check if tool hasn't been used recently
        if tool['name'] not in state.action_history:
            relevance_score += 0.2
        
        # Check if prerequisites are met
        if tool['can_execute']:
            relevance_score += 0.2
        
        return min(1.0, relevance_score)
    
    def _filter_and_rank_candidates(self, state: PlanningState, candidates: List[ActionCandidate]) -> List[ActionCandidate]:
        """Filter and rank action candidates."""
        
        # Filter out candidates that can't be executed
        executable_candidates = [c for c in candidates if c.prerequisites_met]
        
        # Remove duplicates
        unique_candidates = []
        seen_actions = set()
        for candidate in executable_candidates:
            if candidate.action_name not in seen_actions:
                unique_candidates.append(candidate)
                seen_actions.add(candidate.action_name)
        
        # Sort by confidence and cost (higher confidence, lower cost is better)
        unique_candidates.sort(key=lambda c: (-c.confidence, c.estimated_cost))
        
        return unique_candidates
    
    def _get_cache_key(self, state: PlanningState) -> str:
        """Generate cache key for state."""
        return f"{state.task_description}_{len(state.action_history)}_{hash(tuple(state.action_history))}"
    
    def get_expansion_stats(self) -> Dict[str, Any]:
        """Get statistics about action expansion performance."""
        total_requests = self._cache_hits + self._cache_misses
        cache_hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self._action_cache),
            'max_actions': self.max_actions,
            'tool_registry_size': len(self.tool_registry.get_tool_names())
        }
    
    def clear_cache(self):
        """Clear the action expansion cache."""
        self._action_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Cleared action expansion cache")
