"""
Planning State Module

Defines the PlanningState class that represents a node in the A* search tree.
Each state captures the current situation, action history, and context needed
for strategic planning.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid


@dataclass
class PlanningState:
    """
    Represents a state in the A* search tree for strategic planning.
    
    This class captures all the information needed to represent a particular
    point in the planning process, including the task, actions taken so far,
    current observations, and context from SAM's systems.
    """
    
    # Core planning fields
    task_description: str
    """The original user goal or task to accomplish"""
    
    action_history: List[str] = field(default_factory=list)
    """Sequence of actions taken to reach this state"""
    
    current_observation: str = ""
    """Result or observation from the last action taken"""
    
    parent: Optional['PlanningState'] = None
    """Reference to the previous state (for path reconstruction)"""
    
    g_score: int = 0
    """Cost to reach this state (typically len(action_history))"""
    
    # SAM-specific context fields (Phase 1.5 enhancement)
    document_context: Optional[Dict[str, Any]] = None
    """Available documents and their analysis results"""
    
    memory_context: Optional[List[str]] = None
    """Relevant memories from SAM's memory system"""
    
    conversation_context: Optional[Dict[str, Any]] = None
    """Current conversation state and history"""
    
    # Metadata fields
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """Unique identifier for this state"""
    
    created_at: datetime = field(default_factory=datetime.now)
    """Timestamp when this state was created"""
    
    depth: int = 0
    """Depth in the search tree (number of actions from root)"""
    
    def __post_init__(self):
        """Initialize computed fields after object creation."""
        # Set g_score to action history length if not explicitly set
        if self.g_score == 0:
            self.g_score = len(self.action_history)
        
        # Set depth based on action history
        self.depth = len(self.action_history)
    
    def add_action(self, action: str, observation: str = "") -> 'PlanningState':
        """
        Create a new state by adding an action to this state.
        
        Args:
            action: The action to add
            observation: The result of executing the action
            
        Returns:
            New PlanningState with the action added
        """
        new_action_history = self.action_history + [action]
        
        return PlanningState(
            task_description=self.task_description,
            action_history=new_action_history,
            current_observation=observation,
            parent=self,
            g_score=len(new_action_history),
            document_context=self.document_context,
            memory_context=self.memory_context,
            conversation_context=self.conversation_context,
            depth=len(new_action_history)
        )
    
    def get_action_path(self) -> List[str]:
        """
        Reconstruct the full path of actions from root to this state.
        
        Returns:
            List of actions from the initial state to this state
        """
        return self.action_history.copy()
    
    def get_state_summary(self) -> str:
        """
        Generate a concise summary of this state for logging/debugging.
        
        Returns:
            Human-readable summary of the state
        """
        summary_parts = [
            f"Task: {self.task_description[:50]}{'...' if len(self.task_description) > 50 else ''}",
            f"Actions: {len(self.action_history)}",
            f"G-Score: {self.g_score}",
            f"Depth: {self.depth}"
        ]
        
        if self.current_observation:
            obs_preview = self.current_observation[:30]
            summary_parts.append(f"Last Obs: {obs_preview}{'...' if len(self.current_observation) > 30 else ''}")
        
        return " | ".join(summary_parts)
    
    def has_document_context(self) -> bool:
        """Check if this state has document context available."""
        return self.document_context is not None and len(self.document_context) > 0
    
    def has_memory_context(self) -> bool:
        """Check if this state has memory context available."""
        return self.memory_context is not None and len(self.memory_context) > 0
    
    def has_conversation_context(self) -> bool:
        """Check if this state has conversation context available."""
        return self.conversation_context is not None and len(self.conversation_context) > 0
    
    def get_context_summary(self) -> Dict[str, bool]:
        """
        Get a summary of available context types.
        
        Returns:
            Dictionary indicating which context types are available
        """
        return {
            'documents': self.has_document_context(),
            'memory': self.has_memory_context(),
            'conversation': self.has_conversation_context()
        }
    
    def is_goal_state(self, goal_checker=None) -> bool:
        """
        Check if this state represents a goal state.
        
        Args:
            goal_checker: Optional function to check goal conditions
            
        Returns:
            True if this is a goal state, False otherwise
        """
        # Default simple goal check - can be enhanced with custom logic
        if goal_checker:
            return goal_checker(self)
        
        # Simple heuristic: if we have a substantial observation that suggests completion
        if self.current_observation:
            completion_indicators = [
                'completed', 'finished', 'done', 'success', 'accomplished',
                'achieved', 'resolved', 'solved', 'answered'
            ]
            obs_lower = self.current_observation.lower()
            return any(indicator in obs_lower for indicator in completion_indicators)
        
        return False
    
    def __eq__(self, other) -> bool:
        """Check equality based on state content (not metadata)."""
        if not isinstance(other, PlanningState):
            return False
        
        return (
            self.task_description == other.task_description and
            self.action_history == other.action_history and
            self.current_observation == other.current_observation
        )
    
    def __hash__(self) -> int:
        """Generate hash for use in sets/dictionaries."""
        return hash((
            self.task_description,
            tuple(self.action_history),
            self.current_observation
        ))
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"PlanningState(id={self.state_id[:8]}, depth={self.depth}, g_score={self.g_score})"
