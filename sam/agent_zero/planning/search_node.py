"""
Search Node Module

Defines the SearchNode class that wraps PlanningState for use in the A* search algorithm.
Includes heuristic scores and comparison methods for priority queue operations.
"""

from dataclasses import dataclass, field
from typing import Optional
from .state import PlanningState


@dataclass
class SearchNode:
    """
    Wrapper for PlanningState that includes search-specific metadata.
    
    This class adds the heuristic and total scores needed for A* search,
    along with comparison methods for priority queue operations.
    """
    
    state: PlanningState
    """The planning state this node represents"""
    
    h_score: int = 0
    """Heuristic estimate of cost from this state to goal"""
    
    f_score: int = field(init=False)
    """Total estimated cost (g_score + h_score)"""
    
    # Search metadata
    expanded: bool = False
    """Whether this node has been expanded (children generated)"""
    
    in_frontier: bool = True
    """Whether this node is currently in the frontier"""
    
    def __post_init__(self):
        """Calculate f_score after initialization."""
        self.f_score = self.state.g_score + self.h_score
    
    def update_h_score(self, new_h_score: int):
        """
        Update the heuristic score and recalculate f_score.
        
        Args:
            new_h_score: New heuristic score value
        """
        self.h_score = new_h_score
        self.f_score = self.state.g_score + self.h_score
    
    def mark_expanded(self):
        """Mark this node as expanded (children have been generated)."""
        self.expanded = True
        self.in_frontier = False
    
    def get_search_summary(self) -> str:
        """
        Generate a summary of search-specific information.
        
        Returns:
            Human-readable summary of search metadata
        """
        status = []
        if self.expanded:
            status.append("expanded")
        if self.in_frontier:
            status.append("in_frontier")
        
        status_str = ", ".join(status) if status else "processed"
        
        return (
            f"SearchNode(f={self.f_score}, g={self.state.g_score}, h={self.h_score}, "
            f"depth={self.state.depth}, status={status_str})"
        )
    
    def is_better_than(self, other: 'SearchNode') -> bool:
        """
        Check if this node is better than another for A* search.
        
        Args:
            other: Another SearchNode to compare against
            
        Returns:
            True if this node should be explored before the other
        """
        if self.f_score != other.f_score:
            return self.f_score < other.f_score
        
        # Tie-breaking: prefer nodes with lower g_score (shorter paths)
        if self.state.g_score != other.state.g_score:
            return self.state.g_score < other.state.g_score
        
        # Final tie-breaking: prefer newer nodes (exploration diversity)
        return self.state.created_at > other.state.created_at
    
    # Comparison methods for priority queue operations
    def __lt__(self, other: 'SearchNode') -> bool:
        """Less than comparison for priority queue (min-heap)."""
        return self.is_better_than(other)
    
    def __le__(self, other: 'SearchNode') -> bool:
        """Less than or equal comparison."""
        return self.is_better_than(other) or self == other
    
    def __gt__(self, other: 'SearchNode') -> bool:
        """Greater than comparison."""
        return other.is_better_than(self)
    
    def __ge__(self, other: 'SearchNode') -> bool:
        """Greater than or equal comparison."""
        return other.is_better_than(self) or self == other
    
    def __eq__(self, other) -> bool:
        """
        Equality comparison based on state equality.
        
        Args:
            other: Another SearchNode or object to compare
            
        Returns:
            True if the underlying states are equal
        """
        if not isinstance(other, SearchNode):
            return False
        return self.state == other.state
    
    def __hash__(self) -> int:
        """Generate hash based on underlying state."""
        return hash(self.state)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"SearchNode(f={self.f_score}, g={self.state.g_score}, h={self.h_score}, "
            f"state_id={self.state.state_id[:8]})"
        )


class SearchNodeFactory:
    """
    Factory class for creating SearchNode instances with consistent configuration.
    """
    
    @staticmethod
    def create_root_node(task_description: str, 
                        initial_context: Optional[dict] = None) -> SearchNode:
        """
        Create the root node for A* search.
        
        Args:
            task_description: The main task to accomplish
            initial_context: Optional initial context (documents, memory, etc.)
            
        Returns:
            SearchNode representing the initial state
        """
        # Extract context components if provided
        document_context = None
        memory_context = None
        conversation_context = None
        
        if initial_context:
            document_context = initial_context.get('documents')
            memory_context = initial_context.get('memory')
            conversation_context = initial_context.get('conversation')
        
        # Create initial planning state
        initial_state = PlanningState(
            task_description=task_description,
            action_history=[],
            current_observation="Starting task planning",
            parent=None,
            g_score=0,
            document_context=document_context,
            memory_context=memory_context,
            conversation_context=conversation_context
        )
        
        # Create search node with initial heuristic (will be updated by estimator)
        return SearchNode(
            state=initial_state,
            h_score=0,  # Will be set by heuristic estimator
            expanded=False,
            in_frontier=True
        )
    
    @staticmethod
    def create_child_node(parent_node: SearchNode, 
                         action: str, 
                         observation: str = "",
                         h_score: int = 0) -> SearchNode:
        """
        Create a child node from a parent node and action.
        
        Args:
            parent_node: The parent SearchNode
            action: Action taken to reach this state
            observation: Result of executing the action
            h_score: Heuristic score for the new state
            
        Returns:
            New SearchNode representing the child state
        """
        # Create new state by adding action to parent
        child_state = parent_node.state.add_action(action, observation)
        
        # Create search node for child
        return SearchNode(
            state=child_state,
            h_score=h_score,
            expanded=False,
            in_frontier=True
        )
