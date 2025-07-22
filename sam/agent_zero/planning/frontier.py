"""
Frontier Module

Implements the Frontier class that manages the priority queue of nodes to explore
in the A* search algorithm. Provides efficient operations for adding, removing,
and managing search nodes.
"""

import heapq
import logging
from typing import List, Optional, Set, Dict, Any
from collections import defaultdict
from .search_node import SearchNode

logger = logging.getLogger(__name__)


class Frontier:
    """
    Priority queue for managing SearchNodes in A* search.
    
    This class wraps Python's heapq module to provide a clean interface
    for managing the frontier of nodes to explore. It includes optimizations
    for duplicate detection and frontier management.
    """
    
    def __init__(self, max_size: Optional[int] = None):
        """
        Initialize the frontier.
        
        Args:
            max_size: Optional maximum size limit for the frontier
        """
        self._heap: List[SearchNode] = []
        self._node_set: Set[str] = set()  # Track state IDs for duplicate detection
        self._max_size = max_size
        self._total_added = 0
        self._total_popped = 0
        
        # Statistics tracking
        self._size_history: List[int] = []
        self._best_f_score_history: List[int] = []
    
    def add(self, node: SearchNode) -> bool:
        """
        Add a node to the frontier.
        
        Args:
            node: SearchNode to add to the frontier
            
        Returns:
            True if node was added, False if it was a duplicate or frontier is full
        """
        # Check for duplicates
        if node.state.state_id in self._node_set:
            logger.debug(f"Skipping duplicate node: {node.state.state_id[:8]}")
            return False
        
        # Check size limit
        if self._max_size and len(self._heap) >= self._max_size:
            # If new node is better than worst node, replace it
            if self._heap and node.is_better_than(self._heap[0]):
                worst_node = heapq.heappop(self._heap)
                self._node_set.remove(worst_node.state.state_id)
                worst_node.in_frontier = False
                logger.debug(f"Replaced worst node with better node")
            else:
                logger.debug(f"Frontier full, rejecting node: {node.get_search_summary()}")
                return False
        
        # Add node to frontier
        heapq.heappush(self._heap, node)
        self._node_set.add(node.state.state_id)
        node.in_frontier = True
        self._total_added += 1
        
        # Update statistics
        self._size_history.append(len(self._heap))
        if self._heap:
            self._best_f_score_history.append(self._heap[0].f_score)
        
        logger.debug(f"Added node to frontier: {node.get_search_summary()}")
        return True
    
    def pop(self) -> Optional[SearchNode]:
        """
        Remove and return the best node from the frontier.
        
        Returns:
            Best SearchNode according to A* criteria, or None if frontier is empty
        """
        if not self._heap:
            return None
        
        # Get best node
        best_node = heapq.heappop(self._heap)
        self._node_set.remove(best_node.state.state_id)
        best_node.in_frontier = False
        self._total_popped += 1
        
        # Update statistics
        self._size_history.append(len(self._heap))
        
        logger.debug(f"Popped node from frontier: {best_node.get_search_summary()}")
        return best_node
    
    def peek(self) -> Optional[SearchNode]:
        """
        Look at the best node without removing it.
        
        Returns:
            Best SearchNode or None if frontier is empty
        """
        return self._heap[0] if self._heap else None
    
    def is_empty(self) -> bool:
        """
        Check if the frontier is empty.
        
        Returns:
            True if no nodes are in the frontier
        """
        return len(self._heap) == 0
    
    def size(self) -> int:
        """
        Get the current size of the frontier.
        
        Returns:
            Number of nodes currently in the frontier
        """
        return len(self._heap)
    
    def contains_state(self, state_id: str) -> bool:
        """
        Check if a state is already in the frontier.
        
        Args:
            state_id: State ID to check for
            
        Returns:
            True if state is in frontier
        """
        return state_id in self._node_set
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get frontier statistics for monitoring and debugging.
        
        Returns:
            Dictionary with frontier statistics
        """
        stats = {
            'current_size': len(self._heap),
            'max_size_limit': self._max_size,
            'total_added': self._total_added,
            'total_popped': self._total_popped,
            'is_empty': self.is_empty()
        }
        
        if self._heap:
            stats.update({
                'best_f_score': self._heap[0].f_score,
                'best_g_score': self._heap[0].state.g_score,
                'best_h_score': self._heap[0].h_score,
                'best_depth': self._heap[0].state.depth
            })
        
        if self._size_history:
            stats.update({
                'max_size_reached': max(self._size_history),
                'avg_size': sum(self._size_history) / len(self._size_history)
            })
        
        if self._best_f_score_history:
            stats.update({
                'best_f_score_ever': min(self._best_f_score_history),
                'f_score_improvement': (
                    self._best_f_score_history[0] - min(self._best_f_score_history)
                    if len(self._best_f_score_history) > 1 else 0
                )
            })
        
        return stats
    
    def get_frontier_summary(self) -> str:
        """
        Get a human-readable summary of the frontier state.
        
        Returns:
            Summary string for logging/debugging
        """
        if self.is_empty():
            return "Frontier: EMPTY"
        
        best = self._heap[0]
        stats = self.get_statistics()
        
        return (
            f"Frontier: {stats['current_size']} nodes, "
            f"best f={best.f_score} (g={best.state.g_score}, h={best.h_score}), "
            f"depth={best.state.depth}"
        )
    
    def clear(self):
        """Clear all nodes from the frontier."""
        for node in self._heap:
            node.in_frontier = False
        
        self._heap.clear()
        self._node_set.clear()
        logger.debug("Frontier cleared")
    
    def get_nodes_by_f_score(self, max_nodes: int = 10) -> List[SearchNode]:
        """
        Get the best nodes sorted by f_score for analysis.
        
        Args:
            max_nodes: Maximum number of nodes to return
            
        Returns:
            List of best nodes sorted by f_score
        """
        # Create a copy of heap and sort it
        sorted_nodes = sorted(self._heap, key=lambda n: (n.f_score, n.state.g_score))
        return sorted_nodes[:max_nodes]
    
    def prune_high_cost_nodes(self, f_score_threshold: int) -> int:
        """
        Remove nodes with f_score above threshold to manage memory.
        
        Args:
            f_score_threshold: Maximum allowed f_score
            
        Returns:
            Number of nodes removed
        """
        original_size = len(self._heap)
        
        # Filter out high-cost nodes
        filtered_nodes = [node for node in self._heap if node.f_score <= f_score_threshold]
        
        # Update node set
        removed_nodes = [node for node in self._heap if node.f_score > f_score_threshold]
        for node in removed_nodes:
            self._node_set.remove(node.state.state_id)
            node.in_frontier = False
        
        # Rebuild heap
        self._heap = filtered_nodes
        heapq.heapify(self._heap)
        
        removed_count = original_size - len(self._heap)
        if removed_count > 0:
            logger.info(f"Pruned {removed_count} high-cost nodes (f_score > {f_score_threshold})")
        
        return removed_count
    
    def __len__(self) -> int:
        """Return the size of the frontier."""
        return len(self._heap)
    
    def __bool__(self) -> bool:
        """Return True if frontier is not empty."""
        return not self.is_empty()
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"Frontier(size={len(self._heap)}, max_size={self._max_size})"
