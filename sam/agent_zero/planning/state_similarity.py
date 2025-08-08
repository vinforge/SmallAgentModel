"""
State Similarity Detection Module

Provides functionality to detect similar planning states for optimization purposes.
This enables caching of heuristic estimates and avoiding redundant computations
in the A* search algorithm.
"""

from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import hashlib
import logging
from .state import PlanningState

logger = logging.getLogger(__name__)


@dataclass
class SimilarityMetrics:
    """Metrics for measuring state similarity."""
    
    task_similarity: float = 0.0
    """Similarity of task descriptions (0.0 to 1.0)"""
    
    action_similarity: float = 0.0
    """Similarity of action histories (0.0 to 1.0)"""
    
    context_similarity: float = 0.0
    """Similarity of context (documents, memory, etc.) (0.0 to 1.0)"""
    
    observation_similarity: float = 0.0
    """Similarity of current observations (0.0 to 1.0)"""
    
    overall_similarity: float = 0.0
    """Overall similarity score (0.0 to 1.0)"""
    
    def __post_init__(self):
        """Calculate overall similarity as weighted average."""
        # Weights for different similarity components
        weights = {
            'task': 0.3,
            'action': 0.3,
            'context': 0.2,
            'observation': 0.2
        }
        
        self.overall_similarity = (
            weights['task'] * self.task_similarity +
            weights['action'] * self.action_similarity +
            weights['context'] * self.context_similarity +
            weights['observation'] * self.observation_similarity
        )


class StateSimilarityDetector:
    """
    Detects similarity between planning states for optimization.
    
    This class provides methods to compare states and determine if they
    are similar enough to reuse cached heuristic estimates or other
    computed values.
    """
    
    def __init__(self, similarity_threshold: float = 0.8):
        """
        Initialize the similarity detector.
        
        Args:
            similarity_threshold: Minimum similarity score to consider states similar
        """
        self.similarity_threshold = similarity_threshold
        self._state_cache: Dict[str, PlanningState] = {}
        self._similarity_cache: Dict[Tuple[str, str], SimilarityMetrics] = {}
    
    def compute_similarity(self, state1: PlanningState, state2: PlanningState) -> SimilarityMetrics:
        """
        Compute similarity metrics between two planning states.
        
        Args:
            state1: First planning state
            state2: Second planning state
            
        Returns:
            SimilarityMetrics object with detailed similarity scores
        """
        # Check cache first
        cache_key = (state1.state_id, state2.state_id)
        reverse_key = (state2.state_id, state1.state_id)
        
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]
        elif reverse_key in self._similarity_cache:
            return self._similarity_cache[reverse_key]
        
        # Compute similarity metrics
        task_sim = self._compute_task_similarity(state1, state2)
        action_sim = self._compute_action_similarity(state1, state2)
        context_sim = self._compute_context_similarity(state1, state2)
        obs_sim = self._compute_observation_similarity(state1, state2)
        
        metrics = SimilarityMetrics(
            task_similarity=task_sim,
            action_similarity=action_sim,
            context_similarity=context_sim,
            observation_similarity=obs_sim
        )
        
        # Cache the result
        self._similarity_cache[cache_key] = metrics
        
        return metrics
    
    def are_states_similar(self, state1: PlanningState, state2: PlanningState) -> bool:
        """
        Check if two states are similar enough for optimization purposes.
        
        Args:
            state1: First planning state
            state2: Second planning state
            
        Returns:
            True if states are considered similar
        """
        metrics = self.compute_similarity(state1, state2)
        return metrics.overall_similarity >= self.similarity_threshold
    
    def find_similar_states(self, target_state: PlanningState, 
                           candidate_states: List[PlanningState]) -> List[Tuple[PlanningState, float]]:
        """
        Find states similar to the target state from a list of candidates.
        
        Args:
            target_state: State to find similarities for
            candidate_states: List of candidate states to compare
            
        Returns:
            List of (state, similarity_score) tuples for similar states
        """
        similar_states = []
        
        for candidate in candidate_states:
            if candidate.state_id == target_state.state_id:
                continue  # Skip self
            
            metrics = self.compute_similarity(target_state, candidate)
            if metrics.overall_similarity >= self.similarity_threshold:
                similar_states.append((candidate, metrics.overall_similarity))
        
        # Sort by similarity score (highest first)
        similar_states.sort(key=lambda x: x[1], reverse=True)
        
        return similar_states
    
    def _compute_task_similarity(self, state1: PlanningState, state2: PlanningState) -> float:
        """Compute similarity of task descriptions."""
        if state1.task_description == state2.task_description:
            return 1.0
        
        # Simple word-based similarity
        words1 = set(state1.task_description.lower().split())
        words2 = set(state2.task_description.lower().split())
        
        if not words1 and not words2:
            return 1.0
        elif not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _compute_action_similarity(self, state1: PlanningState, state2: PlanningState) -> float:
        """Compute similarity of action histories."""
        actions1 = state1.action_history
        actions2 = state2.action_history
        
        if not actions1 and not actions2:
            return 1.0
        elif not actions1 or not actions2:
            return 0.0
        
        # Compute longest common subsequence similarity
        lcs_length = self._longest_common_subsequence(actions1, actions2)
        max_length = max(len(actions1), len(actions2))
        
        return lcs_length / max_length if max_length > 0 else 1.0
    
    def _compute_context_similarity(self, state1: PlanningState, state2: PlanningState) -> float:
        """Compute similarity of context (documents, memory, conversation)."""
        similarities = []
        
        # Document context similarity
        doc_sim = self._compare_dict_context(state1.document_context, state2.document_context)
        similarities.append(doc_sim)
        
        # Memory context similarity
        mem_sim = self._compare_list_context(state1.memory_context, state2.memory_context)
        similarities.append(mem_sim)
        
        # Conversation context similarity
        conv_sim = self._compare_dict_context(state1.conversation_context, state2.conversation_context)
        similarities.append(conv_sim)
        
        # Return average similarity
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _compute_observation_similarity(self, state1: PlanningState, state2: PlanningState) -> float:
        """Compute similarity of current observations."""
        obs1 = state1.current_observation
        obs2 = state2.current_observation
        
        if obs1 == obs2:
            return 1.0
        
        if not obs1 and not obs2:
            return 1.0
        elif not obs1 or not obs2:
            return 0.0
        
        # Simple word-based similarity for observations
        words1 = set(obs1.lower().split())
        words2 = set(obs2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _longest_common_subsequence(self, seq1: List[str], seq2: List[str]) -> int:
        """Compute length of longest common subsequence."""
        m, n = len(seq1), len(seq2)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _compare_dict_context(self, ctx1: Optional[Dict], ctx2: Optional[Dict]) -> float:
        """Compare dictionary-based context."""
        if ctx1 is None and ctx2 is None:
            return 1.0
        elif ctx1 is None or ctx2 is None:
            return 0.0
        
        # Compare keys
        keys1 = set(ctx1.keys())
        keys2 = set(ctx2.keys())
        
        if not keys1 and not keys2:
            return 1.0
        
        key_intersection = keys1.intersection(keys2)
        key_union = keys1.union(keys2)
        
        key_similarity = len(key_intersection) / len(key_union) if key_union else 0.0
        
        # Compare values for common keys
        value_similarities = []
        for key in key_intersection:
            val1 = str(ctx1[key])
            val2 = str(ctx2[key])
            
            if val1 == val2:
                value_similarities.append(1.0)
            else:
                # Simple string similarity
                words1 = set(val1.lower().split())
                words2 = set(val2.lower().split())
                
                if words1 or words2:
                    intersection = words1.intersection(words2)
                    union = words1.union(words2)
                    value_similarities.append(len(intersection) / len(union) if union else 0.0)
                else:
                    value_similarities.append(1.0)
        
        value_similarity = sum(value_similarities) / len(value_similarities) if value_similarities else 1.0
        
        # Combine key and value similarities
        return (key_similarity + value_similarity) / 2
    
    def _compare_list_context(self, ctx1: Optional[List], ctx2: Optional[List]) -> float:
        """Compare list-based context."""
        if ctx1 is None and ctx2 is None:
            return 1.0
        elif ctx1 is None or ctx2 is None:
            return 0.0
        
        if not ctx1 and not ctx2:
            return 1.0
        
        # Convert to sets for comparison
        set1 = set(str(item) for item in ctx1)
        set2 = set(str(item) for item in ctx2)
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def get_state_signature(self, state: PlanningState) -> str:
        """
        Generate a signature for a state for quick similarity checks.
        
        Args:
            state: Planning state to generate signature for
            
        Returns:
            String signature representing the state
        """
        # Create a hash of key state components
        components = [
            state.task_description,
            str(sorted(state.action_history)),
            state.current_observation,
            str(sorted(state.document_context.keys()) if state.document_context else ""),
            str(sorted(state.memory_context) if state.memory_context else ""),
            str(sorted(state.conversation_context.keys()) if state.conversation_context else "")
        ]
        
        signature_string = "|".join(components)
        return hashlib.md5(signature_string.encode()).hexdigest()
    
    def clear_cache(self):
        """Clear similarity computation cache."""
        self._similarity_cache.clear()
        logger.debug("Cleared state similarity cache")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'similarity_cache_size': len(self._similarity_cache),
            'state_cache_size': len(self._state_cache)
        }
