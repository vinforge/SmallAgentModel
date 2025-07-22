"""
Episodic Memory Enhanced Heuristic Module

Integrates SAM's episodic memory system with the A* planner heuristic estimator
to learn from past planning outcomes and improve future cost estimates.
Acts as the "Experience Engine" for self-improving planning.
"""

import logging
import hashlib
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from .state import PlanningState
from .heuristic_estimator import HeuristicEstimator, HeuristicEstimate

# Import SAM's episodic memory components
try:
    from memory.episodic_store import EpisodicMemoryStore, EpisodicMemory, InteractionType, OutcomeType
    EPISODIC_MEMORY_AVAILABLE = True
except ImportError:
    EPISODIC_MEMORY_AVAILABLE = False
    logging.warning("SAM episodic memory components not available - using fallback implementation")

logger = logging.getLogger(__name__)


@dataclass
class PlanningExperience:
    """Represents a past planning experience for learning."""
    
    experience_id: str
    """Unique identifier for this experience"""
    
    task_type: str
    """Type/category of the planning task"""
    
    action_sequence: List[str]
    """Sequence of actions taken"""
    
    context_features: Dict[str, Any]
    """Context features (documents, memory, etc.)"""
    
    estimated_cost: int
    """Original heuristic estimate"""
    
    actual_cost: int
    """Actual cost to complete the task"""
    
    success: bool
    """Whether the planning was successful"""
    
    outcome_quality: float
    """Quality of the outcome (0.0 to 1.0)"""
    
    timestamp: str
    """When this experience occurred"""
    
    metadata: Dict[str, Any]
    """Additional metadata about the experience"""


@dataclass
class ExperienceAdjustment:
    """Adjustment to heuristic based on past experiences."""
    
    adjustment_value: int
    """Amount to adjust the heuristic estimate"""
    
    confidence: float
    """Confidence in this adjustment (0.0 to 1.0)"""
    
    reasoning: str
    """Explanation for the adjustment"""
    
    similar_experiences: List[PlanningExperience]
    """Past experiences that influenced this adjustment"""


class EpisodicMemoryHeuristic(HeuristicEstimator):
    """
    Enhanced heuristic estimator that learns from past planning experiences.
    
    This class extends the base HeuristicEstimator to incorporate learning
    from SAM's episodic memory system, enabling self-improving planning
    through experience-based cost estimation adjustments.
    """
    
    def __init__(self,
                 llm_interface=None,
                 context_manager=None,
                 episodic_store: Optional[EpisodicMemoryStore] = None,
                 enable_experience_learning: bool = True,
                 experience_weight: float = 0.3,
                 similarity_threshold: float = 0.7):
        """
        Initialize the episodic memory enhanced heuristic estimator.
        
        Args:
            llm_interface: SAM's LLM interface for base estimates
            context_manager: SAM context manager
            episodic_store: SAM's episodic memory store
            enable_experience_learning: Whether to use experience-based learning
            experience_weight: Weight for experience adjustments (0.0 to 1.0)
            similarity_threshold: Threshold for considering experiences similar
        """
        super().__init__(llm_interface, context_manager)
        
        self.enable_experience_learning = enable_experience_learning and EPISODIC_MEMORY_AVAILABLE
        self.experience_weight = experience_weight
        self.similarity_threshold = similarity_threshold
        
        # Initialize episodic memory store
        if self.enable_experience_learning:
            self.episodic_store = episodic_store or self._create_episodic_store()
        else:
            self.episodic_store = None
        
        # Experience cache for performance
        self._experience_cache: Dict[str, List[PlanningExperience]] = {}
        self._cache_expiry_hours = 24
        
        # Learning statistics
        self.total_adjustments = 0
        self.successful_adjustments = 0
        self.experience_queries = 0
        
        logger.info(f"EpisodicMemoryHeuristic initialized (learning: {self.enable_experience_learning})")
    
    def _create_episodic_store(self) -> Optional[EpisodicMemoryStore]:
        """Create episodic memory store if available."""
        try:
            return EpisodicMemoryStore(db_path="memory_store/planning_episodic_memory.db")
        except Exception as e:
            logger.warning(f"Failed to create episodic store: {e}")
            self.enable_experience_learning = False
            return None
    
    def get_detailed_estimate(self, state: PlanningState) -> HeuristicEstimate:
        """
        Get detailed heuristic estimate enhanced with episodic memory learning.
        
        Args:
            state: Planning state to estimate cost for
            
        Returns:
            Enhanced HeuristicEstimate with experience-based adjustments
        """
        # Get base estimate from parent class
        base_estimate = super().get_detailed_estimate(state)
        
        # Apply experience-based learning if enabled
        if self.enable_experience_learning and self.episodic_store:
            experience_adjustment = self._get_experience_adjustment(state)
            
            if experience_adjustment:
                # Apply adjustment to base estimate
                adjusted_cost = max(0, base_estimate.estimated_cost + experience_adjustment.adjustment_value)
                
                # Combine confidence scores
                combined_confidence = (
                    base_estimate.confidence * (1 - self.experience_weight) +
                    experience_adjustment.confidence * self.experience_weight
                )
                
                # Enhanced reasoning
                enhanced_reasoning = (
                    f"{base_estimate.reasoning}\n"
                    f"Experience adjustment: {experience_adjustment.reasoning}"
                )
                
                # Enhanced context factors
                enhanced_factors = base_estimate.context_factors + [
                    f"experience_adjustment_{experience_adjustment.adjustment_value}",
                    f"similar_experiences_{len(experience_adjustment.similar_experiences)}"
                ]
                
                self.total_adjustments += 1
                
                return HeuristicEstimate(
                    estimated_cost=adjusted_cost,
                    confidence=combined_confidence,
                    reasoning=enhanced_reasoning,
                    context_factors=enhanced_factors,
                    fallback_used=base_estimate.fallback_used
                )
        
        return base_estimate
    
    def _get_experience_adjustment(self, state: PlanningState) -> Optional[ExperienceAdjustment]:
        """Get experience-based adjustment for the heuristic estimate."""
        
        # Find similar past experiences
        similar_experiences = self._find_similar_experiences(state)
        
        if not similar_experiences:
            return None
        
        # Calculate adjustment based on past outcomes
        adjustment_value, confidence, reasoning = self._calculate_experience_adjustment(
            state, similar_experiences
        )
        
        if abs(adjustment_value) < 1:  # Skip very small adjustments
            return None
        
        return ExperienceAdjustment(
            adjustment_value=adjustment_value,
            confidence=confidence,
            reasoning=reasoning,
            similar_experiences=similar_experiences
        )
    
    def _find_similar_experiences(self, state: PlanningState) -> List[PlanningExperience]:
        """Find similar past planning experiences."""
        
        # Generate cache key for this state
        cache_key = self._generate_experience_cache_key(state)
        
        # Check cache first
        if cache_key in self._experience_cache:
            cached_experiences, cache_time = self._experience_cache[cache_key]
            if self._is_cache_valid(cache_time):
                return cached_experiences
        
        # Query episodic memory for similar experiences
        similar_experiences = self._query_episodic_memory(state)
        
        # Cache the results
        self._experience_cache[cache_key] = (similar_experiences, datetime.now())
        
        self.experience_queries += 1
        
        return similar_experiences
    
    def _query_episodic_memory(self, state: PlanningState) -> List[PlanningExperience]:
        """Query SAM's episodic memory for similar planning experiences."""
        
        if not self.episodic_store:
            return []
        
        try:
            # Create search criteria based on current state
            task_type = self._classify_task_type(state.task_description)
            context_features = self._extract_context_features(state)
            
            # Search for similar planning memories
            # Note: This is a simplified implementation - real implementation would use
            # more sophisticated similarity matching
            
            memories = self._search_planning_memories(task_type, context_features)
            
            # Convert to PlanningExperience objects
            experiences = []
            for memory in memories:
                experience = self._convert_memory_to_experience(memory)
                if experience and self._is_experience_similar(state, experience):
                    experiences.append(experience)
            
            return experiences[:10]  # Limit to top 10 similar experiences
            
        except Exception as e:
            logger.warning(f"Error querying episodic memory: {e}")
            return []
    
    def _search_planning_memories(self, task_type: str, context_features: Dict[str, Any]) -> List[Any]:
        """Search for planning-related memories in episodic store."""
        
        # This is a simplified implementation
        # Real implementation would use the episodic store's search capabilities
        
        try:
            # Search for memories with planning-related content
            search_terms = [task_type, "planning", "action_sequence"]
            
            # For now, return empty list - would need to implement proper search
            # based on the actual episodic store interface
            return []
            
        except Exception as e:
            logger.warning(f"Error searching planning memories: {e}")
            return []
    
    def _convert_memory_to_experience(self, memory: Any) -> Optional[PlanningExperience]:
        """Convert episodic memory to PlanningExperience."""
        
        try:
            # Extract planning-specific information from memory
            # This would depend on how planning experiences are stored
            
            metadata = getattr(memory, 'metadata', {})
            planning_data = metadata.get('planning_data', {})
            
            if not planning_data:
                return None
            
            return PlanningExperience(
                experience_id=getattr(memory, 'memory_id', ''),
                task_type=planning_data.get('task_type', 'unknown'),
                action_sequence=planning_data.get('action_sequence', []),
                context_features=planning_data.get('context_features', {}),
                estimated_cost=planning_data.get('estimated_cost', 0),
                actual_cost=planning_data.get('actual_cost', 0),
                success=planning_data.get('success', False),
                outcome_quality=planning_data.get('outcome_quality', 0.5),
                timestamp=getattr(memory, 'timestamp', ''),
                metadata=metadata
            )
            
        except Exception as e:
            logger.warning(f"Error converting memory to experience: {e}")
            return None
    
    def _classify_task_type(self, task_description: str) -> str:
        """Classify the type of planning task."""

        task_lower = task_description.lower()

        # Check for research tasks first (more specific)
        if any(word in task_lower for word in ['research', 'search papers', 'find papers', 'investigate']):
            return 'research_task'
        elif any(word in task_lower for word in ['memory', 'remember', 'recall', 'previous']):
            return 'memory_task'
        elif any(word in task_lower for word in ['synthesize', 'combine', 'merge', 'integrate']):
            return 'synthesis_task'
        elif any(word in task_lower for word in ['document', 'paper', 'file', 'analyze']):
            return 'document_analysis'
        else:
            return 'general_task'
    
    def _extract_context_features(self, state: PlanningState) -> Dict[str, Any]:
        """Extract context features for similarity matching."""
        
        features = {
            'has_documents': state.has_document_context(),
            'has_memory': state.has_memory_context(),
            'has_conversation': state.has_conversation_context(),
            'action_count': len(state.action_history),
            'task_length': len(state.task_description.split())
        }
        
        # Add document-specific features
        if state.document_context:
            features['document_count'] = len(state.document_context)
            features['document_types'] = list(set(
                doc.get('type', 'unknown') for doc in state.document_context.values()
            ))
        
        return features
    
    def _is_experience_similar(self, state: PlanningState, experience: PlanningExperience) -> bool:
        """Check if a past experience is similar to the current state."""
        
        # Task type similarity
        current_task_type = self._classify_task_type(state.task_description)
        if current_task_type != experience.task_type:
            return False
        
        # Context similarity
        current_features = self._extract_context_features(state)
        similarity_score = self._calculate_context_similarity(current_features, experience.context_features)
        
        return similarity_score >= self.similarity_threshold
    
    def _calculate_context_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Calculate similarity between context features."""
        
        # Simple similarity calculation
        common_keys = set(features1.keys()).intersection(set(features2.keys()))
        
        if not common_keys:
            return 0.0
        
        matches = 0
        for key in common_keys:
            if features1[key] == features2[key]:
                matches += 1
        
        return matches / len(common_keys)
    
    def _calculate_experience_adjustment(self, 
                                       state: PlanningState, 
                                       experiences: List[PlanningExperience]) -> Tuple[int, float, str]:
        """Calculate heuristic adjustment based on past experiences."""
        
        if not experiences:
            return 0, 0.0, "No similar experiences found"
        
        # Calculate average estimation error from past experiences
        estimation_errors = []
        successful_experiences = []
        
        for exp in experiences:
            if exp.actual_cost > 0:  # Valid actual cost
                error = exp.actual_cost - exp.estimated_cost
                estimation_errors.append(error)
                
                if exp.success:
                    successful_experiences.append(exp)
        
        if not estimation_errors:
            return 0, 0.0, "No experiences with valid cost data"
        
        # Calculate adjustment
        avg_error = sum(estimation_errors) / len(estimation_errors)
        
        # Weight by success rate
        success_rate = len(successful_experiences) / len(experiences)
        confidence = min(0.9, success_rate * len(experiences) / 10.0)  # More experiences = higher confidence
        
        # Apply conservative adjustment
        adjustment = int(avg_error * 0.5)  # Conservative 50% of historical error
        
        reasoning = (
            f"Based on {len(experiences)} similar experiences: "
            f"avg error {avg_error:.1f}, success rate {success_rate:.1%}, "
            f"adjustment {adjustment}"
        )
        
        return adjustment, confidence, reasoning
    
    def record_planning_outcome(self, 
                              state: PlanningState,
                              estimated_cost: int,
                              actual_cost: int,
                              success: bool,
                              outcome_quality: float = 0.5):
        """Record a planning outcome for future learning."""
        
        if not (self.enable_experience_learning and self.episodic_store):
            return
        
        try:
            # Create planning experience
            experience = PlanningExperience(
                experience_id=self._generate_experience_id(state),
                task_type=self._classify_task_type(state.task_description),
                action_sequence=state.action_history,
                context_features=self._extract_context_features(state),
                estimated_cost=estimated_cost,
                actual_cost=actual_cost,
                success=success,
                outcome_quality=outcome_quality,
                timestamp=datetime.now().isoformat(),
                metadata={
                    'task_description': state.task_description,
                    'current_observation': state.current_observation
                }
            )
            
            # Store in episodic memory
            self._store_planning_experience(experience)
            
            # Update learning statistics
            if success:
                self.successful_adjustments += 1
            
            logger.info(f"Recorded planning outcome: {experience.experience_id}")
            
        except Exception as e:
            logger.warning(f"Error recording planning outcome: {e}")
    
    def _store_planning_experience(self, experience: PlanningExperience):
        """Store planning experience in episodic memory."""
        
        # This would integrate with SAM's episodic memory store
        # For now, we'll create a simplified storage approach
        
        try:
            # Create episodic memory entry
            memory = EpisodicMemory(
                memory_id=experience.experience_id,
                session_id="planning_session",
                user_id="system",
                timestamp=experience.timestamp,
                interaction_type=InteractionType.QUERY,
                query=f"Planning task: {experience.task_type}",
                context={'planning_context': experience.context_features},
                response=f"Completed with {len(experience.action_sequence)} actions",
                reasoning_chain=[],
                active_profile="planning",
                outcome_type=OutcomeType.SUCCESS if experience.success else OutcomeType.FAILURE,
                confidence_score=experience.outcome_quality
            )
            
            # Add planning-specific metadata
            memory.meta_reasoning = {
                'planning_data': {
                    'task_type': experience.task_type,
                    'action_sequence': experience.action_sequence,
                    'context_features': experience.context_features,
                    'estimated_cost': experience.estimated_cost,
                    'actual_cost': experience.actual_cost,
                    'success': experience.success,
                    'outcome_quality': experience.outcome_quality
                }
            }
            
            # Store in episodic memory
            self.episodic_store.store_memory(memory)
            
        except Exception as e:
            logger.warning(f"Error storing planning experience: {e}")
    
    def _generate_experience_id(self, state: PlanningState) -> str:
        """Generate unique ID for planning experience."""
        content = f"{state.task_description}_{state.action_history}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _generate_experience_cache_key(self, state: PlanningState) -> str:
        """Generate cache key for experience lookup."""
        task_type = self._classify_task_type(state.task_description)
        context_hash = hashlib.md5(str(self._extract_context_features(state)).encode()).hexdigest()[:8]
        return f"{task_type}_{context_hash}"
    
    def _is_cache_valid(self, cache_time: datetime) -> bool:
        """Check if cache entry is still valid."""
        return (datetime.now() - cache_time).total_seconds() < (self._cache_expiry_hours * 3600)
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics about experience-based learning."""
        
        success_rate = (self.successful_adjustments / max(1, self.total_adjustments)) * 100
        
        return {
            'experience_learning_enabled': self.enable_experience_learning,
            'total_adjustments': self.total_adjustments,
            'successful_adjustments': self.successful_adjustments,
            'success_rate': success_rate,
            'experience_queries': self.experience_queries,
            'cache_size': len(self._experience_cache),
            'experience_weight': self.experience_weight,
            'similarity_threshold': self.similarity_threshold
        }
