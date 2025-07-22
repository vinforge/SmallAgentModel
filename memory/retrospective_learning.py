"""
Retrospective Learning Loop for SAM
Improves responses by reflecting on past reasoning and learning from experience.

Sprint 6 Task 4: Retrospective Learning Loop
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .memory_manager import LongTermMemoryManager
from reasoning.self_decide_framework import SelfDecideSession
from reasoning.tool_executor import ToolResponse

logger = logging.getLogger(__name__)

class ReflectionType(Enum):
    """Types of reflection analysis."""
    CONFIDENCE_CHECK = "confidence_check"
    TOOL_EFFECTIVENESS = "tool_effectiveness"
    RESPONSE_QUALITY = "response_quality"
    KNOWLEDGE_GAP = "knowledge_gap"
    USER_SATISFACTION = "user_satisfaction"

@dataclass
class ReflectionResult:
    """Result of a reflection analysis."""
    reflection_type: ReflectionType
    confidence_score: float
    issues_identified: List[str]
    improvement_suggestions: List[str]
    should_retry: bool
    alternative_approach: Optional[str]
    learning_points: List[str]

@dataclass
class LearningEntry:
    """Entry in the learning history."""
    learning_id: str
    timestamp: str
    original_query: str
    original_response: str
    reflection_results: List[ReflectionResult]
    improvement_made: bool
    improved_response: Optional[str]
    user_feedback: Optional[str]
    effectiveness_score: float

class RetrospectiveLearningManager:
    """
    Manages SAM's retrospective learning and self-improvement capabilities.
    """
    
    def __init__(self, memory_manager: LongTermMemoryManager):
        """
        Initialize the retrospective learning manager.
        
        Args:
            memory_manager: Long-term memory manager instance
        """
        self.memory_manager = memory_manager
        
        # Learning configuration
        self.config = {
            'confidence_threshold': 0.65,  # Retry threshold
            'reflection_enabled': True,
            'auto_retry_enabled': True,
            'learning_rate': 0.1,  # How much to adjust based on feedback
            'max_retry_attempts': 2
        }
        
        # Learning history
        self.learning_history: List[LearningEntry] = []
        
        logger.info("Retrospective learning manager initialized")
    
    def reflect_on_response(self, session: SelfDecideSession, 
                           tool_responses: List[ToolResponse] = None,
                           user_feedback: Optional[str] = None) -> List[ReflectionResult]:
        """
        Perform post-response reflection analysis.
        
        Args:
            session: SELF-DECIDE session to reflect on
            tool_responses: Tool responses from the session
            user_feedback: Optional user feedback
            
        Returns:
            List of reflection results
        """
        try:
            reflections = []
            
            # Confidence check reflection
            confidence_reflection = self._reflect_on_confidence(session)
            reflections.append(confidence_reflection)
            
            # Tool effectiveness reflection
            if tool_responses:
                tool_reflection = self._reflect_on_tool_effectiveness(session, tool_responses)
                reflections.append(tool_reflection)
            
            # Response quality reflection
            quality_reflection = self._reflect_on_response_quality(session)
            reflections.append(quality_reflection)
            
            # Knowledge gap reflection
            gap_reflection = self._reflect_on_knowledge_gaps(session)
            reflections.append(gap_reflection)
            
            # User satisfaction reflection (if feedback available)
            if user_feedback:
                satisfaction_reflection = self._reflect_on_user_satisfaction(user_feedback)
                reflections.append(satisfaction_reflection)
            
            # Store reflection in memory
            self._store_reflection_in_memory(session, reflections)
            
            logger.debug(f"Completed reflection on session {session.session_id}: {len(reflections)} analyses")
            return reflections
            
        except Exception as e:
            logger.error(f"Error in reflection analysis: {e}")
            return []
    
    def should_retry_response(self, reflections: List[ReflectionResult]) -> Tuple[bool, Optional[str]]:
        """
        Determine if response should be retried based on reflections.
        
        Args:
            reflections: List of reflection results
            
        Returns:
            Tuple of (should_retry, alternative_approach)
        """
        try:
            if not self.config['auto_retry_enabled']:
                return False, None
            
            # Check if any reflection suggests retry
            should_retry = any(reflection.should_retry for reflection in reflections)
            
            if should_retry:
                # Find the best alternative approach
                alternative_approaches = [
                    reflection.alternative_approach 
                    for reflection in reflections 
                    if reflection.alternative_approach
                ]
                
                best_approach = alternative_approaches[0] if alternative_approaches else None
                
                logger.info(f"Reflection suggests retry with approach: {best_approach}")
                return True, best_approach
            
            return False, None
            
        except Exception as e:
            logger.error(f"Error determining retry: {e}")
            return False, None
    
    def learn_from_feedback(self, learning_id: str, user_feedback: str, 
                           effectiveness_score: float):
        """
        Learn from user feedback on a previous response.
        
        Args:
            learning_id: ID of the learning entry
            user_feedback: User's feedback
            effectiveness_score: Score from 0.0 to 1.0
        """
        try:
            # Find the learning entry
            learning_entry = next(
                (entry for entry in self.learning_history if entry.learning_id == learning_id),
                None
            )
            
            if learning_entry:
                learning_entry.user_feedback = user_feedback
                learning_entry.effectiveness_score = effectiveness_score
                
                # Extract learning points from feedback
                learning_points = self._extract_learning_points(user_feedback, effectiveness_score)
                
                # Store learning in memory
                learning_content = f"User feedback: {user_feedback} (effectiveness: {effectiveness_score:.2f})"
                self.memory_manager.store_memory(
                    content=learning_content,
                    content_type='user_feedback',
                    tags=['learning', 'feedback', 'improvement'],
                    importance_score=0.8,
                    metadata={
                        'learning_id': learning_id,
                        'effectiveness_score': effectiveness_score,
                        'learning_points': learning_points
                    }
                )
                
                logger.info(f"Learned from feedback for {learning_id}: {effectiveness_score:.2f}")
            
        except Exception as e:
            logger.error(f"Error learning from feedback: {e}")
    
    def get_improvement_suggestions(self, query_type: str, 
                                  tools_used: List[str] = None) -> List[str]:
        """
        Get improvement suggestions based on past learning.
        
        Args:
            query_type: Type of query being processed
            tools_used: Tools being used
            
        Returns:
            List of improvement suggestions
        """
        try:
            suggestions = []
            
            # Search for relevant learning experiences
            search_query = f"{query_type} {' '.join(tools_used or [])}"
            learning_memories = self.memory_manager.search_memories(
                query=search_query,
                content_type='reflection',
                top_k=5
            )
            
            for result in learning_memories:
                memory = result.memory_entry
                learning_points = memory.metadata.get('learning_points', [])
                suggestions.extend(learning_points)
            
            # Remove duplicates and return top suggestions
            unique_suggestions = list(set(suggestions))
            return unique_suggestions[:5]
            
        except Exception as e:
            logger.error(f"Error getting improvement suggestions: {e}")
            return []
    
    def update_memory_with_contradiction(self, memory_id: str, 
                                       contradictory_info: str) -> bool:
        """
        Handle contradictory information by updating or flagging memory.
        
        Args:
            memory_id: ID of the memory with contradictory information
            contradictory_info: The contradictory information
            
        Returns:
            True if memory was updated, False otherwise
        """
        try:
            memory = self.memory_manager.recall_memory(memory_id)
            if not memory:
                return False
            
            # Add contradiction flag to metadata
            memory.metadata['contradiction_detected'] = True
            memory.metadata['contradictory_info'] = contradictory_info
            memory.metadata['contradiction_timestamp'] = datetime.now().isoformat()
            
            # Reduce importance score
            memory.importance_score *= 0.8
            
            # Add contradiction tag
            if 'contradiction' not in memory.tags:
                memory.tags.append('contradiction')
            
            # Store contradiction analysis in memory
            contradiction_content = f"Contradiction detected in memory {memory_id}: {contradictory_info}"
            self.memory_manager.store_memory(
                content=contradiction_content,
                content_type='contradiction',
                tags=['contradiction', 'analysis', 'memory_update'],
                metadata={
                    'original_memory_id': memory_id,
                    'contradictory_info': contradictory_info
                }
            )
            
            logger.info(f"Flagged memory {memory_id} with contradiction")
            return True
            
        except Exception as e:
            logger.error(f"Error updating memory with contradiction: {e}")
            return False
    
    def _reflect_on_confidence(self, session: SelfDecideSession) -> ReflectionResult:
        """Reflect on the confidence level of the response."""
        confidence = session.confidence_score
        issues = []
        suggestions = []
        should_retry = False
        alternative_approach = None
        learning_points = []
        
        if confidence < self.config['confidence_threshold']:
            issues.append(f"Low confidence score: {confidence:.2f}")
            suggestions.append("Consider using additional tools or seeking more information")
            should_retry = True
            alternative_approach = "retry_with_more_tools"
            learning_points.append("Low confidence may indicate insufficient information gathering")
        
        if confidence < 0.3:
            issues.append("Very low confidence indicates significant uncertainty")
            suggestions.append("Acknowledge uncertainty explicitly in response")
            learning_points.append("Very low confidence requires explicit uncertainty communication")
        
        return ReflectionResult(
            reflection_type=ReflectionType.CONFIDENCE_CHECK,
            confidence_score=confidence,
            issues_identified=issues,
            improvement_suggestions=suggestions,
            should_retry=should_retry,
            alternative_approach=alternative_approach,
            learning_points=learning_points
        )
    
    def _reflect_on_tool_effectiveness(self, session: SelfDecideSession, 
                                     tool_responses: List[ToolResponse]) -> ReflectionResult:
        """Reflect on the effectiveness of tools used."""
        issues = []
        suggestions = []
        should_retry = False
        alternative_approach = None
        learning_points = []
        
        # Analyze tool success rates
        successful_tools = [tr for tr in tool_responses if tr.success]
        failed_tools = [tr for tr in tool_responses if not tr.success]
        
        success_rate = len(successful_tools) / len(tool_responses) if tool_responses else 1.0
        
        if success_rate < 0.5:
            issues.append(f"Low tool success rate: {success_rate:.2f}")
            suggestions.append("Consider alternative tools or different approach")
            should_retry = True
            alternative_approach = "retry_with_different_tools"
            learning_points.append("Low tool success rate indicates need for tool selection improvement")
        
        if failed_tools:
            failed_tool_names = [tr.tool_name for tr in failed_tools]
            issues.append(f"Failed tools: {', '.join(failed_tool_names)}")
            suggestions.append(f"Avoid using {', '.join(failed_tool_names)} for similar queries")
            learning_points.append(f"Tools {', '.join(failed_tool_names)} may not be suitable for this query type")
        
        # Check for tool execution times
        slow_tools = [tr for tr in tool_responses if tr.execution_time_ms > 10000]  # > 10 seconds
        if slow_tools:
            slow_tool_names = [tr.tool_name for tr in slow_tools]
            suggestions.append(f"Consider faster alternatives to {', '.join(slow_tool_names)}")
            learning_points.append("Some tools may be too slow for interactive use")
        
        return ReflectionResult(
            reflection_type=ReflectionType.TOOL_EFFECTIVENESS,
            confidence_score=success_rate,
            issues_identified=issues,
            improvement_suggestions=suggestions,
            should_retry=should_retry,
            alternative_approach=alternative_approach,
            learning_points=learning_points
        )
    
    def _reflect_on_response_quality(self, session: SelfDecideSession) -> ReflectionResult:
        """Reflect on the quality of the response."""
        issues = []
        suggestions = []
        should_retry = False
        alternative_approach = None
        learning_points = []
        
        response_length = len(session.final_answer)
        
        # Check response length
        if response_length < 50:
            issues.append("Response may be too brief")
            suggestions.append("Provide more detailed explanation")
            learning_points.append("Brief responses may not fully address user needs")
        elif response_length > 1000:
            issues.append("Response may be too verbose")
            suggestions.append("Consider more concise communication")
            learning_points.append("Overly long responses may overwhelm users")
        
        # Check if response addresses the original query
        query_words = set(session.original_query.lower().split())
        response_words = set(session.final_answer.lower().split())
        overlap = len(query_words & response_words) / len(query_words) if query_words else 0
        
        if overlap < 0.3:
            issues.append("Response may not directly address the query")
            suggestions.append("Ensure response directly answers the user's question")
            should_retry = True
            alternative_approach = "refocus_on_query"
            learning_points.append("Responses should directly address user queries")
        
        # Check for reasoning steps completion
        if len(session.reasoning_steps) < 8:  # Less than most SELF-DECIDE steps
            issues.append("Incomplete reasoning process")
            suggestions.append("Complete all reasoning steps for better quality")
            learning_points.append("Complete reasoning processes lead to better responses")
        
        quality_score = min(1.0, (overlap + (response_length / 500)) / 2)
        
        return ReflectionResult(
            reflection_type=ReflectionType.RESPONSE_QUALITY,
            confidence_score=quality_score,
            issues_identified=issues,
            improvement_suggestions=suggestions,
            should_retry=should_retry,
            alternative_approach=alternative_approach,
            learning_points=learning_points
        )
    
    def _reflect_on_knowledge_gaps(self, session: SelfDecideSession) -> ReflectionResult:
        """Reflect on knowledge gaps identified during reasoning."""
        issues = []
        suggestions = []
        should_retry = False
        alternative_approach = None
        learning_points = []
        
        gap_count = len(session.knowledge_gaps)
        
        if gap_count > 3:
            issues.append(f"Many knowledge gaps identified: {gap_count}")
            suggestions.append("Consider additional information gathering")
            should_retry = True
            alternative_approach = "gather_more_information"
            learning_points.append("Multiple knowledge gaps indicate need for more comprehensive information gathering")
        
        # Analyze gap types
        gap_types = [gap.gap_type for gap in session.knowledge_gaps if hasattr(gap, 'gap_type')]
        if 'critical' in gap_types:
            issues.append("Critical knowledge gaps present")
            suggestions.append("Address critical gaps before responding")
            should_retry = True
            learning_points.append("Critical knowledge gaps must be addressed for accurate responses")
        
        gap_confidence = 1.0 - (gap_count * 0.1)  # Reduce confidence based on gap count
        
        return ReflectionResult(
            reflection_type=ReflectionType.KNOWLEDGE_GAP,
            confidence_score=max(0.0, gap_confidence),
            issues_identified=issues,
            improvement_suggestions=suggestions,
            should_retry=should_retry,
            alternative_approach=alternative_approach,
            learning_points=learning_points
        )
    
    def _reflect_on_user_satisfaction(self, user_feedback: str) -> ReflectionResult:
        """Reflect on user satisfaction based on feedback."""
        issues = []
        suggestions = []
        should_retry = False
        alternative_approach = None
        learning_points = []
        
        feedback_lower = user_feedback.lower()
        
        # Analyze sentiment (simplified)
        positive_indicators = ['good', 'great', 'helpful', 'thanks', 'perfect', 'excellent']
        negative_indicators = ['bad', 'wrong', 'unhelpful', 'confusing', 'unclear', 'poor']
        
        positive_count = sum(1 for word in positive_indicators if word in feedback_lower)
        negative_count = sum(1 for word in negative_indicators if word in feedback_lower)
        
        if negative_count > positive_count:
            issues.append("Negative user feedback received")
            suggestions.append("Analyze feedback to improve future responses")
            learning_points.append("Negative feedback indicates areas for improvement")
        
        if 'unclear' in feedback_lower or 'confusing' in feedback_lower:
            issues.append("User found response unclear")
            suggestions.append("Provide clearer explanations in future")
            learning_points.append("Clarity is essential for user understanding")
        
        if 'wrong' in feedback_lower or 'incorrect' in feedback_lower:
            issues.append("User indicated response was incorrect")
            suggestions.append("Verify information accuracy")
            should_retry = True
            alternative_approach = "verify_and_correct"
            learning_points.append("Accuracy is critical for user trust")
        
        satisfaction_score = max(0.0, (positive_count - negative_count + 1) / 3)
        
        return ReflectionResult(
            reflection_type=ReflectionType.USER_SATISFACTION,
            confidence_score=satisfaction_score,
            issues_identified=issues,
            improvement_suggestions=suggestions,
            should_retry=should_retry,
            alternative_approach=alternative_approach,
            learning_points=learning_points
        )
    
    def _store_reflection_in_memory(self, session: SelfDecideSession, 
                                   reflections: List[ReflectionResult]):
        """Store reflection results in memory for future learning."""
        try:
            reflection_summary = f"Reflection on session {session.session_id}: "
            reflection_summary += f"{len(reflections)} analyses completed, "
            reflection_summary += f"confidence: {session.confidence_score:.2f}"
            
            # Collect all learning points
            all_learning_points = []
            for reflection in reflections:
                all_learning_points.extend(reflection.learning_points)
            
            # Store in memory
            self.memory_manager.store_memory(
                content=reflection_summary,
                content_type='reflection',
                tags=['reflection', 'learning', 'self_improvement'],
                importance_score=0.7,
                metadata={
                    'session_id': session.session_id,
                    'reflection_count': len(reflections),
                    'learning_points': all_learning_points,
                    'original_confidence': session.confidence_score
                }
            )
            
        except Exception as e:
            logger.error(f"Error storing reflection in memory: {e}")
    
    def _extract_learning_points(self, feedback: str, effectiveness_score: float) -> List[str]:
        """Extract learning points from user feedback."""
        learning_points = []
        
        feedback_lower = feedback.lower()
        
        if effectiveness_score < 0.3:
            learning_points.append("Response was not effective for user needs")
        elif effectiveness_score > 0.8:
            learning_points.append("Response was highly effective")
        
        if 'too long' in feedback_lower:
            learning_points.append("User prefers more concise responses")
        elif 'too short' in feedback_lower:
            learning_points.append("User prefers more detailed responses")
        
        if 'more examples' in feedback_lower:
            learning_points.append("User benefits from concrete examples")
        
        if 'technical' in feedback_lower:
            learning_points.append("User appreciates technical depth")
        
        return learning_points

# Global retrospective learning manager instance
_learning_manager = None

def get_learning_manager(memory_manager: LongTermMemoryManager) -> RetrospectiveLearningManager:
    """Get or create a global retrospective learning manager instance."""
    global _learning_manager
    
    if _learning_manager is None:
        _learning_manager = RetrospectiveLearningManager(memory_manager)
    
    return _learning_manager
