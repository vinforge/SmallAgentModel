#!/usr/bin/env python3
"""
Phase 6: Feedback & Correction Loop for SAM
Handles user feedback, corrections, and continuous learning from interactions.

This system enables SAM to:
1. Accept and process user feedback on response quality
2. Store corrections alongside input and profile context
3. Learn from error patterns and improve over time
4. Refine scoring logic using feedback data
5. Provide adaptive responses based on learned corrections
"""

import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter

# Import related components
from memory.episodic_store import EpisodicMemoryStore, EpisodicMemory, InteractionType, OutcomeType
from profiles.user_modeler import UserModelingEngine, UserModel

logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """Types of user feedback."""
    SATISFACTION_RATING = "satisfaction_rating"
    CORRECTION = "correction"
    PREFERENCE_UPDATE = "preference_update"
    ERROR_REPORT = "error_report"
    SUGGESTION = "suggestion"

class CorrectionType(Enum):
    """Types of corrections users can provide."""
    FACTUAL_ERROR = "factual_error"
    TONE_ADJUSTMENT = "tone_adjustment"
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"
    CLARITY = "clarity"
    BIAS_CORRECTION = "bias_correction"

class LearningPriority(Enum):
    """Priority levels for learning from feedback."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class FeedbackEntry:
    """Represents a single feedback entry from a user."""
    feedback_id: str
    memory_id: str  # Links to the original interaction
    user_id: str
    session_id: str
    timestamp: str
    
    # Feedback details
    feedback_type: FeedbackType
    rating: Optional[float] = None  # 0.0 to 1.0 for satisfaction ratings
    correction_text: Optional[str] = None
    correction_type: Optional[CorrectionType] = None
    
    # Context
    original_query: str = ""
    original_response: str = ""
    active_profile: str = ""
    
    # Learning metadata
    learning_priority: LearningPriority = LearningPriority.MEDIUM
    processed: bool = False
    applied_to_model: bool = False
    
    # Validation
    user_verified: bool = False
    expert_reviewed: bool = False

@dataclass
class CorrectionPattern:
    """Represents a pattern in user corrections."""
    pattern_id: str
    pattern_type: str
    description: str
    frequency: int
    confidence: float
    
    # Pattern details
    common_triggers: List[str]
    typical_corrections: List[str]
    affected_profiles: List[str]
    
    # Learning insights
    suggested_improvements: List[str]
    model_adjustments: Dict[str, Any]

@dataclass
class LearningInsight:
    """Represents an insight learned from feedback patterns."""
    insight_id: str
    insight_type: str
    description: str
    confidence: float
    evidence_count: int
    
    # Actionable recommendations
    recommended_actions: List[str]
    profile_adjustments: Dict[str, Dict[str, float]]
    dimension_weight_changes: Dict[str, float]
    
    # Implementation status
    implemented: bool = False
    implementation_date: Optional[str] = None
    effectiveness_score: Optional[float] = None

class FeedbackHandler:
    """
    Advanced feedback and correction system that learns from user interactions
    and continuously improves SAM's responses.
    """
    
    def __init__(self, episodic_store: EpisodicMemoryStore, user_modeler: UserModelingEngine):
        """Initialize the feedback handler."""
        self.episodic_store = episodic_store
        self.user_modeler = user_modeler
        
        # Feedback storage
        self.feedback_store = []  # In production, this would be a database
        self.correction_patterns = []
        self.learning_insights = []
        
        # Learning parameters
        self.min_feedback_for_pattern = 3
        self.correction_confidence_threshold = 0.7
        self.learning_rate = 0.1
        
        # Feedback processing settings
        self.auto_apply_high_confidence_corrections = True
        self.require_expert_review_for_critical = True
        
        logger.info("Feedback Handler initialized")
    
    def submit_feedback(self, 
                       memory_id: str,
                       user_id: str,
                       feedback_type: FeedbackType,
                       rating: Optional[float] = None,
                       correction_text: Optional[str] = None,
                       correction_type: Optional[CorrectionType] = None) -> str:
        """Submit user feedback for a specific interaction."""
        try:
            # Get the original memory
            memories = self.episodic_store.retrieve_memories(user_id, limit=100)
            original_memory = next((m for m in memories if m.memory_id == memory_id), None)
            
            if not original_memory:
                logger.warning(f"Memory {memory_id} not found for feedback")
                return ""
            
            # Create feedback entry
            feedback_id = f"feedback_{int(time.time() * 1000)}"
            
            feedback = FeedbackEntry(
                feedback_id=feedback_id,
                memory_id=memory_id,
                user_id=user_id,
                session_id=original_memory.session_id,
                timestamp=datetime.now().isoformat(),
                feedback_type=feedback_type,
                rating=rating,
                correction_text=correction_text,
                correction_type=correction_type,
                original_query=original_memory.query,
                original_response=original_memory.response,
                active_profile=original_memory.active_profile,
                learning_priority=self._determine_learning_priority(feedback_type, rating, correction_type)
            )
            
            # Store feedback
            self.feedback_store.append(feedback)
            
            # Update the original memory with feedback
            self.episodic_store.update_feedback(
                memory_id=memory_id,
                user_feedback=correction_text or f"{feedback_type.value}: {rating}",
                user_satisfaction=rating,
                correction_applied=correction_text is not None
            )
            
            # Process feedback for immediate learning
            self._process_feedback_immediate(feedback)
            
            logger.info(f"Feedback submitted: {feedback_id}")
            return feedback_id
            
        except Exception as e:
            logger.error(f"Error submitting feedback: {e}")
            return ""
    
    def _determine_learning_priority(self, 
                                   feedback_type: FeedbackType,
                                   rating: Optional[float],
                                   correction_type: Optional[CorrectionType]) -> LearningPriority:
        """Determine learning priority for feedback."""
        
        # Critical priority for factual errors and bias corrections
        if correction_type in [CorrectionType.FACTUAL_ERROR, CorrectionType.BIAS_CORRECTION]:
            return LearningPriority.CRITICAL
        
        # High priority for very low satisfaction ratings
        if rating is not None and rating < 0.3:
            return LearningPriority.HIGH
        
        # High priority for corrections
        if feedback_type == FeedbackType.CORRECTION:
            return LearningPriority.HIGH
        
        # Medium priority for moderate dissatisfaction
        if rating is not None and rating < 0.6:
            return LearningPriority.MEDIUM
        
        # Low priority for general feedback
        return LearningPriority.LOW
    
    def _process_feedback_immediate(self, feedback: FeedbackEntry):
        """Process feedback for immediate learning opportunities."""
        try:
            # For critical feedback, flag for immediate attention
            if feedback.learning_priority == LearningPriority.CRITICAL:
                self._handle_critical_feedback(feedback)
            
            # For corrections, attempt to learn patterns
            if feedback.feedback_type == FeedbackType.CORRECTION and feedback.correction_text:
                self._analyze_correction_pattern(feedback)
            
            # Update user model with satisfaction data
            if feedback.rating is not None:
                self._update_user_satisfaction_model(feedback)
            
            feedback.processed = True
            
        except Exception as e:
            logger.error(f"Error processing immediate feedback: {e}")
    
    def _handle_critical_feedback(self, feedback: FeedbackEntry):
        """Handle critical feedback that requires immediate attention."""
        logger.warning(f"Critical feedback received: {feedback.feedback_id}")
        
        # For factual errors, flag the response for review
        if feedback.correction_type == CorrectionType.FACTUAL_ERROR:
            self._flag_factual_error(feedback)
        
        # For bias corrections, analyze for systemic issues
        if feedback.correction_type == CorrectionType.BIAS_CORRECTION:
            self._analyze_bias_pattern(feedback)
    
    def _flag_factual_error(self, feedback: FeedbackEntry):
        """Flag a factual error for review and correction."""
        # In production, this would trigger alerts and review processes
        logger.error(f"Factual error flagged in memory {feedback.memory_id}: {feedback.correction_text}")
        
        # Create learning insight for factual accuracy improvement
        insight = LearningInsight(
            insight_id=f"factual_error_{feedback.feedback_id}",
            insight_type="factual_accuracy",
            description=f"Factual error identified: {feedback.correction_text}",
            confidence=0.9,
            evidence_count=1,
            recommended_actions=[
                "Review source credibility requirements",
                "Enhance fact-checking processes",
                "Update knowledge base with correct information"
            ],
            profile_adjustments={
                feedback.active_profile: {"credibility": 1.2, "utility": 0.9}
            }
        )
        
        self.learning_insights.append(insight)
    
    def _analyze_bias_pattern(self, feedback: FeedbackEntry):
        """Analyze bias correction for systemic patterns."""
        # Look for similar bias patterns in recent feedback
        recent_bias_feedback = [
            f for f in self.feedback_store 
            if f.correction_type == CorrectionType.BIAS_CORRECTION 
            and f.user_id == feedback.user_id
            and self._is_recent(f.timestamp, days=30)
        ]
        
        if len(recent_bias_feedback) >= 2:
            # Pattern detected - create learning insight
            insight = LearningInsight(
                insight_id=f"bias_pattern_{feedback.user_id}",
                insight_type="bias_correction",
                description=f"Recurring bias pattern detected for user {feedback.user_id}",
                confidence=0.8,
                evidence_count=len(recent_bias_feedback),
                recommended_actions=[
                    "Review response generation for bias",
                    "Enhance perspective diversity",
                    "Implement bias detection checks"
                ],
                profile_adjustments={
                    feedback.active_profile: {"clarity": 1.1, "credibility": 1.1}
                }
            )
            
            self.learning_insights.append(insight)
    
    def _analyze_correction_pattern(self, feedback: FeedbackEntry):
        """Analyze correction for learning patterns."""
        # Look for similar corrections from this user
        user_corrections = [
            f for f in self.feedback_store 
            if f.user_id == feedback.user_id 
            and f.feedback_type == FeedbackType.CORRECTION
            and f.correction_type == feedback.correction_type
        ]
        
        if len(user_corrections) >= self.min_feedback_for_pattern:
            # Create or update correction pattern
            pattern_id = f"correction_pattern_{feedback.user_id}_{feedback.correction_type.value}"
            
            existing_pattern = next((p for p in self.correction_patterns if p.pattern_id == pattern_id), None)
            
            if existing_pattern:
                existing_pattern.frequency += 1
                existing_pattern.typical_corrections.append(feedback.correction_text)
            else:
                pattern = CorrectionPattern(
                    pattern_id=pattern_id,
                    pattern_type=feedback.correction_type.value,
                    description=f"User frequently corrects {feedback.correction_type.value}",
                    frequency=len(user_corrections),
                    confidence=min(0.9, len(user_corrections) / 10),
                    common_triggers=[feedback.original_query],
                    typical_corrections=[feedback.correction_text],
                    affected_profiles=[feedback.active_profile],
                    suggested_improvements=[
                        f"Improve {feedback.correction_type.value} in responses",
                        f"Adjust {feedback.active_profile} profile for better {feedback.correction_type.value}"
                    ],
                    model_adjustments={}
                )
                
                self.correction_patterns.append(pattern)
    
    def _update_user_satisfaction_model(self, feedback: FeedbackEntry):
        """Update user satisfaction model with new rating."""
        # This would integrate with the user modeling engine
        # to update satisfaction patterns and preferences
        pass
    
    def _is_recent(self, timestamp_str: str, days: int = 7) -> bool:
        """Check if timestamp is within recent days."""
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            cutoff = datetime.now() - timedelta(days=days)
            return timestamp >= cutoff
        except Exception:
            return False
    
    def analyze_feedback_patterns(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze feedback patterns for insights."""
        try:
            # Filter feedback by user if specified
            feedback_to_analyze = self.feedback_store
            if user_id:
                feedback_to_analyze = [f for f in feedback_to_analyze if f.user_id == user_id]
            
            if not feedback_to_analyze:
                return {"message": "No feedback data available"}
            
            analysis = {
                "total_feedback": len(feedback_to_analyze),
                "feedback_types": Counter(f.feedback_type.value for f in feedback_to_analyze),
                "correction_types": Counter(f.correction_type.value for f in feedback_to_analyze if f.correction_type),
                "average_satisfaction": np.mean([f.rating for f in feedback_to_analyze if f.rating is not None]),
                "learning_priorities": Counter(f.learning_priority.value for f in feedback_to_analyze),
                "patterns_detected": len(self.correction_patterns),
                "insights_generated": len(self.learning_insights)
            }
            
            # Recent trends
            recent_feedback = [f for f in feedback_to_analyze if self._is_recent(f.timestamp, days=7)]
            if recent_feedback:
                analysis["recent_trends"] = {
                    "feedback_count_last_7_days": len(recent_feedback),
                    "average_satisfaction_recent": np.mean([f.rating for f in recent_feedback if f.rating is not None]),
                    "most_common_correction_recent": Counter(f.correction_type.value for f in recent_feedback if f.correction_type).most_common(1)
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing feedback patterns: {e}")
            return {"error": str(e)}
    
    def get_learning_recommendations(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get learning recommendations based on feedback analysis."""
        recommendations = []
        
        try:
            # Get user-specific insights if user_id provided
            relevant_insights = self.learning_insights
            if user_id:
                # Filter insights relevant to this user
                user_feedback = [f for f in self.feedback_store if f.user_id == user_id]
                relevant_memory_ids = {f.memory_id for f in user_feedback}
                relevant_insights = [i for i in self.learning_insights if any(
                    action for action in i.recommended_actions 
                    if user_id in action or any(mid in action for mid in relevant_memory_ids)
                )]
            
            # Convert insights to recommendations
            for insight in relevant_insights:
                if not insight.implemented:
                    recommendations.append({
                        "insight_id": insight.insight_id,
                        "type": insight.insight_type,
                        "description": insight.description,
                        "confidence": insight.confidence,
                        "actions": insight.recommended_actions,
                        "profile_adjustments": insight.profile_adjustments,
                        "priority": "high" if insight.confidence > 0.8 else "medium"
                    })
            
            # Sort by confidence
            recommendations.sort(key=lambda x: x["confidence"], reverse=True)
            
            return recommendations[:10]  # Return top 10 recommendations
            
        except Exception as e:
            logger.error(f"Error getting learning recommendations: {e}")
            return []
    
    def apply_learning_insight(self, insight_id: str) -> bool:
        """Apply a learning insight to improve the system."""
        try:
            insight = next((i for i in self.learning_insights if i.insight_id == insight_id), None)
            
            if not insight:
                logger.warning(f"Insight {insight_id} not found")
                return False
            
            # Mark as implemented
            insight.implemented = True
            insight.implementation_date = datetime.now().isoformat()
            
            # In production, this would apply the actual model adjustments
            logger.info(f"Applied learning insight: {insight_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error applying learning insight: {e}")
            return False

    def get_user_feedback_history(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get feedback history for a specific user."""
        try:
            user_feedback = []

            for feedback in self.feedback_store:
                if feedback.user_id == user_id:
                    user_feedback.append({
                        'feedback_id': feedback.feedback_id,
                        'memory_id': feedback.memory_id,
                        'timestamp': feedback.timestamp,
                        'feedback_type': feedback.feedback_type.value,
                        'rating': feedback.rating,
                        'correction_text': feedback.correction_text,
                        'correction_type': feedback.correction_type.value if feedback.correction_type else None,
                        'original_query': feedback.original_query,
                        'original_response': feedback.original_response,
                        'learning_priority': feedback.learning_priority.value
                    })

            # Sort by timestamp (most recent first) and limit
            user_feedback.sort(key=lambda x: x['timestamp'], reverse=True)
            return user_feedback[:limit]

        except Exception as e:
            logger.error(f"Error getting user feedback history: {e}")
            return []


# Global feedback handler instance
_feedback_handler = None

def get_feedback_handler() -> FeedbackHandler:
    """Get the global feedback handler instance."""
    global _feedback_handler
    if _feedback_handler is None:
        # Create with mock dependencies for now
        from learning.episodic_memory import EpisodicMemoryStore
        from learning.user_modeling import UserModelingEngine

        try:
            episodic_store = EpisodicMemoryStore()
            user_modeler = UserModelingEngine()
            _feedback_handler = FeedbackHandler(episodic_store, user_modeler)
        except Exception as e:
            # Create minimal feedback handler for testing
            _feedback_handler = MinimalFeedbackHandler()

    return _feedback_handler

class MinimalFeedbackHandler:
    """Minimal feedback handler for testing when full system is not available."""

    def __init__(self):
        self.feedback_store = []
        self.logger = logging.getLogger(__name__)

    def submit_feedback(self, memory_id: str, user_id: str, feedback_type,
                       rating: float = None, correction_text: str = None,
                       correction_type = None) -> str:
        """Submit feedback (minimal implementation)."""
        feedback_id = f"feedback_{len(self.feedback_store)}"

        feedback = {
            'feedback_id': feedback_id,
            'memory_id': memory_id,
            'user_id': user_id,
            'feedback_type': feedback_type,
            'rating': rating,
            'correction_text': correction_text,
            'correction_type': correction_type,
            'timestamp': datetime.now().isoformat()
        }

        self.feedback_store.append(feedback)
        self.logger.info(f"Feedback submitted: {feedback_id}")
        return feedback_id

    def get_user_feedback_history(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get feedback history for a user."""
        user_feedback = [f for f in self.feedback_store if f['user_id'] == user_id]
        return user_feedback[-limit:] if user_feedback else []

# Convenience functions
def create_feedback_handler(episodic_store: EpisodicMemoryStore,
                          user_modeler: UserModelingEngine) -> FeedbackHandler:
    """Create and return a feedback handler instance."""
    return FeedbackHandler(episodic_store, user_modeler)

def submit_user_feedback(handler: FeedbackHandler,
                        memory_id: str,
                        user_id: str,
                        rating: float,
                        correction: Optional[str] = None) -> str:
    """Convenience function to submit user feedback."""
    feedback_type = FeedbackType.CORRECTION if correction else FeedbackType.SATISFACTION_RATING
    correction_type = CorrectionType.COMPLETENESS if correction else None
    
    return handler.submit_feedback(
        memory_id=memory_id,
        user_id=user_id,
        feedback_type=feedback_type,
        rating=rating,
        correction_text=correction,
        correction_type=correction_type
    )
