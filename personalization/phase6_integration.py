#!/usr/bin/env python3
"""
Phase 6: Integration System for SAM Personalization Engine
Integrates episodic memory, user modeling, feedback handling, and adaptive refinement.

This system provides:
1. Unified interface for all Phase 6 capabilities
2. Seamless integration with existing SAM response generation
3. Automatic learning and adaptation from user interactions
4. Personalized profile recommendations and refinements
5. Comprehensive memory and feedback management
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

# Import Phase 6 components
from memory.episodic_store import EpisodicMemoryStore, create_episodic_store, store_interaction_memory
from profiles.user_modeler import UserModelingEngine, create_user_modeler, analyze_user_preferences
from learning.feedback_handler import FeedbackHandler, create_feedback_handler, submit_user_feedback
from profiles.adaptive_refinement import AdaptiveProfileRefinement, create_adaptive_refinement

# Import Phase 5 for enhanced integration
try:
    from reasoning.phase5_integration import Phase5EnhancedResponse
    PHASE5_AVAILABLE = True
except ImportError:
    PHASE5_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class PersonalizedResponse:
    """Enhanced response with Phase 6 personalization capabilities."""
    # Original response data
    original_query: str
    response: str
    user_id: str
    session_id: str

    # Personalization enhancements
    recommended_profile: str
    personalization_applied: bool
    memory_stored: bool

    # Learning insights
    similar_past_queries: List[Dict[str, Any]]
    user_preferences_applied: List[str]
    profile_refinement_suggestions: List[Dict[str, Any]]

    # Metadata
    processing_time_ms: int
    personalization_confidence: float
    timestamp: str

    # Optional fields with defaults
    phase5_enhanced: Optional[Any] = None

class Phase6PersonalizationEngine:
    """
    Comprehensive personalization engine that integrates all Phase 6 capabilities
    to provide adaptive, learning-based personalized AI interactions.
    """
    
    def __init__(self, 
                 db_path: str = "memory_store/episodic_memory.db",
                 enable_auto_refinement: bool = True,
                 enable_feedback_learning: bool = True):
        """Initialize the Phase 6 personalization engine."""
        
        self.enable_auto_refinement = enable_auto_refinement
        self.enable_feedback_learning = enable_feedback_learning
        
        try:
            # Initialize core components
            self.episodic_store = create_episodic_store(db_path)
            self.user_modeler = create_user_modeler(self.episodic_store)
            self.feedback_handler = create_feedback_handler(self.episodic_store, self.user_modeler)
            self.adaptive_refinement = create_adaptive_refinement(
                self.episodic_store, self.user_modeler, self.feedback_handler
            )
            
            self.phase6_available = True
            logger.info("Phase 6 Personalization Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Phase 6 components: {e}")
            self.phase6_available = False
    
    def personalize_response(self,
                           query: str,
                           initial_response: str,
                           user_id: str,
                           session_id: str,
                           context: Optional[Dict[str, Any]] = None,
                           current_profile: Optional[str] = None) -> PersonalizedResponse:
        """
        Apply comprehensive personalization to a response using all Phase 6 capabilities.
        
        Args:
            query: User's query
            initial_response: Initial response from SAM
            user_id: Unique user identifier
            session_id: Session identifier
            context: Additional context including memory results, tool outputs, etc.
            current_profile: Currently active profile
            
        Returns:
            PersonalizedResponse with all Phase 6 enhancements applied
        """
        start_time = time.time()
        
        if not self.phase6_available:
            return self._create_fallback_response(query, initial_response, user_id, session_id)
        
        try:
            # Stage 1: Analyze user behavior and get recommendations
            recommended_profile = self.user_modeler.get_recommended_profile(user_id, context or {})
            
            # Stage 2: Check for similar past queries
            similar_queries = self.episodic_store.find_similar_queries(user_id, query, limit=3)
            similar_past_queries = [
                {
                    "query": mem.query,
                    "response_preview": mem.response[:100] + "...",
                    "satisfaction": mem.user_satisfaction,
                    "timestamp": mem.timestamp
                }
                for mem in similar_queries
            ]
            
            # Stage 3: Apply user preferences if available
            user_preferences_applied = []
            personalization_applied = False
            
            if current_profile != recommended_profile:
                user_preferences_applied.append(f"Recommended profile: {recommended_profile}")
                personalization_applied = True
            
            # Stage 4: Get profile refinement suggestions
            refinement_suggestions = []
            if self.enable_auto_refinement:
                opportunities = self.adaptive_refinement.detect_refinement_opportunities(user_id)
                refinement_suggestions = [
                    {
                        "type": opp["type"],
                        "description": opp["description"],
                        "confidence": opp["confidence"]
                    }
                    for opp in opportunities[:3]  # Top 3 suggestions
                ]
            
            # Stage 5: Store interaction in episodic memory
            memory_stored = False
            try:
                memory = self.episodic_store.create_memory_from_interaction(
                    user_id=user_id,
                    session_id=session_id,
                    query=query,
                    response=initial_response,
                    context=context or {},
                    active_profile=current_profile or recommended_profile
                )
                memory_stored = self.episodic_store.store_memory(memory)
            except Exception as e:
                logger.warning(f"Failed to store memory: {e}")
            
            # Stage 6: Calculate personalization confidence
            personalization_confidence = self._calculate_personalization_confidence(
                user_id, recommended_profile, similar_queries, refinement_suggestions
            )
            
            # Stage 7: Integrate with Phase 5 if available
            phase5_enhanced = None
            if PHASE5_AVAILABLE and context:
                try:
                    from reasoning.phase5_integration import enhance_sam_response
                    phase5_enhanced = enhance_sam_response(
                        query=query,
                        initial_response=initial_response,
                        context=context
                    )
                except Exception as e:
                    logger.warning(f"Phase 5 integration failed: {e}")
            
            processing_time = int((time.time() - start_time) * 1000)
            
            result = PersonalizedResponse(
                original_query=query,
                response=initial_response,
                user_id=user_id,
                session_id=session_id,
                recommended_profile=recommended_profile,
                personalization_applied=personalization_applied,
                memory_stored=memory_stored,
                similar_past_queries=similar_past_queries,
                user_preferences_applied=user_preferences_applied,
                profile_refinement_suggestions=refinement_suggestions,
                phase5_enhanced=phase5_enhanced,
                processing_time_ms=processing_time,
                personalization_confidence=personalization_confidence,
                timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"Response personalized for user {user_id} in {processing_time}ms")
            return result
            
        except Exception as e:
            logger.error(f"Error in personalization: {e}")
            return self._create_fallback_response(query, initial_response, user_id, session_id)
    
    def submit_feedback(self,
                       memory_id: str,
                       user_id: str,
                       rating: float,
                       correction: Optional[str] = None) -> bool:
        """Submit user feedback for learning and adaptation."""
        try:
            if not self.phase6_available or not self.enable_feedback_learning:
                return False
            
            feedback_id = submit_user_feedback(
                self.feedback_handler,
                memory_id=memory_id,
                user_id=user_id,
                rating=rating,
                correction=correction
            )
            
            # Trigger automatic refinement if enabled
            if self.enable_auto_refinement and rating < 0.6:
                self._trigger_automatic_refinement(user_id, rating)
            
            return bool(feedback_id)
            
        except Exception as e:
            logger.error(f"Error submitting feedback: {e}")
            return False
    
    def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user insights and analytics."""
        try:
            if not self.phase6_available:
                return {"error": "Phase 6 not available"}
            
            # Get user preferences analysis
            preferences = analyze_user_preferences(self.user_modeler, user_id)
            
            # Get feedback analytics
            feedback_analysis = self.feedback_handler.analyze_feedback_patterns(user_id)
            
            # Get refinement opportunities
            refinement_opportunities = self.adaptive_refinement.detect_refinement_opportunities(user_id)
            
            # Get episodic memory statistics
            memory_stats = self.episodic_store.get_user_statistics(user_id)
            
            return {
                "user_preferences": preferences,
                "feedback_analytics": feedback_analysis,
                "refinement_opportunities": refinement_opportunities,
                "memory_statistics": memory_stats,
                "personalization_status": {
                    "phase6_available": self.phase6_available,
                    "auto_refinement_enabled": self.enable_auto_refinement,
                    "feedback_learning_enabled": self.enable_feedback_learning
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting user insights: {e}")
            return {"error": str(e)}
    
    def apply_profile_refinement(self, user_id: str, profile_id: str) -> bool:
        """Apply automatic profile refinement for a user."""
        try:
            if not self.phase6_available or not self.enable_auto_refinement:
                return False
            
            from profiles.adaptive_refinement import RefinementTrigger
            
            refinement = self.adaptive_refinement.refine_profile(
                user_id=user_id,
                profile_id=profile_id,
                trigger=RefinementTrigger.MANUAL_REQUEST,
                auto_apply=True
            )
            
            return refinement is not None
            
        except Exception as e:
            logger.error(f"Error applying profile refinement: {e}")
            return False
    
    def get_similar_interactions(self, user_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get similar past interactions for context."""
        try:
            if not self.phase6_available:
                return []
            
            similar_memories = self.episodic_store.find_similar_queries(user_id, query, limit=limit)
            
            return [
                {
                    "memory_id": mem.memory_id,
                    "query": mem.query,
                    "response_preview": mem.response[:150] + "...",
                    "timestamp": mem.timestamp,
                    "satisfaction": mem.user_satisfaction,
                    "profile_used": mem.active_profile
                }
                for mem in similar_memories
            ]
            
        except Exception as e:
            logger.error(f"Error getting similar interactions: {e}")
            return []
    
    def _calculate_personalization_confidence(self,
                                            user_id: str,
                                            recommended_profile: str,
                                            similar_queries: List[Any],
                                            refinement_suggestions: List[Dict[str, Any]]) -> float:
        """Calculate confidence in personalization recommendations."""
        confidence_factors = []
        
        # Factor 1: User interaction history
        try:
            user_stats = self.episodic_store.get_user_statistics(user_id)
            interaction_factor = min(1.0, user_stats.get("total_interactions", 0) / 50)
            confidence_factors.append(interaction_factor)
        except Exception:
            confidence_factors.append(0.3)
        
        # Factor 2: Similar query availability
        similarity_factor = min(1.0, len(similar_queries) / 5)
        confidence_factors.append(similarity_factor)
        
        # Factor 3: Profile recommendation confidence
        if recommended_profile != "general":
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.5)
        
        # Factor 4: Refinement opportunity confidence
        if refinement_suggestions:
            avg_refinement_confidence = sum(s["confidence"] for s in refinement_suggestions) / len(refinement_suggestions)
            confidence_factors.append(avg_refinement_confidence)
        else:
            confidence_factors.append(0.6)
        
        # Calculate overall confidence
        overall_confidence = sum(confidence_factors) / len(confidence_factors)
        return min(1.0, max(0.0, overall_confidence))
    
    def _trigger_automatic_refinement(self, user_id: str, rating: float):
        """Trigger automatic profile refinement based on low satisfaction."""
        try:
            if rating < 0.4:  # Very low satisfaction
                opportunities = self.adaptive_refinement.detect_refinement_opportunities(user_id)
                
                for opp in opportunities[:1]:  # Apply top opportunity
                    if opp["confidence"] > 0.7:
                        self.adaptive_refinement.refine_profile(
                            user_id=user_id,
                            profile_id=opp["profile"],
                            trigger=opp["trigger"],
                            auto_apply=True
                        )
                        logger.info(f"Auto-applied refinement for user {user_id} due to low satisfaction")
                        break
        except Exception as e:
            logger.warning(f"Auto-refinement failed: {e}")
    
    def _create_fallback_response(self, query: str, response: str, user_id: str, session_id: str) -> PersonalizedResponse:
        """Create fallback response when Phase 6 is unavailable."""
        return PersonalizedResponse(
            original_query=query,
            response=response,
            user_id=user_id,
            session_id=session_id,
            recommended_profile="general",
            personalization_applied=False,
            memory_stored=False,
            similar_past_queries=[],
            user_preferences_applied=[],
            profile_refinement_suggestions=[],
            processing_time_ms=0,
            personalization_confidence=0.3,
            timestamp=datetime.now().isoformat()
        )
    
    def get_phase6_status(self) -> Dict[str, Any]:
        """Get Phase 6 system status."""
        return {
            "phase6_available": self.phase6_available,
            "auto_refinement_enabled": self.enable_auto_refinement,
            "feedback_learning_enabled": self.enable_feedback_learning,
            "components": {
                "episodic_store": self.episodic_store is not None,
                "user_modeler": self.user_modeler is not None,
                "feedback_handler": self.feedback_handler is not None,
                "adaptive_refinement": self.adaptive_refinement is not None
            },
            "phase5_integration": PHASE5_AVAILABLE
        }


# Global Phase 6 engine instance
_phase6_engine = None

def get_phase6_engine(db_path: str = "memory_store/episodic_memory.db") -> Phase6PersonalizationEngine:
    """Get or create global Phase 6 engine instance."""
    global _phase6_engine
    
    if _phase6_engine is None:
        _phase6_engine = Phase6PersonalizationEngine(db_path)
    
    return _phase6_engine

def personalize_sam_response(query: str,
                           initial_response: str,
                           user_id: str,
                           session_id: str,
                           context: Optional[Dict[str, Any]] = None,
                           current_profile: Optional[str] = None) -> PersonalizedResponse:
    """Convenience function for personalizing SAM responses with Phase 6."""
    engine = get_phase6_engine()
    return engine.personalize_response(query, initial_response, user_id, session_id, context, current_profile)

def submit_personalization_feedback(memory_id: str,
                                  user_id: str,
                                  rating: float,
                                  correction: Optional[str] = None) -> bool:
    """Convenience function for submitting personalization feedback."""
    engine = get_phase6_engine()
    return engine.submit_feedback(memory_id, user_id, rating, correction)

def get_personalization_insights(user_id: str) -> Dict[str, Any]:
    """Convenience function for getting personalization insights."""
    engine = get_phase6_engine()
    return engine.get_user_insights(user_id)
