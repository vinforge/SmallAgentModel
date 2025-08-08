#!/usr/bin/env python3
"""
Phase 6: User Modeling Engine for SAM
Tracks and adjusts personal profile weights based on user behavior patterns.

This system enables SAM to:
1. Learn user preferences and behavior patterns
2. Adapt profile weights based on usage
3. Generate personalized profiles from interaction history
4. Predict user preferences for new scenarios
5. Provide insights into user behavior evolution
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

# Import episodic memory for user interaction history
from memory.episodic_store import EpisodicMemoryStore, EpisodicMemory, InteractionType

logger = logging.getLogger(__name__)

class PreferenceType(Enum):
    """Types of user preferences that can be learned."""
    DIMENSION_WEIGHT = "dimension_weight"
    PROFILE_PREFERENCE = "profile_preference"
    QUERY_STYLE = "query_style"
    FEEDBACK_PATTERN = "feedback_pattern"
    RISK_TOLERANCE = "risk_tolerance"
    COMPLEXITY_PREFERENCE = "complexity_preference"

class LearningConfidence(Enum):
    """Confidence levels for learned preferences."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class UserPreference:
    """Represents a learned user preference."""
    preference_id: str
    preference_type: PreferenceType
    description: str
    value: Any  # Can be float, dict, string, etc.
    confidence: LearningConfidence
    evidence_count: int
    first_observed: str
    last_updated: str
    supporting_memories: List[str] = field(default_factory=list)

@dataclass
class PersonalizedProfile:
    """A personalized profile generated from user behavior."""
    profile_id: str
    profile_name: str
    base_profile: str  # The original profile this is based on
    user_id: str
    
    # Customized weights and preferences
    dimension_weights: Dict[str, float]
    preference_adjustments: Dict[str, Any]
    
    # Learning metadata
    confidence_score: float
    usage_count: int
    success_rate: float
    created_at: str
    last_used: str
    
    # Performance metrics
    average_satisfaction: float = 0.0
    improvement_over_base: float = 0.0

@dataclass
class UserModel:
    """Complete user model with all learned preferences and patterns."""
    user_id: str
    created_at: str
    last_updated: str
    
    # Core preferences
    preferences: List[UserPreference]
    personalized_profiles: List[PersonalizedProfile]
    
    # Behavioral patterns
    interaction_patterns: Dict[str, Any]
    learning_velocity: float  # How quickly user adapts
    consistency_score: float  # How consistent user behavior is
    
    # Usage statistics
    total_interactions: int
    profile_usage_distribution: Dict[str, int]
    average_session_length: float
    preferred_interaction_times: List[str]

class UserModelingEngine:
    """
    Advanced user modeling engine that learns from interaction patterns
    and generates personalized profiles and preferences.
    """
    
    def __init__(self, episodic_store: EpisodicMemoryStore):
        """Initialize the user modeling engine."""
        self.episodic_store = episodic_store
        
        # Learning parameters
        self.min_interactions_for_learning = 10
        self.preference_confidence_threshold = 0.7
        self.profile_generation_threshold = 20
        
        # Default dimension weights for base profiles
        self.base_profile_weights = {
            "general": {
                "utility": 0.25, "clarity": 0.20, "feasibility": 0.15,
                "credibility": 0.15, "novelty": 0.10, "complexity": 0.10, "danger": 0.05
            },
            "researcher": {
                "novelty": 0.30, "credibility": 0.25, "utility": 0.20,
                "complexity": 0.15, "clarity": 0.05, "feasibility": 0.03, "danger": 0.02
            },
            "business": {
                "utility": 0.35, "feasibility": 0.25, "clarity": 0.15,
                "credibility": 0.10, "danger": 0.08, "novelty": 0.05, "complexity": 0.02
            },
            "legal": {
                "credibility": 0.40, "clarity": 0.25, "danger": 0.15,
                "utility": 0.10, "feasibility": 0.05, "novelty": 0.03, "complexity": 0.02
            }
        }
        
        logger.info("User Modeling Engine initialized")
    
    def analyze_user_behavior(self, user_id: str) -> UserModel:
        """Analyze user behavior and generate comprehensive user model."""
        try:
            # Get user's interaction history
            memories = self.episodic_store.retrieve_memories(user_id, limit=500)
            
            if len(memories) < self.min_interactions_for_learning:
                return self._create_minimal_user_model(user_id, memories)
            
            # Analyze different aspects of user behavior
            preferences = self._learn_user_preferences(memories)
            personalized_profiles = self._generate_personalized_profiles(user_id, memories, preferences)
            interaction_patterns = self._analyze_interaction_patterns(memories)
            
            # Calculate behavioral metrics
            learning_velocity = self._calculate_learning_velocity(memories)
            consistency_score = self._calculate_consistency_score(memories)
            
            # Usage statistics
            profile_usage = Counter(m.active_profile for m in memories)
            avg_session_length = self._calculate_average_session_length(memories)
            preferred_times = self._analyze_interaction_times(memories)
            
            user_model = UserModel(
                user_id=user_id,
                created_at=memories[-1].timestamp if memories else datetime.now().isoformat(),
                last_updated=datetime.now().isoformat(),
                preferences=preferences,
                personalized_profiles=personalized_profiles,
                interaction_patterns=interaction_patterns,
                learning_velocity=learning_velocity,
                consistency_score=consistency_score,
                total_interactions=len(memories),
                profile_usage_distribution=dict(profile_usage),
                average_session_length=avg_session_length,
                preferred_interaction_times=preferred_times
            )
            
            logger.info(f"Generated user model for {user_id} with {len(preferences)} preferences")
            return user_model
            
        except Exception as e:
            logger.error(f"Error analyzing user behavior: {e}")
            return self._create_minimal_user_model(user_id, [])
    
    def _learn_user_preferences(self, memories: List[EpisodicMemory]) -> List[UserPreference]:
        """Learn user preferences from interaction history."""
        preferences = []
        
        # Learn dimension weight preferences
        dimension_preferences = self._learn_dimension_preferences(memories)
        preferences.extend(dimension_preferences)
        
        # Learn profile preferences
        profile_preferences = self._learn_profile_preferences(memories)
        preferences.extend(profile_preferences)
        
        # Learn query style preferences
        query_style_preferences = self._learn_query_style_preferences(memories)
        preferences.extend(query_style_preferences)
        
        # Learn feedback patterns
        feedback_preferences = self._learn_feedback_patterns(memories)
        preferences.extend(feedback_preferences)
        
        return preferences
    
    def _learn_dimension_preferences(self, memories: List[EpisodicMemory]) -> List[UserPreference]:
        """Learn user preferences for different conceptual dimensions."""
        preferences = []
        
        # Analyze dimension scores from successful interactions
        successful_memories = [m for m in memories if m.user_satisfaction and m.user_satisfaction > 0.7]
        
        if len(successful_memories) < 5:
            return preferences
        
        # Calculate average dimension scores for successful interactions
        dimension_totals = defaultdict(list)
        for memory in successful_memories:
            for dim, score in memory.dimension_scores.items():
                dimension_totals[dim].append(score)
        
        # Identify preferred dimensions (those with consistently high scores)
        for dimension, scores in dimension_totals.items():
            if len(scores) >= 3:
                avg_score = np.mean(scores)
                consistency = 1.0 - np.std(scores)  # Higher consistency = lower std dev
                
                if avg_score > 0.7 and consistency > 0.6:
                    confidence = self._calculate_preference_confidence(len(scores), consistency)
                    
                    preferences.append(UserPreference(
                        preference_id=f"dimension_preference_{dimension}",
                        preference_type=PreferenceType.DIMENSION_WEIGHT,
                        description=f"User prefers high {dimension} (avg: {avg_score:.2f})",
                        value={"dimension": dimension, "preferred_score": avg_score, "weight_boost": 1.2},
                        confidence=confidence,
                        evidence_count=len(scores),
                        first_observed=successful_memories[-1].timestamp,
                        last_updated=successful_memories[0].timestamp,
                        supporting_memories=[m.memory_id for m in successful_memories if dimension in m.dimension_scores][:5]
                    ))
        
        return preferences
    
    def _learn_profile_preferences(self, memories: List[EpisodicMemory]) -> List[UserPreference]:
        """Learn user preferences for different reasoning profiles."""
        preferences = []
        
        # Analyze profile usage and satisfaction
        profile_stats = defaultdict(lambda: {"count": 0, "satisfaction": [], "memories": []})
        
        for memory in memories:
            profile = memory.active_profile
            profile_stats[profile]["count"] += 1
            profile_stats[profile]["memories"].append(memory.memory_id)
            
            if memory.user_satisfaction is not None:
                profile_stats[profile]["satisfaction"].append(memory.user_satisfaction)
        
        # Identify preferred profiles
        for profile, stats in profile_stats.items():
            if stats["count"] >= 5:
                usage_frequency = stats["count"] / len(memories)
                avg_satisfaction = np.mean(stats["satisfaction"]) if stats["satisfaction"] else 0.5
                
                if usage_frequency > 0.3 or avg_satisfaction > 0.7:
                    confidence = self._calculate_preference_confidence(stats["count"], avg_satisfaction)
                    
                    preferences.append(UserPreference(
                        preference_id=f"profile_preference_{profile}",
                        preference_type=PreferenceType.PROFILE_PREFERENCE,
                        description=f"User frequently uses {profile} profile ({usage_frequency:.1%} of time)",
                        value={"profile": profile, "usage_frequency": usage_frequency, "satisfaction": avg_satisfaction},
                        confidence=confidence,
                        evidence_count=stats["count"],
                        first_observed=memories[-1].timestamp,
                        last_updated=memories[0].timestamp,
                        supporting_memories=stats["memories"][:5]
                    ))
        
        return preferences
    
    def _learn_query_style_preferences(self, memories: List[EpisodicMemory]) -> List[UserPreference]:
        """Learn user query style preferences."""
        preferences = []
        
        # Analyze query characteristics
        query_lengths = [len(m.query.split()) for m in memories]
        avg_query_length = np.mean(query_lengths)
        
        # Categorize query types
        query_types = []
        for memory in memories:
            query_lower = memory.query.lower()
            if any(word in query_lower for word in ['what', 'define', 'explain']):
                query_types.append("information_seeking")
            elif any(word in query_lower for word in ['how', 'steps', 'process']):
                query_types.append("procedural")
            elif any(word in query_lower for word in ['should', 'recommend', 'suggest']):
                query_types.append("decision_support")
            else:
                query_types.append("general")
        
        # Identify dominant query style
        query_type_counts = Counter(query_types)
        if query_type_counts:
            dominant_type, count = query_type_counts.most_common(1)[0]
            frequency = count / len(memories)
            
            if frequency > 0.4:  # If more than 40% of queries are of this type
                confidence = self._calculate_preference_confidence(count, frequency)
                
                preferences.append(UserPreference(
                    preference_id=f"query_style_{dominant_type}",
                    preference_type=PreferenceType.QUERY_STYLE,
                    description=f"User primarily asks {dominant_type} questions ({frequency:.1%})",
                    value={"query_type": dominant_type, "frequency": frequency, "avg_length": avg_query_length},
                    confidence=confidence,
                    evidence_count=count,
                    first_observed=memories[-1].timestamp,
                    last_updated=memories[0].timestamp,
                    supporting_memories=[m.memory_id for m in memories if self._categorize_query(m.query) == dominant_type][:5]
                ))
        
        return preferences

    def _generate_personalized_profiles(self, user_id: str, memories: List[EpisodicMemory],
                                      preferences: List[UserPreference]) -> List[PersonalizedProfile]:
        """Generate personalized profiles based on user preferences."""
        personalized_profiles = []

        if len(memories) < self.profile_generation_threshold:
            return personalized_profiles

        # Analyze which base profiles work best for the user
        profile_performance = defaultdict(lambda: {"satisfaction": [], "count": 0})

        for memory in memories:
            if memory.user_satisfaction is not None:
                profile_performance[memory.active_profile]["satisfaction"].append(memory.user_satisfaction)
            profile_performance[memory.active_profile]["count"] += 1

        # Generate personalized versions of well-performing profiles
        for base_profile, performance in profile_performance.items():
            if performance["count"] >= 10 and performance["satisfaction"]:
                avg_satisfaction = np.mean(performance["satisfaction"])

                if avg_satisfaction > 0.6:  # Only personalize profiles that work reasonably well
                    personalized_profile = self._create_personalized_profile(
                        user_id, base_profile, memories, preferences, avg_satisfaction
                    )
                    if personalized_profile:
                        personalized_profiles.append(personalized_profile)

        return personalized_profiles

    def _create_personalized_profile(self, user_id: str, base_profile: str,
                                   memories: List[EpisodicMemory], preferences: List[UserPreference],
                                   base_satisfaction: float) -> Optional[PersonalizedProfile]:
        """Create a personalized profile based on user preferences."""
        try:
            # Start with base profile weights
            base_weights = self.base_profile_weights.get(base_profile, self.base_profile_weights["general"])
            personalized_weights = base_weights.copy()

            # Apply dimension preferences
            dimension_prefs = [p for p in preferences if p.preference_type == PreferenceType.DIMENSION_WEIGHT]
            for pref in dimension_prefs:
                if pref.confidence in [LearningConfidence.HIGH, LearningConfidence.VERY_HIGH]:
                    dimension = pref.value["dimension"]
                    boost = pref.value["weight_boost"]

                    if dimension in personalized_weights:
                        personalized_weights[dimension] *= boost

            # Normalize weights to sum to 1.0
            total_weight = sum(personalized_weights.values())
            if total_weight > 0:
                personalized_weights = {k: v / total_weight for k, v in personalized_weights.items()}

            # Calculate improvement metrics
            profile_memories = [m for m in memories if m.active_profile == base_profile]
            usage_count = len(profile_memories)

            # Generate profile
            profile_id = f"{user_id}_{base_profile}_personalized"
            profile_name = f"Personalized {base_profile.title()}"

            personalized_profile = PersonalizedProfile(
                profile_id=profile_id,
                profile_name=profile_name,
                base_profile=base_profile,
                user_id=user_id,
                dimension_weights=personalized_weights,
                preference_adjustments={},
                confidence_score=min(0.9, base_satisfaction + 0.1),
                usage_count=usage_count,
                success_rate=base_satisfaction,
                created_at=datetime.now().isoformat(),
                last_used=profile_memories[0].timestamp if profile_memories else datetime.now().isoformat(),
                average_satisfaction=base_satisfaction,
                improvement_over_base=0.1  # Estimated improvement
            )

            return personalized_profile

        except Exception as e:
            logger.error(f"Error creating personalized profile: {e}")
            return None

    def _analyze_interaction_patterns(self, memories: List[EpisodicMemory]) -> Dict[str, Any]:
        """Analyze user interaction patterns."""
        patterns = {}

        # Session patterns
        sessions = defaultdict(list)
        for memory in memories:
            sessions[memory.session_id].append(memory)

        session_lengths = [len(session_memories) for session_memories in sessions.values()]
        patterns["average_session_length"] = np.mean(session_lengths) if session_lengths else 0
        patterns["session_count"] = len(sessions)

        # Timing patterns
        timestamps = [datetime.fromisoformat(m.timestamp) for m in memories]
        hours = [ts.hour for ts in timestamps]
        patterns["preferred_hours"] = Counter(hours).most_common(3)

        # Query complexity patterns
        query_lengths = [len(m.query.split()) for m in memories]
        patterns["average_query_length"] = np.mean(query_lengths) if query_lengths else 0
        patterns["query_complexity_trend"] = "increasing" if len(query_lengths) > 10 and np.corrcoef(range(len(query_lengths)), query_lengths)[0, 1] > 0.3 else "stable"

        return patterns

    def _calculate_learning_velocity(self, memories: List[EpisodicMemory]) -> float:
        """Calculate how quickly the user adapts and learns."""
        if len(memories) < 10:
            return 0.5  # Default for insufficient data

        # Analyze satisfaction trend over time
        satisfaction_scores = [(i, m.user_satisfaction) for i, m in enumerate(reversed(memories)) if m.user_satisfaction is not None]

        if len(satisfaction_scores) < 5:
            return 0.5

        # Calculate correlation between time and satisfaction
        indices, scores = zip(*satisfaction_scores)
        correlation = np.corrcoef(indices, scores)[0, 1] if len(scores) > 1 else 0

        # Convert correlation to learning velocity (0.0 to 1.0)
        learning_velocity = max(0.0, min(1.0, (correlation + 1) / 2))
        return learning_velocity

    def _calculate_consistency_score(self, memories: List[EpisodicMemory]) -> float:
        """Calculate how consistent the user's behavior is."""
        if len(memories) < 5:
            return 0.5

        # Analyze consistency in profile usage
        profile_usage = [m.active_profile for m in memories]
        profile_counts = Counter(profile_usage)

        # Calculate entropy (lower entropy = more consistent)
        total = len(profile_usage)
        entropy = -sum((count / total) * np.log2(count / total) for count in profile_counts.values())
        max_entropy = np.log2(len(profile_counts))

        # Convert to consistency score (0.0 to 1.0, higher = more consistent)
        consistency = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
        return consistency

    def _calculate_average_session_length(self, memories: List[EpisodicMemory]) -> float:
        """Calculate average session length in interactions."""
        sessions = defaultdict(list)
        for memory in memories:
            sessions[memory.session_id].append(memory)

        session_lengths = [len(session_memories) for session_memories in sessions.values()]
        return np.mean(session_lengths) if session_lengths else 0.0

    def _analyze_interaction_times(self, memories: List[EpisodicMemory]) -> List[str]:
        """Analyze preferred interaction times."""
        timestamps = [datetime.fromisoformat(m.timestamp) for m in memories]
        hours = [ts.hour for ts in timestamps]

        # Get top 3 preferred hours
        hour_counts = Counter(hours)
        preferred_hours = [f"{hour:02d}:00" for hour, _ in hour_counts.most_common(3)]
        return preferred_hours

    def _create_minimal_user_model(self, user_id: str, memories: List[EpisodicMemory]) -> UserModel:
        """Create minimal user model for users with insufficient interaction history."""
        return UserModel(
            user_id=user_id,
            created_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            preferences=[],
            personalized_profiles=[],
            interaction_patterns={},
            learning_velocity=0.5,
            consistency_score=0.5,
            total_interactions=len(memories),
            profile_usage_distribution={},
            average_session_length=0.0,
            preferred_interaction_times=[]
        )

    def _learn_feedback_patterns(self, memories: List[EpisodicMemory]) -> List[UserPreference]:
        """Learn user feedback patterns."""
        preferences = []

        # Analyze feedback patterns
        feedback_memories = [m for m in memories if m.user_feedback or m.user_satisfaction is not None]

        if len(feedback_memories) >= 5:
            # Calculate feedback frequency
            feedback_frequency = len(feedback_memories) / len(memories)

            # Analyze satisfaction patterns
            satisfaction_scores = [m.user_satisfaction for m in feedback_memories if m.user_satisfaction is not None]
            if satisfaction_scores:
                avg_satisfaction = np.mean(satisfaction_scores)
                satisfaction_consistency = 1.0 - np.std(satisfaction_scores)

                confidence = self._calculate_preference_confidence(len(satisfaction_scores), satisfaction_consistency)

                preferences.append(UserPreference(
                    preference_id="feedback_pattern",
                    preference_type=PreferenceType.FEEDBACK_PATTERN,
                    description=f"User provides feedback {feedback_frequency:.1%} of time (avg satisfaction: {avg_satisfaction:.2f})",
                    value={"feedback_frequency": feedback_frequency, "avg_satisfaction": avg_satisfaction},
                    confidence=confidence,
                    evidence_count=len(feedback_memories),
                    first_observed=feedback_memories[-1].timestamp,
                    last_updated=feedback_memories[0].timestamp,
                    supporting_memories=[m.memory_id for m in feedback_memories[:5]]
                ))

        return preferences

    def _categorize_query(self, query: str) -> str:
        """Categorize query type."""
        query_lower = query.lower()
        if any(word in query_lower for word in ['what', 'define', 'explain']):
            return "information_seeking"
        elif any(word in query_lower for word in ['how', 'steps', 'process']):
            return "procedural"
        elif any(word in query_lower for word in ['should', 'recommend', 'suggest']):
            return "decision_support"
        else:
            return "general"

    def _calculate_preference_confidence(self, evidence_count: int, consistency: float) -> LearningConfidence:
        """Calculate confidence level for a learned preference."""
        # Combine evidence count and consistency to determine confidence
        confidence_score = (min(evidence_count / 20, 1.0) * 0.6) + (consistency * 0.4)

        if confidence_score >= 0.8:
            return LearningConfidence.VERY_HIGH
        elif confidence_score >= 0.6:
            return LearningConfidence.HIGH
        elif confidence_score >= 0.4:
            return LearningConfidence.MEDIUM
        else:
            return LearningConfidence.LOW

    def get_recommended_profile(self, user_id: str, query_context: Dict[str, Any]) -> str:
        """Get recommended profile for user based on learned preferences."""
        try:
            user_model = self.analyze_user_behavior(user_id)

            # If user has personalized profiles, recommend the best one
            if user_model.personalized_profiles:
                best_profile = max(user_model.personalized_profiles, key=lambda p: p.average_satisfaction)
                return best_profile.profile_id

            # Otherwise, recommend based on profile preferences
            profile_prefs = [p for p in user_model.preferences if p.preference_type == PreferenceType.PROFILE_PREFERENCE]
            if profile_prefs:
                best_pref = max(profile_prefs, key=lambda p: p.value["satisfaction"])
                return best_pref.value["profile"]

            # Fall back to most used profile
            if user_model.profile_usage_distribution:
                most_used = max(user_model.profile_usage_distribution.items(), key=lambda x: x[1])
                return most_used[0]

            # Default fallback
            return "general"

        except Exception as e:
            logger.error(f"Error getting recommended profile: {e}")
            return "general"


# Convenience functions
def create_user_modeler(episodic_store: EpisodicMemoryStore) -> UserModelingEngine:
    """Create and return a user modeling engine instance."""
    return UserModelingEngine(episodic_store)

def analyze_user_preferences(user_modeler: UserModelingEngine, user_id: str) -> Dict[str, Any]:
    """Convenience function to analyze user preferences."""
    user_model = user_modeler.analyze_user_behavior(user_id)

    return {
        "total_interactions": user_model.total_interactions,
        "preferences_learned": len(user_model.preferences),
        "personalized_profiles": len(user_model.personalized_profiles),
        "learning_velocity": user_model.learning_velocity,
        "consistency_score": user_model.consistency_score,
        "recommended_profile": user_modeler.get_recommended_profile(user_id, {})
    }
