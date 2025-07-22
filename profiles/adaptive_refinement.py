#!/usr/bin/env python3
"""
Phase 6: Adaptive Profile Refinement for SAM
Automatically adjusts profile weights based on user feedback and usage patterns.

This system enables SAM to:
1. Automatically adjust profile weights based on user feedback
2. Generate custom profiles from user patterns
3. Prompt users to save successful configurations
4. Evolve profiles over time based on effectiveness
5. Maintain profile performance metrics
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
from memory.episodic_store import EpisodicMemoryStore, EpisodicMemory
from profiles.user_modeler import UserModelingEngine, UserModel, PersonalizedProfile
from learning.feedback_handler import FeedbackHandler, FeedbackEntry, LearningInsight

logger = logging.getLogger(__name__)

class RefinementTrigger(Enum):
    """Triggers for profile refinement."""
    USER_FEEDBACK = "user_feedback"
    PERFORMANCE_DECLINE = "performance_decline"
    USAGE_PATTERN_CHANGE = "usage_pattern_change"
    PERIODIC_REVIEW = "periodic_review"
    MANUAL_REQUEST = "manual_request"

class RefinementType(Enum):
    """Types of profile refinements."""
    WEIGHT_ADJUSTMENT = "weight_adjustment"
    NEW_PROFILE_CREATION = "new_profile_creation"
    PROFILE_MERGE = "profile_merge"
    PROFILE_DEPRECATION = "profile_deprecation"

class AdaptationStrategy(Enum):
    """Strategies for profile adaptation."""
    CONSERVATIVE = "conservative"  # Small, gradual changes
    MODERATE = "moderate"         # Balanced adaptation
    AGGRESSIVE = "aggressive"     # Rapid adaptation to feedback

@dataclass
class ProfileRefinement:
    """Represents a profile refinement action."""
    refinement_id: str
    user_id: str
    profile_id: str
    timestamp: str
    
    # Refinement details
    trigger: RefinementTrigger
    refinement_type: RefinementType
    description: str
    
    # Changes made
    weight_changes: Dict[str, float]
    new_weights: Dict[str, float]
    confidence: float
    
    # Performance tracking
    performance_before: float
    performance_after: Optional[float] = None
    effectiveness_score: Optional[float] = None
    
    # Metadata
    evidence_count: int = 0
    user_approved: bool = False
    auto_applied: bool = False

@dataclass
class ProfilePerformanceMetrics:
    """Performance metrics for a profile."""
    profile_id: str
    user_id: str
    
    # Usage statistics
    total_uses: int
    recent_uses: int  # Last 30 days
    
    # Satisfaction metrics
    average_satisfaction: float
    satisfaction_trend: List[float]  # Recent satisfaction scores
    
    # Performance indicators
    success_rate: float
    improvement_rate: float
    consistency_score: float
    
    # Comparative metrics
    performance_vs_baseline: float
    performance_vs_other_profiles: float
    
    # Temporal data
    first_used: str
    last_used: str
    last_updated: str

class AdaptiveProfileRefinement:
    """
    Advanced profile refinement system that automatically adapts profiles
    based on user feedback, usage patterns, and performance metrics.
    """
    
    def __init__(self, 
                 episodic_store: EpisodicMemoryStore,
                 user_modeler: UserModelingEngine,
                 feedback_handler: FeedbackHandler,
                 adaptation_strategy: AdaptationStrategy = AdaptationStrategy.MODERATE):
        """Initialize the adaptive profile refinement system."""
        
        self.episodic_store = episodic_store
        self.user_modeler = user_modeler
        self.feedback_handler = feedback_handler
        self.adaptation_strategy = adaptation_strategy
        
        # Refinement history
        self.refinement_history = []
        self.profile_metrics = {}
        
        # Adaptation parameters
        self.adaptation_rates = {
            AdaptationStrategy.CONSERVATIVE: 0.05,
            AdaptationStrategy.MODERATE: 0.1,
            AdaptationStrategy.AGGRESSIVE: 0.2
        }
        
        # Thresholds for automatic refinement
        self.performance_decline_threshold = 0.15
        self.min_interactions_for_refinement = 10
        self.refinement_confidence_threshold = 0.7
        
        # Auto-refinement settings
        self.auto_apply_high_confidence_refinements = True
        self.prompt_user_for_medium_confidence = True
        
        logger.info(f"Adaptive Profile Refinement initialized with {adaptation_strategy.value} strategy")
    
    def analyze_profile_performance(self, user_id: str, profile_id: str) -> ProfilePerformanceMetrics:
        """Analyze performance metrics for a specific profile."""
        try:
            # Get user's memories for this profile
            memories = self.episodic_store.retrieve_memories(user_id, limit=200)
            profile_memories = [m for m in memories if m.active_profile == profile_id]
            
            if not profile_memories:
                return self._create_empty_metrics(user_id, profile_id)
            
            # Calculate usage statistics
            total_uses = len(profile_memories)
            recent_cutoff = datetime.now() - timedelta(days=30)
            recent_memories = [m for m in profile_memories if self._is_recent(m.timestamp, days=30)]
            recent_uses = len(recent_memories)
            
            # Calculate satisfaction metrics
            satisfaction_scores = [m.user_satisfaction for m in profile_memories if m.user_satisfaction is not None]
            average_satisfaction = np.mean(satisfaction_scores) if satisfaction_scores else 0.5
            satisfaction_trend = satisfaction_scores[-10:] if len(satisfaction_scores) >= 10 else satisfaction_scores
            
            # Calculate performance indicators
            success_rate = len([s for s in satisfaction_scores if s > 0.7]) / len(satisfaction_scores) if satisfaction_scores else 0.5
            
            # Calculate improvement rate (trend in satisfaction over time)
            improvement_rate = 0.0
            if len(satisfaction_trend) >= 5:
                x = np.arange(len(satisfaction_trend))
                correlation = np.corrcoef(x, satisfaction_trend)[0, 1] if len(satisfaction_trend) > 1 else 0
                improvement_rate = max(-1.0, min(1.0, correlation))
            
            # Calculate consistency score
            consistency_score = 1.0 - np.std(satisfaction_scores) if len(satisfaction_scores) > 1 else 0.5
            
            # Compare with baseline (general profile)
            general_memories = [m for m in memories if m.active_profile == "general"]
            general_satisfaction = [m.user_satisfaction for m in general_memories if m.user_satisfaction is not None]
            baseline_satisfaction = np.mean(general_satisfaction) if general_satisfaction else 0.5
            performance_vs_baseline = average_satisfaction - baseline_satisfaction
            
            # Compare with other profiles
            other_profiles_satisfaction = []
            for memory in memories:
                if memory.active_profile != profile_id and memory.user_satisfaction is not None:
                    other_profiles_satisfaction.append(memory.user_satisfaction)
            
            other_avg = np.mean(other_profiles_satisfaction) if other_profiles_satisfaction else 0.5
            performance_vs_other_profiles = average_satisfaction - other_avg
            
            metrics = ProfilePerformanceMetrics(
                profile_id=profile_id,
                user_id=user_id,
                total_uses=total_uses,
                recent_uses=recent_uses,
                average_satisfaction=average_satisfaction,
                satisfaction_trend=satisfaction_trend,
                success_rate=success_rate,
                improvement_rate=improvement_rate,
                consistency_score=consistency_score,
                performance_vs_baseline=performance_vs_baseline,
                performance_vs_other_profiles=performance_vs_other_profiles,
                first_used=profile_memories[-1].timestamp,
                last_used=profile_memories[0].timestamp,
                last_updated=datetime.now().isoformat()
            )
            
            # Cache metrics
            self.profile_metrics[f"{user_id}_{profile_id}"] = metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing profile performance: {e}")
            return self._create_empty_metrics(user_id, profile_id)
    
    def detect_refinement_opportunities(self, user_id: str) -> List[Dict[str, Any]]:
        """Detect opportunities for profile refinement."""
        opportunities = []
        
        try:
            # Get user model and analyze all profiles
            user_model = self.user_modeler.analyze_user_behavior(user_id)
            
            for profile_name in user_model.profile_usage_distribution.keys():
                metrics = self.analyze_profile_performance(user_id, profile_name)
                
                # Check for performance decline
                if metrics.improvement_rate < -0.3:
                    opportunities.append({
                        "type": "performance_decline",
                        "profile": profile_name,
                        "description": f"Profile {profile_name} showing declining satisfaction",
                        "trigger": RefinementTrigger.PERFORMANCE_DECLINE,
                        "confidence": abs(metrics.improvement_rate),
                        "recommended_action": "weight_adjustment"
                    })
                
                # Check for low satisfaction
                if metrics.average_satisfaction < 0.6 and metrics.total_uses >= self.min_interactions_for_refinement:
                    opportunities.append({
                        "type": "low_satisfaction",
                        "profile": profile_name,
                        "description": f"Profile {profile_name} has low average satisfaction ({metrics.average_satisfaction:.2f})",
                        "trigger": RefinementTrigger.USER_FEEDBACK,
                        "confidence": 1.0 - metrics.average_satisfaction,
                        "recommended_action": "weight_adjustment"
                    })
                
                # Check for inconsistent performance
                if metrics.consistency_score < 0.5 and metrics.total_uses >= self.min_interactions_for_refinement:
                    opportunities.append({
                        "type": "inconsistent_performance",
                        "profile": profile_name,
                        "description": f"Profile {profile_name} has inconsistent performance",
                        "trigger": RefinementTrigger.USAGE_PATTERN_CHANGE,
                        "confidence": 1.0 - metrics.consistency_score,
                        "recommended_action": "weight_adjustment"
                    })
            
            # Check for new profile creation opportunities
            if self._should_create_new_profile(user_model):
                opportunities.append({
                    "type": "new_profile_creation",
                    "profile": "new_custom_profile",
                    "description": "User patterns suggest a new custom profile would be beneficial",
                    "trigger": RefinementTrigger.USAGE_PATTERN_CHANGE,
                    "confidence": 0.8,
                    "recommended_action": "new_profile_creation"
                })
            
            # Sort by confidence
            opportunities.sort(key=lambda x: x["confidence"], reverse=True)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error detecting refinement opportunities: {e}")
            return []
    
    def refine_profile(self, 
                      user_id: str, 
                      profile_id: str, 
                      trigger: RefinementTrigger,
                      auto_apply: bool = False) -> Optional[ProfileRefinement]:
        """Refine a profile based on user feedback and usage patterns."""
        try:
            # Analyze current performance
            metrics = self.analyze_profile_performance(user_id, profile_id)
            
            if metrics.total_uses < self.min_interactions_for_refinement:
                logger.info(f"Insufficient data for profile refinement: {metrics.total_uses} interactions")
                return None
            
            # Get user preferences for weight adjustments
            user_model = self.user_modeler.analyze_user_behavior(user_id)
            dimension_preferences = [p for p in user_model.preferences if p.preference_type.value == "dimension_weight"]
            
            # Calculate weight adjustments
            weight_changes = self._calculate_weight_adjustments(metrics, dimension_preferences)
            
            if not weight_changes:
                logger.info("No significant weight adjustments needed")
                return None
            
            # Get current profile weights (would need to be stored/retrieved in production)
            current_weights = self._get_current_profile_weights(profile_id)
            new_weights = self._apply_weight_changes(current_weights, weight_changes)
            
            # Calculate confidence in refinement
            confidence = self._calculate_refinement_confidence(metrics, weight_changes, dimension_preferences)
            
            # Create refinement record
            refinement_id = f"refinement_{user_id}_{profile_id}_{int(time.time())}"
            
            refinement = ProfileRefinement(
                refinement_id=refinement_id,
                user_id=user_id,
                profile_id=profile_id,
                timestamp=datetime.now().isoformat(),
                trigger=trigger,
                refinement_type=RefinementType.WEIGHT_ADJUSTMENT,
                description=f"Adjusted weights based on {trigger.value}",
                weight_changes=weight_changes,
                new_weights=new_weights,
                confidence=confidence,
                performance_before=metrics.average_satisfaction,
                evidence_count=metrics.total_uses,
                auto_applied=auto_apply and confidence >= self.refinement_confidence_threshold
            )
            
            # Apply refinement if conditions are met
            if refinement.auto_applied or auto_apply:
                self._apply_refinement(refinement)
            
            # Store refinement
            self.refinement_history.append(refinement)
            
            logger.info(f"Profile refinement created: {refinement_id} (confidence: {confidence:.2f})")
            return refinement
            
        except Exception as e:
            logger.error(f"Error refining profile: {e}")
            return None
    
    def _calculate_weight_adjustments(self, 
                                    metrics: ProfilePerformanceMetrics,
                                    dimension_preferences: List[Any]) -> Dict[str, float]:
        """Calculate weight adjustments based on performance and preferences."""
        weight_changes = {}
        adaptation_rate = self.adaptation_rates[self.adaptation_strategy]
        
        # Adjust based on dimension preferences
        for pref in dimension_preferences:
            if pref.confidence.value in ["high", "very_high"]:
                dimension = pref.value["dimension"]
                preferred_score = pref.value["preferred_score"]
                
                # If user prefers high scores in this dimension, increase its weight
                if preferred_score > 0.7:
                    weight_changes[dimension] = adaptation_rate * (preferred_score - 0.5)
        
        # Adjust based on performance decline
        if metrics.improvement_rate < -0.2:
            # Reduce weights of dimensions that might be causing issues
            # This is a simplified heuristic - in production, would use more sophisticated analysis
            weight_changes["complexity"] = -adaptation_rate * 0.5
            weight_changes["clarity"] = adaptation_rate * 0.3
        
        return weight_changes
    
    def _get_current_profile_weights(self, profile_id: str) -> Dict[str, float]:
        """Get current weights for a profile."""
        # In production, this would retrieve from profile storage
        # For now, return default weights
        default_weights = {
            "utility": 0.25, "clarity": 0.20, "feasibility": 0.15,
            "credibility": 0.15, "novelty": 0.10, "complexity": 0.10, "danger": 0.05
        }
        return default_weights
    
    def _apply_weight_changes(self, current_weights: Dict[str, float], 
                            weight_changes: Dict[str, float]) -> Dict[str, float]:
        """Apply weight changes to current weights."""
        new_weights = current_weights.copy()
        
        # Apply changes
        for dimension, change in weight_changes.items():
            if dimension in new_weights:
                new_weights[dimension] = max(0.01, new_weights[dimension] + change)
        
        # Normalize to sum to 1.0
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            new_weights = {k: v / total_weight for k, v in new_weights.items()}
        
        return new_weights
    
    def _calculate_refinement_confidence(self, 
                                       metrics: ProfilePerformanceMetrics,
                                       weight_changes: Dict[str, float],
                                       dimension_preferences: List[Any]) -> float:
        """Calculate confidence in the proposed refinement."""
        confidence_factors = []
        
        # Evidence count factor
        evidence_factor = min(1.0, metrics.total_uses / 50)
        confidence_factors.append(evidence_factor)
        
        # Performance decline factor
        if metrics.improvement_rate < -0.2:
            confidence_factors.append(0.8)
        
        # Preference alignment factor
        preference_factor = len(dimension_preferences) / 10  # Assume max 10 preferences
        confidence_factors.append(min(1.0, preference_factor))
        
        # Consistency factor
        confidence_factors.append(metrics.consistency_score)
        
        # Calculate overall confidence
        overall_confidence = np.mean(confidence_factors) if confidence_factors else 0.5
        return overall_confidence
    
    def _apply_refinement(self, refinement: ProfileRefinement):
        """Apply a profile refinement."""
        # In production, this would update the actual profile weights
        logger.info(f"Applied refinement {refinement.refinement_id}")
        refinement.user_approved = True
    
    def _should_create_new_profile(self, user_model: UserModel) -> bool:
        """Determine if a new custom profile should be created."""
        # Check if user has diverse preferences that don't fit existing profiles
        if len(user_model.preferences) >= 5 and user_model.consistency_score < 0.6:
            return True
        
        # Check if user frequently switches between profiles
        if len(user_model.profile_usage_distribution) >= 3:
            usage_values = list(user_model.profile_usage_distribution.values())
            if max(usage_values) / sum(usage_values) < 0.5:  # No dominant profile
                return True
        
        return False
    
    def _create_empty_metrics(self, user_id: str, profile_id: str) -> ProfilePerformanceMetrics:
        """Create empty metrics for profiles with no data."""
        return ProfilePerformanceMetrics(
            profile_id=profile_id,
            user_id=user_id,
            total_uses=0,
            recent_uses=0,
            average_satisfaction=0.5,
            satisfaction_trend=[],
            success_rate=0.5,
            improvement_rate=0.0,
            consistency_score=0.5,
            performance_vs_baseline=0.0,
            performance_vs_other_profiles=0.0,
            first_used=datetime.now().isoformat(),
            last_used=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat()
        )
    
    def _is_recent(self, timestamp_str: str, days: int = 30) -> bool:
        """Check if timestamp is within recent days."""
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            cutoff = datetime.now() - timedelta(days=days)
            return timestamp >= cutoff
        except Exception:
            return False
    
    def get_refinement_summary(self, user_id: str) -> Dict[str, Any]:
        """Get summary of profile refinements for a user."""
        user_refinements = [r for r in self.refinement_history if r.user_id == user_id]
        
        if not user_refinements:
            return {"message": "No refinements found for user"}
        
        return {
            "total_refinements": len(user_refinements),
            "auto_applied": len([r for r in user_refinements if r.auto_applied]),
            "user_approved": len([r for r in user_refinements if r.user_approved]),
            "average_confidence": np.mean([r.confidence for r in user_refinements]),
            "recent_refinements": len([r for r in user_refinements if self._is_recent(r.timestamp, days=30)]),
            "refinement_types": Counter(r.refinement_type.value for r in user_refinements),
            "triggers": Counter(r.trigger.value for r in user_refinements)
        }


# Convenience functions
def create_adaptive_refinement(episodic_store: EpisodicMemoryStore,
                             user_modeler: UserModelingEngine,
                             feedback_handler: FeedbackHandler) -> AdaptiveProfileRefinement:
    """Create and return an adaptive profile refinement instance."""
    return AdaptiveProfileRefinement(episodic_store, user_modeler, feedback_handler)
