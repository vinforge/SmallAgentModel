"""
DPO Data Manager

Provides high-level interface for managing Direct Preference Optimization (DPO) 
preference pairs with validation, quality controls, and integration with SAM's 
feedback system.

Author: SAM Development Team
Version: 1.0.0
"""

import logging
import re
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

try:
    from ...memory.episodic_store import EpisodicMemoryStore, DPOPreferenceData
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from memory.episodic_store import EpisodicMemoryStore, DPOPreferenceData

logger = logging.getLogger(__name__)


@dataclass
class DPOValidationResult:
    """Result of DPO preference pair validation."""
    is_valid: bool
    quality_score: float
    issues: List[str]
    recommendations: List[str]


class DPODataManager:
    """
    High-level manager for DPO preference data with validation and quality controls.
    
    Provides:
    - Validation of preference pairs before storage
    - Quality scoring and filtering
    - Duplicate detection and prevention
    - Data export for training
    - Statistics and monitoring
    """
    
    def __init__(self, episodic_store: EpisodicMemoryStore, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DPO Data Manager.
        
        Args:
            episodic_store: Episodic memory store instance
            config: Configuration options
        """
        self.episodic_store = episodic_store
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.DPODataManager")
        
        # Validation configuration
        self.min_confidence_threshold = self.config.get('min_confidence_threshold', 0.7)
        self.min_correction_length = self.config.get('min_correction_length', 5)  # words
        self.max_correction_length = self.config.get('max_correction_length', 500)  # words
        self.min_response_difference = self.config.get('min_response_difference', 0.1)  # similarity threshold
        self.max_similarity_threshold = self.config.get('max_similarity_threshold', 0.9)  # too similar
        
        # Quality scoring weights
        self.quality_weights = self.config.get('quality_weights', {
            'confidence': 0.3,
            'length_appropriateness': 0.2,
            'response_difference': 0.3,
            'linguistic_quality': 0.2
        })
        
        self.logger.info("DPO Data Manager initialized")
    
    def create_preference_pair(self, feedback_id: str, user_id: str, session_id: str,
                              prompt_text: str, original_response: str, corrected_response: str,
                              feedback_confidence_score: float, feedback_type: str,
                              validate: bool = True) -> Tuple[bool, Optional[DPOPreferenceData], List[str]]:
        """
        Create and optionally validate a DPO preference pair.
        
        Args:
            feedback_id: ID of the original feedback
            user_id: User identifier
            session_id: Session identifier
            prompt_text: The original prompt/query
            original_response: The original (losing) response
            corrected_response: The corrected (winning) response
            feedback_confidence_score: Confidence score from feedback system
            feedback_type: Type of feedback
            validate: Whether to validate before creation
            
        Returns:
            Tuple of (success, preference_data, issues)
        """
        try:
            # Create preference data object
            preference_data = DPOPreferenceData(
                id=None,
                feedback_id=feedback_id,
                user_id=user_id,
                session_id=session_id,
                prompt_text=prompt_text,
                original_response=original_response,
                corrected_response=corrected_response,
                feedback_confidence_score=feedback_confidence_score,
                feedback_type=feedback_type,
                created_timestamp=datetime.now().isoformat(),
                is_active_for_tuning=True
            )
            
            issues = []
            
            # Validate if requested
            if validate:
                validation_result = self.validate_preference_pair(preference_data)
                if not validation_result.is_valid:
                    return False, None, validation_result.issues
                
                # Update quality score
                preference_data.quality_score = validation_result.quality_score
                issues.extend(validation_result.recommendations)
            
            # Check for duplicates
            if self._is_duplicate(preference_data):
                issues.append("Duplicate preference pair detected")
                return False, None, issues
            
            return True, preference_data, issues
            
        except Exception as e:
            self.logger.error(f"Error creating preference pair: {e}")
            return False, None, [f"Creation error: {str(e)}"]
    
    def store_preference_pair(self, preference_data: DPOPreferenceData) -> bool:
        """Store a validated preference pair."""
        try:
            return self.episodic_store.store_dpo_preference(preference_data)
        except Exception as e:
            self.logger.error(f"Error storing preference pair: {e}")
            return False
    
    def validate_preference_pair(self, preference_data: DPOPreferenceData) -> DPOValidationResult:
        """
        Validate a DPO preference pair for quality and suitability.
        
        Args:
            preference_data: The preference pair to validate
            
        Returns:
            Validation result with quality score and recommendations
        """
        issues = []
        recommendations = []
        quality_factors = []
        
        # 1. Confidence threshold check
        if preference_data.feedback_confidence_score < self.min_confidence_threshold:
            issues.append(f"Confidence score {preference_data.feedback_confidence_score:.2f} below threshold {self.min_confidence_threshold}")
        
        confidence_factor = min(1.0, preference_data.feedback_confidence_score / self.min_confidence_threshold)
        quality_factors.append(('confidence', confidence_factor))
        
        # 2. Correction length validation
        correction_words = len(preference_data.corrected_response.split())
        original_words = len(preference_data.original_response.split())
        
        if correction_words < self.min_correction_length:
            issues.append(f"Correction too short: {correction_words} words (min: {self.min_correction_length})")
        elif correction_words > self.max_correction_length:
            issues.append(f"Correction too long: {correction_words} words (max: {self.max_correction_length})")
        
        # Length appropriateness factor
        length_factor = 1.0
        if correction_words < self.min_correction_length:
            length_factor = correction_words / self.min_correction_length
        elif correction_words > self.max_correction_length:
            length_factor = self.max_correction_length / correction_words
        
        quality_factors.append(('length_appropriateness', length_factor))
        
        # 3. Response difference validation
        similarity = self._calculate_response_similarity(
            preference_data.original_response, 
            preference_data.corrected_response
        )
        
        if similarity > self.max_similarity_threshold:
            issues.append(f"Responses too similar: {similarity:.2f} (max: {self.max_similarity_threshold})")
            recommendations.append("Consider more substantial corrections for better training signal")
        elif similarity < self.min_response_difference:
            recommendations.append("Good substantial difference between responses")
        
        difference_factor = max(0.0, min(1.0, (1.0 - similarity) / (1.0 - self.min_response_difference)))
        quality_factors.append(('response_difference', difference_factor))
        
        # 4. Linguistic quality check
        linguistic_factor = self._assess_linguistic_quality(preference_data.corrected_response)
        quality_factors.append(('linguistic_quality', linguistic_factor))
        
        if linguistic_factor < 0.5:
            recommendations.append("Corrected response may have linguistic issues")
        
        # Calculate overall quality score
        quality_score = sum(
            factor * self.quality_weights.get(name, 0.25) 
            for name, factor in quality_factors
        )
        
        # Determine if valid
        is_valid = len(issues) == 0 and quality_score >= 0.5
        
        return DPOValidationResult(
            is_valid=is_valid,
            quality_score=quality_score,
            issues=issues,
            recommendations=recommendations
        )
    
    def _calculate_response_similarity(self, response1: str, response2: str) -> float:
        """Calculate similarity between two responses using word overlap."""
        words1 = set(response1.lower().split())
        words2 = set(response2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        elif not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _assess_linguistic_quality(self, text: str) -> float:
        """Assess linguistic quality of text (simplified heuristics)."""
        if not text.strip():
            return 0.0
        
        quality_score = 1.0
        
        # Check for basic sentence structure
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            quality_score -= 0.3
        
        # Check for reasonable word count
        words = text.split()
        if len(words) < 3:
            quality_score -= 0.2
        
        # Check for excessive repetition
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        max_freq = max(word_freq.values()) if word_freq else 0
        if max_freq > len(words) * 0.3:  # More than 30% repetition
            quality_score -= 0.2
        
        # Check for basic punctuation
        if not re.search(r'[.!?]', text):
            quality_score -= 0.1
        
        return max(0.0, quality_score)
    
    def _is_duplicate(self, preference_data: DPOPreferenceData) -> bool:
        """Check if a preference pair is a duplicate."""
        try:
            # Get recent preferences for the user
            existing_preferences = self.episodic_store.get_dpo_preferences(
                user_id=preference_data.user_id,
                min_confidence=0.0,
                active_only=False,
                limit=100
            )
            
            # Check for exact matches
            for existing in existing_preferences:
                if (existing.prompt_text == preference_data.prompt_text and
                    existing.original_response == preference_data.original_response and
                    existing.corrected_response == preference_data.corrected_response):
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking for duplicates: {e}")
            return False
    
    def get_training_dataset(self, user_id: str, min_confidence: float = 0.8,
                           min_quality: float = 0.6, limit: int = 1000) -> List[Dict[str, str]]:
        """
        Get DPO training dataset in the format expected by DPO trainers.
        
        Args:
            user_id: User identifier
            min_confidence: Minimum confidence threshold
            min_quality: Minimum quality score threshold
            limit: Maximum number of pairs to return
            
        Returns:
            List of training examples in DPO format
        """
        try:
            preferences = self.episodic_store.get_dpo_preferences(
                user_id=user_id,
                min_confidence=min_confidence,
                active_only=True,
                limit=limit
            )
            
            # Filter by quality score if available
            if min_quality > 0:
                preferences = [
                    p for p in preferences 
                    if p.quality_score is None or p.quality_score >= min_quality
                ]
            
            # Convert to DPO training format
            training_data = []
            for pref in preferences:
                training_data.append({
                    'prompt': pref.prompt_text,
                    'chosen': pref.corrected_response,  # y_w (winning response)
                    'rejected': pref.original_response,  # y_l (losing response)
                    'metadata': {
                        'feedback_id': pref.feedback_id,
                        'confidence': pref.feedback_confidence_score,
                        'quality_score': pref.quality_score,
                        'feedback_type': pref.feedback_type
                    }
                })
            
            self.logger.info(f"Generated training dataset with {len(training_data)} examples for user {user_id}")
            return training_data
            
        except Exception as e:
            self.logger.error(f"Error generating training dataset: {e}")
            return []
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a user's DPO data."""
        try:
            base_stats = self.episodic_store.get_dpo_stats(user_id)
            
            # Add quality distribution
            preferences = self.episodic_store.get_dpo_preferences(
                user_id=user_id,
                min_confidence=0.0,
                active_only=False,
                limit=1000
            )
            
            quality_scores = [p.quality_score for p in preferences if p.quality_score is not None]
            
            if quality_scores:
                base_stats.update({
                    'avg_quality_score': round(sum(quality_scores) / len(quality_scores), 3),
                    'min_quality_score': min(quality_scores),
                    'max_quality_score': max(quality_scores),
                    'quality_scored_pairs': len(quality_scores)
                })
            else:
                base_stats.update({
                    'avg_quality_score': 0.0,
                    'min_quality_score': 0.0,
                    'max_quality_score': 0.0,
                    'quality_scored_pairs': 0
                })
            
            # Training readiness assessment
            high_quality_pairs = len([p for p in preferences 
                                    if p.is_active_for_tuning and 
                                    p.feedback_confidence_score >= 0.8 and
                                    (p.quality_score is None or p.quality_score >= 0.6)])
            
            base_stats['training_ready_pairs'] = high_quality_pairs
            base_stats['training_readiness'] = 'Ready' if high_quality_pairs >= 10 else 'Needs more data'
            
            return base_stats
            
        except Exception as e:
            self.logger.error(f"Error getting user stats: {e}")
            return {}


# Global instance management
_dpo_manager = None

def get_dpo_data_manager(episodic_store: Optional[EpisodicMemoryStore] = None, 
                        config: Optional[Dict[str, Any]] = None) -> DPODataManager:
    """Get or create a global DPO data manager instance."""
    global _dpo_manager
    
    if _dpo_manager is None:
        if episodic_store is None:
            try:
                from ...memory.episodic_store import create_episodic_store
            except ImportError:
                from memory.episodic_store import create_episodic_store
            episodic_store = create_episodic_store()

        _dpo_manager = DPODataManager(episodic_store, config)
    
    return _dpo_manager
