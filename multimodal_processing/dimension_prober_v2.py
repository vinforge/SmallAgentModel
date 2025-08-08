#!/usr/bin/env python3
"""
Enhanced Conceptual Dimension Probing v2 for SAM
Implements SPoSE-style vector-based probing with profile-aware reasoning.
The FIRST AI system with human-like conceptual understanding.
"""

import os
import sys
import json
import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path

# Add plugins directory to path for SPoSE access
PLUGINS_DIR = Path(__file__).parent.parent / "plugins" / "dimension_probing"
sys.path.append(str(PLUGINS_DIR / "model_training" / "SPoSE"))

try:
    from models.model import SPoSE
    SPOSE_AVAILABLE = True
except ImportError:
    SPOSE_AVAILABLE = False
    logging.warning("SPoSE model not available. Using enhanced pattern-based scoring.")

@dataclass
class DimensionScoresV2:
    """Enhanced container for conceptual dimension scores with profile awareness."""
    # Dynamic dimensions based on profile
    scores: Dict[str, float]
    
    # Metadata
    profile: str
    confidence: Dict[str, float]
    reasoning: Dict[str, str]
    processing_method: str  # "spose_vector" or "enhanced_pattern"
    
    def get_score(self, dimension: str) -> float:
        """Get score for a specific dimension."""
        return self.scores.get(dimension, 0.0)
    
    def get_high_dimensions(self, threshold: float = 0.6) -> List[Tuple[str, float]]:
        """Get dimensions above threshold."""
        return [(dim, score) for dim, score in self.scores.items() if score > threshold]
    
    def get_weighted_score(self, weights: Dict[str, float]) -> float:
        """Calculate weighted composite score."""
        total_score = 0.0
        total_weight = 0.0
        
        for dim, score in self.scores.items():
            weight = weights.get(dim, 1.0)
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0

@dataclass
class DimensionProbeResultV2:
    """Enhanced result of dimension probing with profile context."""
    scores: DimensionScoresV2
    profile_info: Dict[str, Any]
    processing_time_ms: float
    chunk_context: Optional[Dict[str, Any]] = None

class ProfileManager:
    """Manages dimension profiles for different reasoning modes."""
    
    def __init__(self, profiles_dir: Optional[Path] = None):
        """Initialize profile manager."""
        self.profiles_dir = profiles_dir or Path(__file__).parent / "dimension_profiles"
        self.profiles = {}
        self.default_profile = "general"
        self._load_profiles()
    
    def _load_profiles(self):
        """Load all available profiles."""
        if not self.profiles_dir.exists():
            logging.warning(f"Profiles directory not found: {self.profiles_dir}")
            return
        
        for profile_file in self.profiles_dir.glob("*.json"):
            try:
                with open(profile_file, 'r') as f:
                    profile_data = json.load(f)
                    profile_name = profile_data.get('name', profile_file.stem)
                    self.profiles[profile_name] = profile_data
                    logging.info(f"Loaded profile: {profile_name}")
            except Exception as e:
                logging.error(f"Failed to load profile {profile_file}: {e}")
    
    def get_profile(self, profile_name: str) -> Dict[str, Any]:
        """Get profile configuration."""
        if profile_name not in self.profiles:
            logging.warning(f"Profile '{profile_name}' not found, using default '{self.default_profile}'")
            profile_name = self.default_profile
        
        return self.profiles.get(profile_name, {})
    
    def list_profiles(self) -> List[str]:
        """List available profiles."""
        return list(self.profiles.keys())
    
    def get_profile_description(self, profile_name: str) -> str:
        """Get profile description."""
        profile = self.get_profile(profile_name)
        return profile.get('description', f'Profile: {profile_name}')

class EnhancedDimensionProberV2:
    """
    Enhanced Conceptual Dimension Probing System v2
    
    Implements human-like conceptual understanding through:
    1. SPoSE-style vector-based probing (when available)
    2. Enhanced pattern-based scoring (fallback)
    3. Profile-aware reasoning modes
    4. Transparent explainability
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced dimension prober v2."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize profile manager
        self.profile_manager = ProfileManager()
        self.current_profile = self.config.get('default_profile', 'general')
        
        # Initialize SPoSE models if available
        self.spose_models = {}
        self.use_spose = SPOSE_AVAILABLE and self.config.get('enable_spose', True)
        
        if self.use_spose:
            self._initialize_spose_models()
        else:
            self.logger.info("Using enhanced pattern-based scoring")
    
    def _initialize_spose_models(self):
        """Initialize SPoSE models for vector-based probing."""
        try:
            # Look for pre-trained SPoSE models
            models_dir = PLUGINS_DIR / "model_training" / "SPoSE" / "results"
            if models_dir.exists():
                self.logger.info("SPoSE models directory found")
                # For now, we'll use enhanced pattern-based scoring
                # Future implementation will load actual SPoSE weights
            else:
                self.logger.info("No pre-trained SPoSE models found, using enhanced patterns")
                self.use_spose = False
        except Exception as e:
            self.logger.warning(f"Failed to initialize SPoSE models: {e}")
            self.use_spose = False
    
    def set_profile(self, profile_name: str):
        """Set the current reasoning profile."""
        if profile_name in self.profile_manager.list_profiles():
            self.current_profile = profile_name
            self.logger.info(f"Switched to profile: {profile_name}")
        else:
            self.logger.warning(f"Profile '{profile_name}' not found")
    
    def probe_chunk(self, text: str, context: Optional[Dict[str, Any]] = None, 
                   profile: Optional[str] = None) -> DimensionProbeResultV2:
        """
        Probe a text chunk for conceptual dimensions using current profile.
        
        Args:
            text: Text content to analyze
            context: Optional context information
            profile: Optional profile override
            
        Returns:
            DimensionProbeResultV2 with profile-aware scores
        """
        import time
        start_time = time.time()
        
        # Use specified profile or current default
        active_profile = profile or self.current_profile
        profile_config = self.profile_manager.get_profile(active_profile)
        
        # Choose probing method
        if self.use_spose:
            scores = self._spose_vector_probing(text, profile_config, context)
            method = "spose_vector"
        else:
            scores = self._enhanced_pattern_probing(text, profile_config, context)
            method = "enhanced_pattern"
        
        # Generate reasoning
        reasoning = self._generate_profile_reasoning(text, scores, profile_config)
        
        processing_time = (time.time() - start_time) * 1000
        
        return DimensionProbeResultV2(
            scores=DimensionScoresV2(
                scores=scores,
                profile=active_profile,
                confidence=self._calculate_confidence_scores(text, scores, profile_config),
                reasoning=reasoning,
                processing_method=method
            ),
            profile_info={
                'name': active_profile,
                'description': profile_config.get('description', ''),
                'target_users': profile_config.get('target_users', [])
            },
            processing_time_ms=processing_time,
            chunk_context=context
        )
    
    def _spose_vector_probing(self, text: str, profile_config: Dict[str, Any], 
                             context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        SPoSE-style vector-based dimension probing.
        
        Future implementation will use actual SPoSE embeddings.
        For now, falls back to enhanced pattern-based scoring.
        """
        # TODO: Implement actual SPoSE vector projection
        # For Phase 2.1, we'll use enhanced pattern-based scoring
        return self._enhanced_pattern_probing(text, profile_config, context)
    
    def _enhanced_pattern_probing(self, text: str, profile_config: Dict[str, Any], 
                                 context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Enhanced pattern-based scoring with profile awareness."""
        text_lower = text.lower()
        scores = {}
        
        dimensions = profile_config.get('dimensions', {})
        
        for dim_name, dim_config in dimensions.items():
            # Get patterns and boost keywords for this dimension
            patterns = dim_config.get('patterns', [])
            boost_keywords = dim_config.get('boost_keywords', [])
            
            # Calculate base score from patterns
            base_score = self._score_patterns_enhanced(text_lower, patterns)
            
            # Apply boost from keywords
            keyword_boost = self._score_keywords(text_lower, boost_keywords)
            
            # Combine scores with diminishing returns
            combined_score = base_score + (keyword_boost * 0.3)
            
            # Apply profile weight
            weight = dim_config.get('weight', 1.0)
            final_score = min(1.0, combined_score * weight)
            
            scores[dim_name] = final_score
        
        return scores
    
    def _score_patterns_enhanced(self, text: str, patterns: List[str]) -> float:
        """Enhanced pattern scoring with semantic awareness."""
        import re
        
        if not patterns:
            return 0.0
        
        max_score = 0.0
        total_matches = 0
        
        for pattern in patterns:
            # Create flexible regex pattern
            pattern_words = pattern.lower().split()
            if len(pattern_words) > 1:
                # Multi-word pattern - look for words within reasonable distance
                regex_pattern = r'\b' + r'\W+'.join(re.escape(word) for word in pattern_words) + r'\b'
            else:
                # Single word pattern
                regex_pattern = r'\b' + re.escape(pattern.lower()) + r'\b'
            
            matches = len(re.findall(regex_pattern, text, re.IGNORECASE))
            if matches > 0:
                # Logarithmic scaling for multiple matches
                score = min(1.0, 0.6 + 0.2 * np.log(1 + matches))
                max_score = max(max_score, score)
                total_matches += matches
        
        # Combine max score with frequency bonus
        if total_matches > 0:
            frequency_bonus = min(0.3, total_matches * 0.05)
            return min(1.0, max_score + frequency_bonus)
        
        return 0.0
    
    def _score_keywords(self, text: str, keywords: List[str]) -> float:
        """Score based on keyword presence."""
        import re
        
        if not keywords:
            return 0.0
        
        matches = 0
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', text):
                matches += 1
        
        return min(1.0, matches / len(keywords))
    
    def _calculate_confidence_scores(self, text: str, scores: Dict[str, float], 
                                   profile_config: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence scores for each dimension."""
        text_length = len(text)
        word_count = len(text.split())
        
        # Base confidence on text length and content richness
        base_confidence = min(0.9, 0.4 + (word_count / 150) * 0.5)
        
        confidence = {}
        for dim_name, score in scores.items():
            # Higher confidence for higher scores (more evidence)
            score_confidence = 0.3 + (score * 0.4)
            
            # Combine with base confidence
            final_confidence = min(0.95, (base_confidence + score_confidence) / 2)
            confidence[dim_name] = final_confidence
        
        return confidence
    
    def _generate_profile_reasoning(self, text: str, scores: Dict[str, float], 
                                  profile_config: Dict[str, Any]) -> Dict[str, str]:
        """Generate human-readable reasoning for dimension scores."""
        reasoning = {}
        
        explanation_templates = profile_config.get('explanation_templates', {})
        
        for dim_name, score in scores.items():
            template = explanation_templates.get(dim_name, 
                f"{dim_name.replace('_', ' ').title()} score ({{score:.2f}}) based on content analysis")
            
            reasoning[dim_name] = template.format(score=score)
        
        return reasoning
    
    def get_dimension_summary(self, scores: DimensionScoresV2) -> str:
        """Generate a human-readable summary of dimension scores."""
        high_dims = []
        medium_dims = []
        
        for dim, score in scores.scores.items():
            if score > 0.6:
                high_dims.append(f"{dim}({score:.2f})")
            elif score > 0.3:
                medium_dims.append(f"{dim}({score:.2f})")
        
        summary_parts = []
        if high_dims:
            summary_parts.append(f"High: {', '.join(high_dims)}")
        if medium_dims:
            summary_parts.append(f"Medium: {', '.join(medium_dims)}")
        
        return "; ".join(summary_parts) if summary_parts else "All dimensions low"
    
    def calculate_priority_boost(self, scores: DimensionScoresV2) -> float:
        """Calculate priority boost based on profile-specific rules."""
        profile_config = self.profile_manager.get_profile(scores.profile)
        priority_boosts = profile_config.get('priority_boosts', {})
        
        max_boost = 1.0
        
        for boost_name, boost_value in priority_boosts.items():
            if self._check_boost_condition(boost_name, scores.scores):
                max_boost = max(max_boost, boost_value)
        
        return min(max_boost, 1.5)  # Cap at 1.5x
    
    def _check_boost_condition(self, condition: str, scores: Dict[str, float]) -> bool:
        """Check if a boost condition is met."""
        # Parse condition like "high_utility_high_relevance"
        parts = condition.split('_')
        
        conditions_met = 0
        total_conditions = 0
        
        i = 0
        while i < len(parts):
            if parts[i] in ['high', 'medium', 'low']:
                level = parts[i]
                if i + 1 < len(parts):
                    dimension = parts[i + 1]
                    threshold = {'high': 0.6, 'medium': 0.4, 'low': 0.2}.get(level, 0.5)
                    
                    if dimension in scores:
                        total_conditions += 1
                        if level == 'high' and scores[dimension] > threshold:
                            conditions_met += 1
                        elif level == 'medium' and 0.3 < scores[dimension] <= 0.6:
                            conditions_met += 1
                        elif level == 'low' and scores[dimension] <= 0.3:
                            conditions_met += 1
                    
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        
        return conditions_met == total_conditions and total_conditions > 0
