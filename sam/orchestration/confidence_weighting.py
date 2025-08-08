"""
PINN-Inspired Confidence-Based Weighting System

Implements adaptive skill weighting based on intermediate confidence scores,
inspired by Physics-Informed Neural Networks' adaptive loss weighting techniques.
Dynamically adjusts skill execution order and resource allocation.

Key Features:
- Confidence-based skill prioritization
- Dynamic execution order optimization
- Adaptive resource allocation
- Performance-based weight adjustment
- Integration with Loss Balancer

Author: SAM Development Team
Version: 1.0.0
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class SkillCategory(Enum):
    """Categories of skills for weighting purposes."""
    RETRIEVAL = "retrieval"           # Memory and information retrieval
    ANALYSIS = "analysis"             # Conflict detection, reasoning
    SYNTHESIS = "synthesis"           # Knowledge integration, implicit reasoning
    GENERATION = "generation"         # Response generation, output creation
    VALIDATION = "validation"         # Content vetting, quality assurance
    TOOLS = "tools"                   # External tools, calculators, web

@dataclass
class SkillWeight:
    """Weight configuration for a skill."""
    base_weight: float                # Base importance weight (0.0 to 1.0)
    confidence_sensitivity: float     # How much confidence affects weight
    execution_priority: int           # Execution order priority (lower = earlier)
    resource_multiplier: float        # Resource allocation multiplier
    category: SkillCategory           # Skill category for grouping

@dataclass
class WeightingResult:
    """Result of confidence-based weighting calculation."""
    weighted_plan: List[str]          # Reordered plan based on weights
    skill_weights: Dict[str, float]   # Final weights for each skill
    execution_order: List[str]        # Optimized execution order
    confidence_impact: float          # How much confidence affected weighting
    resource_allocation: Dict[str, float]  # Resource allocation per skill

class ConfidenceWeighting:
    """
    PINN-inspired confidence-based weighting system.
    
    Adapts skill execution order and resource allocation based on
    intermediate confidence scores, similar to how PINNs adapt
    loss term weights during training.
    """
    
    def __init__(
        self,
        enable_dynamic_reordering: bool = True,
        confidence_threshold: float = 0.7,
        weight_adaptation_rate: float = 0.3
    ):
        """
        Initialize confidence-based weighting system.
        
        Args:
            enable_dynamic_reordering: Whether to reorder skills based on confidence
            confidence_threshold: Threshold for high confidence adjustments
            weight_adaptation_rate: Rate of weight adaptation (0.0 to 1.0)
        """
        self.enable_dynamic_reordering = enable_dynamic_reordering
        self.confidence_threshold = confidence_threshold
        self.weight_adaptation_rate = weight_adaptation_rate
        
        # Initialize skill weight configurations
        self.skill_weights = self._initialize_skill_weights()
        
        # Performance tracking for weight adaptation
        self.performance_history: List[Dict[str, Any]] = []
        
        logger.info(f"ConfidenceWeighting initialized (reordering: {enable_dynamic_reordering})")
    
    def _initialize_skill_weights(self) -> Dict[str, SkillWeight]:
        """Initialize default skill weight configurations."""
        return {
            # Retrieval Skills
            "MemoryRetrievalSkill": SkillWeight(
                base_weight=0.9,
                confidence_sensitivity=0.3,
                execution_priority=1,
                resource_multiplier=1.0,
                category=SkillCategory.RETRIEVAL
            ),
            
            # Analysis Skills
            "ConflictDetectorSkill": SkillWeight(
                base_weight=0.7,
                confidence_sensitivity=0.5,
                execution_priority=2,
                resource_multiplier=0.8,
                category=SkillCategory.ANALYSIS
            ),
            
            "ImplicitKnowledgeSkill": SkillWeight(
                base_weight=0.8,
                confidence_sensitivity=0.4,
                execution_priority=3,
                resource_multiplier=1.2,
                category=SkillCategory.SYNTHESIS
            ),
            
            # Validation Skills
            "ContentVettingSkill": SkillWeight(
                base_weight=0.6,
                confidence_sensitivity=0.6,
                execution_priority=4,
                resource_multiplier=0.7,
                category=SkillCategory.VALIDATION
            ),
            
            # Tools
            "CalculatorTool": SkillWeight(
                base_weight=0.5,
                confidence_sensitivity=0.2,
                execution_priority=2,
                resource_multiplier=0.5,
                category=SkillCategory.TOOLS
            ),
            
            "AgentZeroWebBrowserTool": SkillWeight(
                base_weight=0.7,
                confidence_sensitivity=0.4,
                execution_priority=3,
                resource_multiplier=1.5,
                category=SkillCategory.TOOLS
            ),
            
            # Generation Skills (usually last)
            "ResponseGenerationSkill": SkillWeight(
                base_weight=1.0,
                confidence_sensitivity=0.1,
                execution_priority=10,
                resource_multiplier=1.0,
                category=SkillCategory.GENERATION
            ),
            
            # MEMOIR Skills
            "MEMOIR_EditSkill": SkillWeight(
                base_weight=0.8,
                confidence_sensitivity=0.3,
                execution_priority=5,
                resource_multiplier=1.3,
                category=SkillCategory.SYNTHESIS
            )
        }
    
    def calculate_weighted_plan(
        self,
        original_plan: List[str],
        current_confidence: float,
        executed_skills: List[str],
        query_complexity: Optional[str] = None
    ) -> WeightingResult:
        """
        Calculate confidence-weighted execution plan.
        
        Args:
            original_plan: Original skill execution plan
            current_confidence: Current confidence level (0.0 to 1.0)
            executed_skills: Skills already executed
            query_complexity: Query complexity level
            
        Returns:
            Weighting result with optimized plan and allocations
        """
        remaining_skills = [skill for skill in original_plan if skill not in executed_skills]
        
        if not remaining_skills:
            return WeightingResult(
                weighted_plan=[],
                skill_weights={},
                execution_order=[],
                confidence_impact=0.0,
                resource_allocation={}
            )
        
        # Calculate confidence-adjusted weights
        adjusted_weights = {}
        confidence_impact = 0.0
        
        for skill_name in remaining_skills:
            base_config = self.skill_weights.get(skill_name)
            if not base_config:
                # Default weight for unknown skills
                adjusted_weights[skill_name] = 0.5
                continue
            
            # Calculate confidence adjustment
            confidence_adjustment = self._calculate_confidence_adjustment(
                base_config, current_confidence
            )
            
            # Apply query complexity adjustment
            complexity_adjustment = self._calculate_complexity_adjustment(
                base_config, query_complexity
            )
            
            # Final weight calculation
            final_weight = (
                base_config.base_weight * 
                (1.0 + confidence_adjustment) * 
                (1.0 + complexity_adjustment)
            )
            
            # Clamp weight to valid range
            final_weight = max(0.1, min(1.0, final_weight))
            adjusted_weights[skill_name] = final_weight
            
            confidence_impact += abs(confidence_adjustment)
        
        # Calculate resource allocation
        resource_allocation = self._calculate_resource_allocation(
            remaining_skills, adjusted_weights, current_confidence
        )
        
        # Determine execution order
        execution_order = self._determine_execution_order(
            remaining_skills, adjusted_weights, current_confidence
        )
        
        # Create weighted plan
        weighted_plan = execution_order if self.enable_dynamic_reordering else remaining_skills
        
        return WeightingResult(
            weighted_plan=weighted_plan,
            skill_weights=adjusted_weights,
            execution_order=execution_order,
            confidence_impact=confidence_impact / len(remaining_skills),
            resource_allocation=resource_allocation
        )
    
    def _calculate_confidence_adjustment(
        self,
        skill_config: SkillWeight,
        confidence: float
    ) -> float:
        """Calculate confidence-based weight adjustment."""
        if confidence > self.confidence_threshold:
            # High confidence - reduce weight for analysis/validation skills
            if skill_config.category in [SkillCategory.ANALYSIS, SkillCategory.VALIDATION]:
                return -skill_config.confidence_sensitivity * (confidence - self.confidence_threshold)
            else:
                return 0.0
        else:
            # Low confidence - increase weight for analysis/validation skills
            if skill_config.category in [SkillCategory.ANALYSIS, SkillCategory.VALIDATION]:
                return skill_config.confidence_sensitivity * (self.confidence_threshold - confidence)
            else:
                return 0.0
    
    def _calculate_complexity_adjustment(
        self,
        skill_config: SkillWeight,
        query_complexity: Optional[str]
    ) -> float:
        """Calculate query complexity-based weight adjustment."""
        if not query_complexity:
            return 0.0
        
        complexity_multipliers = {
            "simple": -0.2,    # Reduce weights for simple queries
            "medium": 0.0,     # No adjustment for medium queries
            "complex": 0.3     # Increase weights for complex queries
        }
        
        base_multiplier = complexity_multipliers.get(query_complexity, 0.0)
        
        # Apply category-specific adjustments
        if skill_config.category == SkillCategory.SYNTHESIS and query_complexity == "complex":
            return base_multiplier * 1.5  # Extra boost for synthesis on complex queries
        elif skill_config.category == SkillCategory.TOOLS and query_complexity == "simple":
            return base_multiplier * 2.0  # Reduce tool usage for simple queries
        else:
            return base_multiplier
    
    def _calculate_resource_allocation(
        self,
        skills: List[str],
        weights: Dict[str, float],
        confidence: float
    ) -> Dict[str, float]:
        """Calculate resource allocation based on weights and confidence."""
        total_weight = sum(weights.values())
        if total_weight == 0:
            return {skill: 1.0 / len(skills) for skill in skills}
        
        # Base allocation proportional to weights
        base_allocation = {
            skill: weights[skill] / total_weight
            for skill in skills
        }
        
        # Adjust based on confidence
        if confidence > self.confidence_threshold:
            # High confidence - concentrate resources on key skills
            max_skill = max(skills, key=lambda s: weights[s])
            for skill in skills:
                if skill == max_skill:
                    base_allocation[skill] *= 1.3
                else:
                    base_allocation[skill] *= 0.8
        
        # Normalize to ensure sum = 1.0
        total_allocation = sum(base_allocation.values())
        return {
            skill: allocation / total_allocation
            for skill, allocation in base_allocation.items()
        }
    
    def _determine_execution_order(
        self,
        skills: List[str],
        weights: Dict[str, float],
        confidence: float
    ) -> List[str]:
        """Determine optimal execution order based on weights and priorities."""
        if not self.enable_dynamic_reordering:
            return skills
        
        # Create skill priority tuples (priority, weight, skill_name)
        skill_priorities = []
        
        for skill in skills:
            config = self.skill_weights.get(skill)
            if config:
                # Adjust priority based on confidence
                adjusted_priority = config.execution_priority
                
                if confidence < 0.5:
                    # Low confidence - prioritize analysis skills
                    if config.category in [SkillCategory.ANALYSIS, SkillCategory.VALIDATION]:
                        adjusted_priority -= 1
                elif confidence > self.confidence_threshold:
                    # High confidence - deprioritize analysis skills
                    if config.category in [SkillCategory.ANALYSIS, SkillCategory.VALIDATION]:
                        adjusted_priority += 2
                
                skill_priorities.append((adjusted_priority, -weights[skill], skill))
            else:
                # Unknown skill - use default priority
                skill_priorities.append((5, -weights.get(skill, 0.5), skill))
        
        # Sort by priority (lower first), then by weight (higher first)
        skill_priorities.sort()
        
        return [skill for _, _, skill in skill_priorities]
    
    def adapt_weights(
        self,
        skill_name: str,
        performance_score: float,
        confidence_accuracy: float
    ) -> None:
        """
        Adapt skill weights based on performance feedback.
        
        Args:
            skill_name: Name of the skill to adapt
            performance_score: Performance score (0.0 to 1.0)
            confidence_accuracy: How accurate the confidence prediction was
        """
        if skill_name not in self.skill_weights:
            return
        
        config = self.skill_weights[skill_name]
        
        # Adjust confidence sensitivity based on accuracy
        if confidence_accuracy > 0.8:
            # High accuracy - increase sensitivity
            config.confidence_sensitivity *= (1.0 + self.weight_adaptation_rate * 0.1)
        elif confidence_accuracy < 0.5:
            # Low accuracy - decrease sensitivity
            config.confidence_sensitivity *= (1.0 - self.weight_adaptation_rate * 0.1)
        
        # Clamp sensitivity to valid range
        config.confidence_sensitivity = max(0.1, min(1.0, config.confidence_sensitivity))
        
        # Record performance for analysis
        self.performance_history.append({
            "skill_name": skill_name,
            "performance_score": performance_score,
            "confidence_accuracy": confidence_accuracy,
            "adapted_sensitivity": config.confidence_sensitivity
        })
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-50:]
    
    def get_weighting_statistics(self) -> Dict[str, Any]:
        """Get statistics about confidence weighting performance."""
        if not self.performance_history:
            return {
                "total_adaptations": 0,
                "average_performance": 0.0,
                "average_confidence_accuracy": 0.0
            }
        
        recent_history = self.performance_history[-20:]  # Last 20 adaptations
        
        avg_performance = sum(h["performance_score"] for h in recent_history) / len(recent_history)
        avg_accuracy = sum(h["confidence_accuracy"] for h in recent_history) / len(recent_history)
        
        # Calculate skill-specific statistics
        skill_stats = {}
        for skill_name, config in self.skill_weights.items():
            skill_history = [h for h in recent_history if h["skill_name"] == skill_name]
            if skill_history:
                skill_stats[skill_name] = {
                    "adaptations": len(skill_history),
                    "avg_performance": sum(h["performance_score"] for h in skill_history) / len(skill_history),
                    "current_sensitivity": config.confidence_sensitivity
                }
        
        return {
            "total_adaptations": len(self.performance_history),
            "recent_average_performance": avg_performance,
            "recent_average_confidence_accuracy": avg_accuracy,
            "skill_statistics": skill_stats,
            "confidence_threshold": self.confidence_threshold,
            "adaptation_rate": self.weight_adaptation_rate,
            "dynamic_reordering": self.enable_dynamic_reordering
        }
