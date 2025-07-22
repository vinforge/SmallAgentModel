"""
PINN-Inspired Loss Balancer for SOF

Implements dynamic effort allocation based on confidence scores, inspired by
Physics-Informed Neural Networks' loss balancing techniques. Adapts cognitive
effort allocation during skill execution based on intermediate results.

Key Features:
- Dynamic effort weighting based on confidence scores
- Adaptive skill execution parameters
- Early termination for high-confidence results
- Resource optimization for complex reasoning chains

Author: SAM Development Team
Version: 1.0.0
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .confidence_weighting import ConfidenceWeighting, WeightingResult

logger = logging.getLogger(__name__)

class EffortLevel(Enum):
    """Effort levels for skill execution."""
    MINIMAL = "minimal"      # 0.2x effort - quick execution
    REDUCED = "reduced"      # 0.5x effort - reduced parameters
    NORMAL = "normal"        # 1.0x effort - standard execution
    ENHANCED = "enhanced"    # 1.5x effort - enhanced parameters
    MAXIMUM = "maximum"      # 2.0x effort - maximum resources

@dataclass
class SkillEffortConfig:
    """Configuration for skill effort allocation."""
    effort_level: EffortLevel
    timeout_multiplier: float
    parameter_adjustments: Dict[str, Any]
    early_termination_threshold: Optional[float] = None

@dataclass
class EffortAllocation:
    """Effort allocation for a skill execution plan."""
    skill_efforts: Dict[str, SkillEffortConfig]
    total_effort_budget: float
    confidence_threshold: float
    early_termination_enabled: bool
    optimized_plan: Optional[List[str]] = None  # Confidence-weighted plan order

class LossBalancer:
    """
    PINN-inspired loss balancer for dynamic effort allocation.
    
    Adapts the computational effort allocated to each skill based on
    intermediate confidence scores, similar to how PINNs balance
    different loss terms based on gradient magnitudes.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.8,
        effort_budget: float = 1.0,
        enable_early_termination: bool = True,
        adaptation_rate: float = 0.3,
        enable_confidence_weighting: bool = True
    ):
        """
        Initialize the loss balancer.

        Args:
            confidence_threshold: Threshold for reducing effort on subsequent skills
            effort_budget: Total computational effort budget
            enable_early_termination: Whether to enable early termination
            adaptation_rate: Rate of effort adaptation (0.0 to 1.0)
            enable_confidence_weighting: Whether to enable confidence-based weighting
        """
        self.confidence_threshold = confidence_threshold
        self.effort_budget = effort_budget
        self.enable_early_termination = enable_early_termination
        self.adaptation_rate = adaptation_rate

        # PINN-inspired confidence weighting system
        self.confidence_weighting = ConfidenceWeighting(
            confidence_threshold=confidence_threshold,
            weight_adaptation_rate=adaptation_rate
        ) if enable_confidence_weighting else None

        # Effort allocation history for learning
        self.allocation_history: List[Dict[str, Any]] = []
        
        # Default effort configurations for different skill types
        self.default_efforts = {
            "MemoryRetrievalSkill": {
                EffortLevel.MINIMAL: {"max_results": 3, "timeout": 5.0},
                EffortLevel.REDUCED: {"max_results": 5, "timeout": 10.0},
                EffortLevel.NORMAL: {"max_results": 10, "timeout": 15.0},
                EffortLevel.ENHANCED: {"max_results": 15, "timeout": 25.0},
                EffortLevel.MAXIMUM: {"max_results": 20, "timeout": 40.0}
            },
            "ConflictDetectorSkill": {
                EffortLevel.MINIMAL: {"analysis_depth": "shallow", "timeout": 3.0},
                EffortLevel.REDUCED: {"analysis_depth": "basic", "timeout": 8.0},
                EffortLevel.NORMAL: {"analysis_depth": "standard", "timeout": 15.0},
                EffortLevel.ENHANCED: {"analysis_depth": "deep", "timeout": 25.0},
                EffortLevel.MAXIMUM: {"analysis_depth": "comprehensive", "timeout": 40.0}
            },
            "ResponseGenerationSkill": {
                EffortLevel.MINIMAL: {"max_tokens": 100, "temperature": 0.3},
                EffortLevel.REDUCED: {"max_tokens": 200, "temperature": 0.5},
                EffortLevel.NORMAL: {"max_tokens": 400, "temperature": 0.7},
                EffortLevel.ENHANCED: {"max_tokens": 600, "temperature": 0.7},
                EffortLevel.MAXIMUM: {"max_tokens": 800, "temperature": 0.8}
            }
        }
        
        logger.info(f"LossBalancer initialized with confidence_threshold={confidence_threshold}")
    
    def allocate_effort(
        self,
        plan: List[str],
        initial_confidence: float = 0.5,
        query_complexity: Optional[str] = None
    ) -> EffortAllocation:
        """
        Allocate effort across skills in the execution plan.
        
        Args:
            plan: List of skill names to execute
            initial_confidence: Initial confidence in the plan
            query_complexity: Estimated query complexity ("simple", "medium", "complex")
            
        Returns:
            Effort allocation configuration
        """
        logger.info(f"Allocating effort for plan: {plan}")
        
        # Apply confidence-based weighting if enabled
        optimized_plan = plan
        if self.confidence_weighting:
            weighting_result = self.confidence_weighting.calculate_weighted_plan(
                original_plan=plan,
                current_confidence=initial_confidence,
                executed_skills=[],
                query_complexity=query_complexity
            )
            optimized_plan = weighting_result.weighted_plan
            logger.info(f"Applied confidence weighting: {plan} → {optimized_plan}")

        # Determine base effort level from query complexity
        base_effort = self._determine_base_effort(query_complexity, initial_confidence)

        # Allocate effort to each skill
        skill_efforts = {}
        remaining_budget = self.effort_budget

        for i, skill_name in enumerate(optimized_plan):
            # Calculate effort for this skill
            effort_config = self._calculate_skill_effort(
                skill_name=skill_name,
                position=i,
                total_skills=len(optimized_plan),
                base_effort=base_effort,
                remaining_budget=remaining_budget
            )

            # Apply confidence weighting to effort if available
            if self.confidence_weighting and hasattr(weighting_result, 'resource_allocation'):
                resource_multiplier = weighting_result.resource_allocation.get(skill_name, 1.0)
                effort_config = self._apply_resource_multiplier(effort_config, resource_multiplier)

            skill_efforts[skill_name] = effort_config
            remaining_budget -= self._get_effort_cost(effort_config.effort_level)

            logger.debug(f"Allocated {effort_config.effort_level.value} effort to {skill_name}")
        
        allocation = EffortAllocation(
            skill_efforts=skill_efforts,
            total_effort_budget=self.effort_budget,
            confidence_threshold=self.confidence_threshold,
            early_termination_enabled=self.enable_early_termination
        )

        # Store optimized plan in allocation for coordinator to use
        allocation.optimized_plan = optimized_plan
        
        return allocation
    
    def adapt_effort(
        self,
        allocation: EffortAllocation,
        executed_skills: List[str],
        intermediate_confidence: float,
        remaining_skills: List[str]
    ) -> EffortAllocation:
        """
        Adapt effort allocation based on intermediate results.
        
        Args:
            allocation: Current effort allocation
            executed_skills: Skills that have been executed
            intermediate_confidence: Confidence after executed skills
            remaining_skills: Skills yet to be executed
            
        Returns:
            Updated effort allocation
        """
        if not remaining_skills:
            return allocation
        
        logger.info(f"Adapting effort based on confidence {intermediate_confidence:.3f}")
        
        # Calculate adaptation factor based on confidence
        if intermediate_confidence > self.confidence_threshold:
            # High confidence - reduce effort on remaining skills
            adaptation_factor = 1.0 - (self.adaptation_rate * 
                                     (intermediate_confidence - self.confidence_threshold))
            adaptation_factor = max(0.2, adaptation_factor)  # Minimum 20% effort
            
            logger.info(f"High confidence detected, reducing effort by factor {adaptation_factor:.2f}")
            
        elif intermediate_confidence < 0.3:
            # Low confidence - increase effort on remaining skills
            adaptation_factor = 1.0 + (self.adaptation_rate * (0.3 - intermediate_confidence))
            adaptation_factor = min(2.0, adaptation_factor)  # Maximum 200% effort
            
            logger.info(f"Low confidence detected, increasing effort by factor {adaptation_factor:.2f}")
            
        else:
            # Medium confidence - maintain current effort
            adaptation_factor = 1.0
        
        # Update effort allocation for remaining skills
        updated_efforts = allocation.skill_efforts.copy()
        
        for skill_name in remaining_skills:
            if skill_name in updated_efforts:
                current_config = updated_efforts[skill_name]
                new_effort_level = self._adjust_effort_level(
                    current_config.effort_level,
                    adaptation_factor
                )
                
                updated_efforts[skill_name] = SkillEffortConfig(
                    effort_level=new_effort_level,
                    timeout_multiplier=current_config.timeout_multiplier * adaptation_factor,
                    parameter_adjustments=self._adjust_parameters(
                        skill_name,
                        current_config.parameter_adjustments,
                        adaptation_factor
                    ),
                    early_termination_threshold=current_config.early_termination_threshold
                )
        
        # Update allocation
        allocation.skill_efforts = updated_efforts
        
        return allocation
    
    def should_terminate_early(
        self,
        allocation: EffortAllocation,
        current_confidence: float,
        executed_skills: List[str],
        remaining_skills: List[str]
    ) -> bool:
        """
        Determine if execution should terminate early based on confidence.
        
        Args:
            allocation: Current effort allocation
            current_confidence: Current confidence level
            executed_skills: Skills that have been executed
            remaining_skills: Skills yet to be executed
            
        Returns:
            True if execution should terminate early
        """
        if not allocation.early_termination_enabled:
            return False
        
        # Check if confidence is very high and we have basic skills completed
        if (current_confidence > 0.9 and 
            len(executed_skills) >= 2 and 
            "MemoryRetrievalSkill" in executed_skills):
            
            logger.info(f"Early termination triggered: confidence {current_confidence:.3f}")
            return True
        
        return False
    
    def _determine_base_effort(
        self,
        query_complexity: Optional[str],
        initial_confidence: float
    ) -> EffortLevel:
        """Determine base effort level from query complexity and confidence."""
        if query_complexity == "simple" or initial_confidence > 0.8:
            return EffortLevel.REDUCED
        elif query_complexity == "complex" or initial_confidence < 0.3:
            return EffortLevel.ENHANCED
        else:
            return EffortLevel.NORMAL
    
    def _calculate_skill_effort(
        self,
        skill_name: str,
        position: int,
        total_skills: int,
        base_effort: EffortLevel,
        remaining_budget: float
    ) -> SkillEffortConfig:
        """Calculate effort configuration for a specific skill."""
        # Adjust effort based on position in plan
        if position == 0:
            # First skill gets normal effort
            effort_level = base_effort
        elif position == total_skills - 1:
            # Last skill (usually ResponseGeneration) gets enhanced effort
            effort_level = EffortLevel.ENHANCED if base_effort != EffortLevel.MINIMAL else EffortLevel.NORMAL
        else:
            # Middle skills get base effort
            effort_level = base_effort
        
        # Adjust based on remaining budget
        effort_cost = self._get_effort_cost(effort_level)
        if effort_cost > remaining_budget:
            effort_level = self._reduce_effort_level(effort_level)
        
        # Get parameters for this skill and effort level
        parameters = self._get_skill_parameters(skill_name, effort_level)
        
        return SkillEffortConfig(
            effort_level=effort_level,
            timeout_multiplier=1.0,
            parameter_adjustments=parameters,
            early_termination_threshold=0.85 if effort_level in [EffortLevel.MINIMAL, EffortLevel.REDUCED] else None
        )
    
    def _get_effort_cost(self, effort_level: EffortLevel) -> float:
        """Get computational cost for an effort level."""
        costs = {
            EffortLevel.MINIMAL: 0.2,
            EffortLevel.REDUCED: 0.5,
            EffortLevel.NORMAL: 1.0,
            EffortLevel.ENHANCED: 1.5,
            EffortLevel.MAXIMUM: 2.0
        }
        return costs.get(effort_level, 1.0)
    
    def _adjust_effort_level(self, current_level: EffortLevel, factor: float) -> EffortLevel:
        """Adjust effort level by a factor."""
        levels = [EffortLevel.MINIMAL, EffortLevel.REDUCED, EffortLevel.NORMAL, 
                 EffortLevel.ENHANCED, EffortLevel.MAXIMUM]
        
        current_index = levels.index(current_level)
        
        if factor < 1.0:
            # Reduce effort
            new_index = max(0, current_index - 1)
        elif factor > 1.0:
            # Increase effort
            new_index = min(len(levels) - 1, current_index + 1)
        else:
            new_index = current_index
        
        return levels[new_index]
    
    def _reduce_effort_level(self, effort_level: EffortLevel) -> EffortLevel:
        """Reduce effort level by one step."""
        return self._adjust_effort_level(effort_level, 0.5)
    
    def _get_skill_parameters(self, skill_name: str, effort_level: EffortLevel) -> Dict[str, Any]:
        """Get parameters for a skill at a specific effort level."""
        if skill_name in self.default_efforts:
            return self.default_efforts[skill_name].get(effort_level, {})
        return {}
    
    def _adjust_parameters(
        self,
        skill_name: str,
        current_params: Dict[str, Any],
        factor: float
    ) -> Dict[str, Any]:
        """Adjust skill parameters based on adaptation factor."""
        adjusted = current_params.copy()
        
        # Adjust numeric parameters
        for key, value in adjusted.items():
            if isinstance(value, (int, float)) and key in ["max_results", "timeout", "max_tokens"]:
                adjusted[key] = max(1, int(value * factor))
        
        return adjusted

    def _apply_resource_multiplier(self, effort_config: SkillEffortConfig, multiplier: float) -> SkillEffortConfig:
        """Apply resource multiplier to effort configuration."""
        # Adjust effort level based on multiplier
        if multiplier > 1.2:
            # High resource allocation - increase effort
            new_effort_level = self._adjust_effort_level(effort_config.effort_level, 1.5)
        elif multiplier < 0.8:
            # Low resource allocation - decrease effort
            new_effort_level = self._adjust_effort_level(effort_config.effort_level, 0.5)
        else:
            new_effort_level = effort_config.effort_level

        # Adjust timeout multiplier
        new_timeout_multiplier = effort_config.timeout_multiplier * multiplier

        # Adjust parameters
        adjusted_params = effort_config.parameter_adjustments.copy()
        for key, value in adjusted_params.items():
            if isinstance(value, (int, float)) and key in ["max_results", "timeout", "max_tokens"]:
                adjusted_params[key] = max(1, int(value * multiplier))

        return SkillEffortConfig(
            effort_level=new_effort_level,
            timeout_multiplier=new_timeout_multiplier,
            parameter_adjustments=adjusted_params,
            early_termination_threshold=effort_config.early_termination_threshold
        )

    def get_effort_statistics(self) -> Dict[str, Any]:
        """Get statistics about effort allocation."""
        base_stats = {
            "total_allocations": len(self.allocation_history),
            "effort_budget": self.effort_budget,
            "confidence_threshold": self.confidence_threshold,
            "adaptation_rate": self.adaptation_rate,
            "early_termination_enabled": self.enable_early_termination
        }

        if not self.allocation_history:
            base_stats.update({
                "average_final_confidence": 0.0,
                "early_termination_rate": 0.0
            })
            return base_stats

        total_allocations = len(self.allocation_history)
        avg_confidence = sum(h.get("final_confidence", 0.5) for h in self.allocation_history) / total_allocations
        early_terminations = sum(1 for h in self.allocation_history if h.get("early_termination", False))

        base_stats.update({
            "average_final_confidence": avg_confidence,
            "early_termination_rate": early_terminations / total_allocations
        })

        # Add confidence weighting statistics if enabled
        if self.confidence_weighting:
            weighting_stats = self.confidence_weighting.get_weighting_statistics()
            base_stats["confidence_weighting"] = weighting_stats

        return base_stats
