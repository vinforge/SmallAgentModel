"""
TPV Controller Module for SAM
Phase 2A - Enhanced TPV Controller with Dissonance Awareness

This module provides intelligent control over reasoning processes
with integrated cognitive dissonance detection and response.
"""

import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from .tpv_monitor import ReasoningTrace, ReasoningStep

logger = logging.getLogger(__name__)

class ControlMode(Enum):
    """Control modes for the reasoning controller."""
    PASSIVE = "passive"      # Monitor only, no intervention
    ACTIVE = "active"        # Active intervention based on conditions
    AGGRESSIVE = "aggressive" # Strict control with low thresholds

class ControlDecision(Enum):
    """Possible control decisions."""
    CONTINUE = "continue"
    STOP_COMPLETION = "stop_completion"
    STOP_PLATEAU = "stop_plateau"
    STOP_DISSONANCE = "stop_dissonance"
    STOP_MAX_TOKENS = "stop_max_tokens"
    STOP_ERROR = "stop_error"

@dataclass
class ControlConfig:
    """Configuration for the reasoning controller."""
    # TPV-based control parameters
    completion_threshold: float = 0.92
    max_tokens: int = 500
    min_steps: int = 2
    plateau_patience: int = 3
    plateau_threshold: float = 0.005
    
    # Dissonance-based control parameters
    dissonance_threshold: float = 0.85
    dissonance_patience: int = 4
    enable_dissonance_control: bool = True
    
    # Advanced control parameters
    adaptive_thresholds: bool = False
    context_aware_control: bool = False
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ControlConfig':
        """Create ControlConfig from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})

@dataclass
class ControlResult:
    """Result of a control decision."""
    decision: ControlDecision
    reason: str
    confidence: float
    metadata: Dict[str, Any]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'decision': self.decision.value,
            'reason': self.reason,
            'confidence': self.confidence,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }

class ReasoningController:
    """
    Intelligent controller for reasoning processes with dissonance awareness.
    
    Monitors reasoning progress and makes decisions about when to continue
    or stop generation based on TPV scores and cognitive dissonance.
    """
    
    def __init__(self, 
                 mode: ControlMode = ControlMode.PASSIVE,
                 config: Optional[ControlConfig] = None):
        """
        Initialize the reasoning controller.
        
        Args:
            mode: Control mode (passive, active, aggressive)
            config: Control configuration
        """
        self.mode = mode
        self.config = config or ControlConfig()
        
        # Statistics tracking
        self.total_decisions = 0
        self.decisions_by_type: Dict[ControlDecision, int] = {
            decision: 0 for decision in ControlDecision
        }
        self.control_history: List[ControlResult] = []
        
        # Performance tracking
        self.average_decision_time = 0.0
        self.decision_times: List[float] = []
        
        logger.info(f"ReasoningController initialized: mode={mode.value}, "
                   f"dissonance_control={self.config.enable_dissonance_control}")
    
    def should_continue(self, trace: ReasoningTrace, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Determine whether reasoning should continue.
        
        Args:
            trace: Current reasoning trace
            context: Optional context for decision making
            
        Returns:
            True if reasoning should continue, False if it should stop
        """
        start_time = time.time()
        
        try:
            # Make control decision
            control_result = self._make_control_decision(trace, context)
            
            # Record decision
            self._record_decision(control_result)
            
            # Log decision
            should_continue = control_result.decision == ControlDecision.CONTINUE
            logger.info(f"Control decision for {trace.query_id}: {control_result.decision.value} "
                       f"({control_result.reason})")
            
            return should_continue
            
        except Exception as e:
            logger.error(f"Error in control decision: {e}")
            # Default to continue on error in passive mode
            return self.mode == ControlMode.PASSIVE
        
        finally:
            # Track decision time
            decision_time = time.time() - start_time
            self.decision_times.append(decision_time)
            if len(self.decision_times) > 100:  # Keep only recent times
                self.decision_times = self.decision_times[-100:]
            
            if self.decision_times:
                self.average_decision_time = sum(self.decision_times) / len(self.decision_times)
    
    def _make_control_decision(self, trace: ReasoningTrace, context: Optional[Dict[str, Any]]) -> ControlResult:
        """
        Make a control decision based on trace analysis.
        
        Args:
            trace: Reasoning trace to analyze
            context: Optional context information
            
        Returns:
            ControlResult with decision and reasoning
        """
        # In passive mode, always continue
        if self.mode == ControlMode.PASSIVE:
            return ControlResult(
                decision=ControlDecision.CONTINUE,
                reason="Passive mode - monitoring only",
                confidence=1.0,
                metadata={'mode': 'passive'},
                timestamp=time.time()
            )
        
        # Check various stop conditions in order of priority
        
        # 1. Check maximum tokens
        total_tokens = sum(step.token_count for step in trace.steps)
        if total_tokens >= self.config.max_tokens:
            return ControlResult(
                decision=ControlDecision.STOP_MAX_TOKENS,
                reason=f"Maximum tokens reached: {total_tokens}/{self.config.max_tokens}",
                confidence=1.0,
                metadata={'total_tokens': total_tokens, 'max_tokens': self.config.max_tokens},
                timestamp=time.time()
            )
        
        # 2. Check minimum steps requirement
        if len(trace.steps) < self.config.min_steps:
            return ControlResult(
                decision=ControlDecision.CONTINUE,
                reason=f"Minimum steps not reached: {len(trace.steps)}/{self.config.min_steps}",
                confidence=0.8,
                metadata={'current_steps': len(trace.steps), 'min_steps': self.config.min_steps},
                timestamp=time.time()
            )
        
        # 3. Check completion threshold
        if trace.current_score >= self.config.completion_threshold:
            return ControlResult(
                decision=ControlDecision.STOP_COMPLETION,
                reason=f"Completion threshold reached: {trace.current_score:.3f}>={self.config.completion_threshold}",
                confidence=0.9,
                metadata={'current_score': trace.current_score, 'threshold': self.config.completion_threshold},
                timestamp=time.time()
            )
        
        # 4. Check for plateau (lack of progress)
        plateau_result = self._check_plateau(trace)
        if plateau_result:
            return plateau_result
        
        # 5. Check for high cognitive dissonance
        if self.config.enable_dissonance_control:
            dissonance_result = self._check_dissonance(trace)
            if dissonance_result:
                return dissonance_result
        
        # 6. Apply adaptive thresholds if enabled
        if self.config.adaptive_thresholds and context:
            adaptive_result = self._apply_adaptive_control(trace, context)
            if adaptive_result:
                return adaptive_result
        
        # Default: continue reasoning
        return ControlResult(
            decision=ControlDecision.CONTINUE,
            reason="All conditions satisfied for continuation",
            confidence=0.7,
            metadata={'current_score': trace.current_score, 'steps': len(trace.steps)},
            timestamp=time.time()
        )
    
    def _check_plateau(self, trace: ReasoningTrace) -> Optional[ControlResult]:
        """
        Check if reasoning has plateaued (no significant progress).
        
        Args:
            trace: Reasoning trace to analyze
            
        Returns:
            ControlResult if plateau detected, None otherwise
        """
        if len(trace.steps) < self.config.plateau_patience:
            return None
        
        # Get recent scores
        recent_scores = [step.tpv_score for step in trace.steps[-self.config.plateau_patience:]]
        
        # Check if all recent scores are within plateau threshold
        max_score = max(recent_scores)
        min_score = min(recent_scores)
        score_range = max_score - min_score
        
        if score_range <= self.config.plateau_threshold:
            return ControlResult(
                decision=ControlDecision.STOP_PLATEAU,
                reason=f"Plateau detected: score range {score_range:.4f} <= {self.config.plateau_threshold}",
                confidence=0.8,
                metadata={
                    'score_range': score_range,
                    'plateau_threshold': self.config.plateau_threshold,
                    'recent_scores': recent_scores
                },
                timestamp=time.time()
            )
        
        return None
    
    def _check_dissonance(self, trace: ReasoningTrace) -> Optional[ControlResult]:
        """
        Check for high cognitive dissonance patterns.
        
        Args:
            trace: Reasoning trace to analyze
            
        Returns:
            ControlResult if high dissonance detected, None otherwise
        """
        if len(trace.steps) < self.config.dissonance_patience:
            return None
        
        # Get recent dissonance scores
        recent_steps = trace.steps[-self.config.dissonance_patience:]
        recent_dissonance = [step.dissonance_score for step in recent_steps 
                           if step.dissonance_score is not None]
        
        if not recent_dissonance:
            return None  # No dissonance data available
        
        # Check if all recent dissonance scores are above threshold
        if all(score > self.config.dissonance_threshold for score in recent_dissonance):
            avg_dissonance = sum(recent_dissonance) / len(recent_dissonance)
            
            return ControlResult(
                decision=ControlDecision.STOP_DISSONANCE,
                reason=f"High cognitive dissonance detected: avg={avg_dissonance:.3f} > {self.config.dissonance_threshold}",
                confidence=0.85,
                metadata={
                    'avg_dissonance': avg_dissonance,
                    'dissonance_threshold': self.config.dissonance_threshold,
                    'recent_dissonance': recent_dissonance,
                    'patience_steps': self.config.dissonance_patience
                },
                timestamp=time.time()
            )
        
        return None
    
    def _apply_adaptive_control(self, trace: ReasoningTrace, context: Dict[str, Any]) -> Optional[ControlResult]:
        """
        Apply adaptive control based on context.
        
        Args:
            trace: Reasoning trace to analyze
            context: Context information
            
        Returns:
            ControlResult if adaptive control triggers, None otherwise
        """
        # Example adaptive control logic
        # This can be extended based on specific requirements
        
        query_complexity = context.get('query_complexity', 'medium')
        user_expertise = context.get('user_expertise', 'general')
        
        # Adjust thresholds based on context
        adjusted_completion_threshold = self.config.completion_threshold
        
        if query_complexity == 'high':
            adjusted_completion_threshold *= 0.95  # Lower threshold for complex queries
        elif query_complexity == 'low':
            adjusted_completion_threshold *= 1.05  # Higher threshold for simple queries
        
        if user_expertise == 'expert':
            adjusted_completion_threshold *= 1.02  # Slightly higher for experts
        elif user_expertise == 'beginner':
            adjusted_completion_threshold *= 0.98  # Slightly lower for beginners
        
        # Check adjusted completion threshold
        if trace.current_score >= adjusted_completion_threshold:
            return ControlResult(
                decision=ControlDecision.STOP_COMPLETION,
                reason=f"Adaptive completion threshold reached: {trace.current_score:.3f}>={adjusted_completion_threshold:.3f}",
                confidence=0.85,
                metadata={
                    'adaptive_threshold': adjusted_completion_threshold,
                    'original_threshold': self.config.completion_threshold,
                    'context': context
                },
                timestamp=time.time()
            )
        
        return None
    
    def _record_decision(self, result: ControlResult):
        """
        Record a control decision for statistics.
        
        Args:
            result: Control result to record
        """
        self.total_decisions += 1
        self.decisions_by_type[result.decision] += 1
        self.control_history.append(result)
        
        # Keep only recent history to manage memory
        if len(self.control_history) > 1000:
            self.control_history = self.control_history[-1000:]
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get controller status and statistics.
        
        Returns:
            Status dictionary
        """
        return {
            'mode': self.mode.value,
            'total_decisions': self.total_decisions,
            'decisions_by_type': {k.value: v for k, v in self.decisions_by_type.items()},
            'average_decision_time': self.average_decision_time,
            'config': {
                'completion_threshold': self.config.completion_threshold,
                'dissonance_threshold': self.config.dissonance_threshold,
                'dissonance_control_enabled': self.config.enable_dissonance_control,
                'max_tokens': self.config.max_tokens,
                'min_steps': self.config.min_steps
            }
        }
    
    def get_control_statistics(self) -> Dict[str, Any]:
        """
        Get detailed control statistics.
        
        Returns:
            Detailed statistics dictionary
        """
        stats = self.get_status()
        
        if self.control_history:
            # Calculate decision distribution
            recent_decisions = self.control_history[-100:]  # Last 100 decisions
            decision_distribution = {}
            for decision in ControlDecision:
                count = sum(1 for result in recent_decisions if result.decision == decision)
                decision_distribution[decision.value] = count / len(recent_decisions)
            
            stats['recent_decision_distribution'] = decision_distribution
            
            # Calculate average confidence by decision type
            confidence_by_decision = {}
            for decision in ControlDecision:
                relevant_results = [r for r in recent_decisions if r.decision == decision]
                if relevant_results:
                    avg_confidence = sum(r.confidence for r in relevant_results) / len(relevant_results)
                    confidence_by_decision[decision.value] = avg_confidence
            
            stats['confidence_by_decision'] = confidence_by_decision
        
        return stats
    
    def update_config(self, new_config: Dict[str, Any]):
        """
        Update controller configuration.
        
        Args:
            new_config: Dictionary of configuration updates
        """
        # Update config attributes
        for key, value in new_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config {key} = {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")
    
    def reset_statistics(self):
        """Reset all statistics and history."""
        self.total_decisions = 0
        self.decisions_by_type = {decision: 0 for decision in ControlDecision}
        self.control_history.clear()
        self.decision_times.clear()
        self.average_decision_time = 0.0
        logger.info("Controller statistics reset")
