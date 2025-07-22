"""
TPV Planning Controller Module

Integrates SAM's TPV (Thinking Progress Verification) system with the A* planner
to provide intelligent planning time control. Acts as the "Governor" to prevent
runaway planning and detect stagnation in the search process.
"""

import logging
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .state import PlanningState
from .search_node import SearchNode
from .frontier import Frontier

# Import SAM's TPV components
try:
    from sam.cognition.tpv.tpv_controller import ReasoningController, ControlMode, ControlConfig
    from sam.cognition.tpv.tpv_monitor import TPVMonitor, ReasoningTrace, ReasoningStep
    TPV_AVAILABLE = True
except ImportError:
    TPV_AVAILABLE = False
    logging.warning("SAM TPV components not available - using fallback implementation")

logger = logging.getLogger(__name__)


class PlanningProgressType(Enum):
    """Types of planning progress indicators."""
    F_SCORE_IMPROVEMENT = "f_score_improvement"
    FRONTIER_DIVERSITY = "frontier_diversity"
    EXPLORATION_EFFICIENCY = "exploration_efficiency"
    SOLUTION_CONVERGENCE = "solution_convergence"


@dataclass
class PlanningProgressMetrics:
    """Metrics for tracking planning progress."""
    
    best_f_score: float
    """Current best f-score in frontier"""
    
    f_score_history: List[float]
    """History of best f-scores"""
    
    frontier_size: int
    """Current frontier size"""
    
    nodes_explored: int
    """Total nodes explored"""
    
    exploration_rate: float
    """Rate of node exploration (nodes/second)"""
    
    diversity_score: float
    """Diversity of solutions in frontier"""
    
    stagnation_count: int
    """Number of iterations without improvement"""
    
    planning_time: float
    """Total planning time elapsed"""


@dataclass
class PlanningControlResult:
    """Result of planning control decision."""
    
    should_continue: bool
    """Whether planning should continue"""
    
    reason: str
    """Reason for the decision"""
    
    confidence: float
    """Confidence in the decision (0.0 to 1.0)"""
    
    progress_metrics: PlanningProgressMetrics
    """Current progress metrics"""
    
    tpv_score: Optional[float] = None
    """TPV score if available"""
    
    metadata: Dict[str, Any] = None
    """Additional metadata"""


class TPVPlanningController:
    """
    TPV-enhanced planning controller for A* search.
    
    This controller integrates SAM's TPV system with A* planning to provide
    intelligent control over planning time and detect when planning should
    stop due to stagnation or completion.
    """
    
    def __init__(self,
                 max_stagnation_iterations: int = 10,
                 min_improvement_threshold: float = 0.01,
                 max_planning_time: float = 300.0,
                 enable_tpv_integration: bool = True,
                 tpv_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the TPV planning controller.
        
        Args:
            max_stagnation_iterations: Max iterations without improvement
            min_improvement_threshold: Minimum f-score improvement to continue
            max_planning_time: Maximum planning time in seconds
            enable_tpv_integration: Whether to use SAM's TPV system
            tpv_config: Configuration for TPV integration
        """
        self.max_stagnation_iterations = max_stagnation_iterations
        self.min_improvement_threshold = min_improvement_threshold
        self.max_planning_time = max_planning_time
        self.enable_tpv_integration = enable_tpv_integration and TPV_AVAILABLE
        
        # Initialize TPV components if available
        self.tpv_monitor = None
        self.reasoning_controller = None
        self.current_query_id = None
        
        if self.enable_tpv_integration:
            self._initialize_tpv_components(tpv_config or {})
        
        # Planning state tracking
        self.planning_start_time = 0.0
        self.progress_history: List[PlanningProgressMetrics] = []
        self.control_decisions: List[PlanningControlResult] = []
        
        logger.info(f"TPVPlanningController initialized (TPV: {self.enable_tpv_integration})")
    
    def _initialize_tpv_components(self, tpv_config: Dict[str, Any]):
        """Initialize SAM's TPV components for planning control."""
        try:
            # Create TPV monitor for planning progress
            self.tpv_monitor = TPVMonitor(
                enable_dissonance_monitoring=tpv_config.get('enable_dissonance', False)
            )
            
            # Create reasoning controller with planning-specific config
            control_config = ControlConfig(
                max_steps=tpv_config.get('max_steps', 100),
                completion_threshold=tpv_config.get('completion_threshold', 0.9),
                plateau_threshold=tpv_config.get('plateau_threshold', 0.01),
                plateau_patience=tpv_config.get('plateau_patience', 5),
                min_steps=tpv_config.get('min_steps', 3)
            )
            
            self.reasoning_controller = ReasoningController(
                mode=ControlMode.ACTIVE,
                config=control_config
            )
            
            logger.info("TPV components initialized for planning control")
            
        except Exception as e:
            logger.warning(f"Failed to initialize TPV components: {e}")
            self.enable_tpv_integration = False
    
    def start_planning_session(self, task_description: str) -> str:
        """
        Start a new planning session with TPV monitoring.
        
        Args:
            task_description: Description of the planning task
            
        Returns:
            Session ID for tracking
        """
        self.planning_start_time = time.time()
        self.progress_history.clear()
        self.control_decisions.clear()
        
        # Start TPV monitoring if available
        if self.enable_tpv_integration and self.tpv_monitor:
            self.current_query_id = self.tpv_monitor.start_monitoring(
                query=f"Planning: {task_description}",
                metadata={'type': 'planning', 'task': task_description}
            )
            logger.info(f"Started TPV planning session: {self.current_query_id}")
        
        return self.current_query_id or "fallback_session"
    
    def should_continue_planning(self,
                               frontier: Frontier,
                               nodes_explored: int,
                               current_best_node: Optional[SearchNode] = None) -> PlanningControlResult:
        """
        Determine whether planning should continue based on progress analysis.
        
        Args:
            frontier: Current planning frontier
            nodes_explored: Number of nodes explored so far
            current_best_node: Current best node (optional)
            
        Returns:
            PlanningControlResult with decision and reasoning
        """
        # Calculate current progress metrics
        progress_metrics = self._calculate_progress_metrics(frontier, nodes_explored, current_best_node)
        
        # Update TPV monitoring if available
        tpv_score = None
        if self.enable_tpv_integration and self.tpv_monitor and self.current_query_id:
            tpv_score = self._update_tpv_monitoring(progress_metrics)
        
        # Make control decision
        control_result = self._make_planning_control_decision(progress_metrics, tpv_score)
        
        # Record decision
        self.progress_history.append(progress_metrics)
        self.control_decisions.append(control_result)
        
        # Log decision
        logger.debug(f"Planning control decision: {control_result.should_continue} "
                    f"({control_result.reason}) - f_score: {progress_metrics.best_f_score:.3f}")
        
        return control_result
    
    def _calculate_progress_metrics(self,
                                  frontier: Frontier,
                                  nodes_explored: int,
                                  current_best_node: Optional[SearchNode]) -> PlanningProgressMetrics:
        """Calculate current planning progress metrics."""
        
        # Get current best f-score
        best_node = frontier.peek()
        best_f_score = best_node.f_score if best_node else float('inf')
        
        # Calculate f-score history
        f_score_history = [metrics.best_f_score for metrics in self.progress_history]
        f_score_history.append(best_f_score)
        
        # Calculate exploration rate
        planning_time = time.time() - self.planning_start_time
        exploration_rate = nodes_explored / max(planning_time, 0.001)
        
        # Calculate diversity score (simplified)
        diversity_score = self._calculate_frontier_diversity(frontier)
        
        # Calculate stagnation count
        stagnation_count = self._calculate_stagnation_count(f_score_history)
        
        return PlanningProgressMetrics(
            best_f_score=best_f_score,
            f_score_history=f_score_history,
            frontier_size=frontier.size(),
            nodes_explored=nodes_explored,
            exploration_rate=exploration_rate,
            diversity_score=diversity_score,
            stagnation_count=stagnation_count,
            planning_time=planning_time
        )
    
    def _calculate_frontier_diversity(self, frontier: Frontier) -> float:
        """Calculate diversity score for frontier nodes."""
        if frontier.size() <= 1:
            return 0.0
        
        # Get sample of nodes for diversity calculation
        sample_nodes = frontier.get_nodes_by_f_score(max_nodes=10)
        
        if len(sample_nodes) <= 1:
            return 0.0
        
        # Calculate diversity based on f-score variance
        f_scores = [node.f_score for node in sample_nodes]
        mean_f_score = sum(f_scores) / len(f_scores)
        variance = sum((score - mean_f_score) ** 2 for score in f_scores) / len(f_scores)
        
        # Normalize diversity score
        diversity_score = min(1.0, variance / 10.0)  # Normalize to reasonable range
        
        return diversity_score
    
    def _calculate_stagnation_count(self, f_score_history: List[float]) -> int:
        """Calculate number of iterations without significant improvement."""
        if len(f_score_history) < 2:
            return 0
        
        stagnation_count = 0
        current_best = f_score_history[-1]
        
        # Count iterations without improvement
        for i in range(len(f_score_history) - 2, -1, -1):
            improvement = f_score_history[i] - current_best
            if improvement < self.min_improvement_threshold:
                stagnation_count += 1
            else:
                break
        
        return stagnation_count
    
    def _update_tpv_monitoring(self, progress_metrics: PlanningProgressMetrics) -> Optional[float]:
        """Update TPV monitoring with current planning progress."""
        if not (self.enable_tpv_integration and self.tpv_monitor and self.current_query_id):
            return None
        
        try:
            # Create progress text for TPV monitoring
            progress_text = self._create_progress_text(progress_metrics)
            
            # Update TPV monitoring
            tpv_score = self.tpv_monitor.predict_progress(
                current_text=progress_text,
                query_id=self.current_query_id,
                token_count=len(progress_text.split())
            )
            
            return tpv_score
            
        except Exception as e:
            logger.warning(f"TPV monitoring update failed: {e}")
            return None
    
    def _create_progress_text(self, progress_metrics: PlanningProgressMetrics) -> str:
        """Create text representation of planning progress for TPV."""
        return (
            f"Planning progress: explored {progress_metrics.nodes_explored} nodes, "
            f"best f-score {progress_metrics.best_f_score:.3f}, "
            f"frontier size {progress_metrics.frontier_size}, "
            f"stagnation {progress_metrics.stagnation_count} iterations, "
            f"time {progress_metrics.planning_time:.2f}s"
        )
    
    def _make_planning_control_decision(self,
                                      progress_metrics: PlanningProgressMetrics,
                                      tpv_score: Optional[float]) -> PlanningControlResult:
        """Make control decision based on progress metrics and TPV score."""
        
        # Check time limit
        if progress_metrics.planning_time > self.max_planning_time:
            return PlanningControlResult(
                should_continue=False,
                reason=f"Time limit exceeded: {progress_metrics.planning_time:.2f}s > {self.max_planning_time}s",
                confidence=0.9,
                progress_metrics=progress_metrics,
                tpv_score=tpv_score,
                metadata={'termination_type': 'time_limit'}
            )
        
        # Check stagnation
        if progress_metrics.stagnation_count >= self.max_stagnation_iterations:
            return PlanningControlResult(
                should_continue=False,
                reason=f"Planning stagnated: {progress_metrics.stagnation_count} iterations without improvement",
                confidence=0.8,
                progress_metrics=progress_metrics,
                tpv_score=tpv_score,
                metadata={'termination_type': 'stagnation'}
            )
        
        # Check TPV controller decision if available
        if self.enable_tpv_integration and self.reasoning_controller and self.current_query_id:
            tpv_decision = self._check_tpv_controller_decision()
            if tpv_decision and not tpv_decision.should_continue:
                return PlanningControlResult(
                    should_continue=False,
                    reason=f"TPV controller decision: {tpv_decision.reason}",
                    confidence=tpv_decision.confidence,
                    progress_metrics=progress_metrics,
                    tpv_score=tpv_score,
                    metadata={'termination_type': 'tpv_controller', 'tpv_decision': tpv_decision.reason}
                )
        
        # Check for very good solution (low f-score)
        if progress_metrics.best_f_score <= 1.0:  # Very good solution found
            return PlanningControlResult(
                should_continue=False,
                reason=f"Excellent solution found: f-score {progress_metrics.best_f_score:.3f}",
                confidence=0.9,
                progress_metrics=progress_metrics,
                tpv_score=tpv_score,
                metadata={'termination_type': 'optimal_solution'}
            )
        
        # Continue planning
        return PlanningControlResult(
            should_continue=True,
            reason="Planning progressing normally",
            confidence=0.7,
            progress_metrics=progress_metrics,
            tpv_score=tpv_score,
            metadata={'status': 'continuing'}
        )
    
    def _check_tpv_controller_decision(self) -> Optional[PlanningControlResult]:
        """Check TPV controller decision for current planning session."""
        if not (self.enable_tpv_integration and self.reasoning_controller and self.current_query_id):
            return None
        
        try:
            # Get current trace
            trace = self.tpv_monitor.get_trace(self.current_query_id)
            if not trace:
                return None
            
            # Check controller decision
            should_continue = self.reasoning_controller.should_continue(trace)
            
            if not should_continue:
                # Get the latest control result for reasoning
                control_history = self.reasoning_controller.control_history
                latest_control = control_history[-1] if control_history else None
                
                reason = latest_control.reason if latest_control else "TPV controller indicated stop"
                confidence = latest_control.confidence if latest_control else 0.7
                
                return PlanningControlResult(
                    should_continue=False,
                    reason=reason,
                    confidence=confidence,
                    progress_metrics=None,  # Will be filled by caller
                    tpv_score=trace.current_score
                )
            
            return None
            
        except Exception as e:
            logger.warning(f"TPV controller check failed: {e}")
            return None
    
    def stop_planning_session(self) -> Optional[Dict[str, Any]]:
        """
        Stop the current planning session and return summary.
        
        Returns:
            Planning session summary
        """
        if self.enable_tpv_integration and self.tpv_monitor and self.current_query_id:
            try:
                trace = self.tpv_monitor.stop_monitoring(self.current_query_id)
                logger.info(f"Stopped TPV planning session: {self.current_query_id}")
                
                return {
                    'session_id': self.current_query_id,
                    'total_time': time.time() - self.planning_start_time,
                    'progress_steps': len(self.progress_history),
                    'control_decisions': len(self.control_decisions),
                    'final_tpv_score': trace.current_score if trace else None,
                    'tpv_steps': len(trace.steps) if trace else 0
                }
            except Exception as e:
                logger.warning(f"Error stopping TPV session: {e}")
        
        return {
            'session_id': 'fallback_session',
            'total_time': time.time() - self.planning_start_time,
            'progress_steps': len(self.progress_history),
            'control_decisions': len(self.control_decisions)
        }
    
    def get_planning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive planning statistics."""
        return {
            'tpv_integration_enabled': self.enable_tpv_integration,
            'total_progress_steps': len(self.progress_history),
            'total_control_decisions': len(self.control_decisions),
            'planning_time': time.time() - self.planning_start_time if self.planning_start_time > 0 else 0,
            'stagnation_threshold': self.max_stagnation_iterations,
            'improvement_threshold': self.min_improvement_threshold,
            'time_limit': self.max_planning_time,
            'current_session': self.current_query_id
        }
