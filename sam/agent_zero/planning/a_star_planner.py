"""
A* Planner Module

Implements the main A* search planner that assembles all components
into a functional strategic planning system for SAM. This is the core
of the LLM-guided A* search implementation.
"""

import logging
from typing import List, Dict, Any, Optional, Callable, Set
from dataclasses import dataclass
from datetime import datetime
import time

from .state import PlanningState
from .search_node import SearchNode, SearchNodeFactory
from .frontier import Frontier
from .heuristic_estimator import HeuristicEstimator
from .action_expander import ActionExpander
from .episodic_memory_heuristic import EpisodicMemoryHeuristic
from .meta_reasoning_validator import MetaReasoningPlanValidator
from .sam_context_manager import SAMContextManager
from .sam_tool_registry import get_sam_tool_registry
from .tpv_planning_controller import TPVPlanningController

logger = logging.getLogger(__name__)


@dataclass
class PlanningResult:
    """Result of A* planning operation."""
    
    success: bool
    """Whether planning was successful"""
    
    plan: List[str]
    """Sequence of actions to execute (empty if failed)"""
    
    total_cost: int
    """Total estimated cost of the plan"""
    
    nodes_explored: int
    """Number of nodes explored during search"""
    
    planning_time: float
    """Time spent planning in seconds"""
    
    goal_reached: bool
    """Whether a goal state was found"""
    
    termination_reason: str
    """Reason why planning terminated"""
    
    search_statistics: Dict[str, Any]
    """Detailed search statistics"""


class AStarPlanner:
    """
    Main A* search planner for SAM.
    
    This class assembles all the A* search components into a functional
    strategic planning system. It provides the main interface for
    generating optimal action sequences for complex tasks.
    """
    
    def __init__(self,
                 llm_interface=None,
                 context_manager: Optional[SAMContextManager] = None,
                 max_nodes: int = 1000,
                 max_time_seconds: int = 300,
                 beam_width: Optional[int] = None,
                 enable_tpv_control: bool = True,
                 tpv_config: Optional[Dict[str, Any]] = None,
                 enable_episodic_learning: bool = True,
                 episodic_store=None,
                 enable_meta_reasoning_validation: bool = True,
                 validation_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the A* planner.

        Args:
            llm_interface: SAM's LLM interface for heuristics and actions
            context_manager: SAM context manager for enhanced planning
            max_nodes: Maximum number of nodes to explore
            max_time_seconds: Maximum planning time in seconds
            beam_width: Optional beam width for beam search variant
            enable_tpv_control: Whether to enable TPV planning control
            tpv_config: Configuration for TPV controller
            enable_episodic_learning: Whether to enable episodic memory learning
            episodic_store: Optional episodic memory store instance
            enable_meta_reasoning_validation: Whether to enable meta-reasoning plan validation
            validation_config: Configuration for meta-reasoning validator
        """
        self.llm_interface = llm_interface
        self.context_manager = context_manager or SAMContextManager()
        self.max_nodes = max_nodes
        self.max_time_seconds = max_time_seconds
        self.beam_width = beam_width
        self.enable_tpv_control = enable_tpv_control
        self.enable_episodic_learning = enable_episodic_learning
        self.enable_meta_reasoning_validation = enable_meta_reasoning_validation
        
        # Initialize components
        self.tool_registry = get_sam_tool_registry()

        # Use episodic memory enhanced heuristic if enabled
        if self.enable_episodic_learning:
            self.heuristic_estimator = EpisodicMemoryHeuristic(
                llm_interface=llm_interface,
                context_manager=context_manager,
                episodic_store=episodic_store,
                enable_experience_learning=True
            )
        else:
            self.heuristic_estimator = HeuristicEstimator(
                llm_interface=llm_interface,
                context_manager=context_manager
            )
        self.action_expander = ActionExpander(
            llm_interface=llm_interface,
            tool_registry=self.tool_registry,
            context_manager=context_manager
        )

        # Initialize TPV planning controller
        if self.enable_tpv_control:
            self.tpv_controller = TPVPlanningController(
                max_planning_time=max_time_seconds,
                enable_tpv_integration=True,
                tpv_config=tpv_config or {}
            )
        else:
            self.tpv_controller = None

        # Initialize meta-reasoning plan validator
        if self.enable_meta_reasoning_validation:
            self.plan_validator = MetaReasoningPlanValidator(
                enable_meta_reasoning=True,
                enable_safety_validation=True,
                **(validation_config or {})
            )
        else:
            self.plan_validator = None
        
        # Planning state
        self.frontier = Frontier(max_size=max_nodes)
        self.visited_states: Set[str] = set()
        self.goal_checker: Optional[Callable[[PlanningState], bool]] = None
        
        # Statistics
        self.nodes_explored = 0
        self.planning_start_time = 0.0
        
        logger.info(f"Initialized AStarPlanner (max_nodes={max_nodes}, max_time={max_time_seconds}s, "
                    f"tpv_control={enable_tpv_control}, episodic_learning={enable_episodic_learning}, "
                    f"meta_validation={enable_meta_reasoning_validation})")
    
    def find_optimal_plan(self, 
                         task_description: str,
                         initial_context: Optional[Dict[str, Any]] = None,
                         goal_checker: Optional[Callable[[PlanningState], bool]] = None) -> PlanningResult:
        """
        Find optimal plan for the given task using A* search.
        
        Args:
            task_description: Description of the task to accomplish
            initial_context: Initial context (documents, memory, etc.)
            goal_checker: Optional function to check if goal is reached
            
        Returns:
            PlanningResult with the optimal plan or failure information
        """
        self.planning_start_time = time.time()
        self.nodes_explored = 0
        self.goal_checker = goal_checker
        
        logger.info(f"Starting A* planning for task: {task_description[:100]}...")
        
        try:
            # Initialize search
            self._initialize_search(task_description, initial_context)

            # Start TPV planning session if enabled
            if self.tpv_controller:
                session_id = self.tpv_controller.start_planning_session(task_description)
                logger.info(f"Started TPV planning session: {session_id}")

            # Main A* search loop
            result = self._execute_search()
            
            # Record planning outcome for episodic learning
            if self.enable_episodic_learning and hasattr(self.heuristic_estimator, 'record_planning_outcome'):
                self._record_planning_outcome(task_description, initial_context, result)

            # Stop TPV session if enabled
            if self.tpv_controller:
                tpv_summary = self.tpv_controller.stop_planning_session()
                logger.info(f"TPV session summary: {tpv_summary}")

            # Log results
            planning_time = time.time() - self.planning_start_time
            logger.info(f"Planning completed in {planning_time:.2f}s: {result.termination_reason}")

            return result
            
        except Exception as e:
            logger.error(f"Planning failed with error: {e}")
            return PlanningResult(
                success=False,
                plan=[],
                total_cost=0,
                nodes_explored=self.nodes_explored,
                planning_time=time.time() - self.planning_start_time,
                goal_reached=False,
                termination_reason=f"Error: {str(e)}",
                search_statistics=self._get_search_statistics()
            )
    
    def _initialize_search(self, task_description: str, initial_context: Optional[Dict[str, Any]]):
        """Initialize the A* search."""
        
        # Clear previous search state
        self.frontier.clear()
        self.visited_states.clear()
        
        # Update context manager with initial context
        if initial_context:
            # This would integrate with SAM's session state
            self.context_manager.update_from_session_state(initial_context)
        
        # Create root node
        planning_context = self.context_manager.get_planning_context()
        root_node = SearchNodeFactory.create_root_node(task_description, planning_context)
        
        # Get initial heuristic estimate
        h_score = self.heuristic_estimator.estimate_cost_to_go(root_node.state)
        root_node.update_h_score(h_score)
        
        # Add to frontier
        self.frontier.add(root_node)
        
        logger.debug(f"Initialized search with root node: f={root_node.f_score}")
    
    def _execute_search(self) -> PlanningResult:
        """Execute the main A* search loop."""
        
        while not self.frontier.is_empty():
            # Check TPV-enhanced termination conditions
            if self._should_terminate_with_tpv():
                return self._create_termination_result()
            
            # Get best node from frontier
            current_node = self.frontier.pop()
            if current_node is None:
                break
            
            self.nodes_explored += 1
            
            # Check if we've reached the goal
            if self._is_goal_state(current_node.state):
                return self._create_success_result(current_node)
            
            # Mark as visited
            state_signature = self._get_state_signature(current_node.state)
            if state_signature in self.visited_states:
                continue  # Skip already visited states
            
            self.visited_states.add(state_signature)
            
            # Expand the node
            self._expand_node(current_node)
            
            # Log progress periodically
            if self.nodes_explored % 50 == 0:
                logger.debug(f"Explored {self.nodes_explored} nodes, frontier: {self.frontier.size()}")
        
        # Search completed without finding goal
        return self._create_failure_result("Search space exhausted")
    
    def _should_terminate_with_tpv(self) -> bool:
        """Check if search should terminate using TPV-enhanced control."""

        # Use TPV controller if available
        if self.tpv_controller:
            control_result = self.tpv_controller.should_continue_planning(
                frontier=self.frontier,
                nodes_explored=self.nodes_explored,
                current_best_node=self.frontier.peek()
            )

            if not control_result.should_continue:
                # Store TPV termination reason for result
                self._tpv_termination_reason = control_result.reason
                self._tpv_termination_confidence = control_result.confidence
                return True

        # Fallback to basic termination checks
        return self._should_terminate()

    def _should_terminate(self) -> bool:
        """Check if search should terminate (basic checks)."""

        # Time limit check
        elapsed_time = time.time() - self.planning_start_time
        if elapsed_time > self.max_time_seconds:
            return True

        # Node limit check
        if self.nodes_explored >= self.max_nodes:
            return True

        return False
    
    def _is_goal_state(self, state: PlanningState) -> bool:
        """Check if the state represents a goal state."""
        
        if self.goal_checker:
            return self.goal_checker(state)
        
        # Default goal checking
        return state.is_goal_state()
    
    def _expand_node(self, node: SearchNode):
        """Expand a node by generating child nodes."""
        
        # Get possible actions
        possible_actions = self.action_expander.get_next_possible_actions(node.state)
        
        # Create child nodes
        child_nodes = []
        for action in possible_actions:
            # Simulate action execution (in real implementation, this would execute the action)
            observation = f"Executed {action}"
            
            # Create child state
            child_state = node.state.add_action(action, observation)
            
            # Check if we've seen this state before
            child_signature = self._get_state_signature(child_state)
            if child_signature in self.visited_states:
                continue
            
            # Get heuristic estimate for child
            h_score = self.heuristic_estimator.estimate_cost_to_go(child_state)
            
            # Create child node
            child_node = SearchNodeFactory.create_child_node(node, action, observation, h_score)
            child_nodes.append(child_node)
        
        # Apply beam search if configured
        if self.beam_width and len(child_nodes) > self.beam_width:
            child_nodes.sort(key=lambda n: n.f_score)
            child_nodes = child_nodes[:self.beam_width]
        
        # Add children to frontier
        for child_node in child_nodes:
            self.frontier.add(child_node)
        
        # Mark parent as expanded
        node.mark_expanded()
        
        logger.debug(f"Expanded node with {len(child_nodes)} children")
    
    def _get_state_signature(self, state: PlanningState) -> str:
        """Get a signature for state deduplication."""
        # Simple signature based on task and action history
        return f"{state.task_description}|{tuple(state.action_history)}"
    
    def _create_success_result(self, goal_node: SearchNode) -> PlanningResult:
        """Create result for successful planning."""

        plan = goal_node.state.get_action_path()
        planning_time = time.time() - self.planning_start_time

        # Validate the plan if meta-reasoning validation is enabled
        validation_result = None
        if self.enable_meta_reasoning_validation and self.plan_validator:
            try:
                # Get initial state for validation
                initial_state = self._get_initial_state_for_validation(goal_node)
                validation_result = self.plan_validator.validate_plan(
                    plan=plan,
                    initial_state=initial_state,
                    context=self.context_manager.get_planning_context()
                )

                # Log validation results
                logger.info(f"Plan validation: valid={validation_result.is_valid}, "
                           f"risk={validation_result.overall_risk_score:.2f}, "
                           f"confidence={validation_result.confidence_score:.2f}")

            except Exception as e:
                logger.warning(f"Plan validation failed: {e}")

        # Create search statistics with validation info
        search_stats = self._get_search_statistics()
        if validation_result:
            search_stats['validation_result'] = {
                'is_valid': validation_result.is_valid,
                'risk_score': validation_result.overall_risk_score,
                'confidence_score': validation_result.confidence_score,
                'issues_count': len(validation_result.issues),
                'recommendations_count': len(validation_result.recommendations)
            }

        return PlanningResult(
            success=True,
            plan=plan,
            total_cost=goal_node.state.g_score,
            nodes_explored=self.nodes_explored,
            planning_time=planning_time,
            goal_reached=True,
            termination_reason="Goal state reached",
            search_statistics=search_stats
        )
    
    def _create_failure_result(self, reason: str) -> PlanningResult:
        """Create result for failed planning."""
        
        planning_time = time.time() - self.planning_start_time
        
        # Try to return best partial plan
        best_node = self.frontier.peek()
        partial_plan = best_node.state.get_action_path() if best_node else []
        
        return PlanningResult(
            success=False,
            plan=partial_plan,
            total_cost=best_node.state.g_score if best_node else 0,
            nodes_explored=self.nodes_explored,
            planning_time=planning_time,
            goal_reached=False,
            termination_reason=reason,
            search_statistics=self._get_search_statistics()
        )
    
    def _create_termination_result(self) -> PlanningResult:
        """Create result for early termination."""

        elapsed_time = time.time() - self.planning_start_time

        # Check if TPV controller caused termination
        if hasattr(self, '_tpv_termination_reason'):
            reason = f"TPV Control: {self._tpv_termination_reason}"
        elif elapsed_time > self.max_time_seconds:
            reason = f"Time limit exceeded ({self.max_time_seconds}s)"
        elif self.nodes_explored >= self.max_nodes:
            reason = f"Node limit exceeded ({self.max_nodes})"
        else:
            reason = "Unknown termination condition"

        return self._create_failure_result(reason)
    
    def _get_search_statistics(self) -> Dict[str, Any]:
        """Get detailed search statistics."""
        
        frontier_stats = self.frontier.get_statistics()
        estimator_stats = self.heuristic_estimator.get_estimation_stats()
        expander_stats = self.action_expander.get_expansion_stats()
        
        stats = {
            'nodes_explored': self.nodes_explored,
            'visited_states': len(self.visited_states),
            'frontier_stats': frontier_stats,
            'estimator_stats': estimator_stats,
            'expander_stats': expander_stats,
            'planning_time': time.time() - self.planning_start_time
        }

        # Add TPV statistics if available
        if self.tpv_controller:
            stats['tpv_stats'] = self.tpv_controller.get_planning_statistics()

        # Add validation statistics if available
        if self.plan_validator:
            stats['validation_stats'] = self.plan_validator.get_validation_statistics()

        return stats

    def _record_planning_outcome(self,
                                task_description: str,
                                initial_context: Optional[Dict[str, Any]],
                                result: PlanningResult):
        """Record planning outcome for episodic learning."""

        try:
            # Create a representative state for the task
            planning_context = self.context_manager.get_planning_context()
            final_state = PlanningState(
                task_description=task_description,
                action_history=result.plan,
                current_observation="Planning completed",
                document_context=planning_context.get('documents'),
                memory_context=planning_context.get('memory'),
                conversation_context=planning_context.get('conversation')
            )

            # Calculate outcome quality based on result
            outcome_quality = self._calculate_outcome_quality(result)

            # Record the outcome
            self.heuristic_estimator.record_planning_outcome(
                state=final_state,
                estimated_cost=result.total_cost,  # This would be the initial estimate
                actual_cost=result.total_cost,
                success=result.success and result.goal_reached,
                outcome_quality=outcome_quality
            )

            logger.debug(f"Recorded planning outcome: success={result.success}, quality={outcome_quality:.2f}")

        except Exception as e:
            logger.warning(f"Error recording planning outcome: {e}")

    def _calculate_outcome_quality(self, result: PlanningResult) -> float:
        """Calculate quality score for planning outcome."""

        quality = 0.5  # Base quality

        # Success bonus
        if result.success and result.goal_reached:
            quality += 0.3

        # Efficiency bonus (fewer nodes explored is better)
        if result.nodes_explored < self.max_nodes * 0.5:
            quality += 0.1

        # Time efficiency bonus
        if result.planning_time < self.max_time_seconds * 0.5:
            quality += 0.1

        # Plan length consideration (shorter plans are often better)
        if len(result.plan) <= 5:
            quality += 0.1
        elif len(result.plan) > 10:
            quality -= 0.1

        return max(0.0, min(1.0, quality))

    def _get_initial_state_for_validation(self, goal_node: SearchNode) -> PlanningState:
        """Get initial state for plan validation."""

        # Reconstruct initial state from goal node
        current_node = goal_node
        while current_node.state.parent is not None:
            current_node = SearchNode(state=current_node.state.parent, h_score=0)

        return current_node.state

    def set_goal_checker(self, goal_checker: Callable[[PlanningState], bool]):
        """Set custom goal checking function."""
        self.goal_checker = goal_checker
    
    def get_planning_statistics(self) -> Dict[str, Any]:
        """Get current planning statistics."""
        return self._get_search_statistics()
    
    def clear_caches(self):
        """Clear all component caches."""
        self.heuristic_estimator.clear_cache()
        self.action_expander.clear_cache()
        logger.info("Cleared all planner caches")


class SAMPlannerIntegration:
    """
    Integration layer between A* planner and SAM's existing systems.

    This class provides the interface for SAM to use the A* planner
    while maintaining compatibility with existing SAM workflows.
    """

    def __init__(self, sam_session_state: Optional[Dict[str, Any]] = None):
        """
        Initialize SAM planner integration.

        Args:
            sam_session_state: SAM's current session state
        """
        self.session_state = sam_session_state or {}
        self.context_manager = SAMContextManager()
        self.planner = None

        # Update context from session state
        if self.session_state:
            self.context_manager.update_from_session_state(self.session_state)

    def create_planner(self, llm_interface=None, **kwargs) -> AStarPlanner:
        """
        Create A* planner configured for SAM.

        Args:
            llm_interface: SAM's LLM interface
            **kwargs: Additional planner configuration

        Returns:
            Configured AStarPlanner instance
        """
        self.planner = AStarPlanner(
            llm_interface=llm_interface,
            context_manager=self.context_manager,
            **kwargs
        )
        return self.planner

    def plan_task(self,
                  task_description: str,
                  goal_checker: Optional[Callable[[PlanningState], bool]] = None) -> PlanningResult:
        """
        Plan a task using A* search with SAM context.

        Args:
            task_description: Task to plan for
            goal_checker: Optional custom goal checker

        Returns:
            Planning result with action sequence
        """
        if not self.planner:
            raise ValueError("Planner not initialized. Call create_planner() first.")

        # Get current SAM context
        initial_context = self.context_manager.get_planning_context()

        # Execute planning
        result = self.planner.find_optimal_plan(
            task_description=task_description,
            initial_context=initial_context,
            goal_checker=goal_checker
        )

        return result

    def execute_plan(self, plan: List[str], execution_engine=None) -> Dict[str, Any]:
        """
        Execute a plan using SAM's execution engine.

        Args:
            plan: List of actions to execute
            execution_engine: SAM's execution engine

        Returns:
            Execution results
        """
        if not execution_engine:
            # Simulate execution for testing
            return {
                'success': True,
                'executed_actions': plan,
                'results': [f"Simulated execution of {action}" for action in plan]
            }

        # In real implementation, this would use SAM's actual execution engine
        results = []
        for action in plan:
            try:
                result = execution_engine.execute_action(action)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to execute action {action}: {e}")
                return {
                    'success': False,
                    'executed_actions': plan[:len(results)],
                    'results': results,
                    'error': str(e)
                }

        return {
            'success': True,
            'executed_actions': plan,
            'results': results
        }

    def plan_and_execute(self,
                        task_description: str,
                        execution_engine=None,
                        goal_checker: Optional[Callable[[PlanningState], bool]] = None) -> Dict[str, Any]:
        """
        Plan and execute a task in one operation.

        Args:
            task_description: Task to plan and execute
            execution_engine: SAM's execution engine
            goal_checker: Optional custom goal checker

        Returns:
            Combined planning and execution results
        """
        # Plan the task
        planning_result = self.plan_task(task_description, goal_checker)

        if not planning_result.success:
            return {
                'success': False,
                'planning_result': planning_result,
                'execution_result': None,
                'error': f"Planning failed: {planning_result.termination_reason}"
            }

        # Execute the plan
        execution_result = self.execute_plan(planning_result.plan, execution_engine)

        return {
            'success': execution_result['success'],
            'planning_result': planning_result,
            'execution_result': execution_result
        }

    def update_session_state(self, new_session_state: Dict[str, Any]):
        """Update the session state and context."""
        self.session_state.update(new_session_state)
        self.context_manager.update_from_session_state(self.session_state)

    def get_planning_suggestions(self, task_description: str) -> List[str]:
        """
        Get planning suggestions without full A* search.

        Args:
            task_description: Task to get suggestions for

        Returns:
            List of suggested actions
        """
        if not self.planner:
            self.create_planner()

        # Create a simple state for the task
        initial_context = self.context_manager.get_planning_context()
        root_node = SearchNodeFactory.create_root_node(task_description, initial_context)

        # Get action suggestions
        suggestions = self.planner.action_expander.get_next_possible_actions(root_node.state)

        return suggestions[:5]  # Return top 5 suggestions
