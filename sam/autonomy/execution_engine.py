"""
Autonomous Goal Execution Engine for SAM
========================================

This module implements the core engine that automatically executes top-priority
goals during idle periods with comprehensive safety oversight and monitoring.

Phase C: Full Autonomy with Monitoring

Author: SAM Development Team
Version: 2.0.0
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from enum import Enum

from .goals import Goal
from .goal_stack import GoalStack
from .motivation_engine import MotivationEngine
from .safety.goal_validator import GoalSafetyValidator
from .idle_processor import IdleTimeProcessor, IdleProcessingConfig
from .system_monitor import SystemLoadMonitor, SystemThresholds, SystemState

# Import SAM orchestration components
try:
    from sam.orchestration.uif import SAM_UIF, UIFStatus
    from sam.orchestration.planner import DynamicPlanner
    from sam.orchestration.coordinator import CoordinatorEngine, ExecutionResult
    ORCHESTRATION_AVAILABLE = True
except ImportError:
    ORCHESTRATION_AVAILABLE = False

logger = logging.getLogger(__name__)

class ExecutionState(Enum):
    """Autonomous execution states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class ExecutionConfig:
    """Configuration for autonomous execution engine."""
    # Idle processing settings
    idle_threshold_seconds: float = 30.0
    max_processing_time_seconds: float = 300.0
    check_interval_seconds: float = 5.0
    max_concurrent_goals: int = 1
    
    # System monitoring settings
    enable_system_monitoring: bool = True
    cpu_threshold_percent: float = 70.0
    memory_threshold_percent: float = 80.0
    monitoring_interval_seconds: float = 5.0
    
    # Safety settings
    enable_safety_validation: bool = True
    max_execution_attempts: int = 3
    execution_timeout_seconds: float = 600.0
    
    # Goal selection settings
    min_goal_priority: float = 0.3
    priority_boost_recent: float = 0.1
    enable_goal_scheduling: bool = True

class AutonomousExecutionEngine:
    """
    Core engine for autonomous goal execution with comprehensive safety oversight.
    
    This engine coordinates all aspects of autonomous operation:
    - Idle time detection and processing
    - System resource monitoring
    - Goal selection and prioritization
    - Safe goal execution with timeout handling
    - Comprehensive monitoring and logging
    - Emergency controls and safety circuits
    
    Features:
    - Integrated idle processing and system monitoring
    - Real goal execution using SAM's orchestration system
    - Comprehensive safety validation and oversight
    - Emergency stop and pause mechanisms
    - Detailed execution logging and statistics
    - Configurable thresholds and behaviors
    """
    
    def __init__(self,
                 goal_stack: GoalStack,
                 motivation_engine: MotivationEngine,
                 safety_validator: GoalSafetyValidator,
                 config: Optional[ExecutionConfig] = None,
                 coordinator: Optional[CoordinatorEngine] = None,
                 planner: Optional[DynamicPlanner] = None):
        """
        Initialize the autonomous execution engine.
        
        Args:
            goal_stack: GoalStack for goal management
            motivation_engine: MotivationEngine for goal generation
            safety_validator: Safety validator for validation
            config: Optional execution configuration
            coordinator: Optional CoordinatorEngine for plan execution
            planner: Optional DynamicPlanner for plan generation
        """
        self.logger = logging.getLogger(f"{__name__}.AutonomousExecutionEngine")
        
        # Core components
        self.goal_stack = goal_stack
        self.motivation_engine = motivation_engine
        self.safety_validator = safety_validator
        self.config = config or ExecutionConfig()
        
        # Orchestration components
        self.coordinator = coordinator
        self.planner = planner
        
        # State management
        self.state = ExecutionState.STOPPED
        self.execution_start_time: Optional[datetime] = None
        
        # Sub-components
        self.system_monitor: Optional[SystemLoadMonitor] = None
        self.idle_processor: Optional[IdleTimeProcessor] = None
        
        # Statistics
        self.stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'timeout_executions': 0,
            'safety_rejections': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0,
            'last_execution_time': None,
            'uptime_seconds': 0.0,
            'goals_executed_by_priority': {},
            'execution_errors': []
        }
        
        # Initialize sub-components
        self._initialize_components()
        
        self.logger.info("AutonomousExecutionEngine initialized")
    
    def _initialize_components(self) -> None:
        """Initialize sub-components."""
        try:
            # Initialize system monitor
            if self.config.enable_system_monitoring:
                thresholds = SystemThresholds(
                    cpu_critical=self.config.cpu_threshold_percent,
                    memory_critical=self.config.memory_threshold_percent
                )
                
                self.system_monitor = SystemLoadMonitor(
                    thresholds=thresholds,
                    monitoring_interval=self.config.monitoring_interval_seconds
                )
                
                # Register alert callbacks
                self.system_monitor.register_alert_callback(
                    SystemState.CRITICAL,
                    self._handle_critical_system_state
                )
            
            # Initialize idle processor
            idle_config = IdleProcessingConfig(
                idle_threshold_seconds=self.config.idle_threshold_seconds,
                max_processing_time_seconds=self.config.max_processing_time_seconds,
                check_interval_seconds=self.config.check_interval_seconds,
                max_concurrent_goals=self.config.max_concurrent_goals,
                enable_system_load_monitoring=self.config.enable_system_monitoring,
                cpu_threshold_percent=self.config.cpu_threshold_percent,
                memory_threshold_percent=self.config.memory_threshold_percent
            )
            
            self.idle_processor = IdleTimeProcessor(
                goal_stack=self.goal_stack,
                motivation_engine=self.motivation_engine,
                safety_validator=self.safety_validator,
                config=idle_config,
                goal_executor=self._execute_goal_with_orchestration
            )
            
            self.logger.info("Sub-components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            self.state = ExecutionState.ERROR
    
    def start(self) -> bool:
        """
        Start autonomous execution.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.state == ExecutionState.RUNNING:
            self.logger.warning("Autonomous execution already running")
            return False
        
        try:
            self.logger.info("Starting autonomous execution engine...")
            self.state = ExecutionState.STARTING
            
            # Start system monitor
            if self.system_monitor:
                if not self.system_monitor.start_monitoring():
                    self.logger.error("Failed to start system monitoring")
                    self.state = ExecutionState.ERROR
                    return False
            
            # Start idle processor
            if self.idle_processor:
                if not self.idle_processor.start():
                    self.logger.error("Failed to start idle processor")
                    self.state = ExecutionState.ERROR
                    return False
            
            # Update state
            self.state = ExecutionState.RUNNING
            self.execution_start_time = datetime.now()
            
            self.logger.info("Autonomous execution engine started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start autonomous execution: {e}")
            self.state = ExecutionState.ERROR
            return False
    
    def stop(self) -> bool:
        """
        Stop autonomous execution.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        if self.state == ExecutionState.STOPPED:
            self.logger.warning("Autonomous execution not running")
            return False
        
        try:
            self.logger.info("Stopping autonomous execution engine...")
            
            # Stop idle processor
            if self.idle_processor:
                self.idle_processor.stop()
            
            # Stop system monitor
            if self.system_monitor:
                self.system_monitor.stop_monitoring()
            
            # Update statistics
            if self.execution_start_time:
                uptime = (datetime.now() - self.execution_start_time).total_seconds()
                self.stats['uptime_seconds'] += uptime
            
            # Update state
            self.state = ExecutionState.STOPPED
            self.execution_start_time = None
            
            self.logger.info("Autonomous execution engine stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping autonomous execution: {e}")
            self.state = ExecutionState.ERROR
            return False
    
    def pause(self) -> None:
        """Pause autonomous execution."""
        if self.idle_processor:
            self.idle_processor.pause()
        
        self.state = ExecutionState.PAUSED
        self.logger.info("Autonomous execution paused")
    
    def resume(self) -> None:
        """Resume autonomous execution."""
        if self.idle_processor:
            self.idle_processor.resume()
        
        self.state = ExecutionState.RUNNING
        self.logger.info("Autonomous execution resumed")
    
    def emergency_stop(self) -> None:
        """Emergency stop all autonomous execution."""
        if self.idle_processor:
            self.idle_processor.emergency_stop()
        
        self.state = ExecutionState.EMERGENCY_STOP
        self.logger.critical("EMERGENCY STOP activated for autonomous execution")
    
    def reset_emergency_stop(self) -> None:
        """Reset emergency stop state."""
        if self.idle_processor:
            self.idle_processor.reset_emergency_stop()
        
        self.state = ExecutionState.RUNNING
        self.logger.info("Emergency stop reset, autonomous execution resumed")
    
    def _execute_goal_with_orchestration(self, goal: Goal) -> bool:
        """
        Execute a goal using SAM's orchestration system.
        
        Args:
            goal: Goal to execute
            
        Returns:
            True if execution successful, False otherwise
        """
        execution_start = datetime.now()
        
        try:
            self.logger.info(f"Executing autonomous goal: {goal.description}")
            
            # Check if orchestration is available
            if not ORCHESTRATION_AVAILABLE:
                self.logger.warning("Orchestration not available, using simulation")
                return self._simulate_goal_execution(goal)
            
            # Create UIF for the goal
            uif = SAM_UIF(input_query=goal.description)
            uif.add_log_entry(f"Autonomous execution of goal: {goal.goal_id}")
            uif.add_log_entry(f"Goal source: {goal.source_skill}")
            uif.add_log_entry(f"Goal priority: {goal.priority}")
            
            # Generate plan
            if self.planner:
                plan_result = self.planner.create_plan(uif, mode="goal_focused")
                
                if not plan_result.plan:
                    self.logger.warning(f"No plan generated for goal: {goal.goal_id}")
                    return False
                
                uif.add_log_entry(f"Generated plan with {len(plan_result.plan)} steps")
            else:
                self.logger.warning("No planner available, cannot generate plan")
                return False
            
            # Execute plan
            if self.coordinator:
                execution_report = self.coordinator.execute_plan(plan_result.plan, uif)
                
                # Check execution result
                success = execution_report.result in [
                    ExecutionResult.SUCCESS, 
                    ExecutionResult.PARTIAL_SUCCESS
                ]
                
                if success:
                    self.logger.info(f"Goal executed successfully: {goal.goal_id}")
                else:
                    self.logger.warning(f"Goal execution failed: {goal.goal_id}")
                
                return success
            else:
                self.logger.warning("No coordinator available, cannot execute plan")
                return False
            
        except Exception as e:
            self.logger.error(f"Error executing goal {goal.goal_id}: {e}")
            
            # Record error
            self.stats['execution_errors'].append({
                'goal_id': goal.goal_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            
            # Keep only last 10 errors
            if len(self.stats['execution_errors']) > 10:
                self.stats['execution_errors'] = self.stats['execution_errors'][-10:]
            
            return False
        
        finally:
            # Update execution statistics
            execution_time = (datetime.now() - execution_start).total_seconds()
            self.stats['total_execution_time'] += execution_time
            self.stats['total_executions'] += 1
            self.stats['last_execution_time'] = datetime.now()
            
            # Update average execution time
            if self.stats['total_executions'] > 0:
                self.stats['average_execution_time'] = (
                    self.stats['total_execution_time'] / self.stats['total_executions']
                )
            
            # Track goals by priority
            priority_bucket = f"{goal.priority:.1f}"
            if priority_bucket not in self.stats['goals_executed_by_priority']:
                self.stats['goals_executed_by_priority'][priority_bucket] = 0
            self.stats['goals_executed_by_priority'][priority_bucket] += 1
    
    def _simulate_goal_execution(self, goal: Goal) -> bool:
        """
        Simulate goal execution when orchestration is not available.
        
        Args:
            goal: Goal to simulate
            
        Returns:
            True if simulation successful, False otherwise
        """
        try:
            self.logger.info(f"Simulating execution of goal: {goal.description}")
            
            # Simulate processing time based on goal complexity
            import time
            processing_time = min(5.0, max(1.0, goal.priority * 3))
            time.sleep(processing_time)
            
            # Simulate success/failure based on priority
            import random
            success_probability = min(0.95, goal.priority + 0.2)
            success = random.random() < success_probability
            
            if success:
                self.logger.info(f"Goal simulation completed successfully: {goal.goal_id}")
            else:
                self.logger.warning(f"Goal simulation failed: {goal.goal_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error in goal simulation: {e}")
            return False
    
    def _handle_critical_system_state(self, state, metrics) -> None:
        """Handle critical system state by pausing execution."""
        self.logger.warning(f"Critical system state detected: {state.value}")
        self.logger.warning(f"System metrics: CPU={metrics.cpu_percent:.1f}%, Memory={metrics.memory_percent:.1f}%")
        
        # Pause execution during critical system state
        if self.state == ExecutionState.RUNNING:
            self.pause()
            self.logger.info("Autonomous execution paused due to critical system state")
    
    def record_activity(self, source: str = "unknown") -> None:
        """Record user activity to reset idle timer."""
        if self.idle_processor:
            self.idle_processor.record_activity(source)
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive execution engine status."""
        status = {
            'state': self.state.value,
            'orchestration_available': ORCHESTRATION_AVAILABLE,
            'execution_start_time': self.execution_start_time.isoformat() if self.execution_start_time else None,
            'config': {
                'idle_threshold': self.config.idle_threshold_seconds,
                'max_processing_time': self.config.max_processing_time_seconds,
                'system_monitoring': self.config.enable_system_monitoring,
                'safety_validation': self.config.enable_safety_validation,
                'max_concurrent_goals': self.config.max_concurrent_goals
            },
            'statistics': self.stats.copy()
        }
        
        # Add sub-component status
        if self.idle_processor:
            status['idle_processor'] = self.idle_processor.get_status()
        
        if self.system_monitor:
            status['system_monitor'] = self.system_monitor.get_current_status()
        
        return status
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        stats = self.stats.copy()
        
        # Add current uptime
        if self.execution_start_time:
            current_uptime = (datetime.now() - self.execution_start_time).total_seconds()
            stats['current_uptime_seconds'] = current_uptime
        
        # Add sub-component statistics
        if self.idle_processor:
            stats['idle_processor_stats'] = self.idle_processor.get_statistics()
        
        if self.system_monitor:
            stats['system_monitor_stats'] = self.system_monitor.get_statistics()
        
        return stats
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update execution configuration.
        
        Args:
            new_config: Dictionary with new configuration values
        """
        for key, value in new_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.info(f"Updated config {key} = {value}")
            else:
                self.logger.warning(f"Unknown config key: {key}")
        
        # Update sub-component configurations
        if self.idle_processor and 'idle_threshold_seconds' in new_config:
            self.idle_processor.update_config({'idle_threshold_seconds': new_config['idle_threshold_seconds']})
        
        if self.system_monitor and any(k.endswith('_threshold_percent') for k in new_config.keys()):
            threshold_updates = {k: v for k, v in new_config.items() if k.endswith('_threshold_percent')}
            self.system_monitor.update_thresholds(threshold_updates)


def create_execution_engine(goal_stack: GoalStack,
                           motivation_engine: MotivationEngine,
                           safety_validator: GoalSafetyValidator,
                           config: Optional[Dict[str, Any]] = None,
                           coordinator: Optional[CoordinatorEngine] = None,
                           planner: Optional[DynamicPlanner] = None) -> AutonomousExecutionEngine:
    """
    Factory function to create an AutonomousExecutionEngine.

    Args:
        goal_stack: GoalStack instance
        motivation_engine: MotivationEngine instance
        safety_validator: GoalSafetyValidator instance
        config: Optional configuration dictionary
        coordinator: Optional CoordinatorEngine for plan execution
        planner: Optional DynamicPlanner for plan generation

    Returns:
        Configured AutonomousExecutionEngine instance
    """
    # Create config object
    if config:
        execution_config = ExecutionConfig(**config)
    else:
        execution_config = ExecutionConfig()

    # Create execution engine
    engine = AutonomousExecutionEngine(
        goal_stack=goal_stack,
        motivation_engine=motivation_engine,
        safety_validator=safety_validator,
        config=execution_config,
        coordinator=coordinator,
        planner=planner
    )

    return engine
