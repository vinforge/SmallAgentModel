"""
Idle Time Processing System for SAM Autonomy
============================================

This module implements the background task scheduler that monitors system idle time
and executes autonomous goals when appropriate, with comprehensive safety oversight.

Phase C: Full Autonomy with Monitoring

Author: SAM Development Team
Version: 2.0.0
"""

import asyncio
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum

from .goals import Goal
from .goal_stack import GoalStack
from .motivation_engine import MotivationEngine
from .safety.goal_validator import GoalSafetyValidator

logger = logging.getLogger(__name__)

class IdleState(Enum):
    """System idle states."""
    ACTIVE = "active"
    IDLE = "idle"
    PROCESSING = "processing"
    PAUSED = "paused"
    ERROR = "error"

@dataclass
class IdleProcessingConfig:
    """Configuration for idle processing system."""
    idle_threshold_seconds: float = 30.0  # Time before considering system idle
    max_processing_time_seconds: float = 300.0  # Max time for autonomous processing
    check_interval_seconds: float = 5.0  # How often to check idle status
    max_concurrent_goals: int = 1  # Max goals to process simultaneously
    enable_system_load_monitoring: bool = True
    cpu_threshold_percent: float = 80.0  # Pause if CPU usage above this
    memory_threshold_percent: float = 85.0  # Pause if memory usage above this
    enable_user_activity_detection: bool = True
    user_activity_sources: list = None  # Sources to monitor for user activity

    def __post_init__(self):
        if self.user_activity_sources is None:
            self.user_activity_sources = ["streamlit", "api", "cli"]

class IdleTimeProcessor:
    """
    Background processor for autonomous goal execution during idle periods.
    
    This system monitors SAM's activity and automatically executes high-priority
    autonomous goals when the system is idle, with comprehensive safety oversight
    and resource monitoring.
    
    Features:
    - Idle time detection and monitoring
    - System resource monitoring (CPU, memory)
    - User activity detection
    - Safe autonomous goal execution
    - Emergency pause mechanisms
    - Comprehensive logging and monitoring
    """
    
    def __init__(self, 
                 goal_stack: GoalStack,
                 motivation_engine: MotivationEngine,
                 safety_validator: GoalSafetyValidator,
                 config: Optional[IdleProcessingConfig] = None,
                 goal_executor: Optional[Callable] = None):
        """
        Initialize the idle time processor.
        
        Args:
            goal_stack: GoalStack for retrieving goals
            motivation_engine: MotivationEngine for goal generation
            safety_validator: Safety validator for validation
            config: Optional configuration
            goal_executor: Optional custom goal executor function
        """
        self.logger = logging.getLogger(f"{__name__}.IdleTimeProcessor")
        
        # Core components
        self.goal_stack = goal_stack
        self.motivation_engine = motivation_engine
        self.safety_validator = safety_validator
        self.config = config or IdleProcessingConfig()
        self.goal_executor = goal_executor
        
        # State management
        self.state = IdleState.ACTIVE
        self.last_activity_time = datetime.now()
        self.processing_start_time: Optional[datetime] = None
        self.current_goal: Optional[Goal] = None
        
        # Control flags
        self._running = False
        self._paused = False
        self._emergency_stop = False
        
        # Threading
        self._processor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Statistics
        self.stats = {
            'total_idle_periods': 0,
            'total_goals_processed': 0,
            'total_processing_time': 0.0,
            'successful_executions': 0,
            'failed_executions': 0,
            'emergency_stops': 0,
            'last_processing_time': None,
            'average_processing_time': 0.0
        }
        
        # Activity tracking
        self.activity_callbacks: Dict[str, Callable] = {}
        
        self.logger.info("IdleTimeProcessor initialized")
    
    def start(self) -> bool:
        """
        Start the idle time processor.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self._running:
            self.logger.warning("IdleTimeProcessor already running")
            return False
        
        try:
            self._running = True
            self._stop_event.clear()
            
            # Start processor thread
            self._processor_thread = threading.Thread(
                target=self._processor_loop,
                name="IdleTimeProcessor",
                daemon=True
            )
            self._processor_thread.start()
            
            self.logger.info("IdleTimeProcessor started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start IdleTimeProcessor: {e}")
            self._running = False
            return False
    
    def stop(self) -> bool:
        """
        Stop the idle time processor.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        if not self._running:
            self.logger.warning("IdleTimeProcessor not running")
            return False
        
        try:
            self.logger.info("Stopping IdleTimeProcessor...")
            
            # Signal stop
            self._running = False
            self._stop_event.set()
            
            # Wait for thread to finish
            if self._processor_thread and self._processor_thread.is_alive():
                self._processor_thread.join(timeout=10.0)
                
                if self._processor_thread.is_alive():
                    self.logger.warning("Processor thread did not stop gracefully")
            
            self.logger.info("IdleTimeProcessor stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping IdleTimeProcessor: {e}")
            return False
    
    def pause(self) -> None:
        """Pause autonomous processing."""
        self._paused = True
        self.state = IdleState.PAUSED
        self.logger.info("IdleTimeProcessor paused")
    
    def resume(self) -> None:
        """Resume autonomous processing."""
        self._paused = False
        self.state = IdleState.ACTIVE
        self.logger.info("IdleTimeProcessor resumed")
    
    def emergency_stop(self) -> None:
        """Emergency stop all autonomous processing."""
        self._emergency_stop = True
        self._paused = True
        self.state = IdleState.ERROR
        self.stats['emergency_stops'] += 1
        self.logger.critical("EMERGENCY STOP activated for IdleTimeProcessor")
    
    def reset_emergency_stop(self) -> None:
        """Reset emergency stop state."""
        self._emergency_stop = False
        self._paused = False
        self.state = IdleState.ACTIVE
        self.logger.info("Emergency stop reset, IdleTimeProcessor resumed")
    
    def record_activity(self, source: str = "unknown") -> None:
        """
        Record user activity to reset idle timer.
        
        Args:
            source: Source of the activity (e.g., "streamlit", "api", "cli")
        """
        self.last_activity_time = datetime.now()
        
        if self.state == IdleState.IDLE:
            self.state = IdleState.ACTIVE
        
        self.logger.debug(f"Activity recorded from source: {source}")
    
    def register_activity_callback(self, source: str, callback: Callable) -> None:
        """
        Register a callback for activity detection.
        
        Args:
            source: Activity source name
            callback: Function to call for activity detection
        """
        self.activity_callbacks[source] = callback
        self.logger.info(f"Activity callback registered for source: {source}")
    
    def _processor_loop(self) -> None:
        """Main processor loop running in background thread."""
        self.logger.info("Processor loop started")
        
        try:
            while self._running and not self._stop_event.is_set():
                try:
                    # Check for emergency stop
                    if self._emergency_stop:
                        self.logger.debug("Emergency stop active, skipping processing")
                        time.sleep(self.config.check_interval_seconds)
                        continue
                    
                    # Check if paused
                    if self._paused:
                        self.logger.debug("Processor paused, skipping processing")
                        time.sleep(self.config.check_interval_seconds)
                        continue
                    
                    # Update state based on activity
                    self._update_idle_state()
                    
                    # Process goals if idle
                    if self.state == IdleState.IDLE:
                        self._process_idle_goals()
                    
                    # Sleep until next check
                    time.sleep(self.config.check_interval_seconds)
                    
                except Exception as e:
                    self.logger.error(f"Error in processor loop: {e}")
                    self.state = IdleState.ERROR
                    time.sleep(self.config.check_interval_seconds * 2)  # Longer sleep on error
        
        except Exception as e:
            self.logger.critical(f"Critical error in processor loop: {e}")
            self.state = IdleState.ERROR
        
        finally:
            self.logger.info("Processor loop ended")
    
    def _update_idle_state(self) -> None:
        """Update the idle state based on activity and system conditions."""
        now = datetime.now()
        time_since_activity = (now - self.last_activity_time).total_seconds()
        
        # Check activity callbacks
        if self.config.enable_user_activity_detection:
            for source, callback in self.activity_callbacks.items():
                try:
                    if callback():  # If callback returns True, activity detected
                        self.record_activity(source)
                        return
                except Exception as e:
                    self.logger.warning(f"Activity callback error for {source}: {e}")
        
        # Check system load if enabled
        if self.config.enable_system_load_monitoring:
            if not self._check_system_resources():
                if self.state == IdleState.IDLE:
                    self.state = IdleState.ACTIVE
                return
        
        # Update state based on time since activity
        if time_since_activity >= self.config.idle_threshold_seconds:
            if self.state == IdleState.ACTIVE:
                self.state = IdleState.IDLE
                self.stats['total_idle_periods'] += 1
                self.logger.info(f"System idle detected after {time_since_activity:.1f}s")
        else:
            if self.state == IdleState.IDLE:
                self.state = IdleState.ACTIVE
    
    def _check_system_resources(self) -> bool:
        """
        Check if system resources are available for processing.
        
        Returns:
            True if resources are available, False otherwise
        """
        try:
            import psutil
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.config.cpu_threshold_percent:
                self.logger.debug(f"CPU usage too high: {cpu_percent:.1f}%")
                return False
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > self.config.memory_threshold_percent:
                self.logger.debug(f"Memory usage too high: {memory.percent:.1f}%")
                return False
            
            return True
            
        except ImportError:
            self.logger.warning("psutil not available, skipping system resource monitoring")
            return True
        except Exception as e:
            self.logger.warning(f"Error checking system resources: {e}")
            return True  # Default to allowing processing if check fails

    def _process_idle_goals(self) -> None:
        """Process autonomous goals during idle time."""
        try:
            # Check if already processing
            if self.state == IdleState.PROCESSING:
                # Check if processing time exceeded
                if (self.processing_start_time and
                    (datetime.now() - self.processing_start_time).total_seconds() >
                    self.config.max_processing_time_seconds):

                    self.logger.warning("Processing time exceeded, stopping current goal")
                    self._stop_current_goal()
                    return
                else:
                    return  # Still processing current goal

            # Check if memory synthesis should run (Task 33, Phase 2)
            if self._should_run_memory_synthesis():
                self._execute_memory_synthesis()
                return

            # Get top priority goals
            top_goals = self.goal_stack.get_top_priority_goals(
                limit=self.config.max_concurrent_goals,
                status="pending"
            )

            if not top_goals:
                self.logger.debug("No pending goals to process")
                return

            # Select goal to process
            goal_to_process = top_goals[0]  # Highest priority

            # Validate goal before processing
            is_valid, error_msg = self.safety_validator.validate_goal(goal_to_process)
            if not is_valid:
                self.logger.warning(f"Goal failed validation: {error_msg}")
                self.goal_stack.update_goal_status(
                    goal_to_process.goal_id,
                    "failed",
                    f"Validation failed: {error_msg}"
                )
                return

            # Start processing
            self._start_goal_processing(goal_to_process)

        except Exception as e:
            self.logger.error(f"Error processing idle goals: {e}")
            self.state = IdleState.ERROR

    def _start_goal_processing(self, goal: Goal) -> None:
        """Start processing a specific goal."""
        try:
            self.logger.info(f"Starting autonomous goal processing: {goal.goal_id}")

            # Update state
            self.state = IdleState.PROCESSING
            self.processing_start_time = datetime.now()
            self.current_goal = goal

            # Update goal status
            self.goal_stack.update_goal_status(goal.goal_id, "active")

            # Execute goal
            success = self._execute_goal(goal)

            # Update statistics and status
            processing_time = (datetime.now() - self.processing_start_time).total_seconds()
            self.stats['total_processing_time'] += processing_time
            self.stats['last_processing_time'] = datetime.now()

            if success:
                self.stats['successful_executions'] += 1
                self.goal_stack.update_goal_status(goal.goal_id, "completed")
                self.logger.info(f"Goal completed successfully: {goal.goal_id}")
            else:
                self.stats['failed_executions'] += 1
                self.goal_stack.update_goal_status(
                    goal.goal_id,
                    "failed",
                    "Execution failed during autonomous processing"
                )
                self.logger.warning(f"Goal execution failed: {goal.goal_id}")

            # Update average processing time
            total_executions = self.stats['successful_executions'] + self.stats['failed_executions']
            if total_executions > 0:
                self.stats['average_processing_time'] = (
                    self.stats['total_processing_time'] / total_executions
                )

            self.stats['total_goals_processed'] += 1

        except Exception as e:
            self.logger.error(f"Error starting goal processing: {e}")
            if self.current_goal:
                self.goal_stack.update_goal_status(
                    self.current_goal.goal_id,
                    "failed",
                    f"Processing error: {str(e)}"
                )

        finally:
            # Reset processing state
            self.state = IdleState.IDLE
            self.processing_start_time = None
            self.current_goal = None

    def _execute_goal(self, goal: Goal) -> bool:
        """
        Execute a specific goal.

        Args:
            goal: Goal to execute

        Returns:
            True if execution successful, False otherwise
        """
        try:
            if self.goal_executor:
                # Use custom executor if provided
                return self.goal_executor(goal)
            else:
                # Default execution logic
                return self._default_goal_execution(goal)

        except Exception as e:
            self.logger.error(f"Goal execution error: {e}")
            return False

    def _default_goal_execution(self, goal: Goal) -> bool:
        """
        Default goal execution implementation.

        This creates a plan for the goal and simulates execution.
        In a full implementation, this would integrate with the
        CoordinatorEngine to actually execute the plan.

        Args:
            goal: Goal to execute

        Returns:
            True if execution successful, False otherwise
        """
        try:
            self.logger.info(f"Executing goal: {goal.description}")

            # Simulate goal execution
            # In real implementation, this would:
            # 1. Create a UIF for the goal
            # 2. Generate a plan using DynamicPlanner
            # 3. Execute the plan using CoordinatorEngine
            # 4. Handle results and update goal status

            # For now, simulate processing time
            import time
            time.sleep(2)  # Simulate work

            # Simulate success/failure based on goal priority
            # Higher priority goals are more likely to succeed
            success_probability = min(0.9, goal.priority + 0.1)
            import random
            success = random.random() < success_probability

            if success:
                self.logger.info(f"Goal execution simulated successfully: {goal.goal_id}")
            else:
                self.logger.warning(f"Goal execution simulated failure: {goal.goal_id}")

            return success

        except Exception as e:
            self.logger.error(f"Default goal execution error: {e}")
            return False

    def _stop_current_goal(self) -> None:
        """Stop processing the current goal."""
        if self.current_goal:
            self.logger.warning(f"Stopping current goal processing: {self.current_goal.goal_id}")

            self.goal_stack.update_goal_status(
                self.current_goal.goal_id,
                "failed",
                "Processing stopped due to timeout or interruption"
            )

            self.current_goal = None

        self.state = IdleState.IDLE
        self.processing_start_time = None

    def get_status(self) -> Dict[str, Any]:
        """
        Get current processor status.

        Returns:
            Dictionary with current status information
        """
        now = datetime.now()
        time_since_activity = (now - self.last_activity_time).total_seconds()

        status = {
            'state': self.state.value,
            'running': self._running,
            'paused': self._paused,
            'emergency_stop': self._emergency_stop,
            'time_since_activity': time_since_activity,
            'is_idle': time_since_activity >= self.config.idle_threshold_seconds,
            'current_goal': self.current_goal.goal_id if self.current_goal else None,
            'processing_time': (
                (now - self.processing_start_time).total_seconds()
                if self.processing_start_time else 0
            ),
            'config': {
                'idle_threshold': self.config.idle_threshold_seconds,
                'max_processing_time': self.config.max_processing_time_seconds,
                'check_interval': self.config.check_interval_seconds,
                'max_concurrent_goals': self.config.max_concurrent_goals,
                'system_monitoring': self.config.enable_system_load_monitoring,
                'cpu_threshold': self.config.cpu_threshold_percent,
                'memory_threshold': self.config.memory_threshold_percent
            },
            'statistics': self.stats.copy()
        }

        return status

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update processor configuration.

        Args:
            new_config: Dictionary with new configuration values
        """
        for key, value in new_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.info(f"Updated config {key} = {value}")
            else:
                self.logger.warning(f"Unknown config key: {key}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()

    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self.stats = {
            'total_idle_periods': 0,
            'total_goals_processed': 0,
            'total_processing_time': 0.0,
            'successful_executions': 0,
            'failed_executions': 0,
            'emergency_stops': 0,
            'last_processing_time': None,
            'average_processing_time': 0.0
        }
        self.logger.info("Processing statistics reset")

    def _should_run_memory_synthesis(self) -> bool:
        """Check if memory synthesis should run during this idle cycle (Task 33, Phase 2)."""
        try:
            # Import memory synthesizer
            from memory.synthesis.conversation_synthesizer import MemorySynthesizer

            # Initialize synthesizer if not exists
            if not hasattr(self, '_memory_synthesizer'):
                self._memory_synthesizer = MemorySynthesizer()

            # Check if synthesis is due
            return self._memory_synthesizer.should_run_synthesis()

        except Exception as e:
            self.logger.warning(f"Memory synthesis check failed: {e}")
            return False

    def _execute_memory_synthesis(self) -> None:
        """Execute memory synthesis during idle time (Task 33, Phase 2)."""
        try:
            self.logger.info("Starting memory synthesis during idle cycle")
            self.state = IdleState.PROCESSING
            self.processing_start_time = datetime.now()

            # Import and run memory synthesizer
            from memory.synthesis.conversation_synthesizer import MemorySynthesizer

            if not hasattr(self, '_memory_synthesizer'):
                self._memory_synthesizer = MemorySynthesizer()

            # Run conversation synthesis (use asyncio.run for sync context)
            import asyncio
            synthesis_result = asyncio.run(self._memory_synthesizer.run_conversation_synthesis())

            # Log results
            if synthesis_result.get('status') == 'completed':
                insights_count = synthesis_result.get('insights_generated', 0)
                self.logger.info(f"Memory synthesis completed: {insights_count} insights generated")
                self.stats['successful_executions'] += 1
            else:
                self.logger.warning(f"Memory synthesis failed: {synthesis_result.get('error', 'Unknown error')}")
                self.stats['failed_executions'] += 1

            # Update state
            self.state = IdleState.IDLE
            self.processing_start_time = None

        except Exception as e:
            self.logger.error(f"Memory synthesis execution failed: {e}")
            self.state = IdleState.ERROR
            self.processing_start_time = None
            self.stats['failed_executions'] += 1


def create_idle_processor(goal_stack: GoalStack,
                         motivation_engine: MotivationEngine,
                         safety_validator: GoalSafetyValidator,
                         config: Optional[Dict[str, Any]] = None,
                         goal_executor: Optional[Callable] = None) -> IdleTimeProcessor:
    """
    Factory function to create an IdleTimeProcessor.

    Args:
        goal_stack: GoalStack instance
        motivation_engine: MotivationEngine instance
        safety_validator: GoalSafetyValidator instance
        config: Optional configuration dictionary
        goal_executor: Optional custom goal executor function

    Returns:
        Configured IdleTimeProcessor instance
    """
    # Create config object
    if config:
        processor_config = IdleProcessingConfig(**config)
    else:
        processor_config = IdleProcessingConfig()

    # Create processor
    processor = IdleTimeProcessor(
        goal_stack=goal_stack,
        motivation_engine=motivation_engine,
        safety_validator=safety_validator,
        config=processor_config,
        goal_executor=goal_executor
    )

    return processor
