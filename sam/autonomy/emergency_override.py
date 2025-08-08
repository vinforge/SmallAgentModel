"""
Emergency Override System for SAM Autonomy
==========================================

This module implements global emergency controls with automatic pause triggers
and safety circuit breakers for autonomous operation.

Phase C: Full Autonomy with Monitoring

Author: SAM Development Team
Version: 2.0.0
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class EmergencyLevel(Enum):
    """Emergency severity levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class OverrideReason(Enum):
    """Reasons for emergency override activation."""
    MANUAL = "manual"
    SYSTEM_OVERLOAD = "system_overload"
    SAFETY_VIOLATION = "safety_violation"
    EXECUTION_FAILURE = "execution_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    EXTERNAL_SIGNAL = "external_signal"
    TIMEOUT = "timeout"
    UNKNOWN_ERROR = "unknown_error"

@dataclass
class EmergencyEvent:
    """Emergency event record."""
    timestamp: datetime
    level: EmergencyLevel
    reason: OverrideReason
    message: str
    source: str
    auto_triggered: bool
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    resolution_message: Optional[str] = None

@dataclass
class SafetyThresholds:
    """Safety thresholds for automatic emergency triggers."""
    max_cpu_percent: float = 90.0
    max_memory_percent: float = 95.0
    max_failure_rate: float = 50.0  # Percentage
    max_execution_time_minutes: float = 30.0
    max_consecutive_failures: int = 5
    min_available_memory_gb: float = 1.0
    max_goal_queue_size: int = 1000

class EmergencyOverrideSystem:
    """
    Global emergency control system for SAM's autonomous operation.
    
    This system provides multiple layers of safety controls:
    - Manual emergency stops and pauses
    - Automatic triggers based on system conditions
    - Safety circuit breakers for critical situations
    - Emergency event logging and tracking
    - Recovery and reset mechanisms
    - Integration with all autonomy components
    
    Features:
    - Multi-level emergency classification
    - Automatic trigger conditions
    - Manual override controls
    - Event logging and history
    - Recovery procedures
    - Integration hooks for all components
    """
    
    def __init__(self, 
                 safety_thresholds: Optional[SafetyThresholds] = None,
                 monitoring_interval: float = 2.0):
        """
        Initialize the emergency override system.
        
        Args:
            safety_thresholds: Safety thresholds for automatic triggers
            monitoring_interval: Seconds between safety checks
        """
        self.logger = logging.getLogger(f"{__name__}.EmergencyOverrideSystem")
        
        # Configuration
        self.safety_thresholds = safety_thresholds or SafetyThresholds()
        self.monitoring_interval = monitoring_interval
        
        # State management
        self.current_level = EmergencyLevel.NONE
        self.is_active = False
        self.override_reason: Optional[OverrideReason] = None
        self.activation_time: Optional[datetime] = None
        
        # Event tracking
        self.emergency_events: List[EmergencyEvent] = []
        self.active_events: List[EmergencyEvent] = []
        
        # Monitoring
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Callback registrations
        self.emergency_callbacks: Dict[EmergencyLevel, List[Callable]] = {
            level: [] for level in EmergencyLevel
        }
        self.recovery_callbacks: List[Callable] = []
        
        # Component references (to be set by registration)
        self.registered_components: Dict[str, Any] = {}
        
        # Statistics
        self.stats = {
            'total_emergencies': 0,
            'manual_activations': 0,
            'auto_activations': 0,
            'false_positives': 0,
            'average_resolution_time': 0.0,
            'last_emergency': None,
            'uptime_since_last_emergency': 0.0
        }
        
        self.logger.info("EmergencyOverrideSystem initialized")
    
    def start_monitoring(self) -> bool:
        """
        Start emergency monitoring.
        
        Returns:
            True if monitoring started successfully, False otherwise
        """
        if self._monitoring:
            self.logger.warning("Emergency monitoring already running")
            return False
        
        try:
            self._monitoring = True
            self._stop_event.clear()
            
            # Start monitoring thread
            self._monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                name="EmergencyOverrideMonitor",
                daemon=True
            )
            self._monitor_thread.start()
            
            self.logger.info("Emergency monitoring started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start emergency monitoring: {e}")
            self._monitoring = False
            return False
    
    def stop_monitoring(self) -> bool:
        """
        Stop emergency monitoring.
        
        Returns:
            True if monitoring stopped successfully, False otherwise
        """
        if not self._monitoring:
            self.logger.warning("Emergency monitoring not running")
            return False
        
        try:
            self.logger.info("Stopping emergency monitoring...")
            
            # Signal stop
            self._monitoring = False
            self._stop_event.set()
            
            # Wait for thread to finish
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=5.0)
                
                if self._monitor_thread.is_alive():
                    self.logger.warning("Monitor thread did not stop gracefully")
            
            self.logger.info("Emergency monitoring stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping emergency monitoring: {e}")
            return False
    
    def activate_emergency(self, 
                          level: EmergencyLevel,
                          reason: OverrideReason,
                          message: str,
                          source: str = "manual",
                          auto_triggered: bool = False) -> bool:
        """
        Activate emergency override.
        
        Args:
            level: Emergency severity level
            reason: Reason for activation
            message: Descriptive message
            source: Source of the emergency
            auto_triggered: Whether this was automatically triggered
            
        Returns:
            True if emergency activated successfully, False otherwise
        """
        try:
            self.logger.critical(f"EMERGENCY ACTIVATED: {level.value} - {message}")
            
            # Create emergency event
            event = EmergencyEvent(
                timestamp=datetime.now(),
                level=level,
                reason=reason,
                message=message,
                source=source,
                auto_triggered=auto_triggered
            )
            
            # Update state
            self.current_level = level
            self.is_active = True
            self.override_reason = reason
            self.activation_time = datetime.now()
            
            # Add to event tracking
            self.emergency_events.append(event)
            self.active_events.append(event)
            
            # Update statistics
            self.stats['total_emergencies'] += 1
            if auto_triggered:
                self.stats['auto_activations'] += 1
            else:
                self.stats['manual_activations'] += 1
            self.stats['last_emergency'] = datetime.now()
            
            # Trigger emergency actions
            self._execute_emergency_actions(level, event)
            
            # Notify callbacks
            self._notify_emergency_callbacks(level, event)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error activating emergency: {e}")
            return False
    
    def deactivate_emergency(self, resolution_message: str = "Manual deactivation") -> bool:
        """
        Deactivate emergency override.
        
        Args:
            resolution_message: Message describing the resolution
            
        Returns:
            True if emergency deactivated successfully, False otherwise
        """
        if not self.is_active:
            self.logger.warning("No active emergency to deactivate")
            return False
        
        try:
            self.logger.info(f"EMERGENCY DEACTIVATED: {resolution_message}")
            
            # Calculate resolution time
            resolution_time = datetime.now()
            if self.activation_time:
                duration = (resolution_time - self.activation_time).total_seconds()
                
                # Update average resolution time
                total_events = len(self.emergency_events)
                if total_events > 0:
                    current_avg = self.stats['average_resolution_time']
                    self.stats['average_resolution_time'] = (
                        (current_avg * (total_events - 1) + duration) / total_events
                    )
            
            # Resolve active events
            for event in self.active_events:
                event.resolved = True
                event.resolution_time = resolution_time
                event.resolution_message = resolution_message
            
            # Reset state
            self.current_level = EmergencyLevel.NONE
            self.is_active = False
            self.override_reason = None
            self.activation_time = None
            self.active_events.clear()
            
            # Execute recovery actions
            self._execute_recovery_actions(resolution_message)
            
            # Notify recovery callbacks
            self._notify_recovery_callbacks(resolution_message)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error deactivating emergency: {e}")
            return False
    
    def register_component(self, name: str, component: Any) -> None:
        """
        Register a component for emergency control.
        
        Args:
            name: Component name
            component: Component instance
        """
        self.registered_components[name] = component
        self.logger.info(f"Component registered for emergency control: {name}")
    
    def register_emergency_callback(self, level: EmergencyLevel, callback: Callable) -> None:
        """
        Register a callback for emergency events.
        
        Args:
            level: Emergency level to monitor
            callback: Function to call when emergency occurs
        """
        self.emergency_callbacks[level].append(callback)
        self.logger.info(f"Emergency callback registered for level: {level.value}")
    
    def register_recovery_callback(self, callback: Callable) -> None:
        """
        Register a callback for recovery events.
        
        Args:
            callback: Function to call when emergency is resolved
        """
        self.recovery_callbacks.append(callback)
        self.logger.info("Recovery callback registered")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop for automatic emergency detection."""
        self.logger.info("Emergency monitoring loop started")
        
        try:
            while self._monitoring and not self._stop_event.is_set():
                try:
                    # Check safety conditions
                    self._check_safety_conditions()
                    
                    # Sleep until next check
                    self._stop_event.wait(self.monitoring_interval)
                    
                except Exception as e:
                    self.logger.error(f"Error in emergency monitoring loop: {e}")
                    self._stop_event.wait(self.monitoring_interval * 2)
        
        except Exception as e:
            self.logger.critical(f"Critical error in emergency monitoring loop: {e}")
        
        finally:
            self.logger.info("Emergency monitoring loop ended")
    
    def _check_safety_conditions(self) -> None:
        """Check all safety conditions for automatic emergency triggers."""
        # Skip if emergency already active
        if self.is_active:
            return
        
        try:
            # Check system resources
            self._check_system_resources()
            
            # Check execution metrics
            self._check_execution_metrics()
            
            # Check component health
            self._check_component_health()
            
        except Exception as e:
            self.logger.error(f"Error checking safety conditions: {e}")
    
    def _check_system_resources(self) -> None:
        """Check system resource conditions."""
        try:
            import psutil
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.safety_thresholds.max_cpu_percent:
                self.activate_emergency(
                    EmergencyLevel.HIGH,
                    OverrideReason.SYSTEM_OVERLOAD,
                    f"CPU usage critical: {cpu_percent:.1f}%",
                    "system_monitor",
                    auto_triggered=True
                )
                return
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > self.safety_thresholds.max_memory_percent:
                self.activate_emergency(
                    EmergencyLevel.HIGH,
                    OverrideReason.SYSTEM_OVERLOAD,
                    f"Memory usage critical: {memory.percent:.1f}%",
                    "system_monitor",
                    auto_triggered=True
                )
                return
            
            # Check available memory
            available_gb = memory.available / (1024**3)
            if available_gb < self.safety_thresholds.min_available_memory_gb:
                self.activate_emergency(
                    EmergencyLevel.MEDIUM,
                    OverrideReason.RESOURCE_EXHAUSTION,
                    f"Available memory low: {available_gb:.1f}GB",
                    "system_monitor",
                    auto_triggered=True
                )
                return
            
        except ImportError:
            # psutil not available, skip system resource checks
            pass
        except Exception as e:
            self.logger.warning(f"Error checking system resources: {e}")
    
    def _check_execution_metrics(self) -> None:
        """Check execution performance metrics."""
        # Check if execution engine is registered
        execution_engine = self.registered_components.get('execution_engine')
        if not execution_engine:
            return
        
        try:
            # Get execution statistics
            stats = execution_engine.get_statistics()
            
            # Check failure rate
            total_executions = stats.get('total_executions', 0)
            failed_executions = stats.get('failed_executions', 0)
            
            if total_executions > 10:  # Only check if we have enough data
                failure_rate = (failed_executions / total_executions) * 100
                
                if failure_rate > self.safety_thresholds.max_failure_rate:
                    self.activate_emergency(
                        EmergencyLevel.MEDIUM,
                        OverrideReason.EXECUTION_FAILURE,
                        f"High failure rate: {failure_rate:.1f}%",
                        "execution_monitor",
                        auto_triggered=True
                    )
                    return
            
        except Exception as e:
            self.logger.warning(f"Error checking execution metrics: {e}")
    
    def _check_component_health(self) -> None:
        """Check health of registered components."""
        for name, component in self.registered_components.items():
            try:
                # Check if component has a health check method
                if hasattr(component, 'get_status'):
                    status = component.get_status()
                    
                    # Look for error states
                    if isinstance(status, dict):
                        state = status.get('state', '').lower()
                        if 'error' in state or 'critical' in state:
                            self.activate_emergency(
                                EmergencyLevel.MEDIUM,
                                OverrideReason.UNKNOWN_ERROR,
                                f"Component {name} in error state: {state}",
                                f"component_{name}",
                                auto_triggered=True
                            )
                            return
                
            except Exception as e:
                self.logger.warning(f"Error checking health of component {name}: {e}")
    
    def _execute_emergency_actions(self, level: EmergencyLevel, event: EmergencyEvent) -> None:
        """Execute emergency actions based on severity level."""
        try:
            # Stop or pause components based on emergency level
            if level in [EmergencyLevel.CRITICAL, EmergencyLevel.HIGH]:
                # Emergency stop all components
                for name, component in self.registered_components.items():
                    try:
                        if hasattr(component, 'emergency_stop'):
                            component.emergency_stop()
                            self.logger.info(f"Emergency stop executed for {name}")
                        elif hasattr(component, 'stop'):
                            component.stop()
                            self.logger.info(f"Stop executed for {name}")
                    except Exception as e:
                        self.logger.error(f"Error stopping component {name}: {e}")
            
            elif level == EmergencyLevel.MEDIUM:
                # Pause components
                for name, component in self.registered_components.items():
                    try:
                        if hasattr(component, 'pause'):
                            component.pause()
                            self.logger.info(f"Pause executed for {name}")
                    except Exception as e:
                        self.logger.error(f"Error pausing component {name}: {e}")
            
        except Exception as e:
            self.logger.error(f"Error executing emergency actions: {e}")
    
    def _execute_recovery_actions(self, resolution_message: str) -> None:
        """Execute recovery actions after emergency deactivation."""
        try:
            # Resume components that were paused
            for name, component in self.registered_components.items():
                try:
                    if hasattr(component, 'resume'):
                        component.resume()
                        self.logger.info(f"Resume executed for {name}")
                    elif hasattr(component, 'reset_emergency_stop'):
                        component.reset_emergency_stop()
                        self.logger.info(f"Emergency stop reset for {name}")
                except Exception as e:
                    self.logger.error(f"Error resuming component {name}: {e}")
            
        except Exception as e:
            self.logger.error(f"Error executing recovery actions: {e}")
    
    def _notify_emergency_callbacks(self, level: EmergencyLevel, event: EmergencyEvent) -> None:
        """Notify registered emergency callbacks."""
        callbacks = self.emergency_callbacks.get(level, [])
        for callback in callbacks:
            try:
                callback(level, event)
            except Exception as e:
                self.logger.error(f"Error in emergency callback: {e}")
    
    def _notify_recovery_callbacks(self, resolution_message: str) -> None:
        """Notify registered recovery callbacks."""
        for callback in self.recovery_callbacks:
            try:
                callback(resolution_message)
            except Exception as e:
                self.logger.error(f"Error in recovery callback: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current emergency system status."""
        return {
            'is_active': self.is_active,
            'current_level': self.current_level.value,
            'override_reason': self.override_reason.value if self.override_reason else None,
            'activation_time': self.activation_time.isoformat() if self.activation_time else None,
            'monitoring': self._monitoring,
            'active_events_count': len(self.active_events),
            'total_events_count': len(self.emergency_events),
            'registered_components': list(self.registered_components.keys()),
            'safety_thresholds': {
                'max_cpu_percent': self.safety_thresholds.max_cpu_percent,
                'max_memory_percent': self.safety_thresholds.max_memory_percent,
                'max_failure_rate': self.safety_thresholds.max_failure_rate
            }
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get emergency system statistics."""
        stats = self.stats.copy()
        
        # Add uptime since last emergency
        if self.stats['last_emergency']:
            uptime = (datetime.now() - self.stats['last_emergency']).total_seconds()
            stats['uptime_since_last_emergency'] = uptime
        
        return stats
    
    def get_emergency_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get emergency event history.
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of emergency events
        """
        events = self.emergency_events
        if limit:
            events = events[-limit:]
        
        return [
            {
                'timestamp': event.timestamp.isoformat(),
                'level': event.level.value,
                'reason': event.reason.value,
                'message': event.message,
                'source': event.source,
                'auto_triggered': event.auto_triggered,
                'resolved': event.resolved,
                'resolution_time': event.resolution_time.isoformat() if event.resolution_time else None,
                'resolution_message': event.resolution_message
            }
            for event in events
        ]
    
    def update_safety_thresholds(self, new_thresholds: Dict[str, float]) -> None:
        """
        Update safety thresholds.
        
        Args:
            new_thresholds: Dictionary with new threshold values
        """
        for key, value in new_thresholds.items():
            if hasattr(self.safety_thresholds, key):
                setattr(self.safety_thresholds, key, value)
                self.logger.info(f"Updated safety threshold {key} = {value}")
            else:
                self.logger.warning(f"Unknown safety threshold key: {key}")


def create_emergency_override_system(safety_thresholds: Optional[Dict[str, float]] = None,
                                   monitoring_interval: float = 2.0) -> EmergencyOverrideSystem:
    """
    Factory function to create an EmergencyOverrideSystem.
    
    Args:
        safety_thresholds: Optional safety threshold configuration
        monitoring_interval: Seconds between safety checks
        
    Returns:
        Configured EmergencyOverrideSystem instance
    """
    # Create thresholds object
    if safety_thresholds:
        threshold_obj = SafetyThresholds(**safety_thresholds)
    else:
        threshold_obj = SafetyThresholds()
    
    # Create emergency system
    emergency_system = EmergencyOverrideSystem(
        safety_thresholds=threshold_obj,
        monitoring_interval=monitoring_interval
    )
    
    return emergency_system
