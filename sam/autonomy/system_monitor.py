"""
System Load Monitoring for SAM Autonomy
=======================================

This module implements comprehensive system resource monitoring to ensure
autonomous processing only occurs when system resources are available.

Phase C: Full Autonomy with Monitoring

Author: SAM Development Team
Version: 2.0.0
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class SystemState(Enum):
    """System resource states."""
    OPTIMAL = "optimal"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNAVAILABLE = "unavailable"

@dataclass
class SystemThresholds:
    """System resource thresholds."""
    cpu_optimal: float = 30.0
    cpu_moderate: float = 50.0
    cpu_high: float = 70.0
    cpu_critical: float = 85.0
    
    memory_optimal: float = 40.0
    memory_moderate: float = 60.0
    memory_high: float = 75.0
    memory_critical: float = 90.0
    
    disk_optimal: float = 50.0
    disk_moderate: float = 70.0
    disk_high: float = 85.0
    disk_critical: float = 95.0
    
    network_optimal: float = 30.0  # MB/s
    network_moderate: float = 50.0
    network_high: float = 80.0
    network_critical: float = 100.0

@dataclass
class SystemMetrics:
    """Current system metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io_mb_s: float
    load_average: Optional[List[float]] = None
    process_count: int = 0
    available_memory_gb: float = 0.0
    
    def get_overall_state(self, thresholds: SystemThresholds) -> SystemState:
        """Determine overall system state based on metrics."""
        states = []
        
        # CPU state
        if self.cpu_percent >= thresholds.cpu_critical:
            states.append(SystemState.CRITICAL)
        elif self.cpu_percent >= thresholds.cpu_high:
            states.append(SystemState.HIGH)
        elif self.cpu_percent >= thresholds.cpu_moderate:
            states.append(SystemState.MODERATE)
        else:
            states.append(SystemState.OPTIMAL)
        
        # Memory state
        if self.memory_percent >= thresholds.memory_critical:
            states.append(SystemState.CRITICAL)
        elif self.memory_percent >= thresholds.memory_high:
            states.append(SystemState.HIGH)
        elif self.memory_percent >= thresholds.memory_moderate:
            states.append(SystemState.MODERATE)
        else:
            states.append(SystemState.OPTIMAL)
        
        # Return worst state
        if SystemState.CRITICAL in states:
            return SystemState.CRITICAL
        elif SystemState.HIGH in states:
            return SystemState.HIGH
        elif SystemState.MODERATE in states:
            return SystemState.MODERATE
        else:
            return SystemState.OPTIMAL

class SystemLoadMonitor:
    """
    Comprehensive system resource monitoring for autonomous processing decisions.
    
    This monitor tracks CPU, memory, disk, and network usage to determine when
    the system is suitable for autonomous goal processing. It provides real-time
    metrics and historical data for decision making.
    
    Features:
    - Real-time resource monitoring
    - Configurable thresholds
    - Historical data tracking
    - Alert system for resource issues
    - Integration with idle processor
    """
    
    def __init__(self, 
                 thresholds: Optional[SystemThresholds] = None,
                 monitoring_interval: float = 5.0,
                 history_size: int = 100):
        """
        Initialize the system load monitor.
        
        Args:
            thresholds: Resource thresholds for state determination
            monitoring_interval: Seconds between monitoring checks
            history_size: Number of historical metrics to keep
        """
        self.logger = logging.getLogger(f"{__name__}.SystemLoadMonitor")
        
        # Configuration
        self.thresholds = thresholds or SystemThresholds()
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        
        # State
        self.current_metrics: Optional[SystemMetrics] = None
        self.current_state = SystemState.UNAVAILABLE
        self.metrics_history: List[SystemMetrics] = []
        
        # Monitoring control
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Alert callbacks
        self.alert_callbacks: Dict[SystemState, List[Callable]] = {
            state: [] for state in SystemState
        }
        
        # Statistics
        self.stats = {
            'monitoring_start_time': None,
            'total_measurements': 0,
            'state_durations': {state.value: 0.0 for state in SystemState},
            'last_state_change': None,
            'alerts_triggered': {state.value: 0 for state in SystemState}
        }
        
        # Check if psutil is available
        try:
            import psutil
            self.psutil_available = True
            self.logger.info("psutil available for system monitoring")
        except ImportError:
            self.psutil_available = False
            self.logger.warning("psutil not available, system monitoring will be limited")
        
        self.logger.info("SystemLoadMonitor initialized")
    
    def start_monitoring(self) -> bool:
        """
        Start system monitoring.
        
        Returns:
            True if monitoring started successfully, False otherwise
        """
        if self._monitoring:
            self.logger.warning("System monitoring already running")
            return False
        
        if not self.psutil_available:
            self.logger.error("Cannot start monitoring: psutil not available")
            return False
        
        try:
            self._monitoring = True
            self._stop_event.clear()
            
            # Start monitoring thread
            self._monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                name="SystemLoadMonitor",
                daemon=True
            )
            self._monitor_thread.start()
            
            self.stats['monitoring_start_time'] = datetime.now()
            self.logger.info("System monitoring started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start system monitoring: {e}")
            self._monitoring = False
            return False
    
    def stop_monitoring(self) -> bool:
        """
        Stop system monitoring.
        
        Returns:
            True if monitoring stopped successfully, False otherwise
        """
        if not self._monitoring:
            self.logger.warning("System monitoring not running")
            return False
        
        try:
            self.logger.info("Stopping system monitoring...")
            
            # Signal stop
            self._monitoring = False
            self._stop_event.set()
            
            # Wait for thread to finish
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=10.0)
                
                if self._monitor_thread.is_alive():
                    self.logger.warning("Monitor thread did not stop gracefully")
            
            self.logger.info("System monitoring stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping system monitoring: {e}")
            return False
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        self.logger.info("System monitoring loop started")
        
        try:
            while self._monitoring and not self._stop_event.is_set():
                try:
                    # Collect metrics
                    metrics = self._collect_metrics()
                    
                    if metrics:
                        # Update current state
                        old_state = self.current_state
                        self.current_metrics = metrics
                        self.current_state = metrics.get_overall_state(self.thresholds)
                        
                        # Add to history
                        self.metrics_history.append(metrics)
                        if len(self.metrics_history) > self.history_size:
                            self.metrics_history.pop(0)
                        
                        # Update statistics
                        self.stats['total_measurements'] += 1
                        
                        # Handle state changes
                        if old_state != self.current_state:
                            self._handle_state_change(old_state, self.current_state)
                        
                        # Trigger alerts if needed
                        self._check_alerts()
                    
                    # Sleep until next check
                    self._stop_event.wait(self.monitoring_interval)
                    
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")
                    self._stop_event.wait(self.monitoring_interval * 2)
        
        except Exception as e:
            self.logger.critical(f"Critical error in monitoring loop: {e}")
        
        finally:
            self.logger.info("System monitoring loop ended")
    
    def _collect_metrics(self) -> Optional[SystemMetrics]:
        """Collect current system metrics."""
        if not self.psutil_available:
            return None
        
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            available_memory_gb = memory.available / (1024**3)
            
            # Disk usage (root partition)
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Network I/O
            network_io = psutil.net_io_counters()
            # Calculate network speed (simplified)
            network_io_mb_s = 0.0  # Would need historical data for accurate calculation
            
            # Load average (Unix-like systems)
            load_average = None
            try:
                load_average = list(psutil.getloadavg())
            except (AttributeError, OSError):
                pass  # Not available on all systems
            
            # Process count
            process_count = len(psutil.pids())
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                network_io_mb_s=network_io_mb_s,
                load_average=load_average,
                process_count=process_count,
                available_memory_gb=available_memory_gb
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return None
    
    def _handle_state_change(self, old_state: SystemState, new_state: SystemState) -> None:
        """Handle system state changes."""
        self.logger.info(f"System state changed: {old_state.value} -> {new_state.value}")
        
        # Update statistics
        self.stats['last_state_change'] = datetime.now()
        
        # Log state change details
        if self.current_metrics:
            self.logger.info(
                f"System metrics: CPU={self.current_metrics.cpu_percent:.1f}%, "
                f"Memory={self.current_metrics.memory_percent:.1f}%, "
                f"Disk={self.current_metrics.disk_percent:.1f}%"
            )
    
    def _check_alerts(self) -> None:
        """Check if alerts should be triggered."""
        if self.current_state in self.alert_callbacks:
            callbacks = self.alert_callbacks[self.current_state]
            if callbacks:
                self.stats['alerts_triggered'][self.current_state.value] += 1
                
                for callback in callbacks:
                    try:
                        callback(self.current_state, self.current_metrics)
                    except Exception as e:
                        self.logger.error(f"Error in alert callback: {e}")
    
    def register_alert_callback(self, state: SystemState, callback: Callable) -> None:
        """
        Register a callback for system state alerts.
        
        Args:
            state: System state to monitor
            callback: Function to call when state is reached
        """
        self.alert_callbacks[state].append(callback)
        self.logger.info(f"Alert callback registered for state: {state.value}")
    
    def is_suitable_for_processing(self) -> bool:
        """
        Check if system is suitable for autonomous processing.
        
        Returns:
            True if system can handle autonomous processing, False otherwise
        """
        if not self.current_metrics:
            return False
        
        # Only allow processing in optimal or moderate states
        return self.current_state in [SystemState.OPTIMAL, SystemState.MODERATE]
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current system status."""
        if not self.current_metrics:
            return {
                'state': SystemState.UNAVAILABLE.value,
                'monitoring': self._monitoring,
                'psutil_available': self.psutil_available,
                'suitable_for_processing': False
            }
        
        return {
            'state': self.current_state.value,
            'monitoring': self._monitoring,
            'psutil_available': self.psutil_available,
            'suitable_for_processing': self.is_suitable_for_processing(),
            'metrics': {
                'cpu_percent': self.current_metrics.cpu_percent,
                'memory_percent': self.current_metrics.memory_percent,
                'disk_percent': self.current_metrics.disk_percent,
                'available_memory_gb': self.current_metrics.available_memory_gb,
                'process_count': self.current_metrics.process_count,
                'timestamp': self.current_metrics.timestamp.isoformat()
            },
            'thresholds': {
                'cpu_critical': self.thresholds.cpu_critical,
                'memory_critical': self.thresholds.memory_critical,
                'disk_critical': self.thresholds.disk_critical
            }
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        stats = self.stats.copy()
        
        # Add current uptime
        if stats['monitoring_start_time']:
            uptime = (datetime.now() - stats['monitoring_start_time']).total_seconds()
            stats['uptime_seconds'] = uptime
        
        return stats
    
    def get_metrics_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get historical metrics data.
        
        Args:
            limit: Maximum number of metrics to return
            
        Returns:
            List of historical metrics
        """
        history = self.metrics_history
        if limit:
            history = history[-limit:]
        
        return [
            {
                'timestamp': m.timestamp.isoformat(),
                'cpu_percent': m.cpu_percent,
                'memory_percent': m.memory_percent,
                'disk_percent': m.disk_percent,
                'state': m.get_overall_state(self.thresholds).value
            }
            for m in history
        ]
    
    def update_thresholds(self, new_thresholds: Dict[str, float]) -> None:
        """
        Update system thresholds.
        
        Args:
            new_thresholds: Dictionary with new threshold values
        """
        for key, value in new_thresholds.items():
            if hasattr(self.thresholds, key):
                setattr(self.thresholds, key, value)
                self.logger.info(f"Updated threshold {key} = {value}")
            else:
                self.logger.warning(f"Unknown threshold key: {key}")


def create_system_monitor(thresholds: Optional[Dict[str, float]] = None,
                         monitoring_interval: float = 5.0,
                         history_size: int = 100) -> SystemLoadMonitor:
    """
    Factory function to create a SystemLoadMonitor.
    
    Args:
        thresholds: Optional threshold configuration
        monitoring_interval: Seconds between monitoring checks
        history_size: Number of historical metrics to keep
        
    Returns:
        Configured SystemLoadMonitor instance
    """
    # Create thresholds object
    if thresholds:
        threshold_obj = SystemThresholds(**thresholds)
    else:
        threshold_obj = SystemThresholds()
    
    # Create monitor
    monitor = SystemLoadMonitor(
        thresholds=threshold_obj,
        monitoring_interval=monitoring_interval,
        history_size=history_size
    )
    
    return monitor
