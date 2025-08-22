"""
Performance Monitor for SAM
===========================

Real-time performance monitoring and metrics collection for SAM's operations.
Tracks system resources, operation timings, and performance trends.

Author: SAM Development Team
Version: 1.0.0
"""

import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import statistics


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""
    timestamp: datetime
    
    # System metrics
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    
    # SAM-specific metrics
    active_operations: int
    avg_operation_time_ms: float
    operations_per_minute: float
    error_rate: float
    
    # Model metrics
    model_memory_mb: Optional[float] = None
    inference_time_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None
    
    # Custom metrics
    custom_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.custom_metrics is None:
            self.custom_metrics = {}


class PerformanceMonitor:
    """
    Real-time performance monitoring system for SAM.
    
    Features:
    - System resource monitoring
    - Operation timing and throughput
    - Performance trend analysis
    - Alert system for performance issues
    - Metrics export and visualization
    """
    
    def __init__(self, 
                 collection_interval: float = 5.0,
                 history_size: int = 1000,
                 enable_alerts: bool = True):
        """
        Initialize the performance monitor.
        
        Args:
            collection_interval: Seconds between metric collections
            history_size: Number of metric snapshots to keep in memory
            enable_alerts: Whether to enable performance alerts
        """
        self.collection_interval = collection_interval
        self.history_size = history_size
        self.enable_alerts = enable_alerts
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=history_size)
        self.operation_timings: defaultdict = defaultdict(list)
        self.operation_counts: defaultdict = defaultdict(int)
        self.error_counts: defaultdict = defaultdict(int)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Alert thresholds
        self.alert_thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "avg_operation_time_ms": 5000.0,
            "error_rate": 0.1
        }
        
        # Alert callbacks
        self.alert_callbacks: List[Callable] = []
        
        # Custom metric collectors
        self.custom_collectors: Dict[str, Callable] = {}
    
    def start_monitoring(self):
        """Start the performance monitoring thread."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        print("🔍 Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop the performance monitoring thread."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=self.collection_interval + 1)
        print("🔍 Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                
                with self._lock:
                    self.metrics_history.append(metrics)
                
                # Check for alerts
                if self.enable_alerts:
                    self._check_alerts(metrics)
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # SAM operation metrics
        with self._lock:
            # Calculate operation statistics
            all_timings = []
            total_operations = 0
            total_errors = 0
            
            for operation, timings in self.operation_timings.items():
                all_timings.extend(timings)
                total_operations += self.operation_counts[operation]
                total_errors += self.error_counts[operation]
            
            avg_operation_time = statistics.mean(all_timings) if all_timings else 0.0
            error_rate = total_errors / max(total_operations, 1)
            
            # Calculate operations per minute
            now = datetime.now()
            one_minute_ago = now - timedelta(minutes=1)
            recent_operations = sum(
                1 for metrics in self.metrics_history 
                if metrics.timestamp > one_minute_ago
            )
            operations_per_minute = recent_operations * (60.0 / self.collection_interval)
        
        # Collect custom metrics
        custom_metrics = {}
        for name, collector in self.custom_collectors.items():
            try:
                custom_metrics[name] = collector()
            except Exception as e:
                print(f"Error collecting custom metric {name}: {e}")
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            memory_available_mb=memory.available / (1024 * 1024),
            active_operations=len(self.operation_timings),
            avg_operation_time_ms=avg_operation_time,
            operations_per_minute=operations_per_minute,
            error_rate=error_rate,
            custom_metrics=custom_metrics
        )
    
    def record_operation(self, operation_name: str, duration_ms: float, success: bool = True):
        """
        Record an operation timing.
        
        Args:
            operation_name: Name of the operation
            duration_ms: Duration in milliseconds
            success: Whether the operation succeeded
        """
        with self._lock:
            # Keep only recent timings (last 100 per operation)
            if len(self.operation_timings[operation_name]) >= 100:
                self.operation_timings[operation_name].pop(0)
            
            self.operation_timings[operation_name].append(duration_ms)
            self.operation_counts[operation_name] += 1
            
            if not success:
                self.error_counts[operation_name] += 1
    
    def add_custom_metric_collector(self, name: str, collector: Callable[[], float]):
        """
        Add a custom metric collector.
        
        Args:
            name: Name of the metric
            collector: Function that returns the metric value
        """
        self.custom_collectors[name] = collector
    
    def add_alert_callback(self, callback: Callable[[str, float, float], None]):
        """
        Add an alert callback function.
        
        Args:
            callback: Function called when alert is triggered
                     (metric_name, current_value, threshold)
        """
        self.alert_callbacks.append(callback)
    
    def _check_alerts(self, metrics: PerformanceMetrics):
        """Check if any metrics exceed alert thresholds."""
        checks = [
            ("cpu_percent", metrics.cpu_percent),
            ("memory_percent", metrics.memory_percent),
            ("avg_operation_time_ms", metrics.avg_operation_time_ms),
            ("error_rate", metrics.error_rate)
        ]
        
        for metric_name, value in checks:
            threshold = self.alert_thresholds.get(metric_name)
            if threshold and value > threshold:
                self._trigger_alert(metric_name, value, threshold)
    
    def _trigger_alert(self, metric_name: str, value: float, threshold: float):
        """Trigger an alert for a metric threshold breach."""
        for callback in self.alert_callbacks:
            try:
                callback(metric_name, value, threshold)
            except Exception as e:
                print(f"Error in alert callback: {e}")
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the most recent metrics snapshot."""
        with self._lock:
            return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, minutes: int = 60) -> List[PerformanceMetrics]:
        """
        Get metrics history for the specified time period.
        
        Args:
            minutes: Number of minutes of history to return
            
        Returns:
            List of metrics snapshots
        """
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self._lock:
            return [
                metrics for metrics in self.metrics_history 
                if metrics.timestamp > cutoff_time
            ]
    
    def get_performance_summary(self, minutes: int = 60) -> Dict[str, Any]:
        """
        Get a performance summary for the specified time period.
        
        Args:
            minutes: Number of minutes to analyze
            
        Returns:
            Performance summary dictionary
        """
        history = self.get_metrics_history(minutes)
        
        if not history:
            return {"message": "No metrics available"}
        
        # Calculate statistics
        cpu_values = [m.cpu_percent for m in history]
        memory_values = [m.memory_percent for m in history]
        operation_times = [m.avg_operation_time_ms for m in history if m.avg_operation_time_ms > 0]
        
        summary = {
            "time_period_minutes": minutes,
            "data_points": len(history),
            "cpu_usage": {
                "avg": statistics.mean(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values)
            },
            "memory_usage": {
                "avg": statistics.mean(memory_values),
                "max": max(memory_values),
                "min": min(memory_values)
            },
            "current_memory_mb": history[-1].memory_used_mb,
            "total_operations": sum(m.operations_per_minute for m in history),
            "avg_error_rate": statistics.mean([m.error_rate for m in history])
        }
        
        if operation_times:
            summary["operation_performance"] = {
                "avg_time_ms": statistics.mean(operation_times),
                "max_time_ms": max(operation_times),
                "min_time_ms": min(operation_times)
            }
        
        return summary
    
    def export_metrics(self, filename: str, format: str = "json"):
        """
        Export metrics history to file.
        
        Args:
            filename: Output filename
            format: Export format ("json" or "csv")
        """
        with self._lock:
            metrics_data = [asdict(m) for m in self.metrics_history]
        
        if format == "json":
            import json
            with open(filename, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
        
        elif format == "csv":
            import csv
            if metrics_data:
                with open(filename, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=metrics_data[0].keys())
                    writer.writeheader()
                    writer.writerows(metrics_data)
        
        print(f"📊 Exported {len(metrics_data)} metrics to {filename}")
    
    def reset_metrics(self):
        """Reset all collected metrics."""
        with self._lock:
            self.metrics_history.clear()
            self.operation_timings.clear()
            self.operation_counts.clear()
            self.error_counts.clear()
        print("🔄 Performance metrics reset")


# Global monitor instance
_performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor
