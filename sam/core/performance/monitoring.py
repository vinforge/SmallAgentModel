#!/usr/bin/env python3
"""
SAM Performance Monitoring Framework
====================================

Comprehensive performance monitoring system for SAM applications to track
performance metrics, identify bottlenecks, and provide optimization insights.

This module provides:
- Real-time performance metrics
- Function execution timing
- Resource usage tracking
- Performance alerts and recommendations
- Historical performance analysis

Author: SAM Development Team
Version: 1.0.0 - Performance Optimization Framework
"""

import streamlit as st
import logging
import time
import threading
import psutil
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from functools import wraps
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Represents a performance metric measurement."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    category: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FunctionProfile:
    """Represents performance profile of a function."""
    name: str
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    last_call: Optional[datetime] = None
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add_execution(self, execution_time: float):
        """Add an execution time measurement."""
        self.call_count += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.avg_time = self.total_time / self.call_count
        self.last_call = datetime.now()
        self.recent_times.append(execution_time)
    
    def get_recent_avg(self) -> float:
        """Get average of recent execution times."""
        if not self.recent_times:
            return 0.0
        return statistics.mean(self.recent_times)
    
    def get_percentile(self, percentile: float) -> float:
        """Get percentile of recent execution times."""
        if not self.recent_times:
            return 0.0
        
        sorted_times = sorted(self.recent_times)
        index = int(len(sorted_times) * percentile / 100)
        return sorted_times[min(index, len(sorted_times) - 1)]


class PerformanceMonitor:
    """Monitors and tracks performance metrics."""
    
    def __init__(self, max_metrics: int = 1000):
        self.metrics: List[PerformanceMetric] = []
        self.max_metrics = max_metrics
        self.function_profiles: Dict[str, FunctionProfile] = {}
        self.alerts: List[str] = []
        self.monitoring_active = True
        self.lock = threading.RLock()
        
        # Performance thresholds
        self.slow_function_threshold = 1.0  # seconds
        self.memory_threshold = 80.0  # percent
        self.cpu_threshold = 80.0  # percent
        
        # Start system monitoring
        self._start_system_monitoring()
    
    def _start_system_monitoring(self):
        """Start system resource monitoring."""
        def monitor_system():
            while self.monitoring_active:
                try:
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.record_metric("cpu_usage", cpu_percent, "percent", "system")
                    
                    # Memory usage
                    memory = psutil.virtual_memory()
                    self.record_metric("memory_usage", memory.percent, "percent", "system")
                    self.record_metric("memory_available", memory.available / (1024**3), "GB", "system")
                    
                    # Disk usage
                    disk = psutil.disk_usage('/')
                    self.record_metric("disk_usage", disk.percent, "percent", "system")
                    
                    # Check thresholds
                    if cpu_percent > self.cpu_threshold:
                        self._add_alert(f"High CPU usage: {cpu_percent:.1f}%")
                    
                    if memory.percent > self.memory_threshold:
                        self._add_alert(f"High memory usage: {memory.percent:.1f}%")
                    
                    time.sleep(30)  # Monitor every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Error in system monitoring: {e}")
                    time.sleep(30)
        
        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()
    
    def record_metric(self, name: str, value: float, unit: str, 
                     category: str = "general", metadata: Dict[str, Any] = None):
        """Record a performance metric."""
        with self.lock:
            metric = PerformanceMetric(
                name=name,
                value=value,
                unit=unit,
                timestamp=datetime.now(),
                category=category,
                metadata=metadata or {}
            )
            
            self.metrics.append(metric)
            
            # Trim old metrics
            if len(self.metrics) > self.max_metrics:
                self.metrics = self.metrics[-self.max_metrics:]
    
    def profile_function(self, func_name: str, execution_time: float):
        """Profile a function execution."""
        with self.lock:
            if func_name not in self.function_profiles:
                self.function_profiles[func_name] = FunctionProfile(func_name)
            
            profile = self.function_profiles[func_name]
            profile.add_execution(execution_time)
            
            # Check for slow functions
            if execution_time > self.slow_function_threshold:
                self._add_alert(f"Slow function: {func_name} took {execution_time:.2f}s")
    
    def _add_alert(self, message: str):
        """Add a performance alert."""
        with self.lock:
            if message not in self.alerts:
                self.alerts.append(message)
                logger.warning(f"Performance alert: {message}")
                
                # Keep only recent alerts
                if len(self.alerts) > 50:
                    self.alerts = self.alerts[-50:]
    
    def get_metrics(self, category: str = None, 
                   since: Optional[datetime] = None) -> List[PerformanceMetric]:
        """Get performance metrics with optional filtering."""
        with self.lock:
            filtered_metrics = self.metrics
            
            if category:
                filtered_metrics = [m for m in filtered_metrics if m.category == category]
            
            if since:
                filtered_metrics = [m for m in filtered_metrics if m.timestamp >= since]
            
            return filtered_metrics.copy()
    
    def get_function_profiles(self) -> Dict[str, FunctionProfile]:
        """Get all function profiles."""
        with self.lock:
            return self.function_profiles.copy()
    
    def get_alerts(self) -> List[str]:
        """Get current performance alerts."""
        with self.lock:
            return self.alerts.copy()
    
    def clear_alerts(self):
        """Clear all performance alerts."""
        with self.lock:
            self.alerts.clear()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics."""
        with self.lock:
            # Recent metrics (last hour)
            recent_cutoff = datetime.now() - timedelta(hours=1)
            recent_metrics = self.get_metrics(since=recent_cutoff)
            
            # Group by category
            by_category = defaultdict(list)
            for metric in recent_metrics:
                by_category[metric.category].append(metric.value)
            
            # Calculate averages
            category_averages = {}
            for category, values in by_category.items():
                if values:
                    category_averages[category] = {
                        'avg': statistics.mean(values),
                        'min': min(values),
                        'max': max(values),
                        'count': len(values)
                    }
            
            # Function performance
            slow_functions = []
            for name, profile in self.function_profiles.items():
                if profile.get_recent_avg() > self.slow_function_threshold:
                    slow_functions.append({
                        'name': name,
                        'avg_time': profile.get_recent_avg(),
                        'call_count': profile.call_count
                    })
            
            return {
                'total_metrics': len(self.metrics),
                'recent_metrics': len(recent_metrics),
                'category_averages': category_averages,
                'function_profiles': len(self.function_profiles),
                'slow_functions': slow_functions,
                'alerts': len(self.alerts)
            }


def performance_timer(category: str = "function", monitor: Optional[PerformanceMonitor] = None):
    """Decorator to time function execution."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = time.time() - start_time
                
                # Record metrics
                perf_monitor = monitor or get_performance_monitor()
                perf_monitor.profile_function(func.__name__, execution_time)
                perf_monitor.record_metric(
                    f"{func.__name__}_execution_time",
                    execution_time,
                    "seconds",
                    category
                )
        
        return wrapper
    return decorator


class PerformanceBenchmark:
    """Benchmarking utilities for performance testing."""
    
    @staticmethod
    def benchmark_function(func: Callable, iterations: int = 100, 
                          *args, **kwargs) -> Dict[str, float]:
        """Benchmark a function over multiple iterations."""
        times = []
        
        for _ in range(iterations):
            start_time = time.time()
            func(*args, **kwargs)
            execution_time = time.time() - start_time
            times.append(execution_time)
        
        return {
            'min_time': min(times),
            'max_time': max(times),
            'avg_time': statistics.mean(times),
            'median_time': statistics.median(times),
            'std_dev': statistics.stdev(times) if len(times) > 1 else 0,
            'total_time': sum(times),
            'iterations': iterations
        }
    
    @staticmethod
    def compare_functions(functions: Dict[str, Callable], iterations: int = 100,
                         *args, **kwargs) -> Dict[str, Dict[str, float]]:
        """Compare performance of multiple functions."""
        results = {}
        
        for name, func in functions.items():
            results[name] = PerformanceBenchmark.benchmark_function(
                func, iterations, *args, **kwargs
            )
        
        return results


# Global performance monitor
_global_monitor = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def render_performance_dashboard():
    """Render a comprehensive performance dashboard."""
    st.subheader("📊 Performance Monitoring Dashboard")
    
    monitor = get_performance_monitor()
    
    # Performance summary
    summary = monitor.get_performance_summary()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Metrics", summary['total_metrics'])
    
    with col2:
        st.metric("Recent Metrics", summary['recent_metrics'])
    
    with col3:
        st.metric("Function Profiles", summary['function_profiles'])
    
    with col4:
        st.metric("Active Alerts", summary['alerts'])
    
    # System metrics
    st.markdown("### 🖥️ System Performance")
    
    recent_metrics = monitor.get_metrics(category="system", 
                                       since=datetime.now() - timedelta(minutes=30))
    
    if recent_metrics:
        import pandas as pd
        import plotly.express as px
        
        # Group metrics by name
        metric_data = defaultdict(list)
        for metric in recent_metrics:
            metric_data[metric.name].append({
                'timestamp': metric.timestamp,
                'value': metric.value
            })
        
        # Plot system metrics
        for metric_name, data in metric_data.items():
            if data:
                df = pd.DataFrame(data)
                fig = px.line(df, x='timestamp', y='value', 
                             title=f"{metric_name.replace('_', ' ').title()}")
                st.plotly_chart(fig, use_container_width=True)
    
    # Function performance
    st.markdown("### ⚡ Function Performance")
    
    profiles = monitor.get_function_profiles()
    if profiles:
        # Sort by recent average time
        sorted_profiles = sorted(profiles.items(), 
                               key=lambda x: x[1].get_recent_avg(), 
                               reverse=True)
        
        for name, profile in sorted_profiles[:10]:  # Top 10 slowest
            with st.expander(f"🔧 {name}", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Calls", profile.call_count)
                    st.metric("Avg Time", f"{profile.avg_time:.3f}s")
                
                with col2:
                    st.metric("Min Time", f"{profile.min_time:.3f}s")
                    st.metric("Max Time", f"{profile.max_time:.3f}s")
                
                with col3:
                    st.metric("Recent Avg", f"{profile.get_recent_avg():.3f}s")
                    st.metric("95th Percentile", f"{profile.get_percentile(95):.3f}s")
    
    # Performance alerts
    alerts = monitor.get_alerts()
    if alerts:
        st.markdown("### ⚠️ Performance Alerts")
        
        for alert in alerts[-10:]:  # Show last 10 alerts
            st.warning(alert)
        
        if st.button("🗑️ Clear Alerts"):
            monitor.clear_alerts()
            st.rerun()
    
    # Slow functions
    slow_functions = summary.get('slow_functions', [])
    if slow_functions:
        st.markdown("### 🐌 Slow Functions")
        
        for func in slow_functions:
            st.error(f"**{func['name']}**: {func['avg_time']:.3f}s avg "
                    f"({func['call_count']} calls)")


def benchmark_ui_component(component_func: Callable, iterations: int = 10):
    """Benchmark a UI component rendering."""
    st.markdown("### 🏃 Component Benchmark")
    
    if st.button("Run Benchmark"):
        with st.spinner(f"Running {iterations} iterations..."):
            results = PerformanceBenchmark.benchmark_function(
                component_func, iterations
            )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Time", f"{results['avg_time']:.3f}s")
            st.metric("Min Time", f"{results['min_time']:.3f}s")
        
        with col2:
            st.metric("Max Time", f"{results['max_time']:.3f}s")
            st.metric("Median Time", f"{results['median_time']:.3f}s")
        
        with col3:
            st.metric("Total Time", f"{results['total_time']:.3f}s")
            st.metric("Std Dev", f"{results['std_dev']:.3f}s")
