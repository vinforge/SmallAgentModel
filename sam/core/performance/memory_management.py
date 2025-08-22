#!/usr/bin/env python3
"""
SAM Memory Management Framework
==============================

Intelligent memory management system for SAM applications to optimize
memory usage, prevent memory leaks, and improve overall performance.

This module provides:
- Memory usage monitoring
- Automatic garbage collection
- Memory leak detection
- Resource cleanup strategies
- Memory optimization recommendations

Author: SAM Development Team
Version: 1.0.0 - Performance Optimization Framework
"""

import streamlit as st
import logging
import gc
import psutil
import threading
import time
import weakref
from typing import Dict, Any, Optional, List, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
import sys

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Represents a memory usage snapshot."""
    timestamp: datetime
    total_memory_mb: float
    available_memory_mb: float
    used_memory_mb: float
    memory_percent: float
    python_memory_mb: float
    gc_objects: int
    gc_collections: Dict[int, int]


class MemoryMonitor:
    """Monitors memory usage and provides optimization recommendations."""
    
    def __init__(self, snapshot_interval: int = 30):
        self.snapshot_interval = snapshot_interval
        self.snapshots: List[MemorySnapshot] = []
        self.max_snapshots = 100
        self.monitoring = False
        self.monitor_thread = None
        self.callbacks: List[Callable[[MemorySnapshot], None]] = []
        
        # Memory thresholds
        self.warning_threshold = 80.0  # Percent
        self.critical_threshold = 90.0  # Percent
        
        # Start monitoring
        self.start_monitoring()
    
    def start_monitoring(self):
        """Start memory monitoring."""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                snapshot = self._take_snapshot()
                self._process_snapshot(snapshot)
                time.sleep(self.snapshot_interval)
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                time.sleep(self.snapshot_interval)
    
    def _take_snapshot(self) -> MemorySnapshot:
        """Take a memory usage snapshot."""
        # System memory
        memory = psutil.virtual_memory()
        
        # Python process memory
        process = psutil.Process()
        python_memory_mb = process.memory_info().rss / (1024 * 1024)
        
        # Garbage collection info
        gc_stats = gc.get_stats()
        gc_collections = {i: stat['collections'] for i, stat in enumerate(gc_stats)}
        
        return MemorySnapshot(
            timestamp=datetime.now(),
            total_memory_mb=memory.total / (1024 * 1024),
            available_memory_mb=memory.available / (1024 * 1024),
            used_memory_mb=memory.used / (1024 * 1024),
            memory_percent=memory.percent,
            python_memory_mb=python_memory_mb,
            gc_objects=len(gc.get_objects()),
            gc_collections=gc_collections
        )
    
    def _process_snapshot(self, snapshot: MemorySnapshot):
        """Process a memory snapshot."""
        # Add to snapshots list
        self.snapshots.append(snapshot)
        
        # Trim old snapshots
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots = self.snapshots[-self.max_snapshots:]
        
        # Check thresholds
        if snapshot.memory_percent >= self.critical_threshold:
            logger.critical(f"Critical memory usage: {snapshot.memory_percent:.1f}%")
            self._trigger_emergency_cleanup()
        elif snapshot.memory_percent >= self.warning_threshold:
            logger.warning(f"High memory usage: {snapshot.memory_percent:.1f}%")
        
        # Notify callbacks
        for callback in self.callbacks:
            try:
                callback(snapshot)
            except Exception as e:
                logger.error(f"Error in memory callback: {e}")
    
    def _trigger_emergency_cleanup(self):
        """Trigger emergency memory cleanup."""
        logger.info("Triggering emergency memory cleanup")
        
        # Force garbage collection
        collected = gc.collect()
        logger.info(f"Garbage collection freed {collected} objects")
        
        # Clear caches if available
        try:
            from sam.core.performance.caching import _global_caches
            for cache in _global_caches.values():
                cache.clear()
            logger.info("Cleared all caches")
        except ImportError:
            pass
    
    def add_callback(self, callback: Callable[[MemorySnapshot], None]):
        """Add a memory monitoring callback."""
        self.callbacks.append(callback)
    
    def get_current_usage(self) -> Optional[MemorySnapshot]:
        """Get current memory usage."""
        if self.snapshots:
            return self.snapshots[-1]
        return self._take_snapshot()
    
    def get_usage_trend(self, minutes: int = 10) -> List[MemorySnapshot]:
        """Get memory usage trend for the last N minutes."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [s for s in self.snapshots if s.timestamp >= cutoff]
    
    def get_recommendations(self) -> List[str]:
        """Get memory optimization recommendations."""
        recommendations = []
        
        if not self.snapshots:
            return recommendations
        
        current = self.snapshots[-1]
        
        # High memory usage
        if current.memory_percent > self.warning_threshold:
            recommendations.append(
                f"Memory usage is high ({current.memory_percent:.1f}%). "
                "Consider clearing caches or reducing data in memory."
            )
        
        # Growing Python memory
        if len(self.snapshots) >= 5:
            recent_python_memory = [s.python_memory_mb for s in self.snapshots[-5:]]
            if recent_python_memory[-1] > recent_python_memory[0] * 1.5:
                recommendations.append(
                    "Python memory usage is growing rapidly. "
                    "Check for memory leaks or large data structures."
                )
        
        # High GC object count
        if current.gc_objects > 100000:
            recommendations.append(
                f"High number of Python objects ({current.gc_objects:,}). "
                "Consider object pooling or data structure optimization."
            )
        
        return recommendations


class ResourceTracker:
    """Tracks resource usage and helps prevent memory leaks."""
    
    def __init__(self):
        self.tracked_objects: Dict[str, Set[weakref.ref]] = defaultdict(set)
        self.creation_counts: Dict[str, int] = defaultdict(int)
        self.cleanup_callbacks: Dict[str, List[Callable]] = defaultdict(list)
    
    def track_object(self, obj: Any, category: str = "general"):
        """Track an object for memory leak detection."""
        def cleanup_callback(ref):
            self.tracked_objects[category].discard(ref)
        
        ref = weakref.ref(obj, cleanup_callback)
        self.tracked_objects[category].add(ref)
        self.creation_counts[category] += 1
    
    def register_cleanup_callback(self, category: str, callback: Callable):
        """Register a cleanup callback for a category."""
        self.cleanup_callbacks[category].append(callback)
    
    def cleanup_category(self, category: str):
        """Cleanup all objects in a category."""
        # Call cleanup callbacks
        for callback in self.cleanup_callbacks[category]:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in cleanup callback for {category}: {e}")
        
        # Clear tracked objects
        self.tracked_objects[category].clear()
        logger.info(f"Cleaned up category: {category}")
    
    def get_live_objects(self, category: str = None) -> Dict[str, int]:
        """Get count of live objects by category."""
        if category:
            # Clean up dead references
            live_refs = {ref for ref in self.tracked_objects[category] if ref() is not None}
            self.tracked_objects[category] = live_refs
            return {category: len(live_refs)}
        
        result = {}
        for cat, refs in self.tracked_objects.items():
            # Clean up dead references
            live_refs = {ref for ref in refs if ref() is not None}
            self.tracked_objects[cat] = live_refs
            result[cat] = len(live_refs)
        
        return result
    
    def detect_potential_leaks(self) -> List[str]:
        """Detect potential memory leaks."""
        leaks = []
        live_objects = self.get_live_objects()
        
        for category, count in live_objects.items():
            creation_count = self.creation_counts[category]
            
            # If more than 50% of created objects are still alive, potential leak
            if creation_count > 10 and count > creation_count * 0.5:
                leaks.append(
                    f"Potential leak in '{category}': {count}/{creation_count} objects still alive"
                )
        
        return leaks


class MemoryOptimizer:
    """Provides memory optimization utilities."""
    
    @staticmethod
    def optimize_dataframe(df) -> Any:
        """Optimize pandas DataFrame memory usage."""
        try:
            import pandas as pd
            
            if not isinstance(df, pd.DataFrame):
                return df
            
            # Optimize numeric columns
            for col in df.select_dtypes(include=['int64']).columns:
                df[col] = pd.to_numeric(df[col], downcast='integer')
            
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = pd.to_numeric(df[col], downcast='float')
            
            # Optimize object columns
            for col in df.select_dtypes(include=['object']).columns:
                if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                    df[col] = df[col].astype('category')
            
            return df
            
        except ImportError:
            return df
    
    @staticmethod
    def clear_streamlit_cache():
        """Clear Streamlit cache."""
        try:
            st.cache_data.clear()
            st.cache_resource.clear()
            logger.info("Cleared Streamlit cache")
        except Exception as e:
            logger.error(f"Error clearing Streamlit cache: {e}")
    
    @staticmethod
    def force_garbage_collection() -> int:
        """Force garbage collection and return number of collected objects."""
        collected = 0
        for generation in range(3):
            collected += gc.collect(generation)
        
        logger.info(f"Garbage collection freed {collected} objects")
        return collected
    
    @staticmethod
    def get_memory_usage_by_type() -> Dict[str, int]:
        """Get memory usage breakdown by object type."""
        type_counts = defaultdict(int)
        
        for obj in gc.get_objects():
            type_counts[type(obj).__name__] += 1
        
        # Sort by count
        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_types[:20])  # Top 20 types


# Global instances
_memory_monitor = None
_resource_tracker = ResourceTracker()


def get_memory_monitor() -> MemoryMonitor:
    """Get the global memory monitor."""
    global _memory_monitor
    if _memory_monitor is None:
        _memory_monitor = MemoryMonitor()
    return _memory_monitor


def get_resource_tracker() -> ResourceTracker:
    """Get the global resource tracker."""
    return _resource_tracker


def track_resource(obj: Any, category: str = "general"):
    """Track a resource for memory leak detection."""
    get_resource_tracker().track_object(obj, category)


def render_memory_dashboard():
    """Render a comprehensive memory management dashboard."""
    st.subheader("🧠 Memory Management Dashboard")
    
    monitor = get_memory_monitor()
    tracker = get_resource_tracker()
    
    # Current memory usage
    current = monitor.get_current_usage()
    if current:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("System Memory", f"{current.memory_percent:.1f}%")
        
        with col2:
            st.metric("Python Memory", f"{current.python_memory_mb:.1f} MB")
        
        with col3:
            st.metric("GC Objects", f"{current.gc_objects:,}")
        
        with col4:
            st.metric("Available", f"{current.available_memory_mb:.0f} MB")
    
    # Memory trend
    trend = monitor.get_usage_trend(minutes=30)
    if trend:
        st.markdown("### 📈 Memory Usage Trend (30 minutes)")
        
        import pandas as pd
        import plotly.express as px
        
        df = pd.DataFrame([
            {
                'Time': s.timestamp,
                'System Memory %': s.memory_percent,
                'Python Memory MB': s.python_memory_mb
            }
            for s in trend
        ])
        
        fig = px.line(df, x='Time', y=['System Memory %', 'Python Memory MB'],
                     title="Memory Usage Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    # Resource tracking
    st.markdown("### 🔍 Resource Tracking")
    live_objects = tracker.get_live_objects()
    
    if live_objects:
        for category, count in live_objects.items():
            st.write(f"**{category}**: {count} live objects")
    
    # Memory leaks
    leaks = tracker.detect_potential_leaks()
    if leaks:
        st.markdown("### ⚠️ Potential Memory Leaks")
        for leak in leaks:
            st.warning(leak)
    
    # Recommendations
    recommendations = monitor.get_recommendations()
    if recommendations:
        st.markdown("### 💡 Optimization Recommendations")
        for rec in recommendations:
            st.info(rec)
    
    # Memory management actions
    st.markdown("### 🔧 Memory Management Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Force GC"):
            collected = MemoryOptimizer.force_garbage_collection()
            st.success(f"Collected {collected} objects")
            st.rerun()
    
    with col2:
        if st.button("🧹 Clear Caches"):
            MemoryOptimizer.clear_streamlit_cache()
            st.success("Caches cleared")
            st.rerun()
    
    with col3:
        if st.button("📊 Object Types"):
            types = MemoryOptimizer.get_memory_usage_by_type()
            st.json(types)
