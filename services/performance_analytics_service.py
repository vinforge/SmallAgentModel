#!/usr/bin/env python3
"""
Performance Analytics Service
Real-time performance monitoring and analytics for SAM Phase 3.
"""

import logging
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import json
import statistics

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Individual performance metric sample."""
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    context: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class PerformanceAlert:
    """Performance alert when thresholds are exceeded."""
    timestamp: datetime
    metric_name: str
    severity: str  # 'warning', 'critical'
    current_value: float
    threshold: float
    message: str
    context: Dict[str, Any] = field(default_factory=dict)

class MetricCollector:
    """Collects and aggregates performance metrics."""
    
    def __init__(self, max_samples: int = 10000):
        self.max_samples = max_samples
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples))
        self._lock = threading.RLock()
    
    def record(self, metric_name: str, value: float, unit: str = "", context: Dict[str, Any] = None):
        """Record a performance metric."""
        with self._lock:
            metric = PerformanceMetric(
                timestamp=datetime.now(),
                metric_name=metric_name,
                value=value,
                unit=unit,
                context=context or {}
            )
            self.metrics[metric_name].append(metric)
    
    def get_recent_metrics(self, metric_name: str, duration: timedelta) -> List[PerformanceMetric]:
        """Get recent metrics within the specified duration."""
        with self._lock:
            cutoff_time = datetime.now() - duration
            return [m for m in self.metrics[metric_name] if m.timestamp >= cutoff_time]
    
    def get_metric_stats(self, metric_name: str, duration: timedelta) -> Dict[str, float]:
        """Get statistical summary of a metric over the specified duration."""
        recent_metrics = self.get_recent_metrics(metric_name, duration)
        
        if not recent_metrics:
            return {}
        
        values = [m.value for m in recent_metrics]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0,
            'p95': statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
            'p99': statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values)
        }

class PerformanceAnalyticsService:
    """
    Advanced performance analytics service for SAM Phase 3.
    
    Features:
    - Real-time metric collection
    - Performance trend analysis
    - Threshold-based alerting
    - Service performance tracking
    - User experience metrics
    - System health monitoring
    """
    
    def __init__(self):
        self.collector = MetricCollector()
        self.alerts: deque = deque(maxlen=1000)
        self.alert_callbacks: List[Callable] = []
        
        # Performance thresholds
        self.thresholds = {
            'search_response_time_ms': {'warning': 1000, 'critical': 3000},
            'cache_hit_rate_percent': {'warning': 70, 'critical': 50},
            'memory_usage_mb': {'warning': 500, 'critical': 1000},
            'error_rate_percent': {'warning': 5, 'critical': 10},
            'service_availability_percent': {'warning': 95, 'critical': 90}
        }
        
        # Service tracking
        self.service_metrics = {
            'document_search_service': defaultdict(list),
            'result_processor_service': defaultdict(list),
            'context_builder_service': defaultdict(list),
            'search_router': defaultdict(list),
            'slp_fallback_service': defaultdict(list),
            'intelligent_cache_service': defaultdict(list)
        }
        
        self._lock = threading.RLock()
        
        logger.info("PerformanceAnalyticsService initialized")
    
    def record_search_performance(self, query: str, response_time_ms: float, 
                                 result_count: int, cache_hit: bool = False,
                                 service_name: str = "unknown"):
        """Record search performance metrics."""
        context = {
            'query_length': len(query),
            'result_count': result_count,
            'cache_hit': cache_hit,
            'service_name': service_name
        }
        
        self.collector.record('search_response_time_ms', response_time_ms, 'ms', context)
        self.collector.record('search_result_count', result_count, 'count', context)
        
        # Check thresholds
        self._check_threshold('search_response_time_ms', response_time_ms, context)
        
        # Track service-specific metrics
        with self._lock:
            self.service_metrics[service_name]['response_times'].append(response_time_ms)
            self.service_metrics[service_name]['result_counts'].append(result_count)
    
    def record_cache_performance(self, hit_rate: float, total_size_mb: float, 
                                avg_access_time_ms: float):
        """Record cache performance metrics."""
        context = {
            'cache_size_mb': total_size_mb,
            'avg_access_time_ms': avg_access_time_ms
        }
        
        self.collector.record('cache_hit_rate_percent', hit_rate, '%', context)
        self.collector.record('cache_size_mb', total_size_mb, 'MB', context)
        self.collector.record('cache_access_time_ms', avg_access_time_ms, 'ms', context)
        
        # Check thresholds
        self._check_threshold('cache_hit_rate_percent', hit_rate, context)
    
    def record_service_error(self, service_name: str, error_type: str, error_message: str):
        """Record service error for error rate tracking."""
        context = {
            'service_name': service_name,
            'error_type': error_type,
            'error_message': error_message[:200]  # Truncate long messages
        }
        
        self.collector.record('service_error', 1, 'count', context)
        
        # Calculate error rate
        recent_errors = self.get_recent_error_count(service_name, timedelta(minutes=5))
        recent_requests = self.get_recent_request_count(service_name, timedelta(minutes=5))
        
        if recent_requests > 0:
            error_rate = (recent_errors / recent_requests) * 100
            self.collector.record('error_rate_percent', error_rate, '%', context)
            self._check_threshold('error_rate_percent', error_rate, context)
    
    def record_user_experience(self, user_id: str, query: str, satisfaction_score: float,
                              total_interaction_time_ms: float):
        """Record user experience metrics."""
        context = {
            'user_id': user_id,
            'query_type': self._classify_query(query),
            'query_length': len(query)
        }
        
        self.collector.record('user_satisfaction', satisfaction_score, 'score', context)
        self.collector.record('interaction_time_ms', total_interaction_time_ms, 'ms', context)
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data."""
        now = datetime.now()
        last_hour = timedelta(hours=1)
        last_day = timedelta(days=1)
        
        dashboard = {
            'timestamp': now.isoformat(),
            'summary': {
                'search_performance': self._get_search_summary(last_hour),
                'cache_performance': self._get_cache_summary(last_hour),
                'service_health': self._get_service_health_summary(),
                'user_experience': self._get_user_experience_summary(last_hour)
            },
            'trends': {
                'hourly': self._get_hourly_trends(),
                'daily': self._get_daily_trends()
            },
            'alerts': {
                'recent': [self._alert_to_dict(alert) for alert in list(self.alerts)[-10:]],
                'active_count': len([a for a in self.alerts if a.timestamp >= now - timedelta(hours=1)])
            },
            'services': self._get_service_breakdown()
        }
        
        return dashboard
    
    def _get_search_summary(self, duration: timedelta) -> Dict[str, Any]:
        """Get search performance summary."""
        response_time_stats = self.collector.get_metric_stats('search_response_time_ms', duration)
        result_count_stats = self.collector.get_metric_stats('search_result_count', duration)
        
        return {
            'avg_response_time_ms': response_time_stats.get('mean', 0),
            'p95_response_time_ms': response_time_stats.get('p95', 0),
            'total_searches': response_time_stats.get('count', 0),
            'avg_result_count': result_count_stats.get('mean', 0)
        }
    
    def _get_cache_summary(self, duration: timedelta) -> Dict[str, Any]:
        """Get cache performance summary."""
        hit_rate_stats = self.collector.get_metric_stats('cache_hit_rate_percent', duration)
        size_stats = self.collector.get_metric_stats('cache_size_mb', duration)
        
        return {
            'avg_hit_rate': hit_rate_stats.get('mean', 0),
            'current_size_mb': size_stats.get('max', 0),
            'cache_efficiency': 'excellent' if hit_rate_stats.get('mean', 0) > 80 else 'good' if hit_rate_stats.get('mean', 0) > 60 else 'needs_improvement'
        }
    
    def _get_service_health_summary(self) -> Dict[str, Any]:
        """Get service health summary."""
        healthy_services = 0
        total_services = len(self.service_metrics)
        
        for service_name, metrics in self.service_metrics.items():
            if metrics.get('response_times'):
                recent_times = metrics['response_times'][-10:]  # Last 10 requests
                avg_time = sum(recent_times) / len(recent_times)
                if avg_time < 1000:  # Under 1 second
                    healthy_services += 1
        
        health_percentage = (healthy_services / total_services * 100) if total_services > 0 else 100
        
        return {
            'healthy_services': healthy_services,
            'total_services': total_services,
            'health_percentage': health_percentage,
            'status': 'healthy' if health_percentage > 90 else 'warning' if health_percentage > 70 else 'critical'
        }
    
    def _get_user_experience_summary(self, duration: timedelta) -> Dict[str, Any]:
        """Get user experience summary."""
        satisfaction_stats = self.collector.get_metric_stats('user_satisfaction', duration)
        interaction_stats = self.collector.get_metric_stats('interaction_time_ms', duration)
        
        return {
            'avg_satisfaction': satisfaction_stats.get('mean', 0),
            'avg_interaction_time_ms': interaction_stats.get('mean', 0),
            'total_interactions': satisfaction_stats.get('count', 0)
        }
    
    def _get_service_breakdown(self) -> Dict[str, Any]:
        """Get per-service performance breakdown."""
        breakdown = {}
        
        for service_name, metrics in self.service_metrics.items():
            if metrics.get('response_times'):
                recent_times = metrics['response_times'][-100:]  # Last 100 requests
                breakdown[service_name] = {
                    'avg_response_time_ms': sum(recent_times) / len(recent_times),
                    'request_count': len(recent_times),
                    'status': 'healthy' if sum(recent_times) / len(recent_times) < 1000 else 'slow'
                }
        
        return breakdown
    
    def _check_threshold(self, metric_name: str, value: float, context: Dict[str, Any]):
        """Check if metric value exceeds thresholds and generate alerts."""
        if metric_name not in self.thresholds:
            return
        
        thresholds = self.thresholds[metric_name]
        
        # Determine severity
        severity = None
        threshold_value = None
        
        if value >= thresholds.get('critical', float('inf')):
            severity = 'critical'
            threshold_value = thresholds['critical']
        elif value >= thresholds.get('warning', float('inf')):
            severity = 'warning'
            threshold_value = thresholds['warning']
        elif metric_name == 'cache_hit_rate_percent' and value <= thresholds.get('critical', 0):
            severity = 'critical'
            threshold_value = thresholds['critical']
        elif metric_name == 'cache_hit_rate_percent' and value <= thresholds.get('warning', 0):
            severity = 'warning'
            threshold_value = thresholds['warning']
        
        if severity:
            alert = PerformanceAlert(
                timestamp=datetime.now(),
                metric_name=metric_name,
                severity=severity,
                current_value=value,
                threshold=threshold_value,
                message=f"{metric_name} {severity}: {value} (threshold: {threshold_value})",
                context=context
            )
            
            self.alerts.append(alert)
            
            # Notify callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
    
    def _classify_query(self, query: str) -> str:
        """Classify query type for analytics."""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ['document', 'file', 'upload', 'sam story']):
            return 'document'
        elif any(term in query_lower for term in ['conversation', 'discuss', 'talk']):
            return 'conversation'
        elif any(term in query_lower for term in ['correct', 'wrong', 'fix']):
            return 'correction'
        elif any(term in query_lower for term in ['explain', 'what is', 'how']):
            return 'knowledge'
        else:
            return 'general'
    
    def _alert_to_dict(self, alert: PerformanceAlert) -> Dict[str, Any]:
        """Convert alert to dictionary for JSON serialization."""
        return {
            'timestamp': alert.timestamp.isoformat(),
            'metric_name': alert.metric_name,
            'severity': alert.severity,
            'current_value': alert.current_value,
            'threshold': alert.threshold,
            'message': alert.message,
            'context': alert.context
        }
    
    def get_recent_error_count(self, service_name: str, duration: timedelta) -> int:
        """Get recent error count for a service."""
        recent_errors = self.collector.get_recent_metrics('service_error', duration)
        return len([e for e in recent_errors if e.context.get('service_name') == service_name])
    
    def get_recent_request_count(self, service_name: str, duration: timedelta) -> int:
        """Get recent request count for a service."""
        with self._lock:
            return len(self.service_metrics[service_name].get('response_times', []))
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add callback for performance alerts."""
        self.alert_callbacks.append(callback)
    
    def _get_hourly_trends(self) -> Dict[str, List[float]]:
        """Get hourly performance trends."""
        # Simplified implementation - would need more sophisticated time bucketing
        return {
            'response_times': [],
            'cache_hit_rates': [],
            'error_rates': []
        }
    
    def _get_daily_trends(self) -> Dict[str, List[float]]:
        """Get daily performance trends."""
        # Simplified implementation - would need more sophisticated time bucketing
        return {
            'avg_response_times': [],
            'total_searches': [],
            'user_satisfaction': []
        }

# Global instance for easy access
_performance_analytics_service = None

def get_performance_analytics_service() -> PerformanceAnalyticsService:
    """Get or create the global performance analytics service instance."""
    global _performance_analytics_service
    if _performance_analytics_service is None:
        _performance_analytics_service = PerformanceAnalyticsService()
    return _performance_analytics_service
