"""
DPO Performance Monitor

Tracks and analyzes personalization effectiveness and system performance
for the SAM Personalized Tuner system.

Author: SAM Development Team
Version: 1.0.0
"""

import time
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
import sqlite3
import threading

logger = logging.getLogger(__name__)


@dataclass
class PersonalizationMetric:
    """Represents a personalization performance metric."""
    timestamp: datetime
    user_id: str
    metric_type: str  # 'response_quality', 'user_satisfaction', 'inference_time', etc.
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceSnapshot:
    """Snapshot of system performance at a point in time."""
    timestamp: datetime
    total_users: int
    active_models: int
    total_requests: int
    personalized_requests: int
    average_inference_time: float
    cache_hit_rate: float
    error_rate: float
    user_satisfaction_avg: float
    personalization_effectiveness: float


class DPOPerformanceMonitor:
    """
    Monitor and analyze personalization performance.
    
    Features:
    - Real-time metrics collection
    - Performance trend analysis
    - User satisfaction tracking
    - System health monitoring
    - Automated reporting
    """
    
    def __init__(self, db_path: str = "./logs/performance_metrics.db"):
        """
        Initialize the performance monitor.
        
        Args:
            db_path: Path to the metrics database
        """
        self.logger = logging.getLogger(f"{__name__}.DPOPerformanceMonitor")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Initialize database
        self._initialize_database()
        
        # Performance tracking
        self.session_start = datetime.now()
        self.last_snapshot = None
        
        self.logger.info("DPO Performance Monitor initialized")
    
    def _initialize_database(self):
        """Initialize the metrics database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Metrics table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS personalization_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        metric_type TEXT NOT NULL,
                        value REAL NOT NULL,
                        metadata TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Performance snapshots table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS performance_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        total_users INTEGER,
                        active_models INTEGER,
                        total_requests INTEGER,
                        personalized_requests INTEGER,
                        average_inference_time REAL,
                        cache_hit_rate REAL,
                        error_rate REAL,
                        user_satisfaction_avg REAL,
                        personalization_effectiveness REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # User feedback table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_feedback_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        prompt TEXT,
                        response TEXT,
                        is_personalized BOOLEAN,
                        model_id TEXT,
                        user_rating INTEGER,
                        feedback_text TEXT,
                        improvement_detected BOOLEAN,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes
                conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_user_time ON personalization_metrics(user_id, timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_type_time ON personalization_metrics(metric_type, timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_time ON performance_snapshots(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_user_time ON user_feedback_metrics(user_id, timestamp)")
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error initializing metrics database: {e}")
            raise
    
    def record_metric(self, user_id: str, metric_type: str, value: float, 
                     metadata: Optional[Dict[str, Any]] = None):
        """
        Record a personalization metric.
        
        Args:
            user_id: User identifier
            metric_type: Type of metric
            value: Metric value
            metadata: Additional metadata
        """
        try:
            with self.lock:
                metric = PersonalizationMetric(
                    timestamp=datetime.now(),
                    user_id=user_id,
                    metric_type=metric_type,
                    value=value,
                    metadata=metadata or {}
                )
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO personalization_metrics 
                        (timestamp, user_id, metric_type, value, metadata)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        metric.timestamp.isoformat(),
                        metric.user_id,
                        metric.metric_type,
                        metric.value,
                        json.dumps(metric.metadata)
                    ))
                    conn.commit()
                
                self.logger.debug(f"Recorded metric: {metric_type}={value} for user {user_id}")
                
        except Exception as e:
            self.logger.error(f"Error recording metric: {e}")
    
    def record_response_feedback(self, user_id: str, prompt: str, response: str,
                               is_personalized: bool, model_id: Optional[str] = None,
                               user_rating: Optional[int] = None, 
                               feedback_text: Optional[str] = None,
                               improvement_detected: Optional[bool] = None):
        """
        Record user feedback on a response.
        
        Args:
            user_id: User identifier
            prompt: Original prompt
            response: Generated response
            is_personalized: Whether response used personalized model
            model_id: Model ID if personalized
            user_rating: User rating (1-5)
            feedback_text: User feedback text
            improvement_detected: Whether improvement was detected
        """
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO user_feedback_metrics 
                        (timestamp, user_id, prompt, response, is_personalized, 
                         model_id, user_rating, feedback_text, improvement_detected)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        datetime.now().isoformat(),
                        user_id,
                        prompt,
                        response,
                        is_personalized,
                        model_id,
                        user_rating,
                        feedback_text,
                        improvement_detected
                    ))
                    conn.commit()
                
                # Record derived metrics
                if user_rating is not None:
                    self.record_metric(user_id, "user_satisfaction", user_rating)
                
                if improvement_detected is not None:
                    self.record_metric(user_id, "improvement_detected", 1.0 if improvement_detected else 0.0)
                
        except Exception as e:
            self.logger.error(f"Error recording response feedback: {e}")
    
    def take_performance_snapshot(self) -> PerformanceSnapshot:
        """
        Take a snapshot of current system performance.
        
        Returns:
            Performance snapshot
        """
        try:
            # Get inference engine status
            inference_status = self._get_inference_engine_status()
            
            # Calculate metrics
            snapshot = PerformanceSnapshot(
                timestamp=datetime.now(),
                total_users=self._count_active_users(),
                active_models=len(inference_status.get('active_models', {})),
                total_requests=inference_status.get('metrics', {}).get('total_requests', 0),
                personalized_requests=inference_status.get('metrics', {}).get('personalized_requests', 0),
                average_inference_time=inference_status.get('metrics', {}).get('average_inference_time', 0.0),
                cache_hit_rate=inference_status.get('metrics', {}).get('cache_hit_rate', 0.0),
                error_rate=self._calculate_error_rate(),
                user_satisfaction_avg=self._calculate_average_satisfaction(),
                personalization_effectiveness=self._calculate_personalization_effectiveness()
            )
            
            # Store snapshot
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO performance_snapshots 
                    (timestamp, total_users, active_models, total_requests, 
                     personalized_requests, average_inference_time, cache_hit_rate,
                     error_rate, user_satisfaction_avg, personalization_effectiveness)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    snapshot.timestamp.isoformat(),
                    snapshot.total_users,
                    snapshot.active_models,
                    snapshot.total_requests,
                    snapshot.personalized_requests,
                    snapshot.average_inference_time,
                    snapshot.cache_hit_rate,
                    snapshot.error_rate,
                    snapshot.user_satisfaction_avg,
                    snapshot.personalization_effectiveness
                ))
                conn.commit()
            
            self.last_snapshot = snapshot
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Error taking performance snapshot: {e}")
            return PerformanceSnapshot(
                timestamp=datetime.now(),
                total_users=0, active_models=0, total_requests=0,
                personalized_requests=0, average_inference_time=0.0,
                cache_hit_rate=0.0, error_rate=1.0,
                user_satisfaction_avg=0.0, personalization_effectiveness=0.0
            )
    
    def _get_inference_engine_status(self) -> Dict[str, Any]:
        """Get status from the inference engine."""
        try:
            from .inference_engine import get_personalized_inference_engine
            engine = get_personalized_inference_engine()
            return engine.get_status()
        except Exception:
            return {}
    
    def _count_active_users(self) -> int:
        """Count users with recent activity."""
        try:
            cutoff = datetime.now() - timedelta(hours=24)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT COUNT(DISTINCT user_id) 
                    FROM personalization_metrics 
                    WHERE timestamp > ?
                """, (cutoff.isoformat(),))
                
                return cursor.fetchone()[0]
                
        except Exception:
            return 0
    
    def _calculate_error_rate(self) -> float:
        """Calculate recent error rate."""
        try:
            cutoff = datetime.now() - timedelta(hours=1)
            
            with sqlite3.connect(self.db_path) as conn:
                # Count total requests
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM personalization_metrics 
                    WHERE metric_type = 'inference_request' AND timestamp > ?
                """, (cutoff.isoformat(),))
                total_requests = cursor.fetchone()[0]
                
                if total_requests == 0:
                    return 0.0
                
                # Count errors
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM personalization_metrics 
                    WHERE metric_type = 'inference_error' AND timestamp > ?
                """, (cutoff.isoformat(),))
                errors = cursor.fetchone()[0]
                
                return errors / total_requests
                
        except Exception:
            return 0.0
    
    def _calculate_average_satisfaction(self) -> float:
        """Calculate average user satisfaction."""
        try:
            cutoff = datetime.now() - timedelta(days=7)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT AVG(user_rating) FROM user_feedback_metrics 
                    WHERE user_rating IS NOT NULL AND timestamp > ?
                """, (cutoff.isoformat(),))
                
                result = cursor.fetchone()[0]
                return result if result is not None else 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_personalization_effectiveness(self) -> float:
        """Calculate personalization effectiveness."""
        try:
            cutoff = datetime.now() - timedelta(days=7)
            
            with sqlite3.connect(self.db_path) as conn:
                # Get average satisfaction for personalized responses
                cursor = conn.execute("""
                    SELECT AVG(user_rating) FROM user_feedback_metrics 
                    WHERE is_personalized = 1 AND user_rating IS NOT NULL AND timestamp > ?
                """, (cutoff.isoformat(),))
                personalized_satisfaction = cursor.fetchone()[0] or 0.0
                
                # Get average satisfaction for base model responses
                cursor = conn.execute("""
                    SELECT AVG(user_rating) FROM user_feedback_metrics 
                    WHERE is_personalized = 0 AND user_rating IS NOT NULL AND timestamp > ?
                """, (cutoff.isoformat(),))
                base_satisfaction = cursor.fetchone()[0] or 0.0
                
                if base_satisfaction == 0:
                    return 0.0
                
                # Calculate improvement ratio
                return (personalized_satisfaction - base_satisfaction) / base_satisfaction
                
        except Exception:
            return 0.0
    
    def get_user_metrics(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        """
        Get metrics for a specific user.
        
        Args:
            user_id: User identifier
            days: Number of days to look back
            
        Returns:
            User metrics dictionary
        """
        try:
            cutoff = datetime.now() - timedelta(days=days)
            
            with sqlite3.connect(self.db_path) as conn:
                # Get metric summary
                cursor = conn.execute("""
                    SELECT metric_type, COUNT(*), AVG(value), MIN(value), MAX(value)
                    FROM personalization_metrics 
                    WHERE user_id = ? AND timestamp > ?
                    GROUP BY metric_type
                """, (user_id, cutoff.isoformat()))
                
                metrics = {}
                for row in cursor.fetchall():
                    metric_type, count, avg_val, min_val, max_val = row
                    metrics[metric_type] = {
                        'count': count,
                        'average': avg_val,
                        'min': min_val,
                        'max': max_val
                    }
                
                # Get feedback summary
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_responses,
                        SUM(CASE WHEN is_personalized = 1 THEN 1 ELSE 0 END) as personalized_responses,
                        AVG(user_rating) as avg_rating,
                        SUM(CASE WHEN improvement_detected = 1 THEN 1 ELSE 0 END) as improvements
                    FROM user_feedback_metrics 
                    WHERE user_id = ? AND timestamp > ?
                """, (user_id, cutoff.isoformat()))
                
                feedback_row = cursor.fetchone()
                feedback_summary = {
                    'total_responses': feedback_row[0] or 0,
                    'personalized_responses': feedback_row[1] or 0,
                    'average_rating': feedback_row[2] or 0.0,
                    'improvements_detected': feedback_row[3] or 0
                }
                
                return {
                    'user_id': user_id,
                    'period_days': days,
                    'metrics': metrics,
                    'feedback_summary': feedback_summary,
                    'personalization_rate': (
                        feedback_summary['personalized_responses'] / max(1, feedback_summary['total_responses'])
                    )
                }
                
        except Exception as e:
            self.logger.error(f"Error getting user metrics: {e}")
            return {'user_id': user_id, 'error': str(e)}
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health metrics.
        
        Returns:
            System health dictionary
        """
        try:
            snapshot = self.take_performance_snapshot()
            
            # Calculate health score (0-100)
            health_factors = [
                min(100, snapshot.cache_hit_rate * 100),  # Cache performance
                max(0, 100 - snapshot.error_rate * 100),  # Error rate (inverted)
                min(100, snapshot.user_satisfaction_avg * 20),  # User satisfaction (1-5 scale)
                max(0, min(100, 50 + snapshot.personalization_effectiveness * 50))  # Effectiveness
            ]
            
            health_score = sum(health_factors) / len(health_factors)
            
            # Determine status
            if health_score >= 80:
                status = "excellent"
            elif health_score >= 60:
                status = "good"
            elif health_score >= 40:
                status = "fair"
            else:
                status = "poor"
            
            return {
                'health_score': health_score,
                'status': status,
                'snapshot': snapshot,
                'uptime_hours': (datetime.now() - self.session_start).total_seconds() / 3600,
                'recommendations': self._generate_recommendations(snapshot)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system health: {e}")
            return {
                'health_score': 0,
                'status': 'error',
                'error': str(e)
            }
    
    def _generate_recommendations(self, snapshot: PerformanceSnapshot) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        if snapshot.cache_hit_rate < 0.5:
            recommendations.append("Consider increasing model cache size to improve performance")
        
        if snapshot.error_rate > 0.1:
            recommendations.append("High error rate detected - check system logs")
        
        if snapshot.user_satisfaction_avg < 3.0:
            recommendations.append("User satisfaction is low - review personalization quality")
        
        if snapshot.personalization_effectiveness < 0.1:
            recommendations.append("Personalization showing limited effectiveness - review training data quality")
        
        if snapshot.average_inference_time > 2.0:
            recommendations.append("Inference time is high - consider model optimization")
        
        if not recommendations:
            recommendations.append("System is performing well - no immediate actions needed")
        
        return recommendations


# Global performance monitor instance
_performance_monitor = None

def get_dpo_performance_monitor(db_path: str = "./logs/performance_metrics.db") -> DPOPerformanceMonitor:
    """Get or create a global DPO performance monitor instance."""
    global _performance_monitor
    
    if _performance_monitor is None:
        _performance_monitor = DPOPerformanceMonitor(db_path)
    
    return _performance_monitor
