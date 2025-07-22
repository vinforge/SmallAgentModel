"""
Test-Time Training (TTT) Performance Monitoring
==============================================

Comprehensive monitoring and metrics collection for TTT operations,
including performance tracking, A/B testing, and optimization insights.
"""

import json
import time
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

import logging

logger = logging.getLogger(__name__)

@dataclass
class TTTPerformanceMetric:
    """Individual TTT performance measurement."""
    session_id: str
    timestamp: datetime
    task_type: str
    examples_count: int
    training_steps: int
    adaptation_time: float
    confidence_score: float
    convergence_score: float
    success: bool
    fallback_reason: Optional[str] = None
    user_feedback: Optional[float] = None  # 1-5 rating
    accuracy_improvement: Optional[float] = None  # vs ICL baseline

@dataclass
class TTTSessionSummary:
    """Summary of TTT performance over a session or time period."""
    total_attempts: int
    successful_adaptations: int
    average_confidence: float
    average_adaptation_time: float
    average_training_steps: float
    fallback_rate: float
    top_task_types: List[Tuple[str, int]]
    performance_trend: str  # "improving", "stable", "declining"

class TTTMetricsCollector:
    """Collects and analyzes TTT performance metrics."""
    
    def __init__(self, db_path: str = "logs/ttt_metrics.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        logger.info(f"TTT metrics collector initialized with DB: {self.db_path}")
    
    def _init_database(self):
        """Initialize the TTT metrics database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ttt_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    task_type TEXT,
                    examples_count INTEGER,
                    training_steps INTEGER,
                    adaptation_time REAL,
                    confidence_score REAL,
                    convergence_score REAL,
                    success BOOLEAN,
                    fallback_reason TEXT,
                    user_feedback REAL,
                    accuracy_improvement REAL,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ttt_ab_tests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    method TEXT NOT NULL,  -- 'TTT' or 'ICL'
                    task_type TEXT,
                    examples_count INTEGER,
                    response_quality REAL,
                    response_time REAL,
                    user_satisfaction REAL,
                    metadata TEXT
                )
            """)
            
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ttt_timestamp ON ttt_metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ttt_session ON ttt_metrics(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ab_test ON ttt_ab_tests(test_id, method)")
    
    def record_ttt_attempt(self, metric: TTTPerformanceMetric, metadata: Dict[str, Any] = None):
        """Record a TTT adaptation attempt."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO ttt_metrics (
                        session_id, timestamp, task_type, examples_count,
                        training_steps, adaptation_time, confidence_score,
                        convergence_score, success, fallback_reason,
                        user_feedback, accuracy_improvement, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metric.session_id,
                    metric.timestamp.isoformat(),
                    metric.task_type,
                    metric.examples_count,
                    metric.training_steps,
                    metric.adaptation_time,
                    metric.confidence_score,
                    metric.convergence_score,
                    metric.success,
                    metric.fallback_reason,
                    metric.user_feedback,
                    metric.accuracy_improvement,
                    json.dumps(metadata or {})
                ))
            
            logger.debug(f"Recorded TTT metric for session {metric.session_id}")
            
        except Exception as e:
            logger.error(f"Failed to record TTT metric: {e}")
    
    def record_ab_test_result(self, test_id: str, session_id: str, method: str,
                            task_type: str, examples_count: int, response_quality: float,
                            response_time: float, user_satisfaction: float = None,
                            metadata: Dict[str, Any] = None):
        """Record A/B test result comparing TTT vs ICL."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO ttt_ab_tests (
                        test_id, session_id, timestamp, method, task_type,
                        examples_count, response_quality, response_time,
                        user_satisfaction, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    test_id,
                    session_id,
                    datetime.now().isoformat(),
                    method,
                    task_type,
                    examples_count,
                    response_quality,
                    response_time,
                    user_satisfaction,
                    json.dumps(metadata or {})
                ))
            
            logger.debug(f"Recorded A/B test result: {method} for test {test_id}")
            
        except Exception as e:
            logger.error(f"Failed to record A/B test result: {e}")
    
    def get_session_summary(self, session_id: str) -> TTTSessionSummary:
        """Get performance summary for a specific session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_attempts,
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                        AVG(confidence_score) as avg_confidence,
                        AVG(adaptation_time) as avg_time,
                        AVG(training_steps) as avg_steps,
                        task_type,
                        COUNT(task_type) as task_count
                    FROM ttt_metrics 
                    WHERE session_id = ?
                    GROUP BY task_type
                    ORDER BY task_count DESC
                """, (session_id,))
                
                results = cursor.fetchall()
                
                if not results:
                    return TTTSessionSummary(0, 0, 0.0, 0.0, 0.0, 0.0, [], "stable")
                
                # Aggregate results
                total_attempts = sum(r[0] for r in results)
                successful_adaptations = sum(r[1] for r in results)
                avg_confidence = np.mean([r[2] for r in results if r[2] is not None])
                avg_time = np.mean([r[3] for r in results if r[3] is not None])
                avg_steps = np.mean([r[4] for r in results if r[4] is not None])
                fallback_rate = 1.0 - (successful_adaptations / total_attempts) if total_attempts > 0 else 0.0
                
                top_task_types = [(r[5], r[6]) for r in results[:5]]
                
                return TTTSessionSummary(
                    total_attempts=total_attempts,
                    successful_adaptations=successful_adaptations,
                    average_confidence=avg_confidence,
                    average_adaptation_time=avg_time,
                    average_training_steps=avg_steps,
                    fallback_rate=fallback_rate,
                    top_task_types=top_task_types,
                    performance_trend="stable"  # TODO: Calculate trend
                )
                
        except Exception as e:
            logger.error(f"Failed to get session summary: {e}")
            return TTTSessionSummary(0, 0, 0.0, 0.0, 0.0, 0.0, [], "unknown")
    
    def get_performance_trends(self, days: int = 7) -> Dict[str, Any]:
        """Get TTT performance trends over the specified number of days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with sqlite3.connect(self.db_path) as conn:
                # Daily success rates
                cursor = conn.execute("""
                    SELECT 
                        DATE(timestamp) as date,
                        COUNT(*) as total,
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                        AVG(confidence_score) as avg_confidence,
                        AVG(adaptation_time) as avg_time
                    FROM ttt_metrics 
                    WHERE timestamp >= ?
                    GROUP BY DATE(timestamp)
                    ORDER BY date
                """, (cutoff_date.isoformat(),))
                
                daily_stats = cursor.fetchall()
                
                # Task type performance
                cursor = conn.execute("""
                    SELECT 
                        task_type,
                        COUNT(*) as total,
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                        AVG(confidence_score) as avg_confidence
                    FROM ttt_metrics 
                    WHERE timestamp >= ?
                    GROUP BY task_type
                    ORDER BY total DESC
                """, (cutoff_date.isoformat(),))
                
                task_stats = cursor.fetchall()
                
                return {
                    "daily_performance": [
                        {
                            "date": row[0],
                            "total_attempts": row[1],
                            "success_rate": row[2] / row[1] if row[1] > 0 else 0,
                            "avg_confidence": row[3] or 0,
                            "avg_adaptation_time": row[4] or 0
                        }
                        for row in daily_stats
                    ],
                    "task_type_performance": [
                        {
                            "task_type": row[0],
                            "total_attempts": row[1],
                            "success_rate": row[2] / row[1] if row[1] > 0 else 0,
                            "avg_confidence": row[3] or 0
                        }
                        for row in task_stats
                    ]
                }
                
        except Exception as e:
            logger.error(f"Failed to get performance trends: {e}")
            return {"daily_performance": [], "task_type_performance": []}
    
    def get_ab_test_results(self, test_id: str) -> Dict[str, Any]:
        """Get A/B test results comparing TTT vs ICL performance."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        method,
                        COUNT(*) as count,
                        AVG(response_quality) as avg_quality,
                        AVG(response_time) as avg_time,
                        AVG(user_satisfaction) as avg_satisfaction
                    FROM ttt_ab_tests 
                    WHERE test_id = ?
                    GROUP BY method
                """, (test_id,))
                
                results = cursor.fetchall()
                
                ab_results = {}
                for row in results:
                    ab_results[row[0]] = {
                        "count": row[1],
                        "avg_quality": row[2] or 0,
                        "avg_response_time": row[3] or 0,
                        "avg_satisfaction": row[4] or 0
                    }
                
                # Calculate improvement metrics
                if "TTT" in ab_results and "ICL" in ab_results:
                    ttt_quality = ab_results["TTT"]["avg_quality"]
                    icl_quality = ab_results["ICL"]["avg_quality"]
                    quality_improvement = ((ttt_quality - icl_quality) / icl_quality * 100) if icl_quality > 0 else 0
                    
                    ab_results["comparison"] = {
                        "quality_improvement_percent": quality_improvement,
                        "time_overhead": ab_results["TTT"]["avg_response_time"] - ab_results["ICL"]["avg_response_time"],
                        "satisfaction_delta": ab_results["TTT"]["avg_satisfaction"] - ab_results["ICL"]["avg_satisfaction"]
                    }
                
                return ab_results
                
        except Exception as e:
            logger.error(f"Failed to get A/B test results: {e}")
            return {}
    
    def export_metrics(self, output_path: str, days: int = 30):
        """Export TTT metrics to JSON file for analysis."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM ttt_metrics 
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                """, (cutoff_date.isoformat(),))
                
                columns = [desc[0] for desc in cursor.description]
                metrics = [dict(zip(columns, row)) for row in cursor.fetchall()]
                
                export_data = {
                    "export_timestamp": datetime.now().isoformat(),
                    "period_days": days,
                    "total_records": len(metrics),
                    "metrics": metrics
                }
                
                with open(output_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                logger.info(f"Exported {len(metrics)} TTT metrics to {output_path}")
                
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")

# Global metrics collector instance
_metrics_collector = None

def get_ttt_metrics_collector() -> TTTMetricsCollector:
    """Get the global TTT metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = TTTMetricsCollector()
    return _metrics_collector
