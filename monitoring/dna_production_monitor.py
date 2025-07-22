"""
DNA Layer Production Monitoring System
======================================

Comprehensive monitoring and telemetry for DNA layer in production.
Tracks routing patterns, efficiency gains, and performance across all users.
"""

import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class DNAProductionMonitor:
    """
    Production monitoring system for DNA layer performance.
    
    Tracks real-world usage patterns, efficiency gains, and routing intelligence
    across the entire SAM user base.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.monitoring_active = True
        
        # Real-time metrics
        self.routing_decisions = deque(maxlen=10000)  # Last 10k routing decisions
        self.efficiency_history = deque(maxlen=1000)   # Last 1k efficiency measurements
        self.performance_metrics = deque(maxlen=1000)  # Last 1k performance measurements
        
        # Aggregated statistics
        self.daily_stats = defaultdict(lambda: {
            'total_requests': 0,
            'total_efficiency': 0.0,
            'routing_patterns': defaultdict(int),
            'user_scenarios': defaultdict(int),
            'performance_data': []
        })
        
        # Alert thresholds
        self.efficiency_threshold = config.get('efficiency_threshold_alert', 0.15)
        self.performance_threshold = config.get('max_forward_time', 0.200)
        
        # Setup logging
        self.setup_monitoring_logs()
        
        logger.info("DNA Production Monitor initialized")
        logger.info(f"Monitoring configuration: {config}")
    
    def setup_monitoring_logs(self):
        """Setup production monitoring logs."""
        log_dir = Path(self.config.get('report_directory', 'logs/dna_routing'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create daily log file
        today = datetime.now().strftime('%Y-%m-%d')
        self.daily_log_file = log_dir / f"dna_production_{today}.log"
        
        # Setup file handler
        file_handler = logging.FileHandler(self.daily_log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add to logger
        production_logger = logging.getLogger('dna_production')
        production_logger.addHandler(file_handler)
        production_logger.setLevel(logging.INFO)
        
        self.production_logger = production_logger
    
    def record_routing_decision(
        self,
        user_id: str,
        session_id: str,
        routing_info: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        user_scenario: str = "unknown"
    ):
        """Record a routing decision from production usage."""
        if not self.monitoring_active:
            return
        
        timestamp = time.time()
        
        # Extract key metrics
        expert_utilization = routing_info.get('expert_utilization', [0, 0, 0, 0])
        identity_usage = expert_utilization[2] if len(expert_utilization) > 2 else 0
        routing_entropy = routing_info.get('routing_entropy', 0)
        forward_time = performance_metrics.get('forward_time', 0)
        
        # Create routing record
        routing_record = {
            'timestamp': timestamp,
            'user_id': user_id,
            'session_id': session_id,
            'identity_usage': identity_usage,
            'routing_entropy': routing_entropy,
            'forward_time': forward_time,
            'user_scenario': user_scenario,
            'expert_distribution': expert_utilization
        }
        
        # Store in real-time queues
        self.routing_decisions.append(routing_record)
        self.efficiency_history.append(identity_usage)
        self.performance_metrics.append(forward_time)
        
        # Update daily statistics
        today = datetime.now().strftime('%Y-%m-%d')
        daily_stat = self.daily_stats[today]
        daily_stat['total_requests'] += 1
        daily_stat['total_efficiency'] += identity_usage
        daily_stat['user_scenarios'][user_scenario] += 1
        daily_stat['performance_data'].append(forward_time)
        
        # Update routing patterns
        dominant_expert = np.argmax(expert_utilization)
        expert_names = ['attention', 'mlp', 'identity', 'normalization']
        daily_stat['routing_patterns'][expert_names[dominant_expert]] += 1
        
        # Log to production logs
        self.production_logger.info(
            f"ROUTING_DECISION user={user_id} scenario={user_scenario} "
            f"efficiency={identity_usage:.3f} entropy={routing_entropy:.3f} "
            f"time={forward_time:.4f}s"
        )
        
        # Check for alerts
        self.check_alerts(routing_record)
    
    def check_alerts(self, routing_record: Dict[str, Any]):
        """Check for performance alerts and anomalies."""
        
        # Efficiency alert
        if routing_record['identity_usage'] < self.efficiency_threshold:
            self.production_logger.warning(
                f"LOW_EFFICIENCY_ALERT user={routing_record['user_id']} "
                f"efficiency={routing_record['identity_usage']:.3f} "
                f"threshold={self.efficiency_threshold}"
            )
        
        # Performance alert
        if routing_record['forward_time'] > self.performance_threshold:
            self.production_logger.warning(
                f"SLOW_PERFORMANCE_ALERT user={routing_record['user_id']} "
                f"time={routing_record['forward_time']:.4f}s "
                f"threshold={self.performance_threshold}s"
            )
        
        # Routing entropy alert (too low = routing collapse)
        if routing_record['routing_entropy'] < 0.5:
            self.production_logger.warning(
                f"ROUTING_COLLAPSE_ALERT user={routing_record['user_id']} "
                f"entropy={routing_record['routing_entropy']:.3f}"
            )
    
    def get_real_time_stats(self) -> Dict[str, Any]:
        """Get real-time production statistics."""
        if not self.efficiency_history:
            return {'status': 'no_data'}
        
        # Calculate current metrics
        current_efficiency = np.mean(list(self.efficiency_history)[-100:])  # Last 100 requests
        current_performance = np.mean(list(self.performance_metrics)[-100:])
        
        # Routing distribution
        recent_decisions = list(self.routing_decisions)[-100:]
        routing_distribution = defaultdict(int)
        
        for decision in recent_decisions:
            expert_dist = decision['expert_distribution']
            dominant_expert = np.argmax(expert_dist)
            expert_names = ['attention', 'mlp', 'identity', 'normalization']
            routing_distribution[expert_names[dominant_expert]] += 1
        
        return {
            'timestamp': time.time(),
            'current_efficiency': current_efficiency,
            'current_performance': current_performance,
            'total_requests_monitored': len(self.routing_decisions),
            'routing_distribution': dict(routing_distribution),
            'efficiency_trend': self._calculate_trend(self.efficiency_history),
            'performance_trend': self._calculate_trend(self.performance_metrics),
            'status': 'healthy' if current_efficiency > self.efficiency_threshold else 'warning'
        }
    
    def _calculate_trend(self, data_queue: deque) -> str:
        """Calculate trend direction for a metric."""
        if len(data_queue) < 20:
            return 'insufficient_data'
        
        recent_data = list(data_queue)[-20:]
        first_half = np.mean(recent_data[:10])
        second_half = np.mean(recent_data[10:])
        
        if second_half > first_half * 1.05:
            return 'improving'
        elif second_half < first_half * 0.95:
            return 'declining'
        else:
            return 'stable'
    
    def generate_daily_report(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive daily report."""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        daily_stat = self.daily_stats[date]
        
        if daily_stat['total_requests'] == 0:
            return {'date': date, 'status': 'no_data'}
        
        # Calculate daily metrics
        avg_efficiency = daily_stat['total_efficiency'] / daily_stat['total_requests']
        avg_performance = np.mean(daily_stat['performance_data']) if daily_stat['performance_data'] else 0
        
        # Routing analysis
        total_routing = sum(daily_stat['routing_patterns'].values())
        routing_percentages = {
            expert: (count / total_routing) * 100
            for expert, count in daily_stat['routing_patterns'].items()
        }
        
        # User scenario analysis
        total_scenarios = sum(daily_stat['user_scenarios'].values())
        scenario_percentages = {
            scenario: (count / total_scenarios) * 100
            for scenario, count in daily_stat['user_scenarios'].items()
        }
        
        report = {
            'date': date,
            'summary': {
                'total_requests': daily_stat['total_requests'],
                'average_efficiency': avg_efficiency,
                'average_performance': avg_performance,
                'efficiency_vs_target': avg_efficiency / 0.219,  # vs 21.9% validation target
                'performance_vs_target': avg_performance / 0.200   # vs 200ms target
            },
            'routing_analysis': {
                'expert_usage_percentages': routing_percentages,
                'identity_module_usage': routing_percentages.get('identity', 0),
                'routing_diversity': len([p for p in routing_percentages.values() if p > 5])
            },
            'user_scenarios': scenario_percentages,
            'performance_metrics': {
                'min_response_time': min(daily_stat['performance_data']) if daily_stat['performance_data'] else 0,
                'max_response_time': max(daily_stat['performance_data']) if daily_stat['performance_data'] else 0,
                'p95_response_time': np.percentile(daily_stat['performance_data'], 95) if daily_stat['performance_data'] else 0
            },
            'health_status': 'healthy' if avg_efficiency > self.efficiency_threshold else 'needs_attention'
        }
        
        # Save report
        if self.config.get('save_routing_reports', True):
            report_path = Path(self.config.get('report_directory', 'logs/dna_routing'))
            report_file = report_path / f"daily_report_{date}.json"
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Daily report saved: {report_file}")
        
        return report
    
    def get_production_health_summary(self) -> Dict[str, Any]:
        """Get overall production health summary."""
        real_time_stats = self.get_real_time_stats()
        
        # Calculate overall health score
        efficiency_score = min(100, (real_time_stats.get('current_efficiency', 0) / 0.219) * 100)
        performance_score = min(100, (0.200 / max(real_time_stats.get('current_performance', 0.001), 0.001)) * 100)
        
        overall_health = (efficiency_score + performance_score) / 2
        
        return {
            'overall_health_score': overall_health,
            'efficiency_score': efficiency_score,
            'performance_score': performance_score,
            'status': 'excellent' if overall_health > 90 else 'good' if overall_health > 75 else 'needs_attention',
            'total_requests_monitored': len(self.routing_decisions),
            'monitoring_duration_hours': (time.time() - (self.routing_decisions[0]['timestamp'] if self.routing_decisions else time.time())) / 3600,
            'real_time_metrics': real_time_stats
        }


# Global production monitor instance
production_monitor = None

def initialize_production_monitoring(config: Dict[str, Any]):
    """Initialize global production monitoring."""
    global production_monitor
    production_monitor = DNAProductionMonitor(config)
    logger.info("DNA Production Monitoring initialized")

def record_production_routing(user_id: str, session_id: str, routing_info: Dict, performance_metrics: Dict, scenario: str = "unknown"):
    """Record routing decision in production."""
    if production_monitor:
        production_monitor.record_routing_decision(user_id, session_id, routing_info, performance_metrics, scenario)

def get_production_health():
    """Get current production health status."""
    if production_monitor:
        return production_monitor.get_production_health_summary()
    return {'status': 'monitoring_not_initialized'}
