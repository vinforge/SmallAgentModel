"""
SLP Metrics Collector
====================

Real-time metrics collection for SLP system performance tracking.
Provides lightweight, efficient collection of execution and performance data.

Phase 1A.2 - Enhanced Performance Tracking (preserving 100% of existing functionality)
"""

import logging
import time
import threading
import psutil
import hashlib
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from collections import deque, defaultdict

from .latent_program import LatentProgram, ExecutionResult
from .latent_program_store import LatentProgramStore

logger = logging.getLogger(__name__)


class SLPMetricsCollector:
    """
    Real-time metrics collection for SLP system.
    
    Provides efficient, non-intrusive collection of performance metrics
    while preserving 100% of existing SLP functionality.
    """
    
    def __init__(self, store: Optional[LatentProgramStore] = None, 
                 collection_interval: int = 60):
        """
        Initialize the metrics collector.
        
        Args:
            store: Optional LatentProgramStore instance
            collection_interval: Interval in seconds for periodic metric collection
        """
        self.store = store or LatentProgramStore()
        self.collection_interval = collection_interval
        self.enabled = True
        
        # Real-time metrics storage
        self.execution_metrics = deque(maxlen=1000)  # Last 1000 executions
        self.pattern_discoveries = deque(maxlen=500)  # Last 500 discoveries
        self.system_metrics = deque(maxlen=100)  # Last 100 system snapshots
        
        # Performance counters
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)
        
        # Thread-safe locks
        self._metrics_lock = threading.Lock()
        self._collection_thread = None
        self._stop_collection = threading.Event()
        
        # Callback functions for real-time updates
        self.callbacks = {
            'execution': [],
            'pattern_discovery': [],
            'system_update': []
        }
        
        logger.info(f"SLP Metrics Collector initialized with {collection_interval}s interval")
    
    def start_collection(self):
        """Start background metrics collection."""
        try:
            if self._collection_thread and self._collection_thread.is_alive():
                logger.warning("Metrics collection already running")
                return
            
            self._stop_collection.clear()
            self._collection_thread = threading.Thread(
                target=self._collection_loop,
                name="SLPMetricsCollector",
                daemon=True
            )
            self._collection_thread.start()
            
            logger.info("Started SLP metrics collection")
            
        except Exception as e:
            logger.error(f"Failed to start metrics collection: {e}")
    
    def stop_collection(self):
        """Stop background metrics collection."""
        try:
            if self._collection_thread and self._collection_thread.is_alive():
                self._stop_collection.set()
                self._collection_thread.join(timeout=5)
                
            logger.info("Stopped SLP metrics collection")
            
        except Exception as e:
            logger.error(f"Failed to stop metrics collection: {e}")
    
    def on_program_execution(self, program: LatentProgram, result: ExecutionResult, 
                           context: Optional[Dict[str, Any]] = None):
        """
        Collect metrics when program executes.
        
        Args:
            program: The executed program
            result: Execution result
            context: Optional execution context
        """
        try:
            if not self.enabled:
                return
            
            execution_time = time.time()
            
            # Calculate context hash for pattern analysis
            context_hash = self._calculate_context_hash(context or {})
            
            # Determine if TPV was used
            tpv_used = context and context.get('tpv_enabled', False)
            
            # Calculate efficiency gain if baseline is available
            efficiency_gain = 0.0
            baseline_time = context.get('baseline_execution_time_ms', 0) if context else 0
            if baseline_time > 0 and result.execution_time_ms > 0:
                efficiency_gain = ((baseline_time - result.execution_time_ms) / baseline_time) * 100
            
            # Create execution metrics record
            execution_metrics = {
                'timestamp': execution_time,
                'program_id': program.id,
                'execution_time_ms': result.execution_time_ms,
                'quality_score': result.quality_score,
                'success': result.success,
                'token_count': result.token_count,
                'context_hash': context_hash,
                'tpv_used': tpv_used,
                'efficiency_gain': efficiency_gain,
                'user_profile': program.active_profile,
                'query_type': self._classify_query_type(context),
                'error_message': result.error_message if not result.success else '',
                'baseline_time_ms': baseline_time,
                'confidence_at_execution': program.confidence_score,
                'memory_usage_mb': self._get_memory_usage(),
                'cpu_usage_percent': self._get_cpu_usage()
            }
            
            # Store in real-time metrics
            with self._metrics_lock:
                self.execution_metrics.append(execution_metrics)
                self.counters['total_executions'] += 1
                if result.success:
                    self.counters['successful_executions'] += 1
                self.timers['execution_times'].append(result.execution_time_ms)
            
            # Store in database for persistence
            self.store.record_enhanced_execution(program.id, execution_metrics)
            
            # Trigger callbacks
            self._trigger_callbacks('execution', execution_metrics)
            
            logger.debug(f"Collected execution metrics for program {program.id}")
            
        except Exception as e:
            logger.error(f"Failed to collect execution metrics: {e}")
    
    def on_pattern_capture(self, pattern_data: Dict[str, Any], success: bool):
        """
        Collect metrics when pattern is captured.
        
        Args:
            pattern_data: Pattern information
            success: Whether capture was successful
        """
        try:
            if not self.enabled:
                return
            
            capture_time = time.time()
            
            # Create pattern discovery record
            discovery_metrics = {
                'timestamp': capture_time,
                'pattern_type': pattern_data.get('pattern_type', 'unknown'),
                'signature_hash': pattern_data.get('signature_hash', ''),
                'capture_success': success,
                'similarity_score': pattern_data.get('similarity_score', 0.0),
                'user_context': pattern_data.get('user_context', ''),
                'query_text': pattern_data.get('query_text', ''),
                'response_quality': pattern_data.get('response_quality', 0.0),
                'capture_reason': pattern_data.get('capture_reason', ''),
                'program_id': pattern_data.get('program_id', ''),
                'user_profile': pattern_data.get('user_profile', 'default'),
                'complexity_level': pattern_data.get('complexity_level', 'medium'),
                'domain_category': pattern_data.get('domain_category', 'general')
            }
            
            # Store in real-time metrics
            with self._metrics_lock:
                self.pattern_discoveries.append(discovery_metrics)
                self.counters['pattern_discoveries'] += 1
                if success:
                    self.counters['successful_captures'] += 1
            
            # Store in database for persistence
            self.store.log_pattern_discovery(discovery_metrics)
            
            # Trigger callbacks
            self._trigger_callbacks('pattern_discovery', discovery_metrics)
            
            logger.debug(f"Collected pattern capture metrics: {pattern_data.get('pattern_type', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Failed to collect pattern capture metrics: {e}")
    
    def on_user_feedback(self, program_id: str, feedback: int, context: Optional[Dict[str, Any]] = None):
        """
        Collect user feedback correlation.
        
        Args:
            program_id: ID of the program that received feedback
            feedback: Feedback value (typically -1, 0, 1)
            context: Optional feedback context
        """
        try:
            if not self.enabled:
                return
            
            feedback_time = time.time()
            
            # Update execution metrics with feedback
            with self._metrics_lock:
                # Find recent execution for this program
                for metrics in reversed(self.execution_metrics):
                    if (metrics['program_id'] == program_id and 
                        feedback_time - metrics['timestamp'] < 300):  # Within 5 minutes
                        metrics['user_feedback'] = feedback
                        break
                
                # Update counters
                if feedback > 0:
                    self.counters['positive_feedback'] += 1
                elif feedback < 0:
                    self.counters['negative_feedback'] += 1
            
            logger.debug(f"Collected user feedback for program {program_id}: {feedback}")
            
        except Exception as e:
            logger.error(f"Failed to collect user feedback: {e}")
    
    def get_real_time_stats(self) -> Dict[str, Any]:
        """Get current real-time statistics."""
        try:
            with self._metrics_lock:
                current_time = time.time()
                
                # Calculate recent metrics (last 5 minutes)
                recent_executions = [
                    m for m in self.execution_metrics 
                    if current_time - m['timestamp'] < 300
                ]
                
                recent_discoveries = [
                    d for d in self.pattern_discoveries 
                    if current_time - d['timestamp'] < 300
                ]
                
                # Calculate hit rate
                total_recent = len(recent_executions)
                successful_recent = sum(1 for m in recent_executions if m['success'])
                hit_rate = (successful_recent / total_recent * 100) if total_recent > 0 else 0
                
                # Calculate average execution time
                recent_times = [m['execution_time_ms'] for m in recent_executions if m['success']]
                avg_execution_time = sum(recent_times) / len(recent_times) if recent_times else 0
                
                # Calculate efficiency gains
                efficiency_gains = [m['efficiency_gain'] for m in recent_executions if m['efficiency_gain'] > 0]
                avg_efficiency_gain = sum(efficiency_gains) / len(efficiency_gains) if efficiency_gains else 0
                
                return {
                    'timestamp': datetime.utcnow().isoformat(),
                    'recent_executions': total_recent,
                    'hit_rate_percent': round(hit_rate, 2),
                    'avg_execution_time_ms': round(avg_execution_time, 2),
                    'avg_efficiency_gain_percent': round(avg_efficiency_gain, 2),
                    'recent_discoveries': len(recent_discoveries),
                    'successful_captures': sum(1 for d in recent_discoveries if d['capture_success']),
                    'total_counters': dict(self.counters),
                    'system_load': {
                        'memory_usage_mb': self._get_memory_usage(),
                        'cpu_usage_percent': self._get_cpu_usage()
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to get real-time stats: {e}")
            return {}
    
    def register_callback(self, event_type: str, callback: Callable):
        """
        Register callback for real-time updates.
        
        Args:
            event_type: Type of event ('execution', 'pattern_discovery', 'system_update')
            callback: Callback function to register
        """
        try:
            if event_type in self.callbacks:
                self.callbacks[event_type].append(callback)
                logger.debug(f"Registered callback for {event_type} events")
            else:
                logger.warning(f"Unknown event type: {event_type}")
                
        except Exception as e:
            logger.error(f"Failed to register callback: {e}")
    
    def _collection_loop(self):
        """Background collection loop for periodic metrics."""
        try:
            while not self._stop_collection.wait(self.collection_interval):
                if not self.enabled:
                    continue
                
                # Collect system metrics
                system_metrics = {
                    'timestamp': time.time(),
                    'memory_usage_mb': self._get_memory_usage(),
                    'cpu_usage_percent': self._get_cpu_usage(),
                    'active_programs': len(self.store.get_all_programs()),
                    'total_executions': self.counters['total_executions'],
                    'successful_executions': self.counters['successful_executions'],
                    'pattern_discoveries': self.counters['pattern_discoveries'],
                    'successful_captures': self.counters['successful_captures']
                }
                
                # Store system metrics
                with self._metrics_lock:
                    self.system_metrics.append(system_metrics)
                
                # Store in database
                self.store.record_system_metrics(system_metrics)
                
                # Trigger system update callbacks
                self._trigger_callbacks('system_update', system_metrics)
                
        except Exception as e:
            logger.error(f"Error in metrics collection loop: {e}")
    
    def _calculate_context_hash(self, context: Dict[str, Any]) -> str:
        """Calculate hash of context for pattern analysis."""
        try:
            # Create a simplified context representation for hashing
            context_str = str(sorted(context.items()))
            return hashlib.md5(context_str.encode()).hexdigest()[:16]
        except Exception:
            return "unknown"
    
    def _classify_query_type(self, context: Optional[Dict[str, Any]]) -> str:
        """Classify query type based on context."""
        try:
            if not context:
                return "general"
            
            # Simple classification based on context
            if context.get('documents'):
                return "document_query"
            elif context.get('web_search'):
                return "web_search"
            elif context.get('calculation'):
                return "calculation"
            else:
                return "general"
                
        except Exception:
            return "unknown"
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return round(process.memory_info().rss / 1024 / 1024, 2)
        except Exception:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return round(psutil.cpu_percent(interval=None), 2)
        except Exception:
            return 0.0
    
    def _trigger_callbacks(self, event_type: str, data: Dict[str, Any]):
        """Trigger registered callbacks for an event type."""
        try:
            for callback in self.callbacks.get(event_type, []):
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in callback for {event_type}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to trigger callbacks: {e}")
    
    def reset_counters(self):
        """Reset performance counters."""
        try:
            with self._metrics_lock:
                self.counters.clear()
                self.timers.clear()
            
            logger.info("Reset SLP metrics counters")
            
        except Exception as e:
            logger.error(f"Failed to reset counters: {e}")
    
    def get_historical_data(self, hours: int = 24) -> Dict[str, Any]:
        """Get historical metrics data."""
        try:
            cutoff_time = time.time() - (hours * 3600)
            
            with self._metrics_lock:
                # Filter historical data
                historical_executions = [
                    m for m in self.execution_metrics 
                    if m['timestamp'] >= cutoff_time
                ]
                
                historical_discoveries = [
                    d for d in self.pattern_discoveries 
                    if d['timestamp'] >= cutoff_time
                ]
                
                historical_system = [
                    s for s in self.system_metrics 
                    if s['timestamp'] >= cutoff_time
                ]
                
                return {
                    'time_range_hours': hours,
                    'executions': historical_executions,
                    'discoveries': historical_discoveries,
                    'system_metrics': historical_system,
                    'summary': {
                        'total_executions': len(historical_executions),
                        'total_discoveries': len(historical_discoveries),
                        'avg_execution_time': sum(m['execution_time_ms'] for m in historical_executions) / len(historical_executions) if historical_executions else 0
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return {}
