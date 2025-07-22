"""
TPV Monitor Module for SAM
Phase 1 - Active Monitoring & Passive Control Integration

This module provides thinking process verification monitoring capabilities
with integrated dissonance detection.
"""

import logging
import time
import uuid
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

# Import the dissonance monitor
from ..dissonance_monitor import DissonanceMonitor, DissonanceScore, DissonanceCalculationMode

logger = logging.getLogger(__name__)

@dataclass
class ReasoningStep:
    """Individual step in the reasoning process."""
    step_number: int
    text_content: str
    tpv_score: float
    dissonance_score: Optional[float] = None
    dissonance_metadata: Optional[Dict[str, Any]] = None
    token_count: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'step_number': self.step_number,
            'text_content': self.text_content,
            'tpv_score': self.tpv_score,
            'dissonance_score': self.dissonance_score,
            'dissonance_metadata': self.dissonance_metadata,
            'token_count': self.token_count,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }

@dataclass
class ReasoningTrace:
    """Complete trace of a reasoning session."""
    query_id: str
    original_query: str
    steps: List[ReasoningStep] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    current_score: float = 0.0
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_step(self, step: ReasoningStep):
        """Add a reasoning step to the trace."""
        self.steps.append(step)
        self.current_score = step.tpv_score
    
    def get_progress_percentage(self) -> float:
        """Get progress as percentage (0-100)."""
        return self.current_score * 100
    
    def get_dissonance_history(self) -> List[float]:
        """Get history of dissonance scores."""
        return [step.dissonance_score for step in self.steps if step.dissonance_score is not None]
    
    def get_latest_dissonance(self) -> Optional[float]:
        """Get the most recent dissonance score."""
        dissonance_scores = self.get_dissonance_history()
        return dissonance_scores[-1] if dissonance_scores else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'query_id': self.query_id,
            'original_query': self.original_query,
            'steps': [step.to_dict() for step in self.steps],
            'start_time': self.start_time,
            'end_time': self.end_time,
            'current_score': self.current_score,
            'is_active': self.is_active,
            'metadata': self.metadata
        }

class TPVMonitor:
    """
    Thinking Process Verification Monitor with integrated dissonance detection.
    
    Monitors reasoning progress and cognitive dissonance in real-time.
    """
    
    def __init__(self, 
                 enable_dissonance_monitoring: bool = True,
                 dissonance_config: Optional[Dict[str, Any]] = None):
        """
        Initialize TPV Monitor.
        
        Args:
            enable_dissonance_monitoring: Whether to enable dissonance monitoring
            dissonance_config: Configuration for dissonance monitor
        """
        self.enable_dissonance_monitoring = enable_dissonance_monitoring
        self.active_traces: Dict[str, ReasoningTrace] = {}
        self.completed_traces: Dict[str, ReasoningTrace] = {}
        self.is_initialized = False
        
        # Initialize dissonance monitor if enabled
        self.dissonance_monitor: Optional[DissonanceMonitor] = None
        if enable_dissonance_monitoring:
            self._initialize_dissonance_monitor(dissonance_config or {})
        
        # Performance tracking
        self.total_sessions = 0
        self.total_steps = 0
        
        logger.info(f"TPVMonitor initialized (dissonance_monitoring={enable_dissonance_monitoring})")
    
    def _initialize_dissonance_monitor(self, config: Dict[str, Any]):
        """
        Initialize the dissonance monitoring system.
        
        Args:
            config: Configuration for dissonance monitor
        """
        try:
            # Default configuration
            default_config = {
                'vocab_size': 32000,  # Default for many models
                'calculation_mode': DissonanceCalculationMode.ENTROPY,
                'fallback_mode': True,
                'enable_profiling': True
            }
            default_config.update(config)
            
            self.dissonance_monitor = DissonanceMonitor(
                vocab_size=default_config['vocab_size'],
                calculation_mode=default_config['calculation_mode'],
                fallback_mode=default_config['fallback_mode'],
                enable_profiling=default_config['enable_profiling']
            )
            
            logger.info("Dissonance monitor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize dissonance monitor: {e}")
            self.enable_dissonance_monitoring = False
    
    def initialize(self) -> bool:
        """
        Initialize the TPV monitoring system.
        
        Returns:
            True if initialization successful
        """
        try:
            # Perform any additional initialization
            self.is_initialized = True
            logger.info("TPV Monitor initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"TPV Monitor initialization failed: {e}")
            return False
    
    def start_monitoring(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start monitoring a new reasoning session.
        
        Args:
            query: The original query being processed
            metadata: Optional metadata for the session
            
        Returns:
            Unique query ID for this monitoring session
        """
        query_id = str(uuid.uuid4())
        
        trace = ReasoningTrace(
            query_id=query_id,
            original_query=query,
            metadata=metadata or {}
        )
        
        self.active_traces[query_id] = trace
        self.total_sessions += 1
        
        logger.info(f"Started TPV monitoring for query: {query_id}")
        return query_id
    
    def predict_progress(self, 
                        current_text: str, 
                        query_id: str, 
                        token_count: int = 0,
                        logits: Optional[Any] = None) -> float:
        """
        Predict reasoning progress and calculate dissonance.
        
        Args:
            current_text: Current generated text
            query_id: Query ID for this session
            token_count: Number of tokens generated
            logits: Model logits for dissonance calculation (if available)
            
        Returns:
            TPV progress score (0.0 to 1.0)
        """
        if query_id not in self.active_traces:
            logger.warning(f"Query ID {query_id} not found in active traces")
            return 0.0
        
        trace = self.active_traces[query_id]
        
        # Calculate TPV score (simplified implementation)
        tpv_score = self._calculate_tpv_score(current_text, trace)
        
        # Calculate dissonance score if enabled and logits available
        dissonance_score = None
        dissonance_metadata = None
        
        if self.enable_dissonance_monitoring and self.dissonance_monitor and logits is not None:
            try:
                dissonance_result = self.dissonance_monitor.calculate_dissonance(logits)
                dissonance_score = dissonance_result.score
                dissonance_metadata = dissonance_result.metadata
            except Exception as e:
                logger.warning(f"Dissonance calculation failed: {e}")
        
        # Create reasoning step
        step = ReasoningStep(
            step_number=len(trace.steps) + 1,
            text_content=current_text,
            tpv_score=tpv_score,
            dissonance_score=dissonance_score,
            dissonance_metadata=dissonance_metadata,
            token_count=token_count
        )
        
        trace.add_step(step)
        self.total_steps += 1
        
        dissonance_str = f"{dissonance_score:.3f}" if dissonance_score is not None else "N/A"
        logger.debug(f"TPV step {step.step_number}: score={tpv_score:.3f}, dissonance={dissonance_str}")
        
        return tpv_score
    
    def _calculate_tpv_score(self, current_text: str, trace: ReasoningTrace) -> float:
        """
        Calculate TPV progress score (simplified implementation).
        
        Args:
            current_text: Current generated text
            trace: Reasoning trace
            
        Returns:
            TPV score between 0.0 and 1.0
        """
        # Simplified TPV calculation based on text length and step count
        # In a real implementation, this would use a trained model
        
        text_length = len(current_text.split())
        step_count = len(trace.steps)
        
        # Basic heuristic: progress increases with text length and steps
        length_score = min(1.0, text_length / 100.0)  # Normalize to 100 words
        step_score = min(1.0, step_count / 10.0)      # Normalize to 10 steps
        
        # Combine scores
        combined_score = (length_score * 0.7 + step_score * 0.3)
        
        # Add some variation to simulate real TPV behavior
        import random
        variation = random.uniform(-0.05, 0.05)
        final_score = max(0.0, min(1.0, combined_score + variation))
        
        return final_score
    
    def stop_monitoring(self, query_id: str) -> Optional[ReasoningTrace]:
        """
        Stop monitoring a reasoning session.
        
        Args:
            query_id: Query ID to stop monitoring
            
        Returns:
            Final reasoning trace or None if not found
        """
        if query_id not in self.active_traces:
            logger.warning(f"Query ID {query_id} not found in active traces")
            return None
        
        trace = self.active_traces.pop(query_id)
        trace.end_time = time.time()
        trace.is_active = False
        
        # Store completed trace
        self.completed_traces[query_id] = trace
        
        logger.info(f"Stopped TPV monitoring for query: {query_id} "
                   f"({len(trace.steps)} steps, final_score={trace.current_score:.3f})")
        
        return trace
    
    def get_trace(self, query_id: str) -> Optional[ReasoningTrace]:
        """
        Get reasoning trace for a query.
        
        Args:
            query_id: Query ID
            
        Returns:
            Reasoning trace or None if not found
        """
        # Check active traces first
        if query_id in self.active_traces:
            return self.active_traces[query_id]
        
        # Check completed traces
        if query_id in self.completed_traces:
            return self.completed_traces[query_id]
        
        return None
    
    def get_active_queries(self) -> List[str]:
        """
        Get list of active query IDs.
        
        Returns:
            List of active query IDs
        """
        return list(self.active_traces.keys())
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get TPV monitor status and statistics.
        
        Returns:
            Status dictionary
        """
        status = {
            'initialized': self.is_initialized,
            'active_sessions': len(self.active_traces),
            'completed_sessions': len(self.completed_traces),
            'total_sessions': self.total_sessions,
            'total_steps': self.total_steps,
            'dissonance_monitoring_enabled': self.enable_dissonance_monitoring
        }
        
        # Add dissonance monitor stats if available
        if self.dissonance_monitor:
            status['dissonance_stats'] = self.dissonance_monitor.get_performance_stats()
        
        return status
    
    def cleanup_old_traces(self, max_age_hours: float = 24.0):
        """
        Clean up old completed traces to manage memory.
        
        Args:
            max_age_hours: Maximum age in hours for keeping traces
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        to_remove = []
        for query_id, trace in self.completed_traces.items():
            if trace.end_time and (current_time - trace.end_time) > max_age_seconds:
                to_remove.append(query_id)
        
        for query_id in to_remove:
            del self.completed_traces[query_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old traces")
    
    def get_dissonance_analysis(self, query_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed dissonance analysis for a query.
        
        Args:
            query_id: Query ID
            
        Returns:
            Dissonance analysis or None if not available
        """
        trace = self.get_trace(query_id)
        if not trace:
            return None
        
        dissonance_scores = trace.get_dissonance_history()
        if not dissonance_scores:
            return None
        
        # Calculate statistics
        scores_array = np.array(dissonance_scores)
        
        analysis = {
            'query_id': query_id,
            'total_steps': len(dissonance_scores),
            'mean_dissonance': float(np.mean(scores_array)),
            'max_dissonance': float(np.max(scores_array)),
            'min_dissonance': float(np.min(scores_array)),
            'std_dissonance': float(np.std(scores_array)),
            'dissonance_trend': self._calculate_trend(scores_array),
            'high_dissonance_steps': [i for i, score in enumerate(dissonance_scores) if score > 0.8],
            'dissonance_spikes': self._find_dissonance_spikes(scores_array)
        }
        
        return analysis
    
    def _calculate_trend(self, scores: Any) -> str:
        """Calculate trend in dissonance scores."""
        if len(scores) < 2:
            return "insufficient_data"
        
        # Simple linear trend
        x = list(range(len(scores)))
        correlation = float(np.corrcoef(x, scores)[0, 1])
        
        if correlation > 0.3:
            return "increasing"
        elif correlation < -0.3:
            return "decreasing"
        else:
            return "stable"
    
    def _find_dissonance_spikes(self, scores: Any) -> List[Dict[str, Any]]:
        """Find significant spikes in dissonance scores."""
        if len(scores) < 3:
            return []
        
        spikes = []
        threshold = float(np.mean(scores) + 2 * np.std(scores))
        
        for i, score in enumerate(scores):
            if score > threshold and score > 0.7:  # High absolute threshold too
                spikes.append({
                    'step': i,
                    'score': float(score),
                    'severity': 'high' if score > 0.9 else 'medium'
                })
        
        return spikes
