"""
TPV Core Module for SAM
Phase 1 - Core TPV Infrastructure

This module provides the core TPV (Thinking Process Verification) infrastructure
with integrated dissonance monitoring capabilities.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .tpv_monitor import TPVMonitor, ReasoningTrace
from .tpv_controller import ReasoningController, ControlMode, ControlConfig
from ..dissonance_monitor import DissonanceMonitor, DissonanceCalculationMode

logger = logging.getLogger(__name__)

@dataclass
class TPVResult:
    """Result of TPV processing."""
    query_id: str
    final_score: float
    dissonance_score: Optional[float]
    control_decision: str
    trace: ReasoningTrace
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'query_id': self.query_id,
            'final_score': self.final_score,
            'dissonance_score': self.dissonance_score,
            'control_decision': self.control_decision,
            'trace': self.trace.to_dict(),
            'metadata': self.metadata
        }

class TPVCore:
    """
    Core TPV system with integrated dissonance monitoring.
    
    Coordinates between monitoring, control, and dissonance detection
    to provide comprehensive reasoning process verification.
    """
    
    def __init__(self, 
                 control_mode: ControlMode = ControlMode.PASSIVE,
                 enable_dissonance: bool = True,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize TPV Core system.
        
        Args:
            control_mode: Control mode for reasoning controller
            enable_dissonance: Whether to enable dissonance monitoring
            config: Configuration dictionary
        """
        self.config = config or {}
        self.enable_dissonance = enable_dissonance
        
        # Initialize components
        self.monitor = self._initialize_monitor()
        self.controller = self._initialize_controller(control_mode)
        
        # System state
        self.is_initialized = False
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.total_sessions = 0
        self.successful_sessions = 0
        self.failed_sessions = 0
        
        logger.info(f"TPVCore initialized: control_mode={control_mode.value}, "
                   f"dissonance_enabled={enable_dissonance}")
    
    def _initialize_monitor(self) -> TPVMonitor:
        """Initialize the TPV monitor with dissonance capabilities."""
        dissonance_config = self.config.get('dissonance_config', {})
        
        # Set default dissonance configuration
        default_dissonance_config = {
            'vocab_size': 32000,
            'calculation_mode': DissonanceCalculationMode.ENTROPY,
            'fallback_mode': True,
            'enable_profiling': True
        }
        default_dissonance_config.update(dissonance_config)
        
        monitor = TPVMonitor(
            enable_dissonance_monitoring=self.enable_dissonance,
            dissonance_config=default_dissonance_config
        )
        
        return monitor
    
    def _initialize_controller(self, control_mode: ControlMode) -> ReasoningController:
        """Initialize the reasoning controller."""
        control_config_dict = self.config.get('control_config', {})
        
        # Set default control configuration
        default_control_config = {
            'completion_threshold': 0.92,
            'max_tokens': 500,
            'min_steps': 2,
            'plateau_patience': 3,
            'plateau_threshold': 0.005,
            'dissonance_threshold': 0.85,
            'dissonance_patience': 4,
            'enable_dissonance_control': self.enable_dissonance
        }
        default_control_config.update(control_config_dict)
        
        control_config = ControlConfig.from_dict(default_control_config)
        
        controller = ReasoningController(
            mode=control_mode,
            config=control_config
        )
        
        return controller
    
    def initialize(self) -> bool:
        """
        Initialize the TPV system.
        
        Returns:
            True if initialization successful
        """
        try:
            # Initialize monitor
            if not self.monitor.initialize():
                logger.error("Failed to initialize TPV monitor")
                return False
            
            # Perform system checks
            if not self._perform_system_checks():
                logger.error("TPV system checks failed")
                return False
            
            self.is_initialized = True
            logger.info("TPV Core system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"TPV Core initialization failed: {e}")
            return False
    
    def _perform_system_checks(self) -> bool:
        """Perform system health checks."""
        try:
            # Test monitor functionality
            test_query_id = self.monitor.start_monitoring("system_check")
            test_score = self.monitor.predict_progress("test", test_query_id, token_count=1)
            test_trace = self.monitor.stop_monitoring(test_query_id)
            
            if test_trace is None or test_score < 0:
                logger.error("Monitor system check failed")
                return False
            
            # Test controller functionality
            should_continue = self.controller.should_continue(test_trace)
            if not isinstance(should_continue, bool):
                logger.error("Controller system check failed")
                return False
            
            logger.info("TPV system checks passed")
            return True
            
        except Exception as e:
            logger.error(f"System checks failed: {e}")
            return False
    
    def start_session(self, 
                     query: str, 
                     session_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start a new TPV session.
        
        Args:
            query: The query to process
            session_metadata: Optional metadata for the session
            
        Returns:
            Session ID
        """
        if not self.is_initialized:
            raise RuntimeError("TPV Core not initialized")
        
        # Start monitoring
        query_id = self.monitor.start_monitoring(query, session_metadata)
        
        # Track session
        self.active_sessions[query_id] = {
            'start_time': time.time(),
            'query': query,
            'metadata': session_metadata or {},
            'status': 'active'
        }
        
        self.total_sessions += 1
        logger.info(f"Started TPV session: {query_id}")
        
        return query_id
    
    def process_step(self, 
                    session_id: str, 
                    current_text: str, 
                    token_count: int = 0,
                    logits: Optional[Any] = None,
                    context: Optional[Dict[str, Any]] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Process a reasoning step.
        
        Args:
            session_id: Session ID
            current_text: Current generated text
            token_count: Number of tokens generated
            logits: Model logits for dissonance calculation
            context: Optional context for control decisions
            
        Returns:
            Tuple of (should_continue, step_info)
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        try:
            # Update monitoring
            tpv_score = self.monitor.predict_progress(
                current_text, session_id, token_count, logits
            )
            
            # Get current trace
            trace = self.monitor.get_trace(session_id)
            if not trace:
                raise RuntimeError(f"No trace found for session {session_id}")
            
            # Make control decision
            should_continue = self.controller.should_continue(trace, context)
            
            # Prepare step information
            step_info = {
                'tpv_score': tpv_score,
                'step_number': len(trace.steps),
                'should_continue': should_continue,
                'dissonance_score': trace.get_latest_dissonance(),
                'token_count': token_count
            }
            
            # Update session status
            if not should_continue:
                self.active_sessions[session_id]['status'] = 'stopping'
            
            return should_continue, step_info
            
        except Exception as e:
            logger.error(f"Error processing step for session {session_id}: {e}")
            self.active_sessions[session_id]['status'] = 'error'
            return False, {'error': str(e)}
    
    def end_session(self, session_id: str) -> TPVResult:
        """
        End a TPV session and get results.
        
        Args:
            session_id: Session ID to end
            
        Returns:
            TPVResult with final analysis
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        try:
            # Stop monitoring
            final_trace = self.monitor.stop_monitoring(session_id)
            if not final_trace:
                raise RuntimeError(f"Failed to get final trace for session {session_id}")
            
            # Get session info
            session_info = self.active_sessions.pop(session_id)
            
            # Calculate final metrics
            final_score = final_trace.current_score
            dissonance_score = final_trace.get_latest_dissonance()
            
            # Get last control decision
            control_stats = self.controller.get_control_statistics()
            last_decision = "unknown"
            if self.controller.control_history:
                last_decision = self.controller.control_history[-1].decision.value
            
            # Create result
            result = TPVResult(
                query_id=session_id,
                final_score=final_score,
                dissonance_score=dissonance_score,
                control_decision=last_decision,
                trace=final_trace,
                metadata={
                    'session_duration': time.time() - session_info['start_time'],
                    'total_steps': len(final_trace.steps),
                    'session_metadata': session_info['metadata'],
                    'control_stats': control_stats
                }
            )
            
            # Update statistics
            if session_info['status'] == 'error':
                self.failed_sessions += 1
            else:
                self.successful_sessions += 1
            
            logger.info(f"Ended TPV session: {session_id} "
                       f"(score={final_score:.3f}, steps={len(final_trace.steps)})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error ending session {session_id}: {e}")
            self.failed_sessions += 1
            
            # Create error result
            return TPVResult(
                query_id=session_id,
                final_score=0.0,
                dissonance_score=None,
                control_decision="error",
                trace=ReasoningTrace(session_id, "error"),
                metadata={'error': str(e)}
            )
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of an active session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session status or None if not found
        """
        if session_id not in self.active_sessions:
            return None
        
        session_info = self.active_sessions[session_id]
        trace = self.monitor.get_trace(session_id)
        
        status = {
            'session_id': session_id,
            'status': session_info['status'],
            'start_time': session_info['start_time'],
            'duration': time.time() - session_info['start_time'],
            'query': session_info['query']
        }
        
        if trace:
            status.update({
                'current_score': trace.current_score,
                'steps_completed': len(trace.steps),
                'latest_dissonance': trace.get_latest_dissonance(),
                'is_active': trace.is_active
            })
        
        return status
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get overall system status and statistics.
        
        Returns:
            System status dictionary
        """
        return {
            'initialized': self.is_initialized,
            'active_sessions': len(self.active_sessions),
            'total_sessions': self.total_sessions,
            'successful_sessions': self.successful_sessions,
            'failed_sessions': self.failed_sessions,
            'success_rate': (self.successful_sessions / max(1, self.total_sessions)),
            'dissonance_enabled': self.enable_dissonance,
            'monitor_status': self.monitor.get_status(),
            'controller_status': self.controller.get_status()
        }
    
    def cleanup_old_data(self, max_age_hours: float = 24.0):
        """
        Clean up old data to manage memory usage.
        
        Args:
            max_age_hours: Maximum age in hours for keeping data
        """
        # Clean up monitor traces
        self.monitor.cleanup_old_traces(max_age_hours)
        
        # Clean up controller history
        if len(self.controller.control_history) > 1000:
            self.controller.control_history = self.controller.control_history[-1000:]
        
        logger.info(f"Cleaned up data older than {max_age_hours} hours")
    
    def update_configuration(self, new_config: Dict[str, Any]):
        """
        Update system configuration.
        
        Args:
            new_config: New configuration parameters
        """
        self.config.update(new_config)
        
        # Update controller config if provided
        if 'control_config' in new_config:
            self.controller.update_config(new_config['control_config'])
        
        # Update dissonance monitor config if provided
        if 'dissonance_config' in new_config and self.monitor.dissonance_monitor:
            self.monitor.dissonance_monitor.update_config(new_config['dissonance_config'])
        
        logger.info("TPV Core configuration updated")
    
    def get_dissonance_analysis(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed dissonance analysis for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Dissonance analysis or None if not available
        """
        return self.monitor.get_dissonance_analysis(session_id)
