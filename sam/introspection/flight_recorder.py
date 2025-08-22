"""
Flight Recorder - Dynamic Trace Logging for SAM
===============================================

Structured logging framework for capturing SAM's step-by-step reasoning process.
Provides thread-safe, high-performance logging with unified JSON schema.

Author: SAM Development Team
Version: 1.0.0
"""

import json
import time
import uuid
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
import logging


class ReasoningStep(Enum):
    """Types of reasoning steps in SAM's cognitive process."""
    QUERY_RECEIVED = "query_received"
    MEMORY_RETRIEVAL = "memory_retrieval"
    CONTEXT_ASSEMBLY = "context_assembly"
    AGENT_ZERO_PLANNING = "agent_zero_planning"
    TOOL_EXECUTION = "tool_execution"
    CODE_INTERPRETER = "code_interpreter"
    MODEL_INFERENCE = "model_inference"
    RESPONSE_GENERATION = "response_generation"
    QUALITY_CHECK = "quality_check"
    RESPONSE_SENT = "response_sent"


class TraceLevel(Enum):
    """Trace detail levels."""
    CRITICAL = "critical"  # Major reasoning steps only
    DETAILED = "detailed"  # Include sub-steps
    VERBOSE = "verbose"    # Include all internal operations


@dataclass
class CognitiveVector:
    """Represents neural activation state at a cognitive moment."""
    step_id: str
    vector_data: List[float]
    dimension: int
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceEvent:
    """Single event in SAM's reasoning trace."""
    event_id: str
    session_id: str
    step_type: ReasoningStep
    timestamp: float
    duration_ms: Optional[float]
    component: str  # e.g., "memory", "agent_zero", "code_interpreter"
    operation: str  # e.g., "retrieve_context", "plan_action", "execute_code"
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    metadata: Dict[str, Any]
    cognitive_vector: Optional[CognitiveVector] = None
    parent_event_id: Optional[str] = None
    thread_id: str = field(default_factory=lambda: str(threading.get_ident()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['step_type'] = self.step_type.value
        return result


@dataclass
class ReasoningTrace:
    """Complete reasoning trace for a query session."""
    session_id: str
    query: str
    response: str
    start_time: float
    end_time: float
    total_duration_ms: float
    events: List[TraceEvent]
    cognitive_trajectory: List[CognitiveVector]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'session_id': self.session_id,
            'query': self.query,
            'response': self.response,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'total_duration_ms': self.total_duration_ms,
            'events': [event.to_dict() for event in self.events],
            'cognitive_trajectory': [asdict(cv) for cv in self.cognitive_trajectory],
            'metadata': self.metadata
        }


class FlightRecorder:
    """
    Thread-safe flight recorder for capturing SAM's reasoning process.
    
    Features:
    - High-performance structured logging
    - Thread-safe operation
    - Configurable trace levels
    - Automatic session management
    - Neural activation capture
    """
    
    def __init__(self, 
                 trace_level: TraceLevel = TraceLevel.DETAILED,
                 max_sessions: int = 100,
                 auto_save: bool = True,
                 save_directory: str = "traces"):
        """
        Initialize the Flight Recorder.
        
        Args:
            trace_level: Level of detail to capture
            max_sessions: Maximum number of sessions to keep in memory
            auto_save: Whether to automatically save traces to disk
            save_directory: Directory to save trace files
        """
        self.trace_level = trace_level
        self.max_sessions = max_sessions
        self.auto_save = auto_save
        self.save_directory = Path(save_directory)
        
        # Thread-safe storage
        self._lock = threading.RLock()
        self._active_sessions: Dict[str, ReasoningTrace] = {}
        self._completed_sessions: Dict[str, ReasoningTrace] = {}
        self._session_events: Dict[str, List[TraceEvent]] = {}
        
        # Create save directory
        if self.auto_save:
            self.save_directory.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(f"{__name__}.FlightRecorder")
        self.logger.info(f"🛩️ Flight Recorder initialized - Level: {trace_level.value}")
    
    def start_session(self, query: str, session_id: Optional[str] = None) -> str:
        """
        Start a new reasoning session.
        
        Args:
            query: The input query
            session_id: Optional custom session ID
            
        Returns:
            str: Session ID
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        with self._lock:
            start_time = time.time()
            
            trace = ReasoningTrace(
                session_id=session_id,
                query=query,
                response="",  # Will be filled when session ends
                start_time=start_time,
                end_time=0.0,
                total_duration_ms=0.0,
                events=[],
                cognitive_trajectory=[],
                metadata={
                    'trace_level': self.trace_level.value,
                    'thread_id': str(threading.get_ident())
                }
            )
            
            self._active_sessions[session_id] = trace
            self._session_events[session_id] = []
            
            # Log session start
            self._log_event(
                session_id=session_id,
                step_type=ReasoningStep.QUERY_RECEIVED,
                component="flight_recorder",
                operation="start_session",
                input_data={"query": query},
                output_data={"session_id": session_id}
            )
        
        self.logger.info(f"🚀 Started reasoning session: {session_id}")
        return session_id
    
    def end_session(self, session_id: str, response: str):
        """
        End a reasoning session.
        
        Args:
            session_id: Session to end
            response: Final response generated
        """
        with self._lock:
            if session_id not in self._active_sessions:
                self.logger.warning(f"Session {session_id} not found")
                return
            
            trace = self._active_sessions[session_id]
            end_time = time.time()
            
            # Update trace
            trace.response = response
            trace.end_time = end_time
            trace.total_duration_ms = (end_time - trace.start_time) * 1000
            trace.events = self._session_events[session_id].copy()
            
            # Log session end
            self._log_event(
                session_id=session_id,
                step_type=ReasoningStep.RESPONSE_SENT,
                component="flight_recorder",
                operation="end_session",
                input_data={"response": response},
                output_data={"duration_ms": trace.total_duration_ms}
            )
            
            # Move to completed sessions
            self._completed_sessions[session_id] = trace
            del self._active_sessions[session_id]
            del self._session_events[session_id]
            
            # Auto-save if enabled
            if self.auto_save:
                self._save_trace(trace)
            
            # Cleanup old sessions
            self._cleanup_old_sessions()
        
        self.logger.info(f"✅ Completed reasoning session: {session_id} ({trace.total_duration_ms:.1f}ms)")
    
    def log_step(self,
                 session_id: str,
                 step_type: ReasoningStep,
                 component: str,
                 operation: str,
                 input_data: Dict[str, Any] = None,
                 output_data: Dict[str, Any] = None,
                 metadata: Dict[str, Any] = None,
                 cognitive_vector: Optional[List[float]] = None,
                 parent_event_id: Optional[str] = None) -> str:
        """
        Log a reasoning step.
        
        Args:
            session_id: Session ID
            step_type: Type of reasoning step
            component: Component performing the step
            operation: Specific operation
            input_data: Input data for the step
            output_data: Output data from the step
            metadata: Additional metadata
            cognitive_vector: Neural activation vector
            parent_event_id: Parent event ID for hierarchical tracing
            
        Returns:
            str: Event ID
        """
        return self._log_event(
            session_id=session_id,
            step_type=step_type,
            component=component,
            operation=operation,
            input_data=input_data or {},
            output_data=output_data or {},
            metadata=metadata or {},
            cognitive_vector=cognitive_vector,
            parent_event_id=parent_event_id
        )
    
    def _log_event(self,
                   session_id: str,
                   step_type: ReasoningStep,
                   component: str,
                   operation: str,
                   input_data: Dict[str, Any],
                   output_data: Dict[str, Any],
                   metadata: Dict[str, Any] = None,
                   cognitive_vector: Optional[List[float]] = None,
                   parent_event_id: Optional[str] = None) -> str:
        """Internal method to log an event."""
        event_id = str(uuid.uuid4())
        timestamp = time.time()
        
        # Create cognitive vector if provided
        cog_vector = None
        if cognitive_vector:
            cog_vector = CognitiveVector(
                step_id=event_id,
                vector_data=cognitive_vector,
                dimension=len(cognitive_vector),
                timestamp=timestamp,
                metadata=metadata or {}
            )
        
        # Create event
        event = TraceEvent(
            event_id=event_id,
            session_id=session_id,
            step_type=step_type,
            timestamp=timestamp,
            duration_ms=None,  # Will be calculated if needed
            component=component,
            operation=operation,
            input_data=input_data,
            output_data=output_data,
            metadata=metadata or {},
            cognitive_vector=cog_vector,
            parent_event_id=parent_event_id
        )
        
        with self._lock:
            if session_id in self._session_events:
                self._session_events[session_id].append(event)
                
                # Add to cognitive trajectory if vector provided
                if cog_vector and session_id in self._active_sessions:
                    self._active_sessions[session_id].cognitive_trajectory.append(cog_vector)
        
        return event_id
    
    def get_session_trace(self, session_id: str) -> Optional[ReasoningTrace]:
        """Get trace for a specific session."""
        with self._lock:
            if session_id in self._completed_sessions:
                return self._completed_sessions[session_id]
            elif session_id in self._active_sessions:
                # Return current state of active session
                trace = self._active_sessions[session_id]
                trace.events = self._session_events[session_id].copy()
                return trace
            return None
    
    def get_all_sessions(self) -> List[str]:
        """Get list of all session IDs."""
        with self._lock:
            return list(self._completed_sessions.keys()) + list(self._active_sessions.keys())
    
    def _save_trace(self, trace: ReasoningTrace):
        """Save trace to disk."""
        try:
            filename = f"trace_{trace.session_id}_{int(trace.start_time)}.json"
            filepath = self.save_directory / filename
            
            with open(filepath, 'w') as f:
                json.dump(trace.to_dict(), f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save trace {trace.session_id}: {e}")
    
    def _cleanup_old_sessions(self):
        """Remove old sessions to prevent memory bloat."""
        if len(self._completed_sessions) > self.max_sessions:
            # Remove oldest sessions
            sorted_sessions = sorted(
                self._completed_sessions.items(),
                key=lambda x: x[1].start_time
            )
            
            to_remove = len(sorted_sessions) - self.max_sessions
            for i in range(to_remove):
                session_id = sorted_sessions[i][0]
                del self._completed_sessions[session_id]


# Global flight recorder instance
_flight_recorder: Optional[FlightRecorder] = None


def get_flight_recorder() -> FlightRecorder:
    """Get the global flight recorder instance."""
    global _flight_recorder
    if _flight_recorder is None:
        _flight_recorder = FlightRecorder()
    return _flight_recorder


def initialize_flight_recorder(trace_level: TraceLevel = TraceLevel.DETAILED,
                              max_sessions: int = 100,
                              auto_save: bool = True,
                              save_directory: str = "traces") -> FlightRecorder:
    """Initialize the global flight recorder."""
    global _flight_recorder
    _flight_recorder = FlightRecorder(
        trace_level=trace_level,
        max_sessions=max_sessions,
        auto_save=auto_save,
        save_directory=save_directory
    )
    return _flight_recorder


# Instrumentation Decorators and Context Managers
import functools
from contextlib import contextmanager


def trace_step(step_type: ReasoningStep,
               component: str,
               operation: str = None,
               capture_args: bool = True,
               capture_result: bool = True):
    """
    Decorator to automatically trace function execution.

    Args:
        step_type: Type of reasoning step
        component: Component name
        operation: Operation name (defaults to function name)
        capture_args: Whether to capture function arguments
        capture_result: Whether to capture function result
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            recorder = get_flight_recorder()
            op_name = operation or func.__name__

            # Try to extract session_id from arguments
            session_id = None
            if args and hasattr(args[0], 'session_id'):
                session_id = args[0].session_id
            elif 'session_id' in kwargs:
                session_id = kwargs['session_id']

            if not session_id:
                # If no session_id found, just execute function normally
                return func(*args, **kwargs)

            # Prepare input data
            input_data = {}
            if capture_args:
                input_data = {
                    'args': [str(arg)[:100] for arg in args[1:]],  # Skip self, truncate long args
                    'kwargs': {k: str(v)[:100] for k, v in kwargs.items() if k != 'session_id'}
                }

            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000

                # Prepare output data
                output_data = {'success': True, 'duration_ms': duration_ms}
                if capture_result and result is not None:
                    output_data['result'] = str(result)[:200]  # Truncate long results

                # Log the step
                recorder.log_step(
                    session_id=session_id,
                    step_type=step_type,
                    component=component,
                    operation=op_name,
                    input_data=input_data,
                    output_data=output_data
                )

                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000

                # Log the error
                recorder.log_step(
                    session_id=session_id,
                    step_type=step_type,
                    component=component,
                    operation=op_name,
                    input_data=input_data,
                    output_data={
                        'success': False,
                        'error': str(e),
                        'duration_ms': duration_ms
                    }
                )

                raise

        return wrapper
    return decorator


@contextmanager
def trace_context(session_id: str,
                  step_type: ReasoningStep,
                  component: str,
                  operation: str,
                  input_data: Dict[str, Any] = None,
                  metadata: Dict[str, Any] = None):
    """
    Context manager for tracing code blocks.

    Args:
        session_id: Session ID
        step_type: Type of reasoning step
        component: Component name
        operation: Operation name
        input_data: Input data
        metadata: Additional metadata
    """
    recorder = get_flight_recorder()
    start_time = time.time()

    event_id = recorder.log_step(
        session_id=session_id,
        step_type=step_type,
        component=component,
        operation=f"{operation}_start",
        input_data=input_data or {},
        metadata=metadata or {}
    )

    try:
        yield event_id

        duration_ms = (time.time() - start_time) * 1000
        recorder.log_step(
            session_id=session_id,
            step_type=step_type,
            component=component,
            operation=f"{operation}_complete",
            input_data={},
            output_data={'success': True, 'duration_ms': duration_ms},
            parent_event_id=event_id
        )

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        recorder.log_step(
            session_id=session_id,
            step_type=step_type,
            component=component,
            operation=f"{operation}_error",
            input_data={},
            output_data={
                'success': False,
                'error': str(e),
                'duration_ms': duration_ms
            },
            parent_event_id=event_id
        )
        raise


class TraceSession:
    """Helper class for managing trace sessions."""

    def __init__(self, query: str, session_id: Optional[str] = None):
        self.session_id = session_id
        self.query = query
        self.recorder = get_flight_recorder()

    def __enter__(self):
        self.session_id = self.recorder.start_session(self.query, self.session_id)
        return self.session_id

    def __exit__(self, exc_type, exc_val, exc_tb):
        response = "Error occurred" if exc_type else "Session completed"
        self.recorder.end_session(self.session_id, response)

    def set_response(self, response: str):
        """Set the final response for the session."""
        self.response = response

    def log_step(self, step_type: ReasoningStep, component: str, operation: str, **kwargs):
        """Log a step in this session."""
        return self.recorder.log_step(
            session_id=self.session_id,
            step_type=step_type,
            component=component,
            operation=operation,
            **kwargs
        )
