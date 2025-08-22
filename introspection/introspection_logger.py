"""
Introspection Logger for SAM
============================

Structured logging system that captures SAM's cognitive processes in a
machine-readable format for analysis and debugging.

Author: SAM Development Team
Version: 1.0.0
"""

import json
import time
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
from contextlib import contextmanager


class EventType(Enum):
    """Types of cognitive events that can be logged."""
    REASONING_START = "reasoning_start"
    REASONING_STEP = "reasoning_step"
    REASONING_END = "reasoning_end"
    TOOL_EXECUTION = "tool_execution"
    MEMORY_ACCESS = "memory_access"
    MEMORY_STORE = "memory_store"
    MODEL_INFERENCE = "model_inference"
    PLANNING_START = "planning_start"
    PLANNING_STEP = "planning_step"
    PLANNING_END = "planning_end"
    ERROR_OCCURRED = "error_occurred"
    DECISION_POINT = "decision_point"
    CONTEXT_SWITCH = "context_switch"
    USER_INTERACTION = "user_interaction"
    SYSTEM_STATE = "system_state"


@dataclass
class CognitiveEvent:
    """
    Represents a single cognitive event in SAM's processing.
    """
    event_id: str
    event_type: EventType
    timestamp: datetime
    session_id: str
    user_id: str
    
    # Core event data
    description: str
    details: Dict[str, Any]
    
    # Context information
    context: Dict[str, Any]
    parent_event_id: Optional[str] = None
    
    # Performance metrics
    duration_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    
    # Cognitive metrics
    confidence_score: Optional[float] = None
    complexity_score: Optional[float] = None
    reasoning_depth: Optional[int] = None
    
    # Metadata
    component: Optional[str] = None
    version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['event_type'] = self.event_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CognitiveEvent':
        """Create event from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['event_type'] = EventType(data['event_type'])
        return cls(**data)


class IntrospectionLogger:
    """
    Centralized logger for SAM's cognitive processes.
    
    Features:
    - Structured JSON logging
    - Real-time event streaming
    - Performance monitoring
    - Context tracking
    - Thread-safe operation
    """
    
    def __init__(self, 
                 log_file: Optional[str] = None,
                 session_id: Optional[str] = None,
                 user_id: str = "default",
                 enable_console: bool = True,
                 enable_file: bool = True,
                 buffer_size: int = 100):
        """
        Initialize the introspection logger.
        
        Args:
            log_file: Path to log file (auto-generated if None)
            session_id: Session identifier (auto-generated if None)
            user_id: User identifier
            enable_console: Whether to log to console
            enable_file: Whether to log to file
            buffer_size: Number of events to buffer before writing
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.user_id = user_id
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.buffer_size = buffer_size
        
        # Set up log file
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"sam_introspection_{timestamp}_{self.session_id[:8]}.jsonl"
        
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Event buffer and threading
        self._event_buffer: List[CognitiveEvent] = []
        self._buffer_lock = threading.Lock()
        
        # Context stack for nested operations
        self._context_stack: List[Dict[str, Any]] = []
        self._context_lock = threading.Lock()
        
        # Performance tracking
        self._active_events: Dict[str, float] = {}  # event_id -> start_time
        
        # Standard logger for console output
        self.logger = logging.getLogger(f"sam.introspection.{self.session_id[:8]}")
        if enable_console:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"🧠 Introspection Logger initialized (Session: {self.session_id[:8]})")
    
    def log_event(self, 
                  event_type: EventType,
                  description: str,
                  details: Optional[Dict[str, Any]] = None,
                  component: Optional[str] = None,
                  confidence_score: Optional[float] = None,
                  complexity_score: Optional[float] = None,
                  parent_event_id: Optional[str] = None) -> str:
        """
        Log a cognitive event.
        
        Args:
            event_type: Type of event
            description: Human-readable description
            details: Additional event details
            component: Component that generated the event
            confidence_score: Confidence in the operation (0-1)
            complexity_score: Complexity of the operation (0-1)
            parent_event_id: ID of parent event for nesting
            
        Returns:
            Event ID for tracking
        """
        event_id = str(uuid.uuid4())
        
        # Get current context
        with self._context_lock:
            current_context = {}
            for ctx in self._context_stack:
                current_context.update(ctx)
        
        # Create event
        event = CognitiveEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.now(),
            session_id=self.session_id,
            user_id=self.user_id,
            description=description,
            details=details or {},
            context=current_context,
            parent_event_id=parent_event_id,
            component=component,
            confidence_score=confidence_score,
            complexity_score=complexity_score
        )
        
        # Add to buffer
        with self._buffer_lock:
            self._event_buffer.append(event)
            
            # Flush buffer if full
            if len(self._event_buffer) >= self.buffer_size:
                self._flush_buffer()
        
        # Console logging
        if self.enable_console:
            level = logging.ERROR if event_type == EventType.ERROR_OCCURRED else logging.INFO
            self.logger.log(level, f"[{event_type.value}] {description}")
        
        return event_id
    
    @contextmanager
    def track_operation(self, 
                       event_type: EventType,
                       description: str,
                       component: Optional[str] = None,
                       **kwargs):
        """
        Context manager for tracking operations with automatic timing.
        
        Args:
            event_type: Type of operation
            description: Description of operation
            component: Component performing operation
            **kwargs: Additional event details
        """
        start_time = time.time()
        event_id = self.log_event(
            event_type=event_type,
            description=f"Starting: {description}",
            component=component,
            details=kwargs
        )
        
        self._active_events[event_id] = start_time
        
        try:
            yield event_id
            
            # Log successful completion
            duration_ms = (time.time() - start_time) * 1000
            self.log_event(
                event_type=EventType.REASONING_END if event_type == EventType.REASONING_START else event_type,
                description=f"Completed: {description}",
                component=component,
                details={**kwargs, "duration_ms": duration_ms, "success": True},
                parent_event_id=event_id
            )
            
        except Exception as e:
            # Log error
            duration_ms = (time.time() - start_time) * 1000
            self.log_event(
                event_type=EventType.ERROR_OCCURRED,
                description=f"Failed: {description}",
                component=component,
                details={
                    **kwargs, 
                    "duration_ms": duration_ms, 
                    "error": str(e),
                    "error_type": type(e).__name__
                },
                parent_event_id=event_id
            )
            raise
        
        finally:
            if event_id in self._active_events:
                del self._active_events[event_id]
    
    @contextmanager
    def context(self, **context_data):
        """
        Context manager for adding context to all events within a block.
        
        Args:
            **context_data: Context key-value pairs
        """
        with self._context_lock:
            self._context_stack.append(context_data)
        
        try:
            yield
        finally:
            with self._context_lock:
                if self._context_stack:
                    self._context_stack.pop()
    
    def log_reasoning_step(self, 
                          step_description: str,
                          input_data: Any,
                          output_data: Any,
                          confidence: Optional[float] = None,
                          component: str = "reasoning_engine"):
        """Log a reasoning step with input/output."""
        return self.log_event(
            event_type=EventType.REASONING_STEP,
            description=step_description,
            details={
                "input": str(input_data)[:1000],  # Truncate long inputs
                "output": str(output_data)[:1000],
                "input_type": type(input_data).__name__,
                "output_type": type(output_data).__name__
            },
            component=component,
            confidence_score=confidence
        )
    
    def log_tool_execution(self, 
                          tool_name: str,
                          parameters: Dict[str, Any],
                          result: Any,
                          success: bool,
                          duration_ms: Optional[float] = None):
        """Log tool execution."""
        return self.log_event(
            event_type=EventType.TOOL_EXECUTION,
            description=f"Executed tool: {tool_name}",
            details={
                "tool_name": tool_name,
                "parameters": parameters,
                "result": str(result)[:1000],
                "success": success,
                "duration_ms": duration_ms
            },
            component="tool_executor"
        )
    
    def log_memory_operation(self, 
                           operation: str,
                           memory_type: str,
                           data_summary: str,
                           success: bool):
        """Log memory operations."""
        event_type = EventType.MEMORY_ACCESS if operation == "read" else EventType.MEMORY_STORE
        return self.log_event(
            event_type=event_type,
            description=f"Memory {operation}: {memory_type}",
            details={
                "operation": operation,
                "memory_type": memory_type,
                "data_summary": data_summary,
                "success": success
            },
            component="memory_manager"
        )
    
    def _flush_buffer(self):
        """Flush the event buffer to file."""
        if not self.enable_file or not self._event_buffer:
            return
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                for event in self._event_buffer:
                    json.dump(event.to_dict(), f, ensure_ascii=False)
                    f.write('\n')
            
            self._event_buffer.clear()
            
        except Exception as e:
            if self.enable_console:
                self.logger.error(f"Failed to flush event buffer: {e}")
    
    def flush(self):
        """Manually flush all buffered events."""
        with self._buffer_lock:
            self._flush_buffer()
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session."""
        with self._buffer_lock:
            return {
                "session_id": self.session_id,
                "user_id": self.user_id,
                "buffered_events": len(self._event_buffer),
                "active_operations": len(self._active_events),
                "context_depth": len(self._context_stack),
                "log_file": str(self.log_file)
            }
    
    def close(self):
        """Close the logger and flush all events."""
        self.flush()
        if self.enable_console:
            self.logger.info(f"🧠 Introspection Logger closed (Session: {self.session_id[:8]})")


# Global logger instance
_introspection_logger = None

def get_introspection_logger() -> IntrospectionLogger:
    """Get the global introspection logger instance."""
    global _introspection_logger
    if _introspection_logger is None:
        _introspection_logger = IntrospectionLogger()
    return _introspection_logger


def set_introspection_logger(logger: IntrospectionLogger):
    """Set the global introspection logger instance."""
    global _introspection_logger
    _introspection_logger = logger
