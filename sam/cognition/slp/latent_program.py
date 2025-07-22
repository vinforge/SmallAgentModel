"""
Latent Program Data Structures
==============================

Core data structures for the SLP system including the enhanced LatentProgram
class with configuration parameters, metadata, and lifecycle management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
import uuid
import json


@dataclass
class LatentProgram:
    """
    Enhanced data structure for storing reasoning patterns and their metadata.
    
    This represents a "cognitive program" - a successful reasoning pattern
    that can be reused for similar tasks.
    """
    
    # Core identification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    signature: Dict[str, Any] = field(default_factory=dict)
    reasoning_trace: Dict[str, Any] = field(default_factory=dict)
    
    # Configuration parameters that worked well
    tpv_config: Dict[str, Any] = field(default_factory=dict)
    active_profile: str = "default"
    prompt_template_used: str = ""
    
    # NEW: Context & Constraints
    context_requirements: Dict[str, Any] = field(default_factory=dict)
    execution_constraints: Dict[str, Any] = field(default_factory=dict)
    
    # Performance & Usage Metadata
    usage_count: int = 0
    avg_latency_ms: float = 0.0
    avg_token_count: int = 0
    user_feedback_score: float = 0.0
    
    # NEW: Enhanced Lifecycle Management
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: datetime = field(default_factory=datetime.utcnow)
    success_rate: float = 1.0
    version: int = 1
    parent_program_id: Optional[str] = None
    
    # Status and flags
    is_active: bool = True
    is_experimental: bool = True  # New programs start as experimental
    confidence_score: float = 0.5  # Confidence in this program's effectiveness
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'id': self.id,
            'signature': self.signature,
            'reasoning_trace': self.reasoning_trace,
            'tpv_config': self.tpv_config,
            'active_profile': self.active_profile,
            'prompt_template_used': self.prompt_template_used,
            'context_requirements': self.context_requirements,
            'execution_constraints': self.execution_constraints,
            'usage_count': self.usage_count,
            'avg_latency_ms': self.avg_latency_ms,
            'avg_token_count': self.avg_token_count,
            'user_feedback_score': self.user_feedback_score,
            'created_at': self.created_at.isoformat(),
            'last_used': self.last_used.isoformat(),
            'success_rate': self.success_rate,
            'version': self.version,
            'parent_program_id': self.parent_program_id,
            'is_active': self.is_active,
            'is_experimental': self.is_experimental,
            'confidence_score': self.confidence_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LatentProgram':
        """Create from dictionary (for loading from storage)."""
        # Handle datetime fields
        created_at = datetime.fromisoformat(data.get('created_at', datetime.utcnow().isoformat()))
        last_used = datetime.fromisoformat(data.get('last_used', datetime.utcnow().isoformat()))
        
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            signature=data.get('signature', {}),
            reasoning_trace=data.get('reasoning_trace', {}),
            tpv_config=data.get('tpv_config', {}),
            active_profile=data.get('active_profile', 'default'),
            prompt_template_used=data.get('prompt_template_used', ''),
            context_requirements=data.get('context_requirements', {}),
            execution_constraints=data.get('execution_constraints', {}),
            usage_count=data.get('usage_count', 0),
            avg_latency_ms=data.get('avg_latency_ms', 0.0),
            avg_token_count=data.get('avg_token_count', 0),
            user_feedback_score=data.get('user_feedback_score', 0.0),
            created_at=created_at,
            last_used=last_used,
            success_rate=data.get('success_rate', 1.0),
            version=data.get('version', 1),
            parent_program_id=data.get('parent_program_id'),
            is_active=data.get('is_active', True),
            is_experimental=data.get('is_experimental', True),
            confidence_score=data.get('confidence_score', 0.5)
        )
    
    def update_performance_metrics(self, latency_ms: float, token_count: int, 
                                 success: bool = True, feedback_score: Optional[float] = None):
        """Update performance metrics after execution."""
        self.usage_count += 1
        self.last_used = datetime.utcnow()
        
        # Update running averages
        if self.usage_count == 1:
            self.avg_latency_ms = latency_ms
            self.avg_token_count = token_count
        else:
            # Exponential moving average with alpha = 0.1
            alpha = 0.1
            self.avg_latency_ms = (1 - alpha) * self.avg_latency_ms + alpha * latency_ms
            self.avg_token_count = int((1 - alpha) * self.avg_token_count + alpha * token_count)
        
        # Update success rate
        if self.usage_count == 1:
            self.success_rate = 1.0 if success else 0.0
        else:
            # Update success rate with exponential moving average
            self.success_rate = (1 - alpha) * self.success_rate + alpha * (1.0 if success else 0.0)
        
        # Update feedback score if provided
        if feedback_score is not None:
            if self.user_feedback_score == 0.0:
                self.user_feedback_score = feedback_score
            else:
                self.user_feedback_score = (1 - alpha) * self.user_feedback_score + alpha * feedback_score
        
        # Update confidence score based on usage and success
        self._update_confidence_score()
        
        # Graduate from experimental status after sufficient usage
        if self.usage_count >= 5 and self.success_rate > 0.7:
            self.is_experimental = False
    
    def _update_confidence_score(self):
        """Update confidence score based on performance metrics."""
        # Base confidence on success rate, usage count, and feedback
        usage_factor = min(self.usage_count / 10.0, 1.0)  # Max at 10 uses
        success_factor = self.success_rate
        feedback_factor = max(self.user_feedback_score / 5.0, 0.1) if self.user_feedback_score > 0 else 0.5
        
        self.confidence_score = (usage_factor * 0.3 + success_factor * 0.5 + feedback_factor * 0.2)
        self.confidence_score = max(0.0, min(1.0, self.confidence_score))


@dataclass
class ExecutionResult:
    """Result of executing a latent program."""
    response: str
    quality_score: float
    execution_time_ms: float
    program_used: str
    success: bool = True
    error_message: Optional[str] = None
    token_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'response': self.response,
            'quality_score': self.quality_score,
            'execution_time_ms': self.execution_time_ms,
            'program_used': self.program_used,
            'success': self.success,
            'error_message': self.error_message,
            'token_count': self.token_count
        }


@dataclass
class ValidationResult:
    """Result of program safety validation."""
    is_safe: bool
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'is_safe': self.is_safe,
            'warnings': self.warnings,
            'recommendations': self.recommendations,
            'risk_score': self.risk_score
        }


class ProgramExecutionError(Exception):
    """Exception raised when program execution fails."""
    pass


class ProgramValidationError(Exception):
    """Exception raised when program validation fails."""
    pass
