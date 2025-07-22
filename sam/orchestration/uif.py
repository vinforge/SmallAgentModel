"""
Universal Interface Format (UIF) for SAM Orchestration Framework
================================================================

The UIF serves as the "standard shipping container" for data moving between skills
in the SAM Orchestration Framework. It provides type-safe, validated communication
with comprehensive error handling and state tracking.

Enhanced with Pydantic for runtime data validation to prevent state corruption.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Literal
from pydantic import BaseModel, Field, validator
from enum import Enum


class UIFStatus(str, Enum):
    """Status enumeration for UIF execution state."""
    PENDING = "pending"
    RUNNING = "running" 
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"


class SAM_UIF(BaseModel):
    """
    Universal Interface Format for SAM skill communication.
    
    This Pydantic model ensures type safety and runtime validation
    for all data flowing between skills in the orchestration framework.
    """
    
    # --- Core Identification ---
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: UIFStatus = Field(default=UIFStatus.PENDING)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    # --- Input Data ---
    input_query: str = Field(..., min_length=1, description="The user's original query")
    source_documents: List[str] = Field(default_factory=list, description="Source document references")
    user_context: Dict[str, Any] = Field(default_factory=dict, description="User session context")
    
    # --- Execution Context ---
    active_profile: str = Field(default="general", description="Active user profile")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    
    # --- Skill Communication ---
    intermediate_data: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Data shared between skills during execution"
    )
    skill_outputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Outputs from individual skills"
    )
    
    # --- Security & Vetting ---
    security_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Security and vetting information"
    )
    requires_vetting: bool = Field(default=False, description="Whether content needs vetting")
    vetted_content: Dict[str, Any] = Field(default_factory=dict, description="Vetted external content")
    
    # --- Execution Results ---
    final_response: Optional[str] = Field(default=None, description="Final response to user")
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Response confidence")
    
    # --- Error Handling ---
    error_details: Optional[str] = Field(default=None, description="Error information if failure occurs")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings")
    
    # --- Execution Tracking ---
    log_trace: List[str] = Field(default_factory=list, description="Execution trace log")
    execution_plan: List[str] = Field(default_factory=list, description="Planned skill execution sequence")
    executed_skills: List[str] = Field(default_factory=list, description="Successfully executed skills")
    
    # --- Performance Metrics ---
    start_time: Optional[str] = Field(default=None, description="Execution start timestamp")
    end_time: Optional[str] = Field(default=None, description="Execution end timestamp")
    skill_timings: Dict[str, float] = Field(default_factory=dict, description="Individual skill execution times")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
        extra = "forbid"  # Prevent adding undefined fields
    
    @validator('updated_at', always=True)
    def set_updated_at(cls, v):
        """Automatically update the timestamp when the UIF is modified."""
        return datetime.now().isoformat()
    
    @validator('confidence_score')
    def validate_confidence(cls, v):
        """Ensure confidence score is within valid range."""
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("Confidence score must be between 0.0 and 1.0")
        return v
    
    def add_log_entry(self, message: str, skill_name: Optional[str] = None) -> None:
        """
        Add an entry to the execution log trace.
        
        Args:
            message: Log message
            skill_name: Optional skill name for context
        """
        timestamp = datetime.now().isoformat()
        if skill_name:
            log_entry = f"[{timestamp}] {skill_name}: {message}"
        else:
            log_entry = f"[{timestamp}] {message}"
        
        self.log_trace.append(log_entry)
        self.updated_at = timestamp
    
    def add_warning(self, warning: str) -> None:
        """
        Add a warning message.
        
        Args:
            warning: Warning message
        """
        self.warnings.append(warning)
        self.add_log_entry(f"WARNING: {warning}")
    
    def set_error(self, error_message: str, skill_name: Optional[str] = None) -> None:
        """
        Set error state and message.
        
        Args:
            error_message: Error description
            skill_name: Optional skill that caused the error
        """
        self.status = UIFStatus.FAILURE
        self.error_details = error_message
        self.add_log_entry(f"ERROR: {error_message}", skill_name)
    
    def mark_skill_complete(self, skill_name: str, execution_time: Optional[float] = None) -> None:
        """
        Mark a skill as successfully completed.
        
        Args:
            skill_name: Name of the completed skill
            execution_time: Optional execution time in seconds
        """
        if skill_name not in self.executed_skills:
            self.executed_skills.append(skill_name)
        
        if execution_time is not None:
            self.skill_timings[skill_name] = execution_time
        
        self.add_log_entry(f"Completed successfully", skill_name)
    
    def get_skill_output(self, skill_name: str) -> Any:
        """
        Get output from a specific skill.
        
        Args:
            skill_name: Name of the skill
            
        Returns:
            Skill output or None if not found
        """
        return self.skill_outputs.get(skill_name)
    
    def set_skill_output(self, skill_name: str, output: Any) -> None:
        """
        Set output for a specific skill.
        
        Args:
            skill_name: Name of the skill
            output: Output data
        """
        self.skill_outputs[skill_name] = output
        self.add_log_entry(f"Output set: {type(output).__name__}", skill_name)
    
    def is_complete(self) -> bool:
        """Check if the UIF execution is complete (success or failure)."""
        return self.status in [UIFStatus.SUCCESS, UIFStatus.FAILURE, UIFStatus.CANCELLED]
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the execution.
        
        Returns:
            Dictionary with execution metrics and status
        """
        return {
            "task_id": self.task_id,
            "status": self.status,
            "executed_skills": self.executed_skills,
            "total_skills": len(self.execution_plan),
            "warnings_count": len(self.warnings),
            "has_errors": self.error_details is not None,
            "confidence_score": self.confidence_score,
            "execution_time_total": sum(self.skill_timings.values()) if self.skill_timings else None
        }
