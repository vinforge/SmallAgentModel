"""
Enhanced Goal Data Structure for SAM Autonomy
=============================================

This module defines the Goal data structure with enhanced safety features,
proper datetime handling, and validation for autonomous goal management.

Author: SAM Development Team
Version: 2.0.0
"""

import uuid
import logging
from datetime import datetime
from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

# Valid skill names that can be referenced in goals
# This will be populated dynamically from the SOF skill registry
VALID_SKILL_NAMES = {
    "MemoryRetrievalSkill",
    "ResponseGenerationSkill", 
    "ConflictDetectorSkill",
    "CalculatorTool",
    "AgentZeroWebBrowserTool",
    "ContentVettingSkill",
    "ImplicitKnowledgeSkill",
    "AutonomousFactualCorrectionSkill",
    "MEMOIR_EditSkill",
    "MEMOIR_RetrievalSkill",
    "MEMOIR_LearningSkill"
}

class Goal(BaseModel):
    """
    Enhanced Goal data structure for SAM's autonomous goal system.
    
    This Pydantic model ensures type safety and runtime validation for all
    autonomous goals generated and managed by SAM's Goal & Motivation Engine.
    
    Features:
    - Unique identification with UUID
    - Comprehensive status tracking
    - Priority management with validation
    - Source skill validation
    - Rich context storage
    - Proper datetime handling
    - Dependency tracking
    - Effort estimation
    """
    
    # Core identification
    goal_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the goal"
    )
    
    description: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Clear, actionable description of the goal"
    )
    
    status: Literal["pending", "active", "completed", "failed", "paused"] = Field(
        default="pending",
        description="Current status of the goal"
    )
    
    priority: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Priority score from 0.0 (lowest) to 1.0 (highest)"
    )
    
    # Source and context
    source_skill: str = Field(
        ...,
        description="Name of the skill that generated this goal"
    )
    
    source_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Context data from the source skill (e.g., conflicting_ids, error_details)"
    )
    
    # Timestamps
    creation_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the goal was created"
    )
    
    last_updated_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the goal was last modified"
    )
    
    # Enhanced features
    estimated_effort: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Estimated effort in hours or complexity score"
    )
    
    dependencies: List[str] = Field(
        default_factory=list,
        description="List of goal_ids that must be completed before this goal"
    )
    
    max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of execution attempts before marking as failed"
    )
    
    attempt_count: int = Field(
        default=0,
        ge=0,
        description="Number of execution attempts made"
    )
    
    failure_reason: Optional[str] = Field(
        default=None,
        description="Reason for failure if status is 'failed'"
    )
    
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorizing and filtering goals"
    )
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
        extra = "forbid"  # Prevent adding undefined fields
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @validator('source_skill')
    def validate_source_skill(cls, v):
        """Validate that the source skill is a known, registered skill."""
        if v not in VALID_SKILL_NAMES:
            logger.warning(f"Goal created with unrecognized source skill: {v}")
            # Don't raise an error to allow for dynamic skill registration
            # but log a warning for monitoring
        return v
    
    @validator('last_updated_timestamp', always=True)
    def set_updated_timestamp(cls, v):
        """Automatically update the timestamp when the goal is modified."""
        return datetime.now()
    
    @validator('description')
    def validate_description(cls, v):
        """Ensure description is meaningful and actionable."""
        if not v.strip():
            raise ValueError("Goal description cannot be empty")
        
        # Check for basic actionable language
        action_words = ['resolve', 'find', 'create', 'update', 'analyze', 'correct', 'improve', 'investigate']
        if not any(word in v.lower() for word in action_words):
            logger.warning(f"Goal description may not be actionable: {v}")
        
        return v.strip()
    
    @validator('tags')
    def validate_tags(cls, v):
        """Validate and normalize tags."""
        if v is None:
            return []
        
        # Normalize tags: lowercase, no spaces, alphanumeric only
        normalized_tags = []
        for tag in v:
            if isinstance(tag, str):
                normalized = ''.join(c.lower() for c in tag if c.isalnum())
                if normalized and normalized not in normalized_tags:
                    normalized_tags.append(normalized)
        
        return normalized_tags
    
    def update_status(self, new_status: str, failure_reason: Optional[str] = None) -> None:
        """
        Update the goal status with proper validation and logging.
        
        Args:
            new_status: New status to set
            failure_reason: Reason for failure if status is 'failed'
        """
        old_status = self.status
        self.status = new_status
        self.last_updated_timestamp = datetime.now()
        
        if new_status == "failed" and failure_reason:
            self.failure_reason = failure_reason
        
        logger.info(f"Goal {self.goal_id} status changed: {old_status} -> {new_status}")
    
    def increment_attempt(self) -> bool:
        """
        Increment the attempt count and check if max attempts reached.
        
        Returns:
            True if more attempts are allowed, False if max attempts reached
        """
        self.attempt_count += 1
        self.last_updated_timestamp = datetime.now()
        
        if self.attempt_count >= self.max_attempts:
            self.update_status("failed", f"Maximum attempts ({self.max_attempts}) reached")
            return False
        
        return True
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the goal."""
        normalized = ''.join(c.lower() for c in tag if c.isalnum())
        if normalized and normalized not in self.tags:
            self.tags.append(normalized)
            self.last_updated_timestamp = datetime.now()
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the goal."""
        normalized = ''.join(c.lower() for c in tag if c.isalnum())
        if normalized in self.tags:
            self.tags.remove(normalized)
            self.last_updated_timestamp = datetime.now()
    
    def is_ready_for_execution(self) -> bool:
        """
        Check if the goal is ready for execution.
        
        Returns:
            True if goal can be executed, False otherwise
        """
        return (
            self.status == "pending" and
            self.attempt_count < self.max_attempts
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert goal to dictionary for serialization."""
        return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Goal':
        """Create goal from dictionary."""
        return cls(**data)
    
    def __str__(self) -> str:
        """String representation of the goal."""
        return f"Goal({self.goal_id[:8]}): {self.description[:50]}... [{self.status}]"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"Goal(id={self.goal_id}, status={self.status}, "
                f"priority={self.priority}, source={self.source_skill})")


def update_valid_skill_names(skill_names: List[str]) -> None:
    """
    Update the list of valid skill names from the SOF registry.
    
    This function should be called during SAM initialization to populate
    the valid skill names from the actual registered skills.
    
    Args:
        skill_names: List of registered skill names from SOF
    """
    global VALID_SKILL_NAMES
    VALID_SKILL_NAMES.update(skill_names)
    logger.info(f"Updated valid skill names: {len(VALID_SKILL_NAMES)} skills registered")


def get_valid_skill_names() -> set:
    """Get the current set of valid skill names."""
    return VALID_SKILL_NAMES.copy()
