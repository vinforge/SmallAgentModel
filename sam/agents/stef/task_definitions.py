#!/usr/bin/env python3
"""
STEF Task Definitions
Core dataclasses for defining structured task execution programs.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum

logger = logging.getLogger(__name__)

class StepStatus(Enum):
    """Status of a task step execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class TaskStep:
    """
    Defines a single step in a TaskProgram, involving one tool call.
    
    This represents an atomic unit of work within a structured task sequence.
    Each step specifies which tool to use, how to format its input, and where
    to store its output for use by subsequent steps.
    """
    step_name: str
    tool_name: str  # Must match a key in SAM's tool registry
    
    # A template to create the tool's input string. Uses f-string style placeholders
    # that will be filled from the execution_context.
    # Example: "Find the current population of {country_name}"
    input_template: str
    
    # The key under which the tool's output will be saved in the execution_context.
    # This allows subsequent steps to reference this step's results.
    output_key: str
    
    # Optional: Validation function to check if step output is acceptable
    validation_function: Optional[Callable[[Any], bool]] = None
    
    # Optional: Whether to retry this step if it fails
    retry_on_failure: bool = True
    max_retries: int = 2
    
    # Optional: Alternative tools to try if primary tool fails
    alternative_tools: List[str] = field(default_factory=list)
    
    # Optional: Whether this step is required for program success
    required: bool = True
    
    # Optional: Description for logging and debugging
    description: str = ""

@dataclass
class ExecutionContext:
    """
    Maintains state and data throughout TaskProgram execution.
    
    This context is passed between steps and accumulates results,
    allowing later steps to use outputs from earlier steps.
    """
    initial_query: str
    data: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[str, Any] = field(default_factory=dict)
    step_statuses: Dict[str, StepStatus] = field(default_factory=dict)
    execution_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the context data."""
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in the context data."""
        self.data[key] = value
    
    def set_step_result(self, step_name: str, result: Any) -> None:
        """Store the result of a step execution."""
        self.step_results[step_name] = result
        self.data[step_name] = result  # Also store in main data for template access
    
    def set_step_status(self, step_name: str, status: StepStatus) -> None:
        """Update the status of a step."""
        self.step_statuses[step_name] = status
    
    def get_step_status(self, step_name: str) -> StepStatus:
        """Get the current status of a step."""
        return self.step_statuses.get(step_name, StepStatus.PENDING)
    
    def format_template(self, template: str) -> str:
        """Format a template string using context data."""
        try:
            # Include both data and initial_query for template formatting
            format_dict = {
                'initial_query': self.initial_query,
                **self.data
            }
            return template.format(**format_dict)
        except KeyError as e:
            logger.error(f"Template formatting failed - missing key: {e}")
            raise ValueError(f"Template requires key '{e}' which is not available in context")

@dataclass
class TaskProgram:
    """
    Defines a complete, structured, multi-step task.
    
    A TaskProgram represents a predefined sequence of tool calls that can be
    executed to handle specific types of queries. It includes trigger conditions,
    execution steps, and synthesis instructions.
    """
    program_name: str
    description: str
    
    # Keywords that the router will use to identify if a query matches this program.
    # Simple keyword matching for Phase 1, will be enhanced with confidence scoring in Phase 2.
    trigger_keywords: List[str]
    
    # The sequence of steps to execute
    steps: List[TaskStep]
    
    # A final prompt template to synthesize the final answer after all steps are complete.
    # This template has access to all step results via the execution context.
    synthesis_prompt_template: str
    
    # Optional: Minimum confidence required to trigger this program (Phase 2 enhancement)
    required_confidence: float = 0.7
    
    # Optional: Compatible intent types from SmartQueryRouter (Phase 2 enhancement)
    compatible_intents: List[str] = field(default_factory=list)
    
    # Optional: Whether this program can handle partial failures
    allow_partial_execution: bool = False
    
    # Optional: Custom error message template for failures
    error_message_template: str = "I encountered an error while executing the {program_name} task: {error_details}"
    
    def matches_query(self, query: str) -> bool:
        """
        Check if this program can handle the given query.
        Phase 1: Simple keyword matching
        Phase 2: Will be enhanced with confidence scoring
        """
        query_lower = query.lower()
        
        # Check if any trigger keywords are present
        for keyword in self.trigger_keywords:
            if keyword.lower() in query_lower:
                logger.info(f"Program '{self.program_name}' matched on keyword: '{keyword}'")
                return True
        
        return False
    
    def validate_program(self) -> List[str]:
        """
        Validate the program definition and return any issues found.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if not self.program_name:
            errors.append("Program name is required")
        
        if not self.steps:
            errors.append("At least one step is required")
        
        if not self.synthesis_prompt_template:
            errors.append("Synthesis prompt template is required")
        
        # Check for duplicate step names
        step_names = [step.step_name for step in self.steps]
        if len(step_names) != len(set(step_names)):
            errors.append("Step names must be unique")
        
        # Check for duplicate output keys
        output_keys = [step.output_key for step in self.steps]
        if len(output_keys) != len(set(output_keys)):
            errors.append("Step output keys must be unique")
        
        # Validate each step
        for i, step in enumerate(self.steps):
            if not step.step_name:
                errors.append(f"Step {i+1}: step_name is required")
            if not step.tool_name:
                errors.append(f"Step {i+1}: tool_name is required")
            if not step.input_template:
                errors.append(f"Step {i+1}: input_template is required")
            if not step.output_key:
                errors.append(f"Step {i+1}: output_key is required")
        
        return errors
    
    def get_required_steps(self) -> List[TaskStep]:
        """Get only the required steps for this program."""
        return [step for step in self.steps if step.required]
    
    def get_optional_steps(self) -> List[TaskStep]:
        """Get only the optional steps for this program."""
        return [step for step in self.steps if not step.required]

@dataclass
class ProgramExecutionResult:
    """
    Result of executing a TaskProgram.
    
    Contains the final response, execution metadata, and success/failure information.
    """
    program_name: str
    success: bool
    final_response: str
    execution_context: ExecutionContext
    execution_time_ms: float
    steps_completed: int
    steps_failed: int
    error_message: Optional[str] = None
    
    @property
    def completion_rate(self) -> float:
        """Calculate the percentage of steps that completed successfully."""
        total_steps = self.steps_completed + self.steps_failed
        if total_steps == 0:
            return 0.0
        return (self.steps_completed / total_steps) * 100.0
