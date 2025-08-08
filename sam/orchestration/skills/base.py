"""
BaseSkillModule - Abstract Base Class for SAM Skills
====================================================

Defines the interface and common functionality that all SAM skills must implement.
Enhanced with dependency declaration system for plan validation and self-documenting
architecture.
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass

from ..uif import SAM_UIF, UIFStatus

logger = logging.getLogger(__name__)


class SkillExecutionError(Exception):
    """Raised when a skill fails to execute properly."""
    pass


class SkillDependencyError(Exception):
    """Raised when skill dependencies are not satisfied."""
    pass


@dataclass
class SkillMetadata:
    """Metadata about a skill's capabilities and requirements."""
    name: str
    description: str
    version: str
    category: str
    requires_external_access: bool = False
    requires_vetting: bool = False
    estimated_execution_time: Optional[float] = None


class BaseSkillModule(ABC):
    """
    Abstract base class for all SAM skills.
    
    All skills must inherit from this class and implement the execute method.
    Skills declare their input requirements and output specifications for
    dependency validation and self-documenting architecture.
    """
    
    # Core skill identification
    skill_name: str = "BaseSkill"
    skill_version: str = "1.0.0"
    skill_description: str = "Base skill class"
    skill_category: str = "core"
    
    # Dependency declarations (NEW: Enhanced dependency system)
    required_inputs: List[str] = []  # Keys that must exist in UIF
    optional_inputs: List[str] = []  # Keys that are optional but used if present
    output_keys: List[str] = []      # Keys this skill writes to UIF
    
    # Skill capabilities and requirements
    requires_external_access: bool = False  # Does this skill access external resources?
    requires_vetting: bool = False          # Does output need security vetting?
    can_run_parallel: bool = True           # Can this skill run in parallel with others?
    
    # Performance characteristics
    estimated_execution_time: Optional[float] = None  # Estimated time in seconds
    max_execution_time: Optional[float] = None        # Maximum allowed time
    
    def __init__(self):
        """Initialize the skill module."""
        self.logger = logging.getLogger(f"{__name__}.{self.skill_name}")
        self._validate_skill_definition()
    
    def _validate_skill_definition(self) -> None:
        """Validate that the skill is properly defined."""
        if not self.skill_name or self.skill_name == "BaseSkill":
            raise ValueError(f"Skill must define a unique skill_name")
        
        if not self.skill_description:
            raise ValueError(f"Skill {self.skill_name} must provide a description")
        
        # Validate no overlap between required and optional inputs
        required_set = set(self.required_inputs)
        optional_set = set(self.optional_inputs)
        overlap = required_set.intersection(optional_set)
        if overlap:
            raise ValueError(f"Skill {self.skill_name} has overlapping required/optional inputs: {overlap}")
    
    @abstractmethod
    def execute(self, uif: SAM_UIF) -> SAM_UIF:
        """
        Execute the skill with the provided UIF.
        
        Args:
            uif: Universal Interface Format containing all execution context
            
        Returns:
            Updated UIF with skill results
            
        Raises:
            SkillExecutionError: If skill execution fails
            SkillDependencyError: If required dependencies are missing
        """
        pass
    
    def validate_dependencies(self, uif: SAM_UIF) -> None:
        """
        Validate that all required dependencies are satisfied.

        Args:
            uif: UIF to validate against

        Raises:
            SkillDependencyError: If dependencies are not satisfied
        """
        missing_inputs = []
        available_inputs = self.get_available_inputs(uif)

        # Check required inputs are available
        for required_key in self.required_inputs:
            if required_key not in available_inputs:
                missing_inputs.append(required_key)

        if missing_inputs:
            raise SkillDependencyError(
                f"Skill {self.skill_name} missing required inputs: {missing_inputs}"
            )
    
    def get_available_inputs(self, uif: SAM_UIF) -> Set[str]:
        """
        Get all available input keys from the UIF.

        Args:
            uif: UIF to examine

        Returns:
            Set of available input keys
        """
        available = set(uif.intermediate_data.keys())

        # Add standard UIF fields that skills might use (only if they have values)
        if uif.input_query:
            available.add('input_query')
        if uif.source_documents:
            available.add('source_documents')
        if uif.user_context:
            available.add('user_context')
        if uif.active_profile:
            available.add('active_profile')
        if uif.session_id:
            available.add('session_id')
        if uif.user_id:
            available.add('user_id')

        return available
    
    def execute_with_monitoring(self, uif: SAM_UIF) -> SAM_UIF:
        """
        Execute the skill with comprehensive monitoring and error handling.
        
        Args:
            uif: Universal Interface Format
            
        Returns:
            Updated UIF with execution results
        """
        start_time = time.time()
        
        try:
            # Pre-execution validation
            self.validate_dependencies(uif)
            
            # Log execution start
            uif.add_log_entry(f"Starting execution", self.skill_name)
            uif.status = UIFStatus.RUNNING
            
            # Execute the skill
            result_uif = self.execute(uif)
            
            # Post-execution validation
            execution_time = time.time() - start_time
            
            # Check execution time limits
            if self.max_execution_time and execution_time > self.max_execution_time:
                result_uif.add_warning(
                    f"Execution time ({execution_time:.2f}s) exceeded limit ({self.max_execution_time}s)"
                )
            
            # Mark skill as completed
            result_uif.mark_skill_complete(self.skill_name, execution_time)
            
            # Validate outputs were produced
            self._validate_outputs(result_uif)
            
            return result_uif
            
        except SkillDependencyError as e:
            uif.set_error(f"Dependency error: {str(e)}", self.skill_name)
            return uif
            
        except SkillExecutionError as e:
            uif.set_error(f"Execution error: {str(e)}", self.skill_name)
            return uif
            
        except Exception as e:
            self.logger.exception(f"Unexpected error in skill {self.skill_name}")
            uif.set_error(f"Unexpected error: {str(e)}", self.skill_name)
            return uif
    
    def _validate_outputs(self, uif: SAM_UIF) -> None:
        """
        Validate that the skill produced its declared outputs.
        
        Args:
            uif: UIF to validate
        """
        missing_outputs = []
        
        for output_key in self.output_keys:
            if output_key not in uif.intermediate_data and output_key not in uif.skill_outputs:
                missing_outputs.append(output_key)
        
        if missing_outputs:
            uif.add_warning(f"Skill {self.skill_name} did not produce expected outputs: {missing_outputs}")
    
    def get_metadata(self) -> SkillMetadata:
        """
        Get metadata about this skill.
        
        Returns:
            SkillMetadata object with skill information
        """
        return SkillMetadata(
            name=self.skill_name,
            description=self.skill_description,
            version=self.skill_version,
            category=self.skill_category,
            requires_external_access=self.requires_external_access,
            requires_vetting=self.requires_vetting,
            estimated_execution_time=self.estimated_execution_time
        )
    
    def can_execute(self, uif: SAM_UIF) -> bool:
        """
        Check if this skill can execute with the current UIF state.
        
        Args:
            uif: UIF to check
            
        Returns:
            True if skill can execute, False otherwise
        """
        try:
            self.validate_dependencies(uif)
            return True
        except SkillDependencyError:
            return False
    
    def get_dependency_info(self) -> Dict[str, Any]:
        """
        Get comprehensive dependency information for this skill.
        
        Returns:
            Dictionary with dependency details
        """
        return {
            "skill_name": self.skill_name,
            "required_inputs": self.required_inputs,
            "optional_inputs": self.optional_inputs,
            "output_keys": self.output_keys,
            "requires_external_access": self.requires_external_access,
            "requires_vetting": self.requires_vetting,
            "can_run_parallel": self.can_run_parallel,
            "estimated_execution_time": self.estimated_execution_time
        }
    
    def __str__(self) -> str:
        """String representation of the skill."""
        return f"{self.skill_name} v{self.skill_version} ({self.skill_category})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the skill."""
        return (f"{self.__class__.__name__}(name='{self.skill_name}', "
                f"version='{self.skill_version}', category='{self.skill_category}')")
