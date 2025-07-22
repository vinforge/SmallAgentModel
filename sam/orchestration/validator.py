"""
Plan Validation Engine for SAM Orchestration Framework
======================================================

Validates execution plans before they are executed by the CoordinatorEngine.
Performs dependency checking, circular dependency detection, and plan optimization.
"""

import logging
from typing import List, Dict, Any, Set, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .uif import SAM_UIF
from .skills.base import BaseSkillModule

logger = logging.getLogger(__name__)


class ValidationResult(str, Enum):
    """Validation result enumeration."""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"


@dataclass
class ValidationIssue:
    """Represents a validation issue found during plan validation."""
    severity: ValidationResult
    skill_name: str
    issue_type: str
    description: str
    suggestion: Optional[str] = None


@dataclass
class PlanValidationReport:
    """Comprehensive validation report for an execution plan."""
    is_valid: bool
    issues: List[ValidationIssue]
    optimized_plan: List[str]
    execution_estimate: float
    dependency_graph: Dict[str, List[str]]
    warnings_count: int
    errors_count: int


class PlanValidationEngine:
    """
    Validates execution plans before they are executed.
    
    Performs comprehensive validation including:
    - Skill existence verification
    - Dependency satisfaction checking
    - Circular dependency detection
    - Plan optimization opportunities
    - Execution time estimation
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.PlanValidationEngine")
        self._registered_skills: Dict[str, BaseSkillModule] = {}
        self._validation_cache: Dict[str, PlanValidationReport] = {}
    
    def register_skill(self, skill: BaseSkillModule) -> None:
        """
        Register a skill for validation.
        
        Args:
            skill: Skill to register
        """
        self._registered_skills[skill.skill_name] = skill
        self.logger.debug(f"Registered skill: {skill.skill_name}")
    
    def register_skills(self, skills: List[BaseSkillModule]) -> None:
        """
        Register multiple skills for validation.
        
        Args:
            skills: List of skills to register
        """
        for skill in skills:
            self.register_skill(skill)
    
    def validate_plan(self, plan: List[str], uif: SAM_UIF) -> PlanValidationReport:
        """
        Validate an execution plan comprehensively.
        
        Args:
            plan: List of skill names to execute in order
            uif: Universal Interface Format with initial state
            
        Returns:
            Comprehensive validation report
        """
        self.logger.info(f"Validating plan with {len(plan)} skills: {plan}")
        
        # Check cache first
        cache_key = self._generate_cache_key(plan, uif)
        if cache_key in self._validation_cache:
            self.logger.debug("Using cached validation result")
            return self._validation_cache[cache_key]
        
        issues = []
        
        # 1. Validate skill existence
        existence_issues = self._validate_skill_existence(plan)
        issues.extend(existence_issues)
        
        # 2. Validate dependencies
        dependency_issues, dependency_graph = self._validate_dependencies(plan, uif)
        issues.extend(dependency_issues)
        
        # 3. Check for circular dependencies
        circular_issues = self._check_circular_dependencies(dependency_graph)
        issues.extend(circular_issues)
        
        # 4. Validate plan length and complexity
        complexity_issues = self._validate_plan_complexity(plan)
        issues.extend(complexity_issues)
        
        # 5. Check for optimization opportunities
        optimization_issues, optimized_plan = self._optimize_plan(plan, dependency_graph)
        issues.extend(optimization_issues)
        
        # 6. Estimate execution time
        execution_estimate = self._estimate_execution_time(optimized_plan)
        
        # Categorize issues
        errors = [issue for issue in issues if issue.severity == ValidationResult.INVALID]
        warnings = [issue for issue in issues if issue.severity == ValidationResult.WARNING]
        
        # Create validation report
        report = PlanValidationReport(
            is_valid=len(errors) == 0,
            issues=issues,
            optimized_plan=optimized_plan,
            execution_estimate=execution_estimate,
            dependency_graph=dependency_graph,
            warnings_count=len(warnings),
            errors_count=len(errors)
        )
        
        # Cache the result
        self._validation_cache[cache_key] = report
        
        self.logger.info(f"Plan validation complete: {'VALID' if report.is_valid else 'INVALID'} "
                        f"({len(errors)} errors, {len(warnings)} warnings)")
        
        return report
    
    def _validate_skill_existence(self, plan: List[str]) -> List[ValidationIssue]:
        """Validate that all skills in the plan exist."""
        issues = []
        
        for skill_name in plan:
            if skill_name not in self._registered_skills:
                issues.append(ValidationIssue(
                    severity=ValidationResult.INVALID,
                    skill_name=skill_name,
                    issue_type="missing_skill",
                    description=f"Skill '{skill_name}' is not registered",
                    suggestion=f"Register skill '{skill_name}' or remove from plan"
                ))
        
        return issues
    
    def _validate_dependencies(self, plan: List[str], uif: SAM_UIF) -> Tuple[List[ValidationIssue], Dict[str, List[str]]]:
        """
        Validate that skill dependencies are satisfied.
        
        Returns:
            Tuple of (issues, dependency_graph)
        """
        issues = []
        dependency_graph = {}
        
        # Track what will be available at each step
        available_inputs = set()
        
        # Add initial UIF inputs
        if uif.input_query:
            available_inputs.add('input_query')
        if uif.source_documents:
            available_inputs.add('source_documents')
        if uif.user_context:
            available_inputs.add('user_context')
        if uif.active_profile:
            available_inputs.add('active_profile')
        if uif.session_id:
            available_inputs.add('session_id')
        if uif.user_id:
            available_inputs.add('user_id')
        
        # Add any existing intermediate data
        available_inputs.update(uif.intermediate_data.keys())
        
        for i, skill_name in enumerate(plan):
            if skill_name not in self._registered_skills:
                continue  # Skip missing skills (handled in existence validation)
            
            skill = self._registered_skills[skill_name]
            dependency_graph[skill_name] = []
            
            # Check required inputs
            missing_inputs = []
            for required_input in skill.required_inputs:
                if required_input not in available_inputs:
                    missing_inputs.append(required_input)
                    
                    # Try to find which skill provides this input
                    provider = self._find_input_provider(required_input, plan[:i])
                    if provider:
                        dependency_graph[skill_name].append(provider)
            
            if missing_inputs:
                issues.append(ValidationIssue(
                    severity=ValidationResult.INVALID,
                    skill_name=skill_name,
                    issue_type="missing_dependencies",
                    description=f"Missing required inputs: {missing_inputs}",
                    suggestion=f"Ensure required inputs are provided by previous skills or initial UIF"
                ))
            
            # Add this skill's outputs to available inputs for next skills
            available_inputs.update(skill.output_keys)
        
        return issues, dependency_graph
    
    def _find_input_provider(self, input_key: str, previous_skills: List[str]) -> Optional[str]:
        """Find which skill provides a specific input."""
        for skill_name in reversed(previous_skills):  # Check in reverse order
            if skill_name in self._registered_skills:
                skill = self._registered_skills[skill_name]
                if input_key in skill.output_keys:
                    return skill_name
        return None
    
    def _check_circular_dependencies(self, dependency_graph: Dict[str, List[str]]) -> List[ValidationIssue]:
        """Check for circular dependencies in the plan."""
        issues = []
        
        def has_cycle(node: str, visited: Set[str], rec_stack: Set[str]) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in dependency_graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        visited = set()
        for skill_name in dependency_graph:
            if skill_name not in visited:
                if has_cycle(skill_name, visited, set()):
                    issues.append(ValidationIssue(
                        severity=ValidationResult.INVALID,
                        skill_name=skill_name,
                        issue_type="circular_dependency",
                        description=f"Circular dependency detected involving '{skill_name}'",
                        suggestion="Reorder skills to break circular dependencies"
                    ))
        
        return issues
    
    def _validate_plan_complexity(self, plan: List[str]) -> List[ValidationIssue]:
        """Validate plan complexity and length."""
        issues = []
        
        # Check plan length
        from .config import get_sof_config
        config = get_sof_config()
        
        if len(plan) > config.max_plan_length:
            issues.append(ValidationIssue(
                severity=ValidationResult.WARNING,
                skill_name="plan",
                issue_type="plan_too_long",
                description=f"Plan length ({len(plan)}) exceeds recommended maximum ({config.max_plan_length})",
                suggestion="Consider breaking into smaller sub-plans or optimizing"
            ))
        
        # Check for duplicate skills
        seen_skills = set()
        for skill_name in plan:
            if skill_name in seen_skills:
                issues.append(ValidationIssue(
                    severity=ValidationResult.WARNING,
                    skill_name=skill_name,
                    issue_type="duplicate_skill",
                    description=f"Skill '{skill_name}' appears multiple times in plan",
                    suggestion="Consider if multiple executions are necessary"
                ))
            seen_skills.add(skill_name)
        
        return issues
    
    def _optimize_plan(self, plan: List[str], dependency_graph: Dict[str, List[str]]) -> Tuple[List[ValidationIssue], List[str]]:
        """
        Optimize the execution plan.
        
        Returns:
            Tuple of (optimization_issues, optimized_plan)
        """
        issues = []
        optimized_plan = plan.copy()
        
        # Check for unnecessary skills
        # (This is a simple optimization - more sophisticated ones can be added)
        
        # For now, just return the original plan
        # Future optimizations could include:
        # - Removing redundant skills
        # - Reordering for better efficiency
        # - Parallel execution opportunities
        
        return issues, optimized_plan
    
    def _estimate_execution_time(self, plan: List[str]) -> float:
        """
        Estimate total execution time for the plan.
        
        Returns:
            Estimated execution time in seconds
        """
        total_time = 0.0
        
        for skill_name in plan:
            if skill_name in self._registered_skills:
                skill = self._registered_skills[skill_name]
                if skill.estimated_execution_time:
                    total_time += skill.estimated_execution_time
                else:
                    total_time += 1.0  # Default estimate
        
        return total_time
    
    def _generate_cache_key(self, plan: List[str], uif: SAM_UIF) -> str:
        """Generate a cache key for the validation result."""
        # Simple cache key based on plan and available inputs
        available_inputs = sorted(uif.intermediate_data.keys())
        plan_str = ",".join(plan)
        inputs_str = ",".join(available_inputs)
        return f"{plan_str}|{inputs_str}"
    
    def clear_cache(self) -> None:
        """Clear the validation cache."""
        self._validation_cache.clear()
        self.logger.debug("Validation cache cleared")
    
    def get_registered_skills(self) -> List[str]:
        """Get list of registered skill names."""
        return list(self._registered_skills.keys())
    
    def get_skill_info(self, skill_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a registered skill.
        
        Args:
            skill_name: Name of the skill
            
        Returns:
            Skill information dictionary or None if not found
        """
        if skill_name in self._registered_skills:
            skill = self._registered_skills[skill_name]
            return skill.get_dependency_info()
        return None
