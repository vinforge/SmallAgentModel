"""
CoordinatorEngine for SAM Orchestration Framework
=================================================

Orchestrates the execution of skills according to validated plans.
Provides comprehensive error handling, fallback mechanisms, and execution monitoring.
"""

import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from .uif import SAM_UIF, UIFStatus
from .skills.base import BaseSkillModule, SkillExecutionError
from .validator import PlanValidationEngine, PlanValidationReport
from .planner import DynamicPlanner, PlanGenerationResult
from .config import get_sof_config
from .loss_balancer import LossBalancer, EffortAllocation
from .domain_constraints import DomainConstraints, ConstraintSeverity

logger = logging.getLogger(__name__)


class ExecutionResult(str, Enum):
    """Execution result enumeration."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL_SUCCESS = "partial_success"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ExecutionReport:
    """Comprehensive execution report."""
    result: ExecutionResult
    uif: SAM_UIF
    executed_skills: List[str]
    failed_skills: List[str]
    execution_time: float
    validation_report: Optional[PlanValidationReport]
    fallback_used: bool
    error_details: Optional[str] = None


class CoordinatorEngine:
    """
    Orchestrates skill execution with validation and error handling.
    
    Features:
    - Plan validation before execution
    - Comprehensive error handling and recovery
    - Fallback mechanisms for failed plans
    - Execution monitoring and metrics
    - Timeout management
    """
    
    def __init__(self, fallback_generator: Optional[Callable[[str], str]] = None,
                 enable_dynamic_planning: bool = True,
                 enable_loss_balancing: bool = True,
                 enable_domain_constraints: bool = True,
                 constraints_config_path: Optional[str] = None,
                 motivation_engine=None):
        """
        Initialize the coordinator engine.

        Args:
            fallback_generator: Optional fallback function for generating responses
            enable_dynamic_planning: Whether to enable dynamic plan generation
            enable_loss_balancing: Whether to enable PINN-inspired loss balancing
            enable_domain_constraints: Whether to enable domain constraint enforcement
            constraints_config_path: Path to domain constraints configuration file
            motivation_engine: Optional MotivationEngine for autonomous goal generation
        """
        self.logger = logging.getLogger(f"{__name__}.CoordinatorEngine")
        self._registered_skills: Dict[str, BaseSkillModule] = {}
        self._validator = PlanValidationEngine()
        # Phase B: Pass goal_stack to DynamicPlanner for goal-informed planning
        goal_stack = getattr(motivation_engine, 'goal_stack', None) if motivation_engine else None
        self._dynamic_planner = DynamicPlanner(goal_stack=goal_stack) if enable_dynamic_planning else None
        self._fallback_generator = fallback_generator
        self._execution_history: List[ExecutionReport] = []
        self._motivation_engine = motivation_engine  # Phase B: Autonomous goal generation

        # PINN-inspired loss balancer for dynamic effort allocation
        self._loss_balancer = LossBalancer() if enable_loss_balancing else None

        # Domain constraints for policy enforcement and safety compliance
        self._domain_constraints = DomainConstraints(
            config_path=constraints_config_path or "config/domain_constraints.yaml"
        ) if enable_domain_constraints else None

        # Load configuration
        self._config = get_sof_config()

        self.logger.info(f"CoordinatorEngine initialized (dynamic planning: {enable_dynamic_planning}, loss balancing: {enable_loss_balancing}, constraints: {enable_domain_constraints})")
    
    def register_skill(self, skill: BaseSkillModule) -> None:
        """
        Register a skill for execution.

        Args:
            skill: Skill to register
        """
        self._registered_skills[skill.skill_name] = skill
        self._validator.register_skill(skill)

        # Register with dynamic planner if available
        if self._dynamic_planner:
            self._dynamic_planner.register_skill(skill)

        self.logger.info(f"Registered skill: {skill.skill_name}")
    
    def register_skills(self, skills: List[BaseSkillModule]) -> None:
        """
        Register multiple skills for execution.
        
        Args:
            skills: List of skills to register
        """
        for skill in skills:
            self.register_skill(skill)
    
    def execute_plan(self, plan: List[str], uif: SAM_UIF, use_dynamic_planning: bool = False) -> ExecutionReport:
        """
        Execute a validated plan with comprehensive error handling.

        Args:
            plan: List of skill names to execute in order
            uif: Universal Interface Format with execution context
            use_dynamic_planning: Whether to use dynamic planning to generate/modify the plan

        Returns:
            Comprehensive execution report
        """
        start_time = time.time()
        
        self.logger.info(f"Starting plan execution: {plan}")
        uif.add_log_entry(f"CoordinatorEngine starting plan execution: {plan}")
        uif.execution_plan = plan.copy()
        uif.start_time = datetime.now().isoformat()
        
        try:
            # Phase 0: Domain constraint validation
            if self._domain_constraints:
                # Validate query constraints
                query_validation = self._domain_constraints.validate_query(uif.input_query, {"user_profile": uif.active_profile})

                if not query_validation.is_valid:
                    critical_violations = [v for v in query_validation.violations if v.severity == ConstraintSeverity.CRITICAL]
                    if critical_violations:
                        error_msg = f"Critical constraint violations: {[v.violation_details for v in critical_violations]}"
                        uif.set_error(error_msg)
                        return self._handle_constraint_violation(plan, uif, query_validation, start_time)

                # Log warnings
                for warning in query_validation.warnings:
                    uif.add_warning(f"Constraint warning: {warning}")

                uif.add_log_entry(f"Query constraint validation completed with {len(query_validation.violations)} violations")

            # Phase 1: Dynamic plan generation if requested
            if use_dynamic_planning and self._dynamic_planner:
                plan_result = self._dynamic_planner.create_plan(uif)
                if plan_result.plan:
                    plan = plan_result.plan
                    uif.add_log_entry(f"Dynamic plan generated: {plan} (confidence: {plan_result.confidence:.2f})")
                else:
                    uif.add_warning("Dynamic planning failed, using provided plan")

            # Phase 2: Plan constraint validation
            if self._domain_constraints:
                plan_validation = self._domain_constraints.validate_plan(plan, {"user_profile": uif.active_profile})

                if not plan_validation.is_valid:
                    critical_violations = [v for v in plan_validation.violations if v.severity == ConstraintSeverity.CRITICAL]
                    if critical_violations:
                        error_msg = f"Critical plan constraint violations: {[v.violation_details for v in critical_violations]}"
                        uif.set_error(error_msg)
                        return self._handle_constraint_violation(plan, uif, plan_validation, start_time)

                # Apply constraint-based plan modifications
                if plan_validation.blocked_skills:
                    original_plan = plan.copy()
                    plan = [skill for skill in plan if skill not in plan_validation.blocked_skills]
                    uif.add_log_entry(f"Removed blocked skills from plan: {original_plan} → {plan}")

                # Log constraint warnings
                for warning in plan_validation.warnings:
                    uif.add_warning(f"Plan constraint warning: {warning}")

            # Phase 3: Validate the plan
            if self._config.enable_plan_validation:
                validation_report = self._validator.validate_plan(plan, uif)

                if not validation_report.is_valid:
                    return self._handle_invalid_plan(plan, uif, validation_report, start_time)

                # Use optimized plan if available
                execution_plan = validation_report.optimized_plan
                uif.add_log_entry(f"Plan validated successfully, executing optimized plan: {execution_plan}")
            else:
                validation_report = None
                execution_plan = plan
                uif.add_log_entry("Plan validation disabled, executing original plan")
            
            # Phase 4: Execute the plan
            execution_result = self._execute_validated_plan(execution_plan, uif)

            # Phase 5: Create execution report
            execution_time = time.time() - start_time
            uif.end_time = datetime.now().isoformat()
            
            report = ExecutionReport(
                result=execution_result,
                uif=uif,
                executed_skills=uif.executed_skills.copy(),
                failed_skills=self._get_failed_skills(uif),
                execution_time=execution_time,
                validation_report=validation_report,
                fallback_used=False
            )
            
            self._execution_history.append(report)
            self.logger.info(f"Plan execution completed: {execution_result} in {execution_time:.2f}s")

            # Record curriculum performance if dynamic planner is available
            if (self._dynamic_planner and
                execution_result in [ExecutionResult.SUCCESS, ExecutionResult.PARTIAL_SUCCESS]):

                success = execution_result == ExecutionResult.SUCCESS
                confidence = getattr(uif, 'current_confidence', 0.5)

                self._dynamic_planner.record_curriculum_performance(
                    plan=plan,
                    success=success,
                    confidence=confidence,
                    execution_time=execution_time
                )

            # Phase B: Generate autonomous goals after successful execution
            if (self._motivation_engine and
                execution_result in [ExecutionResult.SUCCESS, ExecutionResult.PARTIAL_SUCCESS]):
                try:
                    generated_goals = self._motivation_engine.generate_goals_from_uif(uif)
                    if generated_goals:
                        uif.add_log_entry(f"Generated {len(generated_goals)} autonomous goals")
                        self.logger.info(f"Generated {len(generated_goals)} autonomous goals from execution")
                except Exception as e:
                    self.logger.warning(f"Goal generation failed: {e}")
                    uif.add_log_entry(f"Goal generation failed: {str(e)}")

            return report
            
        except Exception as e:
            self.logger.exception("Unexpected error during plan execution")
            return self._handle_execution_error(plan, uif, str(e), start_time)
    
    def _execute_validated_plan(self, plan: List[str], uif: SAM_UIF) -> ExecutionResult:
        """
        Execute a validated plan step by step with PINN-inspired effort allocation.

        Args:
            plan: Validated execution plan
            uif: Universal Interface Format

        Returns:
            Execution result
        """
        uif.status = UIFStatus.RUNNING
        failed_skills = []

        # Initialize effort allocation if loss balancer is enabled
        effort_allocation = None
        execution_plan = plan  # May be modified by confidence weighting

        if self._loss_balancer:
            query_complexity = self._assess_query_complexity(uif.input_query)
            initial_confidence = getattr(uif, 'initial_confidence', 0.5)
            effort_allocation = self._loss_balancer.allocate_effort(
                plan=plan,
                initial_confidence=initial_confidence,
                query_complexity=query_complexity
            )

            # Use optimized plan if confidence weighting provided one
            if hasattr(effort_allocation, 'optimized_plan') and effort_allocation.optimized_plan:
                execution_plan = effort_allocation.optimized_plan
                uif.add_log_entry(f"Using confidence-weighted plan: {execution_plan}", "ConfidenceWeighting")

            uif.add_log_entry(f"Effort allocation initialized for {len(execution_plan)} skills", "LossBalancer")
        
        for i, skill_name in enumerate(execution_plan):
            try:
                # Check timeout
                if self._is_execution_timeout(uif):
                    uif.add_warning("Execution timeout reached")
                    return ExecutionResult.TIMEOUT

                # Check for early termination based on confidence
                if (effort_allocation and i > 0 and
                    self._loss_balancer.should_terminate_early(
                        effort_allocation,
                        getattr(uif, 'current_confidence', 0.5),
                        uif.executed_skills,
                        execution_plan[i:]
                    )):
                    uif.add_log_entry("Early termination triggered by high confidence", "LossBalancer")
                    break

                # Get the skill
                if skill_name not in self._registered_skills:
                    error_msg = f"Skill '{skill_name}' not found during execution"
                    uif.set_error(error_msg)
                    failed_skills.append(skill_name)

                    if not self._config.continue_on_skill_failure:
                        return ExecutionResult.FAILURE
                    continue
                
                skill = self._registered_skills[skill_name]

                # Apply effort configuration if available
                if effort_allocation and skill_name in effort_allocation.skill_efforts:
                    effort_config = effort_allocation.skill_efforts[skill_name]
                    self._apply_effort_configuration(uif, skill, effort_config)
                    uif.add_log_entry(f"Executing skill {i+1}/{len(plan)}: {skill_name} (effort: {effort_config.effort_level.value})")
                else:
                    uif.add_log_entry(f"Executing skill {i+1}/{len(plan)}: {skill_name}")

                # Execute the skill with monitoring
                uif = skill.execute_with_monitoring(uif)

                # Update effort allocation based on intermediate results
                if effort_allocation and self._loss_balancer and i < len(execution_plan) - 1:
                    current_confidence = getattr(uif, 'current_confidence', 0.5)
                    effort_allocation = self._loss_balancer.adapt_effort(
                        effort_allocation,
                        uif.executed_skills,
                        current_confidence,
                        execution_plan[i+1:]
                    )
                
                # Check if skill failed
                if uif.status == UIFStatus.FAILURE:
                    failed_skills.append(skill_name)
                    uif.add_log_entry(f"Skill {skill_name} failed: {uif.error_details}")
                    
                    if not self._config.continue_on_skill_failure:
                        return ExecutionResult.FAILURE
                    
                    # Reset status to continue with next skill
                    uif.status = UIFStatus.RUNNING
                
            except Exception as e:
                error_msg = f"Unexpected error in skill {skill_name}: {str(e)}"
                self.logger.exception(error_msg)
                uif.add_log_entry(error_msg)
                failed_skills.append(skill_name)
                
                if not self._config.continue_on_skill_failure:
                    uif.set_error(error_msg)
                    return ExecutionResult.FAILURE
        
        # Phase 5C: Post-execution processing for SELF-REFLECT
        self._process_self_reflect_results(uif, execution_plan)

        # Determine final result
        if not failed_skills:
            uif.status = UIFStatus.SUCCESS
            return ExecutionResult.SUCCESS
        elif len(failed_skills) < len(plan):
            uif.add_warning(f"Partial execution: {len(failed_skills)} skills failed")
            return ExecutionResult.PARTIAL_SUCCESS
        else:
            uif.set_error("All skills in plan failed")
            return ExecutionResult.FAILURE
    
    def _handle_invalid_plan(self, plan: List[str], uif: SAM_UIF, 
                           validation_report: PlanValidationReport, start_time: float) -> ExecutionReport:
        """Handle execution of an invalid plan."""
        error_details = f"Plan validation failed: {validation_report.errors_count} errors"
        uif.set_error(error_details)
        
        self.logger.warning(f"Plan validation failed: {[issue.description for issue in validation_report.issues]}")
        
        # Try fallback if enabled
        if self._config.enable_fallback_plans:
            fallback_result = self._try_fallback_execution(uif)
            if fallback_result:
                return fallback_result
        
        execution_time = time.time() - start_time
        
        return ExecutionReport(
            result=ExecutionResult.FAILURE,
            uif=uif,
            executed_skills=[],
            failed_skills=plan,
            execution_time=execution_time,
            validation_report=validation_report,
            fallback_used=False,
            error_details=error_details
        )
    
    def _handle_execution_error(self, plan: List[str], uif: SAM_UIF, 
                              error_message: str, start_time: float) -> ExecutionReport:
        """Handle unexpected execution errors."""
        uif.set_error(f"Execution error: {error_message}")
        execution_time = time.time() - start_time
        
        # Try fallback if enabled
        if self._config.enable_fallback_plans:
            fallback_result = self._try_fallback_execution(uif)
            if fallback_result:
                return fallback_result
        
        return ExecutionReport(
            result=ExecutionResult.FAILURE,
            uif=uif,
            executed_skills=uif.executed_skills.copy(),
            failed_skills=self._get_failed_skills(uif),
            execution_time=execution_time,
            validation_report=None,
            fallback_used=False,
            error_details=error_message
        )
    
    def _try_fallback_execution(self, uif: SAM_UIF) -> Optional[ExecutionReport]:
        """
        Try fallback execution mechanisms.
        
        Returns:
            ExecutionReport if fallback succeeds, None otherwise
        """
        self.logger.info("Attempting fallback execution")
        uif.add_log_entry("Attempting fallback execution")
        
        # Try default safe plan
        default_plan = self._get_default_safe_plan()
        if default_plan:
            try:
                # Reset UIF status for fallback attempt
                uif.status = UIFStatus.PENDING
                uif.error_details = None
                
                validation_report = self._validator.validate_plan(default_plan, uif)
                if validation_report.is_valid:
                    result = self._execute_validated_plan(default_plan, uif)
                    
                    if result == ExecutionResult.SUCCESS:
                        return ExecutionReport(
                            result=ExecutionResult.SUCCESS,
                            uif=uif,
                            executed_skills=uif.executed_skills.copy(),
                            failed_skills=[],
                            execution_time=0.0,  # Will be updated by caller
                            validation_report=validation_report,
                            fallback_used=True
                        )
            except Exception as e:
                self.logger.error(f"Fallback execution failed: {e}")
        
        # Try fallback generator if available
        if self._fallback_generator:
            try:
                fallback_response = self._fallback_generator(uif.input_query)
                uif.final_response = fallback_response
                uif.status = UIFStatus.SUCCESS
                uif.add_log_entry("Used fallback generator for response")
                
                return ExecutionReport(
                    result=ExecutionResult.SUCCESS,
                    uif=uif,
                    executed_skills=["fallback_generator"],
                    failed_skills=[],
                    execution_time=0.0,
                    validation_report=None,
                    fallback_used=True
                )
            except Exception as e:
                self.logger.error(f"Fallback generator failed: {e}")
        
        return None

    def _handle_constraint_violation(
        self,
        plan: List[str],
        uif: SAM_UIF,
        validation_result,
        start_time: float
    ) -> ExecutionReport:
        """Handle constraint violations during execution."""
        # Record all violations
        if self._domain_constraints:
            for violation in validation_result.violations:
                self._domain_constraints.record_violation(violation)

        # Create error details
        critical_violations = [v for v in validation_result.violations if v.severity == ConstraintSeverity.CRITICAL]
        error_details = f"Constraint violations: {[v.violation_details for v in critical_violations]}"

        uif.set_error(error_details)
        execution_time = time.time() - start_time

        # Try safe fallback if no critical violations
        if not critical_violations and self._config.enable_fallback_plans:
            fallback_result = self._try_fallback_execution(uif)
            if fallback_result:
                return fallback_result

        return ExecutionReport(
            result=ExecutionResult.FAILURE,
            uif=uif,
            executed_skills=[],
            failed_skills=plan,
            execution_time=execution_time,
            validation_report=None,
            fallback_used=False,
            error_details=error_details
        )

    def _get_default_safe_plan(self) -> List[str]:
        """
        Get a default safe execution plan.
        
        Returns:
            List of skill names for safe execution
        """
        # Default plan: memory retrieval + response generation
        safe_skills = []
        
        if "MemoryRetrievalSkill" in self._registered_skills:
            safe_skills.append("MemoryRetrievalSkill")
        
        if "ResponseGenerationSkill" in self._registered_skills:
            safe_skills.append("ResponseGenerationSkill")
        
        return safe_skills
    
    def _is_execution_timeout(self, uif: SAM_UIF) -> bool:
        """Check if execution has timed out."""
        if not uif.start_time:
            return False

        try:
            start_dt = datetime.fromisoformat(uif.start_time)
            elapsed = (datetime.now() - start_dt).total_seconds()
            return elapsed > self._config.max_execution_time
        except (ValueError, TypeError):
            # If we can't parse the timestamp, assume no timeout
            return False
    
    def _get_failed_skills(self, uif: SAM_UIF) -> List[str]:
        """Get list of skills that failed during execution."""
        # This is a simple implementation - could be enhanced to track failures more precisely
        planned_skills = set(uif.execution_plan)
        executed_skills = set(uif.executed_skills)
        return list(planned_skills - executed_skills)
    
    def get_execution_history(self) -> List[ExecutionReport]:
        """Get execution history."""
        return self._execution_history.copy()
    
    def get_registered_skills(self) -> List[str]:
        """Get list of registered skill names."""
        return list(self._registered_skills.keys())
    
    def clear_execution_history(self) -> None:
        """Clear execution history."""
        self._execution_history.clear()
        self.logger.debug("Execution history cleared")
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics.
        
        Returns:
            Dictionary with execution statistics
        """
        if not self._execution_history:
            return {"total_executions": 0}
        
        total_executions = len(self._execution_history)
        successful_executions = sum(1 for report in self._execution_history 
                                  if report.result == ExecutionResult.SUCCESS)
        failed_executions = sum(1 for report in self._execution_history 
                              if report.result == ExecutionResult.FAILURE)
        fallback_used = sum(1 for report in self._execution_history if report.fallback_used)
        
        avg_execution_time = sum(report.execution_time for report in self._execution_history) / total_executions
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "success_rate": successful_executions / total_executions,
            "fallback_usage_rate": fallback_used / total_executions,
            "average_execution_time": avg_execution_time
        }

    def execute_with_dynamic_planning(self, uif: SAM_UIF) -> ExecutionReport:
        """
        Execute with dynamic plan generation.

        Args:
            uif: Universal Interface Format with query context

        Returns:
            Execution report
        """
        # Use empty plan - will be generated dynamically
        return self.execute_plan([], uif, use_dynamic_planning=True)

    def _assess_query_complexity(self, query: str) -> str:
        """
        Assess the complexity of a query for effort allocation.

        Args:
            query: The input query

        Returns:
            Complexity level: "simple", "medium", or "complex"
        """
        if not query:
            return "medium"

        query_lower = query.lower()

        # Simple query indicators
        simple_indicators = [
            "what is", "who is", "when is", "where is",
            "define", "meaning of", "explain briefly"
        ]

        # Complex query indicators
        complex_indicators = [
            "analyze", "compare", "evaluate", "synthesize",
            "how does", "why does", "what are the implications",
            "relationship between", "pros and cons"
        ]

        # Count indicators
        simple_count = sum(1 for indicator in simple_indicators if indicator in query_lower)
        complex_count = sum(1 for indicator in complex_indicators if indicator in query_lower)

        # Length-based assessment
        word_count = len(query.split())

        if simple_count > 0 and word_count < 10:
            return "simple"
        elif complex_count > 0 or word_count > 20:
            return "complex"
        else:
            return "medium"

    def _apply_effort_configuration(self, uif: SAM_UIF, skill: BaseSkillModule, effort_config) -> None:
        """
        Apply effort configuration to a skill execution.

        Args:
            uif: Universal Interface Format
            skill: The skill to configure
            effort_config: Effort configuration to apply
        """
        # Store effort parameters in UIF intermediate_data for skill to use
        if 'effort_parameters' not in uif.intermediate_data:
            uif.intermediate_data['effort_parameters'] = {}

        uif.intermediate_data['effort_parameters'][skill.skill_name] = effort_config.parameter_adjustments

        # Set timeout if specified
        if effort_config.timeout_multiplier != 1.0:
            current_timeout = uif.intermediate_data.get('execution_timeout', 30.0)
            uif.intermediate_data['execution_timeout'] = current_timeout * effort_config.timeout_multiplier

        # Log effort application
        self.logger.debug(f"Applied {effort_config.effort_level.value} effort to {skill.skill_name}")

    def get_loss_balancer_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get statistics from the loss balancer.

        Returns:
            Loss balancer statistics or None if not enabled
        """
        if self._loss_balancer:
            return self._loss_balancer.get_effort_statistics()
        return None

    def get_curriculum_status(self) -> Optional[Dict[str, Any]]:
        """
        Get curriculum status from the dynamic planner.

        Returns:
            Curriculum status or None if not available
        """
        if self._dynamic_planner:
            return self._dynamic_planner.get_curriculum_status()
        return None

    def get_constraint_statistics(self) -> Optional[Dict[str, Any]]:
        """
        Get constraint enforcement statistics.

        Returns:
            Constraint statistics or None if not enabled
        """
        if self._domain_constraints:
            return self._domain_constraints.get_constraint_statistics()
        return None

    # SELF-REFLECT Post-Processing (Phase 5C)

    def _process_self_reflect_results(self, uif: SAM_UIF, execution_plan: List[str]) -> None:
        """
        Process SELF-REFLECT results and trigger MEMOIR integration.

        This method is called after plan execution to check if AutonomousFactualCorrectionSkill
        was executed and produced corrections that should be fed into MEMOIR.

        Args:
            uif: Universal Interface Format with execution results
            execution_plan: List of executed skills
        """
        try:
            # Check if AutonomousFactualCorrectionSkill was in the execution plan
            if "AutonomousFactualCorrectionSkill" not in execution_plan:
                return

            # Check if SELF-REFLECT produced corrections
            was_revised = uif.intermediate_data.get("was_revised", False)
            if not was_revised:
                return

            # Check if MEMOIR auto-correction is enabled
            config = self._config
            if not getattr(config, 'enable_memoir_auto_correction', True):
                self.logger.info("MEMOIR auto-correction disabled, skipping")
                return

            self.logger.info("Processing SELF-REFLECT results for MEMOIR integration")

            # Initialize MEMOIR feedback handler
            memoir_handler = self._get_memoir_feedback_handler()
            if not memoir_handler:
                self.logger.warning("MEMOIR feedback handler not available")
                return

            # Process autonomous corrections
            result = memoir_handler.process_autonomous_correction(uif)

            if result.get('success', False):
                corrections_processed = result.get('corrections_processed', 0)
                memoir_edits_created = result.get('memoir_edits_created', 0)

                uif.add_log_entry(
                    f"MEMOIR integration: {corrections_processed} corrections processed, "
                    f"{memoir_edits_created} edits created",
                    "CoordinatorEngine"
                )

                self.logger.info(
                    f"Successfully processed SELF-REFLECT corrections: "
                    f"{corrections_processed} processed, {memoir_edits_created} MEMOIR edits created"
                )
            else:
                error_msg = result.get('error', 'Unknown error')
                uif.add_warning(f"MEMOIR integration failed: {error_msg}")
                self.logger.warning(f"MEMOIR integration failed: {error_msg}")

        except Exception as e:
            self.logger.error(f"Error processing SELF-REFLECT results: {e}")
            uif.add_warning(f"SELF-REFLECT post-processing failed: {str(e)}")

    def _get_memoir_feedback_handler(self):
        """
        Get or create MEMOIR feedback handler instance.

        Returns:
            MEMOIRFeedbackHandler instance or None if unavailable
        """
        try:
            # Import here to avoid circular dependencies
            from sam.learning.feedback_handler import MEMOIRFeedbackHandler

            # Create handler instance (could be cached in the future)
            return MEMOIRFeedbackHandler()

        except ImportError as e:
            self.logger.error(f"Failed to import MEMOIRFeedbackHandler: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to create MEMOIRFeedbackHandler: {e}")
            return None
