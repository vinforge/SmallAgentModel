"""
Meta-Reasoning Plan Validator Module

Integrates SAM's meta-reasoning system with the A* planner to provide
comprehensive plan validation including risk assessment, ethical validation,
and strategic analysis. Acts as the "Sanity Check" for planning outcomes.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from .state import PlanningState

# Import SAM's meta-reasoning components
try:
    from reasoning.reflective_meta_reasoning import ReflectiveMetaReasoning, ReflectiveResult
    from sam.autonomy.safety.goal_validator import GoalSafetyValidator
    META_REASONING_AVAILABLE = True
except ImportError:
    # Create placeholder classes for type annotations
    class ReflectiveMetaReasoning:
        pass
    class ReflectiveResult:
        pass
    class GoalSafetyValidator:
        pass
    META_REASONING_AVAILABLE = False
    logging.warning("SAM meta-reasoning components not available - using fallback implementation")

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a validation issue found in a plan."""
    
    severity: ValidationSeverity
    """Severity level of the issue"""
    
    category: str
    """Category of the issue (risk, ethics, strategy, etc.)"""
    
    description: str
    """Human-readable description of the issue"""
    
    action_index: Optional[int] = None
    """Index of the action that caused the issue (if applicable)"""
    
    recommendation: Optional[str] = None
    """Recommended action to address the issue"""
    
    confidence: float = 0.8
    """Confidence in this validation result"""


@dataclass
class PlanValidationResult:
    """Result of comprehensive plan validation."""
    
    is_valid: bool
    """Whether the plan passes validation"""
    
    overall_risk_score: float
    """Overall risk score (0.0 to 1.0, higher is riskier)"""
    
    confidence_score: float
    """Confidence in the plan's effectiveness (0.0 to 1.0)"""
    
    issues: List[ValidationIssue]
    """List of validation issues found"""
    
    recommendations: List[str]
    """High-level recommendations for plan improvement"""
    
    alternative_suggestions: List[str]
    """Alternative action sequences to consider"""
    
    meta_reasoning_analysis: Optional[Dict[str, Any]] = None
    """Detailed meta-reasoning analysis if available"""
    
    validation_time: float = 0.0
    """Time spent on validation"""


class MetaReasoningPlanValidator:
    """
    Meta-reasoning enhanced plan validator for A* search planning.
    
    This validator integrates SAM's meta-reasoning capabilities to provide
    comprehensive validation of planning outcomes, including risk assessment,
    ethical considerations, and strategic analysis.
    """
    
    def __init__(self,
                 enable_meta_reasoning: bool = True,
                 enable_safety_validation: bool = True,
                 risk_threshold: float = 0.7,
                 confidence_threshold: float = 0.6):
        """
        Initialize the meta-reasoning plan validator.
        
        Args:
            enable_meta_reasoning: Whether to use SAM's meta-reasoning system
            enable_safety_validation: Whether to use safety validation
            risk_threshold: Maximum acceptable risk score
            confidence_threshold: Minimum required confidence score
        """
        self.enable_meta_reasoning = enable_meta_reasoning and META_REASONING_AVAILABLE
        self.enable_safety_validation = enable_safety_validation
        self.risk_threshold = risk_threshold
        self.confidence_threshold = confidence_threshold
        
        # Initialize meta-reasoning components
        if self.enable_meta_reasoning:
            self.meta_reasoner = self._initialize_meta_reasoner()
        else:
            self.meta_reasoner = None
        
        if self.enable_safety_validation:
            self.safety_validator = self._initialize_safety_validator()
        else:
            self.safety_validator = None
        
        # Validation statistics
        self.total_validations = 0
        self.plans_rejected = 0
        self.high_risk_plans = 0
        
        logger.info(f"MetaReasoningPlanValidator initialized (meta_reasoning: {self.enable_meta_reasoning}, "
                   f"safety: {self.enable_safety_validation})")
    
    def _initialize_meta_reasoner(self) -> Optional[ReflectiveMetaReasoning]:
        """Initialize SAM's meta-reasoning system."""
        try:
            return ReflectiveMetaReasoning()
        except Exception as e:
            logger.warning(f"Failed to initialize meta-reasoner: {e}")
            self.enable_meta_reasoning = False
            return None
    
    def _initialize_safety_validator(self) -> Optional[GoalSafetyValidator]:
        """Initialize SAM's safety validator."""
        try:
            return GoalSafetyValidator()
        except Exception as e:
            logger.warning(f"Failed to initialize safety validator: {e}")
            self.enable_safety_validation = False
            return None
    
    def validate_plan(self, 
                     plan: List[str],
                     initial_state: PlanningState,
                     context: Optional[Dict[str, Any]] = None) -> PlanValidationResult:
        """
        Perform comprehensive validation of a planning result.
        
        Args:
            plan: List of actions in the plan
            initial_state: Initial planning state
            context: Additional context for validation
            
        Returns:
            PlanValidationResult with comprehensive analysis
        """
        start_time = datetime.now()
        self.total_validations += 1
        
        try:
            # Initialize validation result
            issues = []
            recommendations = []
            alternative_suggestions = []
            meta_reasoning_analysis = None
            
            # 1. Basic plan validation
            basic_issues = self._validate_basic_plan_structure(plan, initial_state)
            issues.extend(basic_issues)
            
            # 2. Risk assessment
            risk_issues, risk_score = self._assess_plan_risks(plan, initial_state, context)
            issues.extend(risk_issues)
            
            # 3. Safety validation
            if self.enable_safety_validation and self.safety_validator:
                safety_issues = self._validate_plan_safety(plan, initial_state)
                issues.extend(safety_issues)
            
            # 4. Meta-reasoning analysis
            if self.enable_meta_reasoning and self.meta_reasoner:
                meta_result = self._perform_meta_reasoning_analysis(plan, initial_state, context)
                if meta_result:
                    issues.extend(meta_result.get('issues', []))
                    recommendations.extend(meta_result.get('recommendations', []))
                    alternative_suggestions.extend(meta_result.get('alternatives', []))
                    meta_reasoning_analysis = meta_result
            
            # 5. Strategic analysis
            strategic_issues, strategic_recommendations = self._analyze_plan_strategy(plan, initial_state)
            issues.extend(strategic_issues)
            recommendations.extend(strategic_recommendations)
            
            # 6. Generate alternative suggestions
            if not alternative_suggestions:
                alternative_suggestions = self._generate_alternative_suggestions(plan, initial_state)
            
            # Calculate overall scores
            overall_risk_score = self._calculate_overall_risk_score(issues, risk_score)
            confidence_score = self._calculate_confidence_score(plan, issues, meta_reasoning_analysis)
            
            # Determine if plan is valid
            is_valid = self._determine_plan_validity(overall_risk_score, confidence_score, issues)
            
            if not is_valid:
                self.plans_rejected += 1
            
            if overall_risk_score > self.risk_threshold:
                self.high_risk_plans += 1
            
            # Calculate validation time
            validation_time = (datetime.now() - start_time).total_seconds()
            
            result = PlanValidationResult(
                is_valid=is_valid,
                overall_risk_score=overall_risk_score,
                confidence_score=confidence_score,
                issues=issues,
                recommendations=list(set(recommendations)),  # Remove duplicates
                alternative_suggestions=alternative_suggestions,
                meta_reasoning_analysis=meta_reasoning_analysis,
                validation_time=validation_time
            )
            
            logger.debug(f"Plan validation completed: valid={is_valid}, risk={overall_risk_score:.2f}, "
                        f"confidence={confidence_score:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during plan validation: {e}")
            return PlanValidationResult(
                is_valid=False,
                overall_risk_score=1.0,
                confidence_score=0.0,
                issues=[ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="validation_error",
                    description=f"Validation failed: {str(e)}",
                    recommendation="Manual review required"
                )],
                recommendations=["Manual review required due to validation error"],
                alternative_suggestions=[],
                validation_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _validate_basic_plan_structure(self, plan: List[str], initial_state: PlanningState) -> List[ValidationIssue]:
        """Validate basic plan structure and consistency."""
        issues = []
        
        # Check for empty plan
        if not plan:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="structure",
                description="Plan is empty - no actions to execute",
                recommendation="Generate a non-empty plan with specific actions"
            ))
        
        # Check for excessively long plans
        elif len(plan) > 15:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="efficiency",
                description=f"Plan is very long ({len(plan)} actions) - may be inefficient",
                recommendation="Consider simplifying the plan or breaking into sub-tasks"
            ))
        
        # Check for duplicate consecutive actions
        for i in range(len(plan) - 1):
            if plan[i] == plan[i + 1]:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="efficiency",
                    description=f"Duplicate consecutive action: {plan[i]}",
                    action_index=i,
                    recommendation="Remove duplicate actions or add variation"
                ))
        
        return issues
    
    def _assess_plan_risks(self, 
                          plan: List[str], 
                          initial_state: PlanningState,
                          context: Optional[Dict[str, Any]]) -> Tuple[List[ValidationIssue], float]:
        """Assess risks associated with the plan."""
        issues = []
        risk_factors = []
        
        # Check for potentially risky actions
        risky_actions = ['delete', 'remove', 'modify', 'overwrite', 'replace']
        for i, action in enumerate(plan):
            action_lower = action.lower()
            for risky_word in risky_actions:
                if risky_word in action_lower:
                    risk_factors.append(0.3)
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="risk",
                        description=f"Potentially risky action: {action}",
                        action_index=i,
                        recommendation="Ensure proper safeguards are in place"
                    ))
                    break
        
        # Check for resource-intensive actions
        intensive_actions = ['deep_analyze', 'comprehensive', 'extensive']
        for i, action in enumerate(plan):
            action_lower = action.lower()
            for intensive_word in intensive_actions:
                if intensive_word in action_lower:
                    risk_factors.append(0.2)
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        category="resource",
                        description=f"Resource-intensive action: {action}",
                        action_index=i,
                        recommendation="Monitor resource usage during execution"
                    ))
                    break
        
        # Calculate overall risk score
        base_risk = 0.1  # Base risk for any plan
        additional_risk = sum(risk_factors)
        overall_risk = min(1.0, base_risk + additional_risk)
        
        return issues, overall_risk
    
    def _validate_plan_safety(self, plan: List[str], initial_state: PlanningState) -> List[ValidationIssue]:
        """Validate plan safety using SAM's safety validator."""
        issues = []
        
        if not self.safety_validator:
            return issues
        
        try:
            # Create a mock goal for safety validation
            # This is a simplified approach - real implementation would be more sophisticated
            plan_description = f"Execute planning sequence: {' -> '.join(plan)}"
            
            # Note: This is a simplified safety check
            # Real implementation would integrate more deeply with SAM's safety systems
            
            # Check for harmful patterns in plan actions
            harmful_patterns = ['delete_all', 'format', 'shutdown', 'terminate']
            for i, action in enumerate(plan):
                action_lower = action.lower()
                for pattern in harmful_patterns:
                    if pattern in action_lower:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.CRITICAL,
                            category="safety",
                            description=f"Potentially harmful action detected: {action}",
                            action_index=i,
                            recommendation="Remove or modify this action for safety"
                        ))
            
        except Exception as e:
            logger.warning(f"Safety validation error: {e}")
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="safety",
                description="Safety validation could not be completed",
                recommendation="Manual safety review recommended"
            ))
        
        return issues
    
    def _perform_meta_reasoning_analysis(self, 
                                       plan: List[str],
                                       initial_state: PlanningState,
                                       context: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Perform meta-reasoning analysis of the plan."""
        
        if not self.meta_reasoner:
            return None
        
        try:
            # Create a description of the plan for meta-reasoning
            plan_description = f"Planning task: {initial_state.task_description}\n"
            plan_description += f"Proposed action sequence: {' -> '.join(plan)}\n"
            plan_description += f"Context: {len(plan)} actions planned"
            
            # Perform reflective reasoning on the plan
            meta_result = self.meta_reasoner.reflective_reasoning_cycle(
                query=initial_state.task_description,
                initial_response=plan_description,
                context=context or {}
            )
            
            # Extract validation-relevant information
            issues = []
            recommendations = []
            alternatives = []
            
            # Analyze meta-reasoning results for validation insights
            if hasattr(meta_result, 'critiques') and meta_result.critiques:
                for critique in meta_result.critiques:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        category="meta_reasoning",
                        description=f"Meta-reasoning critique: {critique}",
                        recommendation="Consider addressing this critique"
                    ))
            
            if hasattr(meta_result, 'alternatives') and meta_result.alternatives:
                alternatives = [alt for alt in meta_result.alternatives if alt != plan_description]
            
            return {
                'issues': issues,
                'recommendations': recommendations,
                'alternatives': alternatives,
                'meta_confidence': getattr(meta_result, 'meta_confidence', 0.5),
                'reasoning_chain': getattr(meta_result, 'reasoning_chain', []),
                'full_result': meta_result
            }
            
        except Exception as e:
            logger.warning(f"Meta-reasoning analysis error: {e}")
            return None
    
    def _analyze_plan_strategy(self, plan: List[str], initial_state: PlanningState) -> Tuple[List[ValidationIssue], List[str]]:
        """Analyze strategic aspects of the plan."""
        issues = []
        recommendations = []
        
        # Check for logical action ordering
        if len(plan) >= 2:
            # Simple heuristic: analysis actions should come before synthesis
            analysis_actions = [i for i, action in enumerate(plan) if 'analyze' in action.lower()]
            synthesis_actions = [i for i, action in enumerate(plan) if 'synthesize' in action.lower()]
            
            if analysis_actions and synthesis_actions:
                if max(analysis_actions) > min(synthesis_actions):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="strategy",
                        description="Analysis actions appear after synthesis actions",
                        recommendation="Consider reordering actions: analyze before synthesize"
                    ))
        
        # Check for task-action alignment
        task_lower = initial_state.task_description.lower()
        if 'document' in task_lower and not any('document' in action.lower() for action in plan):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="alignment",
                description="Task mentions documents but plan has no document-related actions",
                recommendation="Add document analysis actions to the plan"
            ))
        
        # Generate strategic recommendations
        if len(plan) == 1:
            recommendations.append("Consider adding verification or validation steps")
        
        if not any('synthesize' in action.lower() or 'create' in action.lower() for action in plan):
            recommendations.append("Consider adding a synthesis or output generation step")
        
        return issues, recommendations
    
    def _generate_alternative_suggestions(self, plan: List[str], initial_state: PlanningState) -> List[str]:
        """Generate alternative action sequences."""
        alternatives = []
        
        # Simple alternative generation based on task type
        task_lower = initial_state.task_description.lower()
        
        if 'document' in task_lower:
            alternatives.append("Alternative: Start with document structure analysis")
            alternatives.append("Alternative: Include document comparison if multiple documents")
        
        if 'research' in task_lower:
            alternatives.append("Alternative: Begin with broad web search, then narrow to specific sources")
            alternatives.append("Alternative: Include memory search for related past research")
        
        if len(plan) > 5:
            alternatives.append("Alternative: Break into smaller sub-tasks")
        
        return alternatives
    
    def _calculate_overall_risk_score(self, issues: List[ValidationIssue], base_risk: float) -> float:
        """Calculate overall risk score from issues and base risk."""
        
        # Weight issues by severity
        severity_weights = {
            ValidationSeverity.INFO: 0.1,
            ValidationSeverity.WARNING: 0.3,
            ValidationSeverity.ERROR: 0.6,
            ValidationSeverity.CRITICAL: 1.0
        }
        
        issue_risk = sum(severity_weights.get(issue.severity, 0.5) for issue in issues) * 0.1
        
        return min(1.0, base_risk + issue_risk)
    
    def _calculate_confidence_score(self, 
                                  plan: List[str], 
                                  issues: List[ValidationIssue],
                                  meta_analysis: Optional[Dict[str, Any]]) -> float:
        """Calculate confidence score for the plan."""
        
        base_confidence = 0.7  # Base confidence
        
        # Reduce confidence for issues
        critical_issues = sum(1 for issue in issues if issue.severity == ValidationSeverity.CRITICAL)
        error_issues = sum(1 for issue in issues if issue.severity == ValidationSeverity.ERROR)
        
        confidence_reduction = critical_issues * 0.3 + error_issues * 0.2
        
        # Adjust based on meta-reasoning if available
        meta_confidence_boost = 0.0
        if meta_analysis and 'meta_confidence' in meta_analysis:
            meta_confidence_boost = (meta_analysis['meta_confidence'] - 0.5) * 0.2
        
        final_confidence = max(0.0, min(1.0, base_confidence - confidence_reduction + meta_confidence_boost))
        
        return final_confidence
    
    def _determine_plan_validity(self, risk_score: float, confidence_score: float, issues: List[ValidationIssue]) -> bool:
        """Determine if the plan is valid based on validation results."""
        
        # Check for critical issues
        has_critical_issues = any(issue.severity == ValidationSeverity.CRITICAL for issue in issues)
        if has_critical_issues:
            return False
        
        # Check risk and confidence thresholds
        if risk_score > self.risk_threshold:
            return False
        
        if confidence_score < self.confidence_threshold:
            return False
        
        return True
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        
        rejection_rate = (self.plans_rejected / max(1, self.total_validations)) * 100
        high_risk_rate = (self.high_risk_plans / max(1, self.total_validations)) * 100
        
        return {
            'meta_reasoning_enabled': self.enable_meta_reasoning,
            'safety_validation_enabled': self.enable_safety_validation,
            'total_validations': self.total_validations,
            'plans_rejected': self.plans_rejected,
            'high_risk_plans': self.high_risk_plans,
            'rejection_rate': rejection_rate,
            'high_risk_rate': high_risk_rate,
            'risk_threshold': self.risk_threshold,
            'confidence_threshold': self.confidence_threshold
        }
