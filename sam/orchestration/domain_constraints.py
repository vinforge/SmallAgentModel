"""
PINN-Inspired Domain-Informed Constraints System

Implements constraint management framework for policy enforcement and safety compliance,
inspired by Physics-Informed Neural Networks' constraint satisfaction techniques.
Provides flexible, YAML-configurable constraint rules with real-time enforcement.

Key Features:
- YAML-based constraint configuration
- Real-time constraint validation
- Policy enforcement mechanisms
- Safety compliance monitoring
- Constraint violation reporting

Author: SAM Development Team
Version: 1.0.0
"""

import logging
import yaml
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

class ConstraintType(Enum):
    """Types of constraints that can be enforced."""
    CONTENT_FILTER = "content_filter"         # Content-based restrictions
    SKILL_RESTRICTION = "skill_restriction"   # Skill usage limitations
    RESOURCE_LIMIT = "resource_limit"         # Resource usage constraints
    SAFETY_POLICY = "safety_policy"           # Safety and compliance rules
    DOMAIN_SPECIFIC = "domain_specific"       # Domain-specific constraints
    TEMPORAL_LIMIT = "temporal_limit"         # Time-based restrictions

class ConstraintSeverity(Enum):
    """Severity levels for constraint violations."""
    INFO = "info"           # Informational only
    WARNING = "warning"     # Warning but allow continuation
    ERROR = "error"         # Block execution but allow fallback
    CRITICAL = "critical"   # Hard stop, no fallback allowed

@dataclass
class ConstraintRule:
    """Individual constraint rule definition."""
    name: str
    constraint_type: ConstraintType
    severity: ConstraintSeverity
    description: str
    pattern: Optional[str] = None           # Regex pattern for matching
    keywords: Optional[List[str]] = None    # Keywords to check
    max_value: Optional[float] = None       # Maximum allowed value
    min_value: Optional[float] = None       # Minimum allowed value
    allowed_skills: Optional[List[str]] = None  # Allowed skills list
    blocked_skills: Optional[List[str]] = None  # Blocked skills list
    conditions: Optional[Dict[str, Any]] = None  # Additional conditions
    enabled: bool = True

@dataclass
class ConstraintViolation:
    """Constraint violation report."""
    rule_name: str
    constraint_type: ConstraintType
    severity: ConstraintSeverity
    description: str
    violation_details: str
    suggested_action: str
    context: Dict[str, Any]

@dataclass
class ConstraintValidationResult:
    """Result of constraint validation."""
    is_valid: bool
    violations: List[ConstraintViolation]
    warnings: List[str]
    allowed_skills: List[str]
    blocked_skills: List[str]
    resource_limits: Dict[str, float]

class DomainConstraints:
    """
    PINN-inspired domain constraint management system.
    
    Enforces domain-specific constraints and policies similar to how
    PINNs enforce physical laws and boundary conditions during training.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        enable_strict_mode: bool = False,
        default_severity: ConstraintSeverity = ConstraintSeverity.WARNING
    ):
        """
        Initialize domain constraints system.
        
        Args:
            config_path: Path to YAML constraint configuration file
            enable_strict_mode: Whether to enforce all constraints strictly
            default_severity: Default severity for undefined constraints
        """
        self.enable_strict_mode = enable_strict_mode
        self.default_severity = default_severity
        
        # Constraint rules storage
        self.constraint_rules: Dict[str, ConstraintRule] = {}
        
        # Violation tracking
        self.violation_history: List[ConstraintViolation] = []
        
        # Load constraint configuration
        if config_path:
            self.load_constraints_from_file(config_path)
        else:
            self._initialize_default_constraints()
        
        logger.info(f"DomainConstraints initialized with {len(self.constraint_rules)} rules")
    
    def _initialize_default_constraints(self) -> None:
        """Initialize default constraint rules."""
        default_rules = [
            ConstraintRule(
                name="no_harmful_content",
                constraint_type=ConstraintType.CONTENT_FILTER,
                severity=ConstraintSeverity.ERROR,
                description="Block harmful or inappropriate content",
                keywords=["violence", "hate", "illegal", "harmful", "dangerous"],
                enabled=True
            ),
            
            ConstraintRule(
                name="memory_usage_limit",
                constraint_type=ConstraintType.RESOURCE_LIMIT,
                severity=ConstraintSeverity.WARNING,
                description="Limit memory retrieval results",
                max_value=50.0,  # Max memory results
                enabled=True
            ),
            
            ConstraintRule(
                name="execution_time_limit",
                constraint_type=ConstraintType.TEMPORAL_LIMIT,
                severity=ConstraintSeverity.ERROR,
                description="Limit total execution time",
                max_value=300.0,  # 5 minutes max
                enabled=True
            ),
            
            ConstraintRule(
                name="safe_web_browsing",
                constraint_type=ConstraintType.SAFETY_POLICY,
                severity=ConstraintSeverity.WARNING,
                description="Ensure safe web browsing practices",
                pattern=r"https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                enabled=True
            ),
            
            ConstraintRule(
                name="no_system_modification",
                constraint_type=ConstraintType.SKILL_RESTRICTION,
                severity=ConstraintSeverity.CRITICAL,
                description="Block system modification skills",
                blocked_skills=["SystemModificationSkill", "FileSystemSkill"],
                enabled=True
            )
        ]
        
        for rule in default_rules:
            self.constraint_rules[rule.name] = rule
    
    def load_constraints_from_file(self, config_path: str) -> None:
        """
        Load constraint rules from YAML configuration file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Constraint config file not found: {config_path}")
                self._initialize_default_constraints()
                return
            
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if not config_data or 'constraints' not in config_data:
                logger.warning("Invalid constraint configuration format")
                self._initialize_default_constraints()
                return
            
            # Parse constraint rules
            for rule_data in config_data['constraints']:
                rule = self._parse_constraint_rule(rule_data)
                if rule:
                    self.constraint_rules[rule.name] = rule
            
            logger.info(f"Loaded {len(self.constraint_rules)} constraint rules from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading constraint configuration: {e}")
            self._initialize_default_constraints()
    
    def _parse_constraint_rule(self, rule_data: Dict[str, Any]) -> Optional[ConstraintRule]:
        """Parse a constraint rule from configuration data."""
        try:
            return ConstraintRule(
                name=rule_data['name'],
                constraint_type=ConstraintType(rule_data['type']),
                severity=ConstraintSeverity(rule_data.get('severity', 'warning')),
                description=rule_data['description'],
                pattern=rule_data.get('pattern'),
                keywords=rule_data.get('keywords'),
                max_value=rule_data.get('max_value'),
                min_value=rule_data.get('min_value'),
                allowed_skills=rule_data.get('allowed_skills'),
                blocked_skills=rule_data.get('blocked_skills'),
                conditions=rule_data.get('conditions'),
                enabled=rule_data.get('enabled', True)
            )
        except (KeyError, ValueError) as e:
            logger.error(f"Error parsing constraint rule: {e}")
            return None
    
    def validate_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> ConstraintValidationResult:
        """
        Validate a query against all constraint rules.
        
        Args:
            query: The input query to validate
            context: Additional context for validation
            
        Returns:
            Constraint validation result
        """
        violations = []
        warnings = []
        context = context or {}
        
        for rule_name, rule in self.constraint_rules.items():
            if not rule.enabled:
                continue
            
            violation = self._check_content_constraint(query, rule, context)
            if violation:
                violations.append(violation)
                
                if violation.severity == ConstraintSeverity.WARNING:
                    warnings.append(f"{rule_name}: {violation.violation_details}")
        
        is_valid = not any(v.severity in [ConstraintSeverity.ERROR, ConstraintSeverity.CRITICAL] 
                          for v in violations)
        
        return ConstraintValidationResult(
            is_valid=is_valid,
            violations=violations,
            warnings=warnings,
            allowed_skills=[],  # Will be populated by validate_plan
            blocked_skills=[],  # Will be populated by validate_plan
            resource_limits={}  # Will be populated by validate_plan
        )
    
    def validate_plan(self, plan: List[str], context: Optional[Dict[str, Any]] = None) -> ConstraintValidationResult:
        """
        Validate an execution plan against constraint rules.
        
        Args:
            plan: List of skill names in execution plan
            context: Additional context for validation
            
        Returns:
            Constraint validation result
        """
        violations = []
        warnings = []
        allowed_skills = []
        blocked_skills = []
        resource_limits = {}
        context = context or {}
        
        for rule_name, rule in self.constraint_rules.items():
            if not rule.enabled:
                continue
            
            # Check skill restrictions
            if rule.constraint_type == ConstraintType.SKILL_RESTRICTION:
                violation = self._check_skill_constraint(plan, rule, context)
                if violation:
                    violations.append(violation)
                    
                    if rule.blocked_skills:
                        blocked_skills.extend(rule.blocked_skills)
                    
                    if violation.severity == ConstraintSeverity.WARNING:
                        warnings.append(f"{rule_name}: {violation.violation_details}")
            
            # Check resource limits
            elif rule.constraint_type == ConstraintType.RESOURCE_LIMIT:
                if rule.max_value is not None:
                    resource_limits[rule_name] = rule.max_value
            
            # Check allowed skills
            if rule.allowed_skills:
                allowed_skills.extend(rule.allowed_skills)
        
        # Remove duplicates
        allowed_skills = list(set(allowed_skills))
        blocked_skills = list(set(blocked_skills))
        
        is_valid = not any(v.severity in [ConstraintSeverity.ERROR, ConstraintSeverity.CRITICAL] 
                          for v in violations)
        
        return ConstraintValidationResult(
            is_valid=is_valid,
            violations=violations,
            warnings=warnings,
            allowed_skills=allowed_skills,
            blocked_skills=blocked_skills,
            resource_limits=resource_limits
        )
    
    def _check_content_constraint(
        self,
        content: str,
        rule: ConstraintRule,
        context: Dict[str, Any]
    ) -> Optional[ConstraintViolation]:
        """Check content against a constraint rule."""
        if rule.constraint_type != ConstraintType.CONTENT_FILTER:
            return None
        
        content_lower = content.lower()
        
        # Check keywords
        if rule.keywords:
            for keyword in rule.keywords:
                if keyword.lower() in content_lower:
                    return ConstraintViolation(
                        rule_name=rule.name,
                        constraint_type=rule.constraint_type,
                        severity=rule.severity,
                        description=rule.description,
                        violation_details=f"Content contains blocked keyword: '{keyword}'",
                        suggested_action="Rephrase query to avoid blocked content",
                        context=context
                    )
        
        # Check regex pattern
        if rule.pattern:
            if re.search(rule.pattern, content, re.IGNORECASE):
                return ConstraintViolation(
                    rule_name=rule.name,
                    constraint_type=rule.constraint_type,
                    severity=rule.severity,
                    description=rule.description,
                    violation_details=f"Content matches blocked pattern: {rule.pattern}",
                    suggested_action="Modify content to avoid pattern match",
                    context=context
                )
        
        return None
    
    def _check_skill_constraint(
        self,
        plan: List[str],
        rule: ConstraintRule,
        context: Dict[str, Any]
    ) -> Optional[ConstraintViolation]:
        """Check plan against skill constraint rule."""
        if rule.constraint_type != ConstraintType.SKILL_RESTRICTION:
            return None
        
        # Check blocked skills
        if rule.blocked_skills:
            for skill in plan:
                if skill in rule.blocked_skills:
                    return ConstraintViolation(
                        rule_name=rule.name,
                        constraint_type=rule.constraint_type,
                        severity=rule.severity,
                        description=rule.description,
                        violation_details=f"Plan contains blocked skill: '{skill}'",
                        suggested_action="Remove blocked skill from execution plan",
                        context=context
                    )
        
        # Check allowed skills (if specified, only these are allowed)
        if rule.allowed_skills:
            for skill in plan:
                if skill not in rule.allowed_skills:
                    return ConstraintViolation(
                        rule_name=rule.name,
                        constraint_type=rule.constraint_type,
                        severity=rule.severity,
                        description=rule.description,
                        violation_details=f"Plan contains non-allowed skill: '{skill}'",
                        suggested_action="Use only allowed skills in execution plan",
                        context=context
                    )
        
        return None
    
    def record_violation(self, violation: ConstraintViolation) -> None:
        """Record a constraint violation for tracking."""
        self.violation_history.append(violation)
        
        # Keep only recent violations
        if len(self.violation_history) > 1000:
            self.violation_history = self.violation_history[-500:]
        
        # Log violation based on severity
        if violation.severity == ConstraintSeverity.CRITICAL:
            logger.critical(f"CRITICAL constraint violation: {violation.violation_details}")
        elif violation.severity == ConstraintSeverity.ERROR:
            logger.error(f"Constraint violation: {violation.violation_details}")
        elif violation.severity == ConstraintSeverity.WARNING:
            logger.warning(f"Constraint warning: {violation.violation_details}")
        else:
            logger.info(f"Constraint info: {violation.violation_details}")
    
    def get_constraint_statistics(self) -> Dict[str, Any]:
        """Get statistics about constraint enforcement."""
        if not self.violation_history:
            return {
                "total_violations": 0,
                "violations_by_severity": {},
                "violations_by_type": {},
                "most_violated_rules": []
            }
        
        # Count violations by severity
        severity_counts = {}
        for violation in self.violation_history:
            severity = violation.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Count violations by type
        type_counts = {}
        for violation in self.violation_history:
            constraint_type = violation.constraint_type.value
            type_counts[constraint_type] = type_counts.get(constraint_type, 0) + 1
        
        # Find most violated rules
        rule_counts = {}
        for violation in self.violation_history:
            rule_name = violation.rule_name
            rule_counts[rule_name] = rule_counts.get(rule_name, 0) + 1
        
        most_violated = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_violations": len(self.violation_history),
            "violations_by_severity": severity_counts,
            "violations_by_type": type_counts,
            "most_violated_rules": most_violated,
            "total_rules": len(self.constraint_rules),
            "enabled_rules": sum(1 for rule in self.constraint_rules.values() if rule.enabled),
            "strict_mode": self.enable_strict_mode
        }
