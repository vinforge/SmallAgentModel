"""
Program Validator
================

Safety validation system for latent programs to ensure they don't contain
harmful patterns or violate security constraints.
"""

import re
import logging
from typing import List, Dict, Any
from dataclasses import dataclass

from .latent_program import LatentProgram, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class ValidationCheck:
    """Result of a single validation check."""
    name: str
    passed: bool
    warning: str = ""
    recommendation: str = ""
    risk_score: float = 0.0


class ProgramValidator:
    """
    Safety validation system for latent programs.
    
    Ensures programs don't contain harmful patterns, respect resource limits,
    and comply with security and privacy requirements.
    """
    
    def __init__(self):
        """Initialize the validator with safety rules."""
        self.max_execution_time_ms = 30000  # 30 seconds
        self.max_token_count = 4000
        self.max_memory_usage_mb = 100
        
        # Dangerous patterns to detect
        self.dangerous_patterns = [
            r'exec\s*\(',
            r'eval\s*\(',
            r'__import__',
            r'subprocess',
            r'os\.system',
            r'open\s*\(',
            r'file\s*\(',
            r'input\s*\(',
            r'raw_input\s*\('
        ]
        
        # Prompt injection patterns
        self.injection_patterns = [
            r'ignore\s+previous\s+instructions',
            r'forget\s+everything',
            r'new\s+instructions',
            r'system\s*:\s*',
            r'admin\s*:\s*',
            r'override\s+safety',
            r'jailbreak',
            r'developer\s+mode'
        ]
        
        logger.info("Program validator initialized")
    
    def validate_program_safety(self, program: LatentProgram) -> ValidationResult:
        """
        Comprehensive safety validation of a latent program.
        
        Args:
            program: The latent program to validate
            
        Returns:
            ValidationResult with safety assessment
        """
        try:
            checks = []
            
            # Run all validation checks
            checks.append(self._check_resource_limits(program))
            checks.append(self._check_prompt_injection_resistance(program))
            checks.append(self._check_output_safety_constraints(program))
            checks.append(self._check_user_privacy_compliance(program))
            checks.append(self._check_execution_safety(program))
            checks.append(self._check_configuration_safety(program))
            
            # Aggregate results
            all_passed = all(check.passed for check in checks)
            warnings = [check.warning for check in checks if check.warning]
            recommendations = [check.recommendation for check in checks if check.recommendation]
            
            # Calculate overall risk score
            risk_scores = [check.risk_score for check in checks]
            overall_risk = max(risk_scores) if risk_scores else 0.0
            
            result = ValidationResult(
                is_safe=all_passed and overall_risk < 0.7,
                warnings=warnings,
                recommendations=recommendations,
                risk_score=overall_risk
            )
            
            if not result.is_safe:
                logger.warning(f"Program {program.id} failed safety validation")
            
            return result
            
        except Exception as e:
            logger.error(f"Validation error for program {program.id}: {e}")
            return ValidationResult(
                is_safe=False,
                warnings=[f"Validation error: {e}"],
                recommendations=["Manual review required"],
                risk_score=1.0
            )
    
    def _check_resource_limits(self, program: LatentProgram) -> ValidationCheck:
        """Check if program respects resource limits."""
        try:
            warnings = []
            recommendations = []
            risk_score = 0.0
            
            # Check execution time limits
            if program.avg_latency_ms > self.max_execution_time_ms:
                warnings.append(f"Average execution time {program.avg_latency_ms}ms exceeds limit")
                recommendations.append("Consider optimizing program execution")
                risk_score = max(risk_score, 0.3)
            
            # Check token count limits
            if program.avg_token_count > self.max_token_count:
                warnings.append(f"Average token count {program.avg_token_count} exceeds limit")
                recommendations.append("Consider reducing response length")
                risk_score = max(risk_score, 0.2)
            
            # Check execution constraints
            constraints = program.execution_constraints
            if constraints.get('max_tokens', 0) > self.max_token_count:
                warnings.append("Program allows excessive token usage")
                risk_score = max(risk_score, 0.4)
            
            if constraints.get('timeout_seconds', 0) > 60:
                warnings.append("Program allows excessive execution time")
                risk_score = max(risk_score, 0.3)
            
            passed = len(warnings) == 0
            
            return ValidationCheck(
                name="resource_limits",
                passed=passed,
                warning="; ".join(warnings) if warnings else "",
                recommendation="; ".join(recommendations) if recommendations else "",
                risk_score=risk_score
            )
            
        except Exception as e:
            return ValidationCheck(
                name="resource_limits",
                passed=False,
                warning=f"Resource check failed: {e}",
                risk_score=0.5
            )
    
    def _check_prompt_injection_resistance(self, program: LatentProgram) -> ValidationCheck:
        """Check resistance to prompt injection attacks."""
        try:
            warnings = []
            risk_score = 0.0
            
            # Check prompt template for injection vulnerabilities
            template = program.prompt_template_used.lower()
            
            for pattern in self.injection_patterns:
                if re.search(pattern, template, re.IGNORECASE):
                    warnings.append(f"Potential injection pattern detected: {pattern}")
                    risk_score = max(risk_score, 0.8)
            
            # Check reasoning trace for suspicious patterns
            trace_text = str(program.reasoning_trace).lower()
            for pattern in self.injection_patterns:
                if re.search(pattern, trace_text, re.IGNORECASE):
                    warnings.append(f"Injection pattern in reasoning trace: {pattern}")
                    risk_score = max(risk_score, 0.6)
            
            passed = len(warnings) == 0
            recommendation = "Review and sanitize prompt templates" if warnings else ""
            
            return ValidationCheck(
                name="prompt_injection_resistance",
                passed=passed,
                warning="; ".join(warnings) if warnings else "",
                recommendation=recommendation,
                risk_score=risk_score
            )
            
        except Exception as e:
            return ValidationCheck(
                name="prompt_injection_resistance",
                passed=False,
                warning=f"Injection check failed: {e}",
                risk_score=0.5
            )
    
    def _check_output_safety_constraints(self, program: LatentProgram) -> ValidationCheck:
        """Check output safety constraints."""
        try:
            warnings = []
            risk_score = 0.0
            
            # Check for dangerous code execution patterns
            template = program.prompt_template_used
            trace_text = str(program.reasoning_trace)
            
            combined_text = f"{template} {trace_text}".lower()
            
            for pattern in self.dangerous_patterns:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    warnings.append(f"Dangerous execution pattern detected: {pattern}")
                    risk_score = max(risk_score, 0.9)
            
            # Check for file system access patterns
            file_patterns = [r'\.\./', r'/etc/', r'/var/', r'C:\\', r'~/', r'\$HOME']
            for pattern in file_patterns:
                if re.search(pattern, combined_text):
                    warnings.append(f"File system access pattern detected: {pattern}")
                    risk_score = max(risk_score, 0.7)
            
            passed = len(warnings) == 0
            recommendation = "Remove dangerous execution patterns" if warnings else ""
            
            return ValidationCheck(
                name="output_safety",
                passed=passed,
                warning="; ".join(warnings) if warnings else "",
                recommendation=recommendation,
                risk_score=risk_score
            )
            
        except Exception as e:
            return ValidationCheck(
                name="output_safety",
                passed=False,
                warning=f"Output safety check failed: {e}",
                risk_score=0.5
            )
    
    def _check_user_privacy_compliance(self, program: LatentProgram) -> ValidationCheck:
        """Check user privacy compliance."""
        try:
            warnings = []
            risk_score = 0.0
            
            # Check for PII patterns in stored data
            pii_patterns = [
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
                r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'  # Phone number
            ]
            
            combined_text = str(program.to_dict())
            
            for pattern in pii_patterns:
                if re.search(pattern, combined_text):
                    warnings.append("Potential PII detected in program data")
                    risk_score = max(risk_score, 0.8)
                    break
            
            # Check context requirements for privacy concerns
            context_req = program.context_requirements
            if context_req.get('store_user_data', False):
                warnings.append("Program configured to store user data")
                risk_score = max(risk_score, 0.4)
            
            passed = len(warnings) == 0
            recommendation = "Remove or anonymize personal information" if warnings else ""
            
            return ValidationCheck(
                name="privacy_compliance",
                passed=passed,
                warning="; ".join(warnings) if warnings else "",
                recommendation=recommendation,
                risk_score=risk_score
            )
            
        except Exception as e:
            return ValidationCheck(
                name="privacy_compliance",
                passed=False,
                warning=f"Privacy check failed: {e}",
                risk_score=0.5
            )
    
    def _check_execution_safety(self, program: LatentProgram) -> ValidationCheck:
        """Check execution safety parameters."""
        try:
            warnings = []
            risk_score = 0.0
            
            # Check TPV configuration for unsafe settings
            tpv_config = program.tpv_config
            
            if tpv_config.get('allow_code_execution', False):
                warnings.append("Program allows code execution")
                risk_score = max(risk_score, 0.9)
            
            if tpv_config.get('disable_safety_checks', False):
                warnings.append("Program disables safety checks")
                risk_score = max(risk_score, 0.8)
            
            # Check for excessive permissions
            if tpv_config.get('admin_mode', False):
                warnings.append("Program requests admin mode")
                risk_score = max(risk_score, 0.7)
            
            passed = len(warnings) == 0
            recommendation = "Review and restrict execution permissions" if warnings else ""
            
            return ValidationCheck(
                name="execution_safety",
                passed=passed,
                warning="; ".join(warnings) if warnings else "",
                recommendation=recommendation,
                risk_score=risk_score
            )
            
        except Exception as e:
            return ValidationCheck(
                name="execution_safety",
                passed=False,
                warning=f"Execution safety check failed: {e}",
                risk_score=0.5
            )
    
    def _check_configuration_safety(self, program: LatentProgram) -> ValidationCheck:
        """Check configuration safety."""
        try:
            warnings = []
            risk_score = 0.0
            
            # Check for suspicious configuration values
            if program.confidence_score > 1.0 or program.confidence_score < 0.0:
                warnings.append("Invalid confidence score")
                risk_score = max(risk_score, 0.3)
            
            if program.success_rate > 1.0 or program.success_rate < 0.0:
                warnings.append("Invalid success rate")
                risk_score = max(risk_score, 0.3)
            
            if program.user_feedback_score > 5.0 or program.user_feedback_score < 0.0:
                warnings.append("Invalid feedback score")
                risk_score = max(risk_score, 0.2)
            
            # Check for reasonable usage patterns
            if program.usage_count < 0:
                warnings.append("Invalid usage count")
                risk_score = max(risk_score, 0.4)
            
            if program.avg_latency_ms < 0:
                warnings.append("Invalid latency value")
                risk_score = max(risk_score, 0.3)
            
            passed = len(warnings) == 0
            recommendation = "Correct invalid configuration values" if warnings else ""
            
            return ValidationCheck(
                name="configuration_safety",
                passed=passed,
                warning="; ".join(warnings) if warnings else "",
                recommendation=recommendation,
                risk_score=risk_score
            )
            
        except Exception as e:
            return ValidationCheck(
                name="configuration_safety",
                passed=False,
                warning=f"Configuration check failed: {e}",
                risk_score=0.5
            )
