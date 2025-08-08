"""
CalculatorTool - Mathematical Computation Skill
===============================================

Provides secure mathematical computation capabilities using sandboxed execution.
Supports arithmetic, algebraic, and basic statistical operations.
"""

import re
import math
import time
import logging
from typing import Dict, Any, Optional, List, Union
from ..uif import SAM_UIF
from .base import BaseSkillModule, SkillExecutionError
from ..security import get_security_manager, SecurityPolicy

logger = logging.getLogger(__name__)


class CalculatorTool(BaseSkillModule):
    """
    Secure calculator tool for mathematical computations.
    
    Supports:
    - Basic arithmetic operations (+, -, *, /, %, **)
    - Mathematical functions (sin, cos, tan, log, sqrt, etc.)
    - Statistical operations (mean, median, std, etc.)
    - Constants (pi, e)
    - Safe expression evaluation
    """
    
    skill_name = "CalculatorTool"
    skill_version = "1.0.0"
    skill_description = "Performs secure mathematical computations and calculations"
    skill_category = "tools"
    
    # Dependency declarations
    required_inputs = []  # Can extract from input_query if calculation_expression not provided
    optional_inputs = ["calculation_expression", "calculation_context", "precision"]
    output_keys = ["calculation_result", "calculation_steps", "calculation_confidence"]
    
    # Skill characteristics
    requires_external_access = False
    requires_vetting = False
    can_run_parallel = True
    estimated_execution_time = 1.0
    max_execution_time = 5.0
    
    def __init__(self):
        super().__init__()
        self._security_manager = get_security_manager()
        self._setup_security_policy()
        self._math_functions = self._get_safe_math_functions()
    
    def _setup_security_policy(self) -> None:
        """Set up security policy for calculator operations."""
        policy = SecurityPolicy(
            allow_network_access=False,
            allow_file_system_access=False,
            max_execution_time=5.0,
            sandbox_enabled=True,
            allowed_commands=[],
            blocked_commands=["rm", "del", "format", "import", "exec", "eval"]
        )
        
        self._security_manager.register_tool_policy(self.skill_name, policy)
    
    def _get_safe_math_functions(self) -> Dict[str, Any]:
        """
        Get dictionary of safe mathematical functions.
        
        Returns:
            Dictionary mapping function names to implementations
        """
        return {
            # Basic math functions
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'pow': pow,
            
            # Math module functions
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'asin': math.asin,
            'acos': math.acos,
            'atan': math.atan,
            'sinh': math.sinh,
            'cosh': math.cosh,
            'tanh': math.tanh,
            'log': math.log,
            'log10': math.log10,
            'log2': math.log2,
            'exp': math.exp,
            'ceil': math.ceil,
            'floor': math.floor,
            'factorial': math.factorial,
            'degrees': math.degrees,
            'radians': math.radians,
            
            # Constants
            'pi': math.pi,
            'e': math.e,
            'tau': math.tau,
            
            # Statistical functions
            'mean': lambda x: sum(x) / len(x) if x else 0,
            'median': lambda x: sorted(x)[len(x)//2] if x else 0,
            'std': lambda x: math.sqrt(sum((i - sum(x)/len(x))**2 for i in x) / len(x)) if len(x) > 1 else 0,
        }
    
    def execute(self, uif: SAM_UIF) -> SAM_UIF:
        """
        Execute mathematical calculation with security controls.

        Args:
            uif: Universal Interface Format with calculation request

        Returns:
            Updated UIF with calculation results
        """
        # Initialize tracing
        trace_id = uif.intermediate_data.get('trace_id')
        start_time = time.time()

        if trace_id:
            self._log_trace_event(
                trace_id=trace_id,
                event_type="start",
                severity="info",
                message="Starting mathematical calculation",
                payload={
                    "tool": self.skill_name,
                    "input_query": uif.input_query,
                    "has_calculation_expression": "calculation_expression" in uif.intermediate_data
                }
            )

        try:
            # Extract calculation expression
            expression = self._extract_calculation_expression(uif)
            if not expression:
                if trace_id:
                    self._log_trace_event(
                        trace_id=trace_id,
                        event_type="error",
                        severity="error",
                        message="No calculation expression found",
                        payload={"input_query": uif.input_query}
                    )
                raise SkillExecutionError("No calculation expression found")

            if trace_id:
                self._log_trace_event(
                    trace_id=trace_id,
                    event_type="data_in",
                    severity="info",
                    message=f"Extracted calculation expression: {expression}",
                    payload={
                        "expression": expression,
                        "expression_length": len(expression),
                        "contains_functions": any(func in expression for func in ['sin', 'cos', 'log', 'sqrt'])
                    }
                )

            self.logger.info(f"Performing calculation: {expression}")

            # Get calculation context
            context = uif.intermediate_data.get("calculation_context", {})
            precision = uif.intermediate_data.get("precision", 10)

            # Perform secure calculation
            calc_start_time = time.time()
            result = self._perform_secure_calculation(expression, context, precision)
            calc_duration = (time.time() - calc_start_time) * 1000

            if trace_id:
                self._log_trace_event(
                    trace_id=trace_id,
                    event_type="tool_call",
                    severity="info",
                    message=f"Calculation completed: {expression} = {result['value']}",
                    duration_ms=calc_duration,
                    payload={
                        "expression": expression,
                        "result": result["value"],
                        "steps_count": len(result["steps"]),
                        "confidence": result["confidence"],
                        "precision": precision,
                        "context_variables": len(context)
                    }
                )

            # Store results in UIF
            uif.intermediate_data["calculation_result"] = result["value"]
            uif.intermediate_data["calculation_steps"] = result["steps"]
            uif.intermediate_data["calculation_confidence"] = result["confidence"]

            # Set skill outputs
            uif.set_skill_output(self.skill_name, {
                "expression": expression,
                "result": result["value"],
                "steps": result["steps"],
                "confidence": result["confidence"],
                "precision": precision
            })

            total_duration = (time.time() - start_time) * 1000

            if trace_id:
                self._log_trace_event(
                    trace_id=trace_id,
                    event_type="data_out",
                    severity="info",
                    message=f"Calculator tool execution completed successfully",
                    duration_ms=total_duration,
                    payload={
                        "result": result["value"],
                        "result_type": type(result["value"]).__name__,
                        "execution_time_ms": total_duration,
                        "calculation_time_ms": calc_duration,
                        "overhead_ms": total_duration - calc_duration
                    }
                )

            self.logger.info(f"Calculation completed: {expression} = {result['value']}")

            return uif

        except Exception as e:
            total_duration = (time.time() - start_time) * 1000

            if trace_id:
                self._log_trace_event(
                    trace_id=trace_id,
                    event_type="error",
                    severity="error",
                    message=f"Calculator execution failed: {str(e)}",
                    duration_ms=total_duration,
                    payload={
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "execution_time_ms": total_duration
                    }
                )

            self.logger.exception("Error during calculation")
            raise SkillExecutionError(f"Calculation failed: {str(e)}")
    
    def _extract_calculation_expression(self, uif: SAM_UIF) -> Optional[str]:
        """
        Extract calculation expression from UIF or query.
        
        Returns:
            Calculation expression or None if not found
        """
        # Check intermediate data first
        if "calculation_expression" in uif.intermediate_data:
            return uif.intermediate_data["calculation_expression"]
        
        # Try to extract from query
        query = uif.input_query.lower()
        
        # Look for mathematical expressions in the query
        math_patterns = [
            r'calculate\s+(.+)',
            r'compute\s+(.+)',
            r'what\s+is\s+(.+)',
            r'solve\s+(.+)',
            r'evaluate\s+(.+)',
            r'find\s+(.+)',
        ]
        
        for pattern in math_patterns:
            match = re.search(pattern, query)
            if match:
                expression = match.group(1).strip()
                # Clean up the expression
                expression = self._clean_expression(expression)
                if self._is_valid_math_expression(expression):
                    return expression
        
        # Look for direct mathematical expressions
        # Simple pattern for basic math expressions
        math_expr_pattern = r'[\d\+\-\*/\(\)\.\s\^]+'
        if re.match(math_expr_pattern, query.replace(' ', '')):
            cleaned = self._clean_expression(query)
            if self._is_valid_math_expression(cleaned):
                return cleaned
        
        return None
    
    def _clean_expression(self, expression: str) -> str:
        """
        Clean and normalize mathematical expression.
        
        Returns:
            Cleaned expression
        """
        # Remove common words
        words_to_remove = [
            'calculate', 'compute', 'what', 'is', 'the', 'value', 'of',
            'solve', 'evaluate', 'find', 'result', 'answer', '?', '.'
        ]
        
        cleaned = expression.lower()
        for word in words_to_remove:
            cleaned = cleaned.replace(word, ' ')
        
        # Normalize operators
        cleaned = cleaned.replace('^', '**')  # Power operator
        cleaned = cleaned.replace('x', '*')   # Multiplication
        cleaned = cleaned.replace('÷', '/')   # Division
        
        # Remove extra spaces
        cleaned = ' '.join(cleaned.split())
        
        return cleaned.strip()
    
    def _is_valid_math_expression(self, expression: str) -> bool:
        """
        Check if expression is a valid mathematical expression.
        
        Returns:
            True if expression appears to be valid math
        """
        if not expression:
            return False
        
        # Check for dangerous patterns
        dangerous_patterns = [
            'import', 'exec', 'eval', 'open', 'file', '__',
            'subprocess', 'os.', 'sys.', 'globals', 'locals'
        ]
        
        for pattern in dangerous_patterns:
            if pattern in expression.lower():
                return False
        
        # Check for basic math patterns
        valid_chars = set('0123456789+-*/().** abcdefghijklmnopqrstuvwxyz')
        if not all(c in valid_chars for c in expression.lower()):
            return False
        
        return True
    
    def _perform_secure_calculation(self, expression: str, context: Dict[str, Any], precision: int) -> Dict[str, Any]:
        """
        Perform calculation with security controls.
        
        Returns:
            Dictionary with calculation results
        """
        def calculation_func():
            return self._evaluate_expression(expression, context, precision)
        
        # Execute with security manager
        security_result = self._security_manager.execute_tool_safely(
            self.skill_name,
            calculation_func
        )
        
        if not security_result.success:
            if security_result.rate_limited:
                raise SkillExecutionError("Calculator rate limit exceeded")
            elif security_result.security_violations:
                raise SkillExecutionError(f"Security violations: {security_result.security_violations}")
            else:
                raise SkillExecutionError(f"Calculation failed: {security_result.error_message}")
        
        return security_result.output
    
    def _evaluate_expression(self, expression: str, context: Dict[str, Any], precision: int) -> Dict[str, Any]:
        """
        Evaluate mathematical expression safely.
        
        Returns:
            Dictionary with calculation results
        """
        steps = []
        confidence = 0.9
        
        try:
            # Prepare safe namespace
            safe_namespace = self._math_functions.copy()
            
            # Add context variables if provided
            if context:
                for key, value in context.items():
                    if isinstance(value, (int, float)) and not key.startswith('_'):
                        safe_namespace[key] = value
            
            # Record the original expression
            steps.append(f"Original expression: {expression}")
            
            # Preprocess expression for function calls
            processed_expr = self._preprocess_expression(expression)
            steps.append(f"Processed expression: {processed_expr}")
            
            # Evaluate the expression
            result = eval(processed_expr, {"__builtins__": {}}, safe_namespace)
            
            # Handle different result types
            if isinstance(result, complex):
                if result.imag == 0:
                    result = result.real
                else:
                    result = f"{result.real:.{precision}f} + {result.imag:.{precision}f}i"
                    confidence = 0.8
            elif isinstance(result, float):
                if math.isinf(result):
                    result = "∞" if result > 0 else "-∞"
                    confidence = 0.7
                elif math.isnan(result):
                    result = "NaN (Not a Number)"
                    confidence = 0.5
                else:
                    result = round(result, precision)
            elif isinstance(result, int):
                # Keep integers as integers
                pass
            else:
                result = str(result)
                confidence = 0.6
            
            steps.append(f"Final result: {result}")
            
            return {
                "value": result,
                "steps": steps,
                "confidence": confidence
            }
            
        except ZeroDivisionError:
            steps.append("Error: Division by zero")
            return {
                "value": "Error: Division by zero",
                "steps": steps,
                "confidence": 0.0
            }
        except ValueError as e:
            steps.append(f"Error: Invalid value - {str(e)}")
            return {
                "value": f"Error: Invalid value - {str(e)}",
                "steps": steps,
                "confidence": 0.0
            }
        except Exception as e:
            steps.append(f"Error: {str(e)}")
            return {
                "value": f"Error: {str(e)}",
                "steps": steps,
                "confidence": 0.0
            }
    
    def _preprocess_expression(self, expression: str) -> str:
        """
        Preprocess expression to handle function calls and special cases.
        
        Returns:
            Preprocessed expression
        """
        # Handle implicit multiplication (e.g., "2pi" -> "2*pi")
        expression = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expression)
        expression = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', expression)
        
        # Handle parentheses multiplication (e.g., "2(3+4)" -> "2*(3+4)")
        expression = re.sub(r'(\d)\(', r'\1*(', expression)
        expression = re.sub(r'\)(\d)', r')*\1', expression)
        
        return expression
    
    def can_handle_query(self, query: str) -> bool:
        """
        Check if this tool can handle the given query.
        
        Args:
            query: User query to check
            
        Returns:
            True if query appears to be mathematical
        """
        math_keywords = [
            'calculate', 'compute', 'math', 'arithmetic', 'add', 'subtract',
            'multiply', 'divide', 'sum', 'product', 'square', 'root',
            'sin', 'cos', 'tan', 'log', 'exp', 'factorial', 'percentage'
        ]
        
        query_lower = query.lower()
        
        # Check for math keywords
        if any(keyword in query_lower for keyword in math_keywords):
            return True
        
        # Check for mathematical symbols
        math_symbols = ['+', '-', '*', '/', '=', '^', '%', '(', ')']
        if any(symbol in query for symbol in math_symbols):
            return True
        
        # Check for numbers
        if re.search(r'\d', query):
            return True
        
        return False

    def _log_trace_event(self, trace_id: str, event_type: str, severity: str,
                        message: str, duration_ms: Optional[float] = None,
                        payload: Optional[Dict[str, Any]] = None) -> None:
        """Log a trace event for the calculator tool."""
        try:
            from sam.cognition.trace_logger import log_event
            log_event(
                trace_id=trace_id,
                source_module=self.skill_name,
                event_type=event_type,
                severity=severity,
                message=message,
                duration_ms=duration_ms,
                payload=payload or {},
                metadata={
                    "tool_version": self.skill_version,
                    "tool_category": self.skill_category,
                    "requires_external_access": self.requires_external_access
                }
            )
        except ImportError:
            # Tracing not available, continue without logging
            pass
        except Exception as e:
            # Don't let tracing errors break the tool
            self.logger.debug(f"Trace logging failed: {e}")
