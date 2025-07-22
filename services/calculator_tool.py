#!/usr/bin/env python3
"""
Calculator Tool Service
Handles mathematical calculations and operations for the Smart Query Router.
"""

import logging
import re
import math
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CalculationResult:
    """Result of a mathematical calculation."""
    result: Union[float, int, str]
    expression: str
    explanation: str
    success: bool
    error_message: Optional[str] = None

class CalculatorTool:
    """
    Mathematical calculator tool that can handle various types of calculations:
    - Basic arithmetic (+, -, *, /)
    - Percentage calculations
    - Financial calculations (interest, etc.)
    - Unit conversions
    """
    
    def __init__(self):
        self.supported_operations = [
            'addition', 'subtraction', 'multiplication', 'division',
            'percentage', 'compound_interest', 'simple_interest',
            'square_root', 'power', 'logarithm'
        ]
        
        # Safe mathematical functions for eval
        self.safe_functions = {
            'sqrt': math.sqrt,
            'pow': math.pow,
            'log': math.log,
            'log10': math.log10,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'abs': abs,
            'round': round,
            'min': min,
            'max': max
        }
    
    def calculate(self, expression: str, context: Dict[str, Any] = None) -> CalculationResult:
        """
        Perform calculation based on the expression.
        
        Args:
            expression: Mathematical expression or natural language description
            context: Optional context including extracted values from documents
            
        Returns:
            CalculationResult with the computed value and explanation
        """
        context = context or {}
        
        try:
            # Clean and normalize the expression
            normalized_expr = self._normalize_expression(expression)
            
            # Detect calculation type
            calc_type = self._detect_calculation_type(normalized_expr)
            
            # Perform calculation based on type
            if calc_type == 'percentage':
                return self._calculate_percentage(normalized_expr, context)
            elif calc_type == 'basic_arithmetic':
                return self._calculate_arithmetic(normalized_expr, context)
            elif calc_type == 'financial':
                return self._calculate_financial(normalized_expr, context)
            elif calc_type == 'conversion':
                return self._calculate_conversion(normalized_expr, context)
            else:
                return self._calculate_general(normalized_expr, context)
                
        except Exception as e:
            logger.error(f"Calculation failed for '{expression}': {e}")
            return CalculationResult(
                result="Error",
                expression=expression,
                explanation=f"I couldn't calculate that: {str(e)}",
                success=False,
                error_message=str(e)
            )
    
    def _normalize_expression(self, expression: str) -> str:
        """Normalize the expression for calculation."""
        # Remove common words and normalize
        normalized = expression.lower().strip()
        
        # Replace common phrases
        replacements = {
            "what's": "",
            "what is": "",
            "calculate": "",
            "compute": "",
            "how much is": "",
            "percent of": "% of",
            "percentage of": "% of"
        }
        
        for phrase, replacement in replacements.items():
            normalized = normalized.replace(phrase, replacement)
        
        return normalized.strip()
    
    def _detect_calculation_type(self, expression: str) -> str:
        """Detect the type of calculation needed."""
        if re.search(r'\d+%\s+of\s+\d+', expression):
            return 'percentage'
        elif re.search(r'\d+\s*[\+\-\*\/]\s*\d+', expression):
            return 'basic_arithmetic'
        elif any(word in expression for word in ['interest', 'loan', 'investment']):
            return 'financial'
        elif any(word in expression for word in ['convert', 'to', 'from']):
            return 'conversion'
        else:
            return 'general'
    
    def _calculate_percentage(self, expression: str, context: Dict[str, Any]) -> CalculationResult:
        """Calculate percentage operations."""
        # Pattern: X% of Y
        match = re.search(r'(\d+(?:\.\d+)?)%\s+of\s+(\d+(?:\.\d+)?)', expression)
        if match:
            percentage = float(match.group(1))
            number = float(match.group(2))
            result = (percentage / 100) * number
            
            return CalculationResult(
                result=result,
                expression=f"{percentage}% of {number}",
                explanation=f"{percentage}% of {number} = {result}",
                success=True
            )
        
        # Pattern: X percent of Y
        match = re.search(r'(\d+(?:\.\d+)?)\s+percent\s+of\s+(\d+(?:\.\d+)?)', expression)
        if match:
            percentage = float(match.group(1))
            number = float(match.group(2))
            result = (percentage / 100) * number
            
            return CalculationResult(
                result=result,
                expression=f"{percentage}% of {number}",
                explanation=f"{percentage}% of {number} = {result}",
                success=True
            )
        
        raise ValueError("Could not parse percentage calculation")
    
    def _calculate_arithmetic(self, expression: str, context: Dict[str, Any]) -> CalculationResult:
        """Calculate basic arithmetic operations."""
        # Extract arithmetic expression
        match = re.search(r'(\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(\d+(?:\.\d+)?)', expression)
        if match:
            num1 = float(match.group(1))
            operator = match.group(2)
            num2 = float(match.group(3))
            
            operations = {
                '+': lambda x, y: x + y,
                '-': lambda x, y: x - y,
                '*': lambda x, y: x * y,
                '/': lambda x, y: x / y if y != 0 else float('inf')
            }
            
            if operator in operations:
                result = operations[operator](num1, num2)
                
                return CalculationResult(
                    result=result,
                    expression=f"{num1} {operator} {num2}",
                    explanation=f"{num1} {operator} {num2} = {result}",
                    success=True
                )
        
        raise ValueError("Could not parse arithmetic expression")
    
    def _calculate_financial(self, expression: str, context: Dict[str, Any]) -> CalculationResult:
        """Calculate financial operations like interest."""
        # Simple interest: P * R * T / 100
        if 'simple interest' in expression:
            # Try to extract principal, rate, time
            numbers = re.findall(r'\d+(?:\.\d+)?', expression)
            if len(numbers) >= 3:
                principal = float(numbers[0])
                rate = float(numbers[1])
                time = float(numbers[2])
                
                interest = (principal * rate * time) / 100
                
                return CalculationResult(
                    result=interest,
                    expression=f"Simple Interest: P={principal}, R={rate}%, T={time}",
                    explanation=f"Simple Interest = ({principal} × {rate} × {time}) ÷ 100 = {interest}",
                    success=True
                )
        
        # Compound interest: P(1 + r/n)^(nt) - P
        elif 'compound interest' in expression:
            numbers = re.findall(r'\d+(?:\.\d+)?', expression)
            if len(numbers) >= 3:
                principal = float(numbers[0])
                rate = float(numbers[1]) / 100  # Convert percentage
                time = float(numbers[2])
                n = 1  # Compounded annually by default
                
                amount = principal * (1 + rate/n) ** (n * time)
                interest = amount - principal
                
                return CalculationResult(
                    result=interest,
                    expression=f"Compound Interest: P={principal}, R={numbers[1]}%, T={time}",
                    explanation=f"Compound Interest = {principal}(1 + {numbers[1]}/100)^{time} - {principal} = {interest:.2f}",
                    success=True
                )
        
        raise ValueError("Could not parse financial calculation")
    
    def _calculate_conversion(self, expression: str, context: Dict[str, Any]) -> CalculationResult:
        """Calculate unit conversions."""
        # Basic temperature conversions
        if 'celsius' in expression and 'fahrenheit' in expression:
            celsius_match = re.search(r'(\d+(?:\.\d+)?)\s*celsius', expression)
            if celsius_match:
                celsius = float(celsius_match.group(1))
                fahrenheit = (celsius * 9/5) + 32
                
                return CalculationResult(
                    result=fahrenheit,
                    expression=f"{celsius}°C to °F",
                    explanation=f"{celsius}°C = {fahrenheit}°F",
                    success=True
                )
        
        raise ValueError("Could not parse conversion")
    
    def _calculate_general(self, expression: str, context: Dict[str, Any]) -> CalculationResult:
        """Handle general mathematical expressions safely."""
        try:
            # Remove any non-mathematical characters
            clean_expr = re.sub(r'[^\d\+\-\*\/\.\(\)\s]', '', expression)
            
            # Basic safety check
            if not re.match(r'^[\d\+\-\*\/\.\(\)\s]+$', clean_expr):
                raise ValueError("Expression contains invalid characters")
            
            # Evaluate safely
            result = eval(clean_expr, {"__builtins__": {}}, self.safe_functions)
            
            return CalculationResult(
                result=result,
                expression=clean_expr,
                explanation=f"{clean_expr} = {result}",
                success=True
            )
            
        except Exception as e:
            raise ValueError(f"Could not evaluate expression: {e}")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get information about calculator capabilities."""
        return {
            'supported_operations': self.supported_operations,
            'examples': [
                "What's 15% of 250?",
                "Calculate 45 + 67",
                "What is 100 * 1.5?",
                "Simple interest: 1000 at 5% for 2 years",
                "Convert 25 celsius to fahrenheit"
            ],
            'description': "I can perform various mathematical calculations including arithmetic, percentages, and financial calculations."
        }

# Global instance for easy access
_calculator_tool = None

def get_calculator_tool() -> CalculatorTool:
    """Get or create the global calculator tool instance."""
    global _calculator_tool
    if _calculator_tool is None:
        _calculator_tool = CalculatorTool()
    return _calculator_tool
