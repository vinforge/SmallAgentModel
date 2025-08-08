#!/usr/bin/env python3
"""
STEF Programs Package
Contains all available TaskProgram definitions.
"""

from .fact_check_calculator import (
    fact_check_and_calculate_program,
    simple_calculation_program,
    document_calculation_program
)

__all__ = [
    'fact_check_and_calculate_program',
    'simple_calculation_program', 
    'document_calculation_program'
]
