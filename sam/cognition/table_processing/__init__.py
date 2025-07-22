"""
Table Processing Module for SAM
===============================

Enhanced table intelligence system that provides semantic role classification
on detected tables, enabling SAM to understand table structures beyond simple
text extraction.

Core Components:
- TableParser: Multi-modal table detection and extraction
- TableRoleClassifier: Context-aware semantic role classification
- TableValidator: Quality validation and structure verification
- TableEnhancer: Post-processing enhancements
- TokenRoles: 9 semantic role definitions

This implements the Neuro-Symbolic Router for foundational table intelligence
as defined in task23.md Phase 1.
"""

from .table_parser import TableParser, TableObject
from .role_classifier import TableRoleClassifier, RoleClassification
from .table_validator import TableValidator, ValidationResult
from .table_enhancer import TableEnhancer, EnhancementResult
from .token_roles import TokenRole, SEMANTIC_ROLES
from .config import TableProcessingConfig, get_table_config
from .utils import TableUtils, CoordinateSystem

__version__ = "1.0.0"
__author__ = "SAM Development Team"
__phase__ = "Phase 1 - Foundational Table Intelligence"

# Module-level configuration
TABLE_CONFIG = {
    "detection_confidence_threshold": 0.7,
    "classification_confidence_threshold": 0.8,
    "validation_quality_threshold": 0.6,
    "max_table_size": (100, 50),  # (rows, columns)
    "enable_context_awareness": True,
    "enable_quality_validation": True,
    "enable_post_processing": True,
    "model_paths": {
        "role_classifier": "sam/assets/table_role_classifier.bin",
        "table_detector": "sam/assets/table_detector.bin"
    }
}

def get_table_processing_config():
    """Get the current table processing configuration."""
    return TABLE_CONFIG.copy()

def update_table_processing_config(updates: dict):
    """Update table processing configuration."""
    TABLE_CONFIG.update(updates)

def initialize_table_processing():
    """Initialize the table processing system."""
    try:
        # Initialize core components
        parser = TableParser()
        classifier = TableRoleClassifier()
        validator = TableValidator()
        enhancer = TableEnhancer()
        
        return {
            "parser": parser,
            "classifier": classifier,
            "validator": validator,
            "enhancer": enhancer,
            "status": "initialized"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

# Export main classes for easy import
__all__ = [
    'TableParser',
    'TableObject',
    'TableRoleClassifier',
    'RoleClassification',
    'TableValidator',
    'ValidationResult',
    'TableEnhancer',
    'EnhancementResult',
    'TokenRole',
    'SEMANTIC_ROLES',
    'TableProcessingConfig',
    'get_table_config',
    'TableUtils',
    'CoordinateSystem',
    'get_table_processing_config',
    'update_table_processing_config',
    'initialize_table_processing'
]
