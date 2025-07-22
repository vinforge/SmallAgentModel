"""
SAM Autonomy Safety Module
=========================

This module provides safety validation and controls for SAM's autonomous
goal generation and execution system. It implements multiple layers of
protection to prevent harmful autonomous actions.

Components:
- GoalSafetyValidator: Core safety validation for autonomous goals
- Safety policies and deny-lists
- Loop detection and prevention
- Rate limiting controls

Author: SAM Development Team
Version: 2.0.0
"""

from .goal_validator import GoalSafetyValidator

__all__ = [
    'GoalSafetyValidator'
]
