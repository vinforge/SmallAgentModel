"""
TPV (Thinking Process Verification) Module for SAM
Phase 1 - Active Monitoring & Passive Control Integration

This module provides thinking process verification capabilities for SAM,
enabling meta-cognitive reasoning and self-reflection.
"""

__version__ = "0.2.0"
__phase__ = "Phase 1 - Active Monitoring & Passive Control Integration"

from .tpv_core import TPVCore
from .tpv_config import TPVConfig
from .tpv_monitor import TPVMonitor, ReasoningTrace, ReasoningStep
from .tpv_controller import ReasoningController, ControlMode, ControlDecision
from .tpv_trigger import TPVTrigger, UserProfile, QueryIntent, TriggerResult
from .sam_integration import SAMTPVIntegration, TPVEnabledResponse, sam_tpv_integration

__all__ = [
    "TPVCore",
    "TPVConfig",
    "TPVMonitor",
    "ReasoningController",
    "TPVTrigger",
    "SAMTPVIntegration",
    "TPVEnabledResponse",
    "sam_tpv_integration",
    "ReasoningTrace",
    "ReasoningStep",
    "ControlMode",
    "ControlDecision",
    "UserProfile",
    "QueryIntent",
    "TriggerResult"
]
