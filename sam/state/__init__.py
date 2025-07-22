"""
SAM State Management System
==========================

Centralized state management for SAM's persistent data including:
- Discovery cycle state and notifications
- Vetting queue management
- Application configuration state
- User preferences and settings

This module provides thread-safe, atomic operations for state management
with proper error handling and recovery mechanisms.

Part of SAM's Task 27: Automated "Dream & Discover" Engine
Author: SAM Development Team
Version: 1.0.0
"""

from .state_manager import (
    StateManager, StateKey, StateValue, get_state_manager
)
from .vetting_queue import (
    VettingQueueManager, VettingEntry, VettingStatus, get_vetting_queue_manager
)

__all__ = [
    'StateManager',
    'StateKey', 
    'StateValue',
    'get_state_manager',
    'VettingQueueManager',
    'VettingEntry',
    'VettingStatus',
    'get_vetting_queue_manager',
]
