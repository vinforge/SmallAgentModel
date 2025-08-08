"""
SAM Session Management Module
============================

Provides session state management and conversational buffer functionality
for maintaining context within chat sessions.

Part of Task 30: Advanced Conversational Coherence Engine
"""

from .state_manager import (
    SessionManager,
    ConversationTurn,
    get_session_manager
)

__all__ = [
    'SessionManager',
    'ConversationTurn', 
    'get_session_manager'
]
