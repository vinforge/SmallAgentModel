#!/usr/bin/env python3
"""
SAM Session State Manager - Phase 1 Conversational Buffer
=========================================================

Implements the Short-Term Conversational Buffer for maintaining context
within a single chat session. This solves the "connect the dots" problem
by keeping track of recent conversation turns.

Part of Task 30: Advanced Conversational Coherence Engine
Author: SAM Development Team
Version: 1.0.0
"""

import logging
import json
import threading
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Deque
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        """Create from dictionary."""
        return cls(**data)

class SessionManager:
    """
    Manages conversational sessions with short-term memory buffer.
    
    Features:
    - Fixed-size conversation buffer (deque)
    - Automatic session persistence
    - Thread-safe operations
    - Configurable buffer depth
    - Session cleanup and management
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the session manager.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(f"{__name__}.SessionManager")
        
        # Default configuration
        self.config = {
            'conversation_history_depth': 10,
            'session_timeout_minutes': 60,
            'auto_save': True,
            'storage_directory': 'sessions',
            'max_content_length': 2000
        }
        
        if config:
            self.config.update(config)
        
        # Session storage
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        
        # Storage setup
        self.storage_dir = Path(self.config['storage_directory'])
        self.storage_dir.mkdir(exist_ok=True)
        
        # Load existing sessions
        self._load_sessions()
        
        self.logger.info(f"SessionManager initialized with buffer depth: {self.config['conversation_history_depth']}")
    
    def create_session(self, session_id: str, user_id: Optional[str] = None) -> str:
        """
        Create a new conversation session.
        
        Args:
            session_id: Unique session identifier
            user_id: Optional user identifier
            
        Returns:
            Session ID
        """
        with self._lock:
            session_data = {
                'session_id': session_id,
                'user_id': user_id,
                'created_at': datetime.now().isoformat(),
                'last_activity': datetime.now().isoformat(),
                'conversational_buffer': deque(maxlen=self.config['conversation_history_depth']),
                'metadata': {}
            }
            
            self.sessions[session_id] = session_data
            
            if self.config['auto_save']:
                self._save_session(session_id)
            
            self.logger.info(f"Created session: {session_id} for user: {user_id}")
            return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data by ID."""
        with self._lock:
            return self.sessions.get(session_id)
    
    def add_turn(self, session_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a conversation turn to the session buffer.
        
        Args:
            session_id: Session identifier
            role: 'user' or 'assistant'
            content: Message content
            metadata: Optional metadata
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if not session:
                self.logger.warning(f"Session not found: {session_id}")
                return False
            
            # Truncate content if too long
            if len(content) > self.config['max_content_length']:
                content = content[:self.config['max_content_length']] + "... [truncated]"
            
            # Create conversation turn
            turn = ConversationTurn(
                role=role,
                content=content,
                timestamp=datetime.now().isoformat(),
                metadata=metadata or {}
            )
            
            # Add to buffer (deque automatically handles max length)
            session['conversational_buffer'].append(turn.to_dict())
            session['last_activity'] = datetime.now().isoformat()
            
            if self.config['auto_save']:
                self._save_session(session_id)
            
            self.logger.debug(f"Added {role} turn to session {session_id}: {content[:50]}...")
            return True
    
    def get_conversation_history(self, session_id: str, max_turns: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            max_turns: Maximum number of turns to return
            
        Returns:
            List of conversation turns
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if not session:
                return []
            
            buffer = list(session['conversational_buffer'])
            
            if max_turns:
                buffer = buffer[-max_turns:]
            
            return buffer
    
    def format_conversation_history(self, session_id: str, max_turns: Optional[int] = None) -> str:
        """
        Format conversation history as a human-readable string.
        
        Args:
            session_id: Session identifier
            max_turns: Maximum number of turns to include
            
        Returns:
            Formatted conversation history
        """
        history = self.get_conversation_history(session_id, max_turns)
        
        if not history:
            return "No recent conversation history."
        
        formatted_lines = []
        for turn in reversed(history):  # Most recent first
            role = turn['role'].title()
            content = turn['content']
            timestamp = turn['timestamp']
            
            # Parse timestamp for relative time
            try:
                turn_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                time_ago = self._format_time_ago(turn_time)
                formatted_lines.append(f"{role} ({time_ago}): {content}")
            except:
                formatted_lines.append(f"{role}: {content}")
        
        return "\n".join(formatted_lines)
    
    def clear_session(self, session_id: str) -> bool:
        """Clear conversation history for a session."""
        with self._lock:
            session = self.sessions.get(session_id)
            if not session:
                return False
            
            session['conversational_buffer'].clear()
            session['last_activity'] = datetime.now().isoformat()
            
            if self.config['auto_save']:
                self._save_session(session_id)
            
            self.logger.info(f"Cleared session: {session_id}")
            return True
    
    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions based on timeout."""
        with self._lock:
            current_time = datetime.now()
            timeout_delta = timedelta(minutes=self.config['session_timeout_minutes'])
            expired_sessions = []
            
            for session_id, session in self.sessions.items():
                try:
                    last_activity = datetime.fromisoformat(session['last_activity'])
                    if current_time - last_activity > timeout_delta:
                        expired_sessions.append(session_id)
                except:
                    # Invalid timestamp, mark for cleanup
                    expired_sessions.append(session_id)
            
            # Remove expired sessions
            for session_id in expired_sessions:
                del self.sessions[session_id]
                # Remove session file
                session_file = self.storage_dir / f"{session_id}.json"
                if session_file.exists():
                    session_file.unlink()
            
            if expired_sessions:
                self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
            
            return len(expired_sessions)
    
    def _format_time_ago(self, timestamp: datetime) -> str:
        """Format timestamp as relative time."""
        now = datetime.now()
        if timestamp.tzinfo:
            now = now.replace(tzinfo=timestamp.tzinfo)
        
        delta = now - timestamp
        
        if delta.total_seconds() < 60:
            return "just now"
        elif delta.total_seconds() < 3600:
            minutes = int(delta.total_seconds() / 60)
            return f"{minutes}m ago"
        elif delta.total_seconds() < 86400:
            hours = int(delta.total_seconds() / 3600)
            return f"{hours}h ago"
        else:
            days = delta.days
            return f"{days}d ago"
    
    def _save_session(self, session_id: str) -> bool:
        """Save session to disk."""
        try:
            session = self.sessions.get(session_id)
            if not session:
                return False
            
            # Convert deque to list for JSON serialization
            session_copy = session.copy()
            session_copy['conversational_buffer'] = list(session['conversational_buffer'])
            
            session_file = self.storage_dir / f"{session_id}.json"
            with open(session_file, 'w') as f:
                json.dump(session_copy, f, indent=2)
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving session {session_id}: {e}")
            return False
    
    def _load_sessions(self) -> None:
        """Load sessions from disk."""
        try:
            for session_file in self.storage_dir.glob("*.json"):
                try:
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)
                    
                    # Convert list back to deque
                    buffer_data = session_data.get('conversational_buffer', [])
                    session_data['conversational_buffer'] = deque(
                        buffer_data, 
                        maxlen=self.config['conversation_history_depth']
                    )
                    
                    session_id = session_data['session_id']
                    self.sessions[session_id] = session_data
                    
                except Exception as e:
                    self.logger.warning(f"Error loading session file {session_file}: {e}")
            
            self.logger.info(f"Loaded {len(self.sessions)} sessions from disk")
            
        except Exception as e:
            self.logger.error(f"Error loading sessions: {e}")

# Global session manager instance
_session_manager: Optional[SessionManager] = None
_session_manager_lock = threading.Lock()

def get_session_manager(config: Optional[Dict[str, Any]] = None) -> SessionManager:
    """Get the global session manager instance."""
    global _session_manager
    
    with _session_manager_lock:
        if _session_manager is None:
            _session_manager = SessionManager(config)
        return _session_manager
