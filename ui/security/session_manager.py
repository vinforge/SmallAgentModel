#!/usr/bin/env python3
"""
Session Manager for SAM UI
==========================

Handles session management, authentication, and security features
extracted from the monolithic secure_streamlit_app.py.

This module provides:
- Session initialization and management
- Authentication and security checks
- Session timeout handling
- User state management

Author: SAM Development Team
Version: 1.0.0 - Refactored from secure_streamlit_app.py
"""

import streamlit as st
import logging
import time
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages user sessions and security for SAM UI."""
    
    def __init__(self):
        self.session_timeout_minutes = 60
        self.max_failed_attempts = 5
        self.lockout_duration_minutes = 15
    
    def initialize_session(self) -> str:
        """Initialize a new user session."""
        if 'session_id' not in st.session_state:
            # Generate unique session ID
            session_id = self._generate_session_id()
            st.session_state.session_id = session_id
            st.session_state.session_start_time = time.time()
            st.session_state.last_activity_time = time.time()
            st.session_state.is_authenticated = False
            st.session_state.failed_attempts = 0
            st.session_state.lockout_until = None
            
            logger.info(f"New session initialized: {session_id}")
        
        return st.session_state.session_id
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        timestamp = str(time.time())
        random_bytes = secrets.token_bytes(16)
        session_data = f"{timestamp}_{random_bytes.hex()}"
        return hashlib.sha256(session_data.encode()).hexdigest()[:16]
    
    def check_session_timeout(self) -> bool:
        """Check if the current session has timed out."""
        if 'last_activity_time' not in st.session_state:
            return True
        
        current_time = time.time()
        last_activity = st.session_state.last_activity_time
        timeout_seconds = self.session_timeout_minutes * 60
        
        if current_time - last_activity > timeout_seconds:
            logger.info(f"Session {st.session_state.get('session_id', 'unknown')} timed out")
            self.clear_session()
            return True
        
        # Update last activity time
        st.session_state.last_activity_time = current_time
        return False
    
    def clear_session(self):
        """Clear the current session."""
        session_id = st.session_state.get('session_id', 'unknown')
        
        # Clear all session state except for essential UI state
        keys_to_preserve = ['theme', 'sidebar_state']
        preserved_state = {k: st.session_state.get(k) for k in keys_to_preserve if k in st.session_state}
        
        st.session_state.clear()
        
        # Restore preserved state
        for k, v in preserved_state.items():
            st.session_state[k] = v
        
        logger.info(f"Session cleared: {session_id}")
    
    def is_session_valid(self) -> bool:
        """Check if the current session is valid."""
        return (
            'session_id' in st.session_state and
            'session_start_time' in st.session_state and
            not self.check_session_timeout()
        )
    
    def authenticate_user(self, username: str, password: str) -> bool:
        """Authenticate user credentials."""
        # Check if user is locked out
        if self.is_user_locked_out():
            return False
        
        # Simple authentication (in production, use proper auth system)
        if self._verify_credentials(username, password):
            st.session_state.is_authenticated = True
            st.session_state.username = username
            st.session_state.failed_attempts = 0
            st.session_state.lockout_until = None
            
            logger.info(f"User authenticated: {username}")
            return True
        else:
            st.session_state.failed_attempts = st.session_state.get('failed_attempts', 0) + 1
            
            if st.session_state.failed_attempts >= self.max_failed_attempts:
                lockout_until = datetime.now() + timedelta(minutes=self.lockout_duration_minutes)
                st.session_state.lockout_until = lockout_until
                logger.warning(f"User locked out due to failed attempts: {username}")
            
            logger.warning(f"Authentication failed for user: {username}")
            return False
    
    def _verify_credentials(self, username: str, password: str) -> bool:
        """Verify user credentials (placeholder implementation)."""
        # In production, this would check against a proper user database
        # For now, use simple demo credentials
        demo_users = {
            'admin': 'admin123',
            'user': 'user123',
            'demo': 'demo123'
        }
        
        return demo_users.get(username) == password
    
    def is_user_locked_out(self) -> bool:
        """Check if user is currently locked out."""
        lockout_until = st.session_state.get('lockout_until')
        
        if lockout_until is None:
            return False
        
        if datetime.now() < lockout_until:
            return True
        else:
            # Lockout period has expired
            st.session_state.lockout_until = None
            st.session_state.failed_attempts = 0
            return False
    
    def get_lockout_time_remaining(self) -> Optional[int]:
        """Get remaining lockout time in seconds."""
        lockout_until = st.session_state.get('lockout_until')
        
        if lockout_until is None:
            return None
        
        remaining = (lockout_until - datetime.now()).total_seconds()
        return max(0, int(remaining))
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated."""
        return st.session_state.get('is_authenticated', False)
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information."""
        if not self.is_session_valid():
            return {}
        
        session_start = st.session_state.get('session_start_time', 0)
        last_activity = st.session_state.get('last_activity_time', 0)
        
        return {
            'session_id': st.session_state.get('session_id'),
            'username': st.session_state.get('username'),
            'session_duration': time.time() - session_start,
            'time_since_last_activity': time.time() - last_activity,
            'is_authenticated': self.is_authenticated(),
            'failed_attempts': st.session_state.get('failed_attempts', 0),
            'is_locked_out': self.is_user_locked_out()
        }
    
    def render_session_status(self):
        """Render session status in the UI."""
        if not self.is_session_valid():
            st.error("❌ Invalid session. Please refresh the page.")
            return
        
        session_info = self.get_session_info()
        
        with st.expander("🔒 Session Information", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Session ID:** {session_info['session_id']}")
                st.write(f"**Username:** {session_info.get('username', 'Not authenticated')}")
                st.write(f"**Authenticated:** {'✅' if session_info['is_authenticated'] else '❌'}")
            
            with col2:
                duration_minutes = session_info['session_duration'] / 60
                st.write(f"**Session Duration:** {duration_minutes:.1f} minutes")
                
                activity_seconds = session_info['time_since_last_activity']
                st.write(f"**Last Activity:** {activity_seconds:.0f} seconds ago")
                
                if session_info['is_locked_out']:
                    remaining = self.get_lockout_time_remaining()
                    st.write(f"**🔒 Locked Out:** {remaining} seconds remaining")
    
    def render_authentication_form(self) -> bool:
        """Render authentication form and handle login."""
        st.markdown("### 🔐 Authentication Required")
        
        if self.is_user_locked_out():
            remaining = self.get_lockout_time_remaining()
            st.error(f"🔒 Account locked due to failed attempts. Try again in {remaining} seconds.")
            return False
        
        with st.form("auth_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit_button = st.form_submit_button("🔑 Login")
            
            if submit_button:
                if not username or not password:
                    st.error("❌ Please enter both username and password")
                    return False
                
                if self.authenticate_user(username, password):
                    st.success("✅ Authentication successful!")
                    st.rerun()
                    return True
                else:
                    failed_attempts = st.session_state.get('failed_attempts', 0)
                    remaining_attempts = self.max_failed_attempts - failed_attempts
                    
                    if remaining_attempts > 0:
                        st.error(f"❌ Invalid credentials. {remaining_attempts} attempts remaining.")
                    else:
                        st.error("🔒 Account locked due to too many failed attempts.")
                    
                    return False
        
        # Show demo credentials hint
        with st.expander("💡 Demo Credentials", expanded=False):
            st.info("""
            **Demo Users:**
            - Username: `demo`, Password: `demo123`
            - Username: `user`, Password: `user123`
            - Username: `admin`, Password: `admin123`
            """)
        
        return False


# Global session manager instance
_session_manager = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def initialize_secure_session() -> str:
    """Initialize a secure session."""
    return get_session_manager().initialize_session()


def check_authentication() -> bool:
    """Check if user is authenticated, show login form if not."""
    session_manager = get_session_manager()
    
    # Initialize session if needed
    session_manager.initialize_session()
    
    # Check session timeout
    if session_manager.check_session_timeout():
        st.warning("⏰ Session timed out. Please log in again.")
        return False
    
    # Check authentication
    if not session_manager.is_authenticated():
        return session_manager.render_authentication_form()
    
    return True


def render_session_sidebar():
    """Render session information in sidebar."""
    session_manager = get_session_manager()
    
    if session_manager.is_session_valid():
        with st.sidebar:
            session_manager.render_session_status()
            
            if session_manager.is_authenticated():
                if st.button("🚪 Logout"):
                    session_manager.clear_session()
                    st.rerun()


def get_current_session_id() -> Optional[str]:
    """Get the current session ID."""
    return st.session_state.get('session_id')


def get_current_username() -> Optional[str]:
    """Get the current authenticated username."""
    return st.session_state.get('username')
