#!/usr/bin/env python3
"""
Memory Authentication Module
============================

Handles authentication and security for the SAM Memory Control Center.
Extracted from the monolithic memory_app.py.

This module provides:
- Memory-specific authentication
- Security integration with main SAM interface
- Session validation for memory access
- Security status indicators

Author: SAM Development Team
Version: 1.0.0 - Refactored from memory_app.py
"""

import streamlit as st
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class MemoryAuthManager:
    """Manages authentication for the Memory Control Center."""
    
    def __init__(self):
        self.security_manager = None
        self._initialize_security()
    
    def _initialize_security(self):
        """Initialize the security manager."""
        try:
            from security import SecureStateManager
            
            # Try to get shared security manager from session state
            if 'security_manager' in st.session_state:
                self.security_manager = st.session_state.security_manager
                logger.info("✅ Using shared security manager from session state")
            else:
                # Create new instance but check for existing authentication
                self.security_manager = SecureStateManager()
                st.session_state.security_manager = self.security_manager
                logger.info("🆕 Created new security manager instance")
                
        except ImportError as e:
            logger.error(f"Security module not available: {e}")
            self.security_manager = None
    
    def check_authentication(self) -> bool:
        """
        Check if user is authenticated via the secure SAM interface.
        
        Returns:
            bool: True if authenticated, False otherwise
        """
        try:
            if not self.security_manager:
                logger.error("Security manager not available")
                return False
            
            # Check if the security system is unlocked
            is_authenticated = self.security_manager.is_unlocked()
            
            if is_authenticated:
                # Verify session is still valid
                session_info = self.security_manager.get_session_info()
                if session_info['is_unlocked'] and session_info['time_remaining'] > 0:
                    logger.info("✅ Authentication verified - Memory Control Center access granted")
                    return True
                else:
                    # Session expired, lock the application
                    self.security_manager.lock_application()
                    logger.warning("⚠️ Session expired - locking application")
                    return False
            
            logger.info("🔒 Authentication required - user not authenticated")
            return False
            
        except Exception as e:
            logger.error(f"Authentication check failed: {e}")
            return False
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information."""
        if not self.security_manager:
            return {'is_unlocked': False, 'time_remaining': 0}
        
        try:
            return self.security_manager.get_session_info()
        except Exception as e:
            logger.error(f"Failed to get session info: {e}")
            return {'is_unlocked': False, 'time_remaining': 0}
    
    def render_authentication_required(self):
        """Render the authentication required page."""
        st.title("🔒 SAM Memory Control Center")
        st.markdown("*Authentication Required*")
        st.markdown("---")
        
        st.error("🚫 **Access Denied - Authentication Required**")
        st.markdown("""
        The SAM Memory Control Center requires authentication through the secure SAM interface.
        
        **🔐 Security Integration:**
        - Memory Control Center shares the same security system as the main SAM interface
        - Your memory data is encrypted with AES-256-GCM and requires authentication
        - Session management ensures secure access across all SAM components
        
        **📋 To Access the Memory Control Center:**
        1. **Open the Secure SAM Interface**: [http://localhost:8502](http://localhost:8502)
        2. **Enter your master password** to unlock SAM
        3. **Return to this page** or click the Memory Control Center link from the main interface
        
        **🛡️ Why Authentication is Required:**
        - Your personal memory data is encrypted and protected
        - Unauthorized access could compromise your private conversations
        - Session-based security ensures only you can access your memories
        
        **🔧 Troubleshooting:**
        - Ensure the main SAM interface is running on port 8502
        - Check that you've successfully authenticated in the main interface
        - Try refreshing this page after authenticating
        """)
        
        # Show security status
        self.render_security_status()
        
        # Provide direct link to main interface
        st.markdown("### 🚀 Quick Access")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔐 Open SAM Interface", type="primary"):
                st.markdown("[Click here to open SAM](http://localhost:8502)")
        
        with col2:
            if st.button("🔄 Refresh Authentication"):
                st.rerun()
    
    def render_security_status(self):
        """Render security status indicator."""
        st.markdown("### 🛡️ Security Status")
        
        if not self.security_manager:
            st.error("❌ **Security Module**: Not Available")
            st.error("🔒 **Access**: Denied - Security module required")
            return
        
        session_info = self.get_session_info()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if session_info['is_unlocked']:
                st.success("✅ **Authentication**: Verified")
            else:
                st.error("❌ **Authentication**: Required")
        
        with col2:
            time_remaining = session_info.get('time_remaining', 0)
            if time_remaining > 0:
                minutes = int(time_remaining / 60)
                st.info(f"⏰ **Session**: {minutes}m remaining")
            else:
                st.warning("⏰ **Session**: Expired")
        
        with col3:
            if session_info['is_unlocked']:
                st.success("🔓 **Access**: Granted")
            else:
                st.error("🔒 **Access**: Denied")
        
        # Show detailed session information
        with st.expander("🔍 Detailed Security Information", expanded=False):
            st.json(session_info)
    
    def render_security_status_indicator(self) -> str:
        """
        Render a compact security status indicator for the sidebar.
        
        Returns:
            str: Status indicator text
        """
        if not self.security_manager:
            return "🔒 Security: Unavailable"
        
        session_info = self.get_session_info()
        
        if session_info['is_unlocked']:
            time_remaining = session_info.get('time_remaining', 0)
            if time_remaining > 0:
                minutes = int(time_remaining / 60)
                return f"🔓 Authenticated ({minutes}m)"
            else:
                return "⏰ Session Expired"
        else:
            return "🔒 Authentication Required"
    
    def is_dream_canvas_available(self) -> bool:
        """Check if Dream Canvas is available and accessible."""
        try:
            # Check authentication first
            if not self.check_authentication():
                return False
            
            # Check if Dream Canvas module is available
            try:
                from ui.dream_canvas import DreamCanvasApp
                return True
            except ImportError:
                return False
                
        except Exception as e:
            logger.error(f"Error checking Dream Canvas availability: {e}")
            return False
    
    def render_dream_canvas_locked(self):
        """Render Dream Canvas locked message."""
        st.markdown("### 🎨 Dream Canvas")
        st.warning("🔒 **Dream Canvas is locked**")
        st.markdown("""
        **Dream Canvas Access Requirements:**
        - ✅ Authentication through secure SAM interface
        - ✅ Valid session with sufficient time remaining
        - ✅ Dream Canvas module properly installed
        
        **Current Status:**
        """)
        
        # Show current status
        auth_status = "✅ Authenticated" if self.check_authentication() else "❌ Not Authenticated"
        canvas_status = "✅ Available" if self.is_dream_canvas_available() else "❌ Unavailable"
        
        st.write(f"- **Authentication**: {auth_status}")
        st.write(f"- **Dream Canvas Module**: {canvas_status}")
        
        if not self.check_authentication():
            st.info("💡 Please authenticate through the main SAM interface to access Dream Canvas.")
        elif not self.is_dream_canvas_available():
            st.info("💡 Dream Canvas module is not available. Please check your installation.")


# Global memory auth manager instance
_memory_auth_manager = None


def get_memory_auth_manager() -> MemoryAuthManager:
    """Get the global memory authentication manager instance."""
    global _memory_auth_manager
    if _memory_auth_manager is None:
        _memory_auth_manager = MemoryAuthManager()
    return _memory_auth_manager


def check_memory_authentication() -> bool:
    """Check if user is authenticated for memory access."""
    return get_memory_auth_manager().check_authentication()


def render_memory_authentication_required():
    """Render authentication required page for memory access."""
    get_memory_auth_manager().render_authentication_required()


def render_memory_security_status():
    """Render memory security status."""
    get_memory_auth_manager().render_security_status()


def get_memory_security_status_indicator() -> str:
    """Get compact security status indicator text."""
    return get_memory_auth_manager().render_security_status_indicator()
