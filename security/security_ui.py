"""
Security UI Components for SAM

Provides Streamlit-based user interface components for enterprise security features
including password setup, authentication, and session management.

Author: SAM Development Team
Version: 2.0.0
"""

import logging
import time
import streamlit as st
from typing import Optional, Tuple
from datetime import datetime

from .secure_state_manager import SecureStateManager, SecurityState

logger = logging.getLogger(__name__)

class SecurityUI:
    """
    Security user interface manager for SAM.

    Provides Streamlit-based UI components for:
    - Initial password setup
    - Authentication interface
    - Session management
    - Security status display
    """

    def __init__(self, security_manager: SecureStateManager):
        self.security_manager = security_manager
        self.logger = logging.getLogger(f"{__name__}.SecurityUI")
        self.logger.info("SecurityUI initialized with real authentication")

    def render_security_interface(self) -> bool:
        """
        Render the main security interface based on current state.

        Returns:
            True if user is authenticated and can proceed
        """
        try:
            current_state = self.security_manager.get_state()

            if current_state == SecurityState.SETUP_REQUIRED:
                return self._render_setup_interface()
            elif current_state == SecurityState.LOCKED:
                # Check if we should show lockout interface or login with error
                if self.security_manager.is_locked_out():
                    return self._render_lockout_interface()
                else:
                    # Show login interface with error if there were failed attempts
                    show_error = self.security_manager.get_failed_attempts() > 0
                    return self._render_login_interface(show_error=show_error)
            elif current_state == SecurityState.AUTHENTICATED:
                return self._render_authenticated_interface()
            elif current_state == SecurityState.ERROR:
                st.error("Security system error - please restart SAM")
                return False
            else:
                st.error(f"Unknown security state: {current_state}")
                return False

        except Exception as e:
            self.logger.error(f"Security interface error: {e}")
            st.error(f"Security system error: {e}")
            return False

    def _render_setup_interface(self) -> bool:
        """Render the initial password setup interface."""
        st.markdown("### 🔐 SAM Security Setup")
        st.markdown("---")

        st.info("🚀 **Welcome to SAM!** Set up your master password to secure your AI assistant.")

        with st.container():
            st.markdown("#### Master Password Requirements")
            st.markdown("""
            - **Minimum 8 characters** (12+ recommended)
            - **Mix of uppercase, lowercase, numbers, symbols**
            - **Unique password** (don't reuse from other accounts)
            - ⚠️ **Cannot be recovered if lost!**
            """)

            with st.form("password_setup_form"):
                password = st.text_input(
                    "Master Password",
                    type="password",
                    help="This password will encrypt all your SAM data"
                )

                confirm_password = st.text_input(
                    "Confirm Master Password",
                    type="password"
                )

                submitted = st.form_submit_button("🔒 Setup Security", type="primary")

                if submitted:
                    if not password:
                        st.error("❌ Password cannot be empty")
                        return False

                    if len(password) < 8:
                        st.error("❌ Password must be at least 8 characters long")
                        return False

                    if password != confirm_password:
                        st.error("❌ Passwords do not match")
                        return False

                    # Validate password strength
                    is_valid, message = self._validate_password_strength(password)
                    if not is_valid:
                        st.error(f"❌ {message}")
                        return False

                    # Initialize security
                    with st.spinner("🔧 Setting up encryption..."):
                        success = self.security_manager.initialize_security(password)

                    if success:
                        # Mark master password as created in setup status
                        try:
                            from utils.first_time_setup import get_first_time_setup_manager
                            setup_manager = get_first_time_setup_manager()
                            setup_manager.update_setup_status('master_password_created', True)
                        except:
                            pass  # Continue even if setup status update fails

                        st.success("✅ Security setup completed successfully!")
                        st.info("🔄 Please refresh the page to continue")
                        return False  # Require refresh
                    else:
                        st.error("❌ Security setup failed. Please try again.")
                        return False

        return False

    def _render_login_interface(self, show_error: bool = False) -> bool:
        """Render the login interface."""
        st.markdown("### 🔐 SAM Secure Access")
        st.markdown("---")

        if show_error:
            failed_attempts = self.security_manager.get_failed_attempts()
            max_attempts = self.security_manager.max_attempts
            remaining = max_attempts - failed_attempts

            if remaining > 0:
                st.error(f"❌ **Authentication failed.** Attempt {failed_attempts} of {max_attempts}. {remaining} attempts remaining.")
            else:
                st.error("❌ **Too many failed attempts.** Account locked.")

        # Show security status
        metadata = self.security_manager.keystore_manager.get_metadata()
        if metadata:
            st.info(f"🏠 **Installation:** {metadata.installation_id}")

        with st.form("login_form"):
            password = st.text_input(
                "Master Password",
                type="password",
                help="Enter your master password to unlock SAM"
            )

            submitted = st.form_submit_button("🔓 Unlock SAM", type="primary")

            if submitted:
                if not password:
                    st.error("❌ Please enter your password")
                    return False

                with st.spinner("🔍 Verifying password..."):
                    success = self.security_manager.unlock_application(password)

                if success:
                    st.success("✅ Welcome back! SAM is now unlocked.")
                    st.rerun()  # Refresh to show authenticated interface
                else:
                    # Error will be shown on next render
                    st.rerun()

        return False

    def _render_lockout_interface(self) -> bool:
        """Render the lockout interface."""
        remaining_seconds = self.security_manager.get_lockout_remaining()

        st.markdown("### 🚫 Account Locked")
        st.markdown("---")

        failed_attempts = self.security_manager.get_failed_attempts()
        max_attempts = self.security_manager.max_attempts

        st.error(f"🔒 **Account temporarily locked due to {failed_attempts} failed authentication attempts.**")

        if remaining_seconds > 0:
            minutes = remaining_seconds // 60
            seconds = remaining_seconds % 60
            st.warning(f"⏰ **Time remaining:** {minutes}m {seconds}s")

            # Auto-refresh every 10 seconds for better user experience
            time.sleep(1)  # Small delay to prevent excessive refreshing
            st.rerun()
        else:
            # Lockout has expired, redirect to login
            st.success("🔓 **Lockout period has expired.** You may now try logging in again.")
            st.info("🔄 **Refreshing to login screen...**")
            time.sleep(2)
            st.rerun()

        st.info("💡 **Tip:** Use a password manager to securely store your master password.")
        st.info(f"🔢 **Security:** Account locks after {max_attempts} failed attempts for security.")

        return False

    def _render_authenticated_interface(self) -> bool:
        """Render interface for authenticated users."""
        # Update activity
        self.security_manager.update_activity()

        # Show security status in sidebar
        with st.sidebar:
            st.markdown("### 🔐 Security Status")
            st.success("✅ **Authenticated**")

            session_info = self.security_manager.get_session_info()
            if session_info and session_info.get('session_id'):
                st.info(f"🆔 Session: {session_info['session_id'][:8]}...")

                # Show session time
                if session_info.get('started_at'):
                    from datetime import datetime
                    started_at_timestamp = session_info['started_at']

                    # Handle both timestamp (float) and ISO string formats
                    if isinstance(started_at_timestamp, (int, float)):
                        started_at = datetime.fromtimestamp(started_at_timestamp)
                    elif isinstance(started_at_timestamp, str):
                        started_at = datetime.fromisoformat(started_at_timestamp)
                    else:
                        started_at = datetime.now()  # Fallback

                    session_duration = datetime.now() - started_at
                    hours = int(session_duration.total_seconds() // 3600)
                    minutes = int((session_duration.total_seconds() % 3600) // 60)
                    st.info(f"⏰ Active: {hours}h {minutes}m")

                # Show timeout info
                if session_info.get('auto_lock_enabled'):
                    # Calculate timeout from time_remaining
                    time_remaining = session_info.get('time_remaining', 0)
                    timeout_minutes = time_remaining // 60
                    st.info(f"🔒 Auto-lock: {timeout_minutes}m remaining")

            # Lock button
            if st.button("🔒 Lock SAM", type="secondary"):
                self.security_manager.lock_application()
                st.rerun()

        return True  # User is authenticated

    def _validate_password_strength(self, password: str) -> Tuple[bool, str]:
        """Validate password strength."""
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"

        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)

        strength_score = sum([has_upper, has_lower, has_digit, has_special])

        if strength_score < 2:
            return False, "Password should contain uppercase, lowercase, numbers, and symbols"

        return True, "Password strength acceptable"

def create_security_ui(security_manager: SecureStateManager) -> SecurityUI:
    """Create a security UI instance."""
    return SecurityUI(security_manager)
