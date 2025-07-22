"""
SAM Secure State Manager

Manages security state and session management for SAM.
Provides centralized security state tracking and validation.

Author: SAM Development Team
Version: 2.0.0
"""

import os
import json
import time
from enum import Enum
from typing import Optional, Dict, Any
from .keystore_manager import KeystoreManager
from .crypto_utils import CryptoManager


class SecurityState(Enum):
    """Security state enumeration."""
    UNINITIALIZED = "uninitialized"
    SETUP_REQUIRED = "setup_required"
    AUTHENTICATED = "authenticated"
    LOCKED = "locked"
    ERROR = "error"


class SecureStateManager:
    """
    Manages security state and session management for SAM.
    
    Provides centralized security state tracking, session management,
    and security validation for the SAM application.
    """
    
    def __init__(self):
        """Initialize the secure state manager."""
        self.keystore_manager = KeystoreManager()
        self.crypto_manager = None
        self.current_state = SecurityState.UNINITIALIZED
        self.session_start_time = None
        self.session_timeout = 3600  # 1 hour default
        self.last_activity = None

        # Failed attempt tracking
        self.failed_attempts = 0
        self.max_attempts = 5
        self.lockout_start_time = None
        self.lockout_duration = 1800  # 30 minutes in seconds

        # Initialize state
        self._initialize_state()
    
    def _initialize_state(self):
        """Initialize the security state based on current setup."""
        try:
            if not self.keystore_manager.keystore_exists():
                self.current_state = SecurityState.SETUP_REQUIRED
            elif self.keystore_manager.is_locked():
                self.current_state = SecurityState.LOCKED
            else:
                # Check if we have a valid session
                if self._has_valid_session():
                    self.current_state = SecurityState.AUTHENTICATED
                else:
                    self.current_state = SecurityState.LOCKED
                    
        except Exception as e:
            print(f"Error initializing security state: {e}")
            self.current_state = SecurityState.ERROR
    
    def is_setup_required(self) -> bool:
        """Check if initial security setup is required."""
        return self.current_state == SecurityState.SETUP_REQUIRED
    
    def is_authenticated(self) -> bool:
        """Check if user is currently authenticated."""
        if self.current_state != SecurityState.AUTHENTICATED:
            return False

        # Check session timeout
        if not self._has_valid_session():
            self.current_state = SecurityState.LOCKED
            return False

        return True

    def is_unlocked(self) -> bool:
        """Check if user is currently unlocked (alias for is_authenticated)."""
        return self.is_authenticated()
    
    def is_locked(self) -> bool:
        """Check if the system is locked."""
        return self.current_state == SecurityState.LOCKED
    
    def authenticate(self, password: str) -> bool:
        """
        Authenticate user with password.

        Args:
            password: User's master password

        Returns:
            bool: True if authentication successful
        """
        try:
            # Check if account is locked out
            if self.is_locked_out():
                return False

            # Verify password with keystore
            is_valid, session_key = self.keystore_manager.verify_password(password)
            if not is_valid:
                # Increment failed attempts
                self.failed_attempts += 1

                # Lock account if max attempts reached
                if self.failed_attempts >= self.max_attempts:
                    self.lockout_start_time = time.time()
                    self.current_state = SecurityState.LOCKED

                return False

            # Reset failed attempts on successful authentication
            self.failed_attempts = 0
            self.lockout_start_time = None

            # Initialize crypto manager
            self.crypto_manager = CryptoManager()
            # Set session key from keystore verification
            self.crypto_manager.session_key = session_key

            # Start new session
            self.session_start_time = time.time()
            self.last_activity = time.time()
            self.current_state = SecurityState.AUTHENTICATED

            return True

        except Exception as e:
            print(f"Authentication error: {e}")
            self.current_state = SecurityState.ERROR
            return False
    
    def lock_session(self):
        """Lock the current session."""
        self.crypto_manager = None
        self.session_start_time = None
        self.last_activity = None
        self.current_state = SecurityState.LOCKED

    def unlock_application(self, password: str) -> bool:
        """
        Unlock the application with password (alias for authenticate).

        Args:
            password: User's master password

        Returns:
            bool: True if unlock successful
        """
        return self.authenticate(password)

    def lock_application(self):
        """Lock the application (alias for lock_session)."""
        self.lock_session()

    def initialize_security(self, password: str) -> bool:
        """
        Initialize security system (alias for setup_security).

        Args:
            password: Master password to set

        Returns:
            bool: True if initialization successful
        """
        return self.setup_security(password)

    def get_failed_attempts(self) -> int:
        """
        Get number of failed authentication attempts.

        Returns:
            int: Number of failed attempts
        """
        return self.failed_attempts

    def get_lockout_remaining(self) -> int:
        """
        Get remaining lockout time in seconds.

        Returns:
            int: Remaining lockout time in seconds, 0 if not locked
        """
        if not self.lockout_start_time:
            return 0

        elapsed = time.time() - self.lockout_start_time
        remaining = max(0, self.lockout_duration - elapsed)

        # Clear lockout if time has expired
        if remaining == 0:
            self.lockout_start_time = None
            self.failed_attempts = 0

        return int(remaining)

    def is_locked_out(self) -> bool:
        """
        Check if account is currently locked out.

        Returns:
            bool: True if account is locked out
        """
        return self.get_lockout_remaining() > 0

    def reset_failed_attempts(self):
        """Reset failed attempt counter and clear lockout."""
        self.failed_attempts = 0
        self.lockout_start_time = None
        if self.current_state == SecurityState.LOCKED and not self.is_locked_out():
            # If we were locked but lockout expired, return to normal locked state
            pass

    def get_max_attempts(self) -> int:
        """
        Get maximum allowed failed attempts.

        Returns:
            int: Maximum attempts before lockout
        """
        return self.max_attempts
    
    def update_activity(self):
        """Update last activity timestamp."""
        if self.current_state == SecurityState.AUTHENTICATED:
            self.last_activity = time.time()
    
    def _has_valid_session(self) -> bool:
        """Check if current session is valid (not timed out)."""
        if not self.session_start_time or not self.last_activity:
            return False
        
        current_time = time.time()
        
        # Check session timeout
        if current_time - self.last_activity > self.session_timeout:
            return False
        
        return True
    
    def get_crypto_manager(self) -> Optional[CryptoManager]:
        """Get the current crypto manager if authenticated."""
        if self.is_authenticated():
            return self.crypto_manager
        return None
    
    def get_state(self) -> SecurityState:
        """Get current security state."""
        return self.current_state
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information."""
        if not self.is_authenticated():
            return {
                "authenticated": False,
                "state": self.current_state.value
            }

        current_time = time.time()
        session_duration = current_time - self.session_start_time if self.session_start_time else 0
        time_since_activity = current_time - self.last_activity if self.last_activity else 0
        time_remaining = max(0, self.session_timeout - time_since_activity)

        # Generate a session ID based on start time
        session_id = f"sam_session_{int(self.session_start_time)}" if self.session_start_time else "unknown"

        return {
            "authenticated": True,
            "state": self.current_state.value,
            "session_id": session_id,
            "started_at": self.session_start_time,
            "session_duration": session_duration,
            "time_since_activity": time_since_activity,
            "time_remaining": time_remaining,
            "session_timeout": self.session_timeout
        }
    
    def setup_security(self, password: str) -> bool:
        """
        Setup initial security configuration.

        Args:
            password: Master password to set

        Returns:
            bool: True if setup successful
        """
        try:
            # Create keystore
            if not self.keystore_manager.create_keystore(password):
                return False

            # Initialize crypto manager with session key
            self.crypto_manager = CryptoManager()

            # Derive key from password and set session key
            derived_key, salt = self.crypto_manager.derive_key_from_password(password)
            self.crypto_manager.set_session_key(derived_key)

            # Start session
            self.session_start_time = time.time()
            self.last_activity = time.time()
            self.current_state = SecurityState.AUTHENTICATED

            return True

        except Exception as e:
            print(f"Security setup error: {e}")
            self.current_state = SecurityState.ERROR
            return False
    
    def change_password(self, old_password: str, new_password: str) -> bool:
        """
        Change the master password.

        Args:
            old_password: Current password
            new_password: New password to set

        Returns:
            bool: True if password changed successfully
        """
        try:
            # Verify old password
            is_valid, _ = self.keystore_manager.verify_password(old_password)
            if not is_valid:
                return False

            # Update keystore with new password
            if not self.keystore_manager.change_password(old_password, new_password):
                return False

            # Update crypto manager
            self.crypto_manager = CryptoManager()

            return True

        except Exception as e:
            print(f"Password change error: {e}")
            return False
    
    def export_security_status(self) -> Dict[str, Any]:
        """Export current security status for monitoring."""
        return {
            "security_state": self.current_state.value,
            "keystore_exists": self.keystore_manager.keystore_exists(),
            "session_info": self.get_session_info(),
            "setup_required": self.is_setup_required(),
            "authenticated": self.is_authenticated(),
            "locked": self.is_locked()
        }

    def is_initialized(self) -> bool:
        """
        Check if the security system is fully initialized.

        Returns:
            bool: True if keystore exists and crypto manager is ready
        """
        try:
            return (self.keystore_manager.is_initialized() and
                    self.crypto_manager is not None and
                    self.crypto_manager.is_initialized())
        except Exception:
            return False
