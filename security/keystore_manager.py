"""
Keystore Manager for SAM Security Module

Manages secure storage of cryptographic keys and security metadata.
Provides enterprise-grade keystore functionality with audit trails.

Author: SAM Development Team
Version: 2.0.0
"""

import json
import secrets
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from .crypto_utils import CryptoManager

logger = logging.getLogger(__name__)

@dataclass
class KeystoreMetadata:
    """Metadata for the keystore."""
    version: str
    created_at: str
    last_accessed: str
    installation_id: str
    first_setup_completed: bool
    access_count: int
    last_backup: Optional[str] = None

@dataclass
class KDFConfig:
    """Key derivation function configuration."""
    algorithm: str
    iterations: Optional[int] = None
    memory_cost: Optional[int] = None
    parallelism: Optional[int] = None
    time_cost: Optional[int] = None
    salt_length: int = 16
    hash_length: int = 32

class KeystoreManager:
    """
    Manages the SAM security keystore.
    
    Features:
    - Secure key storage with verification
    - Audit trail and access logging
    - Backup and recovery support
    - Installation ID tracking
    - Automatic keystore validation
    """
    
    def __init__(self, keystore_path: Optional[Path] = None):
        self.keystore_path = keystore_path or Path("security/keystore.json")
        self.crypto = CryptoManager()
        self.logger = logging.getLogger(f"{__name__}.KeystoreManager")
        
        # Ensure security directory exists
        self.keystore_path.parent.mkdir(exist_ok=True)
        
        self.logger.info(f"KeystoreManager initialized with path: {self.keystore_path}")
    
    def create_keystore(self, password: str) -> bool:
        """
        Create a new keystore with master password.
        
        Args:
            password: Master password for the keystore
            
        Returns:
            True if keystore created successfully
        """
        try:
            # Generate installation ID
            installation_id = f"sam_{secrets.token_hex(8)}"
            
            # Derive key and get salt
            derived_key, salt = self.crypto.derive_key_from_password(password)
            
            # Create verifier hash for password verification
            verifier_hash = self.crypto.generate_secure_token(32)
            self.crypto.set_session_key(derived_key)
            
            # Encrypt the verifier with the derived key
            verifier_result = self.crypto.encrypt(verifier_hash)
            
            # Determine KDF algorithm used
            try:
                import argon2
                kdf_config = KDFConfig(
                    algorithm="argon2id",
                    time_cost=3,
                    memory_cost=65536,
                    parallelism=4,
                    salt_length=16,
                    hash_length=32
                )
            except ImportError:
                kdf_config = KDFConfig(
                    algorithm="pbkdf2_sha256",
                    iterations=100000,
                    salt_length=16,
                    hash_length=32
                )
            
            # Create metadata
            now = datetime.now().isoformat()
            metadata = KeystoreMetadata(
                version="2.0.0",
                created_at=now,
                last_accessed=now,
                installation_id=installation_id,
                first_setup_completed=True,
                access_count=1
            )
            
            # Create keystore data
            keystore_data = {
                "metadata": asdict(metadata),
                "kdf_config": asdict(kdf_config),
                "salt": salt.hex(),
                "verifier": {
                    "ciphertext": verifier_result.ciphertext.hex(),
                    "nonce": verifier_result.nonce.hex(),
                    "tag": verifier_result.tag.hex(),
                    "metadata": verifier_result.metadata
                },
                "security_settings": {
                    "session_timeout_minutes": 60,
                    "max_failed_attempts": 5,
                    "lockout_duration_minutes": 30,
                    "require_password_on_startup": True,
                    "auto_lock_on_idle": True,
                    "audit_logging_enabled": True
                },
                "audit_log": [
                    {
                        "timestamp": now,
                        "event": "keystore_created",
                        "details": {
                            "installation_id": installation_id,
                            "kdf_algorithm": kdf_config.algorithm
                        }
                    }
                ]
            }
            
            # Save keystore
            with open(self.keystore_path, 'w') as f:
                json.dump(keystore_data, f, indent=2)
            
            # Set secure file permissions (Unix-like systems)
            try:
                import stat
                self.keystore_path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 600 permissions
            except (ImportError, OSError):
                pass  # Windows or permission error
            
            self.logger.info(f"Keystore created successfully: {installation_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create keystore: {e}")
            return False
        finally:
            # Clear session key
            self.crypto.clear_session_key()
    
    def verify_password(self, password: str) -> Tuple[bool, Optional[bytes]]:
        """
        Verify master password and return session key if valid.
        
        Args:
            password: Password to verify
            
        Returns:
            Tuple of (is_valid, session_key)
        """
        try:
            if not self.keystore_path.exists():
                self.logger.error("Keystore file not found")
                return False, None
            
            # Load keystore
            with open(self.keystore_path, 'r') as f:
                keystore_data = json.load(f)
            
            # Extract salt and verifier
            salt = bytes.fromhex(keystore_data['salt'])
            verifier_data = keystore_data['verifier']
            
            # Derive key from password
            derived_key, _ = self.crypto.derive_key_from_password(password, salt)
            
            # Set session key and try to decrypt verifier
            self.crypto.set_session_key(derived_key)
            
            try:
                ciphertext = bytes.fromhex(verifier_data['ciphertext'])
                nonce = bytes.fromhex(verifier_data['nonce'])
                tag = bytes.fromhex(verifier_data['tag'])
                
                # Attempt decryption
                result = self.crypto.decrypt(ciphertext, nonce, tag)
                
                # If we get here, password is correct
                self._update_access_log(keystore_data, "password_verified")
                
                self.logger.info("Password verified successfully")
                return True, derived_key
                
            except ValueError:
                # Decryption failed - wrong password
                self._update_access_log(keystore_data, "password_verification_failed")
                self.logger.warning("Password verification failed")
                return False, None
                
        except Exception as e:
            self.logger.error(f"Password verification error: {e}")
            return False, None
        finally:
            # Don't clear session key here - caller needs it
            pass
    
    def is_setup_required(self) -> bool:
        """Check if keystore setup is required."""
        if not self.keystore_path.exists():
            return True
        
        try:
            with open(self.keystore_path, 'r') as f:
                keystore_data = json.load(f)
            
            metadata = keystore_data.get('metadata', {})
            return not metadata.get('first_setup_completed', False)
            
        except Exception as e:
            self.logger.error(f"Error checking setup status: {e}")
            return True
    
    def validate_keystore(self) -> bool:
        """Validate keystore integrity and structure."""
        try:
            if not self.keystore_path.exists():
                self.logger.error("Keystore file does not exist")
                return False
            
            with open(self.keystore_path, 'r') as f:
                keystore_data = json.load(f)
            
            # Check required fields
            required_fields = ['metadata', 'kdf_config', 'salt', 'verifier']
            for field in required_fields:
                if field not in keystore_data:
                    self.logger.error(f"Missing required field: {field}")
                    return False
            
            # Validate metadata
            metadata = keystore_data['metadata']
            required_metadata = ['version', 'created_at', 'installation_id', 'first_setup_completed']
            for field in required_metadata:
                if field not in metadata:
                    self.logger.error(f"Missing metadata field: {field}")
                    return False
            
            # Validate verifier structure
            verifier = keystore_data['verifier']
            required_verifier = ['ciphertext', 'nonce', 'tag']
            for field in required_verifier:
                if field not in verifier:
                    self.logger.error(f"Missing verifier field: {field}")
                    return False
            
            # Validate hex encoding
            try:
                bytes.fromhex(keystore_data['salt'])
                bytes.fromhex(verifier['ciphertext'])
                bytes.fromhex(verifier['nonce'])
                bytes.fromhex(verifier['tag'])
            except ValueError as e:
                self.logger.error(f"Invalid hex encoding: {e}")
                return False
            
            self.logger.info("Keystore validation passed")
            return True
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Keystore JSON is corrupted: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Keystore validation failed: {e}")
            return False
    
    def get_metadata(self) -> Optional[KeystoreMetadata]:
        """Get keystore metadata."""
        try:
            if not self.keystore_path.exists():
                return None
            
            with open(self.keystore_path, 'r') as f:
                keystore_data = json.load(f)
            
            metadata_dict = keystore_data.get('metadata', {})
            return KeystoreMetadata(**metadata_dict)
            
        except Exception as e:
            self.logger.error(f"Failed to get metadata: {e}")
            return None
    
    def get_security_settings(self) -> Dict[str, Any]:
        """Get security settings from keystore."""
        try:
            if not self.keystore_path.exists():
                return {}
            
            with open(self.keystore_path, 'r') as f:
                keystore_data = json.load(f)
            
            return keystore_data.get('security_settings', {})
            
        except Exception as e:
            self.logger.error(f"Failed to get security settings: {e}")
            return {}
    
    def _update_access_log(self, keystore_data: Dict[str, Any], event: str) -> None:
        """Update access log in keystore."""
        try:
            # Update metadata
            metadata = keystore_data['metadata']
            metadata['last_accessed'] = datetime.now().isoformat()
            metadata['access_count'] = metadata.get('access_count', 0) + 1
            
            # Add audit log entry
            audit_entry = {
                "timestamp": datetime.now().isoformat(),
                "event": event,
                "details": {
                    "access_count": metadata['access_count']
                }
            }
            
            audit_log = keystore_data.get('audit_log', [])
            audit_log.append(audit_entry)
            
            # Keep only last 100 entries
            if len(audit_log) > 100:
                audit_log = audit_log[-100:]
            
            keystore_data['audit_log'] = audit_log
            
            # Save updated keystore
            with open(self.keystore_path, 'w') as f:
                json.dump(keystore_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to update access log: {e}")
    
    def keystore_exists(self) -> bool:
        """Check if keystore file exists."""
        return self.keystore_path.exists()

    def is_locked(self) -> bool:
        """Check if keystore is locked (requires password)."""
        if not self.keystore_exists():
            return True

        # If keystore exists but we don't have a session key, it's locked
        return not hasattr(self.crypto, 'session_key') or self.crypto.session_key is None

    def change_password(self, old_password: str, new_password: str) -> bool:
        """
        Change the master password for the keystore.

        Args:
            old_password: Current password
            new_password: New password to set

        Returns:
            bool: True if password changed successfully
        """
        try:
            # Verify old password first
            is_valid, session_key = self.verify_password(old_password)
            if not is_valid:
                self.logger.error("Old password verification failed")
                return False

            # Load current keystore
            with open(self.keystore_path, 'r') as f:
                keystore_data = json.load(f)

            # Create new crypto manager with new password
            new_crypto = CryptoManager()

            # Generate new salt and derive new key
            new_salt = secrets.token_bytes(16)
            new_key = new_crypto._derive_key(new_password, new_salt)

            # Update keystore with new password hash and salt
            keystore_data['password_hash'] = new_crypto._hash_password(new_password, new_salt).hex()
            keystore_data['salt'] = new_salt.hex()

            # Update metadata
            keystore_data['metadata']['last_accessed'] = datetime.now().isoformat()

            # Save updated keystore
            with open(self.keystore_path, 'w') as f:
                json.dump(keystore_data, f, indent=2)

            # Update current crypto manager
            self.crypto = new_crypto
            self.crypto.session_key = new_key

            self.logger.info("Password changed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to change password: {e}")
            return False

    def backup_keystore(self, backup_path: Optional[Path] = None) -> bool:
        """Create a backup of the keystore."""
        try:
            if not self.keystore_path.exists():
                self.logger.error("Cannot backup: keystore does not exist")
                return False
            
            if backup_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = self.keystore_path.parent / f"keystore_backup_{timestamp}.json"
            
            # Copy keystore
            import shutil
            shutil.copy2(self.keystore_path, backup_path)
            
            # Update backup timestamp in original
            try:
                with open(self.keystore_path, 'r') as f:
                    keystore_data = json.load(f)
                
                keystore_data['metadata']['last_backup'] = datetime.now().isoformat()
                
                with open(self.keystore_path, 'w') as f:
                    json.dump(keystore_data, f, indent=2)
            except Exception:
                pass  # Non-critical if this fails
            
            self.logger.info(f"Keystore backed up to: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            return False

    def is_initialized(self) -> bool:
        """
        Check if the keystore is initialized (exists and has valid structure).

        Returns:
            bool: True if keystore exists and is properly initialized
        """
        try:
            if not self.keystore_path.exists():
                return False

            # Try to load and validate keystore structure
            with open(self.keystore_path, 'r') as f:
                keystore_data = json.load(f)

            # Check for required fields
            required_fields = ['installation_id', 'kdf_config', 'verifier']
            for field in required_fields:
                if field not in keystore_data:
                    return False

            return True

        except Exception:
            return False
