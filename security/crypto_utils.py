"""
Cryptographic Utilities for SAM Security Module

Provides AES-256-GCM authenticated encryption with secure key derivation
using Argon2id. Implements enterprise-grade cryptographic standards.

Author: SAM Development Team
Version: 2.0.0
"""

import os
import secrets
import logging
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import InvalidTag

try:
    import argon2
    ARGON2_AVAILABLE = True
except ImportError:
    ARGON2_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class EncryptionResult:
    """Result of encryption operation."""
    ciphertext: bytes
    nonce: bytes
    tag: bytes
    metadata: Dict[str, Any]

@dataclass
class DecryptionResult:
    """Result of decryption operation."""
    plaintext: bytes
    metadata: Dict[str, Any]

class CryptoManager:
    """
    Enterprise-grade cryptographic manager for SAM.
    
    Features:
    - AES-256-GCM authenticated encryption
    - Argon2id key derivation (fallback to PBKDF2-SHA256)
    - Secure random generation
    - Session key management
    - Metadata protection
    """
    
    def __init__(self):
        self.session_key: Optional[bytes] = None
        self.aesgcm: Optional[AESGCM] = None
        self.logger = logging.getLogger(f"{__name__}.CryptoManager")
        
        # Encryption parameters
        self.key_length = 32  # 256 bits
        self.nonce_length = 12  # 96 bits for GCM
        self.salt_length = 16  # 128 bits
        
        # Argon2id parameters (enterprise grade)
        self.argon2_params = {
            'time_cost': 3,      # Number of iterations
            'memory_cost': 65536,  # Memory usage in KB (64 MB)
            'parallelism': 4,    # Number of parallel threads
            'hash_len': 32,      # Output length in bytes
            'salt_len': 16       # Salt length in bytes
        }
        
        # PBKDF2 parameters (fallback)
        self.pbkdf2_iterations = 100000
        
        self.logger.info("CryptoManager initialized")
    
    def set_session_key(self, key: bytes) -> None:
        """
        Set the session encryption key.
        
        Args:
            key: 32-byte encryption key
        """
        if len(key) != self.key_length:
            raise ValueError(f"Key must be exactly {self.key_length} bytes")
        
        self.session_key = key
        self.aesgcm = AESGCM(key)
        self.logger.info("Session key set and AES-GCM initialized")
    
    def clear_session_key(self) -> None:
        """Clear the session key from memory."""
        if self.session_key:
            # Overwrite key in memory (best effort)
            self.session_key = b'\x00' * len(self.session_key)
        
        self.session_key = None
        self.aesgcm = None
        self.logger.info("Session key cleared from memory")
    
    def derive_key_from_password(self, password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """
        Derive encryption key from password using Argon2id or PBKDF2.
        
        Args:
            password: Master password
            salt: Optional salt (generated if not provided)
            
        Returns:
            Tuple of (derived_key, salt)
        """
        if salt is None:
            salt = secrets.token_bytes(self.salt_length)
        
        if ARGON2_AVAILABLE:
            return self._derive_key_argon2(password, salt), salt
        else:
            self.logger.warning("Argon2 not available, falling back to PBKDF2-SHA256")
            return self._derive_key_pbkdf2(password, salt), salt
    
    def _derive_key_argon2(self, password: str, salt: bytes) -> bytes:
        """Derive key using Argon2id."""
        ph = argon2.PasswordHasher(
            time_cost=self.argon2_params['time_cost'],
            memory_cost=self.argon2_params['memory_cost'],
            parallelism=self.argon2_params['parallelism'],
            hash_len=self.argon2_params['hash_len'],
            salt_len=self.argon2_params['salt_len']
        )
        
        # Use low-level API for key derivation
        return argon2.low_level.hash_secret_raw(
            secret=password.encode('utf-8'),
            salt=salt,
            time_cost=self.argon2_params['time_cost'],
            memory_cost=self.argon2_params['memory_cost'],
            parallelism=self.argon2_params['parallelism'],
            hash_len=self.argon2_params['hash_len'],
            type=argon2.Type.ID
        )
    
    def _derive_key_pbkdf2(self, password: str, salt: bytes) -> bytes:
        """Derive key using PBKDF2-SHA256 (fallback)."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self.key_length,
            salt=salt,
            iterations=self.pbkdf2_iterations,
        )
        return kdf.derive(password.encode('utf-8'))
    
    def encrypt(self, plaintext: str, associated_data: Optional[str] = None) -> EncryptionResult:
        """
        Encrypt plaintext using AES-256-GCM.
        
        Args:
            plaintext: Text to encrypt
            associated_data: Optional associated data for authentication
            
        Returns:
            EncryptionResult with ciphertext, nonce, and metadata
        """
        if not self.aesgcm:
            raise RuntimeError("No session key set. Call set_session_key() first.")
        
        # Generate random nonce
        nonce = secrets.token_bytes(self.nonce_length)
        
        # Convert to bytes
        plaintext_bytes = plaintext.encode('utf-8')
        associated_data_bytes = associated_data.encode('utf-8') if associated_data else None
        
        # Encrypt with authentication
        ciphertext = self.aesgcm.encrypt(nonce, plaintext_bytes, associated_data_bytes)
        
        # Extract tag (last 16 bytes)
        tag = ciphertext[-16:]
        ciphertext_only = ciphertext[:-16]
        
        metadata = {
            'algorithm': 'AES-256-GCM',
            'nonce_length': len(nonce),
            'tag_length': len(tag),
            'encrypted_at': datetime.now().isoformat(),
            'has_associated_data': associated_data is not None
        }
        
        self.logger.debug(f"Encrypted {len(plaintext_bytes)} bytes")
        
        return EncryptionResult(
            ciphertext=ciphertext_only,
            nonce=nonce,
            tag=tag,
            metadata=metadata
        )
    
    def decrypt(self, ciphertext: bytes, nonce: bytes, tag: bytes, 
                associated_data: Optional[str] = None) -> DecryptionResult:
        """
        Decrypt ciphertext using AES-256-GCM.
        
        Args:
            ciphertext: Encrypted data
            nonce: Nonce used for encryption
            tag: Authentication tag
            associated_data: Optional associated data for authentication
            
        Returns:
            DecryptionResult with plaintext and metadata
        """
        if not self.aesgcm:
            raise RuntimeError("No session key set. Call set_session_key() first.")
        
        try:
            # Reconstruct full ciphertext with tag
            full_ciphertext = ciphertext + tag
            
            # Convert associated data
            associated_data_bytes = associated_data.encode('utf-8') if associated_data else None
            
            # Decrypt and verify
            plaintext_bytes = self.aesgcm.decrypt(nonce, full_ciphertext, associated_data_bytes)
            
            metadata = {
                'algorithm': 'AES-256-GCM',
                'decrypted_at': datetime.now().isoformat(),
                'plaintext_length': len(plaintext_bytes)
            }
            
            self.logger.debug(f"Decrypted {len(plaintext_bytes)} bytes")
            
            return DecryptionResult(
                plaintext=plaintext_bytes,
                metadata=metadata
            )
            
        except InvalidTag:
            self.logger.error("Decryption failed: Invalid authentication tag")
            raise ValueError("Decryption failed: Data may be corrupted or tampered with")
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            raise ValueError(f"Decryption failed: {e}")
    
    def encrypt_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encrypt a dictionary, preserving structure for searchable fields.
        
        Args:
            data: Dictionary to encrypt
            
        Returns:
            Dictionary with encrypted sensitive fields
        """
        if not self.aesgcm:
            raise RuntimeError("No session key set. Call set_session_key() first.")
        
        # Define which fields should remain searchable (not encrypted)
        searchable_fields = {
            'id', 'timestamp', 'source_type', 'content_type', 
            'tags', 'importance_score', 'memory_type'
        }
        
        encrypted_data = {}
        
        for key, value in data.items():
            if key in searchable_fields:
                # Keep searchable fields as plaintext
                encrypted_data[key] = value
            else:
                # Encrypt sensitive fields
                if isinstance(value, (str, int, float, bool)):
                    str_value = str(value)
                    result = self.encrypt(str_value, associated_data=key)
                    
                    encrypted_data[f"{key}_encrypted"] = {
                        'ciphertext': result.ciphertext.hex(),
                        'nonce': result.nonce.hex(),
                        'tag': result.tag.hex(),
                        'metadata': result.metadata
                    }
                else:
                    # For complex objects, convert to string first
                    import json
                    str_value = json.dumps(value, default=str)
                    result = self.encrypt(str_value, associated_data=key)
                    
                    encrypted_data[f"{key}_encrypted"] = {
                        'ciphertext': result.ciphertext.hex(),
                        'nonce': result.nonce.hex(),
                        'tag': result.tag.hex(),
                        'metadata': result.metadata,
                        'original_type': type(value).__name__
                    }
        
        return encrypted_data
    
    def decrypt_dict(self, encrypted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decrypt a dictionary that was encrypted with encrypt_dict.
        
        Args:
            encrypted_data: Dictionary with encrypted fields
            
        Returns:
            Dictionary with decrypted fields
        """
        if not self.aesgcm:
            raise RuntimeError("No session key set. Call set_session_key() first.")
        
        decrypted_data = {}
        
        for key, value in encrypted_data.items():
            if key.endswith('_encrypted'):
                # Decrypt encrypted field
                original_key = key[:-10]  # Remove '_encrypted' suffix
                
                if isinstance(value, dict) and 'ciphertext' in value:
                    try:
                        ciphertext = bytes.fromhex(value['ciphertext'])
                        nonce = bytes.fromhex(value['nonce'])
                        tag = bytes.fromhex(value['tag'])
                        
                        result = self.decrypt(ciphertext, nonce, tag, associated_data=original_key)
                        plaintext = result.plaintext.decode('utf-8')
                        
                        # Convert back to original type if specified
                        if 'original_type' in value:
                            original_type = value['original_type']
                            if original_type == 'dict' or original_type == 'list':
                                import json
                                decrypted_data[original_key] = json.loads(plaintext)
                            elif original_type == 'int':
                                decrypted_data[original_key] = int(plaintext)
                            elif original_type == 'float':
                                decrypted_data[original_key] = float(plaintext)
                            elif original_type == 'bool':
                                decrypted_data[original_key] = plaintext.lower() == 'true'
                            else:
                                decrypted_data[original_key] = plaintext
                        else:
                            decrypted_data[original_key] = plaintext
                            
                    except Exception as e:
                        self.logger.error(f"Failed to decrypt field {original_key}: {e}")
                        decrypted_data[original_key] = f"[DECRYPTION_FAILED: {e}]"
            else:
                # Keep non-encrypted fields as-is
                decrypted_data[key] = value
        
        return decrypted_data
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate a cryptographically secure random token."""
        return secrets.token_urlsafe(length)

    def constant_time_compare(self, a: str, b: str) -> bool:
        """Constant-time string comparison to prevent timing attacks."""
        return secrets.compare_digest(a.encode('utf-8'), b.encode('utf-8'))

    def is_initialized(self) -> bool:
        """
        Check if the crypto manager is initialized with a session key.

        Returns:
            bool: True if session key is set and ready for encryption/decryption
        """
        return self.session_key is not None and self.aesgcm is not None

    # Convenience wrappers used by encrypted stores
    def encrypt_data(self, plaintext: str) -> str:
        """Encrypt text and return a JSON string with ciphertext, nonce, tag (hex-encoded)."""
        res = self.encrypt(plaintext)
        payload = {
            'ciphertext': res.ciphertext.hex(),
            'nonce': res.nonce.hex(),
            'tag': res.tag.hex(),
            'meta': res.metadata,
        }
        import json as _json
        return _json.dumps(payload)

    def decrypt_data(self, data: str) -> str:
        """Decrypt JSON string produced by encrypt_data and return plaintext string."""
        import json as _json
        obj = data
        if isinstance(data, str):
            try:
                obj = _json.loads(data)
            except Exception:
                raise ValueError("decrypt_data expects JSON string payload")
        if not isinstance(obj, dict):
            raise ValueError("decrypt_data expects JSON object")
        ciphertext = bytes.fromhex(obj['ciphertext'])
        nonce = bytes.fromhex(obj['nonce'])
        tag = bytes.fromhex(obj['tag'])
        dec = self.decrypt(ciphertext, nonce, tag)
        return dec.plaintext.decode('utf-8')
