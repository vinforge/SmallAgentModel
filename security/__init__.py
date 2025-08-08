"""
SAM Security Module

Provides enterprise-grade security features for SAM including:
- AES-256-GCM authenticated encryption
- Argon2id key derivation
- Secure keystore management
- Session management with automatic timeout
- Security UI components

Author: SAM Development Team
Version: 2.0.0
"""

# Core security components with graceful fallbacks
try:
    from .secure_state_manager import SecureStateManager, SecurityState
    _SECURE_STATE_AVAILABLE = True
except ImportError as e:
    _SECURE_STATE_AVAILABLE = False
    # Create fallback classes
    class SecurityState:
        UNAUTHENTICATED = "unauthenticated"
        AUTHENTICATED = "authenticated"

    class SecureStateManager:
        def __init__(self):
            self.available = False
        def is_authenticated(self):
            return False

try:
    from .crypto_utils import CryptoManager
    _CRYPTO_AVAILABLE = True
except ImportError as e:
    _CRYPTO_AVAILABLE = False
    class CryptoManager:
        def __init__(self):
            self.available = False
        def is_initialized(self):
            return False

try:
    from .keystore_manager import KeystoreManager
    _KEYSTORE_AVAILABLE = True
except ImportError as e:
    _KEYSTORE_AVAILABLE = False
    class KeystoreManager:
        def __init__(self):
            self.available = False

# Conditionally import components that require external dependencies
try:
    from .encrypted_chroma_store import EncryptedChromaStore
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False
    EncryptedChromaStore = None

try:
    from .security_ui import create_security_ui
    _UI_AVAILABLE = True
except ImportError:
    _UI_AVAILABLE = False
    create_security_ui = None

# Base exports (always available)
__all__ = [
    'SecureStateManager',
    'SecurityState',
    'CryptoManager',
    'KeystoreManager'
]

# Add optional components if available
if _CHROMA_AVAILABLE:
    __all__.append('EncryptedChromaStore')

if _UI_AVAILABLE:
    __all__.append('create_security_ui')

# Security status functions
def is_security_available():
    """Check if core security modules are available."""
    return _SECURE_STATE_AVAILABLE and _CRYPTO_AVAILABLE and _KEYSTORE_AVAILABLE

def get_security_status():
    """Get detailed security module status."""
    return {
        'secure_state_available': _SECURE_STATE_AVAILABLE,
        'crypto_available': _CRYPTO_AVAILABLE,
        'keystore_available': _KEYSTORE_AVAILABLE,
        'chroma_available': _CHROMA_AVAILABLE,
        'ui_available': _UI_AVAILABLE,
        'security_ready': is_security_available()
    }

def install_security_dependencies():
    """Install missing security dependencies."""
    import subprocess
    import sys

    packages = [
        'cryptography>=41.0.0',
        'argon2-cffi>=23.1.0',
        'pydantic>=2.0.0'
    ]

    try:
        subprocess.run([
            sys.executable, '-m', 'pip', 'install'
        ] + packages, check=True)
        return True
    except:
        return False

# Add utility functions to exports
__all__.extend(['is_security_available', 'get_security_status', 'install_security_dependencies'])
