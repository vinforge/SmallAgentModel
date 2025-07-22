"""
SAM Web UI Security Middleware

Provides security integration for the Flask web application.
Handles authentication, session management, and secure routing.

Author: SAM Development Team
Version: 1.0.0
"""

import logging
from functools import wraps
from flask import jsonify, request, session
from typing import Callable, Any

logger = logging.getLogger(__name__)

class SecurityMiddleware:
    """Security middleware for SAM web UI."""
    
    def __init__(self):
        self.security_manager = None
        self._initialize_security()
    
    def _initialize_security(self):
        """Initialize security manager."""
        try:
            from security import SecureStateManager
            self.security_manager = SecureStateManager()
            logger.info("Security middleware initialized")
        except ImportError:
            logger.warning("Security module not available")
        except Exception as e:
            logger.error(f"Failed to initialize security middleware: {e}")
    
    def require_unlock(self, f: Callable) -> Callable:
        """Decorator to require security unlock for sensitive operations."""
        @wraps(f)
        def wrapper(*args, **kwargs):
            if not self.security_manager:
                return jsonify({
                    'error': 'Security not available',
                    'requires_setup': True
                }), 500
            
            if not self.security_manager.is_unlocked():
                return jsonify({
                    'error': 'Application is locked',
                    'requires_unlock': True,
                    'setup_required': self.security_manager.is_setup_required(),
                    'security_status': self._get_security_status()
                }), 403
            
            # Extend session on successful access
            self.security_manager.extend_session()
            
            return f(*args, **kwargs)
        return wrapper
    
    def optional_security(self, f: Callable) -> Callable:
        """Decorator for operations that work with or without security."""
        @wraps(f)
        def wrapper(*args, **kwargs):
            # Add security context to request
            request.security_context = {
                'security_available': self.security_manager is not None,
                'is_unlocked': self.security_manager.is_unlocked() if self.security_manager else False,
                'setup_required': self.security_manager.is_setup_required() if self.security_manager else False
            }
            
            return f(*args, **kwargs)
        return wrapper
    
    def _get_security_status(self) -> dict:
        """Get current security status."""
        if not self.security_manager:
            return {'available': False}
        
        session_info = self.security_manager.get_session_info()
        return {
            'available': True,
            'state': session_info['state'],
            'is_unlocked': session_info['is_unlocked'],
            'setup_required': self.security_manager.is_setup_required(),
            'time_remaining': session_info['time_remaining'],
            'failed_attempts': session_info['failed_attempts'],
            'max_attempts': session_info['max_attempts']
        }
    
    def setup_master_password(self, password: str) -> dict:
        """Setup master password."""
        if not self.security_manager:
            return {'success': False, 'error': 'Security not available'}
        
        if not self.security_manager.is_setup_required():
            return {'success': False, 'error': 'Setup already completed'}
        
        try:
            success = self.security_manager.setup_master_password(password)
            
            if success:
                return {
                    'success': True,
                    'message': 'Master password setup successful',
                    'security_status': self._get_security_status()
                }
            else:
                return {'success': False, 'error': 'Setup failed'}
                
        except Exception as e:
            logger.error(f"Master password setup failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def unlock_application(self, password: str) -> dict:
        """Unlock application with password."""
        if not self.security_manager:
            return {'success': False, 'error': 'Security not available'}
        
        try:
            success = self.security_manager.unlock_application(password)
            
            if success:
                return {
                    'success': True,
                    'message': 'Application unlocked successfully',
                    'security_status': self._get_security_status()
                }
            else:
                return {
                    'success': False, 
                    'error': 'Invalid password',
                    'security_status': self._get_security_status()
                }
                
        except Exception as e:
            logger.error(f"Application unlock failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def lock_application(self) -> dict:
        """Lock the application."""
        if not self.security_manager:
            return {'success': False, 'error': 'Security not available'}
        
        try:
            self.security_manager.lock_application()
            return {
                'success': True,
                'message': 'Application locked successfully',
                'security_status': self._get_security_status()
            }
            
        except Exception as e:
            logger.error(f"Application lock failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def extend_session(self) -> dict:
        """Extend the current session."""
        if not self.security_manager:
            return {'success': False, 'error': 'Security not available'}
        
        if not self.security_manager.is_unlocked():
            return {'success': False, 'error': 'Application is locked'}
        
        try:
            self.security_manager.extend_session()
            return {
                'success': True,
                'message': 'Session extended successfully',
                'security_status': self._get_security_status()
            }
            
        except Exception as e:
            logger.error(f"Session extension failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_status(self) -> dict:
        """Get comprehensive security status."""
        return self._get_security_status()

# Global security middleware instance
security_middleware = SecurityMiddleware()

# Convenience decorators
def require_unlock(f: Callable) -> Callable:
    """Decorator to require security unlock."""
    return security_middleware.require_unlock(f)

def optional_security(f: Callable) -> Callable:
    """Decorator for optional security operations."""
    return security_middleware.optional_security(f)

def get_secure_memory_store():
    """Get secure memory store instance."""
    try:
        from memory.secure_memory_vectorstore import get_secure_memory_store, VectorStoreType
        
        return get_secure_memory_store(
            store_type=VectorStoreType.CHROMA,
            storage_directory="web_ui",
            embedding_dimension=384,
            enable_encryption=True
        )
    except Exception as e:
        logger.error(f"Failed to get secure memory store: {e}")
        # Fallback to regular memory store
        from memory.memory_vectorstore import get_memory_store, VectorStoreType
        return get_memory_store(
            store_type=VectorStoreType.CHROMA,
            storage_directory="web_ui",
            embedding_dimension=384
        )

def create_security_routes(app):
    """Create security-related routes for the Flask app."""
    
    @app.route('/api/security/status')
    def security_status():
        """Get security status."""
        return jsonify(security_middleware.get_status())
    
    @app.route('/api/security/setup', methods=['POST'])
    def security_setup():
        """Setup master password."""
        data = request.get_json()
        password = data.get('password')
        
        if not password:
            return jsonify({'success': False, 'error': 'Password required'}), 400
        
        result = security_middleware.setup_master_password(password)
        status_code = 200 if result['success'] else 400
        
        return jsonify(result), status_code
    
    @app.route('/api/security/unlock', methods=['POST'])
    def security_unlock():
        """Unlock application."""
        data = request.get_json()
        password = data.get('password')
        
        if not password:
            return jsonify({'success': False, 'error': 'Password required'}), 400
        
        result = security_middleware.unlock_application(password)
        status_code = 200 if result['success'] else 401
        
        return jsonify(result), status_code
    
    @app.route('/api/security/lock', methods=['POST'])
    def security_lock():
        """Lock application."""
        result = security_middleware.lock_application()
        return jsonify(result)
    
    @app.route('/api/security/extend', methods=['POST'])
    def security_extend():
        """Extend session."""
        result = security_middleware.extend_session()
        status_code = 200 if result['success'] else 403
        
        return jsonify(result), status_code
    
    @app.route('/api/security/dashboard')
    @require_unlock
    def security_dashboard():
        """Get security dashboard data."""
        try:
            # Get comprehensive security information
            security_status = security_middleware.security_manager.get_security_status()
            
            # Get memory store security info
            memory_store = get_secure_memory_store()
            memory_security = memory_store.get_security_status()
            
            dashboard_data = {
                'security_status': security_status,
                'memory_security': memory_security,
                'system_info': {
                    'encryption_algorithm': 'AES-256-GCM',
                    'kdf_algorithm': 'Argon2id',
                    'security_level': 'Enterprise'
                }
            }
            
            return jsonify(dashboard_data)
            
        except Exception as e:
            logger.error(f"Security dashboard failed: {e}")
            return jsonify({'error': str(e)}), 500

def inject_security_context():
    """Inject security context into templates."""
    if security_middleware.security_manager:
        return {
            'security_available': True,
            'is_unlocked': security_middleware.security_manager.is_unlocked(),
            'setup_required': security_middleware.security_manager.is_setup_required()
        }
    else:
        return {
            'security_available': False,
            'is_unlocked': False,
            'setup_required': False
        }
