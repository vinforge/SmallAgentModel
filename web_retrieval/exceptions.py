"""
Custom exceptions for web retrieval operations.

This module defines specific exception types for different failure modes
in the web retrieval system, enabling precise error handling and debugging.
"""

class WebRetrievalError(Exception):
    """Base exception for all web retrieval operations."""
    
    def __init__(self, message: str, url: str = None, details: dict = None):
        super().__init__(message)
        self.url = url
        self.details = details or {}
        
    def to_dict(self) -> dict:
        """Convert exception to dictionary for JSON serialization."""
        return {
            'error_type': self.__class__.__name__,
            'message': str(self),
            'url': self.url,
            'details': self.details
        }


class ProcessIsolationError(WebRetrievalError):
    """Raised when subprocess execution fails."""
    
    def __init__(self, message: str, returncode: int = None, stderr: str = None, **kwargs):
        super().__init__(message, **kwargs)
        self.returncode = returncode
        self.stderr = stderr
        self.details.update({
            'returncode': returncode,
            'stderr': stderr
        })


class ContentExtractionError(WebRetrievalError):
    """Raised when content extraction from web page fails."""
    
    def __init__(self, message: str, extraction_method: str = None, **kwargs):
        super().__init__(message, **kwargs)
        self.extraction_method = extraction_method
        self.details.update({
            'extraction_method': extraction_method
        })


class TimeoutError(WebRetrievalError):
    """Raised when web retrieval operation times out."""
    
    def __init__(self, message: str, timeout_seconds: int = None, **kwargs):
        super().__init__(message, **kwargs)
        self.timeout_seconds = timeout_seconds
        self.details.update({
            'timeout_seconds': timeout_seconds
        })


class ValidationError(WebRetrievalError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, validation_errors: list = None, **kwargs):
        super().__init__(message, **kwargs)
        self.validation_errors = validation_errors or []
        self.details.update({
            'validation_errors': self.validation_errors
        })


class NetworkError(WebRetrievalError):
    """Raised when network connectivity issues occur."""
    
    def __init__(self, message: str, status_code: int = None, **kwargs):
        super().__init__(message, **kwargs)
        self.status_code = status_code
        self.details.update({
            'status_code': status_code
        })


class SecurityError(WebRetrievalError):
    """Raised when security constraints are violated."""
    
    def __init__(self, message: str, security_check: str = None, **kwargs):
        super().__init__(message, **kwargs)
        self.security_check = security_check
        self.details.update({
            'security_check': security_check
        })
