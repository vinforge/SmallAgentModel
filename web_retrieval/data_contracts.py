"""
Data contracts and validation schemas for web retrieval operations.

This module defines the structured data formats used throughout the web
retrieval system, ensuring consistent data handling and validation.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import json
import re
from urllib.parse import urlparse

from .exceptions import ValidationError


@dataclass
class WebContentData:
    """Structured container for web content data."""
    
    url: str
    content: Optional[str]
    timestamp: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate data after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate the web content data structure."""
        errors = []
        
        # Validate URL
        if not self.url:
            errors.append("URL is required")
        elif not self._is_valid_url(self.url):
            errors.append(f"Invalid URL format: {self.url}")
        
        # Validate timestamp
        if not self.timestamp:
            errors.append("Timestamp is required")
        elif not self._is_valid_iso_timestamp(self.timestamp):
            errors.append(f"Invalid timestamp format: {self.timestamp}")
        
        # Validate content constraints
        if self.content and len(self.content) > 1000000:  # 1MB limit
            errors.append("Content exceeds maximum size limit (1MB)")
        
        # Validate metadata
        if not isinstance(self.metadata, dict):
            errors.append("Metadata must be a dictionary")
        
        if errors:
            raise ValidationError("Data validation failed", validation_errors=errors)
    
    @staticmethod
    def _is_valid_url(url: str) -> bool:
        """Check if URL has valid format."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    @staticmethod
    def _is_valid_iso_timestamp(timestamp: str) -> bool:
        """Check if timestamp is valid ISO format."""
        try:
            datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return True
        except Exception:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'url': self.url,
            'content': self.content,
            'timestamp': self.timestamp,
            'error': self.error,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WebContentData':
        """Create instance from dictionary."""
        return cls(
            url=data.get('url', ''),
            content=data.get('content'),
            timestamp=data.get('timestamp', ''),
            error=data.get('error'),
            metadata=data.get('metadata', {})
        )
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'WebContentData':
        """Create instance from JSON string."""
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON format: {e}")


@dataclass
class QuarantineMetadata:
    """Metadata for quarantined web content."""
    
    filename: str
    source_url: str
    fetch_timestamp: str
    content_length: int
    content_type: Optional[str] = None
    fetch_method: str = "manual"
    security_status: str = "untrusted"
    review_status: str = "pending"
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'filename': self.filename,
            'source_url': self.source_url,
            'fetch_timestamp': self.fetch_timestamp,
            'content_length': self.content_length,
            'content_type': self.content_type,
            'fetch_method': self.fetch_method,
            'security_status': self.security_status,
            'review_status': self.review_status,
            'tags': self.tags
        }


def create_timestamp() -> str:
    """Create ISO timestamp for current UTC time."""
    return datetime.now(timezone.utc).isoformat()


def sanitize_filename(url: str, max_length: int = 100) -> str:
    """Create safe filename from URL."""
    parsed = urlparse(url)
    domain = parsed.netloc.replace('www.', '')
    
    # Remove invalid filename characters
    safe_domain = re.sub(r'[<>:"/\\|?*]', '_', domain)
    
    # Truncate if too long
    if len(safe_domain) > max_length - 20:  # Leave room for timestamp
        safe_domain = safe_domain[:max_length - 20]
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    return f"{safe_domain}_{timestamp}.json"
