"""
WebFetcher: Secure, isolated web content retrieval using browser automation.

This module provides the main WebFetcher class that handles web content
retrieval through isolated subprocess execution, ensuring the main SAM
application remains stable even if browser operations fail.
"""

import logging
import subprocess
import sys
import tempfile
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .data_contracts import WebContentData, create_timestamp
from .exceptions import (
    WebRetrievalError,
    ProcessIsolationError,
    ContentExtractionError,
    TimeoutError,
    NetworkError
)


@dataclass
class WebFetchResult:
    """Result container for web fetch operations."""
    
    url: str
    content: Optional[str]
    timestamp: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def success(self) -> bool:
        """Check if fetch was successful."""
        return self.error is None and self.content is not None
    
    def to_web_content_data(self) -> WebContentData:
        """Convert to WebContentData for validation and storage."""
        return WebContentData(
            url=self.url,
            content=self.content,
            timestamp=self.timestamp,
            error=self.error,
            metadata=self.metadata
        )


class WebFetcher:
    """
    Secure web content fetcher with process isolation.
    
    This class provides isolated web content retrieval using browser automation
    technologies. All browser operations run in separate processes to prevent
    crashes or instability from affecting the main SAM application.
    """
    
    def __init__(self, 
                 timeout: int = 30,
                 max_content_length: int = 1000000,
                 user_agent: str = None):
        """
        Initialize WebFetcher.
        
        Args:
            timeout: Maximum time to wait for page load (seconds)
            max_content_length: Maximum content size to retrieve (bytes)
            user_agent: Custom user agent string
        """
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.user_agent = user_agent or self._get_default_user_agent()
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration
        if timeout <= 0:
            raise ValueError("Timeout must be positive")
        if max_content_length <= 0:
            raise ValueError("Max content length must be positive")
    
    def fetch_url_content(self, url: str) -> WebFetchResult:
        """
        Fetch content from URL using isolated subprocess.
        
        This method spawns a separate process to handle browser automation,
        ensuring that any browser crashes or issues don't affect the main
        SAM application. This is our "digital air gap" for web operations.
        
        Args:
            url: Target URL to fetch content from
            
        Returns:
            WebFetchResult containing content or error information
            
        Raises:
            WebRetrievalError: For various retrieval failures
        """
        self.logger.info(f"Starting web fetch for URL: {url}")
        
        try:
            # Validate URL format
            self._validate_url(url)
            
            # Execute fetch in isolated subprocess
            result_data = self._execute_isolated_fetch(url)
            
            # Create result object
            result = WebFetchResult(
                url=url,
                content=result_data.get('content'),
                timestamp=result_data.get('timestamp', create_timestamp()),
                error=result_data.get('error'),
                metadata=result_data.get('metadata', {})
            )
            
            # Validate content length
            if result.content and len(result.content) > self.max_content_length:
                result.content = result.content[:self.max_content_length]
                result.metadata['content_truncated'] = True
                self.logger.warning(f"Content truncated to {self.max_content_length} characters")
            
            if result.success:
                self.logger.info(f"Successfully fetched {len(result.content)} characters from {url}")
            else:
                self.logger.error(f"Failed to fetch content from {url}: {result.error}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Unexpected error fetching {url}: {e}")
            return WebFetchResult(
                url=url,
                content=None,
                timestamp=create_timestamp(),
                error=f"Unexpected error: {str(e)}",
                metadata={'error_type': type(e).__name__}
            )
    
    def _validate_url(self, url: str) -> None:
        """Validate URL format and security constraints."""
        if not url or not isinstance(url, str):
            raise WebRetrievalError("URL must be a non-empty string")
        
        if not url.startswith(('http://', 'https://')):
            raise WebRetrievalError("URL must start with http:// or https://")
        
        # Additional security checks can be added here
        # e.g., blacklist certain domains, check for suspicious patterns
    
    def _execute_isolated_fetch(self, url: str) -> Dict[str, Any]:
        """Execute web fetch in isolated subprocess."""
        # Create the subprocess script
        script_content = self._create_fetch_script(url)
        
        # Write script to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            temp_script = f.name
        
        try:
            # Execute subprocess
            self.logger.debug(f"Executing isolated fetch subprocess for {url}")
            
            result = subprocess.run(
                [sys.executable, temp_script],
                capture_output=True,
                text=True,
                timeout=self.timeout + 10,  # Add buffer for subprocess overhead
                cwd=Path(__file__).parent.parent  # Run from project root
            )
            
            if result.returncode == 0:
                # Parse successful result
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError as e:
                    raise ProcessIsolationError(
                        f"Failed to parse subprocess output: {e}",
                        returncode=result.returncode,
                        stderr=result.stderr
                    )
            else:
                # Handle subprocess failure
                raise ProcessIsolationError(
                    f"Subprocess failed with return code {result.returncode}",
                    returncode=result.returncode,
                    stderr=result.stderr
                )
                
        except subprocess.TimeoutExpired:
            raise TimeoutError(
                f"Web fetch timed out after {self.timeout} seconds",
                timeout_seconds=self.timeout,
                url=url
            )
        finally:
            # Clean up temporary script
            try:
                os.unlink(temp_script)
            except Exception as e:
                self.logger.warning(f"Failed to clean up temporary script: {e}")
    
    def _create_fetch_script(self, url: str) -> str:
        """Create Python script for isolated web fetching."""
        return f'''
import sys
import json
from datetime import datetime, timezone

def fetch_with_requests(url):
    """Fallback method using requests + BeautifulSoup."""
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {{
            'User-Agent': '{self.user_agent}'
        }}

        response = requests.get(url, headers=headers, timeout={self.timeout})
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Extract text content
        text = soup.get_text()

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in str(line).split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        return text

    except Exception as e:
        raise Exception(f"Requests fetch failed: {{e}}")

def main():
    url = "{url}"

    try:
        # Try requests method (more reliable for now)
        content = fetch_with_requests(url)

        result = {{
            "url": url,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": None,
            "metadata": {{
                "content_length": len(content),
                "fetch_method": "requests+beautifulsoup",
                "user_agent": "{self.user_agent}"
            }}
        }}

        print(json.dumps(result))

    except Exception as e:
        error_result = {{
            "url": url,
            "content": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
            "metadata": {{
                "error_type": type(e).__name__,
                "fetch_method": "requests+beautifulsoup"
            }}
        }}
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    def _get_default_user_agent(self) -> str:
        """Get default user agent string."""
        return ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 SAM/1.0")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get fetcher statistics and configuration."""
        return {
            'timeout': self.timeout,
            'max_content_length': self.max_content_length,
            'user_agent': self.user_agent
        }
