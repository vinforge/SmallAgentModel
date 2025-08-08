"""
Web Retrieval Configuration Module
Provides configuration management for the intelligent web retrieval system.
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class WebRetrievalConfig:
    """Configuration for web retrieval system."""
    
    # Provider settings
    web_retrieval_provider: str = "cocoindex"  # cocoindex, legacy
    
    # CocoIndex settings
    cocoindex_num_pages: int = 5
    cocoindex_search_provider: str = "duckduckgo"  # serper, duckduckgo
    
    # API keys
    serper_api_key: Optional[str] = None
    newsapi_api_key: Optional[str] = None
    
    # Request settings
    timeout_seconds: int = 30
    max_retries: int = 3
    rate_limit_delay: float = 1.0
    
    # Content settings
    max_content_length: int = 50000
    min_content_length: int = 100
    
    # Security settings
    allowed_domains: list = None
    blocked_domains: list = None
    
    # Cache settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    
    def __post_init__(self):
        """Initialize default values."""
        if self.allowed_domains is None:
            self.allowed_domains = []
        if self.blocked_domains is None:
            self.blocked_domains = ['malware.com', 'spam.com']

def load_web_config() -> Dict[str, Any]:
    """Load web retrieval configuration from environment and defaults."""
    try:
        # Load from environment variables with fallbacks
        config = {
            'web_retrieval_provider': os.getenv('SAM_WEB_RETRIEVAL_PROVIDER', 'cocoindex'),
            'cocoindex_num_pages': int(os.getenv('SAM_COCOINDEX_NUM_PAGES', '5')),
            'cocoindex_search_provider': os.getenv('SAM_COCOINDEX_SEARCH_PROVIDER', 'duckduckgo'),
            'serper_api_key': os.getenv('SAM_SERPER_API_KEY', ''),
            'newsapi_api_key': os.getenv('SAM_NEWSAPI_API_KEY', ''),
            'timeout_seconds': int(os.getenv('SAM_WEB_TIMEOUT', '30')),
            'max_retries': int(os.getenv('SAM_WEB_MAX_RETRIES', '3')),
            'rate_limit_delay': float(os.getenv('SAM_WEB_RATE_LIMIT', '1.0')),
            'max_content_length': int(os.getenv('SAM_WEB_MAX_CONTENT', '50000')),
            'min_content_length': int(os.getenv('SAM_WEB_MIN_CONTENT', '100')),
            'enable_caching': os.getenv('SAM_WEB_ENABLE_CACHE', 'true').lower() == 'true',
            'cache_ttl_seconds': int(os.getenv('SAM_WEB_CACHE_TTL', '3600')),
            'allowed_domains': [],
            'blocked_domains': ['malware.com', 'spam.com']
        }
        
        logger.info(f"Loaded web retrieval config: provider={config['web_retrieval_provider']}, "
                   f"search_provider={config['cocoindex_search_provider']}, "
                   f"num_pages={config['cocoindex_num_pages']}")
        
        return config
        
    except Exception as e:
        logger.error(f"Error loading web config: {e}")
        # Return safe defaults
        return {
            'web_retrieval_provider': 'cocoindex',
            'cocoindex_num_pages': 5,
            'cocoindex_search_provider': 'duckduckgo',
            'serper_api_key': '',
            'newsapi_api_key': '',
            'timeout_seconds': 30,
            'max_retries': 3,
            'rate_limit_delay': 1.0,
            'max_content_length': 50000,
            'min_content_length': 100,
            'enable_caching': True,
            'cache_ttl_seconds': 3600,
            'allowed_domains': [],
            'blocked_domains': ['malware.com', 'spam.com']
        }

def get_web_config_object() -> WebRetrievalConfig:
    """Get web retrieval configuration as a dataclass object."""
    config_dict = load_web_config()
    
    return WebRetrievalConfig(
        web_retrieval_provider=config_dict['web_retrieval_provider'],
        cocoindex_num_pages=config_dict['cocoindex_num_pages'],
        cocoindex_search_provider=config_dict['cocoindex_search_provider'],
        serper_api_key=config_dict['serper_api_key'] if config_dict['serper_api_key'] else None,
        newsapi_api_key=config_dict['newsapi_api_key'] if config_dict['newsapi_api_key'] else None,
        timeout_seconds=config_dict['timeout_seconds'],
        max_retries=config_dict['max_retries'],
        rate_limit_delay=config_dict['rate_limit_delay'],
        max_content_length=config_dict['max_content_length'],
        min_content_length=config_dict['min_content_length'],
        allowed_domains=config_dict['allowed_domains'],
        blocked_domains=config_dict['blocked_domains'],
        enable_caching=config_dict['enable_caching'],
        cache_ttl_seconds=config_dict['cache_ttl_seconds']
    )

def validate_web_config(config: Dict[str, Any]) -> bool:
    """Validate web retrieval configuration."""
    try:
        required_keys = [
            'web_retrieval_provider',
            'cocoindex_num_pages',
            'cocoindex_search_provider',
            'timeout_seconds',
            'max_retries'
        ]
        
        for key in required_keys:
            if key not in config:
                logger.error(f"Missing required config key: {key}")
                return False
        
        # Validate provider
        if config['web_retrieval_provider'] not in ['cocoindex', 'legacy']:
            logger.error(f"Invalid web_retrieval_provider: {config['web_retrieval_provider']}")
            return False
        
        # Validate search provider
        if config['cocoindex_search_provider'] not in ['serper', 'duckduckgo']:
            logger.error(f"Invalid cocoindex_search_provider: {config['cocoindex_search_provider']}")
            return False
        
        # Validate numeric values
        if config['cocoindex_num_pages'] < 1 or config['cocoindex_num_pages'] > 20:
            logger.error(f"Invalid cocoindex_num_pages: {config['cocoindex_num_pages']}")
            return False
        
        if config['timeout_seconds'] < 5 or config['timeout_seconds'] > 120:
            logger.error(f"Invalid timeout_seconds: {config['timeout_seconds']}")
            return False
        
        logger.info("Web retrieval configuration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Error validating web config: {e}")
        return False

def get_default_web_config() -> Dict[str, Any]:
    """Get default web retrieval configuration."""
    return {
        'web_retrieval_provider': 'cocoindex',
        'cocoindex_num_pages': 5,
        'cocoindex_search_provider': 'duckduckgo',
        'serper_api_key': '',
        'newsapi_api_key': '',
        'timeout_seconds': 30,
        'max_retries': 3,
        'rate_limit_delay': 1.0,
        'max_content_length': 50000,
        'min_content_length': 100,
        'enable_caching': True,
        'cache_ttl_seconds': 3600,
        'allowed_domains': [],
        'blocked_domains': ['malware.com', 'spam.com']
    }

# Global configuration cache
_web_config_cache = None

def get_cached_web_config() -> Dict[str, Any]:
    """Get cached web configuration (loads once, reuses)."""
    global _web_config_cache
    
    if _web_config_cache is None:
        _web_config_cache = load_web_config()
    
    return _web_config_cache

def clear_web_config_cache():
    """Clear the web configuration cache."""
    global _web_config_cache
    _web_config_cache = None
    logger.info("Web configuration cache cleared")
