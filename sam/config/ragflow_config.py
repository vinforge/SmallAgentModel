"""
RAGFlow Integration Configuration
Configuration settings for RAGFlow integration with SAM.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class RAGFlowEnvironment(Enum):
    """RAGFlow deployment environments."""
    LOCAL = "local"
    DOCKER = "docker"
    CLOUD = "cloud"
    CUSTOM = "custom"

@dataclass
class RAGFlowConfig:
    """RAGFlow integration configuration."""
    
    # Connection settings
    enabled: bool = True
    api_base_url: str = "http://localhost:9380/api/v1"
    api_key: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    
    # Knowledge base settings
    default_knowledge_base: str = "sam_documents"
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    default_chunk_template: str = "intelligent"
    default_language: str = "English"
    
    # Processing settings
    enable_ocr: bool = True
    enable_table_extraction: bool = True
    enable_image_analysis: bool = True
    max_file_size_mb: int = 100
    
    # Supported file formats
    supported_formats: list = None
    
    # Retrieval settings
    default_max_results: int = 5
    default_similarity_threshold: float = 0.3
    enable_hybrid_retrieval: bool = True
    
    # Fusion weights for hybrid retrieval
    fusion_weights: Dict[str, float] = None
    
    # Synchronization settings
    enable_sync: bool = True
    sync_interval: int = 300  # 5 minutes
    sync_batch_size: int = 10
    enable_auto_sync: bool = False
    
    # Caching settings
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    cache_max_size: int = 1000
    
    # Performance settings
    enable_parallel_processing: bool = True
    max_concurrent_uploads: int = 3
    chunk_processing_timeout: int = 600  # 10 minutes
    
    # Security settings
    enable_encryption: bool = False
    verify_ssl: bool = True
    
    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.supported_formats is None:
            self.supported_formats = [
                "pdf", "doc", "docx", "txt", "md", "mdx",
                "csv", "xlsx", "xls", "jpeg", "jpg", "png", 
                "tif", "gif", "ppt", "pptx"
            ]
        
        if self.fusion_weights is None:
            self.fusion_weights = {
                'ragflow': 0.6,
                'sam': 0.4,
                'similarity_boost': 0.1,
                'recency_boost': 0.05
            }

# Global configuration instance
_ragflow_config: Optional[RAGFlowConfig] = None

def get_ragflow_config() -> RAGFlowConfig:
    """
    Get RAGFlow configuration instance.
    
    Returns:
        RAGFlow configuration
    """
    global _ragflow_config
    
    if _ragflow_config is None:
        _ragflow_config = load_ragflow_config()
    
    return _ragflow_config

def load_ragflow_config() -> RAGFlowConfig:
    """
    Load RAGFlow configuration from environment variables and defaults.
    
    Returns:
        Loaded RAGFlow configuration
    """
    return RAGFlowConfig(
        # Connection settings
        enabled=os.getenv('RAGFLOW_ENABLED', 'true').lower() == 'true',
        api_base_url=os.getenv('RAGFLOW_API_URL', 'http://localhost:9380/api/v1'),
        api_key=os.getenv('RAGFLOW_API_KEY'),
        timeout=int(os.getenv('RAGFLOW_TIMEOUT', '30')),
        max_retries=int(os.getenv('RAGFLOW_MAX_RETRIES', '3')),
        
        # Knowledge base settings
        default_knowledge_base=os.getenv('RAGFLOW_KNOWLEDGE_BASE', 'sam_documents'),
        embedding_model=os.getenv('RAGFLOW_EMBEDDING_MODEL', 'BAAI/bge-large-en-v1.5'),
        default_chunk_template=os.getenv('RAGFLOW_CHUNK_TEMPLATE', 'intelligent'),
        default_language=os.getenv('RAGFLOW_LANGUAGE', 'English'),
        
        # Processing settings
        enable_ocr=os.getenv('RAGFLOW_ENABLE_OCR', 'true').lower() == 'true',
        enable_table_extraction=os.getenv('RAGFLOW_ENABLE_TABLES', 'true').lower() == 'true',
        enable_image_analysis=os.getenv('RAGFLOW_ENABLE_IMAGES', 'true').lower() == 'true',
        max_file_size_mb=int(os.getenv('RAGFLOW_MAX_FILE_SIZE_MB', '100')),
        
        # Retrieval settings
        default_max_results=int(os.getenv('RAGFLOW_MAX_RESULTS', '5')),
        default_similarity_threshold=float(os.getenv('RAGFLOW_SIMILARITY_THRESHOLD', '0.3')),
        enable_hybrid_retrieval=os.getenv('RAGFLOW_HYBRID_RETRIEVAL', 'true').lower() == 'true',
        
        # Synchronization settings
        enable_sync=os.getenv('RAGFLOW_ENABLE_SYNC', 'true').lower() == 'true',
        sync_interval=int(os.getenv('RAGFLOW_SYNC_INTERVAL', '300')),
        sync_batch_size=int(os.getenv('RAGFLOW_SYNC_BATCH_SIZE', '10')),
        enable_auto_sync=os.getenv('RAGFLOW_AUTO_SYNC', 'false').lower() == 'true',
        
        # Caching settings
        enable_caching=os.getenv('RAGFLOW_ENABLE_CACHING', 'true').lower() == 'true',
        cache_ttl=int(os.getenv('RAGFLOW_CACHE_TTL', '3600')),
        cache_max_size=int(os.getenv('RAGFLOW_CACHE_MAX_SIZE', '1000')),
        
        # Performance settings
        enable_parallel_processing=os.getenv('RAGFLOW_PARALLEL_PROCESSING', 'true').lower() == 'true',
        max_concurrent_uploads=int(os.getenv('RAGFLOW_MAX_CONCURRENT_UPLOADS', '3')),
        chunk_processing_timeout=int(os.getenv('RAGFLOW_CHUNK_TIMEOUT', '600')),
        
        # Security settings
        enable_encryption=os.getenv('RAGFLOW_ENABLE_ENCRYPTION', 'false').lower() == 'true',
        verify_ssl=os.getenv('RAGFLOW_VERIFY_SSL', 'true').lower() == 'true'
    )

def update_ragflow_config(**kwargs) -> RAGFlowConfig:
    """
    Update RAGFlow configuration with new values.
    
    Args:
        **kwargs: Configuration values to update
        
    Returns:
        Updated configuration
    """
    global _ragflow_config
    
    config = get_ragflow_config()
    
    # Update configuration values
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config

def validate_ragflow_config(config: RAGFlowConfig) -> Dict[str, Any]:
    """
    Validate RAGFlow configuration.
    
    Args:
        config: Configuration to validate
        
    Returns:
        Validation result with errors and warnings
    """
    errors = []
    warnings = []
    
    # Validate required settings
    if not config.api_base_url:
        errors.append("RAGFlow API base URL is required")
    
    if config.enabled and not config.api_key:
        warnings.append("RAGFlow API key not set - authentication may fail")
    
    # Validate numeric settings
    if config.timeout <= 0:
        errors.append("Timeout must be positive")
    
    if config.max_retries < 0:
        errors.append("Max retries cannot be negative")
    
    if config.max_file_size_mb <= 0:
        errors.append("Max file size must be positive")
    
    if config.default_similarity_threshold < 0 or config.default_similarity_threshold > 1:
        errors.append("Similarity threshold must be between 0 and 1")
    
    # Validate fusion weights
    if config.fusion_weights:
        total_weight = config.fusion_weights.get('ragflow', 0) + config.fusion_weights.get('sam', 0)
        if abs(total_weight - 1.0) > 0.1:
            warnings.append(f"Fusion weights don't sum to 1.0 (current: {total_weight})")
    
    # Validate file formats
    if not config.supported_formats:
        warnings.append("No supported file formats specified")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }

def get_environment_config() -> Dict[str, Any]:
    """
    Get environment-specific configuration for RAGFlow deployment.
    
    Returns:
        Environment configuration
    """
    # Detect environment
    if os.getenv('DOCKER_CONTAINER'):
        environment = RAGFlowEnvironment.DOCKER
        api_url = "http://ragflow:9380/api/v1"
    elif os.getenv('KUBERNETES_SERVICE_HOST'):
        environment = RAGFlowEnvironment.CLOUD
        api_url = os.getenv('RAGFLOW_SERVICE_URL', 'http://ragflow-service:9380/api/v1')
    else:
        environment = RAGFlowEnvironment.LOCAL
        api_url = "http://localhost:9380/api/v1"
    
    return {
        'environment': environment,
        'api_url': api_url,
        'deployment_type': environment.value,
        'docker_compose_file': 'docker-compose-ragflow.yml' if environment == RAGFlowEnvironment.DOCKER else None
    }

def create_docker_compose_config() -> str:
    """
    Create Docker Compose configuration for RAGFlow integration.
    
    Returns:
        Docker Compose YAML configuration
    """
    return """
version: '3.8'

services:
  ragflow:
    image: infiniflow/ragflow:v0.19.1
    container_name: ragflow-server
    ports:
      - "9380:9380"
    environment:
      - RAGFLOW_API_KEY=${RAGFLOW_API_KEY:-}
      - RAGFLOW_LOG_LEVEL=${RAGFLOW_LOG_LEVEL:-INFO}
    volumes:
      - ragflow_data:/opt/ragflow/data
      - ragflow_logs:/opt/ragflow/logs
    networks:
      - sam_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9380/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  ragflow-db:
    image: postgres:15
    container_name: ragflow-postgres
    environment:
      - POSTGRES_DB=ragflow
      - POSTGRES_USER=ragflow
      - POSTGRES_PASSWORD=${RAGFLOW_DB_PASSWORD:-ragflow123}
    volumes:
      - ragflow_db_data:/var/lib/postgresql/data
    networks:
      - sam_network
    restart: unless-stopped

  ragflow-redis:
    image: redis:7-alpine
    container_name: ragflow-redis
    volumes:
      - ragflow_redis_data:/data
    networks:
      - sam_network
    restart: unless-stopped

volumes:
  ragflow_data:
  ragflow_logs:
  ragflow_db_data:
  ragflow_redis_data:

networks:
  sam_network:
    external: true
"""

# Configuration validation on import
if __name__ == "__main__":
    config = get_ragflow_config()
    validation = validate_ragflow_config(config)
    
    if validation['valid']:
        print("✅ RAGFlow configuration is valid")
    else:
        print("❌ RAGFlow configuration has errors:")
        for error in validation['errors']:
            print(f"  - {error}")
    
    if validation['warnings']:
        print("⚠️ RAGFlow configuration warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
