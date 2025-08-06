"""
Chroma Client Singleton for SAM
Provides cached Chroma client for Streamlit compatibility and performance.
"""

import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any
import streamlit as st

logger = logging.getLogger(__name__)

class ChromaClientManager:
    """Singleton manager for Chroma client with Streamlit caching support."""
    
    _instance = None
    _client = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChromaClientManager, cls).__new__(cls)
        return cls._instance
    
    @staticmethod
    @st.cache_resource
    def get_chroma_client(persist_path: str = "web_ui/chroma_db"):
        """
        Get cached Chroma client for Streamlit compatibility.
        
        Args:
            persist_path: Path to Chroma database directory
            
        Returns:
            ChromaDB client instance
        """
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Ensure directory exists
            chroma_path = Path(persist_path)
            chroma_path.mkdir(parents=True, exist_ok=True)
            
            # Configure Chroma settings
            settings = Settings(
                persist_directory=str(chroma_path),
                anonymized_telemetry=False,
                allow_reset=True
            )
            
            client = chromadb.PersistentClient(
                path=str(chroma_path),
                settings=settings
            )
            
            logger.info(f"Chroma client initialized: {persist_path}")
            return client
            
        except ImportError:
            logger.error("ChromaDB not available. Install with: pip install chromadb")
            raise
        except Exception as e:
            logger.error(f"Error initializing Chroma client: {e}")
            raise
    
    @staticmethod
    @st.cache_resource
    def get_chroma_collection(collection_name: str = "sam_memory_store", 
                             persist_path: str = "web_ui/chroma_db"):
        """
        Get cached Chroma collection for Streamlit compatibility.
        
        Args:
            collection_name: Name of the collection
            persist_path: Path to Chroma database directory
            
        Returns:
            ChromaDB collection instance
        """
        try:
            client = ChromaClientManager.get_chroma_client(persist_path)
            
            # Collection metadata (ChromaDB compatible - no lists)
            collection_metadata = {
                "description": "SAM Enhanced Memory Store with Citation Support",
                "version": "2.0",
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_dimension": "384",
                "distance_function": "cosine",
                "features": "metadata_filtering,hybrid_ranking,rich_citations"
            }
            
            collection = client.get_or_create_collection(
                name=collection_name,
                metadata=collection_metadata
            )
            
            logger.info(f"Chroma collection ready: {collection_name} ({collection.count()} items)")
            return collection
            
        except Exception as e:
            logger.error(f"Error getting Chroma collection: {e}")
            raise
    
    @staticmethod
    def load_config() -> Dict[str, Any]:
        """Load Chroma configuration from config file."""
        try:
            config_path = Path("config/sam_config.json")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    return config.get("memory", {}).get("chroma_config", {})
        except Exception as e:
            logger.warning(f"Could not load Chroma config: {e}")
        
        # Return default configuration
        return {
            "persist_path": "web_ui/chroma_db",
            "collection_name": "sam_memory_store",
            "distance_function": "cosine",
            "batch_size": 100
        }
    
    @staticmethod
    def reset_cache():
        """Reset Streamlit cache for Chroma resources."""
        try:
            ChromaClientManager.get_chroma_client.clear()
            ChromaClientManager.get_chroma_collection.clear()
            logger.info("Chroma cache cleared")
        except Exception as e:
            logger.warning(f"Error clearing Chroma cache: {e}")

# Convenience functions for easy access
def get_chroma_client(persist_path: str = "web_ui/chroma_db"):
    """Get Chroma client instance."""
    return ChromaClientManager.get_chroma_client(persist_path)

def get_chroma_collection(collection_name: str = "sam_memory_store", 
                         persist_path: str = "web_ui/chroma_db"):
    """Get Chroma collection instance."""
    return ChromaClientManager.get_chroma_collection(collection_name, persist_path)

def load_chroma_config():
    """Load Chroma configuration."""
    return ChromaClientManager.load_config()
