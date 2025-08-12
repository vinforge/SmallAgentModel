"""
SAM Encrypted ChromaDB Store

Provides transparent encryption/decryption for ChromaDB collections.
Integrates with SAM's security system for secure memory storage.

Author: SAM Development Team
Version: 1.0.0
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

import chromadb
from chromadb.config import Settings

from .crypto_utils import CryptoManager

logger = logging.getLogger(__name__)

class EncryptedChromaStore:
    """
    Encrypted ChromaDB store with transparent encryption/decryption.
    
    Provides secure storage for memory chunks with metadata encryption
    and seamless integration with SAM's security system.
    """
    
    def __init__(self, collection_name: str = "sam_secure_memory",
                 crypto_manager: CryptoManager = None,
                 storage_path: str = "memory_store/encrypted"):
        """
        Initialize encrypted ChromaDB store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            crypto_manager: CryptoManager instance for encryption
            storage_path: Path for storing encrypted data
        """
        self.collection_name = collection_name
        self.crypto_manager = crypto_manager
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        if not self.crypto_manager:
            raise ValueError("CryptoManager is required for encrypted storage")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.storage_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"✅ Connected to existing encrypted collection: {collection_name}")
        except Exception:
            # Collection doesn't exist or cannot be fetched, create it
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"encrypted": True, "created_at": time.time()}
            )
            logger.info(f"✅ Created new encrypted collection: {collection_name}")
        
        logger.info(f"Encrypted ChromaDB store initialized at {self.storage_path}")
    
    def add_memory_chunk(self, chunk_text: str, metadata: Dict[str, Any],
                        embedding: List[float], chunk_id: str = None) -> str:
        """
        Add encrypted memory chunk to the store.
        
        Args:
            chunk_text: Text content to encrypt and store
            metadata: Metadata dictionary to encrypt
            embedding: Embedding vector (not encrypted)
            chunk_id: Optional chunk ID (generated if not provided)
            
        Returns:
            Chunk ID of the stored memory
        """
        try:
            # Generate chunk ID if not provided
            if not chunk_id:
                chunk_id = f"enc_{hash(chunk_text + str(time.time()))%1000000:06d}"
            
            # Encrypt the content
            encrypted_content = self.crypto_manager.encrypt_data(chunk_text)
            
            # Encrypt sensitive metadata fields
            encrypted_metadata = self._encrypt_metadata(metadata)
            
            # Add timestamp and encryption markers
            encrypted_metadata.update({
                'encrypted': True,
                'created_at': time.time(),
                'chunk_id': chunk_id
            })
            
            # Store in ChromaDB
            self.collection.add(
                documents=[encrypted_content],
                metadatas=[encrypted_metadata],
                embeddings=[embedding],
                ids=[chunk_id]
            )
            
            logger.debug(f"Added encrypted memory chunk: {chunk_id}")
            return chunk_id
            
        except Exception as e:
            logger.error(f"Failed to add encrypted memory chunk: {e}")
            raise
    
    def query_memories(self, query_embedding: List[float], n_results: int = 5,
                      where_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Query encrypted memories and decrypt results.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where_filter: Optional metadata filter
            
        Returns:
            List of decrypted memory results
        """
        try:
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Decrypt and format results
            decrypted_results = []
            
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    try:
                        # Decrypt content
                        decrypted_content = self.crypto_manager.decrypt_data(doc)
                        
                        # Decrypt metadata
                        decrypted_metadata = self._decrypt_metadata(metadata)
                        
                        # Create result object
                        result = {
                            'id': results['ids'][0][i],
                            'content': decrypted_content,
                            'metadata': decrypted_metadata,
                            'distance': distance
                        }
                        
                        decrypted_results.append(result)
                        
                    except Exception as e:
                        logger.warning(f"Failed to decrypt result {i}: {e}")
                        continue
            
            logger.debug(f"Queried {len(decrypted_results)} encrypted memories")
            return decrypted_results
            
        except Exception as e:
            logger.error(f"Failed to query encrypted memories: {e}")
            return []
    
    def get_memory_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Get specific memory by ID and decrypt.
        
        Args:
            chunk_id: ID of the memory chunk
            
        Returns:
            Decrypted memory data or None if not found
        """
        try:
            results = self.collection.get(
                ids=[chunk_id],
                include=['documents', 'metadatas']
            )
            
            if results['documents'] and results['documents'][0]:
                doc = results['documents'][0]
                metadata = results['metadatas'][0]
                
                # Decrypt content and metadata
                decrypted_content = self.crypto_manager.decrypt_data(doc)
                decrypted_metadata = self._decrypt_metadata(metadata)
                
                return {
                    'id': chunk_id,
                    'content': decrypted_content,
                    'metadata': decrypted_metadata
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get memory by ID {chunk_id}: {e}")
            return None
    
    def delete_memory(self, chunk_id: str) -> bool:
        """
        Delete memory chunk by ID.
        
        Args:
            chunk_id: ID of the memory chunk to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=[chunk_id])
            logger.debug(f"Deleted encrypted memory: {chunk_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete memory {chunk_id}: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the encrypted collection.
        
        Returns:
            Collection information dictionary
        """
        try:
            count = self.collection.count()
            
            return {
                'name': self.collection_name,
                'chunk_count': count,
                'encrypted': True,
                'storage_path': str(self.storage_path),
                'searchable_fields': ['chunk_id', 'created_at', 'sam_memory_type'],
                'encrypted_fields': ['content', 'source', 'tags', 'metadata']
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}
    
    def _encrypt_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encrypt sensitive metadata fields.
        
        Args:
            metadata: Original metadata dictionary
            
        Returns:
            Metadata with encrypted sensitive fields
        """
        encrypted_metadata = {}
        
        # Fields that should be encrypted
        sensitive_fields = ['source', 'tags', 'author', 'classification']
        
        # Fields that should remain searchable (not encrypted)
        searchable_fields = ['sam_memory_type', 'importance_score', 'created_at', 'chunk_id']
        
        for key, value in metadata.items():
            if key in sensitive_fields:
                # Encrypt sensitive fields
                try:
                    encrypted_value = self.crypto_manager.encrypt_data(json.dumps(value))
                    encrypted_metadata[f"enc_{key}"] = encrypted_value
                except Exception as e:
                    logger.warning(f"Failed to encrypt metadata field {key}: {e}")
                    encrypted_metadata[key] = value
            elif key in searchable_fields:
                # Keep searchable fields unencrypted
                encrypted_metadata[key] = value
            else:
                # Default: encrypt unknown fields
                try:
                    encrypted_value = self.crypto_manager.encrypt_data(json.dumps(value))
                    encrypted_metadata[f"enc_{key}"] = encrypted_value
                except Exception as e:
                    logger.warning(f"Failed to encrypt metadata field {key}: {e}")
                    encrypted_metadata[key] = value
        
        return encrypted_metadata
    
    def _decrypt_metadata(self, encrypted_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decrypt metadata fields.
        
        Args:
            encrypted_metadata: Metadata with encrypted fields
            
        Returns:
            Decrypted metadata dictionary
        """
        decrypted_metadata = {}
        
        for key, value in encrypted_metadata.items():
            if key.startswith('enc_'):
                # Decrypt encrypted fields
                original_key = key[4:]  # Remove 'enc_' prefix
                try:
                    decrypted_value = self.crypto_manager.decrypt_data(value)
                    decrypted_metadata[original_key] = json.loads(decrypted_value)
                except Exception as e:
                    logger.warning(f"Failed to decrypt metadata field {key}: {e}")
                    decrypted_metadata[original_key] = value
            else:
                # Keep unencrypted fields as-is
                decrypted_metadata[key] = value
        
        return decrypted_metadata
    
    def close(self):
        """Close the encrypted store and cleanup resources."""
        try:
            # ChromaDB client doesn't need explicit closing
            logger.info("Encrypted ChromaDB store closed")
        except Exception as e:
            logger.error(f"Error closing encrypted store: {e}")

# Global encrypted store instance
_encrypted_store_instance = None

def get_encrypted_store() -> Optional[EncryptedChromaStore]:
    """
    Get the global encrypted store instance.

    Returns:
        EncryptedChromaStore instance or None if not initialized
    """
    global _encrypted_store_instance

    if _encrypted_store_instance is None:
        try:
            # Try to initialize with default crypto manager
            from .crypto_utils import CryptoManager
            crypto_manager = CryptoManager()
            _encrypted_store_instance = EncryptedChromaStore(crypto_manager=crypto_manager)
            logger.info("Encrypted store initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize encrypted store: {e}")
            return None

    return _encrypted_store_instance

def initialize_encrypted_store(crypto_manager: CryptoManager = None) -> EncryptedChromaStore:
    """
    Initialize the global encrypted store instance.

    Args:
        crypto_manager: CryptoManager instance for encryption

    Returns:
        EncryptedChromaStore instance
    """
    global _encrypted_store_instance

    if crypto_manager is None:
        from .crypto_utils import CryptoManager
        crypto_manager = CryptoManager()

    _encrypted_store_instance = EncryptedChromaStore(crypto_manager=crypto_manager)
    logger.info("Encrypted store initialized with provided crypto manager")

    return _encrypted_store_instance
