#!/usr/bin/env python3
"""
V2 Storage Schema for SAM MUVERA Retrieval Pipeline
Manages storage of multi-vector embeddings and FDE vectors for efficient retrieval.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# Global storage manager instance
_v2_storage_manager = None

@dataclass
class V2DocumentRecord:
    """Record for a document in the v2 storage system."""
    document_id: str              # Unique document identifier
    filename: str                 # Original filename
    file_path: str               # Path to original file
    text_content: str            # Extracted text content
    fde_vector: np.ndarray       # Fixed dimensional encoding vector
    token_embeddings_path: str   # Path to stored token embeddings
    num_tokens: int              # Number of tokens
    embedding_dim: int           # Original embedding dimension
    fde_dim: int                 # FDE vector dimension
    processing_timestamp: str    # When document was processed
    file_size: int               # Original file size in bytes
    content_hash: str            # Hash of content for deduplication
    metadata: Dict[str, Any]     # Additional metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        # Convert numpy array to list for JSON serialization
        data['fde_vector'] = self.fde_vector.tolist()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'V2DocumentRecord':
        """Create from dictionary."""
        # Convert list back to numpy array
        data['fde_vector'] = np.array(data['fde_vector'])
        return cls(**data)

class V2StorageManager:
    """
    Storage manager for SAM's v2 retrieval pipeline.
    
    Manages:
    - FDE vectors in ChromaDB for fast similarity search
    - Full token embeddings in memory-mapped files for accurate reranking
    - Document metadata and indexing
    """
    
    def __init__(self, 
                 storage_root: str = "uploads",
                 chroma_db_path: str = "chroma_db_v2",
                 collection_name: str = "v2_documents"):
        """
        Initialize the v2 storage manager.
        
        Args:
            storage_root: Root directory for file storage
            chroma_db_path: Path to ChromaDB database
            collection_name: Name of the ChromaDB collection
        """
        self.storage_root = Path(storage_root)
        self.chroma_db_path = Path(chroma_db_path)
        self.collection_name = collection_name
        
        # Create directories
        self.storage_root.mkdir(exist_ok=True)
        self.chroma_db_path.mkdir(exist_ok=True)
        
        # ChromaDB components
        self.chroma_client = None
        self.collection = None
        self.is_initialized = False
        
        logger.info(f"📁 V2StorageManager initialized: {storage_root}")
        logger.info(f"🗄️  ChromaDB path: {chroma_db_path}")
    
    def _initialize_chromadb(self) -> bool:
        """Initialize ChromaDB client and collection."""
        if self.is_initialized:
            return True
        
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Create ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.chroma_db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection(
                    name=self.collection_name
                )
                logger.info(f"✅ Connected to existing collection: {self.collection_name}")
            except Exception:
                # Create new collection
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "SAM v2 documents with FDE vectors"}
                )
                logger.info(f"✅ Created new collection: {self.collection_name}")
            
            self.is_initialized = True
            return True
            
        except ImportError:
            logger.error("❌ ChromaDB not available. Install with: pip install chromadb")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to initialize ChromaDB: {e}")
            return False
    
    def _get_document_storage_path(self, document_id: str) -> Path:
        """Get storage path for a document."""
        return self.storage_root / document_id
    
    def _save_token_embeddings(self, 
                              document_id: str, 
                              token_embeddings: np.ndarray) -> str:
        """Save token embeddings to memory-mapped file."""
        doc_path = self._get_document_storage_path(document_id)
        doc_path.mkdir(exist_ok=True)
        
        embeddings_path = doc_path / "token_embeddings.npy"
        
        # Save as memory-mapped array for efficient loading
        np.save(embeddings_path, token_embeddings)
        
        logger.debug(f"💾 Token embeddings saved: {embeddings_path}")
        return str(embeddings_path)
    
    def _load_token_embeddings(self, embeddings_path: str) -> Optional[np.ndarray]:
        """Load token embeddings from file."""
        try:
            embeddings_path = Path(embeddings_path)
            if not embeddings_path.exists():
                logger.error(f"❌ Token embeddings file not found: {embeddings_path}")
                return None
            
            # Load as memory-mapped array for efficiency
            embeddings = np.load(embeddings_path, mmap_mode='r')
            logger.debug(f"📂 Token embeddings loaded: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Failed to load token embeddings: {e}")
            return None
    
    def store_document(self, 
                      document_id: str,
                      filename: str,
                      file_path: str,
                      text_content: str,
                      token_embeddings: np.ndarray,
                      fde_vector: np.ndarray,
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store a document in the v2 storage system.
        
        Args:
            document_id: Unique document identifier
            filename: Original filename
            file_path: Path to original file
            text_content: Extracted text content
            token_embeddings: Full token embeddings array
            fde_vector: Fixed dimensional encoding vector
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self._initialize_chromadb():
                return False
            
            logger.info(f"💾 Storing document: {document_id}")
            
            # Save token embeddings to file
            embeddings_path = self._save_token_embeddings(document_id, token_embeddings)
            
            # Create document record
            import hashlib
            content_hash = hashlib.md5(text_content.encode()).hexdigest()
            
            record = V2DocumentRecord(
                document_id=document_id,
                filename=filename,
                file_path=file_path,
                text_content=text_content,
                fde_vector=fde_vector,
                token_embeddings_path=embeddings_path,
                num_tokens=token_embeddings.shape[0],
                embedding_dim=token_embeddings.shape[1],
                fde_dim=len(fde_vector),
                processing_timestamp=datetime.now().isoformat(),
                file_size=os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                content_hash=content_hash,
                metadata=metadata or {}
            )
            
            # Save document metadata
            doc_path = self._get_document_storage_path(document_id)
            metadata_path = doc_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(record.to_dict(), f, indent=2)
            
            # Store FDE vector in ChromaDB
            self.collection.add(
                embeddings=[fde_vector.tolist()],
                documents=[text_content[:1000]],  # Store first 1000 chars as document
                metadatas=[{
                    'document_id': document_id,
                    'filename': filename,
                    'num_tokens': record.num_tokens,
                    'embedding_dim': record.embedding_dim,
                    'processing_timestamp': record.processing_timestamp,
                    'content_hash': content_hash
                }],
                ids=[document_id]
            )
            
            logger.info(f"✅ Document stored successfully: {document_id}")
            logger.info(f"📊 Tokens: {record.num_tokens}, FDE dim: {record.fde_dim}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store document {document_id}: {e}")
            return False
    
    def retrieve_document(self, document_id: str) -> Optional[V2DocumentRecord]:
        """Retrieve a document record by ID."""
        try:
            doc_path = self._get_document_storage_path(document_id)
            metadata_path = doc_path / "metadata.json"
            
            if not metadata_path.exists():
                logger.error(f"❌ Document metadata not found: {document_id}")
                return None
            
            with open(metadata_path, 'r') as f:
                data = json.load(f)
            
            record = V2DocumentRecord.from_dict(data)
            logger.debug(f"📂 Document retrieved: {document_id}")
            
            return record
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve document {document_id}: {e}")
            return None
    
    def load_token_embeddings(self, document_id: str) -> Optional[np.ndarray]:
        """Load token embeddings for a document."""
        record = self.retrieve_document(document_id)
        if not record:
            return None
        
        return self._load_token_embeddings(record.token_embeddings_path)
    
    def search_by_fde(self, 
                     query_fde: np.ndarray, 
                     top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search documents by FDE vector similarity.
        
        Args:
            query_fde: Query FDE vector
            top_k: Number of top results to return
            
        Returns:
            List of (document_id, similarity_score) tuples
        """
        try:
            if not self._initialize_chromadb():
                return []
            
            logger.debug(f"🔍 Searching by FDE: top_k={top_k}")
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_fde.tolist()],
                n_results=top_k
            )
            
            # Extract document IDs and distances
            document_ids = results['ids'][0]
            distances = results['distances'][0]
            
            # Convert distances to similarities (assuming cosine distance)
            similarities = [1 - dist for dist in distances]
            
            search_results = list(zip(document_ids, similarities))
            
            logger.debug(f"📄 FDE search found {len(search_results)} results")
            
            return search_results
            
        except Exception as e:
            logger.error(f"❌ FDE search failed: {e}")
            return []
    
    def list_documents(self) -> List[str]:
        """List all stored document IDs."""
        try:
            if not self._initialize_chromadb():
                return []
            
            # Get all document IDs from ChromaDB
            results = self.collection.get()
            document_ids = results['ids']
            
            logger.debug(f"📋 Found {len(document_ids)} documents")
            
            return document_ids
            
        except Exception as e:
            logger.error(f"❌ Failed to list documents: {e}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document from storage."""
        try:
            if not self._initialize_chromadb():
                return False
            
            logger.info(f"🗑️  Deleting document: {document_id}")
            
            # Remove from ChromaDB
            self.collection.delete(ids=[document_id])
            
            # Remove files
            doc_path = self._get_document_storage_path(document_id)
            if doc_path.exists():
                import shutil
                shutil.rmtree(doc_path)
            
            logger.info(f"✅ Document deleted: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete document {document_id}: {e}")
            return False
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            document_ids = self.list_documents()
            total_docs = len(document_ids)
            
            # Calculate storage usage
            total_size = 0
            for doc_id in document_ids:
                doc_path = self._get_document_storage_path(doc_id)
                if doc_path.exists():
                    for file_path in doc_path.rglob('*'):
                        if file_path.is_file():
                            total_size += file_path.stat().st_size
            
            return {
                'total_documents': total_docs,
                'storage_size_bytes': total_size,
                'storage_size_mb': total_size / (1024 * 1024),
                'storage_root': str(self.storage_root),
                'chroma_db_path': str(self.chroma_db_path),
                'collection_name': self.collection_name,
                'is_initialized': self.is_initialized
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get storage stats: {e}")
            return {'error': str(e)}

def get_v2_storage_manager(storage_root: str = "uploads",
                          chroma_db_path: str = "chroma_db_v2",
                          collection_name: str = "v2_documents") -> V2StorageManager:
    """
    Get or create a v2 storage manager instance.
    
    Args:
        storage_root: Root directory for file storage
        chroma_db_path: Path to ChromaDB database
        collection_name: Name of the ChromaDB collection
        
    Returns:
        V2StorageManager instance
    """
    global _v2_storage_manager
    
    if _v2_storage_manager is None:
        _v2_storage_manager = V2StorageManager(
            storage_root=storage_root,
            chroma_db_path=chroma_db_path,
            collection_name=collection_name
        )
    
    return _v2_storage_manager

def create_v2_collections(chroma_db_path: str = "chroma_db_v2") -> bool:
    """
    Create v2 ChromaDB collections.
    
    Args:
        chroma_db_path: Path to ChromaDB database
        
    Returns:
        True if successful, False otherwise
    """
    try:
        storage_manager = get_v2_storage_manager(chroma_db_path=chroma_db_path)
        return storage_manager._initialize_chromadb()
    except Exception as e:
        logger.error(f"❌ Failed to create v2 collections: {e}")
        return False
