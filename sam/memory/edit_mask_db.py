"""
SAM MEMOIR Edit Mask Database

Implementation of FAISS-based database for storing and retrieving
edit masks with similarity search capabilities.

Based on the MEMOIR paper: "Localized Model Editing for Lifelong Learning"
Author: SAM Development Team
Version: 1.0.0
"""

import torch
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime
from pathlib import Path
import json
import pickle

# Try to import FAISS, fall back to simple implementation if not available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available, using simple similarity search")

logger = logging.getLogger(__name__)

class EditMaskDatabase:
    """
    Database for storing and retrieving MEMOIR edit masks.
    
    This class provides efficient storage and similarity-based retrieval
    of sparse edit masks using FAISS for optimized nearest-neighbor search.
    Falls back to simple implementation if FAISS is not available.
    
    Key Features:
    - Fast similarity search using FAISS
    - Metadata storage for each edit
    - Configurable similarity thresholds
    - Persistence to disk
    - Statistics and monitoring
    """
    
    def __init__(
        self,
        hidden_size: int,
        similarity_metric: str = 'cosine',
        storage_dir: Optional[str] = None,
        max_edits: int = 100000
    ):
        """
        Initialize the EditMaskDatabase.
        
        Args:
            hidden_size: Dimension of the edit masks
            similarity_metric: Similarity metric ('cosine', 'l2', 'inner_product')
            storage_dir: Directory for persistent storage
            max_edits: Maximum number of edits to store
        """
        self.hidden_size = hidden_size
        self.similarity_metric = similarity_metric
        self.storage_dir = Path(storage_dir) if storage_dir else None
        self.max_edits = max_edits
        
        # Initialize FAISS index or fallback
        self.index = self._create_index()
        
        # Metadata storage
        self.edit_metadata = {}  # edit_id -> metadata
        self.edit_id_to_index = {}  # edit_id -> index position
        self.index_to_edit_id = {}  # index position -> edit_id
        
        # Statistics
        self.total_edits = 0
        self.total_searches = 0
        self.cache_hits = 0
        
        # Simple cache for recent searches
        self.search_cache = {}
        self.cache_size = 1000
        
        # Create storage directory
        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()
        
        logger.info(f"EditMaskDatabase initialized: {hidden_size}D, metric={similarity_metric}")
    
    def _create_index(self):
        """Create FAISS index or fallback implementation."""
        if FAISS_AVAILABLE:
            if self.similarity_metric == 'cosine':
                # Normalize vectors for cosine similarity
                index = faiss.IndexFlatIP(self.hidden_size)
            elif self.similarity_metric == 'l2':
                index = faiss.IndexFlatL2(self.hidden_size)
            elif self.similarity_metric == 'inner_product':
                index = faiss.IndexFlatIP(self.hidden_size)
            else:
                raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")
            
            logger.info(f"Created FAISS index: {type(index).__name__}")
            return index
        else:
            # Fallback to simple numpy-based implementation
            logger.info("Using fallback numpy-based similarity search")
            return {
                'vectors': [],
                'metric': self.similarity_metric
            }
    
    def add(
        self, 
        edit_id: str, 
        mask: torch.Tensor, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a new edit mask to the database.
        
        Args:
            edit_id: Unique identifier for the edit
            mask: Edit mask tensor [hidden_size] or [..., hidden_size]
            metadata: Optional metadata about the edit
            
        Returns:
            True if successfully added, False otherwise
        """
        if edit_id in self.edit_metadata:
            logger.warning(f"Edit ID '{edit_id}' already exists, skipping")
            return False
        
        if self.total_edits >= self.max_edits:
            logger.warning(f"Maximum edits ({self.max_edits}) reached")
            return False
        
        # Flatten and normalize the mask
        flat_mask = mask.flatten().detach().cpu().numpy().astype(np.float32)
        
        if len(flat_mask) != self.hidden_size:
            raise ValueError(f"Mask size {len(flat_mask)} doesn't match hidden_size {self.hidden_size}")
        
        # Normalize for cosine similarity
        if self.similarity_metric == 'cosine':
            norm = np.linalg.norm(flat_mask)
            if norm > 0:
                flat_mask = flat_mask / norm
        
        # Add to index
        current_index = self.total_edits
        
        if FAISS_AVAILABLE:
            self.index.add(flat_mask.reshape(1, -1))
        else:
            self.index['vectors'].append(flat_mask)
        
        # Store metadata
        self.edit_metadata[edit_id] = {
            'created_at': datetime.now(),
            'mask_norm': float(np.linalg.norm(flat_mask)),
            'mask_sparsity': float((flat_mask == 0).mean()),
            'metadata': metadata or {}
        }
        
        # Update mappings
        self.edit_id_to_index[edit_id] = current_index
        self.index_to_edit_id[current_index] = edit_id
        
        self.total_edits += 1
        
        logger.debug(f"Added edit '{edit_id}' at index {current_index}")
        return True
    
    def find_closest(
        self, 
        query_mask: torch.Tensor, 
        threshold: float = 0.8,
        top_k: int = 1
    ) -> Optional[Tuple[str, torch.Tensor, float]]:
        """
        Find the closest matching edit mask in the database.
        
        Args:
            query_mask: Query mask tensor
            threshold: Minimum similarity threshold
            top_k: Number of top results to consider
            
        Returns:
            Tuple of (edit_id, original_mask, similarity) if found, None otherwise
        """
        if self.total_edits == 0:
            return None
        
        self.total_searches += 1
        
        # Check cache first
        query_hash = self._hash_mask(query_mask)
        if query_hash in self.search_cache:
            self.cache_hits += 1
            return self.search_cache[query_hash]
        
        # Prepare query vector
        flat_query = query_mask.flatten().detach().cpu().numpy().astype(np.float32)
        
        if len(flat_query) != self.hidden_size:
            raise ValueError(f"Query mask size {len(flat_query)} doesn't match hidden_size {self.hidden_size}")
        
        # Normalize for cosine similarity
        if self.similarity_metric == 'cosine':
            norm = np.linalg.norm(flat_query)
            if norm > 0:
                flat_query = flat_query / norm
        
        # Search for similar masks
        if FAISS_AVAILABLE:
            similarities, indices = self.index.search(flat_query.reshape(1, -1), top_k)
            similarities = similarities[0]
            indices = indices[0]
        else:
            similarities, indices = self._fallback_search(flat_query, top_k)
        
        # Find best match above threshold
        best_match = None
        for i, (similarity, idx) in enumerate(zip(similarities, indices)):
            if idx >= 0 and idx < self.total_edits:
                # Convert similarity based on metric
                if self.similarity_metric == 'l2':
                    # Convert L2 distance to similarity (lower is better)
                    similarity = 1.0 / (1.0 + similarity)
                
                if similarity >= threshold:
                    edit_id = self.index_to_edit_id[idx]
                    
                    # Reconstruct original mask
                    if FAISS_AVAILABLE:
                        original_mask_np = self.index.reconstruct(idx)
                    else:
                        original_mask_np = self.index['vectors'][idx]
                    
                    original_mask = torch.from_numpy(original_mask_np).to(query_mask.device)
                    
                    best_match = (edit_id, original_mask, float(similarity))
                    break
        
        # Cache the result
        self._update_search_cache(query_hash, best_match)
        
        return best_match
    
    def _fallback_search(self, query_vector: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fallback similarity search using numpy.
        
        Args:
            query_vector: Query vector
            top_k: Number of top results
            
        Returns:
            Tuple of (similarities, indices)
        """
        if not self.index['vectors']:
            return np.array([]), np.array([])
        
        vectors = np.array(self.index['vectors'])
        
        if self.similarity_metric == 'cosine' or self.similarity_metric == 'inner_product':
            # Compute dot product (cosine similarity for normalized vectors)
            similarities = np.dot(vectors, query_vector)
            # Sort in descending order (higher is better)
            sorted_indices = np.argsort(similarities)[::-1]
        elif self.similarity_metric == 'l2':
            # Compute L2 distances
            distances = np.linalg.norm(vectors - query_vector, axis=1)
            # Sort in ascending order (lower is better)
            sorted_indices = np.argsort(distances)
            similarities = distances
        else:
            raise ValueError(f"Unsupported metric for fallback: {self.similarity_metric}")
        
        # Return top-k results
        top_k = min(top_k, len(sorted_indices))
        top_indices = sorted_indices[:top_k]
        top_similarities = similarities[top_indices]
        
        return top_similarities, top_indices
    
    def _hash_mask(self, mask: torch.Tensor) -> str:
        """Generate hash for mask caching."""
        mask_bytes = mask.flatten().detach().cpu().numpy().tobytes()
        import hashlib
        return hashlib.md5(mask_bytes).hexdigest()
    
    def _update_search_cache(self, query_hash: str, result):
        """Update the search cache."""
        if len(self.search_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.search_cache))
            del self.search_cache[oldest_key]
        
        self.search_cache[query_hash] = result
    
    def remove(self, edit_id: str) -> bool:
        """
        Remove an edit from the database.
        
        Args:
            edit_id: ID of the edit to remove
            
        Returns:
            True if successfully removed, False if not found
        """
        if edit_id not in self.edit_metadata:
            return False
        
        # Note: FAISS doesn't support efficient removal, so we mark as deleted
        # In a production system, you might want to rebuild the index periodically
        
        # Remove from metadata
        del self.edit_metadata[edit_id]
        
        # Remove from mappings
        if edit_id in self.edit_id_to_index:
            index_pos = self.edit_id_to_index[edit_id]
            del self.edit_id_to_index[edit_id]
            del self.index_to_edit_id[index_pos]
        
        # Clear cache
        self.search_cache.clear()
        
        logger.info(f"Removed edit '{edit_id}' (marked as deleted)")
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        cache_hit_ratio = self.cache_hits / max(self.total_searches, 1)
        
        return {
            'total_edits': self.total_edits,
            'total_searches': self.total_searches,
            'cache_hits': self.cache_hits,
            'cache_hit_ratio': cache_hit_ratio,
            'active_edits': len(self.edit_metadata),
            'hidden_size': self.hidden_size,
            'similarity_metric': self.similarity_metric,
            'faiss_available': FAISS_AVAILABLE,
            'storage_dir': str(self.storage_dir) if self.storage_dir else None
        }
    
    def save_to_disk(self) -> bool:
        """Save database to disk."""
        if not self.storage_dir:
            logger.warning("No storage directory specified")
            return False
        
        try:
            # Save metadata
            metadata_file = self.storage_dir / 'edit_metadata.json'
            with open(metadata_file, 'w') as f:
                # Convert datetime objects to strings for JSON serialization
                serializable_metadata = {}
                for edit_id, meta in self.edit_metadata.items():
                    serializable_meta = meta.copy()
                    serializable_meta['created_at'] = meta['created_at'].isoformat()
                    serializable_metadata[edit_id] = serializable_meta
                
                json.dump({
                    'edit_metadata': serializable_metadata,
                    'edit_id_to_index': self.edit_id_to_index,
                    'index_to_edit_id': self.index_to_edit_id,
                    'total_edits': self.total_edits,
                    'hidden_size': self.hidden_size,
                    'similarity_metric': self.similarity_metric
                }, f, indent=2)
            
            # Save FAISS index or fallback vectors
            if FAISS_AVAILABLE:
                index_file = self.storage_dir / 'faiss_index.bin'
                faiss.write_index(self.index, str(index_file))
            else:
                vectors_file = self.storage_dir / 'vectors.pkl'
                with open(vectors_file, 'wb') as f:
                    pickle.dump(self.index, f)
            
            logger.info(f"Database saved to {self.storage_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save database: {e}")
            return False
    
    def _load_from_disk(self) -> bool:
        """Load database from disk."""
        if not self.storage_dir or not self.storage_dir.exists():
            return False
        
        try:
            # Load metadata
            metadata_file = self.storage_dir / 'edit_metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                
                # Restore metadata with datetime conversion
                self.edit_metadata = {}
                for edit_id, meta in data['edit_metadata'].items():
                    restored_meta = meta.copy()
                    restored_meta['created_at'] = datetime.fromisoformat(meta['created_at'])
                    self.edit_metadata[edit_id] = restored_meta
                
                self.edit_id_to_index = data['edit_id_to_index']
                self.index_to_edit_id = {int(k): v for k, v in data['index_to_edit_id'].items()}
                self.total_edits = data['total_edits']
            
            # Load FAISS index or fallback vectors
            if FAISS_AVAILABLE:
                index_file = self.storage_dir / 'faiss_index.bin'
                if index_file.exists():
                    self.index = faiss.read_index(str(index_file))
            else:
                vectors_file = self.storage_dir / 'vectors.pkl'
                if vectors_file.exists():
                    with open(vectors_file, 'rb') as f:
                        self.index = pickle.load(f)
            
            logger.info(f"Database loaded from {self.storage_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load database: {e}")
            return False
