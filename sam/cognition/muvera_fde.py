#!/usr/bin/env python3
"""
MUVERA Fixed Dimensional Encoding (FDE) Implementation
Based on Section 2 of the MUVERA paper for efficient multi-vector retrieval.
"""

import numpy as np
import logging
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Global FDE instance
_muvera_fde = None

@dataclass
class FDEResult:
    """Result from FDE transformation."""
    fde_vector: np.ndarray        # Fixed dimensional encoding
    original_shape: Tuple[int, int]  # Original token embeddings shape
    num_tokens: int               # Number of input tokens
    embedding_dim: int            # Original embedding dimension
    fde_dim: int                  # FDE output dimension
    compression_ratio: float     # Compression achieved
    processing_time: float       # Time taken for transformation
    hash_functions_used: int     # Number of hash functions
    projection_matrices: int     # Number of projection matrices
    metadata: Dict[str, Any]      # Additional metadata

class MuveraFDE:
    """
    MUVERA Fixed Dimensional Encoding transformer.
    
    Implements the randomized hashing and projection method from the MUVERA paper
    to convert variable-length multi-vector embeddings into fixed-dimensional vectors
    suitable for efficient similarity search.
    """
    
    def __init__(self,
                 fde_dim: int = 768,
                 num_hash_functions: int = 8,
                 num_projections: int = 4,
                 aggregation_method: str = "max_pool",
                 random_seed: int = 42):
        """
        Initialize the MUVERA FDE transformer.
        
        Args:
            fde_dim: Output dimension for FDE vectors
            num_hash_functions: Number of hash functions for randomized hashing
            num_projections: Number of random projection matrices
            aggregation_method: Method to aggregate multi-vector representations
            random_seed: Random seed for reproducibility
        """
        self.fde_dim = fde_dim
        self.num_hash_functions = num_hash_functions
        self.num_projections = num_projections
        self.aggregation_method = aggregation_method
        self.random_seed = random_seed
        
        # Initialize random number generator
        self.rng = np.random.RandomState(random_seed)
        
        # Pre-computed projection matrices and hash functions
        self.projection_matrices = []
        self.hash_functions = []
        self.is_initialized = False
        
        logger.info(f"🧠 MuveraFDE initialized: {fde_dim}D output, "
                   f"{num_hash_functions} hash functions, {num_projections} projections")
    
    def _initialize_transforms(self, input_dim: int):
        """Initialize projection matrices and hash functions."""
        if self.is_initialized:
            return
        
        logger.info(f"🔄 Initializing FDE transforms for {input_dim}D input")
        
        # Create random projection matrices
        for i in range(self.num_projections):
            # Random Gaussian projection matrix
            proj_matrix = self.rng.normal(
                0, 1/np.sqrt(input_dim), 
                size=(input_dim, self.fde_dim // self.num_projections)
            )
            self.projection_matrices.append(proj_matrix)
        
        # Create hash functions (using different random seeds)
        for i in range(self.num_hash_functions):
            hash_seed = self.random_seed + i * 1000
            self.hash_functions.append(hash_seed)
        
        self.is_initialized = True
        logger.info(f"✅ FDE transforms initialized")
    
    def _hash_vector(self, vector: np.ndarray, hash_seed: int, num_buckets: int) -> int:
        """Hash a vector to a bucket using LSH-style hashing."""
        # Create deterministic hash based on vector content and seed
        vector_bytes = vector.astype(np.float32).tobytes()
        hash_input = f"{hash_seed}_{vector_bytes}".encode()
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
        return hash_value % num_buckets
    
    def _apply_randomized_hashing(self, token_embeddings: np.ndarray) -> np.ndarray:
        """Apply randomized hashing to group similar tokens."""
        num_tokens, embedding_dim = token_embeddings.shape
        num_buckets = max(1, num_tokens // 4)  # Adaptive bucket count
        
        # Initialize hash-based aggregation
        hash_aggregated = np.zeros((self.num_hash_functions, embedding_dim))
        
        for i, hash_seed in enumerate(self.hash_functions):
            # Create buckets for this hash function
            buckets = {}
            
            # Hash each token to a bucket
            for token_idx, token_embedding in enumerate(token_embeddings):
                bucket = self._hash_vector(token_embedding, hash_seed, num_buckets)
                
                if bucket not in buckets:
                    buckets[bucket] = []
                buckets[bucket].append(token_embedding)
            
            # Aggregate within each bucket and then across buckets
            bucket_aggregates = []
            for bucket_tokens in buckets.values():
                if self.aggregation_method == "max_pool":
                    bucket_agg = np.max(bucket_tokens, axis=0)
                elif self.aggregation_method == "mean_pool":
                    bucket_agg = np.mean(bucket_tokens, axis=0)
                elif self.aggregation_method == "sum_pool":
                    bucket_agg = np.sum(bucket_tokens, axis=0)
                else:
                    bucket_agg = np.max(bucket_tokens, axis=0)  # Default to max
                
                bucket_aggregates.append(bucket_agg)
            
            # Final aggregation across buckets for this hash function
            if bucket_aggregates:
                if self.aggregation_method == "max_pool":
                    hash_aggregated[i] = np.max(bucket_aggregates, axis=0)
                elif self.aggregation_method == "mean_pool":
                    hash_aggregated[i] = np.mean(bucket_aggregates, axis=0)
                else:
                    hash_aggregated[i] = np.max(bucket_aggregates, axis=0)
        
        return hash_aggregated
    
    def _apply_random_projections(self, hash_aggregated: np.ndarray) -> np.ndarray:
        """Apply random projections to create fixed-dimensional encoding."""
        projected_parts = []
        
        for i, proj_matrix in enumerate(self.projection_matrices):
            # Use different hash functions for different projections
            hash_idx = i % len(self.hash_functions)
            input_vector = hash_aggregated[hash_idx]
            
            # Apply projection
            projected = np.dot(input_vector, proj_matrix)
            projected_parts.append(projected)
        
        # Concatenate all projections
        fde_vector = np.concatenate(projected_parts)
        
        # Ensure exact output dimension
        if len(fde_vector) > self.fde_dim:
            fde_vector = fde_vector[:self.fde_dim]
        elif len(fde_vector) < self.fde_dim:
            # Pad with zeros if needed
            padding = np.zeros(self.fde_dim - len(fde_vector))
            fde_vector = np.concatenate([fde_vector, padding])
        
        return fde_vector
    
    def generate_fde(self, token_embeddings: np.ndarray, doc_id: Optional[str] = None) -> Optional[FDEResult]:
        """
        Generate Fixed Dimensional Encoding from multi-vector token embeddings.
        
        Args:
            token_embeddings: Token embeddings array (num_tokens, embedding_dim)
            doc_id: Optional document identifier
            
        Returns:
            FDEResult with the fixed dimensional encoding
        """
        try:
            import time
            start_time = time.time()
            
            if len(token_embeddings.shape) != 2:
                logger.error(f"❌ Invalid token embeddings shape: {token_embeddings.shape}")
                return None
            
            num_tokens, embedding_dim = token_embeddings.shape
            
            if num_tokens == 0:
                logger.error("❌ No tokens to process")
                return None
            
            logger.debug(f"🔄 Generating FDE for {num_tokens} tokens, {embedding_dim}D")
            
            # Initialize transforms if needed
            self._initialize_transforms(embedding_dim)
            
            # Step 1: Apply randomized hashing
            hash_aggregated = self._apply_randomized_hashing(token_embeddings)
            
            # Step 2: Apply random projections
            fde_vector = self._apply_random_projections(hash_aggregated)
            
            # Normalize the FDE vector
            fde_norm = np.linalg.norm(fde_vector)
            if fde_norm > 0:
                fde_vector = fde_vector / fde_norm
            
            processing_time = time.time() - start_time
            
            # Calculate compression ratio
            original_size = num_tokens * embedding_dim
            compressed_size = self.fde_dim
            compression_ratio = original_size / compressed_size
            
            result = FDEResult(
                fde_vector=fde_vector,
                original_shape=(num_tokens, embedding_dim),
                num_tokens=num_tokens,
                embedding_dim=embedding_dim,
                fde_dim=self.fde_dim,
                compression_ratio=compression_ratio,
                processing_time=processing_time,
                hash_functions_used=self.num_hash_functions,
                projection_matrices=self.num_projections,
                metadata={
                    'doc_id': doc_id,
                    'aggregation_method': self.aggregation_method,
                    'random_seed': self.random_seed,
                    'normalized': True
                }
            )
            
            logger.debug(f"✅ FDE generated: {self.fde_dim}D, "
                        f"compression {compression_ratio:.1f}x, {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to generate FDE: {e}")
            return None
    
    def get_config(self) -> Dict[str, Any]:
        """Get FDE configuration."""
        return {
            'fde_dim': self.fde_dim,
            'num_hash_functions': self.num_hash_functions,
            'num_projections': self.num_projections,
            'aggregation_method': self.aggregation_method,
            'random_seed': self.random_seed,
            'is_initialized': self.is_initialized
        }

def get_muvera_fde(fde_dim: int = 768,
                   num_hash_functions: int = 8,
                   num_projections: int = 4,
                   aggregation_method: str = "max_pool",
                   random_seed: int = 42) -> MuveraFDE:
    """
    Get or create a MUVERA FDE instance.
    
    Args:
        fde_dim: Output dimension for FDE vectors
        num_hash_functions: Number of hash functions
        num_projections: Number of projection matrices
        aggregation_method: Aggregation method
        random_seed: Random seed
        
    Returns:
        MuveraFDE instance
    """
    global _muvera_fde
    
    if _muvera_fde is None:
        _muvera_fde = MuveraFDE(
            fde_dim=fde_dim,
            num_hash_functions=num_hash_functions,
            num_projections=num_projections,
            aggregation_method=aggregation_method,
            random_seed=random_seed
        )
    
    return _muvera_fde

def generate_fde(token_embeddings: np.ndarray, doc_id: Optional[str] = None) -> Optional[FDEResult]:
    """
    Convenience function to generate FDE using default settings.
    
    Args:
        token_embeddings: Token embeddings array
        doc_id: Optional document identifier
        
    Returns:
        FDEResult with the fixed dimensional encoding
    """
    fde_transformer = get_muvera_fde()
    return fde_transformer.generate_fde(token_embeddings, doc_id)
