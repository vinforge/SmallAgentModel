#!/usr/bin/env python3
"""
Embedding Utilities for SAM v2 Retrieval Pipeline
Shared utilities for embedding validation, normalization, and analysis.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """Configuration for embedding operations."""
    model_name: str = "colbert-ir/colbertv2.0"
    max_length: int = 512
    embedding_dim: int = 128
    device: str = "auto"
    cache_dir: Optional[str] = None
    normalize_embeddings: bool = True
    use_attention_mask: bool = True
    batch_size: int = 32

def validate_embeddings(embeddings: np.ndarray, 
                       expected_dim: Optional[int] = None,
                       min_tokens: int = 1,
                       max_tokens: int = 512) -> Tuple[bool, str]:
    """
    Validate embedding array format and dimensions.
    
    Args:
        embeddings: Token embeddings array
        expected_dim: Expected embedding dimension
        min_tokens: Minimum number of tokens
        max_tokens: Maximum number of tokens
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check if embeddings is a numpy array
        if not isinstance(embeddings, np.ndarray):
            return False, f"Embeddings must be numpy array, got {type(embeddings)}"
        
        # Check dimensions
        if len(embeddings.shape) != 2:
            return False, f"Embeddings must be 2D array (tokens, dim), got shape {embeddings.shape}"
        
        num_tokens, embedding_dim = embeddings.shape
        
        # Check token count
        if num_tokens < min_tokens:
            return False, f"Too few tokens: {num_tokens} < {min_tokens}"
        
        if num_tokens > max_tokens:
            return False, f"Too many tokens: {num_tokens} > {max_tokens}"
        
        # Check embedding dimension
        if expected_dim is not None and embedding_dim != expected_dim:
            return False, f"Wrong embedding dimension: {embedding_dim} != {expected_dim}"
        
        # Check for NaN or infinite values
        if np.isnan(embeddings).any():
            return False, "Embeddings contain NaN values"
        
        if np.isinf(embeddings).any():
            return False, "Embeddings contain infinite values"
        
        # Check for zero vectors (might indicate issues)
        zero_vectors = np.all(embeddings == 0, axis=1).sum()
        if zero_vectors > 0:
            logger.warning(f"⚠️  Found {zero_vectors} zero vectors in embeddings")
        
        return True, "Valid embeddings"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def normalize_embeddings(embeddings: np.ndarray, 
                        method: str = "l2",
                        epsilon: float = 1e-8) -> np.ndarray:
    """
    Normalize embeddings using specified method.
    
    Args:
        embeddings: Token embeddings array
        method: Normalization method ('l2', 'l1', 'max', 'none')
        epsilon: Small value to prevent division by zero
        
    Returns:
        Normalized embeddings
    """
    try:
        if method == "none":
            return embeddings
        
        if method == "l2":
            # L2 normalization (unit vectors)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, epsilon)  # Prevent division by zero
            return embeddings / norms
        
        elif method == "l1":
            # L1 normalization
            norms = np.sum(np.abs(embeddings), axis=1, keepdims=True)
            norms = np.maximum(norms, epsilon)
            return embeddings / norms
        
        elif method == "max":
            # Max normalization
            max_vals = np.max(np.abs(embeddings), axis=1, keepdims=True)
            max_vals = np.maximum(max_vals, epsilon)
            return embeddings / max_vals
        
        else:
            logger.warning(f"⚠️  Unknown normalization method: {method}, using L2")
            return normalize_embeddings(embeddings, method="l2", epsilon=epsilon)
            
    except Exception as e:
        logger.error(f"❌ Normalization failed: {e}")
        return embeddings

def compute_embedding_stats(embeddings: np.ndarray) -> Dict[str, Any]:
    """
    Compute statistics for embedding analysis.
    
    Args:
        embeddings: Token embeddings array
        
    Returns:
        Dictionary with embedding statistics
    """
    try:
        num_tokens, embedding_dim = embeddings.shape
        
        # Basic statistics
        mean_vals = np.mean(embeddings, axis=0)
        std_vals = np.std(embeddings, axis=0)
        min_vals = np.min(embeddings, axis=0)
        max_vals = np.max(embeddings, axis=0)
        
        # Per-token statistics
        token_norms = np.linalg.norm(embeddings, axis=1)
        token_means = np.mean(embeddings, axis=1)
        
        # Global statistics
        global_mean = np.mean(embeddings)
        global_std = np.std(embeddings)
        global_min = np.min(embeddings)
        global_max = np.max(embeddings)
        
        # Sparsity analysis
        zero_count = np.sum(embeddings == 0)
        sparsity = zero_count / (num_tokens * embedding_dim)
        
        # Dimension analysis
        dim_variances = np.var(embeddings, axis=0)
        effective_dims = np.sum(dim_variances > 1e-6)  # Dimensions with meaningful variance
        
        stats = {
            'shape': {
                'num_tokens': num_tokens,
                'embedding_dim': embedding_dim,
                'total_elements': num_tokens * embedding_dim
            },
            'global_stats': {
                'mean': float(global_mean),
                'std': float(global_std),
                'min': float(global_min),
                'max': float(global_max),
                'sparsity': float(sparsity)
            },
            'dimension_stats': {
                'mean_per_dim': mean_vals.tolist(),
                'std_per_dim': std_vals.tolist(),
                'min_per_dim': min_vals.tolist(),
                'max_per_dim': max_vals.tolist(),
                'variances': dim_variances.tolist(),
                'effective_dimensions': int(effective_dims)
            },
            'token_stats': {
                'norms': token_norms.tolist(),
                'means': token_means.tolist(),
                'avg_norm': float(np.mean(token_norms)),
                'std_norm': float(np.std(token_norms)),
                'min_norm': float(np.min(token_norms)),
                'max_norm': float(np.max(token_norms))
            },
            'quality_indicators': {
                'zero_vectors': int(np.sum(token_norms < 1e-6)),
                'norm_consistency': float(np.std(token_norms) / np.mean(token_norms)),
                'dimension_utilization': float(effective_dims / embedding_dim)
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"❌ Failed to compute embedding stats: {e}")
        return {'error': str(e)}

def save_embeddings(embeddings: np.ndarray, 
                   filepath: str,
                   metadata: Optional[Dict[str, Any]] = None,
                   compress: bool = True) -> bool:
    """
    Save embeddings to disk with optional metadata.
    
    Args:
        embeddings: Token embeddings array
        filepath: Path to save embeddings
        metadata: Optional metadata to save
        compress: Whether to compress the file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if compress:
            # Save compressed
            np.savez_compressed(
                filepath.with_suffix('.npz'),
                embeddings=embeddings,
                metadata=metadata or {}
            )
        else:
            # Save uncompressed
            np.savez(
                filepath.with_suffix('.npz'),
                embeddings=embeddings,
                metadata=metadata or {}
            )
        
        logger.debug(f"✅ Embeddings saved to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to save embeddings: {e}")
        return False

def load_embeddings(filepath: str) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Load embeddings from disk.
    
    Args:
        filepath: Path to load embeddings from
        
    Returns:
        Tuple of (embeddings, metadata) or (None, None) if failed
    """
    try:
        filepath = Path(filepath)
        
        # Try different extensions
        for ext in ['.npz', '.npy']:
            full_path = filepath.with_suffix(ext)
            if full_path.exists():
                if ext == '.npz':
                    data = np.load(full_path, allow_pickle=True)
                    embeddings = data['embeddings']
                    metadata = data.get('metadata', {}).item() if 'metadata' in data else {}
                    return embeddings, metadata
                else:
                    embeddings = np.load(full_path)
                    return embeddings, {}
        
        logger.error(f"❌ Embeddings file not found: {filepath}")
        return None, None
        
    except Exception as e:
        logger.error(f"❌ Failed to load embeddings: {e}")
        return None, None

def compute_similarity_matrix(embeddings1: np.ndarray, 
                            embeddings2: np.ndarray,
                            metric: str = "cosine") -> np.ndarray:
    """
    Compute similarity matrix between two sets of embeddings.
    
    Args:
        embeddings1: First set of embeddings (N1, D)
        embeddings2: Second set of embeddings (N2, D)
        metric: Similarity metric ('cosine', 'dot', 'euclidean')
        
    Returns:
        Similarity matrix (N1, N2)
    """
    try:
        if metric == "cosine":
            # Normalize embeddings for cosine similarity
            norm1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
            norm2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
            return np.dot(norm1, norm2.T)
        
        elif metric == "dot":
            # Dot product similarity
            return np.dot(embeddings1, embeddings2.T)
        
        elif metric == "euclidean":
            # Negative euclidean distance (higher = more similar)
            from scipy.spatial.distance import cdist
            distances = cdist(embeddings1, embeddings2, metric='euclidean')
            return -distances
        
        else:
            logger.warning(f"⚠️  Unknown metric: {metric}, using cosine")
            return compute_similarity_matrix(embeddings1, embeddings2, metric="cosine")
            
    except Exception as e:
        logger.error(f"❌ Failed to compute similarity matrix: {e}")
        return np.zeros((embeddings1.shape[0], embeddings2.shape[0]))
