#!/usr/bin/env python3
"""
Similarity Metrics for MUVERA v2 Retrieval Pipeline
Implements Chamfer Distance and MaxSim scoring for accurate document reranking.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SimilarityResult:
    """Result from similarity computation."""
    score: float                  # Similarity score
    method: str                   # Similarity method used
    query_tokens: int            # Number of query tokens
    doc_tokens: int              # Number of document tokens
    processing_time: float       # Time taken for computation
    detailed_scores: Optional[List[float]]  # Per-token scores if available
    metadata: Dict[str, Any]     # Additional metadata

class ChamferSimilarity:
    """
    Chamfer Distance-based similarity for multi-vector embeddings.
    
    Computes bidirectional similarity between query and document token embeddings,
    providing robust matching for variable-length sequences.
    """
    
    def __init__(self, 
                 distance_metric: str = "cosine",
                 bidirectional: bool = True,
                 normalize_by_length: bool = True):
        """
        Initialize Chamfer similarity calculator.
        
        Args:
            distance_metric: Distance metric ('cosine', 'euclidean', 'dot')
            bidirectional: Whether to compute bidirectional Chamfer distance
            normalize_by_length: Whether to normalize by sequence length
        """
        self.distance_metric = distance_metric
        self.bidirectional = bidirectional
        self.normalize_by_length = normalize_by_length
        
        logger.info(f"🔍 ChamferSimilarity initialized: {distance_metric}, "
                   f"bidirectional={bidirectional}, normalize={normalize_by_length}")
    
    def _compute_distance_matrix(self, 
                                embeddings1: np.ndarray, 
                                embeddings2: np.ndarray) -> np.ndarray:
        """Compute pairwise distance matrix between two sets of embeddings."""
        if self.distance_metric == "cosine":
            # Normalize for cosine similarity
            norm1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
            norm2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
            # Cosine similarity matrix
            sim_matrix = np.dot(norm1, norm2.T)
            # Convert to distance (1 - similarity)
            return 1 - sim_matrix
        
        elif self.distance_metric == "euclidean":
            # Euclidean distance matrix
            from scipy.spatial.distance import cdist
            return cdist(embeddings1, embeddings2, metric='euclidean')
        
        elif self.distance_metric == "dot":
            # Negative dot product (higher dot product = lower distance)
            return -np.dot(embeddings1, embeddings2.T)
        
        else:
            logger.warning(f"⚠️  Unknown distance metric: {self.distance_metric}, using cosine")
            return self._compute_distance_matrix(embeddings1, embeddings2, "cosine")
    
    def compute_similarity(self, 
                          query_embeddings: np.ndarray,
                          doc_embeddings: np.ndarray) -> SimilarityResult:
        """
        Compute Chamfer similarity between query and document embeddings.
        
        Args:
            query_embeddings: Query token embeddings (Q, D)
            doc_embeddings: Document token embeddings (N, D)
            
        Returns:
            SimilarityResult with Chamfer similarity score
        """
        try:
            import time
            start_time = time.time()
            
            if len(query_embeddings.shape) != 2 or len(doc_embeddings.shape) != 2:
                raise ValueError("Embeddings must be 2D arrays")
            
            query_tokens, query_dim = query_embeddings.shape
            doc_tokens, doc_dim = doc_embeddings.shape
            
            if query_dim != doc_dim:
                raise ValueError(f"Embedding dimensions must match: {query_dim} != {doc_dim}")
            
            logger.debug(f"🔄 Computing Chamfer similarity: Q={query_tokens}, D={doc_tokens}")
            
            # Compute distance matrix
            distance_matrix = self._compute_distance_matrix(query_embeddings, doc_embeddings)
            
            # Forward Chamfer: for each query token, find closest document token
            forward_distances = np.min(distance_matrix, axis=1)
            forward_chamfer = np.mean(forward_distances)
            
            if self.bidirectional:
                # Backward Chamfer: for each document token, find closest query token
                backward_distances = np.min(distance_matrix, axis=0)
                backward_chamfer = np.mean(backward_distances)
                
                # Bidirectional Chamfer distance
                chamfer_distance = (forward_chamfer + backward_chamfer) / 2
            else:
                chamfer_distance = forward_chamfer
            
            # Convert distance to similarity (lower distance = higher similarity)
            if self.distance_metric == "cosine":
                # For cosine distance, similarity = 1 - distance
                similarity_score = 1 - chamfer_distance
            elif self.distance_metric == "euclidean":
                # For Euclidean, use negative exponential
                similarity_score = np.exp(-chamfer_distance)
            elif self.distance_metric == "dot":
                # For negative dot product, negate to get similarity
                similarity_score = -chamfer_distance
            else:
                similarity_score = 1 - chamfer_distance
            
            # Normalize by length if requested
            if self.normalize_by_length:
                length_factor = np.sqrt(query_tokens * doc_tokens)
                similarity_score = similarity_score / length_factor
            
            processing_time = time.time() - start_time
            
            result = SimilarityResult(
                score=float(similarity_score),
                method=f"chamfer_{self.distance_metric}",
                query_tokens=query_tokens,
                doc_tokens=doc_tokens,
                processing_time=processing_time,
                detailed_scores=forward_distances.tolist(),
                metadata={
                    'bidirectional': self.bidirectional,
                    'normalize_by_length': self.normalize_by_length,
                    'forward_chamfer': float(forward_chamfer),
                    'backward_chamfer': float(backward_chamfer) if self.bidirectional else None,
                    'distance_metric': self.distance_metric
                }
            )
            
            logger.debug(f"✅ Chamfer similarity: {similarity_score:.4f}, {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to compute Chamfer similarity: {e}")
            return SimilarityResult(
                score=0.0,
                method="chamfer_error",
                query_tokens=0,
                doc_tokens=0,
                processing_time=0.0,
                detailed_scores=None,
                metadata={'error': str(e)}
            )

class MaxSimSimilarity:
    """
    MaxSim similarity for multi-vector embeddings.
    
    Computes maximum similarity between query and document tokens,
    following the ColBERT-style interaction model.
    """
    
    def __init__(self, 
                 similarity_metric: str = "cosine",
                 aggregation: str = "mean"):
        """
        Initialize MaxSim similarity calculator.
        
        Args:
            similarity_metric: Similarity metric ('cosine', 'dot')
            aggregation: How to aggregate max similarities ('mean', 'sum')
        """
        self.similarity_metric = similarity_metric
        self.aggregation = aggregation
        
        logger.info(f"🎯 MaxSimSimilarity initialized: {similarity_metric}, agg={aggregation}")
    
    def compute_similarity(self, 
                          query_embeddings: np.ndarray,
                          doc_embeddings: np.ndarray) -> SimilarityResult:
        """
        Compute MaxSim similarity between query and document embeddings.
        
        Args:
            query_embeddings: Query token embeddings (Q, D)
            doc_embeddings: Document token embeddings (N, D)
            
        Returns:
            SimilarityResult with MaxSim similarity score
        """
        try:
            import time
            start_time = time.time()
            
            if len(query_embeddings.shape) != 2 or len(doc_embeddings.shape) != 2:
                raise ValueError("Embeddings must be 2D arrays")
            
            query_tokens, query_dim = query_embeddings.shape
            doc_tokens, doc_dim = doc_embeddings.shape
            
            if query_dim != doc_dim:
                raise ValueError(f"Embedding dimensions must match: {query_dim} != {doc_dim}")
            
            logger.debug(f"🔄 Computing MaxSim similarity: Q={query_tokens}, D={doc_tokens}")
            
            # Compute similarity matrix
            if self.similarity_metric == "cosine":
                # Normalize embeddings
                query_norm = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
                doc_norm = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
                similarity_matrix = np.dot(query_norm, doc_norm.T)
            
            elif self.similarity_metric == "dot":
                # Dot product similarity
                similarity_matrix = np.dot(query_embeddings, doc_embeddings.T)
            
            else:
                logger.warning(f"⚠️  Unknown similarity metric: {self.similarity_metric}, using cosine")
                query_norm = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
                doc_norm = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
                similarity_matrix = np.dot(query_norm, doc_norm.T)
            
            # For each query token, find maximum similarity with any document token
            max_similarities = np.max(similarity_matrix, axis=1)
            
            # Aggregate the maximum similarities
            if self.aggregation == "mean":
                final_score = np.mean(max_similarities)
            elif self.aggregation == "sum":
                final_score = np.sum(max_similarities)
            else:
                logger.warning(f"⚠️  Unknown aggregation: {self.aggregation}, using mean")
                final_score = np.mean(max_similarities)
            
            processing_time = time.time() - start_time
            
            result = SimilarityResult(
                score=float(final_score),
                method=f"maxsim_{self.similarity_metric}",
                query_tokens=query_tokens,
                doc_tokens=doc_tokens,
                processing_time=processing_time,
                detailed_scores=max_similarities.tolist(),
                metadata={
                    'similarity_metric': self.similarity_metric,
                    'aggregation': self.aggregation,
                    'max_similarity': float(np.max(max_similarities)),
                    'min_similarity': float(np.min(max_similarities)),
                    'std_similarity': float(np.std(max_similarities))
                }
            )
            
            logger.debug(f"✅ MaxSim similarity: {final_score:.4f}, {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to compute MaxSim similarity: {e}")
            return SimilarityResult(
                score=0.0,
                method="maxsim_error",
                query_tokens=0,
                doc_tokens=0,
                processing_time=0.0,
                detailed_scores=None,
                metadata={'error': str(e)}
            )

def compute_chamfer_distance(query_embeddings: np.ndarray,
                           doc_embeddings: np.ndarray,
                           distance_metric: str = "cosine",
                           bidirectional: bool = True) -> float:
    """
    Convenience function to compute Chamfer distance.
    
    Args:
        query_embeddings: Query token embeddings
        doc_embeddings: Document token embeddings
        distance_metric: Distance metric to use
        bidirectional: Whether to use bidirectional distance
        
    Returns:
        Chamfer distance score
    """
    calculator = ChamferSimilarity(
        distance_metric=distance_metric,
        bidirectional=bidirectional
    )
    result = calculator.compute_similarity(query_embeddings, doc_embeddings)
    return result.score

def compute_maxsim_score(query_embeddings: np.ndarray,
                        doc_embeddings: np.ndarray,
                        similarity_metric: str = "cosine",
                        aggregation: str = "mean") -> float:
    """
    Convenience function to compute MaxSim score.
    
    Args:
        query_embeddings: Query token embeddings
        doc_embeddings: Document token embeddings
        similarity_metric: Similarity metric to use
        aggregation: Aggregation method
        
    Returns:
        MaxSim similarity score
    """
    calculator = MaxSimSimilarity(
        similarity_metric=similarity_metric,
        aggregation=aggregation
    )
    result = calculator.compute_similarity(query_embeddings, doc_embeddings)
    return result.score

def get_similarity_calculator(method: str = "maxsim", **kwargs):
    """
    Get a similarity calculator instance.
    
    Args:
        method: Similarity method ('chamfer', 'maxsim')
        **kwargs: Additional arguments for the calculator
        
    Returns:
        Similarity calculator instance
    """
    if method == "chamfer":
        return ChamferSimilarity(**kwargs)
    elif method == "maxsim":
        return MaxSimSimilarity(**kwargs)
    else:
        logger.warning(f"⚠️  Unknown similarity method: {method}, using maxsim")
        return MaxSimSimilarity(**kwargs)
