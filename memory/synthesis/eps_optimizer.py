"""
Epsilon Optimizer for DBSCAN Clustering

This module implements the k-distance graph (elbow method) to find optimal eps values
for DBSCAN clustering on memory embeddings.
"""

import logging
import numpy as np
from typing import List, Tuple, Optional
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

class EpsOptimizer:
    """
    Finds optimal eps values for DBSCAN clustering using the k-distance graph method.
    
    The k-distance graph plots the distance of every point to its k-th nearest neighbor.
    The "elbow" or "knee" of this curve indicates the optimal eps value.
    """
    
    def __init__(self):
        """Initialize the eps optimizer."""
        pass
    
    def find_optimal_eps(self, embeddings: np.ndarray, min_samples: int = 5) -> Tuple[float, List[float]]:
        """
        Find optimal eps value using k-distance graph method.
        
        Args:
            embeddings: Normalized embedding vectors
            min_samples: The k value for k-nearest neighbors (should match DBSCAN min_samples)
            
        Returns:
            Tuple of (optimal_eps, sorted_distances) for plotting
        """
        try:
            logger.info(f"Finding optimal eps for {len(embeddings)} embeddings with k={min_samples}")
            
            # Ensure embeddings are normalized for cosine similarity
            normalized_embeddings = normalize(embeddings, norm='l2')
            
            # Find k-nearest neighbors
            neighbors = NearestNeighbors(
                n_neighbors=min_samples,
                metric='cosine',
                algorithm='auto'
            )
            neighbors.fit(normalized_embeddings)
            
            # Get distances to k-th nearest neighbor for each point
            distances, indices = neighbors.kneighbors(normalized_embeddings)
            
            # Extract k-th nearest neighbor distances (last column)
            k_distances = distances[:, -1]
            
            # Sort distances in ascending order
            sorted_distances = np.sort(k_distances)
            
            # Find the "elbow" using the maximum curvature method
            optimal_eps = self._find_elbow_point(sorted_distances)
            
            logger.info(f"Optimal eps found: {optimal_eps:.4f}")
            logger.info(f"Distance range: {sorted_distances.min():.4f} to {sorted_distances.max():.4f}")
            
            return optimal_eps, sorted_distances.tolist()
            
        except Exception as e:
            logger.error(f"Error finding optimal eps: {e}")
            # Return reasonable defaults
            return 0.5, []
    
    def _find_elbow_point(self, sorted_distances: np.ndarray) -> float:
        """
        Find the elbow point in the k-distance graph using maximum curvature.
        
        Args:
            sorted_distances: Sorted k-distances
            
        Returns:
            Optimal eps value at the elbow point
        """
        try:
            n_points = len(sorted_distances)
            
            # Create point indices
            x = np.arange(n_points)
            y = sorted_distances
            
            # Calculate second derivative (curvature) using finite differences
            if n_points < 10:
                # For small datasets, use a simple heuristic
                # Take the 90th percentile as a reasonable eps
                return float(np.percentile(sorted_distances, 90))
            
            # Calculate first derivative
            dy = np.gradient(y)
            
            # Calculate second derivative (curvature)
            d2y = np.gradient(dy)
            
            # Find the point with maximum curvature (steepest increase)
            # We look in the upper portion of the curve where the elbow typically occurs
            start_idx = int(n_points * 0.5)  # Start from middle
            end_idx = int(n_points * 0.95)   # End before the very tail
            
            if start_idx >= end_idx:
                start_idx = int(n_points * 0.7)
                end_idx = n_points - 1
            
            # Find maximum curvature in the target region
            search_region = d2y[start_idx:end_idx]
            max_curvature_idx = start_idx + np.argmax(search_region)
            
            optimal_eps = sorted_distances[max_curvature_idx]
            
            logger.debug(f"Elbow found at index {max_curvature_idx}/{n_points}, eps={optimal_eps:.4f}")
            
            return float(optimal_eps)
            
        except Exception as e:
            logger.error(f"Error finding elbow point: {e}")
            # Fallback to 90th percentile
            return float(np.percentile(sorted_distances, 90))
    
    def suggest_clustering_params(self, embeddings: np.ndarray, 
                                target_clusters: int = 10) -> dict:
        """
        Suggest complete clustering parameters based on data characteristics.
        
        Args:
            embeddings: Normalized embedding vectors
            target_clusters: Desired approximate number of clusters
            
        Returns:
            Dictionary with suggested parameters
        """
        try:
            n_points = len(embeddings)
            
            # Suggest min_samples based on data size
            if n_points < 100:
                min_samples = 3
            elif n_points < 1000:
                min_samples = 5
            elif n_points < 10000:
                min_samples = 8
            else:
                min_samples = 10
            
            # Find optimal eps
            optimal_eps, distances = self.find_optimal_eps(embeddings, min_samples)
            
            # Suggest min_cluster_size based on data size and target clusters
            min_cluster_size = max(3, n_points // (target_clusters * 10))
            
            # Suggest max_clusters
            max_clusters = min(50, target_clusters * 2)
            
            suggestions = {
                'eps': optimal_eps,
                'min_samples': min_samples,
                'min_cluster_size': min_cluster_size,
                'max_clusters': max_clusters,
                'data_size': n_points,
                'distance_stats': {
                    'min': float(np.min(distances)) if distances else 0.0,
                    'max': float(np.max(distances)) if distances else 1.0,
                    'median': float(np.median(distances)) if distances else 0.5,
                    'percentile_90': float(np.percentile(distances, 90)) if distances else 0.7
                }
            }
            
            logger.info(f"Suggested parameters: eps={optimal_eps:.4f}, min_samples={min_samples}, "
                       f"min_cluster_size={min_cluster_size}, max_clusters={max_clusters}")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error suggesting clustering parameters: {e}")
            return {
                'eps': 0.5,
                'min_samples': 5,
                'min_cluster_size': 8,
                'max_clusters': 15,
                'data_size': len(embeddings),
                'distance_stats': {}
            }
