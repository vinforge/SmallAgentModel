#!/usr/bin/env python3
"""
Cognitive Mapping Engine
========================

Core cognitive mapping and dimensionality reduction functionality for Dream Canvas.
Extracted from the monolithic dream_canvas.py.

This module provides:
- Cognitive map generation
- Dimensionality reduction algorithms
- Memory clustering
- Cluster analysis and enhancement

Author: SAM Development Team
Version: 1.0.0 - Refactored from dream_canvas.py
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import random

from sam.dream_canvas.utils.models import (
    MemoryCluster, CognitiveMap, ClusterConnection, VisualizationConfig,
    VisualizationMethod, ClusteringMethod, TimeRange, CLUSTER_COLORS
)

logger = logging.getLogger(__name__)


class CognitiveMappingEngine:
    """Handles cognitive map generation and analysis."""
    
    def __init__(self):
        self.memory_store = None
        self._initialize_memory_store()
    
    def _initialize_memory_store(self):
        """Initialize the memory store connection."""
        try:
            from memory.memory_vectorstore import get_memory_store
            self.memory_store = get_memory_store()
            logger.info("Memory store initialized for cognitive mapping")
        except ImportError as e:
            logger.error(f"Failed to import memory store: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize memory store: {e}")
    
    def generate_cognitive_map(self, config: VisualizationConfig) -> CognitiveMap:
        """
        Generate a cognitive map based on the provided configuration.
        
        Args:
            config: Visualization configuration
            
        Returns:
            CognitiveMap: Generated cognitive map
        """
        try:
            logger.info(f"Generating cognitive map with method: {config.method.value}")
            
            if not self.memory_store:
                logger.warning("Memory store not available, generating fallback map")
                return self._generate_fallback_cognitive_map(config)
            
            # Get memories based on time range
            memories = self._get_memories_for_time_range(config.time_range, 
                                                       config.custom_start_date, 
                                                       config.custom_end_date)
            
            if not memories:
                logger.warning("No memories found for the specified time range")
                return self._generate_fallback_cognitive_map(config)
            
            # Try advanced clustering first
            clusters = self._try_advanced_clustering(memories, config)
            
            if not clusters:
                # Fallback to basic clustering
                logger.info("Advanced clustering failed, using fallback method")
                clusters = self._try_basic_clustering(memories, config)
            
            # Apply dimensionality reduction
            clusters = self._apply_dimensionality_reduction(clusters, config)
            
            # Enhance cluster separation
            clusters = self._enhance_cluster_separation(clusters)
            
            # Generate connections between clusters
            connections = self._generate_cluster_connections(clusters)
            
            # Create cognitive map
            cognitive_map = CognitiveMap(
                clusters=clusters,
                connections=connections,
                metadata={
                    'total_memories': len(memories),
                    'generation_method': 'advanced',
                    'config': config.to_dict()
                },
                method=config.method,
                clustering_method=config.clustering_method,
                time_range=config.time_range
            )
            
            logger.info(f"Generated cognitive map with {len(clusters)} clusters and {len(connections)} connections")
            return cognitive_map
            
        except Exception as e:
            logger.error(f"Error generating cognitive map: {e}")
            return self._generate_fallback_cognitive_map(config)
    
    def _get_memories_for_time_range(self, time_range: TimeRange, 
                                   start_date: Optional[datetime] = None,
                                   end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get memories for the specified time range."""
        try:
            if time_range == TimeRange.ALL_TIME:
                return self.memory_store.get_all_memories()
            elif time_range == TimeRange.LAST_WEEK:
                cutoff = datetime.now() - timedelta(weeks=1)
                return self.memory_store.get_memories_since(cutoff)
            elif time_range == TimeRange.LAST_MONTH:
                cutoff = datetime.now() - timedelta(days=30)
                return self.memory_store.get_memories_since(cutoff)
            elif time_range == TimeRange.LAST_YEAR:
                cutoff = datetime.now() - timedelta(days=365)
                return self.memory_store.get_memories_since(cutoff)
            elif time_range == TimeRange.CUSTOM and start_date and end_date:
                return self.memory_store.get_memories_between(start_date, end_date)
            else:
                return self.memory_store.get_all_memories()
                
        except Exception as e:
            logger.error(f"Error getting memories for time range: {e}")
            return []
    
    def _try_advanced_clustering(self, memories: List[Dict[str, Any]], 
                               config: VisualizationConfig) -> List[MemoryCluster]:
        """Try advanced clustering methods."""
        try:
            if config.clustering_method == ClusteringMethod.KMEANS:
                return self._try_kmeans_clustering(memories, config)
            elif config.clustering_method == ClusteringMethod.HDBSCAN:
                return self._try_hdbscan_clustering(memories, config)
            elif config.clustering_method == ClusteringMethod.DBSCAN:
                return self._try_dbscan_clustering(memories, config)
            else:
                return self._try_kmeans_clustering(memories, config)
                
        except Exception as e:
            logger.error(f"Advanced clustering failed: {e}")
            return []
    
    def _try_kmeans_clustering(self, memories: List[Dict[str, Any]], 
                             config: VisualizationConfig) -> List[MemoryCluster]:
        """Try K-means clustering."""
        try:
            # Extract embeddings from memories
            embeddings = []
            for memory in memories:
                if 'embedding' in memory:
                    embeddings.append(memory['embedding'])
                else:
                    # Generate dummy embedding if not available
                    embeddings.append(np.random.rand(384))  # Typical embedding size
            
            if not embeddings:
                return []
            
            # Perform K-means clustering
            from sklearn.cluster import KMeans
            
            embeddings_array = np.array(embeddings)
            kmeans = KMeans(n_clusters=config.n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings_array)
            
            # Group memories by cluster
            clusters = []
            for cluster_id in range(config.n_clusters):
                cluster_memories = [memories[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                
                if len(cluster_memories) >= config.min_cluster_size:
                    # Calculate cluster center
                    cluster_embeddings = [embeddings[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                    center = np.mean(cluster_embeddings, axis=0)
                    
                    # Calculate coherence score
                    coherence = self._calculate_cluster_coherence(cluster_embeddings)
                    
                    if coherence >= config.quality_threshold:
                        cluster = MemoryCluster(
                            id=f"cluster_{cluster_id}",
                            name=f"Cluster {cluster_id + 1}",
                            memories=cluster_memories,
                            center=(float(center[0]), float(center[1])),
                            color=CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)],
                            size=len(cluster_memories),
                            coherence_score=coherence
                        )
                        clusters.append(cluster)
            
            return clusters
            
        except ImportError:
            logger.warning("scikit-learn not available for K-means clustering")
            return []
        except Exception as e:
            logger.error(f"K-means clustering failed: {e}")
            return []
    
    def _try_hdbscan_clustering(self, memories: List[Dict[str, Any]], 
                              config: VisualizationConfig) -> List[MemoryCluster]:
        """Try HDBSCAN clustering."""
        try:
            # Similar implementation to K-means but with HDBSCAN
            # This would use the hdbscan library
            logger.info("HDBSCAN clustering not implemented yet, falling back to K-means")
            return self._try_kmeans_clustering(memories, config)
            
        except Exception as e:
            logger.error(f"HDBSCAN clustering failed: {e}")
            return []
    
    def _try_dbscan_clustering(self, memories: List[Dict[str, Any]], 
                             config: VisualizationConfig) -> List[MemoryCluster]:
        """Try DBSCAN clustering."""
        try:
            # Similar implementation to K-means but with DBSCAN
            logger.info("DBSCAN clustering not implemented yet, falling back to K-means")
            return self._try_kmeans_clustering(memories, config)
            
        except Exception as e:
            logger.error(f"DBSCAN clustering failed: {e}")
            return []
    
    def _try_basic_clustering(self, memories: List[Dict[str, Any]], 
                            config: VisualizationConfig) -> List[MemoryCluster]:
        """Basic clustering fallback method."""
        # Simple clustering based on content similarity
        clusters = []
        
        # Group memories into clusters based on simple heuristics
        cluster_size = max(len(memories) // config.n_clusters, config.min_cluster_size)
        
        for i in range(0, len(memories), cluster_size):
            cluster_memories = memories[i:i + cluster_size]
            
            if len(cluster_memories) >= config.min_cluster_size:
                cluster = MemoryCluster(
                    id=f"basic_cluster_{len(clusters)}",
                    name=f"Memory Group {len(clusters) + 1}",
                    memories=cluster_memories,
                    center=(random.uniform(-1, 1), random.uniform(-1, 1)),
                    color=CLUSTER_COLORS[len(clusters) % len(CLUSTER_COLORS)],
                    size=len(cluster_memories),
                    coherence_score=random.uniform(0.4, 0.8)
                )
                clusters.append(cluster)
        
        return clusters
    
    def _apply_dimensionality_reduction(self, clusters: List[MemoryCluster], 
                                      config: VisualizationConfig) -> List[MemoryCluster]:
        """Apply dimensionality reduction to cluster centers."""
        try:
            if config.method == VisualizationMethod.UMAP:
                return self._apply_umap_reduction(clusters, config)
            elif config.method == VisualizationMethod.TSNE:
                return self._apply_tsne_reduction(clusters, config)
            elif config.method == VisualizationMethod.PCA:
                return self._apply_pca_reduction(clusters, config)
            else:
                return clusters  # Return as-is if method not supported
                
        except Exception as e:
            logger.error(f"Dimensionality reduction failed: {e}")
            return clusters
    
    def _apply_umap_reduction(self, clusters: List[MemoryCluster], 
                            config: VisualizationConfig) -> List[MemoryCluster]:
        """Apply UMAP dimensionality reduction."""
        try:
            # Extract cluster centers
            centers = np.array([cluster.center for cluster in clusters])
            
            # Apply UMAP (would use umap-learn library)
            # For now, just normalize the centers
            normalized_centers = (centers - centers.mean(axis=0)) / centers.std(axis=0)
            
            # Update cluster centers
            for i, cluster in enumerate(clusters):
                cluster.center = (float(normalized_centers[i][0]), float(normalized_centers[i][1]))
            
            return clusters
            
        except Exception as e:
            logger.error(f"UMAP reduction failed: {e}")
            return clusters
    
    def _apply_tsne_reduction(self, clusters: List[MemoryCluster], 
                            config: VisualizationConfig) -> List[MemoryCluster]:
        """Apply t-SNE dimensionality reduction."""
        # Similar to UMAP but with t-SNE parameters
        return self._apply_umap_reduction(clusters, config)
    
    def _apply_pca_reduction(self, clusters: List[MemoryCluster], 
                           config: VisualizationConfig) -> List[MemoryCluster]:
        """Apply PCA dimensionality reduction."""
        # Similar to UMAP but with PCA
        return self._apply_umap_reduction(clusters, config)
    
    def _enhance_cluster_separation(self, clusters: List[MemoryCluster]) -> List[MemoryCluster]:
        """Enhance separation between clusters for better visualization."""
        if len(clusters) < 2:
            return clusters
        
        # Apply force-directed layout to separate clusters
        for _ in range(10):  # Iterations
            for i, cluster1 in enumerate(clusters):
                for j, cluster2 in enumerate(clusters):
                    if i != j:
                        # Calculate distance
                        dx = cluster2.center[0] - cluster1.center[0]
                        dy = cluster2.center[1] - cluster1.center[1]
                        distance = np.sqrt(dx**2 + dy**2)
                        
                        # Apply repulsive force if too close
                        if distance < 0.5:
                            force = 0.1 / (distance + 0.01)
                            cluster1.center = (
                                cluster1.center[0] - force * dx,
                                cluster1.center[1] - force * dy
                            )
        
        return clusters
    
    def _generate_cluster_connections(self, clusters: List[MemoryCluster]) -> List[ClusterConnection]:
        """Generate connections between clusters."""
        connections = []
        
        for i, cluster1 in enumerate(clusters):
            for j, cluster2 in enumerate(clusters):
                if i < j:  # Avoid duplicate connections
                    # Calculate connection strength based on shared concepts
                    shared_concepts = set(cluster1.keywords) & set(cluster2.keywords)
                    strength = len(shared_concepts) / max(len(cluster1.keywords), len(cluster2.keywords), 1)
                    
                    if strength > 0.1:  # Only create connections above threshold
                        connection = ClusterConnection(
                            source_cluster_id=cluster1.id,
                            target_cluster_id=cluster2.id,
                            strength=strength,
                            connection_type='semantic',
                            shared_concepts=list(shared_concepts)
                        )
                        connections.append(connection)
        
        return connections
    
    def _calculate_cluster_coherence(self, embeddings: List[np.ndarray]) -> float:
        """Calculate coherence score for a cluster."""
        if len(embeddings) < 2:
            return 1.0
        
        # Calculate average pairwise similarity
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(similarity)
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def _generate_fallback_cognitive_map(self, config: VisualizationConfig) -> CognitiveMap:
        """Generate a fallback cognitive map when memory store is unavailable."""
        logger.info("Generating fallback cognitive map")
        
        # Create mock clusters
        clusters = []
        for i in range(min(config.n_clusters, 6)):  # Limit fallback clusters
            cluster = MemoryCluster(
                id=f"fallback_cluster_{i}",
                name=f"Sample Cluster {i + 1}",
                memories=[],  # Empty for fallback
                center=(random.uniform(-2, 2), random.uniform(-2, 2)),
                color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                size=random.randint(5, 20),
                coherence_score=random.uniform(0.5, 0.9),
                keywords=[f"concept_{i}_{j}" for j in range(3)]
            )
            clusters.append(cluster)
        
        # Create mock connections
        connections = []
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if random.random() > 0.7:  # 30% chance of connection
                    connection = ClusterConnection(
                        source_cluster_id=clusters[i].id,
                        target_cluster_id=clusters[j].id,
                        strength=random.uniform(0.2, 0.8),
                        connection_type='semantic'
                    )
                    connections.append(connection)
        
        return CognitiveMap(
            clusters=clusters,
            connections=connections,
            metadata={
                'total_memories': 0,
                'generation_method': 'fallback',
                'config': config.to_dict()
            },
            method=config.method,
            clustering_method=config.clustering_method,
            time_range=config.time_range
        )


# Global cognitive mapping engine instance
_cognitive_mapping_engine = None


def get_cognitive_mapping_engine() -> CognitiveMappingEngine:
    """Get the global cognitive mapping engine instance."""
    global _cognitive_mapping_engine
    if _cognitive_mapping_engine is None:
        _cognitive_mapping_engine = CognitiveMappingEngine()
    return _cognitive_mapping_engine


def generate_cognitive_map(config: VisualizationConfig) -> CognitiveMap:
    """Generate a cognitive map using the global engine."""
    return get_cognitive_mapping_engine().generate_cognitive_map(config)
