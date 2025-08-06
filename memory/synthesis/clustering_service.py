"""
Clustering Service for SAM's Cognitive Synthesis Engine

This module implements DBSCAN-based clustering to identify dense concept clusters
in SAM's memory store for synthesis into emergent insights.
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from ..memory_vectorstore import MemoryVectorStore, MemoryChunk

logger = logging.getLogger(__name__)

@dataclass
class ConceptCluster:
    """Represents a cluster of related memory concepts."""
    cluster_id: str
    chunk_ids: List[str]
    chunks: List[MemoryChunk]
    centroid: np.ndarray
    coherence_score: float
    size: int
    dominant_themes: List[str]
    metadata: Dict[str, Any]

class ClusteringService:
    """
    Service for clustering memory vectors to identify concept groups for synthesis.
    
    Uses DBSCAN (Density-Based Spatial Clustering) which is ideal for this use case:
    - No need to pre-define number of clusters
    - Handles noise/outliers naturally  
    - Density-based approach perfect for concept clustering
    """
    
    def __init__(self,
                 eps: float = 0.5,
                 min_samples: int = 2,
                 min_cluster_size: int = 3,
                 max_clusters: int = 20,
                 quality_threshold: float = 0.3):
        """
        Initialize the clustering service.
        
        Args:
            eps: Maximum distance between samples in a cluster
            min_samples: Minimum samples in a neighborhood for core point
            min_cluster_size: Minimum size for a meaningful cluster
            max_clusters: Maximum number of clusters to return
            quality_threshold: Minimum coherence score for cluster inclusion
        """
        self.eps = eps
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.quality_threshold = quality_threshold
        
        logger.info(f"ClusteringService initialized with eps={eps}, min_samples={min_samples}")
    
    def find_concept_clusters(self, memory_store: MemoryVectorStore) -> List[ConceptCluster]:
        """
        Find dense concept clusters in the memory store.
        
        Args:
            memory_store: The memory vector store to analyze
            
        Returns:
            List of concept clusters suitable for synthesis
        """
        try:
            logger.info("🧠 Starting concept clustering analysis...")
            
            # Get all memory chunks and embeddings
            all_memories = memory_store.get_all_memories()
            logger.info(f"Retrieved {len(all_memories)} total memories from store")

            if len(all_memories) < self.min_cluster_size:
                logger.warning(f"Insufficient memories for clustering: {len(all_memories)} < {self.min_cluster_size}")
                return []
            
            # Extract embeddings and prepare data
            embeddings, chunk_ids, chunks = self._prepare_clustering_data(all_memories)
            if len(embeddings) == 0:
                logger.warning("No valid embeddings found for clustering")
                return []
            
            # Perform DBSCAN clustering
            clusters = self._perform_dbscan_clustering(embeddings)
            
            # Process clusters into ConceptCluster objects
            concept_clusters = self._process_clusters(clusters, chunk_ids, chunks, embeddings)
            logger.info(f"Processed {len(concept_clusters)} raw concept clusters")

            # Filter and rank clusters by quality
            quality_clusters = self._filter_and_rank_clusters(concept_clusters)
            logger.info(f"After quality filtering: {len(quality_clusters)} high-quality clusters (threshold: {self.quality_threshold})")
            
            logger.info(f"✅ Found {len(quality_clusters)} high-quality concept clusters")
            return quality_clusters
            
        except Exception as e:
            logger.error(f"Error in concept clustering: {e}")
            return []
    
    def _prepare_clustering_data(self, memories: List[MemoryChunk]) -> Tuple[np.ndarray, List[str], List[MemoryChunk]]:
        """Prepare memory data for clustering analysis."""
        embeddings = []
        chunk_ids = []
        valid_chunks = []
        
        for chunk in memories:
            if chunk.embedding is not None and len(chunk.embedding) > 0:
                embeddings.append(chunk.embedding)
                chunk_ids.append(chunk.chunk_id)
                valid_chunks.append(chunk)
        
        if len(embeddings) == 0:
            return np.array([]), [], []
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings)

        # For cosine similarity, we should NOT use StandardScaler as it breaks the angular relationships
        # Instead, we normalize to unit vectors (L2 normalization)
        from sklearn.preprocessing import normalize
        normalized_embeddings = normalize(embeddings_array, norm='l2')

        logger.info(f"Prepared {len(embeddings)} embeddings for clustering (L2 normalized for cosine similarity)")
        return normalized_embeddings, chunk_ids, valid_chunks
    
    def _perform_dbscan_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """Perform DBSCAN clustering on embeddings."""
        logger.info(f"Running DBSCAN with eps={self.eps}, min_samples={self.min_samples}")
        
        # Initialize DBSCAN
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='cosine')
        
        # Fit and predict clusters
        cluster_labels = dbscan.fit_predict(embeddings)
        
        # Log clustering results
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        logger.info(f"DBSCAN results: {n_clusters} clusters, {n_noise} noise points")
        
        return cluster_labels
    
    def _process_clusters(self, cluster_labels: np.ndarray, chunk_ids: List[str], 
                         chunks: List[MemoryChunk], embeddings: np.ndarray) -> List[ConceptCluster]:
        """Process raw cluster labels into ConceptCluster objects."""
        concept_clusters = []
        
        # Group chunks by cluster
        cluster_groups = {}
        for i, label in enumerate(cluster_labels):
            if label != -1:  # Skip noise points
                if label not in cluster_groups:
                    cluster_groups[label] = []
                cluster_groups[label].append(i)
        
        # Create ConceptCluster objects
        logger.info(f"Found {len(cluster_groups)} cluster groups from DBSCAN")
        for cluster_id, indices in cluster_groups.items():
            logger.debug(f"Cluster {cluster_id}: {len(indices)} members (min required: {self.min_cluster_size})")
            if len(indices) >= self.min_cluster_size:
                cluster_chunks = [chunks[i] for i in indices]
                cluster_embeddings = embeddings[indices]
                cluster_chunk_ids = [chunk_ids[i] for i in indices]
                
                # Calculate cluster centroid
                centroid = np.mean(cluster_embeddings, axis=0)
                
                # Calculate coherence score (average cosine similarity to centroid)
                coherence_score = self._calculate_coherence_score(cluster_embeddings, centroid)
                
                # Extract dominant themes
                dominant_themes = self._extract_dominant_themes(cluster_chunks)
                
                concept_cluster = ConceptCluster(
                    cluster_id=f"cluster_{cluster_id:03d}",
                    chunk_ids=cluster_chunk_ids,
                    chunks=cluster_chunks,
                    centroid=centroid,
                    coherence_score=coherence_score,
                    size=len(cluster_chunks),
                    dominant_themes=dominant_themes,
                    metadata={
                        'dbscan_label': cluster_id,
                        'avg_importance': np.mean([chunk.importance_score for chunk in cluster_chunks]),
                        'source_diversity': len(set(chunk.source for chunk in cluster_chunks)),
                        'memory_types': list(set(chunk.memory_type.value for chunk in cluster_chunks))
                    }
                )
                
                concept_clusters.append(concept_cluster)
        
        return concept_clusters
    
    def _calculate_coherence_score(self, embeddings: np.ndarray, centroid: np.ndarray) -> float:
        """Calculate cluster coherence using cosine similarity to centroid."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarities = cosine_similarity(embeddings, centroid.reshape(1, -1))
        return float(np.mean(similarities))
    
    def _extract_dominant_themes(self, chunks: List[MemoryChunk]) -> List[str]:
        """Extract dominant themes from cluster chunks."""
        # Collect all tags from chunks
        all_tags = []
        for chunk in chunks:
            all_tags.extend(chunk.tags)
        
        # Count tag frequency
        tag_counts = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Return top themes (tags that appear in at least 30% of chunks)
        min_frequency = max(1, len(chunks) * 0.3)
        dominant_themes = [tag for tag, count in tag_counts.items() if count >= min_frequency]
        
        # Sort by frequency and return top 5
        dominant_themes.sort(key=lambda x: tag_counts[x], reverse=True)
        return dominant_themes[:5]
    
    def _filter_and_rank_clusters(self, clusters: List[ConceptCluster]) -> List[ConceptCluster]:
        """Filter clusters by quality and rank by synthesis potential."""
        # Filter by quality threshold
        quality_clusters = [
            cluster for cluster in clusters 
            if cluster.coherence_score >= self.quality_threshold
        ]
        
        # Rank by synthesis potential (combination of coherence, size, and importance)
        def synthesis_potential(cluster: ConceptCluster) -> float:
            return (
                cluster.coherence_score * 0.4 +
                min(cluster.size / 10.0, 1.0) * 0.3 +  # Normalize size
                cluster.metadata.get('avg_importance', 0.5) * 0.2 +
                min(cluster.metadata.get('source_diversity', 1) / 5.0, 1.0) * 0.1
            )
        
        quality_clusters.sort(key=synthesis_potential, reverse=True)
        
        # Return top clusters up to max limit
        return quality_clusters[:self.max_clusters]
