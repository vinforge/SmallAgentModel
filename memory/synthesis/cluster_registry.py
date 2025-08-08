"""
Cluster Registry Service for SAM's Synthesis Engine

This service manages the mapping between synthesis cluster IDs and their metadata,
providing a bridge between the synthesis engine and UI visualization systems.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ClusterMetadata:
    """Metadata for a synthesis cluster."""
    cluster_id: str
    size: int
    coherence_score: float
    dominant_themes: List[str]
    memory_count: int
    avg_importance: float
    sources: List[str]
    source_count: int
    insight_generated: bool
    synthesis_run_id: str
    created_at: str

class ClusterRegistry:
    """
    Registry service for managing cluster metadata across synthesis runs.
    
    This service provides:
    1. Cluster metadata persistence and retrieval
    2. Mapping between cluster IDs and their data
    3. Fallback data for UI display when clusters are missing
    """
    
    def __init__(self, synthesis_output_dir: str = "synthesis_output"):
        """Initialize the cluster registry."""
        self.synthesis_output_dir = Path(synthesis_output_dir)
        self.synthesis_output_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for quick access
        self._cluster_cache: Dict[str, ClusterMetadata] = {}
        self._last_cache_update = None
        
        logger.info("ClusterRegistry initialized")
    
    def get_cluster_metadata(self, cluster_id: str) -> Optional[ClusterMetadata]:
        """
        Retrieve metadata for a specific cluster.
        
        Args:
            cluster_id: The cluster ID to look up
            
        Returns:
            ClusterMetadata if found, None otherwise
        """
        try:
            # Check cache first
            if cluster_id in self._cluster_cache:
                return self._cluster_cache[cluster_id]
            
            # Load from synthesis files
            self._refresh_cache()
            
            return self._cluster_cache.get(cluster_id)
            
        except Exception as e:
            logger.error(f"Error retrieving cluster metadata for {cluster_id}: {e}")
            return None
    
    def get_cluster_stats(self, cluster_id: str) -> Dict[str, Any]:
        """
        Get cluster statistics for UI display.
        
        Args:
            cluster_id: The cluster ID to get stats for
            
        Returns:
            Dictionary with cluster statistics
        """
        metadata = self.get_cluster_metadata(cluster_id)
        
        if metadata:
            return {
                "memory_count": metadata.memory_count,
                "avg_importance": metadata.avg_importance,
                "source_count": metadata.source_count,
                "sources": metadata.sources,
                "coherence_score": metadata.coherence_score,
                "dominant_themes": metadata.dominant_themes,
                "exists": True
            }
        else:
            # Return fallback data for missing clusters
            return {
                "memory_count": 0,
                "avg_importance": 0.0,
                "source_count": 0,
                "sources": [],
                "coherence_score": 0.0,
                "dominant_themes": [],
                "exists": False
            }
    
    def register_clusters_from_synthesis(self, synthesis_file: Path) -> int:
        """
        Register clusters from a synthesis output file.
        
        Args:
            synthesis_file: Path to synthesis output JSON file
            
        Returns:
            Number of clusters registered
        """
        try:
            with open(synthesis_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            cluster_metadata = data.get('cluster_metadata', {})
            synthesis_run_id = data.get('synthesis_run_log', {}).get('run_id', 'unknown')
            
            registered_count = 0
            
            for cluster_id, cluster_data in cluster_metadata.items():
                metadata = ClusterMetadata(
                    cluster_id=cluster_id,
                    size=cluster_data.get('size', 0),
                    coherence_score=cluster_data.get('coherence_score', 0.0),
                    dominant_themes=cluster_data.get('dominant_themes', []),
                    memory_count=cluster_data.get('memory_count', 0),
                    avg_importance=cluster_data.get('avg_importance', 0.0),
                    sources=cluster_data.get('sources', []),
                    source_count=cluster_data.get('source_count', 0),
                    insight_generated=cluster_data.get('insight_generated', False),
                    synthesis_run_id=synthesis_run_id,
                    created_at=datetime.now().isoformat()
                )
                
                self._cluster_cache[cluster_id] = metadata
                registered_count += 1
            
            logger.info(f"Registered {registered_count} clusters from {synthesis_file.name}")
            return registered_count
            
        except Exception as e:
            logger.error(f"Error registering clusters from {synthesis_file}: {e}")
            return 0
    
    def _refresh_cache(self):
        """Refresh the cluster cache from synthesis files."""
        try:
            # Find all synthesis files
            synthesis_files = list(self.synthesis_output_dir.glob("synthesis_run_log_*.json"))
            
            if not synthesis_files:
                logger.warning("No synthesis files found for cluster registry")
                return
            
            # Sort by modification time (newest first)
            synthesis_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            # Load clusters from recent files (last 10 runs)
            for synthesis_file in synthesis_files[:10]:
                self.register_clusters_from_synthesis(synthesis_file)
            
            self._last_cache_update = datetime.now()
            logger.info(f"Cluster cache refreshed with {len(self._cluster_cache)} clusters")
            
        except Exception as e:
            logger.error(f"Error refreshing cluster cache: {e}")
    
    def get_all_clusters(self) -> Dict[str, ClusterMetadata]:
        """Get all registered clusters."""
        self._refresh_cache()
        return self._cluster_cache.copy()
    
    def clear_cache(self):
        """Clear the cluster cache."""
        self._cluster_cache.clear()
        self._last_cache_update = None
        logger.info("Cluster cache cleared")

# Global registry instance
_cluster_registry = None

def get_cluster_registry() -> ClusterRegistry:
    """Get the global cluster registry instance."""
    global _cluster_registry
    if _cluster_registry is None:
        _cluster_registry = ClusterRegistry()
    return _cluster_registry

def get_cluster_stats(cluster_id: str) -> Dict[str, Any]:
    """Convenience function to get cluster statistics."""
    registry = get_cluster_registry()
    return registry.get_cluster_stats(cluster_id)
