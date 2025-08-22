#!/usr/bin/env python3
"""
Dream Canvas Data Models
========================

Core data models and structures for the Dream Canvas cognitive visualization.
Extracted from the monolithic dream_canvas.py.

This module provides:
- Memory cluster data structures
- Cognitive map representations
- Visualization data models
- Research insight structures

Author: SAM Development Team
Version: 1.0.0 - Refactored from dream_canvas.py
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class VisualizationMethod(Enum):
    """Available dimensionality reduction methods."""
    UMAP = "umap"
    TSNE = "t-sne"
    PCA = "pca"
    MDS = "mds"


class ClusteringMethod(Enum):
    """Available clustering methods."""
    KMEANS = "kmeans"
    HDBSCAN = "hdbscan"
    DBSCAN = "dbscan"
    HIERARCHICAL = "hierarchical"


class TimeRange(Enum):
    """Available time range filters."""
    ALL_TIME = "all_time"
    LAST_WEEK = "last_week"
    LAST_MONTH = "last_month"
    LAST_YEAR = "last_year"
    CUSTOM = "custom"


@dataclass
class MemoryCluster:
    """Represents a cluster of related memories."""
    id: str
    name: str
    memories: List[Dict[str, Any]]
    center: Tuple[float, float]
    color: str
    size: int
    coherence_score: float
    keywords: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.keywords:
            self.keywords = self._extract_keywords()
    
    def _extract_keywords(self) -> List[str]:
        """Extract keywords from cluster memories."""
        # Placeholder implementation
        keywords = []
        for memory in self.memories[:5]:  # Sample first 5 memories
            content = memory.get('content', '')
            # Simple keyword extraction (would be more sophisticated in practice)
            words = content.split()[:10]
            keywords.extend([word.lower().strip('.,!?') for word in words if len(word) > 3])
        
        # Return unique keywords, limited to top 10
        return list(set(keywords))[:10]
    
    def get_summary(self) -> str:
        """Get a summary of the cluster."""
        return f"Cluster '{self.name}' with {self.size} memories (coherence: {self.coherence_score:.2f})"
    
    def get_memory_count(self) -> int:
        """Get the number of memories in this cluster."""
        return len(self.memories)
    
    def get_dominant_themes(self) -> List[str]:
        """Get the dominant themes in this cluster."""
        return self.keywords[:5]


@dataclass
class ClusterConnection:
    """Represents a connection between two clusters."""
    source_cluster_id: str
    target_cluster_id: str
    strength: float
    connection_type: str
    shared_concepts: List[str] = field(default_factory=list)
    
    def get_connection_strength_label(self) -> str:
        """Get a human-readable connection strength label."""
        if self.strength >= 0.8:
            return "Very Strong"
        elif self.strength >= 0.6:
            return "Strong"
        elif self.strength >= 0.4:
            return "Moderate"
        elif self.strength >= 0.2:
            return "Weak"
        else:
            return "Very Weak"


@dataclass
class CognitiveMap:
    """Represents the cognitive map visualization."""
    clusters: List[MemoryCluster]
    connections: List[ClusterConnection]
    metadata: Dict[str, Any]
    method: VisualizationMethod = VisualizationMethod.UMAP
    clustering_method: ClusteringMethod = ClusteringMethod.KMEANS
    time_range: TimeRange = TimeRange.ALL_TIME
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_cluster_by_id(self, cluster_id: str) -> Optional[MemoryCluster]:
        """Get a cluster by its ID."""
        for cluster in self.clusters:
            if cluster.id == cluster_id:
                return cluster
        return None
    
    def get_total_memories(self) -> int:
        """Get the total number of memories across all clusters."""
        return sum(cluster.get_memory_count() for cluster in self.clusters)
    
    def get_cluster_count(self) -> int:
        """Get the number of clusters."""
        return len(self.clusters)
    
    def get_connection_count(self) -> int:
        """Get the number of connections."""
        return len(self.connections)
    
    def get_average_coherence(self) -> float:
        """Get the average coherence score across all clusters."""
        if not self.clusters:
            return 0.0
        return sum(cluster.coherence_score for cluster in self.clusters) / len(self.clusters)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the cognitive map."""
        return {
            'total_memories': self.get_total_memories(),
            'cluster_count': self.get_cluster_count(),
            'connection_count': self.get_connection_count(),
            'average_coherence': self.get_average_coherence(),
            'method': self.method.value,
            'clustering_method': self.clustering_method.value,
            'time_range': self.time_range.value,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class ResearchInsight:
    """Represents a research insight generated from cluster analysis."""
    id: str
    cluster_id: str
    title: str
    description: str
    keywords: List[str]
    confidence_score: float
    research_papers: List[str] = field(default_factory=list)
    auto_ingested: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_research_status(self) -> str:
        """Get the research status of this insight."""
        if self.research_papers:
            return f"Researched ({len(self.research_papers)} papers)"
        else:
            return "Pending Research"
    
    def add_research_paper(self, paper_path: str):
        """Add a research paper to this insight."""
        if paper_path not in self.research_papers:
            self.research_papers.append(paper_path)


@dataclass
class VisualizationConfig:
    """Configuration for Dream Canvas visualization."""
    method: VisualizationMethod = VisualizationMethod.UMAP
    clustering_method: ClusteringMethod = ClusteringMethod.KMEANS
    time_range: TimeRange = TimeRange.ALL_TIME
    n_components: int = 2
    n_clusters: int = 8
    min_cluster_size: int = 5
    quality_threshold: float = 0.3
    perplexity: int = 30  # For t-SNE
    n_neighbors: int = 15  # For UMAP
    custom_start_date: Optional[datetime] = None
    custom_end_date: Optional[datetime] = None
    
    def validate(self) -> bool:
        """Validate the configuration."""
        if self.n_components < 2 or self.n_components > 3:
            return False
        if self.n_clusters < 2 or self.n_clusters > 50:
            return False
        if self.min_cluster_size < 2:
            return False
        if self.quality_threshold < 0.0 or self.quality_threshold > 1.0:
            return False
        if self.time_range == TimeRange.CUSTOM:
            if not self.custom_start_date or not self.custom_end_date:
                return False
            if self.custom_start_date >= self.custom_end_date:
                return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'method': self.method.value,
            'clustering_method': self.clustering_method.value,
            'time_range': self.time_range.value,
            'n_components': self.n_components,
            'n_clusters': self.n_clusters,
            'min_cluster_size': self.min_cluster_size,
            'quality_threshold': self.quality_threshold,
            'perplexity': self.perplexity,
            'n_neighbors': self.n_neighbors,
            'custom_start_date': self.custom_start_date.isoformat() if self.custom_start_date else None,
            'custom_end_date': self.custom_end_date.isoformat() if self.custom_end_date else None
        }


@dataclass
class DreamCanvasState:
    """Represents the current state of the Dream Canvas."""
    cognitive_map: Optional[CognitiveMap] = None
    config: VisualizationConfig = field(default_factory=VisualizationConfig)
    selected_cluster_id: Optional[str] = None
    research_insights: List[ResearchInsight] = field(default_factory=list)
    is_loading: bool = False
    last_updated: datetime = field(default_factory=datetime.now)
    
    def get_selected_cluster(self) -> Optional[MemoryCluster]:
        """Get the currently selected cluster."""
        if not self.cognitive_map or not self.selected_cluster_id:
            return None
        return self.cognitive_map.get_cluster_by_id(self.selected_cluster_id)
    
    def get_insights_for_cluster(self, cluster_id: str) -> List[ResearchInsight]:
        """Get research insights for a specific cluster."""
        return [insight for insight in self.research_insights if insight.cluster_id == cluster_id]
    
    def add_research_insight(self, insight: ResearchInsight):
        """Add a research insight."""
        self.research_insights.append(insight)
        self.last_updated = datetime.now()
    
    def update_cognitive_map(self, cognitive_map: CognitiveMap):
        """Update the cognitive map."""
        self.cognitive_map = cognitive_map
        self.last_updated = datetime.now()
    
    def clear_state(self):
        """Clear the current state."""
        self.cognitive_map = None
        self.selected_cluster_id = None
        self.research_insights = []
        self.is_loading = False
        self.last_updated = datetime.now()


# Color palettes for visualization
CLUSTER_COLORS = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
    '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
    '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2'
]

CONNECTION_COLORS = {
    'semantic': '#FF6B6B',
    'temporal': '#4ECDC4', 
    'causal': '#45B7D1',
    'associative': '#96CEB4'
}

# Default visualization parameters
DEFAULT_UMAP_PARAMS = {
    'n_neighbors': 15,
    'min_dist': 0.1,
    'metric': 'cosine'
}

DEFAULT_TSNE_PARAMS = {
    'perplexity': 30,
    'learning_rate': 200,
    'max_iter': 1000
}

DEFAULT_CLUSTERING_PARAMS = {
    'kmeans': {'n_clusters': 8, 'random_state': 42},
    'hdbscan': {'min_cluster_size': 5, 'min_samples': 3},
    'dbscan': {'eps': 0.5, 'min_samples': 5}
}
