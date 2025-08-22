"""
Algonauts-Inspired Neural Visualization for SAM
===============================================

Advanced visualization of SAM's neural activation patterns inspired by
the Algonauts project. Provides dimensionality reduction and trajectory
analysis of cognitive states.

Author: SAM Development Team
Version: 1.0.0
"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from .flight_recorder import CognitiveVector


logger = logging.getLogger(__name__)


@dataclass
class ProjectedPoint:
    """A point in the projected cognitive space."""
    step_id: str
    x: float
    y: float
    timestamp: float
    step_type: str
    component: str
    metadata: Dict[str, Any]


@dataclass
class CognitiveTrajectory:
    """Complete cognitive trajectory with projections."""
    session_id: str
    original_dimension: int
    projection_method: str
    projected_points: List[ProjectedPoint]
    explained_variance: Optional[float]
    trajectory_metrics: Dict[str, float]


class AlgonautsVisualizer:
    """
    Algonauts-inspired visualizer for SAM's cognitive processes.
    
    Features:
    - UMAP/PCA/t-SNE dimensionality reduction
    - Trajectory analysis and metrics
    - Cognitive state clustering
    - Interactive visualization data
    """
    
    def __init__(self, 
                 default_method: str = "umap",
                 n_components: int = 2,
                 random_state: int = 42):
        """
        Initialize the Algonauts visualizer.
        
        Args:
            default_method: Default dimensionality reduction method
            n_components: Number of dimensions for projection
            random_state: Random seed for reproducibility
        """
        self.default_method = default_method
        self.n_components = n_components
        self.random_state = random_state
        
        self.logger = logging.getLogger(f"{__name__}.AlgonautsVisualizer")
        
        # Check available methods
        self.available_methods = ["pca", "tsne"]
        if UMAP_AVAILABLE:
            self.available_methods.append("umap")
        
        if default_method not in self.available_methods:
            self.logger.warning(f"Method {default_method} not available, using PCA")
            self.default_method = "pca"
    
    def project_cognitive_vectors(self, 
                                  cognitive_vectors: List[CognitiveVector],
                                  method: Optional[str] = None,
                                  trace_events: Optional[List[Dict]] = None) -> CognitiveTrajectory:
        """
        Project high-dimensional cognitive vectors to 2D space.
        
        Args:
            cognitive_vectors: List of cognitive vectors to project
            method: Dimensionality reduction method to use
            trace_events: Corresponding trace events for context
            
        Returns:
            CognitiveTrajectory: Projected trajectory with metadata
        """
        if not cognitive_vectors:
            raise ValueError("No cognitive vectors provided")
        
        method = method or self.default_method
        if method not in self.available_methods:
            self.logger.warning(f"Method {method} not available, using PCA")
            method = "pca"
        
        self.logger.info(f"🧠 Projecting {len(cognitive_vectors)} cognitive vectors using {method.upper()}")
        
        # Extract vector data
        vector_matrix = np.array([cv.vector_data for cv in cognitive_vectors])
        
        # Perform dimensionality reduction
        projected_data, explained_variance = self._reduce_dimensions(vector_matrix, method)
        
        # Create projected points with metadata
        projected_points = []
        for i, cv in enumerate(cognitive_vectors):
            # Find corresponding trace event for context
            step_type = "unknown"
            component = "unknown"
            if trace_events:
                matching_event = next(
                    (event for event in trace_events if event.get('id') == cv.step_id),
                    None
                )
                if matching_event:
                    step_type = matching_event.get('step_type', 'unknown')
                    component = matching_event.get('component', 'unknown')
            
            point = ProjectedPoint(
                step_id=cv.step_id,
                x=float(projected_data[i, 0]),
                y=float(projected_data[i, 1]),
                timestamp=cv.timestamp,
                step_type=step_type,
                component=component,
                metadata=cv.metadata
            )
            projected_points.append(point)
        
        # Calculate trajectory metrics
        trajectory_metrics = self._calculate_trajectory_metrics(projected_points)
        
        # Create trajectory object
        trajectory = CognitiveTrajectory(
            session_id=cognitive_vectors[0].metadata.get('session_id', 'unknown'),
            original_dimension=cognitive_vectors[0].dimension,
            projection_method=method,
            projected_points=projected_points,
            explained_variance=explained_variance,
            trajectory_metrics=trajectory_metrics
        )
        
        self.logger.info(f"✅ Projection complete - Explained variance: {explained_variance:.3f}")
        return trajectory
    
    def _reduce_dimensions(self, vector_matrix: np.ndarray, method: str) -> Tuple[np.ndarray, float]:
        """
        Perform dimensionality reduction on vector matrix.
        
        Args:
            vector_matrix: Matrix of vectors to reduce
            method: Reduction method to use
            
        Returns:
            Tuple of (projected_data, explained_variance)
        """
        if method == "pca":
            reducer = PCA(n_components=self.n_components, random_state=self.random_state)
            projected = reducer.fit_transform(vector_matrix)
            explained_variance = sum(reducer.explained_variance_ratio_)
            
        elif method == "tsne":
            reducer = TSNE(n_components=self.n_components, random_state=self.random_state, 
                          perplexity=min(30, len(vector_matrix) - 1))
            projected = reducer.fit_transform(vector_matrix)
            explained_variance = 0.0  # t-SNE doesn't provide explained variance
            
        elif method == "umap" and UMAP_AVAILABLE:
            reducer = umap.UMAP(n_components=self.n_components, random_state=self.random_state)
            projected = reducer.fit_transform(vector_matrix)
            explained_variance = 0.0  # UMAP doesn't provide explained variance directly
            
        else:
            raise ValueError(f"Unknown or unavailable method: {method}")
        
        return projected, explained_variance
    
    def _calculate_trajectory_metrics(self, points: List[ProjectedPoint]) -> Dict[str, float]:
        """
        Calculate metrics for the cognitive trajectory.
        
        Args:
            points: List of projected points
            
        Returns:
            Dictionary of trajectory metrics
        """
        if len(points) < 2:
            return {}
        
        # Calculate total path length
        total_distance = 0.0
        for i in range(1, len(points)):
            dx = points[i].x - points[i-1].x
            dy = points[i].y - points[i-1].y
            total_distance += np.sqrt(dx*dx + dy*dy)
        
        # Calculate displacement (start to end distance)
        start_point = points[0]
        end_point = points[-1]
        displacement = np.sqrt(
            (end_point.x - start_point.x)**2 + 
            (end_point.y - start_point.y)**2
        )
        
        # Calculate tortuosity (path length / displacement)
        tortuosity = total_distance / displacement if displacement > 0 else float('inf')
        
        # Calculate average step size
        step_sizes = []
        for i in range(1, len(points)):
            dx = points[i].x - points[i-1].x
            dy = points[i].y - points[i-1].y
            step_sizes.append(np.sqrt(dx*dx + dy*dy))
        
        avg_step_size = np.mean(step_sizes) if step_sizes else 0.0
        step_variance = np.var(step_sizes) if step_sizes else 0.0
        
        # Calculate temporal metrics
        total_time = points[-1].timestamp - points[0].timestamp
        avg_time_per_step = total_time / (len(points) - 1) if len(points) > 1 else 0.0
        
        return {
            'total_distance': total_distance,
            'displacement': displacement,
            'tortuosity': tortuosity,
            'avg_step_size': avg_step_size,
            'step_variance': step_variance,
            'total_time': total_time,
            'avg_time_per_step': avg_time_per_step,
            'num_steps': len(points)
        }
    
    def generate_visualization_data(self, trajectory: CognitiveTrajectory) -> Dict[str, Any]:
        """
        Generate data for interactive visualization.
        
        Args:
            trajectory: Cognitive trajectory to visualize
            
        Returns:
            Dictionary containing visualization data
        """
        # Prepare point data
        points_data = []
        for i, point in enumerate(trajectory.projected_points):
            points_data.append({
                'x': point.x,
                'y': point.y,
                'step_id': point.step_id,
                'step_order': i,
                'timestamp': point.timestamp,
                'step_type': point.step_type,
                'component': point.component,
                'metadata': point.metadata
            })
        
        # Prepare trajectory path
        path_data = {
            'x': [p.x for p in trajectory.projected_points],
            'y': [p.y for p in trajectory.projected_points]
        }
        
        # Color mapping for different step types
        step_types = list(set(p.step_type for p in trajectory.projected_points))
        color_map = {step_type: i for i, step_type in enumerate(step_types)}
        
        # Component grouping
        components = list(set(p.component for p in trajectory.projected_points))
        component_map = {component: i for i, component in enumerate(components)}
        
        return {
            'session_id': trajectory.session_id,
            'projection_method': trajectory.projection_method,
            'original_dimension': trajectory.original_dimension,
            'explained_variance': trajectory.explained_variance,
            'points': points_data,
            'path': path_data,
            'metrics': trajectory.trajectory_metrics,
            'step_types': step_types,
            'components': components,
            'color_maps': {
                'step_types': color_map,
                'components': component_map
            }
        }
    
    def analyze_cognitive_patterns(self, trajectory: CognitiveTrajectory) -> Dict[str, Any]:
        """
        Analyze patterns in the cognitive trajectory.
        
        Args:
            trajectory: Cognitive trajectory to analyze
            
        Returns:
            Dictionary containing pattern analysis
        """
        points = trajectory.projected_points
        
        if len(points) < 3:
            return {'error': 'Insufficient points for pattern analysis'}
        
        # Cluster analysis by step type
        step_type_clusters = {}
        for point in points:
            if point.step_type not in step_type_clusters:
                step_type_clusters[point.step_type] = []
            step_type_clusters[point.step_type].append((point.x, point.y))
        
        # Calculate cluster centroids and spreads
        cluster_analysis = {}
        for step_type, coords in step_type_clusters.items():
            coords_array = np.array(coords)
            centroid = np.mean(coords_array, axis=0)
            spread = np.std(coords_array, axis=0)
            
            cluster_analysis[step_type] = {
                'centroid': centroid.tolist(),
                'spread': spread.tolist(),
                'count': len(coords)
            }
        
        # Identify potential loops or revisits
        loops = self._detect_loops(points)
        
        # Calculate cognitive complexity
        complexity_score = self._calculate_cognitive_complexity(trajectory)
        
        return {
            'cluster_analysis': cluster_analysis,
            'loops_detected': loops,
            'complexity_score': complexity_score,
            'trajectory_summary': {
                'total_steps': len(points),
                'unique_step_types': len(step_type_clusters),
                'spatial_extent': {
                    'x_range': [min(p.x for p in points), max(p.x for p in points)],
                    'y_range': [min(p.y for p in points), max(p.y for p in points)]
                }
            }
        }
    
    def _detect_loops(self, points: List[ProjectedPoint], threshold: float = 0.1) -> List[Dict]:
        """Detect potential loops in the trajectory."""
        loops = []
        
        for i in range(len(points)):
            for j in range(i + 3, len(points)):  # Minimum loop size of 3
                dist = np.sqrt(
                    (points[i].x - points[j].x)**2 + 
                    (points[i].y - points[j].y)**2
                )
                
                if dist < threshold:
                    loops.append({
                        'start_step': i,
                        'end_step': j,
                        'distance': dist,
                        'loop_length': j - i
                    })
        
        return loops
    
    def _calculate_cognitive_complexity(self, trajectory: CognitiveTrajectory) -> float:
        """Calculate a complexity score for the cognitive trajectory."""
        metrics = trajectory.trajectory_metrics
        
        # Combine multiple factors into complexity score
        complexity = 0.0
        
        # Tortuosity contributes to complexity
        if 'tortuosity' in metrics and metrics['tortuosity'] != float('inf'):
            complexity += min(metrics['tortuosity'] / 10.0, 1.0)  # Normalize
        
        # Step variance contributes to complexity
        if 'step_variance' in metrics:
            complexity += min(metrics['step_variance'] * 10.0, 1.0)  # Normalize
        
        # Number of unique step types
        unique_steps = len(set(p.step_type for p in trajectory.projected_points))
        complexity += min(unique_steps / 10.0, 1.0)  # Normalize
        
        return min(complexity, 1.0)  # Cap at 1.0
