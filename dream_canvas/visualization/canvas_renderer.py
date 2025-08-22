#!/usr/bin/env python3
"""
Dream Canvas Visualization Renderer
===================================

Handles the rendering and visualization of cognitive maps in the Dream Canvas.
Extracted from the monolithic dream_canvas.py.

This module provides:
- Interactive cognitive map visualization
- Cluster and connection rendering
- Plotly-based interactive charts
- Visualization controls and interactions

Author: SAM Development Team
Version: 1.0.0 - Refactored from dream_canvas.py
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional

from sam.dream_canvas.utils.models import (
    CognitiveMap, MemoryCluster, ClusterConnection, DreamCanvasState,
    VisualizationConfig, CONNECTION_COLORS
)

logger = logging.getLogger(__name__)


class CanvasRenderer:
    """Handles Dream Canvas visualization rendering."""
    
    def __init__(self):
        self.default_layout = {
            'width': 800,
            'height': 600,
            'showlegend': True,
            'hovermode': 'closest',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)'
        }
    
    def render_cognitive_map(self, cognitive_map: CognitiveMap, 
                           selected_cluster_id: Optional[str] = None) -> go.Figure:
        """
        Render the cognitive map as an interactive Plotly visualization.
        
        Args:
            cognitive_map: The cognitive map to render
            selected_cluster_id: ID of the currently selected cluster
            
        Returns:
            go.Figure: Plotly figure object
        """
        try:
            fig = go.Figure()
            
            # Add cluster connections first (so they appear behind clusters)
            self._add_connections_to_figure(fig, cognitive_map.connections, cognitive_map.clusters)
            
            # Add clusters
            self._add_clusters_to_figure(fig, cognitive_map.clusters, selected_cluster_id)
            
            # Update layout
            fig.update_layout(
                title=f"Cognitive Map - {cognitive_map.get_cluster_count()} Clusters, "
                      f"{cognitive_map.get_total_memories()} Memories",
                **self.default_layout,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error rendering cognitive map: {e}")
            return self._create_error_figure("Failed to render cognitive map")
    
    def _add_clusters_to_figure(self, fig: go.Figure, clusters: List[MemoryCluster], 
                              selected_cluster_id: Optional[str] = None):
        """Add clusters to the figure."""
        for cluster in clusters:
            # Determine if this cluster is selected
            is_selected = cluster.id == selected_cluster_id
            
            # Calculate marker size based on cluster size
            marker_size = min(max(cluster.size * 2, 20), 60)
            
            # Add cluster as scatter point
            fig.add_trace(go.Scatter(
                x=[cluster.center[0]],
                y=[cluster.center[1]],
                mode='markers+text',
                marker=dict(
                    size=marker_size,
                    color=cluster.color,
                    opacity=0.8 if not is_selected else 1.0,
                    line=dict(
                        width=3 if is_selected else 1,
                        color='black' if is_selected else 'white'
                    )
                ),
                text=[cluster.name],
                textposition="middle center",
                textfont=dict(
                    size=10,
                    color='white'
                ),
                hovertemplate=(
                    f"<b>{cluster.name}</b><br>"
                    f"Memories: {cluster.size}<br>"
                    f"Coherence: {cluster.coherence_score:.2f}<br>"
                    f"Keywords: {', '.join(cluster.keywords[:3])}<br>"
                    "<extra></extra>"
                ),
                name=cluster.name,
                customdata=[cluster.id],
                showlegend=False
            ))
    
    def _add_connections_to_figure(self, fig: go.Figure, connections: List[ClusterConnection], 
                                 clusters: List[MemoryCluster]):
        """Add connections between clusters to the figure."""
        # Create a mapping of cluster IDs to positions
        cluster_positions = {cluster.id: cluster.center for cluster in clusters}
        
        for connection in connections:
            source_pos = cluster_positions.get(connection.source_cluster_id)
            target_pos = cluster_positions.get(connection.target_cluster_id)
            
            if source_pos and target_pos:
                # Calculate line width based on connection strength
                line_width = max(connection.strength * 5, 1)
                
                # Get connection color
                color = CONNECTION_COLORS.get(connection.connection_type, '#CCCCCC')
                
                # Add connection line
                fig.add_trace(go.Scatter(
                    x=[source_pos[0], target_pos[0]],
                    y=[source_pos[1], target_pos[1]],
                    mode='lines',
                    line=dict(
                        width=line_width,
                        color=color,
                        dash='solid' if connection.strength > 0.5 else 'dash'
                    ),
                    opacity=0.6,
                    hovertemplate=(
                        f"<b>Connection</b><br>"
                        f"Strength: {connection.strength:.2f}<br>"
                        f"Type: {connection.connection_type}<br>"
                        f"Shared: {', '.join(connection.shared_concepts[:3])}<br>"
                        "<extra></extra>"
                    ),
                    showlegend=False
                ))
    
    def render_cluster_details(self, cluster: MemoryCluster) -> None:
        """Render detailed information about a selected cluster."""
        try:
            st.subheader(f"🔍 {cluster.name} Details")
            
            # Cluster metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Memories", cluster.size)
            
            with col2:
                st.metric("Coherence", f"{cluster.coherence_score:.2f}")
            
            with col3:
                st.metric("Keywords", len(cluster.keywords))
            
            with col4:
                st.metric("Position", f"({cluster.center[0]:.2f}, {cluster.center[1]:.2f})")
            
            # Keywords
            if cluster.keywords:
                st.markdown("**🏷️ Keywords:**")
                keyword_cols = st.columns(min(len(cluster.keywords), 5))
                for i, keyword in enumerate(cluster.keywords[:5]):
                    with keyword_cols[i]:
                        st.code(keyword)
            
            # Memory samples
            if cluster.memories:
                st.markdown("**📝 Memory Samples:**")
                
                # Show first few memories
                for i, memory in enumerate(cluster.memories[:3]):
                    with st.expander(f"Memory {i+1}", expanded=False):
                        content = memory.get('content', 'No content available')
                        st.write(content[:200] + "..." if len(content) > 200 else content)
                        
                        # Show metadata if available
                        if 'timestamp' in memory:
                            st.caption(f"Created: {memory['timestamp']}")
                        if 'source' in memory:
                            st.caption(f"Source: {memory['source']}")
            
        except Exception as e:
            logger.error(f"Error rendering cluster details: {e}")
            st.error("Failed to render cluster details")
    
    def render_map_statistics(self, cognitive_map: CognitiveMap) -> None:
        """Render statistics about the cognitive map."""
        try:
            st.subheader("📊 Map Statistics")
            
            stats = cognitive_map.get_summary_stats()
            
            # Main metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Memories", stats['total_memories'])
            
            with col2:
                st.metric("Clusters", stats['cluster_count'])
            
            with col3:
                st.metric("Connections", stats['connection_count'])
            
            with col4:
                st.metric("Avg Coherence", f"{stats['average_coherence']:.2f}")
            
            # Cluster size distribution
            cluster_sizes = [cluster.size for cluster in cognitive_map.clusters]
            if cluster_sizes:
                st.markdown("**Cluster Size Distribution:**")
                
                # Create histogram
                fig = px.histogram(
                    x=cluster_sizes,
                    nbins=10,
                    title="Distribution of Cluster Sizes",
                    labels={'x': 'Cluster Size', 'y': 'Count'}
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Coherence distribution
            coherence_scores = [cluster.coherence_score for cluster in cognitive_map.clusters]
            if coherence_scores:
                st.markdown("**Coherence Score Distribution:**")
                
                # Create box plot
                fig = px.box(
                    y=coherence_scores,
                    title="Distribution of Coherence Scores",
                    labels={'y': 'Coherence Score'}
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error rendering map statistics: {e}")
            st.error("Failed to render map statistics")
    
    def render_configuration_panel(self, config: VisualizationConfig) -> VisualizationConfig:
        """
        Render the configuration panel and return updated configuration.
        
        Args:
            config: Current configuration
            
        Returns:
            VisualizationConfig: Updated configuration
        """
        try:
            st.subheader("⚙️ Visualization Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Visualization method
                method = st.selectbox(
                    "Visualization Method",
                    options=list(VisualizationMethod),
                    index=list(VisualizationMethod).index(config.method),
                    format_func=lambda x: x.value.upper()
                )
                
                # Clustering method
                clustering_method = st.selectbox(
                    "Clustering Method",
                    options=list(ClusteringMethod),
                    index=list(ClusteringMethod).index(config.clustering_method),
                    format_func=lambda x: x.value.upper()
                )
                
                # Number of clusters
                n_clusters = st.slider(
                    "Number of Clusters",
                    min_value=2,
                    max_value=20,
                    value=config.n_clusters,
                    help="Target number of clusters to generate"
                )
            
            with col2:
                # Time range
                time_range = st.selectbox(
                    "Time Range",
                    options=list(TimeRange),
                    index=list(TimeRange).index(config.time_range),
                    format_func=lambda x: x.value.replace('_', ' ').title()
                )
                
                # Quality threshold
                quality_threshold = st.slider(
                    "Quality Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=config.quality_threshold,
                    step=0.1,
                    help="Minimum coherence score for clusters"
                )
                
                # Minimum cluster size
                min_cluster_size = st.slider(
                    "Minimum Cluster Size",
                    min_value=2,
                    max_value=20,
                    value=config.min_cluster_size,
                    help="Minimum number of memories per cluster"
                )
            
            # Advanced parameters
            with st.expander("🔧 Advanced Parameters", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    n_components = st.slider(
                        "Dimensions",
                        min_value=2,
                        max_value=3,
                        value=config.n_components,
                        help="Number of dimensions for visualization"
                    )
                    
                    perplexity = st.slider(
                        "Perplexity (t-SNE)",
                        min_value=5,
                        max_value=50,
                        value=config.perplexity,
                        help="Perplexity parameter for t-SNE"
                    )
                
                with col2:
                    n_neighbors = st.slider(
                        "Neighbors (UMAP)",
                        min_value=5,
                        max_value=50,
                        value=config.n_neighbors,
                        help="Number of neighbors for UMAP"
                    )
            
            # Create updated configuration
            updated_config = VisualizationConfig(
                method=method,
                clustering_method=clustering_method,
                time_range=time_range,
                n_components=n_components,
                n_clusters=n_clusters,
                min_cluster_size=min_cluster_size,
                quality_threshold=quality_threshold,
                perplexity=perplexity,
                n_neighbors=n_neighbors
            )
            
            return updated_config
            
        except Exception as e:
            logger.error(f"Error rendering configuration panel: {e}")
            st.error("Failed to render configuration panel")
            return config
    
    def _create_error_figure(self, error_message: str) -> go.Figure:
        """Create an error figure when visualization fails."""
        fig = go.Figure()
        
        fig.add_annotation(
            text=f"❌ {error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="red")
        )
        
        fig.update_layout(
            **self.default_layout,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig


# Global canvas renderer instance
_canvas_renderer = None


def get_canvas_renderer() -> CanvasRenderer:
    """Get the global canvas renderer instance."""
    global _canvas_renderer
    if _canvas_renderer is None:
        _canvas_renderer = CanvasRenderer()
    return _canvas_renderer


def render_cognitive_map(cognitive_map: CognitiveMap, 
                        selected_cluster_id: Optional[str] = None) -> go.Figure:
    """Render a cognitive map using the global renderer."""
    return get_canvas_renderer().render_cognitive_map(cognitive_map, selected_cluster_id)


def render_cluster_details(cluster: MemoryCluster) -> None:
    """Render cluster details using the global renderer."""
    get_canvas_renderer().render_cluster_details(cluster)


def render_map_statistics(cognitive_map: CognitiveMap) -> None:
    """Render map statistics using the global renderer."""
    get_canvas_renderer().render_map_statistics(cognitive_map)


def render_configuration_panel(config: VisualizationConfig) -> VisualizationConfig:
    """Render configuration panel using the global renderer."""
    return get_canvas_renderer().render_configuration_panel(config)
