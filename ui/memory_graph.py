"""
Memory Visualization (Graph View) for SAM
Interactive graph showing memory connections and reasoning chains.

Sprint 12 Task 3: Memory Visualization (Graph View)
"""

import logging
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
import math

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.memory_vectorstore import MemoryVectorStore, MemoryType, get_memory_store
from memory.memory_reasoning import MemoryDrivenReasoningEngine, get_memory_reasoning_engine

logger = logging.getLogger(__name__)

class MemoryGraphVisualizer:
    """
    Interactive graph visualization of memory connections and reasoning chains.
    """
    
    def __init__(self):
        """Initialize the memory graph visualizer."""
        self.memory_store = get_memory_store()
        self.memory_reasoning = get_memory_reasoning_engine()
        
        # Graph configuration
        self.config = {
            'max_nodes': 500,
            'similarity_threshold': 0.3,
            'edge_weight_threshold': 0.2,
            'layout_algorithm': 'spring',
            'node_size_factor': 20,
            'edge_width_factor': 5
        }
        
        # Color schemes
        self.memory_type_colors = {
            'document': '#1f77b4',
            'conversation': '#ff7f0e',
            'reasoning': '#2ca02c',
            'insight': '#d62728',
            'fact': '#9467bd',
            'procedure': '#8c564b'
        }
        
        self.importance_colors = {
            'low': '#cccccc',
            'medium': '#ffcc00',
            'high': '#ff6600',
            'critical': '#cc0000'
        }
    
    def render(self):
        """Render the complete memory graph interface."""
        st.subheader("📊 Memory Graph Visualization")
        st.markdown("Explore connections between memories and reasoning chains")
        
        # Graph controls
        self._render_graph_controls()
        
        # Generate and display graph
        if st.button("🔄 Generate Graph", type="primary"):
            with st.spinner("Building memory graph..."):
                graph_data = self._build_memory_graph()
                if graph_data:
                    self._render_interactive_graph(graph_data)
                else:
                    st.warning("No memories found or insufficient connections for graph visualization")
        
        # Graph statistics
        self._render_graph_statistics()
    
    def _render_graph_controls(self):
        """Render graph configuration controls."""
        with st.expander("🎛️ Graph Settings", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                self.config['max_nodes'] = st.slider(
                    "Max Nodes",
                    min_value=50,
                    max_value=1000,
                    value=self.config['max_nodes'],
                    step=50,
                    help="Maximum number of memory nodes to display"
                )
                
                self.config['similarity_threshold'] = st.slider(
                    "Similarity Threshold",
                    min_value=0.1,
                    max_value=0.9,
                    value=self.config['similarity_threshold'],
                    step=0.1,
                    help="Minimum similarity to create edges"
                )
            
            with col2:
                layout_options = ['spring', 'circular', 'random', 'shell']
                self.config['layout_algorithm'] = st.selectbox(
                    "Layout Algorithm",
                    options=layout_options,
                    index=layout_options.index(self.config['layout_algorithm']),
                    help="Graph layout algorithm"
                )
                
                color_scheme = st.selectbox(
                    "Color Scheme",
                    options=['Memory Type', 'Importance', 'Recency', 'User'],
                    index=0,
                    help="Node coloring scheme"
                )
            
            with col3:
                show_labels = st.checkbox("Show Node Labels", value=True)
                show_edge_weights = st.checkbox("Show Edge Weights", value=False)
                
                filter_by_user = st.text_input(
                    "Filter by User ID",
                    placeholder="Leave empty for all users",
                    help="Show only memories from specific user"
                )
        
        return color_scheme, show_labels, show_edge_weights, filter_by_user
    
    def _build_memory_graph(self) -> Optional[Dict[str, Any]]:
        """Build the memory graph data structure."""
        try:
            # Get memories
            memories = list(self.memory_store.memory_chunks.values())
            
            if not memories:
                return None
            
            # Limit number of memories
            if len(memories) > self.config['max_nodes']:
                # Sort by importance and recency
                memories.sort(key=lambda m: (m.importance_score, m.timestamp), reverse=True)
                memories = memories[:self.config['max_nodes']]
            
            # Create NetworkX graph
            G = nx.Graph()
            
            # Add nodes
            for memory in memories:
                G.add_node(
                    memory.chunk_id,
                    memory=memory,
                    label=self._create_node_label(memory),
                    size=self._calculate_node_size(memory),
                    color=self._get_node_color(memory, 'Memory Type')
                )
            
            # Add edges based on similarity
            self._add_similarity_edges(G, memories)
            
            # Add edges based on shared tags
            self._add_tag_edges(G, memories)
            
            # Add edges based on reasoning lineage
            self._add_reasoning_edges(G, memories)
            
            # Calculate layout
            layout = self._calculate_layout(G)
            
            return {
                'graph': G,
                'layout': layout,
                'memories': memories
            }
            
        except Exception as e:
            logger.error(f"Error building memory graph: {e}")
            st.error(f"Error building graph: {e}")
            return None
    
    def _render_interactive_graph(self, graph_data: Dict[str, Any]):
        """Render the interactive graph using Plotly."""
        try:
            G = graph_data['graph']
            layout = graph_data['layout']
            
            # Prepare node data
            node_x = []
            node_y = []
            node_text = []
            node_colors = []
            node_sizes = []
            node_ids = []
            
            for node_id in G.nodes():
                x, y = layout[node_id]
                node_x.append(x)
                node_y.append(y)
                
                node_data = G.nodes[node_id]
                memory = node_data['memory']
                
                # Create hover text
                hover_text = self._create_hover_text(memory)
                node_text.append(hover_text)
                
                node_colors.append(node_data['color'])
                node_sizes.append(node_data['size'])
                node_ids.append(node_id)
            
            # Prepare edge data
            edge_x = []
            edge_y = []
            edge_weights = []
            
            for edge in G.edges(data=True):
                x0, y0 = layout[edge[0]]
                x1, y1 = layout[edge[1]]
                
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_weights.append(edge[2].get('weight', 0.5))
            
            # Create Plotly figure
            fig = go.Figure()
            
            # Add edges
            fig.add_trace(go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines',
                line=dict(width=2, color='rgba(125, 125, 125, 0.5)'),
                hoverinfo='none',
                showlegend=False,
                name='Connections'
            ))
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    line=dict(width=2, color='white'),
                    opacity=0.8
                ),
                text=[G.nodes[node_id]['label'] for node_id in node_ids],
                textposition="middle center",
                textfont=dict(size=8, color='white'),
                hovertext=node_text,
                hoverinfo='text',
                showlegend=False,
                name='Memories'
            ))
            
            # Update layout
            fig.update_layout(
                title="Memory Connection Graph",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="Nodes represent memories, edges show connections",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor='left', yanchor='bottom',
                        font=dict(color='gray', size=12)
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white',
                height=600
            )
            
            # Display the graph
            st.plotly_chart(fig, use_container_width=True)
            
            # Graph insights
            self._render_graph_insights(G)
            
        except Exception as e:
            logger.error(f"Error rendering graph: {e}")
            st.error(f"Error rendering graph: {e}")
    
    def _render_graph_insights(self, G: nx.Graph):
        """Render insights about the graph structure."""
        st.subheader("🔍 Graph Insights")
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Nodes", G.number_of_nodes())
        
        with col2:
            st.metric("Edges", G.number_of_edges())
        
        with col3:
            if G.number_of_nodes() > 0:
                density = nx.density(G)
                st.metric("Density", f"{density:.3f}")
            else:
                st.metric("Density", "0")
        
        with col4:
            if G.number_of_nodes() > 0:
                avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
                st.metric("Avg Connections", f"{avg_degree:.1f}")
            else:
                st.metric("Avg Connections", "0")
        
        # Most connected memories
        if G.number_of_nodes() > 0:
            st.markdown("**Most Connected Memories:**")
            
            degree_centrality = nx.degree_centrality(G)
            top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for node_id, centrality in top_nodes:
                memory = G.nodes[node_id]['memory']
                st.markdown(f"• **{memory.memory_type.value}** - {memory.source} (connections: {centrality:.2f})")
        
        # Memory clusters
        if G.number_of_nodes() > 2:
            st.markdown("**Memory Clusters:**")
            
            try:
                communities = nx.community.greedy_modularity_communities(G)
                st.markdown(f"Found {len(communities)} clusters:")
                
                for i, community in enumerate(communities[:3]):  # Show top 3 clusters
                    cluster_types = [G.nodes[node]['memory'].memory_type.value for node in community]
                    type_counts = {}
                    for t in cluster_types:
                        type_counts[t] = type_counts.get(t, 0) + 1
                    
                    st.markdown(f"Cluster {i+1}: {len(community)} memories - {dict(type_counts)}")
                    
            except Exception as e:
                st.markdown("Cluster analysis not available")
    
    def _render_graph_statistics(self):
        """Render overall graph statistics."""
        st.subheader("📈 Memory Graph Statistics")
        
        try:
            stats = self.memory_store.get_memory_stats()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Memory Distribution:**")
                if stats['memory_types']:
                    # Create bar chart
                    fig = px.bar(
                        x=list(stats['memory_types'].keys()),
                        y=list(stats['memory_types'].values()),
                        title="Memories by Type",
                        color=list(stats['memory_types'].values()),
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Memory Timeline:**")
                
                # Get memory creation dates
                memories = list(self.memory_store.memory_chunks.values())
                if memories:
                    dates = [datetime.fromisoformat(m.timestamp).date() for m in memories]
                    
                    # Count memories by date
                    date_counts = {}
                    for date in dates:
                        date_counts[date] = date_counts.get(date, 0) + 1
                    
                    # Create timeline chart
                    if date_counts:
                        fig = px.line(
                            x=list(date_counts.keys()),
                            y=list(date_counts.values()),
                            title="Memory Creation Timeline"
                        )
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error loading statistics: {e}")
    
    def _create_node_label(self, memory) -> str:
        """Create a short label for a memory node."""
        # Use first few words of content or source
        if len(memory.content) > 20:
            return memory.content[:20] + "..."
        return memory.content
    
    def _calculate_node_size(self, memory) -> float:
        """Calculate node size based on importance and access count."""
        base_size = 10
        importance_factor = memory.importance_score * 20
        access_factor = min(memory.access_count * 2, 10)
        
        return base_size + importance_factor + access_factor
    
    def _get_node_color(self, memory, color_scheme: str) -> str:
        """Get node color based on the selected color scheme."""
        if color_scheme == 'Memory Type':
            return self.memory_type_colors.get(memory.memory_type.value, '#cccccc')
        
        elif color_scheme == 'Importance':
            if memory.importance_score >= 0.8:
                return self.importance_colors['critical']
            elif memory.importance_score >= 0.6:
                return self.importance_colors['high']
            elif memory.importance_score >= 0.4:
                return self.importance_colors['medium']
            else:
                return self.importance_colors['low']
        
        elif color_scheme == 'Recency':
            # Color by how recent the memory is
            memory_date = datetime.fromisoformat(memory.timestamp)
            days_old = (datetime.now() - memory_date).days
            
            if days_old <= 1:
                return '#00ff00'  # Green for very recent
            elif days_old <= 7:
                return '#ffff00'  # Yellow for recent
            elif days_old <= 30:
                return '#ff8800'  # Orange for older
            else:
                return '#ff0000'  # Red for old
        
        else:  # Default
            return '#1f77b4'
    
    def _create_hover_text(self, memory) -> str:
        """Create hover text for a memory node."""
        hover_parts = [
            f"<b>{memory.memory_type.value.title()}</b>",
            f"Source: {memory.source}",
            f"Created: {memory.timestamp[:10]}",
            f"Importance: {memory.importance_score:.2f}",
            f"Access Count: {memory.access_count}",
            "",
            f"Content: {memory.content[:100]}..."
        ]
        
        if memory.tags:
            hover_parts.append(f"Tags: {', '.join(memory.tags[:3])}")
        
        return "<br>".join(hover_parts)
    
    def _add_similarity_edges(self, G: nx.Graph, memories: List):
        """Add edges based on content similarity."""
        try:
            # Calculate similarities between memories
            for i, memory1 in enumerate(memories):
                for j, memory2 in enumerate(memories[i+1:], i+1):
                    if memory1.embedding and memory2.embedding:
                        # Calculate cosine similarity
                        similarity = self._cosine_similarity(memory1.embedding, memory2.embedding)
                        
                        if similarity >= self.config['similarity_threshold']:
                            G.add_edge(
                                memory1.chunk_id,
                                memory2.chunk_id,
                                weight=similarity,
                                edge_type='similarity',
                                label=f"Similarity: {similarity:.2f}"
                            )
                            
        except Exception as e:
            logger.error(f"Error adding similarity edges: {e}")
    
    def _add_tag_edges(self, G: nx.Graph, memories: List):
        """Add edges based on shared tags."""
        try:
            for i, memory1 in enumerate(memories):
                for j, memory2 in enumerate(memories[i+1:], i+1):
                    shared_tags = set(memory1.tags) & set(memory2.tags)
                    
                    if shared_tags:
                        # Weight based on number of shared tags
                        weight = len(shared_tags) / max(len(memory1.tags), len(memory2.tags), 1)
                        
                        if weight >= self.config['edge_weight_threshold']:
                            G.add_edge(
                                memory1.chunk_id,
                                memory2.chunk_id,
                                weight=weight,
                                edge_type='tags',
                                label=f"Shared tags: {', '.join(list(shared_tags)[:2])}"
                            )
                            
        except Exception as e:
            logger.error(f"Error adding tag edges: {e}")
    
    def _add_reasoning_edges(self, G: nx.Graph, memories: List):
        """Add edges based on reasoning lineage."""
        try:
            # Look for memories that reference each other in metadata
            for memory1 in memories:
                for memory2 in memories:
                    if memory1.chunk_id != memory2.chunk_id:
                        # Check if one memory references another
                        if (memory1.metadata.get('parent_memory') == memory2.chunk_id or
                            memory2.chunk_id in memory1.metadata.get('related_memories', [])):
                            
                            G.add_edge(
                                memory1.chunk_id,
                                memory2.chunk_id,
                                weight=0.8,
                                edge_type='reasoning',
                                label="Reasoning connection"
                            )
                            
        except Exception as e:
            logger.error(f"Error adding reasoning edges: {e}")
    
    def _calculate_layout(self, G: nx.Graph) -> Dict[str, Tuple[float, float]]:
        """Calculate node positions using the selected layout algorithm."""
        try:
            if self.config['layout_algorithm'] == 'spring':
                return nx.spring_layout(G, k=1, iterations=50)
            elif self.config['layout_algorithm'] == 'circular':
                return nx.circular_layout(G)
            elif self.config['layout_algorithm'] == 'random':
                return nx.random_layout(G)
            elif self.config['layout_algorithm'] == 'shell':
                return nx.shell_layout(G)
            else:
                return nx.spring_layout(G)
                
        except Exception as e:
            logger.error(f"Error calculating layout: {e}")
            return nx.random_layout(G)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0

def main():
    """Main function to run the memory graph visualizer."""
    visualizer = MemoryGraphVisualizer()
    visualizer.render()

if __name__ == "__main__":
    main()
