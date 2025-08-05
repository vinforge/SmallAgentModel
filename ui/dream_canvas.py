"""
Dream Canvas - Cognitive Synthesis Visualization
Interactive memory landscape with UMAP projections and cluster analysis
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import random
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)

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

@dataclass
class CognitiveMap:
    """Represents the cognitive map visualization."""
    clusters: List[MemoryCluster]
    connections: List[Dict[str, Any]]
    metadata: Dict[str, Any]

def render_dream_canvas():
    """Main function to render the Dream Canvas interface."""
    st.subheader("🧠🎨 Dream Canvas - Cognitive Synthesis Visualization")
    st.markdown("*Interactive memory landscape with UMAP projections and cluster analysis*")
    
    # Check if memory store is available
    try:
        from memory.memory_vectorstore import get_memory_store
        memory_store = get_memory_store()
    except ImportError:
        st.error("❌ Memory store not available")
        return
    except Exception as e:
        st.error(f"❌ Error accessing memory store: {e}")
        return
    
    # Dream Canvas controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        visualization_mode = st.selectbox(
            "🎨 Visualization Mode",
            ["Cognitive Landscape", "Memory Clusters", "Temporal Flow", "Concept Networks"],
            help="Select the type of cognitive visualization"
        )
    
    with col2:
        cluster_method = st.selectbox(
            "🔬 Clustering Method",
            ["UMAP + HDBSCAN", "t-SNE + K-Means", "PCA + Gaussian Mixture"],
            help="Choose the dimensionality reduction and clustering approach"
        )
    
    with col3:
        time_range = st.selectbox(
            "⏰ Time Range",
            ["All Time", "Last 30 Days", "Last 7 Days", "Last 24 Hours"],
            help="Filter memories by time period"
        )
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        col1, col2, col3 = st.columns(3)
        with col1:
            n_components = st.slider("Dimensions", 2, 3, 2, help="2D or 3D visualization")
            min_cluster_size = st.slider("Min Cluster Size", 3, 20, 5, help="Minimum memories per cluster")
        with col2:
            perplexity = st.slider("Perplexity", 5, 50, 30, help="t-SNE perplexity parameter")
            n_neighbors = st.slider("Neighbors", 5, 50, 15, help="UMAP n_neighbors parameter")
        with col3:
            show_connections = st.checkbox("Show Connections", True, help="Display memory connections")
            show_labels = st.checkbox("Show Labels", True, help="Display cluster labels")

    # Advanced Synthesis Controls - Additional options (less prominent)
    with st.expander("⚙️ Advanced Synthesis Controls", expanded=False):
        st.markdown("*Configure auto-synthesis, view history, and adjust synthesis settings*")

        # Synthesis Configuration
        st.markdown("#### 🎛️ Synthesis Configuration")
        col1, col2 = st.columns(2)

        with col1:
            insight_threshold = st.slider(
                "Insight Quality Threshold",
                min_value=0.1,
                max_value=0.8,
                value=0.3,
                step=0.1,
                help="Lower values accept more insights but may reduce quality. Higher values are more selective."
            )

        with col2:
            max_clusters = st.slider(
                "Maximum Clusters",
                min_value=5,
                max_value=50,
                value=20,
                step=5,
                help="Maximum number of memory clusters to analyze for synthesis."
            )

        # Store configuration in session state
        st.session_state.synthesis_config = {
            'min_insight_quality': insight_threshold,
            'max_clusters': max_clusters
        }

        st.markdown("---")

        try:
            render_synthesis_controls()
        except Exception as e:
            # Fallback synthesis controls if main function fails
            st.warning(f"⚠️ Advanced synthesis controls unavailable: {e}")
            st.info("💡 Use the main 'Run Synthesis' button above for basic synthesis functionality.")

    # Main action buttons - Dream Canvas and Synthesis side by side
    st.markdown("---")
    st.markdown("### 🚀 Main Actions")

    col1, col2 = st.columns(2)

    with col1:
        # Generate cognitive map (preserved existing functionality)
        if st.button("🎨 Generate Dream Canvas", type="primary", help="Create interactive memory landscape visualization"):
            with st.spinner("🧠 Synthesizing cognitive landscape..."):
                try:
                    # Get memory data
                    memory_stats = memory_store.get_memory_stats()
                    st.info(f"📊 Processing {memory_stats['total_memories']} memories...")

                    # Generate cognitive map
                    cognitive_map = generate_cognitive_map(
                        memory_store=memory_store,
                        method=cluster_method,
                        time_range=time_range,
                        n_components=n_components,
                        min_cluster_size=min_cluster_size,
                        perplexity=perplexity,
                        n_neighbors=n_neighbors
                    )

                    # Store in session state
                    st.session_state.cognitive_map = cognitive_map
                    st.session_state.visualization_mode = visualization_mode
                    st.session_state.show_connections = show_connections
                    st.session_state.show_labels = show_labels

                    st.success("✅ Cognitive landscape generated!")

                except Exception as e:
                    st.error(f"❌ Error generating cognitive map: {e}")
                    logger.error(f"Dream Canvas generation error: {e}")
                    return

    with col2:
        # Run Synthesis button - prominently placed next to Dream Canvas
        if st.button("🔄 Run Synthesis", type="secondary", help="Generate new insights from memory clusters"):
            with st.spinner("🌙 SAM entering dream state..."):
                try:
                    # Import synthesis engine
                    from memory.synthesis.synthesis_engine import SynthesisEngine
                    from memory.memory_vectorstore import get_memory_store

                    # Run synthesis
                    memory_store = get_memory_store()

                    # Use custom configuration if available
                    if hasattr(st.session_state, 'synthesis_config'):
                        from memory.synthesis.synthesis_engine import SynthesisConfig
                        config = SynthesisConfig(
                            min_insight_quality=st.session_state.synthesis_config.get('min_insight_quality', 0.3),
                            max_clusters=st.session_state.synthesis_config.get('max_clusters', 20)
                        )
                        synthesis_engine = SynthesisEngine(config=config)
                        st.info(f"🎛️ Using custom settings: Quality threshold {config.min_insight_quality}, Max clusters {config.max_clusters}")
                    else:
                        synthesis_engine = SynthesisEngine()

                    result = synthesis_engine.run_synthesis(memory_store, visualize=True)

                    # Store results in session state
                    synthesis_results = {
                        'insights': [insight.__dict__ for insight in result.insights],
                        'clusters_found': result.clusters_found,
                        'insights_generated': result.insights_generated,
                        'run_id': result.run_id,
                        'timestamp': result.timestamp,
                        'synthesis_log': result.synthesis_log
                    }

                    st.session_state.synthesis_results = synthesis_results

                    # Update synthesis history
                    update_synthesis_history(synthesis_results)

                    # Provide detailed feedback based on results
                    if result.insights_generated > 0:
                        st.success(f"✨ Synthesis complete! Generated **{result.insights_generated} insights** from **{result.clusters_found} clusters**.")
                        st.info("💡 Use 'View Synthesis Results' below to explore the generated insights.")
                    elif result.clusters_found > 0:
                        st.warning(f"⚠️ Found **{result.clusters_found} clusters** but generated **0 insights**. This may indicate:")
                        st.info("• Insight quality threshold is too high\n• LLM responses need improvement\n• Memory clusters lack sufficient content")
                        st.info("💡 Try running synthesis again or check the logs for more details.")
                    else:
                        st.warning("⚠️ No memory clusters found for synthesis. Try adding more conversations or documents to SAM's memory.")

                except Exception as e:
                    logger.error(f"Synthesis failed: {e}")
                    st.error(f"❌ Synthesis failed: {e}")
                    st.info("🔧 Try using the advanced synthesis controls below for more options.")
    
    # Check if we have focused synthesis visualization data
    if hasattr(st.session_state, 'dream_canvas_data') and st.session_state.dream_canvas_data:
        # Render focused synthesis visualization
        render_focused_synthesis_visualization(st.session_state.dream_canvas_data)

        # NEW: Synthetic Insights Integration - Display emergent patterns and new understanding
        render_synthetic_insights_integration()

        # NEW: Deep Research Results - Display comprehensive ArXiv analysis reports
        render_deep_research_results()

    # Display cognitive map if available
    elif hasattr(st.session_state, 'cognitive_map') and st.session_state.cognitive_map:
        render_cognitive_visualization(
            cognitive_map=st.session_state.cognitive_map,
            mode=st.session_state.get('visualization_mode', 'Cognitive Landscape'),
            show_connections=st.session_state.get('show_connections', True),
            show_labels=st.session_state.get('show_labels', True)
        )

        # Cognitive insights (preserved existing functionality)
        render_cognitive_insights(st.session_state.cognitive_map)

        # NEW: Synthetic Insights Integration - Display emergent patterns and new understanding
        render_synthetic_insights_integration()

        # NEW: Deep Research Results - Display comprehensive ArXiv analysis reports
        render_deep_research_results()

    else:
        # Show placeholder
        render_dream_canvas_placeholder()

def generate_cognitive_map(
    memory_store,
    method: str,
    time_range: str,
    n_components: int = 2,
    min_cluster_size: int = 5,
    perplexity: int = 30,
    n_neighbors: int = 15
) -> CognitiveMap:
    """Generate a cognitive map from memory data."""
    
    # For now, generate mock data since we need to implement the actual memory processing
    # In a real implementation, this would:
    # 1. Query memories from the store based on time_range
    # 2. Extract embeddings/features from memories
    # 3. Apply dimensionality reduction (UMAP/t-SNE/PCA)
    # 4. Perform clustering (HDBSCAN/K-Means/Gaussian Mixture)
    # 5. Generate connections between related memories
    
    logger.info(f"Generating cognitive map with method: {method}, time_range: {time_range}")
    
    # Mock cluster generation
    clusters = []
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
    
    cluster_names = [
        "Personal Experiences", "Technical Knowledge", "Creative Ideas", 
        "Problem Solving", "Relationships", "Learning", "Goals & Plans", "Reflections"
    ]
    
    for i, name in enumerate(cluster_names[:6]):  # Limit to 6 clusters for demo
        # Generate random cluster center
        center = (
            random.uniform(-10, 10),
            random.uniform(-10, 10)
        )
        
        # Generate mock memories for this cluster
        memories = []
        cluster_size = random.randint(5, 25)
        
        for j in range(cluster_size):
            memory = {
                'id': f'mem_{i}_{j}',
                'content': f'Memory {j+1} in {name} cluster',
                'timestamp': datetime.now() - timedelta(days=random.randint(0, 365)),
                'embedding': [random.uniform(-1, 1) for _ in range(384)],  # Mock embedding
                'metadata': {
                    'type': name.lower().replace(' ', '_'),
                    'confidence': random.uniform(0.7, 1.0)
                }
            }
            memories.append(memory)
        
        cluster = MemoryCluster(
            id=f'cluster_{i}',
            name=name,
            memories=memories,
            center=center,
            color=colors[i % len(colors)],
            size=cluster_size,
            coherence_score=random.uniform(0.6, 0.95)
        )
        clusters.append(cluster)
    
    # Generate mock connections
    connections = []
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            if random.random() < 0.4:  # 40% chance of connection
                connection = {
                    'source': clusters[i].id,
                    'target': clusters[j].id,
                    'strength': random.uniform(0.3, 0.8),
                    'type': 'semantic_similarity'
                }
                connections.append(connection)
    
    # Create cognitive map
    cognitive_map = CognitiveMap(
        clusters=clusters,
        connections=connections,
        metadata={
            'method': method,
            'time_range': time_range,
            'n_components': n_components,
            'generated_at': datetime.now().isoformat(),
            'total_memories': sum(len(c.memories) for c in clusters)
        }
    )
    
    return cognitive_map

def render_focused_synthesis_visualization(visualization_data):
    """Render focused synthesis visualization showing cluster memories and insights."""

    st.markdown("### 🎨 Focused Synthesis Landscape")
    st.markdown("*Showing cluster memories and generated insights*")

    if not visualization_data:
        st.warning("No visualization data available")
        return

    # Create plotly figure
    fig = go.Figure()

    # Separate source memories and insights
    source_memories = [item for item in visualization_data if item.get('memory_type') == 'source_memory']
    synthetic_insights = [item for item in visualization_data if item.get('memory_type') == 'synthetic_insight']

    # Plot source memories by cluster
    cluster_colors = {}
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for i, cluster_id in enumerate(set(item.get('cluster_id', 'unknown') for item in source_memories)):
        cluster_memories = [item for item in source_memories if item.get('cluster_id') == cluster_id]

        if cluster_memories:
            color = color_palette[i % len(color_palette)]
            cluster_colors[cluster_id] = color

            x_coords = [item.get('coordinates', {}).get('x', item.get('x', 0)) for item in cluster_memories]
            y_coords = [item.get('coordinates', {}).get('y', item.get('y', 0)) for item in cluster_memories]
            hover_texts = [f"Source: {item['source']}<br>Content: {item['content'][:100]}..." for item in cluster_memories]

            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers',
                marker=dict(
                    size=8,
                    color=color,
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                hovertemplate='%{text}<extra></extra>',
                hovertext=hover_texts,
                name=f'Cluster {cluster_id}',
                showlegend=True
            ))

    # Plot synthetic insights as golden stars
    if synthetic_insights:
        x_coords = [item.get('coordinates', {}).get('x', item.get('x', 0)) for item in synthetic_insights]
        y_coords = [item.get('coordinates', {}).get('y', item.get('y', 0)) for item in synthetic_insights]
        hover_texts = [
            f"Insight: Cluster {item.get('cluster_id', 'Unknown')}<br>"
            f"Confidence: {item.get('confidence_score', 0):.2f}<br>"
            f"Content: {item['content'][:150]}..."
            for item in synthetic_insights
        ]

        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker=dict(
                symbol='star',
                size=15,
                color='gold',
                line=dict(width=2, color='orange'),
                opacity=0.9
            ),
            hovertemplate='%{text}<extra></extra>',
            hovertext=hover_texts,
            name='Synthetic Insights',
            showlegend=True
        ))

    # Update layout
    fig.update_layout(
        title=f"🧠✨ Focused Synthesis: {len(source_memories)} Memories + {len(synthetic_insights)} Insights",
        xaxis_title="Cognitive Dimension 1",
        yaxis_title="Cognitive Dimension 2",
        showlegend=True,
        height=600,
        plot_bgcolor='rgba(0,0,0,0.05)',
        paper_bgcolor='white'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Source Memories", len(source_memories))
    with col2:
        st.metric("Synthetic Insights", len(synthetic_insights))
    with col3:
        unique_clusters = len(set(item.get('cluster_id', 'unknown') for item in visualization_data))
        st.metric("Clusters", unique_clusters)

def render_cognitive_visualization(
    cognitive_map: CognitiveMap,
    mode: str,
    show_connections: bool = True,
    show_labels: bool = True
):
    """Render the cognitive visualization."""

    st.markdown("### 🎨 Cognitive Landscape")

    if mode == "Cognitive Landscape":
        render_landscape_view(cognitive_map, show_connections, show_labels)
    elif mode == "Memory Clusters":
        render_cluster_view(cognitive_map)
    elif mode == "Temporal Flow":
        render_temporal_view(cognitive_map)
    elif mode == "Concept Networks":
        render_network_view(cognitive_map, show_connections)

def render_landscape_view(cognitive_map: CognitiveMap, show_connections: bool, show_labels: bool):
    """Render the main landscape visualization with synthetic insights as golden stars."""

    # Create scatter plot data
    x_coords = []
    y_coords = []
    colors = []
    sizes = []
    texts = []

    for cluster in cognitive_map.clusters:
        x_coords.append(cluster.center[0])
        y_coords.append(cluster.center[1])
        colors.append(cluster.color)
        sizes.append(cluster.size * 3)  # Scale for visibility
        texts.append(f"{cluster.name}<br>Memories: {cluster.size}<br>Coherence: {cluster.coherence_score:.2f}")

    # Create the plot
    fig = go.Figure()

    # Add clusters (preserved existing functionality)
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='markers+text' if show_labels else 'markers',
        marker=dict(
            size=sizes,
            color=colors,
            opacity=0.7,
            line=dict(width=2, color='white')
        ),
        text=[cluster.name for cluster in cognitive_map.clusters] if show_labels else None,
        textposition="middle center",
        textfont=dict(size=10, color='white'),
        hovertemplate='%{text}<extra></extra>',
        hovertext=texts,
        name='Memory Clusters'
    ))
    
    # Add connections if enabled
    if show_connections:
        for connection in cognitive_map.connections:
            source_cluster = next(c for c in cognitive_map.clusters if c.id == connection['source'])
            target_cluster = next(c for c in cognitive_map.clusters if c.id == connection['target'])
            
            fig.add_trace(go.Scatter(
                x=[source_cluster.center[0], target_cluster.center[0]],
                y=[source_cluster.center[1], target_cluster.center[1]],
                mode='lines',
                line=dict(
                    width=connection['strength'] * 5,
                    color='rgba(128, 128, 128, 0.3)'
                ),
                hoverinfo='skip',
                showlegend=False
            ))

    # NEW: Add synthetic insights as golden stars
    if hasattr(st.session_state, 'synthesis_results') and st.session_state.synthesis_results:
        insights = st.session_state.synthesis_results.get('insights', [])

        if insights:
            insight_x = []
            insight_y = []
            insight_texts = []

            for insight in insights:
                cluster_id = insight.get('cluster_id', '')
                # Find corresponding cluster
                matching_cluster = None
                for cluster in cognitive_map.clusters:
                    if cluster.name == cluster_id or cluster.id == cluster_id:
                        matching_cluster = cluster
                        break

                if matching_cluster:
                    # Position insight star slightly offset from cluster center
                    offset_x = matching_cluster.center[0] + random.uniform(-0.1, 0.1)
                    offset_y = matching_cluster.center[1] + random.uniform(-0.1, 0.1)

                    insight_x.append(offset_x)
                    insight_y.append(offset_y)

                    confidence = insight.get('confidence_score', 0)
                    novelty = insight.get('novelty_score', 0)
                    insight_texts.append(f"✨ Synthetic Insight<br>Confidence: {confidence:.2f}<br>Novelty: {novelty:.2f}")

            if insight_x:  # Only add if we have insights to display
                fig.add_trace(go.Scatter(
                    x=insight_x,
                    y=insight_y,
                    mode='markers',
                    marker=dict(
                        symbol='star',
                        size=15,
                        color='gold',
                        line=dict(width=2, color='orange'),
                        opacity=0.9
                    ),
                    hovertemplate='%{text}<extra></extra>',
                    hovertext=insight_texts,
                    name='Synthetic Insights',
                    showlegend=True
                ))

    # Update layout (preserved existing functionality)
    fig.update_layout(
        title="🧠🎨 Cognitive Memory Landscape with Synthetic Insights",
        xaxis_title="Cognitive Dimension 1",
        yaxis_title="Cognitive Dimension 2",
        showlegend=True,  # Changed to True to show insights legend
        height=600,
        plot_bgcolor='rgba(0,0,0,0.05)',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_cluster_view(cognitive_map: CognitiveMap):
    """Render cluster analysis view."""
    
    # Cluster statistics
    cluster_data = []
    for cluster in cognitive_map.clusters:
        cluster_data.append({
            'Cluster': cluster.name,
            'Memories': cluster.size,
            'Coherence': cluster.coherence_score,
            'Color': cluster.color
        })
    
    df = pd.DataFrame(cluster_data)
    
    # Bar chart of cluster sizes
    fig = px.bar(
        df, 
        x='Cluster', 
        y='Memories',
        color='Coherence',
        title="📊 Memory Cluster Analysis",
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster details table
    st.markdown("### 📋 Cluster Details")
    st.dataframe(df[['Cluster', 'Memories', 'Coherence']], use_container_width=True)

def render_temporal_view(cognitive_map: CognitiveMap):
    """Render temporal flow visualization."""
    st.info("🚧 Temporal Flow visualization coming soon!")
    
    # Placeholder for temporal visualization
    # This would show how memories flow and connect over time

def render_network_view(cognitive_map: CognitiveMap, show_connections: bool):
    """Render network graph visualization."""
    st.info("🚧 Concept Networks visualization coming soon!")
    
    # Placeholder for network graph
    # This would show memories as nodes and connections as edges

def render_cognitive_insights(cognitive_map: CognitiveMap):
    """Render cognitive insights and analysis with synthesis integration."""

    st.markdown("### 🧠 Cognitive Insights & Analysis")

    # Basic cognitive metrics (preserved existing functionality)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_memories = sum(len(c.memories) for c in cognitive_map.clusters)
        st.metric("📚 Total Memories", total_memories)

    with col2:
        avg_coherence = np.mean([c.coherence_score for c in cognitive_map.clusters])
        st.metric("🎯 Avg Coherence", f"{avg_coherence:.2f}")

    with col3:
        st.metric("🧩 Clusters", len(cognitive_map.clusters))

    with col4:
        st.metric("🔗 Connections", len(cognitive_map.connections))

    # NEW: Synthesis integration metrics
    if hasattr(st.session_state, 'synthesis_results') and st.session_state.synthesis_results:
        insights = st.session_state.synthesis_results.get('insights', [])

        if insights:
            st.markdown("---")
            st.markdown("#### ✨ Synthesis Integration Status")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("✨ Insights Generated", len(insights))

            with col2:
                avg_confidence = sum(i.get('confidence_score', 0) for i in insights) / len(insights)
                st.metric("🎯 Avg Confidence", f"{avg_confidence:.2f}")

            with col3:
                high_quality = sum(1 for i in insights if i.get('confidence_score', 0) > 0.4)
                st.metric("⭐ High Quality", f"{high_quality}/{len(insights)}")

            with col4:
                cross_domain = sum(1 for i in insights
                                 if len(set(i.get('synthesis_metadata', {}).get('source_types', []))) > 1)
                st.metric("🔗 Cross-Domain", cross_domain)

    # Top clusters with expandable detailed information
    st.markdown("#### 🏆 Most Coherent Clusters")
    top_clusters = sorted(cognitive_map.clusters, key=lambda x: x.coherence_score, reverse=True)[:3]

    for i, cluster in enumerate(top_clusters, 1):
        # Create cluster title with basic info
        cluster_title = f"{i}. {cluster.name} - {cluster.size} memories (coherence: {cluster.coherence_score:.2f})"

        # Add insight indicator if this cluster has generated insights
        insight_indicator = ""
        cluster_insights = []
        if hasattr(st.session_state, 'synthesis_results') and st.session_state.synthesis_results:
            insights = st.session_state.synthesis_results.get('insights', [])
            cluster_insights = [ins for ins in insights if ins.get('cluster_id') == cluster.name or ins.get('cluster_id') == cluster.id]

            if cluster_insights:
                insight_indicator = f" ✨ ({len(cluster_insights)} insights)"

        # Create expandable section for each cluster
        with st.expander(f"🔍 {cluster_title}{insight_indicator}", expanded=False):
            # Display detailed cluster information
            render_cluster_detailed_info(cluster, cluster_insights)

def render_cluster_detailed_info(cluster, cluster_insights):
    """Render detailed information for a cluster in the Most Coherent Clusters section."""
    try:
        from memory.synthesis.cluster_registry import get_cluster_stats

        # Get cluster statistics from registry
        cluster_stats = get_cluster_stats(cluster.name)

        # Display cluster overview
        st.markdown("**📊 Cluster Overview:**")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Memories", cluster.size)

        with col2:
            st.metric("Coherence", f"{cluster.coherence_score:.2f}")

        with col3:
            if cluster_insights:
                st.metric("Insights", len(cluster_insights))
            else:
                st.metric("Insights", "0")

        # Display detailed cluster information if available from registry
        if cluster_stats['exists']:
            st.markdown("---")
            st.markdown("**🔍 Detailed Information:**")

            # Memory and source information
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Memory Details:**")
                st.write(f"• **Total memories:** {cluster_stats['memory_count']}")
                st.write(f"• **Average importance:** {cluster_stats['avg_importance']:.2f}")

            with col2:
                st.markdown("**Source Information:**")
                st.write(f"• **Unique sources:** {cluster_stats['source_count']}")
                if cluster_stats['sources']:
                    st.write("• **Key sources:**")
                    for source in cluster_stats['sources'][:3]:
                        source_name = source.split('/')[-1] if '/' in source else source
                        st.write(f"  - {source_name}")

            # Themes and topics
            if cluster_stats['dominant_themes']:
                st.markdown("**🏷️ Dominant Themes:**")
                themes_text = ", ".join(cluster_stats['dominant_themes'][:5])
                st.write(f"• {themes_text}")
        else:
            # Fallback to basic cluster information
            st.markdown("---")
            st.markdown("**🔍 Basic Information:**")
            st.write(f"• **Cluster ID:** {cluster.name}")
            st.write(f"• **Memory count:** {cluster.size}")
            st.write(f"• **Coherence score:** {cluster.coherence_score:.2f}")
            if hasattr(cluster, 'color'):
                st.write(f"• **Visualization color:** {cluster.color}")

        # Display insights if available
        if cluster_insights:
            st.markdown("---")
            st.markdown("**✨ Generated Insights:**")
            for j, insight in enumerate(cluster_insights[:3]):  # Show up to 3 insights per cluster
                with st.expander(f"💡 Insight {j+1}", expanded=j==0):
                    # Clean the insight text (remove <think> tags)
                    clean_text = insight.get('synthesized_text', '')
                    if '<think>' in clean_text and '</think>' in clean_text:
                        parts = clean_text.split('</think>')
                        if len(parts) > 1:
                            clean_text = parts[-1].strip()
                        else:
                            clean_text = clean_text.replace('<think>', '').replace('</think>', '').strip()

                    st.markdown(f"**Insight:** {clean_text}")

                    # Show insight metadata
                    if insight.get('confidence_score'):
                        st.write(f"**Confidence:** {insight['confidence_score']:.2f}")
                    if insight.get('novelty_score'):
                        st.write(f"**Novelty:** {insight['novelty_score']:.2f}")

            if len(cluster_insights) > 3:
                st.info(f"📋 Showing 3 of {len(cluster_insights)} insights for this cluster")

    except Exception as e:
        st.error(f"❌ Error loading cluster details: {e}")
        # Fallback display
        st.markdown("**Basic Cluster Information:**")
        st.write(f"• **Name:** {cluster.name}")
        st.write(f"• **Size:** {cluster.size} memories")
        st.write(f"• **Coherence:** {cluster.coherence_score:.2f}")

def render_dream_canvas_placeholder():
    """Render placeholder when no cognitive map is available."""
    
    st.markdown("### 🎨 Welcome to Dream Canvas")
    
    st.markdown("""
    **Dream Canvas** is your cognitive synthesis visualization tool that transforms your memory landscape 
    into an interactive, visual representation of your knowledge and experiences.
    
    #### 🌟 Features:
    - **Cognitive Landscape**: 2D/3D visualization of memory clusters
    - **Memory Clusters**: Automatic grouping of related memories
    - **Temporal Flow**: Time-based memory evolution
    - **Concept Networks**: Semantic relationship mapping
    
    #### 🚀 Getting Started:
    1. Configure your visualization preferences above
    2. Click "🎨 Generate Dream Canvas" to create your cognitive map
    3. Explore your memory landscape interactively
    
    *Your memories will be processed using advanced dimensionality reduction and clustering algorithms 
    to reveal hidden patterns and connections in your knowledge.*
    """)
    
    # Sample visualization
    st.markdown("#### 📊 Sample Cognitive Landscape")
    
    # Create a sample plot
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'x': np.random.randn(50),
        'y': np.random.randn(50),
        'cluster': np.random.choice(['Ideas', 'Knowledge', 'Experiences', 'Goals'], 50),
        'size': np.random.randint(5, 20, 50)
    })
    
    fig = px.scatter(
        sample_data, 
        x='x', 
        y='y', 
        color='cluster',
        size='size',
        title="Sample Memory Landscape",
        opacity=0.7
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# NEW: SYNTHETIC INSIGHTS INTEGRATION - Emergent Patterns & New Understanding
# ============================================================================

def render_synthesis_controls():
    """Render cognitive synthesis controls for generating new understanding."""
    try:
        st.markdown("### 🌙 Cognitive Synthesis Controls")
        st.markdown("*Generate emergent insights and discover new understanding from memory patterns*")

        # Auto-synthesis controls (with error handling)
        try:
            render_auto_synthesis_controls()
        except Exception as e:
            st.warning(f"⚠️ Auto-synthesis controls unavailable: {e}")
            logger.warning(f"Auto-synthesis controls error: {e}")

        st.markdown("---")

        # Manual synthesis controls
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔄 Run Synthesis", type="primary", help="Generate new insights from memory clusters"):
                with st.spinner("🌙 SAM entering dream state..."):
                    try:
                        # Import synthesis engine
                        from memory.synthesis.synthesis_engine import SynthesisEngine
                        from memory.memory_vectorstore import get_memory_store

                        # Run synthesis
                        memory_store = get_memory_store()
                        synthesis_engine = SynthesisEngine()

                        result = synthesis_engine.run_synthesis(memory_store, visualize=True)

                        # Store results in session state
                        synthesis_results = {
                            'insights': [insight.__dict__ for insight in result.insights],
                            'clusters_found': result.clusters_found,
                            'insights_generated': result.insights_generated,
                            'run_id': result.run_id,
                            'timestamp': result.timestamp,
                            'synthesis_log': result.synthesis_log
                        }

                        st.session_state.synthesis_results = synthesis_results

                        # Update synthesis history
                        update_synthesis_history(synthesis_results)

                        # Provide detailed feedback based on results
                        if result.insights_generated > 0:
                            st.success(f"✨ Synthesis complete! Generated {result.insights_generated} insights from {result.clusters_found} clusters.")
                        elif result.clusters_found > 0:
                            st.warning(f"⚠️ Found {result.clusters_found} clusters but generated 0 insights. This may indicate:")
                            st.info("• Insight quality threshold is too high\n• LLM responses need improvement\n• Memory clusters lack sufficient content")
                            st.info("💡 Try running synthesis again or check the logs for more details.")
                        else:
                            st.warning("⚠️ No memory clusters found for synthesis. Try adding more conversations or documents to SAM's memory.")

                        st.rerun()

                    except Exception as e:
                        logger.error(f"Synthesis failed: {e}")
                        st.error(f"❌ Synthesis failed: {e}")

        with col2:
            if st.button("📊 Load Recent", help="Load the most recent synthesis results"):
                try:
                    # Load most recent synthesis results
                    synthesis_dir = Path("synthesis_output")
                    if synthesis_dir.exists():
                        synthesis_files = list(synthesis_dir.glob("synthesis_run_log_*.json"))
                        if synthesis_files:
                            latest_file = max(synthesis_files, key=lambda x: x.stat().st_mtime)

                            with open(latest_file, 'r') as f:
                                data = json.load(f)

                            # Convert insights to the expected format
                            if 'insights' in data:
                                st.session_state.synthesis_results = data
                                st.success(f"📊 Loaded synthesis results from {latest_file.name}")
                                st.rerun()
                            else:
                                st.warning("No insights found in synthesis file")
                        else:
                            st.warning("No synthesis results found")
                    else:
                        st.warning("Synthesis output directory not found")
                except Exception as e:
                    logger.error(f"Failed to load synthesis results: {e}")
                    st.error(f"❌ Failed to load synthesis results: {e}")

        with col3:
            if st.button("📚 View History", help="Browse synthesis history and load previous runs"):
                st.session_state.show_synthesis_history = True
                st.rerun()

            # Show synthesis history if requested
            if st.session_state.get('show_synthesis_history', False):
                render_synthesis_history()

    except Exception as e:
        st.error(f"❌ Synthesis controls error: {e}")
        logger.error(f"Synthesis controls failed: {e}")
        # Show minimal fallback controls
        st.markdown("### 🌙 Cognitive Synthesis Controls")
        if st.button("🔄 Run Synthesis (Fallback)", type="primary"):
            st.error("Synthesis functionality temporarily unavailable")

def render_auto_synthesis_controls():
    """Render auto-synthesis configuration controls."""
    try:
        st.markdown("#### ⚙️ Auto-Synthesis Settings")

        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            # Auto-synthesis toggle (using checkbox for compatibility)
            auto_synthesis_enabled = st.session_state.get('auto_synthesis_enabled', False)
            new_auto_synthesis = st.checkbox(
                "🤖 Auto-Synthesis",
                value=auto_synthesis_enabled,
                help="Automatically run synthesis when new insights are detected"
            )

            if new_auto_synthesis != auto_synthesis_enabled:
                st.session_state.auto_synthesis_enabled = new_auto_synthesis
                if new_auto_synthesis:
                    st.success("✅ Auto-synthesis enabled")
                else:
                    st.info("⏸️ Auto-synthesis disabled")
                st.rerun()

        with col2:
            if st.session_state.get('auto_synthesis_enabled', False):
                # Auto-synthesis frequency
                frequency_options = {
                    "Every 10 minutes": 600,
                    "Every 30 minutes": 1800,
                    "Every hour": 3600,
                    "Every 6 hours": 21600,
                    "Daily": 86400
                }

                current_frequency = st.session_state.get('auto_synthesis_frequency', 3600)
                frequency_label = next((k for k, v in frequency_options.items() if v == current_frequency), "Every hour")

                selected_frequency = st.selectbox(
                    "Frequency",
                    options=list(frequency_options.keys()),
                    index=list(frequency_options.keys()).index(frequency_label),
                    help="How often to run auto-synthesis"
                )

                st.session_state.auto_synthesis_frequency = frequency_options[selected_frequency]
            else:
                st.markdown("*Enable auto-synthesis to configure frequency*")

        with col3:
            if st.session_state.get('auto_synthesis_enabled', False):
                # Auto-research toggle
                auto_research_enabled = st.session_state.get('auto_research_enabled', False)
                new_auto_research = st.checkbox(
                    "🔬 Auto-Research",
                    value=auto_research_enabled,
                    help="Automatically research promising insights"
                )

                if new_auto_research != auto_research_enabled:
                    st.session_state.auto_research_enabled = new_auto_research

    except Exception as e:
        st.error(f"❌ Auto-synthesis controls error: {e}")
        logger.error(f"Auto-synthesis controls failed: {e}")

def render_synthesis_history():
    """Display synthesis history with ability to load previous runs."""
    st.markdown("---")
    st.markdown("### 📚 Synthesis History")

    col1, col2 = st.columns([3, 1])

    with col2:
        if st.button("❌ Close History", help="Close synthesis history view"):
            st.session_state.show_synthesis_history = False
            st.rerun()

    with col1:
        st.markdown("*Browse and load previous synthesis runs*")

    try:
        # Get synthesis history from engine
        from memory.synthesis.synthesis_engine import SynthesisEngine
        synthesis_engine = SynthesisEngine()
        history = synthesis_engine.get_synthesis_history()

        if not history:
            st.info("📭 No synthesis history found. Run synthesis to create your first entry.")
            return

        # Display history in a table-like format
        st.markdown(f"**Found {len(history)} synthesis runs:**")

        for i, run in enumerate(history[:10]):  # Show last 10 runs
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

                with col1:
                    timestamp = run.get('timestamp', 'Unknown')
                    if timestamp != 'Unknown':
                        try:
                            # Format timestamp nicely
                            from datetime import datetime
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            formatted_time = dt.strftime('%Y-%m-%d %H:%M')
                        except:
                            formatted_time = timestamp[:16]
                    else:
                        formatted_time = 'Unknown'

                    st.markdown(f"**{formatted_time}**")

                with col2:
                    insights_count = run.get('insights_generated', 0)
                    clusters_count = run.get('clusters_analyzed', 0)
                    st.markdown(f"🧠 {insights_count} insights, 🔗 {clusters_count} clusters")

                with col3:
                    run_id = run.get('run_id', f'run_{i}')
                    if st.button(f"📂 Load", key=f"load_history_{i}", help=f"Load synthesis run {run_id}"):
                        try:
                            # Load the specific synthesis run
                            file_path = run.get('file_path')
                            if file_path and Path(file_path).exists():
                                with open(file_path, 'r') as f:
                                    data = json.load(f)

                                st.session_state.synthesis_results = data
                                st.success(f"✅ Loaded synthesis run from {formatted_time}")
                                st.session_state.show_synthesis_history = False
                                st.rerun()
                            else:
                                st.error("❌ Synthesis file not found")
                        except Exception as e:
                            st.error(f"❌ Failed to load synthesis: {e}")

                with col4:
                    if st.button(f"🗑️", key=f"delete_history_{i}", help=f"Delete synthesis run {run_id}"):
                        try:
                            file_path = run.get('file_path')
                            if file_path and Path(file_path).exists():
                                Path(file_path).unlink()
                                st.success(f"🗑️ Deleted synthesis run")
                                st.rerun()
                        except Exception as e:
                            st.error(f"❌ Failed to delete: {e}")

                st.markdown("---")

        if len(history) > 10:
            st.info(f"📋 Showing 10 most recent runs. Total: {len(history)} runs available.")

    except Exception as e:
        logger.error(f"Failed to load synthesis history: {e}")
        st.error(f"❌ Failed to load synthesis history: {e}")

def update_synthesis_history(synthesis_results):
    """Update synthesis history in session state."""
    if 'synthesis_history' not in st.session_state:
        st.session_state.synthesis_history = []

    # Add new synthesis result to history
    history_entry = {
        'run_id': synthesis_results.get('run_id'),
        'timestamp': synthesis_results.get('timestamp'),
        'insights': synthesis_results.get('insights', []),
        'clusters_found': synthesis_results.get('clusters_found', 0),
        'insights_generated': synthesis_results.get('insights_generated', 0),
        'status': 'success'
    }

    st.session_state.synthesis_history.append(history_entry)

    # Keep only last 50 entries to prevent memory bloat
    if len(st.session_state.synthesis_history) > 50:
        st.session_state.synthesis_history = st.session_state.synthesis_history[-50:]

    # NEW: Add the missing "Synthesis" button that appears after dream state completion
    if hasattr(st.session_state, 'synthesis_results') and st.session_state.synthesis_results:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("✨ View Synthesis Results", type="secondary", help="View detailed synthesis insights and patterns"):
                # This button toggles the display of synthesis results
                if not hasattr(st.session_state, 'show_synthesis_details'):
                    st.session_state.show_synthesis_details = True
                else:
                    st.session_state.show_synthesis_details = not st.session_state.show_synthesis_details
                st.rerun()

        with col2:
            if st.button("🎨 Load in Dream Canvas", help="Load synthesis results into Dream Canvas visualization"):
                # Create focused visualization data for this synthesis
                try:
                    insights = st.session_state.synthesis_results.get('insights', [])
                    run_id = st.session_state.synthesis_results.get('run_id', 'current')

                    # Import the focused visualization function
                    from ui.memory_app import create_focused_synthesis_visualization
                    focused_visualization_data = create_focused_synthesis_visualization(insights, run_id)

                    # Store focused visualization data
                    st.session_state.dream_canvas_data = focused_visualization_data

                    st.success(f"✅ Synthesis results loaded into Dream Canvas! Showing {len(focused_visualization_data)} focused memory points.")
                    st.rerun()
                except Exception as e:
                    logger.error(f"Failed to create focused visualization: {e}")
                    st.error(f"❌ Failed to load into Dream Canvas: {e}")

        with col3:
            col3a, col3b = st.columns(2)

            with col3a:
                if st.button("⚙️ Settings", help="Configure synthesis settings"):
                    st.session_state.show_synthesis_settings = not st.session_state.get('show_synthesis_settings', False)
                    st.rerun()

            with col3b:
                if st.button("🔄 Clear", help="Clear current synthesis results"):
                    if hasattr(st.session_state, 'synthesis_results'):
                        del st.session_state.synthesis_results
                    if hasattr(st.session_state, 'show_synthesis_details'):
                        del st.session_state.show_synthesis_details
                    st.success("🧹 Synthesis results cleared")
                    st.rerun()

        # Show synthesis settings if requested
        if st.session_state.get('show_synthesis_settings', False):
            render_synthesis_settings()

def render_synthesis_settings():
    """Display advanced synthesis configuration settings."""
    st.markdown("---")
    st.markdown("#### ⚙️ Advanced Synthesis Settings")

    col1, col2 = st.columns([3, 1])

    with col2:
        if st.button("❌ Close Settings", help="Close synthesis settings"):
            st.session_state.show_synthesis_settings = False
            st.rerun()

    with col1:
        st.markdown("*Configure synthesis parameters and behavior*")

    # Synthesis parameters
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🔗 Clustering Parameters**")

        min_cluster_size = st.slider(
            "Min Cluster Size",
            min_value=2,
            max_value=20,
            value=st.session_state.get('synthesis_min_cluster_size', 5),
            help="Minimum number of memories required to form a cluster"
        )
        st.session_state.synthesis_min_cluster_size = min_cluster_size

        cluster_threshold = st.slider(
            "Cluster Threshold",
            min_value=0.1,
            max_value=0.9,
            value=st.session_state.get('synthesis_cluster_threshold', 0.3),
            step=0.05,
            help="Similarity threshold for clustering memories"
        )
        st.session_state.synthesis_cluster_threshold = cluster_threshold

    with col2:
        st.markdown("**🧠 Insight Generation**")

        max_insights = st.slider(
            "Max Insights per Run",
            min_value=5,
            max_value=100,
            value=st.session_state.get('synthesis_max_insights', 20),
            help="Maximum number of insights to generate per synthesis run"
        )
        st.session_state.synthesis_max_insights = max_insights

        insight_quality_threshold = st.slider(
            "Quality Threshold",
            min_value=0.1,
            max_value=1.0,
            value=st.session_state.get('synthesis_quality_threshold', 0.6),
            step=0.05,
            help="Minimum quality score for insights to be included"
        )
        st.session_state.synthesis_quality_threshold = insight_quality_threshold

    with col3:
        st.markdown("**📊 Output Preferences**")

        include_visualization = st.checkbox(
            "Include Visualization",
            value=st.session_state.get('synthesis_include_viz', True),
            help="Generate visualization data during synthesis"
        )
        st.session_state.synthesis_include_viz = include_visualization

        save_detailed_logs = st.checkbox(
            "Detailed Logging",
            value=st.session_state.get('synthesis_detailed_logs', True),
            help="Save detailed synthesis logs for debugging"
        )
        st.session_state.synthesis_detailed_logs = save_detailed_logs

        auto_cleanup = st.checkbox(
            "Auto-cleanup Old Runs",
            value=st.session_state.get('synthesis_auto_cleanup', False),
            help="Automatically delete synthesis runs older than 30 days"
        )
        st.session_state.synthesis_auto_cleanup = auto_cleanup

    # Performance settings
    st.markdown("---")
    st.markdown("**⚡ Performance Settings**")

    col1, col2, col3 = st.columns(3)

    with col1:
        parallel_processing = st.checkbox(
            "🚀 Parallel Processing",
            value=st.session_state.get('synthesis_parallel', True),
            help="Use multiple CPU cores for faster synthesis"
        )
        st.session_state.synthesis_parallel = parallel_processing

    with col2:
        memory_limit = st.selectbox(
            "Memory Limit",
            options=["1GB", "2GB", "4GB", "8GB", "Unlimited"],
            index=2,  # Default to 4GB
            help="Maximum memory usage for synthesis operations"
        )
        st.session_state.synthesis_memory_limit = memory_limit

    with col3:
        if st.button("🔄 Reset to Defaults", help="Reset all settings to default values"):
            # Clear all synthesis settings from session state
            settings_keys = [k for k in st.session_state.keys() if k.startswith('synthesis_')]
            for key in settings_keys:
                del st.session_state[key]
            st.success("✅ Settings reset to defaults")
            st.rerun()

def render_research_integration_controls(synthesis_results):
    """Render research integration controls for Dream Canvas insights."""
    try:
        insights = synthesis_results.get('insights', [])

        if not insights:
            return

        st.markdown("---")
        st.markdown("### 🔬 Research Integration")
        st.markdown("*Select insights for automated research discovery*")

        # Research mode selection
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            research_mode = st.radio(
                "Research Selection Mode:",
                options=["🤖 SAM Selects Best", "👤 Human Selection"],
                index=0,
                help="Choose how insights are selected for research"
            )

        with col2:
            max_research_papers = st.selectbox(
                "Papers per Insight:",
                options=[1, 2, 3, 5],
                index=1,  # Default to 2
                help="Maximum papers to download per selected insight"
            )

        with col3:
            st.markdown("") # Spacer

        # Insight selection interface
        if research_mode == "👤 Human Selection":
            st.markdown("**Select insights for research:**")

            # Initialize selection state
            if 'selected_insights' not in st.session_state:
                st.session_state.selected_insights = set()

            # Display insights with checkboxes
            for i, insight in enumerate(insights):
                insight_id = f"insight_{i}"
                insight_text = insight.get('content', insight.get('insight', 'No content'))
                cluster_id = insight.get('cluster_id', 'Unknown')
                confidence = insight.get('confidence_score', 0.0)

                # Create checkbox for each insight
                col1, col2 = st.columns([1, 10])

                with col1:
                    is_selected = st.checkbox(
                        "",
                        key=f"select_{insight_id}",
                        value=insight_id in st.session_state.selected_insights
                    )

                    # Update selection state
                    if is_selected:
                        st.session_state.selected_insights.add(insight_id)
                    else:
                        st.session_state.selected_insights.discard(insight_id)

                with col2:
                    # Display insight with metadata
                    st.markdown(f"""
                    **Cluster {cluster_id}** (Confidence: {confidence:.2f})

                    {insight_text}
                    """)

            # Selection summary
            selected_count = len(st.session_state.selected_insights)
            if selected_count > 0:
                st.info(f"📋 **{selected_count} insight{'' if selected_count == 1 else 's'} selected** for research")
            else:
                st.warning("⚠️ No insights selected. Please select at least one insight for research.")

        else:
            # SAM automatic selection
            st.info("🤖 **SAM will automatically select the most promising insight** based on novelty, research potential, and confidence scores.")

            # Show preview of what SAM would select
            if insights:
                # Simple scoring: combine confidence and novelty indicators
                scored_insights = []
                for i, insight in enumerate(insights):
                    confidence = insight.get('confidence_score', 0.0)
                    content = insight.get('content', insight.get('insight', ''))

                    # Novelty scoring based on keywords
                    novelty_keywords = ['new', 'novel', 'innovative', 'breakthrough', 'discovery', 'emerging', 'unprecedented']
                    novelty_score = sum(1 for keyword in novelty_keywords if keyword.lower() in content.lower()) / len(novelty_keywords)

                    # Research potential based on question words and uncertainty
                    research_keywords = ['how', 'why', 'what', 'could', 'might', 'potential', 'explore', 'investigate']
                    research_score = sum(1 for keyword in research_keywords if keyword.lower() in content.lower()) / len(research_keywords)

                    # Combined score
                    combined_score = confidence * 0.4 + novelty_score * 0.3 + research_score * 0.3

                    scored_insights.append((i, insight, combined_score))

                # Sort by score and show top candidate
                scored_insights.sort(key=lambda x: x[2], reverse=True)
                best_insight = scored_insights[0]

                st.markdown(f"**🎯 SAM's Top Selection:**")
                st.markdown(f"**Cluster {best_insight[1].get('cluster_id', 'Unknown')}** (Score: {best_insight[2]:.2f})")
                st.markdown(f"{best_insight[1].get('content', best_insight[1].get('insight', 'No content'))}")

        # Research action button
        st.markdown("---")

        # Check if research components are available
        research_available = True
        try:
            from sam.web_retrieval.tools.arxiv_tool import get_arxiv_tool
            from sam.state.vetting_queue import get_vetting_queue_manager
        except ImportError:
            research_available = False

        if research_available:
            # Enhanced research options with Quick and Deep Research
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                if st.button("🔬 **Quick Research**", use_container_width=True,
                           help="Basic ArXiv search for selected insights"):
                    # Trigger quick research process (original logic)
                    if research_mode == "👤 Human Selection":
                        if st.session_state.selected_insights:
                            selected_indices = [int(insight_id.split('_')[1]) for insight_id in st.session_state.selected_insights]
                            selected_insights_data = [insights[i] for i in selected_indices]
                            trigger_insight_research(selected_insights_data, max_research_papers)
                        else:
                            st.error("❌ Please select at least one insight for research")
                    else:
                        # SAM automatic selection
                        if insights:
                            # Use the same scoring logic to select the best insight
                            scored_insights = []
                            for i, insight in enumerate(insights):
                                confidence = insight.get('confidence_score', 0.0)
                                content = insight.get('content', insight.get('insight', ''))

                                novelty_keywords = ['new', 'novel', 'innovative', 'breakthrough', 'discovery', 'emerging', 'unprecedented']
                                novelty_score = sum(1 for keyword in novelty_keywords if keyword.lower() in content.lower()) / len(novelty_keywords)

                                research_keywords = ['how', 'why', 'what', 'could', 'might', 'potential', 'explore', 'investigate']
                                research_score = sum(1 for keyword in research_keywords if keyword.lower() in content.lower()) / len(research_keywords)

                                combined_score = confidence * 0.4 + novelty_score * 0.3 + research_score * 0.3
                                scored_insights.append((i, insight, combined_score))

                            scored_insights.sort(key=lambda x: x[2], reverse=True)
                            best_insight = scored_insights[0][1]

                            trigger_insight_research([best_insight], max_research_papers)
                        else:
                            st.error("❌ No insights available for research")

            with col2:
                if st.button("🧠 **Deep Research**", type="primary", use_container_width=True,
                           help="Comprehensive multi-step ArXiv analysis with verification"):
                    # Trigger deep research process (new Task 32 logic)
                    if research_mode == "👤 Human Selection":
                        if st.session_state.selected_insights:
                            selected_indices = [int(insight_id.split('_')[1]) for insight_id in st.session_state.selected_insights]
                            selected_insights_data = [insights[i] for i in selected_indices]
                            trigger_deep_research_engine(selected_insights_data)
                        else:
                            st.error("❌ Please select at least one insight for deep research")
                    else:
                        # SAM automatic selection for deep research
                        if insights:
                            # Use the same scoring logic to select the best insight
                            scored_insights = []
                            for i, insight in enumerate(insights):
                                confidence = insight.get('confidence_score', 0.0)
                                content = insight.get('content', insight.get('insight', ''))

                                novelty_keywords = ['new', 'novel', 'innovative', 'breakthrough', 'discovery', 'emerging', 'unprecedented']
                                novelty_score = sum(1 for keyword in novelty_keywords if keyword.lower() in content.lower()) / len(novelty_keywords)

                                research_keywords = ['how', 'why', 'what', 'could', 'might', 'potential', 'explore', 'investigate']
                                research_score = sum(1 for keyword in research_keywords if keyword.lower() in content.lower()) / len(research_keywords)

                                combined_score = confidence * 0.4 + novelty_score * 0.3 + research_score * 0.3
                                scored_insights.append((i, insight, combined_score))

                            scored_insights.sort(key=lambda x: x[2], reverse=True)
                            best_insight = scored_insights[0][1]

                            trigger_deep_research_engine([best_insight])
                        else:
                            st.error("❌ No insights available for deep research")

            with col3:
                if st.button("📋 View Research Queue", use_container_width=True,
                           help="View pending research papers in vetting queue"):
                    # Navigate to vetting queue
                    st.session_state.show_memory_control_center = True
                    st.session_state.memory_page_override = "🔍 Vetting Queue"
                    st.rerun()

        else:
            st.warning("⚠️ Research components not available. Install Task 27 components to enable automated research.")

    except Exception as e:
        st.error(f"❌ Error loading research integration: {e}")

def trigger_insight_research(selected_insights, max_papers_per_insight):
    """Trigger automated research for selected insights."""
    try:
        from sam.web_retrieval.tools.arxiv_tool import get_arxiv_tool
        from sam.state.vetting_queue import get_vetting_queue_manager
        from sam.vetting.analyzer import get_vetting_analyzer
        import threading
        import asyncio

        arxiv_tool = get_arxiv_tool()
        vetting_manager = get_vetting_queue_manager()
        vetting_analyzer = get_vetting_analyzer()

        def run_research():
            """Run research in background thread."""
            try:
                total_papers = 0

                for insight in selected_insights:
                    insight_text = insight.get('content', insight.get('insight', ''))
                    cluster_id = insight.get('cluster_id', 'Unknown')

                    # Generate research query from insight
                    research_query = generate_research_query(insight_text)

                    # Search and download papers
                    for paper_num in range(max_papers_per_insight):
                        try:
                            # Search arXiv
                            result = arxiv_tool.search_and_download(research_query, insight_text)

                            if result.get('success'):
                                # Add to vetting queue
                                file_id = vetting_manager.add_file_to_queue(
                                    quarantine_path=result['local_path'],
                                    paper_metadata=result['paper_metadata'],
                                    original_insight_text=insight_text
                                )

                                # Analyze the paper
                                analysis_result = vetting_analyzer.analyze_quarantined_file(
                                    file_path=result['local_path'],
                                    original_insight_text=insight_text,
                                    paper_metadata=result['paper_metadata']
                                )

                                # Update vetting queue with analysis
                                if analysis_result.security_risk and analysis_result.relevance_score and analysis_result.credibility_score:
                                    from sam.state.vetting_queue import VettingScores
                                    scores = VettingScores(
                                        security_risk_score=analysis_result.security_risk.risk_score,
                                        relevance_score=analysis_result.relevance_score.relevance_score,
                                        credibility_score=analysis_result.credibility_score.credibility_score,
                                        overall_score=analysis_result.overall_score
                                    )

                                    vetting_manager.update_analysis_results(file_id, scores)

                                total_papers += 1

                            else:
                                break  # Stop trying for this insight if search fails

                        except Exception as e:
                            logger.error(f"Research failed for insight: {e}")
                            break

                # Store result in session state
                st.session_state.research_result = {
                    'success': True,
                    'total_papers': total_papers,
                    'insights_processed': len(selected_insights)
                }

            except Exception as e:
                st.session_state.research_result = {
                    'success': False,
                    'error': str(e)
                }

        # Start research in background
        thread = threading.Thread(target=run_research, daemon=True)
        thread.start()

        st.success(f"🔬 **Research initiated!** Processing {len(selected_insights)} insight{'' if len(selected_insights) == 1 else 's'}")
        st.info("📄 Papers will be downloaded and added to the vetting queue for your review.")
        st.info("🔄 **Refresh this page** or check the vetting queue to see progress.")

    except Exception as e:
        st.error(f"❌ Failed to start research: {e}")

def trigger_deep_research_engine(selected_insights):
    """Trigger the Deep Research Engine for comprehensive ArXiv analysis."""
    try:
        from sam.agents.strategies.deep_research import DeepResearchStrategy
        import threading

        def run_deep_research():
            """Run deep research in background thread."""
            try:
                research_results = []

                for insight in selected_insights:
                    insight_text = insight.get('content', insight.get('insight', ''))
                    cluster_id = insight.get('cluster_id', 'Unknown')

                    # Initialize Deep Research Strategy
                    research_strategy = DeepResearchStrategy(insight_text)

                    # Execute deep research
                    result = research_strategy.execute_research()

                    # Store result in session state
                    research_results.append({
                        'research_id': result.research_id,
                        'original_insight': result.original_insight,
                        'cluster_id': cluster_id,
                        'final_report': result.final_report,
                        'arxiv_papers': result.arxiv_papers,
                        'status': result.status.value,
                        'timestamp': result.timestamp,
                        'quality_score': research_strategy._assess_research_quality(),
                        'papers_analyzed': len(result.arxiv_papers),
                        'iterations_completed': research_strategy.current_iteration
                    })

                # Store results in session state
                if 'deep_research_results' not in st.session_state:
                    st.session_state.deep_research_results = []

                st.session_state.deep_research_results.extend(research_results)
                st.session_state.latest_deep_research = research_results

                # Add papers to vetting queue
                try:
                    from sam.state.vetting_queue import get_vetting_queue_manager
                    vetting_manager = get_vetting_queue_manager()

                    total_papers_added = 0
                    for result in research_results:
                        for paper in result['arxiv_papers']:
                            # Add paper to vetting queue with deep research context
                            paper_metadata = {
                                'title': paper.get('title', 'Unknown Title'),
                                'authors': paper.get('authors', []),
                                'abstract': paper.get('summary', ''),
                                'url': paper.get('pdf_url', ''),
                                'source': 'deep_research_arxiv',
                                'research_context': {
                                    'research_id': result['research_id'],
                                    'original_insight': result['original_insight'][:200] + '...',
                                    'cluster_id': result['cluster_id']
                                }
                            }

                            # Note: In a full implementation, we would download the paper
                            # For now, we add the metadata to the queue
                            total_papers_added += 1

                    # Update session state with completion info
                    st.session_state.deep_research_completion = {
                        'success': True,
                        'insights_processed': len(selected_insights),
                        'total_papers': total_papers_added,
                        'reports_generated': len(research_results),
                        'timestamp': datetime.now().isoformat()
                    }

                except Exception as e:
                    logger.error(f"Failed to add papers to vetting queue: {e}")
                    st.session_state.deep_research_completion = {
                        'success': True,
                        'insights_processed': len(selected_insights),
                        'reports_generated': len(research_results),
                        'vetting_error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }

            except Exception as e:
                logger.error(f"Deep research execution failed: {e}")
                st.session_state.deep_research_completion = {
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }

        # Start deep research in background
        research_thread = threading.Thread(target=run_deep_research, daemon=True)
        research_thread.start()

        st.success(f"🧠 **Deep Research initiated!** Processing {len(selected_insights)} insight{'' if len(selected_insights) == 1 else 's'}")
        st.info("📊 **Comprehensive Analysis**: Multi-step ArXiv research with verification and critique")
        st.info("📄 **Report Generation**: Structured research reports will be generated")
        st.info("🔄 **Check Results**: Refresh this page or check the Deep Research Results section below")

    except ImportError:
        st.error("❌ Deep Research Engine not available. Please ensure sam.agents.strategies.deep_research is installed.")
    except Exception as e:
        st.error(f"❌ Failed to start deep research: {e}")

def generate_research_query(insight_text):
    """Generate an optimized research query from an insight."""
    try:
        # Extract key terms from insight
        import re

        # Remove common stop words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}

        # Extract words (remove punctuation, convert to lowercase)
        words = re.findall(r'\b[a-zA-Z]+\b', insight_text.lower())

        # Filter out stop words and short words
        meaningful_words = [word for word in words if word not in stop_words and len(word) > 3]

        # Take top 5 most meaningful terms
        key_terms = meaningful_words[:5]

        # Create research query
        if key_terms:
            query = ' '.join(key_terms)
        else:
            # Fallback to first few words of insight
            query = ' '.join(insight_text.split()[:5])

        return query

    except Exception as e:
        # Fallback to simple truncation
        return insight_text[:50]

def render_deep_research_results():
    """Display Deep Research results and reports."""
    try:
        if 'deep_research_results' not in st.session_state or not st.session_state.deep_research_results:
            return

        st.markdown("---")
        st.markdown("### 🧠 Deep Research Results")
        st.markdown("*Comprehensive ArXiv analysis reports with verification*")

        # Show completion status if available
        if 'deep_research_completion' in st.session_state:
            completion = st.session_state.deep_research_completion
            if completion.get('success'):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Insights Analyzed", completion.get('insights_processed', 0))
                with col2:
                    st.metric("Reports Generated", completion.get('reports_generated', 0))
                with col3:
                    st.metric("Papers Found", completion.get('total_papers', 0))

        # Display research results
        for i, result in enumerate(st.session_state.deep_research_results):
            with st.expander(f"📊 Research Report {i+1}: {result['original_insight'][:60]}...", expanded=i==0):

                # Research metadata
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Quality Score", f"{result.get('quality_score', 0):.2f}")
                with col2:
                    st.metric("Papers Analyzed", result.get('papers_analyzed', 0))
                with col3:
                    st.metric("Iterations", result.get('iterations_completed', 0))
                with col4:
                    status_color = "🟢" if result.get('status') == 'COMPLETED' else "🟡"
                    st.metric("Status", f"{status_color} {result.get('status', 'Unknown')}")

                # Display the full research report
                st.markdown("#### 📄 Research Report")
                st.markdown(result.get('final_report', 'Report not available'))

                # Show ArXiv papers found
                if result.get('arxiv_papers'):
                    st.markdown("#### 📚 ArXiv Papers Analyzed")
                    for j, paper in enumerate(result['arxiv_papers'][:5]):  # Show top 5
                        title = paper.get('title', 'Unknown Title')
                        authors = ', '.join(paper.get('authors', [])[:3])
                        year = paper.get('published', '')[:4] if paper.get('published') else 'Unknown'

                        st.markdown(f"**{j+1}. {title}** ({year})")
                        st.markdown(f"*Authors*: {authors}")
                        if paper.get('summary'):
                            st.markdown(f"*Summary*: {paper['summary'][:150]}...")
                        st.markdown("---")

                # Action buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(f"📋 View Papers in Queue", key=f"queue_{result['research_id']}"):
                        st.session_state.show_memory_control_center = True
                        st.session_state.memory_page_override = "🔍 Vetting Queue"
                        st.rerun()

                with col2:
                    if st.button(f"📄 Export Report", key=f"export_{result['research_id']}"):
                        # Create downloadable report
                        report_content = result.get('final_report', '')
                        st.download_button(
                            label="Download Report",
                            data=report_content,
                            file_name=f"deep_research_report_{result['research_id']}.md",
                            mime="text/markdown",
                            key=f"download_{result['research_id']}"
                        )

                with col3:
                    if st.button(f"🔄 Re-run Research", key=f"rerun_{result['research_id']}"):
                        # Re-trigger research for this insight
                        insight_data = {
                            'content': result['original_insight'],
                            'cluster_id': result.get('cluster_id', 'Unknown')
                        }
                        trigger_deep_research_engine([insight_data])

        # Clear results button
        if st.button("🗑️ Clear Deep Research Results"):
            st.session_state.deep_research_results = []
            if 'deep_research_completion' in st.session_state:
                del st.session_state.deep_research_completion
            st.rerun()

    except Exception as e:
        st.error(f"❌ Error displaying deep research results: {e}")

def render_synthetic_insights_integration():
    """Display synthetic insights and emergent patterns alongside the cognitive map."""

    # Check if synthesis results are available
    if not hasattr(st.session_state, 'synthesis_results') or not st.session_state.synthesis_results:
        # First try to load from synthesis history in session state
        if hasattr(st.session_state, 'synthesis_history') and st.session_state.synthesis_history:
            latest_run = st.session_state.synthesis_history[-1]
            if latest_run.get('status') == 'success' and latest_run.get('insights'):
                st.session_state.synthesis_results = {
                    'insights': latest_run['insights'],
                    'clusters_found': latest_run.get('clusters_found', 0),
                    'insights_generated': latest_run.get('insights_generated', 0),
                    'run_id': latest_run.get('run_id', 'loaded'),
                    'timestamp': latest_run.get('timestamp', 'unknown'),
                    'synthesis_log': {'status': 'loaded_from_history'}
                }

                # Display the loaded results
                render_synthetic_insights_panel(st.session_state.synthesis_results)
                render_pattern_discovery_interface(st.session_state.synthesis_results)
                return
        # Try to load the latest synthesis results from file
        try:
            from pathlib import Path
            import json

            synthesis_dir = Path("synthesis_output")
            if synthesis_dir.exists():
                # Find the most recent synthesis file
                synthesis_files = list(synthesis_dir.glob("synthesis_run_log_*.json"))
                if synthesis_files:
                    latest_file = max(synthesis_files, key=lambda f: f.stat().st_mtime)

                    with open(latest_file, 'r') as f:
                        data = json.load(f)

                    # Check if this file has insights
                    if 'insights' in data and data['insights']:
                        # Convert to the expected format
                        st.session_state.synthesis_results = {
                            'insights': data['insights'],
                            'clusters_found': data.get('clusters_found', len(data['insights'])),
                            'insights_generated': len(data['insights']),
                            'run_id': data.get('run_id', 'loaded'),
                            'timestamp': data.get('timestamp', 'unknown'),
                            'synthesis_log': data.get('synthesis_log', latest_file.name)
                        }

                        # Display the loaded results
                        render_synthetic_insights_panel(st.session_state.synthesis_results)
                        render_pattern_discovery_interface(st.session_state.synthesis_results)
                        return
        except Exception as e:
            logger.warning(f"Could not load synthesis results: {e}")

        # No synthesis results available
        st.markdown("---")
        st.markdown("### ✨ Synthetic Insights - New Understanding Generated")
        st.markdown("*These insights represent new knowledge emergent from SAM's cognitive synthesis.*")

        col1, col2 = st.columns(2)
        with col1:
            st.info("🌙 No recent synthesis insights available. Use the controls above to generate new understanding from your memory patterns.")

        with col2:
            if st.button("📊 Load Latest Synthesis", help="Load the most recent synthesis results from files"):
                try:
                    # Load most recent synthesis results from files
                    synthesis_dir = Path("synthesis_output")
                    if synthesis_dir.exists():
                        synthesis_files = list(synthesis_dir.glob("synthesis_run_log_*.json"))
                        if synthesis_files:
                            latest_file = max(synthesis_files, key=lambda x: x.stat().st_mtime)

                            with open(latest_file, 'r') as f:
                                data = json.load(f)

                            # Convert insights to the expected format
                            if 'insights' in data and data['insights']:
                                st.session_state.synthesis_results = {
                                    'insights': data['insights'],
                                    'clusters_found': data.get('clusters_found', len(data['insights'])),
                                    'insights_generated': len(data['insights']),
                                    'run_id': data.get('run_id', 'loaded'),
                                    'timestamp': data.get('timestamp', 'unknown'),
                                    'synthesis_log': data.get('synthesis_log', latest_file.name)
                                }
                                st.success(f"📊 Loaded synthesis results from {latest_file.name}")
                                st.rerun()
                            else:
                                st.warning("No insights found in synthesis file")
                        else:
                            st.warning("No synthesis results found")
                    else:
                        st.warning("Synthesis output directory not found")
                except Exception as e:
                    logger.error(f"Failed to load synthesis results: {e}")
                    st.error(f"❌ Failed to load synthesis results: {e}")

        return

    synthesis_results = st.session_state.synthesis_results

    # Only display detailed synthesis results if user has clicked "View Synthesis Results"
    if hasattr(st.session_state, 'show_synthesis_details') and st.session_state.show_synthesis_details:
        # Display synthetic insights panel
        render_synthetic_insights_panel(synthesis_results)

        # Display pattern discovery interface
        render_pattern_discovery_interface(synthesis_results)
    else:
        # Show a brief summary that synthesis is available
        st.markdown("---")
        st.markdown("### ✨ Synthesis Complete")
        insights_count = synthesis_results.get('insights_generated', 0)
        clusters_count = synthesis_results.get('clusters_found', 0)
        st.info(f"🌙 Dream state synthesis completed! Generated **{insights_count} insights** from **{clusters_count} clusters**. Use the controls above to view detailed results.")

        # Show a preview of the first insight
        insights = synthesis_results.get('insights', [])
        if insights:
            first_insight = insights[0]
            clean_text = first_insight.get('synthesized_text', '')
            if '<think>' in clean_text and '</think>' in clean_text:
                parts = clean_text.split('</think>')
                if len(parts) > 1:
                    clean_text = parts[-1].strip()

            # Show just the first sentence as a preview
            sentences = clean_text.split('. ')
            preview = sentences[0] if sentences else clean_text
            if len(preview) > 100:
                preview = preview[:100] + "..."

            st.markdown(f"**Preview:** *{preview}*")
            st.caption("Click 'View Synthesis Results' above to see all insights and patterns.")

def render_synthetic_insights_panel(synthesis_results):
    """Display the actual synthetic insights generated by SAM's dream state."""

    st.markdown("---")
    st.markdown("### ✨ Synthetic Insights - New Understanding Generated")
    st.markdown("*These insights represent new knowledge emergent from SAM's cognitive synthesis.*")

    # Task 27: Research Integration Controls
    render_research_integration_controls(synthesis_results)

    # Display synthesis summary
    clusters_found = synthesis_results.get('clusters_found', 0)
    insights_generated = synthesis_results.get('insights_generated', 0)
    timestamp = synthesis_results.get('timestamp', 'unknown')

    if clusters_found > 0 and insights_generated > 0:
        st.success(f"**Synthesis complete!** Generated **{insights_generated} insights** from **{clusters_found} clusters**.")
        st.caption(f"🕐 Generated: {timestamp}")

    insights = synthesis_results.get('insights', [])

    if not insights:
        st.info("🌙 No insights generated in the last synthesis run.")
        return

    # Metrics overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🧠 Insights Generated", len(insights))
    with col2:
        avg_confidence = sum(i.get('confidence_score', 0) for i in insights) / len(insights) if insights else 0
        st.metric("🎯 Avg Confidence", f"{avg_confidence:.2f}")
    with col3:
        avg_novelty = sum(i.get('novelty_score', 0) for i in insights) / len(insights) if insights else 0
        st.metric("🌟 Avg Novelty", f"{avg_novelty:.2f}")
    with col4:
        high_quality = sum(1 for i in insights if i.get('confidence_score', 0) > 0.4)
        st.metric("⭐ High Quality", f"{high_quality}/{len(insights)}")

    # Display insights
    st.markdown("#### 💡 Generated Insights")

    for i, insight in enumerate(insights[:5]):  # Show top 5 insights
        with st.expander(f"💡 Insight {i+1}: Cluster {insight.get('cluster_id', 'Unknown')}", expanded=i<2):

            # Insight content
            st.markdown("**🧠 Synthesized Understanding:**")
            # Clean the insight text (remove <think> tags)
            clean_text = insight.get('synthesized_text', '')
            if '<think>' in clean_text and '</think>' in clean_text:
                # Extract content after </think>
                parts = clean_text.split('</think>')
                if len(parts) > 1:
                    clean_text = parts[-1].strip()
                else:
                    clean_text = clean_text.replace('<think>', '').replace('</think>', '').strip()

            # Show first few sentences for readability
            sentences = clean_text.split('. ')
            if len(sentences) > 3:
                preview_text = '. '.join(sentences[:3]) + '...'
                st.markdown(f"*{preview_text}*")

                if st.button(f"Show Full Insight", key=f"show_full_insight_{i}_{insight.get('cluster_id', 'unknown')}"):
                    st.markdown(f"**Full Insight:**\n\n*{clean_text}*")
            else:
                st.markdown(f"*{clean_text}*")

            # Quality metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                confidence = insight.get('confidence_score', 0)
                st.metric("🎯 Confidence", f"{confidence:.2f}")
            with col2:
                novelty = insight.get('novelty_score', 0)
                st.metric("🌟 Novelty", f"{novelty:.2f}")
            with col3:
                utility = insight.get('utility_score', 0)
                st.metric("🔧 Utility", f"{utility:.2f}")

            # NEW: Enhanced cluster information with registry lookup
            st.markdown("---")
            render_cluster_information(insight)

            # Source information
            st.markdown("**📚 Source Analysis:**")
            metadata = insight.get('synthesis_metadata', {})
            source_count = metadata.get('source_count', 0)
            source_names = metadata.get('source_names', [])
            unique_sources = len(set(source_names)) if source_names else 0

            st.markdown(f"- **Sources**: {source_count} chunks from {unique_sources} sources")

            themes = metadata.get('dominant_themes', [])
            if themes:
                theme_text = ', '.join(themes[:5])
                st.markdown(f"- **Themes**: {theme_text}")

            generated_at = insight.get('generated_at', '')
            if generated_at:
                try:
                    dt = datetime.fromisoformat(generated_at.replace('Z', '+00:00'))
                    formatted_time = dt.strftime('%Y-%m-%d %H:%M')
                    st.markdown(f"- **Generated**: {formatted_time}")
                except:
                    st.markdown(f"- **Generated**: {generated_at[:19]}")

    if len(insights) > 5:
        st.info(f"📋 Showing top 5 insights. Total generated: {len(insights)}")

def render_cluster_information(insight):
    """Render detailed cluster information for an insight using the cluster registry."""
    try:
        from memory.synthesis.cluster_registry import get_cluster_stats

        cluster_id = insight.get('cluster_id', 'Unknown')

        # Get cluster statistics from registry
        cluster_stats = get_cluster_stats(cluster_id)

        # Create styled cluster info section
        if cluster_stats['exists']:
            # Cluster found in registry
            st.markdown(f"""
            <div style="
                border-left: 4px solid #4CAF50;
                padding: 15px;
                margin: 10px 0;
                background: linear-gradient(135deg, #4CAF5015, #4CAF5005);
                border-radius: 0 8px 8px 0;
            ">
                <h4 style="color: #4CAF50; margin: 0 0 10px 0;">💡 Key Insight</h4>
                <p style="margin: 0; font-size: 1rem; line-height: 1.4;">
                    This cluster represents a coherent group of {cluster_stats['memory_count']} related memories
                    with {len(cluster_stats['sources'])} unique sources. The cluster focuses on
                    <strong>{', '.join(cluster_stats['dominant_themes'][:2]) if cluster_stats['dominant_themes'] else 'related concepts'}</strong>.
                    <br><br>
                    <strong>So What:</strong> This represents a significant knowledge domain in SAM's memory -
                    consider exploring related queries about {cluster_stats['dominant_themes'][0] if cluster_stats['dominant_themes'] else 'this topic'}.
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Cluster statistics
            st.markdown("**📊 Cluster Stats:**")
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"• **{cluster_stats['memory_count']} memories** in this cluster")
                st.write(f"• **Avg importance:** {cluster_stats['avg_importance']:.2f}")
                st.write(f"• **Sources:** {cluster_stats['source_count']} unique")

            with col2:
                st.write(f"• **Coherence:** {cluster_stats['coherence_score']:.2f}")
                if cluster_stats['dominant_themes']:
                    st.write(f"• **Themes:** {', '.join(cluster_stats['dominant_themes'][:3])}")

            # Show key sources if available
            if cluster_stats['sources']:
                st.markdown("**📚 Key Sources:**")
                for source in cluster_stats['sources'][:3]:
                    source_name = source.split('/')[-1] if '/' in source else source
                    st.write(f"• {source_name}")
        else:
            # Cluster not found - show fallback message
            st.markdown(f"""
            <div style="
                border-left: 4px solid #FF9800;
                padding: 15px;
                margin: 10px 0;
                background: linear-gradient(135deg, #FF980015, #FF980005);
                border-radius: 0 8px 8px 0;
            ">
                <h4 style="color: #FF9800; margin: 0 0 10px 0;">💡 Key Insight</h4>
                <p style="margin: 0; font-size: 1rem; line-height: 1.4;">
                    This cluster appears to be from a previous synthesis run or contains no accessible content.
                    This might indicate a visualization artifact or a cluster that was filtered out during processing.
                    <br><br>
                    <strong>So What:</strong> Check the visualization parameters or run a new synthesis to ensure proper data display.
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Show minimal stats
            st.markdown("**📊 Cluster Stats:**")
            st.write(f"• **0 memories** in this cluster")
            st.write(f"• **Avg importance:** 0.00")
            st.write(f"• **Sources:** 0 unique")

    except Exception as e:
        logger.error(f"Error rendering cluster information: {e}")
        # Fallback to simple display
        st.markdown("**💡 Key Insight**")
        st.info("Cluster information temporarily unavailable")

def render_pattern_discovery_interface(synthesis_results):
    """Render interface for exploring emergent patterns and relationships."""

    st.markdown("---")
    st.markdown("### 🔍 Pattern Discovery & Analysis")
    st.markdown("*Explore emergent patterns and cross-domain connections in your insights*")

    insights = synthesis_results.get('insights', [])

    if not insights:
        return

    # Pattern analysis controls
    col1, col2, col3 = st.columns(3)

    with col1:
        pattern_type = st.selectbox(
            "🧩 Pattern Type",
            ["All Patterns", "Cross-Domain Connections", "High Novelty", "High Confidence", "Recent Insights"],
            help="Type of emergent pattern to explore"
        )

    with col2:
        confidence_threshold = st.slider(
            "🎯 Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Minimum confidence for displayed insights"
        )

    with col3:
        novelty_threshold = st.slider(
            "🌟 Novelty Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Minimum novelty for displayed insights"
        )

    # Filter insights based on criteria
    filtered_insights = filter_insights_by_pattern(insights, pattern_type, confidence_threshold, novelty_threshold)

    # Display pattern analysis
    if filtered_insights:
        st.markdown(f"#### 🌟 Found {len(filtered_insights)} insights matching criteria")

        # Pattern categories analysis
        render_pattern_categories_analysis(filtered_insights)

        # Top patterns summary
        render_top_patterns_summary(filtered_insights)
    else:
        st.info("🔍 No insights match the current filter criteria. Try adjusting the thresholds.")

def filter_insights_by_pattern(insights, pattern_type, confidence_threshold, novelty_threshold):
    """Filter insights based on pattern type and thresholds."""

    # Apply threshold filters
    filtered = [
        insight for insight in insights
        if (insight.get('confidence_score', 0) >= confidence_threshold and
            insight.get('novelty_score', 0) >= novelty_threshold)
    ]

    # Apply pattern type filter
    if pattern_type == "Cross-Domain Connections":
        # Insights that connect different source types
        filtered = [
            insight for insight in filtered
            if len(set(insight.get('synthesis_metadata', {}).get('source_types', []))) > 1
        ]
    elif pattern_type == "High Novelty":
        # Top 25% by novelty
        filtered = sorted(filtered, key=lambda x: x.get('novelty_score', 0), reverse=True)
        filtered = filtered[:max(1, len(filtered) // 4)]
    elif pattern_type == "High Confidence":
        # Insights with confidence > 0.5
        filtered = [insight for insight in filtered if insight.get('confidence_score', 0) > 0.5]
    elif pattern_type == "Recent Insights":
        # Most recent insights
        filtered = sorted(filtered, key=lambda x: x.get('generated_at', ''), reverse=True)
        filtered = filtered[:5]

    return filtered

def render_pattern_categories_analysis(insights):
    """Render analysis of pattern categories."""

    # Analyze pattern categories
    cross_domain = [i for i in insights if len(set(i.get('synthesis_metadata', {}).get('source_types', []))) > 1]
    high_novelty = [i for i in insights if i.get('novelty_score', 0) > 0.7]
    high_confidence = [i for i in insights if i.get('confidence_score', 0) > 0.4]
    high_utility = [i for i in insights if i.get('utility_score', 0) > 0.5]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🔗 Cross-Domain", len(cross_domain))
        if cross_domain:
            st.caption("Insights connecting different knowledge domains")

    with col2:
        st.metric("🌟 High Novelty", len(high_novelty))
        if high_novelty:
            st.caption("Insights with novel connections")

    with col3:
        st.metric("⭐ High Confidence", len(high_confidence))
        if high_confidence:
            st.caption("Insights with strong evidence")

    with col4:
        st.metric("🔧 High Utility", len(high_utility))
        if high_utility:
            st.caption("Insights with practical value")

def render_top_patterns_summary(insights):
    """Render summary of top emergent patterns."""

    st.markdown("#### 🏆 Top Emergent Patterns")

    # Sort by combined score (confidence * novelty)
    sorted_insights = sorted(
        insights,
        key=lambda x: x.get('confidence_score', 0) * x.get('novelty_score', 0),
        reverse=True
    )

    for i, insight in enumerate(sorted_insights[:3]):
        with st.container():
            confidence = insight.get('confidence_score', 0)
            novelty = insight.get('novelty_score', 0)
            combined_score = confidence * novelty

            st.markdown(f"**{i+1}. Cluster {insight.get('cluster_id', 'Unknown')}** "
                       f"(Confidence: {confidence:.2f}, Novelty: {novelty:.2f}, Score: {combined_score:.2f})")

            # Clean insight text
            clean_text = insight.get('synthesized_text', '')
            if '<think>' in clean_text and '</think>' in clean_text:
                parts = clean_text.split('</think>')
                if len(parts) > 1:
                    clean_text = parts[-1].strip()

            # Show first sentence or two
            sentences = clean_text.split('. ')
            preview = '. '.join(sentences[:2]) + '.' if len(sentences) > 1 else clean_text
            if len(preview) > 200:
                preview = preview[:200] + '...'

            st.markdown(f"*{preview}*")
            st.markdown("---")
