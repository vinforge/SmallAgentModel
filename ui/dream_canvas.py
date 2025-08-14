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

    # Initialize defaults for auto research controls
    if 'auto_run_research_on_select' not in st.session_state:
        st.session_state.auto_run_research_on_select = True
    if 'auto_research_mode' not in st.session_state:
        st.session_state.auto_research_mode = 'Deep'

    # Dream Canvas controls
    col1, col2, col3 = st.columns(3)

    with col1:
        visualization_mode = st.selectbox(
            "🎨 Visualization Mode",
            ["Cognitive Landscape", "Document Landscape", "Memory Clusters", "Temporal Flow", "Concept Networks"],
            help="Select the type of cognitive visualization"
        )

    with col2:
        cluster_method = st.selectbox(
            "🔬 Clustering Method",
            ["UMAP + HDBSCAN", "t-SNE + K-Means", "PCA + Gaussian Mixture"],
            help="Choose the dimensionality reduction and clustering approach"
        )

        # Add info button for clustering method descriptions
        if st.button("ℹ️ Method Info", help="Learn about clustering methods"):
            st.session_state.show_clustering_info = True

    # Show clustering method descriptions if requested
    if hasattr(st.session_state, 'show_clustering_info') and st.session_state.show_clustering_info:
        with st.expander("🔬 Clustering Method Descriptions", expanded=True):
            st.markdown("""
            ### 🧠 Understanding Clustering Methods

            **🎯 UMAP + HDBSCAN** *(Recommended)*
            - **UMAP**: Uniform Manifold Approximation and Projection - preserves both local and global structure
            - **HDBSCAN**: Hierarchical Density-Based Clustering - finds clusters of varying densities
            - **Best for**: Large document collections, preserves semantic relationships
            - **Strengths**: Excellent at revealing document topic clusters, handles noise well

            **📊 t-SNE + K-Means**
            - **t-SNE**: t-Distributed Stochastic Neighbor Embedding - emphasizes local similarities
            - **K-Means**: Partitions data into spherical clusters
            - **Best for**: Clear visual separation, when you know approximate number of topics
            - **Strengths**: Creates distinct, well-separated clusters

            **📈 PCA + Gaussian Mixture**
            - **PCA**: Principal Component Analysis - linear dimensionality reduction
            - **Gaussian Mixture**: Assumes data comes from mixture of Gaussian distributions
            - **Best for**: Linear relationships, overlapping topic boundaries
            - **Strengths**: Fast, interpretable dimensions, handles overlapping concepts

            💡 **Tip**: Start with UMAP + HDBSCAN for document clustering, then try others for different perspectives!
            """)

            if st.button("✅ Got it!", key="close_clustering_info"):
                st.session_state.show_clustering_info = False
                st.rerun()

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
            min_cluster_size = st.slider("Min Cluster Size", 2, 15, 3, help="Minimum memories per cluster (lower = more clusters)")
            max_clusters = st.slider("Max Clusters", 10, 100, 50, help="Maximum clusters to display (higher = more detail)")
        with col2:
            perplexity = st.slider("Perplexity", 5, 50, 30, help="t-SNE perplexity parameter")
            n_neighbors = st.slider("Neighbors", 5, 50, 15, help="UMAP n_neighbors parameter")
            clustering_eps = st.slider("Clustering Eps", 0.3, 1.2, 0.8, 0.1, help="DBSCAN eps parameter (higher = fewer, larger clusters)")
        with col3:
            show_connections = st.checkbox("Show Connections", True, help="Display memory connections")
            show_labels = st.checkbox("Show Labels", True, help="Display cluster labels")
            quality_threshold = st.slider("Quality Threshold", 0.05, 0.5, 0.1, 0.05, help="Minimum cluster quality (lower = more clusters)")

        # Add preset buttons for common scenarios
        st.markdown("**🎯 Quick Presets:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📚 Document Rich", help="Many small document clusters"):
                st.session_state.preset_eps = 0.6
                st.session_state.preset_min_size = 3
                st.session_state.preset_max_clusters = 50
                st.session_state.preset_quality = 0.05
        with col2:
            if st.button("🎯 Balanced", help="Moderate number of meaningful clusters"):
                st.session_state.preset_eps = 0.4
                st.session_state.preset_min_size = 5
                st.session_state.preset_max_clusters = 25
                st.session_state.preset_quality = 0.2
        with col3:
            if st.button("🏔️ High Level", help="Few large topic clusters"):
                st.session_state.preset_eps = 0.3
                st.session_state.preset_min_size = 10
                st.session_state.preset_max_clusters = 10
                st.session_state.preset_quality = 0.4

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

    col1, col2, col3 = st.columns(3)

    with col1:
        # Generate cognitive map (preserved existing functionality)
        if st.button("🎨 Generate Dream Canvas", type="primary", help="Create interactive memory landscape visualization"):
            with st.spinner("🧠 Synthesizing cognitive landscape..."):
                try:
                    # Get memory data
                    memory_stats = memory_store.get_memory_stats()
                    total_memories = memory_stats['total_memories']

                    # Create progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    status_text.text(f"📊 Loading {total_memories} memories...")
                    progress_bar.progress(0.1)

                    # Generate cognitive map with progress tracking
                    status_text.text("🔬 Performing clustering analysis...")
                    progress_bar.progress(0.3)

                    cognitive_map = generate_cognitive_map(
                        memory_store=memory_store,
                        method=cluster_method,
                        time_range=time_range,
                        n_components=n_components,
                        min_cluster_size=min_cluster_size,
                        perplexity=perplexity,
                        n_neighbors=n_neighbors,
                        clustering_eps=clustering_eps,
                        max_clusters=max_clusters,
                        quality_threshold=quality_threshold
                    )

                    status_text.text("🎨 Finalizing visualization...")
                    progress_bar.progress(0.9)

                    # Store in session state
                    st.session_state.cognitive_map = cognitive_map
                    st.session_state.visualization_mode = visualization_mode
                    st.session_state.show_connections = show_connections
                    st.session_state.show_labels = show_labels

                    progress_bar.progress(1.0)
                    status_text.text("✅ Processing complete!")

                    # Show final success message with details
                    st.success(f"✅ Generated landscape with {len(cognitive_map.clusters)} clusters from {total_memories} memories!")

                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()

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

                    # Pre-check: Verify we have memories to synthesize
                    memory_stats = memory_store.get_memory_stats()
                    total_memories = memory_stats.get('total_memories', 0)

                    if total_memories == 0:
                        st.warning("⚠️ No memories found in SAM's knowledge base.")
                        st.info("""
                        **To use Dream Canvas synthesis, you need to add some memories first:**

                        📚 **Add Documents**: Go to Memory Center → Document Upload to add PDFs, text files, etc.

                        💬 **Have Conversations**: Chat with SAM to create conversation memories

                        📁 **Bulk Ingestion**: Use Memory Center → Bulk Ingestion to process multiple files

                        🔄 **Import Data**: Use any of SAM's data ingestion features
                        """)
                        return

                    st.info(f"📊 Processing {total_memories} memories for synthesis...")

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

                    # Regenerate cognitive map with real clustering data to ensure consistency
                    try:
                        logger.info("Regenerating cognitive map after synthesis...")
                        updated_cognitive_map = generate_cognitive_map(
                            memory_store=memory_store,
                            method="UMAP + HDBSCAN",  # Match synthesis default
                            time_range="All Time",
                            n_components=2,
                            min_cluster_size=8,  # Match SynthesisConfig.min_cluster_size
                            perplexity=30,
                            n_neighbors=15,
                            clustering_eps=0.3,  # Match SynthesisConfig.clustering_eps
                            max_clusters=10,      # Match SynthesisConfig.max_clusters
                            quality_threshold=0.5,  # Match SynthesisConfig.quality_threshold
                            clustering_min_samples=3,  # Match SynthesisConfig.clustering_min_samples
                            disable_kmeans_fallback=True  # Keep cluster IDs stable for insight mapping
                        )
                        st.session_state.cognitive_map = updated_cognitive_map
                        logger.info("Cognitive map updated with synthesis results")
                    except Exception as map_error:
                        logger.warning(f"Failed to update cognitive map: {map_error}")

                    # Provide detailed feedback based on results
                    if result.insights_generated > 0:
                        st.success(f"✨ Synthesis complete! Generated **{result.insights_generated} insights** from **{result.clusters_found} clusters**.")
                        st.info("💡 To view them, expand a cluster in 'Most Coherent Clusters' and click 📚 Show Insights.")
                        st.session_state.show_insight_archive = True
                        st.session_state.auto_expand_first_insight_cluster = True
                        st.rerun()
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

    with col3:
        # Insight Archive button - shows emergent insights in clusters
        st.markdown("**🔍 Insights**")
        archive_active = hasattr(st.session_state, 'show_insight_archive') and st.session_state.show_insight_archive

        if archive_active:
            # Show active state with option to hide
            if st.button("📚 Hide Insights", type="secondary", help="Hide emergent insights from cluster details", use_container_width=True):
                st.session_state.show_insight_archive = False
                st.rerun()
            st.caption("✨ Insights visible in clusters")
        else:
            # Show inactive state with option to show
            if st.button("📚 Show Insights", type="secondary", help="View emergent insights within cluster details", use_container_width=True):
                st.session_state.show_insight_archive = True
                st.rerun()
            st.caption("💡 Click to reveal insights")

        # Add Archived Insights link
        if st.button("📚 Archived Insights", help="View all archived insights from previous synthesis runs", use_container_width=True):
            # Navigate to archived insights page
            st.session_state.navigate_to_archived_insights = True
            st.info("💡 Navigate to 'Archived Insights' in the main menu to view all archived insights")
            st.rerun()

        # Auto Research controls
        st.markdown("---")
        st.markdown("**🧪 Research Mode**")
        auto_on = st.toggle("Auto-run research on select", key="auto_run_research_on_select", help="If on, selecting an insight's 🔬 checkbox starts research immediately")
        if auto_on:
            # Do not assign to st.session_state["auto_run_research_on_select"] here; widget manages it
            st.session_state.auto_research_mode = st.radio("Mode", ["Deep", "Quick"], index=0, horizontal=True, key="auto_research_mode_selector")
            st.session_state.deep_research_download_limit = st.number_input("Deep Research download limit (top-K)", min_value=1, max_value=10, value=int(st.session_state.get('deep_research_download_limit', 3)), step=1, help="Only the top-K papers will be downloaded and assessed per insight.")
        else:
            # Do not assign to st.session_state["auto_run_research_on_select"] here; widget manages it
            pass

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
    min_cluster_size: int = 3,
    perplexity: int = 30,
    n_neighbors: int = 15,
    clustering_eps: float = 0.8,
    max_clusters: int = 50,
    quality_threshold: float = 0.1,
    clustering_min_samples: int = 2,
    disable_kmeans_fallback: bool = False
) -> CognitiveMap:
    """Generate a cognitive map from real memory data using clustering and dimensionality reduction."""

    logger.info(f"Generating cognitive map with method: {method}, time_range: {time_range}")

    try:
        # Import required modules
        from memory.synthesis.clustering_service import ClusteringService
        import numpy as np

        # Get all memories from the store
        all_memories = memory_store.get_all_memories()
        logger.info(f"Retrieved {len(all_memories)} memories for cognitive map generation")

        if len(all_memories) < 3:
            logger.warning("Insufficient memories for cognitive map generation, using fallback")
            return _generate_fallback_cognitive_map(method, time_range, n_components)

        # Use user-configurable clustering parameters for rich document clustering
        clustering_service = ClusteringService(
            eps=clustering_eps,  # Keep eps consistent with synthesis
            min_samples=clustering_min_samples,  # Keep min_samples consistent with synthesis
            min_cluster_size=min_cluster_size,  # User-configurable minimum size
            max_clusters=max_clusters,  # User-configurable maximum clusters
            quality_threshold=quality_threshold  # User-configurable quality threshold
        )

        logger.info(f"Using clustering parameters: eps={clustering_eps}, min_size={min_cluster_size}, max_clusters={max_clusters}, quality={quality_threshold}")

        # Get concept clusters - try multiple approaches for rich clustering
        concept_clusters = clustering_service.find_concept_clusters(memory_store)
        logger.info(f"Found {len(concept_clusters)} concept clusters for visualization")

        # If DBSCAN produces too few clusters, optionally try K-Means for forced clustering
        if not disable_kmeans_fallback and len(concept_clusters) < 5 and max_clusters >= 10:
            logger.info("DBSCAN produced few clusters, trying K-Means for richer clustering...")
            concept_clusters = _try_kmeans_clustering(memory_store, min(max_clusters, 20), min_cluster_size, quality_threshold)
            logger.info(f"K-Means produced {len(concept_clusters)} clusters")

        if len(concept_clusters) == 0:
            logger.warning("No clusters found, using fallback cognitive map")
            return _generate_fallback_cognitive_map(method, time_range, n_components)

        # Apply dimensionality reduction for visualization
        clusters, connections = _apply_dimensionality_reduction(
            concept_clusters, method, n_components, perplexity, n_neighbors
        )

        # Create cognitive map
        cognitive_map = CognitiveMap(
            clusters=clusters,
            connections=connections,
            metadata={
                'method': method,
                'time_range': time_range,
                'n_components': n_components,
                'generated_at': datetime.now().isoformat(),
                'total_memories': sum(len(c.memories) for c in clusters),
                'real_clustering': True,
                'concept_clusters_found': len(concept_clusters)
            }
        )

        logger.info(f"Generated cognitive map with {len(clusters)} clusters and {len(connections)} connections")
        return cognitive_map

    except Exception as e:
        logger.error(f"Error generating cognitive map: {e}")
        logger.info("Falling back to template cognitive map")
        return _generate_fallback_cognitive_map(method, time_range, n_components)

def _generate_fallback_cognitive_map(method: str, time_range: str, n_components: int) -> CognitiveMap:
    """Generate a fallback cognitive map with template data when real clustering fails."""

    # Template cluster generation
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
            'total_memories': sum(len(c.memories) for c in clusters),
            'fallback_mode': True
        }
    )

    return cognitive_map

def _apply_dimensionality_reduction(concept_clusters, method: str, n_components: int, perplexity: int, n_neighbors: int):
    """Apply dimensionality reduction to concept clusters for visualization."""

    try:
        import numpy as np
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA

        # Extract embeddings from concept clusters
        all_embeddings = []
        cluster_mapping = []

        for cluster_idx, concept_cluster in enumerate(concept_clusters):
            for chunk in concept_cluster.chunks:
                if hasattr(chunk, 'embedding') and chunk.embedding is not None:
                    all_embeddings.append(chunk.embedding)
                    cluster_mapping.append(cluster_idx)

        if len(all_embeddings) == 0:
            logger.warning("No embeddings found in concept clusters")
            return [], []

        embeddings_array = np.array(all_embeddings)
        logger.info(f"Applying {method} dimensionality reduction to {len(embeddings_array)} embeddings")

        # Apply dimensionality reduction with better parameters for visualization
        if method == "UMAP + HDBSCAN":
            try:
                import umap
                # Optimized UMAP parameters for better cluster separation
                reducer = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=min(n_neighbors, len(embeddings_array)//2),
                    min_dist=0.3,  # Increase minimum distance for better separation
                    spread=2.0,    # Increase spread for better distribution
                    random_state=42
                )
                reduced_embeddings = reducer.fit_transform(embeddings_array)
            except ImportError:
                logger.warning("UMAP not available, falling back to t-SNE")
                reducer = TSNE(
                    n_components=n_components,
                    perplexity=min(perplexity, len(embeddings_array)-1),
                    learning_rate=200,  # Better learning rate
                    random_state=42
                )
                reduced_embeddings = reducer.fit_transform(embeddings_array)
        elif method == "t-SNE + K-Means":
            reducer = TSNE(
                n_components=n_components,
                perplexity=min(perplexity, len(embeddings_array)-1),
                learning_rate=200,
                early_exaggeration=12,  # Better early exaggeration
                random_state=42
            )
            reduced_embeddings = reducer.fit_transform(embeddings_array)
        else:  # PCA + Gaussian Mixture
            reducer = PCA(n_components=n_components, random_state=42)
            reduced_embeddings = reducer.fit_transform(embeddings_array)

        # Normalize coordinates to improve visualization
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        reduced_embeddings = scaler.fit_transform(reduced_embeddings)

        # Convert concept clusters to visualization clusters
        clusters = []
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']

        for i, concept_cluster in enumerate(concept_clusters):
            # Find embeddings belonging to this cluster
            cluster_indices = [j for j, cluster_idx in enumerate(cluster_mapping) if cluster_idx == i]

            if len(cluster_indices) == 0:
                continue

            cluster_embeddings = reduced_embeddings[cluster_indices]

            # Calculate cluster center with better positioning
            center = np.mean(cluster_embeddings, axis=0)

            # Add slight separation to prevent overlap
            if len(clusters) > 0:
                # Check distance to existing clusters and adjust if too close
                min_separation = 0.5  # Minimum distance between cluster centers
                for existing_cluster in clusters:
                    existing_center = np.array(existing_cluster.center)
                    distance = np.linalg.norm(center - existing_center)
                    if distance < min_separation:
                        # Push cluster away from existing one
                        direction = center - existing_center
                        if np.linalg.norm(direction) > 0:
                            direction = direction / np.linalg.norm(direction)
                            center = existing_center + direction * min_separation

            # Convert memory chunks to simple memory format
            memories = []
            for chunk in concept_cluster.chunks:
                memory = {
                    'id': chunk.chunk_id,
                    'content': chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content,
                    'timestamp': chunk.timestamp,
                    'embedding': chunk.embedding,
                    'metadata': {
                        'importance': chunk.importance_score,
                        'source': chunk.source
                    }
                }
                memories.append(memory)

            # Create visualization cluster
            cluster = MemoryCluster(
                id=concept_cluster.cluster_id,
                name=f"Cluster {i+1}" if not concept_cluster.dominant_themes else concept_cluster.dominant_themes[0],
                memories=memories,
                center=tuple(center),
                color=colors[i % len(colors)],
                size=concept_cluster.size,
                coherence_score=concept_cluster.coherence_score
            )
            clusters.append(cluster)

        # Apply cluster separation enhancement
        clusters = _enhance_cluster_separation(clusters)

        # Generate connections based on cluster similarity
        connections = []
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                # Calculate distance between cluster centers
                center1 = np.array(clusters[i].center)
                center2 = np.array(clusters[j].center)
                distance = np.linalg.norm(center1 - center2)

                # Create connection if clusters are semantically related but visually separated
                max_distance = 3.0  # Fixed threshold for better control
                if distance < max_distance:
                    strength = 1.0 - (distance / max_distance)  # Closer = stronger
                    connection = {
                        'source': clusters[i].id,
                        'target': clusters[j].id,
                        'strength': strength,
                        'type': 'semantic_similarity'
                    }
                    connections.append(connection)

        logger.info(f"Created {len(clusters)} visualization clusters with {len(connections)} connections")
        return clusters, connections

    except Exception as e:
        logger.error(f"Error in dimensionality reduction: {e}")
        return [], []

def _extract_pdf_files_from_cluster(cluster):
    """Extract PDF file information from cluster memories."""
    import re
    from collections import defaultdict

    pdf_files = defaultdict(lambda: {'chunk_count': 0, 'file_path': None})

    try:
        # Extract PDF info from cluster memories
        for memory in cluster.memories:
            source = memory.get('metadata', {}).get('source', '') or memory.get('source', '')

            if source:
                # Extract filename from various source formats
                filename = _extract_clean_filename(source)

                if filename and filename.lower().endswith('.pdf'):
                    pdf_files[filename]['chunk_count'] += 1

                    # Try to find the actual file path
                    if not pdf_files[filename]['file_path']:
                        file_path = _find_pdf_file_path(source, filename)
                        pdf_files[filename]['file_path'] = file_path

        # Convert to list format sorted by chunk count
        result = []
        for filename, info in pdf_files.items():
            result.append({
                'filename': filename,
                'chunk_count': info['chunk_count'],
                'file_path': info['file_path']
            })

        # Sort by chunk count (most chunks first)
        result.sort(key=lambda x: x['chunk_count'], reverse=True)

        return result

    except Exception as e:
        logger.error(f"Error extracting PDF files from cluster: {e}")
        return []

def _extract_clean_filename(source: str) -> str:
    """Extract clean filename from various source formats."""
    import re

    try:
        # Handle "document:web_ui/uploads/20250606_154557_filename.pdf:block_1"
        match = re.search(r'uploads/\d{8}_\d{6}_([^:]+)', source)
        if match:
            return match.group(1)

        # Handle "document:filename.pdf" or "document:filename.pdf:block_1"
        if source.startswith('document:'):
            filename_part = source[9:]  # Remove "document:" prefix
            filename = filename_part.split(':')[0]  # Remove ":block_X" suffix
            if not filename.startswith('web_ui/'):
                return filename

        # Handle direct filenames
        filename = source.split('/')[-1].split(':')[0]
        if '.' in filename:
            return filename

        return None

    except Exception:
        return None

def _find_pdf_file_path(source: str, filename: str) -> str:
    """Find the actual file path for a PDF."""
    try:
        # Common upload directories
        upload_dirs = [
            Path("web_ui/uploads"),
            Path("uploads"),
            Path("documents"),
            Path("data/documents")
        ]

        # Try to extract path from source
        if 'uploads/' in source:
            # Extract full path from source
            import re
            match = re.search(r'(web_ui/uploads/[^:]+)', source)
            if match:
                potential_path = Path(match.group(1))
                if potential_path.exists():
                    return str(potential_path)

        # Search in common directories
        for upload_dir in upload_dirs:
            if upload_dir.exists():
                # Look for exact filename
                exact_path = upload_dir / filename
                if exact_path.exists():
                    return str(exact_path)

                # Look for files with timestamp prefix
                for file_path in upload_dir.glob(f"*_{filename}"):
                    if file_path.exists():
                        return str(file_path)

                # Look for files containing the filename
                for file_path in upload_dir.glob("*.pdf"):
                    if filename in file_path.name:
                        return str(file_path)

        return None

    except Exception:
        return None

def _render_cluster_deep_research_controls(cluster_id: str):
    """Render Deep Research controls for selected cluster insights (unique keys per cluster)."""
    if 'selected_cluster_insights' not in st.session_state or not st.session_state.selected_cluster_insights:
        return

    selected_count = len(st.session_state.selected_cluster_insights)

    st.markdown("---")
    st.markdown(f"**🔬 Deep Research ({selected_count} insights selected)**")

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        if st.button(
            "🚀 Start Deep Research",
            key=f"start_deep_research_{cluster_id}",
            type="primary",
            help=f"Research {selected_count} selected insights"
        ):
            _execute_cluster_deep_research()

    with col2:
        if st.button(
            "🗑️ Clear Selection",
            key=f"clear_selection_{cluster_id}",
            help="Clear all selected insights"
        ):
            st.session_state.selected_cluster_insights.clear()
            if 'cluster_insight_data' in st.session_state:
                st.session_state.cluster_insight_data.clear()
            st.rerun()

    with col3:
        st.caption(f"📊 {selected_count} selected")

def _execute_cluster_deep_research():
    """Execute Deep Research pipeline for selected cluster insights."""
    try:
        if 'selected_cluster_insights' not in st.session_state or not st.session_state.selected_cluster_insights:
            st.error("No insights selected for research")
            return

        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()

        selected_insights = st.session_state.cluster_insight_data
        total_insights = len(selected_insights)

        for i, (insight_id, insight_data) in enumerate(selected_insights.items()):
            progress = (i + 1) / total_insights
            progress_bar.progress(progress)
            status_text.text(f"🔬 Researching insight {i+1}/{total_insights}...")

            # Execute research for this insight
            _research_single_cluster_insight(insight_id, insight_data)

        progress_bar.progress(1.0)
        status_text.text("✅ Deep Research completed!")

        # Clear selection after successful research
        st.session_state.selected_cluster_insights.clear()
        st.session_state.cluster_insight_data.clear()

        st.success(f"🎉 Deep Research completed for {total_insights} insights!")
        st.rerun()

    except Exception as e:
        logger.error(f"Deep Research execution failed: {e}")
        st.error(f"❌ Deep Research failed: {e}")

def _research_single_cluster_insight(insight_id: str, insight_data: Dict[str, Any]):
    """Research a single cluster insight using the automated pipeline."""
    try:
        insight = insight_data['insight']
        cluster_name = insight_data['cluster_name']

        # Extract weighted keywords from insight
        insight_text = insight.get('synthesized_text', insight.get('content', ''))
        keywords = _extract_weighted_keywords_from_insight(insight_text)

        # Search ArXiv using keywords
        search_query = ' '.join(keywords[:5])  # Use top 5 keywords

        logger.info(f"🔍 Searching ArXiv for: {search_query}")

        # Use ArXiv tool to search and download
        from sam.web_retrieval.tools.arxiv_tool import get_arxiv_tool
        arxiv_tool = get_arxiv_tool()

        result = arxiv_tool.search_and_download(
            query=search_query,
            insight_text=insight_text
        )

        if result['success']:
            # Auto-ingest the downloaded paper
            paper_path = result['paper_metadata']['local_path']
            _auto_ingest_research_paper(paper_path, insight_id, cluster_name, keywords)

            # Store research result for display
            _store_research_result(insight_id, result, keywords)

            # Mark status on original insight if present in session
            try:
                for key, data in (st.session_state.get('cluster_insight_data') or {}).items():
                    if key == insight_id:
                        data['insight']['research_status'] = 'completed'
                        break
            except Exception:
                pass

            logger.info(f"✅ Research completed for insight: {insight_id}")
        else:
            try:
                for key, data in (st.session_state.get('cluster_insight_data') or {}).items():
                    if key == insight_id:
                        data['insight']['research_status'] = 'failed'
                        break
            except Exception:
                pass
            logger.warning(f"⚠️ Research failed for insight {insight_id}: {result['error']}")

    except Exception as e:
        logger.error(f"Single insight research failed: {e}")

def _extract_weighted_keywords_from_insight(insight_text: str) -> List[str]:
    """Extract weighted keywords from insight text for ArXiv search."""
    try:
        import re
        from collections import Counter

        # Clean text
        text = re.sub(r'<[^>]+>', '', insight_text)  # Remove HTML tags
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation

        # Extract words
        words = text.lower().split()

        # Remove stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
            'might', 'can', 'cannot', 'not', 'no', 'yes', 'also', 'very', 'much', 'more', 'most',
            'some', 'any', 'all', 'each', 'every', 'both', 'either', 'neither', 'one', 'two',
            'first', 'second', 'last', 'next', 'previous', 'new', 'old', 'good', 'bad', 'big', 'small'
        }

        # Filter meaningful words
        meaningful_words = [
            word for word in words
            if word not in stop_words and len(word) > 3 and word.isalpha()
        ]

        # Count frequency and apply weights
        word_freq = Counter(meaningful_words)

        # Apply domain-specific weights
        domain_weights = {
            'neural': 2.0, 'network': 2.0, 'learning': 2.0, 'machine': 2.0, 'deep': 2.0,
            'artificial': 2.0, 'intelligence': 2.0, 'algorithm': 2.0, 'model': 2.0,
            'data': 1.5, 'analysis': 1.5, 'method': 1.5, 'approach': 1.5, 'technique': 1.5,
            'research': 1.5, 'study': 1.5, 'experiment': 1.5, 'result': 1.5, 'finding': 1.5
        }

        # Calculate weighted scores
        weighted_scores = {}
        for word, freq in word_freq.items():
            weight = domain_weights.get(word, 1.0)
            weighted_scores[word] = freq * weight

        # Return top keywords sorted by weighted score
        sorted_keywords = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
        return [word for word, score in sorted_keywords[:10]]

    except Exception as e:
        logger.error(f"Keyword extraction failed: {e}")
        return ['machine learning', 'artificial intelligence']  # Fallback keywords

def _auto_ingest_research_paper(paper_path: str, insight_id: str, cluster_name: str, keywords: List[str]):
    """Auto-ingest downloaded research paper with Deep Research metadata."""
    try:
        from sam.ingestion.v2_ingestion_pipeline import ingest_document_v2
        from datetime import datetime

        # Create rich metadata for the research paper
        metadata = {
            'source_type': 'deep_research',
            'research_context': 'cluster_insight_research',
            'insight_id': insight_id,
            'cluster_name': cluster_name,
            'research_keywords': keywords,
            'research_timestamp': datetime.now().isoformat(),
            'priority_score': 0.9,  # High priority for research papers
            'auto_ingested': True,
            'research_session': f"deep_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }

        # Ingest the document
        result = ingest_document_v2(
            file_path=paper_path,
            metadata=metadata
        )

        if result.successful_documents > 0:
            logger.info(f"✅ Auto-ingested research paper: {paper_path}")

            # Trigger auto-synthesis after ingestion
            _trigger_auto_synthesis_after_research()
        else:
            logger.warning(f"⚠️ Auto-ingestion failed for: {paper_path}")

    except Exception as e:
        logger.error(f"Auto-ingestion failed: {e}")

def _trigger_auto_synthesis_after_research():
    """Trigger automatic synthesis after research paper ingestion."""
    try:
        from memory.synthesis.synthesis_engine import SynthesisEngine
        from memory.memory_vectorstore import get_memory_store

        # Get memory store
        memory_store = get_memory_store()

        # Initialize synthesis engine
        synthesis_engine = SynthesisEngine()

        # Run synthesis
        logger.info("🧠 Triggering auto-synthesis after Deep Research...")
        synthesis_result = synthesis_engine.run_synthesis(
            memory_store=memory_store,
            visualize=False,  # Don't show visualization during auto-synthesis
            save_output=True
        )

        logger.info("✅ Auto-synthesis completed after Deep Research")

    except Exception as e:
        logger.error(f"Auto-synthesis failed: {e}")

def _store_research_result(insight_id: str, result: Dict[str, Any], keywords: List[str]):
    """Store research result for display below the original insight."""
    if 'deep_research_results' not in st.session_state:
        st.session_state.deep_research_results = {}

    st.session_state.deep_research_results[insight_id] = {
        'result': result,
        'keywords': keywords,
        'timestamp': datetime.now().isoformat(),
        'status': 'completed'
    }

    # Update belief/evidence using the downloaded paper
    try:
        # Find the original insight object by id
        insight_record = (st.session_state.get('cluster_insight_data') or {}).get(insight_id)
        if insight_record:
            insight = insight_record.get('insight') or {}
            evidence = _llm_assess_evidence(
                insight.get('synthesized_text', insight.get('content', '')),
                result.get('paper_metadata', {}),
                result.get('local_path')
            )
            _apply_evidence_update(insight, evidence)
    except Exception as _e:
        logger.debug(f"Evidence update failed: {_e}")

def _render_deep_research_results_for_insight(insight_id: str):
    """Render Deep Research results below an insight if available."""
    if 'deep_research_results' not in st.session_state:
        return

    if insight_id not in st.session_state.deep_research_results:
        return

    research_data = st.session_state.deep_research_results[insight_id]
    result = research_data['result']
    keywords = research_data['keywords']

    st.markdown("---")
    st.markdown("**🔬 Deep Research Results:**")

    if result['success']:
        paper_metadata = result['paper_metadata']

        # Update any matching insight's status to completed
        try:
            if 'deep_research_results' in st.session_state:
                for key, data in st.session_state.deep_research_results.items():
                    pass  # placeholder for mapping if needed
        except Exception:
            pass

        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(f"📄 **{paper_metadata['title']}**")
            st.caption(f"Authors: {', '.join(paper_metadata['authors'][:3])}")
            st.caption(f"ArXiv ID: {paper_metadata['arxiv_id']}")

            # Show keywords used
            st.caption(f"🔍 Keywords: {', '.join(keywords[:5])}")

        with col2:
            st.success("✅ Downloaded & Ingested")
            st.caption(f"📊 Score: {result.get('relevance_score', 'N/A')}")

        # Show paper summary (avoid nested expanders)
        show_summary_key = f"show_{insight_id}_paper_summary"
        if st.checkbox("📋 Paper Summary", key=show_summary_key):
            summary = paper_metadata.get('summary', 'No summary available')
            st.markdown(summary[:500] + "..." if len(summary) > 500 else summary)
    else:
        st.error(f"❌ Research failed: {result['error']}")

def _try_kmeans_clustering(memory_store, n_clusters: int, min_cluster_size: int, quality_threshold: float):
    """Try K-Means clustering to force a specific number of clusters."""
    try:
        from sklearn.cluster import KMeans
        from memory.synthesis.clustering_service import ConceptCluster
        import numpy as np
        import uuid

        logger.info(f"Attempting K-Means clustering with {n_clusters} clusters")

        # Get all memories and embeddings
        all_memories = memory_store.get_all_memories()
        if len(all_memories) < n_clusters:
            logger.warning(f"Not enough memories ({len(all_memories)}) for {n_clusters} clusters")
            return []

        # Extract embeddings
        embeddings = []
        valid_memories = []

        for memory in all_memories:
            if hasattr(memory, 'embedding') and memory.embedding is not None:
                embeddings.append(memory.embedding)
                valid_memories.append(memory)

        if len(embeddings) < n_clusters:
            logger.warning(f"Not enough embeddings ({len(embeddings)}) for {n_clusters} clusters")
            return []

        embeddings_array = np.array(embeddings)

        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_array)

        # Create ConceptCluster objects
        concept_clusters = []
        for cluster_id in range(n_clusters):
            # Get memories for this cluster
            cluster_indices = np.where(cluster_labels == cluster_id)[0]

            if len(cluster_indices) >= min_cluster_size:
                cluster_memories = [valid_memories[i] for i in cluster_indices]
                cluster_embeddings = embeddings_array[cluster_indices]

                # Calculate centroid and coherence
                centroid = np.mean(cluster_embeddings, axis=0)

                # Calculate coherence (average cosine similarity to centroid)
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(cluster_embeddings, [centroid])
                coherence_score = np.mean(similarities)

                # Only include if meets quality threshold
                if coherence_score >= quality_threshold:
                    # Create chunk objects (simplified)
                    chunks = []
                    chunk_ids = []
                    for memory in cluster_memories:
                        chunk_id = str(uuid.uuid4())
                        chunk_ids.append(chunk_id)
                        # Create a simple chunk-like object with all required attributes
                        chunk = type('MemoryChunk', (), {
                            'chunk_id': chunk_id,
                            'content': getattr(memory, 'content', str(memory)),
                            'source': getattr(memory, 'source', 'Unknown'),
                            'embedding': getattr(memory, 'embedding', None),
                            'timestamp': getattr(memory, 'timestamp', '2024-01-01T00:00:00'),
                            'metadata': getattr(memory, 'metadata', {}),
                            'memory_type': getattr(memory, 'memory_type', 'DOCUMENT'),
                            'importance_score': getattr(memory, 'importance_score', 0.5),
                            'relevance_score': getattr(memory, 'relevance_score', 0.5),
                            'quality_score': getattr(memory, 'quality_score', 0.5)
                        })()
                        chunks.append(chunk)

                    # Extract themes (simplified)
                    themes = [f"Document Cluster {cluster_id + 1}"]

                    concept_cluster = ConceptCluster(
                        cluster_id=f"kmeans_cluster_{cluster_id}",
                        chunk_ids=chunk_ids,
                        chunks=chunks,
                        centroid=centroid,
                        coherence_score=coherence_score,
                        size=len(cluster_memories),
                        dominant_themes=themes,
                        metadata={'method': 'kmeans', 'forced_clustering': True}
                    )

                    concept_clusters.append(concept_cluster)

        logger.info(f"K-Means created {len(concept_clusters)} quality clusters")
        return concept_clusters

    except Exception as e:
        logger.error(f"Error in K-Means clustering: {e}")
        return []

def _enhance_cluster_separation(clusters):
    """Enhance cluster separation to prevent overlapping in visualization."""
    import numpy as np

    if len(clusters) <= 1:
        return clusters

    # Minimum separation distance
    min_separation = 1.0

    # Apply force-directed separation
    for iteration in range(10):  # Multiple iterations for better separation
        moved = False

        for i, cluster_i in enumerate(clusters):
            center_i = np.array(cluster_i.center)

            for j, cluster_j in enumerate(clusters):
                if i >= j:
                    continue

                center_j = np.array(cluster_j.center)
                distance = np.linalg.norm(center_i - center_j)

                if distance < min_separation and distance > 0:
                    # Calculate repulsion force
                    direction = (center_i - center_j) / distance
                    push_distance = (min_separation - distance) / 2

                    # Move clusters apart
                    new_center_i = center_i + direction * push_distance
                    new_center_j = center_j - direction * push_distance

                    # Update cluster centers
                    clusters[i] = MemoryCluster(
                        id=cluster_i.id,
                        name=cluster_i.name,
                        memories=cluster_i.memories,
                        center=tuple(new_center_i),
                        color=cluster_i.color,
                        size=cluster_i.size,
                        coherence_score=cluster_i.coherence_score
                    )

                    clusters[j] = MemoryCluster(
                        id=cluster_j.id,
                        name=cluster_j.name,
                        memories=cluster_j.memories,
                        center=tuple(new_center_j),
                        color=cluster_j.color,
                        size=cluster_j.size,
                        coherence_score=cluster_j.coherence_score
                    )

                    moved = True

        if not moved:
            break

    return clusters

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
    elif mode == "Document Landscape":
        render_document_landscape_view(cognitive_map, show_connections, show_labels)
    elif mode == "Memory Clusters":
        render_cluster_view(cognitive_map)
    elif mode == "Temporal Flow":
        render_temporal_view(cognitive_map)
    elif mode == "Concept Networks":
        render_network_view(cognitive_map, show_connections)

def render_landscape_view(cognitive_map: CognitiveMap, show_connections: bool, show_labels: bool):
    """Render the main landscape visualization with improved scaling and separation."""
    import numpy as np

    # Create scatter plot data with improved scaling
    x_coords = []
    y_coords = []
    colors = []
    sizes = []
    texts = []

    # Calculate better size scaling
    if cognitive_map.clusters:
        cluster_sizes = [cluster.size for cluster in cognitive_map.clusters]
        min_size = min(cluster_sizes)
        max_size = max(cluster_sizes)

        # Normalize sizes to reasonable range (15-60 pixels)
        size_range = max_size - min_size if max_size > min_size else 1

        for cluster in cognitive_map.clusters:
            x_coords.append(cluster.center[0])
            y_coords.append(cluster.center[1])
            colors.append(cluster.color)

            # Better size scaling: normalize to 15-60 pixel range
            normalized_size = 15 + (cluster.size - min_size) / size_range * 45
            sizes.append(max(15, min(60, normalized_size)))  # Clamp to reasonable range

            texts.append(f"{cluster.name}<br>Memories: {cluster.size}<br>Coherence: {cluster.coherence_score:.2f}")

    if not x_coords:
        st.warning("No clusters available for visualization")
        return

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
                # Find corresponding cluster with improved matching
                matching_cluster = None
                for cluster in cognitive_map.clusters:
                    # Try multiple matching strategies
                    if (cluster.id == cluster_id or
                        cluster.name == cluster_id or
                        cluster.id.endswith(cluster_id) or
                        cluster_id.endswith(cluster.id)):
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

    # Calculate axis ranges for better visualization
    if x_coords and y_coords:
        x_range = [min(x_coords) - 1, max(x_coords) + 1]
        y_range = [min(y_coords) - 1, max(y_coords) + 1]

        # Ensure minimum range for visibility
        x_span = x_range[1] - x_range[0]
        y_span = y_range[1] - y_range[0]

        if x_span < 2:
            center_x = (x_range[0] + x_range[1]) / 2
            x_range = [center_x - 1, center_x + 1]

        if y_span < 2:
            center_y = (y_range[0] + y_range[1]) / 2
            y_range = [center_y - 1, center_y + 1]
    else:
        x_range = [-2, 2]
        y_range = [-2, 2]

    # Update layout with improved scaling
    fig.update_layout(
        title="🧠🎨 Cognitive Memory Landscape with Synthetic Insights",
        xaxis=dict(
            title="Cognitive Dimension 1",
            range=x_range,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='rgba(128,128,128,0.3)'
        ),
        yaxis=dict(
            title="Cognitive Dimension 2",
            range=y_range,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='rgba(128,128,128,0.3)'
        ),
        showlegend=True,
        height=700,  # Increased height for better visibility
        plot_bgcolor='rgba(248,249,250,0.8)',
        paper_bgcolor='white',
        hovermode='closest'
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

def render_document_landscape_view(cognitive_map: CognitiveMap, show_connections: bool, show_labels: bool):
    """Render document-level landscape showing individual documents as circles."""
    import plotly.graph_objects as go
    import numpy as np

    st.markdown("### 📄 Document Landscape")
    st.markdown("*Individual documents positioned by semantic similarity*")

    try:
        # Get memory store to access individual documents
        from memory.memory_vectorstore import get_memory_store
        memory_store = get_memory_store()

        # Get all memories (will group by source document)
        all_memories = memory_store.search_memories("", max_results=2000)

        if not all_memories:
            st.warning("No memories found for visualization")
            return

        # Group memories by source document or content type
        doc_groups = {}
        for memory in all_memories:
            # Try multiple ways to get source information
            source = getattr(memory, 'source', None)
            if not source:
                source = getattr(memory, 'metadata', {}).get('source', None)
            if not source:
                # Group by content type or first few words
                content = getattr(memory, 'content', str(memory))[:50]
                source = f"Memory Group {len(doc_groups) + 1}"

            if source not in doc_groups:
                doc_groups[source] = []
            doc_groups[source].append(memory)

        st.info(f"📊 Visualizing {len(doc_groups)} memory groups with {len(all_memories)} total memories")

        # Create document-level visualization
        fig = go.Figure()

        # Get embeddings for documents (use first chunk as representative)
        doc_embeddings = []
        doc_names = []
        doc_sizes = []
        doc_colors = []

        color_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']

        for i, (doc_name, memories) in enumerate(doc_groups.items()):
            # Try to get embedding from memory
            embedding = None
            for memory in memories:
                if hasattr(memory, 'embedding') and memory.embedding:
                    embedding = memory.embedding
                    break
                elif hasattr(memory, 'vector') and memory.vector:
                    embedding = memory.vector
                    break

            if embedding:
                doc_embeddings.append(embedding)
                doc_names.append(doc_name.split('/')[-1] if '/' in doc_name else doc_name)
                doc_sizes.append(len(memories))  # Size based on number of chunks
                doc_colors.append(color_palette[i % len(color_palette)])

        if not doc_embeddings:
            # Fallback: create synthetic visualization based on content similarity
            st.warning("No embeddings found - creating synthetic layout based on content")

            # Create a simple grid layout for memory groups
            import math
            grid_size = math.ceil(math.sqrt(len(doc_groups)))

            x_coords = []
            y_coords = []
            sizes = []
            colors = []
            names = []

            for i, (doc_name, memories) in enumerate(doc_groups.items()):
                x = i % grid_size
                y = i // grid_size
                x_coords.append(x)
                y_coords.append(y)
                sizes.append(len(memories) * 5 + 10)  # Scale size
                colors.append(color_palette[i % len(color_palette)])
                names.append(doc_name.split('/')[-1] if '/' in doc_name else doc_name)

            # Add synthetic scatter plot
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
                text=names if show_labels else None,
                textposition="middle center",
                textfont=dict(size=8, color='white'),
                hovertemplate='<b>%{text}</b><br>Memories: %{marker.size}<br>Position: (%{x}, %{y})<extra></extra>',
                name='Memory Groups'
            ))

            # Update layout for synthetic view
            fig.update_layout(
                title="📄 Memory Groups Layout (Synthetic - No Embeddings Available)",
                xaxis=dict(title="Grid X", showgrid=True),
                yaxis=dict(title="Grid Y", showgrid=True),
                showlegend=True,
                height=700,
                plot_bgcolor='rgba(248,249,250,0.8)',
                paper_bgcolor='white'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Show statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Memory Groups", len(doc_groups))
            with col2:
                st.metric("Total Memories", len(all_memories))
            with col3:
                avg_memories = len(all_memories) / len(doc_groups) if doc_groups else 0
                st.metric("Avg Memories/Group", f"{avg_memories:.1f}")

            return

        # Apply dimensionality reduction
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler

        embeddings_array = np.array(doc_embeddings)

        # Normalize embeddings
        scaler = StandardScaler()
        embeddings_array = scaler.fit_transform(embeddings_array)

        # Apply t-SNE for 2D visualization
        tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings_array)-1), random_state=42)
        reduced_embeddings = tsne.fit_transform(embeddings_array)

        # Create scatter plot for documents
        x_coords = reduced_embeddings[:, 0]
        y_coords = reduced_embeddings[:, 1]

        # Normalize sizes to reasonable range (10-50 pixels)
        min_size = min(doc_sizes)
        max_size = max(doc_sizes)
        size_range = max_size - min_size if max_size > min_size else 1

        normalized_sizes = []
        for size in doc_sizes:
            normalized_size = 10 + (size - min_size) / size_range * 40
            normalized_sizes.append(max(10, min(50, normalized_size)))

        # Add document scatter plot
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers+text' if show_labels else 'markers',
            marker=dict(
                size=normalized_sizes,
                color=doc_colors,
                opacity=0.7,
                line=dict(width=2, color='white')
            ),
            text=doc_names if show_labels else None,
            textposition="middle center",
            textfont=dict(size=8, color='white'),
            hovertemplate='<b>%{text}</b><br>Chunks: %{marker.size}<br>Position: (%{x:.2f}, %{y:.2f})<extra></extra>',
            name='Documents'
        ))

        # Add connections between similar documents if requested
        if show_connections:
            # Calculate pairwise distances and connect close documents
            for i in range(len(x_coords)):
                for j in range(i+1, len(x_coords)):
                    distance = np.sqrt((x_coords[i] - x_coords[j])**2 + (y_coords[i] - y_coords[j])**2)

                    # Connect documents that are close together
                    if distance < np.std(reduced_embeddings) * 1.5:
                        fig.add_trace(go.Scatter(
                            x=[x_coords[i], x_coords[j]],
                            y=[y_coords[i], y_coords[j]],
                            mode='lines',
                            line=dict(width=1, color='rgba(128,128,128,0.3)'),
                            showlegend=False,
                            hoverinfo='skip'
                        ))

        # Update layout
        fig.update_layout(
            title="📄 Document Landscape - Individual Documents by Semantic Similarity",
            xaxis=dict(
                title="Semantic Dimension 1",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)'
            ),
            yaxis=dict(
                title="Semantic Dimension 2",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)'
            ),
            showlegend=True,
            height=700,
            plot_bgcolor='rgba(248,249,250,0.8)',
            paper_bgcolor='white',
            hovermode='closest'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show document statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Documents", len(doc_groups))
        with col2:
            st.metric("Memory Chunks", len(all_memories))
        with col3:
            avg_chunks = len(all_memories) / len(doc_groups) if doc_groups else 0
            st.metric("Avg Chunks/Doc", f"{avg_chunks:.1f}")

        # Show document list
        with st.expander("📋 Document Details"):
            for doc_name, memories in doc_groups.items():
                st.markdown(f"**{doc_name.split('/')[-1]}**: {len(memories)} chunks")

    except Exception as e:
        st.error(f"❌ Error rendering document landscape: {e}")
        logger.error(f"Document landscape error: {e}")

def render_insight_archive_mode():
    """Render the insight archive interface."""
    try:
        from ui.insight_archive_ui import render_insight_archive
        render_insight_archive()
    except ImportError as e:
        st.error(f"❌ Insight Archive UI not available: {e}")
    except Exception as e:
        st.error(f"❌ Error loading Insight Archive: {e}")

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

    # Auto-expand the first cluster once after synthesis, if requested
    auto_expand_first = st.session_state.get('auto_expand_first_insight_cluster', False)
    if auto_expand_first:
        # We will consume this flag after first render to avoid repeated auto-expands
        st.session_state.auto_expand_first_insight_cluster = False

    for i, cluster in enumerate(top_clusters, 1):
        # Create cluster title with basic info
        cluster_title = f"{i}. {cluster.name} - {cluster.size} memories (coherence: {cluster.coherence_score:.2f})"

        # Determine if this expander should be auto-expanded once
        expanded_flag = (auto_expand_first and i == 1)

        # Add insight indicator if this cluster has generated insights
        insight_indicator = ""
        cluster_insights = []
        if hasattr(st.session_state, 'synthesis_results') and st.session_state.synthesis_results:
            insights = st.session_state.synthesis_results.get('insights', [])
            # Robust matching between insight cluster IDs and visualization clusters
            cluster_insights = []
            for ins in insights:
                cid = str(ins.get('cluster_id', ''))
                if cid == cluster.id or cid == cluster.name or cluster.id.endswith(cid) or cid.endswith(cluster.id):
                    cluster_insights.append(ins)

            if cluster_insights:
                insight_indicator = f" ✨ ({len(cluster_insights)} insights)"

        # Create expandable section for each cluster with status badges
        status_badge = ""
        if any(ins.get('research_status') in ('in_progress', 'completed') for ins in cluster_insights or []):
            # If any insight has research status, show a badge summary
            in_prog = sum(1 for ins in cluster_insights if ins.get('research_status') == 'in_progress')
            done = sum(1 for ins in cluster_insights if ins.get('research_status') == 'completed')
            if in_prog or done:
                status_badge = f"  🧪[{in_prog} in progress] ✅[{done} done]"

        with st.expander(f"🔍 {cluster_title}{insight_indicator}{status_badge}", expanded=expanded_flag):
            # Display detailed cluster information
            render_cluster_detailed_info(cluster, cluster_insights)

def render_cluster_detailed_info(cluster, cluster_insights):
    """Render detailed information for a cluster in the Most Coherent Clusters section."""
    try:
        from memory.synthesis.cluster_registry import get_cluster_stats

        # Get cluster statistics from registry
        cluster_stats = get_cluster_stats(cluster.name)

        # Display cluster overview (using simple layout to avoid column nesting)
        st.markdown("**📊 Cluster Overview:**")
        st.markdown(f"• **Memories:** {cluster.size}")
        st.markdown(f"• **Coherence:** {cluster.coherence_score:.2f}")
        if cluster_insights:
            st.markdown(f"• **Insights:** {len(cluster_insights)}")
        else:
            st.markdown(f"• **Insights:** 0")

        # Show emergent insights if Insight Archive is activated
        if hasattr(st.session_state, 'show_insight_archive') and st.session_state.show_insight_archive and cluster_insights:
            st.markdown("---")
            st.markdown("**✨ Emergent Insights:**")

            # Initialize selection state for Deep Research
            if 'selected_cluster_insights' not in st.session_state:
                st.session_state.selected_cluster_insights = set()

            for i, insight in enumerate(cluster_insights, 1):
                insight_id = f"{cluster.id}_insight_{i}"

                # Research status badge and evidence summary for insight
                status = insight.get('research_status', 'idle')
                if status == 'in_progress':
                    badge = " ⏳"
                elif status == 'completed':
                    badge = " ✅"
                elif status == 'failed':
                    badge = " ❌"
                else:
                    badge = ""

                summary = _insight_evidence_summary(insight)
                label = f"💡 Insight {i}: {insight.get('title', 'Untitled')}{badge}  ({summary})" if summary else f"💡 Insight {i}: {insight.get('title', 'Untitled')}{badge}"

                show_insight_details_key = f"show_{insight_id}_details"
                show_insight_details = st.checkbox(
                    label,
                    key=show_insight_details_key
                )
                if show_insight_details:
                    # Add checkbox for Deep Research selection
                    col1, col2 = st.columns([1, 10])

                    with col1:
                        is_selected = st.checkbox(
                            "🔬",
                            key=f"select_{insight_id}",
                            value=insight_id in st.session_state.selected_cluster_insights,
                            help="Select for Deep Research"
                        )

                        # Update selection state
                        if is_selected:
                            st.session_state.selected_cluster_insights.add(insight_id)
                            # Store insight data for research
                            if 'cluster_insight_data' not in st.session_state:
                                st.session_state.cluster_insight_data = {}
                            st.session_state.cluster_insight_data[insight_id] = {
                                'insight': insight,
                                'cluster_id': cluster.id,
                                'cluster_name': cluster.name
                            }

                            # Auto-run research if enabled
                            if st.session_state.get('auto_run_research_on_select', False):
                                mode = st.session_state.get('auto_research_mode', 'Deep')
                                try:
                                    if mode == 'Deep':
                                        # Use Deep Research engine in background
                                        insight_text_for_research = insight.get('synthesized_text', insight.get('content', ''))
                                        simple_insight = {
                                            'content': insight_text_for_research,
                                            'cluster_id': cluster.id,
                                            'insight_id': insight_id
                                        }
                                        trigger_deep_research_engine([simple_insight])
                                        insight['research_status'] = 'in_progress'
                                        st.info("🧠 Deep Research started for this insight...")
                                    else:
                                        # Quick research via arXiv tool in background
                                        import threading
                                        data = {'insight': insight, 'cluster_name': cluster.name}
                                        insight['research_status'] = 'in_progress'
                                        threading.Thread(target=_research_single_cluster_insight, args=(insight_id, data), daemon=True).start()
                                        st.info("🔎 Quick Research started for this insight...")
                                except Exception as e:
                                    logger.warning(f"Auto-run research failed: {e}")
                        else:
                            st.session_state.selected_cluster_insights.discard(insight_id)
                            if 'cluster_insight_data' in st.session_state and insight_id in st.session_state.cluster_insight_data:
                                del st.session_state.cluster_insight_data[insight_id]

                    with col2:
                        # Display insight content
                        insight_text = insight.get('synthesized_text', insight.get('content', ''))

                        # Clean up the insight text (remove thinking tags)
                        if '<think>' in insight_text and '</think>' in insight_text:
                            parts = insight_text.split('</think>')
                            if len(parts) > 1:
                                insight_text = parts[-1].strip()

                        st.markdown(insight_text)

                        # Show insight metadata
                        col1, col2 = st.columns(2)
                        with col1:
                            confidence = insight.get('confidence_score', 0)
                            st.caption(f"🎯 Confidence: {confidence:.2f}")
                        with col2:
                            generated_at = insight.get('generated_at', '')
                            if generated_at:
                                st.caption(f"🕐 Generated: {generated_at[:19]}")

                        # Evidence & Belief (always visible)
                        _render_evidence_and_belief(insight, insight_id)

                        # Show related paragraphs/source content if available
                        _render_insight_source_paragraphs(insight, insight_id)

                        # Show Deep Research results if available
                        _render_deep_research_results_for_insight(insight_id)

            # Deep Research controls (pass cluster id for unique keys)
            _render_cluster_deep_research_controls(cluster.id)

            # Add button to hide insights
            if st.button("🔽 Hide Insights", key=f"hide_insights_{cluster.id}"):
                st.session_state.show_insight_archive = False
                st.rerun()

        # Extract PDF files from cluster memories
        pdf_files = _extract_pdf_files_from_cluster(cluster)

        if pdf_files:
            st.markdown("---")
            st.markdown("**📚 PDF Documents in this Cluster:**")

            # Group by document and show with links
            for pdf_info in pdf_files[:10]:  # Show top 10 PDFs
                filename = pdf_info['filename']
                chunk_count = pdf_info['chunk_count']
                file_path = pdf_info['file_path']

                # Create clickable link if file exists
                if file_path and Path(file_path).exists():
                    # Create download link
                    with open(file_path, 'rb') as f:
                        file_data = f.read()

                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"📄 **{filename}** ({chunk_count} chunks)")
                    with col2:
                        st.download_button(
                            label="📥 Open PDF",
                            data=file_data,
                            file_name=filename,
                            mime="application/pdf",
                            key=f"download_{cluster.id}_{filename}"
                        )
                else:
                    st.markdown(f"📄 **{filename}** ({chunk_count} chunks)")

            if len(pdf_files) > 10:
                st.caption(f"... and {len(pdf_files) - 10} more documents")

        # Display detailed cluster information if available from registry
        if cluster_stats['exists']:
            st.markdown("---")
            st.markdown("**🔍 Basic Information:**")

            # Cluster ID and basic info
            st.markdown(f"• **Cluster ID:** {cluster.name}")
            st.markdown(f"• **Memory count:** {cluster.size}")
            st.markdown(f"• **Coherence score:** {cluster.coherence_score:.2f}")
            st.markdown(f"• **Visualization color:** {cluster.color}")

            # Show source statistics
            if cluster_stats['sources']:
                st.markdown("**📊 Source Statistics:**")
                st.markdown(f"• **Unique sources:** {cluster_stats['source_count']}")
                st.markdown(f"• **Average importance:** {cluster_stats['avg_importance']:.2f}")

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
                # Clean the insight text (remove <think> tags)
                clean_text = insight.get('synthesized_text', '')
                if '<think>' in clean_text and '</think>' in clean_text:
                    parts = clean_text.split('</think>')
                    if len(parts) > 1:
                        clean_text = parts[-1].strip()
                    else:
                        clean_text = clean_text.replace('<think>', '').replace('</think>', '').strip()

                # Display insight without nested expander
                st.markdown(f"**💡 Insight {j+1}:**")
                st.markdown(f"*{clean_text}*")

                # Show insight metadata
                col1, col2 = st.columns(2)
                with col1:
                    if insight.get('confidence_score'):
                        st.write(f"**Confidence:** {insight['confidence_score']:.2f}")
                with col2:
                    if insight.get('novelty_score'):
                        st.write(f"**Novelty:** {insight['novelty_score']:.2f}")

                if j < len(cluster_insights) - 1:  # Add separator between insights
                    st.markdown("---")

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

                        # Pre-check: Verify we have memories to synthesize
                        memory_stats = memory_store.get_memory_stats()
                        total_memories = memory_stats.get('total_memories', 0)

                        if total_memories == 0:
                            st.warning("⚠️ No memories found in SAM's knowledge base.")
                            st.info("💡 Add some documents or have conversations with SAM first, then try synthesis again.")
                            return

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
                            st.info("💡 To view them, expand a cluster in 'Most Coherent Clusters' and click 📚 Show Insights.")
                            # Prepare UI to show insights immediately
                            st.session_state.show_insight_archive = True
                            st.session_state.auto_expand_first_insight_cluster = True
                        elif result.clusters_found > 0:
                            st.warning(f"⚠️ Found {result.clusters_found} clusters but generated 0 insights. This may indicate:")
                            st.info("• Insight quality threshold is too high\n• LLM responses need improvement\n• Memory clusters lack sufficient content")
                            st.info("💡 Try running synthesis again or check the logs for more details.")
                        else:
                            st.warning("⚠️ No memory clusters found for synthesis. Try adding more conversations or documents to SAM's memory.")

                        # Regenerate cognitive map with synthesis-aligned parameters for consistent cluster IDs
                        try:
                            logger.info("Regenerating cognitive map after synthesis (advanced controls)...")
                            updated_cognitive_map = generate_cognitive_map(
                                memory_store=memory_store,
                                method="UMAP + HDBSCAN",
                                time_range="All Time",
                                n_components=2,
                                min_cluster_size=8,
                                perplexity=30,
                                n_neighbors=15,
                                clustering_eps=0.3,
                                max_clusters=10,
                                quality_threshold=0.5,
                                clustering_min_samples=3,
                                disable_kmeans_fallback=True
                            )
                            st.session_state.cognitive_map = updated_cognitive_map
                            logger.info("Cognitive map updated with synthesis results (advanced controls)")
                        except Exception as map_error:
                            logger.warning(f"Failed to update cognitive map (advanced controls): {map_error}")

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

        # Use a simpler layout to avoid column nesting issues
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

            # Auto-research toggle
            auto_research_enabled = st.session_state.get('auto_research_enabled', False)
            new_auto_research = st.checkbox(
                "🔬 Auto-Research",
                value=auto_research_enabled,
                help="Automatically research promising insights"
            )

            if new_auto_research != auto_research_enabled:
                st.session_state.auto_research_enabled = new_auto_research
        else:
            st.markdown("*Enable auto-synthesis to configure frequency and auto-research*")

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
            st.info("💡 Synthesis insights are displayed in the 'Most Coherent Clusters' section above.")

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
            # Use simple layout to avoid column nesting
            if st.button("⚙️ Settings", help="Configure synthesis settings"):
                st.session_state.show_synthesis_settings = not st.session_state.get('show_synthesis_settings', False)
                st.rerun()

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
                    stable_insight_id = insight.get('insight_id')

                    try:
                        # Initialize Deep Research Strategy
                        research_strategy = DeepResearchStrategy(insight_text)

                        # Execute deep research
                        result = research_strategy.execute_research()

                        status_value = result.status.value if hasattr(result, 'status') else 'COMPLETED'
                        arxiv_papers = result.arxiv_papers if hasattr(result, 'arxiv_papers') else []
                        final_report = result.final_report if hasattr(result, 'final_report') else ''
                        timestamp_val = result.timestamp if hasattr(result, 'timestamp') else datetime.now().isoformat()
                        research_id_val = result.research_id if hasattr(result, 'research_id') else f"research_{int(time.time())}"
                        original_insight_val = result.original_insight if hasattr(result, 'original_insight') else insight_text

                    except Exception as e:
                        logger.error(f"Deep research failed for insight_id={stable_insight_id}: {e}")
                        status_value = 'FAILED'
                        arxiv_papers = []
                        final_report = ''
                        timestamp_val = datetime.now().isoformat()
                        research_id_val = f"research_{int(time.time())}"
                        original_insight_val = insight_text

                    # Store result in session state
                    research_results.append({
                        'research_id': research_id_val,
                        'original_insight': original_insight_val,
                        'cluster_id': cluster_id,
                        'insight_id': stable_insight_id,
                        'final_report': final_report,
                        'arxiv_papers': arxiv_papers,
                        'status': status_value,
                        'timestamp': timestamp_val,
                        'quality_score': research_strategy._assess_research_quality() if status_value != 'FAILED' else 0.0,
                        'papers_analyzed': len(arxiv_papers),
                        'iterations_completed': research_strategy.current_iteration if status_value != 'FAILED' else 0
                    })

                # Store results in session state
                if 'deep_research_results' not in st.session_state:
                    st.session_state.deep_research_results = []

                st.session_state.deep_research_results.extend(research_results)
                st.session_state.latest_deep_research = research_results

                # Mark corresponding insights as completed/failed using stable insight_id and show toasts
                try:
                    completed_ids = {r.get('insight_id') for r in research_results if r.get('insight_id') and r.get('status') == 'COMPLETED'}
                    failed_ids = {r.get('insight_id') for r in research_results if r.get('insight_id') and r.get('status') == 'FAILED'}

                    for key, data in (st.session_state.get('cluster_insight_data') or {}).items():
                        if key in completed_ids:
                            insight_obj = data.get('insight') or {}
                            insight_obj['research_status'] = 'completed'
                            st.success(f"Research finished: {insight_obj.get('title', key)}")
                        elif key in failed_ids:
                            insight_obj = data.get('insight') or {}
                            insight_obj['research_status'] = 'failed'
                            st.error(f"Research failed: {insight_obj.get('title', key)}")
                except Exception as _e:
                    logger.debug(f"Could not set deep research completion status: {_e}")

                # Add papers to vetting queue
                try:
                    from sam.state.vetting_queue import get_vetting_queue_manager
                    vetting_manager = get_vetting_queue_manager()

                    total_papers_added = 0
                    download_limit = int(st.session_state.get('deep_research_download_limit', 3))
                    for result in research_results:
                        # Only process top-K papers per insight
                        papers = result['arxiv_papers'][:download_limit] if result.get('arxiv_papers') else []
                        for paper in papers:
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

                            # Download deep research paper to capture filename/local path
                            local_path = None
                            try:
                                from sam.web_retrieval.tools.arxiv_tool import get_arxiv_tool
                                arxiv_tool = get_arxiv_tool()
                                # download expects arxiv_id and title in a dict like the search results
                                dl = arxiv_tool._download_paper({
                                    'arxiv_id': paper.get('arxiv_id'),
                                    'title': paper.get('title')
                                })
                                if dl.get('success'):
                                    local_path = dl.get('local_path')
                            except Exception as _e:
                                logger.debug(f"Deep Research download failed: {_e}")

                            # Produce Evidence (LLM) and Bayesian update per paper for the matching insight
                            try:
                                stable_id = result.get('insight_id')
                                if stable_id and 'cluster_insight_data' in st.session_state:
                                    rec = st.session_state.cluster_insight_data.get(stable_id)
                                    if rec:
                                        insight_obj = rec.get('insight') or {}
                                        ev = _llm_assess_evidence(
                                            insight_obj.get('synthesized_text', insight_obj.get('content', '')),
                                            {
                                                'title': paper.get('title'),
                                                'summary': paper.get('summary'),
                                                'arxiv_id': paper.get('arxiv_id'),
                                                'published': paper.get('published'),
                                                'categories': paper.get('categories'),
                                                'pdf_url': paper.get('pdf_url'),
                                                'selection_score': paper.get('selection_score')
                                            },
                                            local_path
                                        )
                                        _apply_evidence_update(insight_obj, ev)
                            except Exception as _e:
                                logger.debug(f"Deep Research evidence update failed: {_e}")

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
        import re
        text = (insight_text or '')
        # Extract longer alphabetic tokens as keywords
        terms = re.findall(r'\b[a-zA-Z]{4,}\b', text)
        # Deduplicate preserving order
        seen = set()
        keywords = []
        for t in terms:
            tl = t.lower()
            if tl not in seen:
                seen.add(tl)
                keywords.append(t)
            if len(keywords) >= 8:
                break
        if keywords:
            return ' '.join(keywords)
        return text.strip()[:120]
    except Exception:
        return (insight_text or '')[:120]

# --- Evidence & Belief Helpers ---

def _init_insight_belief_fields(insight: Dict[str, Any]):
    """Ensure belief/evidence fields exist on the insight object."""
    try:
        if 'prior_confidence' not in insight:
            insight['prior_confidence'] = float(insight.get('confidence_score', 0.5) or 0.5)
        if 'belief_score' not in insight:
            insight['belief_score'] = float(insight['prior_confidence'])
        if 'evidence_log' not in insight:
            insight['evidence_log'] = []
    except Exception:
        pass


def _compute_paper_quality(paper_metadata: Dict[str, Any]) -> float:
    """Heuristic quality score in [0,1]. Prefer selection_score if available."""
    try:
        sel = paper_metadata.get('selection_score')
        if sel is not None:
            # Assume selection_score roughly in 0..10 range; normalize conservatively
            return max(0.0, min(1.0, float(sel) / 10.0))
        # Fallback
        return 0.7
    except Exception:
        return 0.7


def _assess_paper_against_insight(insight_text: str, paper_metadata: Dict[str, Any], local_path: Optional[str] = None) -> Dict[str, Any]:
    """Quick heuristic stance/strength assessment and evidence packaging."""
    import re
    from pathlib import Path
    text = (insight_text or '').lower()
    title = (paper_metadata.get('title') or '').lower()
    summary = (paper_metadata.get('summary') or '').lower()

    # Simple keyword extraction
    words = re.findall(r'\b[a-z]{4,}\b', text)
    top_terms = list(dict.fromkeys(words))[:10]

    overlap = sum(1 for t in top_terms if t in title or t in summary)
    contradict_terms = ['not', 'fails', 'fail', 'contradict', 'against', 'however']
    has_contradiction = any(ct in summary for ct in contradict_terms)

    # Stance + strength
    if has_contradiction and overlap >= 1:
        stance = 'contradict'
        strength = min(1.0, 0.4 + 0.1 * overlap)
    elif overlap >= 2:
        stance = 'support'
        strength = min(1.0, 0.3 + 0.1 * overlap)
    else:
        stance = 'neutral'
        strength = 0.2

    quality = _compute_paper_quality(paper_metadata)

    # Build evidence record with links
    arxiv_id = paper_metadata.get('arxiv_id')
    pdf_url = paper_metadata.get('pdf_url')
    file_name = None
    if local_path:
        try:
            file_name = Path(local_path).name
        except Exception:
            file_name = None

    return {
        'source': {
            'paper_title': paper_metadata.get('title'),
            'arxiv_id': arxiv_id,
            'pdf_url': pdf_url,
            'local_path': local_path,
            'filename': file_name,
            'published': paper_metadata.get('published'),
            'categories': paper_metadata.get('categories')
        },
        'stance': stance,
        'strength': float(strength),
        'quality': float(quality),
        'excerpt': '',
        'rationale': 'Heuristic assessment based on keyword overlap.'
    }


def _bayesian_update_belief(insight: Dict[str, Any], evidence: Dict[str, Any], k: float = 1.5):
    """Update belief_score using Bayesian odds and a likelihood ratio from evidence."""
    import math
    _init_insight_belief_fields(insight)
    p = float(insight.get('belief_score', insight.get('prior_confidence', 0.5) or 0.5))
    p = max(1e-6, min(1 - 1e-6, p))  # clamp

    stance = evidence.get('stance', 'neutral')
    strength = float(evidence.get('strength', 0.2) or 0.2)
    quality = float(evidence.get('quality', 0.7) or 0.7)

    sign = 0
    if stance == 'support':
        sign = +1
    elif stance == 'contradict':
        sign = -1

    LR = math.exp(sign * k * strength * quality)
    odds = p / (1 - p)
    new_odds = odds * LR
    posterior = new_odds / (1 + new_odds)


def _insight_evidence_summary(insight: Dict[str, Any]) -> str:
    """Return compact summary: Δbelief and counts S/C."""
    try:
        prior = float(insight.get('prior_confidence', insight.get('confidence_score', 0.5) or 0.5))
        belief = float(insight.get('belief_score', prior))
        delta = belief - prior
        ev = insight.get('evidence_log', []) or []
        s = sum(1 for e in ev if e.get('stance') == 'support')
        c = sum(1 for e in ev if e.get('stance') == 'contradict')
        arrow = '↑' if delta > 1e-6 else ('↓' if delta < -1e-6 else '→')
        return f"{arrow}{delta:+.2f} • S:{s} C:{c}"
    except Exception:
        return ""


def _llm_assess_evidence(insight_text: str, paper_metadata: Dict[str, Any], local_path: Optional[str] = None) -> Dict[str, Any]:
    """Use LLM to assess stance, strength, excerpt, rationale from abstract/title/context and top sections."""
    try:
        from services.response_generator_service import get_response_generator_service, GenerationConfig
        rg = get_response_generator_service()
        # Try to load top sections if the paper has been ingested (by filename)
        top_sections = []
        try:
            if local_path:
                from memory.memory_vectorstore import get_memory_store
                m = get_memory_store()
                fname = local_path.split('/')[-1]
                where = {"filename": {"$eq": fname}}
                # Use enhanced retrieval if available; fallback to search_memories
                try:
                    results = m.enhanced_search_memories(insight_text, max_results=3, where_filter=where)
                    top_sections = [r.chunk.content for r in results] if results else []
                except Exception:
                    basic = m.search_memories(insight_text, max_results=3, where_filter=where)
                    top_sections = [r.chunk.content for r in basic] if basic else []
        except Exception:
            pass

        title = paper_metadata.get('title', '')
        abstract = paper_metadata.get('summary', '')
        arxiv_id = paper_metadata.get('arxiv_id', '')
        sections_text = "\n\n".join(top_sections[:3]) if top_sections else ""

        prompt = (
            "You are an evidence assessor. Given an insight and a paper (title + abstract + top sections), "
            "decide if the paper supports, contradicts, or is neutral to the insight. "
            "Return strict JSON with keys: stance (support|contradict|neutral), strength (0..1), "
            "excerpt (<=280 chars from the abstract or section), rationale (<=200 chars).\n\n"
            f"Insight:\n{insight_text}\n\n"
            f"Paper Title: {title}\nArXiv: {arxiv_id}\nAbstract:\n{abstract}\n\n"
            f"Top Sections (truncated):\n{sections_text[:1200]}\n\n"
            "JSON:"
        )

        cfg = GenerationConfig(temperature=0.2, max_tokens=400)
        raw = rg.generate_response(prompt, cfg)

        import json
        parsed = json.loads(raw.strip()) if raw else {}
        stance = parsed.get('stance', 'neutral')
        strength = float(parsed.get('strength', 0.3) or 0.3)
        excerpt = parsed.get('excerpt', '')
        rationale = parsed.get('rationale', '')

    except Exception:
        # Fallback to heuristic
        assessed = _assess_paper_against_insight(insight_text, paper_metadata, local_path)
        # Ensure excerpt/rationale keys exist
        assessed.setdefault('excerpt', '')
        assessed.setdefault('rationale', 'Heuristic fallback')
        return assessed

    # Build evidence dict with links
    ev = {
        'source': {
            'paper_title': paper_metadata.get('title'),
            'arxiv_id': paper_metadata.get('arxiv_id'),
            'pdf_url': paper_metadata.get('pdf_url'),
            'local_path': local_path,
            'filename': (local_path.split('/')[-1] if local_path else None),
            'published': paper_metadata.get('published'),
            'categories': paper_metadata.get('categories')
        },
        'stance': stance,
        'strength': float(max(0.0, min(1.0, strength))),
        'quality': float(_compute_paper_quality(paper_metadata)),
        'excerpt': excerpt,
        'rationale': rationale
    }
    return ev


def _apply_evidence_update(insight: Dict[str, Any], evidence: Dict[str, Any]):
    """Apply Bayesian update and show a 'Belief updated' toast with delta."""
    try:
        prev = float(insight.get('belief_score', insight.get('prior_confidence', 0.5) or 0.5))
        _bayesian_update_belief(insight, evidence)
        new = float(insight.get('belief_score', prev))
        delta = new - prev
        arrow = '↑' if delta > 1e-6 else ('↓' if delta < -1e-6 else '→')
        st.info(f"Belief updated {arrow}{delta:+.2f}")
    except Exception as _e:
        logger.debug(f"Belief update toast failed: {_e}")

    insight['belief_score'] = float(max(0.0, min(1.0, posterior)))
    # Append evidence
    try:
        insight.setdefault('evidence_log', []).append(evidence)
    except Exception:
        pass


def _render_evidence_and_belief(insight: Dict[str, Any], insight_id: str):
    """Always-visible Evidence & Belief section under each insight."""
    _init_insight_belief_fields(insight)

    st.markdown("---")
    st.markdown("**📐 Evidence & Belief**")

    prior = insight.get('prior_confidence', 0.5)
    belief = insight.get('belief_score', prior)
    st.caption(f"Belief: {belief:.2f} (prior {prior:.2f})")

    evidence_log = insight.get('evidence_log', [])
    if not evidence_log:
        st.caption("No evidence yet. New research will populate here.")
        return

    # Small, focused list of recent evidence
    for ev in evidence_log[-3:][::-1]:
        src = ev.get('source', {})
        title = src.get('paper_title') or 'Unknown Title'
        arxiv_id = src.get('arxiv_id') or 'N/A'
        pdf_url = src.get('pdf_url')
        filename = src.get('filename')

        header = f"{title} (arXiv:{arxiv_id})"
        if pdf_url:
            st.markdown(f"- [{header}]({pdf_url})" + (f" — file: `{filename}`" if filename else ""))
        else:
            st.markdown(f"- {header}" + (f" — file: `{filename}`" if filename else ""))

        stance = ev.get('stance', 'neutral')
        strength = ev.get('strength', 0.0)
        st.caption(f"Stance: {stance} • Strength: {strength:.2f}")




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
        # Synthesis results available - insights can be viewed in clusters via Insight Archive button
        pass

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
        # Use container instead of expander to avoid nesting issues
        with st.container():
            # Add insight header
            st.markdown(f"### 💡 Insight {i+1}: Cluster {insight.get('cluster_id', 'Unknown')}")

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

            # Research Mode Controls - Add the missing checkbox and microscope functionality
            st.markdown("---")
            col1, col2 = st.columns([1, 10])

            with col1:
                insight_id = f"insight_{i}_{insight.get('cluster_id', 'unknown')}"
                research_selected = st.checkbox(
                    "🔬",
                    key=f"research_select_{insight_id}",
                    help="Select for Research Mode"
                )

            with col2:
                if research_selected:
                    st.caption("✅ Selected for research")
                    # Store selection in session state for research processing
                    if 'selected_synthesis_insights' not in st.session_state:
                        st.session_state.selected_synthesis_insights = set()
                    st.session_state.selected_synthesis_insights.add(insight_id)

                    # Store insight data for research
                    if 'synthesis_insight_data' not in st.session_state:
                        st.session_state.synthesis_insight_data = {}
                    st.session_state.synthesis_insight_data[insight_id] = insight
                else:
                    if 'selected_synthesis_insights' in st.session_state:
                        st.session_state.selected_synthesis_insights.discard(insight_id)
                    if 'synthesis_insight_data' in st.session_state and insight_id in st.session_state.synthesis_insight_data:
                        del st.session_state.synthesis_insight_data[insight_id]

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

    # Research Mode Controls for selected synthesis insights
    render_synthesis_research_mode_controls()

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

def _render_insight_source_paragraphs(insight: Dict[str, Any], insight_id: str):
    """Render the source paragraphs/chunks that the insight was derived from."""
    try:
        # Check if user wants to see source paragraphs
        show_sources_key = f"show_sources_{insight_id}"

        if st.button("📄 Show Related Paragraphs", key=f"btn_{show_sources_key}", help="View the source content this insight was derived from"):
            st.session_state[show_sources_key] = not st.session_state.get(show_sources_key, False)

        if st.session_state.get(show_sources_key, False):
            st.markdown("**📚 Source Content:**")

            # Try to get source chunks from insight metadata
            source_chunks = insight.get('source_chunks', [])
            source_documents = insight.get('source_documents', [])
            cluster_id = insight.get('cluster_id')

            if source_chunks:
                # Display actual source chunks if available
                for i, chunk in enumerate(source_chunks[:5], 1):
                    show_chunk_key = f"show_{insight_id}_source_{i}"
                    if st.checkbox(f"📄 Source {i}: {chunk.get('source', 'Unknown')}", key=show_chunk_key):
                        content = chunk.get('content', '')
                        if content:
                            st.markdown(f"**Content:**\n\n{content}")

                            # Show metadata if available
                            if chunk.get('memory_type'):
                                st.caption(f"Type: {chunk.get('memory_type')}")
                            if chunk.get('confidence'):
                                st.caption(f"Confidence: {chunk.get('confidence'):.2f}")
                        else:
                            st.info("Content not available for this source")

            elif cluster_id:
                # Try to get cluster memories if source chunks not available
                try:
                    cluster_memories = _get_cluster_memories_for_insight(cluster_id)

                    if cluster_memories:
                        st.markdown("**📋 Related Cluster Content:**")
                        for i, memory in enumerate(cluster_memories[:3], 1):
                            show_mem_key = f"show_{insight_id}_memory_{i}"
                            if st.checkbox(f"📄 Memory {i}: {memory.get('source', 'Unknown')}", key=show_mem_key):
                                content = memory.get('content', '')
                                if content:
                                    if len(content) > 1000:
                                        content = content[:1000] + "..."
                                    st.markdown(f"**Content:**\n\n{content}")

                                    if memory.get('memory_type'):
                                        st.caption(f"Type: {memory.get('memory_type')}")
                                    if memory.get('timestamp'):
                                        st.caption(f"Timestamp: {memory.get('timestamp')}")
                                else:
                                    st.info("Content not available for this memory")
                    else:
                        st.info("No cluster memories found for this insight")

                except Exception as e:
                    logger.warning(f"Failed to retrieve cluster memories: {e}")
                    st.info("Unable to retrieve cluster content at this time")

            elif source_documents:
                # Fallback: show source document names
                st.markdown("**📚 Source Documents:**")
                for doc in source_documents:
                    st.caption(f"• {doc}")
                st.info("💡 Tip: Source content extraction is being enhanced. Document names shown above.")

            else:
                st.info("No source content available for this insight")

    except Exception as e:
        logger.error(f"Error rendering insight source paragraphs: {e}")
        st.error("Unable to display source content at this time")

def _get_cluster_memories_for_insight(cluster_id: str) -> List[Dict[str, Any]]:
    """Get memories for a cluster to display as source content."""
    try:
        # Try multiple approaches to get cluster memories

        # Approach 1: Try cluster registry
        try:
            from memory.synthesis.cluster_registry import get_cluster_registry
            registry = get_cluster_registry()
            cluster_metadata = registry.get_cluster_metadata(cluster_id)

            if cluster_metadata and hasattr(cluster_metadata, 'memories'):
                return cluster_metadata.memories
        except Exception as e:
            logger.debug(f"Cluster registry approach failed: {e}")

        # Approach 2: Try to find cluster in synthesis results
        if hasattr(st.session_state, 'synthesis_results'):
            insights = st.session_state.synthesis_results.get('insights', [])
            for insight in insights:
                if insight.get('cluster_id') == cluster_id:
                    source_chunks = insight.get('source_chunks', [])
                    if source_chunks:
                        # Convert source chunks to memory format
                        memories = []
                        for chunk in source_chunks:
                            memory = {
                                'content': chunk.get('content', ''),
                                'source': chunk.get('source', 'Unknown'),
                                'memory_type': chunk.get('memory_type', 'document'),
                                'timestamp': chunk.get('timestamp', ''),
                                'confidence': chunk.get('importance_score', 0.0)
                            }
                            memories.append(memory)
                        return memories

        # Approach 3: Try to get memories from memory store by cluster
        try:
            from memory.memory_vectorstore import get_memory_store
            memory_store = get_memory_store()

            # Get all memories and filter by cluster (if cluster info is stored)
            all_memories = memory_store.get_all_memories()
            cluster_memories = []

            for memory in all_memories:
                # Check if memory belongs to this cluster (various ways to check)
                memory_cluster_id = None

                # Check metadata for cluster ID
                if hasattr(memory, 'metadata') and memory.metadata:
                    memory_cluster_id = memory.metadata.get('cluster_id')

                # Check if memory content matches cluster theme
                if memory_cluster_id == cluster_id:
                    memory_dict = {
                        'content': getattr(memory, 'content', str(memory)),
                        'source': getattr(memory, 'source', 'Unknown'),
                        'memory_type': getattr(memory, 'memory_type', 'unknown'),
                        'timestamp': getattr(memory, 'timestamp', ''),
                        'confidence': getattr(memory, 'importance_score', 0.0)
                    }
                    cluster_memories.append(memory_dict)

            if cluster_memories:
                return cluster_memories[:5]  # Limit to 5 memories

        except Exception as e:
            logger.debug(f"Memory store approach failed: {e}")

        # Approach 4: Fallback - create sample content
        logger.info(f"No specific memories found for cluster {cluster_id}, using fallback")
        return [
            {
                'content': f"This insight was generated from cluster {cluster_id}. The specific source content is being retrieved...",
                'source': f"Cluster {cluster_id}",
                'memory_type': 'cluster_summary',
                'timestamp': datetime.now().isoformat(),
                'confidence': 0.5
            }
        ]

    except Exception as e:
        logger.error(f"Error getting cluster memories: {e}")
        return []

def render_synthesis_research_mode_controls():
    """Render research mode controls for selected synthesis insights."""
    if 'selected_synthesis_insights' not in st.session_state or not st.session_state.selected_synthesis_insights:
        return

    selected_count = len(st.session_state.selected_synthesis_insights)

    st.markdown("---")
    st.markdown("### 🔬 Research Mode")
    st.markdown(f"*{selected_count} insight(s) selected for research*")

    col1, col2, col3 = st.columns(3)

    with col1:
        research_mode = st.radio(
            "Research Mode:",
            ["Deep Research", "Quick Research"],
            help="Choose research depth"
        )

    with col2:
        max_papers = st.selectbox(
            "Papers per Insight:",
            [1, 2, 3, 5],
            index=1,
            help="Maximum papers to download per insight"
        )

    with col3:
        if st.button("🚀 Start Research", type="primary"):
            trigger_synthesis_insight_research(
                list(st.session_state.selected_synthesis_insights),
                research_mode,
                max_papers
            )

def trigger_synthesis_insight_research(selected_insight_ids, research_mode, max_papers):
    """Trigger research for selected synthesis insights."""
    try:
        # Get the selected insights data
        selected_insights = []
        insight_data = st.session_state.get('synthesis_insight_data', {})

        for insight_id in selected_insight_ids:
            if insight_id in insight_data:
                selected_insights.append(insight_data[insight_id])

        if not selected_insights:
            st.error("❌ No insight data found for selected insights")
            return

        # Trigger research based on mode
        if research_mode == "Deep Research":
            # Use Deep Research engine
            try:
                from sam.agents.strategies.deep_research import DeepResearchStrategy

                for insight in selected_insights:
                    insight_text = insight.get('synthesized_text', '')
                    if insight_text:
                        # Clean the insight text
                        if '<think>' in insight_text and '</think>' in insight_text:
                            parts = insight_text.split('</think>')
                            if len(parts) > 1:
                                insight_text = parts[-1].strip()

                        # Start deep research
                        research_strategy = DeepResearchStrategy(insight_text)
                        st.info(f"🔬 Starting Deep Research for insight from cluster {insight.get('cluster_id', 'Unknown')}")

                        # Note: In a real implementation, this would be run asynchronously
                        # For now, we'll just show that research has been initiated

            except ImportError:
                st.error("❌ Deep Research engine not available")
                return

        else:  # Quick Research
            # Use Quick Research (basic ArXiv search)
            for insight in selected_insights:
                insight_text = insight.get('synthesized_text', '')
                if insight_text:
                    st.info(f"🔍 Starting Quick Research for insight from cluster {insight.get('cluster_id', 'Unknown')}")

        st.success(f"🔬 Research initiated for {len(selected_insights)} insights using {research_mode}")
        st.info("Research results will appear in the Deep Research Results section")

        # Clear selections
        st.session_state.selected_synthesis_insights = set()
        st.session_state.synthesis_insight_data = {}
        st.rerun()

    except Exception as e:
        st.error(f"❌ Failed to start research: {e}")
        logger.error(f"Research trigger error: {e}")
