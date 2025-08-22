#!/usr/bin/env python3
"""
Dream Canvas Application Controller
==================================

Main controller for the Dream Canvas cognitive visualization interface.
Replaces the monolithic dream_canvas.py with a modular architecture.

This module provides:
- Dream Canvas application orchestration
- Component integration and coordination
- State management
- User interaction handling

Author: SAM Development Team
Version: 1.0.0 - Refactored from dream_canvas.py
"""

import streamlit as st
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from sam.dream_canvas.utils.models import (
    DreamCanvasState, VisualizationConfig, VisualizationMethod,
    ClusteringMethod, TimeRange
)
from sam.dream_canvas.handlers.cognitive_mapping import get_cognitive_mapping_engine
from sam.dream_canvas.visualization.canvas_renderer import get_canvas_renderer
from sam.dream_canvas.research.deep_research import get_deep_research_engine

logger = logging.getLogger(__name__)


class DreamCanvasController:
    """Main controller for the Dream Canvas application."""
    
    def __init__(self):
        self.app_name = "Dream Canvas - Cognitive Synthesis Visualization"
        self.version = "2.0.0"
        
        # Initialize components
        self.mapping_engine = get_cognitive_mapping_engine()
        self.canvas_renderer = get_canvas_renderer()
        self.research_engine = get_deep_research_engine()
        
        # Initialize state
        self._initialize_state()
    
    def _initialize_state(self):
        """Initialize Dream Canvas state."""
        if 'dream_canvas_state' not in st.session_state:
            st.session_state.dream_canvas_state = DreamCanvasState()
        
        if 'dream_canvas_initialized' not in st.session_state:
            st.session_state.dream_canvas_initialized = True
            logger.info("Dream Canvas state initialized")
    
    def render_dream_canvas(self):
        """Main function to render the Dream Canvas interface."""
        try:
            # Header
            self._render_header()
            
            # Check memory store availability
            if not self._check_memory_store_availability():
                self._render_memory_store_unavailable()
                return
            
            # Main interface
            self._render_main_interface()
            
        except Exception as e:
            logger.error(f"Error rendering Dream Canvas: {e}")
            st.error(f"❌ Dream Canvas error: {str(e)}")
    
    def _render_header(self):
        """Render the Dream Canvas header."""
        st.subheader("🧠🎨 Dream Canvas - Cognitive Synthesis Visualization")
        st.markdown("*Interactive memory landscape with UMAP projections and cluster analysis*")
        
        # Status indicator
        state = st.session_state.dream_canvas_state
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if state.cognitive_map:
                st.success(f"✅ Map loaded: {state.cognitive_map.get_cluster_count()} clusters")
            else:
                st.info("🎯 Ready to generate cognitive map")
        
        with col2:
            if state.is_loading:
                st.warning("⏳ Processing...")
            else:
                st.success("🚀 Ready")
        
        with col3:
            st.caption(f"v{self.version}")
    
    def _check_memory_store_availability(self) -> bool:
        """Check if memory store is available."""
        try:
            from memory.memory_vectorstore import get_memory_store
            memory_store = get_memory_store()
            return memory_store is not None
        except ImportError:
            return False
        except Exception as e:
            logger.error(f"Error checking memory store: {e}")
            return False
    
    def _render_memory_store_unavailable(self):
        """Render message when memory store is unavailable."""
        st.warning("⚠️ **Memory Store Unavailable**")
        st.markdown("""
        The Dream Canvas requires access to the SAM memory store to function properly.
        
        **Possible solutions:**
        - Ensure the memory system is properly initialized
        - Check that you have memories stored in the system
        - Verify the memory store configuration
        
        **Demo Mode:**
        You can still explore the interface with sample data.
        """)
        
        if st.button("🎭 Enable Demo Mode"):
            self._enable_demo_mode()
    
    def _enable_demo_mode(self):
        """Enable demo mode with sample data."""
        state = st.session_state.dream_canvas_state
        
        # Generate demo cognitive map
        demo_config = VisualizationConfig(
            method=VisualizationMethod.UMAP,
            clustering_method=ClusteringMethod.KMEANS,
            n_clusters=6
        )
        
        with st.spinner("Generating demo cognitive map..."):
            cognitive_map = self.mapping_engine.generate_cognitive_map(demo_config)
            state.update_cognitive_map(cognitive_map)
        
        st.success("✅ Demo mode enabled!")
        st.rerun()
    
    def _render_main_interface(self):
        """Render the main Dream Canvas interface."""
        state = st.session_state.dream_canvas_state
        
        # Configuration panel
        with st.expander("⚙️ Configuration", expanded=not state.cognitive_map):
            updated_config = self.canvas_renderer.render_configuration_panel(state.config)
            
            # Check if configuration changed
            if updated_config.to_dict() != state.config.to_dict():
                state.config = updated_config
            
            # Generate map button
            if st.button("🎨 Generate Cognitive Map", type="primary"):
                self._generate_cognitive_map()
        
        # Main visualization area
        if state.cognitive_map:
            self._render_visualization_area()
        else:
            self._render_welcome_message()
    
    def _generate_cognitive_map(self):
        """Generate a new cognitive map."""
        state = st.session_state.dream_canvas_state
        
        try:
            state.is_loading = True
            
            with st.spinner("🧠 Generating cognitive map..."):
                # Validate configuration
                if not state.config.validate():
                    st.error("❌ Invalid configuration. Please check your settings.")
                    return
                
                # Generate cognitive map
                cognitive_map = self.mapping_engine.generate_cognitive_map(state.config)
                
                if cognitive_map and cognitive_map.clusters:
                    state.update_cognitive_map(cognitive_map)
                    st.success(f"✅ Generated cognitive map with {len(cognitive_map.clusters)} clusters!")
                else:
                    st.error("❌ Failed to generate cognitive map. No clusters found.")
            
        except Exception as e:
            logger.error(f"Error generating cognitive map: {e}")
            st.error(f"❌ Error generating cognitive map: {str(e)}")
        
        finally:
            state.is_loading = False
            st.rerun()
    
    def _render_visualization_area(self):
        """Render the main visualization area."""
        state = st.session_state.dream_canvas_state
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Cognitive Map", "📊 Statistics", "🔬 Research", "⚙️ Settings"])
        
        with tab1:
            self._render_cognitive_map_tab()
        
        with tab2:
            self._render_statistics_tab()
        
        with tab3:
            self._render_research_tab()
        
        with tab4:
            self._render_settings_tab()
    
    def _render_cognitive_map_tab(self):
        """Render the cognitive map visualization tab."""
        state = st.session_state.dream_canvas_state
        
        # Render the cognitive map
        fig = self.canvas_renderer.render_cognitive_map(
            state.cognitive_map, 
            state.selected_cluster_id
        )
        
        # Display the figure
        plotly_events = st.plotly_chart(
            fig, 
            use_container_width=True,
            key="cognitive_map"
        )
        
        # Handle cluster selection (this would need custom Streamlit component)
        # For now, provide cluster selection via selectbox
        cluster_options = ["None"] + [cluster.name for cluster in state.cognitive_map.clusters]
        selected_cluster_name = st.selectbox(
            "Select cluster for details:",
            options=cluster_options,
            index=0
        )
        
        if selected_cluster_name != "None":
            # Find selected cluster
            selected_cluster = None
            for cluster in state.cognitive_map.clusters:
                if cluster.name == selected_cluster_name:
                    selected_cluster = cluster
                    state.selected_cluster_id = cluster.id
                    break
            
            if selected_cluster:
                # Render cluster details
                self.canvas_renderer.render_cluster_details(selected_cluster)
    
    def _render_statistics_tab(self):
        """Render the statistics tab."""
        state = st.session_state.dream_canvas_state
        
        if state.cognitive_map:
            self.canvas_renderer.render_map_statistics(state.cognitive_map)
        else:
            st.info("Generate a cognitive map to view statistics.")
    
    def _render_research_tab(self):
        """Render the research tab."""
        state = st.session_state.dream_canvas_state
        
        if not state.cognitive_map:
            st.info("Generate a cognitive map to access research features.")
            return
        
        st.subheader("🔬 Deep Research")
        
        # Cluster selection for research
        cluster_options = [cluster.name for cluster in state.cognitive_map.clusters]
        selected_cluster_name = st.selectbox(
            "Select cluster for research:",
            options=cluster_options,
            key="research_cluster_select"
        )
        
        # Find selected cluster
        selected_cluster = None
        for cluster in state.cognitive_map.clusters:
            if cluster.name == selected_cluster_name:
                selected_cluster = cluster
                break
        
        if selected_cluster:
            # Render research controls
            self.research_engine.render_cluster_research_controls(selected_cluster)
    
    def _render_settings_tab(self):
        """Render the settings tab."""
        st.subheader("⚙️ Dream Canvas Settings")
        
        # Auto-research settings
        st.markdown("### 🔬 Research Settings")
        
        auto_research = st.checkbox(
            "Enable automatic research",
            value=True,
            help="Automatically generate research insights for new clusters"
        )
        
        auto_ingestion = st.checkbox(
            "Enable automatic paper ingestion",
            value=True,
            help="Automatically ingest relevant research papers"
        )
        
        # Visualization settings
        st.markdown("### 🎨 Visualization Settings")
        
        show_connections = st.checkbox(
            "Show cluster connections",
            value=True,
            help="Display connections between related clusters"
        )
        
        animation_speed = st.slider(
            "Animation speed",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Speed of visualization animations"
        )
        
        # Performance settings
        st.markdown("### ⚡ Performance Settings")
        
        max_clusters = st.slider(
            "Maximum clusters",
            min_value=5,
            max_value=50,
            value=20,
            help="Maximum number of clusters to display"
        )
        
        cache_results = st.checkbox(
            "Cache results",
            value=True,
            help="Cache cognitive maps for faster loading"
        )
    
    def _render_welcome_message(self):
        """Render welcome message when no map is loaded."""
        st.markdown("""
        ### 🎨 Welcome to Dream Canvas
        
        Dream Canvas provides an interactive visualization of your memory landscape using advanced
        dimensionality reduction and clustering techniques.
        
        **Features:**
        - 🧠 **Cognitive Mapping**: UMAP/t-SNE projections of your memory space
        - 🎯 **Smart Clustering**: Automatic grouping of related memories
        - 🔬 **Deep Research**: AI-powered research insights from memory clusters
        - 📊 **Analytics**: Comprehensive statistics and visualizations
        
        **Get Started:**
        1. Configure your visualization parameters above
        2. Click "Generate Cognitive Map" to create your first map
        3. Explore clusters and generate research insights
        
        **Tips:**
        - Start with default settings for your first map
        - Experiment with different clustering methods
        - Use the research features to discover new insights
        """)


def render_dream_canvas():
    """Main function to render the Dream Canvas interface."""
    controller = DreamCanvasController()
    controller.render_dream_canvas()


# For backward compatibility
def main():
    """Main entry point for the Dream Canvas application."""
    render_dream_canvas()


if __name__ == "__main__":
    main()
