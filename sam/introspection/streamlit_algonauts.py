"""
Streamlit Algonauts Integration
==============================

Streamlit components for displaying Algonauts-style cognitive visualizations
directly in the main SAM application interface.

Author: SAM Development Team
Version: 1.0.0
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import json

from .flight_recorder import get_flight_recorder, ReasoningTrace
from .algonauts_visualization import AlgonautsVisualizer


class StreamlitAlgonautsInterface:
    """Streamlit interface for Algonauts cognitive visualizations."""
    
    def __init__(self):
        self.flight_recorder = get_flight_recorder()
        self.visualizer = AlgonautsVisualizer()
    
    def render_algonauts_tab(self):
        """Render the complete Algonauts visualization tab."""
        st.header("🧠 Algonauts: SAM's Cognitive Visualization")
        st.markdown("Explore SAM's reasoning process through interactive cognitive trajectory analysis")

        # Analysis mode selector
        analysis_mode = st.radio(
            "Analysis Mode:",
            ["Single Session", "Comparative Analysis"],
            help="Choose between analyzing a single session or comparing multiple sessions"
        )

        if analysis_mode == "Single Session":
            self._render_single_session_analysis()
        else:
            self._render_comparative_analysis()

    def _render_single_session_analysis(self):
        """Render single session analysis interface."""
        # Get available sessions
        sessions = self.flight_recorder.get_all_sessions()

        if not sessions:
            st.info("🔄 No reasoning sessions found. Start a conversation with SAM to generate cognitive traces.")
            return

        # Session selector
        col1, col2 = st.columns([3, 1])

        with col1:
            selected_session = st.selectbox(
                "Select a reasoning session to analyze:",
                sessions,
                format_func=lambda x: f"Session {x[:8]}..." if x else "No sessions"
            )

        with col2:
            if st.button("🔄 Refresh Sessions"):
                st.rerun()

        if selected_session:
            self._render_session_analysis(selected_session)

    def _render_comparative_analysis(self):
        """Render comparative analysis interface for Algonauts experiment."""
        st.subheader("🔬 Comparative Cognitive Analysis")
        st.markdown("Compare cognitive trajectories between different models or sessions")

        # Get available sessions
        sessions = self.flight_recorder.get_all_sessions()

        if len(sessions) < 2:
            st.warning("⚠️ Need at least 2 sessions for comparative analysis")
            return

        # Session selectors
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Model A (e.g., Mistral)**")
            session_a = st.selectbox(
                "Select first session:",
                sessions,
                format_func=lambda x: f"Session {x[:8]}..." if x else "No sessions",
                key="session_a"
            )

        with col2:
            st.markdown("**Model B (e.g., Jamba)**")
            session_b = st.selectbox(
                "Select second session:",
                sessions,
                format_func=lambda x: f"Session {x[:8]}..." if x else "No sessions",
                key="session_b"
            )

        if session_a and session_b and session_a != session_b:
            self._render_side_by_side_comparison(session_a, session_b)
    
    def _render_session_analysis(self, session_id: str):
        """Render analysis for a specific session."""
        trace = self.flight_recorder.get_session_trace(session_id)
        
        if not trace:
            st.error("❌ Session not found")
            return
        
        if not trace.cognitive_trajectory:
            st.warning("🧠 No cognitive vectors found in this session. Cognitive visualizations require neural activation data.")
            return
        
        # Session overview
        self._render_session_overview(trace)
        
        # Cognitive trajectory visualization
        self._render_cognitive_trajectory(trace)
        
        # Reasoning timeline
        self._render_reasoning_timeline(trace)
        
        # Pattern analysis
        self._render_pattern_analysis(trace)

    def _render_side_by_side_comparison(self, session_a: str, session_b: str):
        """Render side-by-side trajectory comparison for Algonauts experiment."""
        st.subheader("🎯 Side-by-Side Trajectory Comparison")

        # Load both traces
        trace_a = self.flight_recorder.get_session_trace(session_a)
        trace_b = self.flight_recorder.get_session_trace(session_b)

        if not trace_a or not trace_b:
            st.error("❌ Failed to load one or both sessions")
            return

        if not trace_a.cognitive_trajectory or not trace_b.cognitive_trajectory:
            st.warning("🧠 One or both sessions lack cognitive trajectory data")
            return

        # Projection method selector
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            projection_method = st.selectbox(
                "Dimensionality Reduction Method:",
                ["pca", "tsne", "umap"] if hasattr(self.visualizer, 'available_methods') else ["pca"],
                help="Choose how to project high-dimensional cognitive vectors to 2D",
                key="comparison_projection"
            )

        with col2:
            show_paths = st.checkbox("Show Trajectory Paths", value=True, key="comparison_paths")

        with col3:
            overlay_mode = st.checkbox("Overlay Trajectories", value=True, key="overlay_mode")

        try:
            # Project both trajectories
            trajectory_a = self.visualizer.project_cognitive_vectors(
                trace_a.cognitive_trajectory,
                method=projection_method
            )

            trajectory_b = self.visualizer.project_cognitive_vectors(
                trace_b.cognitive_trajectory,
                method=projection_method
            )

            if overlay_mode:
                # Create overlaid visualization
                self._render_overlaid_trajectories(trajectory_a, trajectory_b, session_a, session_b, show_paths)
            else:
                # Create side-by-side visualizations
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**Session A: {session_a[:8]}...**")
                    self._render_single_trajectory(trajectory_a, show_paths, color="blue")

                with col2:
                    st.markdown(f"**Session B: {session_b[:8]}...**")
                    self._render_single_trajectory(trajectory_b, show_paths, color="red")

            # Quantitative comparison
            self._render_trajectory_metrics_comparison(trajectory_a, trajectory_b, session_a, session_b)

        except Exception as e:
            st.error(f"❌ Failed to generate trajectory comparison: {e}")
            st.exception(e)

    def _render_overlaid_trajectories(self, trajectory_a, trajectory_b, session_a: str, session_b: str, show_paths: bool):
        """Render overlaid trajectory visualization."""
        import plotly.graph_objects as go

        fig = go.Figure()

        # Extract points for both trajectories
        points_a = trajectory_a.projected_points
        points_b = trajectory_b.projected_points

        # Add trajectory A (blue)
        x_a = [p.x for p in points_a]
        y_a = [p.y for p in points_a]

        fig.add_trace(go.Scatter(
            x=x_a, y=y_a,
            mode='markers+lines' if show_paths else 'markers',
            name=f'Model A ({session_a[:8]}...)',
            marker=dict(color='blue', size=8, opacity=0.7),
            line=dict(color='blue', width=2, dash='solid'),
            hovertemplate='<b>Model A</b><br>Step: %{pointNumber}<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>'
        ))

        # Add trajectory B (red)
        x_b = [p.x for p in points_b]
        y_b = [p.y for p in points_b]

        fig.add_trace(go.Scatter(
            x=x_b, y=y_b,
            mode='markers+lines' if show_paths else 'markers',
            name=f'Model B ({session_b[:8]}...)',
            marker=dict(color='red', size=8, opacity=0.7),
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate='<b>Model B</b><br>Step: %{pointNumber}<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>'
        ))

        # Mark start and end points
        if points_a:
            fig.add_trace(go.Scatter(
                x=[x_a[0]], y=[y_a[0]],
                mode='markers',
                name='Start A',
                marker=dict(color='blue', size=12, symbol='star'),
                showlegend=False
            ))

        if points_b:
            fig.add_trace(go.Scatter(
                x=[x_b[0]], y=[y_b[0]],
                mode='markers',
                name='Start B',
                marker=dict(color='red', size=12, symbol='star'),
                showlegend=False
            ))

        fig.update_layout(
            title="🧠 Overlaid Cognitive Trajectories",
            xaxis_title="Cognitive Dimension 1",
            yaxis_title="Cognitive Dimension 2",
            height=600,
            hovermode='closest',
            legend=dict(x=0.02, y=0.98)
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_single_trajectory(self, trajectory, show_paths: bool, color: str = "blue"):
        """Render a single trajectory visualization."""
        import plotly.graph_objects as go

        fig = go.Figure()

        points = trajectory.projected_points
        x = [p.x for p in points]
        y = [p.y for p in points]

        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers+lines' if show_paths else 'markers',
            marker=dict(color=color, size=8, opacity=0.7),
            line=dict(color=color, width=2),
            hovertemplate='Step: %{pointNumber}<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>'
        ))

        # Mark start point
        if points:
            fig.add_trace(go.Scatter(
                x=[x[0]], y=[y[0]],
                mode='markers',
                marker=dict(color=color, size=12, symbol='star'),
                showlegend=False
            ))

        fig.update_layout(
            xaxis_title="Cognitive Dimension 1",
            yaxis_title="Cognitive Dimension 2",
            height=400,
            hovermode='closest'
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_trajectory_metrics_comparison(self, trajectory_a, trajectory_b, session_a: str, session_b: str):
        """Render quantitative trajectory metrics comparison."""
        st.subheader("📊 Quantitative Trajectory Analysis")

        # Calculate metrics for both trajectories
        metrics_a = self._calculate_trajectory_metrics(trajectory_a)
        metrics_b = self._calculate_trajectory_metrics(trajectory_b)

        # Display comparison table
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Trajectory Length",
                f"{metrics_a['trajectory_length']:.3f}",
                delta=f"{metrics_a['trajectory_length'] - metrics_b['trajectory_length']:+.3f}",
                help="Total Euclidean distance traveled in 2D space"
            )

        with col2:
            st.metric(
                "State Volatility",
                f"{metrics_a['state_volatility']:.3f}",
                delta=f"{metrics_a['state_volatility'] - metrics_b['state_volatility']:+.3f}",
                help="Average distance between consecutive points"
            )

        with col3:
            st.metric(
                "RAG Influence Score",
                f"{metrics_a['rag_influence']:.3f}",
                delta=f"{metrics_a['rag_influence'] - metrics_b['rag_influence']:+.3f}",
                help="Distance between initial and context processing states"
            )

        # Detailed comparison table
        st.markdown("### Detailed Metrics Comparison")

        comparison_data = {
            "Metric": ["Trajectory Length", "State Volatility", "RAG Influence Score", "Number of Steps"],
            f"Model A ({session_a[:8]}...)": [
                f"{metrics_a['trajectory_length']:.3f}",
                f"{metrics_a['state_volatility']:.3f}",
                f"{metrics_a['rag_influence']:.3f}",
                f"{metrics_a['num_steps']}"
            ],
            f"Model B ({session_b[:8]}...)": [
                f"{metrics_b['trajectory_length']:.3f}",
                f"{metrics_b['state_volatility']:.3f}",
                f"{metrics_b['rag_influence']:.3f}",
                f"{metrics_b['num_steps']}"
            ],
            "Difference (A - B)": [
                f"{metrics_a['trajectory_length'] - metrics_b['trajectory_length']:+.3f}",
                f"{metrics_a['state_volatility'] - metrics_b['state_volatility']:+.3f}",
                f"{metrics_a['rag_influence'] - metrics_b['rag_influence']:+.3f}",
                f"{metrics_a['num_steps'] - metrics_b['num_steps']:+d}"
            ]
        }

        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)

        # Interpretation
        st.markdown("### 🔍 Interpretation")

        interpretations = []

        if metrics_a['trajectory_length'] < metrics_b['trajectory_length']:
            interpretations.append("🎯 **Model A shows more efficient reasoning** (shorter trajectory)")
        else:
            interpretations.append("🎯 **Model B shows more efficient reasoning** (shorter trajectory)")

        if metrics_a['state_volatility'] < metrics_b['state_volatility']:
            interpretations.append("🧘 **Model A demonstrates more stable thinking** (lower volatility)")
        else:
            interpretations.append("🧘 **Model B demonstrates more stable thinking** (lower volatility)")

        if metrics_a['rag_influence'] > metrics_b['rag_influence']:
            interpretations.append("📚 **Model A adapts more strongly to context** (higher RAG influence)")
        else:
            interpretations.append("📚 **Model B adapts more strongly to context** (higher RAG influence)")

        for interpretation in interpretations:
            st.markdown(interpretation)

    def _calculate_trajectory_metrics(self, trajectory):
        """Calculate quantitative metrics for a single trajectory."""
        points = trajectory.projected_points

        if len(points) < 2:
            return {
                'trajectory_length': 0.0,
                'state_volatility': 0.0,
                'rag_influence': 0.0,
                'num_steps': len(points)
            }

        # Calculate trajectory length (total distance)
        total_distance = 0
        step_distances = []

        for i in range(1, len(points)):
            dx = points[i].x - points[i-1].x
            dy = points[i].y - points[i-1].y
            distance = (dx**2 + dy**2)**0.5
            total_distance += distance
            step_distances.append(distance)

        # Calculate state volatility (average step distance)
        avg_volatility = sum(step_distances) / len(step_distances) if step_distances else 0.0

        # Calculate RAG influence (distance from start to second point, assuming context processing)
        rag_influence = 0.0
        if len(points) >= 2:
            dx = points[1].x - points[0].x
            dy = points[1].y - points[0].y
            rag_influence = (dx**2 + dy**2)**0.5

        return {
            'trajectory_length': total_distance,
            'state_volatility': avg_volatility,
            'rag_influence': rag_influence,
            'num_steps': len(points)
        }
    
    def _render_session_overview(self, trace: ReasoningTrace):
        """Render session overview metrics."""
        st.subheader("📊 Session Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Steps", len(trace.events))
        
        with col2:
            st.metric("Duration", f"{trace.total_duration_ms:.1f}ms")
        
        with col3:
            st.metric("Cognitive Vectors", len(trace.cognitive_trajectory))
        
        with col4:
            unique_components = len(set(event.component for event in trace.events))
            st.metric("Components Used", unique_components)
        
        # Query and response
        with st.expander("📝 Query & Response", expanded=False):
            st.markdown(f"**Query:** {trace.query}")
            st.markdown(f"**Response:** {trace.response}")
    
    def _render_cognitive_trajectory(self, trace: ReasoningTrace):
        """Render the main Algonauts cognitive trajectory visualization."""
        st.subheader("🎨 Cognitive Trajectory Visualization")
        
        # Projection method selector
        col1, col2 = st.columns([2, 1])
        
        with col1:
            projection_method = st.selectbox(
                "Dimensionality Reduction Method:",
                ["pca", "tsne", "umap"] if hasattr(self.visualizer, 'available_methods') else ["pca"],
                help="Choose how to project high-dimensional cognitive vectors to 2D"
            )
        
        with col2:
            show_path = st.checkbox("Show Trajectory Path", value=True)
        
        try:
            # Convert trace events for visualizer
            trace_events = []
            for event in trace.events:
                trace_events.append({
                    'id': event.event_id,
                    'step_type': event.step_type.value,
                    'component': event.component
                })
            
            # Project cognitive vectors
            trajectory = self.visualizer.project_cognitive_vectors(
                trace.cognitive_trajectory,
                method=projection_method,
                trace_events=trace_events
            )
            
            # Create the visualization
            fig = self._create_trajectory_plot(trajectory, show_path)
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Trajectory metrics
            self._render_trajectory_metrics(trajectory)
            
        except Exception as e:
            st.error(f"❌ Failed to generate cognitive trajectory: {str(e)}")
    
    def _create_trajectory_plot(self, trajectory, show_path: bool = True):
        """Create the Plotly trajectory visualization."""
        points = trajectory.projected_points
        
        # Prepare data
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]
        step_types = [p.step_type for p in points]
        components = [p.component for p in points]
        step_ids = [p.step_id[:8] for p in points]
        
        # Create color mapping
        unique_step_types = list(set(step_types))
        color_map = {step_type: i for i, step_type in enumerate(unique_step_types)}
        colors = [color_map[step_type] for step_type in step_types]
        
        # Create the main scatter plot
        fig = go.Figure()
        
        # Add trajectory path if requested
        if show_path and len(points) > 1:
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines',
                line=dict(color='rgba(128,128,128,0.5)', width=2),
                name='Trajectory Path',
                hoverinfo='skip'
            ))
        
        # Add scatter points
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker=dict(
                size=12,
                color=colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="Step Type",
                    tickvals=list(range(len(unique_step_types))),
                    ticktext=unique_step_types
                ),
                line=dict(width=2, color='white')
            ),
            text=[f"Step: {sid}<br>Type: {st}<br>Component: {comp}" 
                  for sid, st, comp in zip(step_ids, step_types, components)],
            hovertemplate='%{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>',
            name='Cognitive States'
        ))
        
        # Add step numbers as annotations
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            fig.add_annotation(
                x=x, y=y,
                text=str(i+1),
                showarrow=False,
                font=dict(color="white", size=10),
                bgcolor="rgba(0,0,0,0.7)",
                bordercolor="white",
                borderwidth=1
            )
        
        # Update layout
        fig.update_layout(
            title=f"Cognitive Trajectory ({trajectory.projection_method.upper()} Projection)",
            xaxis_title="Cognitive Dimension 1",
            yaxis_title="Cognitive Dimension 2",
            showlegend=True,
            height=600,
            template="plotly_white"
        )
        
        return fig
    
    def _render_trajectory_metrics(self, trajectory):
        """Render trajectory analysis metrics."""
        metrics = trajectory.trajectory_metrics
        
        if not metrics:
            return
        
        st.subheader("📈 Trajectory Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Path Length", 
                f"{metrics.get('total_distance', 0):.3f}",
                help="Total distance traveled in cognitive space"
            )
        
        with col2:
            st.metric(
                "Displacement", 
                f"{metrics.get('displacement', 0):.3f}",
                help="Direct distance from start to end"
            )
        
        with col3:
            tortuosity = metrics.get('tortuosity', 0)
            if tortuosity != float('inf'):
                st.metric(
                    "Tortuosity", 
                    f"{tortuosity:.2f}",
                    help="Path complexity (path length / displacement)"
                )
            else:
                st.metric("Tortuosity", "∞", help="Infinite (no displacement)")
        
        # Additional metrics in expander
        with st.expander("🔍 Detailed Metrics", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Average Step Size", f"{metrics.get('avg_step_size', 0):.3f}")
                st.metric("Step Variance", f"{metrics.get('step_variance', 0):.3f}")
            
            with col2:
                st.metric("Total Time", f"{metrics.get('total_time', 0):.3f}s")
                st.metric("Avg Time/Step", f"{metrics.get('avg_time_per_step', 0):.3f}s")
    
    def _render_reasoning_timeline(self, trace: ReasoningTrace):
        """Render interactive reasoning timeline."""
        st.subheader("⏱️ Reasoning Timeline")
        
        # Create timeline data
        timeline_data = []
        for i, event in enumerate(trace.events):
            timeline_data.append({
                'Step': i + 1,
                'Type': event.step_type.value.replace('_', ' ').title(),
                'Component': event.component,
                'Operation': event.operation,
                'Duration (ms)': event.duration_ms or 0,
                'Input': str(event.input_data)[:100] + "..." if len(str(event.input_data)) > 100 else str(event.input_data),
                'Output': str(event.output_data)[:100] + "..." if len(str(event.output_data)) > 100 else str(event.output_data)
            })
        
        df = pd.DataFrame(timeline_data)
        
        # Display as interactive table
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )
    
    def _render_pattern_analysis(self, trace: ReasoningTrace):
        """Render cognitive pattern analysis."""
        st.subheader("🔍 Pattern Analysis")
        
        try:
            # Convert trace events for visualizer
            trace_events = []
            for event in trace.events:
                trace_events.append({
                    'id': event.event_id,
                    'step_type': event.step_type.value,
                    'component': event.component
                })
            
            # Project and analyze
            trajectory = self.visualizer.project_cognitive_vectors(
                trace.cognitive_trajectory,
                method="pca",
                trace_events=trace_events
            )
            
            patterns = self.visualizer.analyze_cognitive_patterns(trajectory)
            
            # Display cluster analysis
            if 'cluster_analysis' in patterns:
                st.markdown("**🎯 Step Type Clustering:**")
                
                cluster_data = []
                for step_type, analysis in patterns['cluster_analysis'].items():
                    cluster_data.append({
                        'Step Type': step_type.replace('_', ' ').title(),
                        'Count': analysis['count'],
                        'Centroid X': f"{analysis['centroid'][0]:.3f}",
                        'Centroid Y': f"{analysis['centroid'][1]:.3f}",
                        'Spread X': f"{analysis['spread'][0]:.3f}",
                        'Spread Y': f"{analysis['spread'][1]:.3f}"
                    })
                
                cluster_df = pd.DataFrame(cluster_data)
                st.dataframe(cluster_df, use_container_width=True, hide_index=True)
            
            # Display complexity score
            if 'complexity_score' in patterns:
                complexity = patterns['complexity_score']
                st.metric(
                    "Cognitive Complexity Score", 
                    f"{complexity:.3f}",
                    help="Overall complexity of the reasoning process (0-1 scale)"
                )
            
        except Exception as e:
            st.error(f"❌ Pattern analysis failed: {str(e)}")
    



def render_algonauts_interface():
    """Main function to render the Algonauts interface in Streamlit."""
    interface = StreamlitAlgonautsInterface()
    interface.render_algonauts_tab()
