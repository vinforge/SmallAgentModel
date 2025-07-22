"""
TPV Visualization Module for SAM
Phase 5B - Dissonance-Aware Meta-Reasoning UI

This module provides advanced visualization components for TPV monitoring
including dual-line charts for reasoning progress and cognitive dissonance.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def render_tpv_dissonance_chart(trace_data: Dict[str, Any], 
                               expanded: bool = False,
                               height: int = 400) -> None:
    """
    Render dual-line chart showing TPV progress and cognitive dissonance.
    
    Args:
        trace_data: TPV trace data containing steps and dissonance scores
        expanded: Whether to show expanded view
        height: Chart height in pixels
    """
    try:
        if not trace_data or 'steps' not in trace_data:
            st.info("📊 No TPV trace data available for visualization")
            return
        
        steps = trace_data['steps']
        if not steps:
            st.info("📊 No reasoning steps recorded yet")
            return
        
        # Extract data for visualization
        step_numbers = []
        tpv_scores = []
        dissonance_scores = []
        timestamps = []
        
        for i, step in enumerate(steps):
            step_numbers.append(i + 1)
            tpv_scores.append(step.get('tpv_score', 0.0))
            dissonance_scores.append(step.get('dissonance_score'))
            timestamps.append(step.get('timestamp', datetime.now().timestamp()))
        
        # Create dual-axis chart
        fig = make_subplots(
            specs=[[{"secondary_y": True}]],
            subplot_titles=["🧠 Real-Time Cognitive Analysis"]
        )
        
        # Add TPV progress line
        fig.add_trace(
            go.Scatter(
                x=step_numbers,
                y=tpv_scores,
                mode='lines+markers',
                name='Reasoning Progress (TPV)',
                line=dict(color='#2E86AB', width=3),
                marker=dict(size=8, symbol='circle'),
                hovertemplate='<b>Step %{x}</b><br>' +
                             'TPV Score: %{y:.3f}<br>' +
                             '<extra></extra>'
            ),
            secondary_y=False
        )
        
        # Add dissonance line if data available
        valid_dissonance = [d for d in dissonance_scores if d is not None]
        if valid_dissonance:
            # Filter out None values for plotting
            dissonance_x = [step_numbers[i] for i, d in enumerate(dissonance_scores) if d is not None]
            dissonance_y = [d for d in dissonance_scores if d is not None]
            
            fig.add_trace(
                go.Scatter(
                    x=dissonance_x,
                    y=dissonance_y,
                    mode='lines+markers',
                    name='Cognitive Dissonance',
                    line=dict(color='#F18F01', width=3, dash='dot'),
                    marker=dict(size=8, symbol='diamond'),
                    hovertemplate='<b>Step %{x}</b><br>' +
                                 'Dissonance: %{y:.3f}<br>' +
                                 '<extra></extra>'
                ),
                secondary_y=True
            )
            
            # Add dissonance threshold line
            threshold = trace_data.get('dissonance_threshold', 0.85)
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Dissonance Threshold ({threshold})",
                secondary_y=True
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="🧠 Cognitive Process Monitoring",
                x=0.5,
                font=dict(size=16)
            ),
            height=height,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(t=80, b=40, l=40, r=40)
        )
        
        # Update y-axes
        fig.update_yaxes(
            title_text="Reasoning Progress (0-1)",
            secondary_y=False,
            range=[0, 1.1],
            tickformat='.2f'
        )
        
        if valid_dissonance:
            fig.update_yaxes(
                title_text="Cognitive Dissonance (0-1)",
                secondary_y=True,
                range=[0, 1.1],
                tickformat='.2f'
            )
        
        fig.update_xaxes(
            title_text="Reasoning Step",
            tickmode='linear',
            tick0=1,
            dtick=1
        )
        
        # Display chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed metrics if expanded
        if expanded:
            render_detailed_metrics(trace_data, steps, valid_dissonance)
            
    except Exception as e:
        logger.error(f"Error rendering TPV dissonance chart: {e}")
        st.error(f"❌ Error rendering visualization: {e}")

def render_detailed_metrics(trace_data: Dict[str, Any], 
                           steps: List[Dict[str, Any]], 
                           valid_dissonance: List[float]) -> None:
    """
    Render detailed metrics below the chart.
    
    Args:
        trace_data: Complete trace data
        steps: List of reasoning steps
        valid_dissonance: List of valid dissonance scores
    """
    st.markdown("### 📊 Detailed Analysis")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        final_tpv = steps[-1].get('tpv_score', 0.0) if steps else 0.0
        st.metric(
            "Final TPV Score",
            f"{final_tpv:.3f}",
            help="Final reasoning quality score"
        )
    
    with col2:
        st.metric(
            "Total Steps",
            len(steps),
            help="Number of reasoning steps completed"
        )
    
    with col3:
        if valid_dissonance:
            avg_dissonance = np.mean(valid_dissonance)
            st.metric(
                "Avg Dissonance",
                f"{avg_dissonance:.3f}",
                help="Average cognitive dissonance across steps"
            )
        else:
            st.metric("Avg Dissonance", "N/A", help="No dissonance data available")
    
    with col4:
        if valid_dissonance:
            max_dissonance = max(valid_dissonance)
            st.metric(
                "Peak Dissonance",
                f"{max_dissonance:.3f}",
                help="Highest dissonance score recorded"
            )
        else:
            st.metric("Peak Dissonance", "N/A", help="No dissonance data available")
    
    # Dissonance analysis
    if valid_dissonance:
        render_dissonance_analysis(valid_dissonance, trace_data)

def render_dissonance_analysis(dissonance_scores: List[float], 
                              trace_data: Dict[str, Any]) -> None:
    """
    Render detailed dissonance analysis.
    
    Args:
        dissonance_scores: List of dissonance scores
        trace_data: Complete trace data
    """
    st.markdown("### 🔍 Dissonance Analysis")
    
    # Calculate statistics
    mean_dissonance = np.mean(dissonance_scores)
    std_dissonance = np.std(dissonance_scores)
    threshold = trace_data.get('dissonance_threshold', 0.85)
    
    # Analysis columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Statistical Summary:**")
        st.write(f"• Mean: {mean_dissonance:.3f}")
        st.write(f"• Std Dev: {std_dissonance:.3f}")
        st.write(f"• Min: {min(dissonance_scores):.3f}")
        st.write(f"• Max: {max(dissonance_scores):.3f}")
    
    with col2:
        st.markdown("**Threshold Analysis:**")
        high_dissonance_count = sum(1 for d in dissonance_scores if d > threshold)
        high_dissonance_pct = (high_dissonance_count / len(dissonance_scores)) * 100
        
        st.write(f"• Threshold: {threshold}")
        st.write(f"• High dissonance steps: {high_dissonance_count}")
        st.write(f"• Percentage above threshold: {high_dissonance_pct:.1f}%")
        
        # Warning if high dissonance
        if high_dissonance_pct > 50:
            st.warning("⚠️ High cognitive conflict detected")
        elif high_dissonance_pct > 25:
            st.info("ℹ️ Moderate cognitive uncertainty")
        else:
            st.success("✅ Low cognitive dissonance")

def render_tpv_status_enhanced(tpv_data: Optional[Dict[str, Any]] = None) -> None:
    """
    Enhanced TPV status display with dissonance monitoring.
    
    Args:
        tpv_data: TPV data from the last response
    """
    try:
        if not tpv_data:
            return
        
        if tpv_data.get('tpv_enabled'):
            with st.expander("🧠 Cognitive Process Analysis (Phase 5B: Dissonance-Aware)", expanded=False):
                # Main metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Reasoning Score",
                        f"{tpv_data.get('final_score', 0.0):.3f}",
                        help="Final reasoning quality score (0.0 - 1.0)"
                    )
                
                with col2:
                    st.metric(
                        "TPV Steps",
                        tpv_data.get('tpv_steps', 0),
                        help="Number of reasoning steps monitored"
                    )
                
                with col3:
                    dissonance_score = tpv_data.get('final_dissonance_score')
                    if dissonance_score is not None:
                        st.metric(
                            "Final Dissonance",
                            f"{dissonance_score:.3f}",
                            help="Final cognitive dissonance score (0.0 - 1.0)"
                        )
                    else:
                        st.metric("Final Dissonance", "N/A", help="Dissonance monitoring not available")
                
                with col4:
                    trigger_type = tpv_data.get('trigger_type', 'none')
                    st.metric(
                        "Trigger Type",
                        trigger_type.title(),
                        help="What triggered TPV monitoring"
                    )
                
                # Control decision info
                control_decision = tpv_data.get('control_decision', 'unknown')
                if control_decision != 'continue':
                    if control_decision == 'stop_dissonance':
                        st.warning("🧠 **Stopped due to high cognitive dissonance**")
                    elif control_decision == 'stop_completion':
                        st.success("✅ **Completed successfully**")
                    elif control_decision == 'stop_plateau':
                        st.info("📊 **Stopped due to reasoning plateau**")
                    else:
                        st.info(f"ℹ️ **Control decision**: {control_decision}")
                
                # Render visualization if trace data available
                if 'trace' in tpv_data:
                    render_tpv_dissonance_chart(tpv_data['trace'], expanded=True, height=350)
                
                # Performance metrics
                if 'performance_metrics' in tpv_data:
                    render_performance_metrics(tpv_data['performance_metrics'])
        
        elif tpv_data and not tpv_data.get('tpv_enabled'):
            with st.expander("🧠 Thinking Process Analysis", expanded=False):
                trigger_type = tpv_data.get('trigger_type', 'none')
                st.info(f"🔍 **TPV Not Triggered**: {trigger_type.replace('_', ' ').title()} - Standard response generation used.")
    
    except Exception as e:
        logger.debug(f"Enhanced TPV status display error: {e}")

def render_performance_metrics(metrics: Dict[str, Any]) -> None:
    """
    Render performance metrics for TPV and dissonance monitoring.
    
    Args:
        metrics: Performance metrics dictionary
    """
    st.markdown("### ⚡ Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tpv_time = metrics.get('tpv_processing_time', 0.0)
        st.metric(
            "TPV Processing",
            f"{tpv_time:.3f}s",
            help="Time spent on TPV monitoring"
        )
    
    with col2:
        dissonance_time = metrics.get('dissonance_processing_time', 0.0)
        st.metric(
            "Dissonance Analysis",
            f"{dissonance_time:.3f}s",
            help="Time spent on dissonance calculation"
        )
    
    with col3:
        total_overhead = metrics.get('total_overhead', 0.0)
        st.metric(
            "Total Overhead",
            f"{total_overhead:.3f}s",
            help="Total additional processing time"
        )

def render_real_time_monitor(session_id: Optional[str] = None) -> None:
    """
    Render real-time TPV and dissonance monitoring interface.
    
    Args:
        session_id: Optional session ID for monitoring
    """
    st.markdown("### 🔴 Real-Time Cognitive Monitoring")
    
    if not session_id:
        st.info("💡 Start a conversation to see real-time cognitive monitoring")
        return
    
    # Placeholder for real-time monitoring
    # This would connect to the TPV system for live updates
    placeholder = st.empty()
    
    with placeholder.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Current TPV Score", "0.000", help="Real-time reasoning progress")
        
        with col2:
            st.metric("Current Dissonance", "0.000", help="Real-time cognitive dissonance")
        
        # Real-time chart placeholder
        st.info("📊 Real-time chart will appear here during active reasoning")

def create_sample_trace_data() -> Dict[str, Any]:
    """
    Create sample trace data for testing visualization.
    
    Returns:
        Sample trace data dictionary
    """
    steps = []
    for i in range(8):
        # Simulate realistic TPV and dissonance progression
        tpv_score = min(1.0, 0.1 + (i * 0.12) + np.random.normal(0, 0.02))
        
        # Simulate dissonance - higher at beginning and end
        if i < 2:
            dissonance = 0.7 + np.random.normal(0, 0.1)  # High initial uncertainty
        elif i > 5:
            dissonance = 0.8 + np.random.normal(0, 0.1)  # High final uncertainty
        else:
            dissonance = 0.3 + np.random.normal(0, 0.1)  # Lower middle uncertainty
        
        dissonance = max(0.0, min(1.0, dissonance))
        
        steps.append({
            'tpv_score': tpv_score,
            'dissonance_score': dissonance,
            'timestamp': datetime.now().timestamp() + i,
            'step_number': i + 1
        })
    
    return {
        'steps': steps,
        'dissonance_threshold': 0.85,
        'final_score': steps[-1]['tpv_score'],
        'control_decision': 'stop_completion'
    }

def demo_visualization() -> None:
    """Demo function to test the visualization components."""
    st.markdown("## 🧪 TPV Dissonance Visualization Demo")
    
    # Create sample data
    sample_data = create_sample_trace_data()
    
    # Render the visualization
    render_tpv_dissonance_chart(sample_data, expanded=True)
    
    # Show raw data
    with st.expander("📋 Raw Trace Data", expanded=False):
        st.json(sample_data)
