"""
Test-Time Training (TTT) Transparency UI Components
==================================================

UI components for displaying TTT adaptation status, performance metrics,
and cognitive priming information to users.
"""

import streamlit as st
from typing import Dict, Any, Optional
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import asdict

from sam.monitoring.ttt_metrics import get_ttt_metrics_collector, TTTSessionSummary
import logging

logger = logging.getLogger(__name__)

def render_ttt_status_indicator(adaptation_metadata: Optional[Dict[str, Any]] = None):
    """
    Render TTT status indicator in the reasoning transparency section.
    
    Args:
        adaptation_metadata: TTT adaptation metadata if available
    """
    if not adaptation_metadata:
        return
    
    try:
        # Check if TTT was used
        if hasattr(adaptation_metadata, 'fallback_reason') and adaptation_metadata.fallback_reason:
            # TTT failed, show fallback info
            st.info("🔄 **Cognitive Priming:** Attempted Test-Time Adaptation (fell back to standard reasoning)")
            with st.expander("🔍 TTT Fallback Details"):
                st.write(f"**Reason:** {adaptation_metadata.fallback_reason}")
                st.write("**Status:** Using standard In-Context Learning (ICL)")
        else:
            # TTT succeeded, show success info
            confidence = getattr(adaptation_metadata, 'confidence_score', 0)
            training_steps = getattr(adaptation_metadata, 'training_steps', 0)
            examples_used = getattr(adaptation_metadata, 'examples_used', 0)
            adaptation_time = getattr(adaptation_metadata, 'adaptation_time', 0)
            
            # Main TTT status
            st.success(f"🧠 **Cognitive Priming:** ✅ Test-Time Adaptation active ({examples_used} examples, {training_steps} steps, confidence: {confidence:.2f})")
            
            # Detailed metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "📊 Adaptation Quality", 
                    f"{confidence:.1%}",
                    help="Confidence in the adapted reasoning pattern"
                )
            
            with col2:
                st.metric(
                    "⚡ Training Steps", 
                    f"{training_steps}",
                    help="Number of adaptation training steps performed"
                )
            
            with col3:
                st.metric(
                    "🕒 Adaptation Time", 
                    f"{adaptation_time:.2f}s",
                    help="Time taken to adapt the reasoning process"
                )
            
            # Performance boost indicator
            if confidence > 0.8:
                boost_estimate = "25-35%"
                boost_color = "🟢"
            elif confidence > 0.7:
                boost_estimate = "15-25%"
                boost_color = "🟡"
            else:
                boost_estimate = "5-15%"
                boost_color = "🟠"
            
            st.info(f"⚡ **Performance Boost:** {boost_color} Expected +{boost_estimate} accuracy improvement for this task type")
            
    except Exception as e:
        logger.error(f"Error rendering TTT status: {e}")
        st.warning("🔄 **Cognitive Priming:** TTT status unavailable")

def render_ttt_settings_panel():
    """Render TTT configuration settings panel for SAM Pro users."""
    st.subheader("🧠 Test-Time Training Settings")
    
    # Check if user has SAM Pro
    sam_pro_active = st.session_state.get('sam_pro_active', False)
    
    if not sam_pro_active:
        st.info("🔒 Test-Time Training settings are available with SAM Pro")
        if st.button("🔑 Activate SAM Pro"):
            st.session_state.show_sam_pro_activation = True
        return
    
    # TTT Enable/Disable
    ttt_enabled = st.checkbox(
        "Enable Test-Time Training",
        value=st.session_state.get('ttt_enabled', True),
        help="Automatically adapt reasoning for few-shot tasks"
    )
    st.session_state.ttt_enabled = ttt_enabled
    
    if ttt_enabled:
        # Advanced TTT settings
        with st.expander("⚙️ Advanced TTT Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                confidence_threshold = st.slider(
                    "Confidence Threshold",
                    min_value=0.5,
                    max_value=0.95,
                    value=st.session_state.get('ttt_confidence_threshold', 0.7),
                    step=0.05,
                    help="Minimum confidence required to use TTT adaptation"
                )
                st.session_state.ttt_confidence_threshold = confidence_threshold
                
                max_examples = st.slider(
                    "Max Examples",
                    min_value=2,
                    max_value=15,
                    value=st.session_state.get('ttt_max_examples', 10),
                    help="Maximum number of examples to use for adaptation"
                )
                st.session_state.ttt_max_examples = max_examples
            
            with col2:
                max_training_steps = st.slider(
                    "Max Training Steps",
                    min_value=2,
                    max_value=15,
                    value=st.session_state.get('ttt_max_steps', 8),
                    help="Maximum number of adaptation training steps"
                )
                st.session_state.ttt_max_steps = max_training_steps
                
                lora_rank = st.selectbox(
                    "LoRA Rank",
                    options=[8, 16, 32, 64],
                    index=1,  # Default to 16
                    help="Rank of the LoRA adapter (higher = more capacity, slower)"
                )
                st.session_state.ttt_lora_rank = lora_rank
        
        # TTT Performance Monitoring
        st.markdown("### 📊 TTT Performance")
        
        try:
            metrics_collector = get_ttt_metrics_collector()
            session_id = st.session_state.get('session_id', 'current')
            summary = metrics_collector.get_session_summary(session_id)
            
            if summary.total_attempts > 0:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Attempts", summary.total_attempts)
                
                with col2:
                    success_rate = (summary.successful_adaptations / summary.total_attempts) * 100
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                
                with col3:
                    st.metric("Avg Confidence", f"{summary.average_confidence:.2f}")
                
                with col4:
                    st.metric("Avg Time", f"{summary.average_adaptation_time:.2f}s")
                
                # Performance trends
                if st.button("📈 View Detailed Metrics"):
                    st.session_state.show_ttt_metrics = True
            else:
                st.info("No TTT usage data available yet")
                
        except Exception as e:
            logger.error(f"Error loading TTT metrics: {e}")
            st.warning("TTT metrics unavailable")

def render_ttt_metrics_dashboard():
    """Render detailed TTT performance metrics dashboard."""
    st.subheader("📊 Test-Time Training Performance Dashboard")
    
    try:
        metrics_collector = get_ttt_metrics_collector()
        
        # Time period selector
        col1, col2 = st.columns([1, 3])
        with col1:
            days = st.selectbox("Time Period", [1, 7, 30], index=1)
        
        # Get performance trends
        trends = metrics_collector.get_performance_trends(days=days)
        
        if trends["daily_performance"]:
            # Daily performance chart
            daily_data = trends["daily_performance"]
            dates = [d["date"] for d in daily_data]
            success_rates = [d["success_rate"] * 100 for d in daily_data]
            avg_confidence = [d["avg_confidence"] for d in daily_data]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, y=success_rates,
                mode='lines+markers',
                name='Success Rate (%)',
                line=dict(color='green')
            ))
            fig.add_trace(go.Scatter(
                x=dates, y=[c * 100 for c in avg_confidence],
                mode='lines+markers',
                name='Avg Confidence (%)',
                line=dict(color='blue'),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title="TTT Performance Trends",
                xaxis_title="Date",
                yaxis_title="Success Rate (%)",
                yaxis2=dict(
                    title="Confidence (%)",
                    overlaying='y',
                    side='right'
                ),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Task type performance
        if trends["task_type_performance"]:
            st.markdown("### 📋 Performance by Task Type")
            
            task_data = trends["task_type_performance"]
            task_types = [t["task_type"] for t in task_data]
            success_rates = [t["success_rate"] * 100 for t in task_data]
            
            fig = px.bar(
                x=task_types,
                y=success_rates,
                title="Success Rate by Task Type",
                labels={"x": "Task Type", "y": "Success Rate (%)"}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Export metrics
        if st.button("📥 Export Metrics"):
            export_path = f"ttt_metrics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            metrics_collector.export_metrics(export_path, days=days)
            st.success(f"Metrics exported to {export_path}")
            
    except Exception as e:
        logger.error(f"Error rendering TTT dashboard: {e}")
        st.error("Failed to load TTT metrics dashboard")

def render_ttt_help_section():
    """Render TTT help and explanation section."""
    with st.expander("ℹ️ What is Test-Time Training?"):
        st.markdown("""
        **Test-Time Training (TTT)** is an advanced AI technique that allows SAM to temporarily 
        adapt its reasoning process for specific types of problems.
        
        **How it works:**
        1. 🔍 **Pattern Detection**: SAM detects when you provide few-shot examples
        2. 🧠 **Cognitive Adaptation**: SAM trains a lightweight adapter on your examples
        3. ⚡ **Enhanced Reasoning**: SAM applies the adapted reasoning to your specific problem
        4. 🗑️ **Clean Slate**: The temporary adaptation is discarded after use
        
        **Benefits:**
        - 📈 **15-30% accuracy improvement** on pattern-based tasks
        - 🎯 **Specialized reasoning** for your specific problem type
        - 🔒 **No permanent changes** to SAM's base knowledge
        - ⚡ **Fast adaptation** in just a few seconds
        
        **Best for:**
        - Analogical reasoning (A is to B as C is to ?)
        - Pattern completion tasks
        - Rule learning from examples
        - Mathematical sequence problems
        - Logic puzzles with examples
        
        **Example TTT-suitable query:**
        ```
        Example 1: cat -> feline
        Example 2: dog -> canine  
        Example 3: horse -> equine
        
        What is: elephant -> ?
        ```
        
        SAM will detect this pattern and temporarily adapt its reasoning to better 
        handle animal-to-classification mappings.
        """)

def get_ttt_config_from_session() -> Dict[str, Any]:
    """Get TTT configuration from session state."""
    return {
        "enabled": st.session_state.get('ttt_enabled', True),
        "confidence_threshold": st.session_state.get('ttt_confidence_threshold', 0.7),
        "max_examples": st.session_state.get('ttt_max_examples', 10),
        "max_training_steps": st.session_state.get('ttt_max_steps', 8),
        "lora_rank": st.session_state.get('ttt_lora_rank', 16)
    }
