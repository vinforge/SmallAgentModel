#!/usr/bin/env python3
"""
Phase 6: Memory Control Dashboard for SAM
Advanced dashboard for visualizing and managing personalized learning and episodic memory.

This dashboard provides:
1. Episodic memory visualization and management
2. User modeling insights and preferences
3. Profile performance analytics
4. Feedback and learning trends
5. Personalization controls and settings
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sys
from pathlib import Path

# Add SAM to path
sys.path.append(str(Path(__file__).parent.parent))

# Import Phase 6 components
from memory.episodic_store import EpisodicMemoryStore, create_episodic_store, InteractionType
from profiles.user_modeler import UserModelingEngine, create_user_modeler
from learning.feedback_handler import FeedbackHandler, create_feedback_handler
from profiles.adaptive_refinement import AdaptiveProfileRefinement, create_adaptive_refinement

def initialize_phase6_components():
    """Initialize Phase 6 components."""
    try:
        # Initialize episodic store
        episodic_store = create_episodic_store()
        
        # Initialize user modeler
        user_modeler = create_user_modeler(episodic_store)
        
        # Initialize feedback handler
        feedback_handler = create_feedback_handler(episodic_store, user_modeler)
        
        # Initialize adaptive refinement
        adaptive_refinement = create_adaptive_refinement(episodic_store, user_modeler, feedback_handler)
        
        return episodic_store, user_modeler, feedback_handler, adaptive_refinement
        
    except Exception as e:
        st.error(f"Error initializing Phase 6 components: {e}")
        return None, None, None, None

def render_episodic_memory_section(episodic_store: EpisodicMemoryStore, user_id: str):
    """Render episodic memory visualization section."""
    st.header("📚 Episodic Memory")
    
    col1, col2, col3 = st.columns(3)
    
    # Get user memories
    memories = episodic_store.retrieve_memories(user_id, limit=100)
    
    with col1:
        st.metric("Total Memories", len(memories))
    
    with col2:
        recent_memories = [m for m in memories if _is_recent(m.timestamp, days=7)]
        st.metric("Recent (7 days)", len(recent_memories))
    
    with col3:
        avg_confidence = sum(m.confidence_score for m in memories) / len(memories) if memories else 0
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    # Memory timeline
    if memories:
        st.subheader("Memory Timeline")
        
        # Prepare data for timeline
        timeline_data = []
        for memory in memories[-20:]:  # Last 20 memories
            timeline_data.append({
                "timestamp": memory.timestamp,
                "query": memory.query[:50] + "..." if len(memory.query) > 50 else memory.query,
                "profile": memory.active_profile,
                "confidence": memory.confidence_score,
                "satisfaction": memory.user_satisfaction or 0.5
            })
        
        df = pd.DataFrame(timeline_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create timeline chart
        fig = px.scatter(df, x='timestamp', y='confidence', 
                        color='profile', size='satisfaction',
                        hover_data=['query'],
                        title="Memory Timeline (Last 20 Interactions)")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Memory details table
        st.subheader("Recent Memories")
        display_df = df[['timestamp', 'query', 'profile', 'confidence', 'satisfaction']].copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(display_df, use_container_width=True)

def render_user_modeling_section(user_modeler: UserModelingEngine, user_id: str):
    """Render user modeling insights section."""
    st.header("🧠 User Modeling & Preferences")
    
    # Analyze user behavior
    user_model = user_modeler.analyze_user_behavior(user_id)
    
    if user_model.total_interactions < 5:
        st.info("Insufficient interaction data for comprehensive user modeling. Continue using SAM to build your personalized profile!")
        return
    
    # User statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Interactions", user_model.total_interactions)
    
    with col2:
        st.metric("Preferences Learned", len(user_model.preferences))
    
    with col3:
        st.metric("Learning Velocity", f"{user_model.learning_velocity:.2f}")
    
    with col4:
        st.metric("Consistency Score", f"{user_model.consistency_score:.2f}")
    
    # Profile usage distribution
    if user_model.profile_usage_distribution:
        st.subheader("Profile Usage Distribution")
        
        profile_df = pd.DataFrame(list(user_model.profile_usage_distribution.items()), 
                                columns=['Profile', 'Usage Count'])
        
        fig = px.pie(profile_df, values='Usage Count', names='Profile',
                    title="Profile Usage Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Learned preferences
    if user_model.preferences:
        st.subheader("Learned Preferences")
        
        preferences_data = []
        for pref in user_model.preferences:
            preferences_data.append({
                "Type": pref.preference_type.value.replace('_', ' ').title(),
                "Description": pref.description,
                "Confidence": pref.confidence.value.title(),
                "Evidence": pref.evidence_count
            })
        
        pref_df = pd.DataFrame(preferences_data)
        st.dataframe(pref_df, use_container_width=True)
    
    # Personalized profiles
    if user_model.personalized_profiles:
        st.subheader("Personalized Profiles")
        
        for profile in user_model.personalized_profiles:
            with st.expander(f"📊 {profile.profile_name}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Usage Count", profile.usage_count)
                
                with col2:
                    st.metric("Success Rate", f"{profile.success_rate:.2f}")
                
                with col3:
                    st.metric("Avg Satisfaction", f"{profile.average_satisfaction:.2f}")
                
                # Dimension weights visualization
                weights_df = pd.DataFrame(list(profile.dimension_weights.items()),
                                        columns=['Dimension', 'Weight'])
                
                fig = px.bar(weights_df, x='Dimension', y='Weight',
                           title=f"{profile.profile_name} - Dimension Weights")
                st.plotly_chart(fig, use_container_width=True)

def render_feedback_analytics_section(feedback_handler: FeedbackHandler, user_id: str):
    """Render feedback analytics section."""
    st.header("📊 Feedback & Learning Analytics")
    
    # Get feedback analysis
    feedback_analysis = feedback_handler.analyze_feedback_patterns(user_id)
    
    if feedback_analysis.get("total_feedback", 0) == 0:
        st.info("No feedback data available yet. Start providing feedback to see analytics!")
        return
    
    # Feedback statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Feedback", feedback_analysis.get("total_feedback", 0))
    
    with col2:
        avg_satisfaction = feedback_analysis.get("average_satisfaction", 0)
        st.metric("Avg Satisfaction", f"{avg_satisfaction:.2f}")
    
    with col3:
        patterns_detected = feedback_analysis.get("patterns_detected", 0)
        st.metric("Patterns Detected", patterns_detected)
    
    # Feedback types distribution
    if "feedback_types" in feedback_analysis:
        st.subheader("Feedback Types")
        
        feedback_types = feedback_analysis["feedback_types"]
        types_df = pd.DataFrame(list(feedback_types.items()),
                              columns=['Feedback Type', 'Count'])
        
        fig = px.bar(types_df, x='Feedback Type', y='Count',
                    title="Feedback Types Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Learning recommendations
    recommendations = feedback_handler.get_learning_recommendations(user_id)
    
    if recommendations:
        st.subheader("Learning Recommendations")
        
        for rec in recommendations[:5]:  # Show top 5
            with st.expander(f"💡 {rec['type'].replace('_', ' ').title()} (Confidence: {rec['confidence']:.2f})"):
                st.write(rec['description'])
                
                if rec.get('actions'):
                    st.write("**Recommended Actions:**")
                    for action in rec['actions']:
                        st.write(f"• {action}")
                
                if st.button(f"Apply Recommendation", key=f"apply_{rec['insight_id']}"):
                    if feedback_handler.apply_learning_insight(rec['insight_id']):
                        st.success("Recommendation applied successfully!")
                    else:
                        st.error("Failed to apply recommendation")

def render_profile_performance_section(adaptive_refinement: AdaptiveProfileRefinement, user_id: str):
    """Render profile performance analytics section."""
    st.header("🎯 Profile Performance & Refinement")
    
    # Get refinement opportunities
    opportunities = adaptive_refinement.detect_refinement_opportunities(user_id)
    
    if not opportunities:
        st.info("No refinement opportunities detected. Your profiles are performing well!")
        return
    
    # Refinement opportunities
    st.subheader("Refinement Opportunities")
    
    for opp in opportunities[:3]:  # Show top 3
        with st.expander(f"🔧 {opp['type'].replace('_', ' ').title()} - {opp['profile']}"):
            st.write(opp['description'])
            st.write(f"**Confidence:** {opp['confidence']:.2f}")
            st.write(f"**Recommended Action:** {opp['recommended_action'].replace('_', ' ').title()}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button(f"Apply Refinement", key=f"refine_{opp['profile']}_{opp['type']}"):
                    refinement = adaptive_refinement.refine_profile(
                        user_id=user_id,
                        profile_id=opp['profile'],
                        trigger=opp['trigger'],
                        auto_apply=True
                    )
                    
                    if refinement:
                        st.success("Profile refinement applied!")
                    else:
                        st.error("Failed to apply refinement")
            
            with col2:
                if st.button(f"Dismiss", key=f"dismiss_{opp['profile']}_{opp['type']}"):
                    st.info("Opportunity dismissed")
    
    # Refinement history
    refinement_summary = adaptive_refinement.get_refinement_summary(user_id)
    
    if refinement_summary.get("total_refinements", 0) > 0:
        st.subheader("Refinement History")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Refinements", refinement_summary["total_refinements"])
        
        with col2:
            st.metric("Auto Applied", refinement_summary["auto_applied"])
        
        with col3:
            st.metric("Avg Confidence", f"{refinement_summary['average_confidence']:.2f}")

def render_personalization_controls(user_id: str):
    """Render personalization controls section."""
    st.header("⚙️ Personalization Controls")
    
    # User preferences
    st.subheader("Personalization Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        auto_refinement = st.checkbox("Enable Automatic Profile Refinement", value=True)
        feedback_prompts = st.checkbox("Enable Feedback Prompts", value=True)
        learning_notifications = st.checkbox("Show Learning Notifications", value=True)
    
    with col2:
        adaptation_strategy = st.selectbox(
            "Adaptation Strategy",
            ["Conservative", "Moderate", "Aggressive"],
            index=1
        )
        
        memory_retention = st.slider(
            "Memory Retention (days)",
            min_value=30,
            max_value=365,
            value=180
        )
    
    # Data management
    st.subheader("Data Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Export Personal Data"):
            st.info("Personal data export initiated")
    
    with col2:
        if st.button("Clear Old Memories"):
            st.warning("This will remove memories older than retention period")
    
    with col3:
        if st.button("Reset Personalization"):
            st.error("This will reset all learned preferences")

def _is_recent(timestamp_str: str, days: int = 7) -> bool:
    """Check if timestamp is within recent days."""
    try:
        timestamp = datetime.fromisoformat(timestamp_str)
        cutoff = datetime.now() - timedelta(days=days)
        return timestamp >= cutoff
    except Exception:
        return False

def main():
    """Main Phase 6 Memory Dashboard application."""
    st.set_page_config(
        page_title="SAM Phase 6 - Memory & Personalization Dashboard",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 SAM Phase 6: Memory & Personalization Dashboard")
    st.markdown("**Advanced episodic memory, user modeling, and adaptive personalization**")
    
    # Initialize components
    with st.spinner("Initializing Phase 6 components..."):
        episodic_store, user_modeler, feedback_handler, adaptive_refinement = initialize_phase6_components()
    
    if not all([episodic_store, user_modeler, feedback_handler, adaptive_refinement]):
        st.error("Failed to initialize Phase 6 components")
        return
    
    # User selection
    st.sidebar.header("User Selection")
    user_id = st.sidebar.text_input("User ID", value="default_user")
    
    if not user_id:
        st.warning("Please enter a User ID to continue")
        return
    
    # Navigation
    st.sidebar.header("Navigation")
    section = st.sidebar.selectbox(
        "Select Section",
        [
            "📚 Episodic Memory",
            "🧠 User Modeling",
            "📊 Feedback Analytics",
            "🎯 Profile Performance",
            "⚙️ Personalization Controls"
        ]
    )
    
    # Render selected section
    if section == "📚 Episodic Memory":
        render_episodic_memory_section(episodic_store, user_id)
    elif section == "🧠 User Modeling":
        render_user_modeling_section(user_modeler, user_id)
    elif section == "📊 Feedback Analytics":
        render_feedback_analytics_section(feedback_handler, user_id)
    elif section == "🎯 Profile Performance":
        render_profile_performance_section(adaptive_refinement, user_id)
    elif section == "⚙️ Personalization Controls":
        render_personalization_controls(user_id)
    
    # Footer
    st.markdown("---")
    st.markdown("**SAM Phase 6** - Personalization Engine + Episodic Memory + Lifelong Learning")

if __name__ == "__main__":
    main()
