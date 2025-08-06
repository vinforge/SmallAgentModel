#!/usr/bin/env python3
"""
Insight Archive UI Component for SAM
===================================

Streamlit UI for browsing, searching, and managing archived insights.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import sys
sys.path.append('.')

def render_insight_archive():
    """Render the main insight archive interface."""
    st.markdown("## 📚 Insight Archive")
    st.markdown("*Browse and search your collection of emergent insights*")
    
    try:
        from memory.synthesis.insight_archive import get_insight_archive
        
        archive = get_insight_archive()
        
        # Get archive statistics
        stats = archive.get_archive_stats()
        
        if 'error' in stats:
            st.error(f"❌ Error accessing archive: {stats['error']}")
            return
        
        # Display archive overview
        render_archive_overview(stats)
        
        # Archive controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            view_mode = st.selectbox(
                "📋 View Mode",
                ["Recent Insights", "Search Insights", "Browse by Category", "Quality Rankings"],
                help="Choose how to browse the archive"
            )
        
        with col2:
            if st.button("📊 Archive Analytics", help="View detailed analytics"):
                st.session_state.show_archive_analytics = True
        
        with col3:
            if st.button("📤 Export Archive", help="Export insights to file"):
                st.session_state.show_export_options = True
        
        # Main content area
        if view_mode == "Recent Insights":
            render_recent_insights(archive)
        elif view_mode == "Search Insights":
            render_search_interface(archive)
        elif view_mode == "Browse by Category":
            render_category_browser(archive, stats)
        elif view_mode == "Quality Rankings":
            render_quality_rankings(archive)
        
        # Optional sections
        if hasattr(st.session_state, 'show_archive_analytics') and st.session_state.show_archive_analytics:
            render_archive_analytics(archive, stats)
        
        if hasattr(st.session_state, 'show_export_options') and st.session_state.show_export_options:
            render_export_interface(archive)
            
    except Exception as e:
        st.error(f"❌ Error loading insight archive: {e}")

def render_archive_overview(stats: Dict[str, Any]):
    """Render archive overview with key statistics."""
    st.markdown("### 📊 Archive Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Insights", stats.get('total_insights', 0))
    
    with col2:
        st.metric("High Quality", stats.get('high_quality_insights', 0))
    
    with col3:
        st.metric("Recent (7d)", stats.get('recent_insights_7d', 0))
    
    with col4:
        avg_quality = stats.get('average_quality', 0)
        st.metric("Avg Quality", f"{avg_quality:.2f}")
    
    # Categories breakdown
    categories = stats.get('categories', {})
    if categories:
        st.markdown("**📂 Categories:**")
        category_text = " | ".join([f"{cat}: {count}" for cat, count in categories.items()])
        st.caption(category_text)

def render_recent_insights(archive):
    """Render recent insights view."""
    st.markdown("### 🕒 Recent Insights")
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        days_back = st.slider("Days to look back", 1, 30, 7)
    with col2:
        min_quality = st.slider("Minimum quality", 0.0, 1.0, 0.3, 0.1)
    
    # Get recent insights
    since = datetime.now() - timedelta(days=days_back)
    insights = archive.get_insights(
        limit=20,
        min_quality=min_quality,
        since=since
    )
    
    if not insights:
        st.info(f"No insights found in the last {days_back} days with quality ≥ {min_quality}")
        return
    
    # Display insights
    for i, insight in enumerate(insights):
        render_insight_card(insight, i)

def render_search_interface(archive):
    """Render search interface."""
    st.markdown("### 🔍 Search Insights")
    
    # Search input
    search_query = st.text_input(
        "Search query",
        placeholder="Enter keywords to search insights...",
        help="Search in insight text, summaries, and tags"
    )
    
    if search_query:
        # Perform search
        results = archive.search_insights(search_query, limit=15)
        
        if results:
            st.success(f"Found {len(results)} insights matching '{search_query}'")
            
            for i, insight in enumerate(results):
                render_insight_card(insight, i, highlight_query=search_query)
        else:
            st.warning(f"No insights found matching '{search_query}'")

def render_category_browser(archive, stats: Dict[str, Any]):
    """Render category browser."""
    st.markdown("### 📂 Browse by Category")
    
    categories = stats.get('categories', {})
    if not categories:
        st.info("No categories found in the archive")
        return
    
    # Category selection
    selected_category = st.selectbox(
        "Select category",
        list(categories.keys()),
        help="Browse insights by category"
    )
    
    if selected_category:
        # Get insights for category
        insights = archive.get_insights(
            limit=50,
            category=selected_category,
            min_quality=0.0
        )
        
        st.info(f"Found {len(insights)} insights in category '{selected_category}'")
        
        for i, insight in enumerate(insights):
            render_insight_card(insight, i)

def render_quality_rankings(archive):
    """Render quality rankings view."""
    st.markdown("### 🏆 Quality Rankings")
    
    # Get high-quality insights
    insights = archive.get_insights(
        limit=30,
        min_quality=0.4
    )
    
    if not insights:
        st.info("No high-quality insights found")
        return
    
    # Create quality chart
    quality_data = pd.DataFrame([
        {
            'Insight': f"Insight {i+1}",
            'Quality Score': insight.quality_score,
            'Confidence': insight.confidence_score,
            'Novelty': insight.novelty_score,
            'Utility': insight.utility_score,
            'Category': insight.category,
            'Summary': insight.summary[:50] + "..." if len(insight.summary) > 50 else insight.summary
        }
        for i, insight in enumerate(insights)
    ])
    
    # Quality distribution chart
    fig = px.scatter(
        quality_data,
        x='Confidence',
        y='Novelty',
        size='Utility',
        color='Quality Score',
        hover_data=['Summary'],
        title="Insight Quality Distribution",
        color_continuous_scale='viridis'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Top insights table
    st.markdown("#### 🥇 Top Quality Insights")
    display_df = quality_data[['Insight', 'Quality Score', 'Category', 'Summary']].head(10)
    st.dataframe(display_df, use_container_width=True)

def render_insight_card(insight, index: int, highlight_query: Optional[str] = None):
    """Render an individual insight card."""
    # Use container instead of expander to avoid nesting issues
    with st.container():
        # Add insight header with styling
        st.markdown(f"### 💡 Insight {index + 1}")
        st.markdown(f"**{insight.summary}**")
        st.markdown("---")

        # Metadata row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Quality", f"{insight.quality_score:.2f}")
        with col2:
            st.metric("Confidence", f"{insight.confidence_score:.2f}")
        with col3:
            st.metric("Sources", insight.source_count)
        with col4:
            st.caption(f"Category: {insight.category}")
        
        # Tags
        if insight.tags:
            st.markdown(f"**🏷️ Tags:** {', '.join(insight.tags)}")
        
        # Insight text
        st.markdown("**💭 Insight:**")
        insight_text = insight.synthesized_text
        
        # Highlight search terms if provided
        if highlight_query:
            # Simple highlighting (could be enhanced)
            for term in highlight_query.split():
                insight_text = insight_text.replace(
                    term, f"**{term}**"
                )
        
        st.markdown(insight_text)
        
        # Source documents
        if insight.source_documents:
            st.markdown("**📚 Source Documents:**")
            for doc in insight.source_documents[:5]:
                st.caption(f"• {doc}")
            st.markdown("")  # Add spacing
        
        # Metadata
        st.caption(f"Generated: {insight.generated_at[:10]} | Archived: {insight.archived_at[:10]}")

        # Add separator between insights
        st.markdown("---")
        st.markdown("")  # Add spacing

def render_archive_analytics(archive, stats: Dict[str, Any]):
    """Render detailed archive analytics."""
    st.markdown("---")
    st.markdown("### 📊 Archive Analytics")
    
    # Quality distribution
    insights = archive.get_insights(limit=1000, min_quality=0.0)
    
    if insights:
        # Create analytics dataframe
        analytics_df = pd.DataFrame([
            {
                'Quality Score': insight.quality_score,
                'Confidence': insight.confidence_score,
                'Novelty': insight.novelty_score,
                'Utility': insight.utility_score,
                'Category': insight.category,
                'Source Count': insight.source_count,
                'Generated Date': insight.generated_at[:10]
            }
            for insight in insights
        ])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Quality distribution histogram
            fig1 = px.histogram(
                analytics_df,
                x='Quality Score',
                nbins=20,
                title="Quality Score Distribution"
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Category distribution pie chart
            category_counts = analytics_df['Category'].value_counts()
            fig2 = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Insights by Category"
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Insights over time
        analytics_df['Generated Date'] = pd.to_datetime(analytics_df['Generated Date'])
        daily_counts = analytics_df.groupby(analytics_df['Generated Date'].dt.date).size()
        
        fig3 = px.line(
            x=daily_counts.index,
            y=daily_counts.values,
            title="Insights Generated Over Time"
        )
        st.plotly_chart(fig3, use_container_width=True)

def render_export_interface(archive):
    """Render export interface."""
    st.markdown("---")
    st.markdown("### 📤 Export Archive")
    
    col1, col2 = st.columns(2)
    
    with col1:
        export_format = st.selectbox(
            "Export format",
            ["JSON", "Markdown", "CSV"],
            help="Choose export format"
        )
        
        min_quality_export = st.slider(
            "Minimum quality for export",
            0.0, 1.0, 0.3, 0.1
        )
    
    with col2:
        category_filter = st.selectbox(
            "Category filter",
            ["All"] + list(archive.get_archive_stats().get('categories', {}).keys()),
            help="Filter by category"
        )
        
        max_insights = st.number_input(
            "Maximum insights",
            min_value=1,
            max_value=10000,
            value=100,
            help="Maximum number of insights to export"
        )
    
    if st.button("📥 Export Insights"):
        try:
            filters = {
                'min_quality': min_quality_export,
                'limit': max_insights,
                'category': None if category_filter == "All" else category_filter
            }
            
            output_path = archive.export_insights(
                format=export_format.lower(),
                filters=filters
            )
            
            st.success(f"✅ Exported insights to: {output_path}")
            
            # Provide download link
            with open(output_path, 'rb') as f:
                st.download_button(
                    label=f"📥 Download {export_format} Export",
                    data=f.read(),
                    file_name=output_path.split('/')[-1],
                    mime=f"application/{export_format.lower()}"
                )
                
        except Exception as e:
            st.error(f"❌ Export failed: {e}")
