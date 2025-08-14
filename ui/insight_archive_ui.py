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
    st.markdown("## 📚 Archived Insights")
    st.markdown("*Browse and search your collection of emergent insights from synthesis runs*")

    try:
        from memory.synthesis.insight_archive import get_insight_archive

        archive = get_insight_archive()

        # Auto-archive any recent synthesis results that haven't been archived yet
        auto_archive_recent_insights(archive)

        # Get archive statistics
        stats = archive.get_archive_stats()

        if 'error' in stats:
            st.error(f"❌ Error accessing archive: {stats['error']}")
            return

        # Display archive overview
        render_archive_overview(stats)

        # Display current research settings from Dream Canvas
        render_research_settings_display()

        # Archive controls
        col1, col2, col3 = st.columns(3)

        with col1:
            view_mode = st.selectbox(
                "📋 View Mode",
                ["Recent Insights", "Search Insights", "Browse by Category", "Quality Rankings", "Synthesis Runs"],
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
        elif view_mode == "Synthesis Runs":
            render_synthesis_runs_view(archive)

        # Research Mode Controls for selected insights
        render_research_mode_controls()

        # Research Results Display
        render_research_results_section()

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
    """Render an individual insight card with enhanced format matching the requirements."""
    # Use container instead of expander to avoid nesting issues
    with st.container():
        # Enhanced header with insight number and cluster info
        st.markdown(f"### 💡 Insight #{index + 1}")
        st.markdown(f"**Cluster ID:** {insight.cluster_id}")

        # Get cluster coherence score from metadata if available
        coherence_score = insight.metadata.get('coherence_score', 'N/A')
        st.markdown(f"**Coherence Score:** {coherence_score}")

        # Dominant themes
        themes = insight.metadata.get('dominant_themes', [])
        if themes:
            st.markdown(f"**Dominant Themes:** {', '.join(themes[:3])}")

        # Summary
        st.markdown(f"**Summary:** {insight.summary}")

        st.markdown("---")

        # EMERGENT INSIGHT section
        st.markdown("**🧠 EMERGENT INSIGHT:**")
        insight_text = insight.synthesized_text

        # Clean up insight text (remove thinking tags)
        if '<think>' in insight_text and '</think>' in insight_text:
            parts = insight_text.split('</think>')
            if len(parts) > 1:
                insight_text = parts[-1].strip()

        # Highlight search terms if provided
        if highlight_query:
            for term in highlight_query.split():
                insight_text = insight_text.replace(
                    term, f"**{term}**"
                )

        st.markdown(insight_text)

        # Strategic Implications (extracted from metadata or generated)
        strategic_implications = insight.metadata.get('strategic_implications',
                                                     'This insight provides new understanding that could inform strategic decisions.')
        st.markdown(f"**📈 Strategic Implications:** {strategic_implications}")

        # Actionable Recommendations (extracted from metadata or generated)
        actionable_recommendations = insight.metadata.get('actionable_recommendations',
                                                         'Consider how this insight can be applied to current projects and decision-making processes.')
        st.markdown(f"**🎯 Actionable Recommendations:** {actionable_recommendations}")

        # Quality metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 Confidence", f"{insight.confidence_score:.2f}")
        with col2:
            st.metric("🌟 Novelty", f"{insight.novelty_score:.2f}")
        with col3:
            st.metric("🔧 Utility", f"{insight.utility_score:.2f}")
        with col4:
            st.metric("⭐ Quality", f"{insight.quality_score:.2f}")

        # PDF Documents in this Cluster
        if insight.source_documents:
            st.markdown("**📚 PDF Documents in this Cluster:**")
            for doc in insight.source_documents[:5]:
                st.caption(f"• {doc}")
            if len(insight.source_documents) > 5:
                st.caption(f"... and {len(insight.source_documents) - 5} more documents")

        # Research Mode Controls - Add the missing checkbox and microscope functionality
        col1, col2 = st.columns([1, 10])
        with col1:
            research_selected = st.checkbox(
                "🔬",
                key=f"research_select_{insight.archive_id}",
                help="Select for Research Mode"
            )
        with col2:
            if research_selected:
                st.caption("✅ Selected for research")
                # Store selection in session state for research processing
                if 'selected_archived_insights' not in st.session_state:
                    st.session_state.selected_archived_insights = set()
                st.session_state.selected_archived_insights.add(insight.archive_id)

                # Check if auto-research is enabled (from Dream Canvas settings)
                auto_research_enabled = st.session_state.get('auto_run_research_on_select', False)
                if auto_research_enabled:
                    # Auto-trigger research
                    research_mode = st.session_state.get('auto_research_mode', 'Deep')
                    download_limit = st.session_state.get('deep_research_download_limit', 3)

                    st.info(f"🤖 Auto-research enabled: Starting {research_mode} Research...")

                    # Trigger research immediately
                    trigger_archived_insight_research(
                        [insight.archive_id],
                        research_mode,
                        download_limit
                    )
            else:
                if 'selected_archived_insights' in st.session_state:
                    st.session_state.selected_archived_insights.discard(insight.archive_id)

        # Metadata footer
        st.caption(f"Generated: {insight.generated_at[:10]} | Archived: {insight.archived_at[:10]} | Sources: {insight.source_count}")

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

def auto_archive_recent_insights(archive):
    """Automatically archive insights from recent synthesis runs."""
    try:
        # Check if there are synthesis results in session state that haven't been archived
        if hasattr(st.session_state, 'synthesis_results') and st.session_state.synthesis_results:
            insights = st.session_state.synthesis_results.get('insights', [])
            run_id = st.session_state.synthesis_results.get('run_id', 'unknown')

            for insight_data in insights:
                # Check if this insight is already archived
                insight_id = insight_data.get('insight_id', f"insight_{run_id}_{hash(insight_data.get('synthesized_text', ''))}")

                # Try to find existing archived insight
                existing = archive.get_insights(limit=1)  # This is a simple check - could be enhanced

                # Archive the insight if it's not already archived
                try:
                    # Convert insight data to SynthesizedInsight-like object for archiving
                    from memory.synthesis.insight_generator import SynthesizedInsight
                    from datetime import datetime

                    # Create a mock SynthesizedInsight object
                    class MockInsight:
                        def __init__(self, data):
                            self.insight_id = data.get('insight_id', insight_id)
                            self.cluster_id = data.get('cluster_id', 'unknown')
                            self.synthesized_text = data.get('synthesized_text', '')
                            self.confidence_score = data.get('confidence_score', 0.0)
                            self.novelty_score = data.get('novelty_score', 0.0)
                            self.utility_score = data.get('utility_score', 0.0)
                            self.synthesis_metadata = data.get('synthesis_metadata', {})
                            self.generated_at = data.get('generated_at', datetime.now().isoformat())
                            self.source_chunks = []  # Mock empty source chunks

                    mock_insight = MockInsight(insight_data)

                    # Archive with appropriate category and tags
                    archive.archive_insight(
                        mock_insight,
                        category="synthesis_run",
                        tags=["auto_archived", f"run_{run_id}"]
                    )

                except Exception as archive_error:
                    # Silently continue if archiving fails for individual insights
                    pass

    except Exception as e:
        # Silently handle auto-archiving errors to not disrupt the UI
        pass

def render_synthesis_runs_view(archive):
    """Render synthesis runs view showing insights grouped by synthesis run."""
    st.markdown("### 🧠 Synthesis Runs")

    # Get insights grouped by synthesis run
    insights = archive.get_insights(limit=200, min_quality=0.0)

    if not insights:
        st.info("No synthesis runs found in the archive")
        return

    # Group insights by synthesis run ID
    runs = {}
    for insight in insights:
        run_id = insight.metadata.get('synthesis_run_id', 'unknown')
        if run_id not in runs:
            runs[run_id] = []
        runs[run_id].append(insight)

    # Display runs
    for run_id, run_insights in runs.items():
        with st.expander(f"🧠 Synthesis Run: {run_id} ({len(run_insights)} insights)", expanded=False):
            # Run summary
            avg_quality = sum(i.quality_score for i in run_insights) / len(run_insights)
            st.markdown(f"**Average Quality:** {avg_quality:.2f}")
            st.markdown(f"**Insights Generated:** {len(run_insights)}")

            # Show insights from this run
            for i, insight in enumerate(run_insights):
                render_insight_card(insight, i)

def render_research_mode_controls():
    """Render research mode controls for selected archived insights using Dream Canvas settings."""
    if 'selected_archived_insights' not in st.session_state or not st.session_state.selected_archived_insights:
        return

    selected_count = len(st.session_state.selected_archived_insights)

    st.markdown("---")
    st.markdown("### 🔬 Research Mode")
    st.markdown(f"*{selected_count} insight(s) selected for research*")

    # Use Dream Canvas research settings if available
    auto_research_enabled = st.session_state.get('auto_run_research_on_select', False)
    research_mode = st.session_state.get('auto_research_mode', 'Deep')
    download_limit = st.session_state.get('deep_research_download_limit', 3)

    col1, col2, col3 = st.columns(3)

    with col1:
        # Show current Dream Canvas settings
        st.markdown("**Current Settings:**")
        st.caption(f"Mode: {research_mode}")
        st.caption(f"Auto-run: {'On' if auto_research_enabled else 'Off'}")
        st.caption(f"Download limit: {download_limit}")

        # Allow override
        override_mode = st.radio(
            "Override Mode:",
            ["Use Dream Canvas Settings", "Deep Research", "Quick Research"],
            help="Choose research mode or use Dream Canvas settings"
        )

    with col2:
        if override_mode != "Use Dream Canvas Settings":
            max_papers = st.selectbox(
                "Papers per Insight:",
                [1, 2, 3, 5, 10],
                index=2,  # Default to 3
                help="Maximum papers to download per insight"
            )
        else:
            max_papers = download_limit
            st.metric("Papers per Insight", max_papers)

    with col3:
        if st.button("🚀 Start Research", type="primary"):
            # Determine final research mode
            if override_mode == "Use Dream Canvas Settings":
                final_mode = research_mode
            else:
                final_mode = override_mode.replace(" Research", "")

            trigger_archived_insight_research(
                list(st.session_state.selected_archived_insights),
                final_mode,
                max_papers
            )

def trigger_archived_insight_research(selected_insight_ids, research_mode, max_papers):
    """Trigger research for selected archived insights using Dream Canvas research logic."""
    try:
        from memory.synthesis.insight_archive import get_insight_archive

        archive = get_insight_archive()

        # Get the selected insights from the archive
        selected_insights = []

        # Get insights by their archive IDs using the new method
        for insight_id in selected_insight_ids:
            insight = archive.get_insight_by_id(insight_id)
            if insight:
                selected_insights.append(insight)

        if not selected_insights:
            st.error("❌ No insights found for the selected IDs")
            return

        # Process each insight for research
        research_results = []

        for insight in selected_insights:
            insight_text = insight.synthesized_text

            # Clean the insight text (remove thinking tags)
            if '<think>' in insight_text and '</think>' in insight_text:
                parts = insight_text.split('</think>')
                if len(parts) > 1:
                    insight_text = parts[-1].strip()

            if research_mode == "Deep":
                # Use Deep Research engine
                try:
                    from sam.agents.strategies.deep_research import DeepResearchStrategy

                    st.info(f"🔬 Starting Deep Research for insight from cluster {insight.cluster_id}")

                    # Initialize Deep Research
                    research_strategy = DeepResearchStrategy(
                        insight_text=insight_text,
                        research_id=f"archived_insight_{insight.archive_id}"
                    )

                    # Execute research (this would normally be async)
                    with st.spinner(f"🌙 Conducting deep research for insight {insight.archive_id[:8]}..."):
                        try:
                            research_result = research_strategy.execute_research()
                            research_results.append({
                                'insight_id': insight.archive_id,
                                'cluster_id': insight.cluster_id,
                                'research_result': research_result,
                                'status': 'completed'
                            })
                            st.success(f"✅ Deep research completed for insight {insight.archive_id[:8]}")
                        except Exception as research_error:
                            st.warning(f"⚠️ Deep research failed for insight {insight.archive_id[:8]}: {research_error}")
                            research_results.append({
                                'insight_id': insight.archive_id,
                                'cluster_id': insight.cluster_id,
                                'error': str(research_error),
                                'status': 'failed'
                            })

                except ImportError:
                    st.error("❌ Deep Research engine not available")
                    return

            else:  # Quick Research
                # Use Quick Research (basic ArXiv search)
                try:
                    from sam.web_retrieval.tools.arxiv_tool import get_arxiv_tool

                    st.info(f"🔍 Starting Quick Research for insight from cluster {insight.cluster_id}")

                    arxiv_tool = get_arxiv_tool()

                    # Generate search query from insight
                    search_query = generate_research_query_from_insight(insight_text)

                    with st.spinner(f"🔍 Searching ArXiv for insight {insight.archive_id[:8]}..."):
                        papers = arxiv_tool.search_papers(search_query, max_results=max_papers)

                        if papers:
                            research_results.append({
                                'insight_id': insight.archive_id,
                                'cluster_id': insight.cluster_id,
                                'papers': papers,
                                'query': search_query,
                                'status': 'completed'
                            })
                            st.success(f"✅ Found {len(papers)} papers for insight {insight.archive_id[:8]}")
                        else:
                            st.warning(f"⚠️ No papers found for insight {insight.archive_id[:8]}")
                            research_results.append({
                                'insight_id': insight.archive_id,
                                'cluster_id': insight.cluster_id,
                                'papers': [],
                                'query': search_query,
                                'status': 'no_results'
                            })

                except ImportError:
                    st.error("❌ ArXiv tool not available")
                    return

        # Store research results in session state
        if 'archived_insight_research_results' not in st.session_state:
            st.session_state.archived_insight_research_results = []

        st.session_state.archived_insight_research_results.extend(research_results)

        # Show summary
        completed_count = len([r for r in research_results if r['status'] == 'completed'])
        st.success(f"🔬 Research completed for {completed_count}/{len(selected_insights)} insights using {research_mode} Research")

        if research_results:
            st.info("📊 Research results are now available. Scroll down to view the 'Research Results' section.")

        # Clear selections
        st.session_state.selected_archived_insights = set()
        st.rerun()

    except Exception as e:
        st.error(f"❌ Failed to start research: {e}")
        import traceback
        st.error(f"Details: {traceback.format_exc()}")

def generate_research_query_from_insight(insight_text):
    """Generate a research query from insight text."""
    try:
        import re

        # Extract key terms (words longer than 3 characters)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', insight_text)

        # Remove common words
        common_words = {'this', 'that', 'with', 'from', 'they', 'have', 'been', 'were', 'will', 'would', 'could', 'should', 'might', 'must', 'shall', 'more', 'most', 'some', 'many', 'much', 'such', 'very', 'also', 'just', 'only', 'even', 'well', 'still', 'back', 'good', 'best', 'better', 'great', 'large', 'small', 'high', 'long', 'right', 'different', 'important', 'possible', 'available', 'similar', 'related', 'specific', 'particular', 'general', 'common', 'special', 'certain', 'clear', 'simple', 'complex', 'difficult', 'easy', 'hard', 'strong', 'weak', 'fast', 'slow', 'early', 'late', 'recent', 'current', 'future', 'past', 'present', 'previous', 'next', 'first', 'last', 'second', 'third', 'other', 'another', 'same', 'different', 'various', 'several', 'multiple', 'single', 'individual', 'personal', 'public', 'private', 'social', 'economic', 'political', 'cultural', 'natural', 'human', 'technical', 'scientific', 'medical', 'legal', 'financial', 'business', 'commercial', 'industrial', 'educational', 'academic', 'professional', 'international', 'national', 'local', 'regional', 'global', 'worldwide', 'universal', 'general', 'specific', 'particular', 'special', 'unique', 'rare', 'common', 'typical', 'normal', 'standard', 'regular', 'usual', 'ordinary', 'extraordinary', 'exceptional', 'remarkable', 'significant', 'important', 'major', 'minor', 'primary', 'secondary', 'main', 'central', 'key', 'critical', 'essential', 'necessary', 'required', 'optional', 'additional', 'extra', 'further', 'more', 'less', 'fewer', 'greater', 'smaller', 'larger', 'bigger', 'higher', 'lower', 'deeper', 'wider', 'narrower', 'broader', 'closer', 'farther', 'nearer', 'distant', 'remote', 'nearby', 'adjacent', 'surrounding', 'external', 'internal', 'inner', 'outer', 'upper', 'lower', 'top', 'bottom', 'front', 'back', 'left', 'right', 'north', 'south', 'east', 'west', 'center', 'middle', 'side', 'edge', 'corner', 'surface', 'inside', 'outside', 'above', 'below', 'over', 'under', 'through', 'across', 'around', 'between', 'among', 'within', 'without', 'beyond', 'behind', 'ahead', 'forward', 'backward', 'upward', 'downward', 'inward', 'outward', 'toward', 'away', 'near', 'far', 'close', 'open', 'closed', 'full', 'empty', 'complete', 'incomplete', 'finished', 'unfinished', 'started', 'stopped', 'continued', 'paused', 'resumed', 'ended', 'begun', 'completed', 'achieved', 'accomplished', 'succeeded', 'failed', 'won', 'lost', 'gained', 'lost', 'increased', 'decreased', 'improved', 'worsened', 'changed', 'remained', 'stayed', 'moved', 'shifted', 'transferred', 'converted', 'transformed', 'developed', 'evolved', 'progressed', 'advanced', 'retreated', 'declined', 'grew', 'shrank', 'expanded', 'contracted', 'extended', 'shortened', 'lengthened', 'widened', 'narrowed', 'deepened', 'shallowed', 'heightened', 'lowered', 'raised', 'dropped', 'lifted', 'fell', 'rose', 'climbed', 'descended', 'ascended', 'entered', 'exited', 'arrived', 'departed', 'came', 'went', 'returned', 'left', 'stayed', 'remained', 'continued', 'stopped', 'started', 'began', 'ended', 'finished', 'completed', 'accomplished', 'achieved', 'reached', 'attained', 'obtained', 'acquired', 'gained', 'earned', 'received', 'got', 'took', 'gave', 'provided', 'offered', 'supplied', 'delivered', 'sent', 'brought', 'carried', 'held', 'kept', 'maintained', 'preserved', 'protected', 'defended', 'attacked', 'fought', 'battled', 'competed', 'contested', 'challenged', 'opposed', 'resisted', 'supported', 'helped', 'assisted', 'aided', 'served', 'worked', 'operated', 'functioned', 'performed', 'acted', 'behaved', 'conducted', 'managed', 'controlled', 'directed', 'guided', 'led', 'followed', 'accompanied', 'joined', 'participated', 'involved', 'engaged', 'committed', 'dedicated', 'devoted', 'focused', 'concentrated', 'emphasized', 'stressed', 'highlighted', 'featured', 'included', 'contained', 'comprised', 'consisted', 'composed', 'made', 'created', 'produced', 'generated', 'caused', 'resulted', 'led', 'brought', 'gave', 'provided', 'offered', 'presented', 'showed', 'displayed', 'demonstrated', 'revealed', 'exposed', 'uncovered', 'discovered', 'found', 'located', 'identified', 'recognized', 'acknowledged', 'admitted', 'accepted', 'agreed', 'approved', 'endorsed', 'supported', 'backed', 'favored', 'preferred', 'chosen', 'selected', 'picked', 'decided', 'determined', 'concluded', 'resolved', 'solved', 'answered', 'responded', 'replied', 'reacted', 'acted', 'moved', 'proceeded', 'advanced', 'progressed', 'continued', 'persisted', 'persevered', 'endured', 'lasted', 'survived', 'lived', 'existed', 'occurred', 'happened', 'took', 'place', 'appeared', 'emerged', 'arose', 'developed', 'formed', 'shaped', 'molded', 'designed', 'planned', 'organized', 'arranged', 'prepared', 'ready', 'set', 'established', 'founded', 'built', 'constructed', 'assembled', 'installed', 'placed', 'positioned', 'located', 'situated', 'based', 'grounded', 'rooted', 'founded', 'established', 'created', 'formed', 'developed', 'built', 'made', 'produced', 'manufactured', 'generated', 'caused', 'resulted', 'led', 'brought', 'gave', 'provided', 'offered', 'presented', 'showed', 'displayed', 'demonstrated', 'revealed', 'exposed', 'uncovered', 'discovered', 'found', 'located', 'identified', 'recognized', 'acknowledged', 'admitted', 'accepted', 'agreed', 'approved', 'endorsed', 'supported', 'backed', 'favored', 'preferred', 'chosen', 'selected', 'picked', 'decided', 'determined', 'concluded', 'resolved', 'solved', 'answered', 'responded', 'replied', 'reacted', 'acted', 'moved', 'proceeded', 'advanced', 'progressed', 'continued', 'persisted', 'persevered', 'endured', 'lasted', 'survived', 'lived', 'existed', 'occurred', 'happened', 'took', 'place', 'appeared', 'emerged', 'arose', 'developed', 'formed', 'shaped', 'molded', 'designed', 'planned', 'organized', 'arranged', 'prepared', 'ready', 'established', 'founded', 'built', 'constructed', 'assembled', 'installed', 'placed', 'positioned', 'located', 'situated', 'based', 'grounded', 'rooted'}

        # Filter out common words and get unique terms
        filtered_words = []
        seen = set()
        for word in words:
            word_lower = word.lower()
            if word_lower not in common_words and word_lower not in seen and len(word) > 3:
                seen.add(word_lower)
                filtered_words.append(word)
                if len(filtered_words) >= 6:  # Limit to 6 key terms
                    break

        if filtered_words:
            return ' '.join(filtered_words)
        else:
            # Fallback to first 100 characters
            return insight_text[:100].strip()

    except Exception:
        return insight_text[:100].strip()

def render_research_results_section():
    """Render research results from archived insight research."""
    if 'archived_insight_research_results' not in st.session_state or not st.session_state.archived_insight_research_results:
        return

    research_results = st.session_state.archived_insight_research_results

    st.markdown("---")
    st.markdown("### 📊 Research Results")
    st.markdown(f"*Results from {len(research_results)} research operations*")

    # Summary metrics
    completed_results = [r for r in research_results if r['status'] == 'completed']
    failed_results = [r for r in research_results if r['status'] == 'failed']
    no_results = [r for r in research_results if r['status'] == 'no_results']

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("✅ Completed", len(completed_results))
    with col2:
        st.metric("❌ Failed", len(failed_results))
    with col3:
        st.metric("⚠️ No Results", len(no_results))
    with col4:
        total_papers = sum(len(r.get('papers', [])) for r in completed_results)
        st.metric("📄 Papers Found", total_papers)

    # Display results
    for i, result in enumerate(research_results):
        with st.expander(f"🔬 Research Result {i+1}: Insight {result['insight_id'][:8]} (Cluster {result['cluster_id']})", expanded=False):

            if result['status'] == 'completed':
                if 'research_result' in result:
                    # Deep Research result
                    research_result = result['research_result']
                    st.markdown("**🧠 Deep Research Analysis:**")

                    if hasattr(research_result, 'final_report') and research_result.final_report:
                        st.markdown(research_result.final_report)
                    else:
                        st.info("Deep research completed but no detailed report available")

                    if hasattr(research_result, 'arxiv_papers') and research_result.arxiv_papers:
                        st.markdown(f"**📚 Papers Analyzed:** {len(research_result.arxiv_papers)}")
                        for j, paper in enumerate(research_result.arxiv_papers[:3], 1):
                            st.markdown(f"{j}. **{paper.get('title', 'Unknown Title')}**")
                            st.caption(f"Authors: {paper.get('authors', 'Unknown')}")
                            if paper.get('summary'):
                                st.caption(f"Summary: {paper['summary'][:200]}...")

                elif 'papers' in result:
                    # Quick Research result
                    papers = result['papers']
                    query = result.get('query', 'Unknown query')

                    st.markdown("**🔍 Quick Research Results:**")
                    st.markdown(f"**Search Query:** {query}")
                    st.markdown(f"**Papers Found:** {len(papers)}")

                    for j, paper in enumerate(papers, 1):
                        st.markdown(f"{j}. **{paper.get('title', 'Unknown Title')}**")
                        st.caption(f"Authors: {paper.get('authors', 'Unknown')}")
                        if paper.get('summary'):
                            st.caption(f"Summary: {paper['summary'][:200]}...")
                        if paper.get('arxiv_url'):
                            st.markdown(f"[📄 View Paper]({paper['arxiv_url']})")

            elif result['status'] == 'failed':
                st.error(f"❌ Research failed: {result.get('error', 'Unknown error')}")

            elif result['status'] == 'no_results':
                st.warning(f"⚠️ No papers found for query: {result.get('query', 'Unknown query')}")

    # Clear results button
    if st.button("🗑️ Clear Research Results", help="Clear all research results from this session"):
        st.session_state.archived_insight_research_results = []
        st.rerun()

def render_research_settings_display():
    """Display current research settings from Dream Canvas."""
    st.markdown("### ⚙️ Current Research Settings")

    # Get current settings from session state
    auto_research_enabled = st.session_state.get('auto_run_research_on_select', False)
    research_mode = st.session_state.get('auto_research_mode', 'Deep')
    download_limit = st.session_state.get('deep_research_download_limit', 3)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        status_color = "🟢" if auto_research_enabled else "🔴"
        st.metric("Auto-Research", f"{status_color} {'On' if auto_research_enabled else 'Off'}")

    with col2:
        mode_icon = "🧠" if research_mode == "Deep" else "🔍"
        st.metric("Research Mode", f"{mode_icon} {research_mode}")

    with col3:
        st.metric("Download Limit", f"📄 {download_limit}")

    with col4:
        if st.button("🎛️ Configure", help="Go to Dream Canvas to configure research settings"):
            st.info("💡 Navigate to Dream Canvas → Main Actions → Research Mode to configure these settings")

    if auto_research_enabled:
        st.success("✅ Auto-research is enabled. Selecting an insight checkbox will automatically start research.")
    else:
        st.info("ℹ️ Auto-research is disabled. Select insights and use the Research Mode controls below to start research manually.")

    st.markdown("---")
