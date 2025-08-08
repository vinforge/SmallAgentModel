"""
Interactive Memory Browser UI for SAM
Visual interface for memory search, inspection, and management.

Sprint 12 Task 1: Interactive Memory Browser UI
"""

import logging
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.memory_vectorstore import MemoryVectorStore, MemoryType, get_memory_store
from memory.memory_reasoning import MemoryDrivenReasoningEngine, get_memory_reasoning_engine

logger = logging.getLogger(__name__)

class MemoryBrowserUI:
    """
    Interactive memory browser with search, filtering, and visualization.
    """
    
    def __init__(self):
        """Initialize the memory browser UI."""
        self.memory_store = get_memory_store()
        self.memory_reasoning = get_memory_reasoning_engine()
        
        # UI state
        if 'selected_memory' not in st.session_state:
            st.session_state.selected_memory = None
        if 'search_results' not in st.session_state:
            st.session_state.search_results = []
        if 'filter_settings' not in st.session_state:
            st.session_state.filter_settings = {
                'memory_types': [],
                'tags': [],
                'date_range': None,
                'importance_range': [0.0, 1.0],
                'user_filter': None
            }
    
    def render(self):
        """Render the complete memory browser interface."""
        st.title("🧠 SAM Memory Browser")
        st.markdown("Search, inspect, and manage SAM's long-term memory")
        
        # Sidebar for filters and controls
        self._render_sidebar()
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._render_search_interface()
            self._render_memory_list()
        
        with col2:
            self._render_memory_details()
            self._render_memory_stats()
    
    def _render_sidebar(self):
        """Render the sidebar with filters and controls."""
        with st.sidebar:
            st.header("🔍 Search & Filters")

            # Phase 3.2.3: Enhanced Search Mode Selection
            st.subheader("🎯 Search Mode")
            search_mode = st.selectbox(
                "Search Strategy",
                options=["Enhanced Hybrid", "Semantic Only", "Keyword Only", "Recent First"],
                index=0,
                help="Choose search strategy for optimal results"
            )
            st.session_state.filter_settings['search_mode'] = search_mode

            # Phase 3.2.3: Source-Specific Filtering
            st.subheader("📚 Source Filters")

            # Document source filter
            available_sources = self._get_available_sources()
            selected_sources = st.multiselect(
                "Document Sources",
                options=available_sources,
                default=st.session_state.filter_settings.get('selected_sources', []),
                help="Filter by specific document sources"
            )
            st.session_state.filter_settings['selected_sources'] = selected_sources

            # Source type filter
            source_types = st.multiselect(
                "Source Types",
                options=["PDF Documents", "Web Pages", "Conversations", "System Logs", "User Notes"],
                default=st.session_state.filter_settings.get('source_types', []),
                help="Filter by source type"
            )
            st.session_state.filter_settings['source_types'] = source_types

            # Memory type filter
            memory_types = st.multiselect(
                "Memory Types",
                options=[mt.value for mt in MemoryType],
                default=st.session_state.filter_settings['memory_types'],
                help="Filter by memory type"
            )
            st.session_state.filter_settings['memory_types'] = memory_types
            
            # Phase 3.2.3: Enhanced Date and Quality Filters
            st.subheader("📅 Date & Quality")

            # Date range filter with presets
            date_filter = st.selectbox(
                "Time Period",
                options=["All Time", "Last Hour", "Last 24 Hours", "Last Week", "Last Month", "Last 3 Months", "Custom Range"],
                index=0
            )

            if date_filter == "Custom Range":
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start Date")
                with col2:
                    end_date = st.date_input("End Date")
                st.session_state.filter_settings['date_range'] = (start_date, end_date)
            elif date_filter != "All Time":
                hours_back = {"Last Hour": 1/24, "Last 24 Hours": 1, "Last Week": 7, "Last Month": 30, "Last 3 Months": 90}[date_filter]
                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=hours_back)
                st.session_state.filter_settings['date_range'] = (start_date, end_date)
            else:
                st.session_state.filter_settings['date_range'] = None

            # Phase 3.2.3: Confidence and Ranking Filters
            st.subheader("🎯 Quality Filters")

            # Confidence score filter
            confidence_range = st.slider(
                "Confidence Score",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.filter_settings.get('confidence_range', [0.0, 1.0]),
                step=0.05,
                help="Filter by memory confidence/quality score"
            )
            st.session_state.filter_settings['confidence_range'] = confidence_range

            # Ranking score filter (for enhanced search results)
            ranking_threshold = st.slider(
                "Ranking Threshold",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.filter_settings.get('ranking_threshold', 0.1),
                step=0.05,
                help="Minimum ranking score for enhanced search results"
            )
            st.session_state.filter_settings['ranking_threshold'] = ranking_threshold
            
            # Importance filter
            st.subheader("⭐ Importance")
            importance_range = st.slider(
                "Importance Score",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.filter_settings['importance_range'],
                step=0.1,
                help="Filter by importance score"
            )
            st.session_state.filter_settings['importance_range'] = importance_range
            
            # User filter
            st.subheader("👤 User Filter")
            user_filter = st.text_input(
                "User ID",
                value=st.session_state.filter_settings['user_filter'] or "",
                help="Filter memories by user ID"
            )
            st.session_state.filter_settings['user_filter'] = user_filter if user_filter else None
            
            # Phase 3.2.3: Advanced Filter Controls
            st.subheader("🎛️ Filter Controls")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear Filters", key="clear_filters_button"):
                    st.session_state.filter_settings = {
                        'memory_types': [],
                        'tags': [],
                        'date_range': None,
                        'importance_range': [0.0, 1.0],
                        'user_filter': None,
                        'search_mode': 'Enhanced Hybrid',
                        'selected_sources': [],
                        'source_types': [],
                        'confidence_range': [0.0, 1.0],
                        'ranking_threshold': 0.1
                    }
                    st.rerun()

            with col2:
                if st.button("💾 Save Preset", key="save_preset_button"):
                    self._save_filter_preset()

            # Filter presets
            saved_presets = self._get_saved_presets()
            if saved_presets:
                selected_preset = st.selectbox(
                    "Load Preset",
                    options=["None"] + list(saved_presets.keys()),
                    index=0
                )
                if selected_preset != "None" and st.button("📂 Load"):
                    st.session_state.filter_settings.update(saved_presets[selected_preset])
                    st.rerun()
    
    def _render_search_interface(self):
        """Render the enhanced search interface with real-time capabilities."""
        st.subheader("🔍 Enhanced Memory Search")

        # Phase 3.2.3: Real-time search with advanced options
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            search_query = st.text_input(
                "Search memories",
                placeholder="Enter keywords, topics, or questions...",
                help="Search through memory content using semantic similarity",
                key="main_search"
            )

        with col2:
            real_time_search = st.checkbox("🔄 Real-time", value=False, help="Search as you type")

        with col3:
            search_button = st.button("🔍 Search", type="primary", key="main_search_button")

        # Phase 3.2.3: Source-specific search
        st.markdown("**🎯 Source-Specific Search:**")
        col1, col2 = st.columns(2)

        with col1:
            source_query = st.text_input(
                "Search within specific source",
                placeholder="Enter source name or pattern...",
                help="Search within a specific document or source"
            )

        with col2:
            if st.button("📚 Source Search") and source_query:
                with st.spinner("Searching within sources..."):
                    results = self._search_within_sources(source_query, search_query or "")
                    st.session_state.search_results = results
                    st.success(f"Found {len(results)} memories in matching sources")

        # Perform main search
        if (search_button and search_query) or (real_time_search and search_query and len(search_query) > 2):
            with st.spinner("Searching memories..."):
                results = self._enhanced_search_memories(search_query)
                st.session_state.search_results = results
                if not real_time_search:  # Only show success message for manual search
                    st.success(f"Found {len(results)} memories")

        # Phase 3.2.3: Enhanced quick searches with categories
        st.markdown("**⚡ Quick Searches:**")

        # Categorized quick searches
        search_categories = {
            "📄 Documents": ["Recent uploads", "PDF documents", "Important documents"],
            "💬 Conversations": ["Recent chats", "User questions", "System responses"],
            "🔬 Research": ["AI research", "Technical insights", "Learning notes"],
            "📊 Analytics": ["Performance data", "Usage statistics", "Error logs"]
        }

        for category, searches in search_categories.items():
            with st.expander(category):
                cols = st.columns(len(searches))
                for i, quick_search in enumerate(searches):
                    with cols[i]:
                        if st.button(quick_search, key=f"quick_{category}_{i}"):
                            results = self._enhanced_search_memories(quick_search)
                            st.session_state.search_results = results
    
    def _render_memory_list(self):
        """Render the list of memories."""
        st.subheader("📚 Memory Results")
        
        # Get memories to display
        if st.session_state.search_results:
            memories_to_show = st.session_state.search_results
        else:
            memories_to_show = self._get_recent_memories()
        
        if not memories_to_show:
            st.info("No memories found. Try adjusting your search or filters.")
            return
        
        # Phase 3.2.3: Enhanced memory cards with ranking information
        for i, memory_result in enumerate(memories_to_show):
            # Handle both enhanced and legacy result formats
            if hasattr(memory_result, 'content') and hasattr(memory_result, 'final_score'):
                # Enhanced RankedMemoryResult
                memory = memory_result
                content = memory_result.content
                similarity = getattr(memory_result, 'final_score', 1.0)
                is_enhanced = True
                source_name = memory_result.metadata.get('source_name', 'Unknown')
                confidence = memory_result.confidence_score
            elif hasattr(memory_result, 'chunk'):
                # Legacy MemorySearchResult
                memory = memory_result.chunk
                content = memory.content
                similarity = getattr(memory_result, 'similarity_score', 1.0)
                is_enhanced = False
                source_name = memory.source
                confidence = memory.importance_score
            else:
                # Direct memory object
                memory = memory_result
                content = memory.content
                similarity = 1.0
                is_enhanced = False
                source_name = memory.source
                confidence = memory.importance_score

            with st.container():
                # Phase 3.2.3: Enhanced memory card with ranking indicators
                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    # Enhanced header with ranking badge
                    if is_enhanced:
                        ranking_badge = "🏆" if similarity > 0.7 else "⭐" if similarity > 0.5 else "📄"
                        st.markdown(f"{ranking_badge} **{memory.memory_type.value.title() if hasattr(memory, 'memory_type') else 'Enhanced'}** - {source_name}")
                    else:
                        st.markdown(f"**{memory.memory_type.value.title()}** - {source_name}")

                    # Content preview with highlighting
                    content_preview = content[:150]
                    if len(content) > 150:
                        content_preview += "..."
                    st.markdown(content_preview)

                    # Enhanced tags and metadata
                    if hasattr(memory, 'tags') and memory.tags:
                        tag_html = " ".join([f"<span style='background-color: #e1f5fe; padding: 2px 6px; border-radius: 3px; font-size: 0.8em;'>{tag}</span>" for tag in memory.tags[:5]])
                        st.markdown(tag_html, unsafe_allow_html=True)

                    # Phase 3.2.3: Show enhanced metadata for ranked results
                    if is_enhanced and hasattr(memory_result, 'metadata'):
                        metadata = memory_result.metadata
                        if metadata.get('page_number') or metadata.get('chunk_index'):
                            location_info = []
                            if metadata.get('page_number'):
                                location_info.append(f"p.{metadata['page_number']}")
                            if metadata.get('chunk_index'):
                                location_info.append(f"chunk {metadata['chunk_index']}")
                            st.caption(f"📍 {', '.join(location_info)}")

                with col2:
                    # Enhanced metrics with ranking information
                    if is_enhanced:
                        st.metric("Final Score", f"{similarity:.3f}")
                        st.metric("Confidence", f"{confidence:.2f}")
                        if hasattr(memory_result, 'semantic_score'):
                            st.caption(f"Semantic: {memory_result.semantic_score:.2f}")
                    else:
                        st.metric("Importance", f"{confidence:.2f}")
                        if similarity < 1.0:
                            st.metric("Similarity", f"{similarity:.2f}")

                    if hasattr(memory, 'access_count'):
                        st.caption(f"Access: {memory.access_count}")

                with col3:
                    # Actions with enhanced options
                    if hasattr(memory, 'timestamp'):
                        st.caption(memory.timestamp[:10])

                    memory_id = getattr(memory, 'chunk_id', f"mem_{i}")

                    if st.button("👁️ View", key=f"view_{memory_id}"):
                        st.session_state.selected_memory = memory

                    if st.button("✏️ Edit", key=f"edit_{memory_id}"):
                        self._show_edit_dialog(memory)

                    # Phase 3.2.3: Enhanced actions
                    if is_enhanced and st.button("🔍 Similar", key=f"similar_{memory_id}"):
                        similar_results = self._find_similar_memories(content)
                        st.session_state.search_results = similar_results
                        st.rerun()

                st.divider()
    
    def _render_memory_details(self):
        """Render detailed view of selected memory."""
        st.subheader("📄 Memory Details")
        
        if st.session_state.selected_memory is None:
            st.info("Select a memory from the list to view details")
            return
        
        memory = st.session_state.selected_memory
        
        # Memory header
        st.markdown(f"### {memory.memory_type.value.title()}")
        st.markdown(f"**Source:** {memory.source}")
        st.markdown(f"**Created:** {memory.timestamp}")
        st.markdown(f"**Last Accessed:** {memory.last_accessed}")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Importance", f"{memory.importance_score:.2f}")
        with col2:
            st.metric("Access Count", memory.access_count)
        with col3:
            st.metric("Content Length", len(memory.content))
        
        # Full content
        st.markdown("**Content:**")
        st.text_area("", value=memory.content, height=200, disabled=True)
        
        # Tags
        if memory.tags:
            st.markdown("**Tags:**")
            st.write(", ".join(memory.tags))
        
        # Metadata
        if memory.metadata:
            st.markdown("**Metadata:**")
            st.json(memory.metadata)
        
        # Actions
        st.markdown("**Actions:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Edit Memory"):
                self._show_edit_dialog(memory)
        
        with col2:
            if st.button("Delete Memory", type="secondary"):
                self._show_delete_dialog(memory)
        
        with col3:
            if st.button("Find Similar"):
                similar_results = self._search_memories(memory.content[:100])
                st.session_state.search_results = similar_results
                st.rerun()
    
    def _render_memory_stats(self):
        """Render memory statistics panel."""
        st.subheader("📊 Memory Statistics")
        
        try:
            stats = self.memory_store.get_memory_stats()

            # Overall stats with fallback values
            total_memories = stats.get('total_memories', len(getattr(self.memory_store, 'memory_chunks', {})))
            total_size_mb = stats.get('total_size_mb', 0.0)

            st.metric("Total Memories", total_memories)
            st.metric("Storage Size", f"{total_size_mb:.2f} MB")
            
            # Memory types chart
            memory_types = stats.get('memory_types', {})
            if memory_types:
                st.markdown("**Memory Types:**")

                # Create pie chart
                fig = px.pie(
                    values=list(memory_types.values()),
                    names=list(memory_types.keys()),
                    title="Memory Distribution by Type"
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Recent activity
            if stats.get('newest_memory'):
                st.markdown("**Recent Activity:**")
                st.caption(f"Newest: {stats['newest_memory'][:10]}")
                if stats.get('oldest_memory'):
                    st.caption(f"Oldest: {stats['oldest_memory'][:10]}")
            
        except Exception as e:
            st.error(f"Error loading statistics: {e}")
    
    def _search_memories(self, query: str) -> List:
        """Search memories with current filters."""
        try:
            # Apply memory type filter
            memory_types = None
            if st.session_state.filter_settings['memory_types']:
                memory_types = [MemoryType(mt) for mt in st.session_state.filter_settings['memory_types']]
            
            # Perform search
            results = self.memory_reasoning.search_memories(
                query=query,
                user_id=st.session_state.filter_settings['user_filter'],
                memory_types=memory_types,
                max_results=50
            )
            
            # Apply additional filters
            filtered_results = []
            for result in results:
                memory = result.chunk
                
                # Date filter
                if st.session_state.filter_settings['date_range']:
                    start_date, end_date = st.session_state.filter_settings['date_range']
                    memory_date = datetime.fromisoformat(memory.timestamp).date()
                    if not (start_date <= memory_date <= end_date):
                        continue
                
                # Importance filter
                importance_min, importance_max = st.session_state.filter_settings['importance_range']
                if not (importance_min <= memory.importance_score <= importance_max):
                    continue
                
                filtered_results.append(result)
            
            return filtered_results
            
        except Exception as e:
            st.error(f"Search error: {e}")
            return []
    
    def _get_recent_memories(self, limit: int = 20) -> List:
        """Get recent memories with filters applied."""
        try:
            # Get all memories and sort by timestamp
            all_memories = list(self.memory_store.memory_chunks.values())
            
            # Apply filters
            filtered_memories = []
            for memory in all_memories:
                # Memory type filter
                if (st.session_state.filter_settings['memory_types'] and 
                    memory.memory_type.value not in st.session_state.filter_settings['memory_types']):
                    continue
                
                # User filter
                if (st.session_state.filter_settings['user_filter'] and 
                    memory.metadata.get('user_id') != st.session_state.filter_settings['user_filter']):
                    continue
                
                # Date filter
                if st.session_state.filter_settings['date_range']:
                    start_date, end_date = st.session_state.filter_settings['date_range']
                    memory_date = datetime.fromisoformat(memory.timestamp).date()
                    if not (start_date <= memory_date <= end_date):
                        continue
                
                # Importance filter
                importance_min, importance_max = st.session_state.filter_settings['importance_range']
                if not (importance_min <= memory.importance_score <= importance_max):
                    continue
                
                filtered_memories.append(memory)
            
            # Sort by timestamp (newest first)
            filtered_memories.sort(key=lambda m: m.timestamp, reverse=True)
            
            return filtered_memories[:limit]
            
        except Exception as e:
            st.error(f"Error loading recent memories: {e}")
            return []
    
    def _show_edit_dialog(self, memory):
        """Show memory editing dialog."""
        st.session_state.editing_memory = memory
        # This would open a modal or separate page for editing
        # For now, we'll use a simple form in the sidebar
        
    def _show_delete_dialog(self, memory):
        """Show memory deletion confirmation."""
        st.session_state.deleting_memory = memory
        # This would show a confirmation dialog

    # Phase 3.2.3: Enhanced search methods
    def _enhanced_search_memories(self, query: str) -> List:
        """Enhanced search using hybrid ranking and filtering."""
        try:
            search_mode = st.session_state.filter_settings.get('search_mode', 'Enhanced Hybrid')

            # Use enhanced search if available
            if hasattr(self.memory_store, 'enhanced_search_memories') and search_mode == 'Enhanced Hybrid':
                results = self.memory_store.enhanced_search_memories(
                    query=query,
                    max_results=50,
                    initial_candidates=100
                )
                logger.info(f"Enhanced search returned {len(results)} ranked results")
            else:
                # Fallback to regular search
                results = self._search_memories(query)

            # Apply enhanced filtering
            return self._apply_enhanced_filters(results)

        except Exception as e:
            st.error(f"Enhanced search error: {e}")
            return []

    def _search_within_sources(self, source_pattern: str, content_query: str = "") -> List:
        """Search within specific sources matching the pattern."""
        try:
            all_memories = list(self.memory_store.memory_chunks.values())
            matching_memories = []

            for memory in all_memories:
                # Check if source matches pattern
                source = getattr(memory, 'source', '')
                if source_pattern.lower() in source.lower():
                    # If content query provided, also check content relevance
                    if content_query:
                        if content_query.lower() in memory.content.lower():
                            matching_memories.append(memory)
                    else:
                        matching_memories.append(memory)

            # Apply filters
            return self._apply_enhanced_filters(matching_memories)

        except Exception as e:
            st.error(f"Source search error: {e}")
            return []

    def _find_similar_memories(self, content: str) -> List:
        """Find memories similar to the given content."""
        try:
            # Use the content as a search query
            if hasattr(self.memory_store, 'enhanced_search_memories'):
                results = self.memory_store.enhanced_search_memories(
                    query=content[:200],  # Use first 200 chars as query
                    max_results=10
                )
            else:
                results = self.memory_store.search_memories(content[:200], max_results=10)

            return self._apply_enhanced_filters(results)

        except Exception as e:
            st.error(f"Similar search error: {e}")
            return []

    def _apply_enhanced_filters(self, results: List) -> List:
        """Apply enhanced filtering to search results."""
        try:
            filtered_results = []

            for result in results:
                # Handle different result types
                if hasattr(result, 'content') and hasattr(result, 'final_score'):
                    # Enhanced RankedMemoryResult
                    memory = result
                    content = result.content
                    score = result.final_score
                    confidence = result.confidence_score
                    source = result.metadata.get('source_name', '')
                elif hasattr(result, 'chunk'):
                    # Legacy MemorySearchResult
                    memory = result.chunk
                    content = memory.content
                    score = getattr(result, 'similarity_score', 1.0)
                    confidence = memory.importance_score
                    source = memory.source
                else:
                    # Direct memory object
                    memory = result
                    content = memory.content
                    score = 1.0
                    confidence = memory.importance_score
                    source = memory.source

                # Apply ranking threshold filter
                ranking_threshold = st.session_state.filter_settings.get('ranking_threshold', 0.1)
                if score < ranking_threshold:
                    continue

                # Apply confidence filter
                confidence_range = st.session_state.filter_settings.get('confidence_range', [0.0, 1.0])
                if not (confidence_range[0] <= confidence <= confidence_range[1]):
                    continue

                # Apply source filters
                selected_sources = st.session_state.filter_settings.get('selected_sources', [])
                if selected_sources and not any(src in source for src in selected_sources):
                    continue

                # Apply source type filters
                source_types = st.session_state.filter_settings.get('source_types', [])
                if source_types:
                    source_type_match = False
                    for source_type in source_types:
                        if source_type == "PDF Documents" and ".pdf" in source.lower():
                            source_type_match = True
                        elif source_type == "Web Pages" and ("http" in source.lower() or "web" in source.lower()):
                            source_type_match = True
                        elif source_type == "Conversations" and ("conversation" in source.lower() or "chat" in source.lower()):
                            source_type_match = True
                        elif source_type == "System Logs" and ("log" in source.lower() or "system" in source.lower()):
                            source_type_match = True
                        elif source_type == "User Notes" and ("note" in source.lower() or "user" in source.lower()):
                            source_type_match = True

                    if not source_type_match:
                        continue

                filtered_results.append(result)

            return filtered_results

        except Exception as e:
            logger.error(f"Error applying enhanced filters: {e}")
            return results  # Return unfiltered results on error

    # Phase 3.2.3: Helper methods for enhanced functionality
    def _get_available_sources(self) -> List[str]:
        """Get list of available document sources."""
        try:
            all_memories = list(self.memory_store.memory_chunks.values())
            sources = set()

            for memory in all_memories:
                source = getattr(memory, 'source', '')
                if source:
                    # Extract clean source names
                    if 'uploads/' in source:
                        # Extract filename from upload path
                        import re
                        match = re.search(r'uploads/\d{8}_\d{6}_([^:]+)', source)
                        if match:
                            sources.add(match.group(1))
                    elif source.startswith('document:'):
                        # Extract document name
                        doc_name = source[9:].split(':')[0]
                        if not doc_name.startswith('web_ui/'):
                            sources.add(doc_name)
                    else:
                        # Use source as-is
                        sources.add(source)

            return sorted(list(sources))

        except Exception as e:
            logger.error(f"Error getting available sources: {e}")
            return []

    def _save_filter_preset(self):
        """Save current filter settings as a preset."""
        try:
            preset_name = st.text_input("Preset Name", placeholder="Enter preset name...")
            if preset_name and st.button("💾 Save"):
                if 'filter_presets' not in st.session_state:
                    st.session_state.filter_presets = {}

                st.session_state.filter_presets[preset_name] = st.session_state.filter_settings.copy()
                st.success(f"Preset '{preset_name}' saved!")

        except Exception as e:
            st.error(f"Error saving preset: {e}")

    def _get_saved_presets(self) -> Dict:
        """Get saved filter presets."""
        return getattr(st.session_state, 'filter_presets', {})

def main():
    """Main function to run the memory browser."""
    browser = MemoryBrowserUI()
    browser.render()

if __name__ == "__main__":
    main()
