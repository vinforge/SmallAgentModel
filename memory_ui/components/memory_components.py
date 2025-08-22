#!/usr/bin/env python3
"""
Memory UI Components
===================

Core memory interface components for the SAM Memory Control Center.
Extracted from the monolithic memory_app.py.

This module provides:
- Memory browser interface
- Memory editor interface
- Memory graph visualization
- Command interface
- Role-based access controls

Author: SAM Development Team
Version: 1.0.0 - Refactored from memory_app.py
"""

import streamlit as st
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class MemoryComponentManager:
    """Manages memory UI components and their interactions."""
    
    def __init__(self):
        self.browser = None
        self.editor = None
        self.visualizer = None
        self.command_processor = None
        self.role_filter = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize memory components."""
        try:
            # Import memory components
            from ui.memory_browser import MemoryBrowserUI
            from ui.memory_editor import MemoryEditor
            from ui.memory_graph import MemoryGraphVisualizer
            from ui.memory_commands import MemoryCommandProcessor, get_command_processor
            from ui.role_memory_filter import RoleBasedMemoryFilter, get_role_filter
            
            # Initialize components
            self.browser = MemoryBrowserUI()
            self.editor = MemoryEditor()
            self.visualizer = MemoryGraphVisualizer()
            self.command_processor = get_command_processor()
            self.role_filter = get_role_filter()
            
            logger.info("Memory components initialized successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import memory components: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize memory components: {e}")
    
    def render_memory_browser(self):
        """Render the memory browser interface."""
        try:
            if not self.browser:
                st.error("❌ Memory browser not available")
                return
            
            st.subheader("🧠 Memory Browser")
            self.browser.render()
            
        except Exception as e:
            logger.error(f"Error rendering memory browser: {e}")
            st.error(f"Error loading memory browser: {e}")
    
    def render_memory_editor(self):
        """Render the memory editor interface."""
        try:
            if not self.editor:
                st.error("❌ Memory editor not available")
                return
            
            st.subheader("✏️ Memory Editor")
            
            # Check if we have a memory to edit
            if hasattr(st.session_state, 'editing_memory') and st.session_state.editing_memory:
                self.editor.render_edit_interface(st.session_state.editing_memory)
            elif hasattr(st.session_state, 'deleting_memory') and st.session_state.deleting_memory:
                self.editor.render_delete_interface(st.session_state.deleting_memory)
            else:
                st.info("Select a memory from the Memory Browser to edit or delete it.")
                
                # Show recent edits and deletions
                col1, col2 = st.columns(2)
                
                with col1:
                    self.editor.render_undo_interface()
                
                with col2:
                    self.editor.render_audit_log()
                    
        except Exception as e:
            logger.error(f"Error rendering memory editor: {e}")
            st.error(f"Error loading memory editor: {e}")
    
    def render_memory_graph(self):
        """Render the memory graph visualization."""
        try:
            if not self.visualizer:
                st.error("❌ Memory graph visualizer not available")
                return
            
            st.subheader("📊 Memory Graph")
            self.visualizer.render()
            
        except Exception as e:
            logger.error(f"Error rendering memory graph: {e}")
            st.error(f"Error loading memory graph: {e}")
    
    def render_command_interface(self):
        """Render the command interface."""
        try:
            if not self.command_processor:
                st.error("❌ Command processor not available")
                return
            
            st.subheader("⚡ Memory Commands")
            
            # Command input
            command_input = st.text_input(
                "Enter memory command:",
                placeholder="e.g., search 'machine learning', delete old memories, export memories",
                help="Use natural language commands to interact with your memory"
            )
            
            col1, col2 = st.columns([1, 4])
            
            with col1:
                execute_button = st.button("🚀 Execute", type="primary")
            
            with col2:
                if st.button("❓ Help"):
                    self._show_command_help()
            
            if execute_button and command_input:
                with st.spinner("Processing command..."):
                    try:
                        result = self.command_processor.process_command(command_input)
                        
                        if result.get('success', False):
                            st.success(f"✅ {result.get('message', 'Command executed successfully')}")
                            
                            # Show results if available
                            if 'data' in result:
                                st.json(result['data'])
                        else:
                            st.error(f"❌ {result.get('error', 'Command failed')}")
                            
                    except Exception as e:
                        st.error(f"❌ Command execution failed: {str(e)}")
            
            # Show command history
            self._render_command_history()
            
        except Exception as e:
            logger.error(f"Error rendering command interface: {e}")
            st.error(f"Error loading command interface: {e}")
    
    def render_role_access(self):
        """Render role-based access controls."""
        try:
            if not self.role_filter:
                st.error("❌ Role filter not available")
                return
            
            st.subheader("👥 Role-Based Access")
            
            # Get available roles
            from agents.task_router import AgentRole
            available_roles = list(AgentRole)
            
            # Role selection
            selected_roles = st.multiselect(
                "Select roles to view memories for:",
                options=available_roles,
                default=st.session_state.get('selected_roles', [AgentRole.GENERAL]),
                format_func=lambda x: x.value.title()
            )
            
            # Update session state
            st.session_state.selected_roles = selected_roles
            
            if selected_roles:
                # Apply role filter
                filtered_memories = self.role_filter.filter_memories_by_roles(selected_roles)
                
                # Display filtered results
                st.write(f"**Found {len(filtered_memories)} memories for selected roles**")
                
                # Show role statistics
                role_stats = self.role_filter.get_role_statistics(selected_roles)
                self._render_role_statistics(role_stats)
                
                # Show filtered memories
                if filtered_memories:
                    self._render_filtered_memories(filtered_memories)
            else:
                st.info("Select one or more roles to view their associated memories.")
                
        except Exception as e:
            logger.error(f"Error rendering role access: {e}")
            st.error(f"Error loading role access: {e}")
    
    def _show_command_help(self):
        """Show command help information."""
        st.info("""
        **Available Commands:**
        
        **Search Commands:**
        - `search "keyword"` - Search memories by keyword
        - `find recent` - Find recent memories
        - `list all` - List all memories
        
        **Management Commands:**
        - `delete old` - Delete old memories
        - `export memories` - Export memory data
        - `backup memories` - Create memory backup
        
        **Analysis Commands:**
        - `analyze patterns` - Analyze memory patterns
        - `show stats` - Show memory statistics
        - `summarize memories` - Generate memory summary
        """)
    
    def _render_command_history(self):
        """Render command execution history."""
        if 'command_history' not in st.session_state:
            st.session_state.command_history = []
        
        if st.session_state.command_history:
            with st.expander("📜 Command History", expanded=False):
                for i, cmd in enumerate(reversed(st.session_state.command_history[-10:])):
                    st.write(f"**{i+1}.** {cmd.get('command', 'Unknown')}")
                    if cmd.get('result'):
                        st.caption(f"Result: {cmd['result']}")
    
    def _render_role_statistics(self, role_stats: Dict[str, Any]):
        """Render role-based statistics."""
        st.markdown("### 📊 Role Statistics")
        
        cols = st.columns(len(role_stats))
        
        for i, (role, stats) in enumerate(role_stats.items()):
            with cols[i]:
                st.metric(
                    label=role.title(),
                    value=stats.get('count', 0),
                    delta=stats.get('recent_count', 0)
                )
    
    def _render_filtered_memories(self, memories: List[Dict[str, Any]]):
        """Render filtered memory results."""
        st.markdown("### 🧠 Filtered Memories")
        
        for memory in memories[:20]:  # Limit to first 20 for performance
            with st.expander(f"Memory: {memory.get('id', 'Unknown')}", expanded=False):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Content:** {memory.get('content', 'No content')[:200]}...")
                    st.write(f"**Role:** {memory.get('role', 'Unknown')}")
                
                with col2:
                    st.write(f"**Created:** {memory.get('timestamp', 'Unknown')}")
                    st.write(f"**Type:** {memory.get('type', 'Unknown')}")


# Global memory component manager instance
_memory_component_manager = None


def get_memory_component_manager() -> MemoryComponentManager:
    """Get the global memory component manager instance."""
    global _memory_component_manager
    if _memory_component_manager is None:
        _memory_component_manager = MemoryComponentManager()
    return _memory_component_manager


def render_memory_browser():
    """Render the memory browser interface."""
    get_memory_component_manager().render_memory_browser()


def render_memory_editor():
    """Render the memory editor interface."""
    get_memory_component_manager().render_memory_editor()


def render_memory_graph():
    """Render the memory graph visualization."""
    get_memory_component_manager().render_memory_graph()


def render_command_interface():
    """Render the command interface."""
    get_memory_component_manager().render_command_interface()


def render_role_access():
    """Render role-based access controls."""
    get_memory_component_manager().render_role_access()
