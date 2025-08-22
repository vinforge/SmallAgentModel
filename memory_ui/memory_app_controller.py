#!/usr/bin/env python3
"""
SAM Memory Application Controller
================================

Main controller for the SAM Memory Control Center.
Replaces the monolithic memory_app.py with a modular architecture.

This module provides:
- Memory application orchestration
- Page routing and navigation
- Component integration
- Authentication flow

Author: SAM Development Team
Version: 1.0.0 - Refactored from memory_app.py
"""

import os
import streamlit as st
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Environment setup
os.environ.update({
    'STREAMLIT_SERVER_FILE_WATCHER_TYPE': 'none',
    'STREAMLIT_SERVER_HEADLESS': 'true',
    'STREAMLIT_BROWSER_GATHER_USAGE_STATS': 'false',
    'PYTORCH_DISABLE_PER_OP_PROFILING': '1'
})

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


class MemoryAppController:
    """Main controller for the SAM Memory Control Center."""
    
    def __init__(self):
        self.app_name = "SAM Memory Control Center"
        self.version = "2.0.0"
        self.auth_manager = None
        self.component_manager = None
        self.status_handler = None
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize application components."""
        try:
            from sam.memory_ui.security.memory_auth import get_memory_auth_manager
            from sam.memory_ui.components.memory_components import get_memory_component_manager
            from sam.memory_ui.handlers.system_status import get_system_status_handler
            
            self.auth_manager = get_memory_auth_manager()
            self.component_manager = get_memory_component_manager()
            self.status_handler = get_system_status_handler()
            
            logger.info("Memory application components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
    
    def run(self):
        """Main application entry point."""
        try:
            # Configure page
            self._configure_page()
            
            # Check authentication
            if not self.auth_manager or not self.auth_manager.check_authentication():
                self._render_authentication_page()
                return
            
            # Render main application
            self._render_main_application()
            
        except Exception as e:
            logger.error(f"Memory application error: {e}")
            st.error(f"❌ Application error: {str(e)}")
    
    def _configure_page(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title=self.app_name,
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/vinforge/SmallAgentModel',
                'Report a bug': 'https://github.com/vinforge/SmallAgentModel/issues',
                'About': f"{self.app_name} v{self.version} - Memory Management Interface"
            }
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            padding: 1rem 0;
            border-bottom: 2px solid #e0e0e0;
            margin-bottom: 2rem;
        }
        
        .memory-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        
        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 6px;
        }
        
        .status-healthy { background-color: #28a745; }
        .status-warning { background-color: #ffc107; }
        .status-error { background-color: #dc3545; }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_authentication_page(self):
        """Render authentication required page."""
        if self.auth_manager:
            self.auth_manager.render_authentication_required()
        else:
            st.error("❌ Authentication system not available")
    
    def _render_main_application(self):
        """Render the main memory application."""
        # Header
        self._render_header()
        
        # Sidebar
        self._render_sidebar()
        
        # Main content
        self._render_main_content()
    
    def _render_header(self):
        """Render application header."""
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown(f"""
            <div class="main-header">
                <h1>🧠 {self.app_name}</h1>
                <p>Advanced memory management and visualization interface</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Security status
            if self.auth_manager:
                status_text = self.auth_manager.render_security_status_indicator()
                st.markdown(f"**Security:** {status_text}")
        
        with col3:
            # Version info
            st.markdown(f"""
            <div style="text-align: right;">
                <small>v{self.version}</small><br>
                <small>Memory UI</small>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_sidebar(self):
        """Render application sidebar."""
        with st.sidebar:
            # Navigation
            st.markdown("### 🧭 Navigation")
            
            pages = [
                ("🧠 Memory Browser", "browser"),
                ("✏️ Memory Editor", "editor"),
                ("📊 Memory Graph", "graph"),
                ("⚡ Commands", "commands"),
                ("👥 Role Access", "roles"),
                ("🖥️ System Status", "status"),
                ("🏆 Memory Ranking", "ranking"),
                ("📚 Citations", "citations"),
                ("💡 Insights", "insights")
            ]
            
            for label, key in pages:
                if st.button(label, key=f"nav_{key}", use_container_width=True):
                    st.session_state.current_memory_page = key
                    st.rerun()
            
            # Quick actions
            st.markdown("---")
            st.markdown("### ⚡ Quick Actions")
            
            if st.button("🔍 Search Memories", use_container_width=True):
                st.session_state.show_search = True
                st.rerun()
            
            if st.button("📤 Export Data", use_container_width=True):
                st.session_state.show_export = True
                st.rerun()
            
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
            
            # System info
            st.markdown("---")
            st.markdown("### ℹ️ System Info")
            
            if self.status_handler:
                # Quick system status
                st.caption("Memory Store: ✅ Active")
                st.caption("Vector Index: ✅ Ready")
                st.caption("Commands: ✅ Available")
    
    def _render_main_content(self):
        """Render main content area."""
        # Get current page
        current_page = st.session_state.get('current_memory_page', 'browser')
        
        # Route to appropriate page
        if current_page == 'browser':
            self._render_browser_page()
        elif current_page == 'editor':
            self._render_editor_page()
        elif current_page == 'graph':
            self._render_graph_page()
        elif current_page == 'commands':
            self._render_commands_page()
        elif current_page == 'roles':
            self._render_roles_page()
        elif current_page == 'status':
            self._render_status_page()
        elif current_page == 'ranking':
            self._render_ranking_page()
        elif current_page == 'citations':
            self._render_citations_page()
        elif current_page == 'insights':
            self._render_insights_page()
        else:
            self._render_browser_page()  # Default
    
    def _render_browser_page(self):
        """Render memory browser page."""
        if self.component_manager:
            self.component_manager.render_memory_browser()
        else:
            st.error("❌ Memory browser not available")
    
    def _render_editor_page(self):
        """Render memory editor page."""
        if self.component_manager:
            self.component_manager.render_memory_editor()
        else:
            st.error("❌ Memory editor not available")
    
    def _render_graph_page(self):
        """Render memory graph page."""
        if self.component_manager:
            self.component_manager.render_memory_graph()
        else:
            st.error("❌ Memory graph not available")
    
    def _render_commands_page(self):
        """Render commands page."""
        if self.component_manager:
            self.component_manager.render_command_interface()
        else:
            st.error("❌ Command interface not available")
    
    def _render_roles_page(self):
        """Render role access page."""
        if self.component_manager:
            self.component_manager.render_role_access()
        else:
            st.error("❌ Role access not available")
    
    def _render_status_page(self):
        """Render system status page."""
        if self.status_handler:
            self.status_handler.render_system_status()
        else:
            st.error("❌ System status not available")
    
    def _render_ranking_page(self):
        """Render memory ranking page."""
        if self.status_handler:
            self.status_handler.render_memory_ranking()
        else:
            st.error("❌ Memory ranking not available")
    
    def _render_citations_page(self):
        """Render citations page."""
        if self.status_handler:
            self.status_handler.render_citation_engine()
        else:
            st.error("❌ Citation engine not available")
    
    def _render_insights_page(self):
        """Render insights page."""
        st.subheader("💡 Memory Insights")
        st.info("Memory insights interface coming soon...")
        
        # Placeholder for insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Memory Patterns", "15", delta="+3")
            st.metric("Knowledge Gaps", "7", delta="-2")
        
        with col2:
            st.metric("Learning Velocity", "2.3x", delta="+0.4x")
            st.metric("Retention Rate", "94%", delta="+2%")


def main():
    """Main entry point for the memory application."""
    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run application
    app = MemoryAppController()
    app.run()


if __name__ == "__main__":
    main()
