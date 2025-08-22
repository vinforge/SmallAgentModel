#!/usr/bin/env python3
"""
SAM Application Controller
=========================

Main application controller that orchestrates the refactored SAM UI components.
This replaces the monolithic secure_streamlit_app.py with a modular architecture.

This module provides:
- Application initialization and configuration
- Component orchestration
- Route handling
- State management

Author: SAM Development Team
Version: 2.0.0 - Standardized Architecture
"""

import streamlit as st
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import standardized framework
from sam.core import BaseController, register_component, handle_error, format_duration

# Import refactored components
from sam.ui.security.session_manager import get_session_manager, check_authentication
from sam.ui.components.chat_interface import get_chat_interface, render_chat_interface
from sam.ui.handlers.document_handler import render_document_upload_section, render_uploaded_documents_list

logger = logging.getLogger(__name__)


class SAMAppController(BaseController):
    """Main controller for the SAM Streamlit application using standardized framework."""

    def __init__(self):
        super().__init__("SAM - Small Agent Model", "2.0.0")

    def _initialize_components(self):
        """Initialize SAM-specific components."""
        try:
            # Initialize core components
            self.session_manager = get_session_manager()
            self.chat_interface = get_chat_interface()

            # Register components
            self.components['session_manager'] = self.session_manager
            self.components['chat_interface'] = self.chat_interface

            logger.info("SAM application components initialized")

        except Exception as e:
            logger.error(f"Failed to initialize SAM components: {e}")
            self.components['session_manager'] = None
            self.components['chat_interface'] = None
    
    def _check_prerequisites(self) -> bool:
        """Check SAM application prerequisites."""
        try:
            # Initialize session
            if self.session_manager:
                self.session_manager.initialize_session()

            # Check authentication
            return check_authentication()

        except Exception as e:
            handle_error(e, "SAM Prerequisites")
            return False
    
    def _get_app_description(self) -> str:
        """Get SAM application description."""
        return "Your intelligent AI assistant for document analysis and research"

    def _get_app_status(self) -> Dict[str, str]:
        """Get SAM application status."""
        if not self.session_manager:
            return {"level": "error", "message": "Session Manager Unavailable"}

        if not self.chat_interface:
            return {"level": "warning", "message": "Chat Interface Unavailable"}

        return {"level": "healthy", "message": "Ready"}
    
    def _render_navigation(self):
        """Render SAM navigation."""
        nav_options = [
            ("💬 Chat", "chat"),
            ("📄 Documents", "documents"),
            ("🧠 Memory Center", "memory"),
            ("🔬 Algonauts", "algonauts"),
            ("⚙️ Settings", "settings")
        ]

        self.render_standard_navigation_buttons(nav_options, "current_page")

    def _render_quick_actions(self):
        """Render SAM quick actions."""
        st.markdown("### ⚡ Quick Actions")

        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

        if st.button("🗑️ Clear Session", use_container_width=True):
            # Clear session with confirmation
            if st.button("⚠️ Confirm Clear", use_container_width=True):
                st.session_state.clear()
                st.rerun()
    
    def _render_main_content(self):
        """Render SAM main content area."""
        # Get current page
        current_page = st.session_state.get('current_page', 'chat')

        if current_page == 'chat':
            self._render_chat_page()
        elif current_page == 'documents':
            self._render_documents_page()
        elif current_page == 'memory':
            self._render_memory_page()
        elif current_page == 'algonauts':
            self._render_algonauts_page()
        elif current_page == 'settings':
            self._render_settings_page()
        else:
            self._render_chat_page()  # Default to chat
    
    def _render_session_info(self):
        """Render session information in sidebar."""
        st.markdown("### 👤 Session")
        
        session_info = self.session_manager.get_session_info()
        
        if session_info:
            username = session_info.get('username', 'Anonymous')
            session_duration = format_duration(session_info.get('session_duration', 0))
            
            st.write(f"**User:** {username}")
            st.write(f"**Duration:** {session_duration}")
            
            if st.button("🚪 Logout", key="sidebar_logout"):
                self.session_manager.clear_session()
                st.rerun()
    
    def _render_sidebar_document_section(self):
        """Render document section in sidebar."""
        st.markdown("### 📁 Documents")
        
        # Quick upload
        uploaded_file = st.file_uploader(
            "Quick Upload",
            type=['pdf', 'txt', 'docx', 'md'],
            help="Upload a document for analysis",
            key="sidebar_upload"
        )
        
        if uploaded_file:
            st.success(f"✅ {uploaded_file.name} uploaded")
            
            # Process button
            if st.button("🚀 Process", key="sidebar_process"):
                # This would trigger document processing
                st.info("Processing document...")
        
        # Show uploaded documents count
        uploaded_docs = st.session_state.get('uploaded_documents', [])
        if uploaded_docs:
            st.write(f"📊 **{len(uploaded_docs)}** documents in session")
    
    def _render_sidebar_navigation(self):
        """Render navigation section in sidebar."""
        st.markdown("### 🧭 Navigation")
        
        nav_options = [
            ("💬 Chat", "chat"),
            ("📄 Documents", "documents"),
            ("🧠 Memory Center", "memory"),
            ("🔬 Algonauts", "algonauts"),
            ("⚙️ Settings", "settings")
        ]
        
        for label, key in nav_options:
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state.current_page = key
                st.rerun()
    
    def _render_sidebar_settings(self):
        """Render settings section in sidebar."""
        st.markdown("### ⚙️ Settings")
        
        # Theme toggle
        current_theme = st.session_state.get('theme', 'light')
        if st.button(f"🎨 Theme: {current_theme.title()}", key="theme_toggle"):
            new_theme = 'dark' if current_theme == 'light' else 'light'
            st.session_state.theme = new_theme
            st.rerun()
        
        # Debug mode
        debug_mode = st.session_state.get('debug_mode', False)
        if st.checkbox("🔧 Debug Mode", value=debug_mode, key="debug_toggle"):
            st.session_state.debug_mode = not debug_mode
        
        # Help and about
        if st.button("❓ Help", key="help_button"):
            st.session_state.show_help = True
        
        if st.button("ℹ️ About", key="about_button"):
            st.session_state.show_about = True
    
    def _render_main_content(self):
        """Render main content area."""
        # Get current page
        current_page = st.session_state.get('current_page', 'chat')
        
        if current_page == 'chat':
            self._render_chat_page()
        elif current_page == 'documents':
            self._render_documents_page()
        elif current_page == 'memory':
            self._render_memory_page()
        elif current_page == 'algonauts':
            self._render_algonauts_page()
        elif current_page == 'settings':
            self._render_settings_page()
        else:
            self._render_chat_page()  # Default to chat
    
    def _render_chat_page(self):
        """Render the chat page."""
        st.markdown("## 💬 Chat with SAM")
        
        # Render chat interface
        user_input = render_chat_interface()
        
        # Handle user input
        if user_input:
            self._handle_chat_input(user_input)
    
    def _render_documents_page(self):
        """Render the documents page."""
        st.markdown("## 📄 Document Management")
        
        # Document upload section
        render_document_upload_section()
        
        # Uploaded documents list
        render_uploaded_documents_list()
    
    def _render_memory_page(self):
        """Render the memory center page."""
        st.markdown("## 🧠 Memory Center")
        st.info("Memory Center integration coming soon...")
        
        # Placeholder for memory center integration
        if st.button("🚀 Open Memory Center"):
            st.info("This would open the Memory Control Center in a new tab")
    
    def _render_algonauts_page(self):
        """Render the Algonauts page."""
        st.markdown("## 🔬 Algonauts - Cognitive Analysis")
        st.info("Algonauts cognitive trajectory analysis integration coming soon...")
        
        # Placeholder for Algonauts integration
        if st.button("🧠 Start Cognitive Analysis"):
            st.info("This would launch the Algonauts experiment interface")
    
    def _render_settings_page(self):
        """Render the settings page."""
        st.markdown("## ⚙️ Settings")
        
        # Application settings
        st.markdown("### 🔧 Application Settings")
        
        # Model settings
        st.markdown("### 🤖 Model Configuration")
        
        # Performance settings
        st.markdown("### ⚡ Performance")
        
        # Security settings
        st.markdown("### 🔒 Security")
        
        st.info("Settings interface coming soon...")
    
    def _handle_chat_input(self, user_input: str):
        """Handle user chat input."""
        try:
            # This would integrate with the existing SAM response generation
            # For now, add a placeholder response
            
            with st.spinner("🤖 SAM is thinking..."):
                # Simulate processing time
                import time
                time.sleep(1)
                
                # Add assistant response
                response = f"I received your message: '{user_input}'. This is a placeholder response during refactoring."
                
                self.chat_interface.add_message(
                    "assistant", 
                    response,
                    metadata={'processing_time': 1.0}
                )
            
            st.rerun()
            
        except Exception as e:
            logger.error(f"Error handling chat input: {e}")
            st.error(f"❌ Error processing your message: {str(e)}")


def main():
    """Main application entry point."""
    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run application
    app = SAMAppController()
    app.run()


if __name__ == "__main__":
    main()
