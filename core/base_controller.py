#!/usr/bin/env python3
"""
SAM Base Controller Framework
============================

Unified base controller framework that standardizes patterns across all SAM UI modules.
This establishes consistent interfaces, error handling, and component management.

This module provides:
- Base controller class with standard patterns
- Unified error handling and logging
- Consistent component initialization
- Standard navigation and sidebar patterns
- Common state management utilities

Author: SAM Development Team
Version: 1.0.0 - Standardization Framework
"""

import streamlit as st
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseController(ABC):
    """Base controller class that standardizes patterns across SAM UI modules."""
    
    def __init__(self, app_name: str, version: str = "2.0.0"):
        self.app_name = app_name
        self.version = version
        self.components = {}
        self.initialized = False
        
        # Standard initialization
        self._initialize_base_state()
        self._initialize_components()
        self.initialized = True
    
    def _initialize_base_state(self):
        """Initialize base application state."""
        state_key = f"{self._get_app_key()}_initialized"
        
        if state_key not in st.session_state:
            st.session_state[state_key] = True
            st.session_state[f"{self._get_app_key()}_start_time"] = time.time()
            logger.info(f"{self.app_name} base state initialized")
    
    @abstractmethod
    def _initialize_components(self):
        """Initialize application-specific components. Must be implemented by subclasses."""
        pass
    
    def _get_app_key(self) -> str:
        """Get a unique key for this application."""
        return self.app_name.lower().replace(" ", "_").replace("-", "_")
    
    def run(self):
        """Main application entry point with standardized error handling."""
        try:
            # Configure page
            self._configure_page()
            
            # Check prerequisites
            if not self._check_prerequisites():
                return
            
            # Render main application
            self._render_main_application()
            
        except Exception as e:
            logger.error(f"{self.app_name} error: {e}")
            self._render_error_page(str(e))
    
    def _configure_page(self):
        """Configure Streamlit page with standard settings."""
        # Extract icon from app name (first emoji if present)
        icon = "🤖"
        if self.app_name and len(self.app_name) > 0:
            for char in self.app_name:
                if ord(char) > 127:  # Unicode emoji range
                    icon = char
                    break
        
        st.set_page_config(
            page_title=self.app_name,
            page_icon=icon,
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/vinforge/SmallAgentModel',
                'Report a bug': 'https://github.com/vinforge/SmallAgentModel/issues',
                'About': f"{self.app_name} v{self.version} - SAM Framework"
            }
        )
        
        # Apply standard CSS
        self._apply_standard_css()
    
    def _apply_standard_css(self):
        """Apply standard CSS styling across all SAM applications."""
        st.markdown("""
        <style>
        /* SAM Standard Styling Framework */
        .main-header {
            padding: 1rem 0;
            border-bottom: 2px solid #e0e0e0;
            margin-bottom: 2rem;
        }
        
        .sam-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            border-left: 4px solid #007bff;
        }
        
        .sam-status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 6px;
        }
        
        .sam-status-healthy { background-color: #28a745; }
        .sam-status-warning { background-color: #ffc107; }
        .sam-status-error { background-color: #dc3545; }
        .sam-status-info { background-color: #17a2b8; }
        
        .sam-navigation-button {
            width: 100%;
            margin-bottom: 0.25rem;
        }
        
        .sam-metric-card {
            text-align: center;
            padding: 1rem;
            background: white;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .sam-error-container {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .sam-success-container {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    @abstractmethod
    def _check_prerequisites(self) -> bool:
        """Check application prerequisites. Must be implemented by subclasses."""
        pass
    
    def _render_main_application(self):
        """Render the main application with standard layout."""
        # Header
        self._render_standard_header()
        
        # Sidebar
        self._render_standard_sidebar()
        
        # Main content
        self._render_main_content()
    
    def _render_standard_header(self):
        """Render standardized application header."""
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown(f"""
            <div class="main-header">
                <h1>{self.app_name}</h1>
                <p>{self._get_app_description()}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Status indicator
            status = self._get_app_status()
            status_class = f"sam-status-{status['level']}"
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem;">
                <span class="sam-status-indicator {status_class}"></span>
                <strong>{status['message']}</strong>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Version and uptime
            uptime = self._get_uptime()
            st.markdown(f"""
            <div style="text-align: right; padding: 1rem;">
                <small>v{self.version}</small><br>
                <small>Uptime: {uptime}</small>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_standard_sidebar(self):
        """Render standardized sidebar with navigation."""
        with st.sidebar:
            # App info
            st.markdown(f"### {self.app_name}")
            
            # Navigation
            self._render_navigation()
            
            # Quick actions
            st.markdown("---")
            self._render_quick_actions()
            
            # System info
            st.markdown("---")
            self._render_system_info()
    
    @abstractmethod
    def _render_navigation(self):
        """Render application-specific navigation. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _render_quick_actions(self):
        """Render application-specific quick actions. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _render_main_content(self):
        """Render main content area. Must be implemented by subclasses."""
        pass
    
    def _get_app_description(self) -> str:
        """Get application description for header."""
        return "SAM Framework Application"
    
    def _get_app_status(self) -> Dict[str, str]:
        """Get current application status."""
        if not self.initialized:
            return {"level": "warning", "message": "Initializing..."}
        
        return {"level": "healthy", "message": "Ready"}
    
    def _get_uptime(self) -> str:
        """Get application uptime."""
        start_time_key = f"{self._get_app_key()}_start_time"
        start_time = st.session_state.get(start_time_key, time.time())
        uptime_seconds = time.time() - start_time
        
        if uptime_seconds < 60:
            return f"{uptime_seconds:.0f}s"
        elif uptime_seconds < 3600:
            return f"{uptime_seconds/60:.1f}m"
        else:
            return f"{uptime_seconds/3600:.1f}h"
    
    def _render_system_info(self):
        """Render system information in sidebar."""
        st.markdown("### ℹ️ System Info")
        
        # Component status
        for name, component in self.components.items():
            status = "✅" if component else "❌"
            st.caption(f"{name}: {status}")
    
    def _render_error_page(self, error_message: str):
        """Render error page with standard formatting."""
        st.markdown(f"""
        <div class="sam-error-container">
            <h2>❌ Application Error</h2>
            <p><strong>{self.app_name}</strong> encountered an error:</p>
            <code>{error_message}</code>
        </div>
        """, unsafe_allow_html=True)
        
        # Show debug information in expander
        with st.expander("🔧 Debug Information", expanded=False):
            st.write("**Application State:**")
            st.json({
                "app_name": self.app_name,
                "version": self.version,
                "initialized": self.initialized,
                "components": list(self.components.keys()),
                "timestamp": datetime.now().isoformat()
            })
    
    def render_standard_navigation_buttons(self, pages: List[Tuple[str, str]], 
                                         current_page_key: str = "current_page"):
        """Render standard navigation buttons."""
        st.markdown("### 🧭 Navigation")
        
        for label, key in pages:
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state[current_page_key] = key
                st.rerun()
    
    def render_standard_metrics(self, metrics: Dict[str, Any]):
        """Render metrics in standard format."""
        cols = st.columns(len(metrics))
        
        for i, (label, value) in enumerate(metrics.items()):
            with cols[i]:
                if isinstance(value, dict):
                    st.metric(
                        label=label,
                        value=value.get('value', 'N/A'),
                        delta=value.get('delta')
                    )
                else:
                    st.metric(label=label, value=value)
    
    def render_standard_status_card(self, title: str, status: str, details: Optional[str] = None):
        """Render a standard status card."""
        status_colors = {
            'healthy': '#28a745',
            'warning': '#ffc107', 
            'error': '#dc3545',
            'info': '#17a2b8'
        }
        
        color = status_colors.get(status, '#6c757d')
        
        st.markdown(f"""
        <div class="sam-card" style="border-left-color: {color};">
            <h4>{title}</h4>
            <p><strong>Status:</strong> {status.title()}</p>
            {f"<p>{details}</p>" if details else ""}
        </div>
        """, unsafe_allow_html=True)
    
    def handle_component_error(self, component_name: str, error: Exception):
        """Standard error handling for components."""
        logger.error(f"{component_name} error in {self.app_name}: {error}")
        st.error(f"❌ {component_name} error: {str(error)}")
    
    def get_session_key(self, key: str) -> str:
        """Get a session key scoped to this application."""
        return f"{self._get_app_key()}_{key}"
    
    def set_session_value(self, key: str, value: Any):
        """Set a session value scoped to this application."""
        st.session_state[self.get_session_key(key)] = value
    
    def get_session_value(self, key: str, default: Any = None) -> Any:
        """Get a session value scoped to this application."""
        return st.session_state.get(self.get_session_key(key), default)


class ComponentManager:
    """Base class for managing application components."""
    
    def __init__(self, app_name: str):
        self.app_name = app_name
        self.components = {}
        self.initialized = False
    
    def register_component(self, name: str, component: Any):
        """Register a component."""
        self.components[name] = component
        logger.info(f"Registered component '{name}' for {self.app_name}")
    
    def get_component(self, name: str) -> Optional[Any]:
        """Get a registered component."""
        return self.components.get(name)
    
    def is_component_available(self, name: str) -> bool:
        """Check if a component is available."""
        return name in self.components and self.components[name] is not None
    
    def render_component_status(self):
        """Render status of all components."""
        st.markdown("### 🔧 Component Status")
        
        for name, component in self.components.items():
            status = "✅ Available" if component else "❌ Unavailable"
            st.write(f"**{name}:** {status}")


# Global singleton pattern for component managers
_component_managers = {}


def get_component_manager(app_name: str) -> ComponentManager:
    """Get or create a component manager for an application."""
    if app_name not in _component_managers:
        _component_managers[app_name] = ComponentManager(app_name)
    return _component_managers[app_name]
