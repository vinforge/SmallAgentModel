#!/usr/bin/env python3
"""
SAM Component Interface Standards
=================================

Standardized interfaces and patterns for SAM UI components.
This ensures consistency across all modules and reduces code duplication.

This module provides:
- Base component classes
- Standard component interfaces
- Common component patterns
- Unified error handling for components
- Standard lifecycle management

Author: SAM Development Team
Version: 1.0.0 - Standardization Framework
"""

import streamlit as st
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime

from sam.core.unified_utils import handle_error, safe_execute, render_status_badge

logger = logging.getLogger(__name__)


class BaseComponent(ABC):
    """Base class for all SAM UI components."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.initialized = False
        self.last_error = None
        self.render_count = 0
        
        # Initialize component
        self._initialize()
    
    def _initialize(self):
        """Initialize the component."""
        try:
            self._setup_component()
            self.initialized = True
            logger.info(f"Component '{self.name}' initialized successfully")
        except Exception as e:
            self.last_error = e
            logger.error(f"Failed to initialize component '{self.name}': {e}")
    
    @abstractmethod
    def _setup_component(self):
        """Setup component-specific initialization. Must be implemented by subclasses."""
        pass
    
    def render(self, **kwargs) -> Any:
        """Render the component with error handling."""
        if not self.initialized:
            self._render_initialization_error()
            return None
        
        try:
            self.render_count += 1
            return self._render_component(**kwargs)
        except Exception as e:
            self.last_error = e
            self._render_component_error(e)
            return None
    
    @abstractmethod
    def _render_component(self, **kwargs) -> Any:
        """Render the component content. Must be implemented by subclasses."""
        pass
    
    def _render_initialization_error(self):
        """Render initialization error message."""
        st.error(f"❌ Component '{self.name}' failed to initialize")
        
        if self.last_error:
            with st.expander("Error Details", expanded=False):
                st.code(str(self.last_error))
    
    def _render_component_error(self, error: Exception):
        """Render component error message."""
        handle_error(error, f"Component '{self.name}'")
    
    def get_status(self) -> Dict[str, Any]:
        """Get component status information."""
        return {
            'name': self.name,
            'description': self.description,
            'initialized': self.initialized,
            'render_count': self.render_count,
            'last_error': str(self.last_error) if self.last_error else None,
            'status': 'healthy' if self.initialized and not self.last_error else 'error'
        }
    
    def render_status_indicator(self) -> str:
        """Render a status indicator for this component."""
        status = self.get_status()
        
        if status['status'] == 'healthy':
            return render_status_badge('success', f"{self.name}: Ready")
        else:
            return render_status_badge('error', f"{self.name}: Error")


class RenderableComponent(BaseComponent):
    """Base class for components that render UI elements."""
    
    def __init__(self, name: str, description: str = "", title: str = ""):
        self.title = title or name
        super().__init__(name, description)
    
    def render_with_header(self, **kwargs) -> Any:
        """Render component with standard header."""
        if self.title:
            st.subheader(self.title)
        
        return self.render(**kwargs)
    
    def render_with_container(self, **kwargs) -> Any:
        """Render component within a container."""
        with st.container():
            return self.render_with_header(**kwargs)


class DataComponent(BaseComponent):
    """Base class for components that handle data operations."""
    
    def __init__(self, name: str, description: str = ""):
        self.data_cache = {}
        self.cache_timeout = 300  # 5 minutes
        super().__init__(name, description)
    
    def get_cached_data(self, key: str) -> Optional[Any]:
        """Get cached data if still valid."""
        if key in self.data_cache:
            data, timestamp = self.data_cache[key]
            if (datetime.now() - timestamp).total_seconds() < self.cache_timeout:
                return data
            else:
                # Remove expired cache
                del self.data_cache[key]
        
        return None
    
    def set_cached_data(self, key: str, data: Any):
        """Set cached data with timestamp."""
        self.data_cache[key] = (data, datetime.now())
    
    def clear_cache(self):
        """Clear all cached data."""
        self.data_cache.clear()
        logger.info(f"Cleared cache for component '{self.name}'")


class InteractiveComponent(RenderableComponent):
    """Base class for interactive components with user input."""
    
    def __init__(self, name: str, description: str = "", title: str = ""):
        self.callbacks = {}
        self.state_keys = []
        super().__init__(name, description, title)
    
    def register_callback(self, event: str, callback: Callable):
        """Register a callback for an event."""
        if event not in self.callbacks:
            self.callbacks[event] = []
        self.callbacks[event].append(callback)
    
    def trigger_callback(self, event: str, *args, **kwargs):
        """Trigger callbacks for an event."""
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Callback error in component '{self.name}': {e}")
    
    def get_state_key(self, key: str) -> str:
        """Get a state key scoped to this component."""
        return f"{self.name}_{key}"
    
    def set_state(self, key: str, value: Any):
        """Set component state."""
        state_key = self.get_state_key(key)
        st.session_state[state_key] = value
        
        if state_key not in self.state_keys:
            self.state_keys.append(state_key)
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get component state."""
        state_key = self.get_state_key(key)
        return st.session_state.get(state_key, default)
    
    def clear_state(self):
        """Clear all component state."""
        for state_key in self.state_keys:
            if state_key in st.session_state:
                del st.session_state[state_key]
        
        self.state_keys.clear()
        logger.info(f"Cleared state for component '{self.name}'")


class ComponentRegistry:
    """Registry for managing components across the application."""
    
    def __init__(self):
        self.components = {}
        self.component_groups = {}
    
    def register(self, component: BaseComponent, group: str = "default"):
        """Register a component."""
        self.components[component.name] = component
        
        if group not in self.component_groups:
            self.component_groups[group] = []
        
        if component.name not in self.component_groups[group]:
            self.component_groups[group].append(component.name)
        
        logger.info(f"Registered component '{component.name}' in group '{group}'")
    
    def get(self, name: str) -> Optional[BaseComponent]:
        """Get a component by name."""
        return self.components.get(name)
    
    def get_group(self, group: str) -> List[BaseComponent]:
        """Get all components in a group."""
        if group not in self.component_groups:
            return []
        
        return [self.components[name] for name in self.component_groups[group] 
                if name in self.components]
    
    def render_group(self, group: str, **kwargs):
        """Render all components in a group."""
        components = self.get_group(group)
        
        for component in components:
            component.render(**kwargs)
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get status summary of all components."""
        total_components = len(self.components)
        healthy_components = sum(1 for comp in self.components.values() 
                               if comp.get_status()['status'] == 'healthy')
        
        return {
            'total_components': total_components,
            'healthy_components': healthy_components,
            'error_components': total_components - healthy_components,
            'groups': list(self.component_groups.keys()),
            'component_details': {name: comp.get_status() 
                                for name, comp in self.components.items()}
        }
    
    def render_status_dashboard(self):
        """Render a status dashboard for all components."""
        st.subheader("🔧 Component Status Dashboard")
        
        status = self.get_status_summary()
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Components", status['total_components'])
        
        with col2:
            st.metric("Healthy", status['healthy_components'])
        
        with col3:
            st.metric("Errors", status['error_components'])
        
        # Component details
        for group_name, component_names in self.component_groups.items():
            with st.expander(f"📦 {group_name.title()} Components", expanded=False):
                for comp_name in component_names:
                    if comp_name in self.components:
                        comp = self.components[comp_name]
                        comp_status = comp.get_status()
                        
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write(f"**{comp_name}**: {comp_status['description']}")
                        
                        with col2:
                            status_indicator = comp.render_status_indicator()
                            st.write(status_indicator)


# Global component registry
_component_registry = ComponentRegistry()


def get_component_registry() -> ComponentRegistry:
    """Get the global component registry."""
    return _component_registry


def register_component(component: BaseComponent, group: str = "default"):
    """Register a component in the global registry."""
    _component_registry.register(component, group)


def get_component(name: str) -> Optional[BaseComponent]:
    """Get a component from the global registry."""
    return _component_registry.get(name)


def render_component_group(group: str, **kwargs):
    """Render all components in a group."""
    _component_registry.render_group(group, **kwargs)


def render_component_status_dashboard():
    """Render the global component status dashboard."""
    _component_registry.render_status_dashboard()


# Standard component decorators
def component_error_handler(func):
    """Decorator for component methods to handle errors gracefully."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if args and hasattr(args[0], 'name'):
                component_name = args[0].name
            else:
                component_name = func.__name__
            
            handle_error(e, f"Component '{component_name}'")
            return None
    
    return wrapper


def cached_component_method(cache_key: str = None, timeout: int = 300):
    """Decorator for caching component method results."""
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if not isinstance(self, DataComponent):
                return func(self, *args, **kwargs)
            
            key = cache_key or f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            
            # Try to get cached result
            cached_result = self.get_cached_data(key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(self, *args, **kwargs)
            self.set_cached_data(key, result)
            
            return result
        
        return wrapper
    return decorator
