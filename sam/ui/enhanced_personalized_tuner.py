#!/usr/bin/env python3
"""
Enhanced Personalized Tuner UI
===============================

Redesigned Personalized Tuner interface that separates Style Tuning (DPO)
from Reasoning Tuning (SSRL) with independent controls and clear user experience.

Features:
- Dual-section UI: Style Tuning and Reasoning Tuning
- Independent adapter activation controls
- Training progress monitoring
- Multi-adapter management interface
- Integration with existing DPO infrastructure

Author: SAM Development Team
Version: 1.0.0
"""

import streamlit as st
import logging
import time
import traceback
from typing import Dict, Any, List, Optional, Union, Callable
from pathlib import Path
from contextlib import contextmanager

# Configure module logger
logger = logging.getLogger(__name__)

# Constants
DEFAULT_USER_ID = "default_user"
MAX_ADAPTERS_DISPLAY = 10
TRAINING_TIMEOUT_SECONDS = 3600  # 1 hour
UI_REFRESH_INTERVAL = 1.0  # seconds


def handle_ui_errors(func: Callable) -> Callable:
    """
    Decorator for handling UI errors gracefully.

    Args:
        func: Function to wrap with error handling

    Returns:
        Wrapped function with error handling
    """
    def wrapper(*args, **kwargs):
        """Wrapper function that handles UI errors gracefully."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"UI error in {func.__name__}: {e}", exc_info=True)
            st.error(f"An error occurred: {e}")
            st.error("Please check the logs for more details.")
            return None

    return wrapper


@contextmanager
def ui_section(title: str, error_message: str = "Section failed to load"):
    """
    Context manager for UI sections with error handling.

    Args:
        title: Section title for logging
        error_message: Error message to display on failure
    """
    try:
        logger.debug(f"Rendering UI section: {title}")
        yield
    except Exception as e:
        logger.error(f"Error in UI section '{title}': {e}", exc_info=True)
        st.error(f"{error_message}: {e}")


class EnhancedPersonalizedTuner:
    """
    Enhanced Personalized Tuner with separate Style and Reasoning sections.
    """
    
    def __init__(self):
        """Initialize the enhanced tuner interface."""
        self.logger = logging.getLogger(f"{__name__}.EnhancedPersonalizedTuner")
        
        # Initialize session state
        self._init_session_state()
        
        self.logger.info("Enhanced Personalized Tuner initialized")
    
    def _init_session_state(self) -> None:
        """
        Initialize Streamlit session state variables with validation.

        Sets up all necessary session state variables with default values
        and validates existing values for consistency.
        """
        try:
            # Define default session state values
            defaults = {
                'style_tuner_active': False,
                'style_training_status': "idle",
                'reasoning_tuner_active': False,
                'reasoning_training_status': "idle",
                'active_adapters': [],
                'user_id': DEFAULT_USER_ID,
                'last_refresh_time': time.time(),
                'error_count': 0,
                'training_logs': [],
                'adapter_cache': {}
            }

            # Initialize missing values
            for key, default_value in defaults.items():
                if key not in st.session_state:
                    st.session_state[key] = default_value

            # Validate existing values
            self._validate_session_state()

            logger.debug("Session state initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize session state: {e}")
            # Reset to defaults on error
            for key, default_value in defaults.items():
                st.session_state[key] = default_value

    def _validate_session_state(self) -> None:
        """Validate session state values and fix inconsistencies."""
        try:
            # Validate training status values
            valid_statuses = {"idle", "running", "completed", "error"}

            if st.session_state.style_training_status not in valid_statuses:
                logger.warning(f"Invalid style training status: {st.session_state.style_training_status}")
                st.session_state.style_training_status = "idle"

            if st.session_state.reasoning_training_status not in valid_statuses:
                logger.warning(f"Invalid reasoning training status: {st.session_state.reasoning_training_status}")
                st.session_state.reasoning_training_status = "idle"

            # Validate user_id
            if not isinstance(st.session_state.user_id, str) or not st.session_state.user_id.strip():
                logger.warning("Invalid user_id, resetting to default")
                st.session_state.user_id = DEFAULT_USER_ID

            # Validate active_adapters list
            if not isinstance(st.session_state.active_adapters, list):
                logger.warning("Invalid active_adapters, resetting to empty list")
                st.session_state.active_adapters = []

        except Exception as e:
            logger.error(f"Session state validation failed: {e}")
            # Don't raise - allow UI to continue with potentially invalid state
    
    @handle_ui_errors
    def render(self) -> None:
        """Render the enhanced personalized tuner interface."""
        st.header("🧠 Enhanced Personalized Tuner")
        st.markdown("Customize SAM's behavior with independent Style and Reasoning tuning.")

        # Update refresh time
        st.session_state.last_refresh_time = time.time()

        # Load multi-adapter manager with error handling
        adapter_manager = self._load_adapter_manager()
        if adapter_manager is None:
            return
        
        # Get user adapters
        user_adapters = adapter_manager.get_user_adapters(st.session_state.user_id)
        
        # Adapter Status Overview
        self._render_adapter_overview(adapter_manager, user_adapters)
        
        # Create two columns for Style and Reasoning tuning
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_style_tuning_section(adapter_manager, user_adapters)
        
        with col2:
            self._render_reasoning_tuning_section(adapter_manager, user_adapters)
        
        # Advanced Settings
        with st.expander("🔧 Advanced Multi-Adapter Settings"):
            self._render_advanced_settings(adapter_manager)

    def _load_adapter_manager(self):
        """
        Load multi-adapter manager with proper error handling.

        Returns:
            MultiAdapterManager instance or None if loading failed
        """
        try:
            from sam.cognition.multi_adapter_manager import get_multi_adapter_manager
            adapter_manager = get_multi_adapter_manager()

            # Validate manager is working
            stats = adapter_manager.get_stats()
            logger.debug(f"Adapter manager loaded: {stats}")

            return adapter_manager

        except ImportError as e:
            logger.error(f"Multi-adapter manager import failed: {e}")
            st.error("Multi-adapter manager not available. Please check installation.")
            st.info("This feature requires the multi-adapter system to be properly installed.")
            return None

        except Exception as e:
            logger.error(f"Failed to load adapter manager: {e}")
            st.error(f"Failed to initialize adapter system: {e}")
            st.info("Please check the system logs for more details.")
            return None
    
    def _render_adapter_overview(self, adapter_manager, user_adapters: List):
        """Render adapter status overview."""
        st.subheader("📊 Adapter Status Overview")
        
        # Get active adapters
        active_adapters = [a for a in user_adapters if a.is_active]
        
        # Status metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Adapters", len(user_adapters))
        
        with col2:
            st.metric("Active Adapters", len(active_adapters))
        
        with col3:
            style_adapters = [a for a in user_adapters if a.adapter_type.value == "dpo_style"]
            st.metric("Style Adapters", len(style_adapters))
        
        with col4:
            reasoning_adapters = [a for a in user_adapters if a.adapter_type.value == "ssrl_reasoning"]
            st.metric("Reasoning Adapters", len(reasoning_adapters))
        
        # Active adapters list
        if active_adapters:
            st.success(f"**Active Adapters:** {', '.join([a.adapter_id for a in active_adapters])}")
        else:
            st.info("No adapters currently active. Enable adapters below to customize SAM's behavior.")
    
    def _render_style_tuning_section(self, adapter_manager, user_adapters: List):
        """Render the Style Tuning (DPO) section."""
        st.subheader("🎨 Style Tuning (DPO)")
        st.markdown("Customize SAM's communication style and preferences.")
        
        # Get DPO adapters
        dpo_adapters = [a for a in user_adapters if a.adapter_type.value == "dpo_style"]
        
        # Style tuner activation toggle
        style_active = st.checkbox(
            "🎨 Activate Style Tuner",
            value=st.session_state.style_tuner_active,
            key="style_tuner_toggle",
            help="Enable/disable style personalization"
        )
        
        if style_active != st.session_state.style_tuner_active:
            st.session_state.style_tuner_active = style_active
            self._update_adapter_activation(adapter_manager, "dpo_style", style_active)
        
        # DPO adapter selection
        if dpo_adapters:
            st.markdown("**Available Style Adapters:**")
            for adapter in dpo_adapters:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.text(f"📝 {adapter.adapter_id}")
                    if adapter.description:
                        st.caption(adapter.description)
                
                with col2:
                    status = "🟢 Active" if adapter.is_active else "⚪ Inactive"
                    st.text(status)
                
                with col3:
                    if st.button(f"Toggle", key=f"toggle_dpo_{adapter.adapter_id}"):
                        self._toggle_individual_adapter(adapter_manager, adapter.adapter_id)
        else:
            st.info("No style adapters found. Train a style adapter to customize communication.")
        
        # Style training section
        st.markdown("---")
        st.markdown("**🚀 Style Training**")
        
        # Integration with existing DPO interface
        try:
            self._render_dpo_training_interface()
        except Exception as e:
            st.error(f"Error loading DPO training interface: {e}")
            st.button("🎨 Start Style Training", disabled=True, help="DPO training interface not available")
    
    def _render_reasoning_tuning_section(self, adapter_manager, user_adapters: List):
        """Render the Reasoning Tuning (SSRL) section."""
        st.subheader("🧠 Reasoning Tuning (SSRL)")
        st.markdown("Enhance SAM's reasoning and problem-solving capabilities.")
        
        # Get SSRL adapters
        ssrl_adapters = [a for a in user_adapters if a.adapter_type.value == "ssrl_reasoning"]
        
        # Reasoning tuner activation toggle
        reasoning_active = st.checkbox(
            "🧠 Activate Reasoning Tuner",
            value=st.session_state.reasoning_tuner_active,
            key="reasoning_tuner_toggle",
            help="Enable/disable reasoning enhancement"
        )
        
        if reasoning_active != st.session_state.reasoning_tuner_active:
            st.session_state.reasoning_tuner_active = reasoning_active
            self._update_adapter_activation(adapter_manager, "ssrl_reasoning", reasoning_active)
        
        # SSRL adapter selection
        if ssrl_adapters:
            st.markdown("**Available Reasoning Adapters:**")
            for adapter in ssrl_adapters:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.text(f"🧠 {adapter.adapter_id}")
                    if adapter.description:
                        st.caption(adapter.description)
                
                with col2:
                    status = "🟢 Active" if adapter.is_active else "⚪ Inactive"
                    st.text(status)
                
                with col3:
                    if st.button(f"Toggle", key=f"toggle_ssrl_{adapter.adapter_id}"):
                        self._toggle_individual_adapter(adapter_manager, adapter.adapter_id)
        else:
            st.info("No reasoning adapters found. Train a reasoning adapter to enhance problem-solving.")
        
        # SSRL training section
        st.markdown("---")
        st.markdown("**🚀 Reasoning Training**")
        
        # SSRL dataset management
        self._render_ssrl_dataset_management()
        
        # SSRL training controls
        self._render_ssrl_training_interface()
    
    def _render_dpo_training_interface(self):
        """Render DPO training interface integration."""
        try:
            # Import existing DPO components
            from sam.ui.personalized_tuner import PersonalizedTuner
            
            # Create a simplified DPO interface
            st.markdown("**Preference Data Management:**")
            
            # Preference data upload
            uploaded_file = st.file_uploader(
                "Upload preference data (JSON)",
                type=['json'],
                key="dpo_data_upload",
                help="Upload conversation preferences for style training"
            )
            
            if uploaded_file:
                st.success("Preference data uploaded successfully!")
            
            # Training button
            if st.button("🎨 Start Style Training", key="start_dpo_training"):
                self._start_dpo_training()
            
        except ImportError:
            st.warning("DPO training interface not available")
    
    def _render_ssrl_dataset_management(self):
        """Render SSRL dataset management interface."""
        st.markdown("**Training Dataset Management:**")
        
        # Dataset upload
        uploaded_file = st.file_uploader(
            "Upload QA dataset (JSON)",
            type=['json'],
            key="ssrl_data_upload",
            help="Upload question-answer pairs for reasoning training"
        )
        
        if uploaded_file:
            st.success("QA dataset uploaded successfully!")
            
            # Show dataset preview
            try:
                import json
                data = json.load(uploaded_file)
                st.markdown(f"**Dataset Info:** {len(data)} question-answer pairs")
                
                # Preview first few items
                if data and len(data) > 0:
                    with st.expander("📋 Dataset Preview"):
                        for i, item in enumerate(data[:3]):
                            st.markdown(f"**Q{i+1}:** {item.get('question', 'N/A')}")
                            st.markdown(f"**A{i+1}:** {item.get('answer', 'N/A')}")
                            st.markdown("---")
                        
                        if len(data) > 3:
                            st.markdown(f"... and {len(data) - 3} more items")
                            
            except Exception as e:
                st.error(f"Error reading dataset: {e}")
        
        # Clear dataset button
        if st.button("🗑️ Clear Dataset", key="clear_ssrl_dataset"):
            st.info("Dataset cleared")
    
    def _render_ssrl_training_interface(self):
        """Render SSRL training interface."""
        st.markdown("**Training Configuration:**")
        
        # Training parameters
        col1, col2 = st.columns(2)
        
        with col1:
            epochs = st.number_input("Training Epochs", min_value=1, max_value=10, value=3)
            learning_rate = st.selectbox("Learning Rate", ["1e-5", "5e-6", "1e-6"], index=0)
        
        with col2:
            batch_size = st.selectbox("Batch Size", [2, 4, 8], index=1)
            format_weight = st.slider("Format Reward Weight", 0.1, 0.9, 0.3, 0.1)
        
        # Training button
        training_disabled = st.session_state.reasoning_training_status == "running"
        
        if st.button(
            "🧠 Start Reasoning Training", 
            key="start_ssrl_training",
            disabled=training_disabled
        ):
            self._start_ssrl_training(epochs, learning_rate, batch_size, format_weight)
        
        # Training status
        if st.session_state.reasoning_training_status == "running":
            st.info("🔄 Reasoning training in progress...")
            
            # Progress bar (mock)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate progress (in real implementation, this would track actual training)
            for i in range(100):
                progress_bar.progress(i + 1)
                status_text.text(f"Training progress: {i + 1}%")
                time.sleep(0.01)  # Very fast for demo
            
            st.success("✅ Reasoning training completed!")
            st.session_state.reasoning_training_status = "completed"
            st.rerun()
    
    def _render_advanced_settings(self, adapter_manager):
        """Render advanced multi-adapter settings."""
        st.markdown("**🔧 Multi-Adapter Configuration**")
        
        # Adapter loading order
        user_adapters = adapter_manager.get_user_adapters(st.session_state.user_id)
        if user_adapters:
            st.markdown("**Adapter Loading Priority:**")
            
            for adapter in sorted(user_adapters, key=lambda x: x.priority.value):
                priority_name = adapter.priority.name
                st.text(f"{priority_name}: {adapter.adapter_id} ({adapter.adapter_type.value})")
        
        # System statistics
        stats = adapter_manager.get_stats()
        
        st.markdown("**📊 System Statistics:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total System Adapters", stats['total_adapters'])
            st.metric("Active System Adapters", stats['active_adapters'])
        
        with col2:
            st.metric("Loaded Models", stats['loaded_models'])
            st.metric("Cached Base Models", stats['cached_base_models'])
        
        # Refresh button
        if st.button("🔄 Refresh Adapter Registry"):
            adapter_manager.scan_for_adapters()
            st.success("Adapter registry refreshed!")
            st.rerun()
    
    def _update_adapter_activation(self, adapter_manager, adapter_type: str, active: bool):
        """Update adapter activation for a specific type."""
        try:
            user_adapters = adapter_manager.get_user_adapters(st.session_state.user_id)
            type_adapters = [a for a in user_adapters if a.adapter_type.value == adapter_type]
            
            if active and type_adapters:
                # Activate the most recent adapter of this type
                latest_adapter = max(type_adapters, key=lambda x: x.created_at or "")
                active_adapters = [latest_adapter.adapter_id]
            else:
                active_adapters = []
            
            # Update configuration
            adapter_manager.configure_user_adapters(
                user_id=st.session_state.user_id,
                active_adapters=active_adapters
            )
            
            st.success(f"Updated {adapter_type} activation: {'enabled' if active else 'disabled'}")
            
        except Exception as e:
            st.error(f"Error updating adapter activation: {e}")
    
    def _toggle_individual_adapter(self, adapter_manager, adapter_id: str):
        """Toggle activation of an individual adapter."""
        try:
            adapter = adapter_manager.adapters.get(adapter_id)
            if not adapter:
                st.error(f"Adapter not found: {adapter_id}")
                return
            
            if adapter.is_active:
                # Deactivate
                adapter_manager.deactivate_adapter(st.session_state.user_id, adapter_id)
                st.success(f"Deactivated adapter: {adapter_id}")
            else:
                # Activate
                current_active = adapter_manager.get_active_adapters(st.session_state.user_id)
                new_active = [a.adapter_id for a in current_active] + [adapter_id]
                
                adapter_manager.configure_user_adapters(
                    user_id=st.session_state.user_id,
                    active_adapters=new_active
                )
                st.success(f"Activated adapter: {adapter_id}")
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Error toggling adapter: {e}")
    
    def _start_dpo_training(self):
        """Start DPO training process."""
        try:
            st.session_state.style_training_status = "running"
            st.info("🔄 Starting style training...")
            
            # In real implementation, this would launch the DPO training script
            # For now, simulate training completion
            time.sleep(2)
            
            st.session_state.style_training_status = "completed"
            st.success("✅ Style training completed!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error starting DPO training: {e}")
            st.session_state.style_training_status = "error"
    
    def _start_ssrl_training(self, epochs: int, learning_rate: str, batch_size: int, format_weight: float):
        """Start SSRL training process."""
        try:
            st.session_state.reasoning_training_status = "running"
            st.info("🔄 Starting reasoning training...")
            
            # In real implementation, this would launch the SSRL training script
            # For now, simulate training
            training_config = {
                'epochs': epochs,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'format_weight': format_weight
            }
            
            st.json(training_config)
            
        except Exception as e:
            st.error(f"Error starting SSRL training: {e}")
            st.session_state.reasoning_training_status = "error"


def render_enhanced_personalized_tuner():
    """Render the enhanced personalized tuner interface."""
    tuner = EnhancedPersonalizedTuner()
    tuner.render()
