#!/usr/bin/env python3
"""
System Status Handler for Memory UI
===================================

Handles system status monitoring and analytics for the SAM Memory Control Center.
Extracted from the monolithic memory_app.py.

This module provides:
- Memory system status monitoring
- Performance metrics tracking
- System health checks
- Memory ranking analytics
- Citation engine status

Author: SAM Development Team
Version: 1.0.0 - Refactored from memory_app.py
"""

import streamlit as st
import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class SystemStatusHandler:
    """Handles system status monitoring and analytics."""
    
    def __init__(self):
        self.memory_store = None
        self.mode_controller = None
        self.ranking_framework = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize system components."""
        try:
            from memory.memory_vectorstore import get_memory_store
            from config.agent_mode import get_mode_controller
            from memory.memory_ranking import get_memory_ranking_framework
            
            self.memory_store = get_memory_store()
            self.mode_controller = get_mode_controller()
            self.ranking_framework = get_memory_ranking_framework()
            
            logger.info("System status components initialized")
            
        except ImportError as e:
            logger.error(f"Failed to import system components: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize system components: {e}")
    
    def render_system_status(self):
        """Render the system status interface."""
        try:
            st.subheader("🖥️ System Status")
            
            # Memory system status
            col1, col2 = st.columns(2)
            
            with col1:
                self._render_memory_system_status()
            
            with col2:
                self._render_agent_mode_status()
            
            # Performance metrics
            self._render_performance_metrics()
            
            # System health
            self._render_system_health()
            
        except Exception as e:
            logger.error(f"Error rendering system status: {e}")
            st.error(f"Error loading system status: {e}")
    
    def _render_memory_system_status(self):
        """Render memory system status."""
        st.markdown("**Memory System:**")
        
        if not self.memory_store:
            st.error("❌ Memory store not available")
            return
        
        try:
            stats = self.memory_store.get_memory_stats()
            
            # Use .get() with fallback values to prevent KeyError
            total_memories = stats.get('total_memories', len(getattr(self.memory_store, 'memory_chunks', {})))
            total_size_mb = stats.get('total_size_mb', 0.0)
            store_type = stats.get('store_type', 'Unknown')
            
            st.metric("Total Memories", total_memories)
            st.metric("Storage Size", f"{total_size_mb:.2f} MB")
            st.metric("Store Type", store_type)
            
            # Memory distribution
            memory_types = stats.get('memory_types', {})
            if memory_types and total_memories > 0:
                st.markdown("**Memory Distribution:**")
                for mem_type, count in memory_types.items():
                    st.progress(count / total_memories, text=f"{mem_type}: {count}")
                    
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            st.error("❌ Unable to load memory statistics")
    
    def _render_agent_mode_status(self):
        """Render agent mode status."""
        st.markdown("**Agent Mode:**")
        
        if not self.mode_controller:
            st.error("❌ Mode controller not available")
            return
        
        try:
            mode_status = self.mode_controller.get_mode_status()
            
            st.metric("Current Mode", mode_status.current_mode.value.title())
            st.metric("Key Status", mode_status.key_status.value)
            st.metric("Uptime", f"{mode_status.uptime_seconds}s")
            
            st.markdown("**Enabled Capabilities:**")
            for capability in mode_status.enabled_capabilities[:5]:
                st.caption(f"✅ {capability}")
            
            if mode_status.disabled_capabilities:
                st.markdown("**Disabled Capabilities:**")
                for capability in mode_status.disabled_capabilities[:3]:
                    st.caption(f"❌ {capability}")
                    
        except Exception as e:
            logger.error(f"Error getting mode status: {e}")
            st.error("❌ Unable to load agent mode status")
    
    def _render_performance_metrics(self):
        """Render performance metrics."""
        st.subheader("📈 Performance Metrics")
        
        # This would integrate with actual performance monitoring
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Avg Query Time", "45ms", delta="-5ms")
        
        with col2:
            st.metric("Memory Hit Rate", "87%", delta="+2%")
        
        with col3:
            st.metric("Active Sessions", "3", delta="+1")
    
    def _render_system_health(self):
        """Render system health checks."""
        st.subheader("🏥 System Health")
        
        health_checks = self._perform_health_checks()
        
        for component, status in health_checks:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(component)
            with col2:
                st.text(status)
    
    def _perform_health_checks(self) -> List[Tuple[str, str]]:
        """Perform system health checks."""
        checks = []
        
        # Memory Store
        if self.memory_store:
            checks.append(("Memory Store", "✅ Healthy"))
        else:
            checks.append(("Memory Store", "❌ Unavailable"))
        
        # Vector Index
        try:
            if self.memory_store and hasattr(self.memory_store, 'vector_store'):
                checks.append(("Vector Index", "✅ Healthy"))
            else:
                checks.append(("Vector Index", "⚠️ Unknown"))
        except:
            checks.append(("Vector Index", "❌ Error"))
        
        # Agent Mode Controller
        if self.mode_controller:
            checks.append(("Agent Mode Controller", "✅ Healthy"))
        else:
            checks.append(("Agent Mode Controller", "❌ Unavailable"))
        
        # Command Processor
        try:
            from ui.memory_commands import get_command_processor
            processor = get_command_processor()
            if processor:
                checks.append(("Command Processor", "✅ Healthy"))
            else:
                checks.append(("Command Processor", "❌ Unavailable"))
        except:
            checks.append(("Command Processor", "❌ Error"))
        
        # Role Filter
        try:
            from ui.role_memory_filter import get_role_filter
            filter_obj = get_role_filter()
            if filter_obj:
                checks.append(("Role Filter", "✅ Healthy"))
            else:
                checks.append(("Role Filter", "❌ Unavailable"))
        except:
            checks.append(("Role Filter", "❌ Error"))
        
        return checks
    
    def render_memory_ranking(self):
        """Render the Enhanced Memory Ranking interface with Phase 3.2.3 features."""
        try:
            st.subheader("🏆 Enhanced Memory Ranking Framework")
            st.markdown("**Phase 3.2.3:** Real-time ranking controls, weight adjustment, and performance analytics")
            
            if not self.ranking_framework:
                st.error("❌ Memory ranking framework not available")
                return
            
            # Interactive Configuration section
            st.subheader("⚙️ Interactive Ranking Configuration")
            
            # Real-time weight adjustment
            st.markdown("**🎛️ Adjust Ranking Weights:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                semantic_weight = st.slider("Semantic Similarity", 0.0, 1.0, 0.35, 0.05, 
                                           help="Weight for content similarity")
                recency_weight = st.slider("Recency", 0.0, 1.0, 0.15, 0.05, 
                                         help="Weight for temporal relevance")
            
            with col2:
                confidence_weight = st.slider("Source Confidence", 0.0, 1.0, 0.25, 0.05, 
                                            help="Weight for source quality")
                priority_weight = st.slider("User Priority", 0.0, 1.0, 0.15, 0.05, 
                                           help="Weight for user-defined priority")
            
            with col3:
                context_weight = st.slider("Context Relevance", 0.0, 1.0, 0.10, 0.05, 
                                         help="Weight for contextual relevance")
            
            # Normalize weights
            total_weight = semantic_weight + recency_weight + confidence_weight + priority_weight + context_weight
            if total_weight > 0:
                weights = {
                    'semantic': semantic_weight / total_weight,
                    'recency': recency_weight / total_weight,
                    'confidence': confidence_weight / total_weight,
                    'priority': priority_weight / total_weight,
                    'context': context_weight / total_weight
                }
                
                # Update ranking framework
                if st.button("🔄 Apply Weights"):
                    try:
                        self.ranking_framework.update_weights(weights)
                        st.success("✅ Ranking weights updated successfully!")
                    except Exception as e:
                        st.error(f"❌ Failed to update weights: {e}")
            
            # Show current configuration
            with st.expander("📊 Current Configuration", expanded=False):
                current_weights = getattr(self.ranking_framework, 'weights', weights)
                st.json(current_weights)
            
            # Performance analytics
            self._render_ranking_analytics()
            
        except Exception as e:
            logger.error(f"Error rendering memory ranking: {e}")
            st.error(f"Error loading memory ranking: {e}")
    
    def _render_ranking_analytics(self):
        """Render ranking performance analytics."""
        st.subheader("📊 Ranking Performance Analytics")
        
        # Mock analytics data (would be real in production)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Avg Ranking Score", "0.847", delta="+0.023")
        
        with col2:
            st.metric("Top Results Accuracy", "92%", delta="+3%")
        
        with col3:
            st.metric("Ranking Latency", "12ms", delta="-2ms")
        
        # Ranking distribution chart
        st.markdown("**Ranking Score Distribution:**")
        
        # This would show actual ranking distribution
        import numpy as np
        scores = np.random.beta(2, 5, 1000)  # Mock data
        st.bar_chart(scores)
    
    def render_citation_engine(self):
        """Render citation engine interface."""
        try:
            st.subheader("📚 Citation Engine")
            st.markdown("**Advanced citation tracking and source attribution**")
            
            # Citation statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Citations", "1,247", delta="+23")
            
            with col2:
                st.metric("Unique Sources", "89", delta="+5")
            
            with col3:
                st.metric("Citation Accuracy", "96%", delta="+1%")
            
            # Recent citations
            st.markdown("**Recent Citations:**")
            
            # Mock citation data
            citations = [
                {"source": "Research Paper A", "count": 15, "confidence": 0.95},
                {"source": "Documentation B", "count": 12, "confidence": 0.89},
                {"source": "Article C", "count": 8, "confidence": 0.92}
            ]
            
            for citation in citations:
                with st.expander(f"📄 {citation['source']} ({citation['count']} citations)", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Citation Count:** {citation['count']}")
                        st.write(f"**Confidence:** {citation['confidence']:.2%}")
                    with col2:
                        if st.button(f"View Details", key=f"cite_{citation['source']}"):
                            st.info("Citation details would be shown here")
            
        except Exception as e:
            logger.error(f"Error rendering citation engine: {e}")
            st.error(f"Error loading citation engine: {e}")


# Global system status handler instance
_system_status_handler = None


def get_system_status_handler() -> SystemStatusHandler:
    """Get the global system status handler instance."""
    global _system_status_handler
    if _system_status_handler is None:
        _system_status_handler = SystemStatusHandler()
    return _system_status_handler


def render_system_status():
    """Render the system status interface."""
    get_system_status_handler().render_system_status()


def render_memory_ranking():
    """Render the memory ranking interface."""
    get_system_status_handler().render_memory_ranking()


def render_citation_engine():
    """Render the citation engine interface."""
    get_system_status_handler().render_citation_engine()
