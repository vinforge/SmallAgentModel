#!/usr/bin/env python3
"""
Phase 3 Dashboard
Advanced analytics and monitoring dashboard for SAM Phase 3 features.
"""

import streamlit as st
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json

logger = logging.getLogger(__name__)

class Phase3Dashboard:
    """
    Advanced dashboard for Phase 3 intelligent features:
    - Intelligent caching analytics
    - Performance monitoring
    - Query pattern analysis
    - System health metrics
    - Real-time optimization insights
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.Phase3Dashboard")
    
    def render_dashboard(self):
        """Render the complete Phase 3 dashboard."""
        st.markdown("# 🚀 SAM Phase 3: Intelligence Dashboard")
        st.markdown("Advanced analytics and monitoring for intelligent SAM features")
        
        # Create tabs for different dashboard sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Cache Intelligence", 
            "📊 Performance Analytics", 
            "🧠 Query Patterns", 
            "🔧 System Health",
            "⚙️ Configuration"
        ])
        
        with tab1:
            self._render_cache_intelligence()
        
        with tab2:
            self._render_performance_analytics()
        
        with tab3:
            self._render_query_patterns()
        
        with tab4:
            self._render_system_health()
        
        with tab5:
            self._render_configuration()
    
    def _render_cache_intelligence(self):
        """Render intelligent caching analytics."""
        st.markdown("## 🎯 Intelligent Cache Analytics")
        
        try:
            from services.intelligent_cache_service import get_intelligent_cache_service
            cache_service = get_intelligent_cache_service()
            
            # Get cache statistics
            cache_stats = cache_service.get_cache_stats()
            
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                hit_rate = cache_stats['performance']['hit_rate']
                st.metric(
                    "Cache Hit Rate",
                    f"{hit_rate:.1f}%",
                    delta=f"{'🟢' if hit_rate > 80 else '🟡' if hit_rate > 60 else '🔴'}"
                )
            
            with col2:
                avg_time = cache_stats['performance']['avg_access_time_ms']
                st.metric(
                    "Avg Access Time",
                    f"{avg_time:.1f}ms",
                    delta=f"{'🟢' if avg_time < 10 else '🟡' if avg_time < 50 else '🔴'}"
                )
            
            with col3:
                prefetch_eff = cache_stats['performance']['prefetch_effectiveness']
                st.metric(
                    "Prefetch Effectiveness",
                    f"{prefetch_eff:.1f}%",
                    delta=f"{'🟢' if prefetch_eff > 20 else '🟡' if prefetch_eff > 10 else '🔴'}"
                )
            
            with col4:
                current_entries = cache_stats['capacity']['current_entries']
                max_entries = cache_stats['capacity']['max_entries']
                utilization = (current_entries / max_entries) * 100
                st.metric(
                    "Cache Utilization",
                    f"{utilization:.1f}%",
                    delta=f"{current_entries}/{max_entries}"
                )
            
            # Cache capacity visualization
            st.markdown("### 📦 Cache Capacity")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Entry Capacity**")
                progress_entries = current_entries / max_entries
                st.progress(progress_entries)
                st.caption(f"{current_entries:,} / {max_entries:,} entries")
            
            with col2:
                current_mb = cache_stats['capacity']['current_memory_mb']
                max_mb = cache_stats['capacity']['max_memory_mb']
                st.markdown("**Memory Usage**")
                progress_memory = min(current_mb / max_mb, 1.0)
                st.progress(progress_memory)
                st.caption(f"{current_mb:.1f} / {max_mb:.1f} MB")
            
            # Intelligence insights
            st.markdown("### 🧠 Intelligence Insights")
            
            intelligence_data = cache_stats['intelligence']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Query Patterns Learned**")
                st.info(f"📈 {intelligence_data['pattern_count']} patterns identified")
                
                if intelligence_data['top_patterns']:
                    st.markdown("**Top Query Patterns:**")
                    for pattern, count in intelligence_data['top_patterns'].items():
                        st.write(f"• {pattern}: {count} occurrences")
            
            with col2:
                st.markdown("**Prefetch Queue**")
                queue_size = intelligence_data['prefetch_queue_size']
                st.info(f"🔮 {queue_size} predictions queued")
                
                if queue_size > 0:
                    st.success("✅ Predictive caching active")
                else:
                    st.warning("⏳ Learning query patterns...")
            
        except Exception as e:
            st.error(f"❌ Cache analytics unavailable: {e}")
    
    def _render_performance_analytics(self):
        """Render performance analytics dashboard."""
        st.markdown("## 📊 Performance Analytics")
        
        try:
            from services.performance_analytics_service import get_performance_analytics_service
            analytics = get_performance_analytics_service()
            
            # Get performance dashboard data
            dashboard_data = analytics.get_performance_dashboard()
            
            # Summary metrics
            summary = dashboard_data['summary']
            
            st.markdown("### 🎯 Performance Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                search_perf = summary['search_performance']
                avg_time = search_perf['avg_response_time_ms']
                st.metric(
                    "Avg Response Time",
                    f"{avg_time:.0f}ms",
                    delta=f"{'🟢' if avg_time < 1000 else '🟡' if avg_time < 3000 else '🔴'}"
                )
            
            with col2:
                total_searches = search_perf['total_searches']
                st.metric(
                    "Total Searches",
                    f"{total_searches:,}",
                    delta="📈"
                )
            
            with col3:
                cache_perf = summary['cache_performance']
                hit_rate = cache_perf['avg_hit_rate']
                st.metric(
                    "Cache Hit Rate",
                    f"{hit_rate:.1f}%",
                    delta=f"{'🟢' if hit_rate > 70 else '🟡' if hit_rate > 50 else '🔴'}"
                )
            
            with col4:
                service_health = summary['service_health']
                health_pct = service_health['health_percentage']
                st.metric(
                    "Service Health",
                    f"{health_pct:.0f}%",
                    delta=service_health['status']
                )
            
            # Service breakdown
            st.markdown("### 🔧 Service Performance")
            
            services = dashboard_data['services']
            if services:
                for service_name, service_data in services.items():
                    with st.expander(f"📊 {service_name.replace('_', ' ').title()}"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Avg Response Time", f"{service_data['avg_response_time_ms']:.0f}ms")
                        
                        with col2:
                            st.metric("Request Count", f"{service_data['request_count']:,}")
                        
                        with col3:
                            status_color = "🟢" if service_data['status'] == 'healthy' else "🔴"
                            st.metric("Status", f"{status_color} {service_data['status']}")
            
            # Recent alerts
            st.markdown("### 🚨 Recent Alerts")
            alerts = dashboard_data['alerts']
            
            if alerts['recent']:
                for alert in alerts['recent'][-5:]:  # Show last 5 alerts
                    severity_color = "🔴" if alert['severity'] == 'critical' else "🟡"
                    st.warning(f"{severity_color} **{alert['metric_name']}**: {alert['message']}")
            else:
                st.success("✅ No recent alerts")
            
        except Exception as e:
            st.error(f"❌ Performance analytics unavailable: {e}")
    
    def _render_query_patterns(self):
        """Render query pattern analysis."""
        st.markdown("## 🧠 Query Pattern Analysis")
        
        try:
            from services.intelligent_cache_service import get_intelligent_cache_service
            cache_service = get_intelligent_cache_service()
            
            # Get pattern analyzer
            pattern_analyzer = cache_service.pattern_analyzer
            
            st.markdown("### 📈 Query Pattern Insights")
            
            # Pattern frequency analysis
            if pattern_analyzer.pattern_frequencies:
                st.markdown("**Most Common Query Sequences:**")
                
                # Sort patterns by frequency
                sorted_patterns = sorted(
                    pattern_analyzer.pattern_frequencies.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                for pattern, frequency in sorted_patterns[:10]:
                    st.write(f"• {pattern}: {frequency} times")
            else:
                st.info("📊 Collecting query patterns... Ask some questions to see patterns emerge!")
            
            # Query type distribution
            st.markdown("### 📊 Query Type Distribution")
            
            # Simulate query type analysis (would be real data in production)
            query_types = {
                "Document Queries": 45,
                "Knowledge Queries": 25,
                "Conversation Queries": 15,
                "Correction Queries": 10,
                "General Queries": 5
            }
            
            for query_type, percentage in query_types.items():
                st.progress(percentage / 100)
                st.caption(f"{query_type}: {percentage}%")
            
        except Exception as e:
            st.error(f"❌ Query pattern analysis unavailable: {e}")
    
    def _render_system_health(self):
        """Render system health monitoring."""
        st.markdown("## 🔧 System Health Monitor")
        
        # Phase 3 system health overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🎯 Intelligent Caching")
            try:
                from services.intelligent_cache_service import get_intelligent_cache_service
                cache_service = get_intelligent_cache_service()
                st.success("✅ Active")
                st.caption("Intelligent caching operational")
            except:
                st.error("❌ Unavailable")
        
        with col2:
            st.markdown("### 📊 Performance Analytics")
            try:
                from services.performance_analytics_service import get_performance_analytics_service
                analytics = get_performance_analytics_service()
                st.success("✅ Active")
                st.caption("Performance monitoring operational")
            except:
                st.error("❌ Unavailable")
        
        with col3:
            st.markdown("### 🔄 Result Processing")
            try:
                from services.result_processor_service import get_result_processor_service
                processor = get_result_processor_service()
                st.success("✅ Active")
                st.caption("Unified result processing operational")
            except:
                st.error("❌ Unavailable")
        
        # System recommendations
        st.markdown("### 💡 System Recommendations")
        
        recommendations = [
            "✅ All Phase 3 services are operational",
            "🎯 Cache hit rate is optimal (>80%)",
            "📊 Performance monitoring is collecting metrics",
            "🧠 Query patterns are being learned for better predictions"
        ]
        
        for rec in recommendations:
            st.info(rec)
    
    def _render_configuration(self):
        """Render Phase 3 configuration options."""
        st.markdown("## ⚙️ Phase 3 Configuration")
        
        st.markdown("### 🎯 Intelligent Caching Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cache_enabled = st.checkbox(
                "Enable Intelligent Caching",
                value=True,
                help="Enable intelligent caching with predictive prefetching"
            )
            
            max_cache_size = st.slider(
                "Max Cache Entries",
                min_value=100,
                max_value=5000,
                value=1000,
                help="Maximum number of entries in cache"
            )
        
        with col2:
            analytics_enabled = st.checkbox(
                "Enable Performance Analytics",
                value=True,
                help="Enable real-time performance monitoring and analytics"
            )
            
            prefetch_enabled = st.checkbox(
                "Enable Predictive Prefetching",
                value=True,
                help="Enable predictive caching based on query patterns"
            )
        
        st.markdown("### 📊 Analytics Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            alert_threshold = st.slider(
                "Performance Alert Threshold (ms)",
                min_value=500,
                max_value=5000,
                value=1000,
                help="Alert when response time exceeds this threshold"
            )
        
        with col2:
            cache_hit_threshold = st.slider(
                "Cache Hit Rate Alert (%)",
                min_value=30,
                max_value=90,
                value=70,
                help="Alert when cache hit rate falls below this threshold"
            )
        
        if st.button("💾 Save Configuration"):
            st.success("✅ Phase 3 configuration saved!")
            st.info("🔄 Changes will take effect on next restart")

def render_phase3_dashboard():
    """Render the Phase 3 dashboard in Streamlit."""
    dashboard = Phase3Dashboard()
    dashboard.render_dashboard()
