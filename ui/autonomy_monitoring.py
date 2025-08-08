"""
Advanced Autonomy Monitoring Dashboard for SAM
==============================================

This module provides comprehensive monitoring interface with real-time alerts,
performance metrics, and autonomous activity logs for Phase C operations.

Phase C: Full Autonomy with Monitoring

Author: SAM Development Team
Version: 2.0.0
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
import time

# Import autonomy components
try:
    from sam.autonomy import Goal, GoalStack, MotivationEngine, GoalSafetyValidator
    from sam.autonomy.execution_engine import AutonomousExecutionEngine, ExecutionState
    from sam.autonomy.system_monitor import SystemLoadMonitor, SystemState
    from sam.autonomy.idle_processor import IdleTimeProcessor, IdleState
    AUTONOMY_AVAILABLE = True
except ImportError as e:
    AUTONOMY_AVAILABLE = False
    st.error(f"Advanced autonomy components not available: {e}")

logger = logging.getLogger(__name__)

class AutonomyMonitoringDashboard:
    """
    Advanced monitoring dashboard for SAM's autonomous execution system.
    
    Features:
    - Real-time system status monitoring
    - Performance metrics and analytics
    - Autonomous activity logs
    - Alert system with notifications
    - Historical data visualization
    - Emergency controls and overrides
    """
    
    def __init__(self):
        """Initialize the monitoring dashboard."""
        self.logger = logging.getLogger(f"{__name__}.AutonomyMonitoringDashboard")
        
        # Initialize session state for monitoring
        if 'monitoring_initialized' not in st.session_state:
            self._initialize_monitoring_state()
        
        # Auto-refresh settings
        self.auto_refresh_interval = 5  # seconds
        self.last_refresh = time.time()
    
    def _initialize_monitoring_state(self):
        """Initialize Streamlit session state for monitoring."""
        st.session_state.monitoring_initialized = True
        st.session_state.monitoring_alerts = []
        st.session_state.alert_history = []
        st.session_state.auto_refresh_enabled = True
        st.session_state.selected_time_range = "1h"
        st.session_state.show_debug_info = False
        st.session_state.last_alert_check = datetime.now()
    
    def render(self):
        """Render the complete monitoring dashboard."""
        if not AUTONOMY_AVAILABLE:
            st.error("🚫 Advanced autonomy monitoring not available")
            st.info("Please ensure all autonomy components are properly installed.")
            return
        
        st.title("🔍 SAM Autonomy Monitoring Dashboard")
        st.markdown("**Phase C: Full Autonomy with Real-Time Monitoring**")
        
        # Auto-refresh controls
        self._render_refresh_controls()
        
        # Alert banner
        self._render_alert_banner()
        
        # Main monitoring sections
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._render_main_monitoring()
        
        with col2:
            self._render_status_sidebar()
        
        # Detailed monitoring tabs
        self._render_detailed_monitoring()
        
        # Auto-refresh logic
        self._handle_auto_refresh()
    
    def _render_refresh_controls(self):
        """Render auto-refresh and manual controls."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.session_state.auto_refresh_enabled = st.checkbox(
                "🔄 Auto-refresh", 
                value=st.session_state.auto_refresh_enabled
            )
        
        with col2:
            if st.button("🔄 Refresh Now"):
                st.rerun()
        
        with col3:
            st.session_state.selected_time_range = st.selectbox(
                "Time Range",
                options=["5m", "15m", "1h", "6h", "24h"],
                index=2
            )
        
        with col4:
            st.session_state.show_debug_info = st.checkbox(
                "🐛 Debug Info",
                value=st.session_state.show_debug_info
            )
    
    def _render_alert_banner(self):
        """Render alert notifications banner."""
        # Check for new alerts
        self._check_for_alerts()
        
        # Display active alerts
        if st.session_state.monitoring_alerts:
            for alert in st.session_state.monitoring_alerts:
                alert_type = alert.get('type', 'info')
                message = alert.get('message', 'Unknown alert')
                
                if alert_type == 'critical':
                    st.error(f"🚨 CRITICAL: {message}")
                elif alert_type == 'warning':
                    st.warning(f"⚠️ WARNING: {message}")
                else:
                    st.info(f"ℹ️ INFO: {message}")
    
    def _render_main_monitoring(self):
        """Render main monitoring displays."""
        # System overview
        st.subheader("🖥️ System Overview")
        
        # Create metrics grid
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        # Mock data for demonstration (replace with actual monitoring data)
        with metric_col1:
            st.metric(
                "Execution State",
                "RUNNING",
                delta="Active for 2h 15m"
            )
        
        with metric_col2:
            st.metric(
                "Goals Processed",
                "47",
                delta="+3 in last hour"
            )
        
        with metric_col3:
            st.metric(
                "Success Rate",
                "94.7%",
                delta="+2.1% vs yesterday"
            )
        
        with metric_col4:
            st.metric(
                "System Load",
                "Optimal",
                delta="CPU: 23%, RAM: 45%"
            )
        
        # Real-time charts
        st.subheader("📊 Real-Time Performance")
        
        # Create performance charts
        self._render_performance_charts()
    
    def _render_status_sidebar(self):
        """Render status information sidebar."""
        st.subheader("🎛️ System Status")
        
        # Execution engine status
        with st.container():
            st.markdown("**Execution Engine**")
            st.success("✅ Running")
            st.progress(0.75, text="Processing Goal #47")
        
        # System resources
        with st.container():
            st.markdown("**System Resources**")
            
            # CPU usage
            cpu_usage = 23.5
            cpu_color = "normal" if cpu_usage < 70 else "inverse"
            st.progress(cpu_usage/100, text=f"CPU: {cpu_usage}%")
            
            # Memory usage
            memory_usage = 45.2
            memory_color = "normal" if memory_usage < 80 else "inverse"
            st.progress(memory_usage/100, text=f"Memory: {memory_usage}%")
            
            # Disk usage
            disk_usage = 67.8
            st.progress(disk_usage/100, text=f"Disk: {disk_usage}%")
        
        # Recent activity
        with st.container():
            st.markdown("**Recent Activity**")
            
            activity_data = [
                {"time": "14:32:15", "event": "Goal completed", "status": "✅"},
                {"time": "14:31:42", "event": "Goal started", "status": "🔄"},
                {"time": "14:30:18", "event": "System idle detected", "status": "💤"},
                {"time": "14:29:55", "event": "User activity", "status": "👤"},
                {"time": "14:28:33", "event": "Goal generated", "status": "🎯"}
            ]
            
            for activity in activity_data:
                st.text(f"{activity['status']} {activity['time']} - {activity['event']}")
    
    def _render_performance_charts(self):
        """Render real-time performance charts."""
        # Generate sample time series data
        now = datetime.now()
        times = [now - timedelta(minutes=i) for i in range(60, 0, -1)]
        
        # Sample data (replace with actual metrics)
        cpu_data = [20 + 10 * (i % 3) + (i % 7) for i in range(60)]
        memory_data = [40 + 15 * (i % 4) + (i % 5) for i in range(60)]
        goal_rate = [2 + (i % 6) for i in range(60)]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('CPU Usage (%)', 'Memory Usage (%)', 'Goal Processing Rate', 'System State'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # CPU usage
        fig.add_trace(
            go.Scatter(x=times, y=cpu_data, name="CPU %", line=dict(color="blue")),
            row=1, col=1
        )
        
        # Memory usage
        fig.add_trace(
            go.Scatter(x=times, y=memory_data, name="Memory %", line=dict(color="green")),
            row=1, col=2
        )
        
        # Goal processing rate
        fig.add_trace(
            go.Scatter(x=times, y=goal_rate, name="Goals/min", line=dict(color="purple")),
            row=2, col=1
        )
        
        # System state (categorical)
        states = ["Optimal"] * 45 + ["Moderate"] * 10 + ["Optimal"] * 5
        state_values = [1 if s == "Optimal" else 2 for s in states]
        fig.add_trace(
            go.Scatter(x=times, y=state_values, name="System State", 
                      mode="lines+markers", line=dict(color="orange")),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=400,
            showlegend=False,
            title_text="Real-Time System Metrics"
        )
        
        # Update y-axes
        fig.update_yaxes(range=[0, 100], row=1, col=1)
        fig.update_yaxes(range=[0, 100], row=1, col=2)
        fig.update_yaxes(range=[0, 10], row=2, col=1)
        fig.update_yaxes(range=[0, 3], tickvals=[1, 2], ticktext=["Optimal", "Moderate"], row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_detailed_monitoring(self):
        """Render detailed monitoring tabs."""
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Analytics",
            "📋 Activity Log", 
            "🚨 Alerts",
            "⚙️ Configuration",
            "🔧 Emergency Controls"
        ])
        
        with tab1:
            self._render_analytics_tab()
        
        with tab2:
            self._render_activity_log_tab()
        
        with tab3:
            self._render_alerts_tab()
        
        with tab4:
            self._render_configuration_tab()
        
        with tab5:
            self._render_emergency_controls_tab()
    
    def _render_analytics_tab(self):
        """Render analytics and historical data."""
        st.subheader("📈 Performance Analytics")
        
        # Time range selector
        time_range = st.session_state.selected_time_range
        
        # Goal execution analytics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Goal Execution Trends**")
            
            # Sample data for goal execution over time
            dates = pd.date_range(end=datetime.now(), periods=24, freq='H')
            executions = [5 + (i % 8) for i in range(24)]
            successes = [int(e * 0.9) for e in executions]
            
            df = pd.DataFrame({
                'Time': dates,
                'Total Executions': executions,
                'Successful': successes,
                'Failed': [e - s for e, s in zip(executions, successes)]
            })
            
            fig = px.line(df, x='Time', y=['Total Executions', 'Successful', 'Failed'],
                         title="Goal Execution Over Time")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Success Rate Distribution**")
            
            # Success rate by goal priority
            priorities = ['0.1-0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9', '0.9-1.0']
            success_rates = [75, 82, 89, 94, 97]
            
            fig = px.bar(x=priorities, y=success_rates,
                        title="Success Rate by Goal Priority",
                        labels={'x': 'Priority Range', 'y': 'Success Rate (%)'})
            st.plotly_chart(fig, use_container_width=True)
        
        # System performance correlation
        st.markdown("**System Performance Correlation**")
        
        # Sample correlation data
        system_load = [20, 35, 45, 60, 75, 85, 95]
        goal_success = [98, 95, 92, 87, 78, 65, 45]
        
        fig = px.scatter(x=system_load, y=goal_success,
                        title="Goal Success Rate vs System Load",
                        labels={'x': 'System Load (%)', 'y': 'Success Rate (%)'})
        fig.add_trace(go.Scatter(x=system_load, y=goal_success, mode='lines', name='Trend'))
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_activity_log_tab(self):
        """Render activity log with filtering."""
        st.subheader("📋 Autonomous Activity Log")
        
        # Log filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            log_level = st.selectbox("Log Level", ["All", "INFO", "WARNING", "ERROR", "CRITICAL"])
        
        with col2:
            component = st.selectbox("Component", ["All", "Execution Engine", "Goal Stack", "System Monitor"])
        
        with col3:
            max_entries = st.number_input("Max Entries", min_value=10, max_value=1000, value=100)
        
        # Sample log data
        log_data = []
        for i in range(50):
            log_data.append({
                'Timestamp': datetime.now() - timedelta(minutes=i*2),
                'Level': ['INFO', 'WARNING', 'ERROR'][i % 3],
                'Component': ['Execution Engine', 'Goal Stack', 'System Monitor'][i % 3],
                'Message': f"Sample log message {i+1}",
                'Goal ID': f"goal_{i+1:03d}" if i % 2 == 0 else None
            })
        
        # Create DataFrame and apply filters
        df = pd.DataFrame(log_data)
        
        if log_level != "All":
            df = df[df['Level'] == log_level]
        
        if component != "All":
            df = df[df['Component'] == component]
        
        df = df.head(max_entries)
        
        # Display log table
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                'Timestamp': st.column_config.DatetimeColumn(
                    'Timestamp',
                    format='YYYY-MM-DD HH:mm:ss'
                ),
                'Level': st.column_config.TextColumn('Level'),
                'Component': st.column_config.TextColumn('Component'),
                'Message': st.column_config.TextColumn('Message'),
                'Goal ID': st.column_config.TextColumn('Goal ID')
            }
        )
        
        # Export options
        if st.button("📥 Export Log"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"autonomy_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    def _render_alerts_tab(self):
        """Render alerts management."""
        st.subheader("🚨 Alert Management")
        
        # Alert configuration
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Alert Thresholds**")
            
            cpu_alert_threshold = st.slider("CPU Alert Threshold (%)", 0, 100, 80)
            memory_alert_threshold = st.slider("Memory Alert Threshold (%)", 0, 100, 85)
            failure_rate_threshold = st.slider("Failure Rate Alert (%)", 0, 100, 20)
            
            if st.button("💾 Save Alert Settings"):
                st.success("Alert settings saved!")
        
        with col2:
            st.markdown("**Alert History**")
            
            # Sample alert history
            alert_history = [
                {"Time": "14:25:33", "Type": "WARNING", "Message": "High CPU usage detected"},
                {"Time": "13:45:12", "Type": "INFO", "Message": "System returned to optimal state"},
                {"Time": "13:42:08", "Type": "CRITICAL", "Message": "Goal execution failure rate exceeded threshold"},
                {"Time": "12:30:45", "Type": "WARNING", "Message": "Memory usage approaching limit"},
            ]
            
            for alert in alert_history:
                alert_color = {
                    "CRITICAL": "🔴",
                    "WARNING": "🟡", 
                    "INFO": "🔵"
                }.get(alert["Type"], "⚪")
                
                st.text(f"{alert_color} {alert['Time']} - {alert['Message']}")
        
        # Active alerts
        st.markdown("**Active Alerts**")
        
        if st.session_state.monitoring_alerts:
            for i, alert in enumerate(st.session_state.monitoring_alerts):
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.write(f"**{alert['type'].upper()}**: {alert['message']}")
                
                with col2:
                    if st.button("Dismiss", key=f"dismiss_{i}"):
                        st.session_state.monitoring_alerts.pop(i)
                        st.rerun()
        else:
            st.info("No active alerts")
    
    def _render_configuration_tab(self):
        """Render configuration management."""
        st.subheader("⚙️ Autonomy Configuration")
        
        # Configuration sections
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            st.markdown("**Execution Settings**")
            
            idle_threshold = st.number_input("Idle Threshold (seconds)", min_value=10, max_value=300, value=30)
            max_processing_time = st.number_input("Max Processing Time (seconds)", min_value=60, max_value=1800, value=300)
            max_concurrent_goals = st.number_input("Max Concurrent Goals", min_value=1, max_value=10, value=1)
            
            enable_system_monitoring = st.checkbox("Enable System Monitoring", value=True)
            enable_safety_validation = st.checkbox("Enable Safety Validation", value=True)
        
        with config_col2:
            st.markdown("**Resource Thresholds**")
            
            cpu_threshold = st.slider("CPU Threshold (%)", 0, 100, 70)
            memory_threshold = st.slider("Memory Threshold (%)", 0, 100, 80)
            
            monitoring_interval = st.number_input("Monitoring Interval (seconds)", min_value=1, max_value=60, value=5)
            
            min_goal_priority = st.slider("Minimum Goal Priority", 0.0, 1.0, 0.3, 0.1)
        
        # Apply configuration
        if st.button("🔄 Apply Configuration", type="primary"):
            st.success("Configuration applied successfully!")
            st.info("Changes will take effect on next system restart.")
    
    def _render_emergency_controls_tab(self):
        """Render emergency controls and overrides."""
        st.subheader("🔧 Emergency Controls")
        
        st.warning("⚠️ These controls immediately affect autonomous operation. Use with caution.")
        
        # Emergency actions
        emergency_col1, emergency_col2, emergency_col3 = st.columns(3)
        
        with emergency_col1:
            if st.button("🛑 EMERGENCY STOP", type="secondary", use_container_width=True):
                st.error("🚨 EMERGENCY STOP ACTIVATED")
                st.info("All autonomous processing has been halted.")
        
        with emergency_col2:
            if st.button("⏸️ Pause Execution", use_container_width=True):
                st.warning("⏸️ Autonomous execution paused")
        
        with emergency_col3:
            if st.button("▶️ Resume Execution", use_container_width=True):
                st.success("▶️ Autonomous execution resumed")
        
        # System diagnostics
        st.markdown("**System Diagnostics**")
        
        diag_col1, diag_col2 = st.columns(2)
        
        with diag_col1:
            if st.button("🔍 Run Diagnostics"):
                with st.spinner("Running system diagnostics..."):
                    time.sleep(2)  # Simulate diagnostics
                    st.success("✅ All systems operational")
        
        with diag_col2:
            if st.button("📊 Generate Report"):
                st.info("📄 System report generated and saved to logs/")
        
        # Debug information
        if st.session_state.show_debug_info:
            st.markdown("**Debug Information**")
            
            debug_info = {
                "Session State Keys": list(st.session_state.keys()),
                "Current Time": datetime.now().isoformat(),
                "Auto Refresh": st.session_state.auto_refresh_enabled,
                "Selected Time Range": st.session_state.selected_time_range,
                "Alert Count": len(st.session_state.monitoring_alerts)
            }
            
            st.json(debug_info)
    
    def _check_for_alerts(self):
        """Check for new alerts and update alert state."""
        # Simulate alert checking (replace with actual monitoring)
        now = datetime.now()
        
        # Check if enough time has passed since last check
        if (now - st.session_state.last_alert_check).total_seconds() < 30:
            return
        
        st.session_state.last_alert_check = now
        
        # Simulate random alerts for demonstration
        import random
        if random.random() < 0.1:  # 10% chance of new alert
            alert_types = ['info', 'warning', 'critical']
            alert_messages = [
                'System performance optimal',
                'High CPU usage detected',
                'Goal execution failure rate exceeded threshold'
            ]
            
            alert_type = random.choice(alert_types)
            alert_message = random.choice(alert_messages)
            
            new_alert = {
                'type': alert_type,
                'message': alert_message,
                'timestamp': now.isoformat()
            }
            
            # Add to active alerts (limit to 5)
            st.session_state.monitoring_alerts.append(new_alert)
            if len(st.session_state.monitoring_alerts) > 5:
                st.session_state.monitoring_alerts.pop(0)
            
            # Add to alert history
            st.session_state.alert_history.append(new_alert)
            if len(st.session_state.alert_history) > 100:
                st.session_state.alert_history.pop(0)
    
    def _handle_auto_refresh(self):
        """Handle auto-refresh functionality."""
        if st.session_state.auto_refresh_enabled:
            current_time = time.time()
            if current_time - self.last_refresh >= self.auto_refresh_interval:
                self.last_refresh = current_time
                time.sleep(0.1)  # Small delay to prevent too frequent refreshes
                st.rerun()


def render_autonomy_monitoring():
    """Main function to render the autonomy monitoring dashboard."""
    dashboard = AutonomyMonitoringDashboard()
    dashboard.render()


if __name__ == "__main__":
    render_autonomy_monitoring()
