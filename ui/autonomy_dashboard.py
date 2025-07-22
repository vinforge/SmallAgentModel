"""
SAM Autonomy Dashboard UI
========================

This module provides the user interface for SAM's Goal & Motivation Engine,
allowing manual goal management, monitoring, and testing of autonomous behavior.

Phase B: Cautious Integration & Manual Triggers

Author: SAM Development Team
Version: 2.0.0
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging

# Import autonomy components
try:
    from sam.autonomy import Goal, GoalStack, MotivationEngine, GoalSafetyValidator
    from sam.orchestration.uif import SAM_UIF, UIFStatus
    AUTONOMY_AVAILABLE = True
except ImportError as e:
    AUTONOMY_AVAILABLE = False
    st.error(f"Autonomy components not available: {e}")

logger = logging.getLogger(__name__)

class AutonomyDashboard:
    """
    Streamlit-based dashboard for SAM's autonomy system.

    Features:
    - Goal management and monitoring
    - Manual goal triggers
    - Safety validation controls
    - Statistics and analytics
    - Emergency controls
    """

    def __init__(self):
        """Initialize the autonomy dashboard."""
        self.logger = logging.getLogger(f"{__name__}.AutonomyDashboard")

        # Initialize session state
        if 'autonomy_initialized' not in st.session_state:
            self._initialize_session_state()

        # Initialize autonomy components
        self._initialize_autonomy_components()

    def _initialize_session_state(self):
        """Initialize Streamlit session state for autonomy dashboard."""
        st.session_state.autonomy_initialized = True
        st.session_state.autonomy_paused = False
        st.session_state.selected_goal_id = None
        st.session_state.show_goal_details = False
        st.session_state.last_refresh = datetime.now()
        st.session_state.manual_trigger_count = 0

    def _initialize_autonomy_components(self):
        """Initialize autonomy system components."""
        if not AUTONOMY_AVAILABLE:
            return

        try:
            # Initialize components if not already in session state
            if 'goal_stack' not in st.session_state:
                safety_validator = GoalSafetyValidator()
                goal_stack = GoalStack(
                    db_path="memory/autonomy_goals.db",
                    safety_validator=safety_validator
                )
                motivation_engine = MotivationEngine(
                    goal_stack=goal_stack,
                    safety_validator=safety_validator
                )

                st.session_state.goal_stack = goal_stack
                st.session_state.motivation_engine = motivation_engine
                st.session_state.safety_validator = safety_validator

                self.logger.info("Autonomy components initialized successfully")

        except Exception as e:
            st.error(f"Failed to initialize autonomy components: {e}")
            self.logger.error(f"Autonomy initialization failed: {e}")

    def render(self):
        """Render the complete autonomy dashboard."""
        if not AUTONOMY_AVAILABLE:
            st.error("🚫 Autonomy system not available")
            st.info("Please ensure the autonomy module is properly installed.")
            return

        st.title("🤖 SAM Autonomy Dashboard")
        st.markdown("**Phase B: Cautious Integration & Manual Triggers**")

        # Emergency controls at the top
        self._render_emergency_controls()

        # Main dashboard tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Goal Management",
            "🧠 Manual Triggers",
            "📊 Analytics",
            "🛡️ Safety Monitor",
            "⚙️ Settings"
        ])

        with tab1:
            self._render_goal_management()

        with tab2:
            self._render_manual_triggers()

        with tab3:
            self._render_analytics()

        with tab4:
            self._render_safety_monitor()

        with tab5:
            self._render_settings()

    def _render_emergency_controls(self):
        """Render emergency pause/resume controls."""
        st.markdown("### 🚨 Emergency Controls")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("⏸️ PAUSE AUTONOMY", type="secondary", use_container_width=True):
                st.session_state.autonomy_paused = True
                st.success("🛑 Autonomy system paused")
                self.logger.warning("Autonomy system manually paused")

        with col2:
            if st.button("▶️ RESUME AUTONOMY", type="primary", use_container_width=True):
                st.session_state.autonomy_paused = False
                st.success("✅ Autonomy system resumed")
                self.logger.info("Autonomy system manually resumed")

        with col3:
            if st.button("🔄 REFRESH DATA", use_container_width=True):
                st.session_state.last_refresh = datetime.now()
                st.cache_data.clear()
                st.rerun()

        with col4:
            # Status indicator
            if st.session_state.autonomy_paused:
                st.error("🛑 PAUSED")
            else:
                st.success("✅ ACTIVE")

        st.markdown("---")

    def _render_goal_management(self):
        """Render goal management interface."""
        st.header("🎯 Goal Management")

        if 'goal_stack' not in st.session_state:
            st.error("Goal stack not initialized")
            return

        goal_stack = st.session_state.goal_stack

        # Goal statistics
        col1, col2, col3, col4 = st.columns(4)

        try:
            stats = goal_stack.get_statistics()
            goals_by_status = stats.get('goals_by_status', {})

            with col1:
                st.metric("Pending Goals", goals_by_status.get('pending', 0))

            with col2:
                st.metric("Active Goals", goals_by_status.get('active', 0))

            with col3:
                st.metric("Completed Goals", goals_by_status.get('completed', 0))

            with col4:
                st.metric("Failed Goals", goals_by_status.get('failed', 0))

        except Exception as e:
            st.error(f"Error loading goal statistics: {e}")

        # Goal list
        st.subheader("📋 Current Goals")

        try:
            all_goals = goal_stack.get_all_goals()

            if not all_goals:
                st.info("No goals currently in the system.")
                return

            # Create DataFrame for display
            goal_data = []
            for goal in all_goals:
                goal_data.append({
                    'ID': goal.goal_id[:8],
                    'Description': goal.description[:50] + "..." if len(goal.description) > 50 else goal.description,
                    'Status': goal.status,
                    'Priority': f"{goal.priority:.2f}",
                    'Source': goal.source_skill,
                    'Created': goal.creation_timestamp.strftime('%Y-%m-%d %H:%M'),
                    'Attempts': f"{goal.attempt_count}/{goal.max_attempts}"
                })

            df = pd.DataFrame(goal_data)

            # Interactive goal selection
            selected_indices = st.dataframe(
                df,
                use_container_width=True,
                selection_mode="single-row",
                on_select="rerun"
            )

            # Goal details
            if selected_indices and selected_indices.selection.rows:
                selected_idx = selected_indices.selection.rows[0]
                selected_goal = all_goals[selected_idx]
                self._render_goal_details(selected_goal, goal_stack)

        except Exception as e:
            st.error(f"Error loading goals: {e}")
            self.logger.error(f"Goal loading error: {e}")

    def _render_goal_details(self, goal: Goal, goal_stack: GoalStack):
        """Render detailed view of a selected goal."""
        st.subheader(f"🔍 Goal Details: {goal.goal_id[:8]}")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Description:**", goal.description)
            st.write("**Status:**", goal.status)
            st.write("**Priority:**", f"{goal.priority:.2f}")
            st.write("**Source Skill:**", goal.source_skill)
            st.write("**Attempts:**", f"{goal.attempt_count}/{goal.max_attempts}")

        with col2:
            st.write("**Created:**", goal.creation_timestamp.strftime('%Y-%m-%d %H:%M:%S'))
            st.write("**Last Updated:**", goal.last_updated_timestamp.strftime('%Y-%m-%d %H:%M:%S'))
            st.write("**Tags:**", ", ".join(goal.tags) if goal.tags else "None")
            if goal.failure_reason:
                st.write("**Failure Reason:**", goal.failure_reason)

        # Goal actions
        st.subheader("🎮 Goal Actions")

        action_col1, action_col2, action_col3 = st.columns(3)

        with action_col1:
            if st.button(f"🧠 Plan to Address This Goal", key=f"plan_{goal.goal_id}"):
                self._trigger_goal_planning(goal)

        with action_col2:
            if goal.status == "pending":
                if st.button(f"▶️ Activate Goal", key=f"activate_{goal.goal_id}"):
                    goal_stack.update_goal_status(goal.goal_id, "active")
                    st.success("Goal activated!")
                    st.rerun()

        with action_col3:
            if goal.status in ["pending", "active"]:
                if st.button(f"❌ Cancel Goal", key=f"cancel_{goal.goal_id}"):
                    goal_stack.update_goal_status(goal.goal_id, "failed", "Manually cancelled")
                    st.success("Goal cancelled!")
                    st.rerun()

        # Source context
        if goal.source_context:
            with st.expander("📋 Source Context"):
                st.json(goal.source_context)

    def _trigger_goal_planning(self, goal: Goal):
        """Trigger planning for a specific goal."""
        try:
            # Create a UIF focused on this goal
            uif = SAM_UIF(input_query=f"Address autonomous goal: {goal.description}")
            uif.add_log_entry(f"Manual trigger for goal: {goal.goal_id}")

            # This would integrate with the DynamicPlanner in goal_focused mode
            st.success(f"🧠 Planning triggered for goal: {goal.description[:50]}...")
            st.info("This would create a plan to address the goal and execute it.")

            # Increment manual trigger count
            st.session_state.manual_trigger_count += 1

            self.logger.info(f"Manual planning triggered for goal: {goal.goal_id}")

        except Exception as e:
            st.error(f"Failed to trigger planning: {e}")
            self.logger.error(f"Planning trigger failed: {e}")

    def _render_manual_triggers(self):
        """Render manual trigger interface for testing autonomous behavior."""
        st.header("🧠 Manual Triggers")
        st.markdown("Test autonomous goal generation and planning manually.")

        # Test UIF creation
        st.subheader("🧪 Test Goal Generation")

        col1, col2 = st.columns(2)

        with col1:
            test_query = st.text_area(
                "Test Query",
                value="Find conflicting information about climate change",
                help="Enter a query that might trigger autonomous goal generation"
            )

        with col2:
            # Test scenario selection
            test_scenario = st.selectbox(
                "Test Scenario",
                options=[
                    "Conflict Detection",
                    "Low Confidence Inference",
                    "Learning Failure",
                    "Factual Error",
                    "Knowledge Gap",
                    "Web Search Failure",
                    "Memory Inconsistency",
                    "Custom Query"
                ]
            )

        # Scenario-specific parameters
        if test_scenario != "Custom Query":
            st.info(f"**{test_scenario}**: This will simulate a {test_scenario.lower()} scenario to test goal generation.")

        # Manual trigger buttons
        st.subheader("🎮 Manual Triggers")

        trigger_col1, trigger_col2, trigger_col3 = st.columns(3)

        with trigger_col1:
            if st.button("🎯 Generate Goals from Query", type="primary", use_container_width=True):
                self._trigger_goal_generation(test_query, test_scenario)

        with trigger_col2:
            if st.button("🧠 Plan Goal-Informed Response", use_container_width=True):
                self._trigger_goal_informed_planning(test_query)

        with trigger_col3:
            if st.button("🔄 Run Maintenance Tasks", use_container_width=True):
                self._trigger_maintenance_tasks()

        # Manual trigger statistics
        st.subheader("📊 Manual Trigger Statistics")

        stats_col1, stats_col2, stats_col3 = st.columns(3)

        with stats_col1:
            st.metric("Manual Triggers", st.session_state.manual_trigger_count)

        with stats_col2:
            st.metric("Last Refresh", st.session_state.last_refresh.strftime('%H:%M:%S'))

        with stats_col3:
            if 'goal_stack' in st.session_state:
                try:
                    stats = st.session_state.goal_stack.get_statistics()
                    total_goals = sum(stats.get('goals_by_status', {}).values())
                    st.metric("Total Goals", total_goals)
                except:
                    st.metric("Total Goals", "Error")

    def _trigger_goal_generation(self, query: str, scenario: str):
        """Trigger goal generation from a test query."""
        if 'motivation_engine' not in st.session_state:
            st.error("Motivation engine not initialized")
            return

        try:
            # Create test UIF
            uif = SAM_UIF(input_query=query)

            # Add scenario-specific intermediate data
            if scenario == "Conflict Detection":
                uif.intermediate_data['conflict_detected'] = {
                    'conflicting_ids': ['chunk1', 'chunk2'],
                    'conflict_type': 'factual_disagreement'
                }
            elif scenario == "Low Confidence Inference":
                uif.intermediate_data['implicit_knowledge_summary'] = {
                    'inference': 'Low confidence relationship detected',
                    'confidence': 0.3
                }
            elif scenario == "Learning Failure":
                uif.intermediate_data['learning_stall'] = {
                    'edit_details': 'Failed to update knowledge base'
                }
            elif scenario == "Factual Error":
                uif.intermediate_data['factual_error_detected'] = {
                    'error_details': 'Incorrect date found in memory'
                }
            elif scenario == "Knowledge Gap":
                uif.intermediate_data['knowledge_gap_identified'] = {
                    'gap_description': 'Missing information about topic'
                }
            elif scenario == "Web Search Failure":
                uif.intermediate_data['web_search_failed'] = {
                    'search_query': query,
                    'error': 'Search timeout'
                }
            elif scenario == "Memory Inconsistency":
                uif.intermediate_data['memory_inconsistency'] = {
                    'inconsistency_details': 'Conflicting memory chunks detected'
                }

            # Generate goals
            motivation_engine = st.session_state.motivation_engine
            generated_goals = motivation_engine.generate_goals_from_uif(uif)

            if generated_goals:
                st.success(f"✅ Generated {len(generated_goals)} autonomous goals!")

                # Display generated goals
                for i, goal in enumerate(generated_goals):
                    with st.expander(f"Goal {i+1}: {goal.description[:50]}..."):
                        st.write("**Full Description:**", goal.description)
                        st.write("**Priority:**", f"{goal.priority:.2f}")
                        st.write("**Source Skill:**", goal.source_skill)
                        st.write("**Tags:**", ", ".join(goal.tags))
            else:
                st.warning("No goals were generated from this scenario.")

            st.session_state.manual_trigger_count += 1

        except Exception as e:
            st.error(f"Goal generation failed: {e}")
            self.logger.error(f"Manual goal generation failed: {e}")

    def _trigger_goal_informed_planning(self, query: str):
        """Trigger goal-informed planning mode."""
        try:
            st.info("🧠 Goal-informed planning would be triggered here.")
            st.markdown("This would:")
            st.markdown("- Retrieve top priority goals")
            st.markdown("- Create a plan that considers background goals")
            st.markdown("- Execute with goal context")

            # Show what goals would be considered
            if 'goal_stack' in st.session_state:
                top_goals = st.session_state.goal_stack.get_top_priority_goals(limit=3)
                if top_goals:
                    st.subheader("🎯 Goals that would be considered:")
                    for goal in top_goals:
                        st.write(f"- **{goal.description[:60]}...** (Priority: {goal.priority:.2f})")
                else:
                    st.info("No pending goals to consider.")

            st.session_state.manual_trigger_count += 1

        except Exception as e:
            st.error(f"Goal-informed planning failed: {e}")

    def _trigger_maintenance_tasks(self):
        """Trigger maintenance tasks like priority decay and archiving."""
        if 'goal_stack' not in st.session_state:
            st.error("Goal stack not initialized")
            return

        try:
            goal_stack = st.session_state.goal_stack

            # Run maintenance tasks
            decayed_count = goal_stack.decay_priorities()
            archived_count = goal_stack.archive_completed_goals()

            st.success(f"✅ Maintenance completed!")
            st.info(f"- Priority decay applied to {decayed_count} goals")
            st.info(f"- Archived {archived_count} completed goals")

            st.session_state.manual_trigger_count += 1

        except Exception as e:
            st.error(f"Maintenance tasks failed: {e}")
            self.logger.error(f"Maintenance tasks failed: {e}")

    def _render_analytics(self):
        """Render analytics and statistics dashboard."""
        st.header("📊 Autonomy Analytics")

        if 'goal_stack' not in st.session_state:
            st.error("Goal stack not initialized")
            return

        try:
            goal_stack = st.session_state.goal_stack
            motivation_engine = st.session_state.motivation_engine
            safety_validator = st.session_state.safety_validator

            # Get statistics
            goal_stats = goal_stack.get_statistics()
            engine_stats = motivation_engine.get_statistics()
            safety_stats = safety_validator.get_validation_stats()

            # Goal status distribution
            st.subheader("🎯 Goal Status Distribution")

            goals_by_status = goal_stats.get('goals_by_status', {})
            if goals_by_status:
                fig_pie = px.pie(
                    values=list(goals_by_status.values()),
                    names=list(goals_by_status.keys()),
                    title="Goal Status Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No goal data available for visualization.")

            # Goal sources
            st.subheader("🔧 Goals by Source Skill")

            goals_by_source = goal_stats.get('goals_by_source', {})
            if goals_by_source:
                fig_bar = px.bar(
                    x=list(goals_by_source.keys()),
                    y=list(goals_by_source.values()),
                    title="Goals Generated by Source Skill"
                )
                fig_bar.update_layout(xaxis_title="Source Skill", yaxis_title="Goal Count")
                st.plotly_chart(fig_bar, use_container_width=True)

            # Statistics tables
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🧠 Motivation Engine Stats")
                engine_df = pd.DataFrame([
                    {"Metric": "Total Analyses", "Value": engine_stats.get('total_analyses', 0)},
                    {"Metric": "Goals Generated", "Value": engine_stats.get('goals_generated', 0)},
                    {"Metric": "Goals Rejected", "Value": engine_stats.get('goals_rejected', 0)},
                ])
                st.dataframe(engine_df, use_container_width=True)

            with col2:
                st.subheader("🛡️ Safety Validator Stats")
                safety_df = pd.DataFrame([
                    {"Metric": "Goals Last Minute", "Value": safety_stats.get('goals_last_minute', 0)},
                    {"Metric": "Goals Last Hour", "Value": safety_stats.get('goals_last_hour', 0)},
                    {"Metric": "Rate Limit (min)", "Value": safety_stats.get('rate_limit_per_minute', 0)},
                ])
                st.dataframe(safety_df, use_container_width=True)

            # Rule trigger frequency
            st.subheader("📋 Rule Trigger Frequency")
            rules_triggered = engine_stats.get('rules_triggered', {})
            if rules_triggered:
                rules_df = pd.DataFrame([
                    {"Rule": rule, "Triggers": count}
                    for rule, count in rules_triggered.items()
                ])
                st.dataframe(rules_df, use_container_width=True)
            else:
                st.info("No rule trigger data available.")

        except Exception as e:
            st.error(f"Error loading analytics: {e}")
            self.logger.error(f"Analytics error: {e}")

    def _render_safety_monitor(self):
        """Render safety monitoring and controls."""
        st.header("🛡️ Safety Monitor")

        if 'safety_validator' not in st.session_state:
            st.error("Safety validator not initialized")
            return

        safety_validator = st.session_state.safety_validator

        # Safety status overview
        st.subheader("🚨 Safety Status")

        try:
            stats = safety_validator.get_validation_stats()

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                goals_last_minute = stats.get('goals_last_minute', 0)
                rate_limit = stats.get('rate_limit_per_minute', 10)
                if goals_last_minute >= rate_limit:
                    st.error(f"⚠️ Rate Limit\n{goals_last_minute}/{rate_limit}")
                else:
                    st.success(f"✅ Rate OK\n{goals_last_minute}/{rate_limit}")

            with col2:
                goals_last_hour = stats.get('goals_last_hour', 0)
                hourly_limit = stats.get('rate_limit_per_hour', 100)
                if goals_last_hour >= hourly_limit:
                    st.error(f"⚠️ Hourly Limit\n{goals_last_hour}/{hourly_limit}")
                else:
                    st.success(f"✅ Hourly OK\n{goals_last_hour}/{hourly_limit}")

            with col3:
                if st.session_state.autonomy_paused:
                    st.error("🛑 PAUSED")
                else:
                    st.success("✅ ACTIVE")

            with col4:
                last_validation = stats.get('last_validation', 'Never')
                if last_validation != 'Never':
                    try:
                        last_time = datetime.fromisoformat(last_validation)
                        time_diff = datetime.now() - last_time
                        if time_diff.total_seconds() < 300:  # 5 minutes
                            st.success(f"✅ Recent\n{time_diff.seconds}s ago")
                        else:
                            st.warning(f"⚠️ Stale\n{time_diff.seconds//60}m ago")
                    except:
                        st.info(f"📊 Last Check\n{last_validation}")
                else:
                    st.info("📊 No Data")

        except Exception as e:
            st.error(f"Error loading safety stats: {e}")

        # Safety controls
        st.subheader("🔧 Safety Controls")

        control_col1, control_col2 = st.columns(2)

        with control_col1:
            if st.button("🔄 Reset Safety Counters", use_container_width=True):
                try:
                    safety_validator.reset_counters()
                    st.success("Safety counters reset!")
                except Exception as e:
                    st.error(f"Failed to reset counters: {e}")

        with control_col2:
            if st.button("📊 Update Safety Config", use_container_width=True):
                st.info("Safety configuration update would be triggered here.")

        # Safety patterns and rules
        st.subheader("🚫 Safety Patterns")

        with st.expander("View Harmful Action Patterns"):
            st.markdown("**Detected Patterns:**")
            harmful_patterns = [
                "delete.*config", "remove.*security", "modify.*auth",
                "disable.*safety", "bypass.*validation", "override.*security"
            ]
            for pattern in harmful_patterns:
                st.code(pattern)

        with st.expander("View Protected Resources"):
            st.markdown("**Protected File Patterns:**")
            protected_patterns = [
                ".*\\.key$", ".*\\.pem$", ".*config.*\\.json$",
                ".*\\.env$", "sam/autonomy/safety/.*"
            ]
            for pattern in protected_patterns:
                st.code(pattern)

    def _render_settings(self):
        """Render autonomy system settings and configuration."""
        st.header("⚙️ Autonomy Settings")

        # System configuration
        st.subheader("🔧 System Configuration")

        config_col1, config_col2 = st.columns(2)

        with config_col1:
            st.markdown("**Goal Stack Settings:**")

            max_active_goals = st.number_input(
                "Max Active Goals",
                min_value=1,
                max_value=1000,
                value=100,
                help="Maximum number of active goals allowed"
            )

            archive_after_days = st.number_input(
                "Archive After (days)",
                min_value=1,
                max_value=365,
                value=30,
                help="Archive completed goals after N days"
            )

            priority_decay_rate = st.slider(
                "Priority Decay Rate",
                min_value=0.001,
                max_value=0.1,
                value=0.01,
                step=0.001,
                help="Daily priority decay rate"
            )

        with config_col2:
            st.markdown("**Safety Settings:**")

            max_goals_per_minute = st.number_input(
                "Max Goals per Minute",
                min_value=1,
                max_value=100,
                value=10,
                help="Rate limit for goal creation"
            )

            max_goals_per_hour = st.number_input(
                "Max Goals per Hour",
                min_value=1,
                max_value=1000,
                value=100,
                help="Hourly rate limit for goal creation"
            )

            enable_harmful_detection = st.checkbox(
                "Enable Harmful Action Detection",
                value=True,
                help="Detect and block harmful goal patterns"
            )

        # Motivation engine settings
        st.subheader("🧠 Motivation Engine Settings")

        engine_col1, engine_col2 = st.columns(2)

        with engine_col1:
            max_goals_per_analysis = st.number_input(
                "Max Goals per Analysis",
                min_value=1,
                max_value=20,
                value=5,
                help="Maximum goals generated per UIF analysis"
            )

            min_confidence_threshold = st.slider(
                "Min Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Minimum confidence for goal generation"
            )

        with engine_col2:
            st.markdown("**Enable Goal Generation Rules:**")

            enable_conflict_goals = st.checkbox("Conflict Detection", value=True)
            enable_inference_goals = st.checkbox("Low Confidence Inference", value=True)
            enable_learning_goals = st.checkbox("Learning Failures", value=True)
            enable_error_goals = st.checkbox("Factual Errors", value=True)

        # Apply settings button
        if st.button("💾 Apply Settings", type="primary", use_container_width=True):
            try:
                # Update configurations
                if 'goal_stack' in st.session_state:
                    goal_stack = st.session_state.goal_stack
                    goal_stack.config.update({
                        'max_active_goals': max_active_goals,
                        'archive_after_days': archive_after_days,
                        'priority_decay_rate': priority_decay_rate
                    })

                if 'safety_validator' in st.session_state:
                    safety_validator = st.session_state.safety_validator
                    safety_validator.update_config({
                        'max_goals_per_minute': max_goals_per_minute,
                        'max_goals_per_hour': max_goals_per_hour,
                        'enable_harmful_detection': enable_harmful_detection
                    })

                if 'motivation_engine' in st.session_state:
                    motivation_engine = st.session_state.motivation_engine
                    motivation_engine.update_config({
                        'max_goals_per_analysis': max_goals_per_analysis,
                        'min_confidence_threshold': min_confidence_threshold,
                        'enable_conflict_goals': enable_conflict_goals,
                        'enable_inference_goals': enable_inference_goals,
                        'enable_learning_goals': enable_learning_goals,
                        'enable_error_goals': enable_error_goals
                    })

                st.success("✅ Settings applied successfully!")
                self.logger.info("Autonomy settings updated via dashboard")

            except Exception as e:
                st.error(f"Failed to apply settings: {e}")
                self.logger.error(f"Settings update failed: {e}")

        # System information
        st.subheader("ℹ️ System Information")

        info_col1, info_col2 = st.columns(2)

        with info_col1:
            st.markdown("**Component Status:**")
            st.write("- Goal Stack:", "✅ Initialized" if 'goal_stack' in st.session_state else "❌ Not Available")
            st.write("- Motivation Engine:", "✅ Initialized" if 'motivation_engine' in st.session_state else "❌ Not Available")
            st.write("- Safety Validator:", "✅ Initialized" if 'safety_validator' in st.session_state else "❌ Not Available")

        with info_col2:
            st.markdown("**Database Information:**")
            if 'goal_stack' in st.session_state:
                try:
                    goal_stack = st.session_state.goal_stack
                    st.write("- Database Path:", goal_stack.db_path)
                    st.write("- Cache TTL:", f"{goal_stack.config.get('cache_ttl_seconds', 60)}s")
                except:
                    st.write("- Database:", "Error loading info")
            else:
                st.write("- Database:", "Not initialized")


def render_autonomy_dashboard():
    """Main function to render the autonomy dashboard."""
    dashboard = AutonomyDashboard()
    dashboard.render()


if __name__ == "__main__":
    render_autonomy_dashboard()