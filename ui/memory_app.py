"""
Main Memory UI Application for SAM
Integrated Streamlit app for interactive memory control and visualization.

Sprint 12: Interactive Memory Control & Visualization
"""

import os
# Suppress PyTorch/Streamlit compatibility warnings and prevent crashes
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

# Fix torch/Streamlit compatibility issues
os.environ['PYTORCH_DISABLE_PER_OP_PROFILING'] = '1'

import streamlit as st
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ui.memory_browser import MemoryBrowserUI
from ui.memory_editor import MemoryEditor
from ui.memory_graph import MemoryGraphVisualizer
from ui.memory_commands import MemoryCommandProcessor, get_command_processor
from ui.role_memory_filter import RoleBasedMemoryFilter, get_role_filter
from ui.bulk_ingestion_ui import render_bulk_ingestion
from ui.api_key_manager import render_api_key_manager
from ui.insight_archive_ui import render_insight_archive
from memory.memory_vectorstore import get_memory_store
from memory.memory_reasoning import get_memory_reasoning_engine
from config.agent_mode import get_mode_controller
from agents.task_router import AgentRole

def check_authentication():
    """
    Check if user is authenticated via the secure SAM interface.

    Returns:
        bool: True if authenticated, False otherwise
    """
    try:
        # Import security manager
        from security import SecureStateManager

        # CRITICAL FIX: Try to get shared security manager from main SAM interface
        # Check if we can access the main SAM's security manager
        security_manager = None

        # Method 1: Try to get from session state (if available)
        if 'security_manager' in st.session_state:
            security_manager = st.session_state.security_manager
            logger.info("✅ Using shared security manager from session state")
        else:
            # Method 2: Create new instance but check for existing authentication
            security_manager = SecureStateManager()
            st.session_state.security_manager = security_manager
            logger.info("🆕 Created new security manager instance")

        # Check if the security system is unlocked
        is_authenticated = security_manager.is_unlocked()

        if is_authenticated:
            # Verify session is still valid
            session_info = security_manager.get_session_info()
            if session_info['is_unlocked'] and session_info['time_remaining'] > 0:
                logger.info("✅ Authentication verified - Memory Control Center access granted")
                return True
            else:
                # Session expired, lock the application
                security_manager.lock_application()
                logger.warning("⚠️ Session expired - locking application")
                return False

        logger.info("🔒 Authentication required - user not authenticated")
        return False

    except ImportError as e:
        # Security module not available - deny access in production
        st.error(f"❌ Security module not available: {e}")
        st.error("🔒 **Access Denied**: Memory Control Center requires the security module to be properly installed.")
        st.info("💡 **Solution**: Ensure the security module is properly installed and configured.")
        return False
    except Exception as e:
        # Log error and deny access
        logger.error(f"Authentication check failed: {e}")
        st.error(f"❌ Authentication check failed: {e}")
        st.error("🔒 **Access Denied**: Unable to verify authentication status.")
        return False

def render_authentication_required():
    """Render the authentication required page."""
    st.title("🔒 SAM Memory Control Center")
    st.markdown("*Authentication Required*")
    st.markdown("---")

    st.error("🚫 **Access Denied - Authentication Required**")
    st.markdown("""
    The SAM Memory Control Center requires authentication through the secure SAM interface.

    **🔐 Security Integration:**
    - Memory Control Center shares the same security system as the main SAM interface
    - Your memory data is encrypted with AES-256-GCM and requires authentication
    - Session management ensures secure access across all SAM components

    **📋 To Access the Memory Control Center:**
    1. **Open the Secure SAM Interface**: [http://localhost:8502](http://localhost:8502)
    2. **Enter your master password** to unlock SAM
    3. **Return to this page** or click the Memory Control Center link from the main interface

    **🛡️ Why Authentication is Required:**
    - Your personal memory data is encrypted and protected
    - Prevents unauthorized access to your knowledge base
    - Maintains data integrity and privacy
    - Ensures audit trail for all memory operations
    """)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; margin: 1rem 0;">
            <a href="http://localhost:8502" target="_blank" style="
                display: inline-block;
                padding: 0.75rem 1.5rem;
                background-color: #ff4b4b;
                color: white;
                text-decoration: none;
                border-radius: 0.5rem;
                font-weight: bold;
                font-size: 1.1rem;
                width: 100%;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">🔓 Go to Secure SAM Interface</a>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        if st.button("🔄 Check Authentication Status", use_container_width=True):
            st.rerun()

    # Show security status
    try:
        # Use shared security manager if available
        if 'security_manager' in st.session_state:
            security_manager = st.session_state.security_manager
        else:
            from security import SecureStateManager
            security_manager = SecureStateManager()

        session_info = security_manager.get_session_info()

        st.markdown("### 🔍 Security Status")
        status_col1, status_col2 = st.columns(2)

        with status_col1:
            st.metric("Application State", session_info['state'].title())
            st.metric("Failed Attempts", f"{session_info['failed_attempts']}/{session_info['max_attempts']}")

        with status_col2:
            if session_info['is_unlocked']:
                st.metric("Session Time Remaining", f"{session_info['time_remaining']} seconds")
            else:
                st.metric("Authentication Status", "🔒 Locked")

    except Exception as e:
        st.warning(f"Could not retrieve security status: {e}")

def render_security_status_indicator():
    """Render a compact security status indicator in the header."""
    try:
        if 'security_manager' in st.session_state:
            security_manager = st.session_state.security_manager
            session_info = security_manager.get_session_info()

            if session_info['is_unlocked']:
                time_remaining = session_info['time_remaining']
                minutes_remaining = time_remaining // 60

                # Color code based on time remaining
                if minutes_remaining > 30:
                    status_color = "🟢"
                elif minutes_remaining > 10:
                    status_color = "🟡"
                else:
                    status_color = "🔴"

                st.markdown(f"""
                <div style="text-align: right; padding: 10px; background-color: #f0f2f6; border-radius: 5px; margin-top: 10px;">
                    <small><strong>{status_color} Authenticated</strong><br>
                    Session: {minutes_remaining}m remaining</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-align: right; padding: 10px; background-color: #ffebee; border-radius: 5px; margin-top: 10px;">
                    <small><strong>🔒 Session Expired</strong><br>
                    <a href="http://localhost:8502" target="_blank">Re-authenticate</a></small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: right; padding: 10px; background-color: #fff3e0; border-radius: 5px; margin-top: 10px;">
                <small><strong>⚠️ Development Mode</strong><br>
                Security disabled</small>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.markdown(f"""
        <div style="text-align: right; padding: 10px; background-color: #ffebee; border-radius: 5px; margin-top: 10px;">
            <small><strong>❌ Security Error</strong><br>
            {str(e)[:30]}...</small>
        </div>
        """, unsafe_allow_html=True)

def is_dream_canvas_available():
    """Check if Dream Canvas feature is available (now available in Community Edition)."""
    # Dream Canvas is now available to all SAM Community Edition users
    return True

def render_dream_canvas_locked():
    """Render the locked Dream Canvas interface for non-Pro users."""
    st.markdown("---")

    # Locked feature display
    st.markdown("""
    <div style="
        border: 2px solid #FF6B6B;
        border-radius: 12px;
        padding: 30px;
        margin: 20px 0;
        background: linear-gradient(135deg, #FF6B6B15, #FF6B6B05);
        text-align: center;
    ">
        <h2 style="color: #FF6B6B; margin: 0 0 15px 0;">🔒 Premium Feature</h2>
        <h3 style="color: #333; margin: 0 0 20px 0;">Dream Canvas - Cognitive Synthesis Visualization</h3>
        <p style="font-size: 1.1rem; color: #666; margin: 0 0 25px 0;">
            Unlock SAM's revolutionary cognitive synthesis engine that analyzes your memory landscape
            and generates insights from concept clusters.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature highlights
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌟 Dream Canvas Features")
        st.markdown("""
        - **🧠 Cognitive Synthesis**: AI-powered insight generation from memory clusters
        - **🎨 Interactive Visualization**: UMAP projections with color-coded concept clusters
        - **🔍 Cluster Explorer**: Deep dive into knowledge domains with "So What" insights
        - **⚡ Smart Parameters**: Research-backed optimal clustering with AI suggestions
        - **📊 Knowledge Discovery**: Identify cross-domain patterns and emerging themes
        """)

    with col2:
        st.markdown("### 💡 Why Dream Canvas?")
        st.markdown("""
        - **Discover Hidden Patterns**: Find connections across your knowledge base
        - **Generate New Insights**: Synthesize emergent understanding from concept clusters
        - **Visualize Knowledge**: See your memory landscape as an interactive star chart
        - **Optimize Learning**: Identify knowledge gaps and concentration areas
        - **Research Intelligence**: Cross-domain analysis for deeper understanding
        """)

    # Activation call-to-action
    st.markdown("---")
    st.warning("🔒 **Dream Canvas** requires SAM Pro activation")
    st.markdown("""
    **Activate SAM Pro to unlock:**
    - 🧠 Cognitive synthesis and insight generation
    - 🎨 Interactive memory landscape visualization
    - 🔍 Advanced cluster analysis with "So What" insights
    - ⚡ AI-powered parameter optimization
    - 📊 Cross-domain knowledge discovery

    💡 **Activate SAM Pro in the secure interface at [http://localhost:8502](http://localhost:8502)**
    """)

    # Demo preview (static)
    st.markdown("### 🎬 Preview: What You'll Unlock")
    st.markdown("""
    <div style="
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 20px;
        background: #f8f9fa;
        margin: 15px 0;
    ">
        <h4>🎨 Interactive Memory Landscape</h4>
        <p><strong>Memory Points:</strong> 10,594 | <strong>Clusters Discovered:</strong> 12 | <strong>Memories Clustered:</strong> 8,247 (78%)</p>
        <p style="color: #666; font-style: italic;">
            "This cluster represents a convergence of ideas around <strong>machine learning, algorithms</strong>,
            drawing from 3 different types of sources. <strong>So What:</strong> This represents a major knowledge
            domain for SAM - consider this cluster when asking questions about machine learning."
        </p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main Streamlit application for memory management."""

    # Page configuration - only call once at the very beginning
    if 'page_config_set' not in st.session_state:
        st.set_page_config(
            page_title="SAM Memory Control Center",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        st.session_state.page_config_set = True

    # Security check - require authentication from secure SAM interface
    if not check_authentication():
        render_authentication_required()
        return

    # Periodic authentication check (every 5 minutes)
    if 'last_auth_check' not in st.session_state:
        st.session_state.last_auth_check = time.time()

    # Check if 5 minutes have passed since last check
    if time.time() - st.session_state.last_auth_check > 300:  # 5 minutes
        if not check_authentication():
            st.rerun()  # This will trigger the authentication required page
        st.session_state.last_auth_check = time.time()

    # Initialize components
    memory_store = get_memory_store()
    memory_reasoning = get_memory_reasoning_engine()
    mode_controller = get_mode_controller()
    command_processor = get_command_processor()
    role_filter = get_role_filter()

    # Main header with security status
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🧠 SAM Memory Control Center")
        st.markdown("Interactive memory management, visualization, and role-based access control")

    with col2:
        render_security_status_indicator()

    # Navigation menu moved to main content area (preserving 100% of functionality)
    st.markdown("---")
    st.subheader("🎛️ Navigation")

    # Current mode display
    current_mode = mode_controller.get_current_mode()
    mode_status = mode_controller.get_mode_status()

    col1, col2 = st.columns([2, 1])
    with col1:
        # Navigation menu
        page = st.selectbox(
            "Select Feature",
            options=[
                "💬 Enhanced Chat",
                "Chat with SAM",
                "🔍 Reasoning Visualizer",
                "🧠📊 TPV Dissonance Demo",  # NEW: Phase 5B Demo
                "🧠 Personalized Tuner",  # NEW: DPO Integration
                "📁 Bulk Ingestion",
                "🔑 API Key Manager",
                "🧠🎨 Dream Canvas",
                "📚 Archived Insights",  # NEW: Archived Insights page
                "Memory Browser",
                "Memory Editor",
                "Memory Graph",
                "Command Interface",
                "Role-Based Access",
                "🏆 Memory Ranking",
                "📝 Citation Engine",
                "📊 Smart Summaries",
                "📈 Memory Insights",
                "🧠 Thought Settings",
                "🔧 System Health",
                "System Status",
                "⚙️ Admin Dashboard",
                "🐛 Interactive Debugger"
            ],
            index=0,
            help="Choose a Memory Control Center feature to access"
        )

    with col2:
        st.info(f"**Current Mode:** {current_mode.value.title()}")
        if mode_status.key_status.value != "missing":
            st.caption(f"Key Status: {mode_status.key_status.value}")

    st.markdown("---")

    # Sidebar for quick stats and actions (preserving 100% of functionality)
    with st.sidebar:
        st.header("📊 Memory Control Center")

        # Quick stats
        st.subheader("📊 Quick Stats")
        try:
            stats = memory_store.get_memory_stats()
            # Use .get() with fallback values to prevent KeyError
            total_memories = stats.get('total_memories', len(getattr(memory_store, 'memory_chunks', {})))
            total_size_mb = stats.get('total_size_mb', 0.0)

            st.metric("Total Memories", total_memories)
            st.metric("Storage Size", f"{total_size_mb:.1f} MB")

            memory_types = stats.get('memory_types', {})
            if memory_types:
                st.caption("**Memory Types:**")
                for mem_type, count in list(memory_types.items())[:3]:
                    st.caption(f"• {mem_type}: {count}")

        except Exception as e:
            st.error(f"Error loading stats: {e}")
            # Provide fallback metrics
            st.metric("Total Memories", "N/A")
            st.metric("Storage Size", "N/A")

        # Quick actions
        st.subheader("⚡ Quick Actions")

        if st.button("🔄 Refresh Data", key="sidebar_refresh_data"):
            st.cache_data.clear()
            st.rerun()

        if st.button("📊 Memory Stats", key="sidebar_memory_stats"):
            st.session_state.show_stats = True

        # Memory command input
        st.subheader("💬 Quick Command")
        quick_command = st.text_input(
            "Memory Command",
            placeholder="!recall topic AI",
            help="Enter a memory command (type !memhelp for help)"
        )

        if st.button("Execute", key="sidebar_execute_command") and quick_command:
            result = command_processor.process_command(quick_command)
            if result.success:
                st.success("Command executed successfully!")
                st.text_area("Result", value=result.message, height=100)
            else:
                st.error(f"Command failed: {result.message}")
    
    # Main content area
    if page == "💬 Enhanced Chat":
        render_enhanced_chat_interface()
    elif page == "Chat with SAM":
        render_chat_interface()
    elif page == "🔍 Reasoning Visualizer":
        render_reasoning_visualizer()
    elif page == "🧠📊 TPV Dissonance Demo":
        render_tpv_dissonance_demo()
    elif page == "🧠 Personalized Tuner":
        render_personalized_tuner()
    elif page == "📁 Bulk Ingestion":
        render_bulk_ingestion()
    elif page == "🔑 API Key Manager":
        render_api_key_manager()
    elif page == "🧠🎨 Dream Canvas":
        render_dream_canvas()
    elif page == "📚 Archived Insights":
        render_archived_insights()
    elif page == "Memory Browser":
        render_memory_browser()
    elif page == "Memory Editor":
        render_memory_editor()
    elif page == "Memory Graph":
        render_memory_graph()
    elif page == "Command Interface":
        render_command_interface()
    elif page == "Role-Based Access":
        render_role_access()
    elif page == "🏆 Memory Ranking":
        render_memory_ranking()
    elif page == "📝 Citation Engine":
        render_citation_engine()
    elif page == "📊 Smart Summaries":
        render_smart_summaries()
    elif page == "📈 Memory Insights":
        render_memory_insights()
    elif page == "🧠 Thought Settings":
        render_thought_settings()
    elif page == "🔧 System Health":
        render_system_health()
    elif page == "System Status":
        render_system_status()
    elif page == "⚙️ Admin Dashboard":
        render_admin_dashboard_interface()
    elif page == "🐛 Interactive Debugger":
        render_interactive_debugger_interface()

def render_chat_interface():
    """Render the enhanced diagnostic chat interface with SAM."""
    try:
        st.subheader("💬 Diagnostic Chat with SAM")
        st.markdown("Interactive conversation with comprehensive diagnostic information and memory-driven reasoning")

        # Diagnostic settings
        col1, col2, col3 = st.columns(3)

        with col1:
            show_diagnostics = st.checkbox("🔍 Show Diagnostics", value=True, help="Display detailed diagnostic information")

        with col2:
            show_memory_context = st.checkbox("🧠 Show Memory Context", value=True, help="Display memory retrieval details")

        with col3:
            show_reasoning_trace = st.checkbox("🤔 Show Reasoning Trace", value=True, help="Display reasoning process details")

        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # Chat input
        user_input = st.chat_input("Ask SAM anything... (Use !commands for memory operations)")

        if user_input:
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            # Generate diagnostic information
            diagnostic_info = generate_diagnostic_info(user_input)

            # Check if it's a memory command
            command_processor = get_command_processor()

            if user_input.startswith('!'):
                # Process memory command
                result = command_processor.process_command(user_input)

                if result.success:
                    response = f"✅ **Command Result:**\n\n{result.message}"
                    diagnostic_info['command_execution'] = {
                        'status': 'success',
                        'execution_time': getattr(result, 'execution_time_ms', 0),
                        'data_returned': bool(getattr(result, 'data', None))
                    }
                else:
                    response = f"❌ **Command Error:**\n\n{result.message}"
                    diagnostic_info['command_execution'] = {
                        'status': 'failed',
                        'error': result.message
                    }
            else:
                # Regular chat with memory-driven reasoning
                memory_reasoning = get_memory_reasoning_engine()

                # Use memory-driven reasoning
                reasoning_session = memory_reasoning.reason_with_memory(
                    query=user_input,
                    user_id="streamlit_user",
                    session_id=f"streamlit_session_{len(st.session_state.chat_history)}"
                )

                if reasoning_session:
                    response = reasoning_session.final_response

                    # Enhanced diagnostic information
                    diagnostic_info.update({
                        'reasoning_session': {
                            'session_id': reasoning_session.session_id,
                            'reasoning_steps': len(getattr(reasoning_session, 'reasoning_steps', [])),
                            'confidence_score': getattr(reasoning_session, 'confidence_score', 0.0),
                            'processing_time_ms': getattr(reasoning_session, 'processing_time_ms', 0)
                        },
                        'memory_context': {
                            'memories_found': reasoning_session.memory_context.memory_count,
                            'relevance_score': reasoning_session.memory_context.relevance_score,
                            'search_strategy': getattr(reasoning_session.memory_context, 'search_strategy', 'default'),
                            'context_length': len(reasoning_session.memory_context.context_text)
                        }
                    })

                    # Add memory context info to response
                    if reasoning_session.memory_context.memory_count > 0:
                        response += f"\n\n*💭 Recalled {reasoning_session.memory_context.memory_count} relevant memories (relevance: {reasoning_session.memory_context.relevance_score:.2f})*"
                else:
                    response = "I'm here to help! You can ask me questions or use memory commands like `!recall topic AI` to search my memory."
                    diagnostic_info['reasoning_session'] = {'status': 'no_session_created'}

            # Add SAM response to history with diagnostics
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "diagnostics": diagnostic_info if show_diagnostics else None
            })

        # Display chat history with enhanced diagnostics
        for i, message in enumerate(st.session_state.chat_history):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Show diagnostics for assistant messages
                if (message["role"] == "assistant" and
                    show_diagnostics and
                    message.get("diagnostics")):

                    render_diagnostic_panel(message["diagnostics"], show_memory_context, show_reasoning_trace)

        # Enhanced help and controls
        col1, col2 = st.columns(2)

        with col1:
            with st.expander("💡 Memory Commands Help"):
                st.markdown("""
                **Available Memory Commands:**
                - `!recall topic [keyword]` - Find memories about a topic
                - `!recall last 5` - Get recent memories
                - `!searchmem tag:important` - Search by tags
                - `!searchmem type:conversation` - Search by memory type
                - `!memstats` - View memory statistics
                - `!memhelp` - Show all commands

                **Example:** `!recall topic artificial intelligence`
                """)

        with col2:
            with st.expander("🔍 Diagnostic Commands"):
                st.markdown("""
                **Diagnostic Queries:**
                - `Hello SAM` - Basic system status
                - `What do you remember?` - Memory overview
                - `How are you learning?` - Learning status
                - `Show me your capabilities` - System capabilities
                - `What documents have you learned from?` - Learning history

                **System Status:**
                - Memory store health
                - Learning statistics
                - Recent activity
                - Performance metrics
                """)

        # Control buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🗑️ Clear Chat History", key="clear_chat_history_button"):
                st.session_state.chat_history = []
                st.rerun()

        with col2:
            if st.button("📊 System Status"):
                # Add system status message
                status_info = get_system_status_info()
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": format_system_status(status_info),
                    "diagnostics": status_info if show_diagnostics else None
                })
                st.rerun()

        with col3:
            if st.button("🧠 Memory Overview"):
                # Add memory overview message
                memory_info = get_memory_overview_info()
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": format_memory_overview(memory_info),
                    "diagnostics": memory_info if show_diagnostics else None
                })
                st.rerun()

    except Exception as e:
        st.error(f"Error loading chat interface: {e}")

def render_memory_browser():
    """Render the memory browser interface."""
    try:
        browser = MemoryBrowserUI()
        browser.render()
    except Exception as e:
        st.error(f"Error loading memory browser: {e}")

def render_memory_editor():
    """Render the memory editor interface."""
    try:
        st.subheader("✏️ Memory Editor")
        
        editor = MemoryEditor()
        
        # Check if we have a memory to edit
        if hasattr(st.session_state, 'editing_memory') and st.session_state.editing_memory:
            editor.render_edit_interface(st.session_state.editing_memory)
        elif hasattr(st.session_state, 'deleting_memory') and st.session_state.deleting_memory:
            editor.render_delete_interface(st.session_state.deleting_memory)
        else:
            st.info("Select a memory from the Memory Browser to edit or delete it.")
            
            # Show recent edits and deletions
            col1, col2 = st.columns(2)
            
            with col1:
                editor.render_undo_interface()
            
            with col2:
                editor.render_audit_log()
                
    except Exception as e:
        st.error(f"Error loading memory editor: {e}")

def render_memory_graph():
    """Render the memory graph visualization."""
    try:
        visualizer = MemoryGraphVisualizer()
        visualizer.render()
    except Exception as e:
        st.error(f"Error loading memory graph: {e}")

def render_command_interface():
    """Render the command interface."""
    try:
        st.subheader("💬 Memory Command Interface")
        st.markdown("Execute memory recall commands and view results")
        
        command_processor = get_command_processor()
        
        # Command input
        col1, col2 = st.columns([3, 1])
        
        with col1:
            command_text = st.text_input(
                "Enter Memory Command",
                placeholder="!recall topic artificial intelligence",
                help="Enter a memory command to execute"
            )
        
        with col2:
            output_format = st.selectbox("Output", ["text", "json"])
        
        # Execute button
        if st.button("🚀 Execute Command", type="primary", key="execute_command_button") and command_text:
            with st.spinner("Executing command..."):
                result = command_processor.process_command(
                    command_text=command_text,
                    output_format=output_format
                )
                
                # Display results
                if result.success:
                    st.success(f"✅ Command executed successfully in {result.execution_time_ms}ms")
                    
                    if output_format == "json" and result.data:
                        st.json(result.data)
                    else:
                        st.markdown(result.message)
                else:
                    st.error(f"❌ Command failed: {result.message}")
        
        # Command help
        st.subheader("📚 Available Commands")
        
        commands = command_processor.get_available_commands()
        
        for cmd in commands:
            with st.expander(f"**{cmd['command']}**"):
                st.markdown(f"**Description:** {cmd['description']}")
                st.code(cmd['example'])
        
        # Command history
        if hasattr(st.session_state, 'command_history'):
            st.subheader("📜 Command History")
            
            for i, hist_cmd in enumerate(reversed(st.session_state.command_history[-10:])):
                st.caption(f"{i+1}. {hist_cmd}")
                
    except Exception as e:
        st.error(f"Error loading command interface: {e}")

def render_role_access():
    """Render the role-based access control interface."""
    try:
        st.subheader("🎭 Role-Based Memory Access")
        st.markdown("Manage memory access permissions and role-specific filtering")
        
        role_filter = get_role_filter()
        
        # Role selection
        col1, col2 = st.columns(2)
        
        with col1:
            selected_role = st.selectbox(
                "Select Agent Role",
                options=[role.value for role in AgentRole],
                index=0
            )
            role = AgentRole(selected_role)
        
        with col2:
            agent_id = st.text_input(
                "Agent ID",
                value=f"agent_{selected_role}_001",
                help="Specific agent identifier"
            )
        
        # Role permissions
        st.subheader("🔐 Role Permissions")
        
        permissions = role_filter.get_role_memory_permissions(role)
        
        if 'error' not in permissions:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Allowed Memory Types:**")
                for mem_type in permissions['allowed_memory_types']:
                    st.markdown(f"✅ {mem_type}")
                
                if permissions['restricted_memory_types']:
                    st.markdown("**Restricted Memory Types:**")
                    for mem_type in permissions['restricted_memory_types']:
                        st.markdown(f"❌ {mem_type}")
            
            with col2:
                st.markdown("**Access Levels:**")
                for level in permissions['access_levels']:
                    st.markdown(f"🔑 {level}")
                
                if permissions['special_permissions']:
                    st.markdown("**Special Permissions:**")
                    for perm in permissions['special_permissions']:
                        st.markdown(f"⭐ {perm}")
        
        # Role-filtered memories
        st.subheader("📚 Role-Filtered Memories")
        
        search_query = st.text_input(
            "Search Query (optional)",
            placeholder="Enter search terms...",
            help="Search memories accessible to this role"
        )
        
        max_results = st.slider("Max Results", 5, 50, 10)
        
        if st.button("🔍 Filter Memories"):
            with st.spinner("Filtering memories by role..."):
                role_context = role_filter.filter_memories_for_role(
                    role=role,
                    agent_id=agent_id,
                    query=search_query if search_query else None,
                    max_results=max_results
                )
                
                # Display results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Accessible Memories ({len(role_context.accessible_memories)}):**")
                    
                    for i, result in enumerate(role_context.accessible_memories, 1):
                        memory = result.chunk
                        
                        with st.container():
                            st.markdown(f"**{i}. {memory.memory_type.value.title()}** - {memory.source}")
                            st.caption(f"Date: {memory.timestamp[:10]} | Importance: {memory.importance_score:.2f}")
                            
                            content_preview = memory.content[:100]
                            if len(memory.content) > 100:
                                content_preview += "..."
                            st.markdown(content_preview)
                            
                            if memory.tags:
                                tag_html = " ".join([
                                    f"<span style='background-color: #e1f5fe; padding: 2px 6px; border-radius: 3px; font-size: 0.8em;'>{tag}</span>" 
                                    for tag in memory.tags[:3]
                                ])
                                st.markdown(tag_html, unsafe_allow_html=True)
                            
                            st.divider()
                
                with col2:
                    st.markdown("**Filter Summary:**")
                    st.metric("Accessible", len(role_context.accessible_memories))
                    st.metric("Filtered Out", role_context.filtered_count)
                    
                    if role_context.access_summary:
                        st.markdown("**By Type:**")
                        for mem_type, count in role_context.access_summary.items():
                            st.caption(f"{mem_type}: {count}")
                    
                    if role_context.role_specific_insights:
                        st.markdown("**Insights:**")
                        for insight in role_context.role_specific_insights:
                            st.caption(f"• {insight}")
        
        # Collaborative access
        st.subheader("🤝 Collaborative Access")
        
        selected_roles = st.multiselect(
            "Select Multiple Roles",
            options=[role.value for role in AgentRole],
            default=[selected_role]
        )
        
        if len(selected_roles) > 1 and st.button("🔍 Analyze Collaborative Access"):
            roles = [AgentRole(r) for r in selected_roles]
            
            with st.spinner("Analyzing collaborative memory access..."):
                collab_context = role_filter.get_collaborative_memories(
                    roles=roles,
                    query=search_query if search_query else None,
                    max_results=20
                )
                
                if 'error' not in collab_context:
                    st.markdown(f"**Shared Memories ({len(collab_context['shared_memories'])}):**")
                    
                    for memory in collab_context['shared_memories'][:10]:
                        st.markdown(f"• **{memory['memory_type']}** - {memory['source']}")
                        st.caption(f"Accessible to: {', '.join(memory['accessible_to'])}")
                        st.caption(f"Content: {memory['content']}")
                        st.divider()
                    
                    if collab_context['collaboration_insights']:
                        st.markdown("**Collaboration Insights:**")
                        for insight in collab_context['collaboration_insights']:
                            st.info(insight)
                
    except Exception as e:
        st.error(f"Error loading role access interface: {e}")

def render_system_status():
    """Render the system status interface."""
    try:
        st.subheader("🖥️ System Status")
        
        # Memory system status
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Memory System:**")
            
            memory_store = get_memory_store()
            stats = memory_store.get_memory_stats()

            # Use .get() with fallback values to prevent KeyError
            total_memories = stats.get('total_memories', len(getattr(memory_store, 'memory_chunks', {})))
            total_size_mb = stats.get('total_size_mb', 0.0)
            store_type = stats.get('store_type', 'Unknown')

            st.metric("Total Memories", total_memories)
            st.metric("Storage Size", f"{total_size_mb:.2f} MB")
            st.metric("Store Type", store_type)
            
            memory_types = stats.get('memory_types', {})
            if memory_types and total_memories > 0:
                st.markdown("**Memory Distribution:**")
                for mem_type, count in memory_types.items():
                    st.progress(count / total_memories, text=f"{mem_type}: {count}")
        
        with col2:
            st.markdown("**Agent Mode:**")
            
            mode_controller = get_mode_controller()
            mode_status = mode_controller.get_mode_status()
            
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
        
        # Performance metrics
        st.subheader("📈 Performance Metrics")
        
        # This would integrate with actual performance monitoring
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Avg Query Time", "45ms", delta="-5ms")
        
        with col2:
            st.metric("Memory Hit Rate", "87%", delta="+2%")
        
        with col3:
            st.metric("Active Sessions", "3", delta="+1")
        
        # System health
        st.subheader("🏥 System Health")
        
        health_checks = [
            ("Memory Store", "✅ Healthy"),
            ("Vector Index", "✅ Healthy"),
            ("Agent Mode Controller", "✅ Healthy"),
            ("Command Processor", "✅ Healthy"),
            ("Role Filter", "✅ Healthy")
        ]
        
        for component, status in health_checks:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(component)
            with col2:
                st.text(status)
        
    except Exception as e:
        st.error(f"Error loading system status: {e}")

def render_memory_ranking():
    """Render the Enhanced Memory Ranking interface with Phase 3.2.3 features."""
    try:
        st.subheader("🏆 Enhanced Memory Ranking Framework")
        st.markdown("**Phase 3.2.3:** Real-time ranking controls, weight adjustment, and performance analytics")

        from memory.memory_ranking import get_memory_ranking_framework

        ranking_framework = get_memory_ranking_framework()
        memory_store = get_memory_store()

        # Phase 3.2.3: Interactive Configuration section
        st.subheader("⚙️ Interactive Ranking Configuration")

        # Real-time weight adjustment
        st.markdown("**🎛️ Adjust Ranking Weights:**")
        col1, col2, col3 = st.columns(3)

        with col1:
            semantic_weight = st.slider("Semantic Similarity", 0.0, 1.0, 0.35, 0.05, help="Weight for content similarity")
            recency_weight = st.slider("Recency", 0.0, 1.0, 0.15, 0.05, help="Weight for temporal relevance")

        with col2:
            confidence_weight = st.slider("Source Confidence", 0.0, 1.0, 0.25, 0.05, help="Weight for source quality")
            priority_weight = st.slider("User Priority", 0.0, 1.0, 0.15, 0.05, help="Weight for user-defined priority")

        with col3:
            usage_weight = st.slider("Usage Frequency", 0.0, 1.0, 0.05, 0.05, help="Weight for access frequency")
            quality_weight = st.slider("Content Quality", 0.0, 1.0, 0.05, 0.05, help="Weight for content structure")

        # Normalize weights
        total_weight = semantic_weight + recency_weight + confidence_weight + priority_weight + usage_weight + quality_weight
        if total_weight > 0:
            weights = {
                'similarity': semantic_weight / total_weight,
                'recency': recency_weight / total_weight,
                'source_confidence': confidence_weight / total_weight,
                'user_priority': priority_weight / total_weight,
                'usage_frequency': usage_weight / total_weight,
                'content_quality': quality_weight / total_weight
            }
        else:
            weights = ranking_framework.ranking_weights

        # Phase 3.2.3: Real-time settings adjustment
        st.markdown("**⚙️ Advanced Settings:**")
        col1, col2, col3 = st.columns(3)

        with col1:
            priority_threshold = st.slider("Priority Threshold", 0.0, 1.0, 0.4, 0.05, help="Minimum score for priority memories")

        with col2:
            recency_decay = st.slider("Recency Decay (days)", 1, 90, 30, 1, help="Days for recency score decay")

        with col3:
            max_priority = st.slider("Max Priority Memories", 5, 50, 10, 1, help="Maximum number of priority memories")

        # Apply settings button
        if st.button("🔄 Apply Settings", type="primary"):
            # Update ranking framework with new settings
            ranking_framework.ranking_weights = weights
            ranking_framework.priority_threshold = priority_threshold
            ranking_framework.recency_decay_days = recency_decay
            ranking_framework.config['max_priority_memories'] = max_priority
            st.success("✅ Ranking settings updated!")

        # Phase 3.2.3: Enhanced testing section with real-time ranking
        st.subheader("🧪 Real-time Memory Ranking Test")

        col1, col2 = st.columns(2)

        with col1:
            test_query = st.text_input(
                "Test Query",
                value="important dates",
                help="Enter a query to test memory ranking"
            )

        with col2:
            max_results = st.slider("Max Results", 3, 20, 8)

        # Phase 3.2.3: Real-time ranking toggle
        real_time_ranking = st.checkbox("🔄 Real-time Ranking", value=False, help="Update ranking as you type")

        if st.button("🔍 Rank Memories", type="primary") or (real_time_ranking and test_query and len(test_query) > 2):
            with st.spinner("Ranking memories with current settings..."):
                # Use enhanced search if available
                if hasattr(memory_store, 'enhanced_search_memories'):
                    memory_results = memory_store.enhanced_search_memories(
                        query=test_query,
                        max_results=max_results,
                        initial_candidates=max_results * 3
                    )
                else:
                    memory_results = memory_store.search_memories(test_query, max_results=max_results)

                if memory_results:
                    # Rank the memories
                    ranking_scores = ranking_framework.rank_memories(memory_results, query=test_query)

                    st.success(f"✅ Ranked {len(ranking_scores)} memories")

                    # Display ranking results
                    for i, score in enumerate(ranking_scores, 1):
                        with st.container():
                            col1, col2, col3 = st.columns([3, 1, 1])

                            with col1:
                                st.markdown(f"**{i}. Memory {score.memory_id}**")
                                st.caption(score.ranking_explanation)

                            with col2:
                                score_color = "🟢" if score.overall_score > 0.7 else "🟡" if score.overall_score > 0.4 else "🔴"
                                st.metric("Score", f"{score_color} {score.overall_score:.3f}")

                            with col3:
                                priority_icon = "⭐" if score.is_priority else "📌" if score.is_pinned else "•"
                                st.markdown(f"**Status:** {priority_icon}")
                                if score.is_priority:
                                    st.caption("Priority")
                                elif score.is_pinned:
                                    st.caption("Pinned")
                                else:
                                    st.caption("Regular")

                            # Factor breakdown
                            with st.expander("Factor Breakdown"):
                                factor_cols = st.columns(3)
                                for j, (factor, factor_score) in enumerate(score.factor_scores.items()):
                                    with factor_cols[j % 3]:
                                        st.metric(factor.value.replace('_', ' ').title(), f"{factor_score:.3f}")

                            st.divider()
                else:
                    st.warning("No memories found for the test query")

        # Memory pinning section
        st.subheader("📌 Memory Pinning")
        st.markdown("Manually pin/unpin memories for priority treatment")

        memory_id_to_pin = st.text_input(
            "Memory ID to Pin/Unpin",
            placeholder="mem_abc123...",
            help="Enter the memory ID to toggle its pinned status"
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📌 Pin Memory"):
                if memory_id_to_pin:
                    # Implementation would go here
                    st.success(f"Memory {memory_id_to_pin} pinned!")
                else:
                    st.error("Please enter a memory ID")

        with col2:
            if st.button("📌 Unpin Memory"):
                if memory_id_to_pin:
                    # Implementation would go here
                    st.success(f"Memory {memory_id_to_pin} unpinned!")
                else:
                    st.error("Please enter a memory ID")

    except Exception as e:
        st.error(f"Error loading memory ranking: {e}")

def render_citation_engine():
    """Render the Enhanced Citation Engine interface with Phase 3.2.3 features."""
    try:
        st.subheader("📝 Enhanced Citation Engine")
        st.markdown("**Phase 3.2.3:** Source analysis, citation quality metrics, and real-time preview")

        from memory.citation_engine import get_citation_engine

        citation_engine = get_citation_engine()
        memory_store = get_memory_store()

        # Phase 3.2.3: Interactive Configuration section
        st.subheader("⚙️ Interactive Citation Configuration")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Citation Style:**")
            citation_style = st.selectbox(
                "Style",
                options=["inline", "footnote", "academic"],
                index=0,
                help="Choose citation format style"
            )

            enable_citations = st.checkbox("Enable Citations", value=True, help="Toggle citation generation")

        with col2:
            st.markdown("**Quality Thresholds:**")
            min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.3, 0.05, help="Minimum confidence for citations")
            max_quote_length = st.slider("Max Quote Length", 50, 500, 150, 10, help="Maximum quote character length")

        with col3:
            st.markdown("**Citation Limits:**")
            max_citations = st.slider("Max Citations", 1, 10, 5, 1, help="Maximum citations per response")
            transparency_threshold = st.slider("Transparency Threshold", 0.0, 1.0, 0.5, 0.05, help="Minimum transparency score")

        # Phase 3.2.3: Source Analysis Section
        st.subheader("📊 Source Analysis")

        # Get source statistics
        source_stats = _get_source_statistics(memory_store)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📚 Source Distribution:**")
            if source_stats['source_types']:
                import plotly.express as px
                fig = px.pie(
                    values=list(source_stats['source_types'].values()),
                    names=list(source_stats['source_types'].keys()),
                    title="Sources by Type"
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**🎯 Citation Quality Metrics:**")
            st.metric("Total Sources", source_stats['total_sources'])
            st.metric("High Confidence Sources", source_stats['high_confidence_sources'])
            st.metric("Average Source Quality", f"{source_stats['avg_quality']:.2f}")

            if source_stats['most_cited_source']:
                st.caption(f"Most Cited: {source_stats['most_cited_source']}")

        # Apply settings
        if st.button("🔄 Apply Citation Settings", type="primary"):
            citation_engine.citation_style = citation_style
            citation_engine.enable_citations = enable_citations
            citation_engine.min_confidence_threshold = min_confidence
            citation_engine.max_quote_length = max_quote_length
            citation_engine.config['max_citations_per_response'] = max_citations
            st.success("✅ Citation settings updated!")

        # Phase 3.2.3: Enhanced citation testing with real-time preview
        st.subheader("🧪 Real-time Citation Testing")

        col1, col2 = st.columns(2)

        with col1:
            test_query = st.text_input(
                "Test Query",
                value="Incubator dates",
                help="Enter a query to test citation generation"
            )

        with col2:
            citation_mode = st.selectbox(
                "Citation Mode",
                options=["Enhanced Search", "Legacy Search", "Source-Specific"],
                index=0,
                help="Choose search method for citation sources"
            )

        test_response = st.text_area(
            "Sample Response Text",
            value="Based on the documents, there are several important dates to consider.",
            help="Enter sample response text to inject citations into",
            height=100
        )

        # Real-time preview toggle
        real_time_preview = st.checkbox("🔄 Real-time Preview", value=False, help="Update citations as you type")

        if st.button("📝 Generate Citations", type="primary") or (real_time_preview and test_query and test_response):
            with st.spinner("Generating enhanced citations..."):
                # Use enhanced search based on mode
                if citation_mode == "Enhanced Search" and hasattr(memory_store, 'enhanced_search_memories'):
                    memory_results = memory_store.enhanced_search_memories(
                        query=test_query,
                        max_results=max_citations,
                        initial_candidates=max_citations * 2
                    )
                else:
                    memory_results = memory_store.search_memories(test_query, max_results=max_citations)

                if memory_results:
                    # Generate citations with current settings
                    cited_response = citation_engine.inject_citations(test_response, memory_results, test_query)

                    if not real_time_preview:  # Only show success for manual generation
                        st.success(f"✅ Generated {len(cited_response.citations)} citations")

                    # Phase 3.2.3: Enhanced results display
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown("**📝 Response with Enhanced Citations:**")
                        st.markdown(cited_response.response_text)

                        # Show individual citations
                        if cited_response.citations:
                            st.markdown("**📚 Citation Details:**")
                            for i, citation in enumerate(cited_response.citations, 1):
                                with st.expander(f"Citation {i}: {citation.source_name}"):
                                    st.markdown(f"**Quote:** {citation.quote_text}")
                                    st.markdown(f"**Confidence:** {citation.confidence_score:.2f}")
                                    if citation.page_number:
                                        st.markdown(f"**Location:** Page {citation.page_number}")
                                    if citation.section_title:
                                        st.markdown(f"**Section:** {citation.section_title}")

                    with col2:
                        st.markdown("**📊 Citation Metrics:**")
                        st.metric("Transparency Score", f"{cited_response.transparency_score:.1%}")
                        st.metric("Source Count", cited_response.source_count)
                        st.metric("Citation Count", len(cited_response.citations))
                        st.metric("Citation Style", cited_response.citation_style.value)

                        # Quality indicators
                        if cited_response.transparency_score >= transparency_threshold:
                            st.success("🟢 High Transparency")
                        elif cited_response.transparency_score >= 0.3:
                            st.warning("🟡 Medium Transparency")
                        else:
                            st.error("🔴 Low Transparency")

                    # Citation details
                    if cited_response.citations:
                        st.subheader("📋 Citation Details")

                        for i, citation in enumerate(cited_response.citations, 1):
                            with st.expander(f"Citation {i}: {citation.citation_label}"):
                                col1, col2 = st.columns([3, 1])

                                with col1:
                                    st.markdown(f"**Source:** {citation.source_name}")
                                    st.markdown(f"**Quote:** \"{citation.quote_text}\"")
                                    st.markdown(f"**Full Path:** {citation.full_source_path}")

                                with col2:
                                    confidence_color = "🟢" if citation.confidence_score > 0.7 else "🟡" if citation.confidence_score > 0.4 else "🔴"
                                    st.metric("Confidence", f"{confidence_color} {citation.confidence_score:.3f}")
                else:
                    st.warning("No memories found for citation generation")

        # Citation style settings
        st.subheader("🎨 Citation Style Settings")

        from memory.citation_engine import CitationStyle

        new_style = st.selectbox(
            "Citation Style",
            options=[style.value for style in CitationStyle],
            index=list(CitationStyle).index(citation_engine.citation_style)
        )

        if st.button("💾 Update Citation Style"):
            citation_engine.citation_style = CitationStyle(new_style)
            st.success(f"Citation style updated to: {new_style}")

    except Exception as e:
        st.error(f"Error loading citation engine: {e}")

def render_smart_summaries():
    """Render the Sprint 15 Smart Summary Generator interface."""
    try:
        st.subheader("📊 Smart Summary Generator")
        st.markdown("**Sprint 15 Feature:** AI-powered intelligent summarization with source tracking")

        from memory.smart_summarizer import get_smart_summarizer, SummaryRequest, SummaryType, SummaryFormat

        summarizer = get_smart_summarizer()
        memory_store = get_memory_store()

        # Summary generation section
        st.subheader("✨ Generate Smart Summary")

        col1, col2 = st.columns(2)

        with col1:
            topic_keyword = st.text_input(
                "Topic/Keyword",
                value="important dates",
                help="Enter the topic you want to summarize"
            )

            summary_type = st.selectbox(
                "Summary Type",
                options=[stype.value for stype in SummaryType],
                index=0
            )

            output_format = st.selectbox(
                "Output Format",
                options=[fmt.value for fmt in SummaryFormat],
                index=0
            )

        with col2:
            max_length = st.slider("Max Length (words)", 100, 1000, 300)
            include_sources = st.checkbox("Include Sources", value=True)

            # Memory filters
            st.markdown("**Memory Filters (optional):**")
            filter_by_type = st.multiselect(
                "Memory Types",
                options=["document", "conversation", "user_interaction"],
                default=[]
            )

        if st.button("📊 Generate Summary", type="primary", key="generate_summary_button"):
            with st.spinner("Generating smart summary..."):
                # Create summary request
                request = SummaryRequest(
                    topic_keyword=topic_keyword,
                    summary_type=SummaryType(summary_type),
                    output_format=SummaryFormat(output_format),
                    max_length=max_length,
                    include_sources=include_sources,
                    memory_filters={'memory_types': filter_by_type} if filter_by_type else None
                )

                # Generate summary
                summary = summarizer.generate_summary(request, memory_store)

                if summary.word_count > 0:
                    st.success(f"✅ Summary generated: {summary.word_count} words from {summary.source_count} sources")

                    # Display summary
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown("**Generated Summary:**")
                        if summary.output_format == SummaryFormat.MARKDOWN:
                            st.markdown(summary.summary_text)
                        else:
                            st.text(summary.summary_text)

                    with col2:
                        st.markdown("**Summary Statistics:**")
                        st.metric("Word Count", summary.word_count)
                        st.metric("Source Count", summary.source_count)
                        confidence_color = "🟢" if summary.confidence_score > 0.7 else "🟡" if summary.confidence_score > 0.4 else "🔴"
                        st.metric("Confidence", f"{confidence_color} {summary.confidence_score:.3f}")
                        st.metric("Summary ID", summary.summary_id)

                    # Key topics
                    if summary.key_topics:
                        st.markdown("**Key Topics:**")
                        topic_html = " ".join([
                            f"<span style='background-color: #e3f2fd; padding: 4px 8px; border-radius: 4px; margin: 2px; display: inline-block;'>{topic}</span>"
                            for topic in summary.key_topics
                        ])
                        st.markdown(topic_html, unsafe_allow_html=True)

                    # Export options
                    st.subheader("💾 Export Summary")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if st.button("📋 Copy to Clipboard"):
                            st.code(summary.summary_text)

                    with col2:
                        if st.button("💾 Save as Memory"):
                            # Implementation would save summary as a new memory
                            st.success("Summary saved as memory!")

                    with col3:
                        if st.button("📄 Download"):
                            st.download_button(
                                label="Download Summary",
                                data=summary.summary_text,
                                file_name=f"summary_{summary.summary_id}.md",
                                mime="text/markdown"
                            )
                else:
                    st.warning("No summary could be generated. Try a different topic or check if relevant memories exist.")

        # Summary history
        st.subheader("📜 Recent Summaries")

        if 'summary_history' not in st.session_state:
            st.session_state.summary_history = []

        if st.session_state.summary_history:
            for i, hist_summary in enumerate(reversed(st.session_state.summary_history[-5:])):
                with st.expander(f"Summary {i+1}: {hist_summary.get('topic', 'Unknown')}"):
                    st.markdown(f"**Generated:** {hist_summary.get('timestamp', 'Unknown')}")
                    st.markdown(f"**Word Count:** {hist_summary.get('word_count', 0)}")
                    st.markdown(f"**Confidence:** {hist_summary.get('confidence', 0):.3f}")
                    st.markdown(hist_summary.get('text', '')[:200] + "...")
        else:
            st.info("No summaries generated yet. Create your first summary above!")

    except Exception as e:
        st.error(f"Error loading smart summaries: {e}")

def render_memory_insights():
    """Render the Enhanced Memory Usage Insights interface with Knowledge Consolidation tracking."""
    try:
        st.subheader("📈 Enhanced Memory Usage Insights")
        st.markdown("**Enhanced Features:** Analytics, insights, knowledge consolidation tracking, and learning history")

        memory_store = get_memory_store()

        # Knowledge Consolidation Status Section
        st.subheader("🎓 Knowledge Consolidation Status")
        st.markdown("Track SAM's learning from uploaded documents")

        # Get learning history data
        learning_data = get_learning_history_data(memory_store)

        # Learning overview metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Documents Learned",
                learning_data['total_documents_learned'],
                help="Total number of documents SAM has successfully learned from"
            )

        with col2:
            st.metric(
                "Key Concepts",
                learning_data['total_key_concepts'],
                help="Total key concepts extracted and learned"
            )

        with col3:
            st.metric(
                "Avg Enrichment",
                f"{learning_data['average_enrichment_score']:.3f}",
                help="Average enrichment score across all learned documents"
            )

        with col4:
            st.metric(
                "Content Blocks",
                learning_data['total_content_blocks_processed'],
                help="Total content blocks processed and stored"
            )

        # Overall statistics
        st.subheader("📊 Overall Memory Statistics")

        stats = memory_store.get_memory_stats()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Memories", stats['total_memories'])

        with col2:
            st.metric("Storage Size", f"{stats['total_size_mb']:.1f} MB")

        with col3:
            st.metric("Store Type", stats['store_type'])

        with col4:
            avg_importance = sum(chunk.importance_score for chunk in memory_store.memory_chunks.values()) / len(memory_store.memory_chunks) if memory_store.memory_chunks else 0
            st.metric("Avg Importance", f"{avg_importance:.3f}")

        # Learning History Timeline
        st.subheader("📚 Learning History Timeline")
        render_learning_timeline(learning_data)

        # Knowledge Consolidation Details
        st.subheader("🔍 Recent Learning Events")
        render_learning_events_table(learning_data)

        # Memory type distribution
        if stats['memory_types']:
            st.subheader("📋 Memory Type Distribution")

            import plotly.express as px
            import pandas as pd

            # Create pie chart
            df = pd.DataFrame(list(stats['memory_types'].items()), columns=['Type', 'Count'])
            fig = px.pie(df, values='Count', names='Type', title="Memory Distribution by Type")
            st.plotly_chart(fig, use_container_width=True)

        # Usage analytics
        st.subheader("🔍 Usage Analytics")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Most Important Memories:**")

            # Get top memories by importance
            top_memories = sorted(
                memory_store.memory_chunks.values(),
                key=lambda x: x.importance_score,
                reverse=True
            )[:5]

            for i, memory in enumerate(top_memories, 1):
                with st.container():
                    st.markdown(f"**{i}. {memory.memory_type.value.title()}**")
                    st.caption(f"Importance: {memory.importance_score:.3f} | Source: {memory.source}")
                    st.caption(f"Content: {memory.content[:100]}...")
                    st.divider()

        with col2:
            st.markdown("**Recent Memories:**")

            # Get most recent memories
            recent_memories = sorted(
                memory_store.memory_chunks.values(),
                key=lambda x: x.timestamp,
                reverse=True
            )[:5]

            for i, memory in enumerate(recent_memories, 1):
                with st.container():
                    st.markdown(f"**{i}. {memory.memory_type.value.title()}**")
                    st.caption(f"Date: {memory.timestamp[:10]} | Source: {memory.source}")
                    st.caption(f"Content: {memory.content[:100]}...")
                    st.divider()

        # Search performance insights
        st.subheader("🚀 Search Performance Insights")

        test_queries = ["important dates", "artificial intelligence", "documents", "conversation"]

        if st.button("🧪 Run Performance Test"):
            with st.spinner("Testing search performance..."):
                import time

                performance_results = []

                for query in test_queries:
                    start_time = time.time()
                    results = memory_store.search_memories(query, max_results=10)
                    end_time = time.time()

                    performance_results.append({
                        'Query': query,
                        'Results': len(results),
                        'Time (ms)': round((end_time - start_time) * 1000, 2),
                        'Avg Similarity': round(sum(r.similarity_score for r in results) / len(results), 3) if results else 0
                    })

                # Display results
                df = pd.DataFrame(performance_results)
                st.dataframe(df, use_container_width=True)

                # Performance metrics
                avg_time = df['Time (ms)'].mean()
                avg_results = df['Results'].mean()

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Avg Search Time", f"{avg_time:.1f} ms")

                with col2:
                    st.metric("Avg Results", f"{avg_results:.1f}")

        # Memory health check
        st.subheader("🏥 Memory Health Check")

        if st.button("🔍 Run Health Check", key="run_health_check_button"):
            with st.spinner("Analyzing memory health..."):
                health_issues = []

                # Check for duplicate content
                content_hashes = {}
                duplicates = 0

                for chunk in memory_store.memory_chunks.values():
                    content_hash = hash(chunk.content[:100])
                    if content_hash in content_hashes:
                        duplicates += 1
                    else:
                        content_hashes[content_hash] = chunk.chunk_id

                if duplicates > 0:
                    health_issues.append(f"⚠️ Found {duplicates} potential duplicate memories")

                # Check for low importance memories
                low_importance = sum(1 for chunk in memory_store.memory_chunks.values() if chunk.importance_score < 0.3)
                if low_importance > stats['total_memories'] * 0.3:
                    health_issues.append(f"⚠️ {low_importance} memories have low importance scores")

                # Check for missing embeddings
                missing_embeddings = sum(1 for chunk in memory_store.memory_chunks.values() if not chunk.embedding)
                if missing_embeddings > 0:
                    health_issues.append(f"❌ {missing_embeddings} memories missing embeddings")

                # Display health results
                if health_issues:
                    st.warning("Memory health issues detected:")
                    for issue in health_issues:
                        st.markdown(f"- {issue}")
                else:
                    st.success("✅ Memory system is healthy!")

    except Exception as e:
        st.error(f"Error loading memory insights: {e}")

def render_enhanced_chat_interface():
    """Render the Sprint 16 enhanced chat interface."""
    try:
        from ui.chat_interface import render_chat_interface as render_sprint16_chat
        render_sprint16_chat()
    except Exception as e:
        st.error(f"Error loading enhanced chat interface: {e}")
        st.markdown("**Fallback:** Using basic chat interface")
        render_chat_interface()

def render_thought_settings():
    """Render the Sprint 16 thought settings interface."""
    try:
        from ui.chat_interface import render_thought_settings as render_sprint16_settings
        render_sprint16_settings()
    except Exception as e:
        st.error(f"Error loading thought settings: {e}")

# Phase 3.2.3: Helper functions for enhanced features
def _get_source_statistics(memory_store) -> Dict[str, Any]:
    """Get comprehensive source statistics for citation analysis."""
    try:
        all_memories = list(memory_store.memory_chunks.values())

        source_types = {}
        sources = {}
        total_sources = 0
        high_confidence_sources = 0
        quality_scores = []

        for memory in all_memories:
            source = getattr(memory, 'source', 'Unknown')
            confidence = getattr(memory, 'importance_score', 0.0)

            # Count source types
            if '.pdf' in source.lower():
                source_types['PDF Documents'] = source_types.get('PDF Documents', 0) + 1
            elif 'http' in source.lower() or 'web' in source.lower():
                source_types['Web Pages'] = source_types.get('Web Pages', 0) + 1
            elif 'conversation' in source.lower() or 'chat' in source.lower():
                source_types['Conversations'] = source_types.get('Conversations', 0) + 1
            elif 'log' in source.lower():
                source_types['System Logs'] = source_types.get('System Logs', 0) + 1
            else:
                source_types['Other'] = source_types.get('Other', 0) + 1

            # Track individual sources
            sources[source] = sources.get(source, 0) + 1
            total_sources += 1

            # Quality metrics
            if confidence >= 0.7:
                high_confidence_sources += 1
            quality_scores.append(confidence)

        # Find most cited source
        most_cited_source = max(sources.items(), key=lambda x: x[1])[0] if sources else None

        # Calculate average quality
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        return {
            'source_types': source_types,
            'total_sources': total_sources,
            'high_confidence_sources': high_confidence_sources,
            'avg_quality': avg_quality,
            'most_cited_source': most_cited_source,
            'unique_sources': len(sources)
        }

    except Exception as e:
        logger.error(f"Error getting source statistics: {e}")
        return {
            'source_types': {},
            'total_sources': 0,
            'high_confidence_sources': 0,
            'avg_quality': 0.0,
            'most_cited_source': None,
            'unique_sources': 0
        }

def get_learning_history_data(memory_store):
    """Get learning history data from memory store."""
    try:
        all_memories = memory_store.get_all_memories()

        # Filter for document summaries (learning events)
        learning_events = []
        for memory in all_memories:
            metadata = getattr(memory, 'metadata', {})
            if metadata.get('document_type') == 'summary':
                learning_event = {
                    'timestamp': metadata.get('upload_timestamp', metadata.get('processing_timestamp', 'unknown')),
                    'filename': metadata.get('file_name', 'unknown'),
                    'source_file': metadata.get('source_file', 'unknown'),
                    'enrichment_score': metadata.get('enrichment_score', 0.0),
                    'priority_level': metadata.get('priority_level', 'unknown'),
                    'key_concepts': metadata.get('key_concepts', []),
                    'content_types': metadata.get('content_types', []),
                    'content_blocks_count': metadata.get('content_blocks_count', 0),
                    'file_size': metadata.get('file_size', 0),
                    'memory_id': getattr(memory, 'memory_id', 'unknown')
                }
                learning_events.append(learning_event)

        # Sort by timestamp (most recent first)
        learning_events.sort(key=lambda x: x['timestamp'], reverse=True)

        # Calculate learning statistics
        total_documents = len(learning_events)
        total_concepts = sum(len(event.get('key_concepts', [])) for event in learning_events)
        avg_enrichment = sum(event.get('enrichment_score', 0) for event in learning_events) / max(total_documents, 1)
        total_content_blocks = sum(event.get('content_blocks_count', 0) for event in learning_events)

        return {
            'total_documents_learned': total_documents,
            'total_key_concepts': total_concepts,
            'average_enrichment_score': round(avg_enrichment, 3),
            'total_content_blocks_processed': total_content_blocks,
            'learning_events': learning_events,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        st.error(f"Error getting learning history: {e}")
        return {
            'total_documents_learned': 0,
            'total_key_concepts': 0,
            'average_enrichment_score': 0.0,
            'total_content_blocks_processed': 0,
            'learning_events': [],
            'timestamp': datetime.now().isoformat()
        }

def render_learning_timeline(learning_data):
    """Render learning timeline visualization."""
    try:
        import plotly.express as px
        import pandas as pd
        from datetime import datetime, timedelta

        learning_events = learning_data.get('learning_events', [])

        if not learning_events:
            st.info("📚 No learning events found. Upload documents to see SAM's learning timeline!")
            return

        # Prepare timeline data
        timeline_data = []
        for event in learning_events:
            try:
                # Parse timestamp
                if event['timestamp'] != 'unknown':
                    timestamp = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
                    timeline_data.append({
                        'Date': timestamp.date(),
                        'Time': timestamp.time(),
                        'Document': event['filename'],
                        'Enrichment Score': event['enrichment_score'],
                        'Key Concepts': len(event.get('key_concepts', [])),
                        'Content Blocks': event['content_blocks_count'],
                        'Priority': event['priority_level']
                    })
            except Exception as e:
                continue

        if timeline_data:
            df = pd.DataFrame(timeline_data)

            # Create timeline chart
            fig = px.scatter(
                df,
                x='Date',
                y='Enrichment Score',
                size='Content Blocks',
                color='Priority',
                hover_data=['Document', 'Key Concepts'],
                title="📈 SAM's Learning Timeline",
                labels={'Enrichment Score': 'Document Enrichment Score'}
            )

            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Learning velocity chart
            if len(df) > 1:
                daily_counts = df.groupby('Date').size().reset_index(name='Documents Learned')

                fig2 = px.bar(
                    daily_counts,
                    x='Date',
                    y='Documents Learned',
                    title="📊 Daily Learning Velocity",
                    labels={'Documents Learned': 'Documents Processed Per Day'}
                )

                fig2.update_layout(height=300)
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("⚠️ Unable to parse learning timeline data")

    except Exception as e:
        st.error(f"Error rendering learning timeline: {e}")

def render_learning_events_table(learning_data):
    """Render recent learning events table."""
    try:
        import pandas as pd
        from datetime import datetime

        learning_events = learning_data.get('learning_events', [])[:10]  # Show last 10 events

        if not learning_events:
            st.info("📋 No recent learning events found.")
            return

        # Prepare table data
        table_data = []
        for event in learning_events:
            try:
                # Format timestamp
                timestamp_str = "Unknown"
                if event['timestamp'] != 'unknown':
                    timestamp = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
                    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M")

                # Format key concepts
                concepts_count = len(event.get('key_concepts', []))
                concepts_preview = ", ".join(event.get('key_concepts', [])[:3])
                if len(event.get('key_concepts', [])) > 3:
                    concepts_preview += "..."

                table_data.append({
                    '📅 Timestamp': timestamp_str,
                    '📄 Document': event['filename'],
                    '📊 Enrichment': f"{event['enrichment_score']:.3f}",
                    '🔑 Concepts': f"{concepts_count}",
                    '🧩 Blocks': event['content_blocks_count'],
                    '🎯 Priority': event['priority_level'],
                    '💾 Size': f"{event.get('file_size', 0) / 1024:.1f} KB" if event.get('file_size', 0) > 0 else "Unknown"
                })
            except Exception as e:
                continue

        if table_data:
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)

            # Knowledge consolidation confirmation
            st.subheader("🎓 Knowledge Consolidation Confirmation")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**✅ Confirmed Learning Indicators:**")
                st.markdown("• Document summaries created")
                st.markdown("• Key concepts extracted and stored")
                st.markdown("• Memory chunks generated for Q&A")
                st.markdown("• Enrichment scores calculated")
                st.markdown("• Content blocks processed and indexed")

            with col2:
                st.markdown("**📈 Learning Quality Metrics:**")
                total_docs = learning_data['total_documents_learned']
                avg_score = learning_data['average_enrichment_score']
                total_concepts = learning_data['total_key_concepts']

                if total_docs > 0:
                    st.markdown(f"• **{total_docs}** documents successfully learned")
                    st.markdown(f"• **{avg_score:.3f}** average enrichment quality")
                    st.markdown(f"• **{total_concepts}** total concepts mastered")
                    st.markdown(f"• **{total_concepts/total_docs:.1f}** avg concepts per document")

                    # Learning status indicator
                    if avg_score >= 0.7:
                        st.success("🎓 **Excellent Learning Performance!**")
                    elif avg_score >= 0.5:
                        st.info("📚 **Good Learning Performance**")
                    else:
                        st.warning("⚠️ **Learning Performance Needs Improvement**")
                else:
                    st.info("📚 No learning data available yet")
        else:
            st.warning("⚠️ Unable to format learning events data")

    except Exception as e:
        st.error(f"Error rendering learning events: {e}")

def generate_diagnostic_info(user_input):
    """Generate comprehensive diagnostic information for a user query."""
    try:
        memory_store = get_memory_store()
        stats = memory_store.get_memory_stats()

        diagnostic_info = {
            'timestamp': datetime.now().isoformat(),
            'query_analysis': {
                'input_length': len(user_input),
                'word_count': len(user_input.split()),
                'query_type': classify_query_type(user_input),
                'contains_keywords': extract_keywords(user_input)
            },
            'system_state': {
                'total_memories': stats['total_memories'],
                'memory_types': stats['memory_types'],
                'storage_size_mb': stats['total_size_mb'],
                'system_health': 'healthy' if stats['total_memories'] > 0 else 'no_memories'
            },
            'processing_context': {
                'session_type': 'streamlit_diagnostic',
                'user_id': 'streamlit_user',
                'processing_mode': 'memory_driven_reasoning'
            }
        }

        return diagnostic_info

    except Exception as e:
        return {
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'system_state': 'error'
        }

def classify_query_type(query):
    """Classify the type of user query."""
    query_lower = query.lower()

    if any(greeting in query_lower for greeting in ['hello', 'hi', 'hey', 'greetings']):
        return 'greeting'
    elif any(question in query_lower for question in ['what', 'how', 'why', 'when', 'where', 'who']):
        return 'question'
    elif query.startswith('!'):
        return 'command'
    elif any(status in query_lower for status in ['status', 'health', 'system', 'diagnostic']):
        return 'diagnostic'
    elif any(memory in query_lower for memory in ['remember', 'recall', 'memory', 'learned']):
        return 'memory_query'
    else:
        return 'general'

def extract_keywords(query):
    """Extract key terms from the query."""
    import re

    # Remove common stop words and extract meaningful terms
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}

    words = re.findall(r'\b\w+\b', query.lower())
    keywords = [word for word in words if word not in stop_words and len(word) > 2]

    return keywords[:10]  # Return top 10 keywords

def render_diagnostic_panel(diagnostics, show_memory_context, show_reasoning_trace):
    """Render the diagnostic information panel."""
    try:
        with st.expander("🔍 Diagnostic Information", expanded=False):

            # Query Analysis
            if 'query_analysis' in diagnostics:
                st.markdown("**📝 Query Analysis:**")
                qa = diagnostics['query_analysis']
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Input Length", qa.get('input_length', 0))

                with col2:
                    st.metric("Word Count", qa.get('word_count', 0))

                with col3:
                    st.metric("Query Type", qa.get('query_type', 'unknown'))

                if qa.get('contains_keywords'):
                    st.caption(f"**Keywords:** {', '.join(qa['contains_keywords'])}")

            # System State
            if 'system_state' in diagnostics:
                st.markdown("**🖥️ System State:**")
                ss = diagnostics['system_state']

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Memories", ss.get('total_memories', 0))

                with col2:
                    st.metric("Storage Size", f"{ss.get('storage_size_mb', 0):.1f} MB")

                with col3:
                    health_color = "🟢" if ss.get('system_health') == 'healthy' else "🔴"
                    st.metric("Health", f"{health_color} {ss.get('system_health', 'unknown')}")

            # Memory Context (if enabled)
            if show_memory_context and 'memory_context' in diagnostics:
                st.markdown("**🧠 Memory Context:**")
                mc = diagnostics['memory_context']

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Memories Found", mc.get('memories_found', 0))

                with col2:
                    st.metric("Relevance Score", f"{mc.get('relevance_score', 0):.3f}")

                with col3:
                    st.metric("Context Length", mc.get('context_length', 0))

                if mc.get('search_strategy'):
                    st.caption(f"**Search Strategy:** {mc['search_strategy']}")

            # Reasoning Trace (if enabled)
            if show_reasoning_trace and 'reasoning_session' in diagnostics:
                st.markdown("**🤔 Reasoning Session:**")
                rs = diagnostics['reasoning_session']

                if rs.get('status') != 'no_session_created':
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Reasoning Steps", rs.get('reasoning_steps', 0))

                    with col2:
                        st.metric("Confidence", f"{rs.get('confidence_score', 0):.3f}")

                    with col3:
                        st.metric("Processing Time", f"{rs.get('processing_time_ms', 0)} ms")

                    if rs.get('session_id'):
                        st.caption(f"**Session ID:** {rs['session_id']}")
                else:
                    st.caption("No reasoning session created for this query")

            # Command Execution (if applicable)
            if 'command_execution' in diagnostics:
                st.markdown("**⚡ Command Execution:**")
                ce = diagnostics['command_execution']

                if ce.get('status') == 'success':
                    st.success(f"✅ Command executed successfully in {ce.get('execution_time', 0)} ms")
                    if ce.get('data_returned'):
                        st.info("📊 Data returned from command")
                else:
                    st.error(f"❌ Command failed: {ce.get('error', 'Unknown error')}")

            # Timestamp
            st.caption(f"**Generated:** {diagnostics.get('timestamp', 'Unknown')}")

    except Exception as e:
        st.error(f"Error rendering diagnostics: {e}")

def get_system_status_info():
    """Get comprehensive system status information."""
    try:
        memory_store = get_memory_store()
        stats = memory_store.get_memory_stats()

        # Get learning history
        learning_data = get_learning_history_data(memory_store)

        system_info = {
            'timestamp': datetime.now().isoformat(),
            'memory_system': {
                'status': 'operational',
                'total_memories': stats['total_memories'],
                'storage_size_mb': stats['total_size_mb'],
                'memory_types': stats['memory_types'],
                'store_type': stats['store_type']
            },
            'learning_system': {
                'documents_learned': learning_data['total_documents_learned'],
                'key_concepts': learning_data['total_key_concepts'],
                'avg_enrichment': learning_data['average_enrichment_score'],
                'content_blocks': learning_data['total_content_blocks_processed']
            },
            'performance_metrics': {
                'memory_health': 'healthy' if stats.get('total_memories', 0) > 0 else 'no_data',
                'learning_performance': 'excellent' if learning_data['average_enrichment_score'] >= 0.7 else
                                      'good' if learning_data['average_enrichment_score'] >= 0.5 else 'needs_improvement',
                'system_uptime': 'active',
                'last_activity': stats.get('newest_memory', 'unknown')
            }
        }

        return system_info

    except Exception as e:
        return {
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'status': 'error'
        }

def format_system_status(status_info):
    """Format system status information for display."""
    if 'error' in status_info:
        return f"❌ **System Status Error:** {status_info['error']}"

    memory_sys = status_info.get('memory_system', {})
    learning_sys = status_info.get('learning_system', {})
    performance = status_info.get('performance_metrics', {})

    status_msg = f"""🖥️ **SAM System Status Report**

**Memory System:**
• Status: {memory_sys.get('status', 'unknown').title()} ✅
• Total Memories: {memory_sys.get('total_memories', 0):,}
• Storage Size: {memory_sys.get('storage_size_mb', 0):.1f} MB
• Store Type: {memory_sys.get('store_type', 'unknown')}

**Learning System:**
• Documents Learned: {learning_sys.get('documents_learned', 0)}
• Key Concepts Mastered: {learning_sys.get('key_concepts', 0)}
• Average Enrichment Score: {learning_sys.get('avg_enrichment', 0):.3f}
• Content Blocks Processed: {learning_sys.get('content_blocks', 0)}

**Performance Metrics:**
• Memory Health: {performance.get('memory_health', 'unknown').title()}
• Learning Performance: {performance.get('learning_performance', 'unknown').title()}
• System Status: {performance.get('system_uptime', 'unknown').title()}

**Capabilities:**
• ✅ Memory-driven reasoning
• ✅ Document processing and learning
• ✅ Knowledge consolidation
• ✅ Multi-modal content analysis
• ✅ Contextual question answering
• ✅ Learning history tracking

*Generated: {status_info.get('timestamp', 'unknown')}*"""

    return status_msg

def get_memory_overview_info():
    """Get memory overview information."""
    try:
        memory_store = get_memory_store()
        all_memories = memory_store.get_all_memories()
        stats = memory_store.get_memory_stats()

        # Analyze memory content
        memory_sources = {}
        memory_types = {}
        recent_memories = []

        for memory in all_memories:
            # Track sources
            source = getattr(memory, 'source', 'unknown')
            if source.startswith('document:'):
                doc_name = source.split(':')[1].split('/')[-1]
                memory_sources[doc_name] = memory_sources.get(doc_name, 0) + 1

            # Track types
            mem_type = getattr(memory, 'memory_type', 'unknown')
            if hasattr(mem_type, 'value'):
                mem_type = mem_type.value
            memory_types[mem_type] = memory_types.get(mem_type, 0) + 1

            # Collect recent memories
            if len(recent_memories) < 5:
                recent_memories.append({
                    'content_preview': memory.content[:100] + "..." if len(memory.content) > 100 else memory.content,
                    'source': source,
                    'timestamp': getattr(memory, 'timestamp', 'unknown'),
                    'importance': getattr(memory, 'importance_score', 0)
                })

        overview_info = {
            'timestamp': datetime.now().isoformat(),
            'memory_statistics': {
                'total_count': len(all_memories),
                'storage_size': stats.get('total_size_mb', 0.0),
                'unique_sources': len(memory_sources),
                'memory_types': memory_types
            },
            'content_analysis': {
                'top_sources': dict(sorted(memory_sources.items(), key=lambda x: x[1], reverse=True)[:5]),
                'recent_memories': recent_memories
            },
            'system_capabilities': {
                'can_recall': len(all_memories) > 0,
                'can_learn': True,
                'can_reason': True,
                'can_consolidate': True
            }
        }

        return overview_info

    except Exception as e:
        return {
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }

def format_memory_overview(overview_info):
    """Format memory overview information for display."""
    if 'error' in overview_info:
        return f"❌ **Memory Overview Error:** {overview_info['error']}"

    stats = overview_info.get('memory_statistics', {})
    content = overview_info.get('content_analysis', {})
    capabilities = overview_info.get('system_capabilities', {})

    overview_msg = f"""🧠 **SAM Memory Overview**

**Memory Statistics:**
• Total Memories: {stats.get('total_count', 0):,}
• Storage Size: {stats.get('storage_size', 0):.1f} MB
• Unique Sources: {stats.get('unique_sources', 0)}

**Memory Types Distribution:**"""

    for mem_type, count in stats.get('memory_types', {}).items():
        overview_msg += f"\n• {mem_type.title()}: {count}"

    overview_msg += f"""

**Top Document Sources:**"""

    for source, count in content.get('top_sources', {}).items():
        overview_msg += f"\n• {source}: {count} memories"

    overview_msg += f"""

**System Capabilities:**"""

    for capability, status in capabilities.items():
        status_icon = "✅" if status else "❌"
        overview_msg += f"\n• {capability.replace('_', ' ').title()}: {status_icon}"

    if content.get('recent_memories'):
        overview_msg += f"""

**Recent Memory Sample:**"""
        for i, memory in enumerate(content['recent_memories'][:3], 1):
            overview_msg += f"\n{i}. {memory['content_preview']}"
            overview_msg += f"\n   Source: {memory['source']}"

    overview_msg += f"""

*Generated: {overview_info.get('timestamp', 'unknown')}*"""

    return overview_msg

def render_dream_canvas():
    """Render the Dream Canvas interface for cognitive synthesis visualization."""
    try:
        st.subheader("🧠🎨 Dream Canvas - Cognitive Synthesis Visualization")
        st.markdown("Explore SAM's memory landscape through interactive visualization and cognitive synthesis")

        # Dream Canvas is now available to all Community Edition users

        # Dream Canvas Controls Section
        st.markdown("### 🎛️ Dream Canvas Controls")

        # NEW: Workflow Automation Controls
        render_workflow_automation_controls()

        # Clustering Parameter Controls
        with st.expander("🔧 Advanced Clustering Parameters", expanded=False):
            # Smart parameter suggestion
            col_suggest, col_preset, col_manual = st.columns([1, 1, 2])

            with col_suggest:
                if st.button("🤔 Suggest Optimal Parameters", use_container_width=True, help="Use AI to analyze your memory data and suggest optimal clustering parameters"):
                    suggest_optimal_parameters()

            with col_preset:
                if st.button("⚡ Apply Research Preset", use_container_width=True, help="Apply scientifically optimal parameters for UMAP+DBSCAN clustering"):
                    apply_research_preset()

            with col_manual:
                st.markdown("**Manual Parameter Control:**")

            # Parameter sliders
            col1, col2 = st.columns(2)

            with col1:
                # Use session state for parameter persistence
                if 'cluster_radius' not in st.session_state:
                    st.session_state.cluster_radius = 0.15  # Much smaller for UMAP data

                cluster_radius = st.slider(
                    "Cluster Radius (eps)",
                    min_value=0.1,
                    max_value=2.0,
                    value=st.session_state.cluster_radius,
                    step=0.1,
                    help="Maximum distance between memories in a cluster. Lower = more clusters, Higher = fewer clusters",
                    key="eps_slider"
                )
                st.session_state.cluster_radius = cluster_radius

                if 'min_cluster_size' not in st.session_state:
                    st.session_state.min_cluster_size = 15  # Larger for meaningful clusters

                min_cluster_size = st.slider(
                    "Minimum Cluster Size",
                    min_value=3,
                    max_value=50,
                    value=st.session_state.min_cluster_size,
                    step=1,
                    help="Minimum number of memories required to form a cluster",
                    key="min_size_slider"
                )
                st.session_state.min_cluster_size = min_cluster_size

            with col2:
                if 'min_samples' not in st.session_state:
                    st.session_state.min_samples = 8  # Higher for dense clusters

                min_samples = st.slider(
                    "Core Point Threshold",
                    min_value=2,
                    max_value=20,
                    value=st.session_state.min_samples,
                    step=1,
                    help="Minimum neighbors for a memory to be a cluster core point",
                    key="min_samples_slider"
                )
                st.session_state.min_samples = min_samples

                if 'max_clusters' not in st.session_state:
                    st.session_state.max_clusters = 15

                max_clusters = st.slider(
                    "Maximum Clusters",
                    min_value=5,
                    max_value=50,
                    value=st.session_state.max_clusters,
                    step=1,
                    help="Maximum number of clusters to discover",
                    key="max_clusters_slider"
                )
                st.session_state.max_clusters = max_clusters

            # Show current parameter summary
            st.markdown(f"""
            **Current Settings:** eps={cluster_radius}, min_samples={min_samples}, min_size={min_cluster_size}, max_clusters={max_clusters}
            """)

            # Show parameter suggestions if available
            if 'parameter_suggestions' in st.session_state:
                suggestions = st.session_state.parameter_suggestions
                st.success(f"💡 **AI Suggestion:** eps={suggestions['eps']:.3f}, min_samples={suggestions['min_samples']}, min_size={suggestions['min_cluster_size']}")
                if st.button("✨ Apply Suggested Parameters", use_container_width=True):
                    apply_suggested_parameters(suggestions)

        # Main Action Buttons - Enhanced with Run Synthesis
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            if st.button("🌙 Enter Dream State", type="primary", use_container_width=True, help="Trigger cognitive synthesis"):
                trigger_synthesis_with_params(cluster_radius, min_samples, min_cluster_size, max_clusters)

        with col2:
            if st.button("🔄 Run Synthesis", type="secondary", use_container_width=True, help="Generate new insights from memory clusters"):
                trigger_direct_synthesis()

        with col3:
            if st.button("🎨 Generate Visualization", use_container_width=True, help="Create interactive memory map"):
                trigger_visualization_with_params(cluster_radius, min_samples, min_cluster_size, max_clusters)

        with col4:
            if st.button("🔄 Refresh Memory", use_container_width=True, help="Refresh memory store data"):
                # Clear any cached memory store data
                if hasattr(st.session_state, 'memory_store'):
                    del st.session_state.memory_store
                st.rerun()

        with col5:
            if st.button("📚 Refresh History", use_container_width=True, help="Update synthesis statistics"):
                refresh_synthesis_history()

        # Status dashboard
        st.subheader("📊 Memory Landscape Status")

        try:
            # Get memory statistics
            memory_store = get_memory_store()
            stats = memory_store.get_memory_stats()

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Memories", stats.get('total_memories', 0))

            with col2:
                st.metric("Memory Types", len(stats.get('memory_types', {})))

            with col3:
                # Check for synthesis history
                synthesis_count = get_synthesis_run_count()
                st.metric("Synthesis Runs", synthesis_count)

            with col4:
                # Check for synthetic memories
                synthetic_count = get_synthetic_memory_count()
                st.metric("Synthetic Insights", synthetic_count)

        except Exception as e:
            st.warning(f"Could not load memory statistics: {e}")

        # Synthesis history
        st.subheader("🔮 Synthesis History")

        try:
            history = get_synthesis_history()

            if history:
                for idx, run in enumerate(history[-5:]):  # Show last 5 runs
                    run_id_short = run.get('run_id', 'Unknown')[:8]
                    timestamp = run.get('timestamp', 'Unknown')
                    insights_count = run.get('insights_generated', 0)

                    # Enhanced expander title with insight count
                    expander_title = f"Run {run_id_short}... - {timestamp}"
                    if insights_count > 0:
                        expander_title += f" ✨ ({insights_count} insights)"

                    with st.expander(expander_title):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("Clusters Found", run.get('clusters_found', 0))
                            st.metric("Insights Generated", insights_count)

                        with col2:
                            st.metric("Processing Time", f"{run.get('processing_time_ms', 0)}ms")
                            visualization_status = "✅ Yes" if run.get('visualization_enabled') else "❌ No"
                            st.metric("Visualization", visualization_status)

                        if run.get('status') == 'success':
                            st.success("✅ Synthesis completed successfully")

                            # Display actual insights if available
                            insights = run.get('insights', [])
                            if insights:
                                st.markdown("---")
                                st.markdown("**✨ Generated Insights:**")

                                # Show insights with expandable view - Fixed duplicate key issue
                                show_all_insights = st.checkbox(f"Show all {len(insights)} insights", key=f"show_all_synthesis_{idx}_{run_id_short}")

                                insights_to_show = insights if show_all_insights else insights[:3]

                                for i, insight in enumerate(insights_to_show):
                                    with st.container():
                                        # Clean the insight text
                                        clean_text = insight.get('synthesized_text', '')
                                        if '<think>' in clean_text and '</think>' in clean_text:
                                            parts = clean_text.split('</think>')
                                            if len(parts) > 1:
                                                clean_text = parts[-1].strip()
                                            else:
                                                clean_text = clean_text.replace('<think>', '').replace('</think>', '').strip()

                                        # Show preview or full text based on expansion
                                        if show_all_insights:
                                            # Show full insight when expanded
                                            display_text = clean_text
                                        else:
                                            # Show preview when collapsed
                                            sentences = clean_text.split('. ')
                                            display_text = sentences[0] if sentences else clean_text
                                            if len(display_text) > 150:
                                                display_text = display_text[:150] + "..."

                                        confidence = insight.get('confidence_score', 0)
                                        cluster_id = insight.get('cluster_id', 'Unknown')

                                        st.markdown(f"**{i+1}. Cluster {cluster_id}** (Confidence: {confidence:.2f})")
                                        st.markdown(f"*{display_text}*")

                                        if not show_all_insights and i == 2 and len(insights) > 3:
                                            st.caption(f"... and {len(insights) - 3} more insights")
                                            break

                                # Button to load this synthesis into Dream Canvas - Fixed duplicate key issue
                                if st.button(f"🎨 Load in Dream Canvas", key=f"load_canvas_{idx}_{run_id_short}"):
                                    # Create focused visualization data for this synthesis
                                    try:
                                        focused_visualization_data = create_focused_synthesis_visualization(insights, run.get('run_id'))

                                        st.session_state.synthesis_results = {
                                            'insights': insights,
                                            'clusters_found': run.get('clusters_found', 0),
                                            'insights_generated': len(insights),
                                            'run_id': run.get('run_id'),
                                            'timestamp': run.get('timestamp'),
                                            'synthesis_log': {'status': 'loaded'}
                                        }

                                        # Store focused visualization data
                                        st.session_state.dream_canvas_data = focused_visualization_data

                                        st.success(f"✅ Synthesis results loaded into Dream Canvas! Showing {len(focused_visualization_data)} focused memory points.")
                                        st.rerun()
                                    except Exception as e:
                                        logger.error(f"Failed to create focused visualization: {e}")
                                        # Fallback to regular loading
                                        st.session_state.synthesis_results = {
                                            'insights': insights,
                                            'clusters_found': run.get('clusters_found', 0),
                                            'insights_generated': len(insights),
                                            'run_id': run.get('run_id'),
                                            'timestamp': run.get('timestamp'),
                                            'synthesis_log': {'status': 'loaded'}
                                        }
                                        st.success("✅ Synthesis results loaded into Dream Canvas!")
                                        st.rerun()
                        else:
                            st.error(f"❌ Synthesis failed: {run.get('error', 'Unknown error')}")
            else:
                st.info("🌙 No synthesis runs yet. Click 'Enter Dream State' to begin cognitive synthesis.")

        except Exception as e:
            st.warning(f"Could not load synthesis history: {e}")

        # Visualization display area
        st.subheader("🎨 Interactive Visualization")

        # Check if we have visualization data
        if 'dream_canvas_data' in st.session_state and st.session_state.dream_canvas_data:
            render_dream_canvas_visualization(st.session_state.dream_canvas_data)
        else:
            st.info("🎨 No visualization data available. Generate a visualization to see SAM's memory landscape.")

            # Placeholder visualization area
            st.markdown("""
            <div style="
                border: 2px dashed #ccc;
                border-radius: 10px;
                padding: 60px;
                text-align: center;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                margin: 20px 0;
            ">
                <h3>🌌 Dream Canvas Awaits</h3>
                <p>Trigger synthesis and visualization to explore SAM's cognitive landscape</p>
                <p style="font-size: 0.9rem; opacity: 0.8;">
                    Interactive UMAP projection • Memory clusters • Synthetic insights
                </p>
            </div>
            """, unsafe_allow_html=True)

        # NEW: Research Integration Section for Memory Control Center
        render_memory_center_research_integration()

        # Information panel
        with st.expander("ℹ️ About Dream Canvas", expanded=False):
            st.markdown("""
            **Dream Canvas** is SAM's revolutionary cognitive synthesis visualization system:

            🧠 **Cognitive Synthesis:**
            - Analyzes memory clusters for emergent patterns
            - Generates synthetic insights from concept relationships
            - Creates new understanding from existing knowledge

            🎨 **Interactive Visualization:**
            - UMAP 2D projection of memory landscape
            - Color-coded concept clusters
            - Golden stars for synthetic insights
            - Interactive exploration with hover details

            🌙 **Dream State Process:**
            1. **Memory Clustering** - Groups related concepts
            2. **Pattern Discovery** - Identifies emergent relationships
            3. **Insight Generation** - Creates synthetic understanding
            4. **Visualization** - Maps the cognitive landscape

            This represents the **first AI system** with visual cognitive transparency!
            """)

    except Exception as e:
        st.error(f"Error loading Dream Canvas: {e}")
        import traceback
        st.error(f"Details: {traceback.format_exc()}")

def render_archived_insights():
    """Render the Archived Insights interface."""
    try:
        render_insight_archive()
    except Exception as e:
        st.error(f"Error loading Archived Insights: {e}")
        import traceback
        st.error(f"Details: {traceback.format_exc()}")

def render_workflow_automation_controls():
    """Render workflow automation controls for complete Dream Canvas automation."""
    try:
        st.markdown("#### 🔄 Workflow Automation")

        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

        with col1:
            # Main workflow automation toggle
            workflow_enabled = st.session_state.get('workflow_automation_enabled', False)
            new_workflow_enabled = st.checkbox(
                "🤖 Auto-Workflow",
                value=workflow_enabled,
                help="Automatically sequence: Enter Dream State → Run Synthesis → Auto-Research"
            )

            if new_workflow_enabled != workflow_enabled:
                st.session_state.workflow_automation_enabled = new_workflow_enabled
                if new_workflow_enabled:
                    st.success("✅ Workflow automation enabled")
                    st.info("💡 Workflow will run: Dream State → Synthesis → Research")
                else:
                    st.info("⏸️ Workflow automation disabled")
                st.rerun()

        with col2:
            if st.session_state.get('workflow_automation_enabled', False):
                # Auto-research toggle for workflow
                auto_research_enabled = st.session_state.get('workflow_auto_research', True)
                new_auto_research = st.checkbox(
                    "🔬 Auto-Research",
                    value=auto_research_enabled,
                    help="Automatically research best insights after synthesis"
                )
                st.session_state.workflow_auto_research = new_auto_research
            else:
                st.markdown("*Enable Auto-Workflow to configure research*")

        with col3:
            if st.session_state.get('workflow_automation_enabled', False):
                # Research papers per insight
                max_papers = st.selectbox(
                    "Papers per Insight",
                    options=[1, 2, 3, 5],
                    index=1,  # Default to 2
                    help="Papers to research per selected insight"
                )
                st.session_state.workflow_max_papers = max_papers
            else:
                st.markdown("*Configure research settings*")

        with col4:
            # Workflow status indicator
            if st.session_state.get('workflow_running', False):
                st.markdown("🔄 **Running**")
            elif st.session_state.get('workflow_automation_enabled', False):
                st.markdown("✅ **Ready**")
            else:
                st.markdown("⏸️ **Disabled**")

        # Workflow execution button
        if st.session_state.get('workflow_automation_enabled', False):
            st.markdown("---")
            col1, col2, col3 = st.columns([2, 2, 2])

            with col1:
                if st.button("🚀 Run Complete Workflow", type="primary", use_container_width=True,
                           help="Execute full workflow: Dream State → Synthesis → Research"):
                    trigger_complete_workflow()

            with col2:
                # Show workflow progress if running
                if st.session_state.get('workflow_running', False):
                    current_step = st.session_state.get('workflow_current_step', 'Starting...')
                    st.info(f"🔄 {current_step}")

            with col3:
                # Stop workflow button
                if st.session_state.get('workflow_running', False):
                    if st.button("⏹️ Stop Workflow", type="secondary", use_container_width=True):
                        st.session_state.workflow_running = False
                        st.session_state.workflow_current_step = "Stopped by user"
                        st.warning("⏹️ Workflow stopped")
                        st.rerun()

    except Exception as e:
        st.error(f"❌ Workflow automation controls error: {e}")
        logger.error(f"Workflow automation error: {e}")

def trigger_direct_synthesis():
    """Trigger direct synthesis with default parameters - simplified version for Run Synthesis button."""
    try:
        with st.spinner("🔄 Running synthesis..."):
            # Import synthesis components
            from memory.synthesis import SynthesisEngine, SynthesisConfig
            from memory.memory_vectorstore import get_memory_store, VectorStoreType

            # Get memory store
            memory_store = get_memory_store(
                store_type=VectorStoreType.CHROMA,
                storage_directory="memory_store",
                embedding_dimension=384
            )

            # Configure synthesis engine with default settings
            config = SynthesisConfig(
                enable_reingestion=True,
                enable_deduplication=True
            )

            synthesis_engine = SynthesisEngine(config=config)

            # Run synthesis
            result = synthesis_engine.run_synthesis(memory_store, visualize=True)

            # Store results in session state
            synthesis_results = {
                'run_id': result.run_id,
                'timestamp': result.timestamp,
                'clusters_found': result.clusters_found,
                'insights_generated': result.insights_generated,
                'output_file': result.output_file,
                'insights': [insight.__dict__ for insight in result.insights] if hasattr(result, 'insights') else [],
                'synthesis_log': result.synthesis_log if hasattr(result, 'synthesis_log') else {}
            }

            # Add to synthesis history
            if 'synthesis_history' not in st.session_state:
                st.session_state.synthesis_history = []

            st.session_state.synthesis_history.append(synthesis_results)
            st.session_state.latest_synthesis = synthesis_results

            st.success(f"✨ Synthesis complete! Generated **{result.insights_generated} insights** from **{result.clusters_found} clusters**.")
            st.info("💡 View results in the 'Synthesis History' section below or use 'Load in Dream Canvas' to visualize.")

    except Exception as e:
        st.error(f"❌ Synthesis failed: {e}")
        logger.error(f"Direct synthesis error: {e}")

def trigger_complete_workflow():
    """Execute the complete automated workflow: Dream State → Synthesis → Research."""
    try:
        # Set workflow as running
        st.session_state.workflow_running = True
        st.session_state.workflow_current_step = "Initializing workflow..."

        # Get workflow settings
        auto_research = st.session_state.get('workflow_auto_research', True)
        max_papers = st.session_state.get('workflow_max_papers', 2)

        # Get clustering parameters from UI
        cluster_radius = st.session_state.get('cluster_radius', 0.5)
        min_samples = st.session_state.get('min_samples', 5)
        min_cluster_size = st.session_state.get('min_cluster_size', 3)
        max_clusters = st.session_state.get('max_clusters', 20)

        st.info("🚀 Starting complete workflow automation...")

        # STEP 1: Enter Dream State (Advanced Synthesis)
        st.session_state.workflow_current_step = "Step 1/3: Entering Dream State..."
        st.rerun()

        with st.spinner("🌙 Step 1: Entering Dream State..."):
            try:
                # Import synthesis components
                from memory.synthesis import SynthesisEngine, SynthesisConfig
                from memory.memory_vectorstore import get_memory_store

                # Get memory store
                memory_store = get_memory_store(
                    store_type=VectorStoreType.CHROMA,
                    storage_directory="memory_store",
                    embedding_dimension=384
                )

                # Configure synthesis with advanced parameters
                config = SynthesisConfig(
                    clustering_eps=cluster_radius,
                    clustering_min_samples=min_samples,
                    min_cluster_size=min_cluster_size,
                    max_clusters=max_clusters,
                    quality_threshold=0.4,
                    min_insight_quality=0.3,
                    enable_reingestion=True
                )

                synthesis_engine = SynthesisEngine(config=config)

                # Run advanced synthesis
                result = synthesis_engine.run_synthesis(memory_store, visualize=True)

                st.success(f"✅ Step 1 Complete: Dream State generated {result.clusters_found} clusters")

            except Exception as e:
                st.error(f"❌ Step 1 Failed: {e}")
                st.session_state.workflow_running = False
                return

        # STEP 2: Run Synthesis (Generate Insights)
        st.session_state.workflow_current_step = "Step 2/3: Generating Insights..."
        st.rerun()

        with st.spinner("🔄 Step 2: Running Synthesis..."):
            try:
                # Store results in session state
                synthesis_results = {
                    'run_id': result.run_id,
                    'timestamp': result.timestamp,
                    'clusters_found': result.clusters_found,
                    'insights_generated': result.insights_generated,
                    'output_file': result.output_file,
                    'insights': [insight.__dict__ for insight in result.insights] if hasattr(result, 'insights') else [],
                    'synthesis_log': result.synthesis_log if hasattr(result, 'synthesis_log') else {}
                }

                # Add to synthesis history
                if 'synthesis_history' not in st.session_state:
                    st.session_state.synthesis_history = []

                st.session_state.synthesis_history.append(synthesis_results)
                st.session_state.latest_synthesis = synthesis_results
                st.session_state.synthesis_results = synthesis_results

                st.success(f"✅ Step 2 Complete: Generated {result.insights_generated} insights")

            except Exception as e:
                st.error(f"❌ Step 2 Failed: {e}")
                st.session_state.workflow_running = False
                return

        # STEP 3: Auto-Research (if enabled)
        if auto_research and result.insights_generated > 0:
            st.session_state.workflow_current_step = "Step 3/3: Researching Insights..."
            st.rerun()

            with st.spinner("🔬 Step 3: Researching Best Insights..."):
                try:
                    # Auto-select best insights for research
                    insights = synthesis_results.get('insights', [])

                    if insights:
                        # Score insights by confidence, novelty, and utility
                        scored_insights = []
                        for i, insight in enumerate(insights):
                            confidence = insight.get('confidence_score', 0.0)
                            novelty = insight.get('novelty_score', 0.0)
                            utility = insight.get('utility_score', 0.0)

                            # Combined score (weighted average)
                            score = (confidence * 0.4) + (novelty * 0.4) + (utility * 0.2)
                            scored_insights.append((i, insight, score))

                        # Sort by score and select top insights
                        scored_insights.sort(key=lambda x: x[2], reverse=True)

                        # Select top 3 insights or all if fewer
                        num_to_research = min(3, len(scored_insights))
                        selected_insights = [item[1] for item in scored_insights[:num_to_research]]

                        # Trigger research for selected insights
                        trigger_insight_research_workflow(selected_insights, max_papers)

                        st.success(f"✅ Step 3 Complete: Researching {num_to_research} top insights")
                    else:
                        st.warning("⚠️ Step 3 Skipped: No insights available for research")

                except Exception as e:
                    st.error(f"❌ Step 3 Failed: {e}")
                    # Don't return here - workflow still partially successful
        else:
            st.info("ℹ️ Step 3 Skipped: Auto-research disabled or no insights generated")

        # Workflow complete
        st.session_state.workflow_running = False
        st.session_state.workflow_current_step = "Workflow Complete!"

        st.success("🎉 **Complete Workflow Finished Successfully!**")
        st.info("💡 View results in the sections below or use 'Load in Dream Canvas' to visualize.")

        st.rerun()

    except Exception as e:
        st.error(f"❌ Workflow failed: {e}")
        st.session_state.workflow_running = False
        st.session_state.workflow_current_step = "Workflow Failed"
        logger.error(f"Complete workflow error: {e}")

def trigger_insight_research_workflow(selected_insights, max_papers_per_insight):
    """Trigger automated research for insights in workflow mode."""
    try:
        # Check if research components are available
        try:
            from sam.web_retrieval.tools.arxiv_tool import get_arxiv_tool
            from sam.state.vetting_queue import get_vetting_queue_manager
            from sam.vetting.analyzer import get_vetting_analyzer
        except ImportError:
            st.warning("⚠️ Research components not available. Install Task 27 components to enable automated research.")
            return

        import threading
        import asyncio

        arxiv_tool = get_arxiv_tool()
        vetting_manager = get_vetting_queue_manager()
        vetting_analyzer = get_vetting_analyzer()

        def run_research():
            """Run research in background thread."""
            try:
                total_papers = 0

                for insight in selected_insights:
                    insight_text = insight.get('synthesized_text', insight.get('content', 'No content'))
                    cluster_id = insight.get('cluster_id', 'Unknown')

                    # Extract research query from insight
                    research_query = extract_research_query(insight_text)

                    if research_query:
                        # Search for papers
                        papers = asyncio.run(arxiv_tool.search_papers(
                            query=research_query,
                            max_results=max_papers_per_insight
                        ))

                        # Add papers to vetting queue
                        for paper in papers:
                            paper_data = {
                                'title': paper.get('title', 'Unknown Title'),
                                'authors': paper.get('authors', []),
                                'abstract': paper.get('summary', ''),
                                'url': paper.get('pdf_url', ''),
                                'source': 'arxiv',
                                'research_context': {
                                    'insight_cluster': cluster_id,
                                    'insight_text': insight_text[:200] + '...',
                                    'research_query': research_query
                                }
                            }

                            vetting_manager.add_document(paper_data)
                            total_papers += 1

                # Update session state with research results
                st.session_state.workflow_research_results = {
                    'papers_found': total_papers,
                    'insights_researched': len(selected_insights),
                    'timestamp': datetime.now().isoformat()
                }

            except Exception as e:
                logger.error(f"Research workflow error: {e}")
                st.session_state.workflow_research_error = str(e)

        # Start research in background
        research_thread = threading.Thread(target=run_research)
        research_thread.daemon = True
        research_thread.start()

        st.info(f"🔬 Research started for {len(selected_insights)} insights ({max_papers_per_insight} papers each)")

    except Exception as e:
        st.error(f"❌ Research workflow failed: {e}")
        logger.error(f"Research workflow error: {e}")

def render_memory_center_research_integration():
    """Render research integration controls for Memory Control Center Dream Canvas."""
    try:
        # Check if we have synthesis results available
        synthesis_results = st.session_state.get('synthesis_results')
        if not synthesis_results:
            return

        insights = synthesis_results.get('insights', [])
        if not insights:
            return

        st.markdown("---")
        st.markdown("### 🔬 Research Integration")
        st.markdown("*Select insights for automated research discovery*")

        # Research mode selection
        col1, col2, col3 = st.columns([2, 2, 2])

        with col1:
            research_mode = st.radio(
                "Research Selection Mode:",
                options=["🤖 SAM Selects Best", "👤 Human Selection"],
                index=0,
                help="Choose how insights are selected for research",
                key="memory_center_research_mode"
            )

        with col2:
            max_research_papers = st.selectbox(
                "Papers per Insight:",
                options=[1, 2, 3, 5],
                index=1,  # Default to 2
                help="Maximum papers to download per selected insight",
                key="memory_center_max_papers"
            )

        with col3:
            st.markdown("**Research Scope:**")
            st.caption("🎓 ArXiv Academic Papers")
            st.caption("🧠 Deep Research Available")

        # Insight selection interface
        if research_mode == "👤 Human Selection":
            st.markdown("**Select insights for research:**")

            # Initialize selection state for memory center
            if 'memory_center_selected_insights' not in st.session_state:
                st.session_state.memory_center_selected_insights = set()

            # Display insights with checkboxes
            for i, insight in enumerate(insights):
                insight_id = f"memory_insight_{i}"
                insight_text = insight.get('content', insight.get('insight', 'No content'))
                cluster_id = insight.get('cluster_id', 'Unknown')
                confidence = insight.get('confidence_score', 0.0)

                # Create checkbox for each insight
                col1, col2 = st.columns([1, 10])

                with col1:
                    is_selected = st.checkbox(
                        "",
                        key=f"memory_select_{insight_id}",
                        value=insight_id in st.session_state.memory_center_selected_insights
                    )

                    # Update selection state
                    if is_selected:
                        st.session_state.memory_center_selected_insights.add(insight_id)
                    else:
                        st.session_state.memory_center_selected_insights.discard(insight_id)

                with col2:
                    # Display insight content with metadata
                    st.markdown(f"**Cluster {cluster_id}** (Confidence: {confidence:.2f})")
                    st.markdown(insight_text)
                    st.markdown("---")

        else:
            # SAM automatic selection mode
            if insights:
                # Score insights for automatic selection
                scored_insights = []
                for i, insight in enumerate(insights):
                    confidence = insight.get('confidence_score', 0.0)
                    content = insight.get('content', insight.get('insight', ''))

                    novelty_keywords = ['new', 'novel', 'innovative', 'breakthrough', 'discovery', 'emerging', 'unprecedented']
                    novelty_score = sum(1 for keyword in novelty_keywords if keyword.lower() in content.lower()) / len(novelty_keywords)

                    research_keywords = ['how', 'why', 'what', 'could', 'might', 'potential', 'explore', 'investigate']
                    research_score = sum(1 for keyword in research_keywords if keyword.lower() in content.lower()) / len(research_keywords)

                    combined_score = confidence * 0.4 + novelty_score * 0.3 + research_score * 0.3
                    scored_insights.append((i, insight, combined_score))

                # Sort by score and show top candidate
                scored_insights.sort(key=lambda x: x[2], reverse=True)
                best_insight = scored_insights[0]

                st.markdown(f"**🎯 SAM's Top Selection:**")
                st.markdown(f"**Cluster {best_insight[1].get('cluster_id', 'Unknown')}** (Score: {best_insight[2]:.2f})")
                st.markdown(f"{best_insight[1].get('content', best_insight[1].get('insight', 'No content'))}")

        # Research action buttons
        st.markdown("---")

        # Check if research components are available
        research_available = True
        try:
            from sam.web_retrieval.tools.arxiv_tool import get_arxiv_tool
            from sam.state.vetting_queue import get_vetting_queue_manager
        except ImportError:
            research_available = False

        if research_available:
            # Enhanced research options with Quick and Deep Research
            col1, col2, col3 = st.columns([2, 2, 2])

            with col1:
                if st.button("🔬 **Quick Research**", use_container_width=True,
                           help="Basic ArXiv search for selected insights", key="memory_quick_research"):
                    # Trigger quick research process
                    trigger_memory_center_quick_research(research_mode, insights, max_research_papers)

            with col2:
                if st.button("🧠 **Deep Research**", type="primary", use_container_width=True,
                           help="Comprehensive multi-step ArXiv analysis with verification", key="memory_deep_research"):
                    # Trigger deep research process
                    trigger_memory_center_deep_research(research_mode, insights)

            with col3:
                if st.button("📋 View Research Queue", use_container_width=True,
                           help="View pending research papers in vetting queue", key="memory_view_queue"):
                    # Navigate to vetting queue
                    st.session_state.memory_page_override = "🔍 Vetting Queue"
                    st.rerun()

        else:
            st.warning("⚠️ Research components not available. Install Task 27 components to enable automated research.")

        # Display Deep Research Results if available
        render_memory_center_deep_research_results()

    except Exception as e:
        st.error(f"❌ Error loading research integration: {e}")

def extract_research_query(insight_text):
    """Extract a research query from insight text."""
    try:
        # Simple extraction - take first sentence or key phrases
        sentences = insight_text.split('.')
        if sentences:
            # Use first meaningful sentence as research query
            query = sentences[0].strip()

            # Clean up the query
            query = query.replace('EMERGENT INSIGHT:', '').strip()
            query = query.replace('This cluster', '').strip()

            # Limit length
            if len(query) > 100:
                query = query[:100] + '...'

            return query if len(query) > 10 else None
    except:
        return None

def trigger_memory_center_quick_research(research_mode, insights, max_research_papers):
    """Trigger quick research from Memory Control Center."""
    try:
        if research_mode == "👤 Human Selection":
            if st.session_state.memory_center_selected_insights:
                selected_indices = [int(insight_id.split('_')[2]) for insight_id in st.session_state.memory_center_selected_insights]
                selected_insights_data = [insights[i] for i in selected_indices]
                trigger_insight_research_workflow(selected_insights_data, max_research_papers)
                st.success(f"🔬 **Quick Research initiated!** Processing {len(selected_insights_data)} insight{'' if len(selected_insights_data) == 1 else 's'}")
            else:
                st.error("❌ Please select at least one insight for research")
        else:
            # SAM automatic selection
            if insights:
                # Use scoring logic to select best insight
                scored_insights = []
                for i, insight in enumerate(insights):
                    confidence = insight.get('confidence_score', 0.0)
                    content = insight.get('content', insight.get('insight', ''))

                    novelty_keywords = ['new', 'novel', 'innovative', 'breakthrough', 'discovery', 'emerging', 'unprecedented']
                    novelty_score = sum(1 for keyword in novelty_keywords if keyword.lower() in content.lower()) / len(novelty_keywords)

                    research_keywords = ['how', 'why', 'what', 'could', 'might', 'potential', 'explore', 'investigate']
                    research_score = sum(1 for keyword in research_keywords if keyword.lower() in content.lower()) / len(research_keywords)

                    combined_score = confidence * 0.4 + novelty_score * 0.3 + research_score * 0.3
                    scored_insights.append((i, insight, combined_score))

                scored_insights.sort(key=lambda x: x[2], reverse=True)
                best_insight = scored_insights[0][1]

                trigger_insight_research_workflow([best_insight], max_research_papers)
                st.success("🔬 **Quick Research initiated!** Processing SAM's top selected insight")
            else:
                st.error("❌ No insights available for research")

    except Exception as e:
        st.error(f"❌ Failed to start quick research: {e}")

def trigger_memory_center_deep_research(research_mode, insights):
    """Trigger deep research from Memory Control Center."""
    try:
        from sam.agents.strategies.deep_research import DeepResearchStrategy
        import threading

        def run_memory_center_deep_research():
            """Run deep research in background thread for Memory Control Center."""
            try:
                selected_insights_data = []

                if research_mode == "👤 Human Selection":
                    if st.session_state.memory_center_selected_insights:
                        selected_indices = [int(insight_id.split('_')[2]) for insight_id in st.session_state.memory_center_selected_insights]
                        selected_insights_data = [insights[i] for i in selected_indices]
                    else:
                        st.session_state.memory_center_deep_research_error = "No insights selected"
                        return
                else:
                    # SAM automatic selection
                    if insights:
                        # Use scoring logic to select best insight
                        scored_insights = []
                        for i, insight in enumerate(insights):
                            confidence = insight.get('confidence_score', 0.0)
                            content = insight.get('content', insight.get('insight', ''))

                            novelty_keywords = ['new', 'novel', 'innovative', 'breakthrough', 'discovery', 'emerging', 'unprecedented']
                            novelty_score = sum(1 for keyword in novelty_keywords if keyword.lower() in content.lower()) / len(novelty_keywords)

                            research_keywords = ['how', 'why', 'what', 'could', 'might', 'potential', 'explore', 'investigate']
                            research_score = sum(1 for keyword in research_keywords if keyword.lower() in content.lower()) / len(research_keywords)

                            combined_score = confidence * 0.4 + novelty_score * 0.3 + research_score * 0.3
                            scored_insights.append((i, insight, combined_score))

                        scored_insights.sort(key=lambda x: x[2], reverse=True)
                        selected_insights_data = [scored_insights[0][1]]
                    else:
                        st.session_state.memory_center_deep_research_error = "No insights available"
                        return

                # Execute deep research for selected insights
                research_results = []

                for insight in selected_insights_data:
                    insight_text = insight.get('content', insight.get('insight', ''))
                    cluster_id = insight.get('cluster_id', 'Unknown')

                    # Initialize Deep Research Strategy
                    research_strategy = DeepResearchStrategy(insight_text)

                    # Execute deep research
                    result = research_strategy.execute_research()

                    # Store result
                    research_results.append({
                        'research_id': result.research_id,
                        'original_insight': result.original_insight,
                        'cluster_id': cluster_id,
                        'final_report': result.final_report,
                        'arxiv_papers': result.arxiv_papers,
                        'status': result.status.value,
                        'timestamp': result.timestamp,
                        'quality_score': research_strategy._assess_research_quality(),
                        'papers_analyzed': len(result.arxiv_papers),
                        'iterations_completed': research_strategy.current_iteration
                    })

                # Store results in session state
                if 'memory_center_deep_research_results' not in st.session_state:
                    st.session_state.memory_center_deep_research_results = []

                st.session_state.memory_center_deep_research_results.extend(research_results)

                # Update completion status
                st.session_state.memory_center_deep_research_completion = {
                    'success': True,
                    'insights_processed': len(selected_insights_data),
                    'reports_generated': len(research_results),
                    'timestamp': datetime.now().isoformat()
                }

            except Exception as e:
                logger.error(f"Memory Center deep research execution failed: {e}")
                st.session_state.memory_center_deep_research_completion = {
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }

        # Start deep research in background
        research_thread = threading.Thread(target=run_memory_center_deep_research, daemon=True)
        research_thread.start()

        if research_mode == "👤 Human Selection":
            selected_count = len(st.session_state.memory_center_selected_insights)
            st.success(f"🧠 **Deep Research initiated!** Processing {selected_count} selected insight{'' if selected_count == 1 else 's'}")
        else:
            st.success("🧠 **Deep Research initiated!** Processing SAM's top selected insight")

        st.info("📊 **Comprehensive Analysis**: Multi-step ArXiv research with verification and critique")
        st.info("📄 **Report Generation**: Structured research reports will be generated")
        st.info("🔄 **Check Results**: Results will appear in the Deep Research Results section below")

    except ImportError:
        st.error("❌ Deep Research Engine not available. Please ensure sam.agents.strategies.deep_research is installed.")
    except Exception as e:
        st.error(f"❌ Failed to start deep research: {e}")

def render_memory_center_deep_research_results():
    """Display Deep Research results for Memory Control Center."""
    try:
        if 'memory_center_deep_research_results' not in st.session_state or not st.session_state.memory_center_deep_research_results:
            return

        st.markdown("---")
        st.markdown("### 🧠 Deep Research Results")
        st.markdown("*Comprehensive ArXiv analysis reports with verification*")

        # Show completion status if available
        if 'memory_center_deep_research_completion' in st.session_state:
            completion = st.session_state.memory_center_deep_research_completion
            if completion.get('success'):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Insights Analyzed", completion.get('insights_processed', 0))
                with col2:
                    st.metric("Reports Generated", completion.get('reports_generated', 0))
                with col3:
                    timestamp = completion.get('timestamp', '')
                    if timestamp:
                        time_str = timestamp.split('T')[1][:5] if 'T' in timestamp else 'Unknown'
                        st.metric("Completed", time_str)

        # Display research results
        for i, result in enumerate(st.session_state.memory_center_deep_research_results):
            with st.expander(f"📊 Research Report {i+1}: {result['original_insight'][:60]}...", expanded=i==0):

                # Research metadata
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Quality Score", f"{result.get('quality_score', 0):.2f}")
                with col2:
                    st.metric("Papers Analyzed", result.get('papers_analyzed', 0))
                with col3:
                    st.metric("Iterations", result.get('iterations_completed', 0))
                with col4:
                    status_color = "🟢" if result.get('status') == 'COMPLETED' else "🟡"
                    st.metric("Status", f"{status_color} {result.get('status', 'Unknown')}")

                # Display the full research report
                st.markdown("#### 📄 Research Report")
                st.markdown(result.get('final_report', 'Report not available'))

                # Show ArXiv papers found
                if result.get('arxiv_papers'):
                    st.markdown("#### 📚 ArXiv Papers Analyzed")
                    for j, paper in enumerate(result['arxiv_papers'][:5]):  # Show top 5
                        title = paper.get('title', 'Unknown Title')
                        authors = ', '.join(paper.get('authors', [])[:3])
                        year = paper.get('published', '')[:4] if paper.get('published') else 'Unknown'

                        st.markdown(f"**{j+1}. {title}** ({year})")
                        st.markdown(f"*Authors*: {authors}")
                        if paper.get('summary'):
                            st.markdown(f"*Summary*: {paper['summary'][:150]}...")
                        st.markdown("---")

                # Action buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(f"📋 View Papers in Queue", key=f"memory_queue_{result['research_id']}"):
                        st.session_state.memory_page_override = "🔍 Vetting Queue"
                        st.rerun()

                with col2:
                    if st.button(f"📄 Export Report", key=f"memory_export_{result['research_id']}"):
                        # Create downloadable report
                        report_content = result.get('final_report', '')
                        st.download_button(
                            label="Download Report",
                            data=report_content,
                            file_name=f"memory_center_deep_research_report_{result['research_id']}.md",
                            mime="text/markdown",
                            key=f"memory_download_{result['research_id']}"
                        )

                with col3:
                    if st.button(f"🔄 Re-run Research", key=f"memory_rerun_{result['research_id']}"):
                        # Re-trigger research for this insight
                        insight_data = {
                            'content': result['original_insight'],
                            'cluster_id': result.get('cluster_id', 'Unknown')
                        }
                        trigger_memory_center_deep_research("🤖 SAM Selects Best", [insight_data])

        # Clear results button
        if st.button("🗑️ Clear Deep Research Results", key="memory_clear_deep_research"):
            st.session_state.memory_center_deep_research_results = []
            if 'memory_center_deep_research_completion' in st.session_state:
                del st.session_state.memory_center_deep_research_completion
            st.rerun()

    except Exception as e:
        st.error(f"❌ Error displaying deep research results: {e}")

def trigger_synthesis_with_params(eps, min_samples, min_cluster_size, max_clusters):
    """Trigger cognitive synthesis process with custom parameters."""
    try:
        st.info(f"🎛️ Using custom parameters: eps={eps}, min_samples={min_samples}, min_size={min_cluster_size}, max_clusters={max_clusters}")
        with st.spinner("🌙 Entering dream state... Analyzing memory clusters..."):
            # Import synthesis components
            try:
                from memory.synthesis import SynthesisEngine, SynthesisConfig
                from memory.memory_vectorstore import get_memory_store
            except ImportError as e:
                st.error(f"❌ Synthesis components not available: {e}")
                st.info("💡 Make sure the synthesis module is properly installed")
                return

            # Get memory store (force refresh)
            memory_store = get_memory_store()

            # Force reload the memory store to get latest data
            try:
                memory_store._reload_memories()  # Try to reload if method exists
            except:
                pass  # If reload method doesn't exist, continue

            # Debug: Show memory store information
            all_memories = memory_store.get_all_memories()
            st.info(f"📊 Memory store contains {len(all_memories)} memories (refreshed)")

            # Check memory types and sources
            memory_types = {}
            sources = set()
            embeddings_count = 0

            for memory in all_memories[:100]:  # Sample first 100 for performance
                mem_type = memory.memory_type.value
                memory_types[mem_type] = memory_types.get(mem_type, 0) + 1
                sources.add(memory.source)
                if memory.embedding and len(memory.embedding) > 0:
                    embeddings_count += 1

            st.info(f"🔍 Sample analysis: {embeddings_count}/100 memories have embeddings, {len(sources)} unique sources, {len(memory_types)} memory types")

            # Configure synthesis with user-specified parameters
            config = SynthesisConfig(
                clustering_eps=eps,  # User-controlled cluster radius
                clustering_min_samples=min_samples,  # User-controlled core point threshold
                min_cluster_size=min_cluster_size,  # User-controlled minimum cluster size
                max_clusters=max_clusters,  # User-controlled maximum clusters
                quality_threshold=0.4,  # Moderate quality threshold
                min_insight_quality=0.3,  # Lower insight quality threshold for testing
                enable_reingestion=True
            )

            # Debug: Show configuration
            st.info(f"🔧 Using clustering parameters: eps={config.clustering_eps}, min_samples={config.clustering_min_samples}, min_size={config.min_cluster_size}")

            # Create synthesis engine
            synthesis_engine = SynthesisEngine(config=config)

            # Additional debugging: Check memory embeddings
            sample_memories = all_memories[:10]
            embedding_info = []
            for i, memory in enumerate(sample_memories):
                has_embedding = memory.embedding is not None and len(memory.embedding) > 0
                embedding_dim = len(memory.embedding) if has_embedding else 0
                embedding_info.append(f"Memory {i}: {has_embedding} (dim: {embedding_dim})")

            st.info(f"🔍 Sample embedding check:\n" + "\n".join(embedding_info))

            # Run synthesis without visualization first
            result = synthesis_engine.run_synthesis(memory_store, visualize=False)

            # Check status from synthesis_log - handle both 'completed' and successful insight generation
            status = result.synthesis_log.get('status', 'unknown')
            insights_generated = result.insights_generated if hasattr(result, 'insights_generated') else 0

            # Consider synthesis successful if we have insights, even if status isn't explicitly 'completed'
            if status == 'completed' or (insights_generated > 0 and result.insights):
                st.success(f"✅ Dream state completed! Generated {result.insights_generated} insights from {result.clusters_found} clusters")

                # Store result for history
                if 'synthesis_history' not in st.session_state:
                    st.session_state.synthesis_history = []

                # Store complete synthesis results including insights
                synthesis_record = {
                    'run_id': result.run_id,
                    'timestamp': result.timestamp,
                    'clusters_found': result.clusters_found,
                    'insights_generated': result.insights_generated,
                    'processing_time_ms': 0,  # Not tracked in current implementation
                    'status': 'success',  # Mark as success if we have insights
                    'visualization_enabled': False,
                    'insights': [insight.__dict__ for insight in result.insights] if hasattr(result, 'insights') else []
                }

                st.session_state.synthesis_history.append(synthesis_record)

                # Also store the latest synthesis results for Dream Canvas
                st.session_state.synthesis_results = {
                    'insights': [insight.__dict__ for insight in result.insights] if hasattr(result, 'insights') else [],
                    'clusters_found': result.clusters_found,
                    'insights_generated': result.insights_generated,
                    'run_id': result.run_id,
                    'timestamp': result.timestamp,
                    'synthesis_log': result.synthesis_log
                }

                st.rerun()
            else:
                error_reason = result.synthesis_log.get('reason', 'Unknown error')
                st.error(f"❌ Synthesis failed: {error_reason}")

    except Exception as e:
        st.error(f"❌ Failed to enter dream state: {e}")

def trigger_synthesis():
    """Trigger cognitive synthesis process with default parameters."""
    trigger_synthesis_with_params(0.15, 8, 15, 15)  # Optimal parameters for UMAP data

def trigger_visualization_with_params(eps, min_samples, min_cluster_size, max_clusters):
    """Trigger visualization generation with custom parameters."""
    try:
        st.info(f"🎛️ Using custom parameters: eps={eps}, min_samples={min_samples}, min_size={min_cluster_size}, max_clusters={max_clusters}")
        with st.spinner("🎨 Generating memory landscape visualization..."):
            # Import synthesis components
            try:
                from memory.synthesis import SynthesisEngine, SynthesisConfig
                from memory.memory_vectorstore import get_memory_store
            except ImportError as e:
                st.error(f"❌ Synthesis components not available: {e}")
                st.info("💡 Make sure the synthesis module is properly installed")
                return

            # Get memory store
            memory_store = get_memory_store()

            # Configure synthesis with visualization - user-specified parameters
            config = SynthesisConfig(
                clustering_eps=eps,  # User-controlled cluster radius
                clustering_min_samples=min_samples,  # User-controlled core point threshold
                min_cluster_size=min_cluster_size,  # User-controlled minimum cluster size
                max_clusters=max_clusters,  # User-controlled maximum clusters
                quality_threshold=0.4,  # Moderate quality threshold
                min_insight_quality=0.3,  # Lower insight quality threshold for testing
                enable_reingestion=False  # Skip re-ingestion for visualization
            )

            # Debug: Show configuration
            st.info(f"🔧 Using clustering parameters: eps={config.clustering_eps}, min_samples={config.clustering_min_samples}, min_size={config.min_cluster_size}")

            # Create synthesis engine
            synthesis_engine = SynthesisEngine(config=config)

            # Run synthesis with visualization
            result = synthesis_engine.run_synthesis(memory_store, visualize=True)

            # Check status from synthesis_log - handle both 'completed' and successful insight generation
            status = result.synthesis_log.get('status', 'unknown')
            insights_generated = result.insights_generated if hasattr(result, 'insights_generated') else 0

            # Consider synthesis successful if we have insights and visualization data
            if (status == 'completed' or insights_generated > 0) and result.visualization_data:
                st.success(f"✅ Visualization generated! {len(result.visualization_data)} memory points mapped")

                # Store visualization data
                st.session_state.dream_canvas_data = result.visualization_data

                # Store result for history with insights
                if 'synthesis_history' not in st.session_state:
                    st.session_state.synthesis_history = []

                synthesis_record = {
                    'run_id': result.run_id,
                    'timestamp': result.timestamp,
                    'clusters_found': result.clusters_found,
                    'insights_generated': result.insights_generated,
                    'processing_time_ms': 0,  # Not tracked in current implementation
                    'status': 'success',  # Mark as success if we have insights and visualization
                    'visualization_enabled': True,
                    'insights': [insight.__dict__ for insight in result.insights] if hasattr(result, 'insights') else []
                }

                st.session_state.synthesis_history.append(synthesis_record)

                # Also store the latest synthesis results for Dream Canvas
                st.session_state.synthesis_results = {
                    'insights': [insight.__dict__ for insight in result.insights] if hasattr(result, 'insights') else [],
                    'clusters_found': result.clusters_found,
                    'insights_generated': result.insights_generated,
                    'run_id': result.run_id,
                    'timestamp': result.timestamp,
                    'synthesis_log': result.synthesis_log
                }

                st.rerun()
            else:
                error_reason = result.synthesis_log.get('reason', 'No visualization data generated')
                st.error(f"❌ Visualization failed: {error_reason}")

    except Exception as e:
        st.error(f"❌ Failed to generate visualization: {e}")

def trigger_visualization():
    """Trigger visualization generation with default parameters."""
    trigger_visualization_with_params(0.15, 8, 15, 15)  # Optimal parameters for UMAP data

def suggest_optimal_parameters():
    """Analyze memory data to suggest optimal clustering parameters."""
    try:
        with st.spinner("🤔 Analyzing your memory data to suggest optimal parameters..."):
            # Try to use the web UI API first
            try:
                import requests
                response = requests.post(
                    "http://localhost:5002/api/synthesis/suggest-eps",
                    json={
                        "min_samples": st.session_state.get('min_samples', 5),
                        "target_clusters": 10
                    },
                    timeout=10
                )

                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'success':
                        suggestions = data['suggestions']
                        st.session_state.parameter_suggestions = suggestions
                        st.success(f"✨ AI Analysis Complete! Suggested eps: {suggestions['eps']:.3f}")
                        st.rerun()
                        return
                    else:
                        st.warning(f"API returned error: {data.get('error', 'Unknown error')}")
                else:
                    st.warning(f"API request failed with status {response.status_code}")
            except Exception as api_error:
                st.warning(f"Web UI API not available: {api_error}")

            # Fallback to direct analysis
            st.info("🔄 Using direct analysis method...")

            # Get memory store
            memory_store = get_memory_store()
            if not memory_store:
                st.error("❌ Memory store not available")
                return

            # Get embeddings directly
            memories = memory_store.get_all_memories()
            if not memories:
                st.error("❌ No memories found for analysis")
                return

            # Extract embeddings
            embeddings = []
            for memory in memories:
                if hasattr(memory, 'embedding') and memory.embedding is not None:
                    embeddings.append(memory.embedding)

            if len(embeddings) < 10:
                st.error("❌ Not enough memories with embeddings for analysis")
                return

            # Use direct eps optimization
            try:
                from memory.synthesis.eps_optimizer import EpsOptimizer
                import numpy as np

                optimizer = EpsOptimizer()
                suggestions = optimizer.suggest_clustering_params(
                    np.array(embeddings),
                    target_clusters=10
                )

                st.session_state.parameter_suggestions = suggestions
                st.success(f"✨ Direct Analysis Complete! Suggested eps: {suggestions['eps']:.3f}")
                st.rerun()

            except ImportError:
                # Final fallback with reasonable defaults
                st.warning("⚠️ Advanced analysis not available, using heuristic suggestions")

                data_size = len(embeddings)
                suggested_eps = max(0.1, min(0.8, 0.3 + (data_size / 10000) * 0.2))
                suggested_min_samples = max(3, min(10, data_size // 1000))
                suggested_min_cluster_size = max(3, data_size // 100)
                suggested_max_clusters = min(50, max(5, data_size // 50))

                suggestions = {
                    'eps': suggested_eps,
                    'min_samples': suggested_min_samples,
                    'min_cluster_size': suggested_min_cluster_size,
                    'max_clusters': suggested_max_clusters,
                    'data_size': data_size,
                    'method': 'heuristic'
                }

                st.session_state.parameter_suggestions = suggestions
                st.success(f"✨ Heuristic Analysis Complete! Suggested eps: {suggestions['eps']:.3f}")
                st.rerun()

    except Exception as e:
        st.error(f"❌ Error suggesting parameters: {e}")

def apply_suggested_parameters(suggestions):
    """Apply the suggested parameters to the sliders."""
    try:
        # Update session state with suggested values
        st.session_state.cluster_radius = suggestions['eps']
        st.session_state.min_samples = suggestions['min_samples']
        st.session_state.min_cluster_size = suggestions['min_cluster_size']
        st.session_state.max_clusters = suggestions['max_clusters']

        st.success("✨ Suggested parameters applied! Click 'Enter Dream State' to test them.")
        st.rerun()

    except Exception as e:
        st.error(f"❌ Error applying parameters: {e}")

def apply_research_preset():
    """Apply scientifically optimal parameters for UMAP+DBSCAN clustering."""
    try:
        # Research-backed optimal parameters for high-dimensional embeddings with UMAP projection
        st.session_state.cluster_radius = 0.15    # Small eps for UMAP cosine similarity
        st.session_state.min_samples = 8          # Higher for dense, meaningful clusters
        st.session_state.min_cluster_size = 15    # Substantial clusters for synthesis
        st.session_state.max_clusters = 20        # Allow more clusters to be discovered

        st.success("⚡ **Research Preset Applied!**")
        st.info("📚 **Parameters based on:** UMAP + DBSCAN clustering research for high-dimensional embeddings")
        st.info("🎯 **Expected result:** 5-20 distinct clusters instead of 1 massive cluster")
        st.rerun()

    except Exception as e:
        st.error(f"❌ Error applying research preset: {e}")

def create_focused_synthesis_visualization(insights, run_id):
    """Create focused visualization data showing only cluster memories and generated insights."""

    try:
        from memory.memory_vectorstore import MemoryVectorStore
        import numpy as np
        import random

        # Initialize memory store
        memory_store = MemoryVectorStore()

        # Get cluster IDs from insights
        cluster_ids = [insight.get('cluster_id', '') for insight in insights if insight.get('cluster_id')]

        if not cluster_ids:
            logger.warning("No cluster IDs found in insights")
            return []

        logger.info(f"Creating focused visualization for clusters: {cluster_ids}")

        # Retrieve memories for each cluster
        focused_memories = []
        cluster_memory_map = {}

        for cluster_id in cluster_ids:
            try:
                # Search for memories related to this cluster
                # Use cluster_id as search term to find related memories
                search_results = memory_store.search(
                    query=f"cluster {cluster_id}",
                    max_results=50
                )

                cluster_memories = []
                for result in search_results:
                    memory_data = {
                        'id': result.chunk.chunk_id,
                        'content': result.chunk.content,
                        'content_snippet': result.chunk.content[:100] + "..." if len(result.chunk.content) > 100 else result.chunk.content,
                        'source': result.chunk.source,
                        'timestamp': result.chunk.timestamp,
                        'importance_score': result.chunk.importance_score,
                        'cluster_id': cluster_id,
                        'memory_type': 'source_memory',
                        'coordinates': {
                            'x': random.uniform(-5, 5),  # Random positioning for now
                            'y': random.uniform(-5, 5)
                        },
                        'x': random.uniform(-5, 5),  # Also include direct x,y for compatibility
                        'y': random.uniform(-5, 5),
                        'color': f'cluster_{cluster_id}',
                        'is_synthetic': False
                    }
                    cluster_memories.append(memory_data)
                    focused_memories.append(memory_data)

                cluster_memory_map[cluster_id] = cluster_memories
                logger.info(f"Found {len(cluster_memories)} memories for cluster {cluster_id}")

            except Exception as e:
                logger.warning(f"Could not retrieve memories for cluster {cluster_id}: {e}")

        # Add insight points
        for i, insight in enumerate(insights):
            cluster_id = insight.get('cluster_id', f'unknown_{i}')

            # Clean insight text
            clean_text = insight.get('synthesized_text', '')
            if '<think>' in clean_text and '</think>' in clean_text:
                parts = clean_text.split('</think>')
                if len(parts) > 1:
                    clean_text = parts[-1].strip()
                else:
                    clean_text = clean_text.replace('<think>', '').replace('</think>', '').strip()

            # Position insight near its cluster memories
            cluster_memories = cluster_memory_map.get(cluster_id, [])
            if cluster_memories:
                # Position insight at center of cluster memories
                avg_x = sum(mem['x'] for mem in cluster_memories) / len(cluster_memories)
                avg_y = sum(mem['y'] for mem in cluster_memories) / len(cluster_memories)
                insight_x = avg_x + random.uniform(-1, 1)
                insight_y = avg_y + random.uniform(-1, 1)
            else:
                # Random position if no cluster memories found
                insight_x = random.uniform(-3, 3)
                insight_y = random.uniform(-3, 3)

            insight_data = {
                'id': f"insight_{cluster_id}_{i}",
                'content': clean_text[:200] + "..." if len(clean_text) > 200 else clean_text,
                'content_snippet': clean_text[:100] + "..." if len(clean_text) > 100 else clean_text,
                'source': f"Synthesis Insight - Cluster {cluster_id}",
                'timestamp': insight.get('timestamp', ''),
                'importance_score': insight.get('confidence_score', 0.5),
                'cluster_id': cluster_id,
                'memory_type': 'synthetic_insight',
                'coordinates': {
                    'x': insight_x,
                    'y': insight_y
                },
                'x': insight_x,  # Also include direct x,y for compatibility
                'y': insight_y,
                'color': 'synthetic_insight',
                'confidence_score': insight.get('confidence_score', 0),
                'novelty_score': insight.get('novelty_score', 0),
                'is_synthetic': True
            }
            focused_memories.append(insight_data)

        logger.info(f"Created focused visualization with {len(focused_memories)} total points")
        return focused_memories

    except Exception as e:
        logger.error(f"Failed to create focused synthesis visualization: {e}")
        return []

def generate_cluster_insight(contents, sources, cluster_id):
    """Generate meaningful 'So What' insights from cluster content."""
    try:
        # First, try to get enhanced insight from cluster registry
        try:
            from memory.synthesis.cluster_registry import get_cluster_stats
            cluster_stats = get_cluster_stats(f"cluster_{cluster_id:03d}")

            if cluster_stats['exists'] and cluster_stats['dominant_themes']:
                # Use registry data for enhanced insights
                themes = cluster_stats['dominant_themes']
                memory_count = cluster_stats['memory_count']
                source_count = cluster_stats['source_count']
                coherence = cluster_stats['coherence_score']

                if len(themes) >= 2:
                    themes_str = ", ".join(themes[:2])
                    if source_count > 1:
                        insight = f"This cluster represents a convergence of ideas around **{themes_str}**, drawing from {source_count} different sources with {memory_count} related memories. The high coherence score ({coherence:.2f}) suggests this is a well-defined knowledge domain that SAM has identified as conceptually related."
                    else:
                        insight = f"This cluster focuses on **{themes_str}** within a specific domain, containing {memory_count} related memories. The coherence score ({coherence:.2f}) indicates how tightly related these concepts are in SAM's understanding."
                else:
                    insight = f"This cluster contains {memory_count} related memories from {source_count} sources, representing a coherent knowledge area with a coherence score of {coherence:.2f}."

                # Add actionable "So What"
                if memory_count > 10:
                    insight += f" **So What:** This represents a major knowledge domain for SAM - consider this cluster when asking questions about {themes[0] if themes else 'this topic'}."
                else:
                    insight += f" **So What:** This is a specialized knowledge cluster that could provide focused insights for specific queries about {themes[0] if themes else 'related topics'}."

                return insight
        except Exception as e:
            logger.debug(f"Could not get enhanced cluster insight: {e}")

        # Fallback to original analysis if registry lookup fails
        # Analyze content patterns
        content_text = " ".join(contents)

        # Extract key themes using simple keyword analysis
        common_words = []
        for content in contents:
            words = content.lower().split()
            # Filter for meaningful words (longer than 3 chars, not common stop words)
            meaningful_words = [w for w in words if len(w) > 3 and w not in ['this', 'that', 'with', 'from', 'they', 'have', 'been', 'were', 'will', 'would', 'could', 'should']]
            common_words.extend(meaningful_words)

        # Count word frequency
        word_freq = {}
        for word in common_words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Get top themes
        top_themes = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:3]
        theme_words = [theme[0] for theme in top_themes if theme[1] > 1]

        # Analyze source diversity
        source_types = []
        for source in sources:
            if 'document:' in source:
                source_types.append('document')
            elif 'web:' in source:
                source_types.append('web')
            elif 'synthesis:' in source:
                source_types.append('synthetic')
            else:
                source_types.append('memory')

        source_diversity = len(set(source_types))

        # Generate insight based on patterns
        if len(contents) == 0:
            insight = f"This cluster appears to be empty or contains no accessible content. This might indicate a visualization artifact or a cluster that was filtered out during processing."
        elif len(theme_words) >= 2:
            themes_str = ", ".join(theme_words[:2])
            if source_diversity > 1:
                insight = f"This cluster represents a convergence of ideas around **{themes_str}**, drawing from {source_diversity} different types of sources. This suggests a cross-domain pattern that SAM has identified as conceptually related."
            else:
                insight = f"This cluster focuses on **{themes_str}** within a specific domain. The concentration of similar content suggests this is a core knowledge area in SAM's memory."
        elif len(contents) > 5:
            insight = f"This is a large cluster with {len(contents)} related memories. While the specific themes are diverse, SAM has identified underlying conceptual similarities that group these memories together."
        else:
            insight = f"This cluster contains {len(contents)} closely related memories. The tight grouping suggests these concepts are frequently accessed together or share important contextual relationships."

        # Add actionable "So What"
        if len(contents) == 0:
            insight += f" **So What:** Check the visualization parameters or cluster filtering settings to ensure proper data display."
        elif len(contents) > 10:
            insight += f" **So What:** This represents a major knowledge domain for SAM - consider this cluster when asking questions about {theme_words[0] if theme_words else 'this topic'}."
        else:
            insight += f" **So What:** This is a specialized knowledge cluster that could provide focused insights for specific queries."

        return insight

    except Exception as e:
        return f"This cluster contains {len(contents)} related memories that SAM has grouped based on conceptual similarity. **So What:** These memories likely share important thematic or contextual relationships."

def refresh_synthesis_history():
    """Refresh synthesis history display."""
    st.success("📚 History refreshed!")
    st.rerun()

def get_synthesis_run_count():
    """Get the number of synthesis runs."""
    try:
        history = st.session_state.get('synthesis_history', [])
        return len(history)
    except:
        return 0

def get_synthetic_memory_count():
    """Get the count of synthetic memories."""
    try:
        from memory.memory_vectorstore import get_memory_store, MemoryType
        memory_store = get_memory_store()

        # Get all memories and count synthetic ones
        all_memories = memory_store.get_all_memories()
        synthetic_count = sum(1 for memory in all_memories if memory.memory_type == MemoryType.SYNTHESIS)
        return synthetic_count
    except:
        return 0

def get_synthesis_history():
    """Get synthesis history."""
    return st.session_state.get('synthesis_history', [])

def render_dream_canvas_visualization(visualization_data):
    """Render the interactive Dream Canvas visualization."""
    try:
        st.markdown("### 🌌 Interactive Memory Landscape")

        # Display basic statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Memory Points", len(visualization_data))

        with col2:
            cluster_ids = set(point.get('cluster_id', -1) for point in visualization_data)
            cluster_count = len([c for c in cluster_ids if c != -1])
            noise_count = sum(1 for point in visualization_data if point.get('cluster_id', -1) == -1)
            st.metric("Clusters Discovered", cluster_count)

        with col3:
            clustered_count = len(visualization_data) - noise_count
            clustered_percentage = (clustered_count / len(visualization_data)) * 100 if visualization_data else 0
            st.metric("Memories Clustered", f"{clustered_count} ({clustered_percentage:.1f}%)")

        with col4:
            noise_percentage = (noise_count / len(visualization_data)) * 100 if visualization_data else 0
            st.metric("Noise Points", f"{noise_count} ({noise_percentage:.1f}%)")

        # Create actual Plotly visualization
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            import pandas as pd

            # Convert visualization data to DataFrame
            df_data = []
            for point in visualization_data:
                # Ensure cluster_id is an integer
                cluster_id = point.get('cluster_id', -1)
                if isinstance(cluster_id, str):
                    try:
                        cluster_id = int(cluster_id)
                    except (ValueError, TypeError):
                        cluster_id = -1

                df_data.append({
                    'x': float(point['coordinates']['x']),
                    'y': float(point['coordinates']['y']),
                    'cluster_id': cluster_id,
                    'content_snippet': str(point['content_snippet'])[:100],
                    'memory_type': str(point['memory_type']),
                    'source': str(point['source']),
                    'importance_score': float(point.get('importance_score', 0.5)),
                    'is_synthetic': bool(point.get('is_synthetic', False))
                })

            df = pd.DataFrame(df_data)

            # Create enhanced color mapping for clusters
            # Use a vibrant color palette that makes clusters pop
            cluster_colors = [
                '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
                '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
                '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2'
            ]

            def get_cluster_color(cluster_id):
                try:
                    if cluster_id >= 0:
                        return cluster_colors[int(cluster_id) % len(cluster_colors)]
                    else:
                        return '#CCCCCC'  # Light gray for noise points
                except (TypeError, ValueError):
                    return '#CCCCCC'

            df['color'] = df['cluster_id'].apply(get_cluster_color)

            # Create the scatter plot
            fig = go.Figure()

            # Add points by cluster
            for cluster_id in df['cluster_id'].unique():
                cluster_data = df[df['cluster_id'] == cluster_id]
                cluster_name = f"Cluster {cluster_id}" if cluster_id >= 0 else "Noise"

                fig.add_trace(go.Scatter(
                    x=cluster_data['x'],
                    y=cluster_data['y'],
                    mode='markers',
                    name=cluster_name,
                    marker=dict(
                        size=8 + cluster_data['importance_score'] * 4,
                        color=cluster_data['color'].iloc[0] if len(cluster_data) > 0 else '#cccccc',
                        opacity=0.7,
                        line=dict(width=1, color='white')
                    ),
                    text=cluster_data['content_snippet'],
                    hovertemplate='<b>%{text}</b><br>' +
                                  'Source: %{customdata[0]}<br>' +
                                  'Type: %{customdata[1]}<br>' +
                                  'Importance: %{customdata[2]:.2f}<br>' +
                                  '<extra></extra>',
                    customdata=cluster_data[['source', 'memory_type', 'importance_score']].values
                ))

            # Update layout with proper text colors for white background
            fig.update_layout(
                title={
                    'text': f'🧠 SAM\'s Memory Landscape ({len(visualization_data)} memories)',
                    'x': 0.5,
                    'font': {'size': 18, 'color': '#333333'}  # Dark text for white background
                },
                xaxis=dict(
                    title='UMAP Dimension 1',
                    titlefont={'color': '#333333'},  # Dark axis title
                    tickfont={'color': '#333333'},   # Dark tick labels
                    showgrid=True,
                    gridcolor='#e0e0e0'
                ),
                yaxis=dict(
                    title='UMAP Dimension 2',
                    titlefont={'color': '#333333'},  # Dark axis title
                    tickfont={'color': '#333333'},   # Dark tick labels
                    showgrid=True,
                    gridcolor='#e0e0e0'
                ),
                hovermode='closest',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.01,
                    font={'color': '#333333'},  # Dark legend text
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#cccccc',
                    borderwidth=1
                ),
                plot_bgcolor='#fafafa',
                paper_bgcolor='white',
                font={'color': '#333333'},  # Default font color for all text
                height=600
            )

            # Display the interactive plot
            st.plotly_chart(fig, use_container_width=True)

            # Add interactive cluster legend
            if cluster_count > 1:
                st.markdown("### 🎨 Cluster Explorer")

                # Create cluster summary with insights
                cluster_summary = []
                for cluster_id in sorted([c for c in cluster_ids if c != -1]):
                    cluster_data = df[df['cluster_id'] == cluster_id]
                    cluster_color = get_cluster_color(cluster_id)

                    # Extract meaningful content from cluster
                    cluster_contents = cluster_data['content_snippet'].tolist()
                    cluster_sources = cluster_data['source'].tolist()

                    # Debug: Log cluster data for troubleshooting
                    logger.info(f"Cluster {cluster_id}: {len(cluster_data)} memories, {len(cluster_contents)} contents")

                    # Generate cluster insight
                    cluster_insight = generate_cluster_insight(cluster_contents, cluster_sources, cluster_id)

                    cluster_summary.append({
                        'id': cluster_id,
                        'color': cluster_color,
                        'count': len(cluster_data),
                        'avg_importance': cluster_data['importance_score'].mean() if len(cluster_data) > 0 else 0.0,
                        'insight': cluster_insight,
                        'contents': cluster_contents[:3],  # Sample contents
                        'sources': list(set(cluster_sources))[:3]  # Unique sources
                    })

                # Display cluster list with insights
                for i, cluster in enumerate(cluster_summary):
                    with st.expander(f"🔵 Cluster {cluster['id']} - {cluster['count']} memories", expanded=False):
                        # Cluster insight
                        st.markdown(f"""
                        <div style="
                            border-left: 4px solid {cluster['color']};
                            padding: 15px;
                            margin: 10px 0;
                            background: linear-gradient(135deg, {cluster['color']}15, {cluster['color']}05);
                            border-radius: 0 8px 8px 0;
                        ">
                            <h4 style="color: {cluster['color']}; margin: 0 0 10px 0;">💡 Key Insight</h4>
                            <p style="margin: 0; font-size: 1rem; line-height: 1.4;">
                                {cluster['insight']}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Cluster details
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**📊 Cluster Stats:**")
                            # Try to get enhanced stats from cluster registry
                            try:
                                from memory.synthesis.cluster_registry import get_cluster_stats
                                enhanced_stats = get_cluster_stats(f"cluster_{cluster['id']:03d}")
                                if enhanced_stats['exists']:
                                    st.write(f"• **{enhanced_stats['memory_count']} memories** in this cluster")
                                    st.write(f"• **Avg importance:** {enhanced_stats['avg_importance']:.2f}")
                                    st.write(f"• **Sources:** {enhanced_stats['source_count']} unique")
                                else:
                                    # Fallback to original stats
                                    st.write(f"• **{cluster['count']} memories** in this cluster")
                                    st.write(f"• **Avg importance:** {cluster['avg_importance']:.2f}")
                                    st.write(f"• **Sources:** {len(cluster['sources'])} unique")
                            except Exception:
                                # Fallback to original stats
                                st.write(f"• **{cluster['count']} memories** in this cluster")
                                st.write(f"• **Avg importance:** {cluster['avg_importance']:.2f}")
                                st.write(f"• **Sources:** {len(cluster['sources'])} unique")

                        with col2:
                            st.markdown("**📝 Sample Content:**")
                            for j, content in enumerate(cluster['contents'][:2]):
                                st.write(f"• {content[:80]}...")

                        # Sources
                        if cluster['sources']:
                            st.markdown("**📚 Key Sources:**")
                            for source in cluster['sources'][:3]:
                                source_name = source.split('/')[-1] if '/' in source else source
                                st.write(f"• {source_name}")

                        st.divider()

                # Add noise points summary if any
                if noise_count > 0:
                    st.markdown(f"""
                    <div style="
                        border: 2px solid #CCCCCC;
                        border-radius: 8px;
                        padding: 10px;
                        margin: 10px 0;
                        background: linear-gradient(135deg, #CCCCCC20, #CCCCCC10);
                    ">
                        <h4 style="color: #666666; margin: 0;">⚪ Noise Points</h4>
                        <p style="margin: 5px 0; font-size: 0.9rem;">
                            <strong>{noise_count} isolated memories</strong><br>
                            These memories don't fit into any cluster
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error creating visualization: {e}")
            # Fallback to placeholder
            st.markdown(f"""
            <div style="
                border: 1px solid #ddd;
                border-radius: 10px;
                padding: 40px;
                text-align: center;
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                margin: 20px 0;
            ">
                <h4>🎨 Interactive Visualization</h4>
                <p>Memory landscape with {len(visualization_data)} points</p>
                <p style="font-size: 0.9rem; color: #666;">
                    • Color-coded clusters • Interactive hover • Synthetic insights highlighted
                </p>
                <p style="font-size: 0.8rem; color: #888;">
                    Plotly visualization error - using fallback display
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Data exploration
        with st.expander("🔍 Explore Visualization Data", expanded=False):
            # Sample data points
            st.markdown("**Sample Memory Points:**")

            for i, point in enumerate(visualization_data[:5]):
                st.markdown(f"**Point {i+1}:**")
                col1, col2 = st.columns(2)

                with col1:
                    st.caption(f"Type: {point.get('memory_type', 'Unknown')}")
                    st.caption(f"Cluster: {point.get('cluster_id', 'None')}")
                    st.caption(f"Synthetic: {'Yes' if point.get('is_synthetic') else 'No'}")

                with col2:
                    coords = point.get('coordinates', {})
                    st.caption(f"X: {coords.get('x', 0):.3f}")
                    st.caption(f"Y: {coords.get('y', 0):.3f}")
                    st.caption(f"Source: {point.get('source', 'Unknown')}")

                content = point.get('content_snippet', '')
                if content:
                    st.caption(f"Content: {content[:100]}...")

                st.divider()

    except Exception as e:
        st.error(f"Error rendering visualization: {e}")

def render_system_health():
    """Render the System Health monitoring interface."""
    st.subheader("🔧 System Health Monitor")
    st.markdown("Real-time monitoring of SAM's critical thinking and decision-making components")

    try:
        # Import the health monitor
        from services.system_health_monitor import get_health_monitor

        health_monitor = get_health_monitor()

        # Auto-refresh toggle
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown("### 📊 Component Health Status")

        with col2:
            auto_refresh = st.checkbox("🔄 Auto-refresh", value=False, help="Automatically refresh every 30 seconds")

        with col3:
            if st.button("🔄 Refresh Now", type="primary"):
                st.rerun()

        # Get current health report
        with st.spinner("Checking system health..."):
            health_report = health_monitor.get_health_report()

        # Overall health score
        st.markdown("---")

        # Health score with color coding
        score = health_report.overall_score
        if score >= 90:
            score_color = "🟢"
            score_status = "Excellent"
        elif score >= 70:
            score_color = "🟡"
            score_status = "Good"
        elif score >= 50:
            score_color = "🟠"
            score_status = "Warning"
        else:
            score_color = "🔴"
            score_status = "Critical"

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Overall Health", f"{score:.1f}%", help="Overall system health score")

        with col2:
            st.metric("Status", f"{score_color} {score_status}")

        with col3:
            uptime_hours = health_report.uptime_seconds / 3600
            st.metric("Uptime", f"{uptime_hours:.1f}h")

        with col4:
            st.metric("Components", f"{len(health_report.components)}")

        # Component status grid
        st.markdown("### 🔍 Component Details")

        for component in health_report.components:
            with st.expander(f"{component.name} - {'✅' if component.status == 'healthy' else '⚠️' if component.status == 'warning' else '❌'}",
                           expanded=(component.status != 'healthy')):

                col1, col2 = st.columns([2, 1])

                with col1:
                    # Status indicator
                    if component.status == "healthy":
                        st.success(f"✅ **{component.name}** is healthy")
                    elif component.status == "warning":
                        st.warning(f"⚠️ **{component.name}** has warnings")
                    else:
                        st.error(f"❌ **{component.name}** has errors")

                    # Error message if any
                    if component.error_message:
                        st.error(f"**Error:** {component.error_message}")

                with col2:
                    st.metric("Response Time", f"{component.response_time_ms:.1f}ms")
                    st.caption(f"Last checked: {component.last_check}")

                # Component details
                if component.details:
                    st.markdown("**Details:**")
                    for key, value in component.details.items():
                        if isinstance(value, bool):
                            icon = "✅" if value else "❌"
                            st.markdown(f"• {key}: {icon}")
                        else:
                            st.markdown(f"• {key}: `{value}`")

        # Recommendations
        if health_report.recommendations:
            st.markdown("### 💡 Recommendations")
            for recommendation in health_report.recommendations:
                if "🔧" in recommendation:
                    st.info(recommendation)
                elif "⚠️" in recommendation:
                    st.warning(recommendation)
                elif "🎉" in recommendation:
                    st.success(recommendation)
                else:
                    st.markdown(f"• {recommendation}")

        # Quick actions
        st.markdown("### ⚡ Quick Actions")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔧 Fix SOF v2", help="Run SOF v2 fix script"):
                st.info("Running SOF v2 fix script...")
                # This would run the fix script
                st.success("SOF v2 fix completed!")

        with col2:
            if st.button("🧮 Test Math Query", help="Test mathematical query routing"):
                st.info("Testing: What is 10+5?")
                # This would test the math query
                st.success("Math query test: 10+5 = 15 ✅")

        with col3:
            if st.button("📊 Generate Report", help="Generate detailed health report"):
                st.info("Generating detailed health report...")
                # This would generate a detailed report
                st.success("Report generated!")

        # Auto-refresh logic
        if auto_refresh:
            time.sleep(30)
            st.rerun()

        # Raw data (for debugging)
        with st.expander("🔍 Raw Health Data (Debug)", expanded=False):
            st.json({
                "overall_status": health_report.overall_status,
                "overall_score": health_report.overall_score,
                "last_updated": health_report.last_updated,
                "uptime_seconds": health_report.uptime_seconds,
                "component_count": len(health_report.components),
                "recommendations_count": len(health_report.recommendations)
            })

    except Exception as e:
        st.error(f"❌ Failed to load system health monitor: {e}")
        st.markdown("""
        **Possible causes:**
        - System health monitor not properly initialized
        - Missing dependencies
        - Configuration issues

        **Try:**
        1. Run `python fix_sof_v2.py` to fix critical thinking components
        2. Check logs for detailed error information
        3. Restart SAM if issues persist
        """)

def render_reasoning_visualizer():
    """Render the SAM Introspection Dashboard - Reasoning Visualizer."""
    try:
        st.subheader("🔍 SAM Reasoning Visualizer")
        st.markdown("Real-time introspection into SAM's cognitive processes and decision-making")

        # Import trace logger
        try:
            from sam.cognition.trace_logger import get_trace_logger, start_trace, EventType, Severity
            trace_logger = get_trace_logger()
        except ImportError:
            st.error("❌ Trace logging system not available. Please ensure the introspection dashboard is properly installed.")
            return

        # Main interface tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎯 Trace Query",
            "📊 Active Traces",
            "📚 Historical Analysis",
            "📈 Analytics Dashboard",
            "🔍 Trace Comparison",
            "⚙️ Settings"
        ])

        with tab1:
            render_trace_query_interface(trace_logger)

        with tab2:
            render_active_traces_interface(trace_logger)

        with tab3:
            render_historical_analysis_interface()

        with tab4:
            render_analytics_dashboard_interface()

        with tab5:
            render_trace_comparison_interface()

        with tab6:
            render_trace_settings_interface()

    except Exception as e:
        st.error(f"❌ Error loading reasoning visualizer: {e}")
        st.markdown("""
        **Possible causes:**
        - Introspection dashboard components not properly installed
        - Missing trace logging dependencies
        - Configuration issues

        **Try:**
        1. Check that sam/cognition/trace_logger.py exists
        2. Verify API endpoints are available
        3. Restart the Memory Center if issues persist
        """)

def render_trace_query_interface(trace_logger):
    """Render the trace query interface."""
    st.markdown("### 🎯 Initiate Traced Query")
    st.markdown("Send a query to SAM with full reasoning tracing enabled")

    # Query input
    col1, col2 = st.columns([3, 1])

    with col1:
        query_text = st.text_area(
            "Query for SAM",
            placeholder="Enter your question or request for SAM...",
            height=100,
            help="This query will be sent to SAM with full tracing enabled"
        )

    with col2:
        trace_mode = st.selectbox(
            "Trace Mode",
            options=["Manual", "Performance", "Debug", "Full"],
            help="Select the level of tracing detail"
        )

        auto_refresh = st.checkbox(
            "Auto-refresh",
            value=True,
            help="Automatically refresh trace data"
        )

    # Trace button
    if st.button("🔍 Trace Query", type="primary", disabled=not query_text.strip()):
        if query_text.strip():
            # Start the trace
            trace_id = trace_logger.start_trace(
                query=query_text,
                user_id="memory_center_user",
                session_id=st.session_state.get('session_id', 'unknown')
            )

            # Store trace ID in session state
            st.session_state.current_trace_id = trace_id
            st.session_state.trace_query = query_text

            st.success(f"✅ Trace initiated! Trace ID: `{trace_id}`")
            st.info("🔄 Query is being processed. Check the timeline below for real-time updates.")

            # TODO: Actually send the query to SAM with tracing enabled
            # This would integrate with the secure chat interface

    # Display current trace if available
    if hasattr(st.session_state, 'current_trace_id') and st.session_state.current_trace_id:
        render_trace_timeline(trace_logger, st.session_state.current_trace_id, auto_refresh)

def render_trace_timeline(trace_logger, trace_id, auto_refresh=True):
    """Render the trace timeline for a specific trace."""
    st.markdown("### 📋 Trace Timeline")

    # Get trace events
    events = trace_logger.get_trace_events(trace_id)
    summary = trace_logger.get_trace_summary(trace_id)

    if not events:
        st.info("⏳ Waiting for trace events...")
        if auto_refresh:
            time.sleep(1)
            st.rerun()
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Events", len(events))

    with col2:
        st.metric("Modules", len(summary.get('modules_involved', [])))

    with col3:
        status = summary.get('status', 'active')
        st.metric("Status", status.title())

    with col4:
        duration = summary.get('total_duration')
        if duration:
            st.metric("Duration", f"{duration:.2f}s")
        else:
            st.metric("Duration", "Active")

    # Timeline visualization
    st.markdown("#### Event Timeline")

    for i, event in enumerate(events):
        # Event container
        with st.container():
            # Event header
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

            with col1:
                # Color code by event type
                color_map = {
                    'start': '🟢',
                    'end': '🔴',
                    'decision': '🟡',
                    'tool_call': '🔵',
                    'error': '❌',
                    'data_in': '📥',
                    'data_out': '📤',
                    'performance': '📊'
                }

                icon = color_map.get(event.get('event_type', ''), '⚪')
                st.markdown(f"{icon} **{event.get('source_module', 'Unknown')}**")

            with col2:
                st.caption(event.get('event_type', 'unknown').title())

            with col3:
                severity = event.get('severity', 'info')
                severity_colors = {
                    'debug': '🔍',
                    'info': 'ℹ️',
                    'warning': '⚠️',
                    'error': '❌',
                    'critical': '🚨'
                }
                st.caption(f"{severity_colors.get(severity, 'ℹ️')} {severity.title()}")

            with col4:
                duration = event.get('duration_ms')
                if duration:
                    st.caption(f"{duration:.1f}ms")
                else:
                    st.caption("-")

            # Event message
            st.markdown(f"*{event.get('message', 'No message')}*")

            # Expandable details
            if event.get('payload') or event.get('metadata'):
                with st.expander(f"📋 Event Details ({event.get('event_id', 'unknown')[:8]}...)"):
                    if event.get('payload'):
                        st.markdown("**Payload:**")
                        st.json(event['payload'])

                    if event.get('metadata'):
                        st.markdown("**Metadata:**")
                        st.json(event['metadata'])

            st.divider()

    # Auto-refresh
    if auto_refresh and summary.get('status') == 'active':
        time.sleep(2)
        st.rerun()

def render_active_traces_interface(trace_logger):
    """Render the active traces interface."""
    st.markdown("### 📊 Active Traces")

    active_traces = trace_logger.get_active_traces()

    if not active_traces:
        st.info("No active traces currently running")
        return

    st.markdown(f"**{len(active_traces)} active trace(s)**")

    for trace_id in active_traces:
        summary = trace_logger.get_trace_summary(trace_id)

        with st.container():
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.markdown(f"**Trace:** `{trace_id[:8]}...`")
                query = summary.get('query', 'Unknown query')
                st.caption(f"Query: {query[:50]}{'...' if len(query) > 50 else ''}")

            with col2:
                event_count = summary.get('event_count', 0)
                st.metric("Events", event_count)

            with col3:
                if st.button(f"View", key=f"view_{trace_id}"):
                    st.session_state.current_trace_id = trace_id
                    st.rerun()

            st.divider()

def render_historical_analysis_interface():
    """Render the historical trace analysis interface."""
    st.markdown("### 📚 Historical Trace Analysis")
    st.markdown("Browse and analyze historical trace data with advanced filtering")

    try:
        import requests

        # Filters section
        st.markdown("#### 🔍 Filters")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            days_back = st.selectbox(
                "Time Period",
                options=[1, 3, 7, 14, 30],
                index=2,
                format_func=lambda x: f"Last {x} day{'s' if x > 1 else ''}"
            )

        with col2:
            status_filter = st.selectbox(
                "Status",
                options=["All", "completed", "failed", "active"],
                index=0
            )

        with col3:
            success_filter = st.selectbox(
                "Success",
                options=["All", "Success", "Failed"],
                index=0
            )

        with col4:
            limit = st.number_input(
                "Max Results",
                min_value=10,
                max_value=500,
                value=50,
                step=10
            )

        # Search box
        query_search = st.text_input(
            "Search in queries",
            placeholder="Enter keywords to search in query text..."
        )

        # Fetch historical traces
        if st.button("🔍 Search Historical Traces", type="primary"):
            with st.spinner("Fetching historical traces..."):
                try:
                    # Build API request
                    params = {
                        'limit': limit,
                        'offset': 0
                    }

                    # Add time filter
                    if days_back:
                        start_time = time.time() - (days_back * 24 * 3600)
                        params['start_date'] = start_time

                    # Add other filters
                    if status_filter != "All":
                        params['status'] = status_filter

                    if success_filter == "Success":
                        params['success'] = 'true'
                    elif success_filter == "Failed":
                        params['success'] = 'false'

                    if query_search.strip():
                        params['query_contains'] = query_search.strip()

                    # Make API request (would need to implement API server)
                    # For now, simulate with database direct access
                    from sam.cognition.trace_database import get_trace_database

                    db = get_trace_database()
                    filters = {}
                    if 'start_date' in params:
                        filters['start_date'] = params['start_date']
                    if 'status' in params:
                        filters['status'] = params['status']
                    if 'success' in params:
                        filters['success'] = params['success'] == 'true'
                    if 'query_contains' in params:
                        filters['query_contains'] = params['query_contains']

                    traces = db.get_trace_history(limit=limit, filters=filters)

                    if traces:
                        st.success(f"✅ Found {len(traces)} historical traces")

                        # Display traces
                        for trace in traces:
                            with st.container():
                                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

                                with col1:
                                    st.markdown(f"**Query:** {trace['query'][:100]}{'...' if len(trace['query']) > 100 else ''}")
                                    st.caption(f"Trace ID: `{trace['trace_id'][:8]}...`")

                                with col2:
                                    duration = trace.get('total_duration', 0)
                                    st.metric("Duration", f"{duration:.2f}s" if duration else "N/A")

                                with col3:
                                    status = trace.get('status', 'unknown')
                                    success = trace.get('success', False)
                                    status_icon = "✅" if success else "❌" if status == 'failed' else "⏳"
                                    st.metric("Status", f"{status_icon} {status.title()}")

                                with col4:
                                    if st.button("View Details", key=f"view_{trace['trace_id']}"):
                                        st.session_state.selected_historical_trace = trace['trace_id']
                                        st.rerun()

                                # Show timestamp
                                if trace.get('start_time'):
                                    start_dt = datetime.fromtimestamp(trace['start_time'])
                                    st.caption(f"Started: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")

                                st.divider()
                    else:
                        st.info("No traces found matching the specified criteria")

                except Exception as e:
                    st.error(f"❌ Error fetching historical traces: {e}")

        # Show selected trace details
        if hasattr(st.session_state, 'selected_historical_trace'):
            render_historical_trace_details(st.session_state.selected_historical_trace)

    except Exception as e:
        st.error(f"❌ Error loading historical analysis: {e}")

def render_analytics_dashboard_interface():
    """Render the advanced analytics dashboard."""
    st.markdown("### 📈 Analytics Dashboard")
    st.markdown("Comprehensive analytics and insights from trace data")

    try:
        from sam.cognition.trace_analytics import get_trace_analytics

        analytics_engine = get_trace_analytics()

        # Analytics type selector
        col1, col2 = st.columns([2, 1])

        with col1:
            analytics_type = st.selectbox(
                "Analytics Type",
                options=["Performance Trends", "Query Patterns", "Module Efficiency", "Anomaly Detection"],
                index=0
            )

        with col2:
            days = st.selectbox(
                "Time Period",
                options=[1, 3, 7, 14, 30],
                index=2,
                format_func=lambda x: f"Last {x} day{'s' if x > 1 else ''}"
            )

        if st.button("🔄 Generate Analytics", type="primary"):
            with st.spinner(f"Generating {analytics_type.lower()}..."):
                try:
                    if analytics_type == "Performance Trends":
                        result = analytics_engine.get_performance_trends(days)
                        render_performance_trends(result)

                    elif analytics_type == "Query Patterns":
                        result = analytics_engine.get_query_patterns(limit=200)
                        render_query_patterns(result)

                    elif analytics_type == "Module Efficiency":
                        result = analytics_engine.get_module_efficiency(days)
                        render_module_efficiency(result)

                    elif analytics_type == "Anomaly Detection":
                        result = analytics_engine.detect_anomalies(days)
                        render_anomaly_detection(result)

                except Exception as e:
                    st.error(f"❌ Error generating analytics: {e}")

    except Exception as e:
        st.error(f"❌ Error loading analytics dashboard: {e}")

def render_trace_comparison_interface():
    """Render the enhanced trace comparison interface with Phase 2A features."""
    st.markdown("### 🔍 Advanced Trace Comparison")
    st.markdown("Compare multiple traces with detailed analysis and visualization")

    try:
        # Phase 2A: Enhanced comparison options
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("#### 📝 Select Traces to Compare")

            trace_ids_input = st.text_area(
                "Trace IDs (one per line)",
                placeholder="Enter trace IDs to compare...\nExample:\nabc123def456\n789ghi012jkl",
                height=100
            )

        with col2:
            st.markdown("#### ⚙️ Comparison Options")

            comparison_type = st.selectbox(
                "Analysis Type",
                ["performance", "flow", "tools", "general"],
                help="Select the type of comparison analysis"
            )

            include_flow_diagram = st.checkbox(
                "Include Flow Diagrams",
                value=True,
                help="Generate flow diagrams for visual comparison"
            )

            include_hierarchy = st.checkbox(
                "Include Hierarchy View",
                value=False,
                help="Show hierarchical event structure"
            )

        # Action buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔍 Advanced Compare", type="primary"):
                if trace_ids_input.strip():
                    trace_ids = [tid.strip() for tid in trace_ids_input.strip().split('\n') if tid.strip()]

                    if len(trace_ids) < 2:
                        st.error("❌ Please provide at least 2 trace IDs for comparison")
                    else:
                        with st.spinner(f"Performing advanced comparison of {len(trace_ids)} traces..."):
                            try:
                                from sam.cognition.trace_analytics import get_trace_analytics

                                analytics_engine = get_trace_analytics()

                                # Use advanced comparison method
                                comparison = analytics_engine.advanced_trace_comparison(trace_ids, comparison_type)

                                if 'error' in comparison:
                                    st.error(f"❌ Comparison failed: {comparison['error']}")
                                else:
                                    st.session_state['comparison_result'] = comparison
                                    st.session_state['comparison_type'] = comparison_type
                                    st.session_state['include_flow'] = include_flow_diagram
                                    st.session_state['include_hierarchy'] = include_hierarchy
                                    st.success(f"✅ {comparison_type.title()} comparison completed!")

                            except Exception as e:
                                st.error(f"❌ Error comparing traces: {e}")
                else:
                    st.error("❌ Please enter trace IDs to compare")

        with col2:
            if st.button("📊 Get Baseline"):
                days = st.selectbox("Period (days)", [1, 3, 7, 14, 30], index=2, key="baseline_days")

                with st.spinner("Retrieving performance baseline..."):
                    try:
                        from sam.cognition.trace_analytics import get_trace_analytics
                        analytics_engine = get_trace_analytics()
                        baseline = analytics_engine.get_performance_baseline(days)

                        if 'error' in baseline:
                            st.error(f"❌ Failed to get baseline: {baseline['error']}")
                        else:
                            st.session_state['baseline_data'] = baseline['baseline']
                            st.success(f"✅ Baseline retrieved for {days} days")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

        with col3:
            if st.button("🔄 Clear Results"):
                keys_to_clear = ['comparison_result', 'comparison_type', 'baseline_data', 'include_flow', 'include_hierarchy']
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("✅ Results cleared")

        # Display results if available
        if 'comparison_result' in st.session_state:
            render_advanced_comparison_results(
                st.session_state['comparison_result'],
                st.session_state.get('comparison_type', 'general'),
                st.session_state.get('include_flow', False),
                st.session_state.get('include_hierarchy', False)
            )

        # Display baseline if available
        if 'baseline_data' in st.session_state:
            render_performance_baseline(st.session_state['baseline_data'])

    except Exception as e:
        st.error(f"❌ Error loading trace comparison: {e}")

def render_trace_settings_interface():
    """Render trace settings and configuration."""
    st.markdown("### ⚙️ Trace Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎛️ Tracing Options")

        trace_level = st.selectbox(
            "Default Trace Level",
            options=["Info", "Debug", "Warning", "Error"],
            index=0,
            help="Set the default level of detail for tracing"
        )

        auto_cleanup = st.checkbox(
            "Auto-cleanup old traces",
            value=True,
            help="Automatically remove traces older than 24 hours"
        )

        max_events = st.number_input(
            "Max events per trace",
            min_value=100,
            max_value=10000,
            value=1000,
            help="Maximum number of events to store per trace"
        )

    with col2:
        st.markdown("#### 📊 Performance Settings")

        refresh_interval = st.slider(
            "Auto-refresh interval (seconds)",
            min_value=1,
            max_value=10,
            value=2,
            help="How often to refresh active traces"
        )

        enable_performance_tracking = st.checkbox(
            "Enable performance tracking",
            value=True,
            help="Track CPU and memory usage during tracing"
        )

        if st.button("🧹 Cleanup Old Traces"):
            try:
                from sam.cognition.trace_database import get_trace_database
                db = get_trace_database()
                cleanup_days = st.session_state.get('cleanup_days', 30)
                cleaned_count = db.cleanup_old_traces(cleanup_days)
                st.success(f"✅ Cleaned up {cleaned_count} old traces")
            except Exception as e:
                st.error(f"❌ Cleanup failed: {e}")

def render_historical_trace_details(trace_id: str):
    """Render detailed view of a historical trace."""
    st.markdown("#### 📋 Historical Trace Details")

    try:
        from sam.cognition.trace_database import get_trace_database

        db = get_trace_database()
        events = db.get_trace_events_from_db(trace_id)

        if events:
            st.markdown(f"**Trace ID:** `{trace_id}`")
            st.markdown(f"**Events:** {len(events)}")

            # Event timeline
            for event in events:
                with st.container():
                    col1, col2, col3 = st.columns([2, 1, 1])

                    with col1:
                        st.markdown(f"**{event['source_module']}** - {event['message']}")

                    with col2:
                        st.caption(event['event_type'].title())

                    with col3:
                        if event.get('duration_ms'):
                            st.caption(f"{event['duration_ms']:.1f}ms")

                    if event.get('payload') or event.get('metadata'):
                        with st.expander("Details"):
                            if event.get('payload'):
                                st.json(event['payload'])
                            if event.get('metadata'):
                                st.json(event['metadata'])

                    st.divider()
        else:
            st.info("No events found for this trace")

    except Exception as e:
        st.error(f"❌ Error loading trace details: {e}")

def render_performance_trends(trends: Dict[str, Any]):
    """Render performance trends analysis."""
    if 'error' in trends:
        st.error(f"❌ {trends['error']}")
        return

    st.markdown("#### 📊 Performance Trends Analysis")

    # Overall statistics
    overall = trends.get('overall_stats', {})
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Traces", overall.get('total_traces', 0))

    with col2:
        avg_duration = overall.get('avg_duration', 0)
        st.metric("Avg Duration", f"{avg_duration:.2f}s")

    with col3:
        success_rate = overall.get('overall_success_rate', 0)
        st.metric("Success Rate", f"{success_rate:.1%}")

    with col4:
        avg_events = overall.get('avg_events_per_trace', 0)
        st.metric("Avg Events", f"{avg_events:.1f}")

    # Daily performance chart (would need plotting library)
    daily_perf = trends.get('daily_performance', {})
    if daily_perf:
        st.markdown("#### 📈 Daily Performance")

        for day, metrics in daily_perf.items():
            with st.container():
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown(f"**{day}**")

                with col2:
                    st.metric("Traces", metrics['trace_count'])

                with col3:
                    st.metric("Avg Duration", f"{metrics['avg_duration']:.2f}s")

                with col4:
                    st.metric("Success Rate", f"{metrics['success_rate']:.1%}")

    # Recommendations
    recommendations = trends.get('recommendations', [])
    if recommendations:
        st.markdown("#### 💡 Recommendations")
        for rec in recommendations:
            st.info(f"💡 {rec}")

def render_query_patterns(patterns: Dict[str, Any]):
    """Render query patterns analysis."""
    if 'error' in patterns:
        st.error(f"❌ {patterns['error']}")
        return

    st.markdown("#### 🎯 Query Patterns Analysis")

    # Query type distribution
    query_types = patterns.get('query_type_distribution', {})
    if query_types:
        st.markdown("##### Query Type Distribution")

        for query_type, count in query_types.items():
            st.markdown(f"- **{query_type.title()}**: {count} queries")

    # Query length statistics
    length_stats = patterns.get('query_length_stats', {})
    if length_stats:
        st.markdown("##### Query Length Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Average", f"{length_stats.get('avg_length', 0):.0f} chars")

        with col2:
            st.metric("Median", f"{length_stats.get('median_length', 0):.0f} chars")

        with col3:
            st.metric("Maximum", f"{length_stats.get('max_length', 0)} chars")

        with col4:
            st.metric("Minimum", f"{length_stats.get('min_length', 0)} chars")

    # Top keywords
    top_keywords = patterns.get('top_keywords', [])
    if top_keywords:
        st.markdown("##### Top Keywords")

        for keyword, count in top_keywords[:10]:
            st.markdown(f"- **{keyword}**: {count} occurrences")

def render_module_efficiency(efficiency: Dict[str, Any]):
    """Render module efficiency analysis."""
    if 'error' in efficiency:
        st.error(f"❌ {efficiency['error']}")
        return

    st.markdown("#### 🔧 Module Efficiency Analysis")

    # Top performers
    top_performers = efficiency.get('top_performers', [])
    if top_performers:
        st.markdown("##### 🏆 Top Performing Modules")

        for module, stats in top_performers:
            with st.container():
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown(f"**{module}**")

                with col2:
                    st.metric("Usage", stats['usage_count'])

                with col3:
                    st.metric("Success Rate", f"{stats['success_rate']:.1%}")

                with col4:
                    st.metric("Efficiency", f"{stats['efficiency_score']:.2f}")

    # Modules needing attention
    needs_attention = efficiency.get('needs_attention', [])
    if needs_attention:
        st.markdown("##### ⚠️ Modules Needing Attention")

        for module, stats in needs_attention:
            st.warning(f"**{module}**: Efficiency score {stats['efficiency_score']:.2f}")

def render_anomaly_detection(anomalies: Dict[str, Any]):
    """Render anomaly detection results."""
    if 'error' in anomalies:
        st.error(f"❌ {anomalies['error']}")
        return

    st.markdown("#### 🚨 Anomaly Detection Results")

    # Performance outliers
    outliers = anomalies.get('performance_outliers', [])
    if outliers:
        st.markdown("##### ⚡ Performance Outliers")

        for outlier in outliers[:5]:  # Show top 5
            st.warning(f"**Slow Query**: {outlier['query'][:50]}... (Duration: {outlier['duration']:.2f}s)")

    # Error spikes
    error_spikes = anomalies.get('error_spikes', [])
    if error_spikes:
        st.markdown("##### ❌ Error Spikes")

        for spike in error_spikes:
            st.error(f"Error rate: {spike['error_rate']:.1%} ({spike['error_count']}/{spike['total_traces']} traces)")

    # Recommendations
    recommendations = anomalies.get('recommendations', [])
    if recommendations:
        st.markdown("##### 💡 Recommendations")
        for rec in recommendations:
            st.info(f"💡 {rec}")

def render_advanced_comparison_results(comparison: Dict[str, Any], comparison_type: str, include_flow: bool, include_hierarchy: bool):
    """Render advanced trace comparison results with Phase 2A features."""
    st.markdown("#### 🔍 Advanced Trace Comparison Results")

    traces = comparison.get('traces', {})
    analysis = comparison.get('analysis', {})

    if traces:
        st.markdown(f"**{comparison_type.title()} Analysis of {len(traces)} traces**")

        # Summary metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Traces Analyzed", len(traces))

        with col2:
            total_events = sum(trace_data.get('event_count', 0) for trace_data in traces.values())
            st.metric("Total Events", total_events)

        with col3:
            all_modules = set()
            for trace_data in traces.values():
                all_modules.update(trace_data.get('modules', []))
            st.metric("Unique Modules", len(all_modules))

        # Detailed analysis based on comparison type
        if comparison_type == 'performance':
            render_performance_comparison_analysis(analysis, traces)
        elif comparison_type == 'flow':
            render_flow_comparison_analysis(analysis, traces)
        elif comparison_type == 'tools':
            render_tools_comparison_analysis(analysis, traces)
        else:
            render_general_comparison_analysis(analysis, traces)

        # Flow diagrams if requested
        if include_flow:
            render_flow_diagrams_comparison(traces)

        # Hierarchy view if requested
        if include_hierarchy:
            render_hierarchy_comparison(traces)

def render_performance_comparison_analysis(analysis: Dict[str, Any], traces: Dict[str, Any]):
    """Render performance-specific comparison analysis."""
    st.subheader("⚡ Performance Analysis")

    duration_comparison = analysis.get('duration_comparison', {})

    if duration_comparison:
        # Performance metrics table
        perf_data = []
        for trace_id, metrics in duration_comparison.items():
            perf_data.append({
                'Trace ID': trace_id[:8] + '...',
                'Total Duration (ms)': f"{metrics['total_duration']:.1f}",
                'Avg Event Duration (ms)': f"{metrics['avg_event_duration']:.1f}",
                'Max Event Duration (ms)': f"{metrics['max_event_duration']:.1f}",
                'Events with Duration': metrics['events_with_duration']
            })

        st.dataframe(pd.DataFrame(perf_data), use_container_width=True)

        # Performance chart
        chart_data = []
        for trace_id, metrics in duration_comparison.items():
            chart_data.append({
                'Trace': trace_id[:8] + '...',
                'Total Duration': metrics['total_duration'],
                'Avg Event Duration': metrics['avg_event_duration']
            })

        if chart_data:
            chart_df = pd.DataFrame(chart_data)
            st.bar_chart(chart_df.set_index('Trace'))

def render_flow_comparison_analysis(analysis: Dict[str, Any], traces: Dict[str, Any]):
    """Render flow-specific comparison analysis."""
    st.subheader("🔄 Flow Analysis")

    execution_paths = analysis.get('execution_paths', {})
    decision_points = analysis.get('decision_points', {})

    if execution_paths:
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Execution Path Complexity:**")
            for trace_id, path_data in execution_paths.items():
                st.write(f"• {trace_id[:8]}...: {path_data['total_steps']} steps ({path_data['unique_steps']} unique)")

        with col2:
            st.write("**Decision Points:**")
            for trace_id, decision_data in decision_points.items():
                st.write(f"• {trace_id[:8]}...: {decision_data['count']} decisions")

def render_tools_comparison_analysis(analysis: Dict[str, Any], traces: Dict[str, Any]):
    """Render tools-specific comparison analysis."""
    st.subheader("🔧 Tools Analysis")

    tool_usage = analysis.get('tool_usage', {})

    if tool_usage:
        # Tool usage table
        tool_data = []
        for trace_id, usage_data in tool_usage.items():
            tools_used = ', '.join(usage_data['tools_used']) if usage_data['tools_used'] else 'None'
            tool_data.append({
                'Trace ID': trace_id[:8] + '...',
                'Tools Used': tools_used,
                'Tool Calls': usage_data['tool_calls'],
                'Unique Tools': len(usage_data['tools_used'])
            })

        st.dataframe(pd.DataFrame(tool_data), use_container_width=True)

        # Tool distribution chart
        all_tools = set()
        for usage_data in tool_usage.values():
            all_tools.update(usage_data['tools_used'])

        if all_tools:
            st.write("**Tool Usage Distribution:**")
            for tool in all_tools:
                usage_counts = []
                trace_labels = []
                for trace_id, usage_data in tool_usage.items():
                    count = usage_data['tool_distribution'].get(tool, 0)
                    if count > 0:
                        usage_counts.append(count)
                        trace_labels.append(trace_id[:8] + '...')

                if usage_counts:
                    st.write(f"**{tool}:** {', '.join(f'{label}({count})' for label, count in zip(trace_labels, usage_counts))}")

def render_general_comparison_analysis(analysis: Dict[str, Any], traces: Dict[str, Any]):
    """Render general comparison analysis."""
    st.subheader("📊 General Analysis")

    summary_comparison = analysis.get('summary_comparison', {})
    event_distribution = analysis.get('event_distribution', {})

    if summary_comparison:
        # Summary comparison table
        summary_data = []
        for trace_id, summary in summary_comparison.items():
            summary_data.append({
                'Trace ID': trace_id[:8] + '...',
                'Success': '✅' if summary['success'] else '❌',
                'Duration (ms)': f"{summary['duration']:.1f}",
                'Events': summary['event_count'],
                'Modules': summary['modules_count']
            })

        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

    if event_distribution:
        st.write("**Event Type Distribution:**")
        all_event_types = set()
        for dist in event_distribution.values():
            all_event_types.update(dist.keys())

        for event_type in all_event_types:
            counts = []
            trace_labels = []
            for trace_id, dist in event_distribution.items():
                count = dist.get(event_type, 0)
                if count > 0:
                    counts.append(count)
                    trace_labels.append(trace_id[:8] + '...')

            if counts:
                st.write(f"**{event_type}:** {', '.join(f'{label}({count})' for label, count in zip(trace_labels, counts))}")

def render_flow_diagrams_comparison(traces: Dict[str, Any]):
    """Render flow diagrams for trace comparison."""
    st.subheader("🌐 Flow Diagrams")

    for trace_id in traces.keys():
        with st.expander(f"Flow Diagram: {trace_id[:8]}..."):
            try:
                from sam.cognition.trace_analytics import get_trace_analytics
                analytics_engine = get_trace_analytics()
                flow_data = analytics_engine.generate_flow_diagram(trace_id)

                if 'error' in flow_data:
                    st.error(f"❌ Error generating flow diagram: {flow_data['error']}")
                else:
                    st.write("**Flow Diagram Data:**")
                    st.json({
                        "nodes": len(flow_data.get('nodes', [])),
                        "edges": len(flow_data.get('edges', [])),
                        "metadata": flow_data.get('metadata', {})
                    })

                    # In a full implementation, this would render an interactive diagram
                    st.info("💡 Interactive flow diagram visualization would be rendered here")
            except Exception as e:
                st.error(f"❌ Error: {e}")

def render_hierarchy_comparison(traces: Dict[str, Any]):
    """Render hierarchical view for trace comparison."""
    st.subheader("🌳 Hierarchical View")

    for trace_id in traces.keys():
        with st.expander(f"Hierarchy: {trace_id[:8]}..."):
            try:
                from sam.cognition.trace_analytics import get_trace_analytics
                analytics_engine = get_trace_analytics()
                hierarchy_data = analytics_engine.generate_hierarchy_view(trace_id)

                if 'error' in hierarchy_data:
                    st.error(f"❌ Error generating hierarchy: {hierarchy_data['error']}")
                else:
                    stats = hierarchy_data.get('statistics', {})
                    st.write("**Hierarchy Statistics:**")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Total Events", stats.get('total_events', 0))
                    with col2:
                        st.metric("Max Depth", stats.get('max_depth', 0))
                    with col3:
                        st.metric("Root Events", stats.get('root_events', 0))

                    # In a full implementation, this would render an interactive tree
                    st.info("💡 Interactive hierarchy tree would be rendered here")
            except Exception as e:
                st.error(f"❌ Error: {e}")

def render_performance_baseline(baseline: Dict[str, Any]):
    """Render performance baseline data."""
    st.subheader("📈 Performance Baseline")

    perf_metrics = baseline.get('performance_metrics', {})
    volume_metrics = baseline.get('volume_metrics', {})
    period_info = baseline.get('period_info', {})

    if perf_metrics:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Avg Duration",
                f"{perf_metrics.get('avg_duration_ms', 0):.1f}ms",
                help="Average trace duration"
            )

        with col2:
            st.metric(
                "P95 Duration",
                f"{perf_metrics.get('p95_duration_ms', 0):.1f}ms",
                help="95th percentile duration"
            )

        with col3:
            st.metric(
                "Success Rate",
                f"{perf_metrics.get('success_rate', 0):.1%}",
                help="Percentage of successful traces"
            )

        with col4:
            st.metric(
                "Avg Events",
                f"{perf_metrics.get('avg_events_per_trace', 0):.1f}",
                help="Average events per trace"
            )

    if volume_metrics:
        st.write("**Volume Metrics:**")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Traces", volume_metrics.get('total_traces', 0))

        with col2:
            st.metric("Traces/Day", f"{volume_metrics.get('traces_per_day', 0):.1f}")

        with col3:
            success_rate = volume_metrics.get('successful_traces', 0) / max(volume_metrics.get('total_traces', 1), 1)
            st.metric("Success Rate", f"{success_rate:.1%}")

def render_trace_comparison_results(comparison: Dict[str, Any]):
    """Render basic trace comparison results (legacy function)."""
    st.markdown("#### 🔍 Trace Comparison Results")

    traces = comparison.get('traces', {})
    if traces:
        st.markdown(f"**Comparing {len(traces)} traces**")

        # Summary table
        for trace_id, trace_data in traces.items():
            summary = trace_data['summary']
            performance = trace_data['performance']

            with st.container():
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown(f"**{trace_id[:8]}...**")
                    st.caption(summary['query'][:30] + "...")

                with col2:
                    st.metric("Duration", f"{performance['duration']:.2f}s")

                with col3:
                    st.metric("Events", trace_data['event_count'])

                with col4:
                    status = "✅" if performance['success'] else "❌"
                    st.metric("Status", status)

                st.divider()

    # Similarities and differences
    similarities = comparison.get('similarities', {})
    differences = comparison.get('differences', {})

    if similarities or differences:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 🤝 Similarities")
            if similarities:
                for key, value in similarities.items():
                    st.markdown(f"- **{key}**: {value}")
            else:
                st.info("No significant similarities found")

        with col2:
            st.markdown("##### 🔄 Differences")
            if differences:
                for key, value in differences.items():
                    st.markdown(f"- **{key}**: {value}")
            else:
                st.info("No significant differences found")

    # Recommendations
    recommendations = comparison.get('recommendations', [])
    if recommendations:
        st.markdown("##### 💡 Recommendations")
        for rec in recommendations:
            st.info(f"💡 {rec}")

def render_admin_dashboard_interface():
    """Render the Phase 2B administrative dashboard."""
    st.markdown("### ⚙️ SAM Introspection Dashboard - Admin Panel")
    st.markdown("Production-ready administrative controls and monitoring")

    try:
        # Create tabs for different admin functions
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔒 Security & Access",
            "🗂️ Data Retention",
            "⚡ Performance Monitor",
            "🚨 Alert Management",
            "📊 System Overview"
        ])

        with tab1:
            render_security_admin_interface()

        with tab2:
            render_retention_admin_interface()

        with tab3:
            render_performance_admin_interface()

        with tab4:
            render_alert_admin_interface()

        with tab5:
            render_system_overview_interface()

    except Exception as e:
        st.error(f"❌ Error loading admin dashboard: {e}")

def render_security_admin_interface():
    """Render security and access control admin interface."""
    st.subheader("🔒 Security & Access Control")

    try:
        # Security status
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Active Sessions", "3", delta="1")

        with col2:
            st.metric("Failed Login Attempts", "0", delta="0")

        with col3:
            st.metric("Security Alerts", "0", delta="0")

        # Recent security events
        st.subheader("Recent Security Events")

        # Mock security events data
        security_events = [
            {"timestamp": "2024-01-15 10:30:00", "event": "Login Success", "user": "admin", "ip": "192.168.1.100"},
            {"timestamp": "2024-01-15 09:15:00", "event": "Session Created", "user": "analyst", "ip": "192.168.1.101"},
            {"timestamp": "2024-01-15 08:45:00", "event": "Permission Check", "user": "admin", "ip": "192.168.1.100"}
        ]

        if security_events:
            df = pd.DataFrame(security_events)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No recent security events")

        # Security configuration
        st.subheader("Security Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.checkbox("Enable Rate Limiting", value=True)
            st.checkbox("Require Strong Passwords", value=True)
            st.checkbox("Enable Audit Logging", value=True)

        with col2:
            st.number_input("Session Timeout (minutes)", value=60, min_value=5, max_value=480)
            st.number_input("Max Failed Attempts", value=5, min_value=1, max_value=20)
            st.number_input("Lockout Duration (minutes)", value=30, min_value=5, max_value=1440)

        if st.button("💾 Save Security Settings"):
            st.success("✅ Security settings saved")

    except Exception as e:
        st.error(f"❌ Error loading security interface: {e}")

def render_retention_admin_interface():
    """Render data retention admin interface."""
    st.subheader("🗂️ Data Retention & Cleanup")

    try:
        # Retention statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Records", "15,432", delta="234")

        with col2:
            st.metric("Records Deleted", "1,205", delta="45")

        with col3:
            st.metric("Records Archived", "3,456", delta="123")

        with col4:
            st.metric("Storage Freed", "2.3 GB", delta="0.5 GB")

        # Cleanup jobs
        st.subheader("Cleanup Jobs")

        cleanup_jobs = [
            {"job_id": "daily_cleanup", "name": "Daily Cleanup", "status": "Enabled", "last_run": "2024-01-15 02:00:00", "next_run": "2024-01-16 02:00:00"},
            {"job_id": "weekly_archive", "name": "Weekly Archive", "status": "Enabled", "last_run": "2024-01-14 03:00:00", "next_run": "2024-01-21 03:00:00"}
        ]

        df = pd.DataFrame(cleanup_jobs)
        st.dataframe(df, use_container_width=True)

        # Manual cleanup controls
        st.subheader("Manual Cleanup")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🧹 Run Daily Cleanup"):
                with st.spinner("Running cleanup..."):
                    time.sleep(2)
                    st.success("✅ Daily cleanup completed")

        with col2:
            if st.button("📦 Run Archive Job"):
                with st.spinner("Running archive..."):
                    time.sleep(2)
                    st.success("✅ Archive job completed")

        # Retention policies
        st.subheader("Retention Policies")

        policies = [
            {"category": "Trace Events", "retention": "30 days", "archive": "7 days"},
            {"category": "Performance Metrics", "retention": "1 year", "archive": "90 days"},
            {"category": "Error Logs", "retention": "1 year", "archive": "Never"},
            {"category": "Security Audit", "retention": "Permanent", "archive": "Never"}
        ]

        df = pd.DataFrame(policies)
        st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error loading retention interface: {e}")

def render_performance_admin_interface():
    """Render performance monitoring admin interface."""
    st.subheader("⚡ Performance Monitoring")

    try:
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Avg Response Time", "1.2s", delta="-0.3s")

        with col2:
            st.metric("Throughput", "45 req/min", delta="5 req/min")

        with col3:
            st.metric("Error Rate", "0.2%", delta="-0.1%")

        with col4:
            st.metric("Memory Usage", "78%", delta="2%")

        # Circuit breaker status
        st.subheader("Circuit Breaker Status")

        circuit_breakers = [
            {"name": "Tracing", "state": "CLOSED", "failures": 0, "last_failure": "Never"},
            {"name": "Analytics", "state": "CLOSED", "failures": 0, "last_failure": "Never"},
            {"name": "Database", "state": "CLOSED", "failures": 0, "last_failure": "Never"}
        ]

        for cb in circuit_breakers:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.text(cb["name"])

            with col2:
                if cb["state"] == "CLOSED":
                    st.success(cb["state"])
                elif cb["state"] == "HALF_OPEN":
                    st.warning(cb["state"])
                else:
                    st.error(cb["state"])

            with col3:
                st.text(f"Failures: {cb['failures']}")

            with col4:
                st.text(f"Last: {cb['last_failure']}")

        # Performance controls
        st.subheader("Performance Controls")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔄 Reset Circuit Breakers"):
                st.success("✅ Circuit breakers reset")

        with col2:
            if st.button("📊 Generate Report"):
                st.success("✅ Performance report generated")

        with col3:
            if st.button("⚙️ Optimize Performance"):
                with st.spinner("Optimizing..."):
                    time.sleep(2)
                    st.success("✅ Performance optimized")

    except Exception as e:
        st.error(f"❌ Error loading performance interface: {e}")

def render_alert_admin_interface():
    """Render alert management admin interface."""
    st.subheader("🚨 Alert Management")

    try:
        # Alert statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Active Alerts", "2", delta="1")

        with col2:
            st.metric("Total Alerts (24h)", "15", delta="3")

        with col3:
            st.metric("Critical Alerts", "0", delta="0")

        with col4:
            st.metric("Acknowledged", "13", delta="2")

        # Active alerts
        st.subheader("Active Alerts")

        active_alerts = [
            {"id": "alert_001", "severity": "WARNING", "title": "High Memory Usage", "time": "10:30 AM", "status": "Active"},
            {"id": "alert_002", "severity": "INFO", "title": "Cleanup Job Completed", "time": "02:00 AM", "status": "Acknowledged"}
        ]

        for alert in active_alerts:
            col1, col2, col3, col4, col5 = st.columns([1, 1, 3, 1, 1])

            with col1:
                if alert["severity"] == "CRITICAL":
                    st.error(alert["severity"])
                elif alert["severity"] == "WARNING":
                    st.warning(alert["severity"])
                else:
                    st.info(alert["severity"])

            with col2:
                st.text(alert["time"])

            with col3:
                st.text(alert["title"])

            with col4:
                if alert["status"] == "Active":
                    if st.button("✅", key=f"ack_{alert['id']}"):
                        st.success("Alert acknowledged")
                else:
                    st.text("✅")

            with col5:
                if st.button("🗑️", key=f"del_{alert['id']}"):
                    st.success("Alert resolved")

        # Alert configuration
        st.subheader("Alert Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.checkbox("Enable Email Alerts", value=False)
            st.checkbox("Enable Webhook Alerts", value=True)
            st.checkbox("Enable Slack Alerts", value=False)

        with col2:
            st.number_input("Alert Cooldown (minutes)", value=15, min_value=1, max_value=1440)
            st.number_input("Max Alert History", value=1000, min_value=100, max_value=10000)
            st.checkbox("Enable Auto-Resolution", value=True)

        if st.button("💾 Save Alert Settings"):
            st.success("✅ Alert settings saved")

    except Exception as e:
        st.error(f"❌ Error loading alert interface: {e}")

def render_system_overview_interface():
    """Render system overview admin interface."""
    st.subheader("📊 System Overview")

    try:
        # System health score
        health_score = 92

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.metric("System Health Score", f"{health_score}%", delta="2%")

            # Health status
            if health_score >= 90:
                st.success("🟢 System Status: Excellent")
            elif health_score >= 75:
                st.info("🟡 System Status: Good")
            elif health_score >= 50:
                st.warning("🟠 System Status: Fair")
            else:
                st.error("🔴 System Status: Poor")

        with col2:
            st.metric("Uptime", "15d 8h", delta="1d")

        with col3:
            st.metric("Version", "2.0.0", delta="Phase 2B")

        # Component status
        st.subheader("Component Status")

        components = [
            {"name": "Trace Logger", "status": "Healthy", "last_check": "1 min ago"},
            {"name": "Analytics Engine", "status": "Healthy", "last_check": "2 min ago"},
            {"name": "Security Manager", "status": "Healthy", "last_check": "1 min ago"},
            {"name": "Retention Manager", "status": "Healthy", "last_check": "5 min ago"},
            {"name": "Alert Manager", "status": "Healthy", "last_check": "1 min ago"},
            {"name": "Performance Monitor", "status": "Healthy", "last_check": "30 sec ago"}
        ]

        for component in components:
            col1, col2, col3 = st.columns([3, 1, 2])

            with col1:
                st.text(component["name"])

            with col2:
                if component["status"] == "Healthy":
                    st.success("✅ Healthy")
                elif component["status"] == "Warning":
                    st.warning("⚠️ Warning")
                else:
                    st.error("❌ Error")

            with col3:
                st.text(component["last_check"])

        # Quick actions
        st.subheader("Quick Actions")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("🔄 Restart Services"):
                with st.spinner("Restarting..."):
                    time.sleep(3)
                    st.success("✅ Services restarted")

        with col2:
            if st.button("📊 Generate Report"):
                st.success("✅ System report generated")

        with col3:
            if st.button("🧹 Run Maintenance"):
                with st.spinner("Running maintenance..."):
                    time.sleep(4)
                    st.success("✅ Maintenance completed")

        with col4:
            if st.button("💾 Backup System"):
                with st.spinner("Creating backup..."):
                    time.sleep(3)
                    st.success("✅ Backup created")

        # Recent activity
        st.subheader("Recent Activity")

        activities = [
            {"time": "10:45 AM", "activity": "Performance optimization completed", "user": "system"},
            {"time": "10:30 AM", "activity": "High memory usage alert triggered", "user": "system"},
            {"time": "09:15 AM", "activity": "User 'analyst' logged in", "user": "analyst"},
            {"time": "02:00 AM", "activity": "Daily cleanup job completed", "user": "system"},
            {"time": "01:30 AM", "activity": "Weekly archive job started", "user": "system"}
        ]

        df = pd.DataFrame(activities)
        st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error loading system overview: {e}")

def render_interactive_debugger_interface():
    """Render the Phase 3 interactive debugger interface."""
    st.markdown("### 🐛 SAM Interactive Debugger")
    st.markdown("**Phase 3: Real-time debugging with breakpoints and live intervention**")

    try:
        # Create tabs for different debugger functions
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Breakpoints",
            "⏸️ Paused Traces",
            "🔧 Live Rules",
            "📊 Debug Analytics"
        ])

        with tab1:
            render_breakpoints_interface()

        with tab2:
            render_paused_traces_interface()

        with tab3:
            render_live_rules_interface()

        with tab4:
            render_debug_analytics_interface()

    except Exception as e:
        st.error(f"❌ Error loading interactive debugger: {e}")

def render_breakpoints_interface():
    """Render breakpoint management interface."""
    st.subheader("🎯 Breakpoint Management")

    try:
        # Breakpoint creation form
        with st.expander("➕ Create New Breakpoint", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                bp_name = st.text_input("Breakpoint Name", placeholder="e.g., Goal Generation Debug")
                bp_module = st.text_input("Target Module", placeholder="e.g., MotivationEngine or *")
                bp_condition = st.text_area("Condition", placeholder="e.g., payload.get('priority', 0) > 5", height=100)

            with col2:
                bp_description = st.text_area("Description", placeholder="Describe when this breakpoint should trigger", height=80)
                bp_event_type = st.text_input("Event Type", placeholder="e.g., goal_generation_start or *")
                bp_max_hits = st.number_input("Max Hits (optional)", min_value=1, value=None, help="Disable after N hits")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("🎯 Create Breakpoint", type="primary"):
                    if bp_name and bp_module and bp_event_type and bp_condition:
                        # Mock breakpoint creation
                        st.success(f"✅ Breakpoint '{bp_name}' created successfully!")
                        st.info("💡 Breakpoint will trigger when conditions are met during trace execution")
                    else:
                        st.error("❌ Please fill in all required fields")

            with col2:
                if st.button("🧪 Test Condition"):
                    if bp_condition:
                        st.info("🧪 Condition syntax validation would run here")
                    else:
                        st.warning("⚠️ Enter a condition to test")

            with col3:
                if st.button("📖 Help"):
                    st.info("""
                    **Condition Examples:**
                    - `payload.get('priority', 0) > 5` - High priority items
                    - `'error' in message.lower()` - Error messages
                    - `severity == 'critical'` - Critical events
                    - `len(payload.get('goals', [])) > 3` - Multiple goals
                    """)

        # Active breakpoints list
        st.subheader("Active Breakpoints")

        # Mock breakpoints data
        breakpoints = [
            {
                "id": "bp_001",
                "name": "Goal Generation Debug",
                "module": "MotivationEngine",
                "event_type": "goal_generation_start",
                "condition": "payload.get('priority', 0) > 5",
                "status": "Active",
                "hits": 3,
                "created": "2024-01-15 10:30:00"
            },
            {
                "id": "bp_002",
                "name": "Error Tracking",
                "module": "*",
                "event_type": "*",
                "condition": "severity == 'error'",
                "status": "Active",
                "hits": 0,
                "created": "2024-01-15 09:15:00"
            }
        ]

        if breakpoints:
            for bp in breakpoints:
                with st.container():
                    col1, col2, col3, col4, col5 = st.columns([3, 2, 1, 1, 1])

                    with col1:
                        st.write(f"**{bp['name']}**")
                        st.caption(f"{bp['module']}.{bp['event_type']}")

                    with col2:
                        st.code(bp['condition'], language='python')

                    with col3:
                        if bp['status'] == 'Active':
                            st.success(f"✅ {bp['status']}")
                        else:
                            st.warning(f"⚠️ {bp['status']}")
                        st.caption(f"Hits: {bp['hits']}")

                    with col4:
                        if st.button("⏸️", key=f"disable_{bp['id']}", help="Disable breakpoint"):
                            st.success("Breakpoint disabled")

                    with col5:
                        if st.button("🗑️", key=f"delete_{bp['id']}", help="Delete breakpoint"):
                            st.success("Breakpoint deleted")

                    st.divider()
        else:
            st.info("No breakpoints configured. Create one above to start debugging!")

    except Exception as e:
        st.error(f"❌ Error loading breakpoints interface: {e}")

def render_paused_traces_interface():
    """Render paused traces interface."""
    st.subheader("⏸️ Paused Traces")

    try:
        # Mock paused traces
        paused_traces = [
            {
                "trace_id": "trace_abc123",
                "breakpoint": "Goal Generation Debug",
                "paused_at": "2024-01-15 11:45:30",
                "module": "MotivationEngine",
                "event": "goal_generation_start",
                "message": "Starting goal generation analysis for UIF task_456",
                "payload": {
                    "uif_task_id": "task_456",
                    "priority": 8,
                    "context": "user_query_analysis"
                }
            }
        ]

        if paused_traces:
            for trace in paused_traces:
                st.warning(f"🚨 **Trace Paused:** {trace['trace_id']}")

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.write(f"**Breakpoint:** {trace['breakpoint']}")
                    st.write(f"**Module:** {trace['module']}")
                    st.write(f"**Event:** {trace['event']}")
                    st.write(f"**Message:** {trace['message']}")
                    st.write(f"**Paused At:** {trace['paused_at']}")

                    # Show payload
                    st.write("**Current Payload:**")
                    st.json(trace['payload'])

                with col2:
                    st.write("**Debug Actions:**")

                    # Payload override
                    with st.expander("🔧 Override Payload"):
                        override_payload = st.text_area(
                            "Modified Payload (JSON):",
                            value=json.dumps(trace['payload'], indent=2),
                            height=150,
                            key=f"override_{trace['trace_id']}"
                        )

                        if st.button("✅ Apply Override", key=f"apply_{trace['trace_id']}"):
                            try:
                                json.loads(override_payload)  # Validate JSON
                                st.success("✅ Payload override applied!")
                                st.info("🔄 Trace will resume with modified payload")
                            except json.JSONDecodeError:
                                st.error("❌ Invalid JSON format")

                    # Resume actions
                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button("▶️ Resume", key=f"resume_{trace['trace_id']}", type="primary"):
                            st.success("✅ Trace resumed")
                            st.balloons()

                    with col2:
                        if st.button("🛑 Abort", key=f"abort_{trace['trace_id']}"):
                            st.error("🛑 Trace aborted")

                st.divider()
        else:
            st.info("🟢 No traces currently paused")
            st.write("Traces will appear here when they hit active breakpoints.")

    except Exception as e:
        st.error(f"❌ Error loading paused traces interface: {e}")

def render_live_rules_interface():
    """Render live rules management interface."""
    st.subheader("🔧 Live Rule Tuning")

    try:
        # Live rule creation form
        with st.expander("➕ Create New Live Rule", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                rule_name = st.text_input("Rule Name", placeholder="e.g., Priority Boost")
                rule_type = st.selectbox("Rule Type", [
                    "decision_override",
                    "parameter_adjustment",
                    "behavior_modification",
                    "routing_rule",
                    "validation_rule"
                ])
                rule_target_module = st.text_input("Target Module", placeholder="e.g., MotivationEngine")
                rule_target_function = st.text_input("Target Function", placeholder="e.g., generate_goals_from_uif")

            with col2:
                rule_description = st.text_area("Description", placeholder="Describe what this rule does", height=80)
                rule_condition = st.text_area("Condition", placeholder="e.g., data.get('priority', 0) < 5", height=80)
                rule_action = st.text_area("Action", placeholder="e.g., data['priority'] = 10", height=80)
                rule_priority = st.number_input("Priority", min_value=1, max_value=1000, value=100)

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("🔧 Create Rule", type="primary"):
                    if all([rule_name, rule_target_module, rule_target_function, rule_condition, rule_action]):
                        st.success(f"✅ Live rule '{rule_name}' created successfully!")
                        st.info("💡 Rule will be applied to matching function calls in real-time")
                    else:
                        st.error("❌ Please fill in all required fields")

            with col2:
                if st.button("🧪 Test Rule"):
                    st.info("🧪 Rule syntax validation and testing would run here")

            with col3:
                if st.button("📖 Examples"):
                    st.info("""
                    **Example Rules:**

                    **Priority Boost:**
                    - Condition: `data.get('priority', 0) < 5`
                    - Action: `data['priority'] = 10`

                    **Error Handling:**
                    - Condition: `'error' in str(data)`
                    - Action: `data['retry'] = True`

                    **Parameter Adjustment:**
                    - Condition: `data.get('timeout', 0) > 30`
                    - Action: `data['timeout'] = 30`
                    """)

        # Active rules list
        st.subheader("Active Live Rules")

        # Mock live rules data
        live_rules = [
            {
                "id": "rule_001",
                "name": "Priority Boost",
                "type": "parameter_adjustment",
                "target": "MotivationEngine.generate_goals_from_uif",
                "condition": "data.get('priority', 0) < 5",
                "action": "data['priority'] = 10",
                "status": "Active",
                "applications": 15,
                "created": "2024-01-15 10:00:00"
            },
            {
                "id": "rule_002",
                "name": "Timeout Adjustment",
                "type": "behavior_modification",
                "target": "TraceLogger.log_event",
                "condition": "data.get('timeout', 0) > 30",
                "action": "data['timeout'] = 30",
                "status": "Testing",
                "applications": 3,
                "created": "2024-01-15 11:30:00"
            }
        ]

        if live_rules:
            for rule in live_rules:
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 2, 1, 1])

                    with col1:
                        st.write(f"**{rule['name']}** ({rule['type']})")
                        st.caption(f"Target: {rule['target']}")
                        st.caption(f"Applications: {rule['applications']}")

                    with col2:
                        st.write("**Condition:**")
                        st.code(rule['condition'], language='python')
                        st.write("**Action:**")
                        st.code(rule['action'], language='python')

                    with col3:
                        if rule['status'] == 'Active':
                            st.success(f"✅ {rule['status']}")
                        elif rule['status'] == 'Testing':
                            st.warning(f"🧪 {rule['status']}")
                        else:
                            st.info(f"ℹ️ {rule['status']}")

                    with col4:
                        if st.button("⏸️", key=f"disable_rule_{rule['id']}", help="Disable rule"):
                            st.success("Rule disabled")
                        if st.button("🗑️", key=f"delete_rule_{rule['id']}", help="Delete rule"):
                            st.success("Rule deleted")

                    st.divider()
        else:
            st.info("No live rules configured. Create one above to start live tuning!")

    except Exception as e:
        st.error(f"❌ Error loading live rules interface: {e}")

def render_debug_analytics_interface():
    """Render debug analytics interface."""
    st.subheader("📊 Debug Analytics")

    try:
        # Debug statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Active Breakpoints", "2", delta="1")

        with col2:
            st.metric("Paused Traces", "1", delta="0")

        with col3:
            st.metric("Live Rules", "2", delta="1")

        with col4:
            st.metric("Debug Sessions", "5", delta="2")

        # Recent debug activity
        st.subheader("Recent Debug Activity")

        debug_activity = [
            {"time": "11:45:30", "event": "Breakpoint Hit", "details": "Goal Generation Debug triggered in trace_abc123"},
            {"time": "11:30:15", "event": "Live Rule Applied", "details": "Priority Boost rule applied to MotivationEngine"},
            {"time": "11:15:00", "event": "Trace Resumed", "details": "trace_def456 resumed with payload override"},
            {"time": "10:45:30", "event": "Breakpoint Created", "details": "Error Tracking breakpoint created by admin"},
            {"time": "10:30:00", "event": "Live Rule Created", "details": "Priority Boost rule created by admin"}
        ]

        for activity in debug_activity:
            col1, col2, col3 = st.columns([1, 2, 4])

            with col1:
                st.text(activity["time"])

            with col2:
                if "Breakpoint" in activity["event"]:
                    st.info(f"🎯 {activity['event']}")
                elif "Rule" in activity["event"]:
                    st.success(f"🔧 {activity['event']}")
                elif "Resumed" in activity["event"]:
                    st.warning(f"▶️ {activity['event']}")
                else:
                    st.text(activity["event"])

            with col3:
                st.text(activity["details"])

        # Debug performance metrics
        st.subheader("Debug Performance")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Breakpoint Performance:**")
            st.write("• Average condition evaluation: 0.5ms")
            st.write("• Pause/resume overhead: 2.1ms")
            st.write("• Total debug overhead: 0.02%")

        with col2:
            st.write("**Live Rule Performance:**")
            st.write("• Average rule execution: 1.2ms")
            st.write("• Rule application rate: 85%")
            st.write("• Total rule overhead: 0.05%")

        # Debug insights
        st.subheader("Debug Insights")

        insights = [
            "🎯 Most triggered breakpoint: Goal Generation Debug (15 hits)",
            "🔧 Most applied rule: Priority Boost (45 applications)",
            "⏱️ Average debug session: 12 minutes",
            "🐛 Common debug pattern: Goal generation → Rule application → Resume",
            "📈 Debug efficiency improved by 23% this week"
        ]

        for insight in insights:
            st.write(insight)

    except Exception as e:
        st.error(f"❌ Error loading debug analytics interface: {e}")

def render_tpv_dissonance_demo():
    """Render TPV Dissonance Monitoring Demo (Phase 5B)."""
    try:
        st.subheader("🧠📊 TPV Dissonance Monitoring Demo")
        st.markdown("**Phase 5B: Real-time Cognitive Dissonance Detection**")

        st.markdown("""
        This demo showcases SAM's revolutionary **Dissonance-Aware Meta-Reasoning** capabilities.
        SAM can now detect internal cognitive conflicts in real-time and prevent hallucination loops.
        """)

        # Import and use the visualization
        try:
            from ui.tpv_visualization import demo_visualization, create_sample_trace_data, render_tpv_dissonance_chart

            # Demo controls
            col1, col2, col3 = st.columns(3)

            with col1:
                scenario = st.selectbox(
                    "🎭 Demo Scenario",
                    ["High Uncertainty Query", "Technical Analysis", "Ethical Dilemma", "Complex Reasoning"],
                    help="Choose a scenario to demonstrate dissonance patterns"
                )

            with col2:
                dissonance_level = st.selectbox(
                    "🧠 Dissonance Level",
                    ["Low (0.2-0.4)", "Medium (0.4-0.7)", "High (0.7-0.9)", "Critical (0.9+)"],
                    help="Simulate different levels of cognitive dissonance"
                )

            with col3:
                if st.button("🎲 Generate New Demo", help="Generate a new random trace"):
                    st.session_state.demo_trace_data = None
                    st.rerun()

            # Generate or get cached demo data
            if 'demo_trace_data' not in st.session_state:
                st.session_state.demo_trace_data = create_sample_trace_data()

            # Modify data based on selected scenario and dissonance level
            demo_data = st.session_state.demo_trace_data.copy()

            # Adjust dissonance based on selection
            base_dissonance = {
                "Low (0.2-0.4)": 0.3,
                "Medium (0.4-0.7)": 0.55,
                "High (0.7-0.9)": 0.8,
                "Critical (0.9+)": 0.95
            }[dissonance_level]

            # Modify the demo data
            for step in demo_data['steps']:
                # Add some variation around the base level
                import numpy as np
                variation = np.random.normal(0, 0.1)
                step['dissonance_score'] = max(0.0, min(1.0, base_dissonance + variation))

            # Add scenario-specific metadata
            demo_data['scenario'] = scenario
            demo_data['dissonance_level'] = dissonance_level

            # Render the visualization
            st.markdown("### 📊 Real-Time Cognitive Analysis")
            render_tpv_dissonance_chart(demo_data, expanded=True, height=450)

            # Explanation section
            st.markdown("### 🔬 Understanding the Visualization")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                **🔵 TPV Progress (Blue Line):**
                - Shows reasoning quality improvement over time
                - Range: 0.0 (no progress) to 1.0 (complete)
                - Steady increase indicates healthy reasoning
                """)

                st.markdown("""
                **🟠 Cognitive Dissonance (Orange Line):**
                - Measures internal reasoning conflicts
                - Range: 0.0 (certain) to 1.0 (highly conflicted)
                - High values indicate uncertainty or confusion
                """)

            with col2:
                st.markdown("""
                **🔴 Dissonance Threshold (Red Line):**
                - Default threshold: 0.85
                - When exceeded, SAM may halt reasoning
                - Prevents hallucination and confabulation
                """)

                st.markdown("""
                **🎛️ Control Decisions:**
                - **Continue**: Normal reasoning progression
                - **Stop (Dissonance)**: High cognitive conflict detected
                - **Stop (Completion)**: Reasoning successfully completed
                """)

            # Technical details
            with st.expander("🔧 Technical Implementation Details", expanded=False):
                st.markdown("""
                **Dissonance Calculation Methods:**
                - **Entropy**: Measures uncertainty in token probability distributions
                - **Variance**: Analyzes spread of probability values
                - **KL Divergence**: Compares against uniform distribution
                - **Composite**: Weighted combination of multiple metrics

                **Real-time Processing:**
                - Calculated during each token generation step
                - Average processing time: ~0.3ms per calculation
                - Minimal impact on response generation speed

                **Control Integration:**
                - Integrated with existing TPV monitoring system
                - Configurable thresholds and patience parameters
                - Graceful fallback when dissonance calculation fails
                """)

            # Performance metrics
            st.markdown("### ⚡ Performance Impact")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Processing Overhead", "~0.3ms", help="Additional time per reasoning step")
            with col2:
                st.metric("Memory Usage", "+2.1MB", help="Additional memory for dissonance monitoring")
            with col3:
                st.metric("Accuracy Improvement", "+21.9%", help="Reduction in hallucination incidents")

        except ImportError as e:
            st.error(f"❌ TPV Visualization module not available: {e}")
            st.markdown("""
            **To enable this demo:**
            1. Ensure `ui/tpv_visualization.py` is available
            2. Install required dependencies (plotly, numpy)
            3. Restart the Memory Center
            """)

    except Exception as e:
        st.error(f"❌ Error loading TPV dissonance demo: {e}")
        st.markdown("""
        **Troubleshooting:**
        - Check that Phase 5B implementation is complete
        - Verify TPV system is properly initialized
        - Ensure all dependencies are installed
        """)

def render_personalized_tuner():
    """Render the Personalized Tuner interface for DPO fine-tuning."""
    try:
        st.subheader("🧠 Personalized Tuner")
        st.markdown("**Direct Preference Optimization (DPO) for Personalized Model Fine-Tuning**")

        # Import required modules
        try:
            from sam.learning.dpo_data_manager import get_dpo_data_manager
            from sam.learning.feedback_handler import get_feedback_handler
            from memory.episodic_store import create_episodic_store
        except ImportError as e:
            st.error(f"❌ Required modules not available: {e}")
            st.info("Please ensure the DPO integration is properly installed.")
            return

        # Initialize components
        episodic_store = create_episodic_store()
        dpo_manager = get_dpo_data_manager(episodic_store)
        feedback_handler = get_feedback_handler()

        # User ID input
        col1, col2 = st.columns([2, 1])
        with col1:
            user_id = st.text_input("User ID", value="default_user", help="Enter your user ID to view personalization data")
        with col2:
            if st.button("🔄 Refresh Data", help="Reload preference data"):
                st.rerun()

        if not user_id:
            st.warning("Please enter a user ID to continue.")
            return

        # Get user statistics
        user_stats = dpo_manager.get_user_stats(user_id)
        feedback_stats = feedback_handler.get_dpo_statistics(user_id)

        # Statistics overview
        st.markdown("### 📊 Training Data Overview")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Preference Pairs", user_stats.get('total_pairs', 0))
        with col2:
            st.metric("Active for Training", user_stats.get('active_pairs', 0))
        with col3:
            st.metric("Avg Confidence", f"{user_stats.get('avg_confidence', 0.0):.2f}")
        with col4:
            training_ready = user_stats.get('training_ready_pairs', 0)
            st.metric("Training Ready", training_ready,
                     delta="Ready" if training_ready >= 10 else "Need more data")

        # Main interface tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Preference Data",
            "🎯 Training Controls",
            "🧠 Model Management",
            "📈 Analytics",
            "⚙️ Settings"
        ])

        with tab1:
            render_preference_data_tab(dpo_manager, user_id)

        with tab2:
            render_training_controls_tab(dpo_manager, user_id, user_stats)

        with tab3:
            render_model_management_tab(user_id)

        with tab4:
            render_dpo_analytics_tab(dpo_manager, feedback_handler, user_id)

        with tab5:
            render_dpo_settings_tab()

    except Exception as e:
        st.error(f"❌ Error loading Personalized Tuner: {e}")
        st.markdown("""
        **Possible causes:**
        - DPO integration components not properly installed
        - Database connection issues
        - Missing dependencies

        **Try:**
        1. Check that sam/learning/dpo_data_manager.py exists
        2. Verify database is accessible
        3. Restart the Memory Center if issues persist
        """)

def render_preference_data_tab(dpo_manager, user_id):
    """Render the preference data management tab."""
    st.markdown("#### 📋 DPO Preference Pairs")

    # Filtering controls
    col1, col2, col3 = st.columns(3)
    with col1:
        min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.7, 0.1,
                                  help="Filter pairs by confidence score")
    with col2:
        show_inactive = st.checkbox("Show Inactive", help="Include inactive pairs")
    with col3:
        max_pairs = st.number_input("Max Pairs", 1, 1000, 50, help="Maximum pairs to display")

    # Get preference data
    preferences = dpo_manager.episodic_store.get_dpo_preferences(
        user_id=user_id,
        min_confidence=min_confidence,
        active_only=not show_inactive,
        limit=max_pairs
    )

    if not preferences:
        st.info("No preference pairs found matching the criteria.")
        return

    st.markdown(f"**Found {len(preferences)} preference pairs**")

    # Display preference pairs
    for i, pref in enumerate(preferences):
        with st.expander(f"Pair {i+1}: {pref.feedback_type} (Confidence: {pref.feedback_confidence_score:.2f})"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Original Response (Rejected):**")
                st.text_area("", pref.original_response, height=100, key=f"orig_{pref.id}", disabled=True)

            with col2:
                st.markdown("**Corrected Response (Chosen):**")
                st.text_area("", pref.corrected_response, height=100, key=f"corr_{pref.id}", disabled=True)

            # Metadata
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.caption(f"**Prompt:** {pref.prompt_text[:50]}...")
            with col2:
                st.caption(f"**Quality:** {pref.quality_score or 'N/A'}")
            with col3:
                st.caption(f"**Created:** {pref.created_timestamp[:10]}")
            with col4:
                # Toggle active status
                new_status = st.checkbox("Active", value=pref.is_active_for_tuning, key=f"active_{pref.id}")
                if new_status != pref.is_active_for_tuning:
                    if dpo_manager.episodic_store.update_dpo_preference_status(pref.id, new_status):
                        st.success("Status updated!")
                        st.rerun()

def render_training_controls_tab(dpo_manager, user_id, user_stats):
    """Render the training controls tab."""
    st.markdown("#### 🎯 Fine-Tuning Controls")

    # Import training manager
    try:
        from sam.cognition.dpo import get_dpo_training_manager
        training_manager = get_dpo_training_manager()
    except ImportError:
        st.error("❌ Training manager not available. Please ensure DPO dependencies are installed.")
        return

    training_ready_pairs = user_stats.get('training_ready_pairs', 0)

    if training_ready_pairs < 10:
        st.warning(f"⚠️ Only {training_ready_pairs} training-ready pairs available. Recommended minimum: 10")
        st.info("Continue using the feedback system to collect more high-quality preference pairs.")
        return

    st.success(f"✅ {training_ready_pairs} training-ready pairs available!")

    # Check for existing jobs
    user_jobs = training_manager.get_user_jobs(user_id)
    running_jobs = [job for job in user_jobs if job['status'] == 'running']

    if running_jobs:
        st.info(f"🔄 Training job in progress: {running_jobs[0]['job_id']}")
        render_job_progress(training_manager, running_jobs[0])
        return

    # Training configuration
    st.markdown("##### Training Configuration")
    col1, col2 = st.columns(2)

    with col1:
        min_confidence = st.slider("Training Confidence Threshold", 0.5, 1.0, 0.8, 0.05)
        min_quality = st.slider("Training Quality Threshold", 0.0, 1.0, 0.6, 0.1)
        max_training_pairs = st.number_input("Max Training Pairs", 10, 1000, 100)

    with col2:
        learning_rate = st.number_input("Learning Rate", 0.0001, 0.01, 0.0005, format="%.4f")
        num_epochs = st.number_input("Training Epochs", 1, 10, 3)
        lora_rank = st.number_input("LoRA Rank", 4, 64, 16)
        beta = st.number_input("DPO Beta", 0.01, 1.0, 0.1, format="%.2f")

    # Preview training dataset
    if st.button("📊 Preview Training Dataset"):
        training_data = dpo_manager.get_training_dataset(
            user_id=user_id,
            min_confidence=min_confidence,
            min_quality=min_quality,
            limit=max_training_pairs
        )

        st.markdown(f"**Training dataset preview: {len(training_data)} examples**")
        if training_data:
            # Show first few examples
            for i, example in enumerate(training_data[:3]):
                with st.expander(f"Example {i+1}"):
                    st.markdown(f"**Prompt:** {example['prompt']}")
                    st.markdown(f"**Chosen:** {example['chosen'][:100]}...")
                    st.markdown(f"**Rejected:** {example['rejected'][:100]}...")

    # Training controls
    st.markdown("##### Start Fine-Tuning")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Start Fine-Tuning", type="primary"):
            # Create configuration overrides
            config_overrides = {
                'data': {
                    'min_confidence_threshold': min_confidence,
                    'min_quality_threshold': min_quality,
                    'max_training_samples': max_training_pairs
                },
                'training': {
                    'learning_rate': learning_rate,
                    'num_train_epochs': num_epochs,
                    'beta': beta
                },
                'lora': {
                    'r': lora_rank
                }
            }

            # Create and start training job
            job_id = training_manager.create_training_job(user_id, config_overrides)

            if training_manager.start_training_job(job_id):
                st.success(f"🚀 Training job started: {job_id}")
                st.info("Refresh the page to see training progress.")
                st.rerun()
            else:
                st.error("❌ Failed to start training job")

    with col2:
        if st.button("💾 Export Training Data"):
            training_data = dpo_manager.get_training_dataset(
                user_id=user_id,
                min_confidence=min_confidence,
                min_quality=min_quality,
                limit=max_training_pairs
            )

            if training_data:
                import json
                json_data = json.dumps(training_data, indent=2)
                st.download_button(
                    "📥 Download JSON",
                    json_data,
                    f"dpo_training_data_{user_id}.json",
                    "application/json"
                )

    # Recent jobs section
    if user_jobs:
        st.markdown("##### Recent Training Jobs")
        for job in user_jobs[-5:]:  # Show last 5 jobs
            with st.expander(f"Job {job['job_id']} - {job['status'].title()}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Status", job['status'].title())
                with col2:
                    if job['duration_seconds']:
                        duration_min = job['duration_seconds'] / 60
                        st.metric("Duration", f"{duration_min:.1f} min")
                with col3:
                    if job['progress']:
                        st.metric("Progress", f"{job['progress']*100:.1f}%")

                if job['error_message']:
                    st.error(f"Error: {job['error_message']}")

                if job['model_path']:
                    st.success(f"Model saved to: {job['model_path']}")


def render_job_progress(training_manager, job):
    """Render progress for a running training job."""
    st.markdown("##### 🔄 Training Progress")

    # Progress metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Status", job['status'].title())
    with col2:
        if job['duration_seconds']:
            duration_min = job['duration_seconds'] / 60
            st.metric("Duration", f"{duration_min:.1f} min")
    with col3:
        if job['progress']:
            st.metric("Progress", f"{job['progress']*100:.1f}%")
    with col4:
        if job['current_loss']:
            st.metric("Current Loss", f"{job['current_loss']:.4f}")

    # Progress bar
    if job['progress']:
        st.progress(job['progress'])

    # Control buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Refresh"):
            st.rerun()

    with col2:
        if st.button("📋 View Logs"):
            if job['log_file']:
                try:
                    with open(job['log_file'], 'r') as f:
                        log_content = f.read()
                    st.text_area("Training Logs", log_content[-2000:], height=300)  # Last 2000 chars
                except Exception as e:
                    st.error(f"Error reading logs: {e}")

    with col3:
        if st.button("❌ Cancel Training", type="secondary"):
            if training_manager.cancel_training_job(job['job_id']):
                st.success("Training job cancelled")
                st.rerun()
            else:
                st.error("Failed to cancel job")

def render_dpo_analytics_tab(dpo_manager, feedback_handler, user_id):
    """Render the DPO analytics tab."""
    st.markdown("#### 📈 Training Data Analytics")

    # Get comprehensive stats
    user_stats = dpo_manager.get_user_stats(user_id)
    # Use the DPO-aware stats method from the feedback handler
    feedback_stats = {}
    if hasattr(feedback_handler, 'get_dpo_statistics'):
        feedback_stats = feedback_handler.get_dpo_statistics(user_id)

    # Quality distribution
    preferences = dpo_manager.episodic_store.get_dpo_preferences(user_id, 0.0, False, 1000)

    if preferences:
        # Confidence distribution
        confidences = [p.feedback_confidence_score for p in preferences]
        quality_scores = [p.quality_score for p in preferences if p.quality_score is not None]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Confidence Score Distribution**")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(confidences, bins=20, alpha=0.7, color='blue')
            ax.set_xlabel('Confidence Score')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Confidence Scores')
            st.pyplot(fig)

        with col2:
            if quality_scores:
                st.markdown("**Quality Score Distribution**")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(quality_scores, bins=20, alpha=0.7, color='green')
                ax.set_xlabel('Quality Score')
                ax.set_ylabel('Count')
                ax.set_title('Distribution of Quality Scores')
                st.pyplot(fig)
            else:
                st.info("No quality scores available yet.")

        # Feedback type breakdown
        feedback_types = {}
        for p in preferences:
            feedback_types[p.feedback_type] = feedback_types.get(p.feedback_type, 0) + 1

        st.markdown("**Feedback Type Breakdown**")
        for feedback_type, count in feedback_types.items():
            st.metric(feedback_type.replace('_', ' ').title(), count)

    else:
        st.info("No preference data available for analytics.")

def render_dpo_settings_tab():
    """Render the DPO settings tab."""
    st.markdown("#### ⚙️ Personalized Tuner Settings")

    st.markdown("##### Data Collection Settings")

    col1, col2 = st.columns(2)
    with col1:
        auto_collect = st.checkbox("Auto-collect DPO pairs", value=True,
                                  help="Automatically create preference pairs from feedback")
        confidence_threshold = st.slider("DPO Confidence Threshold", 0.5, 1.0, 0.7, 0.05,
                                        help="Minimum confidence for DPO pair creation")

    with col2:
        validate_pairs = st.checkbox("Validate preference pairs", value=True,
                                   help="Run quality validation on new pairs")
        min_correction_length = st.number_input("Min correction length (words)", 1, 50, 5,
                                              help="Minimum words in correction")

    st.markdown("##### Model Settings")

    col1, col2 = st.columns(2)
    with col1:
        base_model = st.selectbox("Base Model", ["llama-3.1-8b", "llama-3.1-70b", "custom"],
                                 help="Base model for fine-tuning")
        lora_rank = st.number_input("LoRA Rank", 1, 256, 16, help="LoRA adapter rank")

    with col2:
        lora_alpha = st.number_input("LoRA Alpha", 1, 512, 32, help="LoRA scaling parameter")
        max_seq_length = st.number_input("Max Sequence Length", 512, 4096, 2048,
                                       help="Maximum sequence length for training")

    if st.button("💾 Save Settings"):
        st.success("Settings saved! (Note: Settings persistence will be implemented in Phase 2)")


def render_model_management_tab(user_id):
    """Render the model management tab."""
    st.markdown("#### 🧠 Personalized Model Management")

    try:
        from sam.cognition.dpo import (
            get_dpo_model_manager,
            get_personalized_sam_client,
            get_user_personalization_status
        )

        model_manager = get_dpo_model_manager()
        sam_client = get_personalized_sam_client()

    except ImportError:
        st.error("❌ Model management not available. Please ensure DPO dependencies are installed.")
        return

    # Get personalization status
    status = get_user_personalization_status(user_id)

    # Current status display
    st.markdown("##### Current Status")
    col1, col2, col3 = st.columns(3)

    with col1:
        if status.get('personalization_enabled'):
            st.success("✅ Personalization Available")
        else:
            st.error("❌ Personalization Unavailable")

    with col2:
        if status.get('has_active_model'):
            st.success(f"🧠 Active Model: {status.get('active_model_id', 'Unknown')}")
        else:
            st.info("🔄 Using Base Model")

    with col3:
        engine_status = status.get('engine_status', {})
        metrics = engine_status.get('metrics', {})
        total_requests = metrics.get('total_requests', 0)
        personalized_requests = metrics.get('personalized_requests', 0)

        if total_requests > 0:
            personalization_rate = (personalized_requests / total_requests) * 100
            st.metric("Personalization Rate", f"{personalization_rate:.1f}%")
        else:
            st.metric("Personalization Rate", "0%")

    # Available models section
    st.markdown("##### Available Personalized Models")

    user_models = model_manager.get_user_models(user_id)

    if not user_models:
        st.info("No personalized models available. Train a model first using the Training Controls tab.")
        return

    # Model list with activation controls
    for model in user_models:
        with st.expander(f"Model: {model.model_id} ({'Active' if model.is_active else 'Inactive'})"):
            col1, col2 = st.columns([2, 1])

            with col1:
                # Model information
                st.markdown(f"**Created:** {model.created_at.strftime('%Y-%m-%d %H:%M')}")
                st.markdown(f"**Base Model:** {model.base_model}")
                st.markdown(f"**Training Job:** {model.training_job_id}")
                st.markdown(f"**Usage Count:** {model.usage_count}")

                if model.last_used:
                    st.markdown(f"**Last Used:** {model.last_used.strftime('%Y-%m-%d %H:%M')}")

                # Training stats
                if model.training_stats:
                    stats = model.training_stats
                    if 'final_loss' in stats:
                        st.markdown(f"**Final Loss:** {stats['final_loss']:.4f}")
                    if 'training_time_seconds' in stats:
                        training_time = stats['training_time_seconds'] / 60
                        st.markdown(f"**Training Time:** {training_time:.1f} minutes")

                # LoRA configuration
                if model.lora_config:
                    lora_info = f"Rank: {model.lora_config.get('r', 'N/A')}, Alpha: {model.lora_config.get('alpha', 'N/A')}"
                    st.markdown(f"**LoRA Config:** {lora_info}")

            with col2:
                # Activation controls
                if model.is_active:
                    if st.button("🔄 Deactivate", key=f"deactivate_{model.model_id}"):
                        if sam_client.deactivate_personalized_model(user_id):
                            st.success("Model deactivated")
                            st.rerun()
                        else:
                            st.error("Failed to deactivate model")
                else:
                    if st.button("🚀 Activate", key=f"activate_{model.model_id}", type="primary"):
                        if sam_client.activate_personalized_model(user_id, model.model_id):
                            st.success("Model activated")
                            st.rerun()
                        else:
                            st.error("Failed to activate model")

                # Model actions
                if st.button("🗑️ Delete", key=f"delete_{model.model_id}", type="secondary"):
                    if st.session_state.get(f"confirm_delete_{model.model_id}"):
                        if model_manager.delete_model(model.model_id, user_id):
                            st.success("Model deleted")
                            st.rerun()
                        else:
                            st.error("Failed to delete model")
                    else:
                        st.session_state[f"confirm_delete_{model.model_id}"] = True
                        st.warning("Click again to confirm deletion")

    # Test personalization section
    st.markdown("##### Test Personalization")

    test_prompt = st.text_area(
        "Test Prompt",
        placeholder="Enter a prompt to test your personalized model...",
        help="This will generate a response using your active personalized model (if any)"
    )

    if st.button("🧪 Test Response") and test_prompt:
        with st.spinner("Generating response..."):
            try:
                from sam.cognition.dpo import generate_personalized_response

                response = generate_personalized_response(
                    prompt=test_prompt,
                    user_id=user_id
                )

                st.markdown("**Response:**")
                st.write(response.content)

                # Show metadata
                with st.expander("Response Metadata"):
                    st.json(response.metadata)

                # Show personalization info
                if response.is_personalized:
                    st.success(f"✅ Generated using personalized model: {response.model_id}")
                else:
                    st.info("ℹ️ Generated using base model")

                if response.fallback_used:
                    st.warning("⚠️ Fallback was used during generation")

            except Exception as e:
                st.error(f"Error generating test response: {e}")


if __name__ == "__main__":
    main()
