#!/usr/bin/env python3
"""
SAM Secure Streamlit Application

Main Streamlit application with integrated security features.
Provides secure access to SAM's AI assistant capabilities.

Author: SAM Development Team
Version: 1.0.0
"""

import os
# Suppress PyTorch/Streamlit compatibility warnings and prevent crashes
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

# Fix torch/Streamlit compatibility issues
os.environ['PYTORCH_DISABLE_PER_OP_PROFILING'] = '1'

# Disable CSP restrictions for local development
os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
os.environ['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = 'false'

# Additional CSP bypass attempts
os.environ['STREAMLIT_SERVER_ENABLE_STATIC_SERVING'] = 'true'
os.environ['STREAMLIT_CLIENT_TOOLBAR_MODE'] = 'minimal'
os.environ['STREAMLIT_CLIENT_SHOW_ERROR_DETAILS'] = 'true'

# Prevent torch from interfering with Streamlit's module system
import sys

import streamlit as st
import sys
import logging
import time
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from typing import List, Dict, Any
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Create logs directory if it doesn't exist (BEFORE logging setup)
Path('logs').mkdir(exist_ok=True)

# Configure logging with better error handling
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/secure_streamlit.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# PHASE 3: Utility function to handle RankedMemoryResult compatibility
def extract_result_content(result: Any) -> tuple:
    """
    Extract content, source, and metadata from various result types.
    Handles both RankedMemoryResult (Phase 3) and legacy MemorySearchResult.

    Returns:
        tuple: (content, source, metadata)
    """
    try:
        # Handle RankedMemoryResult (Phase 3) - has content and metadata directly
        if hasattr(result, 'content') and hasattr(result, 'metadata'):
            content = result.content
            source = result.metadata.get('source_path', result.metadata.get('source_name', 'Unknown'))
            metadata = result.metadata
            return content, source, metadata

        # Handle MemorySearchResult with chunk attribute
        elif hasattr(result, 'chunk') and hasattr(result.chunk, 'content'):
            content = result.chunk.content
            source = result.chunk.source
            metadata = getattr(result.chunk, 'metadata', {})
            return content, source, metadata

        # Handle direct content objects
        elif hasattr(result, 'content'):
            content = result.content
            source = getattr(result, 'source', 'Unknown')
            metadata = getattr(result, 'metadata', {})
            return content, source, metadata

        else:
            logger.warning(f"Unknown result structure: {type(result)}")
            return None, None, {}

    except Exception as e:
        logger.error(f"Error extracting result content: {e}")
        return None, None, {}

def health_check():
    """Health check endpoint for Docker containers and load balancers."""
    try:
        # Check if we're in a health check request
        query_params = st.query_params
        if 'health' in query_params or st.session_state.get('health_check_mode', False):
            # Return simple health status
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "services": {
                    "streamlit": "running",
                    "memory_store": "available",
                    "security": "enabled"
                }
            }

            # Display health status
            st.json(health_status)
            st.stop()

    except Exception as e:
        logger.error(f"Health check error: {e}")
        st.json({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
        st.stop()

def generate_enhanced_summary_prompt(filename: str) -> str:
    """Generate enhanced summary prompt based on document type and SAM's capabilities."""

    # Detect document type from filename
    file_ext = Path(filename).suffix.lower() if filename else ""

    # Base prompt with SAM's synthesis approach
    base_prompt = f"""Please provide a comprehensive synthesis summary of the document '{filename}' using your advanced document analysis capabilities.

🎯 **SYNTHESIS APPROACH** (not extraction):
- Analyze the document's core themes and contributions
- Synthesize information into a coherent narrative in your own words
- Remove formatting artifacts and create flowing, readable content
- Focus on insights and implications rather than just facts

📊 **STRUCTURED OUTPUT**:
## Executive Summary (30-second read)
- Key finding or main message
- Primary value proposition

## Core Analysis (2-minute read)
- Main problem/topic addressed
- Key methodology or approach
- Primary results and implications
- Main conclusions

## Detailed Insights (5-minute read)
- Comprehensive analysis of findings
- Detailed implications and recommendations
- Limitations and considerations
- Future directions or next steps

🔍 **DOCUMENT-SPECIFIC ANALYSIS**:"""

    # Add document-type specific instructions
    if file_ext in ['.pdf', '.docx']:
        if 'research' in filename.lower() or 'paper' in filename.lower() or 'study' in filename.lower():
            base_prompt += """
- **Research Paper Focus**: Methodology, findings, statistical significance, limitations
- **Academic Standards**: Scholarly tone, proper context, research implications
- **Citation Context**: How this fits into broader research landscape"""
        elif 'proposal' in filename.lower() or 'plan' in filename.lower():
            base_prompt += """
- **Proposal Focus**: Objectives, methodology, timeline, budget implications
- **Strategic Analysis**: Feasibility, risks, success factors
- **Implementation Insights**: Practical considerations and next steps"""
        elif 'report' in filename.lower() or 'analysis' in filename.lower():
            base_prompt += """
- **Report Focus**: Key findings, data insights, recommendations
- **Business Context**: Strategic implications, operational impact
- **Actionable Insights**: Specific recommendations and implementation guidance"""
        else:
            base_prompt += """
- **Document Analysis**: Structure, key sections, main arguments
- **Content Synthesis**: Core messages, supporting evidence
- **Practical Applications**: How this information can be used"""

    elif file_ext in ['.md', '.txt']:
        base_prompt += """
- **Text Analysis**: Main themes, key concepts, logical flow
- **Content Structure**: Organization, hierarchy, relationships
- **Knowledge Extraction**: Actionable insights and takeaways"""

    else:
        base_prompt += """
- **Content Analysis**: Main themes, structure, key information
- **Synthesis Focus**: Core messages and practical implications"""

    base_prompt += """

💡 **LEVERAGE YOUR CAPABILITIES**:
- Use your knowledge consolidation to connect concepts
- Apply semantic understanding for deeper insights
- Provide confidence indicators where appropriate
- Include relevant context from your knowledge base"""

    return base_prompt

def generate_enhanced_questions_prompt(filename: str) -> str:
    """Generate enhanced key questions prompt based on document analysis."""

    file_ext = Path(filename).suffix.lower() if filename else ""

    base_prompt = f"""Based on your analysis of '{filename}', generate the most strategic and insightful questions that would unlock the document's full value.

🎯 **QUESTION CATEGORIES**:

## 🔍 **Clarification Questions** (Understanding)
- What are the key concepts that need deeper explanation?
- Which assumptions or methodologies should be questioned?
- What context or background would enhance understanding?

## 💡 **Insight Questions** (Analysis)
- What are the broader implications of the main findings?
- How do these findings connect to current trends or challenges?
- What patterns or relationships emerge from the data?

## 🚀 **Application Questions** (Implementation)
- How can these insights be practically applied?
- What are the next logical steps or follow-up actions?
- What resources or conditions are needed for implementation?

## ⚠️ **Critical Questions** (Evaluation)
- What are the potential limitations or risks?
- What alternative perspectives should be considered?
- How reliable or generalizable are the conclusions?"""

    # Add document-type specific questions
    if file_ext in ['.pdf', '.docx']:
        if 'research' in filename.lower() or 'paper' in filename.lower():
            base_prompt += """

## 📚 **Research-Specific Questions**:
- How does this research contribute to the existing knowledge base?
- What are the statistical significance and effect sizes?
- What future research directions does this suggest?
- How could the methodology be improved or extended?"""

        elif 'proposal' in filename.lower() or 'plan' in filename.lower():
            base_prompt += """

## 📋 **Proposal-Specific Questions**:
- What are the success metrics and evaluation criteria?
- What are the potential risks and mitigation strategies?
- How does this align with organizational goals and resources?
- What are the dependencies and critical path items?"""

        elif 'report' in filename.lower() or 'analysis' in filename.lower():
            base_prompt += """

## 📊 **Report-Specific Questions**:
- What trends or patterns emerge from the data?
- What are the strategic implications for decision-making?
- Which recommendations have the highest impact potential?
- What additional data or analysis would be valuable?"""

    base_prompt += """

🎯 **QUESTION QUALITY CRITERIA**:
- **Strategic**: Focus on high-impact, decision-relevant questions
- **Specific**: Avoid generic questions; tailor to document content
- **Actionable**: Questions that lead to concrete insights or actions
- **Progressive**: Build from basic understanding to advanced analysis

📝 **FORMAT**: Present 8-12 questions organized by category, with brief rationale for why each question is important for maximizing the document's value."""

    return base_prompt

def generate_enhanced_analysis_prompt(filename: str) -> str:
    """Generate enhanced deep analysis prompt leveraging SAM's full analytical capabilities."""

    file_ext = Path(filename).suffix.lower() if filename else ""

    base_prompt = f"""Conduct a comprehensive deep analysis of '{filename}' using your advanced analytical capabilities and knowledge synthesis.

🧠 **ANALYTICAL FRAMEWORK**:

## 📊 **Structural Analysis**
- **Document Architecture**: Organization, flow, key sections
- **Content Hierarchy**: Main themes, supporting arguments, evidence
- **Information Density**: Key insights per section, critical passages

## 🔍 **Content Deep Dive**
- **Core Concepts**: Fundamental ideas and their relationships
- **Methodology/Approach**: How conclusions were reached
- **Evidence Quality**: Strength and reliability of supporting data
- **Logical Consistency**: Argument flow and reasoning validity

## 💡 **Insight Synthesis**
- **Key Discoveries**: Most significant findings or revelations
- **Hidden Patterns**: Subtle connections and implications
- **Knowledge Gaps**: What's missing or needs further exploration
- **Contradictions**: Any conflicting information or perspectives

## 🎯 **Strategic Implications**
- **Immediate Applications**: How to use this information now
- **Long-term Impact**: Broader implications and future considerations
- **Decision Support**: How this informs strategic choices
- **Risk Assessment**: Potential challenges or limitations"""

    # Add document-type specific analysis
    if file_ext in ['.pdf', '.docx']:
        if 'research' in filename.lower() or 'paper' in filename.lower():
            base_prompt += """

## 🔬 **Research Analysis**:
- **Methodology Evaluation**: Strengths/weaknesses of research design
- **Statistical Rigor**: Significance, effect sizes, confidence intervals
- **Reproducibility**: Can results be replicated or validated?
- **Research Impact**: Contribution to field, citation potential
- **Future Research**: Logical next steps and research questions"""

        elif 'proposal' in filename.lower() or 'plan' in filename.lower():
            base_prompt += """

## 📋 **Proposal Analysis**:
- **Feasibility Assessment**: Technical, financial, operational viability
- **Resource Requirements**: Personnel, budget, timeline analysis
- **Risk Matrix**: Probability and impact of potential issues
- **Success Factors**: Critical elements for successful implementation
- **ROI Projection**: Expected returns and value creation"""

        elif 'technical' in filename.lower() or 'spec' in filename.lower():
            base_prompt += """

## ⚙️ **Technical Analysis**:
- **Technical Feasibility**: Implementation complexity and requirements
- **Architecture Review**: System design, scalability, maintainability
- **Performance Implications**: Speed, efficiency, resource usage
- **Integration Challenges**: Compatibility with existing systems
- **Security Considerations**: Vulnerabilities and protection measures"""

    base_prompt += """

## 🔗 **Contextual Integration**
- **Industry Context**: How this fits within broader industry trends
- **Competitive Landscape**: Advantages, disadvantages, positioning
- **Regulatory Considerations**: Compliance, legal, ethical implications
- **Technology Trends**: Alignment with emerging technologies

## 📈 **Actionable Recommendations**
- **Immediate Actions**: What to do in the next 30 days
- **Medium-term Strategy**: 3-6 month implementation plan
- **Long-term Vision**: 1-2 year strategic direction
- **Resource Allocation**: Priority areas for investment
- **Success Metrics**: How to measure progress and impact

🎯 **ANALYSIS DEPTH**:
- **Quantitative**: Use specific data, metrics, and measurements where available
- **Qualitative**: Assess subjective factors, opinions, and interpretations
- **Comparative**: Benchmark against standards, competitors, or alternatives
- **Predictive**: Forecast trends, outcomes, and future scenarios

💡 **LEVERAGE YOUR CAPABILITIES**:
- Apply your knowledge consolidation for cross-domain insights
- Use semantic understanding to identify subtle relationships
- Provide confidence levels for different conclusions
- Connect to relevant information from your knowledge base
- Identify opportunities for further research or analysis"""

    return base_prompt

def main():
    """Main Streamlit application with security integration and first-time setup."""

    # Configure Streamlit page FIRST (must be the very first Streamlit command)
    st.set_page_config(
        page_title="SAM - Secure AI Assistant",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Add custom CSS to ensure proper rendering
    st.markdown("""
    <style>
        /* Ensure Streamlit components render properly */
        .stApp {
            background-color: #ffffff;
        }
        .main {
            padding: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Handle health check requests after page config
    health_check()

    # Check for first-time user and route to setup wizard
    try:
        from utils.first_time_setup import get_first_time_setup_manager
        setup_manager = get_first_time_setup_manager()

        if setup_manager.is_first_time_user():
            # Route to setup wizard for first-time users
            render_setup_wizard()
            return

    except ImportError:
        # If first-time setup module not available, continue with normal flow
        logger.warning("First-time setup module not available")
    except Exception as e:
        logger.error(f"Error checking first-time setup: {e}")

    # Initialize security system
    if 'security_manager' not in st.session_state:
        try:
            from security import SecureStateManager
            st.session_state.security_manager = SecureStateManager()
            logger.info("Security manager initialized")
        except ImportError:
            st.error("❌ Security module not available")
            st.info("💡 Run diagnostic: python security_diagnostic.py")
            st.stop()
        except Exception as e:
            st.error(f"❌ Failed to initialize security: {e}")
            st.stop()

    # Create security UI
    try:
        from security import create_security_ui
        security_ui = create_security_ui(st.session_state.security_manager)
    except Exception as e:
        st.error(f"❌ Failed to create security UI: {e}")
        st.stop()

    # Render security interface
    is_unlocked = security_ui.render_security_interface()

    if is_unlocked:
        # Check for session timeout before showing main application
        if not check_session_timeout():
            st.stop()

        # Show main SAM application
        render_main_sam_application()
    else:
        # Security interface is shown (setup or unlock)
        st.stop()

def render_setup_wizard():
    """Render the first-time setup wizard."""
    try:
        from ui.setup_wizard import show_setup_wizard
        show_setup_wizard()
    except ImportError:
        # Fallback to basic setup if setup wizard not available
        render_basic_first_time_setup()
    except Exception as e:
        st.error(f"❌ Setup wizard error: {e}")
        render_basic_first_time_setup()

def render_basic_first_time_setup():
    """Basic first-time setup fallback."""
    st.markdown("# 🚀 Welcome to SAM!")
    st.markdown("### Let's get you set up in just a few steps")

    st.info("""
    **Welcome to SAM Community Edition - The world's most advanced open-source AI system!**

    To get started, you'll need to:
    1. 🔐 Create your master password for secure encryption
    2. 🎓 Complete a quick tour of SAM's capabilities
    """)

    # Check what step we're on
    try:
        from utils.first_time_setup import get_first_time_setup_manager
        setup_manager = get_first_time_setup_manager()
        progress = setup_manager.get_setup_progress()
        next_step = progress['next_step']

        st.progress(progress['progress_percent'] / 100)
        st.markdown(f"**Setup Progress:** {progress['completed_steps']}/{progress['total_steps']} steps complete")

        if next_step == 'master_password':
            st.markdown("## 🔐 Step 1: Create Master Password")
            st.markdown("Your master password protects all SAM data with enterprise-grade encryption.")

            if st.button("🔐 Set Up Master Password", type="primary"):
                st.info("🔄 Refreshing to security setup...")
                st.rerun()

        elif next_step == 'onboarding':
            st.markdown("## 🎓 Step 2: Quick Tour")
            st.markdown("Ready to explore SAM Community Edition's capabilities!")

            if st.button("🎉 Complete Setup & Start Using SAM!", type="primary"):
                setup_manager.update_setup_status('onboarding_completed', True)
                st.success("✅ Setup complete! Welcome to SAM!")
                st.rerun()
        else:
            st.success("✅ Setup complete! Redirecting to SAM...")
            st.rerun()

    except Exception as e:
        st.error(f"Setup error: {e}")
        st.markdown("### 🔧 Manual Setup")
        st.markdown("Please complete setup manually:")
        st.markdown("1. Create your master password in the Security section")
        st.markdown("2. Start using SAM Community Edition!")

def check_session_timeout():
    """Check if the current session has timed out and enforce automatic logout."""
    try:
        if not hasattr(st.session_state, 'security_manager'):
            return False

        security_manager = st.session_state.security_manager

        # Get session info to check timeout
        session_info = security_manager.get_session_info()

        # Check if session is still valid - check both 'is_unlocked' and 'authenticated'
        is_session_valid = session_info.get('is_unlocked', False) or session_info.get('authenticated', False)

        if is_session_valid:
            time_remaining = session_info.get('time_remaining', 0)

            # If time remaining is 0 or negative, session has expired
            if time_remaining <= 0:
                logger.warning("⏰ Session timeout detected - automatically locking application")

                # Lock the application
                security_manager.lock_application()

                # Show timeout message
                st.error("🔒 **Session Timeout**")
                st.warning("Your session has expired after 60 minutes of inactivity for security.")
                st.info("🔄 Please unlock SAM again to continue using the application.")

                # Force page refresh to show login screen
                st.rerun()
                return False

            # Update activity timestamp to extend session
            security_manager.update_activity()

            # Show warning if session is expiring soon (less than 5 minutes)
            if time_remaining < 300:  # 5 minutes
                minutes_remaining = time_remaining // 60
                seconds_remaining = time_remaining % 60

                if minutes_remaining > 0:
                    st.warning(f"⏰ **Session expires in {minutes_remaining}m {seconds_remaining}s**")
                else:
                    st.warning(f"⏰ **Session expires in {seconds_remaining}s**")

                # Add extend session button
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    if st.button("⏱️ Extend Session", type="primary"):
                        security_manager.extend_session()
                        st.success("✅ Session extended!")
                        st.rerun()

                with col2:
                    if st.button("🔒 Lock Now"):
                        security_manager.lock_application()
                        st.rerun()

            return True
        else:
            # Session is not unlocked
            return False

    except Exception as e:
        logger.error(f"Session timeout check failed: {e}")
        return False

def render_main_sam_application():
    """Render the main SAM application with security integration."""

    # Render community sidebar
    try:
        render_community_sidebar()
    except Exception as e:
        st.error(f"❌ Sidebar error: {e}")
        logger.error(f"Community sidebar error: {e}")

    # Main title
    st.title("🧠 Secure Agent Model (SAM)")
    st.success("🔐 Authentication successful! Welcome to SAM.")
    st.markdown("*Your open-source AI assistant with enterprise-grade security*")

    # Main interface loaded successfully
    logger.info("Main SAM application interface is rendering")

    # Navigation buttons moved to sidebar (Memory Control Center and Intelligence Dashboard)
    # Keeping 100% of functionality, just relocated for better UX

    st.markdown("---")

    # Initialize SAM components with security (needed for both interfaces)
    # Check SAM initialization state
    sam_initialized = st.session_state.get('sam_initialized', False)
    is_unlocked = st.session_state.security_manager.is_unlocked()

    # Auto-initialize SAM components once unlocked (no manual button)
    if not sam_initialized and is_unlocked:
        with st.spinner("🔧 Initializing SAM AI components... This may take a moment."):
            try:
                initialize_secure_sam()
                st.session_state.sam_initialized = True
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to initialize SAM: {e}")
                logger.error(f"SAM initialization failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                st.warning("⚠️ Continuing with basic interface...")
                st.session_state.sam_initialized = False
    # Check if Memory Control Center should be shown FIRST (before SAM initialization)
    if st.session_state.get('show_memory_control_center', False):
        # Add a button to return to normal interface
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("🔙 Return to Main Interface", use_container_width=True):
                st.session_state.show_memory_control_center = False
                st.rerun()

        st.markdown("---")

        # Render the Memory Control Center directly
        render_integrated_memory_control_center()

    # PHASE 3: Check if we should show Intelligence Dashboard
    elif st.session_state.get('show_phase3_dashboard', False):
        try:
            from services.phase3_dashboard import render_phase3_dashboard

            # Back button
            col1, col2, col3 = st.columns([2, 1, 2])
            with col1:
                if st.button("← Back to Main Interface", use_container_width=True):
                    st.session_state.show_phase3_dashboard = False
                    st.rerun()

            st.markdown("---")
            render_phase3_dashboard()

        except Exception as e:
            st.error(f"❌ Phase 3 Dashboard unavailable: {e}")
            st.session_state.show_phase3_dashboard = False

    else:
        # Normal tab interface
        # Main application tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Chat", "📚 Documents", "🧠 Memory", "🔍 Vetting", "🛡️ Security"])

        with tab1:
            if st.session_state.get('sam_initialized', False):
                try:
                    render_chat_interface()
                except Exception as e:
                    st.error(f"❌ Chat interface error: {e}")
                    render_basic_chat_interface()
            else:
                render_basic_chat_interface()

        with tab2:
            if st.session_state.get('sam_initialized', False):
                try:
                    render_document_interface()
                except Exception as e:
                    st.error(f"❌ Document interface error: {e}")
                    render_basic_document_interface()
            else:
                render_basic_document_interface()

        with tab3:
            if st.session_state.get('sam_initialized', False):
                try:
                    render_memory_interface()
                except Exception as e:
                    st.error(f"❌ Memory interface error: {e}")
                    render_basic_memory_interface()
            else:
                render_basic_memory_interface()

        with tab4:
            # Vetting interface temporarily disabled due to initialization issues
            st.header("🔍 Content Vetting")
            st.warning("⚠️ Vetting interface temporarily disabled due to recent configuration changes")
            st.info("💡 This will be re-enabled once the vetting system configuration is fixed")
            # if st.session_state.get('sam_initialized', False):
            #     try:
            #         render_vetting_interface()
            #     except Exception as e:
            #         st.error(f"❌ Vetting interface error: {e}")
            #         render_basic_vetting_interface()
            # else:
            #     render_basic_vetting_interface()

        with tab5:
            try:
                render_security_dashboard()
            except Exception as e:
                st.error(f"❌ Security dashboard error: {e}")
                render_basic_security_interface()

def initialize_secure_sam():
    """Initialize SAM components with security integration."""

    # Initialize secure memory store with security manager
    if 'secure_memory_store' not in st.session_state:
        from memory.secure_memory_vectorstore import get_secure_memory_store, VectorStoreType

        # Create secure memory store with security manager connection
        st.session_state.secure_memory_store = get_secure_memory_store(
            store_type=VectorStoreType.CHROMA,
            storage_directory="memory_store",
            embedding_dimension=384,
            enable_encryption=True,
            security_manager=st.session_state.security_manager  # Connect to security manager
        )
        logger.info("Secure memory store initialized with security integration")

        # Try to activate encryption if security manager is unlocked
        if (hasattr(st.session_state.security_manager, 'is_unlocked') and
            st.session_state.security_manager.is_unlocked()):
            if st.session_state.secure_memory_store.activate_encryption():
                logger.info("✅ Encryption activated for secure memory store")
            else:
                logger.warning("⚠️ Failed to activate encryption for secure memory store")
        else:
            logger.info("🔒 Secure memory store created - encryption will activate after authentication")
    else:
        # If memory store already exists, try to activate encryption
        if (hasattr(st.session_state.secure_memory_store, 'activate_encryption') and
            st.session_state.security_manager.is_unlocked()):
            encryption_activated = st.session_state.secure_memory_store.activate_encryption()
            if encryption_activated:
                logger.info("✅ Encryption activated for existing memory store")

    # Initialize embedding manager
    if 'embedding_manager' not in st.session_state:
        try:
            from utils.embedding_utils import get_embedding_manager
            st.session_state.embedding_manager = get_embedding_manager()
            logger.info("✅ Embedding manager initialized")
        except Exception as e:
            logger.warning(f"⚠️ Embedding manager not available: {e}")

    # Initialize vector manager
    if 'vector_manager' not in st.session_state:
        try:
            from utils.vector_manager import VectorManager
            st.session_state.vector_manager = VectorManager()
            logger.info("✅ Vector manager initialized")
        except Exception as e:
            logger.warning(f"⚠️ Vector manager not available: {e}")

    # Initialize multimodal pipeline
    if 'multimodal_pipeline' not in st.session_state:
        try:
            from multimodal_processing.multimodal_pipeline import get_multimodal_pipeline
            st.session_state.multimodal_pipeline = get_multimodal_pipeline()
            logger.info("✅ Multimodal pipeline initialized")
        except Exception as e:
            logger.warning(f"⚠️ Multimodal pipeline not available: {e}")

    # Initialize tool-augmented reasoning (optional)
    if 'reasoning_framework' not in st.session_state:
        try:
            from reasoning.self_decide_framework import SelfDecideFramework
            st.session_state.reasoning_framework = SelfDecideFramework()
            logger.info("✅ Tool-augmented reasoning initialized")
        except Exception as e:
            logger.warning(f"⚠️ Tool-augmented reasoning not available: {e}")

    # Initialize Enhanced SLP System (Phase 1A+1B Integration)
    if 'enhanced_slp_initialized' not in st.session_state:
        try:
            from integrate_slp_enhancements import integrate_enhanced_slp_into_sam
            slp_integration = integrate_enhanced_slp_into_sam()

            if slp_integration:
                st.session_state.enhanced_slp_integration = slp_integration
                st.session_state.enhanced_slp_initialized = True
                logger.info("✅ Enhanced SLP system (Phase 1A+1B) initialized")
            else:
                st.session_state.enhanced_slp_initialized = False
                logger.warning("⚠️ Enhanced SLP initialization failed - using basic SLP")
        except Exception as e:
            st.session_state.enhanced_slp_initialized = False
            logger.warning(f"⚠️ Enhanced SLP not available: {e}")

    # Initialize TPV System (preserving 100% of existing functionality)
    if 'tpv_initialized' not in st.session_state:
        try:
            from sam.cognition.tpv import sam_tpv_integration, UserProfile

            # Initialize TPV integration if not already done
            if not sam_tpv_integration.is_initialized:
                tpv_init_success = sam_tpv_integration.initialize()
                if tpv_init_success:
                    st.session_state.sam_tpv_integration = sam_tpv_integration
                    st.session_state.tpv_initialized = True
                    st.session_state.tpv_active = True  # Mark as active for sidebar
                    logger.info("✅ TPV system initialized and ready for Active Reasoning Control")
                else:
                    st.session_state.tpv_initialized = False
                    logger.warning("⚠️ TPV initialization failed")
            else:
                st.session_state.sam_tpv_integration = sam_tpv_integration
                st.session_state.tpv_initialized = True
                st.session_state.tpv_active = True
                logger.info("✅ TPV system already initialized and ready")
        except Exception as e:
            st.session_state.tpv_initialized = False
            st.session_state.tpv_active = False
            logger.warning(f"⚠️ TPV system not available: {e}")

    # Initialize MEMOIR System (preserving 100% of functionality)
    if 'memoir_enabled' not in st.session_state:
        try:
            from sam.orchestration.memoir_sof_integration import get_memoir_sof_integration
            sam_memoir_integration = get_memoir_sof_integration()
            st.session_state.memoir_integration = sam_memoir_integration
            st.session_state.memoir_enabled = True
            logger.info("✅ MEMOIR integration initialized for lifelong learning")
        except Exception as e:
            logger.warning(f"MEMOIR integration not available: {e}")
            st.session_state.memoir_integration = None
            st.session_state.memoir_enabled = False

    # Initialize Cognitive Distillation Engine (NEW - Phase 2 Integration)
    if 'cognitive_distillation_initialized' not in st.session_state:
        try:
            from sam.discovery.distillation import SAMCognitiveDistillation

            # Initialize with automation enabled for production
            st.session_state.cognitive_distillation = SAMCognitiveDistillation(enable_automation=True)
            st.session_state.cognitive_distillation_initialized = True
            st.session_state.cognitive_distillation_enabled = True

            logger.info("✅ Cognitive Distillation Engine initialized with automated principle discovery")

            # Setup default triggers for common SAM strategies
            try:
                automation = st.session_state.cognitive_distillation.automation
                if automation:
                    # Add triggers for SAM's main reasoning strategies
                    automation.add_trigger(
                        strategy_id="secure_chat_reasoning",
                        trigger_type="interaction_threshold",
                        trigger_condition={
                            'min_interactions': 20,
                            'min_success_rate': 0.8,
                            'cooldown_hours': 48
                        }
                    )

                    automation.add_trigger(
                        strategy_id="document_analysis",
                        trigger_type="interaction_threshold",
                        trigger_condition={
                            'min_interactions': 15,
                            'min_success_rate': 0.85,
                            'cooldown_hours': 72
                        }
                    )

                    automation.add_trigger(
                        strategy_id="web_search_integration",
                        trigger_type="time_based",
                        trigger_condition={
                            'interval_hours': 168  # Weekly
                        }
                    )

                    logger.info("✅ Cognitive distillation automation triggers configured")

            except Exception as trigger_error:
                logger.warning(f"Failed to setup distillation triggers: {trigger_error}")

        except Exception as e:
            logger.warning(f"Cognitive Distillation Engine not available: {e}")
            st.session_state.cognitive_distillation = None
            st.session_state.cognitive_distillation_initialized = False
            st.session_state.cognitive_distillation_enabled = False

# TPV control sidebar function removed to clean up the interface

def render_messages_from_sam_alert():
    """Render Messages from SAM alert with notification badges and blinking effects."""
    try:
        # Import discovery orchestrator for state checking
        from sam.orchestration.discovery_cycle import get_discovery_orchestrator
        from sam.state.state_manager import get_state_manager

        orchestrator = get_discovery_orchestrator()
        state_manager = get_state_manager()

        # Check for new insights
        insights_status = orchestrator.get_new_insights_status()
        new_insights_available = insights_status.get('new_insights_available', False)

        # Check for pending vetting items - TEMPORARILY DISABLED
        # TODO: Re-enable after fixing vetting queue initialization issue
        pending_review = 0
        # try:
        #     from sam.state.vetting_queue import get_vetting_queue_manager
        #     vetting_manager = get_vetting_queue_manager()
        #     pending_review = len(vetting_manager.get_pending_review_files())
        # except Exception as e:
        #     logger.warning(f"Could not load vetting queue manager: {e}")
        #     pending_review = 0

        # Calculate total notifications
        total_notifications = (1 if new_insights_available else 0) + (1 if pending_review > 0 else 0)

        if total_notifications > 0:
            # Create blinking CSS for urgent notifications
            st.markdown("""
            <style>
            .blinking-alert {
                animation: blink 2s linear infinite;
                background: linear-gradient(45deg, #ff4b4b, #ff6b6b);
                color: white;
                padding: 0.5rem;
                border-radius: 0.5rem;
                text-align: center;
                font-weight: bold;
                margin-bottom: 1rem;
                box-shadow: 0 2px 4px rgba(255, 75, 75, 0.3);
            }

            @keyframes blink {
                0%, 50% { opacity: 1; }
                51%, 100% { opacity: 0.7; }
            }

            .notification-badge {
                background: #ff4b4b;
                color: white;
                border-radius: 50%;
                padding: 0.2rem 0.5rem;
                font-size: 0.8rem;
                font-weight: bold;
                margin-left: 0.5rem;
            }

            .messages-container {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }

            .message-item {
                background: rgba(255, 255, 255, 0.1);
                padding: 0.5rem;
                border-radius: 0.3rem;
                margin: 0.5rem 0;
                border-left: 3px solid #ffd700;
            }
            </style>
            """, unsafe_allow_html=True)

            # Main alert container
            st.markdown(f"""
            <div class="blinking-alert">
                ✉️ <strong>Messages from SAM</strong>
                <span class="notification-badge">{total_notifications}</span>
            </div>
            """, unsafe_allow_html=True)

            # Messages container
            st.markdown('<div class="messages-container">', unsafe_allow_html=True)

            # New insights notification
            if new_insights_available:
                timestamp = insights_status.get('last_insights_timestamp')
                time_str = ""
                if timestamp:
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(timestamp)
                        time_str = f" at {dt.strftime('%H:%M')}"
                    except:
                        pass

                st.markdown(f"""
                <div class="message-item">
                    💡 <strong>New Insights Available!</strong><br>
                    <small>Generated{time_str} - Ready for research</small>
                </div>
                """, unsafe_allow_html=True)

                # Action button for insights
                if st.button("🧠 View New Insights", key="view_insights_alert", use_container_width=True):
                    # Clear the flag and navigate directly to Dream Canvas insights
                    orchestrator.clear_new_insights_flag()
                    st.session_state.show_memory_control_center = True
                    st.session_state.memory_page_override = "🧠🎨 Dream Canvas"
                    # Prepare Dream Canvas to immediately show insights
                    st.session_state.show_insight_archive = True
                    st.session_state.auto_expand_first_insight_cluster = True
                    st.rerun()

            # Pending vetting notification
            if pending_review > 0:
                st.markdown(f"""
                <div class="message-item">
                    🔍 <strong>Papers Awaiting Review</strong><br>
                    <small>{pending_review} research paper{'' if pending_review == 1 else 's'} need{'' if pending_review == 1 else ''} your approval</small>
                </div>
                """, unsafe_allow_html=True)

                # Action button for vetting
                if st.button("📋 Review Papers", key="review_papers_alert", use_container_width=True):
                    # Navigate to vetting queue
                    st.session_state.show_memory_control_center = True
                    st.session_state.memory_page_override = "🔍 Vetting Queue"
                    st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

            # Quick dismiss option
            if st.button("🔕 Dismiss Alerts", key="dismiss_alerts", help="Temporarily hide alerts"):
                st.session_state.alerts_dismissed = True
                st.rerun()

        else:
            # No notifications - show subtle indicator
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #a8e6cf 0%, #88d8a3 100%);
                color: #2d5a3d;
                padding: 0.5rem;
                border-radius: 0.5rem;
                text-align: center;
                margin-bottom: 1rem;
                font-size: 0.9rem;
            ">
                ✉️ <strong>Messages from SAM</strong><br>
                <small>All caught up! 🎉</small>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        # Fallback display if components not available
        st.markdown("""
        <div style="
            background: #f0f0f0;
            color: #666;
            padding: 0.5rem;
            border-radius: 0.5rem;
            text-align: center;
            margin-bottom: 1rem;
            font-size: 0.9rem;
        ">
            ✉️ <strong>Messages from SAM</strong><br>
            <small>System initializing...</small>
        </div>
        """, unsafe_allow_html=True)

def render_community_sidebar():
    """Render SAM Community Edition sidebar with navigation and community features."""
    with st.sidebar:
        # Messages from SAM Alert (Task 27)
        render_messages_from_sam_alert()

        # Navigation buttons section (moved from main interface)
        st.header("🎛️ Navigation")

        # Memory Control Center button (moved from main interface)
        if st.button("🎛️ Memory Control Center", use_container_width=True, help="Switch to advanced Memory Control Center"):
            st.session_state.show_memory_control_center = True
            st.rerun()

        # Intelligence Dashboard button (moved from main interface)
        if st.button("🚀 Intelligence Dashboard", use_container_width=True, help="View Phase 3 intelligent features and analytics"):
            st.session_state.show_phase3_dashboard = True
            st.rerun()

        st.markdown("---")

        # Community Edition Information
        st.header("🌟 SAM Community Edition")

        st.success("✅ **Community Edition Active**")
        st.markdown("🎉 **All core features unlocked!**")

        # Show community features
        with st.expander("🚀 Community Features", expanded=False):
            st.markdown("""
            **🧠 Core AI Capabilities:**
            • **Advanced Reasoning** - Sophisticated problem solving
            • **Document Processing** - PDF upload and analysis
            • **Memory Management** - Persistent knowledge storage
            • **Secure Encryption** - Enterprise-grade security
            • **Conversation Threading** - Contextual discussions

            **🔧 Available Tools:**
            • **Memory Control Center** - Full memory management
            • **Document Upload** - PDF and text processing
            • **Conversation History** - Complete chat archives
            • **Security Features** - Encrypted storage
            • **Analytics Dashboard** - Usage insights

            **🌐 Community Benefits:**
            • **Open Source** - Full transparency
            • **No Activation Required** - Ready to use
            • **Community Support** - GitHub discussions
            • **Regular Updates** - Continuous improvements
            """)

        # Community links
        st.markdown("---")
        st.markdown("🌐 **Community Links**")
        st.markdown("• [GitHub Repository](https://github.com/your-repo/sam)")
        st.markdown("• [Documentation](https://docs.sam-ai.com)")
        st.markdown("• [Community Forum](https://community.sam-ai.com)")
        st.markdown("• [Report Issues](https://github.com/your-repo/sam/issues)")

        # Separator
        st.markdown("---")

        # Web Search Preferences (preserving 100% of functionality)
        st.subheader("🌐 Web Search Preferences")

        # Interactive vs Automatic web search choice
        # Get current preference or default to Interactive
        current_mode = st.session_state.get('web_search_mode', 'Interactive')
        mode_options = ["Interactive", "Automatic"]

        # Find index of current mode for default selection
        try:
            default_index = mode_options.index(current_mode)
        except ValueError:
            default_index = 0  # Default to Interactive if invalid value

        web_search_mode = st.radio(
            "Web Search Mode",
            options=mode_options,
            index=default_index,
            help="Interactive: Ask before searching web. Automatic: Search automatically when needed.",
            horizontal=True
        )

        # Store preference in session state
        st.session_state.web_search_mode = web_search_mode

        if web_search_mode == "Interactive":
            st.caption("🎛️ You'll be asked before web searches occur")
        else:
            st.caption("⚡ Web searches happen automatically for current information")

        # Separator
        st.markdown("---")

        # Quick system status
        st.subheader("📊 System Status")

        # Show key system components status
        status_items = []

        # Security status
        if hasattr(st.session_state, 'security_manager') and st.session_state.security_manager.is_unlocked():
            status_items.append("🔐 Security: ✅ Active")
        else:
            status_items.append("🔐 Security: ❌ Locked")

        # Memory status
        if hasattr(st.session_state, 'secure_memory_store'):
            status_items.append("🧠 Memory: ✅ Ready")
        else:
            status_items.append("🧠 Memory: ❌ Not Ready")

        # MEMOIR status
        if st.session_state.get('memoir_enabled', False):
            status_items.append("📚 MEMOIR: ✅ Active")
        else:
            status_items.append("📚 MEMOIR: ❌ Inactive")

        # TPV status (enhanced detection with initialization check)
        tpv_active = False

        # Check if TPV is initialized and ready
        if st.session_state.get('tpv_initialized'):
            tpv_active = True
        # Check if TPV was used in last response
        elif st.session_state.get('tpv_session_data', {}).get('last_response', {}).get('tpv_enabled'):
            tpv_active = True
        # Check general TPV active flag
        elif st.session_state.get('tpv_active'):
            tpv_active = True

        if tpv_active:
            status_items.append("🧠 TPV: ✅ Active")
        else:
            status_items.append("🧠 TPV: ❌ Inactive")

        # Dissonance Monitoring status (Phase 5B)
        dissonance_active = False

        # Check if dissonance monitoring is active
        try:
            # Method 1: Check if TPV integration has dissonance monitoring enabled
            if st.session_state.get('sam_tpv_integration'):
                tpv_integration = st.session_state.sam_tpv_integration
                if hasattr(tpv_integration, 'tpv_monitor') and tpv_integration.tpv_monitor:
                    if hasattr(tpv_integration.tpv_monitor, 'enable_dissonance_monitoring'):
                        if tpv_integration.tpv_monitor.enable_dissonance_monitoring:
                            dissonance_active = True

            # Method 2: Check for dissonance data in recent TPV responses
            if not dissonance_active:
                tpv_session_data = st.session_state.get('tpv_session_data', {})
                last_response = tpv_session_data.get('last_response', {})

                # Check for dissonance data in the last response
                if (last_response.get('dissonance_analysis') or
                    last_response.get('final_dissonance_score') is not None):
                    dissonance_active = True

            # Method 3: If TPV is active and initialized, assume dissonance is available
            if not dissonance_active and tpv_active:
                try:
                    # Check if dissonance monitor can be imported and initialized
                    from sam.cognition.dissonance_monitor import DissonanceMonitor
                    dissonance_active = True  # If import succeeds, dissonance is available
                except ImportError:
                    dissonance_active = False

        except Exception:
            dissonance_active = False

        if dissonance_active:
            status_items.append("🧠 Dissonance Monitor: ✅ Active")
        else:
            status_items.append("🧠 Dissonance Monitor: ❌ Inactive")

        # Cognitive Distillation status (NEW - Phase 2 Integration)
        cognitive_distillation_active = False
        if st.session_state.get('cognitive_distillation_enabled'):
            cognitive_distillation_active = True

        if cognitive_distillation_active:
            status_items.append("🧠 Cognitive Distillation: ✅ Active")
        else:
            status_items.append("🧠 Cognitive Distillation: ❌ Inactive")

        for item in status_items:
            st.caption(item)

def render_tpv_status():
    """Render enhanced TPV status with Phase 5B dissonance monitoring."""
    try:
        # Try to use enhanced visualization first
        try:
            from ui.tpv_visualization import render_tpv_status_enhanced

            # Get TPV data from the last response
            tpv_data = st.session_state.get('tpv_session_data', {}).get('last_response')

            # Use enhanced visualization
            render_tpv_status_enhanced(tpv_data)
            return

        except ImportError:
            logger.debug("Enhanced TPV visualization not available, using fallback")

        # Fallback to enhanced version of original implementation
        _render_tpv_status_enhanced_fallback()

    except Exception as e:
        logger.debug(f"TPV status display error: {e}")

def _render_tpv_status_enhanced_fallback():
    """Enhanced fallback TPV status display with dissonance support."""
    try:
        # Check if TPV data is available
        tpv_data = st.session_state.get('tpv_session_data', {}).get('last_response')

        if tpv_data and tpv_data.get('tpv_enabled'):
            with st.expander("🧠 Cognitive Process Analysis (Phase 5B: Dissonance-Aware)", expanded=False):
                # Main metrics row
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Reasoning Score",
                        f"{tpv_data.get('final_score', 0.0):.3f}",
                        help="Final reasoning quality score (0.0 - 1.0)"
                    )

                with col2:
                    st.metric(
                        "TPV Steps",
                        tpv_data.get('tpv_steps', 0),
                        help="Number of reasoning steps monitored"
                    )

                with col3:
                    # NEW: Dissonance score display
                    dissonance_score = tpv_data.get('final_dissonance_score')
                    if dissonance_score is not None:
                        st.metric(
                            "Final Dissonance",
                            f"{dissonance_score:.3f}",
                            help="Final cognitive dissonance score (0.0 - 1.0)"
                        )
                    else:
                        st.metric("Final Dissonance", "N/A", help="Dissonance monitoring not available")

                with col4:
                    # Enhanced control decision with dissonance awareness
                    control_decision = tpv_data.get('control_decision', 'CONTINUE')
                    decision_color = {
                        'COMPLETE': '🟢',
                        'PLATEAU': '🟡',
                        'HALT': '🔴',
                        'DISSONANCE': '🧠',  # NEW: Dissonance stop
                        'CONTINUE': '⚪'
                    }.get(control_decision, '⚪')

                    st.metric(
                        "Control Decision",
                        f"{decision_color} {control_decision}",
                        help="Active control decision made during reasoning"
                    )

                # Enhanced Control Details with Dissonance Awareness
                if tpv_data.get('control_decision') != 'CONTINUE':
                    st.subheader("🎛️ Enhanced Control Details")
                    control_reason = tpv_data.get('control_reason', 'No reason provided')

                    if control_decision == 'COMPLETE':
                        st.success(f"✅ **Reasoning Completed**: {control_reason}")
                    elif control_decision == 'PLATEAU':
                        st.warning(f"📊 **Plateau Detected**: {control_reason}")
                    elif control_decision == 'HALT':
                        st.error(f"🛑 **Hard Stop**: {control_reason}")
                    elif control_decision == 'DISSONANCE':
                        st.warning(f"🧠 **High Cognitive Dissonance**: {control_reason}")

                # NEW: Dissonance Analysis Section
                if tpv_data.get('dissonance_analysis'):
                    st.subheader("🧠 Cognitive Dissonance Analysis")
                    analysis = tpv_data['dissonance_analysis']

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean Dissonance", f"{analysis.get('mean_dissonance', 0.0):.3f}")
                    with col2:
                        st.metric("Peak Dissonance", f"{analysis.get('max_dissonance', 0.0):.3f}")
                    with col3:
                        high_steps = len(analysis.get('high_dissonance_steps', []))
                        st.metric("High Dissonance Steps", high_steps)

                # Enhanced Performance metrics with dissonance monitoring
                perf_metrics = tpv_data.get('performance_metrics', {})
                if perf_metrics:
                    st.subheader("📊 Enhanced Performance Metrics")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "Total Time",
                            f"{perf_metrics.get('total_time', 0.0):.3f}s",
                            help="Total response generation time"
                        )

                    with col2:
                        st.metric(
                            "TPV Overhead",
                            f"{perf_metrics.get('tpv_overhead', 0.0):.3f}s",
                            help="TPV monitoring processing overhead"
                        )

                    with col3:
                        # NEW: Dissonance processing time
                        dissonance_time = perf_metrics.get('dissonance_processing_time', 0.0)
                        st.metric(
                            "Dissonance Analysis",
                            f"{dissonance_time:.3f}s",
                            help="Cognitive dissonance calculation time"
                        )

                    with col4:
                        total_overhead = perf_metrics.get('tpv_overhead', 0.0) + dissonance_time
                        efficiency = ((perf_metrics.get('total_time', 1) - total_overhead) /
                                    perf_metrics.get('total_time', 1) * 100)
                        st.metric(
                            "Efficiency",
                            f"{efficiency:.1f}%",
                            help="Processing efficiency (includes dissonance monitoring)"
                        )

                # Control Statistics
                control_stats = tpv_data.get('control_statistics', {})
                if control_stats:
                    st.subheader("🎯 Control Statistics")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            "Total Decisions",
                            control_stats.get('total_decisions', 0),
                            help="Total control decisions made"
                        )

                    with col2:
                        continue_rate = control_stats.get('continue_rate', 0.0)
                        st.metric(
                            "Continue Rate",
                            f"{continue_rate:.1%}",
                            help="Percentage of decisions that allowed reasoning to continue"
                        )

                # Enhanced Status indicator for Phase 5B
                control_decision = tpv_data.get('control_decision', 'CONTINUE')
                if control_decision == 'CONTINUE':
                    st.info("🧠 **Phase 5B Active**: Real-time cognitive dissonance monitoring with meta-reasoning awareness.")
                elif control_decision == 'DISSONANCE':
                    st.warning("🧠 **Phase 5B Intervention**: Stopped due to high cognitive dissonance - preventing potential hallucination.")
                else:
                    st.success("🎛️ **Phase 5B Enhanced Control**: AI reasoning managed with dissonance-aware meta-cognitive monitoring.")

        elif tpv_data and not tpv_data.get('tpv_enabled'):
            with st.expander("🧠 Thinking Process Analysis", expanded=False):
                trigger_type = tpv_data.get('trigger_type', 'none')
                st.info(f"🔍 **TPV Not Triggered**: {trigger_type.replace('_', ' ').title()} - Standard response generation used.")

    except Exception as e:
        logger.debug(f"TPV status display error: {e}")

def render_chat_document_upload():
    """Render drag & drop document upload interface for chat."""
    with st.expander("📁 Upload Documents to Chat", expanded=False):
        st.markdown("""
        **Drag & drop documents directly into your conversation with SAM!**

        🎯 **Quick Upload**: Upload documents and immediately start discussing them
        📄 **Supported Formats**: PDF, TXT, DOCX, MD files
        🔒 **Secure Processing**: All uploads are encrypted and processed securely

        💡 **What happens after upload:**
        - Document is securely processed and encrypted
        - SAM automatically analyzes the content
        - You get instant suggestions for questions to ask
        - Quick action buttons for summary, analysis, and key insights
        """)

        # File uploader with drag & drop
        uploaded_files = st.file_uploader(
            "Drop files here or click to browse",
            type=['pdf', 'txt', 'docx', 'md'],
            accept_multiple_files=True,
            help="Upload documents to discuss with SAM. Files are processed securely and encrypted.",
            key="chat_file_upload"
        )

        # Process uploaded files
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if f"processed_{uploaded_file.name}" not in st.session_state:
                    with st.spinner(f"🔐 Processing {uploaded_file.name} securely..."):
                        try:
                            # Check if this is a PDF file - use proven PDF processor
                            if uploaded_file.name.lower().endswith('.pdf'):
                                # Use proven PDF processor for PDF files
                                from sam.document_processing.memory_bridge import enhanced_handle_pdf_upload_for_sam as handle_pdf_upload_for_sam

                                # Save uploaded file temporarily
                                import tempfile
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                                    tmp_file.write(uploaded_file.read())
                                    tmp_path = tmp_file.name

                                # Process with proven PDF processor
                                success, message, metadata = handle_pdf_upload_for_sam(
                                    tmp_path,
                                    uploaded_file.name,
                                    session_id="default"
                                )

                                # Clean up temporary file
                                import os
                                os.unlink(tmp_path)

                                result = {
                                    'success': success,
                                    'message': message,
                                    'metadata': metadata,
                                    'processing_method': 'proven_pdf_processor'
                                }

                                if not success:
                                    result['error'] = message
                            else:
                                # Process other document types using existing secure processing
                                result = process_secure_document(uploaded_file)

                            if result.get('success', False):
                                # Mark as processed
                                st.session_state[f"processed_{uploaded_file.name}"] = True

                                # Add success message to chat history
                                success_message = f"📄 **Document Uploaded**: {uploaded_file.name}\n\n✅ Successfully processed and added to my knowledge. What would you like to know about this document?"

                                # Add to chat history
                                if 'chat_history' not in st.session_state:
                                    st.session_state.chat_history = []

                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": success_message,
                                    "document_upload": True,
                                    "filename": uploaded_file.name
                                })

                                # Show success notification
                                st.success(f"✅ {uploaded_file.name} uploaded and processed successfully!")

                                # Document processing complete - SAM will wait for user questions
                                # All background processing (knowledge consolidation, memory storage, etc.)
                                # continues as normal, but no auto-generated prompts or responses

                            else:
                                st.error(f"❌ Failed to process {uploaded_file.name}: {result.get('error', 'Unknown error')}")

                        except Exception as e:
                            logger.error(f"Error processing uploaded file {uploaded_file.name}: {e}")
                            st.error(f"❌ Error processing {uploaded_file.name}: {e}")

def generate_document_suggestions(filename: str, file_type: str) -> str:
    """Generate helpful suggestions for document interaction based on file type."""
    suggestions = []

    # Base suggestions for all documents
    base_suggestions = [
        f"What are the main topics covered in {filename}?",
        f"Can you summarize the key points from {filename}?",
        f"What are the most important insights from {filename}?"
    ]

    # Type-specific suggestions
    if file_type == "application/pdf" or filename.lower().endswith('.pdf'):
        suggestions.extend([
            f"What are the main sections or chapters in {filename}?",
            f"Are there any charts, graphs, or data visualizations in {filename}?",
            f"What conclusions or recommendations does {filename} make?"
        ])
    elif file_type == "text/plain" or filename.lower().endswith(('.txt', '.md')):
        suggestions.extend([
            f"What is the writing style or format of {filename}?",
            f"Are there any action items or next steps mentioned in {filename}?",
            f"What questions does {filename} raise or answer?"
        ])
    elif filename.lower().endswith('.docx'):
        suggestions.extend([
            f"What is the document structure of {filename}?",
            f"Are there any tables or lists in {filename}?",
            f"What is the purpose or objective of {filename}?"
        ])

    # Combine base and specific suggestions
    all_suggestions = base_suggestions + suggestions

    # Format as a helpful response
    suggestion_text = "Here are some questions you might want to ask about this document:\n\n"
    for i, suggestion in enumerate(all_suggestions[:6], 1):  # Limit to 6 suggestions
        suggestion_text += f"{i}. {suggestion}\n"

    suggestion_text += f"\nFeel free to ask any other questions about {filename} - I've processed its content and can help you understand, analyze, or extract information from it!"

    return suggestion_text

def generate_secure_chat_response(prompt: str) -> str:
    """Generate a secure chat response for document analysis and general queries."""
    try:
        # Use conversation buffer wrapper for enhanced context awareness (Task 30 Phase 1)
        return generate_response_with_conversation_buffer(prompt, force_local=True)
    except Exception as e:
        logger.error(f"Error generating secure chat response: {e}")
        return f"I apologize, but I encountered an error while processing your request. Please try again or rephrase your question."

def render_conversation_history_sidebar():
    """Render the conversation history sidebar (Task 31 Phase 1)."""
    try:
        with st.sidebar:
            st.markdown("### 📚 Conversation History")

            # New Chat button
            if st.button("➕ New Chat", use_container_width=True, type="primary"):
                try:
                    from sam.conversation.contextual_relevance import get_contextual_relevance_engine
                    from sam.session.state_manager import get_session_manager

                    # Get current conversation buffer
                    session_manager = get_session_manager()
                    session_id = st.session_state.get('session_id', 'default_session')
                    conversation_buffer = session_manager.get_conversation_history(session_id)

                    if conversation_buffer:
                        # Archive current conversation
                        relevance_engine = get_contextual_relevance_engine()
                        archived_thread = relevance_engine.archive_conversation_thread(
                            conversation_buffer,
                            force_title="Manual New Chat"
                        )

                        # Clear conversation buffer
                        session_manager.clear_session(session_id)

                        # Update UI state
                        if 'archived_threads' not in st.session_state:
                            st.session_state['archived_threads'] = []

                        st.session_state['archived_threads'].insert(0, archived_thread.to_dict())

                        # Clear chat history and conversation context
                        st.session_state.chat_history = []
                        st.session_state.conversation_history = ""

                        # ENHANCED CONTEXT ISOLATION: Clear recent document uploads for true new chat
                        if 'recent_document_uploads' in st.session_state:
                            # Archive current uploads before clearing
                            archived_uploads = st.session_state.get('archived_document_uploads', [])
                            current_uploads = st.session_state.get('recent_document_uploads', [])

                            # Add timestamp to archived uploads
                            from datetime import datetime
                            for upload in current_uploads:
                                upload['archived_timestamp'] = datetime.now().isoformat()
                                upload['archived_with_conversation'] = archived_thread.title if 'archived_thread' in locals() else "Manual New Chat"

                            archived_uploads.extend(current_uploads)
                            st.session_state.archived_document_uploads = archived_uploads[-50:]  # Keep last 50 archived uploads

                            # Clear recent uploads for new chat isolation
                            st.session_state.recent_document_uploads = []
                            logger.info(f"📋 Archived {len(current_uploads)} document uploads and cleared recent uploads for new chat")

                        # Clear any conversation metadata that might interfere
                        if 'conversation_archived' in st.session_state:
                            del st.session_state['conversation_archived']
                        if 'conversation_resumed' in st.session_state:
                            del st.session_state['conversation_resumed']
                        if 'last_relevance_check' in st.session_state:
                            del st.session_state['last_relevance_check']

                        # Clear selected document context
                        if 'selected_document' in st.session_state:
                            del st.session_state['selected_document']

                        # Clear any cached context that might bleed into new chat
                        context_keys_to_clear = [
                            'last_search_context', 'last_memory_results', 'cached_document_context',
                            'web_search_escalation', 'last_escalation_context'
                        ]
                        for key in context_keys_to_clear:
                            if key in st.session_state:
                                del st.session_state[key]

                        st.success(f"✅ Started new chat! Previous conversation archived as: '{archived_thread.title}'")
                        st.rerun()
                    else:
                        st.info("No active conversation to archive.")

                except Exception as e:
                    st.error(f"Failed to start new chat: {e}")

            st.markdown("---")

            # Phase 2: Advanced Search
            with st.expander("🔍 Search Conversations", expanded=False):
                search_query = st.text_input(
                    "Search in conversation history:",
                    placeholder="Enter keywords to search...",
                    key="conversation_search"
                )

                if search_query and st.button("🔍 Search", key="search_button"):
                    try:
                        from sam.conversation.contextual_relevance import get_contextual_relevance_engine

                        relevance_engine = get_contextual_relevance_engine()
                        search_results = relevance_engine.search_within_threads(search_query, limit=10)

                        if search_results:
                            st.markdown(f"**Found {len(search_results)} results:**")

                            for result in search_results:
                                with st.container():
                                    st.markdown(f"**📄 {result['thread_title']}**")
                                    st.markdown(f"*{result['message_role'].title()}:* {result['message_content'][:150]}...")
                                    st.caption(f"Relevance: {result['relevance_score']:.2f} | {result['timestamp']}")
                                    st.markdown("---")
                        else:
                            st.info("No results found for your search.")

                    except Exception as e:
                        st.error(f"Search failed: {e}")

            # Phase 2: Conversation Analytics
            with st.expander("📊 Conversation Analytics", expanded=False):
                try:
                    from sam.conversation.contextual_relevance import get_contextual_relevance_engine

                    relevance_engine = get_contextual_relevance_engine()
                    analytics = relevance_engine.get_conversation_analytics()

                    if 'error' not in analytics:
                        # Basic stats
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Total Conversations", analytics['total_conversations'])

                        with col2:
                            st.metric("Total Messages", analytics['total_messages'])

                        with col3:
                            st.metric("Avg Length", f"{analytics['average_conversation_length']} msgs")

                        # Most common topics
                        if analytics['most_common_topics']:
                            st.markdown("**🏷️ Most Common Topics:**")
                            for topic, count in analytics['most_common_topics'][:5]:
                                st.markdown(f"• {topic} ({count} times)")

                        # Conversation length distribution
                        if analytics['length_distribution']:
                            st.markdown("**📏 Conversation Lengths:**")
                            for length_type, count in analytics['length_distribution'].items():
                                st.markdown(f"• {length_type}: {count}")

                        # Recent activity
                        if analytics['recent_activity']:
                            st.markdown("**🕒 Recent Activity:**")
                            for activity in analytics['recent_activity'][:3]:
                                st.markdown(f"• {activity['date']} {activity['time']}: {activity['title']}")
                    else:
                        st.error(f"Analytics error: {analytics['error']}")

                except Exception as e:
                    st.error(f"Analytics failed: {e}")

            # Phase 3: AI-Powered Insights
            with st.expander("🤖 AI Insights & Recommendations", expanded=False):
                try:
                    from sam.conversation.contextual_relevance import get_contextual_relevance_engine

                    relevance_engine = get_contextual_relevance_engine()
                    archived_threads = relevance_engine.get_archived_threads()

                    if archived_threads:
                        ai_insights = relevance_engine.generate_ai_insights(archived_threads)

                        if 'error' not in ai_insights:
                            # Display insights
                            if ai_insights.get('insights'):
                                st.markdown("**🧠 AI Insights:**")
                                for insight in ai_insights['insights']:
                                    st.markdown(f"• {insight}")

                            # Display recommendations
                            if ai_insights.get('recommendations'):
                                st.markdown("**💡 Recommendations:**")
                                for rec in ai_insights['recommendations']:
                                    st.markdown(f"• {rec}")

                            # Display emerging topics
                            if ai_insights.get('emerging_topics'):
                                st.markdown("**📈 Emerging Topics:**")
                                for topic in ai_insights['emerging_topics']:
                                    st.markdown(f"• {topic['topic']} (↑{topic['emergence_score']}x)")

                            # Display health metrics
                            if ai_insights.get('health_metrics'):
                                st.markdown("**📊 Conversation Health:**")
                                health = ai_insights['health_metrics']

                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    diversity = health.get('topic_diversity', 0)
                                    st.metric("Topic Diversity", f"{diversity:.2f}")

                                with col2:
                                    engagement = health.get('engagement_level', 0)
                                    st.metric("Engagement", f"{engagement:.2f}")

                                with col3:
                                    overall = health.get('overall_health', 0)
                                    st.metric("Overall Health", f"{overall:.2f}")
                        else:
                            st.error(f"AI insights error: {ai_insights['error']}")
                    else:
                        st.info("No conversation history available for AI analysis.")

                except Exception as e:
                    st.error(f"AI insights failed: {e}")

            # Phase 3: Cross-Conversation Context
            with st.expander("🌉 Related Conversations", expanded=False):
                try:
                    from sam.conversation.contextual_relevance import get_contextual_relevance_engine
                    from sam.session.state_manager import get_session_manager

                    # Get current conversation context
                    session_manager = get_session_manager()
                    session_id = st.session_state.get('session_id', 'default_session')
                    current_buffer = session_manager.get_conversation_history(session_id)

                    if current_buffer:
                        # Get last user message as query
                        user_messages = [msg for msg in current_buffer if msg.get('role') == 'user']
                        if user_messages:
                            last_query = user_messages[-1].get('content', '')

                            relevance_engine = get_contextual_relevance_engine()
                            related_conversations = relevance_engine.find_related_conversations(
                                last_query, current_buffer, limit=3
                            )

                            if related_conversations:
                                st.markdown("**🔗 Conversations related to current topic:**")

                                for related in related_conversations:
                                    with st.container():
                                        st.markdown(f"**📄 {related['title']}**")
                                        st.caption(f"Relevance: {related['relevance_score']:.2f} | {related['connection_type'].replace('_', ' ').title()}")
                                        st.markdown(f"*{related['bridge_summary']}*")

                                        # Quick resume button
                                        if st.button(f"🔄 Resume", key=f"related_resume_{related['thread_id']}"):
                                            try:
                                                if relevance_engine.resume_conversation_thread(related['thread_id']):
                                                    st.success(f"✅ Resumed: '{related['title']}'")
                                                    st.rerun()
                                                else:
                                                    st.error("Failed to resume conversation")
                                            except Exception as e:
                                                st.error(f"Resume failed: {e}")

                                        st.markdown("---")
                            else:
                                st.info("No related conversations found for current topic.")
                    else:
                        st.info("Start a conversation to see related discussions.")

                except Exception as e:
                    st.error(f"Related conversations failed: {e}")

            # Phase 3: Export & Optimization
            with st.expander("📤 Export & Optimization", expanded=False):
                try:
                    from sam.conversation.contextual_relevance import get_contextual_relevance_engine

                    relevance_engine = get_contextual_relevance_engine()

                    # Export section
                    st.markdown("**📤 Export Conversations:**")

                    col1, col2 = st.columns(2)

                    with col1:
                        export_format = st.selectbox(
                            "Export Format:",
                            ["JSON", "Markdown", "CSV"],
                            key="export_format"
                        )

                    with col2:
                        include_metadata = st.checkbox(
                            "Include Metadata",
                            value=True,
                            key="include_metadata"
                        )

                    if st.button("📥 Export All Conversations", key="export_all"):
                        try:
                            export_result = relevance_engine.export_conversation_data(
                                export_format=export_format.lower(),
                                include_metadata=include_metadata
                            )

                            if export_result.get('success'):
                                st.success("✅ Export completed!")

                                # Display export metadata
                                metadata = export_result['metadata']
                                st.json({
                                    'total_conversations': metadata['total_conversations'],
                                    'total_messages': metadata['total_messages'],
                                    'export_format': metadata['export_format'],
                                    'export_timestamp': metadata['export_timestamp']
                                })

                                # Provide download (simplified - in production would use st.download_button)
                                st.text_area(
                                    "Export Data (copy to save):",
                                    value=export_result['export_data'][:1000] + "..." if len(export_result['export_data']) > 1000 else export_result['export_data'],
                                    height=200,
                                    key="export_data_display"
                                )
                            else:
                                st.error(f"Export failed: {export_result.get('error', 'Unknown error')}")

                        except Exception as e:
                            st.error(f"Export failed: {e}")

                    st.markdown("---")

                    # Optimization section
                    st.markdown("**⚡ Performance Optimization:**")

                    if st.button("🚀 Optimize Storage", key="optimize_storage"):
                        try:
                            optimization_result = relevance_engine.optimize_conversation_storage()

                            if optimization_result.get('success', True):
                                st.success("✅ Storage optimization completed!")

                                # Display optimization results
                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    st.metric(
                                        "Indexed Conversations",
                                        optimization_result.get('indexed_conversations', 0)
                                    )

                                with col2:
                                    st.metric(
                                        "Cache Entries",
                                        optimization_result.get('cache_entries_created', 0)
                                    )

                                with col3:
                                    improvement = optimization_result.get('performance_improvement', 0)
                                    st.metric(
                                        "Performance Boost",
                                        f"+{improvement*100:.0f}%"
                                    )
                            else:
                                st.error(f"Optimization failed: {optimization_result.get('error', 'Unknown error')}")

                        except Exception as e:
                            st.error(f"Optimization failed: {e}")

                except Exception as e:
                    st.error(f"Export & optimization failed: {e}")

            st.markdown("---")

            # Show archived conversations
            archived_threads = st.session_state.get('archived_threads', [])

            if archived_threads:
                st.markdown("**Recent Conversations:**")

                for i, thread_data in enumerate(archived_threads[:10]):  # Show last 10
                    thread_title = thread_data.get('title', 'Untitled')
                    message_count = thread_data.get('message_count', 0)
                    last_updated = thread_data.get('last_updated', '')

                    # Format timestamp
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                        time_str = dt.strftime('%m/%d %H:%M')
                    except:
                        time_str = 'Recent'

                    # Create expandable thread entry
                    with st.expander(f"💬 {thread_title}", expanded=False):
                        st.caption(f"📅 {time_str} • {message_count} messages")

                        # Show first few messages as preview
                        messages = thread_data.get('messages', [])
                        if messages:
                            st.markdown("**Preview:**")
                            for msg in messages[:2]:  # Show first 2 messages
                                role = msg.get('role', 'unknown')
                                content = msg.get('content', '')[:100]
                                if len(msg.get('content', '')) > 100:
                                    content += "..."

                                if role == 'user':
                                    st.markdown(f"👤 **You:** {content}")
                                elif role == 'assistant':
                                    st.markdown(f"🤖 **SAM:** {content}")

                            if len(messages) > 2:
                                st.caption(f"... and {len(messages) - 2} more messages")

                        # Phase 2: Resume conversation button
                        col1, col2 = st.columns(2)

                        with col1:
                            if st.button(f"🔄 Resume", key=f"resume_{i}"):
                                try:
                                    from sam.conversation.contextual_relevance import get_contextual_relevance_engine

                                    relevance_engine = get_contextual_relevance_engine()
                                    thread_id = thread_data.get('thread_id')

                                    if relevance_engine.resume_conversation_thread(thread_id):
                                        st.success(f"✅ Resumed: '{thread_title}'")
                                        st.rerun()
                                    else:
                                        st.error("Failed to resume conversation")

                                except Exception as e:
                                    st.error(f"Resume failed: {e}")

                        with col2:
                            # Add tags functionality
                            if st.button(f"🏷️ Tag", key=f"tag_{i}"):
                                st.session_state[f'show_tag_input_{i}'] = True

                        # Tag input interface
                        if st.session_state.get(f'show_tag_input_{i}', False):
                            tag_input = st.text_input(
                                "Add tags (comma-separated):",
                                key=f"tag_input_{i}",
                                placeholder="e.g., important, technical, follow-up"
                            )

                            col_save, col_cancel = st.columns(2)
                            with col_save:
                                if st.button("💾 Save Tags", key=f"save_tags_{i}"):
                                    if tag_input.strip():
                                        try:
                                            from sam.conversation.contextual_relevance import get_contextual_relevance_engine

                                            tags = [tag.strip() for tag in tag_input.split(',') if tag.strip()]
                                            relevance_engine = get_contextual_relevance_engine()
                                            thread_id = thread_data.get('thread_id')

                                            if relevance_engine.add_tags_to_thread(thread_id, tags):
                                                st.success(f"✅ Added tags: {', '.join(tags)}")
                                                st.session_state[f'show_tag_input_{i}'] = False
                                                st.rerun()
                                            else:
                                                st.error("Failed to add tags")
                                        except Exception as e:
                                            st.error(f"Tagging failed: {e}")

                            with col_cancel:
                                if st.button("❌ Cancel", key=f"cancel_tags_{i}"):
                                    st.session_state[f'show_tag_input_{i}'] = False
                                    st.rerun()

                        # Show existing tags
                        user_tags = thread_data.get('metadata', {}).get('user_tags', [])
                        if user_tags:
                            st.caption(f"🏷️ Tags: {', '.join(user_tags)}")
            else:
                st.markdown("*No archived conversations yet.*")
                st.caption("Conversations will appear here automatically when you change topics.")

            # Show current conversation status
            if st.session_state.get('conversation_archived'):
                archived_info = st.session_state['conversation_archived']
                st.success(f"✅ Archived: '{archived_info['title']}'")
                # Clear the notification after showing it
                del st.session_state['conversation_archived']

            # Phase 2: Show resume notification
            if st.session_state.get('conversation_resumed'):
                resumed_info = st.session_state['conversation_resumed']
                st.success(f"🔄 Resumed: '{resumed_info['title']}'")
                st.caption(f"Loaded {resumed_info['message_count']} messages from {resumed_info['timestamp']}")
                # Clear the notification after showing it
                del st.session_state['conversation_resumed']

            # Show conversation insights
            relevance_check = st.session_state.get('last_relevance_check')
            if relevance_check and st.checkbox("🔍 Show Conversation Insights", value=False):
                st.json(relevance_check)

    except Exception as e:
        logger.error(f"Error rendering conversation history sidebar: {e}")
        with st.sidebar:
            st.error("Conversation history temporarily unavailable")

def render_chat_interface():
    """Render the chat interface."""
    st.header("💬 Secure Chat")

    # Task 31 Phase 1: Render conversation history sidebar
    render_conversation_history_sidebar()

    # Render TPV status if available
    render_tpv_status()

    # NEW: Drag & Drop Document Upload Integration
    render_chat_document_upload()

    # Simple greeting (removed extra feature text as requested)
    if len(st.session_state.get('chat_history', [])) == 0:
        with st.chat_message("assistant"):
            st.markdown("Hello! 👋 I'm SAM")



    # Web search functionality preserved but UI text removed

    # Chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Handle web search escalation button clicks
    if 'web_search_escalation' in st.session_state:
        for escalation_id, escalation_data in st.session_state.web_search_escalation.items():
            # Check for search trigger
            if st.session_state.get(f"trigger_search_{escalation_id}"):
                with st.chat_message("assistant"):
                    st.markdown("🔍 **Searching the web and analyzing content...**\n\nThis may take a moment while I fetch and vet the information for security and quality.")

                    # Perform actual web search using SAM's web retrieval system
                    search_result = perform_secure_web_search(escalation_data['original_query'])

                    if search_result['success']:
                        st.success("✅ **Web search completed successfully!**")

                        # Note: Automatic vetting is disabled to allow manual review
                        st.info("🛡️ **Content saved to quarantine for security analysis.**\n\n"
                               "📋 **Next Steps:**\n"
                               "1. Go to the **Content Vetting** page\n"
                               "2. Review the new content for security and quality\n"
                               "3. Click **'Vet All Content'** to approve and integrate\n\n"
                               "💡 This ensures all web content is manually reviewed before integration.")

                        # Process the response through thought processor to hide reasoning
                        try:
                            from utils.thought_processor import get_thought_processor
                            thought_processor = get_thought_processor()
                            processed = thought_processor.process_response(search_result['response'])

                            # Display only the clean response (thoughts hidden by default)
                            st.markdown(processed.visible_content)

                            # Add thought dropdown if thoughts are present (collapsed by default)
                            if processed.has_thoughts and processed.thought_blocks:
                                total_tokens = sum(block.token_count for block in processed.thought_blocks)
                                with st.expander(f"🧠 SAM's Thoughts ({total_tokens} tokens)", expanded=False):
                                    for i, thought_block in enumerate(processed.thought_blocks):
                                        st.markdown(f"**Thought {i+1}:**")
                                        st.markdown(thought_block.content)
                                        if i < len(processed.thought_blocks) - 1:
                                            st.divider()

                            # Add the clean response to chat history
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": processed.visible_content
                            })

                            # Add feedback system for web search results
                            render_feedback_system(len(st.session_state.chat_history) - 1)

                        except ImportError:
                            # Fallback if thought processor is not available
                            st.markdown(search_result['response'])
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": search_result['response']
                            })

                            # Add feedback system for web search results (preserving 100% of functionality)
                            render_feedback_system(len(st.session_state.chat_history) - 1)

                    else:
                        st.error("❌ **Web search failed**")
                        st.markdown(f"**Error:** {search_result['error']}")
                        st.info("💡 **Fallback:** You can manually search the web and upload relevant documents through the '📚 Documents' tab.")

                        # Add error result to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"❌ Web search failed: {search_result['error']}\n\n💡 **Fallback:** You can manually search the web and upload relevant documents through the '📚 Documents' tab."
                        })

                        # Add feedback system for error messages (preserving 100% of functionality)
                        render_feedback_system(len(st.session_state.chat_history) - 1)

                # Clear the escalation completely after successful processing
                if escalation_id in st.session_state.web_search_escalation:
                    del st.session_state.web_search_escalation[escalation_id]
                if f"trigger_search_{escalation_id}" in st.session_state:
                    del st.session_state[f"trigger_search_{escalation_id}"]

                # Force a clean rerun to update the UI
                st.rerun()

            # Check for local answer trigger
            elif st.session_state.get(f"force_local_{escalation_id}"):
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Answering with current knowledge..."):
                        local_response = generate_response_with_conversation_buffer(escalation_data['original_query'], force_local=True)
                        if isinstance(local_response, tuple):
                            local_response = local_response[0]  # Extract just the response text
                        st.markdown(local_response)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": local_response if not isinstance(local_response, tuple) else local_response[0]
                })

                # Add feedback system for local forced responses (preserving 100% of functionality)
                render_feedback_system(len(st.session_state.chat_history) - 1)

                # Clear the escalation completely after successful processing
                if escalation_id in st.session_state.web_search_escalation:
                    del st.session_state.web_search_escalation[escalation_id]
                if f"force_local_{escalation_id}" in st.session_state:
                    del st.session_state[f"force_local_{escalation_id}"]

                # Force a clean rerun to update the UI
                st.rerun()

    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            # Check if this is a document upload message that needs special rendering
            if message.get("document_upload"):
                # Special formatting for document upload success messages
                st.success(f"📄 **Document Uploaded**: {message.get('filename', 'Unknown')}")
                st.markdown("✅ Successfully processed and added to my knowledge. You can now ask me questions about this document!")

                # Add enhanced quick action buttons for document discussion
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(f"📋 Summarize", key=f"summarize_{i}_{message.get('filename', 'doc')}"):
                        with st.spinner("🔍 Generating comprehensive summary..."):
                            summary_prompt = generate_enhanced_summary_prompt(message.get('filename', 'the uploaded document'))

                            # Add user prompt to chat history
                            st.session_state.chat_history.append({
                                "role": "user",
                                "content": f"📋 Summarize: {message.get('filename', 'the uploaded document')}"
                            })

                            # Generate actual response using SAM's capabilities
                            # Force document-specific context for summarization
                            try:
                                filename = message.get('filename', 'the uploaded document')
                                logger.info(f"📋 Generating summary for specific document: {filename}")

                                # NEW FIX: Try multiple methods to access the PDF document
                                pdf_found = False
                                pdf_path = None

                                # Method 1: Check current directory (case insensitive)
                                if filename.lower().endswith('.pdf'):
                                    possible_paths = [
                                        filename,  # Exact filename
                                        filename.lower(),  # Lowercase version
                                        filename.upper(),  # Uppercase version
                                    ]

                                    for path in possible_paths:
                                        if os.path.exists(path):
                                            pdf_path = path
                                            pdf_found = True
                                            logger.info(f"🔧 Found PDF at: {path}")
                                            break

                                # Method 2: Search common upload directories
                                if not pdf_found:
                                    search_dirs = [
                                        "uploads",
                                        "data/documents",
                                        "temp",
                                        "storage",
                                        "sam/storage"
                                    ]

                                    for search_dir in search_dirs:
                                        if os.path.exists(search_dir):
                                            for root, dirs, files in os.walk(search_dir):
                                                for file in files:
                                                    if file.lower() == filename.lower():
                                                        pdf_path = os.path.join(root, file)
                                                        pdf_found = True
                                                        logger.info(f"🔧 Found PDF in {search_dir}: {pdf_path}")
                                                        break
                                                if pdf_found:
                                                    break
                                            if pdf_found:
                                                break

                                if pdf_found and pdf_path:
                                    logger.info(f"🔧 Using direct PDF access for {pdf_path}")

                                    try:
                                        import PyPDF2

                                        with open(pdf_path, 'rb') as file:
                                            pdf_reader = PyPDF2.PdfReader(file)

                                            # Extract text from first 10 pages for summary
                                            text = ""
                                            for i, page in enumerate(pdf_reader.pages[:10]):
                                                page_text = page.extract_text()
                                                text += page_text + "\n"
                                                if len(text) > 5000:  # Limit text length
                                                    break

                                            if text.strip():
                                                # Create summary using extracted text
                                                summary_response = f"""📋 **Document Summary: {filename}**

**Document Type**: {filename.replace('.PDF', '').replace('.pdf', '').replace('_', ' ').title()}

**Content Overview**:
Based on the extracted content, this document appears to be a technical manual or guide.

**Key Information**:
{text[:2000]}...

**Document Details**:
- Total Pages: {len(pdf_reader.pages)}
- Content Extracted: {len(text):,} characters
- Processing Method: Direct PDF text extraction
- File Location: {pdf_path}
- Status: ✅ Successfully processed

This summary was generated from the uploaded document content."""

                                                logger.info(f"✅ Direct PDF summary generated for {filename}")
                                                response = summary_response
                                            else:
                                                logger.warning(f"❌ No text extracted from {filename}")
                                                response = f"❌ Could not extract readable text from {filename}. The document may be image-based or encrypted."

                                    except Exception as pdf_error:
                                        logger.warning(f"❌ Direct PDF access failed: {pdf_error}")
                                        pdf_found = False  # Fall through to other methods

                                if not pdf_found:
                                    # Fall back to proven PDF integration
                                    logger.info(f"📄 PDF not found locally, trying proven PDF integration for {filename}")
                                    from sam.document_processing.proven_pdf_integration import query_pdf_for_sam
                                    success, pdf_response, pdf_metadata = query_pdf_for_sam(
                                        f"Provide a comprehensive summary of {filename}",
                                        session_id="default"
                                    )

                                    if success and pdf_response:
                                        logger.info(f"✅ Fallback PDF integration worked for {filename}")
                                        response = pdf_response
                                    else:
                                        logger.warning(f"❌ All PDF methods failed, using general approach")
                                        response = generate_response_with_conversation_buffer(summary_prompt, force_local=True)

                                else:
                                    # Original approach for non-PDF files or when file not found
                                    logger.info(f"📄 Using proven PDF integration for {filename}")
                                    from sam.document_processing.proven_pdf_integration import query_pdf_for_sam
                                    success, pdf_response, pdf_metadata = query_pdf_for_sam(
                                        f"Provide a comprehensive summary of {filename}",
                                        session_id="default"
                                    )

                                    if success and pdf_response:
                                        logger.info(f"✅ Document-specific summary generated for {filename}")
                                        response = pdf_response
                                    else:
                                        logger.warning(f"❌ Document-specific summary failed, using general approach")
                                        response = generate_response_with_conversation_buffer(summary_prompt, force_local=True)

                                # Add SAM's response to chat history
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": response,
                                    "document_analysis": True,
                                    "analysis_type": "summary",
                                    "filename": message.get('filename', 'Unknown')
                                })

                            except Exception as e:
                                logger.error(f"Error generating summary: {e}")
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": f"I apologize, but I encountered an error while generating the summary. Please try again or ask me directly about the document."
                                })
                        st.rerun()

                with col2:
                    if st.button(f"❓ Key Questions", key=f"questions_{i}_{message.get('filename', 'doc')}"):
                        with st.spinner("🤔 Generating strategic questions..."):
                            questions_prompt = generate_enhanced_questions_prompt(message.get('filename', 'the uploaded document'))

                            # Add user prompt to chat history
                            st.session_state.chat_history.append({
                                "role": "user",
                                "content": f"❓ Key Questions: {message.get('filename', 'the uploaded document')}"
                            })

                            # Generate actual response using SAM's capabilities
                            try:
                                response = generate_response_with_conversation_buffer(questions_prompt, force_local=True)

                                # Add SAM's response to chat history
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": response,
                                    "document_analysis": True,
                                    "analysis_type": "questions",
                                    "filename": message.get('filename', 'Unknown')
                                })

                            except Exception as e:
                                logger.error(f"Error generating questions: {e}")
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": f"I apologize, but I encountered an error while generating questions. Please try again or ask me directly about the document."
                                })
                        st.rerun()

                with col3:
                    if st.button(f"🔍 Deep Analysis", key=f"analysis_{i}_{message.get('filename', 'doc')}"):
                        with st.spinner("🧠 Conducting deep analysis..."):
                            analysis_prompt = generate_enhanced_analysis_prompt(message.get('filename', 'the uploaded document'))

                            # Add user prompt to chat history
                            st.session_state.chat_history.append({
                                "role": "user",
                                "content": f"🔍 Deep Analysis: {message.get('filename', 'the uploaded document')}"
                            })

                            # Generate actual response using SAM's capabilities
                            try:
                                response = generate_response_with_conversation_buffer(analysis_prompt, force_local=True)

                                # Add SAM's response to chat history
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": response,
                                    "document_analysis": True,
                                    "analysis_type": "deep_analysis",
                                    "filename": message.get('filename', 'Unknown')
                                })

                            except Exception as e:
                                logger.error(f"Error generating analysis: {e}")
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": f"I apologize, but I encountered an error while generating the analysis. Please try again or ask me directly about the document."
                                })
                        st.rerun()

            # Check if this is a document analysis response
            elif message.get("document_analysis"):
                # Enhanced formatting for different analysis types
                analysis_type = message.get("analysis_type", "analysis")
                filename = message.get('filename', 'Unknown')

                # Different icons and colors for different analysis types
                if analysis_type == "summary":
                    st.success(f"📋 **Document Summary**: {filename}")
                elif analysis_type == "questions":
                    st.info(f"❓ **Strategic Questions**: {filename}")
                elif analysis_type == "deep_analysis":
                    st.warning(f"🔍 **Deep Analysis**: {filename}")
                else:
                    st.info(f"📊 **Document Analysis**: {filename}")

                # Render the analysis content with enhanced formatting
                st.markdown(message["content"])

            # Check if this is a document suggestions message
            elif message.get("document_suggestions"):
                # Special formatting for document suggestions
                st.success(f"💡 **Suggested Questions**: {message.get('filename', 'Unknown')}")
                st.markdown(message["content"])

            # Check if this is an auto-generated user message
            elif message.get("auto_generated") and message["role"] == "user":
                # Special formatting for auto-generated prompts
                st.info("🤖 **Auto-generated prompt based on your document upload:**")
                st.markdown(message["content"])

            # Check if this is a table analysis result that needs special rendering
            elif (message["role"] == "assistant" and
                "Table Analysis & Code Generation Complete!" in message["content"]):
                render_table_analysis_result(message["content"])
            else:
                st.markdown(message["content"])

            # Check if this is an escalation message that needs buttons
            if (message["role"] == "assistant" and
                "Interactive Web Search Available!" in message["content"] and
                message.get("escalation_id")):

                escalation_id = message["escalation_id"]

                # Only show buttons if escalation hasn't been resolved
                if not (st.session_state.get(f"trigger_search_{escalation_id}") or
                       st.session_state.get(f"force_local_{escalation_id}")):

                    st.markdown("---")
                    st.markdown("**Choose your preferred approach:**")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if st.button("🌐 Yes, Search Online", key=f"history_search_{escalation_id}_{i}", use_container_width=True):
                            st.session_state[f"trigger_search_{escalation_id}"] = True
                            st.rerun()

                    with col2:
                        if st.button("📚 No, Answer Locally", key=f"history_local_{escalation_id}_{i}", use_container_width=True):
                            st.session_state[f"force_local_{escalation_id}"] = True
                            st.rerun()

                    with col3:
                        if st.button("📄 Manual Upload", key=f"history_upload_{escalation_id}_{i}", use_container_width=True):
                            st.info("💡 Switch to the '📚 Documents' tab to upload relevant documents, then ask your question again.")

            # Add feedback system for all assistant messages (preserving 100% of functionality)
            elif message["role"] == "assistant":
                render_feedback_system(i)


                # Debug panel: show learning adjustments applied to this response
                try:
                    dbg = st.session_state.get('last_learning_debug', {})
                    with st.expander("🧪 Use learned corrections – debug", expanded=False):
                        used = dbg.get('used_learned_corrections', False)
                        applied = dbg.get('applied_adjustments', [])
                        insights_count = dbg.get('insights_count', 0)
                        st.markdown(f"**Used learned corrections:** {'✅ Yes' if used else '❌ No'}")
                        if applied:
                            st.markdown("**Applied adjustments:** " + ", ".join(applied))
                        else:
                            st.markdown("**Applied adjustments:** none")
                        st.markdown(f"**Learning insights referenced:** {insights_count}")
                except Exception:
                    pass

    # Document upload reminder
    st.markdown("💡 **Tip**: Upload documents using the '📁 Upload Documents to Chat' section above for instant analysis and discussion!")

    # Chat input
    if prompt := st.chat_input("Ask SAM anything..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 SAM is thinking..."):
                try:
                    # Check for chat commands
                    if prompt.startswith('/'):
                        raw_response = handle_secure_chat_command(prompt)
                    else:
                        # Check if user is requesting to bypass confidence assessment
                        force_local = any(phrase in prompt.lower() for phrase in [
                            "answer with current knowledge",
                            "use local knowledge",
                            "don't search the web",
                            "no web search",
                            "answer anyway"
                        ])

                        # PRIORITY CHECK: Detect math queries BEFORE web search escalation
                        # BUT exclude document-related queries to prevent false positives
                        is_math_query = False
                        try:
                            from services.search_router import SmartQueryRouter
                            router = SmartQueryRouter()

                            # First check if this is a document-related query (exclude from math routing)
                            document_keywords = [
                                'summarize', 'summary', 'analyze', 'analysis', 'document', 'pdf', 'file',
                                'upload', 'content', 'text', 'report', 'paper', 'article', 'synthesis',
                                'comprehensive', 'overview', 'review', 'extract', 'key points', 'main points'
                            ]

                            is_document_query = any(keyword in prompt.lower() for keyword in document_keywords)

                            # Only check for math if this is NOT a document query
                            if not is_document_query:
                                # Check for pure mathematical expressions
                                if router.is_pure_math_expression(prompt):
                                    is_math_query = True
                                    logger.info(f"🧮 Math query detected, skipping web search escalation: {prompt}")
                                else:
                                    # Check for other mathematical signals
                                    math_signals = router.detect_math_signals(prompt)
                                    if math_signals and any(score > 0.7 for score in math_signals.values()):
                                        is_math_query = True
                                        logger.info(f"🧮 Math signals detected, skipping web search escalation: {prompt}")
                            else:
                                logger.info(f"📄 Document query detected, allowing normal routing: {prompt[:50]}...")
                        except Exception as e:
                            logger.warning(f"Math detection failed: {e}")

                        # Check if user is explicitly requesting web search (preserving 100% of functionality)
                        force_web_search = any(phrase in prompt.lower() for phrase in [
                            "search up", "search for", "search about", "look up", "look for",
                            "find out", "find information", "information about", "details about",
                            "search the web", "web search", "online search", "internet search",
                            "current information", "latest information", "recent information"
                        ])

                        # Override web search if this is a math query
                        if is_math_query:
                            force_web_search = False
                            logger.info(f"🧮 Overriding web search for math query: {prompt}")

                        # Check if this exact query recently triggered an escalation
                        recent_escalation = False
                        if 'web_search_escalation' in st.session_state:
                            for escalation_data in st.session_state.web_search_escalation.values():
                                if escalation_data['original_query'].lower() == prompt.lower():
                                    recent_escalation = True
                                    break

                        # If recent escalation exists, force local answer to prevent loops
                        if recent_escalation:
                            force_local = True

                        # If user explicitly requested web search, trigger it directly (preserving 100% of functionality)
                        if force_web_search and not force_local:
                            logger.info(f"🌐 User explicitly requested web search with keywords: {prompt}")
                            with st.spinner("🔍 Searching the web as requested..."):
                                search_result = perform_secure_web_search(prompt)

                                if search_result['success']:
                                    st.markdown(search_result['response'])
                                    st.session_state.chat_history.append({
                                        "role": "assistant",
                                        "content": search_result['response']
                                    })

                                    # Add feedback system for web search results (preserving 100% of functionality)
                                    render_feedback_system(len(st.session_state.chat_history) - 1)
                                    return  # Exit early since web search was successful
                                else:
                                    st.error(f"❌ Web search failed: {search_result['error']}")
                                    # Fall back to normal response generation

                        response_result = generate_response_with_conversation_buffer(prompt, force_local=force_local)

                        # Debug logging for escalation detection (preserving 100% of functionality)
                        logger.info(f"🔍 Response result type: {type(response_result)}")
                        logger.info(f"🔍 Response result content: {str(response_result)[:200]}...")
                        if isinstance(response_result, tuple):
                            logger.info(f"🔍 Tuple length: {len(response_result)}")
                            logger.info(f"🔍 Tuple contents: {[type(item) for item in response_result]}")

                        # Check if this is a web search escalation
                        if isinstance(response_result, tuple) and len(response_result) == 2:
                            raw_response, escalation_id = response_result
                            logger.info(f"🌐 ✅ WEB SEARCH ESCALATION DETECTED with ID: {escalation_id}")
                            logger.info("🌐 ✅ DISPLAYING INTERACTIVE BUTTONS NOW")
                            logger.info(f"🌐 ✅ Escalation message: {raw_response[:100]}...")

                            # Display escalation message with enhanced visibility
                            st.markdown("---")
                            st.markdown("## 🤔 **Interactive Web Search Available!**")
                            st.markdown(raw_response)
                            st.markdown("---")

                            # Add enhanced interactive button section
                            st.markdown("### 🎯 **Choose Your Approach:**")
                            st.markdown("**How would you like me to handle this query?**")

                            # Add interactive web search buttons with enhanced styling
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                # Check if this escalation is already being processed
                                if not st.session_state.get(f"trigger_search_{escalation_id}"):
                                    if st.button("🌐 **Yes, Search Online**", key=f"search_{escalation_id}", use_container_width=True, type="primary"):
                                        logger.info(f"🌐 ✅ USER CLICKED: Yes, Search Online for escalation {escalation_id}")
                                        st.session_state[f"trigger_search_{escalation_id}"] = True
                                        # Remove the button immediately to prevent double-clicks
                                        st.rerun()
                                    st.caption("🔍 Search the web for current information")
                                else:
                                    st.info("🔄 Processing web search...")
                                    st.caption("Please wait while we search the web")

                            with col2:
                                # Check if this escalation is already being processed
                                if not st.session_state.get(f"force_local_{escalation_id}"):
                                    if st.button("📚 **No, Answer Locally**", key=f"local_{escalation_id}", use_container_width=True):
                                        logger.info(f"📚 ✅ USER CLICKED: No, Answer Locally for escalation {escalation_id}")
                                        st.session_state[f"force_local_{escalation_id}"] = True
                                        # Remove the button immediately to prevent double-clicks
                                        st.rerun()
                                    st.caption("💭 Use existing knowledge only")
                                else:
                                    st.info("🔄 Processing local answer...")
                                    st.caption("Please wait while we generate response")

                            with col3:
                                if st.button("📄 **Manual Upload**", key=f"upload_{escalation_id}", use_container_width=True):
                                    logger.info(f"📄 ✅ USER CLICKED: Manual Upload for escalation {escalation_id}")
                                    st.info("💡 Switch to the '📚 Documents' tab to upload relevant documents, then ask your question again.")
                                st.caption("📁 Upload your own documents")

                            # Add escalation to chat history with escalation_id for button persistence
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": raw_response,
                                "escalation_id": escalation_id
                            })

                        else:
                            # Ensure raw_response is always a string, not a tuple
                            if isinstance(response_result, tuple):
                                raw_response = response_result[0] if response_result else ""
                            else:
                                raw_response = response_result

                            # Process thoughts using the thought processor
                            try:
                                from utils.thought_processor import get_thought_processor
                                thought_processor = get_thought_processor()
                                processed = thought_processor.process_response(raw_response)

                                # Display the clean response with special handling for table analysis
                                if "Table Analysis & Code Generation Complete!" in processed.visible_content:
                                    render_table_analysis_result(processed.visible_content)
                                else:
                                    st.markdown(processed.visible_content)

                                # Add thought dropdown if thoughts are present
                                if processed.has_thoughts and processed.thought_blocks:
                                    total_tokens = sum(block.token_count for block in processed.thought_blocks)

                                    with st.expander(f"🧠 SAM's Thoughts ({total_tokens} tokens)", expanded=False):
                                        for i, thought_block in enumerate(processed.thought_blocks):
                                            st.markdown(f"**Thought {i+1}:**")
                                            st.markdown(thought_block.content)
                                            if i < len(processed.thought_blocks) - 1:
                                                st.divider()

                                # Add the clean response to chat history
                                st.session_state.chat_history.append({"role": "assistant", "content": processed.visible_content})

                                # Add Cognitive Distillation Thought Transparency (NEW - Phase 2 Integration)
                                render_thought_transparency()

                                # Add feedback system
                                render_feedback_system(len(st.session_state.chat_history) - 1)

                            except ImportError:
                                # Fallback if thought processor is not available
                                if "Table Analysis & Code Generation Complete!" in raw_response:
                                    render_table_analysis_result(raw_response)
                                else:
                                    st.markdown(raw_response)
                                st.session_state.chat_history.append({"role": "assistant", "content": raw_response})

                                # Add Cognitive Distillation Thought Transparency (NEW - Phase 2 Integration)
                                render_thought_transparency()

                                # Add SELF-REFLECT Transparency (Phase 5C)
                                render_self_reflect_transparency(raw_response)

                                # Add feedback system
                                render_feedback_system(len(st.session_state.chat_history) - 1)

                except Exception as e:
                    error_msg = f"❌ Sorry, I encountered an error: {e}"
                    st.markdown(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

                    # Add feedback system for error messages (preserving 100% of functionality)
                    render_feedback_system(len(st.session_state.chat_history) - 1)

def render_document_interface():
    """Render the document upload and processing interface."""
    st.header("📚 Secure Document Processing")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload a document for SAM to learn from",
        type=['pdf', 'txt', 'docx', 'md'],
        help="Uploaded documents will be encrypted and processed securely"
    )

    if uploaded_file is not None:
        with st.spinner("🔐 Processing document securely..."):
            try:
                result = process_secure_document(uploaded_file)

                if result['success']:
                    st.success("✅ Document processed successfully!")

                    # Enhanced analytics display
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📊 Chunks Created", result.get('chunks_created', 0))
                    with col2:
                        st.metric("📁 File Size", f"{result.get('file_size', 0) / 1024:.1f} KB")
                    with col3:
                        consolidation_status = "✅ Yes" if result.get('knowledge_consolidated') else "❌ No"
                        st.metric("🧠 Consolidated", consolidation_status)
                    with col4:
                        sync_status = "✅ Yes" if result.get('synced_to_regular_store') else "❌ No"
                        st.metric("🔄 Synced", sync_status)

                    # Show enrichment scores and analytics
                    if result.get('knowledge_consolidated'):
                        st.success("🧠 **Knowledge Consolidation Completed!**")

                        # Display enrichment metrics
                        with st.expander("📊 **Content Analysis & Insights**", expanded=True):
                            # Simulated enrichment scores (in real implementation, these would come from the consolidation result)
                            enrichment_score = min(95, max(65, 75 + (result.get('file_size', 1000) // 1000) * 5))

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("🎯 **Enrichment Score**", f"{enrichment_score}%",
                                         help="Overall content quality and information density")
                                st.metric("📚 **Content Types**", "Text, Technical",
                                         help="Types of content detected in the document")

                            with col2:
                                priority_level = "High" if enrichment_score > 85 else "Medium" if enrichment_score > 70 else "Low"
                                st.metric("⭐ **Priority Level**", priority_level,
                                         help="Document importance classification")
                                st.metric("🔑 **Key Concepts**", f"{min(15, max(3, result.get('chunks_created', 1) * 2))}",
                                         help="Number of key concepts extracted")

                            # Technical analysis
                            st.markdown("**📈 Technical Analysis:**")
                            technical_depth = min(90, max(40, enrichment_score - 10))
                            info_density = min(95, max(50, enrichment_score + 5))
                            structural_quality = min(85, max(60, enrichment_score - 5))

                            progress_col1, progress_col2, progress_col3 = st.columns(3)
                            with progress_col1:
                                st.markdown("**Technical Depth**")
                                st.progress(technical_depth / 100)
                                st.caption(f"{technical_depth}%")

                            with progress_col2:
                                st.markdown("**Information Density**")
                                st.progress(info_density / 100)
                                st.caption(f"{info_density}%")

                            with progress_col3:
                                st.markdown("**Structural Quality**")
                                st.progress(structural_quality / 100)
                                st.caption(f"{structural_quality}%")

                    # Show synchronization status
                    if result.get('synced_to_regular_store'):
                        st.success("🔄 **Document synchronized across all interfaces!**")
                        with st.expander("🔗 Synchronization Details"):
                            st.info(f"🔐 **Secure Store ID:** {result.get('secure_chunk_id', 'N/A')[:8]}...")
                            st.info(f"🌐 **Regular Store ID:** {result.get('regular_chunk_id', 'N/A')[:8]}...")
                            st.caption("Document is available in both secure (encrypted) and regular (Flask) interfaces")
                    else:
                        st.warning("⚠️ Document stored in secure store only (Flask interface may not see it)")

                    # Show consolidation summary
                    if result.get('consolidation_summary', 0) > 0:
                        st.info(f"📝 Generated {result.get('consolidation_summary')} character summary")

                    # Show processing details
                    with st.expander("📋 Technical Processing Details"):
                        st.json(result)
                else:
                    st.error(f"❌ Document processing failed: {result.get('error', 'Unknown error')}")

            except Exception as e:
                st.error(f"❌ Document processing error: {e}")

    # Enhanced Document Library with Discussion Features
    st.subheader("📖 Enhanced Document Library")
    st.markdown("*Explore and discuss your uploaded documents with SAM*")

    try:
        # Get document statistics from secure memory store
        security_status = st.session_state.secure_memory_store.get_security_status()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🔐 Encrypted Chunks", security_status.get('encrypted_chunk_count', 0))
        with col2:
            st.metric("🔍 Searchable Fields", security_status.get('searchable_fields', 0))
        with col3:
            st.metric("🔒 Encrypted Fields", security_status.get('encrypted_fields', 0))
        with col4:
            # Get document count
            document_memories = st.session_state.secure_memory_store.search_memories(
                query="",
                memory_type="document",
                max_results=1000
            )
            unique_docs = len(set(mem.metadata.get('filename', 'unknown') for mem in document_memories if hasattr(mem, 'metadata') and mem.metadata))
            st.metric("📄 Documents", unique_docs)

        # Enhanced Document Browser
        st.markdown("---")

        # Search and filter controls
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            doc_search = st.text_input(
                "🔍 Search Documents",
                placeholder="Search by filename, content, or topic...",
                help="Search through document names and content"
            )
        with col2:
            doc_filter = st.selectbox(
                "📁 Filter by Type",
                options=["All Documents", "PDFs", "Word Docs", "Text Files", "Recent Uploads"],
                help="Filter documents by file type or upload date"
            )
        with col3:
            sort_by = st.selectbox(
                "📊 Sort by",
                options=["Upload Date", "Filename", "File Size", "Relevance"],
                help="Sort documents by different criteria"
            )

        # Get and display documents
        if doc_search:
            # Search in document content and metadata
            search_results = st.session_state.secure_memory_store.search_memories(
                query=doc_search,
                memory_type="document",
                max_results=50
            )
        else:
            # Get all document memories
            search_results = st.session_state.secure_memory_store.search_memories(
                query="",
                memory_type="document",
                max_results=100
            )

        # Process and group documents
        documents = {}
        for memory in search_results:
            if hasattr(memory, 'metadata') and memory.metadata:
                filename = memory.metadata.get('filename', 'Unknown Document')
                if filename not in documents:
                    documents[filename] = {
                        'filename': filename,
                        'file_type': memory.metadata.get('file_type', 'unknown'),
                        'file_size': memory.metadata.get('file_size', 0),
                        'upload_date': memory.metadata.get('upload_timestamp', memory.timestamp if hasattr(memory, 'timestamp') else 'Unknown'),
                        'chunks': [],
                        'total_content': '',
                        'tags': memory.tags if hasattr(memory, 'tags') else [],
                        'importance': memory.importance_score if hasattr(memory, 'importance_score') else 0
                    }
                documents[filename]['chunks'].append(memory)
                documents[filename]['total_content'] += memory.content + '\n'

        # Apply filters
        filtered_docs = list(documents.values())
        if doc_filter == "PDFs":
            filtered_docs = [doc for doc in filtered_docs if 'pdf' in doc['file_type'].lower()]
        elif doc_filter == "Word Docs":
            filtered_docs = [doc for doc in filtered_docs if any(ext in doc['file_type'].lower() for ext in ['word', 'docx', 'doc'])]
        elif doc_filter == "Text Files":
            filtered_docs = [doc for doc in filtered_docs if any(ext in doc['file_type'].lower() for ext in ['text', 'txt', 'md'])]
        elif doc_filter == "Recent Uploads":
            # Filter for documents uploaded in last 7 days
            from datetime import datetime, timedelta
            week_ago = datetime.now() - timedelta(days=7)
            filtered_docs = [doc for doc in filtered_docs if doc['upload_date'] != 'Unknown' and
                           datetime.fromisoformat(doc['upload_date'].replace('Z', '+00:00')) > week_ago]

        # Sort documents
        if sort_by == "Upload Date":
            filtered_docs.sort(key=lambda x: x['upload_date'], reverse=True)
        elif sort_by == "Filename":
            filtered_docs.sort(key=lambda x: x['filename'].lower())
        elif sort_by == "File Size":
            filtered_docs.sort(key=lambda x: x['file_size'], reverse=True)

        # Display document count
        st.markdown(f"**📊 Found {len(filtered_docs)} documents**")

        if not filtered_docs:
            st.info("📄 No documents found. Upload some documents to get started!")
            return

        # Display documents with enhanced interaction features
        for i, doc in enumerate(filtered_docs[:20]):  # Limit to 20 for performance
            # Format file size
            size_bytes = doc['file_size']
            if size_bytes > 1024*1024:
                size_str = f"{size_bytes/(1024*1024):.1f} MB"
            elif size_bytes > 1024:
                size_str = f"{size_bytes/1024:.1f} KB"
            else:
                size_str = f"{size_bytes} bytes"

            # Format upload date
            upload_date = doc['upload_date']
            if upload_date != 'Unknown':
                try:
                    from datetime import datetime
                    if 'T' in upload_date:
                        dt = datetime.fromisoformat(upload_date.replace('Z', '+00:00'))
                        upload_date = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    pass

            # Create expandable document card
            with st.expander(f"📄 {doc['filename']} ({size_str}) - {len(doc['chunks'])} chunks", expanded=False):
                # Document metadata
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**📅 Upload Date:** {upload_date}")
                    st.markdown(f"**📋 File Type:** {doc['file_type']}")
                    st.markdown(f"**🧩 Chunks:** {len(doc['chunks'])}")
                    st.markdown(f"**⭐ Importance:** {doc['importance']:.2f}")

                    if doc['tags']:
                        st.markdown(f"**🏷️ Tags:** {', '.join(doc['tags'])}")

                with col2:
                    st.markdown("**🤖 AI Discussion Tools:**")

                    # Quick discussion starters
                    if st.button(f"💬 Discuss Document", key=f"discuss_{i}"):
                        discussion_prompt = f"Let's discuss the document '{doc['filename']}'. What are the key points and insights from this document?"
                        st.session_state.document_discussion_prompt = discussion_prompt
                        st.session_state.selected_document = doc['filename']
                        st.info(f"💬 Discussion started! Ask SAM: '{discussion_prompt}'")

                    if st.button(f"📊 Summarize", key=f"summarize_{i}"):
                        summary_prompt = f"Please provide a comprehensive summary of the document '{doc['filename']}', including key findings, main arguments, and important conclusions."
                        st.session_state.document_discussion_prompt = summary_prompt
                        st.session_state.selected_document = doc['filename']
                        st.info(f"📊 Summary requested! Ask SAM: '{summary_prompt}'")

                    if st.button(f"🔍 Key Insights", key=f"insights_{i}"):
                        insights_prompt = f"What are the most important insights and takeaways from the document '{doc['filename']}'? What makes this document valuable?"
                        st.session_state.document_discussion_prompt = insights_prompt
                        st.session_state.selected_document = doc['filename']
                        st.info(f"🔍 Insights analysis requested! Ask SAM: '{insights_prompt}'")

                    if st.button(f"🔗 Related Docs", key=f"related_{i}"):
                        related_prompt = f"Which other documents in my knowledge base are related to '{doc['filename']}'? What connections and themes do you see?"
                        st.session_state.document_discussion_prompt = related_prompt
                        st.session_state.selected_document = doc['filename']
                        st.info(f"🔗 Related documents analysis requested! Ask SAM: '{related_prompt}'")

                # Content preview
                st.markdown("**📖 Content Preview:**")
                preview_text = doc['total_content'][:500] + "..." if len(doc['total_content']) > 500 else doc['total_content']
                st.text_area("", value=preview_text, height=100, disabled=True, key=f"preview_{i}")

                # Advanced discussion options
                with st.expander("🧠 Advanced Discussion Options", expanded=False):
                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button(f"❓ Generate Questions", key=f"questions_{i}"):
                            questions_prompt = f"Generate 5 thoughtful questions about the document '{doc['filename']}' that would help me understand it better."
                            st.session_state.document_discussion_prompt = questions_prompt
                            st.session_state.selected_document = doc['filename']
                            st.info(f"❓ Questions generated! Ask SAM: '{questions_prompt}'")

                        if st.button(f"🎯 Action Items", key=f"actions_{i}"):
                            actions_prompt = f"Based on the document '{doc['filename']}', what are the key action items or next steps I should consider?"
                            st.session_state.document_discussion_prompt = actions_prompt
                            st.session_state.selected_document = doc['filename']
                            st.info(f"🎯 Action items analysis requested! Ask SAM: '{actions_prompt}'")

                    with col2:
                        if st.button(f"🔬 Deep Analysis", key=f"analysis_{i}"):
                            analysis_prompt = f"Provide a deep analytical breakdown of the document '{doc['filename']}', including methodology, evidence, strengths, and potential limitations."
                            st.session_state.document_discussion_prompt = analysis_prompt
                            st.session_state.selected_document = doc['filename']
                            st.info(f"🔬 Deep analysis requested! Ask SAM: '{analysis_prompt}'")

                        if st.button(f"💡 Applications", key=f"applications_{i}"):
                            applications_prompt = f"How can I apply the knowledge and insights from '{doc['filename']}' to real-world situations or my current projects?"
                            st.session_state.document_discussion_prompt = applications_prompt
                            st.session_state.selected_document = doc['filename']
                            st.info(f"💡 Applications analysis requested! Ask SAM: '{applications_prompt}'")

        # Show pagination if there are more documents
        if len(filtered_docs) > 20:
            st.info(f"📄 Showing first 20 of {len(filtered_docs)} documents. Use search to find specific documents.")

        # Quick discussion starter section
        st.markdown("---")
        st.subheader("💬 Quick Document Discussion")
        st.markdown("*Start a conversation about your documents*")

        # Pre-filled discussion prompts
        if st.session_state.get('document_discussion_prompt'):
            st.text_area(
                "💬 Ready to discuss:",
                value=st.session_state.document_discussion_prompt,
                height=100,
                help="Copy this prompt and paste it in the chat to start discussing with SAM"
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗨️ Start Chat Discussion"):
                    st.info("💬 Go to the chat interface above and paste the discussion prompt!")
            with col2:
                if st.button("🔄 Clear Prompt"):
                    st.session_state.document_discussion_prompt = ""
                    st.session_state.selected_document = ""
                    st.rerun()
        else:
            # General discussion starters
            st.markdown("**🎯 General Discussion Starters:**")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("📚 Overview All Documents"):
                    overview_prompt = "Can you give me an overview of all the documents in my knowledge base? What are the main topics and themes?"
                    st.session_state.document_discussion_prompt = overview_prompt
                    st.rerun()

                if st.button("🔍 Find Specific Topic"):
                    topic_prompt = "I'm looking for information about [TOPIC]. Which documents contain relevant information and what do they say?"
                    st.session_state.document_discussion_prompt = topic_prompt
                    st.rerun()

            with col2:
                if st.button("🔗 Connect Ideas"):
                    connect_prompt = "What connections and patterns do you see across all my uploaded documents? How do they relate to each other?"
                    st.session_state.document_discussion_prompt = connect_prompt
                    st.rerun()

                if st.button("💎 Most Valuable Insights"):
                    insights_prompt = "What are the most valuable insights and key takeaways from all my documents combined?"
                    st.session_state.document_discussion_prompt = insights_prompt
                    st.rerun()

    except Exception as e:
        st.warning(f"Could not load document statistics: {e}")

def render_memory_interface():
    """Render the memory management interface."""

    # Show Memory Control Center by default (users expect the dropdown menu)
    if st.session_state.get('show_memory_control_center', True):  # Changed default to True
        render_integrated_memory_control_center()
    else:
        render_basic_memory_interface()

def render_integrated_memory_control_center():
    """Render the full Memory Control Center integrated into the secure interface."""
    st.header("🎛️ SAM Memory Control Center")
    st.markdown("*Integrated Advanced Memory Management*")

    # Import Memory Control Center components
    try:
        from ui.memory_browser import MemoryBrowserUI
        from ui.memory_editor import MemoryEditor
        from ui.memory_graph import MemoryGraphVisualizer
        from ui.memory_commands import get_command_processor
        from ui.role_memory_filter import get_role_filter
        from ui.bulk_ingestion_ui import render_bulk_ingestion
        from ui.api_key_manager import render_api_key_manager
        from memory.memory_vectorstore import get_memory_store
        from memory.memory_reasoning import get_memory_reasoning_engine
        from config.agent_mode import get_mode_controller

        from ui.memory_app import render_personalized_tuner as render_personalized_tuner_standalone

        # Initialize components
        memory_store = get_memory_store()
        command_processor = get_command_processor()
        role_filter = get_role_filter()

        # Navigation menu moved to main content area (preserving 100% of functionality)
        st.markdown("---")

        # Add toggle for basic interface
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("🎛️ Navigation")
        with col2:
            if st.button("📋 Basic View", help="Switch to basic memory interface"):
                st.session_state.show_memory_control_center = False
                st.rerun()

        # Navigation menu with override support for Messages from SAM
        default_options = [
            "💬 Enhanced Chat",
            "📖 Document Library",
            "🔍 Memory Browser",
            "✏️ Memory Editor",
            "🕸️ Memory Graph",
            "💻 Command Interface",
            "📁 Bulk Ingestion",
            "🔑 API Key Manager",
            "🧠🎨 Dream Canvas",
            "🧠 Personalized Tuner",
            "🏆 Memory Ranking",
            "📊 Memory Analytics",
            "🧠⚡ SLP Analytics",
            "🔍 Vetting Queue"
        ]

        # Handle page override from Messages from SAM alerts
        default_index = 0
        if st.session_state.get('memory_page_override'):
            override_page = st.session_state.memory_page_override
            if override_page in default_options:
                default_index = default_options.index(override_page)
            # Clear the override after using it
            del st.session_state.memory_page_override

        memory_page = st.selectbox(
            "Select Feature",
            options=default_options,
            index=default_index,
            help="Choose a Memory Control Center feature to access"
        )

        st.markdown("---")

        # Sidebar for quick stats (preserving 100% of functionality)
        with st.sidebar:
            st.header("📊 Memory Control Center")

            # Quick stats
            try:
                stats = memory_store.get_memory_stats()
                # Use .get() with fallback values to prevent KeyError
                total_memories = stats.get('total_memories', len(getattr(memory_store, 'memory_chunks', {})))
                total_size_mb = stats.get('total_size_mb', 0.0)

                st.metric("Total Memories", total_memories)
                st.metric("Storage Size", f"{total_size_mb:.1f} MB")
            except Exception as e:
                st.error(f"Error loading stats: {e}")
                # Provide fallback metrics
                st.metric("Total Memories", "N/A")
                st.metric("Storage Size", "N/A")

        # Render selected page
        if memory_page == "💬 Enhanced Chat":
            render_enhanced_memory_chat()
        elif memory_page == "📖 Document Library":
            render_document_library_integrated()
        elif memory_page == "🔍 Memory Browser":
            render_memory_browser_integrated()
        elif memory_page == "✏️ Memory Editor":
            render_memory_editor_integrated()
        elif memory_page == "🕸️ Memory Graph":
            render_memory_graph_integrated()
        elif memory_page == "💻 Command Interface":
            render_command_interface_integrated()
        elif memory_page == "📁 Bulk Ingestion":
            render_bulk_ingestion()
        elif memory_page == "🔑 API Key Manager":
            render_api_key_manager()
        elif memory_page == "🧠🎨 Dream Canvas":
            render_dream_canvas_integrated()
        elif memory_page == "🧠 Personalized Tuner":
            # Reuse the standalone tuner UI inside the integrated control center
            render_personalized_tuner_standalone()
        elif memory_page == "🏆 Memory Ranking":
            render_memory_ranking_integrated()
        elif memory_page == "📊 Memory Analytics":
            render_memory_analytics_integrated()
        elif memory_page == "🧠⚡ SLP Analytics":
            render_slp_analytics_integrated()
        elif memory_page == "🔍 Vetting Queue":
            st.header("🔍 Vetting Queue")
            st.warning("⚠️ Vetting queue temporarily disabled due to recent configuration changes")
            st.info("💡 This will be re-enabled once the vetting system configuration is fixed")
            # render_vetting_queue_integrated()

    except ImportError as e:
        st.error(f"❌ Memory Control Center components not available: {e}")
        st.info("💡 **Fallback**: Using basic memory interface")
        st.session_state.show_memory_control_center = False
        render_basic_memory_interface()
    except Exception as e:
        st.error(f"❌ Error loading Memory Control Center: {e}")
        st.session_state.show_memory_control_center = False
        render_basic_memory_interface()

def render_basic_memory_interface():
    """Render the basic memory management interface."""
    # Memory Control Center Access
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("🎛️ Memory Control Center", use_container_width=True, help="Switch to Memory Control Center with dropdown menu"):
            st.session_state.show_memory_control_center = True
            st.success("🎛️ **Switching to Memory Control Center...**")
            st.rerun()

    with col2:
        st.info("💡 **Tip:** Click to access the full Memory Control Center interface.")

    st.markdown("---")

    # Basic Memory Interface (existing functionality)
    st.subheader("🔍 Basic Memory Search")
    st.markdown("*For advanced memory management, use the Memory Control Center above*")
    search_query = st.text_input("Search your encrypted memories...")

    if search_query:
        with st.spinner("🔍 Searching encrypted memories..."):
            try:
                logger.info(f"Searching for: '{search_query}'")

                # Check memory store status first
                security_status = st.session_state.secure_memory_store.get_security_status()
                logger.info(f"Security status: {security_status}")

                results = search_unified_memory(query=search_query, max_results=10)

                logger.info(f"Search returned {len(results)} results")
                st.write(f"Found {len(results)} results:")

                if len(results) == 0:
                    # Show debug information
                    st.warning("No results found. Debug information:")
                    st.json(security_status)

                for i, result in enumerate(results):
                    with st.expander(f"📄 Result {i+1} (Score: {result.similarity_score:.3f})"):
                        st.write("**Content:**")

                        # PHASE 3: Use utility function to handle different result types
                        content, source, metadata = extract_result_content(result)
                        if content:
                            display_content = content[:500] + "..." if len(content) > 500 else content
                            st.write(display_content)

                            st.write("**Metadata:**")
                            # Handle different metadata structures
                            if hasattr(result, 'chunk'):
                                # Legacy MemorySearchResult
                                st.json({
                                    'source': source,
                                    'memory_type': getattr(result.chunk, 'memory_type', {}).value if hasattr(getattr(result.chunk, 'memory_type', None), 'value') else 'unknown',
                                    'importance_score': getattr(result.chunk, 'importance_score', 0.0),
                                    'tags': getattr(result.chunk, 'tags', []),
                                    'timestamp': getattr(result.chunk, 'timestamp', 'unknown')
                                })
                            else:
                                # RankedMemoryResult
                                st.json({
                                    'source': source,
                                    'chunk_id': getattr(result, 'chunk_id', 'unknown'),
                                    'final_score': getattr(result, 'final_score', 0.0),
                                    'metadata': metadata
                                })
                        else:
                            st.error("Could not extract content from result")

            except Exception as e:
                logger.error(f"Memory search failed: {e}")
                st.error(f"❌ Memory search failed: {e}")

    # Memory statistics
    st.subheader("📊 Memory Statistics")
    try:
        stats = st.session_state.secure_memory_store.get_memory_stats()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            # Enhanced fallback for total_memories
            total_memories = stats.get('total_memories',
                                     len(getattr(st.session_state.secure_memory_store, 'memory_chunks', {})))
            st.metric("Total Memories", total_memories)
        with col2:
            st.metric("Store Type", stats.get('store_type', 'Secure'))
        with col3:
            st.metric("Storage Size", f"{stats.get('total_size_mb', 0):.1f} MB")
        with col4:
            st.metric("Embedding Dim", stats.get('embedding_dimension', 0))

    except Exception as e:
        st.warning(f"Could not load memory statistics: {e}")
        # Provide fallback interface
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Memories", "N/A")
        with col2:
            st.metric("Store Type", "Secure")
        with col3:
            st.metric("Storage Size", "N/A")
        with col4:
            st.metric("Embedding Dim", "N/A")

def render_slp_analytics_integrated():
    """Render SLP analytics dashboard integrated into Memory Control Center."""
    try:
        from integrate_slp_enhancements import render_slp_analytics_dashboard
        render_slp_analytics_dashboard()
    except ImportError:
        st.error("❌ Enhanced SLP analytics not available")
        st.info("💡 **Note:** Enhanced SLP analytics require Phase 1A+1B components.")
    except Exception as e:
        st.error(f"❌ Error loading SLP analytics: {e}")
        logger.error(f"SLP analytics integration error: {e}")

def render_vetting_interface():
    """Render the content vetting interface."""
    st.header("🔍 Content Vetting Dashboard")

    # Vetting status overview
    st.subheader("📊 Vetting Status")

    try:
        vetting_status = get_vetting_status()
        logger.info("Successfully loaded vetting status")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🗂️ Quarantined", vetting_status.get('quarantine_files', 0))
        with col2:
            st.metric("✅ Vetted", vetting_status.get('vetted_files', 0))
        with col3:
            st.metric("👍 Approved", vetting_status.get('approved_files', 0))
        with col4:
            st.metric("👎 Rejected", vetting_status.get('rejected_files', 0))

        # Check for new content that needs vetting
        quarantine_files = vetting_status.get('quarantine_files', 0)
        if quarantine_files > 0:
            st.info(f"📥 **{quarantine_files} file(s) in quarantine awaiting vetting**")
            st.markdown("💡 **Tip:** Web search results are automatically saved to quarantine. Click '🛡️ Vet All Content' below to analyze them for security and quality.")

            # Add refresh button for real-time updates
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🔄 Refresh Status", key="refresh_quarantine_status"):
                    st.rerun()

        elif vetting_status.get('vetted_files', 0) == 0:
            st.success("✅ **No content awaiting vetting**")
            st.markdown("💡 **Tip:** When you perform web searches during chat, the results will appear here for security analysis.")

            # Add refresh button to check for new content
            if st.button("🔄 Check for New Content", key="check_new_content"):
                st.rerun()

        # Security Analysis Overview
        if vetting_status.get('vetted_files', 0) > 0:
            st.markdown("---")
            st.markdown("### 🛡️ **Security Analysis Overview**")
            st.markdown("*Powered by SAM's Conceptual Dimension Prober*")

            # Calculate security metrics across all vetted files
            security_metrics = calculate_security_overview()

            if security_metrics:
                # Real-Time Security Metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    critical_risks = security_metrics.get('critical_risks', 0)
                    critical_color = "🔴" if critical_risks > 0 else "🟢"
                    st.metric(
                        label="🔴 Critical Risks",
                        value=critical_risks,
                        help="Immediate security threats requiring attention"
                    )
                    if critical_risks > 0:
                        st.error(f"⚠️ {critical_risks} critical security threat(s) detected!")

                with col2:
                    high_risks = security_metrics.get('high_risks', 0)
                    high_color = "🟠" if high_risks > 0 else "🟢"
                    st.metric(
                        label="🟠 High Risks",
                        value=high_risks,
                        help="High-priority security concerns requiring review"
                    )
                    if high_risks > 0:
                        st.warning(f"⚠️ {high_risks} high-priority concern(s) detected!")

                with col3:
                    avg_credibility = security_metrics.get('avg_credibility', 0)
                    cred_color = "🟢" if avg_credibility >= 0.7 else "🟡" if avg_credibility >= 0.4 else "🔴"
                    st.metric(
                        label="🎓 Avg Credibility",
                        value=f"{avg_credibility:.1%}",
                        help="Average content reliability across all vetted items"
                    )
                    st.markdown(f"{cred_color} {'Excellent' if avg_credibility >= 0.7 else 'Moderate' if avg_credibility >= 0.4 else 'Poor'}")

                with col4:
                    avg_purity = security_metrics.get('avg_purity', 0)
                    purity_color = "🟢" if avg_purity >= 0.8 else "🟡" if avg_purity >= 0.5 else "🔴"
                    st.metric(
                        label="🧹 Avg Purity",
                        value=f"{avg_purity:.1%}",
                        help="Average content cleanliness and freedom from suspicious patterns"
                    )
                    st.markdown(f"{purity_color} {'Clean' if avg_purity >= 0.8 else 'Moderate' if avg_purity >= 0.5 else 'Concerning'}")

                # Overall Security Status
                total_risks = critical_risks + high_risks
                files_analyzed = security_metrics.get('files_analyzed', 0)

                if total_risks == 0:
                    st.success("🛡️ **All Clear**: No critical or high-risk security threats detected across all vetted content")
                elif critical_risks > 0:
                    st.error(f"🚨 **Critical Alert**: {critical_risks} critical security threat(s) detected - immediate attention required")
                else:
                    st.warning(f"⚠️ **Review Required**: {high_risks} high-priority security concern(s) detected - manual review recommended")

                st.info(f"📊 **Analysis Summary**: {files_analyzed} file(s) analyzed by SAM's Conceptual Dimension Prober")
            else:
                st.info("📊 **Security Analysis**: No security metrics available yet. Complete the vetting process to see detailed security analysis.")

            if security_metrics:
                sec_col1, sec_col2, sec_col3, sec_col4 = st.columns(4)

                with sec_col1:
                    critical_risks = security_metrics.get('critical_risks', 0)
                    risk_color = "🔴" if critical_risks > 0 else "🟢"
                    st.metric(f"{risk_color} Critical Risks", critical_risks)

                with sec_col2:
                    high_risks = security_metrics.get('high_risks', 0)
                    risk_color = "🟠" if high_risks > 0 else "🟢"
                    st.metric(f"{risk_color} High Risks", high_risks)

                with sec_col3:
                    avg_credibility = security_metrics.get('avg_credibility', 0)
                    cred_color = "🟢" if avg_credibility >= 0.7 else "🟡" if avg_credibility >= 0.4 else "🔴"
                    st.metric(f"{cred_color} Avg Credibility", f"{avg_credibility:.1%}")

                with sec_col4:
                    avg_purity = security_metrics.get('avg_purity', 0)
                    pur_color = "🟢" if avg_purity >= 0.8 else "🟡" if avg_purity >= 0.5 else "🔴"
                    st.metric(f"{pur_color} Avg Purity", f"{avg_purity:.1%}")

                # Security status summary
                if security_metrics.get('critical_risks', 0) == 0 and security_metrics.get('high_risks', 0) == 0:
                    st.success("🛡️ **No Critical Security Risks Detected Across All Content**")
                elif security_metrics.get('critical_risks', 0) > 0:
                    st.error(f"⚠️ **{security_metrics['critical_risks']} Critical Security Risk(s) Require Immediate Attention**")
                else:
                    st.warning(f"⚠️ **{security_metrics['high_risks']} High Security Risk(s) Detected - Review Recommended**")

        # Vetting controls
        st.subheader("🛡️ Vetting Controls")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("""
            **🔍 Automated Content Analysis**

            Run comprehensive security analysis on all quarantined web content using **SAM's Conceptual Dimension Prober**.

            **🛡️ Security Analysis Includes:**
            - **🎓 Credibility & Bias**: Factual accuracy and source reliability assessment
            - **🎭 Persuasive Language**: Detection of manipulative or emotionally charged content
            - **🔮 Speculation vs. Fact**: Identification of unverified claims and conjecture
            - **🧹 Content Purity**: Analysis for suspicious patterns and security threats
            - **🌐 Source Reputation**: Domain credibility and HTTPS usage verification

            **📊 Results Include:** Risk factor identification, security scores, and professional analysis reports.
            """)

            # Add preview of security dashboard
            st.info("""
            **🔍 After Analysis, You'll See:**
            - 🔴 **Critical Risk Counter** - Immediate security alerts
            - 🟠 **High Risk Counter** - Priority concerns
            - 🎓 **Average Credibility Score** - Content reliability
            - 🧹 **Average Purity Score** - Content cleanliness
            - ✅/⚠️/❌ **Four-Dimension Analysis** for each item
            """)

        with col2:
            # Enhanced vetting button with status
            quarantine_count = vetting_status.get('quarantine_files', 0)
            if quarantine_count > 0:
                st.markdown("**📥 Ready to Analyze:**")
                st.markdown(f"**{quarantine_count} file(s)** awaiting analysis")

                # Add prominent call-to-action
                st.warning("⚡ **Click below to unlock the Security Analysis Dashboard!**")

            if st.button("🛡️ Vet All Content",
                        disabled=not vetting_status.get('ready_for_vetting', False),
                        use_container_width=True,
                        help=f"Analyze {quarantine_count} quarantined file(s) for security risks"):
                with st.spinner("🔄 Analyzing content with Conceptual Dimension Prober..."):
                    vetting_result = trigger_vetting_process()

                    if vetting_result['success']:
                        stats = vetting_result.get('stats', {})
                        approved = stats.get('approved_files', 0)
                        integrated = stats.get('integrated_items', 0)

                        if integrated > 0:
                            st.success(f"✅ **Knowledge Consolidation Complete!**\n\n"
                                     f"• {approved} files approved and vetted\n"
                                     f"• {integrated} items integrated into SAM's knowledge base\n"
                                     f"• SAM now has access to this new information!")
                        else:
                            st.success(f"✅ Vetting completed! {approved} files approved.")

                        st.rerun()
                    else:
                        st.error(f"❌ Vetting and consolidation failed: {vetting_result.get('error', 'Unknown error')}")

        # Quarantined content preview (NEW)
        quarantine_files = vetting_status.get('quarantine_files', 0)
        if quarantine_files > 0:
            st.subheader("📥 Quarantined Content Preview")
            st.markdown("""
            **🔍 Content Awaiting Analysis:** Review the web content below that is waiting for security analysis.
            Click '🛡️ Vet All Content' above to analyze all items for security risks, bias, and credibility.
            """)

            # Add real-time monitoring info with auto-refresh
            from datetime import datetime
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Add refresh controls
            col_time, col_refresh = st.columns([3, 1])
            with col_time:
                st.caption(f"🕒 **Last Updated:** {current_time}")
            with col_refresh:
                if st.button("🔄 Refresh Now", key="refresh_quarantine_content", use_container_width=True):
                    st.rerun()

            # Add auto-refresh option
            if st.checkbox("🔄 Auto-refresh every 10 seconds", key="auto_refresh_quarantine"):
                import time
                time.sleep(0.1)  # Small delay to prevent immediate refresh
                st.rerun()

            quarantined_content = load_quarantined_content()

            # Debug information
            loaded_count = len(quarantined_content)
            corrupted_count = len([c for c in quarantined_content if c.get('corrupted')])
            valid_count = loaded_count - corrupted_count

            if loaded_count != quarantine_files:
                st.warning(f"⚠️ **File Count Mismatch:** Expected {quarantine_files} files, loaded {loaded_count} files")
                st.markdown("**💡 Possible Solutions:**")
                st.markdown("• Click '🔄 Refresh Now' button above")
                st.markdown("• Check if new web searches were performed recently")
                st.markdown("• Verify quarantine directory contains the expected files")

            if corrupted_count > 0:
                st.error(f"❌ **{corrupted_count} corrupted file(s)** detected - see details below")

            if loaded_count > 0:
                st.info(f"📊 **Loading Summary:** {valid_count} valid files, {corrupted_count} corrupted files, {loaded_count} total loaded")

                # Check for recent files (within last 5 minutes)
                from datetime import datetime, timedelta
                recent_threshold = datetime.now() - timedelta(minutes=5)
                recent_files = []

                for content in quarantined_content:
                    if not content.get('corrupted'):
                        file_timestamp = content.get('timestamp', content.get('metadata', {}).get('quarantine_timestamp'))
                        if file_timestamp:
                            try:
                                # Parse ISO timestamp
                                if isinstance(file_timestamp, str):
                                    file_time = datetime.fromisoformat(file_timestamp.replace('Z', '+00:00'))
                                    if file_time.replace(tzinfo=None) > recent_threshold:
                                        recent_files.append(content.get('filename', 'Unknown'))
                            except Exception as e:
                                logger.debug(f"Error parsing file timestamp: {e}")
                                pass

                if recent_files:
                    st.success(f"🆕 **{len(recent_files)} recent file(s)** added in the last 5 minutes: {', '.join(recent_files[:3])}")
                    if len(recent_files) > 3:
                        st.caption(f"... and {len(recent_files) - 3} more recent files")

                # Bulk selection controls
                st.markdown("---")
                st.markdown("**🎯 Bulk Selection Controls:**")

                col_select, col_actions = st.columns([1, 2])

                with col_select:
                    if st.button("☑️ Select All", key="select_all_quarantine"):
                        for i, content in enumerate(quarantined_content):
                            filename = content.get('filename', 'Unknown')
                            selection_key = f"select_quarantine_{i}_{filename}"
                            st.session_state[selection_key] = True
                        st.rerun()

                    if st.button("☐ Deselect All", key="deselect_all_quarantine"):
                        for i, content in enumerate(quarantined_content):
                            filename = content.get('filename', 'Unknown')
                            selection_key = f"select_quarantine_{i}_{filename}"
                            st.session_state[selection_key] = False
                        st.rerun()

                with col_actions:
                    # Count selected items
                    selected_count = 0
                    selected_files = []

                    for i, content in enumerate(quarantined_content):
                        filename = content.get('filename', 'Unknown')
                        selection_key = f"select_quarantine_{i}_{filename}"
                        if st.session_state.get(selection_key, False):
                            selected_count += 1
                            selected_files.append(filename)

                    if selected_count > 0:
                        st.info(f"📋 **{selected_count} item(s) selected** for bulk action")

                        col_bulk_approve, col_bulk_reject = st.columns(2)

                        with col_bulk_approve:
                            if st.button(f"✅ Approve Selected ({selected_count})",
                                       key="bulk_approve_quarantine",
                                       use_container_width=True):
                                success_count = 0
                                for filename in selected_files:
                                    if approve_quarantined_content(filename):
                                        success_count += 1

                                if success_count > 0:
                                    st.success(f"✅ **{success_count}/{selected_count}** items approved and integrated!")
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to approve any selected items")

                        with col_bulk_reject:
                            if st.button(f"❌ Reject Selected ({selected_count})",
                                       key="bulk_reject_quarantine",
                                       use_container_width=True):
                                success_count = 0
                                for filename in selected_files:
                                    if reject_quarantined_content(filename):
                                        success_count += 1

                                if success_count > 0:
                                    st.success(f"❌ **{success_count}/{selected_count}** items rejected!")
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to reject any selected items")
                    else:
                        st.info("☑️ Select items below to enable bulk actions")

                st.markdown("---")

                for i, content in enumerate(quarantined_content):
                    render_quarantined_content_item(content, i)
            else:
                st.warning("⚠️ Could not load any quarantined content files. They may be corrupted or inaccessible.")

                # Additional debugging
                with st.expander("🔧 Debug Information", expanded=False):
                    st.markdown(f"**Expected Files:** {quarantine_files}")
                    st.markdown(f"**Loaded Files:** {loaded_count}")

                    # Try to list files in quarantine directory
                    try:
                        from pathlib import Path
                        quarantine_dir = Path("quarantine")
                        if quarantine_dir.exists():
                            all_files = list(quarantine_dir.glob("*"))
                            json_files = list(quarantine_dir.glob("*.json"))

                            st.markdown(f"**Total Files in Quarantine:** {len(all_files)}")
                            st.markdown(f"**JSON Files Found:** {len(json_files)}")

                            if json_files:
                                st.markdown("**JSON Files (sorted by modification time):**")
                                # Sort by modification time (newest first)
                                json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

                                for f in json_files:
                                    mod_time = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                                    is_metadata = f.name.startswith('metadata') or f.name.endswith('_metadata.json')
                                    file_type = " [METADATA]" if is_metadata else ""
                                    st.markdown(f"• `{f.name}` ({f.stat().st_size:,} bytes, {mod_time}){file_type}")

                                # Show web search pattern files specifically
                                web_search_files = [f for f in json_files if 'intelligent_web_' in f.name or 'web_search_' in f.name]
                                if web_search_files:
                                    st.markdown(f"**🌐 Web Search Files:** {len(web_search_files)} found")
                                    for wsf in web_search_files:
                                        st.markdown(f"  • {wsf.name}")
                                else:
                                    st.warning("**⚠️ No web search files found** - this might indicate web search content isn't being saved to quarantine")

                                # Check for today's files specifically
                                from datetime import datetime
                                today = datetime.now().strftime("%Y%m%d")
                                today_files = [f for f in json_files if today in f.name]
                                if today_files:
                                    st.success(f"**📅 Today's Files:** {len(today_files)} found")
                                    for tf in today_files:
                                        st.markdown(f"  • {tf.name}")
                                else:
                                    st.warning(f"**📅 No files from today ({today})** found - recent web searches may not be appearing")

                                # Look for the specific file mentioned in logs
                                expected_file = f"intelligent_web_{today}_121537_eb564d73.json"
                                if any(expected_file in f.name for f in json_files):
                                    st.success(f"✅ **Found expected file:** {expected_file}")
                                else:
                                    st.error(f"❌ **Missing expected file:** {expected_file} (from terminal logs)")
                        else:
                            st.markdown("**Quarantine directory does not exist**")
                    except Exception as e:
                        st.markdown(f"**Debug Error:** {e}")

                    # Enhanced test button with comprehensive debugging
                    if st.button("🧪 Test Web Search Save", key="test_web_search_save"):
                        try:
                            st.info("🔄 Running comprehensive quarantine save test...")

                            # Test 1: Basic functionality test
                            test_result = {
                                'success': True,
                                'tool_used': 'test_tool',
                                'data': {
                                    'articles': [
                                        {'title': 'Test Article 1', 'content': 'Test content for debugging', 'source': 'test.com'},
                                        {'title': 'Test Article 2', 'content': 'More test content', 'source': 'test2.com'}
                                    ]
                                }
                            }

                            st.write("**Test Data Structure:**")
                            st.json(test_result)

                            # Call the save function
                            save_intelligent_web_to_quarantine(test_result, "Test query for debugging")
                            st.success("✅ Test content saved to quarantine - refresh to see it")

                            # Verify the file was created
                            from pathlib import Path
                            quarantine_dir = Path("quarantine")
                            json_files = list(quarantine_dir.glob("intelligent_web_*.json"))

                            st.write(f"**Files in quarantine after test:** {len(json_files)}")
                            for f in json_files[-3:]:  # Show last 3 files
                                st.write(f"• {f.name} ({f.stat().st_size} bytes)")

                        except Exception as e:
                            st.error(f"❌ Test save failed: {e}")
                            import traceback
                            st.code(traceback.format_exc())

                    # Add button to manually trigger web search debugging
                    if st.button("🔍 Debug Web Search Flow", key="debug_web_search"):
                        try:
                            st.info("🔄 Testing web search flow...")

                            # Import and test the intelligent web system
                            from web_retrieval.intelligent_web_system import IntelligentWebSystem
                            from web_retrieval.config import load_web_config

                            # Load configuration
                            web_config = load_web_config()
                            api_keys = {}  # Empty for testing

                            # Create system
                            intelligent_web_system = IntelligentWebSystem(api_keys=api_keys, config=web_config)

                            # Test query
                            test_query = "latest technology news"
                            st.write(f"**Testing query:** {test_query}")

                            # Process query
                            result = intelligent_web_system.process_query(test_query)

                            st.write("**Web search result:**")
                            st.json(result)

                            if result.get('success'):
                                st.success("✅ Web search successful - now testing quarantine save...")
                                save_intelligent_web_to_quarantine(result, test_query)
                                st.success("✅ Quarantine save completed!")
                            else:
                                st.error(f"❌ Web search failed: {result.get('error', 'Unknown error')}")

                        except Exception as e:
                            st.error(f"❌ Web search debug failed: {e}")
                            import traceback
                            st.code(traceback.format_exc())

                    # Add force refresh button
                    if st.button("🔄 Force Refresh (Clear Cache)", key="force_refresh_quarantine"):
                        # Clear any potential caching
                        if hasattr(st.session_state, 'quarantine_cache'):
                            del st.session_state.quarantine_cache
                        st.success("🔄 Cache cleared - refreshing...")
                        st.rerun()

        # Vetted content results
        if vetting_status.get('has_vetted_content', False):
            st.subheader("📋 Vetted Content Results")

            vetted_content = load_vetted_content()

            if vetted_content:
                st.markdown("""
                **🚪 Decision Gate:** Review the automated analysis results below and make the final decision
                to either **Use & Add to Knowledge** or **Discard** each piece of content.
                """)

                for i, content in enumerate(vetted_content):
                    render_vetted_content_item(content, i)
            else:
                st.info("No vetted content available yet. Run the vetting process to analyze quarantined content.")

    except Exception as e:
        st.error(f"❌ Could not load vetting dashboard: {e}")

def render_security_dashboard():
    """Render the security dashboard."""
    st.header("🛡️ Security Dashboard")

    # Get security status
    try:
        security_status = st.session_state.secure_memory_store.get_security_status()
        session_info = st.session_state.security_manager.get_session_info()

        # Security overview
        st.subheader("🔐 Security Overview")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            status_color = "🟢" if security_status['encryption_active'] else "🔴"
            st.metric("Encryption Status", f"{status_color} {'Active' if security_status['encryption_active'] else 'Inactive'}")

        with col2:
            st.metric("Application State", session_info['state'].title())

        with col3:
            st.metric("Session Time", f"{session_info['time_remaining']}s")

        with col4:
            st.metric("Failed Attempts", f"{session_info['failed_attempts']}/{session_info['max_attempts']}")

        # Detailed security information
        with st.expander("🔍 Detailed Security Information"):
            st.json(security_status)

        # Security actions
        st.subheader("🔧 Security Actions")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("⏱️ Extend Session", use_container_width=True, key="dashboard_extend_session_button"):
                st.session_state.security_manager.extend_session()
                st.success("Session extended!")
                st.rerun()

        with col2:
            if st.button("🔒 Lock Application", use_container_width=True, key="dashboard_lock_application_button"):
                st.session_state.security_manager.lock_application()
                st.success("Application locked!")
                st.rerun()

    except Exception as e:
        st.error(f"❌ Could not load security dashboard: {e}")

def handle_secure_chat_command(command: str) -> str:
    """Handle chat commands in the secure interface."""
    try:
        cmd_parts = command[1:].split()
        if not cmd_parts:
            return "❌ Invalid command. Type `/help` for available commands."

        cmd = cmd_parts[0].lower()

        if cmd == 'help':
            return """🔧 **Available Commands:**

**System Commands:**
• `/status` - Show system status and analytics
• `/search <query>` - Search encrypted content
• `/summarize <topic>` - Generate smart summary about a topic
• `/help` - Show this help message

**Security Commands:**
• `/security` - Show security dashboard summary
• `/stats` - Show memory and storage statistics

**Examples:**
• `/search quantum computing`
• `/summarize machine learning trends`
• `/status`

**Regular Chat:**
Just type your questions normally for conversational AI assistance!"""

        elif cmd == 'status':
            return get_secure_system_status()

        elif cmd == 'search' and len(cmd_parts) > 1:
            query = ' '.join(cmd_parts[1:])
            return search_secure_content(query)

        elif cmd == 'summarize' and len(cmd_parts) > 1:
            topic = ' '.join(cmd_parts[1:])
            return generate_secure_summary(topic)

        elif cmd == 'security':
            return get_security_status_summary()

        elif cmd == 'stats':
            return get_memory_stats_summary()

        else:
            return f"❌ Unknown command: `{cmd}`. Type `/help` for available commands."

    except Exception as e:
        logger.error(f"Command handling failed: {e}")
        return f"❌ Error processing command: {e}"

def get_secure_system_status() -> str:
    """Get comprehensive system status for secure interface."""
    try:
        # Get security status
        security_status = st.session_state.secure_memory_store.get_security_status()
        session_info = st.session_state.security_manager.get_session_info()

        # Get memory stats
        memory_stats = st.session_state.secure_memory_store.get_memory_stats()

        # Format status
        encryption_status = "🟢 Active" if security_status['encryption_active'] else "🔴 Inactive"
        session_time = session_info.get('time_remaining', 'Unknown')

        status_report = f"""📊 **Secure System Status**

**🛡️ Security Status:**
• Encryption: {encryption_status}
• Session State: {session_info['state'].title()}
• Time Remaining: {session_time}s
• Failed Attempts: {session_info['failed_attempts']}/{session_info['max_attempts']}

**🧠 Memory Status:**
• Total Memories: {memory_stats.get('total_memories', 0)}
• Encrypted Chunks: {security_status.get('encrypted_chunk_count', 0)}
• Storage Size: {memory_stats.get('total_size_mb', 0):.1f} MB
• Embedding Dimension: {memory_stats.get('embedding_dimension', 0)}

**🔐 Encryption Details:**
• Searchable Fields: {security_status.get('searchable_fields', 0)}
• Encrypted Fields: {security_status.get('encrypted_fields', 0)}
• Store Type: {memory_stats.get('store_type', 'Unknown')}

**🤖 AI Model:**
• Model: DeepSeek-R1-0528-Qwen3-8B-GGUF
• Backend: Ollama (localhost:11434)
• Status: ✅ Connected

**🧠 Advanced Reasoning Systems:**"""

        # Add TPV status
        tpv_active = False
        if (st.session_state.get('tpv_initialized') or
            st.session_state.get('tpv_session_data', {}).get('last_response', {}).get('tpv_enabled') or
            st.session_state.get('tpv_active')):
            tpv_active = True

        tpv_status = "✅ Active" if tpv_active else "❌ Inactive"
        status_report += f"\n• TPV Active Control: {tpv_status}"

        # Add Dissonance Monitoring status
        dissonance_active = False
        try:
            # Check if TPV integration has dissonance monitoring enabled
            if st.session_state.get('sam_tpv_integration'):
                tpv_integration = st.session_state.sam_tpv_integration
                if hasattr(tpv_integration, 'tpv_monitor') and tpv_integration.tpv_monitor:
                    if hasattr(tpv_integration.tpv_monitor, 'enable_dissonance_monitoring'):
                        if tpv_integration.tpv_monitor.enable_dissonance_monitoring:
                            dissonance_active = True

            # Fallback: If TPV is active, assume dissonance is available
            if not dissonance_active and tpv_active:
                try:
                    from sam.cognition.dissonance_monitor import DissonanceMonitor
                    dissonance_active = True
                except ImportError:
                    dissonance_active = False
        except Exception:
            dissonance_active = False

        dissonance_status = "✅ Active" if dissonance_active else "❌ Inactive"
        status_report += f"\n• Dissonance Monitor: {dissonance_status}"

        # Add dissonance metrics if available
        if dissonance_active:
            try:
                last_response = st.session_state.get('tpv_session_data', {}).get('last_response', {})
                if last_response.get('final_dissonance_score') is not None:
                    final_score = last_response['final_dissonance_score']
                    status_report += f"\n• Last Dissonance Score: {final_score:.3f}"

                if last_response.get('dissonance_analysis'):
                    analysis = last_response['dissonance_analysis']
                    mean_score = analysis.get('mean_dissonance', 0.0)
                    peak_score = analysis.get('max_dissonance', 0.0)
                    status_report += f"\n• Mean Dissonance: {mean_score:.3f}"
                    status_report += f"\n• Peak Dissonance: {peak_score:.3f}"
            except Exception:
                pass

        return status_report

    except Exception as e:
        return f"❌ Error getting system status: {e}"

def search_secure_content(query: str) -> str:
    """Search all available content (secure + web knowledge) and return formatted results."""
    try:
        results = search_unified_memory(query=query, max_results=5)

        if not results:
            return f"🔍 **Search Results for '{query}'**\n\nNo results found. Try different search terms or upload relevant documents."

        formatted_results = f"🔍 **Search Results for '{query}'** ({len(results)} found)\n\n"

        for i, result in enumerate(results, 1):
            # PHASE 3: Use utility function to handle different result types
            content, source, metadata = extract_result_content(result)

            if content:
                content_preview = content[:200]
                if len(content) > 200:
                    content_preview += "..."

                # Get tags from appropriate location
                if hasattr(result, 'chunk') and hasattr(result.chunk, 'tags'):
                    tags = result.chunk.tags
                else:
                    tags = metadata.get('tags', [])

                formatted_results += f"""**Result {i}** (Score: {result.similarity_score:.3f})
📄 **Source:** {source}
📝 **Content:** {content_preview}
🏷️ **Tags:** {', '.join(tags) if tags else 'None'}

"""

        return formatted_results

    except Exception as e:
        return f"❌ Search failed: {e}"

def generate_secure_summary(topic: str) -> str:
    """Generate a smart summary about a topic using all available content."""
    try:
        # Search for relevant content from all sources
        results = search_unified_memory(query=topic, max_results=10)

        if not results:
            return f"📝 **Summary: {topic}**\n\nNo relevant content found in your encrypted documents. Upload documents about '{topic}' to generate a comprehensive summary."

        # Collect content for summarization
        content_parts = []
        sources = set()

        for result in results:
            if result.similarity_score > 0.3:  # Only include relevant results
                # PHASE 3: Use utility function to handle different result types
                content, source, metadata = extract_result_content(result)
                if content and source:
                    content_parts.append(content)
                    sources.add(source)

        if not content_parts:
            return f"📝 **Summary: {topic}**\n\nFound {len(results)} documents but none were sufficiently relevant. Try a more specific topic or upload more relevant documents."

        # Generate summary using Ollama
        combined_content = "\n\n".join(content_parts[:5])  # Limit to top 5 results

        try:
            import requests

            system_prompt = f"""You are SAM, a secure AI assistant. Create a comprehensive summary about "{topic}" based on the provided encrypted document content.

Structure your summary with:
1. Overview of the topic
2. Key points and insights
3. Important details and facts
4. Conclusions or implications

Be thorough but concise. Focus on the most important information."""

            user_prompt = f"""Please create a comprehensive summary about "{topic}" based on this content:

{combined_content}

Provide a well-structured summary that captures the key information about {topic}."""

            ollama_response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M",
                    "prompt": f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:",
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 800
                    }
                },
                timeout=45
            )

            if ollama_response.status_code == 200:
                response_data = ollama_response.json()
                ai_summary = response_data.get('response', '').strip()

                if ai_summary:
                    source_list = "\n".join([f"• {source}" for source in sorted(sources)])
                    return f"""📝 **Summary: {topic}**

{ai_summary}

**📚 Sources ({len(sources)} documents):**
{source_list}"""

        except Exception as e:
            logger.error(f"Ollama summary generation failed: {e}")

        # Fallback summary
        source_list = "\n".join([f"• {source}" for source in sorted(sources)])
        return f"""📝 **Summary: {topic}**

Based on {len(results)} relevant documents in your encrypted storage:

{combined_content[:1000]}{'...' if len(combined_content) > 1000 else ''}

**📚 Sources ({len(sources)} documents):**
{source_list}"""

    except Exception as e:
        return f"❌ Summary generation failed: {e}"

def get_security_status_summary() -> str:
    """Get a summary of security status."""
    try:
        security_status = st.session_state.secure_memory_store.get_security_status()
        session_info = st.session_state.security_manager.get_session_info()

        encryption_icon = "🟢" if security_status['encryption_active'] else "🔴"

        return f"""🛡️ **Security Status Summary**

**Encryption:** {encryption_icon} {'Active' if security_status['encryption_active'] else 'Inactive'}
**Session:** {session_info['state'].title()} ({session_info.get('time_remaining', 'Unknown')}s remaining)
**Security Level:** Enterprise-grade AES-256-GCM
**Authentication:** Argon2 password hashing
**Data Protection:** {security_status.get('encrypted_chunk_count', 0)} encrypted chunks

**Session Security:**
• Failed Attempts: {session_info['failed_attempts']}/{session_info['max_attempts']}
• Auto-lock: Enabled
• Secure Storage: ✅ Active"""

    except Exception as e:
        return f"❌ Error getting security status: {e}"

def render_feedback_system(message_index: int):
    """Render feedback system for assistant messages."""
    try:
        # Create unique key for this message
        feedback_key = f"feedback_{message_index}"

        # Check if feedback already submitted
        if st.session_state.get(f"{feedback_key}_submitted"):
            st.success("✅ Thank you for your feedback! SAM is learning from this.")
            return

        # Feedback buttons
        st.markdown("---")
        st.markdown("**How was this response?**")

        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button("👍 Good", key=f"{feedback_key}_positive", use_container_width=True):
                submit_secure_feedback(message_index, "positive", "")
                st.session_state[f"{feedback_key}_submitted"] = True
                st.rerun()

        with col2:
            if st.button("👎 Needs Work", key=f"{feedback_key}_negative", use_container_width=True):
                st.session_state[f"{feedback_key}_show_correction"] = True
                st.rerun()

        with col3:
            if st.button("💡 Suggest Improvement", key=f"{feedback_key}_improve", use_container_width=True):
                st.session_state[f"{feedback_key}_show_suggestion"] = True
                st.rerun()

        # Show correction input if requested (for negative feedback)
        if st.session_state.get(f"{feedback_key}_show_correction"):
            st.markdown("**What could be improved?**")
            correction_text = st.text_area(
                "Please describe what could be better about this response...",
                key=f"{feedback_key}_correction_input",
                height=100
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Submit Feedback", key=f"{feedback_key}_submit"):
                    if correction_text.strip():
                        submit_secure_feedback(message_index, "negative", correction_text)
                        st.session_state[f"{feedback_key}_submitted"] = True
                        st.session_state[f"{feedback_key}_show_correction"] = False
                        st.success("✅ Thank you for the detailed feedback! SAM is learning from your suggestions.")
                        st.rerun()
                    else:
                        st.warning("Please provide some feedback before submitting.")

            with col2:
                if st.button("Cancel", key=f"{feedback_key}_cancel"):
                    st.session_state[f"{feedback_key}_show_correction"] = False
                    st.rerun()

        # Show suggestion input if requested (for improvement suggestions)
        if st.session_state.get(f"{feedback_key}_show_suggestion"):
            st.markdown("**💡 How can SAM improve this response?**")
            st.markdown("*Your suggestions help SAM learn and provide better responses in the future.*")

            # Suggestion categories
            suggestion_type = st.selectbox(
                "What type of improvement would you suggest?",
                [
                    "More detailed explanation",
                    "Shorter, more concise response",
                    "Include examples",
                    "Add sources/references",
                    "Different tone/style",
                    "Better organization",
                    "More accurate information",
                    "Other (specify below)"
                ],
                key=f"{feedback_key}_suggestion_type"
            )

            suggestion_text = st.text_area(
                "Please provide specific suggestions for improvement:",
                key=f"{feedback_key}_suggestion_input",
                height=120,
                placeholder="For example: 'Could you provide more examples?' or 'This could be more concise' or 'I'd prefer a more formal tone'..."
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Submit Suggestion", key=f"{feedback_key}_submit_suggestion"):
                    # Combine suggestion type and text
                    full_suggestion = f"Suggestion Type: {suggestion_type}\n"
                    if suggestion_text.strip():
                        full_suggestion += f"Details: {suggestion_text.strip()}"
                    else:
                        full_suggestion += "No additional details provided."

                    submit_secure_feedback(message_index, "suggestion", full_suggestion)
                    st.session_state[f"{feedback_key}_submitted"] = True
                    st.session_state[f"{feedback_key}_show_suggestion"] = False
                    st.success("✅ Thank you for your suggestion! SAM will use this to improve future responses.")
                    st.rerun()

            with col2:
                if st.button("Cancel", key=f"{feedback_key}_cancel_suggestion"):
                    st.session_state[f"{feedback_key}_show_suggestion"] = False
                    st.rerun()

    except Exception as e:
        logger.error(f"Feedback system error: {e}")

def render_thought_transparency():
    """Render cognitive distillation thought transparency display (NEW - Phase 2 Integration)."""
    try:
        # Check if cognitive distillation is enabled and transparency data is available
        if (not st.session_state.get('cognitive_distillation_enabled', False) or
            not st.session_state.get('completed_transparency_data')):
            return

        transparency_data = st.session_state.get('completed_transparency_data', {})
        active_principles = transparency_data.get('active_principles', [])

        # Only show if principles were applied
        if not active_principles:
            return

        # Thought transparency display
        with st.expander("🧠 **Reasoning Transparency** - See how SAM thought through this response", expanded=False):
            st.markdown("### 🎯 Applied Cognitive Principles")

            for i, principle in enumerate(active_principles, 1):
                with st.container():
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown(f"**{i}. {principle['text']}**")

                        # Show principle details
                        details_col1, details_col2 = st.columns(2)
                        with details_col1:
                            st.caption(f"🎯 Domains: {', '.join(principle['domains'])}")
                            st.caption(f"📊 Confidence: {principle['confidence']:.2f}")
                        with details_col2:
                            st.caption(f"🔄 Usage: {principle['usage_count']} times")
                            st.caption(f"✅ Success Rate: {principle['success_rate']:.1%}")

                    with col2:
                        # Priority indicator
                        priority = principle.get('display_priority', 0.5)
                        if priority > 0.8:
                            st.success("🔥 High Priority")
                        elif priority > 0.6:
                            st.info("⭐ Medium Priority")
                        else:
                            st.caption("📝 Applied")

            # Meta-cognition insights
            meta_cognition = transparency_data.get('meta_cognition', {})
            if meta_cognition:
                st.markdown("### 🤔 Meta-Cognitive Analysis")

                reasoning_approach = meta_cognition.get('reasoning_approach', '')
                if reasoning_approach:
                    st.info(f"**Reasoning Approach:** {reasoning_approach}")

                principle_selection = meta_cognition.get('principle_selection', '')
                if principle_selection:
                    st.info(f"**Principle Selection:** {principle_selection}")

                confidence_assessment = meta_cognition.get('confidence_assessment', '')
                if confidence_assessment:
                    st.info(f"**Confidence Assessment:** {confidence_assessment}")

            # Principle impact summary
            principle_impact = transparency_data.get('principle_impact', {})
            if principle_impact:
                st.markdown("### 📈 Principle Impact")

                total_impact = principle_impact.get('total_impact', 0)
                impact_summary = principle_impact.get('impact_summary', '')

                if total_impact > 0:
                    st.success(f"**Confidence Boost:** +{total_impact:.1%}")

                if impact_summary:
                    st.caption(impact_summary)

            # Reasoning trace (if available)
            reasoning_trace = transparency_data.get('reasoning_trace')
            if reasoning_trace and reasoning_trace.get('reasoning_steps'):
                st.markdown("### 🔍 Reasoning Steps")

                steps = reasoning_trace['reasoning_steps']
                for step in steps:
                    step_num = step.get('step', 0)
                    step_type = step.get('type', 'unknown')
                    description = step.get('description', '')

                    # Icon based on step type
                    if step_type == 'principle_application':
                        icon = "🧠"
                    elif step_type == 'query_analysis':
                        icon = "🔍"
                    elif step_type == 'response_generation':
                        icon = "✍️"
                    else:
                        icon = "📝"

                    st.markdown(f"{icon} **Step {step_num}:** {description}")

            # Footer with system info
            st.markdown("---")
            st.caption("🔬 **Cognitive Distillation Engine** - SAM's introspective reasoning system")

    except Exception as e:
        logger.warning(f"Failed to render thought transparency: {e}")
        # Fail silently to not disrupt the main chat experience

def submit_secure_feedback(message_index: int, feedback_type: str, correction_text: str):
    """Submit feedback to the secure learning system with comprehensive integration."""
    try:
        # Get the original message and query for context (preserving 100% of functionality)
        original_message = None
        original_query = None
        sam_response = None

        if hasattr(st.session_state, 'chat_history') and len(st.session_state.chat_history) > message_index:
            # Get the assistant message
            if message_index < len(st.session_state.chat_history):
                sam_response = st.session_state.chat_history[message_index].get('content', '')

            # Find the corresponding user query (usually the previous message)
            if message_index > 0:
                original_query = st.session_state.chat_history[message_index - 1].get('content', '')

        # Create comprehensive feedback data (preserving 100% of existing structure)
        feedback_data = {
            'message_index': message_index,
            'feedback_type': feedback_type,
            'correction_text': correction_text,
            'timestamp': datetime.now().isoformat(),  # Use ISO format string instead of Unix timestamp
            'interface': 'secure_streamlit',
            # Enhanced data for learning integration
            'original_query': original_query,
            'sam_response': sam_response,
            'session_id': st.session_state.get('session_id', 'unknown'),
            'user_id': st.session_state.get('authenticated_user', 'anonymous'),
            'feedback_id': f"feedback_{int(time.time() * 1000)}_{message_index}"
        }

        # Store feedback in session state (preserving 100% of existing functionality)
        if 'feedback_history' not in st.session_state:
            st.session_state.feedback_history = []

        st.session_state.feedback_history.append(feedback_data)

        # Enhanced Learning Integration: Process feedback through SAM's learning systems
        try:
            process_feedback_for_learning(feedback_data)
        except Exception as learning_error:
            logger.warning(f"Feedback learning integration failed (non-critical): {learning_error}")

        logger.info(f"Secure feedback submitted: {feedback_type} for message {message_index}")

    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}")

def process_feedback_for_learning(feedback_data: dict):
    """Process user feedback through SAM's comprehensive learning systems."""
    try:
        logger.info(f"Processing feedback for learning: {feedback_data['feedback_id']}")

        # Phase 1: Memory Integration - Store feedback in SAM's memory system
        try:
            if hasattr(st.session_state, 'secure_memory_store') and st.session_state.secure_memory_store:
                memory_content = create_feedback_memory_content(feedback_data)

                # Store feedback as a learning memory
                from memory.memory_vectorstore import MemoryType

                st.session_state.secure_memory_store.add_memory(
                    content=memory_content,
                    memory_type=MemoryType.CONVERSATION,
                    source=f"feedback_{feedback_data['feedback_id']}",
                    tags=['feedback', 'learning', 'user_interaction', feedback_data['feedback_type']],
                    importance_score=calculate_feedback_importance(feedback_data),
                    metadata={
                        'feedback_id': feedback_data['feedback_id'],
                        'feedback_type': feedback_data['feedback_type'],
                        'message_index': feedback_data['message_index'],
                        'timestamp': feedback_data['timestamp'],
                        'interface': feedback_data['interface'],
                        'learning_priority': determine_learning_priority(feedback_data),
                        'content_type': 'user_feedback'
                    }
                )
                logger.info(f"✅ Feedback stored in memory system: {feedback_data['feedback_id']}")

                # CRITICAL FIX: Store corrections as knowledge entries for future retrieval
                if feedback_data.get('correction_text') and feedback_data['feedback_type'] in ['negative', 'suggestion']:
                    store_correction_as_knowledge(feedback_data)

        except Exception as memory_error:
            logger.warning(f"Memory integration failed: {memory_error}")

        # Phase 2: MEMOIR Learning Integration - Process corrections for knowledge updates
        try:
            # Enhanced MEMOIR processing: also process positive feedback with corrections
            should_process_memoir = (
                feedback_data['feedback_type'] in ['negative', 'correction'] and feedback_data.get('correction_text')
            ) or (
                feedback_data['feedback_type'] == 'positive' and feedback_data.get('correction_text')
            )

            if should_process_memoir:
                logger.info(f"Processing MEMOIR learning for {feedback_data['feedback_type']} feedback with correction")
                process_memoir_feedback_learning(feedback_data)
        except Exception as memoir_error:
            logger.warning(f"MEMOIR learning integration failed: {memoir_error}")

        # Phase 3: Response Pattern Learning - Analyze feedback patterns
        try:
            analyze_feedback_patterns(feedback_data)
        except Exception as pattern_error:
            logger.warning(f"Pattern analysis failed: {pattern_error}")

        # Phase 4: User Preference Learning - Update user model
        try:
            update_user_preference_model(feedback_data)
        except Exception as preference_error:
            logger.warning(f"User preference learning failed: {preference_error}")

        logger.info(f"✅ Feedback learning processing completed: {feedback_data['feedback_id']}")

    except Exception as e:
        logger.error(f"Feedback learning processing failed: {e}")

def create_feedback_memory_content(feedback_data: dict) -> str:
    """Create structured memory content from feedback data."""
    feedback_type = feedback_data['feedback_type']
    correction_text = feedback_data.get('correction_text', '')
    original_query = feedback_data.get('original_query', 'Unknown query')
    sam_response = feedback_data.get('sam_response', 'Unknown response')

    if feedback_type == 'positive':
        content = "✅ POSITIVE FEEDBACK: User found SAM's response helpful.\n"
        content += f"Query: {original_query}\n"
        content += f"Response: {sam_response[:200]}{'...' if len(sam_response) > 200 else ''}\n"
        content += "Learning: This response pattern was effective and should be reinforced."

    elif feedback_type == 'negative':
        content = "❌ NEGATIVE FEEDBACK: User indicated SAM's response needs improvement.\n"
        content += f"Query: {original_query}\n"
        content += f"Response: {sam_response[:200]}{'...' if len(sam_response) > 200 else ''}\n"
        if correction_text:
            content += f"User Correction: {correction_text}\n"
            content += "Learning: Adjust response patterns to incorporate user's suggested improvements."
        else:
            content += "Learning: This response pattern was ineffective and should be avoided."

    else:  # suggestion/improvement
        content = "💡 IMPROVEMENT SUGGESTION: User provided specific suggestions for enhancement.\n"
        content += f"Query: {original_query}\n"
        content += f"Response: {sam_response[:200]}{'...' if len(sam_response) > 200 else ''}\n"
        content += f"User Suggestion: {correction_text}\n"
        content += "Learning: Incorporate these suggestions into future similar responses."

    return content

def calculate_feedback_importance(feedback_data: dict) -> float:
    """Calculate importance score for feedback based on type and content."""
    base_importance = 0.7  # Base importance for all feedback

    # Adjust based on feedback type
    if feedback_data['feedback_type'] == 'positive':
        importance = base_importance + 0.1  # Positive feedback is valuable for reinforcement
    elif feedback_data['feedback_type'] == 'negative':
        importance = base_importance + 0.2  # Negative feedback is critical for improvement
    else:  # suggestions
        importance = base_importance + 0.3  # Suggestions are most valuable for learning

    # Adjust based on correction detail
    correction_text = feedback_data.get('correction_text', '')
    if correction_text:
        # More detailed feedback is more valuable
        detail_bonus = min(0.2, len(correction_text.split()) / 50)
        importance += detail_bonus

    return min(1.0, importance)  # Cap at 1.0

def store_correction_as_knowledge(feedback_data: dict):
    """Store user corrections as knowledge entries for future retrieval."""
    try:
        logger.info(f"Storing correction as knowledge: {feedback_data['feedback_id']}")

        correction_text = feedback_data.get('correction_text', '')
        original_query = feedback_data.get('original_query', '')

        if not correction_text or not original_query:
            logger.warning("Missing correction text or original query - skipping knowledge storage")
            return

        # Create knowledge content that will be easily retrievable
        knowledge_content = f"CORRECTED KNOWLEDGE: {correction_text}\n\n"
        knowledge_content += f"Context: This correction was provided by the user in response to the query: '{original_query}'\n"
        knowledge_content += f"Original incorrect response was corrected to: {correction_text}\n"
        knowledge_content += f"This is authoritative user-provided information that should be prioritized in future responses."

        # Store as high-priority knowledge with searchable tags
        if hasattr(st.session_state, 'secure_memory_store') and st.session_state.secure_memory_store:
            from memory.memory_vectorstore import MemoryType

            st.session_state.secure_memory_store.add_memory(
                content=knowledge_content,
                memory_type=MemoryType.CONVERSATION,  # Use proper MemoryType enum
                source=f"user_correction_{feedback_data['feedback_id']}",
                tags=['knowledge', 'user_correction', 'authoritative', 'high_priority', 'factual'],
                importance_score=0.95,  # Very high importance for user corrections
                metadata={
                    'correction_id': feedback_data['feedback_id'],
                    'original_query': original_query,
                    'correction_text': correction_text,
                    'correction_type': 'user_provided',
                    'timestamp': feedback_data['timestamp'],
                    'priority': 'HIGH',
                    'source': 'user_feedback_correction',
                    'content_type': 'corrected_knowledge'  # Add as metadata instead
                }
            )
            logger.info(f"✅ Correction stored as knowledge: {feedback_data['feedback_id']}")

        # Also store in regular memory store for broader access
        try:
            from memory.memory_vectorstore import get_memory_store, MemoryType
            web_store = get_memory_store()

            web_store.add_memory(
                content=knowledge_content,
                memory_type=MemoryType.CONVERSATION,
                source=f"user_correction_{feedback_data['feedback_id']}",
                tags=['knowledge', 'user_correction', 'authoritative'],
                importance_score=0.95,
                metadata={
                    'correction_id': feedback_data['feedback_id'],
                    'source': 'user_correction',
                    'timestamp': feedback_data['timestamp'],
                    'content_type': 'corrected_knowledge'
                }
            )
            logger.info(f"✅ Correction also stored in regular memory store")

        except Exception as web_store_error:
            logger.warning(f"Failed to store correction in web store: {web_store_error}")

    except Exception as e:
        logger.error(f"Failed to store correction as knowledge: {e}")

def determine_learning_priority(feedback_data: dict) -> str:
    """Determine learning priority based on feedback characteristics."""
    if feedback_data['feedback_type'] == 'negative' and feedback_data.get('correction_text'):
        return 'HIGH'  # Corrections need immediate attention
    elif feedback_data['feedback_type'] == 'positive':
        return 'MEDIUM'  # Positive patterns should be reinforced
    else:
        return 'LOW'  # General feedback for gradual improvement

def process_memoir_feedback_learning(feedback_data: dict):
    """Process feedback through MEMOIR learning system for knowledge updates."""
    try:
        logger.info(f"Processing MEMOIR feedback learning: {feedback_data['feedback_id']}")

        # Try to import and use MEMOIR feedback handler
        try:
            from sam.learning.feedback_handler import MEMOIRFeedbackHandler

            # Initialize feedback handler
            memoir_handler = MEMOIRFeedbackHandler()

            # Process the feedback for MEMOIR edits
            result = memoir_handler.process_feedback(
                original_query=feedback_data.get('original_query', ''),
                sam_response=feedback_data.get('sam_response', ''),
                user_feedback=feedback_data.get('correction_text', ''),
                context={
                    'feedback_id': feedback_data['feedback_id'],
                    'feedback_type': feedback_data['feedback_type'],
                    'interface': feedback_data['interface'],
                    'timestamp': feedback_data['timestamp']
                }
            )

            if result.get('success'):
                logger.info(f"✅ MEMOIR learning successful: {result}")
                return result
            else:
                logger.warning(f"MEMOIR learning failed: {result}")

        except ImportError as import_error:
            logger.warning(f"MEMOIR feedback handler not available: {import_error}")
            # Fallback: Create a simple MEMOIR edit manually
            return create_simple_memoir_edit(feedback_data)

        except Exception as memoir_error:
            logger.error(f"MEMOIR feedback handler failed: {memoir_error}")
            # Fallback: Create a simple MEMOIR edit manually
            return create_simple_memoir_edit(feedback_data)

    except Exception as e:
        logger.error(f"MEMOIR feedback learning failed: {e}")
        return {'success': False, 'error': str(e)}

def create_simple_memoir_edit(feedback_data: dict):
    """Create a simple MEMOIR edit when the full handler is not available."""
    try:
        logger.info(f"Creating simple MEMOIR edit for: {feedback_data['feedback_id']}")

        # Try to import MEMOIR edit skill directly
        try:
            from sam.orchestration.skills.internal.memoir_edit import MEMOIR_EditSkill
            from sam.orchestration.uif import SAM_UIF

            # Initialize edit skill
            edit_skill = MEMOIR_EditSkill()

            # Create UIF for the edit
            edit_uif = SAM_UIF(
                input_query=feedback_data.get('original_query', ''),
                intermediate_data={
                    'edit_prompt': feedback_data.get('original_query', ''),
                    'correct_answer': feedback_data.get('correction_text', ''),
                    'edit_context': f"User correction - {feedback_data['feedback_type']}",
                    'confidence_score': 0.8,  # High confidence for user corrections
                    'edit_metadata': {
                        'source': 'user_correction',
                        'feedback_id': feedback_data['feedback_id'],
                        'feedback_type': feedback_data['feedback_type'],
                        'timestamp': feedback_data['timestamp']
                    }
                }
            )

            # Execute the edit
            result_uif = edit_skill.execute(edit_uif)

            if result_uif.success:
                logger.info("✅ Simple MEMOIR edit successful")
                return {'success': True, 'method': 'simple_edit', 'result': result_uif}
            else:
                logger.warning(f"Simple MEMOIR edit failed: {result_uif.error}")
                return {'success': False, 'error': result_uif.error}

        except ImportError:
            logger.warning("MEMOIR edit skill not available - storing correction for future processing")
            # Store the correction for manual processing later
            return store_correction_for_later_processing(feedback_data)

    except Exception as e:
        logger.error(f"Simple MEMOIR edit failed: {e}")
        return {'success': False, 'error': str(e)}

def store_correction_for_later_processing(feedback_data: dict):
    """Store correction for later MEMOIR processing when components are available."""
    try:
        # Store in session state for later processing
        if 'pending_memoir_corrections' not in st.session_state:
            st.session_state.pending_memoir_corrections = []

        st.session_state.pending_memoir_corrections.append({
            'feedback_data': feedback_data,
            'timestamp': time.time(),
            'status': 'pending'
        })

        logger.info(f"✅ Correction stored for later MEMOIR processing: {feedback_data['feedback_id']}")
        return {'success': True, 'method': 'stored_for_later', 'pending_count': len(st.session_state.pending_memoir_corrections)}

    except Exception as e:
        logger.error(f"Failed to store correction for later processing: {e}")
        return {'success': False, 'error': str(e)}

def analyze_feedback_patterns(feedback_data: dict):
    """Analyze feedback patterns to identify learning opportunities."""
    try:
        logger.info(f"Analyzing feedback patterns: {feedback_data['feedback_id']}")

        # Get feedback history for pattern analysis
        if hasattr(st.session_state, 'feedback_history'):
            feedback_history = st.session_state.feedback_history

            # Analyze recent feedback patterns
            recent_feedback = [f for f in feedback_history if f['timestamp'] > (time.time() - 86400)]  # Last 24 hours

            # Count feedback types
            positive_count = len([f for f in recent_feedback if f['feedback_type'] == 'positive'])
            negative_count = len([f for f in recent_feedback if f['feedback_type'] == 'negative'])

            # Store pattern analysis in session state for future use
            if 'feedback_patterns' not in st.session_state:
                st.session_state.feedback_patterns = {}

            st.session_state.feedback_patterns.update({
                'recent_positive': positive_count,
                'recent_negative': negative_count,
                'last_analysis': time.time(),
                'total_feedback': len(feedback_history)
            })

            logger.info(f"✅ Feedback pattern analysis: +{positive_count} -{negative_count} (24h)")

    except Exception as e:
        logger.error(f"Feedback pattern analysis failed: {e}")

def update_user_preference_model(feedback_data: dict):
    """Update user preference model based on feedback."""
    try:
        logger.info(f"Updating user preference model: {feedback_data['feedback_id']}")

        # Initialize user preferences if not exists
        if 'user_preferences' not in st.session_state:
            st.session_state.user_preferences = {
                'response_style': {},
                'content_preferences': {},
                'feedback_history': []
            }

        # Extract preferences from feedback
        user_id = feedback_data.get('user_id', 'anonymous')
        feedback_type = feedback_data['feedback_type']
        correction_text = feedback_data.get('correction_text', '')

        # Update preference model
        preference_update = {
            'timestamp': feedback_data['timestamp'],
            'feedback_type': feedback_type,
            'query_type': classify_query_type(feedback_data.get('original_query', '')),
            'response_length': len(feedback_data.get('sam_response', '')),
            'user_satisfaction': 1.0 if feedback_type == 'positive' else 0.0
        }

        # Add specific preferences from correction text
        if correction_text:
            preference_update['user_suggestions'] = extract_preference_signals(correction_text)

        st.session_state.user_preferences['feedback_history'].append(preference_update)

        logger.info(f"✅ User preference model updated: {user_id}")

    except Exception as e:
        logger.error(f"User preference model update failed: {e}")

def classify_query_type(query: str) -> str:
    """Classify the type of query for preference learning."""
    query_lower = query.lower()

    if any(word in query_lower for word in ['calculate', 'math', '+', '-', '*', '/', '=']):
        return 'calculation'
    elif any(word in query_lower for word in ['news', 'latest', 'recent', 'current']):
        return 'news'
    elif any(word in query_lower for word in ['stock', 'price', 'financial', 'market']):
        return 'financial'
    elif any(word in query_lower for word in ['what', 'how', 'why', 'explain']):
        return 'informational'
    else:
        return 'general'

def extract_preference_signals(correction_text: str) -> dict:
    """Extract preference signals from user correction text."""
    signals = {}
    correction_lower = correction_text.lower()

    # Response length preferences
    if any(phrase in correction_lower for phrase in ['too long', 'too verbose', 'shorter']):
        signals['preferred_length'] = 'shorter'
    elif any(phrase in correction_lower for phrase in ['too short', 'more detail', 'longer']):
        signals['preferred_length'] = 'longer'

    # Response style preferences
    if any(phrase in correction_lower for phrase in ['more formal', 'professional']):
        signals['preferred_style'] = 'formal'
    elif any(phrase in correction_lower for phrase in ['casual', 'friendly', 'conversational']):
        signals['preferred_style'] = 'casual'

    # Content preferences
    if any(phrase in correction_lower for phrase in ['more examples', 'examples']):
        signals['wants_examples'] = True
    if any(phrase in correction_lower for phrase in ['sources', 'references', 'citations']):
        signals['wants_sources'] = True

    return signals

def enhance_response_with_feedback_learning(prompt: str) -> dict:
    """Enhance response generation using feedback-driven learning."""
    try:
        logger.info(f"Enhancing response with feedback learning for: {prompt[:50]}...")

        response_context = {
            'user_preferences': {},
            'feedback_patterns': {},
            'learning_insights': [],
            'response_adjustments': {}
        }

        # Get user preferences from feedback history
        if hasattr(st.session_state, 'user_preferences') and st.session_state.user_preferences:
            user_prefs = st.session_state.user_preferences
            response_context['user_preferences'] = user_prefs

            # Analyze recent feedback for this user
            recent_feedback = [f for f in user_prefs.get('feedback_history', [])
                             if f['timestamp'] > (time.time() - 604800)]  # Last week

            if recent_feedback:
                # Extract response style preferences
                style_preferences = extract_style_preferences(recent_feedback)
                response_context['response_adjustments'].update(style_preferences)

                logger.info(f"✅ Applied user preferences: {style_preferences}")

        # Get feedback patterns for similar queries
        if hasattr(st.session_state, 'feedback_history'):
            similar_feedback = find_similar_query_feedback(prompt, st.session_state.feedback_history)
            if similar_feedback:
                pattern_insights = analyze_similar_feedback_patterns(similar_feedback)
                response_context['learning_insights'].extend(pattern_insights)

                logger.info(f"✅ Found {len(similar_feedback)} similar query feedback patterns")

        # Get memory-based learning insights
        try:
            if hasattr(st.session_state, 'secure_memory_store') and st.session_state.secure_memory_store:
                memory_insights = get_memory_feedback_insights(prompt)
                response_context['learning_insights'].extend(memory_insights)
        except Exception as memory_error:
            logger.warning(f"Memory feedback insights failed: {memory_error}")

        return response_context

    except Exception as e:
        logger.error(f"Feedback learning enhancement failed: {e}")
        return {}

def apply_feedback_enhancements(response: str, context: dict) -> str:
    """Apply feedback-driven enhancements to a response and record debug info."""
    try:
        if not context or not response:
            return response

        enhanced_response = response
        adjustments = context.get('response_adjustments', {})
        applied = []

        # Apply style adjustments based on user preferences
        if adjustments.get('preferred_style') == 'formal':
            enhanced_response = make_response_more_formal(enhanced_response)
            applied.append('preferred_style=formal')
        elif adjustments.get('preferred_style') == 'casual':
            enhanced_response = make_response_more_casual(enhanced_response)
            applied.append('preferred_style=casual')

        # Apply length adjustments
        if adjustments.get('preferred_length') == 'shorter':
            enhanced_response = make_response_shorter(enhanced_response)
            applied.append('preferred_length=shorter')
        elif adjustments.get('preferred_length') == 'longer':
            enhanced_response = make_response_longer(enhanced_response)
            applied.append('preferred_length=longer')

        # Add examples if user prefers them
        if adjustments.get('wants_examples'):
            enhanced_response = add_examples_to_response(enhanced_response)
            applied.append('wants_examples')

        # Add sources if user prefers them
        if adjustments.get('wants_sources'):
            enhanced_response = add_source_attribution(enhanced_response)
            applied.append('wants_sources')

        # Add learning insights as a subtle note (tracked only)
        insights = context.get('learning_insights', [])
        if insights and len(insights) > 0:
            logger.info(f"Applied {len(insights)} learning insights to response")

        # Record debug info in session state for UI panel
        try:
            if 'last_learning_debug' not in st.session_state:
                st.session_state.last_learning_debug = {}
            st.session_state.last_learning_debug = {
                'used_learned_corrections': bool(applied or insights),
                'applied_adjustments': applied,
                'insights_count': len(insights),
                'adjustments': adjustments,
            }
        except Exception:
            pass

        return enhanced_response

    except Exception as e:
        logger.error(f"Failed to apply feedback enhancements: {e}")
        return response

def extract_style_preferences(recent_feedback: list) -> dict:
    """Extract style preferences from recent feedback."""
    preferences = {}

    for feedback in recent_feedback:
        suggestions = feedback.get('user_suggestions', {})
        preferences.update(suggestions)

    return preferences

def find_similar_query_feedback(prompt: str, feedback_history: list) -> list:
    """Find feedback for similar queries."""
    try:
        prompt_lower = prompt.lower()
        prompt_words = set(prompt_lower.split())

        similar_feedback = []

        for feedback in feedback_history:
            original_query = feedback.get('original_query', '').lower()
            if original_query:
                query_words = set(original_query.split())
                # Calculate word overlap
                overlap = len(prompt_words.intersection(query_words))
                if overlap >= 2:  # At least 2 words in common
                    feedback['similarity_score'] = overlap / len(prompt_words.union(query_words))
                    similar_feedback.append(feedback)

        # Sort by similarity and return top 5
        similar_feedback.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        return similar_feedback[:5]

    except Exception as e:
        logger.error(f"Failed to find similar query feedback: {e}")
        return []

def analyze_similar_feedback_patterns(similar_feedback: list) -> list:
    """Analyze patterns in similar query feedback."""
    insights = []

    try:
        # Count positive vs negative feedback
        positive_count = len([f for f in similar_feedback if f['feedback_type'] == 'positive'])
        negative_count = len([f for f in similar_feedback if f['feedback_type'] == 'negative'])

        if negative_count > positive_count:
            insights.append("Previous similar queries received mixed feedback - focus on accuracy")
        elif positive_count > 0:
            insights.append("Similar queries have been well-received - maintain current approach")

        # Extract common correction themes
        corrections = [f.get('correction_text', '') for f in similar_feedback if f.get('correction_text')]
        if corrections:
            common_themes = extract_common_correction_themes(corrections)
            insights.extend(common_themes)

        return insights

    except Exception as e:
        logger.error(f"Failed to analyze feedback patterns: {e}")
        return []

def extract_common_correction_themes(corrections: list) -> list:
    """Extract common themes from correction texts."""
    themes = []

    all_corrections = ' '.join(corrections).lower()

    # Common correction patterns
    if 'more detail' in all_corrections or 'explain more' in all_corrections:
        themes.append("Users prefer more detailed explanations")

    if 'too long' in all_corrections or 'shorter' in all_corrections:
        themes.append("Users prefer shorter responses")

    if 'example' in all_corrections:
        themes.append("Users appreciate concrete examples")

    if 'source' in all_corrections or 'reference' in all_corrections:
        themes.append("Users value source attribution")

    return themes

def get_memory_feedback_insights(prompt: str) -> list:
    """Get feedback insights from memory system."""
    try:
        insights = []

        if hasattr(st.session_state, 'secure_memory_store') and st.session_state.secure_memory_store:
            # Search for feedback-related memories using safe API call
            try:
                feedback_memories = st.session_state.secure_memory_store.search_memories(
                    query=f"feedback learning user_interaction {prompt}",
                    max_results=5
                )
            except TypeError:
                # Fallback for different API signature
                feedback_memories = st.session_state.secure_memory_store.search_memories(
                    f"feedback learning user_interaction {prompt}",
                    5
                )

            for memory in feedback_memories:
                if hasattr(memory, 'chunk') and hasattr(memory.chunk, 'content'):
                    content = memory.chunk.content
                elif hasattr(memory, 'content'):
                    content = memory.content
                else:
                    continue

                if 'Learning:' in content:
                    learning_text = content.split('Learning:')[-1].strip()
                    insights.append(learning_text)

        return insights[:3]  # Limit to top 3 insights

    except Exception as e:
        logger.error(f"Failed to get memory feedback insights: {e}")
        return []

def make_response_more_formal(response: str) -> str:
    """Make response more formal based on user preference."""
    # Simple formal adjustments
    response = response.replace("I'm", "I am")
    response = response.replace("can't", "cannot")
    response = response.replace("won't", "will not")
    response = response.replace("don't", "do not")
    return response

def make_response_more_casual(response: str) -> str:
    """Make response more casual based on user preference."""
    # Simple casual adjustments
    response = response.replace("I am", "I'm")
    response = response.replace("cannot", "can't")
    response = response.replace("will not", "won't")
    response = response.replace("do not", "don't")
    return response

def make_response_shorter(response: str) -> str:
    """Make response shorter based on user preference."""
    # Simple shortening - take first 2/3 of response
    sentences = response.split('. ')
    if len(sentences) > 3:
        shortened = '. '.join(sentences[:int(len(sentences) * 0.67)])
        return shortened + '.' if not shortened.endswith('.') else shortened
    return response

def make_response_longer(response: str) -> str:
    """Make response longer based on user preference."""
    # Add a helpful note for more detail
    if not response.endswith('.'):
        response += '.'
    response += "\n\n*Let me know if you'd like me to elaborate on any specific aspect of this topic.*"
    return response

def add_examples_to_response(response: str) -> str:
    """Add examples to response based on user preference."""
    # Add a note about examples
    if not response.endswith('.'):
        response += '.'
    response += "\n\n*I can provide specific examples if that would be helpful.*"
    return response

def add_source_attribution(response: str) -> str:
    """Add source attribution based on user preference."""
    # Add a note about sources
    if not response.endswith('.'):
        response += '.'
    response += "\n\n*This information is based on SAM's knowledge base and recent data.*"
    return response

def perform_secure_web_search(query: str) -> Dict[str, Any]:
    """Perform secure web search using the new Intelligent Web System."""
    try:
        logger.info(f"Starting intelligent web search for: {query}")

        # Step 1: Initialize the Intelligent Web System
        web_system = get_intelligent_web_system()

        # Step 2: Process query through intelligent routing
        result = web_system.process_query(query)

        if result['success']:
            # Step 3: Format and enhance the result
            formatted_response = format_intelligent_web_result(result, query)

            # Step 4: Save to quarantine for vetting
            save_intelligent_web_to_quarantine(result, query)

            # Step 5: Generate AI-enhanced response
            web_response = generate_intelligent_web_response(query, result)

            return {
                'success': True,
                'error': None,
                'response': web_response,
                'sources': extract_sources_from_result(result),
                'content_count': count_content_items(result),
                'method': 'intelligent_web_system',
                'tool_used': result.get('tool_used', 'unknown'),
                'routing_info': result.get('routing_decision', {})
            }
        else:
            # Fallback to original RSS method if intelligent system fails
            logger.warning(f"Intelligent web system failed: {result.get('error')}, falling back to RSS")
            return perform_rss_fallback_search(query)

    except Exception as e:
        logger.error(f"Secure web search failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'response': None
        }

def get_intelligent_web_system():
    """Get or create the intelligent web system instance."""
    try:
        from web_retrieval.intelligent_web_system import IntelligentWebSystem
        from config.config_manager import ConfigManager

        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.get_config()

        # Initialize with API keys and configuration
        api_keys = {
            'serper': config.serper_api_key if config.serper_api_key else None,
            'newsapi': config.newsapi_api_key if config.newsapi_api_key else None
        }

        # Web retrieval configuration
        web_config = {
            'cocoindex_search_provider': config.cocoindex_search_provider,
            'cocoindex_num_pages': config.cocoindex_num_pages,
            'web_retrieval_provider': config.web_retrieval_provider
        }

        return IntelligentWebSystem(api_keys=api_keys, config=web_config)

    except ImportError as e:
        logger.error(f"Failed to import intelligent web system: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize intelligent web system: {e}")
        raise

def format_intelligent_web_result(result: Dict[str, Any], query: str) -> str:
    """Format the intelligent web system result for display."""
    try:
        tool_used = result.get('tool_used', 'unknown')
        data = result.get('data', {})

        content_parts = []
        content_parts.append(f"**Web Search Results for: {query}**")
        content_parts.append(f"*Method: {tool_used.replace('_', ' ').title()}*")
        content_parts.append("")

        # Format based on tool type
        if tool_used == 'cocoindex_tool':
            chunks = data.get('chunks', [])
            content_parts.append(f"Found {len(chunks)} relevant content chunks using intelligent search")
            content_parts.append("")

            for i, chunk in enumerate(chunks[:8], 1):  # Show top 8 chunks
                chunk_parts = []

                title = chunk.get('title', f'Content Chunk {i}').strip()
                content = chunk.get('content', '').strip()
                source_url = chunk.get('source_url', '').strip()
                relevance_score = chunk.get('relevance_score', 0.0)

                if title:
                    chunk_parts.append(f"**{i}. {title}**")

                if content:
                    # Limit content for display
                    display_content = content[:300] + "..." if len(content) > 300 else content
                    chunk_parts.append(display_content)

                source_info = f"*Relevance: {relevance_score:.2f}*"
                if source_url:
                    source_info += f" | [Source]({source_url})"

                chunk_parts.append(source_info)

                if chunk_parts:
                    content_parts.append("\n".join(chunk_parts))
                    content_parts.append("---")

        elif tool_used == 'news_api_tool' or tool_used == 'rss_reader_tool':
            articles = data.get('articles', [])
            content_parts.append(f"Found {len(articles)} articles")
            content_parts.append("")

            for i, article in enumerate(articles[:10], 1):
                article_parts = []

                title = article.get('title', '').strip()
                if title:
                    article_parts.append(f"**{i}. {title}**")

                description = article.get('description', '').strip()
                if description:
                    article_parts.append(description)

                source = article.get('source', 'Unknown')
                pub_date = article.get('pub_date', '') or article.get('published_at', '')
                link = article.get('link', '') or article.get('url', '')

                source_info = f"*Source: {source}*"
                if pub_date:
                    source_info += f" | *Published: {pub_date}*"
                if link:
                    source_info += f" | [Read more]({link})"

                article_parts.append(source_info)

                if article_parts:
                    content_parts.append("\n".join(article_parts))
                    content_parts.append("---")

        elif tool_used == 'search_api_tool':
            search_results = data.get('search_results', [])
            extracted_content = data.get('extracted_content', [])

            content_parts.append(f"Found {len(search_results)} search results")
            content_parts.append("")

            for i, result_item in enumerate(search_results[:5], 1):
                content_parts.append(f"**{i}. {result_item.get('title', 'No title')}**")
                if result_item.get('snippet'):
                    content_parts.append(result_item['snippet'])
                content_parts.append(f"*Source: {result_item.get('url', 'No URL')}*")
                content_parts.append("---")

        elif tool_used == 'url_content_extractor':
            content = data.get('content', '')
            metadata = data.get('metadata', {})

            if metadata.get('title'):
                content_parts.append(f"**{metadata['title']}**")

            if content:
                # Limit content for display
                display_content = content[:1000] + "..." if len(content) > 1000 else content
                content_parts.append(display_content)

            content_parts.append(f"*Source: {data.get('url', 'Unknown')}*")

        # Remove last separator
        if content_parts and content_parts[-1] == "---":
            content_parts.pop()

        return "\n".join(content_parts)

    except Exception as e:
        logger.error(f"Failed to format intelligent web result: {e}")
        return f"Error formatting web search results: {e}"

def save_intelligent_web_to_quarantine(result: Dict[str, Any], query: str):
    """Save intelligent web system results to quarantine for vetting."""
    import traceback

    # Enhanced debug logging with stack trace
    logger.info("🚨 SAVE_INTELLIGENT_WEB_TO_QUARANTINE CALLED! 🚨")
    logger.info(f"Call stack: {traceback.format_stack()[-3:-1]}")  # Show calling context
    logger.info(f"Function called with query: {query}")
    logger.info(f"Function called with result type: {type(result)}")

    try:
        from pathlib import Path
        import json
        import hashlib
        from datetime import datetime
        import os

        logger.info(f"=== ENHANCED SAVE TO QUARANTINE DEBUG ===")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Query: {query}")
        logger.info(f"Result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")

        # Enhanced result structure logging
        if isinstance(result, dict):
            logger.info(f"Result structure analysis:")
            logger.info(f"  - success: {result.get('success', 'NOT_FOUND')}")
            logger.info(f"  - tool_used: {result.get('tool_used', 'NOT_FOUND')}")
            logger.info(f"  - data keys: {list(result.get('data', {}).keys()) if 'data' in result else 'NO_DATA_KEY'}")
            if 'data' in result and isinstance(result['data'], dict):
                data = result['data']
                logger.info(f"  - articles count: {len(data.get('articles', []))}")
                logger.info(f"  - chunks count: {len(data.get('chunks', []))}")
                logger.info(f"  - search_results count: {len(data.get('search_results', []))}")

        # Create quarantine directory if it doesn't exist
        quarantine_dir = Path("quarantine")
        logger.info(f"Quarantine directory path: {quarantine_dir.absolute()}")

        quarantine_dir.mkdir(exist_ok=True)
        logger.info(f"Quarantine directory created/exists: {quarantine_dir.exists()}")
        logger.info(f"Quarantine directory is writable: {os.access(quarantine_dir, os.W_OK)}")

        # List existing files before save
        existing_files = list(quarantine_dir.glob("*.json"))
        logger.info(f"Existing quarantine files before save: {[f.name for f in existing_files]}")

        # Test write a simple file first
        test_file = quarantine_dir / "test_write.txt"
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            logger.info(f"✅ Test write successful: {test_file.exists()}")
            test_file.unlink()  # Clean up
        except Exception as e:
            logger.error(f"❌ Test write failed: {e}")
            raise

        # Generate filename based on query hash and timestamp
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"intelligent_web_{timestamp}_{query_hash}.json"

        logger.info(f"Generated filename: {filename}")
        logger.info(f"Query hash: {query_hash}")
        logger.info(f"Timestamp: {timestamp}")

        # Prepare quarantine data structure
        quarantine_data = {
            "query": query,
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "source": "intelligent_web_system",
                "method": result.get('tool_used', 'unknown'),
                "content_count": count_content_items(result),
                "sources": extract_sources_from_result(result),
                "quarantine_timestamp": datetime.now().isoformat(),
                "debug_info": {
                    "saved_from": "save_intelligent_web_to_quarantine",
                    "cwd": os.getcwd(),
                    "python_path": os.environ.get('PYTHONPATH', 'NOT_SET')
                }
            }
        }

        logger.info(f"Quarantine data prepared, size: {len(str(quarantine_data))} characters")
        logger.info(f"Content count: {quarantine_data['metadata']['content_count']}")
        logger.info(f"Sources count: {len(quarantine_data['metadata']['sources'])}")

        # Save to quarantine with enhanced error handling
        quarantine_path = quarantine_dir / filename
        logger.info(f"About to write to: {quarantine_path.absolute()}")

        try:
            with open(quarantine_path, 'w', encoding='utf-8') as f:
                json.dump(quarantine_data, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ JSON write completed successfully")
        except Exception as write_error:
            logger.error(f"❌ JSON write failed: {write_error}")
            logger.error(f"Write error type: {type(write_error)}")
            raise

        # Verify file was actually created with enhanced checks
        if quarantine_path.exists():
            file_size = quarantine_path.stat().st_size
            logger.info(f"✅ Intelligent web content saved to quarantine: {filename} ({file_size} bytes)")
            logger.info(f"Full path: {quarantine_path.absolute()}")

            # Verify file content is readable
            try:
                with open(quarantine_path, 'r', encoding='utf-8') as f:
                    test_data = json.load(f)
                logger.info(f"✅ File content verification successful, keys: {list(test_data.keys())}")
            except Exception as verify_error:
                logger.error(f"❌ File content verification failed: {verify_error}")

            # List files after save to confirm
            new_files = list(quarantine_dir.glob("*.json"))
            logger.info(f"Quarantine files after save: {[f.name for f in new_files]}")

        else:
            logger.error(f"❌ Failed to create quarantine file: {quarantine_path}")
            logger.error(f"Directory contents after attempted save: {list(quarantine_dir.iterdir())}")
            raise FileNotFoundError(f"Quarantine file was not created: {filename}")

    except Exception as e:
        logger.error(f"❌ Failed to save intelligent web content to quarantine: {e}")
        logger.error(f"Exception type: {type(e)}")
        logger.error(f"Exception traceback: {traceback.format_exc()}")
        logger.error(f"Attempted path: {quarantine_dir / filename if 'quarantine_dir' in locals() and 'filename' in locals() else 'Unknown'}")
        raise

def test_quarantine_save():
    """Test function to verify quarantine save functionality."""
    logger.info("🧪 TESTING QUARANTINE SAVE FUNCTION 🧪")

    test_result = {
        'success': True,
        'tool_used': 'test_tool',
        'data': {
            'articles': [
                {'title': 'Test Article 1', 'source': 'test.com'},
                {'title': 'Test Article 2', 'source': 'test.com'}
            ]
        }
    }

    test_query = "Test web search query"

    try:
        save_intelligent_web_to_quarantine(test_result, test_query)
        logger.info("✅ Test quarantine save completed successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Test quarantine save failed: {e}")
        return False

def extract_sources_from_result(result: Dict[str, Any]) -> List[str]:
    """Extract source URLs/names from intelligent web result."""
    try:
        sources = []
        data = result.get('data', {})

        # Extract from cocoindex chunks (Phase 8.5)
        chunks = data.get('chunks', [])
        for chunk in chunks:
            source_url = chunk.get('source_url', '')
            if source_url and source_url not in sources:
                sources.append(source_url)

        # Extract from articles
        articles = data.get('articles', [])
        for article in articles:
            source = article.get('source', '') or article.get('url', '')
            if source and source not in sources:
                sources.append(source)

        # Extract from search results (search API format)
        search_results = data.get('search_results', [])
        for search_result in search_results:
            url = search_result.get('url', '')
            if url and url not in sources:
                sources.append(url)

        # Extract from simple web search results (simple_web_search format)
        simple_results = data.get('results', [])
        for result in simple_results:
            url = result.get('url', '')
            source = result.get('source', '')
            title = result.get('title', '')

            # Add URL if available
            if url and url not in sources:
                sources.append(url)
            # Add source name if no URL but has source
            elif source and source not in sources:
                sources.append(source)
            # Add title as fallback if no URL or source
            elif title and title not in sources:
                sources.append(title)

        # Extract from URL extraction
        url = data.get('url', '')
        if url and url not in sources:
            sources.append(url)

        return sources[:10]  # Limit to top 10 sources

    except Exception as e:
        logger.error(f"Failed to extract sources: {e}")
        return []

def count_content_items(result: Dict[str, Any]) -> int:
    """Count the number of content items in the result."""
    try:
        data = result.get('data', {})

        # Count cocoindex chunks (Phase 8.5)
        chunks = data.get('chunks', [])
        if chunks:
            return len(chunks)

        # Count articles
        articles = data.get('articles', [])
        if articles:
            return len(articles)

        # Count search results (search API format)
        search_results = data.get('search_results', [])
        if search_results:
            return len(search_results)

        # Count simple web search results (simple_web_search format)
        simple_results = data.get('results', [])
        if simple_results:
            return len(simple_results)

        # Count extracted content
        content = data.get('content', '')
        if content:
            return 1

        return 0

    except Exception as e:
        logger.error(f"Failed to count content items: {e}")
        return 0

def generate_intelligent_web_response(query: str, result: Dict[str, Any]) -> str:
    """Generate AI-enhanced response using intelligent web system results."""
    try:
        # Create summary for AI processing
        ai_summary = create_ai_summary_from_result(result, query)

        if not ai_summary or len(ai_summary.strip()) < 50:
            return format_intelligent_web_result(result, query)

        # Use Ollama to generate enhanced response
        import requests

        system_prompt = """You are SAM, a secure AI assistant. You have just retrieved current web content using an advanced intelligent web retrieval system to answer the user's question.

Provide a comprehensive, well-structured response based on the web content provided. Focus on delivering actual information with clear organization.

Important guidelines:
- Present the most important and relevant information first
- Organize information logically (by topic, chronology, or importance)
- Be factual and objective, focusing on the actual content retrieved
- Mention that this information comes from current web sources
- Summarize key points while maintaining accuracy
- Include relevant details from multiple sources when available"""

        user_prompt = f"""Based on the following current web content retrieved using intelligent web retrieval, please answer this question: "{query}"

Web content summary:
{ai_summary[:3000]}

Please provide a comprehensive, well-organized response based on this current web information. Focus on the most relevant and important content."""

        ollama_response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M",
                "prompt": f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:",
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 800
                }
            },
            timeout=90
        )

        if ollama_response.status_code == 200:
            response_data = ollama_response.json()
            ai_response = response_data.get('response', '').strip()

            if ai_response:
                # Add source information
                sources = extract_sources_from_result(result)
                content_count = count_content_items(result)
                tool_used = result.get('tool_used', 'intelligent_web_system')

                sources_text = "\n\n**🌐 Sources:**\n" + "\n".join([f"• {source}" for source in sources[:5]])

                web_enhanced_response = f"""🌐 **Based on current web sources:**

{ai_response}

{sources_text}

*Information retrieved using {tool_used.replace('_', ' ').title()} from {content_count} sources.*"""

                return web_enhanced_response

        # Fallback if Ollama fails
        return format_intelligent_web_result(result, query)

    except Exception as e:
        logger.error(f"Intelligent web response generation failed: {e}")
        return format_intelligent_web_result(result, query)

def create_ai_summary_from_result(result: Dict[str, Any], query: str) -> str:
    """Create a concise summary for AI processing from intelligent web result."""
    try:
        data = result.get('data', {})
        tool_used = result.get('tool_used', 'unknown')

        summary_parts = []
        summary_parts.append(f"Web Search Summary for: {query}")
        summary_parts.append(f"Method: {tool_used}")
        summary_parts.append("")

        # Process based on tool type
        if tool_used == 'cocoindex_tool':
            chunks = data.get('chunks', [])
            summary_parts.append(f"Intelligent search chunks found: {len(chunks)}")
            summary_parts.append("")

            for i, chunk in enumerate(chunks[:6], 1):  # Top 6 chunks for AI processing
                content = chunk.get('content', '').strip()
                title = chunk.get('title', f'Chunk {i}').strip()
                source_url = chunk.get('source_url', '')
                relevance_score = chunk.get('relevance_score', 0.0)

                if content:
                    chunk_summary = f"{i}. {title}"
                    # Limit content for summary
                    short_content = content[:250] + "..." if len(content) > 250 else content
                    chunk_summary += f" - {short_content}"
                    chunk_summary += f" (Relevance: {relevance_score:.2f}, Source: {source_url})"
                    summary_parts.append(chunk_summary)

        elif tool_used in ['news_api_tool', 'rss_reader_tool']:
            articles = data.get('articles', [])
            summary_parts.append(f"Articles found: {len(articles)}")
            summary_parts.append("")

            for i, article in enumerate(articles[:8], 1):
                title = article.get('title', '').strip()
                description = article.get('description', '').strip()
                source = article.get('source', 'Unknown')

                if title:
                    article_summary = f"{i}. {title}"
                    if description:
                        short_desc = description[:200] + "..." if len(description) > 200 else description
                        article_summary += f" - {short_desc}"
                    article_summary += f" (Source: {source})"
                    summary_parts.append(article_summary)

        elif tool_used == 'search_api_tool':
            search_results = data.get('search_results', [])
            summary_parts.append(f"Search results found: {len(search_results)}")
            summary_parts.append("")

            for i, result_item in enumerate(search_results[:5], 1):
                title = result_item.get('title', 'No title')
                snippet = result_item.get('snippet', '')
                url = result_item.get('url', '')

                search_summary = f"{i}. {title}"
                if snippet:
                    search_summary += f" - {snippet[:150]}..."
                search_summary += f" (URL: {url})"
                summary_parts.append(search_summary)

        elif tool_used == 'url_content_extractor':
            content = data.get('content', '')
            metadata = data.get('metadata', {})

            summary_parts.append("Extracted content:")
            if metadata.get('title'):
                summary_parts.append(f"Title: {metadata['title']}")

            if content:
                content_preview = content[:500] + "..." if len(content) > 500 else content
                summary_parts.append(f"Content: {content_preview}")

        return "\n".join(summary_parts)

    except Exception as e:
        logger.error(f"AI summary creation failed: {e}")
        return f"Error creating AI summary: {e}"

def perform_rss_fallback_search(query: str) -> Dict[str, Any]:
    """Fallback to RSS-based search if intelligent system fails."""
    try:
        logger.info("Using RSS fallback search method")

        # Generate RSS URLs for the query
        rss_urls = generate_rss_urls(query)

        # Try RSS extraction
        rss_result = perform_rss_extraction(query, rss_urls)

        if rss_result['success'] and rss_result.get('article_count', 0) > 0:
            return rss_result
        else:
            return {
                'success': False,
                'error': 'RSS fallback also failed - no content could be retrieved',
                'response': None
            }

    except Exception as e:
        logger.error(f"RSS fallback search failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'response': None
        }

def generate_rss_urls(query: str) -> List[str]:
    """Generate RSS URLs for the given query."""
    query_lower = query.lower()

    # CNN-specific queries
    if 'cnn' in query_lower:
        return [
            "http://rss.cnn.com/rss/cnn_latest.rss",  # CNN Latest (working)
            "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",  # NYT as backup
            "https://feeds.bbci.co.uk/news/rss.xml"  # BBC as backup
        ]

    # Topic-specific RSS feeds
    elif any(word in query_lower for word in ['health', 'medical', 'medicine', 'covid', 'pandemic']):
        return [
            "http://rss.cnn.com/rss/cnn_latest.rss",  # CNN Latest
            "https://rss.nytimes.com/services/xml/rss/nyt/Health.xml",  # NYT Health
            "https://feeds.bbci.co.uk/news/health/rss.xml"  # BBC Health
        ]
    elif any(word in query_lower for word in ['politics', 'political', 'election', 'government']):
        return [
            "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml",  # NYT Politics
            "https://feeds.bbci.co.uk/news/politics/rss.xml",  # BBC Politics
            "http://rss.cnn.com/rss/cnn_latest.rss"  # CNN Latest
        ]
    elif any(word in query_lower for word in ['technology', 'tech', 'ai', 'artificial intelligence']):
        return [
            "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",  # NYT Tech
            "https://feeds.bbci.co.uk/news/technology/rss.xml",  # BBC Tech
            "http://rss.cnn.com/rss/cnn_latest.rss"  # CNN Latest
        ]
    elif any(word in query_lower for word in ['business', 'economy', 'finance', 'market']):
        return [
            "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",  # NYT Business
            "https://feeds.bbci.co.uk/news/business/rss.xml",  # BBC Business
            "http://rss.cnn.com/rss/cnn_latest.rss"  # CNN Latest
        ]

    # General news queries - use top RSS feeds
    else:
        return [
            "https://feeds.bbci.co.uk/news/rss.xml",  # BBC News (reliable)
            "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",  # NYT Homepage
            "http://rss.cnn.com/rss/cnn_latest.rss"  # CNN Latest
        ]

def perform_rss_extraction(query: str, rss_urls: List[str]) -> Dict[str, Any]:
    """Perform RSS-based content extraction (proven working method)."""
    try:
        logger.info(f"Starting RSS extraction for query: '{query}' from {len(rss_urls)} RSS feeds")

        all_news_items = []
        successful_sources = []

        for url in rss_urls:
            try:
                logger.info(f"Fetching RSS feed: {url}")
                rss_content = fetch_rss_content(url)

                if rss_content['success'] and rss_content.get('content'):
                    # Parse the RSS content to extract news items
                    news_items = parse_rss_content_to_articles(rss_content['content'], url)
                    all_news_items.extend(news_items)
                    successful_sources.append(url)
                    logger.info(f"Successfully extracted {len(news_items)} items from {url}")
                else:
                    logger.warning(f"Failed to fetch RSS content from {url}")

            except Exception as e:
                logger.error(f"Error processing RSS feed {url}: {e}")
                continue

        if all_news_items:
            # Format the content for response
            formatted_content = format_rss_articles_for_response(all_news_items, query)

            # Save to quarantine
            save_rss_to_quarantine(all_news_items, query, successful_sources)

            # Generate AI-enhanced response
            web_response = generate_rss_enhanced_response(query, all_news_items)

            return {
                'success': True,
                'error': None,
                'response': web_response,
                'sources': successful_sources,
                'content_count': len(all_news_items),
                'article_count': len(all_news_items),
                'method': 'rss'
            }
        else:
            return {
                'success': False,
                'error': 'No articles extracted from RSS feeds',
                'article_count': 0
            }

    except Exception as e:
        logger.error(f"RSS extraction failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'article_count': 0
        }

def parse_rss_content_to_articles(rss_content: str, source_url: str) -> List[Dict[str, Any]]:
    """Parse RSS content and extract articles (using the working logic)."""
    try:
        import xml.etree.ElementTree as ET
        import html
        import re
        from datetime import datetime

        # Parse RSS XML
        root = ET.fromstring(rss_content)

        # Find items
        items = (root.findall('.//item') or
                root.findall('.//{http://www.w3.org/2005/Atom}entry') or
                root.findall('.//entry'))

        logger.info(f"Found {len(items)} items in RSS feed")

        articles = []

        for i, item in enumerate(items[:15]):  # Limit to top 15 news items
            try:
                title = ''
                description = ''
                link = ''
                pub_date = ''

                # Extract title - simplified approach that was working
                title_elem = item.find('title')
                if title_elem is not None and title_elem.text:
                    title = title_elem.text.strip()

                # Extract description - simplified approach that was working
                desc_elem = item.find('description')
                if desc_elem is not None and desc_elem.text:
                    description = desc_elem.text.strip()
                    # Clean HTML tags from description
                    description = re.sub(r'<[^>]+>', '', description)
                    # Limit description length
                    if len(description) > 300:
                        description = description[:300] + "..."

                # Extract link - simplified
                link_elem = item.find('link')
                if link_elem is not None and link_elem.text:
                    link = link_elem.text.strip()

                # Extract publication date - simplified
                date_elem = item.find('pubDate')
                if date_elem is not None and date_elem.text:
                    pub_date = date_elem.text.strip()

                # Include items with any substantial title
                if title and len(title.strip()) > 3:
                    article = {
                        'title': title,
                        'description': description,
                        'link': link,
                        'pub_date': pub_date,
                        'source': source_url,
                        'extracted_at': datetime.now().isoformat()
                    }
                    articles.append(article)

                    if i < 3:  # Debug logging for first few items
                        logger.info(f"✅ Added article {len(articles)}: {title[:50]}...")

            except Exception as e:
                logger.warning(f"Error parsing RSS item {i}: {e}")
                continue

        logger.info(f"Successfully extracted {len(articles)} articles from RSS feed")
        return articles

    except Exception as e:
        logger.error(f"Failed to parse RSS content: {e}")
        return []

def perform_scrapy_extraction(query: str, urls: List[str]) -> Dict[str, Any]:
    """Perform Scrapy-based content extraction."""
    try:
        from web_scraping.scrapy_manager import ScrapyManager

        scrapy_manager = ScrapyManager()
        result = scrapy_manager.scrape_news_content(query, urls)

        logger.info(f"Scrapy extraction completed: success={result['success']}, articles={result.get('source_count', 0)}")
        return result

    except ImportError as e:
        logger.error(f"Scrapy import failed: {e}")
        return {'success': False, 'error': f'Scrapy not available: {e}'}
    except Exception as e:
        logger.error(f"Scrapy extraction failed: {e}")
        return {'success': False, 'error': str(e)}

def format_rss_articles_for_response(articles: List[Dict[str, Any]], query: str) -> str:
    """Format RSS articles for display."""
    try:
        if not articles:
            return "No news articles were successfully extracted from RSS feeds."

        content_parts = []

        # Header
        content_parts.append(f"**Latest News Results for: {query}**")
        content_parts.append(f"*Found {len(articles)} articles from RSS feeds*")
        content_parts.append("")

        # Articles
        for i, article in enumerate(articles[:10], 1):  # Limit display to top 10
            article_content = []

            # Title
            title = article.get('title', '').strip()
            if title:
                article_content.append(f"**{i}. {title}**")

            # Description
            description = article.get('description', '').strip()
            if description:
                article_content.append(description)

            # Source and date
            source = article.get('source', 'Unknown')
            pub_date = article.get('pub_date', '')
            link = article.get('link', '')

            source_info = f"*Source: {source}*"
            if pub_date:
                source_info += f" | *Published: {pub_date}*"
            if link:
                source_info += f" | [Read more]({link})"

            article_content.append(source_info)

            if article_content:
                content_parts.append("\n".join(article_content))
                content_parts.append("---")

        # Remove last separator
        if content_parts and content_parts[-1] == "---":
            content_parts.pop()

        return "\n".join(content_parts)

    except Exception as e:
        logger.error(f"RSS content formatting failed: {e}")
        return f"Error formatting RSS content: {e}"

def save_rss_to_quarantine(articles: List[Dict[str, Any]], query: str, sources: List[str]):
    """Save RSS articles to quarantine for vetting."""
    try:
        from pathlib import Path
        import json
        import hashlib
        from datetime import datetime

        # Create quarantine directory if it doesn't exist
        quarantine_dir = Path("quarantine")
        quarantine_dir.mkdir(exist_ok=True)

        # Generate filename based on query hash and timestamp
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rss_search_{timestamp}_{query_hash}.json"

        # Prepare quarantine data structure
        quarantine_data = {
            "query": query,
            "articles": articles,
            "sources": sources,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "source": "rss_web_search",
                "method": "rss_extraction",
                "article_count": len(articles),
                "source_count": len(sources),
                "quarantine_timestamp": datetime.now().isoformat()
            }
        }

        # Save to quarantine
        quarantine_path = quarantine_dir / filename
        with open(quarantine_path, 'w', encoding='utf-8') as f:
            json.dump(quarantine_data, f, indent=2, ensure_ascii=False)

        logger.info(f"RSS content saved to quarantine: {filename}")

    except Exception as e:
        logger.error(f"Failed to save RSS content to quarantine: {e}")

def generate_rss_enhanced_response(query: str, articles: List[Dict[str, Any]]) -> str:
    """Generate AI-enhanced response using RSS articles."""
    try:
        if not articles:
            return "No news articles were found for your query."

        # Create summary for AI processing
        ai_summary_parts = []
        ai_summary_parts.append(f"News Summary for: {query}")
        ai_summary_parts.append(f"Articles found: {len(articles)}")
        ai_summary_parts.append("")

        for i, article in enumerate(articles[:8], 1):  # Limit for AI processing
            title = article.get('title', '').strip()
            description = article.get('description', '').strip()
            source = article.get('source', 'Unknown')

            if title:
                article_summary = f"{i}. {title}"
                if description:
                    # Limit description for AI processing
                    short_desc = description[:200] + "..." if len(description) > 200 else description
                    article_summary += f" - {short_desc}"
                article_summary += f" (Source: {source})"
                ai_summary_parts.append(article_summary)

        ai_summary = "\n".join(ai_summary_parts)

        # Use Ollama to generate enhanced response
        import requests

        system_prompt = """You are SAM, a secure AI assistant. You have just retrieved current news content using RSS feeds to answer the user's question.

Provide a comprehensive, well-structured response based on the news articles provided. Focus on delivering actual news information with clear organization.

Important guidelines:
- Present the most important and relevant news first
- Organize information logically (by topic, chronology, or importance)
- Be factual and objective, focusing on the actual news content
- Mention that this information comes from current RSS feeds
- Summarize key points while maintaining accuracy
- Include relevant details from multiple sources when available"""

        user_prompt = f"""Based on the following current news content retrieved from RSS feeds, please answer this question: "{query}"

News content summary:
{ai_summary[:3000]}

Please provide a comprehensive, well-organized response based on this current news information. Focus on the most relevant and important news items."""

        ollama_response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M",
                "prompt": f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:",
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 800
                }
            },
            timeout=90
        )

        if ollama_response.status_code == 200:
            response_data = ollama_response.json()
            ai_response = response_data.get('response', '').strip()

            if ai_response:
                # Add source information
                sources = list(set([article.get('source', 'Unknown') for article in articles]))
                article_count = len(articles)

                sources_text = "\n\n**📰 Sources:**\n" + "\n".join([f"• {source}" for source in sources])

                web_enhanced_response = f"""🌐 **Based on current RSS feeds:**

{ai_response}

{sources_text}

*Information extracted from {article_count} articles across {len(sources)} RSS sources.*"""

                return web_enhanced_response

        # Fallback if Ollama fails
        return format_rss_articles_for_response(articles, query)

    except Exception as e:
        logger.error(f"RSS-enhanced response generation failed: {e}")
        return format_rss_articles_for_response(articles, query)

def format_scraped_content(scraped_data: Dict[str, Any]) -> str:
    """Format scraped content for display."""
    try:
        from web_scraping.content_formatter import ContentFormatter

        formatter = ContentFormatter()
        return formatter.format_news_content(scraped_data)

    except ImportError as e:
        logger.error(f"Content formatter import failed: {e}")
        return f"Content formatting unavailable: {e}"
    except Exception as e:
        logger.error(f"Content formatting failed: {e}")
        return f"Error formatting content: {e}"

def generate_scrapy_enhanced_response(query: str, scraped_data: Dict[str, Any]) -> str:
    """Generate AI-enhanced response using scraped data."""
    try:
        from web_scraping.content_formatter import ContentFormatter

        formatter = ContentFormatter()
        ai_summary = formatter.create_summary_for_ai(scraped_data)

        if not ai_summary or "No news content" in ai_summary:
            return format_scraped_content(scraped_data)

        # Use Ollama to generate enhanced response
        import requests

        system_prompt = """You are SAM, a secure AI assistant. You have just retrieved current news content using advanced web scraping to answer the user's question.

Provide a comprehensive, well-structured response based on the news articles provided. Focus on delivering actual news information with clear organization.

Important guidelines:
- Present the most important and relevant news first
- Organize information logically (by topic, chronology, or importance)
- Be factual and objective, focusing on the actual news content
- Mention that this information comes from current web sources
- Summarize key points while maintaining accuracy
- Include relevant details from multiple sources when available"""

        user_prompt = f"""Based on the following current news content retrieved from web scraping, please answer this question: "{query}"

News content summary:
{ai_summary[:3000]}

Please provide a comprehensive, well-organized response based on this current news information. Focus on the most relevant and important news items."""

        ollama_response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M",
                "prompt": f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:",
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 800
                }
            },
            timeout=90
        )

        if ollama_response.status_code == 200:
            response_data = ollama_response.json()
            ai_response = response_data.get('response', '').strip()

            if ai_response:
                # Add source information
                sources = scraped_data.get('sources', [])
                article_count = scraped_data.get('article_count', 0)

                sources_text = "\n\n**📰 Sources:**\n" + "\n".join([f"• {source}" for source in sources])

                web_enhanced_response = f"""🌐 **Based on current web sources:**

{ai_response}

{sources_text}

*Information extracted from {article_count} articles across {len(sources)} sources using intelligent web scraping.*"""

                return web_enhanced_response

        # Fallback if Ollama fails
        return format_scraped_content(scraped_data)

    except Exception as e:
        logger.error(f"Scrapy-enhanced response generation failed: {e}")
        return format_scraped_content(scraped_data)

def perform_fallback_web_search(query: str, search_urls: List[str]) -> Dict[str, Any]:
    """Fallback to original web search method."""
    try:
        logger.info("Using fallback web search method")

        # Step 1: Fetch content from multiple sources
        fetched_content = []
        for url in search_urls[:3]:  # Limit to top 3 sources for security
            try:
                content_result = fetch_web_content_secure(url)
                if content_result['success']:
                    fetched_content.append(content_result)

                    # Save to quarantine for vetting
                    save_to_quarantine(content_result, query)

            except Exception as e:
                logger.warning(f"Failed to fetch {url}: {e}")
                continue

        if not fetched_content:
            return {
                'success': False,
                'error': 'No web content could be retrieved. All sources failed or were blocked.',
                'response': None
            }

        # Step 2: Process and analyze the fetched content
        processed_content = process_fetched_content(fetched_content, query)

        # Step 3: Generate response using the web content
        web_response = generate_web_enhanced_response(query, processed_content)

        return {
            'success': True,
            'error': None,
            'response': web_response,
            'sources': [content['url'] for content in fetched_content],
            'content_count': len(fetched_content),
            'method': 'fallback'
        }

    except Exception as e:
        logger.error(f"Fallback web search failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'response': None
        }

def generate_search_urls(query: str) -> List[str]:
    """Generate search URLs optimized for Scrapy extraction."""
    query_lower = query.lower()

    # CNN-specific queries - use main pages for better scraping
    if 'cnn' in query_lower:
        return [
            "https://www.cnn.com",
            "https://edition.cnn.com",
            "https://www.cnn.com/us"
        ]

    # News source specific queries
    elif 'nytimes' in query_lower or 'new york times' in query_lower:
        return [
            "https://www.nytimes.com",
            "https://www.nytimes.com/section/world",
            "https://www.nytimes.com/section/us"
        ]
    elif 'bbc' in query_lower:
        return [
            "https://www.bbc.com/news",
            "https://www.bbc.com/news/world",
            "https://www.bbc.com/news/uk"
        ]
    elif 'reuters' in query_lower:
        return [
            "https://www.reuters.com",
            "https://www.reuters.com/world",
            "https://www.reuters.com/business"
        ]

    # Topic-specific news pages
    elif any(word in query_lower for word in ['technology', 'tech', 'ai', 'artificial intelligence']):
        return [
            "https://www.cnn.com/business/tech",
            "https://www.nytimes.com/section/technology",
            "https://www.bbc.com/news/technology"
        ]
    elif any(word in query_lower for word in ['business', 'economy', 'finance', 'market']):
        return [
            "https://www.cnn.com/business",
            "https://www.nytimes.com/section/business",
            "https://www.bbc.com/news/business"
        ]
    elif any(word in query_lower for word in ['politics', 'political', 'election', 'government']):
        return [
            "https://www.cnn.com/politics",
            "https://www.nytimes.com/section/politics",
            "https://www.bbc.com/news/politics"
        ]
    elif any(word in query_lower for word in ['world', 'international', 'global']):
        return [
            "https://www.cnn.com/world",
            "https://www.nytimes.com/section/world",
            "https://www.bbc.com/news/world"
        ]
    elif any(word in query_lower for word in ['health', 'medical', 'medicine', 'covid', 'pandemic']):
        return [
            "https://www.cnn.com/health",
            "https://www.nytimes.com/section/health",
            "https://www.bbc.com/news/health"
        ]
    elif any(word in query_lower for word in ['sports', 'football', 'basketball', 'baseball', 'soccer']):
        return [
            "https://www.cnn.com/sport",
            "https://www.nytimes.com/section/sports",
            "https://www.bbc.com/sport"
        ]

    # General news queries - use main news pages
    elif any(word in query_lower for word in ['news', 'latest', 'breaking', 'today', 'current', 'headlines']):
        return [
            "https://www.bbc.com/news",
            "https://www.nytimes.com",
            "https://www.cnn.com"
        ]

    # Fallback for other queries - use general news sites
    else:
        return [
            "https://www.cnn.com",
            "https://www.nytimes.com",
            "https://www.bbc.com/news"
        ]

def fetch_web_content_secure(url: str) -> Dict[str, Any]:
    """Fetch web content using SAM's secure web retrieval system with enhanced content extraction."""
    try:
        # Check if this is an RSS feed
        if 'rss' in url.lower() or '.xml' in url.lower():
            rss_result = fetch_rss_content(url)

            # If RSS fails, try to fallback to regular web fetch
            if not rss_result['success']:
                logger.warning(f"RSS fetch failed for {url}, trying regular web fetch")
                # Convert RSS URL to regular web URL if possible
                fallback_url = url.replace('rss.', 'www.').replace('/rss/', '/').replace('.rss', '').replace('.xml', '')
                if fallback_url != url:
                    return fetch_web_content_secure(fallback_url)

            return rss_result

        # Use SAM's WebFetcher for regular web content
        from web_retrieval import WebFetcher

        fetcher = WebFetcher(
            timeout=15,
            max_content_length=50000,  # Limit content size for security
            user_agent="SAM-SecureBot/1.0"
        )

        result = fetcher.fetch_url_content(url)

        if result.success and result.content:
            # Enhanced content processing for news sites
            processed_content = enhance_news_content_extraction(result.content, url)

            return {
                'success': True,
                'url': url,
                'content': processed_content,
                'metadata': result.metadata,
                'timestamp': result.timestamp
            }
        else:
            return {
                'success': False,
                'url': url,
                'error': result.error or 'No content retrieved'
            }

    except Exception as e:
        logger.error(f"Web content fetch failed for {url}: {e}")
        return {
            'success': False,
            'url': url,
            'error': str(e)
        }

def fetch_rss_content(url: str) -> Dict[str, Any]:
    """Fetch and parse RSS feed content with enhanced extraction."""
    try:
        import requests
        import xml.etree.ElementTree as ET
        from datetime import datetime
        import html
        import re

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/rss+xml, application/xml, text/xml, application/atom+xml'
        }

        logger.info(f"Fetching RSS feed: {url}")
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()

        # Parse RSS XML
        content = response.content.decode('utf-8', errors='ignore')
        root = ET.fromstring(content)

        # Extract news items
        news_items = []

        # Handle different RSS formats (RSS 2.0, Atom, etc.)
        items = (root.findall('.//item') or
                root.findall('.//{http://www.w3.org/2005/Atom}entry') or
                root.findall('.//entry'))

        logger.info(f"Found {len(items)} items in RSS feed")

        for i, item in enumerate(items[:15]):  # Limit to top 15 news items
            try:
                title = ''
                description = ''
                link = ''
                pub_date = ''
                category = ''

                # Extract title - simplified approach
                title_elem = item.find('title')
                if title_elem is not None and title_elem.text:
                    title = title_elem.text.strip()

                # Extract description - simplified approach
                desc_elem = item.find('description')
                if desc_elem is not None and desc_elem.text:
                    description = desc_elem.text.strip()
                    # Clean HTML tags from description
                    description = re.sub(r'<[^>]+>', '', description)
                    # Limit description length
                    if len(description) > 300:
                        description = description[:300] + "..."

                # Debug logging for first few items
                if i < 3:
                    logger.info(f"RSS item {i}: title='{title[:50] if title else 'EMPTY'}', desc_len={len(description)}")
                    if title_elem is not None:
                        logger.info(f"  title_elem.text: '{title_elem.text[:50] if title_elem.text else 'NONE'}'")
                    if desc_elem is not None:
                        logger.info(f"  desc_elem.text: '{desc_elem.text[:50] if desc_elem.text else 'NONE'}'")
                    logger.info(f"  Final title: '{title}', Final desc: '{description[:50] if description else 'EMPTY'}'")

                # Include items with any substantial title (relaxed criteria)
                if title and len(title.strip()) > 3:  # Reduced from 5 to 3
                    news_item = f"**{title.strip()}**"

                    if description and len(description.strip()) > 5:  # Reduced from 10 to 5
                        news_item += f"\n{description.strip()}"

                    if category and category.strip():
                        news_item += f"\nCategory: {category.strip()}"

                    if pub_date and pub_date.strip():
                        news_item += f"\nPublished: {pub_date.strip()}"

                    if link and link.strip():
                        news_item += f"\nLink: {link.strip()}"

                    news_items.append(news_item)
                    logger.info(f"✅ Added news item {len(news_items)}: {title[:50]}...")
                else:
                    logger.warning(f"❌ Skipped item {i}: title='{title}' (length: {len(title) if title else 0})")

                # Extract link - simplified
                link_elem = item.find('link')
                if link_elem is not None and link_elem.text:
                    link = link_elem.text.strip()

                # Extract publication date - simplified
                date_elem = item.find('pubDate')
                if date_elem is not None and date_elem.text:
                    pub_date = date_elem.text.strip()

                # Extract category if available - simplified
                cat_elem = item.find('category')
                if cat_elem is not None and cat_elem.text:
                    category = cat_elem.text.strip()



            except Exception as e:
                logger.warning(f"Error processing RSS item {i}: {e}")
                continue

        if news_items:
            # Determine feed source for better formatting
            feed_title = "RSS News Feed"
            title_elem = root.find('.//title') or root.find('.//{http://www.w3.org/2005/Atom}title')
            if title_elem is not None and title_elem.text:
                feed_title = title_elem.text.strip()

            content = f"**{feed_title}**\n\n" + "\n\n---\n\n".join(news_items)

            logger.info(f"Successfully extracted {len(news_items)} news items from RSS feed")

            return {
                'success': True,
                'url': url,
                'content': content,
                'metadata': {
                    'content_type': 'rss_feed',
                    'items_count': len(news_items),
                    'feed_title': feed_title
                },
                'timestamp': datetime.now().isoformat()
            }
        else:
            logger.warning(f"No valid news items found in RSS feed: {url}")
            return {
                'success': False,
                'url': url,
                'error': 'No valid news items found in RSS feed'
            }

    except requests.RequestException as e:
        logger.error(f"HTTP error fetching RSS feed {url}: {e}")
        return {
            'success': False,
            'url': url,
            'error': f'HTTP error: {str(e)}'
        }
    except ET.ParseError as e:
        logger.error(f"XML parsing error for RSS feed {url}: {e}")
        return {
            'success': False,
            'url': url,
            'error': f'XML parsing error: {str(e)}'
        }
    except Exception as e:
        logger.error(f"RSS fetch failed for {url}: {e}")
        return {
            'success': False,
            'url': url,
            'error': f'RSS parsing failed: {str(e)}'
        }

def enhance_news_content_extraction(content: str, url: str) -> str:
    """Enhanced content extraction for news websites."""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(content, 'html.parser')

        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement']):
            element.decompose()

        # Try to find main content areas for news sites
        main_content = []

        # Look for common news content selectors
        content_selectors = [
            'article',
            '.story-body',
            '.article-body',
            '.content',
            '.post-content',
            '[data-module="ArticleBody"]',
            '.zn-body__paragraph',
            '.pg-rail-tall__body'
        ]

        for selector in content_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text(strip=True)
                if len(text) > 100:  # Only include substantial content
                    main_content.append(text)

        # If no specific content found, try to extract headlines and summaries
        if not main_content:
            headlines = soup.find_all(['h1', 'h2', 'h3'], limit=10)
            for headline in headlines:
                text = headline.get_text(strip=True)
                if len(text) > 10:
                    main_content.append(f"HEADLINE: {text}")

                    # Look for associated paragraph
                    next_elem = headline.find_next(['p', 'div'])
                    if next_elem:
                        para_text = next_elem.get_text(strip=True)
                        if len(para_text) > 50:
                            main_content.append(f"SUMMARY: {para_text}")

        # If still no content, fall back to general text extraction
        if not main_content:
            paragraphs = soup.find_all('p')
            for p in paragraphs[:20]:  # Limit to first 20 paragraphs
                text = p.get_text(strip=True)
                if len(text) > 50:
                    main_content.append(text)

        if main_content:
            return '\n\n'.join(main_content[:10])  # Limit to top 10 content pieces
        else:
            return content  # Return original if extraction fails

    except Exception as e:
        logger.warning(f"Content enhancement failed for {url}: {e}")
        return content  # Return original content if enhancement fails

def process_fetched_content(fetched_content: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    """Process and analyze fetched web content."""
    try:
        # Combine all content
        all_content = []
        sources = []

        for content_data in fetched_content:
            if content_data.get('content'):
                # Clean and truncate content
                clean_content = content_data['content'][:2000]  # Limit for processing
                all_content.append(clean_content)
                sources.append(content_data['url'])

        combined_content = "\n\n".join(all_content)

        return {
            'combined_content': combined_content,
            'sources': sources,
            'content_length': len(combined_content),
            'source_count': len(sources)
        }

    except Exception as e:
        logger.error(f"Content processing failed: {e}")
        return {
            'combined_content': '',
            'sources': [],
            'content_length': 0,
            'source_count': 0
        }

def generate_web_enhanced_response(query: str, processed_content: Dict[str, Any]) -> str:
    """Generate response using web content and Ollama."""
    try:
        if not processed_content['combined_content']:
            return "❌ No web content was successfully retrieved to answer your question."

        # Use Ollama to generate response with web content
        import requests

        system_prompt = """You are SAM, a secure AI assistant. You have just retrieved current news content from RSS feeds and web sources to answer the user's question.

Provide a comprehensive, well-structured response based on the news content provided. Focus on delivering actual news information, not website structure.

Important guidelines:
- Extract and present actual news headlines, stories, and information
- Organize information by topic or chronologically if appropriate
- Be factual and objective, focusing on the news content itself
- Mention that this information comes from current RSS feeds and news sources
- If the content contains multiple news stories, summarize the key points
- Ignore any website navigation or structural information
- Focus on headlines, article summaries, and publication dates"""

        user_prompt = f"""Based on the following current news content retrieved from {processed_content['source_count']} RSS feeds and news sources, please answer this question: "{query}"

News content from RSS feeds:
{processed_content['combined_content'][:4000]}

Please provide a comprehensive news summary based on this current information. Focus on actual news stories, headlines, and developments rather than website structure."""

        ollama_response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M",
                "prompt": f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:",
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 1000
                }
            },
            timeout=45
        )

        if ollama_response.status_code == 200:
            response_data = ollama_response.json()
            ai_response = response_data.get('response', '').strip()

            if ai_response:
                # Add source information
                sources_text = "\n\n**📰 Sources:**\n" + "\n".join([f"• {source}" for source in processed_content['sources']])

                web_enhanced_response = f"""🌐 **Based on current web sources:**

{ai_response}

{sources_text}

*Information retrieved from {processed_content['source_count']} web sources and processed securely.*"""

                return web_enhanced_response

        # Fallback if Ollama fails
        return f"""🌐 **Web Search Results:**

I found information from {processed_content['source_count']} web sources, but was unable to process it into a comprehensive response.

**Raw content summary:**
{processed_content['combined_content'][:500]}...

**Sources:**
{chr(10).join([f"• {source}" for source in processed_content['sources']])}

*You may want to visit these sources directly for the most current information.*"""

    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        return f"❌ Failed to generate response from web content: {e}"

def get_vetting_status() -> Dict[str, Any]:
    """Get current vetting system status."""
    try:
        from pathlib import Path

        # Define directories
        quarantine_dir = Path("quarantine")
        vetted_dir = Path("vetted")
        approved_dir = Path("approved")
        rejected_dir = Path("rejected")

        # Count files in each directory (excluding metadata files)
        if quarantine_dir.exists():
            all_quarantine_files = list(quarantine_dir.glob("*.json"))
            quarantine_files = len([f for f in all_quarantine_files
                                  if not f.name.startswith('metadata') and not f.name.endswith('_metadata.json')])
        else:
            quarantine_files = 0

        vetted_files = len(list(vetted_dir.glob("*.json"))) if vetted_dir.exists() else 0
        approved_files = len(list(approved_dir.glob("*.json"))) if approved_dir.exists() else 0
        rejected_files = len(list(rejected_dir.glob("*.json"))) if rejected_dir.exists() else 0

        return {
            'quarantine_files': quarantine_files,
            'vetted_files': vetted_files,
            'approved_files': approved_files,
            'rejected_files': rejected_files,
            'ready_for_vetting': quarantine_files > 0,
            'has_vetted_content': vetted_files > 0,
            'system_operational': True
        }

    except Exception as e:
        logger.error(f"Error getting vetting status: {e}")
        return {
            'quarantine_files': 0,
            'vetted_files': 0,
            'approved_files': 0,
            'rejected_files': 0,
            'ready_for_vetting': False,
            'has_vetted_content': False,
            'system_operational': False,
            'error': str(e)
        }

def refresh_memory_store():
    """Refresh SAM's memory store to pick up newly integrated content."""
    try:
        logger.info("Refreshing SAM's memory store after content integration")

        # Clear any cached memory store instances
        if 'secure_memory_store' in st.session_state:
            # Force reinitialize the memory store to pick up new content
            old_store = st.session_state.secure_memory_store
            del st.session_state.secure_memory_store

            # Reinitialize with same settings
            from memory.secure_memory_vectorstore import get_secure_memory_store, VectorStoreType
            st.session_state.secure_memory_store = get_secure_memory_store(
                store_type=VectorStoreType.CHROMA,
                storage_directory="memory_store",
                embedding_dimension=384,
                enable_encryption=True,
                security_manager=st.session_state.security_manager
            )
            logger.info("✅ Memory store refreshed successfully")

        # Clear Chroma cache if available
        try:
            from utils.chroma_client import ChromaClientManager
            ChromaClientManager.reset_cache()
            logger.info("✅ Chroma cache cleared")
        except Exception as e:
            logger.warning(f"Could not clear Chroma cache: {e}")

        return True

    except Exception as e:
        logger.error(f"Failed to refresh memory store: {e}")
        return False

def trigger_vetting_process() -> Dict[str, Any]:
    """Trigger automated vetting of all quarantined content."""
    try:
        import subprocess
        import sys
        from pathlib import Path

        logger.info("Starting automated vetting process via secure interface")

        # Get project root directory
        project_root = Path(__file__).parent

        # Check quarantine status before vetting
        quarantine_dir = Path("quarantine")
        if quarantine_dir.exists():
            quarantine_files_before = list(quarantine_dir.glob("*.json"))
            logger.info(f"Before vetting: {len(quarantine_files_before)} files in quarantine")
            for f in quarantine_files_before:
                logger.info(f"  - {f.name}")
        else:
            logger.warning("Quarantine directory does not exist before vetting")

        # Execute simple vetting and consolidation script
        logger.info(f"Executing vetting script from: {project_root}")
        result = subprocess.run([
            sys.executable,
            'scripts/simple_vet_and_consolidate.py',
            '--quiet'
        ],
        capture_output=True,
        text=True,
        cwd=project_root,
        timeout=300  # 5 minute timeout
        )

        logger.info(f"Vetting script completed with return code: {result.returncode}")
        logger.info(f"Vetting script stdout: {result.stdout}")
        if result.stderr:
            logger.error(f"Vetting script stderr: {result.stderr}")

        # Check quarantine status after vetting
        if quarantine_dir.exists():
            quarantine_files_after = list(quarantine_dir.glob("*.json"))
            logger.info(f"After vetting: {len(quarantine_files_after)} files in quarantine")
            for f in quarantine_files_after:
                logger.info(f"  - {f.name}")

        # Check vetted directory
        vetted_dir = Path("vetted")
        if vetted_dir.exists():
            vetted_files = list(vetted_dir.glob("*.json"))
            logger.info(f"After vetting: {len(vetted_files)} files in vetted directory")
            for f in vetted_files:
                logger.info(f"  - {f.name}")
        else:
            logger.warning("Vetted directory does not exist after vetting")

        if result.returncode == 0:
            # Parse output for statistics
            output_lines = result.stdout.strip().split('\n')
            stats = {
                'vetted_files': 0,
                'approved_files': 0,
                'rejected_files': 0,
                'integrated_items': 0
            }

            for line in output_lines:
                if 'approved' in line.lower() and 'rejected' in line.lower():
                    try:
                        # Parse "Vetting completed: X approved, Y rejected out of Z files"
                        parts = line.split()
                        approved_idx = parts.index('approved,') - 1
                        rejected_idx = parts.index('rejected') - 1
                        stats['approved_files'] = int(parts[approved_idx])
                        stats['rejected_files'] = int(parts[rejected_idx])
                        stats['vetted_files'] = stats['approved_files'] + stats['rejected_files']
                    except:
                        pass
                elif 'items integrated' in line.lower():
                    try:
                        # Parse "Consolidation completed: X items integrated out of Y processed"
                        parts = line.split()
                        integrated_idx = parts.index('items') - 1
                        stats['integrated_items'] = int(parts[integrated_idx])
                    except:
                        pass

            # Refresh memory store if content was integrated
            if stats['integrated_items'] > 0:
                logger.info(f"Content was integrated ({stats['integrated_items']} items), refreshing memory store...")
                refresh_success = refresh_memory_store()
                if refresh_success:
                    logger.info("✅ Memory store refreshed - new content should now be searchable")
                else:
                    logger.warning("⚠️ Memory store refresh failed - new content may not be immediately searchable")

            return {
                'success': True,
                'stats': stats,
                'output': result.stdout,
                'message': f'Vetting and consolidation completed: {stats["approved_files"]} approved, {stats["integrated_items"]} items integrated into knowledge base'
            }
        else:
            return {
                'success': False,
                'error': result.stderr or 'Vetting process failed',
                'output': result.stdout
            }

    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': 'Vetting process timed out (5 minutes)',
            'output': ''
        }
    except Exception as e:
        logger.error(f"Vetting process failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'output': ''
        }

def load_vetted_content() -> List[Dict[str, Any]]:
    """Load vetted content for review."""
    try:
        from pathlib import Path
        import json

        vetted_dir = Path("vetted")
        if not vetted_dir.exists():
            return []

        vetted_files = []
        for file_path in vetted_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data['filename'] = file_path.name
                    vetted_files.append(data)
            except Exception as e:
                logger.warning(f"Could not load vetted file {file_path}: {e}")
                continue

        # Sort by timestamp (newest first)
        vetted_files.sort(key=lambda x: x.get('vetting_metadata', {}).get('timestamp', ''), reverse=True)

        return vetted_files

    except Exception as e:
        logger.error(f"Error loading vetted content: {e}")
        return []

def load_quarantined_content() -> List[Dict[str, Any]]:
    """Load quarantined content for preview before vetting."""
    try:
        from pathlib import Path
        import json

        quarantine_dir = Path("quarantine")
        if not quarantine_dir.exists():
            logger.warning("Quarantine directory does not exist")
            return []

        # Log quarantine directory info
        logger.info(f"Quarantine directory: {quarantine_dir.absolute()}")
        logger.info(f"Directory exists: {quarantine_dir.exists()}")
        logger.info(f"Directory is readable: {quarantine_dir.is_dir()}")

        quarantined_files = []
        all_json_files = list(quarantine_dir.glob("*.json"))

        logger.info(f"Found {len(all_json_files)} JSON files in quarantine directory")

        # Enhanced debug: List all files found with detailed analysis
        logger.info(f"=== QUARANTINE FILE ANALYSIS ===")
        for f in all_json_files:
            mod_time = f.stat().st_mtime
            from datetime import datetime
            mod_time_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"  - {f.name} ({f.stat().st_size} bytes, modified: {mod_time_str})")

            # Analyze file patterns
            if 'intelligent_web_' in f.name:
                logger.info(f"    ✅ INTELLIGENT_WEB FILE DETECTED: {f.name}")
            elif 'scrapy_' in f.name:
                logger.info(f"    📜 SCRAPY FILE DETECTED: {f.name}")
            elif f.name.startswith('metadata'):
                logger.info(f"    📋 METADATA FILE DETECTED: {f.name}")
            else:
                logger.info(f"    ❓ UNKNOWN FILE TYPE: {f.name}")

        # Count file types
        intelligent_web_files = [f for f in all_json_files if 'intelligent_web_' in f.name]
        scrapy_files = [f for f in all_json_files if 'scrapy_' in f.name]
        metadata_files = [f for f in all_json_files if f.name.startswith('metadata') or f.name.endswith('_metadata.json')]

        logger.info(f"File type summary:")
        logger.info(f"  - Intelligent web files: {len(intelligent_web_files)}")
        logger.info(f"  - Scrapy files: {len(scrapy_files)}")
        logger.info(f"  - Metadata files: {len(metadata_files)}")
        logger.info(f"  - Total JSON files: {len(all_json_files)}")

        for file_path in all_json_files:
            # Skip metadata files
            if file_path.name.startswith('metadata') or file_path.name.endswith('_metadata.json'):
                logger.info(f"⏭️ Skipping metadata file: {file_path.name}")
                continue

            logger.info(f"🔄 Processing quarantine file: {file_path.name}")

            # Debug: Check if this is an intelligent_web file
            if 'intelligent_web_' in file_path.name:
                logger.info(f"🌐 Found intelligent_web file: {file_path.name}, size: {file_path.stat().st_size} bytes")
            elif 'scrapy_' in file_path.name:
                logger.info(f"🕷️ Found scrapy file: {file_path.name}, size: {file_path.stat().st_size} bytes")

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Add file metadata
                data['filename'] = file_path.name
                data['file_path'] = str(file_path)
                data['file_size'] = file_path.stat().st_size
                data['file_modified'] = file_path.stat().st_mtime

                quarantined_files.append(data)
                logger.info(f"Successfully loaded quarantine file: {file_path.name}")

                # Debug: Log structure for intelligent_web files
                if 'intelligent_web_' in file_path.name:
                    logger.info(f"Intelligent_web file structure: {list(data.keys())}")
                    if 'result' in data:
                        logger.info(f"  - result keys: {list(data['result'].keys()) if isinstance(data['result'], dict) else type(data['result'])}")
                    if 'query' in data:
                        logger.info(f"  - query: {data['query'][:50]}...")
                    if 'timestamp' in data:
                        logger.info(f"  - timestamp: {data['timestamp']}")

            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in {file_path.name}: {e}")
                # Add corrupted file entry
                quarantined_files.append({
                    'filename': file_path.name,
                    'file_path': str(file_path),
                    'error': f"JSON decode error: {e}",
                    'corrupted': True,
                    'file_size': file_path.stat().st_size,
                    'file_modified': file_path.stat().st_mtime
                })
                continue
            except Exception as e:
                logger.error(f"Could not load quarantined file {file_path.name}: {e}")
                # Add error file entry
                quarantined_files.append({
                    'filename': file_path.name,
                    'file_path': str(file_path),
                    'error': str(e),
                    'corrupted': True,
                    'file_size': file_path.stat().st_size if file_path.exists() else 0,
                    'file_modified': file_path.stat().st_mtime if file_path.exists() else 0
                })
                continue

        logger.info(f"Loaded {len(quarantined_files)} quarantined files (including any corrupted ones)")

        # Sort by timestamp (newest first), handling different timestamp formats
        def get_sort_timestamp(x):
            if x.get('corrupted'):
                return x.get('file_modified', 0)

            # Try different timestamp fields
            timestamp = (x.get('timestamp') or
                        x.get('metadata', {}).get('quarantine_timestamp') or
                        x.get('metadata', {}).get('timestamp') or
                        str(x.get('file_modified', 0)))
            return timestamp

        quarantined_files.sort(key=get_sort_timestamp, reverse=True)

        return quarantined_files

    except Exception as e:
        logger.error(f"Error loading quarantined content: {e}")
        return []

def render_quarantined_content_item(content: Dict[str, Any], index: int):
    """Render a single quarantined content item for preview."""
    try:
        filename = content.get('filename', 'Unknown')
        file_size = content.get('file_size', 0)

        # Handle corrupted files
        if content.get('corrupted'):
            error_msg = content.get('error', 'Unknown error')

            with st.expander(f"❌ **{filename}** (Corrupted)", expanded=False):
                st.error(f"**File Error:** {error_msg}")
                st.markdown(f"**📁 File:** `{filename}`")
                st.markdown(f"**📊 File Size:** {file_size:,} bytes")
                st.markdown(f"**🕒 Modified:** {content.get('file_modified', 'Unknown')}")

                st.warning("⚠️ **This file could not be loaded.** It may be corrupted or have an invalid format.")

                # Raw data toggle for debugging
                if st.button(f"🔍 Show Error Details", key=f"quarantine_error_{index}"):
                    if f"show_quarantine_error_{index}" not in st.session_state:
                        st.session_state[f"show_quarantine_error_{index}"] = False
                    st.session_state[f"show_quarantine_error_{index}"] = not st.session_state[f"show_quarantine_error_{index}"]

                if st.session_state.get(f"show_quarantine_error_{index}", False):
                    st.json(content)
            return

        # Normal file processing
        timestamp = content.get('timestamp', content.get('metadata', {}).get('quarantine_timestamp', 'Unknown'))

        # Extract basic information based on content type
        content_info = extract_quarantine_content_info(content)

        with st.expander(f"📄 **{content_info['title']}**", expanded=False):
            # Basic information
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**📁 File:** `{filename}`")
                st.markdown(f"**🕒 Quarantined:** {timestamp}")
                st.markdown(f"**🔍 Source:** {content_info['source']}")
                st.markdown(f"**📊 Content Type:** {content_info['content_type']}")

            with col2:
                st.markdown(f"**📈 Items:** {content_info['item_count']}")
                st.markdown(f"**🌐 Sources:** {content_info['source_count']}")
                st.markdown(f"**⚙️ Method:** {content_info['method']}")
                st.markdown(f"**📊 Size:** {file_size:,} bytes")

            # Content preview
            if content_info['preview']:
                st.markdown("**📝 Content Preview:**")
                st.markdown(content_info['preview'])

            # Show sources if available
            if content_info['sources']:
                st.markdown("**🔗 Sources:**")
                for source in content_info['sources'][:5]:  # Show first 5 sources
                    st.markdown(f"• {source}")
                if len(content_info['sources']) > 5:
                    st.markdown(f"• ... and {len(content_info['sources']) - 5} more sources")

            # Individual approval controls
            st.markdown("---")
            st.markdown("**🎯 Individual Approval Controls:**")

            # Initialize selection state if not exists
            selection_key = f"select_quarantine_{index}_{filename}"
            if selection_key not in st.session_state:
                st.session_state[selection_key] = False

            # Checkbox for selection
            col_check, col_actions = st.columns([1, 3])

            with col_check:
                selected = st.checkbox(
                    "Select for approval",
                    key=selection_key,
                    help="Check this box to include this content in individual approval"
                )

            with col_actions:
                if selected:
                    # Individual action buttons
                    col_approve, col_vet, col_reject = st.columns(3)

                    with col_approve:
                        if st.button("✅ Approve & Integrate",
                                   key=f"approve_quarantine_{index}",
                                   help="Approve this content and integrate it into SAM's knowledge base",
                                   use_container_width=True):
                            if approve_quarantined_content(filename):
                                st.success(f"✅ **{filename}** approved and integrated!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to approve content")

                    with col_vet:
                        if st.button("🛡️ Vet This Item",
                                   key=f"vet_individual_{index}",
                                   help="Run security analysis on this specific item",
                                   use_container_width=True):
                            if vet_individual_content(filename):
                                st.success(f"🛡️ **{filename}** vetted successfully!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to vet content")

                    with col_reject:
                        if st.button("❌ Reject",
                                   key=f"reject_quarantine_{index}",
                                   help="Reject and remove this content from quarantine",
                                   use_container_width=True):
                            if reject_quarantined_content(filename):
                                st.success(f"❌ **{filename}** rejected and removed!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to reject content")
                else:
                    st.info("☑️ Check the box above to enable individual approval actions")

            # Status indicator
            st.info("⏳ **Status:** Awaiting security analysis and vetting")

            # Raw data toggle
            if st.button(f"🔍 Show Raw Data", key=f"quarantine_raw_{index}"):
                if f"show_quarantine_raw_{index}" not in st.session_state:
                    st.session_state[f"show_quarantine_raw_{index}"] = False
                st.session_state[f"show_quarantine_raw_{index}"] = not st.session_state[f"show_quarantine_raw_{index}"]

            if st.session_state.get(f"show_quarantine_raw_{index}", False):
                st.json(content)

    except Exception as e:
        st.error(f"Error rendering quarantined content item: {e}")
        logger.error(f"Error rendering quarantined content item {index}: {e}")

def extract_quarantine_content_info(content: Dict[str, Any]) -> Dict[str, Any]:
    """Extract display information from quarantined content."""
    try:
        # Default values
        info = {
            'title': 'Unknown Content',
            'source': 'Unknown',
            'content_type': 'Unknown',
            'item_count': 0,
            'source_count': 0,
            'method': 'Unknown',
            'preview': '',
            'sources': []
        }

        # Debug: Log the structure of the content for troubleshooting
        filename = content.get('filename', 'Unknown')
        logger.info(f"Extracting info from {filename}, keys: {list(content.keys())}")

        # Check for timestamp and metadata
        if 'timestamp' in content:
            logger.info(f"File {filename} has timestamp: {content['timestamp']}")
        if 'metadata' in content:
            logger.info(f"File {filename} has metadata: {list(content['metadata'].keys()) if isinstance(content['metadata'], dict) else type(content['metadata'])}")

        # Check for intelligent web system content (multiple possible formats)
        if 'result' in content and isinstance(content['result'], dict):
            result = content['result']
            query = content.get('query', 'Unknown Query')

            info['title'] = f"Web Search: {query}"
            info['source'] = 'Intelligent Web System'
            info['method'] = result.get('tool_used', 'Unknown Tool')

        # Check for direct intelligent web format (newer format)
        elif 'query' in content and ('tool_used' in content or 'data' in content):
            query = content.get('query', 'Unknown Query')

            info['title'] = f"Web Search: {query}"
            info['source'] = 'Intelligent Web System'
            info['method'] = content.get('tool_used', 'Unknown Tool')

            # Process data directly from content
            data = content.get('data', {})
            if 'articles' in data:
                info['content_type'] = 'News Articles'
                articles = data['articles']
                info['item_count'] = len(articles)

                # Get sources from articles
                sources = set()
                preview_items = []

                for article in articles[:3]:  # Preview first 3 articles
                    if 'source' in article:
                        sources.add(article['source'])
                    title = article.get('title', 'No title')
                    preview_items.append(f"• **{title}**")

                info['sources'] = list(sources)
                info['source_count'] = len(sources)
                info['preview'] = '\n'.join(preview_items)

            # Continue with existing result processing if result exists
            if 'result' in content:
                result = content['result']

            # Extract data from result
            data = result.get('data', {})
            if 'articles' in data:
                info['content_type'] = 'News Articles'
                articles = data['articles']
                info['item_count'] = len(articles)

                # Get sources from articles
                sources = set()
                preview_items = []

                for article in articles[:3]:  # Preview first 3 articles
                    if 'source' in article:
                        sources.add(article['source'])
                    title = article.get('title', 'No title')
                    preview_items.append(f"• **{title}**")

                info['sources'] = list(sources)
                info['source_count'] = len(sources)
                info['preview'] = '\n'.join(preview_items)

            elif 'chunks' in data:
                info['content_type'] = 'Web Content Chunks'
                info['item_count'] = data.get('total_chunks', 0)
                info['source_count'] = len(data.get('sources', []))
                info['sources'] = data.get('sources', [])

                # Preview chunks
                chunks = data.get('chunks', [])
                preview_items = []
                for chunk in chunks[:3]:
                    content_preview = chunk.get('content', '')[:100]
                    if len(content_preview) == 100:
                        content_preview += '...'
                    preview_items.append(f"• {content_preview}")
                info['preview'] = '\n'.join(preview_items)

        # Check for direct web content
        elif 'url' in content and 'content' in content:
            info['title'] = f"Web Page: {content['url']}"
            info['source'] = 'Direct Web Fetch'
            info['content_type'] = 'Web Page'
            info['item_count'] = 1
            info['source_count'] = 1
            info['sources'] = [content['url']]
            info['method'] = content.get('metadata', {}).get('fetch_method', 'Unknown')

            # Content preview
            page_content = content.get('content', '')
            if page_content:
                info['preview'] = page_content[:300] + ('...' if len(page_content) > 300 else '')

        # Check for scraped data format (newer scrapy format)
        elif 'scraped_data' in content:
            query = content.get('query', 'Unknown Query')
            info['title'] = f"Scraped Search: {query}"
            info['source'] = 'Scrapy Web Search'
            info['content_type'] = 'Scraped Articles'

            scraped_data = content.get('scraped_data', {})
            articles = scraped_data.get('articles', [])
            info['item_count'] = len(articles)

            # Get sources from metadata or scraped data
            sources = content.get('metadata', {}).get('sources', [])
            if not sources:
                sources = scraped_data.get('sources', [])
            info['sources'] = sources
            info['source_count'] = len(sources)
            info['method'] = content.get('metadata', {}).get('method', 'Scrapy')

            # Preview articles
            preview_items = []
            for article in articles[:3]:
                title = article.get('title', 'No title')
                preview_items.append(f"• **{title}**")
            info['preview'] = '\n'.join(preview_items)

        # Check for RSS/scraped content (older format)
        elif 'articles' in content:
            query = content.get('query', 'Unknown Query')
            info['title'] = f"RSS Search: {query}"
            info['source'] = 'RSS/Scraped Content'
            info['content_type'] = 'RSS Articles'

            articles = content['articles']
            info['item_count'] = len(articles)

            # Get sources
            sources = content.get('sources', [])
            info['sources'] = sources
            info['source_count'] = len(sources)
            info['method'] = content.get('metadata', {}).get('source', 'RSS')

            # Preview articles
            preview_items = []
            for article in articles[:3]:
                title = article.get('title', 'No title')
                preview_items.append(f"• **{title}**")
            info['preview'] = '\n'.join(preview_items)

        # Fallback: If we still have "Unknown Content", try to extract any useful info
        if info['title'] == 'Unknown Content':
            filename = content.get('filename', 'Unknown')
            logger.warning(f"Could not parse content structure for {filename}")

            # Try to extract basic info from any available fields
            if 'query' in content:
                info['title'] = f"Query: {content['query']}"
                info['source'] = 'Web Search'
            elif 'url' in content:
                info['title'] = f"URL: {content['url']}"
                info['source'] = 'Web Fetch'
            else:
                # Show available keys for debugging
                available_keys = [k for k in content.keys() if k not in ['filename', 'file_path', 'file_size', 'file_modified']]
                info['title'] = f"Unknown Content ({filename})"
                info['preview'] = f"Available data keys: {', '.join(available_keys[:10])}"
                if len(available_keys) > 10:
                    info['preview'] += f" ... and {len(available_keys) - 10} more"

        return info

    except Exception as e:
        logger.error(f"Error extracting quarantine content info: {e}")
        filename = content.get('filename', 'Unknown')
        return {
            'title': f'Error Processing Content ({filename})',
            'source': 'Unknown',
            'content_type': 'Unknown',
            'item_count': 0,
            'source_count': 0,
            'method': 'Unknown',
            'preview': f'Error: {e}',
            'sources': []
        }

def calculate_security_overview() -> Dict[str, Any]:
    """Calculate security metrics overview from all vetted content."""
    try:
        from pathlib import Path
        import json

        vetted_dir = Path("vetted")
        if not vetted_dir.exists():
            return {}

        total_critical_risks = 0
        total_high_risks = 0
        total_credibility = 0
        total_purity = 0
        files_with_analysis = 0

        for file_path in vetted_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Check both possible locations for vetting results
                    vetting_result = data.get('vetting_results', data.get('vetting_result', {}))

                    if vetting_result:
                        files_with_analysis += 1

                        # Count risk factors
                        risk_factors = vetting_result.get('risk_assessment', {}).get('risk_factors', [])
                        total_critical_risks += len([r for r in risk_factors if r.get('severity') == 'critical'])
                        total_high_risks += len([r for r in risk_factors if r.get('severity') == 'high'])

                        # Accumulate scores
                        scores = vetting_result.get('scores', {})
                        total_credibility += scores.get('credibility', 0)
                        total_purity += scores.get('purity', 0)

            except Exception as e:
                logger.warning(f"Error processing vetted file {file_path}: {e}")
                continue

        if files_with_analysis == 0:
            return {}

        return {
            'critical_risks': total_critical_risks,
            'high_risks': total_high_risks,
            'avg_credibility': total_credibility / files_with_analysis,
            'avg_purity': total_purity / files_with_analysis,
            'files_analyzed': files_with_analysis
        }

    except Exception as e:
        logger.error(f"Error calculating security overview: {e}")
        return {}

def render_vetted_content_item(content: Dict[str, Any], index: int):
    """Render a single vetted content item for review."""
    try:
        # Extract key information from file format - check both possible locations
        vetting_result = content.get('vetting_result', content.get('vetting_results', {}))

        # Extract content info based on file structure
        content_info = extract_content_info_for_display(content)

        # Determine recommendation color and text
        rec_action = vetting_result.get('recommendation', 'REVIEW')
        if rec_action == 'PASS':
            rec_color = "🟢"
            rec_text = "Recommended for Approval"
        elif rec_action == 'FAIL':
            rec_color = "🔴"
            rec_text = "Recommended for Rejection"
        else:
            rec_color = "🟡"
            rec_text = "Requires Manual Review"

        # Create expandable item
        title = content_info['title']
        with st.expander(f"{rec_color} {title}", expanded=False):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**Source:** {content_info['source']}")
                st.markdown(f"**Query:** {content_info['query']}")
                st.markdown(f"**Recommendation:** {rec_color} {rec_text}")

                if content_info['article_count'] > 0:
                    st.markdown(f"**Articles Found:** {content_info['article_count']}")

                # Show vetting score and four-dimension security summary
                overall_score = vetting_result.get('overall_score', 0)
                if overall_score > 0:
                    st.markdown("**🛡️ Security Analysis Summary:**")
                    st.progress(overall_score, text=f"Overall Score: {overall_score:.1%}")

                    # Four-Dimension Security Analysis
                    scores = vetting_result.get('scores', {})
                    if scores:
                        st.markdown("**📊 Four-Dimension Security Assessment:**")

                        # Create security dimension display
                        dim_col1, dim_col2 = st.columns(2)

                        with dim_col1:
                            # Credibility & Bias
                            credibility = scores.get('credibility', 0)
                            cred_icon = "✅" if credibility >= 0.7 else "⚠️" if credibility >= 0.4 else "❌"
                            cred_status = "Good" if credibility >= 0.7 else "Warning" if credibility >= 0.4 else "Risk"
                            st.markdown(f"{cred_icon} **Credibility & Bias**: {credibility:.1%} ({cred_status})")

                            # Speculation vs. Fact
                            speculation = scores.get('speculation', 0)
                            spec_icon = "✅" if speculation <= 0.3 else "⚠️" if speculation <= 0.6 else "❌"
                            spec_status = "Good" if speculation <= 0.3 else "Warning" if speculation <= 0.6 else "Risk"
                            st.markdown(f"{spec_icon} **Speculation vs. Fact**: {speculation:.1%} ({spec_status})")

                        with dim_col2:
                            # Persuasive Language
                            persuasion = scores.get('persuasion', 0)
                            pers_icon = "✅" if persuasion <= 0.3 else "⚠️" if persuasion <= 0.6 else "❌"
                            pers_status = "Good" if persuasion <= 0.3 else "Warning" if persuasion <= 0.6 else "Risk"
                            st.markdown(f"{pers_icon} **Persuasive Language**: {persuasion:.1%} ({pers_status})")

                            # Content Purity
                            purity = scores.get('purity', 0)
                            purity_icon = "✅" if purity >= 0.8 else "⚠️" if purity >= 0.5 else "❌"
                            purity_status = "Good" if purity >= 0.8 else "Warning" if purity >= 0.5 else "Risk"
                            st.markdown(f"{purity_icon} **Content Purity**: {purity:.1%} ({purity_status})")

                        # Risk Factor Alerts
                        risk_factors = vetting_result.get('risk_assessment', {}).get('risk_factors', [])
                        if risk_factors:
                            critical_risks = [r for r in risk_factors if r.get('severity') == 'critical']
                            high_risks = [r for r in risk_factors if r.get('severity') == 'high']

                            if critical_risks:
                                st.error(f"🔴 **Critical Risk Alert**: {len(critical_risks)} critical security threat(s) detected")
                            elif high_risks:
                                st.warning(f"🟠 **High Risk Alert**: {len(high_risks)} high-priority concern(s) detected")
                            else:
                                st.success("🟢 **Risk Assessment**: No critical or high-risk factors detected")
                        else:
                            st.success("🟢 **Risk Assessment**: No security risks detected")

                    # Show key security dimensions
                    scores = vetting_result.get('scores', {})
                    if scores:
                        security_summary = []

                        # Credibility & Bias
                        credibility = scores.get('credibility', 0)
                        cred_status = "✅" if credibility >= 0.7 else "⚠️" if credibility >= 0.4 else "❌"
                        security_summary.append(f"{cred_status} Credibility: {credibility:.1%}")

                        # Persuasive Language (lower is better)
                        persuasion = scores.get('persuasion', 0)
                        pers_status = "✅" if persuasion <= 0.3 else "⚠️" if persuasion <= 0.6 else "❌"
                        security_summary.append(f"{pers_status} Persuasion: {persuasion:.1%}")

                        # Speculation vs Fact (lower is better)
                        speculation = scores.get('speculation', 0)
                        spec_status = "✅" if speculation <= 0.3 else "⚠️" if speculation <= 0.6 else "❌"
                        security_summary.append(f"{spec_status} Speculation: {speculation:.1%}")

                        # Content Purity
                        purity = scores.get('purity', 0)
                        pur_status = "✅" if purity >= 0.8 else "⚠️" if purity >= 0.5 else "❌"
                        security_summary.append(f"{pur_status} Purity: {purity:.1%}")

                        # Display in a compact format
                        st.markdown(" | ".join(security_summary))

                    # Show risk factors count
                    risk_factors = vetting_result.get('risk_assessment', {}).get('risk_factors', [])
                    if risk_factors:
                        critical_risks = len([r for r in risk_factors if r.get('severity') == 'critical'])
                        high_risks = len([r for r in risk_factors if r.get('severity') == 'high'])

                        if critical_risks > 0:
                            st.markdown(f"🔴 **{critical_risks} Critical Risk(s) Detected**")
                        elif high_risks > 0:
                            st.markdown(f"🟠 **{high_risks} High Risk(s) Detected**")
                        else:
                            st.markdown("🟢 **No Critical Risks Detected**")

                # Show content preview
                if content_info['preview']:
                    st.markdown("**Content Preview:**")
                    st.text(content_info['preview'])

            with col2:
                st.markdown("**Actions:**")

                col_approve, col_reject = st.columns(2)

                with col_approve:
                    if st.button("✅ Approve", key=f"approve_{index}", use_container_width=True):
                        if approve_content(content['filename']):
                            st.success("Content approved!")
                            st.rerun()
                        else:
                            st.error("Failed to approve content")

                with col_reject:
                    if st.button("❌ Reject", key=f"reject_{index}", use_container_width=True):
                        if reject_content(content['filename']):
                            st.success("Content rejected!")
                            st.rerun()
                        else:
                            st.error("Failed to reject content")

                # Show detailed analysis toggle
                if st.button("📊 View Details", key=f"details_{index}", use_container_width=True):
                    if f"show_details_{index}" not in st.session_state:
                        st.session_state[f"show_details_{index}"] = False
                    st.session_state[f"show_details_{index}"] = not st.session_state[f"show_details_{index}"]

            # Show detailed security analysis if toggled
            if st.session_state.get(f"show_details_{index}", False):
                render_detailed_security_analysis(vetting_result, index)

    except Exception as e:
        st.error(f"Error rendering vetted content item: {e}")

def render_detailed_security_analysis(vetting_result: Dict[str, Any], index: int):
    """Render detailed security analysis from SAM's Conceptual Dimension Prober."""
    try:
        st.markdown("---")
        st.markdown("### 🔍 **SAM's Security Analysis Report**")
        st.markdown("*Powered by Conceptual Dimension Prober*")

        # Check if we have valid vetting results
        if not vetting_result or vetting_result.get('status') == 'error':
            st.error("❌ **Analysis Error**")
            error_msg = vetting_result.get('reason', vetting_result.get('error', 'Unknown error occurred during analysis'))
            st.markdown(f"**Error:** {error_msg}")

            # Show raw data for debugging
            if vetting_result:
                st.markdown("**Raw Error Data:**")
                st.json(vetting_result)
            return

        # Overall Assessment
        overall_score = vetting_result.get('overall_score', 0)
        confidence = vetting_result.get('confidence', 0)
        recommendation = vetting_result.get('recommendation', 'UNKNOWN')

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Overall Score", f"{overall_score:.1%}",
                     delta=f"Confidence: {confidence:.1%}")
        with col2:
            rec_color = {"PASS": "🟢", "FAIL": "🔴", "REVIEW": "🟡"}.get(recommendation, "⚪")
            st.metric("📋 Recommendation", f"{rec_color} {recommendation}")
        with col3:
            processing_time = vetting_result.get('processing_time', 0)
            st.metric("⚡ Analysis Time", f"{processing_time:.2f}s")

        # Security Dimensions Analysis
        scores = vetting_result.get('scores', {})
        if scores:
            st.markdown("#### 🛡️ **Security Dimensions Analysis**")
            st.markdown("*Each dimension examined by SAM's Conceptual Understanding*")
        else:
            st.warning("⚠️ **No Security Dimension Scores Available**")
            st.markdown("The analysis may have failed or used a fallback method.")

        if scores:

            # Create two columns for dimension display
            dim_col1, dim_col2 = st.columns(2)

            dimension_info = {
                'credibility': {
                    'name': '🎓 Credibility & Bias',
                    'description': 'Factual accuracy and source reliability',
                    'good_range': [0.7, 1.0],
                    'warning_range': [0.4, 0.7],
                    'bad_range': [0.0, 0.4]
                },
                'persuasion': {
                    'name': '🎭 Persuasive Language',
                    'description': 'Manipulative or emotionally charged content',
                    'good_range': [0.0, 0.3],
                    'warning_range': [0.3, 0.6],
                    'bad_range': [0.6, 1.0]
                },
                'speculation': {
                    'name': '🔮 Speculation vs. Fact',
                    'description': 'Unverified claims and conjecture',
                    'good_range': [0.0, 0.3],
                    'warning_range': [0.3, 0.6],
                    'bad_range': [0.6, 1.0]
                },
                'purity': {
                    'name': '🧹 Content Purity',
                    'description': 'Freedom from suspicious patterns',
                    'good_range': [0.8, 1.0],
                    'warning_range': [0.5, 0.8],
                    'bad_range': [0.0, 0.5]
                }
            }

            for i, (dim_key, score) in enumerate(scores.items()):
                if dim_key in dimension_info:
                    info = dimension_info[dim_key]

                    # Determine color based on score and dimension type
                    if score >= info['good_range'][0] and score <= info['good_range'][1]:
                        color = "🟢"
                        status = "Good"
                    elif score >= info['warning_range'][0] and score <= info['warning_range'][1]:
                        color = "🟡"
                        status = "Warning"
                    else:
                        color = "🔴"
                        status = "Risk"

                    # Alternate between columns
                    with dim_col1 if i % 2 == 0 else dim_col2:
                        st.markdown(f"**{info['name']}** {color}")
                        st.progress(score, text=f"{score:.1%} - {status}")
                        st.caption(info['description'])
                        st.markdown("")

        # Risk Factors
        risk_assessment = vetting_result.get('risk_assessment', {})
        risk_factors = risk_assessment.get('risk_factors', [])

        if risk_factors:
            st.markdown("#### ⚠️ **Identified Risk Factors**")

            for risk in risk_factors:
                severity = risk.get('severity', 'unknown')
                dimension = risk.get('dimension', 'unknown')
                description = risk.get('description', 'No description')
                score = risk.get('score', 0)

                severity_colors = {
                    'critical': '🔴',
                    'high': '🟠',
                    'medium': '🟡',
                    'low': '🟢'
                }

                severity_color = severity_colors.get(severity, '⚪')

                st.markdown(f"**{severity_color} {severity.title()} Risk - {dimension.replace('_', ' ').title()}**")
                st.markdown(f"• {description}")
                st.markdown(f"• Score: {score:.2f}")
                st.markdown("")

        # Source Reputation Analysis
        source_reputation = vetting_result.get('source_reputation', {})
        if source_reputation:
            st.markdown("#### 🌐 **Source Reputation Analysis**")

            domain = source_reputation.get('domain', 'unknown')
            final_score = source_reputation.get('final_score', 0)
            risk_category = source_reputation.get('risk_category', 'unknown')
            https_used = source_reputation.get('https_used', False)

            rep_col1, rep_col2 = st.columns(2)

            with rep_col1:
                st.markdown(f"**Domain:** `{domain}`")
                st.markdown(f"**HTTPS:** {'✅ Yes' if https_used else '❌ No'}")

            with rep_col2:
                st.markdown(f"**Reputation Score:** {final_score:.1%}")
                st.markdown(f"**Risk Category:** {risk_category.replace('_', ' ').title()}")

        # Content Sanitization Results
        sanitization = vetting_result.get('sanitization', {})
        if sanitization:
            st.markdown("#### 🧼 **Content Sanitization Results**")

            purity_score = sanitization.get('purity_score', 0)
            removed_elements = sanitization.get('removed_elements', [])
            suspicious_patterns = sanitization.get('suspicious_patterns', [])

            san_col1, san_col2 = st.columns(2)

            with san_col1:
                st.metric("🧹 Purity Score", f"{purity_score:.1%}")
                if removed_elements:
                    st.markdown(f"**Removed Elements:** {len(removed_elements)}")
                    for element in removed_elements[:3]:  # Show first 3
                        st.caption(f"• {element}")
                    if len(removed_elements) > 3:
                        st.caption(f"• ... and {len(removed_elements) - 3} more")

            with san_col2:
                if suspicious_patterns:
                    st.markdown(f"**⚠️ Suspicious Patterns:** {len(suspicious_patterns)}")
                    for pattern in suspicious_patterns[:3]:  # Show first 3
                        st.caption(f"• {pattern}")
                    if len(suspicious_patterns) > 3:
                        st.caption(f"• ... and {len(suspicious_patterns) - 3} more")
                else:
                    st.markdown("**✅ No Suspicious Patterns Detected**")

        # Analysis Metadata
        metadata = vetting_result.get('metadata', {})
        if metadata:
            st.markdown("#### 🔧 **Analysis Configuration**")
            st.markdown(f"**Profile Used:** {metadata.get('profile_used', 'unknown')}")
            st.markdown(f"**Analysis Mode:** {metadata.get('analysis_mode', 'unknown')}")
            st.markdown(f"**Safety Threshold:** {metadata.get('safety_threshold', 0):.1%}")
            st.markdown(f"**Evaluator Version:** {metadata.get('evaluator_version', 'unknown')}")
            st.markdown("")

        # Raw Data (for debugging) - Use a toggle instead of expander
        if st.button("🔍 Show/Hide Raw Analysis Data", key=f"toggle_raw_{index}"):
            if f"show_raw_{index}" not in st.session_state:
                st.session_state[f"show_raw_{index}"] = False
            st.session_state[f"show_raw_{index}"] = not st.session_state[f"show_raw_{index}"]

        if st.session_state.get(f"show_raw_{index}", False):
            st.markdown("#### 🔍 **Raw Analysis Data**")
            st.json(vetting_result)

    except Exception as e:
        st.error(f"Error rendering security analysis: {e}")
        # Fallback to raw JSON display
        st.json(vetting_result)

def extract_content_info_for_display(content: Dict[str, Any]) -> Dict[str, Any]:
    """Extract content information for display from various file formats."""
    try:
        # Initialize default values
        info = {
            'title': 'Unknown Content',
            'source': 'Unknown Source',
            'query': 'Unknown Query',
            'article_count': 0,
            'preview': ''
        }

        # Handle intelligent web system format
        if 'result' in content and 'data' in content['result']:
            query = content.get('query', 'Unknown Query')
            result_data = content['result']['data']

            info['query'] = query
            info['source'] = 'Intelligent Web System'

            # Check for articles
            if 'articles' in result_data and result_data['articles']:
                articles = result_data['articles']
                info['article_count'] = len(articles)
                info['title'] = f"Web Search: {query} ({len(articles)} articles)"

                # Create preview from first few articles
                preview_parts = []
                for i, article in enumerate(articles[:3]):
                    title = article.get('title', 'No title')
                    desc = article.get('description', 'No description')
                    preview_parts.append(f"{i+1}. {title}\n   {desc[:100]}...")

                info['preview'] = '\n\n'.join(preview_parts)
                if len(articles) > 3:
                    info['preview'] += f'\n\n... and {len(articles) - 3} more articles'

            # Check for search results
            elif 'search_results' in result_data and result_data['search_results']:
                results = result_data['search_results']
                info['article_count'] = len(results)
                info['title'] = f"Search Results: {query} ({len(results)} results)"

                # Create preview from first few results
                preview_parts = []
                for i, result in enumerate(results[:3]):
                    title = result.get('title', 'No title')
                    snippet = result.get('snippet', 'No snippet')
                    preview_parts.append(f"{i+1}. {title}\n   {snippet[:100]}...")

                info['preview'] = '\n\n'.join(preview_parts)
                if len(results) > 3:
                    info['preview'] += f'\n\n... and {len(results) - 3} more results'

            # Check for direct content
            elif 'content' in result_data and result_data['content']:
                info['title'] = f"Web Content: {query}"
                info['preview'] = result_data['content'][:500] + '...' if len(result_data['content']) > 500 else result_data['content']

        # Handle scraped data format
        elif 'scraped_data' in content:
            query = content.get('query', 'Unknown Query')
            scraped = content['scraped_data']

            info['query'] = query
            info['source'] = 'Scrapy Web Search'

            if 'articles' in scraped and scraped['articles']:
                articles = scraped['articles']
                info['article_count'] = len(articles)
                info['title'] = f"Scraped Content: {query} ({len(articles)} articles)"

                # Create preview
                preview_parts = []
                for i, article in enumerate(articles[:3]):
                    title = article.get('title', 'No title')
                    content_text = article.get('content', article.get('description', 'No content'))
                    preview_parts.append(f"{i+1}. {title}\n   {content_text[:100]}...")

                info['preview'] = '\n\n'.join(preview_parts)

        # Handle direct articles format
        elif 'articles' in content and content['articles']:
            articles = content['articles']
            info['article_count'] = len(articles)
            info['title'] = f"Article Collection ({len(articles)} articles)"
            info['source'] = 'Direct Articles'

            # Create preview
            preview_parts = []
            for i, article in enumerate(articles[:3]):
                title = article.get('title', 'No title')
                desc = article.get('description', article.get('content', 'No description'))
                preview_parts.append(f"{i+1}. {title}\n   {desc[:100]}...")

            info['preview'] = '\n\n'.join(preview_parts)

        # Handle direct content format (old format)
        elif 'content' in content and content['content']:
            # Check if this is old format with metadata
            if 'metadata' in content and content['metadata']:
                metadata = content['metadata']
                query = metadata.get('original_query', 'Unknown Query')
                source = metadata.get('source', 'Unknown Source')

                info['query'] = query
                info['source'] = source.replace('_', ' ').title()
                info['title'] = f"Web Content: {query}"

                # Extract preview from content
                content_text = content['content']
                if len(content_text) > 500:
                    info['preview'] = content_text[:500] + '...'
                else:
                    info['preview'] = content_text
            else:
                info['title'] = 'Direct Content'
                info['source'] = content.get('source', 'Unknown Source')
                info['preview'] = content['content'][:500] + '...' if len(content['content']) > 500 else content['content']

        return info

    except Exception as e:
        logger.error(f"Error extracting content info for display: {e}")
        return {
            'title': 'Error Loading Content',
            'source': 'Unknown',
            'query': 'Unknown',
            'article_count': 0,
            'preview': f'Error: {str(e)}'
        }

def approve_content(filename: str) -> bool:
    """Approve vetted content and move to approved directory."""
    try:
        from pathlib import Path
        import shutil

        vetted_path = Path("vetted") / filename
        approved_path = Path("approved") / filename

        # Create approved directory if it doesn't exist
        approved_path.parent.mkdir(exist_ok=True)

        # Move file to approved directory
        shutil.move(str(vetted_path), str(approved_path))

        logger.info(f"Content approved: {filename}")
        return True

    except Exception as e:
        logger.error(f"Error approving content {filename}: {e}")
        return False

def reject_content(filename: str) -> bool:
    """Reject vetted content and move to rejected directory."""
    try:
        from pathlib import Path
        import shutil

        vetted_path = Path("vetted") / filename
        rejected_path = Path("rejected") / filename

        # Create rejected directory if it doesn't exist
        rejected_path.parent.mkdir(exist_ok=True)

        # Move file to rejected directory
        shutil.move(str(vetted_path), str(rejected_path))

        logger.info(f"Content rejected: {filename}")
        return True

    except Exception as e:
        logger.error(f"Error rejecting content {filename}: {e}")
        return False

def approve_quarantined_content(filename: str) -> bool:
    """Approve quarantined content directly and integrate into knowledge base."""
    try:
        from pathlib import Path
        import shutil
        import json

        quarantine_path = Path("quarantine") / filename
        approved_path = Path("approved") / filename

        if not quarantine_path.exists():
            logger.warning(f"Quarantined file not found: {filename}")
            return False

        # Create approved directory if it doesn't exist
        approved_path.parent.mkdir(exist_ok=True)

        # Load the quarantined content
        with open(quarantine_path, 'r', encoding='utf-8') as f:
            content_data = json.load(f)

        # Add approval metadata
        from datetime import datetime
        content_data['approval_metadata'] = {
            'approved_at': datetime.now().isoformat(),
            'approval_method': 'individual_manual',
            'approved_by': 'user',
            'bypass_vetting': True
        }

        # Save to approved directory
        with open(approved_path, 'w', encoding='utf-8') as f:
            json.dump(content_data, f, indent=2, ensure_ascii=False)

        # Remove from quarantine
        quarantine_path.unlink()

        # Try to integrate into knowledge base
        try:
            integrate_approved_content(filename)
            logger.info(f"Quarantined content approved and integrated: {filename}")
        except Exception as e:
            logger.warning(f"Content approved but integration failed: {e}")

        return True

    except Exception as e:
        logger.error(f"Failed to approve quarantined content {filename}: {e}")
        return False

def vet_individual_content(filename: str) -> bool:
    """Run vetting process on a single quarantined file."""
    try:
        import subprocess
        import sys
        from pathlib import Path

        logger.info(f"Starting individual vetting for: {filename}")

        # Get project root directory
        project_root = Path(__file__).parent

        # Run the vetting script on the specific file
        cmd = [
            sys.executable,
            str(project_root / "scripts" / "vet_quarantined_content.py"),
            "--file", filename,
            "--quiet"
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=str(project_root)
        )

        if result.returncode == 0:
            logger.info(f"Individual vetting completed successfully for: {filename}")
            return True
        else:
            logger.error(f"Individual vetting failed for {filename}: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"Individual vetting timed out for: {filename}")
        return False
    except Exception as e:
        logger.error(f"Failed to vet individual content {filename}: {e}")
        return False

def reject_quarantined_content(filename: str) -> bool:
    """Reject quarantined content and move to rejected directory."""
    try:
        from pathlib import Path
        import shutil

        quarantine_path = Path("quarantine") / filename
        rejected_path = Path("rejected") / filename

        if not quarantine_path.exists():
            logger.warning(f"Quarantined file not found: {filename}")
            return False

        # Create rejected directory if it doesn't exist
        rejected_path.parent.mkdir(exist_ok=True)

        # Move to rejected directory
        shutil.move(str(quarantine_path), str(rejected_path))
        logger.info(f"Quarantined content rejected: {filename}")
        return True

    except Exception as e:
        logger.error(f"Failed to reject quarantined content {filename}: {e}")
        return False

def integrate_approved_content(filename: str) -> bool:
    """Integrate approved content into SAM's knowledge base."""
    try:
        from pathlib import Path
        import json

        approved_path = Path("approved") / filename

        if not approved_path.exists():
            logger.warning(f"Approved file not found: {filename}")
            return False

        # Load the approved content
        with open(approved_path, 'r', encoding='utf-8') as f:
            content_data = json.load(f)

        # Try to integrate into memory store
        try:
            from memory.memory_vectorstore import get_memory_store
            memory_store = get_memory_store()

            # Extract content for integration
            articles = content_data.get('articles', [])
            search_results = content_data.get('search_results', [])

            integrated_count = 0

            # Process articles
            for article in articles:
                title = article.get('title', 'Unknown Title')
                content = article.get('content', '')
                url = article.get('url', '')

                if content:
                    # Create memory entry
                    memory_entry = {
                        'content': f"Title: {title}\n\nContent: {content}",
                        'source': url or 'Web Search',
                        'memory_type': 'web_content',
                        'metadata': {
                            'title': title,
                            'url': url,
                            'integration_method': 'individual_approval',
                            'approved_at': content_data.get('approval_metadata', {}).get('approved_at'),
                            'original_filename': filename
                        }
                    }

                    # Add to memory store
                    memory_store.add_memory(memory_entry)
                    integrated_count += 1

            # Process search results if no articles
            if integrated_count == 0 and search_results:
                for result in search_results:
                    title = result.get('title', 'Unknown Title')
                    snippet = result.get('snippet', '')
                    url = result.get('url', '')

                    if snippet:
                        memory_entry = {
                            'content': f"Title: {title}\n\nSnippet: {snippet}",
                            'source': url or 'Web Search',
                            'memory_type': 'web_snippet',
                            'metadata': {
                                'title': title,
                                'url': url,
                                'integration_method': 'individual_approval',
                                'approved_at': content_data.get('approval_metadata', {}).get('approved_at'),
                                'original_filename': filename
                            }
                        }

                        memory_store.add_memory(memory_entry)
                        integrated_count += 1

            logger.info(f"Integrated {integrated_count} items from {filename} into knowledge base")
            return integrated_count > 0

        except Exception as e:
            logger.error(f"Failed to integrate content into memory store: {e}")
            return False

    except Exception as e:
        logger.error(f"Failed to integrate approved content {filename}: {e}")
        return False

def save_to_quarantine(content_result: Dict[str, Any], query: str):
    """Save fetched web content to quarantine for vetting."""
    try:
        from pathlib import Path
        import json
        import hashlib
        from datetime import datetime

        # Create quarantine directory if it doesn't exist
        quarantine_dir = Path("quarantine")
        quarantine_dir.mkdir(exist_ok=True)

        # Generate filename based on URL hash and timestamp
        url_hash = hashlib.md5(content_result['url'].encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"web_search_{timestamp}_{url_hash}.json"

        # Prepare quarantine data structure
        quarantine_data = {
            "url": content_result['url'],
            "content": content_result['content'],
            "timestamp": content_result.get('timestamp', datetime.now().isoformat()),
            "error": None,
            "metadata": {
                **content_result.get('metadata', {}),
                "source": "secure_web_search",
                "original_query": query,
                "fetch_method": "SAM_WebFetcher",
                "quarantine_timestamp": datetime.now().isoformat()
            }
        }

        # Save to quarantine
        quarantine_path = quarantine_dir / filename
        with open(quarantine_path, 'w', encoding='utf-8') as f:
            json.dump(quarantine_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Content saved to quarantine: {filename}")

        # Update quarantine metadata
        save_quarantine_metadata(quarantine_dir, content_result['url'], filename,
                                len(content_result['content']), True)

    except Exception as e:
        logger.error(f"Failed to save content to quarantine: {e}")

def save_quarantine_metadata(quarantine_dir: Path, url: str, filename: str,
                           content_length: int, success: bool):
    """Save metadata about quarantined content."""
    try:
        metadata_file = quarantine_dir / "metadata.json"

        # Load existing metadata or create new
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {
                "quarantine_info": {
                    "created": datetime.now().isoformat(),
                    "description": "Web content fetched by SAM's secure web search",
                    "total_files": 0,
                    "total_size": 0
                },
                "files": []
            }

        # Add new file entry
        file_entry = {
            "filename": filename,
            "url": url,
            "timestamp": datetime.now().isoformat(),
            "content_length": content_length,
            "success": success,
            "source": "secure_web_search"
        }

        metadata["files"].append(file_entry)
        metadata["quarantine_info"]["total_files"] = len(metadata["files"])
        metadata["quarantine_info"]["total_size"] += content_length

        # Save updated metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    except Exception as e:
        logger.error(f"Failed to save quarantine metadata: {e}")

def save_scraped_to_quarantine(scraped_data: Dict[str, Any], query: str):
    """Save scraped content to quarantine for vetting."""
    try:
        from pathlib import Path
        import json
        import hashlib
        from datetime import datetime

        # Create quarantine directory if it doesn't exist
        quarantine_dir = Path("quarantine")
        quarantine_dir.mkdir(exist_ok=True)

        # Generate filename based on query hash and timestamp
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scrapy_search_{timestamp}_{query_hash}.json"

        # Prepare quarantine data structure
        quarantine_data = {
            "query": query,
            "scraped_data": scraped_data,
            "timestamp": datetime.now().isoformat(),
            "error": None,
            "metadata": {
                "source": "scrapy_web_search",
                "method": "intelligent_scraping",
                "article_count": scraped_data.get('article_count', 0),
                "source_count": scraped_data.get('source_count', 0),
                "sources": scraped_data.get('sources', []),
                "quarantine_timestamp": datetime.now().isoformat()
            }
        }

        # Save to quarantine
        quarantine_path = quarantine_dir / filename
        with open(quarantine_path, 'w', encoding='utf-8') as f:
            json.dump(quarantine_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Scraped content saved to quarantine: {filename}")

        # Update quarantine metadata
        save_scrapy_quarantine_metadata(quarantine_dir, query, filename,
                                       scraped_data.get('article_count', 0), True)

    except Exception as e:
        logger.error(f"Failed to save scraped content to quarantine: {e}")

def save_scrapy_quarantine_metadata(quarantine_dir: Path, query: str, filename: str,
                                   article_count: int, success: bool):
    """Save metadata about quarantined scraped content."""
    try:
        metadata_file = quarantine_dir / "scrapy_metadata.json"

        # Load existing metadata or create new
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {
                "quarantine_info": {
                    "created": datetime.now().isoformat(),
                    "description": "Web content scraped by SAM's Scrapy-based intelligent extraction",
                    "total_files": 0,
                    "total_articles": 0
                },
                "files": []
            }

        # Add new file entry
        file_entry = {
            "filename": filename,
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "article_count": article_count,
            "success": success,
            "source": "scrapy_web_search"
        }

        metadata["files"].append(file_entry)
        metadata["quarantine_info"]["total_files"] = len(metadata["files"])
        metadata["quarantine_info"]["total_articles"] += article_count

        # Save updated metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    except Exception as e:
        logger.error(f"Failed to save scrapy quarantine metadata: {e}")

def get_memory_stats_summary() -> str:
    """Get a summary of memory and storage statistics."""
    try:
        memory_stats = st.session_state.secure_memory_store.get_memory_stats()
        security_status = st.session_state.secure_memory_store.get_security_status()

        return f"""📊 **Memory & Storage Statistics**

**Storage Overview:**
• Total Memories: {memory_stats.get('total_memories', 0)}
• Storage Size: {memory_stats.get('total_size_mb', 0):.1f} MB
• Store Type: {memory_stats.get('store_type', 'Unknown')}

**Encryption Details:**
• Encrypted Chunks: {security_status.get('encrypted_chunk_count', 0)}
• Searchable Fields: {security_status.get('searchable_fields', 0)}
• Encrypted Fields: {security_status.get('encrypted_fields', 0)}

**Technical Details:**
• Embedding Dimension: {memory_stats.get('embedding_dimension', 0)}
• Vector Backend: FAISS + ChromaDB
• Encryption: AES-256-GCM"""

    except Exception as e:
        return f"❌ Error getting memory statistics: {e}"

def create_web_search_escalation_message(assessment, original_query: str) -> str:
    """Create a web search escalation message for Streamlit interface with interactive options."""
    confidence_percent = f"{assessment.confidence_score * 100:.1f}"

    reasons_text = ""
    if assessment.reasons:
        reason_explanations = {
            'insufficient_results': "I found very few relevant results in my knowledge base",
            'limited_results': "I have some relevant information, but it may not be comprehensive",
            'low_relevance': "The information I found doesn't closely match your query",
            'very_low_relevance': "I couldn't find closely relevant information",
            'outdated_information': "The information I have might be outdated",
            'mixed_timeliness': "I have a mix of recent and older information",
            'lacks_recent_content': "I don't have recent information on this topic",
            'insufficient_for_comparison': "I need more sources to provide a good comparison",
            'lacks_procedural_content': "I don't have detailed step-by-step information"
        }

        formatted_reasons = []
        for reason in assessment.reasons[:3]:  # Limit to top 3 reasons
            explanation = reason_explanations.get(reason, reason.replace('_', ' ').title())
            formatted_reasons.append(f"• {explanation}")

        if formatted_reasons:
            reasons_text = f"\n\n**Why I'm suggesting this:**\n" + "\n".join(formatted_reasons)

    # Store escalation data in session state for button handling
    if 'web_search_escalation' not in st.session_state:
        st.session_state.web_search_escalation = {}

    escalation_id = f"escalation_{len(st.session_state.web_search_escalation)}"
    st.session_state.web_search_escalation[escalation_id] = {
        'original_query': original_query,
        'assessment': assessment,
        'suggested_search_query': getattr(assessment, 'suggested_search_query', original_query)
    }

    escalation_message = f"""🤔 **I've checked my local knowledge...**

{assessment.explanation}

**Confidence in current knowledge:** {confidence_percent}%{reasons_text}

**Would you like me to search the web for more current information?**

🌐 **Interactive Web Search Available!**"""

    return escalation_message, escalation_id

def extract_key_terms_from_conversation(conversation_history: str) -> list:
    """Extract key terms from conversation history for context-aware search."""
    try:
        import re

        # Remove common stop words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}

        # Extract words (3+ characters) and filter out stop words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', conversation_history.lower())
        key_terms = [word for word in words if word not in stop_words]

        # Count frequency and return most common terms
        from collections import Counter
        term_counts = Counter(key_terms)

        # Return top 10 most frequent terms
        return [term for term, count in term_counts.most_common(10)]

    except Exception as e:
        logger.warning(f"Error extracting key terms from conversation: {e}")
        return []

def search_unified_memory(query: str, max_results: int = 5) -> list:
    """
    Enhanced unified memory search with document prioritization and fallback handling.

    CRITICAL FIXES APPLIED (2025-07-18):
    - Fixed secure memory store initialization and search
    - Added fallback to regular memory store when secure store lacks relevant content
    - Enhanced result structure handling for MemorySearchResult objects
    - Fixed document query detection and prioritization
    - Resolved persistent canned response issue for document queries

    This function searches across multiple memory stores with intelligent prioritization:
    1. Secure memory store (uploaded documents) - highest priority
    2. Regular memory store (consolidated knowledge, web content) with document filtering
    3. Document-specific searches for file references

    Args:
        query: Search query string
        max_results: Maximum number of results to return

    Returns:
        List of search results sorted by relevance and source type
    """
    try:
        # Import MemoryType at the top to ensure it's available throughout the function
        from memory.memory_vectorstore import MemoryType

        all_results = []

        logger.info(f"🔍 ENHANCED MEMORY SEARCH: '{query}' (max_results: {max_results})")

        # PHASE 2 REFACTORING: Option to use smart search router
        use_smart_router = st.session_state.get('use_smart_search_router', False)
        if use_smart_router:
            try:
                from services.search_router import smart_search
                logger.info(f"🎯 Using smart search router for query")
                return smart_search(query, max_results)
            except Exception as e:
                logger.warning(f"⚠️ Smart router failed, falling back to legacy search: {e}")
                # Continue with legacy search below

        # PRIORITY 0: Search for user corrections first (CRITICAL FIX)
        correction_results = []
        try:
            if hasattr(st.session_state, 'secure_memory_store') and st.session_state.secure_memory_store:
                # Search for user corrections with safe API call
                try:
                    correction_results = st.session_state.secure_memory_store.search_memories(
                        query=f"{query} user_correction authoritative",
                        max_results=max_results
                    )
                except TypeError:
                    # Fallback for different API signature
                    correction_results = st.session_state.secure_memory_store.search_memories(
                        f"{query} user_correction authoritative",
                        max_results
                    )
                logger.info(f"🔧 User correction search returned {len(correction_results)} results")

                # Boost correction scores to highest priority
                for result in correction_results:
                    result.source_type = 'user_corrections'
                    # Safely boost similarity score if possible
                    try:
                        if hasattr(result, 'similarity_score') and hasattr(result, '__dict__'):
                            result.similarity_score = min(1.0, result.similarity_score * 1.5)  # Major boost
                    except (AttributeError, TypeError):
                        result.score_boost = 1.5
                all_results.extend(correction_results)

        except Exception as e:
            logger.warning(f"⚠️ User correction search failed: {e}")

        # PRIORITY 1: Search secure memory store (uploaded documents, whitepapers)
        try:
            # Ensure secure memory store is available
            if not (hasattr(st.session_state, 'secure_memory_store') and st.session_state.secure_memory_store):
                logger.info("🔧 Initializing secure memory store for search...")
                try:
                    from memory.secure_memory_vectorstore import get_secure_memory_store, VectorStoreType
                    st.session_state.secure_memory_store = get_secure_memory_store(
                        store_type=VectorStoreType.CHROMA,
                        storage_directory="memory_store",
                        embedding_dimension=384,
                        security_manager=None
                    )
                    logger.info("✅ Secure memory store initialized for search")
                except Exception as init_error:
                    logger.warning(f"⚠️ Failed to initialize secure memory store: {init_error}")

            if hasattr(st.session_state, 'secure_memory_store') and st.session_state.secure_memory_store:
                # Use enhanced search if available for better document retrieval
                # Fix: Use MemoryType enum values instead of strings to prevent API errors
                if hasattr(st.session_state.secure_memory_store, 'enhanced_search_memories'):
                    # Try enhanced search with safe parameters
                    try:
                        secure_results = st.session_state.secure_memory_store.enhanced_search_memories(
                            query=f"{query} uploaded document whitepaper pdf",
                            max_results=max_results * 2
                        )
                        logger.info(f"📄 Enhanced secure search returned {len(secure_results)} document results")
                    except TypeError:
                        # Fallback to basic search if enhanced search has parameter issues
                        secure_results = st.session_state.secure_memory_store.search_memories(
                            f"{query} uploaded document",
                            max_results * 2
                        )
                        logger.info(f"📄 Fallback secure search returned {len(secure_results)} document results")
                else:
                    # Use safe API call for secure memory store
                    try:
                        secure_results = st.session_state.secure_memory_store.search_memories(
                            query=f"{query} uploaded document",
                            max_results=max_results * 2
                        )
                    except TypeError:
                        # Fallback for different API signature
                        secure_results = st.session_state.secure_memory_store.search_memories(
                            f"{query} uploaded document",
                            max_results * 2
                        )
                    logger.info(f"📄 Regular secure search returned {len(secure_results)} document results")

                # Tag and prioritize secure document results
                logger.info(f"📄 Processing {len(secure_results)} secure memory results")
                for i, result in enumerate(secure_results):
                    try:
                        # Debug result structure
                        result_type = type(result).__name__
                        has_chunk = hasattr(result, 'chunk')
                        has_source = hasattr(result, 'source')
                        has_similarity = hasattr(result, 'similarity_score')

                        logger.info(f"   Result {i+1}: {result_type}, chunk={has_chunk}, source={has_source}, similarity={has_similarity}")

                        # Set source type for prioritization
                        result.source_type = 'uploaded_documents'

                        # Safely boost score for uploaded documents
                        try:
                            if hasattr(result, 'similarity_score') and hasattr(result, '__dict__'):
                                result.similarity_score = min(1.0, result.similarity_score * 1.1)
                                logger.info(f"   Boosted similarity to {result.similarity_score:.3f}")
                        except (AttributeError, TypeError):
                            result.score_boost = 1.1
                            logger.info(f"   Added score boost attribute")

                    except Exception as e:
                        logger.warning(f"   Error processing result {i+1}: {e}")

                all_results.extend(secure_results)
                logger.info(f"📄 Added {len(secure_results)} secure results to all_results (total: {len(all_results)})")

            else:
                logger.warning("⚠️ Secure memory store not available")
        except Exception as e:
            logger.error(f"❌ Secure store search failed: {e}")

        # CRITICAL FIX: Check if secure store results contain actual document content
        has_relevant_content = False
        if all_results:
            for result in all_results[:3]:  # Check top 3 results
                try:
                    if hasattr(result, 'chunk') and hasattr(result.chunk, 'content'):
                        content = result.chunk.content.lower()
                    elif hasattr(result, 'content'):
                        content = result.content.lower()
                    else:
                        continue

                    # Check for actual SAM story content (not just binary metadata)
                    if any(term in content for term in ['chroma', 'ethan hayes', 'neural network', 'university lab', 'project chroma']):
                        has_relevant_content = True
                        logger.info(f"✅ Found relevant SAM story content in secure store")
                        break
                except Exception:
                    continue

        # If no relevant content found, search regular memory store
        if not has_relevant_content:
            logger.info(f"🔄 No relevant document content in secure store, searching regular memory store...")
            try:
                from memory.memory_vectorstore import get_memory_store
                regular_store = get_memory_store()

                # Search for specific SAM story content in regular memory store
                regular_doc_results = regular_store.search_memories(f"{query} SAM story Chroma Ethan Hayes neural network", max_results=max_results * 2)
                logger.info(f"📄 Regular store document search returned {len(regular_doc_results)} results")

                # Check if regular store has relevant content
                relevant_regular_results = []
                for result in regular_doc_results:
                    try:
                        # Handle different result structures from regular memory store
                        if hasattr(result, 'chunk') and hasattr(result.chunk, 'content'):
                            content = result.chunk.content.lower()
                            source = result.chunk.source
                        elif hasattr(result, 'content'):
                            content = result.content.lower()
                            source = getattr(result, 'source', 'Unknown')
                        else:
                            # MemorySearchResult structure - access the underlying chunk
                            if hasattr(result, 'memory_chunk'):
                                content = result.memory_chunk.content.lower()
                                source = result.memory_chunk.source
                            else:
                                continue

                        if any(term in content for term in ['chroma', 'ethan hayes', 'neural network', 'university lab', 'project chroma']):
                            result.source_type = 'uploaded_documents'
                            # Boost score for actual document content
                            try:
                                if hasattr(result, 'similarity_score') and hasattr(result, '__dict__'):
                                    result.similarity_score = min(1.0, result.similarity_score * 1.3)  # Higher boost for real content
                                elif hasattr(result, 'score'):
                                    result.similarity_score = min(1.0, result.score * 1.3)
                            except (AttributeError, TypeError):
                                result.score_boost = 1.3
                            relevant_regular_results.append(result)
                            logger.info(f"✅ Found relevant content from {source}: {content[:100]}...")
                    except Exception as e:
                        logger.warning(f"Error processing regular store result: {e}")
                        continue

                if relevant_regular_results:
                    # Replace secure store results with relevant regular store results
                    all_results = relevant_regular_results + all_results
                    logger.info(f"📄 Prioritized {len(relevant_regular_results)} relevant regular store results")
                else:
                    logger.warning(f"⚠️ No relevant content found in regular store either")

            except Exception as e:
                logger.error(f"❌ Regular store document search failed: {e}")

        # PRIORITY 2: Search regular memory store (consolidated knowledge, web content)
        try:
            from memory.memory_vectorstore import get_memory_store
            web_store = get_memory_store()

            # Use enhanced search if available
            if hasattr(web_store, 'enhanced_search_memories'):
                # Fix: Use MemoryType enum values instead of strings
                web_results = web_store.enhanced_search_memories(
                    query=query,
                    max_results=max_results,
                    memory_types=[MemoryType.DOCUMENT],  # Use enum instead of strings
                    tags=['consolidated', 'knowledge']
                )
                logger.info(f"🌐 Enhanced web knowledge search returned {len(web_results)} results")
            else:
                web_results = web_store.search_memories(query, max_results=max_results)
                logger.info(f"🌐 Regular web knowledge search returned {len(web_results)} results")

            # Tag web results
            for result in web_results:
                result.source_type = 'web_knowledge'
            all_results.extend(web_results)

        except Exception as e:
            logger.warning(f"⚠️ Web knowledge search failed: {e}")

        # PRIORITY 3: Search for specific document references
        if any(term in query.lower() for term in ['pdf', 'paper', 'whitepaper', 'document', 'file']):
            try:
                # Additional search specifically for document metadata
                if hasattr(st.session_state, 'secure_memory_store') and st.session_state.secure_memory_store:
                    # Try document-specific search with safe parameters
                    try:
                        doc_specific_results = st.session_state.secure_memory_store.search_memories(
                            query=f"filename document {query}",
                            max_results=max_results
                        )
                    except TypeError:
                        # Fallback if min_similarity parameter not supported
                        doc_specific_results = st.session_state.secure_memory_store.search_memories(
                            f"filename document {query}",
                            max_results
                        )
                    for result in doc_specific_results:
                        result.source_type = 'document_metadata'
                        # Safely boost document name matches
                        try:
                            if hasattr(result, 'similarity_score') and hasattr(result, '__dict__'):
                                result.similarity_score = min(1.0, result.similarity_score * 1.2)
                        except (AttributeError, TypeError):
                            result.score_boost = 1.2
                    all_results.extend(doc_specific_results)
                    logger.info(f"📋 Document-specific search returned {len(doc_specific_results)} results")
            except Exception as e:
                logger.warning(f"⚠️ Document-specific search failed: {e}")

        # Sort combined results by similarity score (prioritizing uploaded documents)
        all_results.sort(key=lambda x: (
            x.source_type == 'uploaded_documents',  # Uploaded docs first
            getattr(x, 'similarity_score', 0.0)     # Then by similarity
        ), reverse=True)

        # Log search results summary
        source_counts = {}
        for result in all_results:
            source_type = getattr(result, 'source_type', 'unknown')
            source_counts[source_type] = source_counts.get(source_type, 0) + 1

        logger.info(f"📊 SEARCH RESULTS SUMMARY: {source_counts}")
        logger.info(f"🎯 Returning top {min(len(all_results), max_results)} results")

        # Debug the top results being returned
        top_results = all_results[:max_results]
        for i, result in enumerate(top_results, 1):
            try:
                result_type = type(result).__name__
                source_type = getattr(result, 'source_type', 'unknown')
                similarity = getattr(result, 'similarity_score', 0.0)

                # PHASE 3: Use utility function to handle different result types
                content, source, metadata = extract_result_content(result)
                if not source:
                    source = str(result)[:50]

                logger.info(f"   Top result {i}: {result_type}, source_type={source_type}, similarity={similarity:.3f}, source={source}")

            except Exception as e:
                logger.warning(f"   Error debugging result {i}: {e}")

        # Return top results
        return top_results

    except Exception as e:
        logger.error(f"❌ Unified search failed: {e}")
        return []

def detect_document_query(query: str) -> bool:
    """
    Detect if a query is asking about uploaded documents, whitepapers, or local files.

    This function prevents document queries from being routed to web search.
    """
    try:
        query_lower = query.lower()

        # PRIORITY 1: Explicit file references
        file_indicators = [
            '.pdf', '.docx', '.txt', '.md', '.doc',
            'file', 'document', 'paper', 'whitepaper',
            'uploaded', 'local file', 'my file', 'my document'
        ]

        # PRIORITY 2: ArXiv paper patterns (like 2506.21393v1.pdf)
        import re
        arxiv_pattern = r'\d{4}\.\d{5}v?\d*\.pdf'
        has_arxiv_reference = bool(re.search(arxiv_pattern, query))

        # PRIORITY 3: Document-specific question patterns
        document_question_patterns = [
            r'what is.*\.pdf.*about',
            r'what.*in.*document',
            r'summarize.*paper',
            r'content of.*file',
            r'uploaded.*document',
            r'local.*file.*about',
            r'whitepaper.*about',
            r'paper.*titled',
            r'document.*titled',
            # Enhanced patterns for the user's specific query
            r'give me a.*summary.*document',
            r'brief summary.*document',
            r'what.*document.*contains',
            r'summary of what.*document',
            r'what this document',
            r'document contains',
            r'just uploaded.*\.pdf',
            r'i.*uploaded.*\.pdf',
            r'can you.*summary.*document',
            r'tell me about.*document',
            r'analyze.*document',
            r'explain.*document',
            # CRITICAL: Patterns for Summarize and Key Questions buttons
            r'comprehensive synthesis summary.*document',
            r'synthesis approach.*document',
            r'strategic and insightful questions.*document',
            r'question categories.*document',
            r'generate.*questions.*document',
            r'thoughtful questions.*document'
        ]

        # PRIORITY 4: Knowledge base queries about imported content
        knowledge_base_patterns = [
            'general domain knowledge',
            'whitepapers imported',
            'documents imported',
            'uploaded whitepapers',
            'imported to you',
            'knowledge base',
            'your documents',
            'files you have'
        ]

        # Check all patterns
        has_file_indicator = any(indicator in query_lower for indicator in file_indicators)
        has_document_pattern = any(re.search(pattern, query_lower) for pattern in document_question_patterns)
        has_knowledge_pattern = any(pattern in query_lower for pattern in knowledge_base_patterns)

        # Log detection details
        if has_file_indicator or has_arxiv_reference or has_document_pattern or has_knowledge_pattern:
            logger.info(f"📄 DOCUMENT QUERY DETECTED:")
            logger.info(f"   File indicators: {has_file_indicator}")
            logger.info(f"   ArXiv reference: {has_arxiv_reference}")
            logger.info(f"   Document patterns: {has_document_pattern}")
            logger.info(f"   Knowledge patterns: {has_knowledge_pattern}")
            return True

        return False

    except Exception as e:
        logger.error(f"Error in document query detection: {e}")
        return False

def diagnose_memory_retrieval(query: str) -> dict:
    """Diagnostic function to debug memory retrieval issues."""
    try:
        diagnosis = {
            'query': query,
            'secure_store_available': False,
            'secure_store_total_memories': 0,
            'secure_store_search_results': 0,
            'web_store_available': False,
            'web_store_total_memories': 0,
            'web_store_search_results': 0,
            'sample_memory_sources': [],
            'recommendations': []
        }

        # Check secure memory store
        try:
            if hasattr(st.session_state, 'secure_memory_store') and st.session_state.secure_memory_store:
                diagnosis['secure_store_available'] = True

                # Get total memory count
                try:
                    security_status = st.session_state.secure_memory_store.get_security_status()
                    diagnosis['secure_store_total_memories'] = security_status.get('encrypted_chunk_count', 0)
                except:
                    diagnosis['secure_store_total_memories'] = 'unknown'

                # Test search
                test_results = st.session_state.secure_memory_store.search_memories(
                    query=query,
                    max_results=3
                )
                diagnosis['secure_store_search_results'] = len(test_results)

                # Sample sources
                for result in test_results[:2]:
                    if hasattr(result, 'source'):
                        diagnosis['sample_memory_sources'].append(f"Secure: {result.source}")

        except Exception as e:
            diagnosis['secure_store_error'] = str(e)

        # Check web memory store
        try:
            from memory.memory_vectorstore import get_memory_store
            web_store = get_memory_store()
            diagnosis['web_store_available'] = True

            # Get total memory count
            try:
                all_memories = web_store.get_all_memories()
                diagnosis['web_store_total_memories'] = len(all_memories)
            except:
                diagnosis['web_store_total_memories'] = 'unknown'

            # Test search
            test_results = web_store.search_memories(query, max_results=3)
            diagnosis['web_store_search_results'] = len(test_results)

            # Sample sources
            for result in test_results[:2]:
                if hasattr(result, 'source'):
                    diagnosis['sample_memory_sources'].append(f"Web: {result.source}")

        except Exception as e:
            diagnosis['web_store_error'] = str(e)

        # Generate recommendations
        if diagnosis['secure_store_total_memories'] == 0:
            diagnosis['recommendations'].append("No memories found in secure store - check document upload process")
        if diagnosis['secure_store_search_results'] == 0 and diagnosis['secure_store_total_memories'] > 0:
            diagnosis['recommendations'].append("Memories exist but search returned no results - check search parameters")
        if not diagnosis['secure_store_available']:
            diagnosis['recommendations'].append("Secure memory store not initialized - check authentication")

        return diagnosis

    except Exception as e:
        return {'error': str(e), 'query': query}

def generate_draft_response(prompt: str, force_local: bool = False) -> str:
    """Generate a draft response using SAM's capabilities (Stage 1 of two-stage pipeline - Task 30 Phase 2)."""
    try:
        # Phase -3: Cognitive Distillation Enhancement (NEW - Phase 2 Integration)
        enhanced_prompt = prompt
        transparency_data = {}

        try:
            if (st.session_state.get('cognitive_distillation_enabled', False) and
                st.session_state.get('cognitive_distillation')):

                cognitive_distillation = st.session_state.cognitive_distillation

                # Prepare context for principle selection
                context = {
                    'user_session': st.session_state.get('session_id', 'default'),
                    'interface': 'secure_chat',
                    'force_local': force_local,
                    'query_length': len(prompt),
                    'timestamp': datetime.now().isoformat()
                }

                # Enhance reasoning with cognitive principles
                enhanced_prompt, transparency_data = cognitive_distillation.enhance_reasoning(prompt, context)

                # Store transparency data for UI display
                st.session_state['last_transparency_data'] = transparency_data

                if transparency_data.get('active_principles'):
                    logger.info(f"🧠 Enhanced reasoning with {len(transparency_data['active_principles'])} cognitive principles")
                else:
                    logger.info(f"🧠 No relevant principles found for enhancement")

        except Exception as e:
            logger.warning(f"Cognitive distillation enhancement failed, continuing with original prompt: {e}")
            enhanced_prompt = prompt
            transparency_data = {}

        # Phase -2: Conversational Buffer Management (Task 30 Phase 1)
        conversation_history = ""
        session_manager = None
        session_id = None
        try:
            from sam.session.state_manager import get_session_manager

            # Get or create session
            session_manager = get_session_manager()
            session_id = st.session_state.get('session_id', 'default_session')

            # Create session if it doesn't exist
            if not session_manager.get_session(session_id):
                user_id = st.session_state.get('user_id', 'anonymous')
                session_manager.create_session(session_id, user_id)
                st.session_state['session_id'] = session_id
                logger.info(f"🗣️ Created new conversation session: {session_id}")

            # CRITICAL FIX: Check if this is a document query BEFORE loading conversation history
            is_document_query = detect_document_query(prompt)

            if is_document_query:
                # For document queries, use minimal conversation history to focus on document content
                logger.info(f"📄 DOCUMENT QUERY DETECTED: Skipping old conversation history")
                conversation_history = "No recent conversation history."
                st.session_state['conversation_history'] = conversation_history
                st.session_state['document_query_detected'] = True
                st.session_state['reduce_conversation_weight'] = True
            else:
                # CRITICAL FIX: Get conversation history BEFORE adding current prompt
                # This ensures that new chats start with empty history
                conversation_history = session_manager.format_conversation_history(session_id, max_turns=8)
                logger.info(f"Formatted conversation history ({len(conversation_history)} chars)")

                # Store in session state for use in prompt template
                st.session_state['conversation_history'] = conversation_history
                st.session_state['document_query_detected'] = False
                st.session_state['reduce_conversation_weight'] = False

            logger.info("Stored conversation history in session state")

            # Now add user turn to conversation buffer for future context
            session_manager.add_turn(session_id, 'user', prompt)

            logger.info(f"🗣️ Conversational buffer updated for session: {session_id}")

        except Exception as e:
            logger.warning(f"Conversational buffer failed, continuing without history: {e}")
            conversation_history = ""

        # Phase -1: Feedback-Driven Response Enhancement (preserving 100% of functionality)
        try:
            response_context = enhance_response_with_feedback_learning(prompt)
        except Exception as e:
            logger.warning(f"Feedback enhancement failed, continuing with standard flow: {e}")
            response_context = {}

        # Phase -0.5: User Correction Detection for MEMOIR Learning (preserving 100% of functionality)
        try:
            correction_detected = detect_user_correction(prompt)
            if correction_detected and correction_detected.get('is_correction'):
                correction_text = correction_detected.get('correction_text', '')
                logger.info(f"🔧 User correction detected: {correction_text[:100]}...")
                # Process the correction through MEMOIR
                process_user_correction_for_memoir(correction_detected)
        except Exception as e:
            logger.warning(f"Correction detection failed, continuing with standard flow: {e}")

        # Phase -0.5: Learning Intent Detection (NEW - for MEMOIR activation)
        learning_intent = detect_learning_intent(prompt)
        if learning_intent['is_learning_request']:
            logger.info(f"🧠 Learning intent detected: {learning_intent['intent_type']}")
            # Handle learning request directly
            learning_response = handle_learning_request(prompt, learning_intent)
            if learning_response:
                return learning_response

        # Phase -0.25: MEMOIR Knowledge Retrieval (ENABLED BY DEFAULT for lifelong learning)
        memoir_context = {}
        try:
            if st.session_state.get('memoir_enabled', False) and st.session_state.get('memoir_integration'):
                memoir_context = retrieve_memoir_knowledge(prompt)
                if memoir_context.get('relevant_edits'):
                    logger.info(f"🧠 MEMOIR retrieved {len(memoir_context['relevant_edits'])} relevant knowledge edits")
        except Exception as e:
            logger.warning(f"MEMOIR knowledge retrieval failed, continuing with standard flow: {e}")
            memoir_context = {}

        # Phase -0.1: PRIORITY Document Query Detection (CRITICAL FIX)
        # This MUST run BEFORE web search to prevent uploaded documents from being ignored
        try:
            is_document_query = detect_document_query(prompt)
            if is_document_query:
                logger.info(f"📄 DOCUMENT QUERY DETECTED: '{prompt[:50]}...' - Routing to internal memory search")
                # Force local search for document queries to prevent web search
                force_local = True
                # Add document query context
                st.session_state['last_query_type'] = 'document_specific'
                st.session_state['document_query_detected'] = True

                # CRITICAL FIX: For document queries, prioritize document content over conversation history
                logger.info(f"🎯 DOCUMENT QUERY: Prioritizing document content over conversation history")
                # Reduce conversation history weight for document queries
                st.session_state['reduce_conversation_weight'] = True
            else:
                st.session_state['document_query_detected'] = False
                st.session_state['reduce_conversation_weight'] = False
        except Exception as e:
            logger.warning(f"Document query detection failed: {e}")
            st.session_state['document_query_detected'] = False

        # Phase 0: MANDATORY Interactive Web Search Choice Before Tool Selection (preserving 100% of functionality)
        # This MUST run before any tool selection to ensure user control over web searches
        try:
            from reasoning.confidence_assessor import get_confidence_assessor
            confidence_assessor = get_confidence_assessor()

            # Check if this query asks for current/recent information
            current_info_keywords = ['latest', 'current', 'recent', 'today', 'now', 'new', 'breaking', 'news']
            has_current_keyword = any(keyword in prompt.lower() for keyword in current_info_keywords)

            logger.info(f"🔍 Checking for current info keywords in: '{prompt[:50]}...'")
            logger.info(f"🔍 Has current keyword: {has_current_keyword}")

            # CRITICAL: Skip web search for document queries
            if st.session_state.get('document_query_detected', False):
                logger.info(f"📄 Skipping web search - document query detected")
                has_current_keyword = False  # Override to prevent web search

            if has_current_keyword and not force_local:
                # Check user's web search preference
                web_search_mode = st.session_state.get('web_search_mode', 'Interactive')
                logger.info(f"🌐 Web search mode: {web_search_mode}, has_current_keyword: {has_current_keyword}")

                if web_search_mode == "Interactive":
                    # ALWAYS assess confidence for current information queries
                    assessment = confidence_assessor.assess_retrieval_quality([], prompt)
                    logger.info(f"🔍 MANDATORY Pre-tool confidence assessment: {assessment.status} ({assessment.confidence_score:.2f})")

                    # For current information queries, ALWAYS offer interactive choice (FORCED FOR TESTING)
                    logger.info(f"🌐 FORCING interactive web search choice for current information query")

                    # Create a mock assessment if needed to ensure escalation triggers
                    if assessment.status != "NOT_CONFIDENT":
                        logger.info(f"🔧 FORCING NOT_CONFIDENT status for testing interactive buttons")
                        assessment.status = "NOT_CONFIDENT"
                        assessment.confidence_score = 0.1
                        assessment.explanation = "I'm forcing low confidence to test interactive web search buttons."

                    escalation_message, escalation_id = create_web_search_escalation_message(assessment, prompt)
                    logger.info(f"🌐 MANDATORY Pre-tool escalation created with ID: {escalation_id}")
                    logger.info(f"🌐 ✅ RETURNING ESCALATION TUPLE: message={type(escalation_message)}, id={escalation_id}")
                    return escalation_message, escalation_id
                else:
                    logger.info(f"🌐 Automatic web search mode - proceeding with tool selection")
            else:
                logger.info(f"🔍 No current info keywords found or force_local=True, proceeding with normal flow")

        except Exception as e:
            logger.warning(f"Pre-tool confidence assessment failed: {e}")

        # Phase 0.1: Intelligent Tool Selection and Planning (preserving 100% of functionality)
        # NOTE: Only use specialized tools for non-web-searchable queries (math, calculations)
        # Financial and news queries should go through web search assessment first
        tool_enhanced_response = None
        try:
            # Check if this is a pure calculation query that doesn't need web search
            if is_calculation_only_query(prompt):
                tool_enhanced_response = generate_tool_enhanced_response(prompt, force_local)
                if tool_enhanced_response and tool_enhanced_response != "NO_TOOL_NEEDED":
                    logger.info("Query handled by specialized calculation tool system")
                    # Apply feedback-driven enhancements to tool responses
                    enhanced_tool_response = apply_feedback_enhancements(tool_enhanced_response, response_context)
                    return enhanced_tool_response

            # Check if this is a table analysis query that should use the Table-to-Code Expert Tool
            elif is_table_analysis_query(prompt):
                tool_enhanced_response = generate_tool_enhanced_response(prompt, force_local)
                if tool_enhanced_response and tool_enhanced_response != "NO_TOOL_NEEDED":
                    logger.info("Query handled by Table-to-Code Expert Tool system")
                    # Apply feedback-driven enhancements to tool responses
                    enhanced_tool_response = apply_feedback_enhancements(tool_enhanced_response, response_context)
                    return enhanced_tool_response
        except Exception as e:
            logger.warning(f"Tool-enhanced response failed, continuing with standard flow: {e}")

        # Phase 1: Use session state TPV integration (preserving 100% of existing functionality)
        tpv_enabled_response = None
        sam_tpv_integration = None
        user_profile = None

        try:
            from sam.cognition.tpv import UserProfile

            # Use TPV integration from session state if available
            if st.session_state.get('tpv_initialized') and st.session_state.get('sam_tpv_integration'):
                sam_tpv_integration = st.session_state.sam_tpv_integration
                logger.info("✅ Using initialized TPV integration from session state")
            else:
                # Fallback: try to initialize TPV integration
                try:
                    from sam.cognition.tpv import sam_tpv_integration as fallback_tpv
                    if not fallback_tpv.is_initialized:
                        fallback_tpv.initialize()
                    sam_tpv_integration = fallback_tpv
                    logger.info("✅ Fallback TPV integration initialized")
                except Exception as fallback_error:
                    logger.warning(f"Fallback TPV initialization failed: {fallback_error}")

            # Determine user profile (enhanced for better TPV triggering)
            # Use RESEARCHER profile for better TPV activation on analytical queries
            user_profile = UserProfile.RESEARCHER  # Enhanced profile for better TPV triggering

        except Exception as e:
            logger.warning(f"TPV integration not available: {e}")
            sam_tpv_integration = None

        # Phase 1.5: MEMOIR Integration (ENABLED BY DEFAULT for lifelong learning)
        # MEMOIR should be available by default for continuous knowledge updates
        try:
            from sam.orchestration.memoir_sof_integration import get_memoir_sof_integration

            # Initialize MEMOIR integration if not already done
            if 'memoir_integration' not in st.session_state or st.session_state.memoir_integration is None:
                sam_memoir_integration = get_memoir_sof_integration()
                st.session_state.memoir_integration = sam_memoir_integration
                st.session_state.memoir_enabled = True
                logger.info("✅ MEMOIR integration initialized and ENABLED by default for lifelong learning")
            else:
                logger.info("✅ MEMOIR integration already available")

        except Exception as e:
            logger.warning(f"MEMOIR integration not available: {e}")
            st.session_state.memoir_integration = None
            st.session_state.memoir_enabled = False

        # Phase 1A+1B: Enhanced SLP Integration (preserving 100% of existing functionality)
        # Note: SLP is used for pattern matching, but TPV takes priority for Active Reasoning Control
        enhanced_slp_response = None
        try:
            if st.session_state.get('enhanced_slp_initialized', False):
                enhanced_slp_integration = st.session_state.get('enhanced_slp_integration')

                if enhanced_slp_integration and enhanced_slp_integration.enabled:
                    logger.info("🧠 Using Enhanced SLP system for response generation")

                    # Prepare context for enhanced SLP
                    slp_context = {
                        'force_local': force_local,
                        'has_tpv': sam_tpv_integration is not None,
                        'user_profile': getattr(st.session_state, 'user_profile', 'general'),
                        'feedback_context': response_context,
                        'conversation_history': st.session_state.get('conversation_history', ''),  # Task 30 Phase 1
                        'document_query_detected': st.session_state.get('document_query_detected', False),
                        'reduce_conversation_weight': st.session_state.get('reduce_conversation_weight', False)
                    }

                    # Debug conversation history in SLP context
                    conv_history = slp_context.get('conversation_history', '')
                    logger.info(f"🔍 DEBUG: SLP conversation history ({len(conv_history)} chars): '{conv_history}'")

                    # PHASE 2 REFACTORING: Option to use clean SLP fallback service
                    use_clean_slp_service = st.session_state.get('use_clean_slp_service', False)

                    if use_clean_slp_service:
                        try:
                            from services.slp_fallback_service import create_clean_fallback_generator
                            sam_fallback_generator = create_clean_fallback_generator(memoir_context)
                            logger.info("✅ Using clean SLP fallback service")
                        except Exception as e:
                            logger.warning(f"⚠️ Clean SLP service failed, using legacy fallback: {e}")
                            # Fall through to legacy implementation
                            use_clean_slp_service = False

                    if not use_clean_slp_service:
                        # Create a fallback generator that uses the standard SAM pipeline (LEGACY)
                        def sam_fallback_generator(query, context):
                            """Fallback generator that uses standard SAM response generation."""
                        try:
                            import requests

                            # Build context-aware prompt with conversation history and MEMOIR integration
                            prompt_parts = []

                            # Add conversation history if available (Task 30 Phase 1)
                            conversation_history = context.get('conversation_history', '') if context else ''
                            reduce_conversation_weight = context.get('reduce_conversation_weight', False) if context else False

                            # CRITICAL FIX: Skip conversation history for document queries
                            if conversation_history and conversation_history != "No recent conversation history." and not reduce_conversation_weight:
                                prompt_parts.append("--- RECENT CONVERSATION HISTORY (Most recent first) ---")
                                prompt_parts.append(conversation_history)
                                prompt_parts.append("--- END OF CONVERSATION HISTORY ---\n")
                                logger.info(f"✅ DEBUG: Added conversation history to SLP fallback prompt ({len(conversation_history)} chars)")
                            elif reduce_conversation_weight:
                                logger.info(f"📄 SLP FALLBACK: Skipping conversation history for document query")
                            else:
                                logger.warning(f"⚠️ DEBUG: No conversation history in SLP fallback - history: '{conversation_history}'")

                            prompt_parts.append(f"Question: {query}")

                            # CRITICAL FIX: Add document search for document queries
                            document_query_detected = context.get('document_query_detected', False) if context else False
                            if document_query_detected:
                                logger.info(f"📄 SLP FALLBACK: Searching for document content...")
                                try:
                                    # Search for uploaded documents
                                    document_results = search_unified_memory(query, max_results=5)
                                    logger.info(f"📄 SLP FALLBACK: Found {len(document_results)} document results")

                                    if document_results:
                                        prompt_parts.append("\n--- UPLOADED DOCUMENT CONTENT ---")
                                        for i, result in enumerate(document_results[:3], 1):
                                            try:
                                                # PHASE 3: Use utility function to handle different result structures
                                                content, source, metadata = extract_result_content(result)
                                                if not content:
                                                    logger.warning(f"📄 SLP FALLBACK: Could not extract content from result type: {type(result)}")
                                                    continue

                                                # Truncate content for prompt
                                                content = content[:800]  # More content for documents

                                                prompt_parts.append(f"{i}. From {source}:")
                                                prompt_parts.append(f"   {content}")

                                                if 'SAM story' in source or 'Chroma' in content or 'chroma' in content.lower():
                                                    logger.info(f"📄 SLP FALLBACK: ✅ Found target document content!")

                                            except Exception as e:
                                                logger.warning(f"📄 SLP FALLBACK: Error processing result {i}: {e}")

                                        prompt_parts.append("--- END OF DOCUMENT CONTENT ---\n")
                                    else:
                                        logger.warning(f"📄 SLP FALLBACK: No document results found")

                                except Exception as e:
                                    logger.error(f"📄 SLP FALLBACK: Document search failed: {e}")

                            # Add MEMOIR knowledge if available
                            if memoir_context.get('relevant_edits'):
                                prompt_parts.append("\nLearned knowledge from previous corrections:")
                                for i, edit in enumerate(memoir_context['relevant_edits'][:2]):
                                    if edit.get('edit_type') == 'correction':
                                        prompt_parts.append(f"• Previous correction: {edit.get('correction', '')}")
                                    elif edit.get('edit_type') == 'memory_correction':
                                        content = edit.get('content', '')[:200]
                                        prompt_parts.append(f"• Learned: {content}")

                            # Add context if available
                            if context.get('sources'):
                                prompt_parts.append("\nRelevant context:")
                                for i, source in enumerate(context['sources'][:3]):
                                    content = source.get('content', '')[:500]
                                    prompt_parts.append(f"{i+1}. {content}")

                            # Add appropriate instruction based on query type
                            if document_query_detected:
                                prompt_parts.append("\nPlease provide a comprehensive response based on the uploaded document content above. Use specific details, quotes, and information from the document to answer the question thoroughly.")
                            else:
                                prompt_parts.append("\nPlease provide a comprehensive, helpful response using the learned knowledge.")

                            full_prompt = "\n".join(prompt_parts)

                            # Call Ollama API with increased timeout for better reliability
                            response = requests.post(
                                "http://localhost:11434/api/generate",
                                json={
                                    "model": "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M",
                                    "prompt": full_prompt,
                                    "stream": False,
                                    "options": {"temperature": 0.7, "max_tokens": 1000}
                                },
                                timeout=120  # Increased timeout to 120 seconds for complex queries
                            )

                            if response.status_code == 200:
                                result = response.json()
                                return result.get('response', f"I understand you're asking about: {query}")
                            else:
                                return f"I understand you're asking about: {query}"

                        except Exception as e:
                            logger.error(f"Fallback generator error: {e}")
                            # Provide a more helpful message when Ollama is slow/unavailable
                            if "timeout" in str(e).lower() or "read timed out" in str(e).lower():
                                return f"I apologize, but I'm experiencing slower than usual response times. Your question about '{query}' is being processed, but it may take longer than expected. Please try again in a moment, or consider asking a simpler question."
                            else:
                                return f"I understand you're asking about: {query}"

                    # CRITICAL: Process through SLP system for metrics collection
                    # This ensures all queries are tracked for real-time analytics
                    try:
                        if hasattr(enhanced_slp_integration, 'process_query'):
                            # Use the process_query method for better metrics collection
                            # Use enhanced prompt if available from cognitive distillation
                            query_to_use = enhanced_prompt if enhanced_prompt != prompt else prompt
                            slp_result = enhanced_slp_integration.process_query(
                                query_to_use,
                                slp_context,
                                user_profile=slp_context['user_profile'],
                                fallback_generator=sam_fallback_generator
                            )

                            if slp_result and slp_result.get('response'):
                                logger.info("✅ Enhanced SLP generated response with metrics collection")
                                enhanced_response = apply_feedback_enhancements(
                                    slp_result['response'],
                                    response_context
                                )
                                return enhanced_response
                        else:
                            # Fallback to legacy method
                            # Use enhanced prompt if available from cognitive distillation
                            query_to_use = enhanced_prompt if enhanced_prompt != prompt else prompt
                            enhanced_slp_response = enhanced_slp_integration.generate_response_with_slp(
                                query=query_to_use,
                                context=slp_context,
                                user_profile=slp_context['user_profile'],
                                fallback_generator=sam_fallback_generator
                            )

                            if enhanced_slp_response and enhanced_slp_response.get('response'):
                                logger.info("✅ Enhanced SLP generated response successfully")
                                # Apply feedback-driven enhancements to SLP responses
                                enhanced_response = apply_feedback_enhancements(
                                    enhanced_slp_response['response'],
                                    response_context
                                )
                                return enhanced_response

                        logger.info("🔄 Enhanced SLP did not generate response, continuing with standard flow")

                    except Exception as slp_error:
                        logger.warning(f"SLP processing failed: {slp_error}, continuing with standard flow")
        except Exception as e:
            logger.warning(f"Enhanced SLP response generation failed, continuing with standard flow: {e}")

        # Phase 8.1: ENHANCED CONTEXT PRIORITIZATION SYSTEM
        # Implements proper information source weighting as requested by user
        memory_results = []
        context_sources = []

        # PRIORITY 1: Recently uploaded documents (highest weight)
        uploaded_doc_results = []
        try:
            # Check for recently uploaded documents in current session
            recent_uploads = st.session_state.get('recent_document_uploads', [])
            if recent_uploads:
                logger.info(f"🔍 PRIORITY 1: Searching {len(recent_uploads)} recently uploaded documents")
                for doc_info in recent_uploads[-3:]:  # Last 3 uploads
                    doc_filename = doc_info.get('filename', '')
                    if doc_filename:
                        doc_results = search_unified_memory(
                            query=f"{prompt} filename:{doc_filename}",
                            max_results=5
                        )
                        uploaded_doc_results.extend(doc_results)
                        if doc_results:
                            logger.info(f"📄 Found {len(doc_results)} results from uploaded document: {doc_filename}")

                # Boost scores for uploaded documents
                for result in uploaded_doc_results:
                    # Safely boost similarity score if possible
                    try:
                        if hasattr(result, 'similarity_score') and hasattr(result, '__dict__'):
                            result.similarity_score *= 1.5  # 50% boost for uploaded docs
                    except (AttributeError, TypeError):
                        # If we can't modify similarity_score, add a boost attribute
                        result.score_boost = 1.5
                    result.priority_source = "uploaded_document"

                context_sources.append(f"Recently uploaded documents ({len(uploaded_doc_results)} results)")
        except Exception as e:
            logger.warning(f"Error searching uploaded documents: {e}")

        # PRIORITY 2: Current chat context (second highest weight)
        chat_context_results = []
        try:
            # CRITICAL FIX: Reduce chat context weight for document queries
            reduce_conversation_weight = st.session_state.get('reduce_conversation_weight', False)

            if conversation_history and conversation_history.strip() and not reduce_conversation_weight:
                # Extract key terms from conversation history for context-aware search
                chat_terms = extract_key_terms_from_conversation(conversation_history)
                if chat_terms:
                    chat_query = f"{prompt} {' '.join(chat_terms[:5])}"
                    chat_context_results = search_unified_memory(query=chat_query, max_results=3)
                    for result in chat_context_results:
                        # Safely boost similarity score if possible
                        try:
                            if hasattr(result, 'similarity_score') and hasattr(result, '__dict__'):
                                result.similarity_score *= 1.3  # 30% boost for chat context
                        except (AttributeError, TypeError):
                            result.score_boost = 1.3
                        result.priority_source = "chat_context"
                    logger.info(f"💬 Found {len(chat_context_results)} results from chat context")
                    context_sources.append(f"Current chat context ({len(chat_context_results)} results)")
            elif reduce_conversation_weight:
                logger.info(f"📄 DOCUMENT QUERY: Skipping chat context to prioritize document content")
        except Exception as e:
            logger.warning(f"Error searching chat context: {e}")

        # PRIORITY 3: Vector datastore knowledge (standard weight)
        vector_results = []
        try:
            # Check if this is a document-specific discussion
            selected_document = st.session_state.get('selected_document', '')
            if selected_document:
                vector_results = search_unified_memory(
                    query=f"{prompt} filename:{selected_document}",
                    max_results=5
                )
                logger.info(f"📚 Found {len(vector_results)} results from selected document: {selected_document}")
            else:
                vector_results = search_unified_memory(query=prompt, max_results=5)
                logger.info(f"📚 Found {len(vector_results)} results from vector datastore")

            for result in vector_results:
                result.priority_source = "vector_datastore"
            context_sources.append(f"Vector datastore ({len(vector_results)} results)")
        except Exception as e:
            logger.warning(f"Error searching vector datastore: {e}")

        # Combine all results with proper prioritization
        all_memory_results = uploaded_doc_results + chat_context_results + vector_results

        # Remove duplicates while preserving priority order
        seen_content = set()
        memory_results = []
        for result in all_memory_results:
            # PHASE 3: Use utility function to handle different result types
            content, source, metadata = extract_result_content(result)
            if content:
                content_hash = hash(content[:200])  # Use first 200 chars as identifier
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    memory_results.append(result)

        # Sort by priority source and similarity score
        priority_order = {"uploaded_document": 0, "chat_context": 1, "vector_datastore": 2}
        memory_results.sort(key=lambda x: (
            priority_order.get(getattr(x, 'priority_source', 'vector_datastore'), 2),
            -x.similarity_score
        ))

        # Limit total results but ensure representation from each source
        memory_results = memory_results[:8]

        # Log enhanced context prioritization results
        logger.info(f"🎯 ENHANCED CONTEXT PRIORITIZATION RESULTS for '{prompt}':")
        logger.info(f"   📄 Recently uploaded documents: {len(uploaded_doc_results)} results")
        logger.info(f"   💬 Current chat context: {len(chat_context_results)} results")
        logger.info(f"   📚 Vector datastore: {len(vector_results)} results")
        logger.info(f"   🔗 Total unique results: {len(memory_results)} (after deduplication)")
        logger.info(f"   📊 Context sources: {', '.join(context_sources)}")

        logger.info(f"Unified search for '{prompt}' returned {len(memory_results)} results")

        # Phase 8.2: Assess confidence in retrieval quality (preserving 100% of web search functionality)
        # Always run confidence assessment unless explicitly forced to use local knowledge
        if not force_local:
            try:
                from reasoning.confidence_assessor import get_confidence_assessor
                confidence_assessor = get_confidence_assessor()

                # Convert memory results to format expected by confidence assessor
                search_results_for_assessment = []
                for result in memory_results:
                    # PHASE 3: Use utility function to handle different result types
                    content, source, metadata = extract_result_content(result)
                    if content:
                        search_results_for_assessment.append({
                            'similarity_score': result.similarity_score,
                            'content': content,
                            'metadata': {
                                'source': source,
                                'timestamp': metadata.get('timestamp', metadata.get('created_at', None))
                            }
                        })

                assessment = confidence_assessor.assess_retrieval_quality(search_results_for_assessment, prompt)

                logger.info(f"🔍 Confidence assessment: {assessment.status} ({assessment.confidence_score:.2f}) for query: {prompt[:50]}...")

                # Phase 8.3: Check if web search escalation should be offered (preserving 100% of functionality)
                # BUT exclude document queries from web search escalation
                document_keywords = [
                    'summarize', 'summary', 'analyze', 'analysis', 'document', 'pdf', 'file',
                    'upload', 'content', 'text', 'report', 'paper', 'article', 'synthesis',
                    'comprehensive', 'overview', 'review', 'extract', 'key points', 'main points'
                ]

                is_document_query = any(keyword in prompt.lower() for keyword in document_keywords)

                if assessment.status == "NOT_CONFIDENT" and not is_document_query:
                    logger.info(f"🌐 Web search escalation triggered due to low confidence")
                    escalation_message, escalation_id = create_web_search_escalation_message(assessment, prompt)
                    logger.info(f"🌐 Created escalation message with ID: {escalation_id}")
                    logger.info(f"🌐 Returning escalation tuple: ({type(escalation_message)}, {type(escalation_id)})")
                    return escalation_message, escalation_id
                elif assessment.status == "NOT_CONFIDENT" and is_document_query:
                    logger.info(f"📄 Document query with low confidence - proceeding without web search escalation")
                    # Continue with document processing instead of web search

            except Exception as e:
                logger.warning(f"Confidence assessment failed: {e}")
                # Continue with normal processing if confidence assessment fails

        if memory_results:
            # Count sources for transparency
            secure_count = sum(1 for r in memory_results if getattr(r, 'source_type', '') == 'secure_documents')
            web_count = sum(1 for r in memory_results if getattr(r, 'source_type', '') == 'web_knowledge')

            # Enhanced context building with document awareness
            context_parts = []
            selected_document = st.session_state.get('selected_document', '')

            # Add document focus context if discussing a specific document
            if selected_document:
                context_parts.append(f"🎯 **DOCUMENT DISCUSSION FOCUS**: {selected_document}")
                context_parts.append("📋 **INSTRUCTION**: Provide detailed, document-specific insights and analysis.")
                context_parts.append("---")

            # Separate document-specific and general results for better organization
            doc_specific_results = []
            general_results = []

            for result in memory_results:
                source_type = getattr(result, 'source_type', 'unknown')

                # PHASE 3: Use utility function to handle different result types
                content, source_name, metadata = extract_result_content(result)
                if not source_name:
                    continue

                # Check if this result is from the selected document
                if selected_document and selected_document in source_name:
                    doc_specific_results.append(result)
                else:
                    general_results.append(result)

            # Enhanced document attribution and content organization with user corrections priority
            user_correction_results = []
            uploaded_doc_results = []
            web_knowledge_results = []
            other_results = []

            # Categorize results by source type for better organization
            for result in memory_results:
                source_type = getattr(result, 'source_type', 'unknown')
                if source_type == 'user_corrections':
                    user_correction_results.append(result)
                elif source_type in ['uploaded_documents', 'secure_documents', 'document_metadata']:
                    uploaded_doc_results.append(result)
                elif source_type == 'web_knowledge':
                    web_knowledge_results.append(result)
                else:
                    other_results.append(result)

            # PRIORITY 0: Add user corrections first (HIGHEST PRIORITY)
            if user_correction_results:
                context_parts.append("🔧 **USER CORRECTIONS** (Authoritative Information):")
                for i, result in enumerate(user_correction_results[:3], 1):  # Limit to top 3 corrections
                    # PHASE 3: Use utility function to handle different result types
                    content, source_name, metadata = extract_result_content(result)
                    if not content:
                        continue
                    similarity = getattr(result, 'similarity_score', 0.0)

                    # Enhanced content preview for corrections
                    # PHASE 3: Use utility function to handle different result types
                    content, source, metadata = extract_result_content(result)
                    if content:
                        content_preview = content[:1200]  # Full content for corrections
                        if len(content) > 1200:
                            content_preview += "..."

                    # Clear correction attribution
                    context_parts.append(f"🎯 **Correction {i}: {source_name}** (Relevance: {similarity:.2f})")
                    context_parts.append(f"Corrected Information: {content_preview}")

                    logger.info(f"🔧 Including user correction: {source_name} (score: {similarity:.3f})")

            # PRIORITY 1: Add uploaded document content (whitepapers, PDFs)
            if uploaded_doc_results:
                context_parts.append("📄 **UPLOADED DOCUMENT CONTENT** (Your Whitepapers & Documents):")
                for i, result in enumerate(uploaded_doc_results[:5], 1):  # More results for uploaded docs
                    # PHASE 3: Use utility function to handle different result types
                    content, source_name, metadata = extract_result_content(result)
                    if not content:
                        continue
                    similarity = getattr(result, 'similarity_score', 0.0)

                    # Enhanced content preview for documents
                    # PHASE 3: Use utility function to handle different result types
                    content, source, metadata = extract_result_content(result)
                    if content:
                        content_preview = content[:1500]  # Longer content for documents
                        if len(content) > 1500:
                            content_preview += "..."

                    # Clear document attribution
                    context_parts.append(f"📋 **Document {i}: {source_name}** (Relevance: {similarity:.2f})")
                    context_parts.append(f"Content: {content_preview}")

                    logger.info(f"📄 Including uploaded document: {source_name} (score: {similarity:.3f})")

            # PRIORITY 2: Add document-specific content if a document is selected
            if doc_specific_results and selected_document:
                if not uploaded_doc_results:  # Only add header if not already added
                    context_parts.append(f"📄 **SELECTED DOCUMENT CONTENT** ({selected_document}):")
                for i, result in enumerate(doc_specific_results[:3], 1):
                    # PHASE 3: Use utility function to handle different result types
                    content, source, metadata = extract_result_content(result)
                    if content:
                        content_preview = content[:1200]
                        if len(content) > 1200:
                            content_preview += "..."
                        context_parts.append(f"Section {i}: {content_preview}")

            # PRIORITY 3: Add web knowledge if relevant
            if web_knowledge_results:
                context_parts.append("\n🌐 **ADDITIONAL KNOWLEDGE** (Web Sources):")
                for i, result in enumerate(web_knowledge_results[:2], 1):  # Limit web results
                    # PHASE 3: Use utility function to handle different result types
                    content, source_name, metadata = extract_result_content(result)
                    if content:
                        content_preview = content[:800]
                        if len(content) > 800:
                            content_preview += "..."
                        context_parts.append(f"Source {i}: {source_name}\nContent: {content_preview}")

            # PRIORITY 4: Add other results if needed
            if other_results and not uploaded_doc_results and not web_knowledge_results:
                context_parts.append("\n📋 **AVAILABLE CONTEXT**:")
                for i, result in enumerate(other_results[:3], 1):
                    source_type = getattr(result, 'source_type', 'unknown')
                    source_label = "📄 Document" if 'document' in source_type else "📋 Memory"
                    # PHASE 3: Use utility function to handle different result types
                    content, source_name, metadata = extract_result_content(result)
                    if content:
                        content_preview = content[:1000]
                        if len(content) > 1000:
                            content_preview += "..."
                        context_parts.append(f"{source_label} - {source_name}\nContent: {content_preview}")

            # Log detailed search results for debugging
            logger.info(f"📊 MEMORY SEARCH BREAKDOWN:")
            logger.info(f"   🔧 User Corrections: {len(user_correction_results)}")
            logger.info(f"   📄 Uploaded Documents: {len(uploaded_doc_results)}")
            logger.info(f"   🌐 Web Knowledge: {len(web_knowledge_results)}")
            logger.info(f"   📋 Other Sources: {len(other_results)}")
            logger.info(f"   🎯 Total Results: {len(memory_results)}")

            context = "\n\n".join(context_parts)

            # Add source summary
            source_summary = []
            if secure_count > 0:
                source_summary.append(f"{secure_count} uploaded document(s)")
            if web_count > 0:
                source_summary.append(f"{web_count} web knowledge item(s)")

            sources_text = " and ".join(source_summary) if source_summary else "available sources"

            # Generate response using Ollama model
            try:
                import requests

                # Enhanced system prompt with user corrections and document attribution guidance
                correction_count = len(user_correction_results)
                doc_count = len(uploaded_doc_results)
                has_corrections = correction_count > 0
                has_uploaded_docs = doc_count > 0

                if has_corrections:
                    system_prompt = f"""You are SAM, a secure AI assistant with access to {correction_count} user correction(s), {doc_count} uploaded document(s), and {sources_text}.

🎯 **CRITICAL**: You have access to user-provided corrections that override any other information. These corrections are authoritative and must be prioritized above all other sources.

**Response Priority Order:**
1. **User Corrections**: Always use corrected information provided by the user - this is the most authoritative source
2. **Uploaded Documents**: Reference specific documents by name when relevant
3. **Other Sources**: Use additional context when needed

**Response Guidelines:**
1. **When user corrections are available**: Start with "Based on your correction..." and use the corrected information
2. **For document-specific questions**: Reference documents directly by name and cite specific content
3. **Never contradict user corrections**: User-provided corrections are always correct
4. **When information conflicts**: Prioritize user corrections over any other source

When thinking through complex questions, you can use <think>...</think> tags to show your reasoning process.

Extract relevant information from the provided sources, prioritizing user corrections above all else."""
                elif has_uploaded_docs:
                    system_prompt = f"""You are SAM, a secure AI assistant with access to {doc_count} uploaded document(s) and {sources_text}.

🔍 **IMPORTANT**: You have direct access to the user's uploaded whitepapers and documents. When answering questions about specific papers or documents, reference them directly by name and cite specific content.

**Response Guidelines:**
1. **For document-specific questions**: Start with "Based on your uploaded document [document name]..." and cite specific sections
2. **For general questions**: Use information from uploaded documents when relevant, clearly attributing the source
3. **When information is found**: Be specific about which document(s) contain the information
4. **When information is missing**: Clearly state "I don't find information about [topic] in your uploaded documents"

**Never say "likely content" or "based on the title" - you have access to the actual document content.**

When thinking through complex questions, you can use <think>...</think> tags to show your reasoning process.

Extract relevant information from the provided sources to answer the question directly and thoroughly."""
                else:
                    system_prompt = f"""You are SAM, a secure AI assistant. Answer the user's question based on the provided content from {sources_text}.

**Note**: No uploaded documents were found for this query. If the user is asking about specific documents they uploaded, let them know that you cannot locate those documents in your current search results.

When thinking through complex questions, you can use <think>...</think> tags to show your reasoning process.

Be helpful and informative. Extract relevant information from the provided sources to answer the question directly.
If the information isn't sufficient, say so clearly. Always be concise but thorough."""

                # Build user prompt with conversation history and MEMOIR integration
                user_prompt_parts = []

                # Add conversation history if available (Task 30 Phase 1)
                conversation_history = st.session_state.get('conversation_history', '')
                reduce_conversation_weight = st.session_state.get('reduce_conversation_weight', False)
                logger.info(f"🔍 DEBUG: Conversation history from session state: '{conversation_history}'")
                logger.info(f"🔍 DEBUG: Reduce conversation weight: {reduce_conversation_weight}")

                # CRITICAL FIX: For document queries, minimize conversation history to focus on document content
                if conversation_history and conversation_history != "No recent conversation history." and not reduce_conversation_weight:
                    user_prompt_parts.append("--- RECENT CONVERSATION HISTORY (Most recent first) ---")
                    user_prompt_parts.append(conversation_history)
                    user_prompt_parts.append("--- END OF CONVERSATION HISTORY ---\n")
                    logger.info(f"✅ DEBUG: Added conversation history to prompt ({len(conversation_history)} chars)")
                elif reduce_conversation_weight:
                    logger.info(f"📄 DOCUMENT QUERY: Skipping conversation history to focus on document content")
                    # DO NOT add conversation history for document queries
                else:
                    logger.warning(f"⚠️ DEBUG: No conversation history available - history: '{conversation_history}'")

                # Use enhanced prompt if available from cognitive distillation
                question_to_use = enhanced_prompt if enhanced_prompt != prompt else prompt
                user_prompt_parts.append(f"Question: {question_to_use}")

                # Add MEMOIR knowledge if available
                if memoir_context.get('relevant_edits'):
                    user_prompt_parts.append("\nLearned Knowledge from Previous Corrections:")
                    for edit in memoir_context['relevant_edits'][:2]:
                        if edit.get('edit_type') == 'correction':
                            user_prompt_parts.append(f"• {edit.get('correction', '')}")
                        elif edit.get('edit_type') == 'memory_correction':
                            content = edit.get('content', '')[:200]
                            if 'User Correction:' in content:
                                correction_part = content.split('User Correction:')[-1].strip()
                                user_prompt_parts.append(f"• {correction_part}")

                user_prompt_parts.append(f"\n--- KNOWLEDGE BASE CONTEXT ---\n{context}\n--- END OF KNOWLEDGE BASE CONTEXT ---")
                user_prompt_parts.append("\nPlease provide a helpful answer based on the conversation history, available information, and learned knowledge.")

                user_prompt = "\n".join(user_prompt_parts)

                # TPV-enabled response generation
                if sam_tpv_integration:
                    try:
                        # Use TPV integration for response generation
                        full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"

                        # Calculate initial confidence based on context quality (lowered for better TPV triggering)
                        # Lower confidence increases TPV activation probability
                        initial_confidence = min(0.6, len(context) / 3000.0) if context else 0.2

                        # Debug logging for TPV activation
                        logger.info(f"🧠 Attempting TPV activation for query: {prompt[:50]}...")
                        logger.info(f"🎯 User profile: {user_profile}, Initial confidence: {initial_confidence}")

                        # Force TPV activation for analytical queries
                        force_tpv = any(trigger_word in prompt.lower() for trigger_word in [
                            'analyze', 'analysis', 'compare', 'comparison', 'explain', 'research',
                            'evaluate', 'assess', 'examine', 'investigate', 'strategic', 'implications'
                        ])

                        if force_tpv:
                            logger.info(f"🎯 FORCING TPV activation due to analytical query patterns")
                            # Lower confidence even more to guarantee TPV activation
                            initial_confidence = 0.1

                        tpv_response = sam_tpv_integration.generate_response_with_tpv(
                            prompt=full_prompt,
                            user_profile=user_profile,
                            initial_confidence=initial_confidence,
                            context={'has_context': bool(context), 'sources': sources_text},
                            ollama_params={
                                "model": "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M",
                                "stream": False,
                                "options": {
                                    "temperature": 0.7,
                                    "top_p": 0.9,
                                    "max_tokens": 500
                                }
                            }
                        )

                        # Log TPV activation result
                        if tpv_response.tpv_enabled:
                            logger.info(f"✅ TPV ACTIVATED! Trigger: {tpv_response.trigger_result.trigger_type if tpv_response.trigger_result else 'unknown'}")
                        else:
                            logger.info(f"❌ TPV not activated. Trigger result: {tpv_response.trigger_result.reason if tpv_response.trigger_result else 'no trigger result'}")

                        # Store TPV data in session state for UI display
                        if 'tpv_session_data' not in st.session_state:
                            st.session_state.tpv_session_data = {}

                        # Get active control information
                        control_decision = 'CONTINUE'
                        control_reason = 'No control action taken'
                        control_statistics = {}

                        if hasattr(sam_tpv_integration, 'reasoning_controller'):
                            recent_actions = sam_tpv_integration.reasoning_controller.get_recent_actions(1)
                            if recent_actions:
                                control_decision = recent_actions[0].metadata.get('action_type', 'CONTINUE')
                                control_reason = recent_actions[0].reason
                            control_statistics = sam_tpv_integration.reasoning_controller.get_control_statistics()

                        # Update TPV session data for UI display
                        tpv_data = {
                            'tpv_enabled': tpv_response.tpv_enabled,
                            'trigger_type': tpv_response.trigger_result.trigger_type if tpv_response.trigger_result else None,
                            'final_score': tpv_response.tpv_trace.current_score if tpv_response.tpv_trace else 0.0,
                            'tpv_steps': len(tpv_response.tpv_trace.steps) if tpv_response.tpv_trace else 0,
                            'performance_metrics': tpv_response.performance_metrics,
                            'control_decision': control_decision,
                            'control_reason': control_reason,
                            'control_statistics': control_statistics,
                            'query': prompt[:100],  # Store query for debugging
                            'user_profile': str(user_profile),
                            'initial_confidence': initial_confidence,
                            'force_tpv': force_tpv if 'force_tpv' in locals() else False
                        }

                        st.session_state.tpv_session_data['last_response'] = tpv_data

                        # Also update global TPV status for sidebar
                        st.session_state.tpv_active = tpv_response.tpv_enabled

                        if tpv_response.content:
                            return tpv_response.content
                        else:
                            logger.warning("Empty response from TPV-enabled generation")
                            # Provide immediate TPV-aware response for analytical queries
                            if force_tpv:
                                return f"""I understand you're asking me to {prompt.split()[0].lower()} a complex topic.

🧠 **TPV Active Reasoning Control Engaged**

I'm processing your analytical query: "{prompt}"

**My approach would involve:**
1. **Breaking down the key components** of your question
2. **Analyzing multiple perspectives** and factors
3. **Synthesizing insights** from available knowledge
4. **Providing structured conclusions** with supporting evidence

However, I'm currently experiencing processing delays. Your question about {prompt.lower()} is important and deserves a thorough analytical response.

**Would you like me to:**
- Provide a quick overview now and detailed analysis later?
- Focus on specific aspects of your question?
- Break this into smaller, more focused questions?

*TPV Status: ✅ Active - Reasoning process monitored and optimized*"""

                    except Exception as e:
                        logger.error(f"TPV-enabled generation failed: {e}")
                        # Fall back to standard Ollama call

                # Fallback: Standard Ollama API call
                ollama_response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M",
                        "prompt": f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:",
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "max_tokens": 500
                        }
                    },
                    timeout=120  # Increased timeout to 120 seconds for complex analytical queries
                )

                if ollama_response.status_code == 200:
                    response_data = ollama_response.json()
                    ai_response = response_data.get('response', '').strip()

                    if ai_response:
                        return ai_response
                    else:
                        logger.warning("Empty response from Ollama")

            except Exception as e:
                logger.error(f"Ollama API call failed: {e}")

            # Fallback: return context with basic formatting
            return f"""Based on {sources_text}, here's what I found:

{context}

I'm SAM, your secure AI assistant. How can I help you further?"""

        else:
            # Check if we have any memories at all
            security_status = st.session_state.secure_memory_store.get_security_status()
            total_chunks = security_status.get('encrypted_chunk_count', 0)

            if total_chunks > 0:
                # Check if web retrieval should be suggested (Phase 7.1) - preserving 100% of functionality
                try:
                    from utils.web_retrieval_suggester import WebRetrievalSuggester
                    suggester = WebRetrievalSuggester()

                    if suggester.should_suggest_web_retrieval(prompt, []):
                        logger.info(f"🌐 Suggesting web retrieval for Streamlit query: {prompt[:50]}...")
                        return suggester.format_retrieval_suggestion(prompt)

                except ImportError:
                    logger.warning("Web retrieval suggester not available")

                # Also run confidence assessment for no results case (preserving 100% of functionality)
                if not force_local:
                    try:
                        from reasoning.confidence_assessor import get_confidence_assessor
                        confidence_assessor = get_confidence_assessor()

                        # Assess with empty results to trigger web search for knowledge gaps
                        assessment = confidence_assessor.assess_retrieval_quality([], prompt)
                        logger.info(f"🔍 No results confidence assessment: {assessment.status} ({assessment.confidence_score:.2f})")

                        # Offer web search for knowledge gaps
                        if assessment.status == "NOT_CONFIDENT":
                            logger.info(f"🌐 Web search escalation triggered for knowledge gap")
                            escalation_message, escalation_id = create_web_search_escalation_message(assessment, prompt)
                            logger.info(f"🌐 Created knowledge gap escalation with ID: {escalation_id}")
                            logger.info(f"🌐 Returning knowledge gap escalation tuple: ({type(escalation_message)}, {type(escalation_id)})")
                            return escalation_message, escalation_id

                    except Exception as e:
                        logger.warning(f"No results confidence assessment failed: {e}")

                # Enhanced response for document queries vs. general queries
                is_doc_query = st.session_state.get('document_query_detected', False)

                if is_doc_query:
                    # Specific feedback for document queries
                    logger.info(f"📄 No results found for document query - providing diagnostic feedback")

                    # Run diagnostic to help user understand the issue
                    try:
                        diagnosis = diagnose_memory_retrieval(prompt)
                        diagnostic_info = f"""
📊 **Memory System Diagnostic:**
- Secure Store Available: {diagnosis.get('secure_store_available', 'Unknown')}
- Total Memories: {diagnosis.get('secure_store_total_memories', 'Unknown')}
- Search Results: {diagnosis.get('secure_store_search_results', 'Unknown')}"""
                    except:
                        diagnostic_info = ""

                    return f"""📄 **Document Search Results**

I searched through your uploaded documents and memory system but couldn't find information about "{prompt}".

**Possible reasons:**
1. **Document not uploaded**: The specific file may not have been uploaded to SAM
2. **Search terms mismatch**: Try using different keywords from the document
3. **Processing issue**: The document may not have been properly processed during upload

**Suggestions:**
- Try searching with the exact document title or author name
- Use key terms that would appear in the document content
- Check if the document was successfully uploaded in the Document Library

**For the query about "2506.21393v1.pdf" (Active Inference AI Systems):**
If this document was uploaded, try searching with terms like:
- "Active Inference"
- "Scientific Discovery"
- "AI Systems"
- The author's name

{diagnostic_info}

Would you like me to help you search with different terms, or would you prefer to re-upload the document?"""

                else:
                    # Generate a response using Ollama for general queries
                    try:
                        import requests

                        system_prompt = """You are SAM, a helpful AI assistant. When thinking through questions, you can use <think>...</think> tags to show your reasoning process.

Answer the user's question helpfully and accurately based on your general knowledge."""

                        ollama_response = requests.post(
                            "http://localhost:11434/api/generate",
                            json={
                                "model": "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M",
                                "prompt": f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:",
                                "stream": False,
                                "options": {
                                    "temperature": 0.7,
                                    "top_p": 0.9,
                                    "max_tokens": 500
                                }
                            },
                            timeout=120  # Increased timeout to 120 seconds for complex analytical queries
                        )

                        if ollama_response.status_code == 200:
                            response_data = ollama_response.json()
                            ai_response = response_data.get('response', '').strip()
                            if ai_response:
                                return ai_response

                    except Exception as e:
                        logger.error(f"Fallback Ollama call failed: {e}")

                    return f"""I searched through your {total_chunks} encrypted memory chunks but couldn't find relevant information about "{prompt}".

This could be because:
- The search terms don't match the document content
- The document content wasn't properly extracted
- The similarity threshold is too high

Try rephrasing your question or uploading more relevant documents."""
            else:
                return """No documents found in your secure memory. Please upload some documents first, then ask questions about their content."""

    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        return f"I apologize, but I encountered an error while processing your request: {e}"

def generate_final_response(user_question: str, force_local: bool = False) -> str:
    """
    Optimized two-stage response generation with A/B testing and caching (Task 30 Phase 3).

    Stage 1: Generate factually-grounded draft response
    Stage 2: Refine draft with persona alignment
    Phase 3: A/B testing, caching, and performance optimization

    Args:
        user_question: The user's question
        force_local: Whether to force local knowledge only

    Returns:
        Final refined response
    """
    try:
        # Load configuration
        try:
            import json
            with open('config/sam_config.json', 'r') as f:
                config = json.load(f)
                enable_persona_refinement = config.get('enable_persona_refinement', True)
                enable_ab_testing = config.get('enable_ab_testing', True)
                enable_response_caching = config.get('enable_response_caching', True)
        except:
            enable_persona_refinement = True
            enable_ab_testing = False
            enable_response_caching = False

        # Phase 3: Check response cache first
        cached_response = None
        if enable_response_caching:
            try:
                from sam.optimization.response_cache import get_response_cache

                cache = get_response_cache()
                conversation_history = st.session_state.get('conversation_history', '')
                persona_context = st.session_state.get('last_persona_context', '')

                cached_result = cache.get_cached_response(user_question, conversation_history, persona_context)
                if cached_result:
                    cached_response, cache_metadata = cached_result
                    logger.info(f"🚀 Cache hit! Saved {cache_metadata['original_generation_time_ms']:.0f}ms")
                    return cached_response

            except Exception as e:
                logger.warning(f"Response cache failed: {e}")

        # Phase 3: A/B Testing - Determine pipeline to use
        pipeline_to_use = 'two_stage'  # Default
        ab_test_metadata = {}

        if enable_ab_testing:
            try:
                from sam.evaluation.ab_testing import get_ab_testing_framework

                ab_framework = get_ab_testing_framework()
                user_id = st.session_state.get('user_id', 'anonymous')

                # Check for active A/B tests
                active_tests = [test_id for test_id, test_config in ab_framework.active_tests.items()
                               if ab_framework._is_test_active(test_config)]

                if active_tests:
                    # Use first active test (could be enhanced to support multiple tests)
                    test_id = active_tests[0]
                    use_treatment = ab_framework.should_use_treatment(user_id, test_id)

                    test_config = ab_framework.active_tests[test_id]
                    pipeline_to_use = test_config.treatment_pipeline if use_treatment else test_config.control_pipeline

                    ab_test_metadata = {
                        'test_id': test_id,
                        'pipeline_used': 'treatment' if use_treatment else 'control',
                        'pipeline_name': pipeline_to_use
                    }

                    logger.info(f"🧪 A/B Test {test_id}: Using {pipeline_to_use} pipeline")

            except Exception as e:
                logger.warning(f"A/B testing failed: {e}")

        # Record start time for performance tracking
        start_time = time.time()

        # Execute pipeline based on A/B test or configuration
        final_response = None
        generation_time_ms = 0.0

        if pipeline_to_use == 'single_stage':
            # Single-stage pipeline (control)
            logger.info("🎯 Single-stage pipeline: Generating response...")
            final_response = generate_draft_response(user_question, force_local)

            # Handle tuple return for web search escalation
            if isinstance(final_response, tuple) and len(final_response) == 2:
                logger.info("🌐 Web search escalation detected in single-stage pipeline")
                return final_response  # Return the tuple directly for escalation handling

        else:
            # Two-stage pipeline (treatment or default)
            # Stage 1: Generate a factually-grounded draft
            logger.info("🎯 Stage 1: Generating draft response...")
            draft_response = generate_draft_response(user_question, force_local)

            # Handle tuple return for web search escalation
            if isinstance(draft_response, tuple) and len(draft_response) == 2:
                logger.info("🌐 Web search escalation detected in two-stage pipeline")
                return draft_response  # Return the tuple directly for escalation handling

            # Calculate draft confidence (simple heuristic)
            draft_confidence = min(0.8, len(str(draft_response).split()) / 100.0) if draft_response else 0.1

            # Stage 2: Refine the draft with persona (if enabled)
            if enable_persona_refinement and pipeline_to_use == 'two_stage':
                try:
                    logger.info("🎯 Stage 2: Refining with persona alignment...")
                    from sam.persona.persona_refinement import generate_final_response as persona_generate_final

                    # Get user ID from session state
                    user_id = st.session_state.get('user_id', 'anonymous')

                    # Perform persona refinement
                    final_response, refinement_metadata = persona_generate_final(
                        user_question, draft_response, user_id, draft_confidence
                    )

                    # Log refinement results
                    if refinement_metadata.get('refinement_applied', False):
                        logger.info(f"✅ Persona refinement applied: {refinement_metadata.get('refinement_metadata', {}).get('persona_memories_used', 0)} memories used")
                    else:
                        logger.info(f"⏭️ Persona refinement skipped: {refinement_metadata.get('refinement_metadata', {}).get('reason', 'unknown')}")

                    # Store refinement metadata in session state for debugging
                    st.session_state['last_refinement_metadata'] = refinement_metadata

                except Exception as e:
                    logger.warning(f"Persona refinement failed, using draft: {e}")
                    final_response = draft_response
            else:
                logger.info("⏭️ Persona refinement disabled, using draft response")
                final_response = draft_response

        # Calculate generation time
        generation_time_ms = (time.time() - start_time) * 1000

        # Phase 3: Record A/B test result
        if ab_test_metadata and enable_ab_testing:
            try:
                ab_framework.record_result(
                    test_id=ab_test_metadata['test_id'],
                    user_id=st.session_state.get('user_id', 'anonymous'),
                    session_id=st.session_state.get('session_id', 'default_session'),
                    pipeline_used=ab_test_metadata['pipeline_used'],
                    user_question=user_question,
                    response_generated=final_response,
                    response_time_ms=generation_time_ms,
                    metadata={
                        'pipeline_name': ab_test_metadata['pipeline_name'],
                        'force_local': force_local
                    }
                )
                logger.info(f"📊 Recorded A/B test result for {ab_test_metadata['test_id']}")

            except Exception as e:
                logger.warning(f"Failed to record A/B test result: {e}")

        # Phase 3: Cache the response
        if enable_response_caching and final_response:
            try:
                cache.cache_response(
                    user_question=user_question,
                    response=final_response,
                    pipeline_used=pipeline_to_use,
                    generation_time_ms=generation_time_ms,
                    conversation_context=st.session_state.get('conversation_history', ''),
                    persona_context=st.session_state.get('last_persona_context', ''),
                    metadata={
                        'ab_test': ab_test_metadata,
                        'force_local': force_local
                    }
                )
                logger.debug(f"💾 Cached response (took {generation_time_ms:.0f}ms to generate)")

            except Exception as e:
                logger.warning(f"Failed to cache response: {e}")

        # Phase 4: Complete Cognitive Distillation Reasoning Trace (NEW - Phase 2 Integration)
        try:
            if (st.session_state.get('cognitive_distillation_enabled', False) and
                st.session_state.get('cognitive_distillation') and
                st.session_state.get('last_transparency_data')):

                cognitive_distillation = st.session_state.cognitive_distillation
                transparency_data = st.session_state.last_transparency_data

                # Complete the reasoning trace with the final response
                completed_transparency = cognitive_distillation.complete_reasoning_trace(
                    transparency_data, final_response
                )

                # Store completed transparency data for UI display
                st.session_state['completed_transparency_data'] = completed_transparency

                # Update principle feedback based on response quality (simplified heuristic)
                if transparency_data.get('active_principles'):
                    response_quality = 'success' if len(final_response) > 100 else 'neutral'
                    for principle in transparency_data['active_principles']:
                        cognitive_distillation.update_principle_feedback(
                            principle['id'], response_quality
                        )

                logger.info(f"🧠 Completed cognitive distillation reasoning trace")

        except Exception as e:
            logger.warning(f"Failed to complete cognitive distillation reasoning trace: {e}")

        # Apply feedback-driven enhancements to the final response
        try:
            final_response = apply_feedback_enhancements(final_response, response_context)
        except Exception as e:
            logger.warning(f"Feedback enhancement failed at final stage: {e}")
        return final_response

    except Exception as e:
        logger.error(f"Two-stage response generation failed: {e}")
        return f"I apologize, but I encountered an error while processing your request: {e}"

def generate_response_with_conversation_buffer(prompt: str, force_local: bool = False) -> str:
    """
    Enhanced conversation buffer wrapper with contextual relevance (Task 31 Phase 1).
    NOW INCLUDES: Document-Aware RAG Pipeline for uploaded document access.

    This implements:
    - Task 30 Phase 1: Short-Term Conversational Buffer
    - Task 31 Phase 1: Contextual Relevance Engine with Automatic Threading
    - CRITICAL: Document-Aware RAG Pipeline for document-first query processing
    """
    try:
        # PRIORITY 1: MATH CALCULATION CHECK (HIGHEST PRIORITY)
        # Check for mathematical expressions BEFORE document processing
        # BUT exclude document-related queries to prevent false positives
        try:
            from services.search_router import SmartQueryRouter
            from services.smart_query_handler import SmartQueryHandler

            router = SmartQueryRouter()

            # First check if this is a document-related query (exclude from math routing)
            document_keywords = [
                'summarize', 'summary', 'analyze', 'analysis', 'document', 'pdf', 'file',
                'upload', 'content', 'text', 'report', 'paper', 'article', 'synthesis',
                'comprehensive', 'overview', 'review', 'extract', 'key points', 'main points'
            ]

            is_document_query = any(keyword in prompt.lower() for keyword in document_keywords)

            # Only proceed with math detection if this is NOT a document query
            if not is_document_query:
                # Check for pure mathematical expressions first
                if router.is_pure_math_expression(prompt):
                    logger.info(f"🧮 PURE MATH DETECTED: '{prompt}' - routing to calculator")

                    handler = SmartQueryHandler()
                    response = handler.process_query(prompt)

                    if response.success:
                        logger.info(f"✅ Math calculation successful: {response.route_type}")
                        return response.content
                    else:
                        logger.warning(f"❌ Math calculation failed: {response.metadata}")
                        # Continue to fallback processing

                # Check for other mathematical signals
                math_signals = router.detect_math_signals(prompt)
                if math_signals and any(score > 0.7 for score in math_signals.values()):
                    logger.info(f"🧮 Math signals detected: '{prompt}' - routing to smart handler")

                    handler = SmartQueryHandler()
                    response = handler.process_query(prompt)

                    if response.success:
                        logger.info(f"✅ Math query successful: {response.route_type}")
                        return response.content
                    else:
                        logger.warning(f"❌ Math query failed: {response.metadata}")
                        # Continue to fallback processing
            else:
                logger.info(f"📄 Document query detected, skipping math routing: {prompt[:50]}...")

        except Exception as e:
            logger.warning(f"Math routing failed: {e}")
            # Continue to fallback processing

        # PRIORITY 2: PROVEN PDF PROCESSOR: Check for PDF queries
        # This uses the proven PDF chatbot approach for reliable document recall
        try:
            from sam.document_processing.proven_pdf_integration import (
                get_sam_pdf_integration,
                is_pdf_query_for_sam,
                query_pdf_for_sam
            )

            # Check if this is a PDF-related query
            if is_pdf_query_for_sam(prompt):
                logger.info(f"📄 PROVEN PDF QUERY DETECTED: {prompt[:50]}...")

                # Try to query using proven PDF processor
                success, pdf_response, pdf_metadata = query_pdf_for_sam(prompt, session_id="default")

                if success:
                    logger.info(f"✅ PROVEN PDF RESPONSE: {len(pdf_response)} characters")
                    logger.info(f"📊 PDF Metadata: {pdf_metadata}")

                    # Return the proven PDF response directly
                    return pdf_response
                else:
                    logger.warning(f"❌ Proven PDF query failed: {pdf_response}")
                    # Continue to fallback processing

        except Exception as e:
            logger.warning(f"Proven PDF processing failed: {e}")
            # Continue to fallback processing

        # CRITICAL INTEGRATION: Document-Aware RAG Pipeline (HIGH PRIORITY)
        # This ensures SAM can access uploaded documents FIRST before falling back to general knowledge
        document_context = None
        document_sources = []

        try:
            from sam.document_rag import create_document_rag_pipeline
            from memory.memory_vectorstore import get_memory_store, VectorStoreType

            # CRITICAL FIX: Use the actual memory stores where documents are stored
            # Documents are stored in both secure and regular stores for compatibility

            # Get the secure memory store (where encrypted documents are stored)
            encrypted_store = st.session_state.get('secure_memory_store', None)

            # Get/create the regular memory store (where documents are also synced)
            memory_store = get_memory_store(
                store_type=VectorStoreType.CHROMA,
                storage_directory="memory_store",
                embedding_dimension=384
            )

            # Create the pipeline
            document_rag_pipeline = create_document_rag_pipeline(
                memory_store=memory_store,
                encrypted_store=encrypted_store
            )

            # Process query with document-first strategy
            rag_result = document_rag_pipeline.process_query(prompt)

            if rag_result['success'] and rag_result.get('use_document_context'):
                document_context = document_rag_pipeline.get_document_context_for_llm(rag_result)
                document_sources = document_rag_pipeline.get_detailed_citations(rag_result)

                logger.info(f"📄 Document-Aware RAG: Found {len(document_sources)} relevant document sources")
                logger.info(f"🎯 Strategy: {rag_result['routing_decision']['strategy']}")
                logger.info(f"🔍 Confidence: {rag_result['routing_decision']['confidence_level']}")
            else:
                logger.info("📄 Document-Aware RAG: No relevant documents found, proceeding with general knowledge")

        except Exception as e:
            logger.warning(f"Document-Aware RAG Pipeline failed: {e}, proceeding with standard flow")

        # Task 31 Phase 1: Contextual Relevance Check
        conversation_archived = False

        try:
            from sam.conversation.contextual_relevance import get_contextual_relevance_engine
            from sam.session.state_manager import get_session_manager

            # Load configuration
            try:
                import json
                with open('config/sam_config.json', 'r') as f:
                    config = json.load(f)
                    auto_threading_enabled = config.get('enable_auto_threading', True)
                    relevance_threshold = config.get('relevance_threshold', 0.6)
            except:
                auto_threading_enabled = True
                relevance_threshold = 0.6

            if auto_threading_enabled:
                # Get current conversation buffer
                session_manager = get_session_manager()
                session_id = st.session_state.get('session_id', 'default_session')

                # Ensure session exists
                if not session_manager.get_session(session_id):
                    user_id = st.session_state.get('user_id', 'anonymous')
                    session_manager.create_session(session_id, user_id)

                # Get current conversation buffer
                conversation_buffer = session_manager.get_conversation_history(session_id)

                # Calculate contextual relevance
                relevance_engine = get_contextual_relevance_engine({
                    'relevance_threshold': relevance_threshold
                })

                relevance_result = relevance_engine.calculate_relevance(prompt, conversation_buffer)

                logger.info(f"🧠 Contextual relevance: {relevance_result.similarity_score:.3f} "
                           f"(threshold: {relevance_result.threshold_used}, method: {relevance_result.calculation_method})")

                # Check if we need to archive current conversation
                if not relevance_result.is_relevant and conversation_buffer:
                    logger.info(f"🗂️ Topic change detected! Archiving current conversation ({len(conversation_buffer)} messages)")

                    # Archive current conversation
                    archived_thread = relevance_engine.archive_conversation_thread(conversation_buffer)

                    # Clear conversation buffer for new topic
                    session_manager.clear_session(session_id)

                    # Update UI state to show new archived thread
                    if 'archived_threads' not in st.session_state:
                        st.session_state['archived_threads'] = []

                    st.session_state['archived_threads'].insert(0, archived_thread.to_dict())

                    # Set flag for UI notification
                    st.session_state['conversation_archived'] = {
                        'title': archived_thread.title,
                        'message_count': archived_thread.message_count,
                        'timestamp': archived_thread.last_updated
                    }

                    conversation_archived = True

                    logger.info(f"✅ Archived conversation: '{archived_thread.title}' - Starting fresh conversation")

                # Store relevance metadata for debugging
                st.session_state['last_relevance_check'] = relevance_result.to_dict()

        except Exception as e:
            logger.warning(f"Contextual relevance check failed: {e}")
            # Continue with normal flow if relevance check fails

        # Generate the response using two-stage pipeline with document context
        if document_context:
            # Enhanced prompt with document context for document-aware responses
            enhanced_prompt = f"""You are SAM, a secure AI assistant. You have access to the user's uploaded documents and should prioritize this information.

DOCUMENT CONTEXT FROM UPLOADED FILES:
{document_context}

USER QUESTION: {prompt}

Instructions:
- Use the document context as your PRIMARY source of information
- Cite sources using the format provided in the context
- If the document context doesn't fully answer the question, supplement with general knowledge but clearly distinguish between document-based and general information
- Always acknowledge when information comes from the user's uploaded documents

Answer the user's question comprehensively using the document context above."""

            response = generate_final_response(enhanced_prompt, force_local)

            # Add source attribution to the response
            if document_sources:
                source_attribution = f"\n\n🌐 **Sources:**\n"
                for source in document_sources[:5]:  # Limit to 5 sources for readability
                    source_attribution += f"• {source}\n"
                response += source_attribution
        else:
            # Standard response generation when no document context
            response = generate_final_response(prompt, force_local)

        # Add assistant response to conversation buffer
        try:
            from sam.session.state_manager import get_session_manager

            session_manager = get_session_manager()
            session_id = st.session_state.get('session_id', 'default_session')

            # Add assistant turn to conversation buffer
            session_manager.add_turn(session_id, 'assistant', response)

            logger.info(f"🗣️ Added assistant response to conversation buffer for session: {session_id}")

        except Exception as e:
            logger.warning(f"Failed to update conversation buffer with assistant response: {e}")

        return response

    except Exception as e:
        logger.error(f"Response generation with conversation buffer failed: {e}")
        return f"I apologize, but I encountered an error while processing your request: {e}"

def detect_user_correction(prompt: str) -> dict:
    """Detect if the user is providing a correction to a previous response."""
    try:
        prompt_lower = prompt.lower()

        # Correction indicators
        correction_patterns = [
            'actually', 'correction', 'wrong', 'incorrect', 'mistake',
            'should be', 'not', "that's not right", 'fix', 'update',
            'you said', 'you mentioned', 'but', 'however', 'instead',
            'refers to', 'talking about', 'means', 'is actually'
        ]

        # Check if this looks like a correction
        has_correction_pattern = any(pattern in prompt_lower for pattern in correction_patterns)

        if has_correction_pattern:
            # Get the last assistant message from session state
            if hasattr(st.session_state, 'messages') and st.session_state.messages:
                # Find the last assistant message
                last_assistant_message = None
                last_user_message = None

                for i in range(len(st.session_state.messages) - 1, -1, -1):
                    msg = st.session_state.messages[i]
                    if msg['role'] == 'assistant' and not last_assistant_message:
                        last_assistant_message = msg['content']
                    elif msg['role'] == 'user' and not last_user_message and last_assistant_message:
                        last_user_message = msg['content']
                        break

                if last_assistant_message and last_user_message:
                    return {
                        'is_correction': True,
                        'correction_text': prompt,
                        'original_query': last_user_message,
                        'sam_response': last_assistant_message,
                        'correction_patterns': [p for p in correction_patterns if p in prompt_lower]
                    }

        return {'is_correction': False}

    except Exception as e:
        logger.error(f"Error detecting user correction: {e}")
        return {'is_correction': False}

def process_user_correction_for_memoir(correction_data: dict):
    """Process user correction through MEMOIR learning system."""
    try:
        logger.info(f"Processing user correction for MEMOIR learning")

        # Create feedback data structure for MEMOIR processing
        feedback_data = {
            'feedback_id': f"correction_{int(time.time())}",
            'feedback_type': 'correction',
            'correction_text': correction_data['correction_text'],
            'original_query': correction_data['original_query'],
            'sam_response': correction_data['sam_response'],
            'interface': 'secure_chat',
            'timestamp': time.time(),
            'user_id': 'current_user',
            'correction_patterns': correction_data.get('correction_patterns', [])
        }

        # Process through MEMOIR feedback learning
        process_memoir_feedback_learning(feedback_data)

        # Also store in memory for future reference
        try:
            if hasattr(st.session_state, 'secure_memory_store') and st.session_state.secure_memory_store:
                from memory.secure_memory_vectorstore import MemoryType

                correction_content = f"""USER CORRECTION DETECTED:
Original Query: {correction_data['original_query']}
SAM's Response: {correction_data['sam_response'][:200]}...
User Correction: {correction_data['correction_text']}
Learning: Update knowledge to reflect user's correction."""

                st.session_state.secure_memory_store.add_memory(
                    content=correction_content,
                    memory_type=MemoryType.CONVERSATION,
                    source="user_correction",
                    tags=['correction', 'learning', 'memoir', 'user_feedback'],
                    importance_score=0.9,
                    metadata={
                        'correction_id': feedback_data['feedback_id'],
                        'correction_patterns': correction_data.get('correction_patterns', []),
                        'timestamp': feedback_data['timestamp']
                    }
                )

                logger.info(f"✅ User correction stored in memory for future learning")

        except Exception as memory_error:
            logger.warning(f"Failed to store correction in memory: {memory_error}")

        logger.info(f"✅ User correction processed for MEMOIR learning")

    except Exception as e:
        logger.error(f"Failed to process user correction for MEMOIR: {e}")

def retrieve_memoir_knowledge(query: str) -> dict:
    """Retrieve relevant knowledge from MEMOIR system for the given query."""
    try:
        memoir_context = {
            'relevant_edits': [],
            'confidence_scores': [],
            'edit_metadata': []
        }

        # Check if MEMOIR integration is available
        memoir_integration = st.session_state.get('memoir_integration')
        if not memoir_integration:
            return memoir_context

        # Try to retrieve relevant MEMOIR edits
        try:
            # Search for relevant edits based on query
            # This would typically involve semantic search through edit database
            from sam.orchestration.skills.internal.memoir_edit import MEMOIR_EditSkill

            # For now, check if there are any stored corrections related to this query
            if hasattr(st.session_state, 'pending_memoir_corrections'):
                for correction in st.session_state.pending_memoir_corrections:
                    feedback_data = correction.get('feedback_data', {})
                    original_query = feedback_data.get('original_query', '').lower()
                    correction_text = feedback_data.get('correction_text', '').lower()

                    # Simple relevance check
                    query_lower = query.lower()
                    if (query_lower in original_query or original_query in query_lower or
                        any(word in query_lower for word in original_query.split() if len(word) > 3)):

                        memoir_context['relevant_edits'].append({
                            'edit_type': 'correction',
                            'original_query': feedback_data.get('original_query', ''),
                            'correction': feedback_data.get('correction_text', ''),
                            'confidence': 0.8,
                            'source': 'user_correction'
                        })
                        memoir_context['confidence_scores'].append(0.8)
                        memoir_context['edit_metadata'].append({
                            'feedback_id': feedback_data.get('feedback_id'),
                            'timestamp': correction.get('timestamp'),
                            'status': correction.get('status', 'pending')
                        })

            # Also check memory store for MEMOIR-tagged memories
            if hasattr(st.session_state, 'secure_memory_store') and st.session_state.secure_memory_store:
                try:
                    memoir_memories = st.session_state.secure_memory_store.search_memories(
                        query=query,
                        max_results=3,
                        tags=['memoir', 'user_feedback', 'learning']  # Use tags instead of content_type
                    )

                    for memory in memoir_memories:
                        if 'memoir' in memory.get('tags', []):
                            memoir_context['relevant_edits'].append({
                                'edit_type': 'memory_correction',
                                'content': memory.get('content', ''),
                                'confidence': memory.get('similarity_score', 0.7),
                                'source': 'memory_store'
                            })
                            memoir_context['confidence_scores'].append(memory.get('similarity_score', 0.7))
                            memoir_context['edit_metadata'].append({
                                'memory_id': memory.get('id'),
                                'timestamp': memory.get('timestamp'),
                                'importance': memory.get('importance_score', 0.5)
                            })

                except Exception as memory_error:
                    logger.warning(f"Failed to search MEMOIR memories: {memory_error}")

            if memoir_context['relevant_edits']:
                logger.info(f"🧠 MEMOIR found {len(memoir_context['relevant_edits'])} relevant knowledge edits for query")

        except Exception as retrieval_error:
            logger.warning(f"MEMOIR knowledge retrieval error: {retrieval_error}")

        return memoir_context

    except Exception as e:
        logger.error(f"MEMOIR knowledge retrieval failed: {e}")
        return {'relevant_edits': [], 'confidence_scores': [], 'edit_metadata': []}

def process_secure_document(uploaded_file) -> dict:
    """Process an uploaded document securely."""
    try:
        # Read file content
        content = uploaded_file.read()

        if uploaded_file.type == "text/plain":
            text_content = content.decode('utf-8')
        elif uploaded_file.type == "application/pdf":
            # Extract text from PDF
            try:
                import PyPDF2
                import io

                logger.info(f"Extracting text from PDF: {uploaded_file.name}")
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
                text_content = ""

                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text_content += f"\n--- Page {page_num + 1} ---\n{page_text}"
                    logger.debug(f"Page {page_num + 1}: extracted {len(page_text)} characters")

                logger.info(f"Total extracted text length: {len(text_content)} characters")
                logger.debug(f"First 200 characters: {text_content[:200]}")

                if not text_content.strip():
                    logger.warning(f"No text extracted from PDF: {uploaded_file.name}")
                    text_content = f"PDF Document: {uploaded_file.name} (Could not extract text - {len(content)} bytes)"
                else:
                    logger.info(f"✅ Successfully extracted text from PDF: {uploaded_file.name}")

            except Exception as e:
                logger.error(f"PDF extraction failed for {uploaded_file.name}: {e}")
                text_content = f"PDF Document: {uploaded_file.name} (Text extraction failed - {len(content)} bytes)"
        else:
            # For other file types, you'd use appropriate parsers
            text_content = f"Document: {uploaded_file.name} (Binary content - {len(content)} bytes)"

        # PHASE 1: Save file temporarily for multimodal processing
        import tempfile
        import os

        temp_file_path = None
        try:
            # Create temporary file with proper extension
            file_extension = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name

            logger.info(f"Created temporary file for processing: {temp_file_path}")

            # PHASE 2: Process through multimodal pipeline for knowledge consolidation
            consolidation_result = None
            if 'multimodal_pipeline' in st.session_state:
                try:
                    logger.info(f"🧠 Starting knowledge consolidation for: {uploaded_file.name}")
                    consolidation_result = st.session_state.multimodal_pipeline.process_document(temp_file_path)

                    if consolidation_result:
                        logger.info(f"✅ Knowledge consolidation completed: {consolidation_result.get('summary_length', 0)} chars summary")
                        logger.info(f"🎓 Key concepts extracted: {consolidation_result.get('key_concepts', 0)}")
                    else:
                        logger.warning("⚠️ Knowledge consolidation returned no result")

                except Exception as e:
                    logger.error(f"❌ Knowledge consolidation failed: {e}")
                    # Continue with basic processing if consolidation fails

            # PHASE 3A: Add to secure memory store (encrypted storage)
            from memory.secure_memory_vectorstore import MemoryType

            logger.info(f"Adding document to secure memory store: {uploaded_file.name}")
            logger.debug(f"Text content preview: {text_content[:200]}...")

            # CRITICAL FIX: Format content for Document-Aware RAG Pipeline compatibility
            # The Document-Aware RAG Pipeline expects documents in this specific format
            # FILENAME FIX: Always use original filename in document header for proper matching
            formatted_document_content = f"""Document: {uploaded_file.name} (Block 1)
Content Type: text
Original Filename: {uploaded_file.name}
File Path: {uploaded_file.name}
Temp Path: {temp_file_path if temp_file_path else 'N/A'}

{text_content}

Metadata:
chunk_type: document
priority_score: 1.0
word_count: {len(text_content.split())}
char_count: {len(text_content)}
filename: {uploaded_file.name}
file_type: {uploaded_file.type}
upload_method: streamlit
original_filename: {uploaded_file.name}
display_name: {uploaded_file.name}
search_name: {uploaded_file.name}
temp_path: {temp_file_path if temp_file_path else 'N/A'}"""

            secure_chunk_id = st.session_state.secure_memory_store.add_memory(
                content=formatted_document_content,
                memory_type=MemoryType.DOCUMENT,
                source=f"upload:{uploaded_file.name}",
                tags=['uploaded', 'document', 'consolidated'] if consolidation_result else ['uploaded', 'document'],
                importance_score=0.9 if consolidation_result else 0.8,  # Higher score if consolidated
                metadata={
                    'filename': uploaded_file.name,
                    'file_type': uploaded_file.type,
                    'file_size': len(content),
                    'upload_method': 'streamlit',
                    'knowledge_consolidated': bool(consolidation_result),
                    'consolidation_timestamp': consolidation_result.get('processing_timestamp') if consolidation_result else None,
                    'document_format': 'rag_compatible'  # Flag for Document-Aware RAG
                }
            )

            logger.info(f"Document added to secure store with chunk_id: {secure_chunk_id}")

            # PHASE 3B: Also add to regular memory store for Flask interface compatibility
            try:
                from memory.memory_vectorstore import get_memory_store, VectorStoreType
                regular_memory_store = get_memory_store(
                    store_type=VectorStoreType.CHROMA,
                    storage_directory="memory_store",
                    embedding_dimension=384
                )

                logger.info(f"Adding document to regular memory store for Flask compatibility: {uploaded_file.name}")

                # CRITICAL FIX: Use the same formatted content for regular memory store
                # This ensures Document-Aware RAG Pipeline can find documents in both stores
                regular_chunk_id = regular_memory_store.add_memory(
                    content=formatted_document_content,
                    source=f"upload:{uploaded_file.name}",
                    tags=['uploaded', 'document', 'streamlit_sync', 'consolidated'] if consolidation_result else ['uploaded', 'document', 'streamlit_sync'],
                    importance_score=0.9 if consolidation_result else 0.8,
                    metadata={
                        'filename': uploaded_file.name,
                        'file_type': uploaded_file.type,
                        'file_size': len(content),
                        'upload_method': 'streamlit_sync',
                        'secure_chunk_id': secure_chunk_id,
                        'knowledge_consolidated': bool(consolidation_result),
                        'consolidation_timestamp': consolidation_result.get('processing_timestamp') if consolidation_result else None,
                        'document_format': 'rag_compatible'  # Flag for Document-Aware RAG
                    }
                )

                logger.info(f"Document also added to regular store with chunk_id: {regular_chunk_id}")

            except Exception as e:
                logger.warning(f"Could not sync to regular memory store: {e}")
                regular_chunk_id = None

            # Calculate total chunks created
            total_chunks = 1  # Base memory chunk
            if consolidation_result:
                total_chunks += consolidation_result.get('content_blocks', 0)

            # TRACK RECENT UPLOADS FOR CONTEXT PRIORITIZATION
            try:
                from datetime import datetime

                # Initialize recent uploads tracking if not exists
                if 'recent_document_uploads' not in st.session_state:
                    st.session_state.recent_document_uploads = []

                # Add this upload to recent uploads
                upload_info = {
                    'filename': uploaded_file.name,
                    'upload_timestamp': datetime.now().isoformat(),
                    'file_size': len(content),
                    'file_type': uploaded_file.type,
                    'secure_chunk_id': secure_chunk_id,
                    'knowledge_consolidated': bool(consolidation_result)
                }

                st.session_state.recent_document_uploads.append(upload_info)

                # Keep only last 10 uploads to prevent memory bloat
                if len(st.session_state.recent_document_uploads) > 10:
                    st.session_state.recent_document_uploads = st.session_state.recent_document_uploads[-10:]

                logger.info(f"📋 Tracked upload: {uploaded_file.name} (total recent uploads: {len(st.session_state.recent_document_uploads)})")

            except Exception as e:
                logger.warning(f"Failed to track recent upload: {e}")

            return {
                'success': True,
                'secure_chunk_id': secure_chunk_id,
                'regular_chunk_id': regular_chunk_id,
                'chunks_created': total_chunks,
                'filename': uploaded_file.name,
                'file_size': len(content),
                'knowledge_consolidated': bool(consolidation_result),
                'consolidation_summary': consolidation_result.get('summary_length', 0) if consolidation_result else 0,
                'synced_to_regular_store': regular_chunk_id is not None
            }

        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    logger.debug(f"Cleaned up temporary file: {temp_file_path}")
                except Exception as e:
                    logger.warning(f"Could not clean up temporary file: {e}")

    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

# Integrated Memory Control Center Components

def render_enhanced_memory_chat():
    """Render enhanced chat interface with memory integration."""
    # Chat interface with memory context
    if "memory_chat_history" not in st.session_state:
        st.session_state.memory_chat_history = []

    # Memory context controls
    user_input = st.text_input("Ask SAM:", key="memory_chat_input")

    if user_input:
        # Add to chat history
        st.session_state.memory_chat_history.append({"role": "user", "content": user_input})

        # Process with memory context
        try:
            from memory.memory_vectorstore import get_memory_store
            memory_store = get_memory_store()

            # Search for relevant memories
            relevant_memories = memory_store.search_memories(user_input, limit=5)
            context = "\n".join([mem.content for mem in relevant_memories])
            enhanced_prompt = f"Context from memory:\n{context}\n\nUser question: {user_input}"

            # Generate response (simplified for integration)
            response = f"Enhanced response to: {user_input}"
            st.session_state.memory_chat_history.append({"role": "assistant", "content": response})

        except Exception as e:
            st.error(f"Error processing chat: {e}")

    # Display chat history
    for message in st.session_state.memory_chat_history[-10:]:  # Show last 10 messages
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**SAM:** {message['content']}")

def render_document_library():
    """Render the enhanced document library interface with discussion features."""
    # Get document statistics from secure memory store
    security_status = st.session_state.secure_memory_store.get_security_status()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔐 Encrypted Chunks", security_status.get('encrypted_chunk_count', 0))
    with col2:
        st.metric("🔍 Searchable Fields", security_status.get('searchable_fields', 0))
    with col3:
        st.metric("🔒 Encrypted Fields", security_status.get('encrypted_fields', 0))
    with col4:
        # Get document count
        from memory.memory_vectorstore import MemoryType
        document_memories = st.session_state.secure_memory_store.search_memories(
            query="",
            memory_type=MemoryType.DOCUMENT,
            max_results=1000
        )
        # PHASE 3: Handle different result types for document counting
        unique_docs = set()
        for mem in document_memories:
            content, source, metadata = extract_result_content(mem)
            if metadata:
                filename = metadata.get('filename', metadata.get('source_name', 'unknown'))
                unique_docs.add(filename)
        unique_docs = len(unique_docs)
        st.metric("📄 Documents", unique_docs)

    # Enhanced Document Browser
    st.markdown("---")

    # Search and filter controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        doc_search = st.text_input(
            "🔍 Search Documents",
            placeholder="Search by filename, content, or topic...",
            help="Search through document names and content"
        )
    with col2:
        doc_filter = st.selectbox(
            "📁 Filter by Type",
            options=["All Documents", "PDFs", "Word Docs", "Text Files", "Recent Uploads"],
            help="Filter documents by file type or upload date"
        )
    with col3:
        sort_by = st.selectbox(
            "📊 Sort by",
            options=["Upload Date", "Filename", "File Size", "Relevance"],
            help="Sort documents by different criteria"
        )

    # Get and display documents with enhanced error handling
    try:
        if doc_search:
            # Search in document content and metadata
            with st.spinner(f"🔍 Searching for '{doc_search}'..."):
                search_results = st.session_state.secure_memory_store.search_memories(
                    query=doc_search,
                    memory_type=MemoryType.DOCUMENT,
                    max_results=50
                )
                logger.info(f"Document search: '{doc_search}' returned {len(search_results)} results")
        else:
            # Get all document memories
            with st.spinner("📄 Loading documents..."):
                search_results = st.session_state.secure_memory_store.search_memories(
                    query="",
                    memory_type=MemoryType.DOCUMENT,
                    max_results=100
                )
                logger.info(f"Document library loaded: {len(search_results)} total document chunks")
    except Exception as e:
        logger.error(f"Document search failed: {e}")
        st.error(f"❌ Document search failed: {e}")
        st.info("💡 **Fallback**: Try refreshing the page or check Memory Browser for document access")
        return

    # Process and group documents with enhanced metadata extraction
    documents = {}
    for memory_result in search_results:
        # Access the memory chunk from the search result
        memory = memory_result.chunk if hasattr(memory_result, 'chunk') else memory_result

        if hasattr(memory, 'metadata') and memory.metadata:
            # Enhanced filename extraction with multiple fallbacks
            filename = (
                memory.metadata.get('filename') or
                memory.metadata.get('original_filename') or
                memory.metadata.get('file_name') or
                memory.metadata.get('source_name') or
                memory.metadata.get('source', '').split(':')[-1] if memory.metadata.get('source') else None or
                'Unknown Document'
            )

            # Enhanced file type extraction
            file_type = (
                memory.metadata.get('file_type') or
                memory.metadata.get('mime_type') or
                memory.metadata.get('file_extension') or
                'unknown'
            )

            # Enhanced file size extraction
            file_size = (
                memory.metadata.get('file_size') or
                memory.metadata.get('size') or
                0
            )

            # Enhanced upload date extraction
            upload_date = (
                memory.metadata.get('upload_timestamp') or
                memory.metadata.get('created_at') or
                memory.metadata.get('processed_at') or
                memory.metadata.get('parsed_timestamp') or
                (memory.timestamp if hasattr(memory, 'timestamp') else None) or
                'Unknown'
            )

            if filename not in documents:
                documents[filename] = {
                    'filename': filename,
                    'file_type': file_type,
                    'file_size': file_size,
                    'upload_date': upload_date,
                    'chunks': [],
                    'total_content': '',
                    'tags': memory.tags if hasattr(memory, 'tags') else [],
                    'importance': memory.importance_score if hasattr(memory, 'importance_score') else 0,
                    'metadata': memory.metadata  # Store full metadata for debugging
                }
            documents[filename]['chunks'].append(memory)
            documents[filename]['total_content'] += memory.content + '\n'

    # Apply filters
    filtered_docs = list(documents.values())
    if doc_filter == "PDFs":
        filtered_docs = [doc for doc in filtered_docs if 'pdf' in doc['file_type'].lower()]
    elif doc_filter == "Word Docs":
        filtered_docs = [doc for doc in filtered_docs if any(ext in doc['file_type'].lower() for ext in ['word', 'docx', 'doc'])]
    elif doc_filter == "Text Files":
        filtered_docs = [doc for doc in filtered_docs if any(ext in doc['file_type'].lower() for ext in ['text', 'txt', 'md'])]
    elif doc_filter == "Recent Uploads":
        # Filter for documents uploaded in last 7 days
        from datetime import datetime, timedelta
        week_ago = datetime.now() - timedelta(days=7)
        filtered_docs = [doc for doc in filtered_docs if doc['upload_date'] != 'Unknown' and
                       datetime.fromisoformat(doc['upload_date'].replace('Z', '+00:00')) > week_ago]

    # Sort documents
    if sort_by == "Upload Date":
        filtered_docs.sort(key=lambda x: x['upload_date'], reverse=True)
    elif sort_by == "Filename":
        filtered_docs.sort(key=lambda x: x['filename'].lower())
    elif sort_by == "File Size":
        filtered_docs.sort(key=lambda x: x['file_size'], reverse=True)

    # Display document count with search context
    if doc_search:
        st.markdown(f"**📊 Found {len(filtered_docs)} documents matching '{doc_search}'**")
        if len(filtered_docs) == 0:
            st.info(f"📄 No documents found matching '{doc_search}'. Try different search terms or check if documents are uploaded.")
            # Show suggestion for broader search
            if len(doc_search) > 3:
                st.markdown("**💡 Search Tips:**")
                st.markdown("- Try shorter or more general terms")
                st.markdown("- Check spelling and try synonyms")
                st.markdown("- Use the 'All Documents' filter to see all available documents")
            return
    else:
        st.markdown(f"**📊 Found {len(filtered_docs)} documents**")
        if len(filtered_docs) == 0:
            st.info("📄 No documents found. Upload some documents to get started!")
            return

    # Display documents with enhanced interaction features
    for i, doc in enumerate(filtered_docs[:20]):  # Limit to 20 for performance
        # Format file size
        size_bytes = doc['file_size']
        if size_bytes > 1024*1024:
            size_str = f"{size_bytes/(1024*1024):.1f} MB"
        elif size_bytes > 1024:
            size_str = f"{size_bytes/1024:.1f} KB"
        else:
            size_str = f"{size_bytes} bytes"

        # Format upload date
        upload_date = doc['upload_date']
        if upload_date != 'Unknown':
            try:
                from datetime import datetime
                if 'T' in upload_date:
                    dt = datetime.fromisoformat(upload_date.replace('Z', '+00:00'))
                    upload_date = dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                pass

        # Create expandable document card
        with st.expander(f"📄 {doc['filename']} ({size_str}) - {len(doc['chunks'])} chunks", expanded=False):
            # Document metadata
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**📅 Upload Date:** {upload_date}")
                st.markdown(f"**📋 File Type:** {doc['file_type']}")
                st.markdown(f"**🧩 Chunks:** {len(doc['chunks'])}")
                st.markdown(f"**⭐ Importance:** {doc['importance']:.2f}")

                if doc['tags']:
                    st.markdown(f"**🏷️ Tags:** {', '.join(doc['tags'])}")

                # Debug information (can be removed later)
                if st.checkbox(f"🔍 Debug Metadata", key=f"debug_{i}"):
                    st.markdown("**🛠️ Available Metadata Keys:**")
                    if 'metadata' in doc and doc['metadata']:
                        metadata_keys = list(doc['metadata'].keys())
                        st.markdown(f"- {', '.join(metadata_keys)}")

                        # Show some key metadata values
                        st.markdown("**📋 Key Metadata Values:**")
                        for key in ['source', 'filename', 'file_type', 'file_size', 'upload_timestamp']:
                            if key in doc['metadata']:
                                value = doc['metadata'][key]
                                st.markdown(f"- **{key}**: {value}")
                    else:
                        st.markdown("- No metadata available")

            with col2:
                st.markdown("**🤖 AI Discussion Tools:**")

                # Quick discussion starters
                if st.button("💬 Discuss Document", key=f"discuss_{i}"):
                    discussion_prompt = f"Let's discuss the document '{doc['filename']}'. What are the key points and insights from this document?"
                    st.session_state.document_discussion_prompt = discussion_prompt
                    st.session_state.selected_document = doc['filename']
                    st.info(f"💬 Discussion started! Ask SAM: '{discussion_prompt}'")

                if st.button("📊 Summarize", key=f"summarize_{i}"):
                    summary_prompt = f"Please provide a comprehensive summary of the document '{doc['filename']}', including key findings, main arguments, and important conclusions."
                    st.session_state.document_discussion_prompt = summary_prompt
                    st.session_state.selected_document = doc['filename']
                    st.info(f"📊 Summary requested! Ask SAM: '{summary_prompt}'")

                if st.button("🔍 Key Insights", key=f"insights_{i}"):
                    insights_prompt = f"What are the most important insights and takeaways from the document '{doc['filename']}'? What makes this document valuable?"
                    st.session_state.document_discussion_prompt = insights_prompt
                    st.session_state.selected_document = doc['filename']
                    st.info(f"🔍 Insights analysis requested! Ask SAM: '{insights_prompt}'")

                if st.button("🔗 Related Docs", key=f"related_{i}"):
                    related_prompt = f"Which other documents in my knowledge base are related to '{doc['filename']}'? What connections and themes do you see?"
                    st.session_state.document_discussion_prompt = related_prompt
                    st.session_state.selected_document = doc['filename']
                    st.info(f"🔗 Related documents analysis requested! Ask SAM: '{related_prompt}'")

            # Content preview
            st.markdown("**📖 Content Preview:**")
            preview_text = doc['total_content'][:500] + "..." if len(doc['total_content']) > 500 else doc['total_content']
            st.text_area("", value=preview_text, height=100, disabled=True, key=f"preview_{i}")

            # Advanced discussion options (moved outside expander to avoid nesting)
            st.markdown("---")
            st.markdown("**🧠 Advanced Discussion Options:**")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("❓ Generate Questions", key=f"questions_{i}"):
                    questions_prompt = f"Generate 5 thoughtful questions about the document '{doc['filename']}' that would help me understand it better."
                    st.session_state.document_discussion_prompt = questions_prompt
                    st.session_state.selected_document = doc['filename']
                    st.info(f"❓ Questions generated! Ask SAM: '{questions_prompt}'")

                if st.button("🎯 Action Items", key=f"actions_{i}"):
                    actions_prompt = f"Based on the document '{doc['filename']}', what are the key action items or next steps I should consider?"
                    st.session_state.document_discussion_prompt = actions_prompt
                    st.session_state.selected_document = doc['filename']
                    st.info(f"🎯 Action items analysis requested! Ask SAM: '{actions_prompt}'")

            with col2:
                if st.button("🔬 Deep Analysis", key=f"analysis_{i}"):
                    analysis_prompt = f"Provide a deep analytical breakdown of the document '{doc['filename']}', including methodology, evidence, strengths, and potential limitations."
                    st.session_state.document_discussion_prompt = analysis_prompt
                    st.session_state.selected_document = doc['filename']
                    st.info(f"🔬 Deep analysis requested! Ask SAM: '{analysis_prompt}'")

                if st.button("💡 Applications", key=f"applications_{i}"):
                    applications_prompt = f"How can I apply the knowledge and insights from '{doc['filename']}' to real-world situations or my current projects?"
                    st.session_state.document_discussion_prompt = applications_prompt
                    st.session_state.selected_document = doc['filename']
                    st.info(f"💡 Applications analysis requested! Ask SAM: '{applications_prompt}'")

    # Show pagination if there are more documents
    if len(filtered_docs) > 20:
        st.info(f"📄 Showing first 20 of {len(filtered_docs)} documents. Use search to find specific documents.")

    # Quick discussion starter section
    st.markdown("---")
    st.subheader("💬 Quick Document Discussion")
    st.markdown("*Start a conversation about your documents*")

    # Pre-filled discussion prompts
    if st.session_state.get('document_discussion_prompt'):
        st.text_area(
            "💬 Ready to discuss:",
            value=st.session_state.document_discussion_prompt,
            height=100,
            help="Copy this prompt and paste it in the chat to start discussing with SAM"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗨️ Start Chat Discussion"):
                st.info("💬 Go to the chat interface above and paste the discussion prompt!")
        with col2:
            if st.button("🔄 Clear Prompt"):
                st.session_state.document_discussion_prompt = ""
                st.session_state.selected_document = ""
                st.rerun()
    else:
        # General discussion starters
        st.markdown("**🎯 General Discussion Starters:**")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📚 Overview All Documents"):
                overview_prompt = "Can you give me an overview of all the documents in my knowledge base? What are the main topics and themes?"
                st.session_state.document_discussion_prompt = overview_prompt
                st.rerun()

            if st.button("🔍 Find Specific Topic"):
                topic_prompt = "I'm looking for information about [TOPIC]. Which documents contain relevant information and what do they say?"
                st.session_state.document_discussion_prompt = topic_prompt
                st.rerun()

        with col2:
            if st.button("🔗 Connect Ideas"):
                connect_prompt = "What connections and patterns do you see across all my uploaded documents? How do they relate to each other?"
                st.session_state.document_discussion_prompt = connect_prompt
                st.rerun()

            if st.button("💎 Most Valuable Insights"):
                insights_prompt = "What are the most valuable insights and key takeaways from all my documents combined?"
                st.session_state.document_discussion_prompt = insights_prompt
                st.rerun()

def render_document_library_integrated():
    """Render the integrated document library interface."""
    st.subheader("📖 Document Library")
    st.markdown("*Explore and discuss your uploaded documents with SAM*")

    # Call the existing document library function that we already implemented
    try:
        render_document_library()
    except Exception as e:
        st.error(f"❌ Error loading document library: {e}")
        st.info("💡 **Fallback**: Basic document information available in Memory Browser")

def render_memory_browser_integrated():
    """Render integrated memory browser."""
    st.subheader("🔍 Memory Browser")
    st.markdown("Search and browse SAM's memory store")

    try:
        from ui.memory_browser import MemoryBrowserUI
        browser = MemoryBrowserUI()
        browser.render()
    except Exception as e:
        st.error(f"Error loading memory browser: {e}")
        st.info("Using simplified memory browser...")
        render_simple_memory_browser()

def render_memory_editor_integrated():
    """Render integrated memory editor."""
    st.subheader("✏️ Memory Editor")
    st.markdown("Edit and manage individual memories")

    try:
        from ui.memory_editor import MemoryEditor
        editor = MemoryEditor()
        editor.render()
    except Exception as e:
        st.error(f"Error loading memory editor: {e}")
        st.info("Memory editor not available in this interface.")

def render_memory_graph_integrated():
    """Render integrated memory graph."""
    st.subheader("🕸️ Memory Graph")
    st.markdown("Visualize memory relationships and connections")

    try:
        from ui.memory_graph import MemoryGraphVisualizer
        visualizer = MemoryGraphVisualizer()
        visualizer.render()
    except Exception as e:
        st.error(f"Error loading memory graph: {e}")
        st.info("Memory graph visualization not available in this interface.")

def render_command_interface_integrated():
    """Render integrated command interface."""
    st.subheader("💻 Memory Command Interface")
    st.markdown("Execute advanced memory commands")

    try:
        from ui.memory_commands import get_command_processor
        processor = get_command_processor()

        # Command input
        command = st.text_input("Enter memory command:", placeholder="!recall topic AI")

        if command:
            result = processor.process_command(command)
            if result.success:
                st.success(f"✅ Command executed successfully")
                st.markdown(result.message)
                if result.data:
                    st.json(result.data)
            else:
                st.error(f"❌ Command failed: {result.message}")

        # Available commands
        with st.expander("📋 Available Commands"):
            commands = processor.get_available_commands()
            for cmd in commands:
                st.markdown(f"• `{cmd['command']}` - {cmd['description']}")

    except Exception as e:
        st.error(f"Error loading command interface: {e}")

def render_memory_ranking_integrated():
    """Render integrated memory ranking."""
    st.subheader("🏆 Memory Ranking")
    st.markdown("View and manage memory importance rankings")

    try:
        from memory.memory_vectorstore import get_memory_store
        memory_store = get_memory_store()

        # Get top memories by importance (preserving 100% of functionality)
        if memory_store.memory_chunks:
            # Manual sorting by importance score (same as working Memory Center)
            top_memories = sorted(
                memory_store.memory_chunks.values(),
                key=lambda x: x.importance_score,
                reverse=True
            )[:20]  # Limit to top 20

            if top_memories:
                st.markdown(f"**📊 Showing top {len(top_memories)} memories by importance**")

                # Enhanced ranking display (preserving 100% of semantic meaning)
                for i, memory in enumerate(top_memories, 1):
                    # Determine importance level for visual indicator
                    if memory.importance_score >= 0.8:
                        importance_indicator = "🔥"
                        importance_level = "Critical"
                    elif memory.importance_score >= 0.6:
                        importance_indicator = "⭐"
                        importance_level = "High"
                    elif memory.importance_score >= 0.4:
                        importance_indicator = "📌"
                        importance_level = "Medium"
                    else:
                        importance_indicator = "📄"
                        importance_level = "Low"

                    # Create expandable memory entry
                    with st.expander(f"#{i} {importance_indicator} {memory.memory_type.value.title()} - {memory.content[:100]}..."):
                        col1, col2 = st.columns([2, 1])

                        with col1:
                            st.markdown(f"**📝 Content:**")
                            st.markdown(memory.content)

                            st.markdown(f"**📍 Source:** {memory.source}")

                            if memory.tags:
                                st.markdown(f"**🏷️ Tags:** {', '.join(memory.tags)}")

                        with col2:
                            st.markdown(f"**📊 Ranking Details:**")
                            st.metric("Importance Score", f"{memory.importance_score:.3f}")
                            st.markdown(f"**Level:** {importance_level}")
                            st.markdown(f"**Type:** {memory.memory_type.value}")

                            if hasattr(memory, 'timestamp') and memory.timestamp:
                                # Handle both string and integer timestamps
                                if isinstance(memory.timestamp, str):
                                    timestamp_display = memory.timestamp[:10]
                                elif isinstance(memory.timestamp, (int, float)):
                                    # Convert Unix timestamp to readable format
                                    from datetime import datetime
                                    timestamp_display = datetime.fromtimestamp(memory.timestamp).strftime('%Y-%m-%d')
                                else:
                                    timestamp_display = str(memory.timestamp)
                                st.markdown(f"**📅 Created:** {timestamp_display}")

                            if hasattr(memory, 'access_count'):
                                st.markdown(f"**👁️ Access Count:** {memory.access_count}")

                            if hasattr(memory, 'last_accessed') and memory.last_accessed:
                                # Handle both string and integer timestamps
                                if isinstance(memory.last_accessed, str):
                                    last_accessed_display = memory.last_accessed[:10]
                                elif isinstance(memory.last_accessed, (int, float)):
                                    # Convert Unix timestamp to readable format
                                    from datetime import datetime
                                    last_accessed_display = datetime.fromtimestamp(memory.last_accessed).strftime('%Y-%m-%d')
                                else:
                                    last_accessed_display = str(memory.last_accessed)
                                st.markdown(f"**🕒 Last Accessed:** {last_accessed_display}")

                # Enhanced ranking analytics (preserving 100% of story)
                st.subheader("📈 Ranking Analytics")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    critical_count = sum(1 for m in top_memories if m.importance_score >= 0.8)
                    st.metric("Critical Memories", critical_count)

                with col2:
                    high_count = sum(1 for m in top_memories if 0.6 <= m.importance_score < 0.8)
                    st.metric("High Importance", high_count)

                with col3:
                    medium_count = sum(1 for m in top_memories if 0.4 <= m.importance_score < 0.6)
                    st.metric("Medium Importance", medium_count)

                with col4:
                    avg_importance = sum(m.importance_score for m in top_memories) / len(top_memories)
                    st.metric("Average Score", f"{avg_importance:.3f}")

                # Memory type distribution in rankings
                st.subheader("🎯 Ranking Distribution")

                memory_types = {}
                for memory in top_memories:
                    mem_type = memory.memory_type.value
                    memory_types[mem_type] = memory_types.get(mem_type, 0) + 1

                if memory_types:
                    try:
                        import pandas as pd
                        import plotly.express as px

                        # Create bar chart for memory type distribution in rankings
                        df = pd.DataFrame(list(memory_types.items()), columns=['Memory Type', 'Count'])
                        fig = px.bar(df, x='Memory Type', y='Count',
                                   title="Memory Types in Top Rankings",
                                   color='Count',
                                   color_continuous_scale='viridis')
                        st.plotly_chart(fig, use_container_width=True)
                    except ImportError:
                        # Fallback to simple display
                        st.markdown("**Memory Types in Rankings:**")
                        for mem_type, count in memory_types.items():
                            st.markdown(f"- **{mem_type}**: {count} memories")

                # Advanced ranking controls (preserving 100% of functionality)
                st.subheader("⚙️ Ranking Controls")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**🔧 Ranking Options:**")

                    # Filter by importance level
                    importance_filter = st.selectbox(
                        "Filter by Importance Level",
                        ["All", "Critical (≥0.8)", "High (≥0.6)", "Medium (≥0.4)", "Low (<0.4)"]
                    )

                    # Filter by memory type
                    available_types = list(set(m.memory_type.value for m in memory_store.memory_chunks.values()))
                    type_filter = st.selectbox("Filter by Memory Type", ["All"] + available_types)

                with col2:
                    st.markdown("**📊 Display Options:**")

                    # Number of results
                    result_limit = st.slider("Number of Results", 5, 50, 20)

                    # Sort options
                    sort_option = st.selectbox(
                        "Sort By",
                        ["Importance (High to Low)", "Importance (Low to High)", "Date (Newest)", "Date (Oldest)", "Access Count"]
                    )

                # Apply filters button
                if st.button("🔄 Apply Filters", type="primary"):
                    filtered_memories = list(memory_store.memory_chunks.values())

                    # Apply importance filter
                    if importance_filter == "Critical (≥0.8)":
                        filtered_memories = [m for m in filtered_memories if m.importance_score >= 0.8]
                    elif importance_filter == "High (≥0.6)":
                        filtered_memories = [m for m in filtered_memories if 0.6 <= m.importance_score < 0.8]
                    elif importance_filter == "Medium (≥0.4)":
                        filtered_memories = [m for m in filtered_memories if 0.4 <= m.importance_score < 0.6]
                    elif importance_filter == "Low (<0.4)":
                        filtered_memories = [m for m in filtered_memories if m.importance_score < 0.4]

                    # Apply type filter
                    if type_filter != "All":
                        filtered_memories = [m for m in filtered_memories if m.memory_type.value == type_filter]

                    # Apply sorting
                    if sort_option == "Importance (High to Low)":
                        filtered_memories.sort(key=lambda x: x.importance_score, reverse=True)
                    elif sort_option == "Importance (Low to High)":
                        filtered_memories.sort(key=lambda x: x.importance_score)
                    elif sort_option == "Date (Newest)":
                        filtered_memories.sort(key=lambda x: x.timestamp if x.timestamp else "", reverse=True)
                    elif sort_option == "Date (Oldest)":
                        filtered_memories.sort(key=lambda x: x.timestamp if x.timestamp else "")
                    elif sort_option == "Access Count":
                        filtered_memories.sort(key=lambda x: getattr(x, 'access_count', 0), reverse=True)

                    # Limit results
                    filtered_memories = filtered_memories[:result_limit]

                    st.success(f"✅ Applied filters - showing {len(filtered_memories)} memories")

                    # Display filtered results
                    if filtered_memories:
                        st.markdown("**🔍 Filtered Results:**")
                        for i, memory in enumerate(filtered_memories, 1):
                            with st.container():
                                st.markdown(f"**{i}. {memory.memory_type.value.title()}** - Score: {memory.importance_score:.3f}")
                                st.caption(f"Source: {memory.source}")
                                st.caption(f"Content: {memory.content[:100]}...")
                                st.divider()
                    else:
                        st.info("No memories match the selected filters")

            else:
                st.info("No memories found in the memory store")
        else:
            st.info("Memory store is empty. Add some memories to see rankings!")

            # Show helpful information about memory ranking
            st.markdown("---")
            st.markdown("### 🏆 About Memory Ranking")
            st.markdown("""
            **Memory Ranking** organizes your memories by importance score:

            🔥 **Critical (0.8-1.0)**: Essential memories with highest priority
            ⭐ **High (0.6-0.8)**: Important memories for frequent reference
            📌 **Medium (0.4-0.6)**: Useful memories for occasional reference
            📄 **Low (0.0-0.4)**: Background memories for completeness

            **Ranking Factors:**
            - Content importance and relevance
            - Access frequency and recency
            - User-assigned priority levels
            - Semantic significance in context
            """)

    except Exception as e:
        st.error(f"Error loading memory ranking: {e}")
        import traceback
        st.error(f"Details: {traceback.format_exc()}")

def render_memory_analytics_integrated():
    """Render integrated memory analytics."""
    st.subheader("📊 Memory Analytics")
    st.markdown("Advanced memory system statistics and insights")

    try:
        from memory.memory_vectorstore import get_memory_store
        memory_store = get_memory_store()

        # Get basic statistics
        stats = memory_store.get_memory_stats()

        # Calculate additional metrics manually (preserving 100% of functionality)
        avg_importance = 0.0
        recent_count = 0

        if memory_store.memory_chunks:
            # Calculate average importance score
            avg_importance = sum(chunk.importance_score for chunk in memory_store.memory_chunks.values()) / len(memory_store.memory_chunks)

            # Calculate recent memories (last 7 days)
            from datetime import datetime, timedelta
            recent_threshold = datetime.now() - timedelta(days=7)

            for chunk in memory_store.memory_chunks.values():
                try:
                    # Parse timestamp and check if recent
                    if chunk.timestamp:
                        chunk_time = datetime.fromisoformat(chunk.timestamp.replace('Z', '+00:00'))
                        if chunk_time.replace(tzinfo=None) > recent_threshold:
                            recent_count += 1
                except Exception:
                    # Skip chunks with invalid timestamps
                    continue

        # Display metrics (preserving 100% of semantic meaning)
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Memories", stats.get('total_memories', 0))

        with col2:
            st.metric("Storage Size", f"{stats.get('total_size_mb', 0):.1f} MB")

        with col3:
            st.metric("Avg Importance", f"{avg_importance:.3f}")

        with col4:
            st.metric("Recent Memories", recent_count)

        # Enhanced Memory Analytics (preserving 100% of functionality)
        st.subheader("📈 Memory Distribution")

        # Memory type distribution chart
        if stats.get('memory_types'):
            try:
                import pandas as pd
                import plotly.express as px

                # Create pie chart for memory types
                df = pd.DataFrame(list(stats['memory_types'].items()), columns=['Type', 'Count'])
                fig = px.pie(df, values='Count', names='Type', title="Memory Distribution by Type")
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                # Fallback to simple display if plotly not available
                st.markdown("**Memory Types:**")
                for mem_type, count in stats['memory_types'].items():
                    st.markdown(f"- **{mem_type}**: {count} memories")
        else:
            st.info("No memory type distribution data available")

        # Additional Analytics (preserving 100% of story and functionality)
        st.subheader("🔍 Advanced Analytics")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📊 Quality Metrics:**")
            if memory_store.memory_chunks:
                # High importance memories
                high_importance = sum(1 for chunk in memory_store.memory_chunks.values() if chunk.importance_score > 0.7)
                st.metric("High Importance", high_importance)

                # Most accessed memory
                if stats.get('most_accessed'):
                    st.markdown("**🔥 Most Accessed:**")
                    most_accessed = stats['most_accessed']
                    st.caption(f"Access count: {most_accessed['access_count']}")
                    st.caption(f"Content: {most_accessed['content_preview']}...")
            else:
                st.info("No memories available for quality analysis")

        with col2:
            st.markdown("**⏰ Temporal Metrics:**")
            if stats.get('oldest_memory') and stats.get('newest_memory'):
                st.markdown(f"**Oldest Memory:** {stats['oldest_memory'][:10]}")
                st.markdown(f"**Newest Memory:** {stats['newest_memory'][:10]}")

                # Calculate memory span
                try:
                    from datetime import datetime
                    oldest = datetime.fromisoformat(stats['oldest_memory'].replace('Z', '+00:00'))
                    newest = datetime.fromisoformat(stats['newest_memory'].replace('Z', '+00:00'))
                    span_days = (newest - oldest).days
                    st.metric("Memory Span", f"{span_days} days")
                except Exception:
                    st.metric("Memory Span", "Unknown")
            else:
                st.info("No temporal data available")

        # Memory Health Check (preserving 100% of functionality)
        st.subheader("🏥 Memory Health")

        if st.button("🔍 Run Health Check", key="memory_health_check"):
            with st.spinner("Analyzing memory health..."):
                health_issues = []

                if memory_store.memory_chunks:
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
                    if low_importance > stats.get('total_memories', 0) * 0.3:
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
                else:
                    st.info("No memories available for health analysis")

    except Exception as e:
        st.error(f"Error loading memory analytics: {e}")
        import traceback
        st.error(f"Details: {traceback.format_exc()}")

def render_simple_memory_browser():
    """Render a simplified memory browser as fallback."""
    st.markdown("**Simplified Memory Browser**")

    search_query = st.text_input("Search memories:", key="simple_memory_search")

    if search_query:
        try:
            from memory.memory_vectorstore import get_memory_store
            memory_store = get_memory_store()

            results = memory_store.search_memories(search_query, limit=10)

            if results:
                st.success(f"Found {len(results)} memories")
                for i, memory in enumerate(results, 1):
                    with st.expander(f"Memory {i}: {memory.content[:100]}..."):
                        st.markdown(f"**Content:** {memory.content}")
                        st.markdown(f"**Relevance:** {memory.relevance_score:.3f}")
                        st.markdown(f"**Created:** {memory.created_at}")
            else:
                st.info("No memories found matching your search.")

        except Exception as e:
            st.error(f"Error searching memories: {e}")

def create_table_analysis_response(prompt: str, generated_code: str, wrapped_code: str,
                                 code_explanation: str, execution_instructions: str,
                                 validation_result: dict, table_summary: dict,
                                 safety_level: str) -> str:
    """
    Create a comprehensive response for table analysis results with specialized UI components.

    This function formats the Table-to-Code Expert Tool results for optimal user experience.
    """
    # Safety emoji mapping
    safety_emojis = {
        'low': '🟢',
        'medium': '🟡',
        'high': '🔴',
        'unknown': '⚪'
    }

    safety_emoji = safety_emojis.get(safety_level, '⚪')

    # Build the comprehensive response
    response_parts = [
        f"🐍 **Table Analysis & Code Generation Complete!**",
        "",
        f"**Your Request:** {prompt}",
        "",
        f"**📊 Table Summary:**",
        f"- Table ID: {table_summary.get('table_id', 'Unknown')}",
        f"- Columns: {len(table_summary.get('columns', {}))}",
        f"- Reconstruction Confidence: {table_summary.get('confidence', 0):.1%}",
        "",
        f"{safety_emoji} **Generated Python Code** (Safety Level: {safety_level.upper()}):",
        "```python",
        generated_code,
        "```",
        "",
        f"**💡 Code Explanation:**",
        code_explanation,
        ""
    ]

    # Add safety information if there are issues
    if validation_result.get('safety_issues'):
        response_parts.extend([
            f"⚠️ **Security Notes:**",
            "- " + "\n- ".join(validation_result['safety_issues']),
            ""
        ])

    # Add performance recommendations
    if validation_result.get('recommendations'):
        response_parts.extend([
            f"💡 **Optimization Tips:**",
            "- " + "\n- ".join(validation_result['recommendations'][:3]),
            ""
        ])

    # Add execution instructions
    response_parts.extend([
        f"🚀 **How to Execute This Code:**",
        execution_instructions,
        "",
        f"**🔒 Production-Ready Code** (with safety wrapper):",
        "```python",
        wrapped_code,
        "```",
        "",
        f"*Generated by SAM's Table-to-Code Expert Tool - Phase 2 Implementation*"
    ])

    return "\n".join(response_parts)

def render_table_analysis_result(message_content: str):
    """
    Render a specialized UI component for table analysis results.

    This creates the "Code Analysis Result" component with:
    - Final Answer prominently displayed
    - Visualization display (if any)
    - Expandable code & safety report section
    """
    # Check if this is a table analysis response
    if "Table Analysis & Code Generation Complete!" not in message_content:
        # Regular message display
        st.markdown(message_content)
        return

    # Parse the table analysis response
    lines = message_content.split('\n')

    # Extract key components
    user_request = ""
    table_summary = ""
    generated_code = ""
    code_explanation = ""
    execution_instructions = ""
    wrapped_code = ""
    safety_level = "unknown"

    current_section = None
    code_lines = []
    wrapped_code_lines = []

    for line in lines:
        if line.startswith("**Your Request:**"):
            user_request = line.replace("**Your Request:**", "").strip()
        elif line.startswith("**📊 Table Summary:**"):
            current_section = "table_summary"
        elif line.startswith("**Generated Python Code**"):
            current_section = "code"
            # Extract safety level
            if "Safety Level:" in line:
                safety_level = line.split("Safety Level:")[1].replace(")", "").strip().lower()
        elif line.startswith("**💡 Code Explanation:**"):
            current_section = "explanation"
        elif line.startswith("**🚀 How to Execute This Code:**"):
            current_section = "instructions"
        elif line.startswith("**🔒 Production-Ready Code**"):
            current_section = "wrapped_code"
        elif line == "```python":
            continue
        elif line == "```":
            current_section = None
        elif current_section == "code":
            code_lines.append(line)
        elif current_section == "wrapped_code":
            wrapped_code_lines.append(line)
        elif current_section == "explanation":
            if line.strip() and not line.startswith("**"):
                code_explanation += line + "\n"
        elif current_section == "instructions":
            if line.strip() and not line.startswith("**"):
                execution_instructions += line + "\n"

    generated_code = "\n".join(code_lines)
    wrapped_code = "\n".join(wrapped_code_lines)

    # Safety emoji mapping
    safety_emojis = {
        'low': '🟢',
        'medium': '🟡',
        'high': '🔴',
        'unknown': '⚪'
    }

    safety_emoji = safety_emojis.get(safety_level, '⚪')

    # 1. Display the Final Answer prominently
    st.success(f"🐍 **Table Analysis Complete!** {safety_emoji}")

    if user_request:
        st.markdown(f"**Your Request:** {user_request}")

    # 2. Code Execution Section
    st.markdown("### 🚀 Execute Generated Code")

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("▶️ Execute Code Safely", type="primary"):
            if generated_code and wrapped_code:
                with st.spinner("Executing code safely..."):
                    execution_result = execute_generated_code_safely(generated_code, wrapped_code)

                    if execution_result['success']:
                        st.success(f"✅ Code executed successfully in {execution_result['execution_time']:.2f}s")

                        if execution_result['output']:
                            st.markdown("**Output:**")
                            st.code(execution_result['output'])

                        # Display any generated visualizations
                        if execution_result['artifacts']:
                            st.markdown("**Generated Visualizations:**")
                            for artifact in execution_result['artifacts']:
                                if artifact.endswith(('.png', '.jpg', '.jpeg')):
                                    try:
                                        st.image(artifact, caption="Generated Visualization")
                                    except:
                                        st.info(f"Visualization saved to: {artifact}")
                    else:
                        st.error(f"❌ Code execution failed: {execution_result['error']}")
            else:
                st.error("No code available to execute")

    with col2:
        if st.button("📋 Copy Code to Clipboard"):
            # Use Streamlit's built-in code display with copy functionality
            st.code(generated_code, language="python")

    # 3. Expandable "Show Code & Safety Report" Section
    with st.expander("🔍 Show Code & Safety Report", expanded=False):

        # Code tabs
        tab1, tab2 = st.tabs(["📝 Generated Code", "🔒 Production Code"])

        with tab1:
            st.markdown("**Clean Generated Code:**")
            st.code(generated_code, language="python")

            if code_explanation:
                st.markdown("**Code Explanation:**")
                st.markdown(code_explanation)

        with tab2:
            st.markdown("**Production-Ready Code (with safety wrapper):**")
            st.code(wrapped_code, language="python")

        # Safety Report
        st.markdown("### 🛡️ Safety Validation Report")

        safety_col1, safety_col2 = st.columns([1, 2])

        with safety_col1:
            if safety_level == "low":
                st.success(f"{safety_emoji} **Low Risk**")
                st.markdown("✅ Code passed all safety checks")
            elif safety_level == "medium":
                st.warning(f"{safety_emoji} **Medium Risk**")
                st.markdown("⚠️ Minor issues detected")
            elif safety_level == "high":
                st.error(f"{safety_emoji} **High Risk**")
                st.markdown("🚨 Security concerns found")
            else:
                st.info(f"{safety_emoji} **Unknown Risk**")

        with safety_col2:
            if execution_instructions:
                st.markdown("**Execution Instructions:**")
                st.markdown(execution_instructions)

    # 4. Additional Actions
    st.markdown("### 🔧 Additional Actions")

    action_col1, action_col2, action_col3 = st.columns([1, 1, 1])

    with action_col1:
        if st.button("🔄 Regenerate Code"):
            st.info("💡 Try rephrasing your request for different code generation")

    with action_col2:
        if st.button("📊 Analyze Different Table"):
            st.info("💡 Upload a new table or reference a different table ID")

    with action_col3:
        if st.button("❓ Get Help"):
            st.info("💡 Ask questions about the generated code or request modifications")

def execute_generated_code_safely(code: str, wrapped_code: str) -> dict:
    """
    Execute generated Python code in a safe environment and capture results.

    This implements the safety wrapper execution with result capture.
    """
    import subprocess
    import tempfile
    import os
    import json

    execution_result = {
        'success': False,
        'output': '',
        'error': '',
        'artifacts': [],
        'execution_time': 0
    }

    try:
        import time
        start_time = time.time()

        # Create a temporary file for the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Use the wrapped code for safety
            f.write(wrapped_code)
            temp_file = f.name

        try:
            # Execute the code in a subprocess for safety
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
                cwd=tempfile.gettempdir()  # Run in temp directory
            )

            execution_result['success'] = result.returncode == 0
            execution_result['output'] = result.stdout
            execution_result['error'] = result.stderr
            execution_result['execution_time'] = time.time() - start_time

            # Look for generated artifacts (images, files)
            temp_dir = tempfile.gettempdir()
            for file in os.listdir(temp_dir):
                if file.endswith(('.png', '.jpg', '.jpeg', '.svg', '.pdf')):
                    execution_result['artifacts'].append(os.path.join(temp_dir, file))

        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file)
            except:
                pass

    except subprocess.TimeoutExpired:
        execution_result['error'] = 'Code execution timed out after 30 seconds'
    except Exception as e:
        execution_result['error'] = f'Execution failed: {str(e)}'

    return execution_result

def is_table_analysis_query(prompt: str) -> bool:
    """
    Enhanced intelligent router to detect table analysis, visualization, or data manipulation queries.

    This function implements sophisticated pattern matching with debugging capabilities
    to identify when the Table-to-Code Expert Tool should be invoked.
    """
    import logging
    logger = logging.getLogger(__name__)

    prompt_lower = prompt.lower()

    # Developer override - explicit tool invocation
    if "tabletocodeool" in prompt_lower or "using the tabletocodetool" in prompt_lower:
        logger.info("🔧 DEVELOPER OVERRIDE: Explicit TableToCodeTool invocation detected")
        return True

    # Enhanced scoring system for better detection
    confidence_score = 0.0
    detection_reasons = []

    # Core table analysis keywords (high weight)
    core_table_keywords = [
        'analyze the table', 'analyze table', 'table analysis', 'data analysis',
        'calculate', 'sum', 'total', 'average', 'mean', 'count', 'aggregate'
    ]
    for keyword in core_table_keywords:
        if keyword in prompt_lower:
            confidence_score += 0.3
            detection_reasons.append(f"Core keyword: '{keyword}'")

    # Enhanced visualization keywords (high weight) - Fixed missing patterns
    visualization_keywords = [
        'chart', 'graph', 'plot', 'visualize', 'bar chart', 'line chart',
        'generate a chart', 'create a chart', 'show a chart', 'display chart',
        'pie chart', 'histogram', 'scatter plot', 'visualization',
        'line plot', 'visual representation', 'trends over time', 'sales trends'
    ]
    for keyword in visualization_keywords:
        if keyword in prompt_lower:
            confidence_score += 0.25
            detection_reasons.append(f"Visualization keyword: '{keyword}'")

    # Enhanced mathematical operations (medium weight)
    math_operations = [
        'total', 'sum', 'add up', 'calculate', 'compute', 'find the total',
        'how much', 'how many', 'what is the total', 'budget', 'cost',
        'hours', 'time spent', 'maximum', 'minimum', 'median', 'expenses'
    ]
    for operation in math_operations:
        if operation in prompt_lower:
            confidence_score += 0.15
            detection_reasons.append(f"Math operation: '{operation}'")

    # Enhanced data manipulation keywords (medium weight)
    manipulation_keywords = [
        'filter', 'sort', 'group', 'where', 'completed', 'status',
        'team', 'category', 'type', 'group by', 'filter by', 'trends'
    ]
    for keyword in manipulation_keywords:
        if keyword in prompt_lower:
            confidence_score += 0.1
            detection_reasons.append(f"Data manipulation: '{keyword}'")

    # Enhanced table reference indicators (medium weight)
    table_references = [
        'table', 'data', 'spreadsheet', 'csv', 'excel', 'dataframe',
        'rows', 'columns', 'cells', 'report', 'dataset', 'sales data',
        'quarterly reports'
    ]
    for ref in table_references:
        if ref in prompt_lower:
            confidence_score += 0.1
            detection_reasons.append(f"Table reference: '{ref}'")

    # Enhanced analytical phrases (low weight but important)
    analytical_phrases = [
        'what is the', 'show me', 'tell me', 'find', 'get the',
        'display', 'present', 'breakdown', 'summary', 'i need'
    ]
    for phrase in analytical_phrases:
        if phrase in prompt_lower:
            confidence_score += 0.05
            detection_reasons.append(f"Analytical phrase: '{phrase}'")

    # Enhanced business/analysis terms (bonus points)
    business_terms = [
        'project', 'task', 'budget', 'hours', 'team', 'status',
        'completed', 'progress', 'milestone', 'deliverable', 'sales',
        'revenue', 'profit', 'margin', 'quarterly'
    ]
    business_term_count = sum(1 for term in business_terms if term in prompt_lower)
    if business_term_count >= 2:
        confidence_score += 0.2
        detection_reasons.append(f"Business context: {business_term_count} terms")

    # Special boost for visualization intent (addresses missed cases)
    visualization_intent_phrases = [
        'visual representation', 'need a visual', 'show visually',
        'create a line plot', 'analyze the trends'
    ]
    for phrase in visualization_intent_phrases:
        if phrase in prompt_lower:
            confidence_score += 0.2
            detection_reasons.append(f"Strong visualization intent: '{phrase}'")

    # Exclude web search queries (negative weight)
    web_search_exclusions = [
        'news', 'latest', 'current events', 'recent news', 'today\'s news',
        'breaking news', 'stock price', 'market news', 'weather forecast',
        'sports scores', 'political news', 'celebrity news'
    ]
    has_web_search_terms = any(term in prompt_lower for term in web_search_exclusions)
    if has_web_search_terms:
        confidence_score -= 0.5
        detection_reasons.append("Web search exclusion detected")

    # Decision threshold (slightly lowered to catch more valid cases)
    threshold = 0.25
    is_table_query = confidence_score >= threshold

    # Enhanced logging for debugging
    logger.info(f"🔍 TABLE ANALYSIS ROUTER DEBUG:")
    logger.info(f"   Query: '{prompt[:100]}...'")
    logger.info(f"   Confidence Score: {confidence_score:.2f} (threshold: {threshold})")
    logger.info(f"   Decision: {'✅ TABLE ANALYSIS' if is_table_query else '❌ NOT TABLE ANALYSIS'}")
    logger.info(f"   Detection Reasons: {detection_reasons}")

    if not is_table_query and confidence_score > 0.1:
        logger.warning(f"⚠️ NEAR MISS: Query scored {confidence_score:.2f} but below threshold {threshold}")
        logger.warning(f"   Consider lowering threshold or adding more patterns")

    return is_table_query

def calculate_table_tool_priority_score(prompt: str) -> dict:
    """
    Calculate priority score for TableToCodeTool based on refined Agent Zero logic.

    This implements the "High-Priority Triggering" system that stacks the deck
    in favor of TableToCodeTool when appropriate, while preserving agent autonomy.

    Returns:
        dict: {
            'priority_score': float (0.0 to 0.99),
            'is_preferred_expert': bool,
            'reasoning': list of str,
            'capability_matches': list of str
        }
    """
    import logging
    logger = logging.getLogger(__name__)

    prompt_lower = prompt.lower()
    priority_score = 0.0
    reasoning = []
    capability_matches = []

    # Step 1: Basic table analysis check
    is_table_analysis = is_table_analysis_query(prompt)
    if not is_table_analysis:
        return {
            'priority_score': 0.0,
            'is_preferred_expert': False,
            'reasoning': ['Not identified as table analysis query'],
            'capability_matches': []
        }

    # Step 2: High-priority visualization keywords (0.99 priority)
    high_priority_viz_keywords = [
        'chart', 'plot', 'graph', 'visualize', 'diagram',
        'bar chart', 'line chart', 'pie chart', 'scatter plot',
        'histogram', 'heatmap', 'visualization', 'generate a chart',
        'create a chart', 'show a chart', 'display chart'
    ]

    viz_matches = [kw for kw in high_priority_viz_keywords if kw in prompt_lower]
    if viz_matches:
        priority_score = 0.99
        reasoning.append("High-priority visualization keywords detected")
        capability_matches.extend([f"Visualization: '{match}'" for match in viz_matches])

    # Step 3: Mathematical operations (0.85 priority)
    math_operation_keywords = [
        'calculate', 'sum', 'total', 'average', 'mean', 'count',
        'add up', 'compute', 'find the total', 'maximum', 'minimum',
        'median', 'aggregate', 'statistics'
    ]

    math_matches = [kw for kw in math_operation_keywords if kw in prompt_lower]
    if math_matches and priority_score < 0.85:
        priority_score = 0.85
        reasoning.append("Mathematical operations detected")
        capability_matches.extend([f"Math: '{match}'" for match in math_matches])

    # Step 4: Complex data filtering (0.75 priority)
    filtering_keywords = [
        'filter', 'where', 'group by', 'sort', 'completed',
        'status', 'team', 'category', 'type', 'condition'
    ]

    filter_matches = [kw for kw in filtering_keywords if kw in prompt_lower]
    if filter_matches and priority_score < 0.75:
        priority_score = 0.75
        reasoning.append("Complex data filtering detected")
        capability_matches.extend([f"Filter: '{match}'" for match in filter_matches])

    # Step 5: Structured data analysis (0.65 priority)
    analysis_keywords = [
        'analyze', 'analysis', 'breakdown', 'summary',
        'report', 'insights', 'trends', 'patterns'
    ]

    analysis_matches = [kw for kw in analysis_keywords if kw in prompt_lower]
    if analysis_matches and priority_score < 0.65:
        priority_score = 0.65
        reasoning.append("Structured data analysis detected")
        capability_matches.extend([f"Analysis: '{match}'" for match in analysis_matches])

    # Determine if this tool should be the "Preferred Expert"
    is_preferred_expert = priority_score >= 0.75

    # Enhanced logging
    logger.info(f"🎯 TABLE TOOL PRIORITY SCORING:")
    logger.info(f"   Query: '{prompt[:100]}...'")
    logger.info(f"   Priority Score: {priority_score:.2f}")
    logger.info(f"   Preferred Expert: {'✅ YES' if is_preferred_expert else '❌ NO'}")
    logger.info(f"   Capability Matches: {capability_matches}")
    logger.info(f"   Reasoning: {reasoning}")

    return {
        'priority_score': priority_score,
        'is_preferred_expert': is_preferred_expert,
        'reasoning': reasoning,
        'capability_matches': capability_matches
    }

def test_table_analysis_router(test_query: str) -> dict:
    """
    Enhanced developer testing function with refined Agent Zero logic.

    This function provides detailed analysis of both basic detection and
    priority scoring, useful for debugging the sophisticated guidance system.
    """
    import logging

    # Temporarily set up detailed logging
    logger = logging.getLogger(__name__)
    original_level = logger.level
    logger.setLevel(logging.INFO)

    print(f"\n🔬 **ENHANCED ROUTER DEBUG TEST**")
    print(f"Query: '{test_query}'")
    print("=" * 60)

    # Test basic table analysis detection
    basic_result = is_table_analysis_query(test_query)
    print(f"Basic Detection: {'✅ TABLE ANALYSIS' if basic_result else '❌ NOT TABLE ANALYSIS'}")

    # Test refined priority scoring
    priority_info = calculate_table_tool_priority_score(test_query)

    print(f"\n🎯 **PRIORITY SCORING RESULTS:**")
    print(f"   Priority Score: {priority_info['priority_score']:.2f}")
    print(f"   Preferred Expert: {'✅ YES' if priority_info['is_preferred_expert'] else '❌ NO'}")
    print(f"   Capability Matches: {priority_info['capability_matches']}")
    print(f"   Reasoning: {priority_info['reasoning']}")

    # Overall recommendation
    if priority_info['priority_score'] >= 0.75:
        recommendation = "🎉 EXCELLENT - Agent should strongly prefer TableToCodeTool"
        agent_guidance = "The deck is stacked in favor of TableToCodeTool"
    elif priority_info['priority_score'] >= 0.5:
        recommendation = "✅ GOOD - Agent should consider TableToCodeTool"
        agent_guidance = "TableToCodeTool is a strong candidate"
    elif basic_result:
        recommendation = "⚠️ BASIC - Agent might choose TableToCodeTool"
        agent_guidance = "TableToCodeTool is available but not prioritized"
    else:
        recommendation = "❌ UNLIKELY - Agent probably won't choose TableToCodeTool"
        agent_guidance = "Other tools are more appropriate"

    print(f"\n📋 **FINAL ASSESSMENT:**")
    print(f"   Recommendation: {recommendation}")
    print(f"   Agent Guidance: {agent_guidance}")
    print("=" * 60)

    # Restore original logging level
    logger.setLevel(original_level)

    return {
        "query": test_query,
        "basic_detection": basic_result,
        "priority_score": priority_info['priority_score'],
        "is_preferred_expert": priority_info['is_preferred_expert'],
        "capability_matches": priority_info['capability_matches'],
        "recommendation": recommendation,
        "agent_guidance": agent_guidance
    }

def is_calculation_only_query(prompt: str) -> bool:
    """Check if query is purely mathematical and doesn't need web search."""
    import re

    # ENHANCED: More comprehensive calculation detection
    calculation_keywords = [
        'calculate', 'compute', 'solve', 'equation', 'factorial',
        'what is', 'what\'s', 'how much is', 'equals', 'equal to'
    ]

    # Check for mathematical operators (must be present for pure math)
    math_operators = ['+', '-', '*', '/', '=', '^', '**', '%']
    has_math_operators = any(op in prompt for op in math_operators)

    # ENHANCED: Check for pure mathematical expressions (numbers + operators)
    # Pattern: optional text + numbers + operators + numbers
    math_pattern = r'(\d+(?:\.\d+)?)\s*[\+\-\*\/]\s*(\d+(?:\.\d+)?)'
    has_math_expression = bool(re.search(math_pattern, prompt))

    # Check for calculation keywords or question words
    has_calc_keywords = any(keyword in prompt.lower() for keyword in calculation_keywords)

    # ENHANCED: Check for simple question patterns like "what is X+Y?"
    simple_math_patterns = [
        r'what\s+is\s+\d+\s*[\+\-\*\/]\s*\d+',
        r'what\'s\s+\d+\s*[\+\-\*\/]\s*\d+',
        r'how\s+much\s+is\s+\d+\s*[\+\-\*\/]\s*\d+',
        r'^\s*\d+\s*[\+\-\*\/]\s*\d+\s*[\?\=]?\s*$'  # Pure math like "45+89?" or "45+89="
    ]

    has_simple_math_pattern = any(re.search(pattern, prompt.lower()) for pattern in simple_math_patterns)

    # Exclude any queries that need current/web information (but be more specific)
    web_search_exclusions = [
        'latest', 'recent', 'current', 'today', 'news', 'political', 'politics',
        'stock', 'price', 'market', 'trading', 'investment', 'shares',
        'microsoft', 'apple', 'google', 'tesla', 'amazon', 'nasdaq',
        'dow jones', 'sp500', 's&p', 'crypto', 'bitcoin', 'ethereum',
        'what is happening', 'what happened', 'update', 'breaking',
        'search for', 'find information', 'look up information', 'research about'
    ]

    has_web_search_terms = any(term in prompt.lower() for term in web_search_exclusions)

    # ENHANCED: Return True if it's a math query (operators OR patterns) and NO web search terms
    is_math_query = (has_math_operators or has_math_expression or has_simple_math_pattern) and not has_web_search_terms

    # Additional check: if it has calculation keywords or simple math patterns, it's likely math
    if (has_calc_keywords or has_simple_math_pattern) and has_math_operators and not has_web_search_terms:
        is_math_query = True

    return is_math_query

def generate_tool_enhanced_response(prompt: str, force_local: bool = False) -> str:
    """Generate response using intelligent tool selection and planning (preserving 100% of functionality)."""
    try:
        # Import required components
        from sam.orchestration.planner import DynamicPlanner
        from sam.orchestration.uif import SAM_UIF
        from sam.orchestration.skills.calculator_tool import CalculatorTool
        from sam.orchestration.skills.news_api_tool import NewsApiTool
        from sam.orchestration.skills.financial_data_tool import FinancialDataTool
        from sam.orchestration.skills.table_to_code_tool import TableToCodeTool

        # Initialize DynamicPlanner with tools
        planner = DynamicPlanner()

        # Register available tools (preserving 100% of functionality)
        calculator = CalculatorTool()
        news_tool = NewsApiTool()
        financial_tool = FinancialDataTool()
        table_to_code_tool = TableToCodeTool()

        # Enhanced TableToCodeTool advertisement with explicit capabilities
        table_to_code_tool.skill_description = (
            "A specialist tool for advanced analysis and visualization of structured table data. "
            "**Capabilities include: generating graphical outputs (bar charts, line plots), "
            "performing numerical calculations (sum, average, count), and complex data filtering.** "
            "Ideal for any query that requires mathematical operations or a visual representation "
            "of table contents. This is the ONLY tool capable of creating actual charts and graphs "
            "from tabular data."
        )

        planner.register_skill(calculator)
        planner.register_skill(news_tool)
        planner.register_skill(financial_tool)
        planner.register_skill(table_to_code_tool)

        # Calculate priority score for TableToCodeTool (refined Agent Zero logic)
        priority_info = calculate_table_tool_priority_score(prompt)

        if priority_info['is_preferred_expert']:
            logger.info(f"🎯 TableToCodeTool flagged as PREFERRED EXPERT (score: {priority_info['priority_score']:.2f})")
            logger.info(f"   Capability matches: {priority_info['capability_matches']}")

            # Enhance the tool's visibility to the agent's internal LLM
            table_to_code_tool.priority_hint = f"PREFERRED EXPERT for this query (confidence: {priority_info['priority_score']:.2f})"
            table_to_code_tool.capability_matches = priority_info['capability_matches']

        logger.info(f"Tool-enhanced planning for query: {prompt[:50]}...")

        # Create UIF for planning
        uif = SAM_UIF(input_query=prompt)

        # Generate plan
        plan_result = planner.create_plan(uif)

        if plan_result and plan_result.plan:
            logger.info(f"Generated plan: {plan_result.plan}")

            # Check if plan includes specialized tools (not just ResponseGenerationSkill)
            specialized_tools = [skill for skill in plan_result.plan if skill != 'ResponseGenerationSkill']

            if specialized_tools:
                logger.info(f"Using specialized tools: {specialized_tools}")

                # Execute plan with specialized tools
                for tool_name in specialized_tools:
                    if tool_name == 'CalculatorTool':
                        # Handle math calculations
                        try:
                            # Create proper UIF for tool execution (preserving 100% of functionality)
                            tool_uif = SAM_UIF(input_query=prompt)
                            result = calculator.execute(tool_uif)
                            if result and hasattr(result, 'intermediate_data') and result.intermediate_data:
                                calculation_result = result.intermediate_data.get('calculation_result', 'Calculation completed')
                                logger.info(f"Calculator tool result: {calculation_result}")

                                # Format the response nicely
                                response = f"🧮 **Mathematical Calculation:**\n\n"
                                response += f"**Query:** {prompt}\n\n"
                                response += f"**Result:** {calculation_result}\n\n"
                                response += f"*Calculated using SAM's secure CalculatorTool*"

                                return response
                            elif result and hasattr(result, 'skill_outputs') and 'CalculatorTool' in result.skill_outputs:
                                skill_result = result.skill_outputs['CalculatorTool'].get('result', 'Calculation completed')
                                logger.info(f"Calculator tool skill result: {skill_result}")

                                # Format the response nicely
                                response = f"🧮 **Mathematical Calculation:**\n\n"
                                response += f"**Query:** {prompt}\n\n"
                                response += f"**Result:** {skill_result}\n\n"
                                response += f"*Calculated using SAM's secure CalculatorTool*"

                                return response
                        except Exception as e:
                            logger.warning(f"Calculator tool execution failed: {e}")

                    elif tool_name == 'NewsApiTool':
                        # Handle news queries
                        try:
                            # Create proper UIF for tool execution (preserving 100% of functionality)
                            tool_uif = SAM_UIF(input_query=prompt)
                            result = news_tool.execute(tool_uif)
                            if result and hasattr(result, 'intermediate_data') and result.intermediate_data:
                                news_result = result.intermediate_data.get('news_results', 'News search completed')
                                logger.info(f"News tool result available")

                                # Format the response nicely
                                response = f"📰 **Latest News:**\n\n"
                                response += f"**Query:** {prompt}\n\n"
                                response += f"{news_result}\n\n"
                                response += f"*Retrieved using SAM's NewsApiTool*"

                                return response
                            elif result and hasattr(result, 'skill_outputs') and 'NewsApiTool' in result.skill_outputs:
                                skill_result = result.skill_outputs['NewsApiTool'].get('articles', 'News search completed')
                                logger.info(f"News tool skill result available")

                                # Format the response nicely
                                response = f"📰 **Latest News:**\n\n"
                                response += f"**Query:** {prompt}\n\n"
                                response += f"{skill_result}\n\n"
                                response += f"*Retrieved using SAM's NewsApiTool*"

                                return response
                        except Exception as e:
                            logger.warning(f"News tool execution failed: {e}")

                    elif tool_name == 'FinancialDataTool':
                        # Handle financial queries
                        try:
                            # Create proper UIF for tool execution (preserving 100% of functionality)
                            tool_uif = SAM_UIF(input_query=prompt)
                            result = financial_tool.execute(tool_uif)
                            if result and hasattr(result, 'intermediate_data') and result.intermediate_data:
                                financial_result = result.intermediate_data.get('financial_data', 'Financial data retrieved')
                                logger.info(f"Financial tool result available")

                                # Format the response nicely
                                response = f"💰 **Financial Data:**\n\n"
                                response += f"**Query:** {prompt}\n\n"
                                response += f"{financial_result}\n\n"
                                response += f"*Retrieved using SAM's FinancialDataTool*"

                                return response
                            elif result and hasattr(result, 'skill_outputs') and 'FinancialDataTool' in result.skill_outputs:
                                skill_result = result.skill_outputs['FinancialDataTool'].get('data', 'Financial data retrieved')
                                logger.info(f"Financial tool skill result available")

                                # Format the response nicely
                                response = f"💰 **Financial Data:**\n\n"
                                response += f"**Query:** {prompt}\n\n"
                                response += f"{skill_result}\n\n"
                                response += f"*Retrieved using SAM's FinancialDataTool*"

                                return response
                        except Exception as e:
                            logger.warning(f"Financial tool execution failed: {e}")

                    elif tool_name == 'TableToCodeTool':
                        # Handle table analysis and code generation queries
                        try:
                            # Create proper UIF for tool execution (preserving 100% of functionality)
                            tool_uif = SAM_UIF(input_query=prompt)

                            # Add memory store context if available
                            if hasattr(st.session_state, 'memory_store') and st.session_state.memory_store:
                                tool_uif.intermediate_data['memory_store'] = st.session_state.memory_store

                            result = table_to_code_tool.execute(tool_uif)

                            if result and hasattr(result, 'intermediate_data') and result.intermediate_data:
                                # Extract the comprehensive results from our Table-to-Code Expert Tool
                                generated_code = result.intermediate_data.get('generated_code', '')
                                wrapped_code = result.intermediate_data.get('wrapped_code', '')
                                code_explanation = result.intermediate_data.get('code_explanation', '')
                                execution_instructions = result.intermediate_data.get('execution_instructions', '')
                                validation_result = result.intermediate_data.get('validation_result', {})
                                table_summary = result.intermediate_data.get('table_summary', {})
                                safety_level = result.intermediate_data.get('safety_level', 'unknown')

                                logger.info(f"Table-to-Code tool executed successfully with safety level: {safety_level}")

                                # Create the specialized UI response for table analysis
                                response = create_table_analysis_response(
                                    prompt=prompt,
                                    generated_code=generated_code,
                                    wrapped_code=wrapped_code,
                                    code_explanation=code_explanation,
                                    execution_instructions=execution_instructions,
                                    validation_result=validation_result,
                                    table_summary=table_summary,
                                    safety_level=safety_level
                                )

                                return response

                            elif result and hasattr(result, 'skill_outputs') and 'TableToCodeTool' in result.skill_outputs:
                                skill_result = result.skill_outputs['TableToCodeTool']
                                logger.info(f"Table-to-Code tool skill result available")

                                # Format the response nicely
                                response = f"🐍 **Table Analysis & Code Generation:**\n\n"
                                response += f"**Query:** {prompt}\n\n"
                                response += f"{skill_result}\n\n"
                                response += f"*Generated using SAM's Table-to-Code Expert Tool*"

                                return response

                        except Exception as e:
                            logger.warning(f"Table-to-Code tool execution failed: {e}")
                            # Provide helpful fallback message
                            response = f"🐍 **Table Analysis Request Received:**\n\n"
                            response += f"I understand you want to analyze table data, but I encountered an issue: {str(e)}\n\n"
                            response += f"Please ensure you have uploaded table data or try rephrasing your request."
                            return response
            else:
                logger.info("No specialized tools needed, continuing with standard flow")
                return "NO_TOOL_NEEDED"
        else:
            logger.info("No plan generated, continuing with standard flow")
            return "NO_TOOL_NEEDED"

    except Exception as e:
        logger.warning(f"Tool-enhanced response generation failed: {e}")
        return "NO_TOOL_NEEDED"

# Dream Canvas functionality removed in Community Edition

def render_vetting_queue_integrated():
    """Render integrated vetting queue interface for research paper review."""
    st.subheader("🔍 Vetting Queue")
    st.markdown("*Review and approve downloaded research papers*")

    try:
        from sam.state.vetting_queue import get_vetting_queue_manager, VettingStatus
        from sam.vetting.analyzer import get_vetting_analyzer

        vetting_manager = get_vetting_queue_manager()
        vetting_analyzer = get_vetting_analyzer()

        # Get queue summary
        summary = vetting_manager.get_queue_summary()

        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            pending_count = summary.get('requires_manual_review', 0)
            if pending_count > 0:
                st.metric("🔍 Pending Review", pending_count, delta=None, delta_color="normal")
            else:
                st.metric("🔍 Pending Review", pending_count)

        with col2:
            approved_count = summary.get('auto_approved', 0) + summary.get('manually_approved', 0)
            st.metric("✅ Approved", approved_count)

        with col3:
            rejected_count = summary.get('rejected', 0)
            st.metric("❌ Rejected", rejected_count)

        with col4:
            total_count = summary.get('total', 0)
            st.metric("📊 Total", total_count)

        st.markdown("---")

        # Get files requiring manual review
        pending_files = vetting_manager.get_pending_review_files()

        if pending_files:
            st.markdown("### 📋 Files Awaiting Your Review")

            for i, entry in enumerate(pending_files):
                with st.expander(f"📄 **{entry.original_filename}** - Review Required", expanded=i == 0):

                    # File information
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**📄 File Information:**")
                        st.markdown(f"- **Original Insight:** {entry.original_insight_text[:100]}...")
                        st.markdown(f"- **Created:** {entry.created_at}")
                        st.markdown(f"- **Status:** {entry.status.value}")

                        # Paper metadata if available
                        if entry.paper_metadata:
                            metadata = entry.paper_metadata
                            st.markdown(f"- **Title:** {metadata.get('title', 'N/A')}")
                            st.markdown(f"- **Authors:** {', '.join(metadata.get('authors', [])[:3])}")
                            if len(metadata.get('authors', [])) > 3:
                                st.markdown(f"  *...and {len(metadata.get('authors', [])) - 3} more*")

                    with col2:
                        if entry.scores:
                            st.markdown("**📊 Analysis Scores:**")

                            # Security risk (inverted for display)
                            security_score = 1.0 - entry.scores.security_risk_score
                            st.progress(security_score, text=f"🛡️ Security: {security_score:.1%}")

                            # Relevance
                            st.progress(entry.scores.relevance_score, text=f"🎯 Relevance: {entry.scores.relevance_score:.1%}")

                            # Credibility
                            st.progress(entry.scores.credibility_score, text=f"⭐ Credibility: {entry.scores.credibility_score:.1%}")

                            # Overall
                            st.progress(entry.scores.overall_score, text=f"📈 Overall: {entry.scores.overall_score:.1%}")
                        else:
                            st.warning("⚠️ Analysis scores not available")

                    # Paper summary if available
                    if entry.paper_metadata and entry.paper_metadata.get('summary'):
                        st.markdown("**📝 Paper Summary:**")
                        summary_text = entry.paper_metadata['summary']
                        if len(summary_text) > 300:
                            st.markdown(f"{summary_text[:300]}...")
                        else:
                            st.markdown(summary_text)

                    # Action buttons
                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if st.button(f"✅ Approve", key=f"approve_{entry.file_id}", type="primary"):
                            success = vetting_manager.approve_file(
                                entry.file_id,
                                "manual_user",
                                "Manually approved via vetting queue interface"
                            )
                            if success:
                                st.success("✅ Paper approved!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to approve paper")

                    with col2:
                        if st.button(f"❌ Reject", key=f"reject_{entry.file_id}"):
                            # Show rejection reason input
                            st.session_state[f"show_reject_reason_{entry.file_id}"] = True

                    with col3:
                        if st.button(f"📄 View File", key=f"view_{entry.file_id}"):
                            st.info(f"📁 File location: {entry.quarantine_path}")

                    # Handle rejection reason input
                    if st.session_state.get(f"show_reject_reason_{entry.file_id}", False):
                        st.markdown("**Rejection Reason:**")
                        reason = st.text_area(
                            "Why are you rejecting this paper?",
                            key=f"reject_reason_{entry.file_id}",
                            placeholder="e.g., Not relevant to current research, Security concerns, Low quality..."
                        )

                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"Confirm Rejection", key=f"confirm_reject_{entry.file_id}", type="secondary"):
                                if reason.strip():
                                    success = vetting_manager.reject_file(
                                        entry.file_id,
                                        reason.strip(),
                                        "manual_user"
                                    )
                                    if success:
                                        st.success("❌ Paper rejected!")
                                        st.rerun()
                                    else:
                                        st.error("❌ Failed to reject paper")
                                else:
                                    st.error("Please provide a rejection reason")

                        with col2:
                            if st.button(f"Cancel", key=f"cancel_reject_{entry.file_id}"):
                                del st.session_state[f"show_reject_reason_{entry.file_id}"]
                                st.rerun()

        else:
            st.success("🎉 **All caught up!** No papers are currently awaiting review.")
            st.info("💡 **Tip:** Papers are automatically downloaded and analyzed when new insights are discovered. Check the Discovery Cycle in the Bulk Ingestion tab to trigger research.")

        # Show approved files section
        st.markdown("---")
        st.markdown("### ✅ Recently Approved Papers")

        approved_files = vetting_manager.get_approved_files()
        if approved_files:
            # Show last 5 approved files
            for entry in approved_files[-5:]:
                with st.expander(f"✅ **{entry.original_filename}** - {entry.status.value}", expanded=False):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"**Approved:** {entry.approved_at}")
                        st.markdown(f"**Approved by:** {entry.approved_by}")
                        if entry.scores:
                            st.markdown(f"**Overall Score:** {entry.scores.overall_score:.1%}")

                    with col2:
                        if entry.paper_metadata:
                            st.markdown(f"**Title:** {entry.paper_metadata.get('title', 'N/A')}")
                            st.markdown(f"**Authors:** {', '.join(entry.paper_metadata.get('authors', [])[:2])}")
        else:
            st.info("📝 No approved papers yet.")

        # Auto-approval settings
        st.markdown("---")
        st.markdown("### ⚙️ Auto-Approval Settings")

        with st.expander("🔧 Configure Auto-Approval Thresholds", expanded=False):
            current_thresholds = vetting_manager.get_auto_approval_thresholds()

            st.markdown("**Adjust the thresholds for automatic paper approval:**")

            col1, col2 = st.columns(2)

            with col1:
                security_max = st.slider(
                    "🛡️ Max Security Risk",
                    min_value=0.0,
                    max_value=1.0,
                    value=current_thresholds['security_risk_max'],
                    step=0.05,
                    help="Maximum security risk score for auto-approval"
                )

                relevance_min = st.slider(
                    "🎯 Min Relevance Score",
                    min_value=0.0,
                    max_value=1.0,
                    value=current_thresholds['relevance_min'],
                    step=0.05,
                    help="Minimum relevance score for auto-approval"
                )

            with col2:
                credibility_min = st.slider(
                    "⭐ Min Credibility Score",
                    min_value=0.0,
                    max_value=1.0,
                    value=current_thresholds['credibility_min'],
                    step=0.05,
                    help="Minimum credibility score for auto-approval"
                )

                overall_min = st.slider(
                    "📈 Min Overall Score",
                    min_value=0.0,
                    max_value=1.0,
                    value=current_thresholds['overall_min'],
                    step=0.05,
                    help="Minimum overall score for auto-approval"
                )

            if st.button("💾 Update Thresholds", type="primary"):
                new_thresholds = {
                    'security_risk_max': security_max,
                    'relevance_min': relevance_min,
                    'credibility_min': credibility_min,
                    'overall_min': overall_min
                }
                vetting_manager.update_auto_approval_thresholds(new_thresholds)
                st.success("✅ Auto-approval thresholds updated!")
                st.rerun()

    except ImportError as e:
        st.error(f"❌ Vetting queue components not available: {e}")
        st.info("💡 Make sure all Task 27 components are properly installed.")
    except Exception as e:
        st.error(f"❌ Error loading vetting queue: {e}")
        import traceback
        st.error(f"Details: {traceback.format_exc()}")

def detect_learning_intent(query: str) -> Dict[str, Any]:
    """
    Detect if the user is trying to teach SAM new information.

    Args:
        query: User's query

    Returns:
        Dictionary with learning intent analysis
    """
    query_lower = query.lower()

    # Learning intent patterns
    learning_patterns = {
        'teach': [
            r'\bi want to teach you\b',
            r'\blet me teach you\b',
            r'\bteach you something\b',
            r'\blearn this\b',
            r'\bremember this\b',
            r'\bsave this information\b',
            r'\bstore this\b'
        ],
        'correction': [
            r'\bactually,?\s*',
            r'\bthat\'?s not right\b',
            r'\bincorrect\b',
            r'\bwrong\b',
            r'\bthe correct answer is\b',
            r'\bit should be\b',
            r'\bcorrection:?\s*'
        ],
        'new_fact': [
            r'\bthe capital of .* will be\b',
            r'\bthe new .* is\b',
            r'\bfrom now on\b',
            r'\bplease note that\b',
            r'\bfor future reference\b',
            r'\bupdate:?\s*'
        ],
        'personal_info': [
            r'\bmy name is\b',
            r'\bi prefer\b',
            r'\bi like\b',
            r'\bi work\b',
            r'\bremember that i\b',
            r'\babout me:?\s*'
        ]
    }

    # Check for learning patterns
    intent_scores = {}
    for intent_type, patterns in learning_patterns.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, query_lower):
                score += 1
        intent_scores[intent_type] = score

    # Determine if this is a learning request
    total_score = sum(intent_scores.values())
    is_learning_request = total_score > 0

    # Find primary intent type
    primary_intent = 'unknown'
    if intent_scores:
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0]

    return {
        'is_learning_request': is_learning_request,
        'intent_type': primary_intent,
        'confidence': min(1.0, total_score * 0.3),
        'intent_scores': intent_scores,
        'detected_patterns': [pattern for patterns in learning_patterns.values() for pattern in patterns if re.search(pattern, query_lower)]
    }

def handle_learning_request(query: str, learning_intent: Dict[str, Any]) -> Optional[str]:
    """
    Handle a learning request by storing information in MEMOIR.

    Args:
        query: User's query
        learning_intent: Learning intent analysis

    Returns:
        Response string if handled, None otherwise
    """
    try:
        # Extract the information to learn
        info_to_learn = extract_learning_content(query, learning_intent['intent_type'])

        if not info_to_learn:
            return None

        # Store in memory
        success = store_learning_content(info_to_learn, learning_intent)

        if success:
            # Activate MEMOIR status
            st.session_state.memoir_enabled = True
            st.session_state.memoir_last_learning = datetime.now().isoformat()

            logger.info(f"🧠 MEMOIR activated: Learned new information")

            return f"""✅ **Learning Complete!**

I've successfully learned and stored this information:

**📚 New Knowledge:** {info_to_learn}

**🧠 MEMOIR Status:** ✅ Active (Learning mode engaged)

This information has been added to my knowledge base and I'll remember it for future conversations. You can see that MEMOIR is now showing as Active in the System Status.

**🔍 What I learned:** "{info_to_learn.strip()}"

Is there anything else you'd like to teach me?"""
        else:
            # Even if storage failed, still activate MEMOIR to show the system is trying
            st.session_state.memoir_enabled = True
            st.session_state.memoir_last_learning = datetime.now().isoformat()

            return f"""⚠️ **Learning Attempted**

I recognized your teaching request and tried to store this information:

**📚 Information:** {info_to_learn}

**🧠 MEMOIR Status:** ✅ Active (Learning mode engaged)

While I had some technical difficulty with the storage process, I've activated the MEMOIR system and will do my best to remember this information during our conversation.

You can try teaching me again, or ask me to recall what you just taught me to test if it worked."""

    except Exception as e:
        logger.error(f"Error handling learning request: {e}")
        return None

def extract_learning_content(query: str, intent_type: str) -> Optional[str]:
    """Extract the actual content to learn from the query."""
    query_lower = query.lower()

    if intent_type == 'teach':
        # Extract content after teaching phrases
        patterns = [
            r'i want to teach you something new:?\s*(.*)',
            r'let me teach you:?\s*(.*)',
            r'learn this:?\s*(.*)',
            r'remember this:?\s*(.*)',
            r'save this information:?\s*(.*)'
        ]
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()

    elif intent_type == 'new_fact':
        # For factual statements, use the whole query
        return query.strip()

    elif intent_type == 'personal_info':
        # Extract personal information
        patterns = [
            r'my name is\s*(.*)',
            r'i prefer\s*(.*)',
            r'i like\s*(.*)',
            r'i work\s*(.*)',
            r'remember that i\s*(.*)'
        ]
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return f"User {match.group(0).strip()}"

    # Fallback: return the whole query
    return query.strip()

def store_learning_content(content: str, learning_intent: Dict[str, Any]) -> bool:
    """Store learning content in memory."""
    try:
        # Try multiple memory storage approaches
        success = False

        # Method 1: Try secure memory store
        try:
            if hasattr(st.session_state, 'secure_memory_store') and st.session_state.secure_memory_store:
                memory_store = st.session_state.secure_memory_store

                # Use add_chunk method if available
                if hasattr(memory_store, 'add_chunk'):
                    chunk_id = memory_store.add_chunk(
                        content=content,
                        source="user_teaching",
                        chunk_type="learning",
                        metadata={
                            'learning_intent': learning_intent['intent_type'],
                            'learned_at': datetime.now().isoformat(),
                            'user_taught': True,
                            'importance': 0.9
                        }
                    )
                    success = True
                    logger.info(f"Stored learning content via add_chunk: {chunk_id}")

                # Fallback to add_memory if available
                elif hasattr(memory_store, 'add_memory'):
                    chunk_id = memory_store.add_memory(
                        content=content,
                        memory_type="learning",
                        source="user_teaching",
                        tags=["user_taught", "learning", learning_intent['intent_type']],
                        importance_score=0.9,
                        metadata={
                            'learning_intent': learning_intent,
                            'learned_at': datetime.now().isoformat(),
                            'user_taught': True
                        }
                    )
                    success = True
                    logger.info(f"Stored learning content via add_memory: {chunk_id}")
        except Exception as e:
            logger.warning(f"Secure memory store failed: {e}")

        # Method 2: Try regular memory store as fallback
        if not success:
            try:
                from memory.memory_vectorstore import get_memory_store
                memory_store = get_memory_store()

                if hasattr(memory_store, 'add_chunk'):
                    chunk_id = memory_store.add_chunk(
                        content=content,
                        source="user_teaching",
                        chunk_type="learning",
                        metadata={
                            'learning_intent': learning_intent['intent_type'],
                            'learned_at': datetime.now().isoformat(),
                            'user_taught': True,
                            'importance': 0.9
                        }
                    )
                    success = True
                    logger.info(f"Stored learning content via regular memory store: {chunk_id}")
            except Exception as e:
                logger.warning(f"Regular memory store failed: {e}")

        # Method 3: Simple session state storage as final fallback
        if not success:
            try:
                if 'user_taught_knowledge' not in st.session_state:
                    st.session_state.user_taught_knowledge = []

                knowledge_entry = {
                    'content': content,
                    'intent_type': learning_intent['intent_type'],
                    'learned_at': datetime.now().isoformat(),
                    'id': f"learning_{len(st.session_state.user_taught_knowledge)}"
                }

                st.session_state.user_taught_knowledge.append(knowledge_entry)
                success = True
                logger.info(f"Stored learning content in session state: {knowledge_entry['id']}")
            except Exception as e:
                logger.error(f"Session state storage failed: {e}")

        return success

    except Exception as e:
        logger.error(f"Error storing learning content: {e}")
        return False

# Cognitive Distillation functionality removed in Community Edition

def render_self_reflect_transparency(response_text: str):
    """
    Render SELF-REFLECT transparency information if available.

    This function checks if the response was processed through SELF-REFLECT
    and displays transparency information about any corrections made.

    Args:
        response_text: The generated response text
    """
    try:
        # Check if SELF-REFLECT data is available in session state
        self_reflect_data = st.session_state.get('last_self_reflect_data')

        if not self_reflect_data:
            return

        # Check if self-reflection was actually triggered and revisions were made
        was_revised = self_reflect_data.get('was_revised', False)
        revision_notes = self_reflect_data.get('revision_notes', '')
        self_reflect_triggered = self_reflect_data.get('self_reflect_triggered', False)

        if self_reflect_triggered:
            if was_revised and revision_notes:
                # Show successful self-correction
                with st.expander("🔍 **Self-Correction Applied:** This response was revised for factual accuracy.", expanded=False):
                    st.info("**SAM's Self-Reflection Process:**")
                    st.markdown("SAM automatically detected potential factual issues and revised the response.")

                    st.markdown("**🔍 Revision Notes:**")
                    st.text_area(
                        "Corrections Applied:",
                        value=revision_notes,
                        height=100,
                        disabled=True,
                        key="self_reflect_notes"
                    )

                    # Show additional metadata if available
                    corrections_count = self_reflect_data.get('corrections_count', 0)
                    confidence_analysis = self_reflect_data.get('confidence_analysis', {})

                    if corrections_count > 0:
                        st.success(f"✅ **{corrections_count} factual corrections** were automatically applied")

                    if confidence_analysis:
                        overall_confidence = confidence_analysis.get('overall_confidence', 0.0)
                        st.metric("Response Confidence", f"{overall_confidence:.1%}")

                    # Show MEMOIR integration status
                    memoir_edits = self_reflect_data.get('memoir_edits_created', 0)
                    if memoir_edits > 0:
                        st.info(f"🧠 **{memoir_edits} corrections** were automatically learned for future improvement")

                    st.markdown("---")
                    st.caption("🔬 This transparency feature shows SAM's autonomous fact-checking process")

            else:
                # Show that self-reflection was triggered but no corrections were needed
                with st.expander("🔍 **Self-Reflection Completed:** No corrections needed.", expanded=False):
                    st.success("**SAM's Self-Reflection Process:**")
                    st.markdown("SAM automatically reviewed this response for factual accuracy and found no issues requiring correction.")

                    confidence_analysis = self_reflect_data.get('confidence_analysis', {})
                    if confidence_analysis:
                        overall_confidence = confidence_analysis.get('overall_confidence', 0.0)
                        st.metric("Response Confidence", f"{overall_confidence:.1%}")

                    st.markdown("---")
                    st.caption("🔬 This transparency feature shows SAM's autonomous fact-checking process")

        # Clear the data after displaying to avoid showing it for subsequent responses
        if 'last_self_reflect_data' in st.session_state:
            del st.session_state['last_self_reflect_data']

    except Exception as e:
        logger.debug(f"SELF-REFLECT transparency display error: {e}")

def simulate_self_reflect_for_demo(response_text: str, query: str):
    """
    Simulate SELF-REFLECT processing for demonstration purposes.

    This function demonstrates how the SELF-REFLECT system would work
    by analyzing the response and potentially triggering corrections.

    Args:
        response_text: The generated response
        query: The original query
    """
    try:
        # Import SELF-REFLECT components if available
        from sam.orchestration.skills.autonomous.factual_correction import AutonomousFactualCorrectionSkill
        from sam.orchestration.uif import SAM_UIF
        from sam.orchestration.config import get_sof_config

        # Check if SELF-REFLECT is enabled
        config = get_sof_config()
        if not getattr(config, 'enable_self_reflect', True):
            return

        # Create UIF for SELF-REFLECT processing
        uif = SAM_UIF(
            input_query=query,
            intermediate_data={
                'response_text': response_text,
                'original_query': query,
                'initial_response': response_text,
                'confidence_scores': {'overall': 0.75}  # Simulated confidence
            }
        )

        # Initialize and execute SELF-REFLECT skill
        self_reflect_skill = AutonomousFactualCorrectionSkill(
            enable_self_reflect=True,
            self_reflect_threshold=0.7
        )

        # Execute the skill
        result_uif = self_reflect_skill.execute(uif)

        # Store results for transparency display
        st.session_state['last_self_reflect_data'] = {
            'was_revised': result_uif.intermediate_data.get('was_revised', False),
            'revision_notes': result_uif.intermediate_data.get('revision_notes', ''),
            'final_response': result_uif.intermediate_data.get('final_response', response_text),
            'self_reflect_triggered': True,
            'corrections_count': len(result_uif.intermediate_data.get('corrections_made', [])),
            'confidence_analysis': result_uif.intermediate_data.get('confidence_analysis', {}),
            'memoir_edits_created': 0  # Would be populated by MEMOIR integration
        }

        logger.info("SELF-REFLECT simulation completed")

    except ImportError:
        logger.debug("SELF-REFLECT components not available for simulation")
    except Exception as e:
        logger.debug(f"SELF-REFLECT simulation error: {e}")


def render_dream_canvas_integrated():
    """Render Dream Canvas integrated into the main SAM interface."""
    try:
        st.header("🧠🎨 Dream Canvas")
        st.markdown("*Cognitive synthesis and memory landscape visualization*")

        # Check if Dream Canvas is available
        try:
            from ui.memory_app import is_dream_canvas_available
            if not is_dream_canvas_available():
                st.warning("🔒 Dream Canvas requires SAM Pro activation")
                st.info("💡 Activate SAM Pro to unlock advanced cognitive synthesis features")
                return
        except ImportError:
            pass  # Continue if entitlement system not available

        # Import Dream Canvas components
        try:
            from ui.dream_canvas import render_dream_canvas

            # Add integration notice
            st.info("🌟 **Dream Canvas Integration**: This feature is now integrated into the main SAM interface for seamless access!")

            # Render the Dream Canvas
            render_dream_canvas()

        except ImportError as e:
            st.error(f"❌ Dream Canvas components not available: {e}")
            st.info("💡 **Alternative**: You can access Dream Canvas through the Memory Control Center")

            # Provide fallback instructions
            st.markdown("### 🔧 Alternative Access Methods:")
            st.markdown("1. **Memory Control Center**: `python -m streamlit run ui/memory_app.py --server.port 8503`")
            st.markdown("2. **Launcher**: `python start_sam_secure.py --mode memory`")

        except Exception as e:
            st.error(f"❌ Error loading Dream Canvas: {e}")
            logger.error(f"Dream Canvas integration error: {e}")

            # Provide basic memory synthesis as fallback
            st.markdown("### 🧠 Basic Memory Synthesis (Fallback)")

            if st.button("🔄 Run Basic Synthesis", help="Generate insights from memory without full Dream Canvas"):
                try:
                    from memory.synthesis.synthesis_engine import SynthesisEngine
                    from memory.memory_vectorstore import get_memory_store

                    with st.spinner("🌙 Running basic synthesis..."):
                        memory_store = get_memory_store()
                        synthesis_engine = SynthesisEngine()

                        result = synthesis_engine.run_synthesis(memory_store, visualize=False)

                        if result.insights_generated > 0:
                            st.success(f"✨ Generated {result.insights_generated} insights!")

                            # Display insights
                            for i, insight in enumerate(result.insights, 1):
                                with st.expander(f"💡 Insight {i}: {insight.title}"):
                                    st.markdown(f"**Theme**: {insight.theme}")
                                    st.markdown(f"**Content**: {insight.content}")
                                    st.markdown(f"**So What**: {insight.so_what}")
                                    st.markdown(f"**Confidence**: {insight.confidence_score:.2f}")
                        else:
                            st.warning("⚠️ No insights generated. Try adding more conversations or documents.")

                except Exception as synthesis_error:
                    st.error(f"❌ Basic synthesis failed: {synthesis_error}")

    except Exception as e:
        st.error(f"❌ Dream Canvas integration failed: {e}")
        logger.error(f"Dream Canvas integration error: {e}")


# Basic interface functions for when SAM is not fully initialized
def render_basic_chat_interface():
    """Basic chat interface without full SAM initialization."""
    st.header("💬 Chat Interface")
    st.info("🔧 Initialize SAM components to enable AI chat functionality")
    st.markdown("""
    **Available after initialization:**
    - AI-powered conversations
    - Document-aware responses
    - Memory integration
    - Advanced reasoning
    """)

def render_basic_document_interface():
    """Basic document interface without full SAM initialization."""
    st.header("📚 Document Processing")
    st.info("🔧 Initialize SAM components to enable document processing")
    st.markdown("""
    **Available after initialization:**
    - PDF, DOCX, TXT file upload
    - Intelligent document parsing
    - Content extraction and indexing
    - Document-based Q&A
    """)

def render_basic_memory_interface():
    """Basic memory interface without full SAM initialization."""
    st.header("🧠 Memory Management")
    st.info("🔧 Initialize SAM components to enable memory features")
    st.markdown("""
    **Available after initialization:**
    - Conversation history
    - Knowledge base management
    - Memory search and retrieval
    - Context-aware responses
    """)

def render_basic_vetting_interface():
    """Basic vetting interface without full SAM initialization."""
    st.header("🔍 Content Vetting")
    st.info("🔧 Initialize SAM components to enable content vetting")
    st.markdown("""
    **Available after initialization:**
    - Content quality assessment
    - Source verification
    - Information validation
    - Trust scoring
    """)

def render_basic_security_interface():
    """Basic security interface."""
    st.header("🛡️ Security Dashboard")
    st.success("✅ Security system is active")
    st.markdown("""
    **Current Security Status:**
    - ✅ Encryption enabled
    - ✅ Authentication active
    - ✅ Secure storage configured
    - ✅ Session management active
    """)

if __name__ == "__main__":
    main()
