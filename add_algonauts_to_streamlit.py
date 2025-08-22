"""
Add Algonauts to Streamlit App
==============================

Simple integration helper to add Algonauts cognitive visualizations
to your existing Streamlit application.

Author: SAM Development Team
Version: 1.0.0
"""

import streamlit as st
import sys
from pathlib import Path

# Add SAM to path
sam_path = Path(__file__).parent / "SmallAgentModel-main"
if str(sam_path) not in sys.path:
    sys.path.insert(0, str(sam_path))

def add_algonauts_tab():
    """
    Add Algonauts visualization tab to your Streamlit app.
    
    Usage in your secure_streamlit_app.py:
    
    ```python
    from add_algonauts_to_streamlit import add_algonauts_tab
    
    # In your main app, add this to your tabs:
    tab1, tab2, tab3, tab_algonauts = st.tabs(["Chat", "Memory", "Other", "🧠 Algonauts"])
    
    with tab_algonauts:
        add_algonauts_tab()
    ```
    """
    try:
        from sam.introspection.streamlit_algonauts import render_algonauts_interface
        render_algonauts_interface()
    except ImportError as e:
        st.error(f"❌ Algonauts components not found: {e}")
        st.info("💡 Make sure the SAM introspection module is properly installed")
    except Exception as e:
        st.error(f"❌ Failed to load Algonauts interface: {e}")


def add_algonauts_sidebar():
    """
    Add Algonauts quick access to sidebar.
    
    Usage in your secure_streamlit_app.py:
    
    ```python
    from add_algonauts_to_streamlit import add_algonauts_sidebar
    
    # In your sidebar:
    add_algonauts_sidebar()
    ```
    """
    try:
        from sam.introspection.flight_recorder import get_flight_recorder
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🧠 Cognitive Traces")
        
        recorder = get_flight_recorder()
        sessions = recorder.get_all_sessions()
        
        if sessions:
            st.sidebar.success(f"📊 {len(sessions)} reasoning sessions captured")
            
            if st.sidebar.button("🔬 View Algonauts Analysis"):
                st.session_state.show_algonauts = True
        else:
            st.sidebar.info("🔄 No cognitive traces yet")
            
            if st.sidebar.button("🎯 Generate Demo Traces"):
                st.session_state.generate_demo = True
                
    except Exception as e:
        st.sidebar.error(f"❌ Algonauts sidebar error: {e}")


def initialize_flight_recorder_for_streamlit():
    """
    Initialize the Flight Recorder for Streamlit session.
    Call this once at the start of your app.
    """
    try:
        from sam.introspection.flight_recorder import initialize_flight_recorder, TraceLevel
        
        # Initialize with Streamlit-friendly settings
        recorder = initialize_flight_recorder(
            trace_level=TraceLevel.DETAILED,
            max_sessions=50,  # Reasonable limit for web app
            auto_save=False,  # Don't save to disk in web app
            save_directory="streamlit_traces"
        )
        
        return recorder
        
    except Exception as e:
        st.error(f"❌ Failed to initialize Flight Recorder: {e}")
        return None


# Example integration code for your secure_streamlit_app.py
INTEGRATION_EXAMPLE = '''
# Add this to your secure_streamlit_app.py

import sys
from pathlib import Path

# Add SAM path
sam_path = Path(__file__).parent / "SmallAgentModel-main"
sys.path.insert(0, str(sam_path))

from add_algonauts_to_streamlit import (
    add_algonauts_tab, 
    add_algonauts_sidebar, 
    initialize_flight_recorder_for_streamlit
)

# Initialize Flight Recorder (call once at app start)
if 'flight_recorder' not in st.session_state:
    st.session_state.flight_recorder = initialize_flight_recorder_for_streamlit()

# Add to your main tabs
tab1, tab2, tab3, tab_algonauts = st.tabs(["💬 Chat", "🧠 Memory", "⚙️ Settings", "🔬 Algonauts"])

with tab1:
    # Your existing chat interface
    pass

with tab2:
    # Your existing memory interface  
    pass

with tab3:
    # Your existing settings
    pass

with tab_algonauts:
    add_algonauts_tab()

# Add to sidebar
add_algonauts_sidebar()
'''


def show_integration_instructions():
    """Show integration instructions in Streamlit."""
    st.title("🔬 Algonauts Integration for Streamlit")
    
    st.markdown("""
    ## 🚀 Quick Integration
    
    Add Algonauts cognitive visualizations to your existing Streamlit app in 3 simple steps:
    """)
    
    st.markdown("### Step 1: Add the Integration Code")
    st.code(INTEGRATION_EXAMPLE, language="python")
    
    st.markdown("### Step 2: Instrument SAM Components")
    st.code('''
# In your SAM reasoning components, add cognitive tracing:

from sam.introspection import TraceSession, ReasoningStep, trace_step

# For each user query:
with TraceSession(user_query) as session_id:
    
    # Add @trace_step decorators to your methods:
    @trace_step(ReasoningStep.MEMORY_RETRIEVAL, "memory")
    def search_memory(self, query):
        # Your existing memory search code
        return results
    
    # Or use context managers:
    with trace_context(session_id, ReasoningStep.MODEL_INFERENCE, "model", "generate"):
        response = model.generate(prompt)
    
    # Capture cognitive vectors (neural activations):
    recorder.log_step(
        session_id=session_id,
        step_type=ReasoningStep.MEMORY_RETRIEVAL,
        component="memory",
        operation="search",
        cognitive_vector=embedding_vector.tolist(),  # ← Key for Algonauts viz
        input_data={"query": query},
        output_data={"results": results}
    )
''', language="python")
    
    st.markdown("### Step 3: Run Your App")
    st.code("streamlit run secure_streamlit_app.py --server.port 8501 --server.address localhost")
    
    st.success("✅ That's it! Algonauts visualizations will appear in the new tab.")
    
    st.markdown("---")
    st.markdown("## 🎯 Test the Integration")
    
    if st.button("🧠 Generate Demo Traces"):
        with st.spinner("Creating demo cognitive traces..."):
            try:
                # Initialize flight recorder
                initialize_flight_recorder_for_streamlit()
                
                # Create demo traces
                from enable_algonauts_demo import create_sample_reasoning_traces
                success = create_sample_reasoning_traces()
                
                if success:
                    st.success("✅ Demo traces created!")
                    st.info("💡 Now you can test the Algonauts tab above")
                else:
                    st.error("❌ Failed to create demo traces")
                    
            except Exception as e:
                st.error(f"❌ Demo creation failed: {str(e)}")
    
    # Show the actual Algonauts interface
    st.markdown("---")
    st.markdown("## 🔬 Live Algonauts Interface")
    add_algonauts_tab()


if __name__ == "__main__":
    show_integration_instructions()
