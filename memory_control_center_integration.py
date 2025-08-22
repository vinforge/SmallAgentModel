"""
Memory Control Center Integration for Architecture Explorer
==========================================================

Integration component to add the Architecture Explorer as a tab
in SAM's Memory Control Center.

Author: SAM Development Team
Version: 1.0.0
"""

import streamlit as st
import requests
from pathlib import Path

def add_architecture_explorer_tab():
    """Add Architecture Explorer tab to Memory Control Center."""
    
    st.header("🗺️ Architecture Explorer")
    st.markdown("Interactive visualization of SAM's static architecture")
    
    # Check if Architecture Explorer service is running
    try:
        response = requests.get("http://localhost:5001/api/architecture", timeout=5)
        if response.status_code == 200:
            # Service is running, embed it
            st.markdown("""
            <iframe src="http://localhost:5001" 
                    width="100%" 
                    height="800" 
                    frameborder="0">
            </iframe>
            """, unsafe_allow_html=True)
        else:
            st.error("Architecture Explorer service not responding")
            show_launch_instructions()
            
    except requests.RequestException:
        st.warning("Architecture Explorer service not running")
        show_launch_instructions()

def show_launch_instructions():
    """Show instructions for launching the Architecture Explorer."""
    st.markdown("""
    ### 🚀 Launch Architecture Explorer
    
    To use the Architecture Explorer, run this command in your terminal:
    
    ```bash
    python launch_architecture_explorer.py
    ```
    
    Then refresh this page to see the interactive visualization.
    """)
    
    if st.button("🔄 Refresh Page"):
        st.experimental_rerun()

# Usage in Memory Control Center:
# Add this to your main Memory Control Center file:
#
# from memory_control_center_integration import add_architecture_explorer_tab
# 
# # In your tab creation section:
# with st.tabs(["Memory", "🗺️ Architecture Explorer", "Other Tabs"])[1]:
#     add_architecture_explorer_tab()
