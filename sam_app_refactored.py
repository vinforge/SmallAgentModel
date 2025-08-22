#!/usr/bin/env python3
"""
SAM Refactored Application
=========================

New streamlined SAM application using the refactored modular architecture.
This replaces the monolithic secure_streamlit_app.py (14,403 lines) with
a clean, maintainable structure.

Key improvements:
- Modular component architecture
- Separation of concerns
- Improved maintainability
- Better error handling
- Enhanced security
- Cleaner code organization

Usage:
    streamlit run sam_app_refactored.py

Author: SAM Development Team
Version: 2.0.0 - Refactored Architecture
"""

import os
import sys
from pathlib import Path

# Environment setup for Streamlit compatibility
os.environ.update({
    'STREAMLIT_SERVER_FILE_WATCHER_TYPE': 'none',
    'STREAMLIT_SERVER_HEADLESS': 'true',
    'STREAMLIT_BROWSER_GATHER_USAGE_STATS': 'false',
    'PYTORCH_DISABLE_PER_OP_PROFILING': '1',
    'STREAMLIT_SERVER_ENABLE_CORS': 'false',
    'STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION': 'false',
    'STREAMLIT_SERVER_ENABLE_STATIC_SERVING': 'true',
    'STREAMLIT_CLIENT_TOOLBAR_MODE': 'minimal',
    'STREAMLIT_CLIENT_SHOW_ERROR_DETAILS': 'true'
})

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Create logs directory
Path('logs').mkdir(exist_ok=True)

# Import the refactored application controller
from sam.ui import main

if __name__ == "__main__":
    main()
