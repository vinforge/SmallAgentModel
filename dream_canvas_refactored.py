#!/usr/bin/env python3
"""
SAM Dream Canvas - Refactored
=============================

New streamlined SAM Dream Canvas using the refactored modular architecture.
This replaces the monolithic dream_canvas.py (4,484 lines) with a clean, 
maintainable structure.

Key improvements:
- Modular component architecture
- Separation of concerns (mapping, visualization, research)
- Improved maintainability and testability
- Better error handling and logging
- Enhanced cognitive mapping algorithms
- Cleaner code organization

Usage:
    streamlit run dream_canvas_refactored.py

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
    'PYTORCH_DISABLE_PER_OP_PROFILING': '1'
})

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Create logs directory
Path('logs').mkdir(exist_ok=True)

# Import the refactored Dream Canvas controller
from sam.dream_canvas import main

if __name__ == "__main__":
    main()
