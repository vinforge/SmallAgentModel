#!/usr/bin/env python3
"""
SAM Memory Control Center - Refactored
======================================

New streamlined SAM Memory Control Center using the refactored modular architecture.
This replaces the monolithic memory_app.py (8,434 lines) with a clean, 
maintainable structure.

Key improvements:
- Modular component architecture
- Separation of concerns (auth, components, handlers)
- Improved maintainability and testability
- Better error handling and logging
- Enhanced security integration
- Cleaner code organization

Usage:
    streamlit run memory_app_refactored.py

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

# Import the refactored memory application controller
from sam.memory_ui import main

if __name__ == "__main__":
    main()
