#!/usr/bin/env python3
"""
SAM Memory UI Launcher
Launch the interactive memory control and visualization interface.

Sprint 12: Interactive Memory Control & Visualization
"""

import sys
import subprocess
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'streamlit',
        'plotly',
        'networkx',
        'pandas',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.info("Install them with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def launch_memory_ui():
    """Launch the Streamlit memory UI application."""
    try:
        # Check dependencies
        if not check_dependencies():
            return False
        
        # Get the path to the memory app
        app_path = Path(__file__).parent / "ui" / "memory_app.py"
        
        if not app_path.exists():
            logger.error(f"Memory app not found at: {app_path}")
            return False
        
        logger.info("🚀 Launching SAM Memory Control Center...")
        logger.info("🔗 Server starting at: http://localhost:8501")
        logger.info("🔐 Authentication required - unlock SAM at http://localhost:8502 first")
        logger.info("📱 Navigate to http://localhost:8501 after authentication")
        logger.info("⏹️  Press Ctrl+C to stop the server")
        
        # Launch Streamlit (no auto-open since authentication is required)
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(app_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false",
            "--server.headless", "true"
        ]
        
        subprocess.run(cmd)
        
        return True
        
    except KeyboardInterrupt:
        logger.info("\n👋 Memory UI stopped by user")
        return True
    except Exception as e:
        logger.error(f"❌ Error launching memory UI: {e}")
        return False

def main():
    """Main launcher function."""
    print("🧠 SAM Memory Control Center Launcher")
    print("=" * 50)
    print("Interactive Memory Control & Visualization")
    print("Sprint 12 Implementation")
    print("=" * 50)

    print("\n⚠️  **SECURITY NOTICE:**")
    print("   The Memory Control Center requires authentication.")
    print("   Make sure you have unlocked SAM at http://localhost:8502 first.")
    print("   If not authenticated, you will be redirected to the secure interface.")
    print("=" * 50)

    # Add current directory to Python path
    sys.path.insert(0, str(Path(__file__).parent))

    success = launch_memory_ui()

    if success:
        print("\n✅ Memory UI session completed successfully")
        return 0
    else:
        print("\n❌ Memory UI launch failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
