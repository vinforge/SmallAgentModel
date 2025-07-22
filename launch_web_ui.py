#!/usr/bin/env python3
"""
Launch script for SAM's Web UI
"""

import os
import sys
from pathlib import Path

def main():
    """Launch the SAM Web UI."""
    print("🚀 Starting SAM Web UI...")
    print("=" * 50)
    
    # Add current directory and web_ui directory to Python path
    current_dir = Path(__file__).parent
    web_ui_dir = current_dir / "web_ui"

    sys.path.insert(0, str(current_dir))
    sys.path.insert(0, str(web_ui_dir))

    # Change to web_ui directory
    os.chdir(web_ui_dir)

    try:
        # Import and run the Flask app
        from app import app, initialize_sam
        
        # Initialize SAM components
        print("🔧 Initializing SAM components...")
        if initialize_sam():
            print("✅ SAM components initialized successfully!")
            print("🌐 Starting web server...")
            print("📱 Access the interface at: http://localhost:5001")
            print("=" * 50)

            # Run the Flask app
            app.run(debug=False, host='0.0.0.0', port=5001)
        else:
            print("❌ Failed to initialize SAM components")
            print("Please check that Ollama is running and the model is available")
            
    except KeyboardInterrupt:
        print("\n👋 SAM Web UI shutting down...")
    except Exception as e:
        print(f"❌ Error starting web UI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
