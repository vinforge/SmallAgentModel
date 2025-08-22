#!/usr/bin/env python3
"""
Launch SAM Architecture Explorer
===============================

Launches the interactive Architecture Explorer UI for visualizing
SAM's static code architecture.

Author: SAM Development Team
Version: 1.0.0
"""

import sys
import argparse
from pathlib import Path

# Add SAM to path
sys.path.insert(0, str(Path(__file__).parent / "SmallAgentModel-main"))

def main():
    """Launch the Architecture Explorer."""
    parser = argparse.ArgumentParser(description="Launch SAM Architecture Explorer")
    parser.add_argument("--port", type=int, default=5001, help="Port to run the server on")
    parser.add_argument("--sam-root", type=str, default=None, help="Path to SAM root directory")
    parser.add_argument("--export", type=str, default=None, help="Generate static HTML export to file")
    
    args = parser.parse_args()
    
    # Determine SAM root path
    sam_root = args.sam_root or str(Path(__file__).parent / "SmallAgentModel-main")
    
    print("🗺️ SAM Architecture Explorer")
    print("=" * 35)
    print(f"📂 SAM Root: {sam_root}")
    
    try:
        from sam.introspection.architecture_explorer_ui import ArchitectureExplorerUI
        
        # Initialize UI
        ui = ArchitectureExplorerUI(sam_root, port=args.port)
        
        if args.export:
            # Generate static export
            print(f"📄 Generating static export to: {args.export}")
            ui.generate_static_export(args.export)
            print("✅ Static export complete!")
        else:
            # Launch interactive server
            print(f"🚀 Starting Architecture Explorer on port {args.port}")
            print(f"🌐 Open http://localhost:{args.port} in your browser")
            print("Press Ctrl+C to stop")
            ui.run(debug=False)
            
    except KeyboardInterrupt:
        print("\n👋 Architecture Explorer stopped")
    except Exception as e:
        print(f"❌ Failed to start Architecture Explorer: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
