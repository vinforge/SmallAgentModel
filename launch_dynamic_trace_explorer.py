#!/usr/bin/env python3
"""
Launch SAM Dynamic Trace Explorer
=================================

Launches the complete Dynamic Trace Explorer system including
Flight Recorder and Trace Visualization UI.

Author: SAM Development Team
Version: 1.0.0
"""

import sys
import argparse
from pathlib import Path

# Add SAM to path
sys.path.insert(0, str(Path(__file__).parent / "SmallAgentModel-main"))

def main():
    """Launch the Dynamic Trace Explorer."""
    parser = argparse.ArgumentParser(description="Launch SAM Dynamic Trace Explorer")
    parser.add_argument("--port", type=int, default=5003, help="Port to run the server on")
    parser.add_argument("--trace-level", choices=["critical", "detailed", "verbose"], 
                       default="detailed", help="Trace detail level")
    parser.add_argument("--max-sessions", type=int, default=100, 
                       help="Maximum number of sessions to keep in memory")
    parser.add_argument("--save-traces", action="store_true", 
                       help="Save traces to disk automatically")
    parser.add_argument("--trace-dir", type=str, default="traces", 
                       help="Directory to save trace files")
    
    args = parser.parse_args()
    
    print("🔬 SAM Dynamic Trace Explorer")
    print("=" * 35)
    print(f"🛩️ Trace Level: {args.trace_level}")
    print(f"💾 Auto-save: {'Yes' if args.save_traces else 'No'}")
    print(f"📂 Trace Directory: {args.trace_dir}")
    
    try:
        from sam.introspection.flight_recorder import initialize_flight_recorder, TraceLevel
        from sam.introspection.trace_visualization_ui import TraceVisualizationUI
        
        # Map string to enum
        trace_level_map = {
            "critical": TraceLevel.CRITICAL,
            "detailed": TraceLevel.DETAILED,
            "verbose": TraceLevel.VERBOSE
        }
        
        # Initialize Flight Recorder
        print("🛩️ Initializing Flight Recorder...")
        recorder = initialize_flight_recorder(
            trace_level=trace_level_map[args.trace_level],
            max_sessions=args.max_sessions,
            auto_save=args.save_traces,
            save_directory=args.trace_dir
        )
        
        # Initialize Trace Visualization UI
        print("🌐 Initializing Trace Visualization UI...")
        ui = TraceVisualizationUI(port=args.port)
        
        print(f"🚀 Starting Dynamic Trace Explorer on port {args.port}")
        print(f"🌐 Open http://localhost:{args.port} in your browser")
        print("📊 Features available:")
        print("   • Interactive reasoning timeline")
        print("   • Cognitive trajectory visualization")
        print("   • Step-by-step trace analysis")
        print("   • Algonauts-inspired neural projections")
        print("\nPress Ctrl+C to stop")
        
        ui.run(debug=False)
            
    except KeyboardInterrupt:
        print("\n👋 Dynamic Trace Explorer stopped")
    except Exception as e:
        print(f"❌ Failed to start Dynamic Trace Explorer: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
