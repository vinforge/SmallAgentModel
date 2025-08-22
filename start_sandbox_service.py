#!/usr/bin/env python3
"""
Start Sandbox Service
=====================

Start the Code Interpreter sandbox service on port 6821.

This service provides secure Python code execution for data analysis.

Author: SAM Development Team
Version: 1.0.0
"""

import sys
import logging
from pathlib import Path

# Add SAM to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Start the sandbox service."""
    print("🚀 Starting SAM Code Interpreter Sandbox Service")
    print("=" * 55)
    print()
    print("📊 **Purpose**: Secure Python code execution for data analysis")
    print("🔒 **Security**: Sandboxed environment with restricted access")
    print("🌐 **Port**: 6821 (avoiding common port conflicts)")
    print("📈 **Capabilities**: pandas, numpy, matplotlib, data science")
    print()
    
    try:
        # Import and start the sandbox service
        from sam.code_interpreter.sandbox_service import create_sandbox_service
        
        print("🔧 Initializing sandbox service...")
        service = create_sandbox_service()
        
        print("✅ Sandbox service initialized successfully")
        print("🌐 Starting web service on http://localhost:6821")
        print()
        print("💡 **Usage**: This service enables CSV data analysis queries like:")
        print("   • 'Calculate the average salary for the entire company'")
        print("   • 'Show me correlations in the data'")
        print("   • 'Create a visualization of department statistics'")
        print()
        print("🔄 **Status**: Service is running... (Press Ctrl+C to stop)")
        print("=" * 55)
        
        # Start the service
        service.run(host="localhost", port=6821, debug=False)
        
    except KeyboardInterrupt:
        print("\n🛑 Sandbox service stopped by user")
    except ImportError as e:
        print(f"❌ Failed to import sandbox service: {e}")
        print("💡 Make sure you're in the SAM directory and dependencies are installed")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to start sandbox service: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
