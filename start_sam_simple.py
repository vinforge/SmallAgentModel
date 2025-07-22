#!/usr/bin/env python3
"""
SAM Launch Script
================

Convenient script to start SAM with proper error handling.
"""

import subprocess
import sys
import webbrowser
import time

def main():
    print("Starting SAM...")

    try:
        # Start SAM using streamlit run command
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "secure_streamlit_app.py",
            "--server.port", "8502",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])

        # Wait a moment for startup
        time.sleep(3)

        # Open browser
        print("Opening browser...")
        webbrowser.open("http://localhost:8502")

        print("SAM is running!")
        print("Access SAM at: http://localhost:8502")
        print("Press Ctrl+C to stop SAM")

        # Wait for process
        process.wait()

    except KeyboardInterrupt:
        print("\nStopping SAM...")
        process.terminate()
    except Exception as e:
        print(f"Error starting SAM: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
