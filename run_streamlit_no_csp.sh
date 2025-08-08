#!/bin/bash

# Kill any existing Streamlit processes on port 8502
echo "Stopping any existing Streamlit processes..."
lsof -ti:8502 | xargs kill -9 2>/dev/null || true

# Wait a moment for the port to be freed
sleep 2

# Change to the script's directory to ensure correct file paths
cd "$(dirname "$0")" || exit

# Set environment variables to disable CSP and other security features
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
export STREAMLIT_SERVER_ENABLE_STATIC_SERVING=true
export STREAMLIT_CLIENT_TOOLBAR_MODE=minimal
export STREAMLIT_CLIENT_SHOW_ERROR_DETAILS=true
export STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Additional environment variables to bypass CSP
export STREAMLIT_SERVER_RUN_ON_SAVE=false
export STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false
export STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION=false

# Python environment variables
export PYTHONUNBUFFERED=1

echo "Starting SAM Streamlit app with CSP disabled..."
echo "Access the app at: http://localhost:8502"

# Run Streamlit with additional flags to disable security features
python -m streamlit run secure_streamlit_app.py \
    --server.port=8502 \
    --server.address=localhost \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --server.enableStaticServing=true \
    --browser.gatherUsageStats=false \
    --server.fileWatcherType=none \
    --global.developmentMode=false \
    --client.toolbarMode=minimal
