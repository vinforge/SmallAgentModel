#!/bin/bash

# Disable CSP restrictions for Streamlit
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
export STREAMLIT_SERVER_ENABLE_STATIC_SERVING=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_HEADLESS=true

# Additional environment variables to bypass CSP
export STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION=false
export STREAMLIT_CLIENT_TOOLBAR_MODE=minimal

# CSP-specific overrides
export STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
export STREAMLIT_GLOBAL_DEVELOPMENT_MODE=true

# Run Streamlit with additional flags to disable CSP
echo "Starting Streamlit with CSP disabled..."
python -m streamlit run secure_streamlit_app.py \
    --server.port=8502 \
    --server.address=localhost \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --global.developmentMode=true \
    --server.fileWatcherType=none \
    --runner.fastReruns=true
