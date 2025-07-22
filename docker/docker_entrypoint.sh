#!/bin/bash
# SAM Docker Entrypoint Script
# Handles initialization, health checks, and startup

set -e

echo "🐳 SAM Docker Container Starting..."
echo "=================================="

# Environment setup
export SAM_DOCKER=true
export PYTHONPATH="/app:$PYTHONPATH"

# Function to wait for service
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local max_attempts=30
    local attempt=1

    echo "⏳ Waiting for $service_name at $host:$port..."
    
    while [ $attempt -le $max_attempts ]; do
        if nc -z "$host" "$port" 2>/dev/null; then
            echo "✅ $service_name is ready!"
            return 0
        fi
        echo "   Attempt $attempt/$max_attempts - $service_name not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "❌ Failed to connect to $service_name after $max_attempts attempts"
    return 1
}

# Function to initialize SAM data directories
initialize_sam_directories() {
    echo "📁 Initializing SAM directories..."
    
    # Create required directories if they don't exist
    mkdir -p /app/data/{documents,uploads,vector_store,knowledge_base,archives}
    mkdir -p /app/memory_store/{encrypted,chroma_db}
    mkdir -p /app/logs
    mkdir -p /app/cache/{distillation}
    mkdir -p /app/backups
    mkdir -p /app/security
    mkdir -p /app/config
    mkdir -p /app/chroma_db
    
    # Set proper permissions
    chmod 755 /app/data /app/memory_store /app/logs /app/cache /app/backups
    chmod 700 /app/security  # More restrictive for security files
    
    echo "✅ Directories initialized"
}

# Function to check and initialize SAM configuration
initialize_sam_config() {
    echo "⚙️ Initializing SAM configuration..."
    
    # Check if Docker-specific config exists
    if [ ! -f "/app/config/sam_config.json" ]; then
        echo "📝 Creating default Docker configuration..."
        cat > /app/config/sam_config.json << 'EOF'
{
    "environment": "docker",
    "data_directory": "/app/data",
    "memory_store_directory": "/app/memory_store",
    "logs_directory": "/app/logs",
    "cache_directory": "/app/cache",
    "security_directory": "/app/security",
    "chroma_db_directory": "/app/chroma_db",
    "redis_url": "redis://redis:6379/0",
    "chroma_host": "chroma",
    "chroma_port": 8000,
    "streamlit_config": {
        "server.port": 8502,
        "server.address": "0.0.0.0",
        "server.headless": true,
        "server.enableCORS": false,
        "server.enableXsrfProtection": false
    },
    "docker_mode": true,
    "auto_setup": true
}
EOF
    fi
    
    echo "✅ Configuration initialized"
}

# Function to run SAM health check
health_check() {
    echo "🏥 Running health check..."
    
    # Check if Python can import SAM modules
    if ! python -c "import sys; sys.path.insert(0, '/app'); import secure_streamlit_app" 2>/dev/null; then
        echo "❌ SAM modules not importable"
        return 1
    fi
    
    # Check if required directories exist
    for dir in "/app/data" "/app/memory_store" "/app/logs"; do
        if [ ! -d "$dir" ]; then
            echo "❌ Required directory $dir missing"
            return 1
        fi
    done
    
    echo "✅ Health check passed"
    return 0
}

# Function to setup SAM for first run
setup_sam_first_run() {
    echo "🎯 Setting up SAM for first run..."
    
    # Check if this is a first-time setup
    if [ ! -f "/app/security/setup_status.json" ]; then
        echo "🔧 First-time setup detected"
        
        # Create minimal setup status to prevent setup loops
        mkdir -p /app/security
        cat > /app/security/setup_status.json << 'EOF'
{
    "setup_completed": false,
    "docker_mode": true,
    "setup_timestamp": null,
    "master_password_set": false,
    "encryption_enabled": false
}
EOF
        
        echo "✅ First-time setup prepared"
    fi
}

# Main initialization sequence
main() {
    echo "🚀 Starting SAM initialization sequence..."
    
    # Step 1: Initialize directories
    initialize_sam_directories
    
    # Step 2: Initialize configuration
    initialize_sam_config
    
    # Step 3: Wait for dependent services
    if [ "${SAM_ENVIRONMENT:-development}" = "production" ]; then
        wait_for_service "redis" "6379" "Redis"
        wait_for_service "chroma" "8000" "ChromaDB"
    fi
    
    # Step 4: Setup SAM for first run
    setup_sam_first_run
    
    # Step 5: Run health check
    if ! health_check; then
        echo "❌ Health check failed, exiting..."
        exit 1
    fi
    
    echo "✅ SAM initialization complete!"
    echo "🌟 Starting SAM application..."
    echo "=================================="
    
    # Execute the main command
    exec "$@"
}

# Handle signals gracefully
trap 'echo "🛑 Received shutdown signal, stopping SAM..."; exit 0' SIGTERM SIGINT

# Run main function
main "$@"
