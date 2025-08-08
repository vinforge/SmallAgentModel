#!/bin/bash
# SAM Docker Release Builder
# Creates production-ready Docker images for distribution

set -e

# Configuration
DOCKER_REGISTRY="ghcr.io"
GITHUB_ORG="forge-1825"
IMAGE_NAME="sam"
VERSION=${1:-"latest"}
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}=================================="
    echo -e "🐳 SAM Docker Release Builder"
    echo -e "==================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Validate environment
validate_environment() {
    print_info "Validating build environment..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    # Check if we're in the right directory
    if [ ! -f "Dockerfile" ]; then
        print_error "Dockerfile not found. Run from SAM root directory."
        exit 1
    fi
    
    # Check if we're in a git repository
    if [ ! -d ".git" ]; then
        print_error "Not in a git repository"
        exit 1
    fi
    
    print_success "Environment validation passed"
}

# Build multi-architecture images
build_images() {
    print_info "Building SAM Docker images..."
    
    # Create builder instance for multi-arch builds
    docker buildx create --name sam-builder --use 2>/dev/null || docker buildx use sam-builder
    
    # Build arguments
    BUILD_ARGS="--build-arg BUILD_DATE=${BUILD_DATE} --build-arg GIT_COMMIT=${GIT_COMMIT} --build-arg VERSION=${VERSION}"
    
    # Image tags
    FULL_IMAGE_NAME="${DOCKER_REGISTRY}/${GITHUB_ORG}/${IMAGE_NAME}"
    
    # Build for multiple architectures
    print_info "Building multi-architecture images..."
    docker buildx build \
        --platform linux/amd64,linux/arm64 \
        ${BUILD_ARGS} \
        -t "${FULL_IMAGE_NAME}:${VERSION}" \
        -t "${FULL_IMAGE_NAME}:latest" \
        --push \
        .
    
    print_success "Multi-architecture images built and pushed"
}

# Build local development image
build_local() {
    print_info "Building local development image..."
    
    docker build \
        --build-arg BUILD_DATE="${BUILD_DATE}" \
        --build-arg GIT_COMMIT="${GIT_COMMIT}" \
        --build-arg VERSION="${VERSION}" \
        -t "sam:${VERSION}" \
        -t "sam:latest" \
        .
    
    print_success "Local image built: sam:${VERSION}"
}

# Create release artifacts
create_release_artifacts() {
    print_info "Creating release artifacts..."
    
    # Create release directory
    RELEASE_DIR="release/sam-docker-${VERSION}"
    mkdir -p "${RELEASE_DIR}"
    
    # Copy essential files for Docker deployment
    cp docker-compose.yml "${RELEASE_DIR}/"
    cp docker/manage_sam.sh "${RELEASE_DIR}/"
    cp docker/sam_docker_config.json "${RELEASE_DIR}/"
    cp DOCKER_DEPLOYMENT_GUIDE.md "${RELEASE_DIR}/"
    cp LICENSE "${RELEASE_DIR}/"
    
    # Create simplified docker-compose for release
    cat > "${RELEASE_DIR}/docker-compose.yml" << EOF
version: '3.8'

services:
  # SAM Main Application
  sam-app:
    image: ${DOCKER_REGISTRY}/${GITHUB_ORG}/${IMAGE_NAME}:${VERSION}
    container_name: sam-main
    restart: unless-stopped
    ports:
      - "8502:8502"  # Secure Streamlit App
      - "8501:8501"  # Memory Control Center
      - "8503:8503"  # Welcome Setup
    volumes:
      # Persistent data volumes
      - sam_data:/app/data
      - sam_memory:/app/memory_store
      - sam_logs:/app/logs
      - sam_chroma:/app/chroma_db
      - sam_uploads:/app/uploads
      - sam_cache:/app/cache
      - sam_backups:/app/backups
      - sam_security:/app/security
    environment:
      - SAM_DOCKER=true
      - SAM_ENVIRONMENT=production
      - SAM_DATA_DIR=/app/data
      - SAM_LOGS_DIR=/app/logs
      - REDIS_URL=redis://redis:6379/0
      - CHROMA_HOST=chroma
      - CHROMA_PORT=8000
    depends_on:
      - redis
      - chroma
    networks:
      - sam-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8502/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  # Redis for session management and caching
  redis:
    image: redis:7-alpine
    container_name: sam-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - sam_redis:/data
    command: redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy allkeys-lru
    networks:
      - sam-network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 5s
      retries: 3

  # ChromaDB Vector Database
  chroma:
    image: chromadb/chroma:latest
    container_name: sam-chroma
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - sam_chroma_data:/chroma/chroma
    environment:
      - CHROMA_SERVER_HOST=0.0.0.0
      - CHROMA_SERVER_HTTP_PORT=8000
      - PERSIST_DIRECTORY=/chroma/chroma
    networks:
      - sam-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
      interval: 30s
      timeout: 10s
      retries: 3

# Named volumes for data persistence
volumes:
  sam_data:
    driver: local
  sam_memory:
    driver: local
  sam_logs:
    driver: local
  sam_chroma:
    driver: local
  sam_chroma_data:
    driver: local
  sam_uploads:
    driver: local
  sam_cache:
    driver: local
  sam_backups:
    driver: local
  sam_security:
    driver: local
  sam_redis:
    driver: local

# Networks
networks:
  sam-network:
    driver: bridge
EOF

    # Create quick start script
    cat > "${RELEASE_DIR}/quick_start.sh" << 'EOF'
#!/bin/bash
# SAM Docker Quick Start Script

set -e

echo "🐳 SAM Docker Quick Start"
echo "========================="

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose found"

# Pull images
echo "📥 Pulling SAM Docker images..."
docker-compose pull

# Start services
echo "🚀 Starting SAM services..."
docker-compose up -d

echo "⏳ Waiting for services to start..."
sleep 15

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    echo "✅ SAM is now running!"
    echo ""
    echo "🌟 Access SAM at:"
    echo "   Main Interface:     http://localhost:8502"
    echo "   Memory Center:      http://localhost:8501"
    echo "   Setup Page:         http://localhost:8503"
    echo ""
    echo "📚 For more information, see DOCKER_DEPLOYMENT_GUIDE.md"
else
    echo "❌ Failed to start SAM services"
    docker-compose logs
    exit 1
fi
EOF

    chmod +x "${RELEASE_DIR}/quick_start.sh"
    chmod +x "${RELEASE_DIR}/manage_sam.sh"
    
    # Create README for Docker release
    cat > "${RELEASE_DIR}/README.md" << EOF
# SAM Docker Release

This is the containerized version of SAM (Secure AI Memory) - an advanced AI assistant with human-like conceptual understanding.

## Quick Start

1. **Prerequisites**: Install Docker and Docker Compose
2. **Start SAM**: Run \`./quick_start.sh\`
3. **Access**: Open http://localhost:8502

## What's Included

- \`docker-compose.yml\` - Complete SAM stack configuration
- \`quick_start.sh\` - One-command startup script
- \`manage_sam.sh\` - Advanced management tools
- \`DOCKER_DEPLOYMENT_GUIDE.md\` - Complete documentation

## System Requirements

- **RAM**: 4GB minimum, 8GB+ recommended
- **Storage**: 10GB+ free space
- **Ports**: 8502, 8501, 8503, 6379, 8000

## Support

- **Documentation**: See DOCKER_DEPLOYMENT_GUIDE.md
- **Issues**: https://github.com/forge-1825/SAM/issues
- **License**: See LICENSE file

## Version Information

- **Version**: ${VERSION}
- **Build Date**: ${BUILD_DATE}
- **Git Commit**: ${GIT_COMMIT}
EOF

    # Create archive
    cd release
    tar -czf "sam-docker-${VERSION}.tar.gz" "sam-docker-${VERSION}/"
    cd ..
    
    print_success "Release artifacts created in release/sam-docker-${VERSION}/"
}

# Generate release notes
generate_release_notes() {
    print_info "Generating release notes..."
    
    cat > "release/RELEASE_NOTES_${VERSION}.md" << EOF
# SAM Docker Release ${VERSION}

## 🐳 Docker Container Version

This release provides SAM as a containerized application for easy deployment and distribution.

### 📦 What's New in Docker Version

- **Complete containerization** of SAM application
- **Multi-service architecture** with Redis and ChromaDB
- **Production-ready** configuration with health checks
- **Persistent data volumes** for data safety
- **One-command deployment** with quick start script
- **Multi-architecture support** (AMD64, ARM64)

### 🚀 Quick Start

\`\`\`bash
# Download and extract
wget https://github.com/forge-1825/SAM/releases/download/${VERSION}/sam-docker-${VERSION}.tar.gz
tar -xzf sam-docker-${VERSION}.tar.gz
cd sam-docker-${VERSION}

# Start SAM
./quick_start.sh

# Access at http://localhost:8502
\`\`\`

### 🏗️ Architecture

- **SAM Main App**: Streamlit application (Port 8502)
- **Memory Center**: Advanced memory management (Port 8501)
- **Setup Interface**: First-time setup (Port 8503)
- **Redis**: Session and cache management
- **ChromaDB**: Vector database for embeddings

### 💾 Data Persistence

All user data is preserved in Docker volumes:
- Application data and documents
- Memory store and knowledge base
- Security configurations
- Logs and cache

### 🔧 Management

Use the included \`manage_sam.sh\` script for:
- Starting/stopping services
- Viewing logs and status
- Creating backups
- Updating SAM

### 📋 System Requirements

- **OS**: Linux, macOS, Windows (with Docker)
- **RAM**: 4GB minimum, 8GB+ recommended
- **Storage**: 10GB+ free space
- **Docker**: Version 20.10+
- **Docker Compose**: Version 2.0+

### 🔒 Security Features

- Non-root container execution
- Isolated network configuration
- Encrypted data storage
- Secure session management

### 📚 Documentation

Complete documentation available in \`DOCKER_DEPLOYMENT_GUIDE.md\`

### 🐛 Known Issues

None at this time.

### 🔄 Upgrade Path

From previous versions:
1. Backup your data: \`./manage_sam.sh backup\`
2. Stop services: \`./manage_sam.sh stop\`
3. Update to new version
4. Start services: \`./manage_sam.sh start\`

---

**Build Information**:
- Version: ${VERSION}
- Build Date: ${BUILD_DATE}
- Git Commit: ${GIT_COMMIT}
- Architectures: linux/amd64, linux/arm64
EOF

    print_success "Release notes generated"
}

# Main build process
main() {
    print_header
    
    if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
        echo "Usage: $0 [version] [--local-only]"
        echo ""
        echo "Options:"
        echo "  version      Version tag (default: latest)"
        echo "  --local-only Build only local image, don't push"
        echo ""
        echo "Examples:"
        echo "  $0 v1.0.0              # Build and push v1.0.0"
        echo "  $0 v1.0.0 --local-only # Build v1.0.0 locally only"
        exit 0
    fi
    
    validate_environment
    
    if [ "$2" = "--local-only" ]; then
        print_info "Building local image only..."
        build_local
    else
        print_info "Building and pushing release images..."
        build_images
    fi
    
    create_release_artifacts
    generate_release_notes
    
    print_success "🎉 SAM Docker release ${VERSION} build complete!"
    echo ""
    print_info "Next steps:"
    echo "  1. Test the release: cd release/sam-docker-${VERSION} && ./quick_start.sh"
    echo "  2. Create GitHub release with release/sam-docker-${VERSION}.tar.gz"
    echo "  3. Upload release notes from release/RELEASE_NOTES_${VERSION}.md"
}

# Run main function
main "$@"
