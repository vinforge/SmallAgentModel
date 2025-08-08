#!/bin/bash
# Script to prepare SAM Docker repository
# Creates a clean Docker-focused repository structure

set -e

echo "🐳 Preparing SAM Docker Repository"
echo "=================================="

# Create temporary directory for Docker repo
DOCKER_REPO_DIR="SAMinDocker"
rm -rf "$DOCKER_REPO_DIR"
mkdir -p "$DOCKER_REPO_DIR"

echo "📦 Copying essential Docker files..."

# Core Docker files
cp Dockerfile "$DOCKER_REPO_DIR/"
cp docker-compose.yml "$DOCKER_REPO_DIR/"
cp .dockerignore "$DOCKER_REPO_DIR/"

# Docker-specific scripts and configs
cp -r docker "$DOCKER_REPO_DIR/"

# Documentation
cp README_DOCKER.md "$DOCKER_REPO_DIR/README.md"
cp DOCKER_DEPLOYMENT_GUIDE.md "$DOCKER_REPO_DIR/"
cp WINDOWS_INSTALLATION_GUIDE.md "$DOCKER_REPO_DIR/"
cp LICENSE "$DOCKER_REPO_DIR/"

# Essential application files (minimal set for Docker)
echo "📋 Copying essential application files..."

# Main application
cp secure_streamlit_app.py "$DOCKER_REPO_DIR/"
cp requirements.txt "$DOCKER_REPO_DIR/"

# Core directories (essential only)
cp -r core "$DOCKER_REPO_DIR/"
cp -r memory "$DOCKER_REPO_DIR/"
cp -r ui "$DOCKER_REPO_DIR/"
cp -r security "$DOCKER_REPO_DIR/"
cp -r config "$DOCKER_REPO_DIR/"
cp -r utils "$DOCKER_REPO_DIR/"
cp -r reasoning "$DOCKER_REPO_DIR/"
cp -r synthesis "$DOCKER_REPO_DIR/"
cp -r multimodal "$DOCKER_REPO_DIR/"
cp -r web_retrieval "$DOCKER_REPO_DIR/"
cp -r tools "$DOCKER_REPO_DIR/"
cp -r agents "$DOCKER_REPO_DIR/"
cp -r prompts "$DOCKER_REPO_DIR/"

# Essential scripts
cp setup_sam.py "$DOCKER_REPO_DIR/"
cp start_sam.py "$DOCKER_REPO_DIR/"

# Create empty directories for Docker volumes
mkdir -p "$DOCKER_REPO_DIR/data"
mkdir -p "$DOCKER_REPO_DIR/memory_store"
mkdir -p "$DOCKER_REPO_DIR/logs"
mkdir -p "$DOCKER_REPO_DIR/uploads"
mkdir -p "$DOCKER_REPO_DIR/cache"
mkdir -p "$DOCKER_REPO_DIR/backups"

# Create Docker-specific README
cat > "$DOCKER_REPO_DIR/README.md" << 'EOF'
# 🐳 SAM Docker - Containerized AI Assistant

**SAM (Secure AI Memory)** - Advanced AI assistant with human-like conceptual understanding, packaged as a Docker container for easy deployment.

## 🚀 Quick Start

### Prerequisites
- **Docker Desktop** (Windows/macOS) or **Docker Engine** (Linux)
- **4GB RAM** minimum (8GB+ recommended)
- **10GB+ free disk space**

### One-Command Deployment

#### **Windows (PowerShell)**
```powershell
# Clone the repository
git clone https://github.com/ro0TuX777/SAMinDocker.git
cd SAMinDocker

# Start SAM
.\docker\quick_start.bat
```

#### **Linux/macOS (Terminal)**
```bash
# Clone the repository
git clone https://github.com/ro0TuX777/SAMinDocker.git
cd SAMinDocker

# Start SAM
./docker/quick_start.sh
```

**Access SAM at: http://localhost:8502**

## 📦 What's Included

- **🧠 Complete SAM AI Assistant**: Advanced AI with memory capabilities
- **🐳 Docker Container Stack**: SAM + Redis + ChromaDB
- **🪟 Cross-Platform Support**: Windows, Linux, macOS
- **📚 Comprehensive Documentation**: Complete setup and usage guides
- **🛠️ Management Tools**: Easy container management scripts

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   SAM Main App  │    │     Redis       │    │    ChromaDB     │
│   (Streamlit)   │◄──►│    (Cache)      │    │   (Vectors)     │
│   Port: 8502    │    │   Port: 6379    │    │   Port: 8000    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

**Services**:
- **SAM Main App**: Core AI assistant (Port 8502)
- **Memory Control Center**: Advanced memory management (Port 8501)
- **Setup Interface**: First-time configuration (Port 8503)
- **Redis**: Session and cache management
- **ChromaDB**: Vector database for embeddings

## 🛠️ Management

### Quick Commands
```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Check status
docker-compose ps
```

### Advanced Management
```bash
# Use the management script
./docker/manage_sam.sh start          # Start services
./docker/manage_sam.sh stop           # Stop services
./docker/manage_sam.sh status         # Check status
./docker/manage_sam.sh logs           # View logs
./docker/manage_sam.sh backup         # Create backup
./docker/manage_sam.sh update         # Update SAM
```

## 💾 Data Persistence

All data is preserved in Docker volumes:
- **Application data**: Documents and knowledge base
- **Memory store**: Conversation history and learned concepts
- **Security configs**: Encryption keys and authentication
- **Logs and cache**: Performance and diagnostic data

## 📚 Documentation

- **[Docker Deployment Guide](DOCKER_DEPLOYMENT_GUIDE.md)**: Complete deployment documentation
- **[Windows Installation Guide](WINDOWS_INSTALLATION_GUIDE.md)**: Windows-specific instructions
- **[Management Scripts](docker/)**: Container management tools

## 🔒 Security Features

- **Encrypted data storage** with secure key management
- **Isolated container environment** with controlled access
- **Non-root container execution** for enhanced security
- **Secure session management** with Redis

## 📋 System Requirements

### Minimum
- **OS**: Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+)
- **RAM**: 4GB
- **Storage**: 10GB free space
- **CPU**: 2 cores

### Recommended
- **RAM**: 8GB+
- **Storage**: 50GB+ SSD
- **CPU**: 4+ cores
- **Network**: Stable internet connection

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/ro0TuX777/SAMinDocker/issues)
- **Documentation**: Complete guides included
- **Original Project**: [SAM Main Repository](https://github.com/forge-1825/SAM)

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

---

**🎉 Experience the future of AI assistance with SAM Docker!** 🚀

*Containerized for your convenience, powered by advanced AI technology.*
EOF

# Create .gitignore for Docker repo
cat > "$DOCKER_REPO_DIR/.gitignore" << 'EOF'
# SAM Docker - Git Ignore

# Data directories (will be in Docker volumes)
data/
memory_store/
logs/
uploads/
cache/
backups/
chroma_db/

# Python cache
__pycache__/
*.py[cod]
*$py.class
*.so

# Virtual environments
venv/
env/
ENV/

# IDE files
.vscode/
.idea/
*.swp
*.swo

# OS files
.DS_Store
Thumbs.db

# Security files
security/keystore.json
security/setup_status.json
sam_pro_key.txt

# Temporary files
*.tmp
*.log

# Docker override files
docker-compose.override.yml
EOF

echo "✅ Docker repository prepared in: $DOCKER_REPO_DIR"
echo ""
echo "📋 Next steps:"
echo "   1. cd $DOCKER_REPO_DIR"
echo "   2. git init"
echo "   3. git add ."
echo "   4. git commit -m 'Initial SAM Docker repository'"
echo "   5. git remote add origin https://github.com/ro0TuX777/SAMinDocker.git"
echo "   6. git push -u origin main"
echo ""
echo "🎯 Repository structure:"
ls -la "$DOCKER_REPO_DIR"
EOF
