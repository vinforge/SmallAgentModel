# 🐳 SAM Docker Distribution

**SAM (Secure AI Memory)** - Advanced AI assistant with human-like conceptual understanding, now available as a containerized application for easy deployment.

## 🚀 Quick Start

### Prerequisites
- **Docker** (version 20.10+)
- **Docker Compose** (version 2.0+)
- **4GB RAM** minimum (8GB+ recommended)
- **10GB+ free disk space**

### One-Command Deployment

#### **Windows (PowerShell)**
```powershell
# Download the latest release
Invoke-WebRequest -Uri "https://github.com/forge-1825/SAM/releases/latest/download/sam-docker-latest.tar.gz" -OutFile "sam-docker-latest.tar.gz"

# Extract and start
tar -xzf sam-docker-latest.tar.gz
cd sam-docker-latest
./quick_start.sh
```

#### **Linux/macOS (Terminal)**
```bash
# Download the latest release
wget https://github.com/forge-1825/SAM/releases/latest/download/sam-docker-latest.tar.gz
# OR use curl if wget is not available
curl -L -o sam-docker-latest.tar.gz https://github.com/forge-1825/SAM/releases/latest/download/sam-docker-latest.tar.gz

# Extract and start
tar -xzf sam-docker-latest.tar.gz
cd sam-docker-latest
./quick_start.sh
```

#### **Alternative: Manual Download**
1. Visit: https://github.com/forge-1825/SAM/releases/latest
2. Download `sam-docker-latest.tar.gz`
3. Extract the archive
4. Run `quick_start.sh` (or `quick_start.bat` on Windows)

**That's it!** SAM will be available at:
- **Main Interface**: http://localhost:8502
- **Memory Control Center**: http://localhost:8501
- **Setup Page**: http://localhost:8503

## 📦 What You Get

### Complete AI Assistant Stack
- **🧠 SAM Core**: Advanced AI with memory capabilities
- **💾 Vector Database**: ChromaDB for semantic search
- **⚡ Redis Cache**: Fast session management
- **🔒 Security**: Encrypted data storage
- **📊 Health Monitoring**: Built-in health checks

### Two Deployment Options
1. **Traditional Python**: Full source code installation
2. **Docker Container**: Pre-built, ready-to-run containers ← **You are here**

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   SAM Main App  │    │     Redis       │    │    ChromaDB     │
│   (Streamlit)   │◄──►│    (Cache)      │    │   (Vectors)     │
│   Port: 8502    │    │   Port: 6379    │    │   Port: 8000    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲
         │
┌─────────────────┐
│     Nginx       │
│  (Load Balancer)│
│   Port: 80/443  │
└─────────────────┘
```

## 🛠️ Management Commands

Use the included `manage_sam.sh` script for advanced operations:

```bash
# Service Management
./manage_sam.sh start          # Start all services
./manage_sam.sh stop           # Stop all services
./manage_sam.sh restart        # Restart services
./manage_sam.sh status         # Check service status

# Monitoring
./manage_sam.sh logs           # View all logs
./manage_sam.sh logs sam-app   # View specific service logs

# Data Management
./manage_sam.sh backup         # Create data backup
./manage_sam.sh restore <path> # Restore from backup

# Maintenance
./manage_sam.sh update         # Update to latest version
./manage_sam.sh cleanup        # Clean up unused resources
```

## 💾 Data Persistence

All your data is safely stored in Docker volumes:

- **📄 Documents**: Uploaded files and knowledge base
- **🧠 Memory**: Conversation history and learned concepts
- **🔐 Security**: Encryption keys and authentication
- **📊 Logs**: Application logs and diagnostics
- **⚡ Cache**: Performance optimization data

**Data survives container restarts and updates!**

## 🔧 Configuration

### Environment Variables

```bash
# Core Settings
SAM_ENVIRONMENT=production
SAM_LOG_LEVEL=INFO
SAM_MAX_WORKERS=2

# Security
SAM_MASTER_PASSWORD=your_secure_password
SAM_ENCRYPTION_ENABLED=true

# Performance
SAM_MEMORY_LIMIT=4GB
SAM_CPU_LIMIT=2
```

### Custom Configuration

Edit `sam_docker_config.json` to customize SAM behavior:

```json
{
  "performance": {
    "memory_limit": "4GB",
    "cpu_limit": "2",
    "worker_processes": 2
  },
  "security": {
    "encryption_enabled": true,
    "auto_setup": true
  },
  "features": {
    "auto_backup": true,
    "health_monitoring": true
  }
}
```

## 🌐 Production Deployment

### SSL/HTTPS Setup

```bash
# Create SSL certificates directory
mkdir -p ssl

# Add your certificates
cp your_cert.pem ssl/cert.pem
cp your_key.pem ssl/key.pem

# Start with production profile
docker-compose --profile production up -d
```

### Cloud Deployment

**AWS ECS**:
```bash
# Use the pre-built image
docker pull ghcr.io/forge-1825/sam:latest
```

**Google Cloud Run**:
```bash
gcloud run deploy sam --image ghcr.io/forge-1825/sam:latest
```

**Azure Container Instances**:
```bash
az container create --resource-group myResourceGroup \
  --name sam --image ghcr.io/forge-1825/sam:latest
```

## 🔍 Troubleshooting

### Common Issues

**Port Already in Use**:
```bash
# Check what's using the port
sudo lsof -i :8502
# Kill the process or change port in docker-compose.yml
```

**Out of Memory**:
```bash
# Check memory usage
docker stats
# Increase memory limits or add swap space
```

**Container Won't Start**:
```bash
# Check logs
./manage_sam.sh logs sam-app
# Rebuild if needed
docker-compose build --no-cache
```

### Debug Mode

```bash
# Start in debug mode
SAM_DEBUG=true docker-compose up

# Access container shell
docker-compose exec sam-app bash
```

## 📊 Monitoring

### Health Checks

```bash
# Application health
curl http://localhost:8502/health

# Service status
./manage_sam.sh status

# Resource usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

### Backup Strategy

```bash
# Manual backup
./manage_sam.sh backup

# Automated daily backups (add to crontab)
0 2 * * * cd /path/to/sam && ./manage_sam.sh backup
```

## 🔄 Updates

### Automatic Updates

```bash
# Update to latest version
./manage_sam.sh update
```

### Manual Updates

```bash
# Pull latest images
docker-compose pull

# Restart with new images
docker-compose up -d
```

## 📋 System Requirements

### Minimum Requirements
- **OS**: Linux, macOS, Windows (with Docker Desktop)
- **RAM**: 4GB
- **Storage**: 10GB free space
- **CPU**: 2 cores
- **Network**: Internet connection for initial setup

### Recommended Requirements
- **RAM**: 8GB+
- **Storage**: 50GB+ SSD
- **CPU**: 4+ cores
- **Network**: Stable broadband connection

## 🆚 Docker vs Traditional Installation

| Feature | Docker Version | Traditional Python |
|---------|----------------|-------------------|
| **Setup Time** | 5 minutes | 15-30 minutes |
| **Dependencies** | Included | Manual installation |
| **Isolation** | Complete | System-dependent |
| **Updates** | One command | Multi-step process |
| **Portability** | Run anywhere | Environment-specific |
| **Resource Usage** | Optimized | Variable |
| **Backup** | Built-in tools | Manual process |

## 📞 Support

### Documentation
- **Complete Guide**: `DOCKER_DEPLOYMENT_GUIDE.md`
- **GitHub Repository**: https://github.com/forge-1825/SAM
- **Issues**: https://github.com/forge-1825/SAM/issues

### Community
- **Discussions**: GitHub Discussions
- **Updates**: Watch the repository for releases

### Professional Support
- **Email**: vin@forge1825.net
- **Enterprise**: Custom deployment assistance available

## 📄 License

SAM is released under the MIT License. See `LICENSE` file for details.

---

## 🎉 Ready to Get Started?

```bash
# Download and start SAM in under 5 minutes
wget https://github.com/forge-1825/SAM/releases/latest/download/sam-docker-latest.tar.gz
tar -xzf sam-docker-latest.tar.gz
cd sam-docker-latest
./quick_start.sh

# Then visit: http://localhost:8502
```

**Welcome to the future of AI assistance!** 🚀
