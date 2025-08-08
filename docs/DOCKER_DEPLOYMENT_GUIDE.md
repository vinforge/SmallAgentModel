# 🐳 SAM Docker Deployment Guide

Complete guide for deploying SAM using Docker containers for production-ready, scalable deployments.

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Management](#management)
- [Production Deployment](#production-deployment)
- [Troubleshooting](#troubleshooting)

## 🌟 Overview

SAM Docker deployment provides:

- **🔧 Consistent Environment**: Same setup across development, staging, and production
- **📦 Easy Deployment**: One-command deployment with Docker Compose
- **🚀 Scalability**: Horizontal scaling with container orchestration
- **🔒 Security**: Isolated processes and controlled resource access
- **💾 Data Persistence**: Persistent volumes for data and configurations
- **🔄 Easy Updates**: Rolling updates and rollbacks

## 📋 Prerequisites

### System Requirements

- **OS**: Linux, macOS, or Windows with WSL2
- **RAM**: Minimum 4GB, Recommended 8GB+
- **Storage**: Minimum 10GB free space
- **CPU**: 2+ cores recommended

### Software Requirements

#### **Windows**
1. **Install Docker Desktop**:
   - Download from: https://docs.docker.com/desktop/windows/
   - Run the installer and restart your computer
   - Start Docker Desktop from Start Menu
   - Docker Compose is included with Docker Desktop

2. **Verify Installation** (PowerShell):
   ```powershell
   docker --version
   docker-compose --version
   ```

#### **Linux**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker --version
docker-compose --version
```

#### **macOS**
1. **Install Docker Desktop**:
   - Download from: https://docs.docker.com/desktop/mac/
   - Drag Docker to Applications folder
   - Start Docker Desktop
   - Docker Compose is included with Docker Desktop

2. **Verify Installation** (Terminal):
   ```bash
   docker --version
   docker-compose --version
   ```

## 🚀 Quick Start

### 1. Download SAM Docker Release

#### **Windows (PowerShell)**
```powershell
# Download the latest release
Invoke-WebRequest -Uri "https://github.com/forge-1825/SAM/releases/latest/download/sam-docker-latest.tar.gz" -OutFile "sam-docker-latest.tar.gz"

# Extract
tar -xzf sam-docker-latest.tar.gz
cd sam-docker-latest
```

#### **Linux/macOS (Terminal)**
```bash
# Download the latest release
wget https://github.com/forge-1825/SAM/releases/latest/download/sam-docker-latest.tar.gz
# OR: curl -L -o sam-docker-latest.tar.gz https://github.com/forge-1825/SAM/releases/latest/download/sam-docker-latest.tar.gz

# Extract
tar -xzf sam-docker-latest.tar.gz
cd sam-docker-latest
```

### 2. Start SAM

#### **Windows**
```batch
# Run the Windows batch file
quick_start.bat

# OR use PowerShell/WSL
./quick_start.sh
```

#### **Linux/macOS**
```bash
# Run the startup script
./quick_start.sh
```

### 3. Access SAM

- **Main Interface**: http://localhost:8502
- **Memory Control Center**: http://localhost:8501  
- **Setup/Welcome**: http://localhost:8503
- **Health Check**: http://localhost:8502/health

### 4. First-Time Setup

1. Navigate to http://localhost:8503 for initial setup
2. Create master password
3. Configure SAM Pro key (optional)
4. Complete security setup

## 🏗️ Architecture

### Container Services

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
│  (Reverse Proxy)│
│   Port: 80/443  │
└─────────────────┘
```

### Data Persistence

- **sam_data**: Application data and documents
- **sam_memory**: Memory store and knowledge base
- **sam_chroma**: Vector database storage
- **sam_security**: Encryption keys and security configs
- **sam_logs**: Application logs
- **sam_cache**: Temporary cache data

## ⚙️ Configuration

### Environment Variables

```bash
# Core settings
SAM_DOCKER=true
SAM_ENVIRONMENT=production
SAM_DATA_DIR=/app/data
SAM_LOGS_DIR=/app/logs

# Service connections
REDIS_URL=redis://redis:6379/0
CHROMA_HOST=chroma
CHROMA_PORT=8000

# Security
SAM_MASTER_PASSWORD=your_secure_password
SAM_ENCRYPTION_KEY=your_encryption_key
```

### Custom Configuration

Edit `docker/sam_docker_config.json` to customize:

```json
{
  "environment": "production",
  "security": {
    "encryption_enabled": true,
    "auto_setup": false
  },
  "performance": {
    "memory_limit": "4GB",
    "cpu_limit": "2",
    "worker_processes": 2
  }
}
```

## 🛠️ Management

### Using Management Script

```bash
# Start services
./docker/manage_sam.sh start

# Check status
./docker/manage_sam.sh status

# View logs
./docker/manage_sam.sh logs
./docker/manage_sam.sh logs sam-app

# Backup data
./docker/manage_sam.sh backup

# Restore from backup
./docker/manage_sam.sh restore ./backups/sam_backup_20240101_120000

# Update SAM
./docker/manage_sam.sh update

# Stop services
./docker/manage_sam.sh stop
```

### Manual Docker Commands

```bash
# Build image
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f sam-app

# Scale services
docker-compose up -d --scale sam-app=3

# Stop services
docker-compose down
```

## 🌐 Production Deployment

### 1. Enable Production Profile

```bash
# Start with Nginx reverse proxy
docker-compose --profile production up -d
```

### 2. SSL/HTTPS Setup

```bash
# Create SSL directory
mkdir -p docker/ssl

# Add your SSL certificates
cp your_cert.pem docker/ssl/cert.pem
cp your_key.pem docker/ssl/key.pem

# Update nginx.conf to enable HTTPS
# Uncomment HTTPS server block in docker/nginx.conf
```

### 3. Environment-Specific Configs

```bash
# Production environment file
cat > .env.production << EOF
SAM_ENVIRONMENT=production
SAM_DEBUG=false
SAM_LOG_LEVEL=WARNING
SAM_MAX_WORKERS=4
SAM_REDIS_URL=redis://redis:6379/0
EOF

# Load environment
docker-compose --env-file .env.production up -d
```

### 4. Monitoring and Logging

```bash
# View resource usage
docker stats

# Monitor logs
docker-compose logs -f --tail=100

# Health checks
curl http://localhost:8502/health
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Check what's using the port
sudo lsof -i :8502

# Kill the process or change port in docker-compose.yml
```

#### 2. Permission Denied
```bash
# Fix Docker permissions
sudo usermod -aG docker $USER
newgrp docker

# Fix file permissions
sudo chown -R $USER:$USER ./docker
```

#### 3. Out of Memory
```bash
# Check memory usage
docker stats

# Increase memory limits in docker-compose.yml
# Or add swap space to your system
```

#### 4. Container Won't Start
```bash
# Check logs for errors
docker-compose logs sam-app

# Rebuild image
docker-compose build --no-cache sam-app

# Reset volumes if needed
docker-compose down -v
```

### Debug Mode

```bash
# Start in debug mode
SAM_DEBUG=true docker-compose up

# Access container shell
docker-compose exec sam-app bash

# Check container health
docker-compose exec sam-app python -c "import secure_streamlit_app; print('OK')"
```

### Performance Tuning

```bash
# Optimize for production
echo 'vm.max_map_count=262144' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

# Monitor performance
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

## 📊 Monitoring

### Health Checks

```bash
# Application health
curl http://localhost:8502/health

# Service health
docker-compose ps
docker-compose exec redis redis-cli ping
docker-compose exec chroma curl http://localhost:8000/api/v1/heartbeat
```

### Backup Strategy

```bash
# Automated daily backups
echo "0 2 * * * cd /path/to/SAM && ./docker/manage_sam.sh backup" | crontab -

# Backup to remote storage
./docker/manage_sam.sh backup
rsync -av ./backups/ user@backup-server:/backups/sam/
```

## 🔄 Updates and Maintenance

### Rolling Updates

```bash
# Update with zero downtime
./docker/manage_sam.sh update

# Manual rolling update
docker-compose pull
docker-compose up -d --no-deps sam-app
```

### Maintenance Mode

```bash
# Enable maintenance mode
docker-compose stop sam-app
# Deploy maintenance page via nginx

# Perform maintenance
./docker/manage_sam.sh backup
# Update configurations, etc.

# Resume service
docker-compose start sam-app
```

---

## 📞 Support

For issues and questions:
- **GitHub Issues**: https://github.com/forge-1825/SAM/issues
- **Documentation**: Check README.md and other guides
- **Community**: Join our community discussions

**Happy Deploying! 🚀**
