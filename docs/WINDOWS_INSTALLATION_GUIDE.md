# 🪟 SAM Docker for Windows - Complete Installation Guide

This guide provides Windows-specific instructions for installing and running SAM using Docker containers.

## 📋 Prerequisites

### System Requirements
- **Windows 10/11** (64-bit)
- **4GB RAM** minimum (8GB+ recommended)
- **10GB+ free disk space**
- **Internet connection** for downloading

### Required Software
1. **Docker Desktop for Windows**
   - Download: https://docs.docker.com/desktop/windows/
   - Includes Docker Compose automatically
   - Requires Windows Subsystem for Linux (WSL2)

## 🚀 Installation Steps

### Step 1: Install Docker Desktop

1. **Download Docker Desktop**:
   - Visit: https://docs.docker.com/desktop/windows/
   - Click "Download for Windows"
   - Run the installer (`Docker Desktop Installer.exe`)

2. **Installation Process**:
   - Accept the license agreement
   - Choose "Use WSL 2 instead of Hyper-V" (recommended)
   - Complete the installation
   - **Restart your computer** when prompted

3. **Start Docker Desktop**:
   - Open Docker Desktop from Start Menu
   - Wait for it to start completely (whale icon in system tray)
   - Accept any terms of service

4. **Verify Installation**:
   Open PowerShell and run:
   ```powershell
   docker --version
   docker-compose --version
   ```

### Step 2: Download SAM Docker Release

#### **Option A: PowerShell Download (Recommended)**
```powershell
# Open PowerShell as Administrator (optional but recommended)
# Navigate to your desired folder (e.g., Desktop)
cd $env:USERPROFILE\Desktop

# Download SAM Docker release
Invoke-WebRequest -Uri "https://github.com/forge-1825/SAM/releases/download/v1.0.0-docker/sam-docker-v1.0.0-docker.tar.gz" -OutFile "sam-docker-v1.0.0-docker.tar.gz"

# Extract the archive
tar -xzf sam-docker-v1.0.0-docker.tar.gz

# Navigate to the extracted folder
cd sam-docker-v1.0.0-docker
```

#### **Option B: Manual Download**
1. Visit: https://github.com/forge-1825/SAM/releases/download/v1.0.0-docker/sam-docker-v1.0.0-docker.tar.gz
2. Save the file to your Desktop or preferred location
3. Right-click the file and extract using:
   - Built-in Windows extractor
   - 7-Zip (if installed)
   - WinRAR (if installed)
4. Open the extracted `sam-docker-v1.0.0-docker` folder

### Step 3: Start SAM

#### **Option A: Windows Batch File (Easiest)**
1. **Double-click** `quick_start.bat` in the extracted folder
2. The script will:
   - Check if Docker is running
   - Download required images
   - Start SAM services
   - Show access URLs

#### **Option B: PowerShell/Command Prompt**
```powershell
# In the sam-docker-v1.0.0-docker folder
./quick_start.sh
```

#### **Option C: Manual Docker Commands**
```powershell
# Pull images
docker-compose pull

# Start services
docker-compose up -d

# Check status
docker-compose ps
```

### Step 4: Access SAM

Once started, SAM will be available at:
- **Main Interface**: http://localhost:8502
- **Memory Control Center**: http://localhost:8501
- **Setup Page**: http://localhost:8503

## 🛠️ Management Commands

### Using the Management Script
```powershell
# Start services
./manage_sam.sh start

# Stop services
./manage_sam.sh stop

# Check status
./manage_sam.sh status

# View logs
./manage_sam.sh logs

# Create backup
./manage_sam.sh backup

# Update SAM
./manage_sam.sh update
```

### Direct Docker Commands
```powershell
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs

# Check status
docker-compose ps

# Restart services
docker-compose restart
```

## 🔧 Troubleshooting

### Common Issues

#### **1. "Docker is not running"**
**Solution**:
- Open Docker Desktop from Start Menu
- Wait for the whale icon to appear in system tray
- Ensure it shows "Docker Desktop is running"

#### **2. "Port already in use"**
**Solution**:
```powershell
# Check what's using the ports
netstat -ano | findstr :8502
netstat -ano | findstr :8501

# Kill the process using the port (replace PID with actual process ID)
taskkill /PID <PID> /F
```

#### **3. "Cannot connect to Docker daemon"**
**Solutions**:
- Restart Docker Desktop
- Restart your computer
- Check Windows Services for Docker services
- Reinstall Docker Desktop if needed

#### **4. "WSL 2 installation is incomplete"**
**Solution**:
- Install WSL 2: https://docs.microsoft.com/en-us/windows/wsl/install
- Restart computer
- Start Docker Desktop again

#### **5. "Insufficient disk space"**
**Solution**:
- Free up at least 10GB of disk space
- Clean Docker images: `docker system prune -a`

### Performance Tips

1. **Allocate More Resources**:
   - Open Docker Desktop Settings
   - Go to Resources → Advanced
   - Increase Memory to 4GB+ and CPU to 2+ cores

2. **Enable File Sharing**:
   - Go to Resources → File Sharing
   - Ensure your drive is shared

3. **Optimize Windows**:
   - Close unnecessary applications
   - Disable Windows Defender real-time scanning for Docker folder (temporarily)

## 📊 System Monitoring

### Check Resource Usage
```powershell
# View container resource usage
docker stats

# View Docker Desktop resource usage
# Check Docker Desktop → Dashboard
```

### View Logs
```powershell
# All services
docker-compose logs

# Specific service
docker-compose logs sam-app
docker-compose logs redis
docker-compose logs chroma
```

## 🔄 Updates and Maintenance

### Update SAM
```powershell
# Stop current version
docker-compose down

# Download new release (repeat Step 2 with new version)
# Start new version
./quick_start.bat
```

### Backup Data
```powershell
# Create backup
./manage_sam.sh backup

# Backup location will be shown in output
```

### Clean Up
```powershell
# Remove old containers and images
docker system prune -a

# Remove SAM completely
docker-compose down -v
docker rmi $(docker images | grep sam | awk '{print $3}')
```

## 🆘 Getting Help

### Documentation
- **Complete Guide**: `DOCKER_DEPLOYMENT_GUIDE.md`
- **Docker README**: `README_DOCKER.md`
- **GitHub**: https://github.com/forge-1825/SAM

### Support Channels
- **GitHub Issues**: https://github.com/forge-1825/SAM/issues
- **Email**: vin@forge1825.net

### Community
- **Discussions**: GitHub Discussions
- **Updates**: Watch the repository for releases

## 💡 Tips for Windows Users

1. **Use PowerShell**: More powerful than Command Prompt
2. **Enable WSL 2**: Better performance for Docker
3. **Windows Terminal**: Better terminal experience
4. **File Paths**: Use forward slashes (/) in Docker commands
5. **Antivirus**: Add Docker folder to exclusions for better performance

---

**🎉 You're ready to experience SAM on Windows!** 

The containerized version provides the same powerful AI capabilities with the convenience of Windows-native installation. 🚀
