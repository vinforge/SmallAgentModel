# SAM Installation Guide - Lazy Provisioning
## Complete Setup Guide with Docker Auto-Management

### 🎯 **Overview**

SAM now features **Lazy Provisioning** - an intelligent system that automatically configures the optimal environment based on your system capabilities. Docker is completely optional and auto-managed when available.

---

## 🚀 **Quick Start (Recommended)**

### **For Most Users: Immediate Setup**

1. **Clone and Install**
   ```bash
   git clone https://github.com/vinforge/SmallAgentModel.git
   cd SmallAgentModel
   pip install -r requirements.txt
   ```

2. **Start SAM with Auto-Configuration**
   ```bash
   python start_sam_enhanced.py
   ```

3. **Choose Option 1**: "Start SAM immediately"

**That's it!** SAM will automatically:
- Detect your system capabilities
- Configure optimal execution mode
- Start with full CSV analysis capabilities
- Auto-manage Docker if available

---

## 📋 **Installation Options**

### **Option 1: Basic Installation (Works Everywhere)**

**Requirements:**
- Python 3.10+
- 2GB+ RAM
- No Docker required

**Setup:**
```bash
git clone https://github.com/vinforge/SmallAgentModel.git
cd SmallAgentModel
pip install -r requirements.txt
python setup_models.py
python setup_encryption.py
python start_sam.py
```

**Result:** Full CSV analysis with local execution (medium security)

### **Option 2: Enhanced Installation (Recommended)**

**Requirements:**
- Python 3.10+
- 4GB+ RAM
- Docker Desktop (optional but recommended)

**Setup:**
```bash
git clone https://github.com/vinforge/SmallAgentModel.git
cd SmallAgentModel
pip install -r requirements.txt
pip install -r requirements_docker.txt  # Optional for Docker features
python setup_models.py
python setup_encryption.py
python start_sam_enhanced.py
```

**Result:** Maximum security with Docker auto-provisioning + fallbacks

### **Option 3: Docker-First Installation**

**Requirements:**
- Python 3.10+
- 4GB+ RAM
- Docker Desktop installed and running

**Setup:**
```bash
git clone https://github.com/vinforge/SmallAgentModel.git
cd SmallAgentModel
pip install -r requirements.txt
pip install -r requirements_docker.txt

# Ensure Docker is running
docker --version
docker info

python setup_models.py
python setup_encryption.py
python start_sam_enhanced.py
```

**Result:** Full Docker mode with maximum security from start

---

## 🐳 **Docker Integration Guide**

### **Docker is Optional!**

SAM works perfectly without Docker using our Lazy Provisioning system:

- **Without Docker**: Local enhanced execution (good security + performance)
- **With Docker**: Isolated containers (maximum security)
- **Auto-Detection**: SAM automatically uses Docker when available

### **Installing Docker (Optional)**

#### **macOS:**
1. Download [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/)
2. Install and start Docker Desktop
3. Restart SAM - it will automatically detect Docker

#### **Windows:**
1. Download [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
2. Install and start Docker Desktop
3. Restart SAM - it will automatically detect Docker

#### **Linux:**
1. Follow [Docker Desktop for Linux](https://docs.docker.com/desktop/install/linux-install/) guide
2. Or install Docker Engine: `sudo apt install docker.io` (Ubuntu/Debian)
3. Start Docker service: `sudo systemctl start docker`
4. Restart SAM - it will automatically detect Docker

### **Docker Auto-Management**

When Docker is available, SAM automatically:
- ✅ Detects Docker installation
- ✅ Starts Docker daemon if needed
- ✅ Provisions sandbox containers
- ✅ Routes data analysis to secure containers
- ✅ Falls back gracefully if Docker fails

**No manual Docker management required!**

---

## 🔧 **System Requirements**

### **Minimum Requirements (Basic Mode)**
- **OS**: Windows 10+, macOS 10.15+, Linux (Ubuntu 18.04+)
- **Python**: 3.10 or higher
- **RAM**: 2GB minimum
- **Storage**: 2GB free space
- **Network**: Internet for initial setup

### **Recommended Requirements (Enhanced Mode)**
- **OS**: Windows 11, macOS 12+, Linux (Ubuntu 20.04+)
- **Python**: 3.11 or 3.12
- **RAM**: 4GB or more
- **Storage**: 5GB free space
- **Docker**: Docker Desktop (optional)

### **Optimal Requirements (Full Docker Mode)**
- **OS**: Latest versions
- **Python**: 3.12
- **RAM**: 8GB or more
- **Storage**: 10GB free space
- **Docker**: Docker Desktop with 4GB+ allocated

---

## 📊 **CSV Analysis Capabilities**

### **What You Get Out of the Box**

Regardless of your installation option, SAM provides:

- ✅ **CSV File Upload**: Through secure chat interface
- ✅ **Automatic Data Analysis**: Shape, columns, statistics
- ✅ **Correlation Detection**: Identifies relationships in data
- ✅ **Smart Suggestions**: Context-aware analysis recommendations
- ✅ **Professional Results**: Formatted analysis with insights

### **Example Capabilities**

```
Upload: employee_data.csv
Ask: "Calculate the average salary for the entire company"
Get: Professional analysis with statistics and insights
```

**Supported Analysis Types:**
- Statistical calculations (mean, median, correlation)
- Grouped analysis (by department, category)
- Data profiling and quality assessment
- Visualization recommendations
- Business intelligence insights

---

## 🔍 **Troubleshooting**

### **Common Issues**

#### **"Docker not found" Warning**
- **Solution**: This is normal! SAM works without Docker
- **Optional**: Install Docker Desktop for enhanced security
- **Result**: SAM uses local execution (still very capable)

#### **"Permission denied" on Linux**
- **Solution**: Use virtual environment or add user to docker group
- **Command**: `sudo usermod -aG docker $USER` (logout/login required)

#### **"Port already in use"**
- **Solution**: SAM auto-detects available ports
- **Manual**: Change port in startup script if needed

#### **Memory Issues**
- **Solution**: SAM automatically adjusts based on available memory
- **Recommendation**: Close other applications for better performance

### **Getting Help**

1. **Check Status**: Run `python check_lazy_provisioning_status.py`
2. **View Logs**: Check console output for detailed error messages
3. **Fallback Mode**: SAM always provides basic functionality
4. **Documentation**: See additional guides in `docs/` folder

---

## 🎯 **Deployment Modes Explained**

### **Full Docker Mode** 🐳
- **When**: Docker available + 4GB+ RAM
- **Security**: Maximum (isolated containers)
- **Features**: All capabilities including large datasets
- **Setup**: Automatic when Docker detected

### **Local Enhanced Mode** ⚡
- **When**: 2GB+ RAM, no Docker required
- **Security**: Good (restricted local execution)
- **Features**: Full CSV analysis, visualizations
- **Setup**: Default for most systems

### **Basic Mode** 📱
- **When**: Minimal system resources
- **Security**: Basic (safety checks only)
- **Features**: Essential CSV analysis
- **Setup**: Fallback for older systems

---

## 🎉 **Success Indicators**

After installation, you should see:

```
✅ SAM initialized successfully!
🔧 Mode: Local Enhanced (or Full Docker)
🔒 Security: Medium (or High)
🐳 Docker: Available (or Not required)
⚡ Fallback: Available
```

**Ready to use!** Upload CSV files and start analyzing data immediately.

---

## 📚 **Next Steps**

1. **Upload a CSV file** through the secure chat interface
2. **Ask data questions** like "What's the average salary?"
3. **Explore capabilities** with different types of analysis
4. **Optional**: Install Docker later for enhanced security
5. **Read guides** in the `docs/` folder for advanced features

**SAM adapts to your environment - no complex configuration needed!** 🚀
