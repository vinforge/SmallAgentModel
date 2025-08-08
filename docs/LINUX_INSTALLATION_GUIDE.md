# 🐧 SAM Linux Installation Guide

**Complete guide for installing SAM on Linux systems with PEP 668 compliance**

---

## 📋 **Prerequisites**

- **Python 3.10+** (tested with Python 3.12)
- **git** for cloning the repository
- **4GB+ RAM** recommended
- **Internet connection** for model downloads
- **[Ollama](https://ollama.ai)** installed and running

### **Install Prerequisites (Ubuntu/Debian)**

```bash
sudo apt update
sudo apt install python3 python3-venv python3-dev build-essential git curl
```

### **Install Ollama**

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &  # Start Ollama service
```

---

## 🚀 **Installation Methods**

### **Method 1: Automated Script (Recommended)**

```bash
# Clone repository
git clone https://github.com/vinforge/SmallAgentModel.git
cd SmallAgentModel

# Make script executable and run
chmod +x start_sam.sh
./start_sam.sh
```

**The script automatically:**
- ✅ Creates virtual environment
- ✅ Installs dependencies
- ✅ Handles PEP 668 restrictions
- ✅ Starts SAM

### **Method 2: Manual Installation**

```bash
# 1. Clone repository
git clone https://github.com/vinforge/SmallAgentModel.git
cd SmallAgentModel

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install core dependencies
pip install streamlit==1.42.0 cryptography>=41.0.0,<43.0.0 numpy pandas requests

# 5. Set up AI models
python setup_models.py

# 6. Set up encryption
python setup_encryption.py

# 7. Start SAM
python start_sam.py
```

### **Method 3: Linux Preparation Script**

```bash
# Clone repository
git clone https://github.com/vinforge/SmallAgentModel.git
cd SmallAgentModel

# Run Linux preparation script
python3 prepare_linux.py

# Follow the setup prompts
python setup_models.py
python setup_encryption.py
python start_sam.py
```

---

## 🔧 **Troubleshooting**

### **PEP 668 "Externally Managed Environment" Error**

**Problem:**
```
error: externally-managed-environment

× This environment is externally managed
╰─> To install Python packages system-wide, try apt install
    python3-xyz, where xyz is the package you are trying to
    install.
```

**Solution:**
Always use virtual environments on modern Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install [packages]
```

### **Virtual Environment Creation Fails**

**Problem:**
```
The virtual environment was not created successfully
```

**Solution:**
Install python3-venv:

```bash
sudo apt update
sudo apt install python3-venv python3-dev
```

### **Permission Denied Errors**

**Problem:**
```
Permission denied: '/usr/local/lib/python3.x/site-packages'
```

**Solution:**
Never use `sudo pip install`. Use virtual environments:

```bash
# ❌ Don't do this
sudo pip install streamlit

# ✅ Do this instead
source .venv/bin/activate
pip install streamlit
```

### **Missing Build Dependencies**

**Problem:**
```
error: Microsoft Visual C++ 14.0 is required
error: Failed building wheel for cryptography
```

**Solution:**
Install build tools:

```bash
sudo apt install build-essential python3-dev libffi-dev libssl-dev
```

### **Ollama Not Found**

**Problem:**
```
ollama: command not found
```

**Solution:**
Install Ollama:

```bash
curl -fsSL https://ollama.ai/install.sh | sh
# Or manually download from https://ollama.ai/download
```

### **Port Already in Use**

**Problem:**
```
OSError: [Errno 98] Address already in use
```

**Solution:**
Find and kill the process using the port:

```bash
# Find process using port 8502
sudo lsof -i :8502

# Kill the process (replace PID with actual process ID)
kill -9 [PID]

# Or use a different port
python start_sam.py --port 8503
```

---

## 🔄 **Restarting SAM**

After initial setup, restart SAM with:

```bash
cd SmallAgentModel
source .venv/bin/activate
python start_sam.py
```

**Or use the launcher script:**

```bash
./start_sam.sh
```

---

## 🎯 **Distribution-Specific Notes**

### **Ubuntu 22.04+ / Debian 12+**
- PEP 668 enforced - virtual environment required
- Use `python3-venv` package

### **CentOS/RHEL/Fedora**
```bash
# Install prerequisites
sudo dnf install python3 python3-venv python3-devel gcc git

# Or on older systems
sudo yum install python3 python3-venv python3-devel gcc git
```

### **Arch Linux**
```bash
# Install prerequisites
sudo pacman -S python python-virtualenv base-devel git
```

### **Alpine Linux**
```bash
# Install prerequisites
sudo apk add python3 python3-dev py3-virtualenv build-base git
```

---

## 🌟 **Performance Tips**

### **Use System Packages When Possible**
```bash
# Install heavy packages via system package manager first
sudo apt install python3-numpy python3-pandas

# Then install remaining packages in virtual environment
source .venv/bin/activate
pip install streamlit cryptography
```

### **Enable GPU Acceleration (NVIDIA)**
```bash
# Install NVIDIA drivers and CUDA
sudo apt install nvidia-driver-535 nvidia-cuda-toolkit

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🆘 **Getting Help**

### **Check Logs**
```bash
# SAM logs
tail -f logs/sam.log

# System logs
journalctl -f
```

### **Verify Installation**
```bash
source .venv/bin/activate
python -c "import streamlit, cryptography, numpy, pandas; print('All packages imported successfully')"
```

### **Common Commands**
```bash
# Check Python version
python --version

# Check virtual environment
which python

# List installed packages
pip list

# Check SAM status
ps aux | grep python
```

---

## 🎉 **Success!**

Once installed, SAM will be available at:
- **Main Interface**: http://localhost:8502
- **Memory Interface**: http://localhost:8501

**Your Linux SAM installation is complete and PEP 668 compliant!** 🐧✨
