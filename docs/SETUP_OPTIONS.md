# 🚀 SAM Setup Options for New Users

**Choose the setup method that works best for you!**

---

## 🎯 **Option 1: Interactive Script (Recommended)**

**Perfect for first-time users and those who want a guided experience.**

### **What it does:**
- ✅ **Guided setup wizard** with step-by-step instructions
- ✅ **Automatic dependency detection** and installation
- ✅ **Interactive encryption setup** with password validation
- ✅ **System optimization** and configuration
- ✅ **Verification checks** to ensure everything works
- ✅ **Beginner-friendly** with helpful explanations

### **How to use:**
```bash
# Navigate to SAM directory
cd SAM

# Run the interactive setup
python setup_sam.py
```

### **What to expect:**
1. **System Requirements Check** - Verifies Python, RAM, disk space
2. **Dependency Installation** - Installs all required packages
3. **AI Model Setup** - Guides you through Ollama installation
4. **Security Configuration** - Interactive encryption setup
5. **Configuration Wizard** - Customizes SAM for your needs
6. **Final Verification** - Ensures everything is working

**⏱️ Time:** ~10-15 minutes  
**💡 Difficulty:** Beginner-friendly

---

## ⚡ **Option 2: Quick Setup**

**For users who want to get started fast with default settings.**

### **What it does:**
- ✅ **Fast installation** with minimal prompts
- ✅ **Default configuration** (can be customized later)
- ✅ **Basic encryption** setup
- ✅ **Standard ports** and settings

### **How to use:**
```bash
# Navigate to SAM directory
cd SAM

# Run the installer and choose Quick Launch
python install_sam.py
# Then select option 2: Quick Launch
```

**⏱️ Time:** ~5 minutes  
**💡 Difficulty:** Easy

---

## 🔧 **Option 3: Manual Installation**

**For advanced users who want full control over the setup process.**

### **What it does:**
- ✅ **Complete control** over all settings
- ✅ **Custom configuration** options
- ✅ **Advanced security** settings
- ✅ **Documentation-guided** process

### **How to use:**
```bash
# 1. Install dependencies manually
pip install streamlit chromadb sentence-transformers
pip install argon2-cffi cryptography requests
pip install beautifulsoup4 PyPDF2 python-docx

# 2. Install Ollama
# Visit: https://ollama.ai/download
# Download the AI model:
ollama pull hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M

# 3. Setup encryption
python setup_encryption.py

# 4. Launch SAM
python start_sam_secure.py --mode full
```

**⏱️ Time:** ~20-30 minutes  
**💡 Difficulty:** Advanced

---

## 🔐 **Option 4: Encryption Only Setup**

**For users who already have SAM installed but need to add encryption.**

### **What it does:**
- ✅ **Encryption setup only** for existing installations
- ✅ **Master password creation** with validation
- ✅ **Keystore initialization** with enterprise-grade security
- ✅ **Migration support** for existing data

### **How to use:**
```bash
# Setup encryption for existing SAM installation
python setup_encryption.py
```

**⏱️ Time:** ~5 minutes  
**💡 Difficulty:** Easy

---

## 🎯 **Which Option Should I Choose?**

### **👶 New to SAM or AI assistants?**
**→ Choose Option 1: Interactive Script**
- Guides you through everything
- Explains each step
- Ensures optimal setup

### **⚡ Want to get started quickly?**
**→ Choose Option 2: Quick Setup**
- Minimal questions
- Default settings
- Fast deployment

### **🔧 Advanced user with specific needs?**
**→ Choose Option 3: Manual Installation**
- Full customization
- Advanced configuration
- Complete control

### **🔄 Already have SAM but need encryption?**
**→ Choose Option 4: Encryption Only**
- Adds security to existing setup
- Preserves current configuration
- Quick encryption deployment

---

## 🆘 **Need Help?**

### **📖 Documentation:**
- **Quick Start**: [`docs/QUICK_ENCRYPTION_SETUP.md`](docs/QUICK_ENCRYPTION_SETUP.md)
- **Complete Guide**: [`docs/ENCRYPTION_SETUP_GUIDE.md`](docs/ENCRYPTION_SETUP_GUIDE.md)
- **Installation Guide**: [`docs/README_SECURE_INSTALLATION.md`](docs/README_SECURE_INSTALLATION.md)
- **Main README**: [`docs/README.md`](docs/README.md)

### **🐛 Troubleshooting:**
- Check `logs/sam.log` for error details
- Ensure Python 3.8+ is installed
- Verify internet connection for downloads
- Check available disk space (5GB+ recommended)

### **💬 Common Issues:**
- **"Ollama not found"**: Install from https://ollama.ai/download
- **"Permission denied"**: Run `chmod +x setup_sam.py`
- **"Module not found"**: Run `pip install -r requirements.txt`
- **"Port in use"**: Choose different ports in configuration

---

## 🎉 **After Setup**

Once setup is complete, you can:

1. **Access SAM** at http://localhost:8502 (secure interface)
2. **Upload documents** and start chatting
3. **Explore features** like Dream Canvas and Memory Center
4. **Customize settings** through the web interface

### **🔑 Daily Usage:**
```bash
# Start SAM
python start_sam_secure.py --mode full

# Access points:
# • Secure Chat: http://localhost:8502
# • Memory Center: http://localhost:8501
# • Standard Chat: http://localhost:5001
```

---

## ⚠️ **Important Reminders**

- **🔑 Remember your master password** - No recovery possible
- **💾 Backup important documents** separately
- **🔒 Use a password manager** to store your master password
- **🏠 Keep SAM local** - Don't expose to internet without VPN
- **🔄 Regular backups** of encrypted data recommended

---

**🎯 Ready to get started? Choose your preferred option above and begin your SAM journey!**
