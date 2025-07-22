# SAM Community Edition - Setup Guide

Welcome to SAM (Smart Assistant Memory) Community Edition Beta! This guide will help you get SAM up and running on your system.

## 🎯 Quick Start (5 minutes)

### Prerequisites
- **Python 3.8+** (Python 3.9+ recommended)
- **4GB+ RAM** (8GB+ recommended)
- **2GB+ free disk space**
- **Internet connection** (for downloading models)

### Installation Steps

1. **Download SAM**
   ```bash
   # If you have git:
   git clone https://github.com/forge-1825/SAM.git
   cd SAM
   
   # Or download and extract the ZIP file
   ```

2. **Run the Installer**
   ```bash
   python install.py
   ```
   
   The installer will:
   - Check system requirements
   - Install Python dependencies
   - Set up Ollama (local AI model)
   - Download the language model
   - Create configuration files
   - Set up directory structure

3. **Start SAM**
   ```bash
   python start_sam.py
   ```
   
   Or use the launcher script:
   ```bash
   # On Linux/macOS:
   ./start_sam.sh
   
   # On Windows:
   start_sam.bat
   ```

4. **Access SAM**
   - **Chat Interface**: http://localhost:5001
   - **Memory Control Center**: http://localhost:8501

## 🔧 Manual Installation

If the automatic installer doesn't work, follow these manual steps:

### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Install Ollama

**macOS:**
```bash
# Using Homebrew:
brew install ollama

# Or download from: https://ollama.ai/download
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
- Download from: https://ollama.ai/download
- Run the installer

### Step 3: Download the AI Model
```bash
# Start Ollama service
ollama serve

# In another terminal, download the model:
ollama pull hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M
```

### Step 4: Create Configuration
```bash
mkdir -p config logs memory_store
```

Create `config/sam_config.json`:
```json
{
  "version": "1.0.0-beta",
  "model": {
    "provider": "ollama",
    "model_name": "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M",
    "api_url": "http://localhost:11434"
  },
  "ui": {
    "chat_port": 5001,
    "memory_ui_port": 8501,
    "host": "0.0.0.0"
  },
  "features": {
    "show_thoughts": true,
    "document_upload": true,
    "memory_management": true
  }
}
```

### Step 5: Start SAM
```bash
python start_sam.py
```

## 🚀 Using SAM

### Chat Interface (Port 5001)
- **Ask Questions**: Type naturally and SAM will respond
- **Upload Documents**: Drag and drop PDF files to add to SAM's knowledge
- **View Thoughts**: Toggle the "SAM's Thoughts" button to see reasoning
- **Memory**: SAM remembers your conversations and uploaded documents

### Memory Control Center (Port 8501)
- **Browse Memories**: View all stored memories and documents
- **Search**: Find specific information in SAM's memory
- **Statistics**: See memory usage and system stats
- **Management**: Delete or organize memories

### Key Features
- **Document Processing**: Upload PDFs and SAM will remember their content
- **Persistent Memory**: SAM remembers across sessions
- **Local AI**: Everything runs on your computer (privacy-focused)
- **Thought Transparency**: See how SAM reasons through problems

## 🔧 Configuration

### Basic Settings
Edit `config/sam_config.json` to customize:

```json
{
  "ui": {
    "chat_port": 5001,        // Change web interface port
    "auto_open_browser": true  // Auto-open browser on startup
  },
  "memory": {
    "max_memories": 10000,    // Maximum memories to store
    "backend": "simple"       // Memory storage type
  },
  "features": {
    "show_thoughts": true,    // Enable thought display
    "web_search": false       // Enable web search (future)
  }
}
```

### Advanced Configuration
- **Memory Backend**: Choose between `simple`, `faiss`, or `chroma`
- **Model Settings**: Configure different AI models
- **Security**: Enable authentication for multi-user setups

## 🆘 Troubleshooting

### Common Issues

**1. "Ollama not found" Error**
```bash
# Check if Ollama is installed:
ollama --version

# If not installed, install it:
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.ai/install.sh | sh
# Windows: Download from ollama.ai
```

**2. "Model not found" Error**
```bash
# Download the required model:
ollama pull hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M
```

**3. "Port already in use" Error**
```bash
# Check what's using the port:
lsof -i :5001

# Or change the port in config/sam_config.json
```

**4. Python Import Errors**
```bash
# Reinstall dependencies:
pip install -r requirements.txt --force-reinstall
```

**5. Memory Issues**
- Restart SAM: `Ctrl+C` then `python start_sam.py`
- Check logs: `tail -f logs/sam.log`
- Clear memory: Delete files in `memory_store/` directory

### Getting Help
1. **Check Logs**: Look at `logs/sam.log` for error details
2. **System Status**: Visit http://localhost:5001/health for system status
3. **Documentation**: Read `README.md` for detailed information
4. **Reset**: Delete `memory_store/` and `config/` to start fresh

## 📊 System Requirements

### Minimum Requirements
- **CPU**: 2+ cores
- **RAM**: 4GB
- **Storage**: 2GB free space
- **OS**: Windows 10+, macOS 10.15+, Linux (Ubuntu 18.04+)

### Recommended Requirements
- **CPU**: 4+ cores
- **RAM**: 8GB+
- **Storage**: 10GB+ free space
- **GPU**: Optional (for faster processing)

## 🔒 Privacy & Security

SAM is designed with privacy in mind:
- **Local Processing**: All AI processing happens on your computer
- **No Data Sharing**: Your conversations and documents stay on your device
- **Open Source**: You can inspect and modify the code
- **Optional Authentication**: Enable user authentication if needed

## 📈 Performance Tips

1. **More RAM**: Allocate more memory for better performance
2. **SSD Storage**: Use SSD for faster document processing
3. **Close Other Apps**: Free up system resources
4. **Regular Cleanup**: Periodically clean old memories

## 🔄 Updates

To update SAM:
1. **Backup**: Export your memories from the Memory Control Center
2. **Update**: Download the latest version
3. **Restore**: Import your memories back

## 📞 Support

- **Documentation**: Check `README.md` and `DEPLOYMENT.md`
- **Logs**: Review `logs/sam.log` for error details
- **Community**: Join our community for help and discussions
- **Issues**: Report bugs and request features

---

**Welcome to SAM Community Edition Beta!** 🎉

Start chatting with SAM and explore its capabilities. Upload some documents to see how SAM learns and remembers information across conversations.
