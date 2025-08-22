# 🤖 SAM - Small Agent Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

**SAM (Small Agent Model)** is an advanced AI assistant with sophisticated memory management, document processing, and autonomous research capabilities. Built for researchers, developers, and knowledge workers who need intelligent document analysis and automated research discovery.

---

## 🌟 **What is SAM?**

SAM (Secure Agent Model) is a revolutionary open-source AI assistant that combines advanced reasoning capabilities with enterprise-grade security. Unlike other AI assistants, SAM encrypts all your data with your personal Master Password, ensuring complete privacy and control.

### **🎯 Key Features**

- **🔐 Enterprise-Grade Security**: All data encrypted with your Master Password
- **📄 Document Processing**: Upload and analyze PDFs, TXT, DOCX, MD, and **CSV files**
- **📊 Data Science Capabilities**: Professional CSV analysis with automatic insights
- **🐳 Smart Docker Integration**: Automatic Docker provisioning when available (optional)
- **🧠 Advanced Memory**: Persistent conversation history and knowledge storage
- **💬 Contextual Conversations**: Maintains context across multiple sessions
- **🔍 Smart Search**: Intelligent routing between local knowledge and web search
- **⚡ Lazy Provisioning**: Works immediately, enhances automatically
- **🎨 Modern Interface**: Clean, intuitive web-based interface
- **🌐 Cross-Platform**: Works on Windows, macOS, and Linux

---

## 🚀 **Quick Start**

### **Prerequisites**

#### **Required (Minimum Setup)**
- **Python 3.10+** (tested with Python 3.12)
- **2GB+ RAM** (4GB+ recommended)
- **Modern web browser**
- **Internet connection** (for initial model downloads)
- **git** for cloning the repository
- **[Ollama](https://ollama.ai)** installed and running for AI responses

#### **Optional (Enhanced Features)**
- **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** for maximum security
  - **Auto-detected and managed** - no manual configuration needed
  - **Provides isolated execution** for CSV data analysis
  - **SAM works perfectly without Docker** using local execution

### **Installation**

#### **🪟 Windows / 🍎 macOS**

1. **Clone the repository:**

   ```bash
   git clone https://github.com/vinforge/SmallAgentModel.git
   cd SmallAgentModel
   ```

2. **Install dependencies (choose one):**

   ```bash
   # Core/runtime only
   pip install -r requirements.txt

   # Core + DPO add-ons
   pip install -r requirements.txt -r requirements_dpo.txt
   # or
   pip install -r requirements_all.txt

   # Dev tools (linters/tests)
   pip install -r requirements_dev.txt
   ```

3. **Set up AI models (first time only):**

   ```bash
   python setup_models.py
   ```

   *Downloads DeepSeek-R1 Qwen 8B model (~4.3GB) and sentence-transformers for document processing*

4. **Set up your Master Password:**

   ```bash
   python setup_encryption.py
   ```

   *This will also offer to set up models if you skipped step 3*

5. **Start SAM:**

   **Option A: Standard Startup**
   ```bash
   python start_sam.py
   ```

   **Option B: Enhanced Startup (Recommended)**
   ```bash
   python start_sam_enhanced.py
   ```

   The enhanced startup provides:
   - Auto-detection of system capabilities
   - Intelligent Docker management (optional)
   - Optimal configuration for your environment
   - Setup guidance for new users

#### **🐧 Linux (Recommended - Virtual Environment)**

**⚠️ Modern Linux systems require virtual environments due to PEP 668 restrictions.**

1. **Clone the repository:**

   ```bash
   git clone https://github.com/vinforge/SmallAgentModel.git
   cd SmallAgentModel
   ```

2. **Create and activate virtual environment:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Upgrade pip and install core dependencies:**

   ```bash
   pip install --upgrade pip
   pip install streamlit==1.42.0 cryptography>=41.0.0,<43.0.0 numpy pandas requests "PyPDF2>=3.0.0,<4.0.0"
   ```

4. **Install optional packages for enhanced PDF processing:**

   ```bash
   pip install langchain faiss-cpu
   ```

   *Note: If these fail, SAM will use fallback PDF processing (still functional)*

5. **Set up AI models:**

   ```bash
   python setup_models.py
   ```

6. **Set up encryption:**

   ```bash
   python setup_encryption.py
   ```

7. **Start SAM:**

   ```bash
   python start_sam.py
   ```

**💡 To restart SAM later:**

```bash
source .venv/bin/activate
python start_sam.py
```

#### **🌐 Access SAM**

- Navigate to `http://localhost:8502`
- Enter your Master Password
- Start chatting with SAM!

---

## 🔧 **Linux Troubleshooting**

### **PEP 668 "Externally Managed Environment" Error**

If you see this error:

```text
error: externally-managed-environment
```

**Solution**: Use a virtual environment (required on modern Linux):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install streamlit==1.42.0 cryptography numpy pandas requests
```

### **Missing python3-venv**

If virtual environment creation fails:

```bash
sudo apt update
sudo apt install python3-venv python3-dev build-essential
```

### **Permission Issues**

Never use `sudo pip install` - always use virtual environments:

```bash
# ❌ Don't do this
sudo pip install streamlit

# ✅ Do this instead
python3 -m venv .venv
source .venv/bin/activate
pip install streamlit
```

---

## 📚 **Documentation**



### **Essential Guides**

- [📖 Master Password Setup Guide](docs/MASTER_PASSWORD_SETUP_GUIDE.md) - Complete setup instructions
- [🚀 Quick Start Guide](docs/QUICK_START.md) - Get up and running fast
- [🔧 Installation Guide](docs/SETUP_GUIDE.md) - Detailed installation instructions
- [⚡ Lazy Provisioning Guide](docs/INSTALLATION_GUIDE_LAZY_PROVISIONING.md) - **NEW!** Auto-configuring setup
- [🐳 Docker Deployment](docs/DOCKER_DEPLOYMENT_GUIDE.md) - Run SAM in Docker

### **New Features & Migration**

- [📊 CSV Upload Capabilities](docs/CSV_UPLOAD_CAPABILITIES.md) - **NEW!** Data science features
- [🔄 Migration Guide](docs/MIGRATION_TO_LAZY_PROVISIONING.md) - **NEW!** Upgrade existing installations
- [🐳 Docker Strategy](docs/DOCKER_STRATEGY_FOR_NEW_USERS.md) - **NEW!** Docker auto-management

### **Advanced Features**

- [📁 Bulk Document Ingestion](docs/BULK_INGESTION_GUIDE.md) - Process multiple documents
- [🔑 API Key Management](docs/API_KEY_MANAGER_GUIDE.md) - Manage external API keys
- [🛡️ Security Features](docs/ENCRYPTION_SETUP_GUIDE.md) - Advanced security configuration

---

## 🔐 **Security & Privacy**

SAM is designed with **privacy-first** principles:

- **🔒 Local Encryption**: All data encrypted with your Master Password
- **🏠 Runs Locally**: No data sent to external servers (unless you choose web search)
- **🔑 You Control Keys**: Master Password never stored, only you know it
- **🛡️ Enterprise Security**: AES-256 encryption, secure key management
- **📝 Audit Trail**: Complete logging of all security operations

### **What's Encrypted?**

- All conversation history
- Uploaded documents and their content
- Memory and knowledge storage
- Session data and preferences
- API keys and configuration

---

## 🎯 **Use Cases**

### **Personal Assistant**
- Research and analysis
- Document summarization
- Question answering
- Creative writing assistance

### **Professional Work**
- Technical documentation review
- Code analysis and explanation
- Meeting notes and summaries
- Project planning assistance

### **Education & Learning**
- Study material analysis
- Research paper review
- Concept explanation
- Learning progress tracking

### **Enterprise**
- Secure document processing
- Internal knowledge management
- Compliance-friendly AI assistance
- Private data analysis

---

## 🛠️ **Architecture**

SAM is built with a modular, secure architecture:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │  Security Layer │    │  AI Processing  │
│   (Streamlit)   │◄──►│  (Encryption)   │◄──►│   (Local LLM)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │    │   Encrypted     │    │    Memory       │
│   Processing    │    │   Storage       │    │   Management    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### **Core Components**
- **Security Layer**: Master Password encryption, key management
- **Document Processing**: PDF parsing, content extraction, proven integration
- **Memory System**: Persistent storage, conversation threading
- **AI Engine**: Local reasoning, response generation
- **Web Interface**: Modern, responsive UI

---

## 🤝 **Contributing**

We welcome contributions from the community! Here's how you can help:

### **Ways to Contribute**
- 🐛 **Bug Reports**: Found an issue? [Open an issue](https://github.com/your-username/sam/issues)
- 💡 **Feature Requests**: Have an idea? [Start a discussion](https://github.com/your-username/sam/discussions)
- 🔧 **Code Contributions**: Submit pull requests for improvements
- 📚 **Documentation**: Help improve our guides and documentation
- 🧪 **Testing**: Test new features and report feedback

### **Development Setup**
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request with a clear description

### **Code Standards**
- Follow PEP 8 Python style guidelines
- Add docstrings to all functions and classes
- Include tests for new features
- Update documentation as needed

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### **What this means:**
- ✅ **Free to use** for personal and commercial projects
- ✅ **Modify and distribute** as you see fit
- ✅ **No warranty** - use at your own risk
- ✅ **Attribution appreciated** but not required

---

## 🆘 **Support**

### **Getting Help**
- 📖 **Documentation**: Check our comprehensive guides in `/docs`
- 💬 **Community**: Join discussions on GitHub
- 🐛 **Issues**: Report bugs via GitHub Issues
- 📧 **Contact**: Reach out to the maintainers

### **Common Issues**
- **Master Password forgotten**: No recovery possible - this is by design for security
- **Installation problems**: Check Python version and dependencies
- **Performance issues**: Ensure adequate RAM and storage space
- **Security concerns**: Review our security documentation

---

## 🎉 **Acknowledgments**

SAM is built on the shoulders of giants. Special thanks to:

- **Streamlit** - For the amazing web framework
- **LangChain** - For document processing capabilities
- **ChromaDB** - For vector storage and retrieval
- **Cryptography** - For enterprise-grade encryption
- **The Open Source Community** - For inspiration and contributions

---

## 🚀 **What's Next?**

SAM is actively developed with exciting features planned:

- 🔮 **Enhanced AI Models**: Support for more local LLM options
- 🌐 **Multi-language Support**: Interface localization
- 📱 **Mobile Interface**: Responsive design improvements
- 🔗 **API Access**: RESTful API for integrations
- 🎨 **Themes**: Customizable interface themes
- 📊 **Analytics**: Usage insights and performance metrics

---

**🧠 Experience the future of secure AI assistance with SAM!**

*Built with ❤️ by the open-source community*
