# 🧠 Secure Agent Model (SAM)

**The world's most advanced open-source AI assistant with enterprise-grade security**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

---

## 🌟 **What is SAM?**

SAM (Secure Agent Model) is a revolutionary open-source AI assistant that combines advanced reasoning capabilities with enterprise-grade security. Unlike other AI assistants, SAM encrypts all your data with your personal Master Password, ensuring complete privacy and control.

### **🎯 Key Features**

- **🔐 Enterprise-Grade Security**: All data encrypted with your Master Password
- **📄 Document Processing**: Upload and analyze PDFs with intelligent content recall
- **🧠 Advanced Memory**: Persistent conversation history and knowledge storage
- **💬 Contextual Conversations**: Maintains context across multiple sessions
- **🔍 Smart Search**: Intelligent routing between local knowledge and web search
- **🎨 Modern Interface**: Clean, intuitive web-based interface
- **🌐 Cross-Platform**: Works on Windows, macOS, and Linux

---

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.8 or higher
- 4GB+ RAM recommended
- Modern web browser
- Internet connection (for initial model downloads)
- **Optional**: [Ollama](https://ollama.ai) for enhanced AI responses

### **Installation**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/sam.git
   cd sam
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
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
   ```bash
   python start_sam.py
   ```

6. **Open your browser:**
   - Go to `http://localhost:8502`
   - Enter your Master Password
   - Start chatting with SAM!

---

## 📚 **Documentation**

### **Essential Guides**
- [📖 Master Password Setup Guide](docs/MASTER_PASSWORD_SETUP_GUIDE.md) - Complete setup instructions
- [🚀 Quick Start Guide](docs/QUICK_START.md) - Get up and running fast
- [🔧 Installation Guide](docs/SETUP_GUIDE.md) - Detailed installation instructions
- [🐳 Docker Deployment](docs/DOCKER_DEPLOYMENT_GUIDE.md) - Run SAM in Docker

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
