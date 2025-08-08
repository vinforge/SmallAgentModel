# SAM v2 Linux Installation Guide

**🐧 Complete setup guide for SAM v2 MUVERA on Linux systems**

## 🚀 Quick Start (Recommended)

### Option 1: Automated Installation
```bash
# Clone the repository
git clone https://github.com/vinforge/SmallAgentModel.git
cd SmallAgentModel

# Run the automated installer
./install_sam_v2_linux.sh

# Activate the environment
source ./activate_sam_v2.sh

# Test the installation
python test_v2_integration.py
```

### Option 2: Quick Installation (Minimal)
```bash
# Clone the repository
git clone https://github.com/vinforge/SmallAgentModel.git
cd SmallAgentModel

# Quick setup for testing
./quick_install_v2.sh

# Activate the environment
source activate_v2.sh

# Test basic functionality
python test_v2_phase1.py
```

## 📋 What the Installation Scripts Do

### Full Installation (`install_sam_v2_linux.sh`)
1. **System Dependencies**: Installs build tools, Python dev headers, BLAS/LAPACK
2. **Virtual Environment**: Creates and activates `.venv` with Python 3.8+
3. **PyTorch**: Auto-detects GPU and installs CUDA or CPU version
4. **SAM v2 Dependencies**: Installs ColBERT, ChromaDB, and all required packages
5. **Testing**: Validates installation with component tests
6. **Scripts**: Creates activation and uninstall scripts

### Quick Installation (`quick_install_v2.sh`)
1. **Virtual Environment**: Creates `.venv` with minimal setup
2. **Core Dependencies**: Installs essential v2 packages only
3. **CPU PyTorch**: Installs CPU-only version for compatibility
4. **Basic Testing**: Validates core imports

## 🔧 Manual Installation

If you prefer manual control or the scripts don't work on your system:

### Step 1: System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv python3-dev \
    build-essential git curl wget libffi-dev libssl-dev \
    libblas-dev liblapack-dev gfortran pkg-config cmake \
    libhdf5-dev libopenblas-dev
```

**RHEL/CentOS/Fedora:**
```bash
sudo dnf install -y python3 python3-pip python3-devel gcc gcc-c++ \
    make git curl wget openssl-devel libffi-devel \
    blas-devel lapack-devel cmake hdf5-devel openblas-devel
```

### Step 2: Virtual Environment
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Step 3: Install Dependencies
```bash
# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Or for CUDA support (if you have NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install SAM v2 core dependencies
pip install colbert-ai>=0.2.19
pip install chromadb>=0.4.0
pip install numpy>=1.21.0
pip install scipy>=1.7.0

# Install document processing
pip install python-docx PyPDF2>=3.0.0

# Install all requirements
pip install -r requirements.txt
```

### Step 4: Test Installation
```bash
# Test basic imports
python3 -c "
import torch
import numpy as np
import chromadb
from colbert import Indexer
print('✅ All dependencies imported successfully')
"

# Test SAM v2 components
python test_v2_integration.py
```

## 🎯 Usage After Installation

### Activate Environment
```bash
# Using the generated script
source ./activate_sam_v2.sh

# Or manually
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Upload Documents
```python
from sam.document_processing.v2_upload_handler import ingest_v2_document

success, message, metadata = ingest_v2_document("document.pdf")
print(f"Upload: {success} - {message}")
```

### Query Documents
```python
from sam.document_processing.v2_query_handler import query_v2_documents

success, response, metadata = query_v2_documents("What is this document about?")
print(f"Response: {response}")
```

### Switch Pipelines
```python
from sam.document_processing.v2_query_handler import switch_pipeline

# Use v2 MUVERA pipeline
switch_pipeline("v2_muvera")

# Use v1 chunking pipeline
switch_pipeline("v1_chunking")
```

## 🧪 Testing

### Component Tests
```bash
# Test foundation components
python test_v2_phase1.py

# Test ingestion pipeline
python test_v2_phase2.py

# Test RAG pipeline
python test_v2_phase3.py

# Test end-to-end integration
python test_v2_integration.py
```

### Expected Results
- ✅ Multi-Vector Embeddings (ColBERTv2 token-level embeddings)
- ✅ MUVERA FDE Transformation (fixed dimensional encoding)
- ✅ Two-Stage Retrieval (fast FDE search + accurate reranking)
- ✅ RAG Pipeline Integration (complete document Q&A)
- ✅ Chat Interface Compatibility (seamless integration)

## 🔧 Troubleshooting

### Common Issues

**1. ColBERT Installation Fails**
```bash
# Try with no cache
pip install colbert-ai --no-cache-dir

# Or use conda
conda install -c conda-forge colbert-ai
```

**2. ChromaDB Connection Error**
```bash
# Clear database and recreate
rm -rf chroma_db_v2/
python -c "from sam.storage import create_v2_collections; create_v2_collections()"
```

**3. Memory Issues**
```bash
# Reduce batch size in configuration
export SAM_V2_BATCH_SIZE=2
export SAM_V2_MAX_WORKERS=1
```

**4. Permission Issues**
```bash
# Make sure scripts are executable
chmod +x install_sam_v2_linux.sh quick_install_v2.sh

# Check virtual environment permissions
ls -la .venv/
```

### Debug Mode
```bash
# Enable detailed logging
export SAM_DEBUG=1
python test_v2_integration.py
```

## 🗑️ Uninstallation

```bash
# Using the generated script
./uninstall_sam_v2.sh

# Or manually
rm -rf .venv chroma_db_v2 uploads
rm -f activate_sam_v2.sh uninstall_sam_v2.sh activate_v2.sh
```

## 📊 System Requirements

### Minimum Requirements
- **OS**: Ubuntu 18.04+, Debian 10+, RHEL 8+, CentOS 8+, Fedora 32+
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB for dependencies, 5GB+ for document storage
- **CPU**: x86_64 architecture

### Recommended Requirements
- **RAM**: 16GB+ for large document collections
- **GPU**: NVIDIA GPU with CUDA 11.8+ for faster processing
- **Storage**: SSD for better I/O performance
- **CPU**: Multi-core processor for parallel processing

## 🚀 Performance Expectations

### With CPU Only
- **Document Upload**: 2-5 seconds per document
- **Query Response**: 100-500ms per query
- **Concurrent Users**: 5-10 simultaneous queries

### With GPU (CUDA)
- **Document Upload**: 0.5-2 seconds per document
- **Query Response**: 50-200ms per query
- **Concurrent Users**: 20-50 simultaneous queries

## 📞 Support

If you encounter issues:

1. **Check Logs**: Enable debug mode and review error messages
2. **Test Components**: Run individual test scripts to isolate issues
3. **Check Dependencies**: Verify all packages are correctly installed
4. **System Resources**: Ensure adequate RAM and storage
5. **GitHub Issues**: Report bugs with system info and error logs

---

**🎉 Ready to experience 90% faster document retrieval with 10% higher accuracy on Linux!**
