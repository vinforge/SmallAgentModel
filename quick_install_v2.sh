#!/bin/bash

# SAM v2 Quick Installation Script
# Minimal setup for testing v2 features

set -e

echo "🚀 SAM v2 Quick Installation"
echo "============================"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
print_status "Python $PYTHON_VERSION detected"

# Create virtual environment
print_status "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch (CPU version for compatibility)
print_status "Installing PyTorch (CPU version)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install v2 minimal requirements
print_status "Installing SAM v2 dependencies..."
if [ -f "requirements_v2_minimal.txt" ]; then
    pip install -r requirements_v2_minimal.txt
else
    # Fallback to manual installation
    pip install numpy>=1.21.0 scipy>=1.7.0 scikit-learn>=1.0.0
    pip install transformers>=4.20.0 sentence-transformers>=2.2.0
    pip install colbert-ai>=0.2.19 chromadb>=0.4.0
    pip install python-docx>=0.8.11 PyPDF2>=3.0.0
    pip install flask>=2.0.0 requests>=2.25.0
    pip install pyyaml>=6.0 tqdm>=4.62.0
fi

# Test basic imports
print_status "Testing installation..."
python3 -c "
import torch
import numpy as np
import chromadb
from colbert import Indexer
print('✅ All core dependencies imported successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'NumPy version: {np.__version__}')
"

# Create simple activation script
cat > activate_v2.sh << 'EOF'
#!/bin/bash
echo "🚀 Activating SAM v2 environment..."
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
echo "✅ Environment activated. To deactivate: deactivate"
EOF
chmod +x activate_v2.sh

print_success "🎉 SAM v2 quick installation completed!"
echo
echo "📋 Next steps:"
echo "1. Activate environment: source activate_v2.sh"
echo "2. Test v2 features: python test_v2_integration.py"
echo
echo "💡 For full installation with GPU support, use: ./install_sam_v2_linux.sh"
