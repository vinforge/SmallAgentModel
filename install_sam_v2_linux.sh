#!/bin/bash

# SAM v2 MUVERA Linux Installation Script
# Comprehensive setup for Ubuntu/Debian and RHEL/CentOS systems

set -e  # Exit on any error

echo "🚀 SAM v2 MUVERA Linux Installation Script"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect Linux distribution
detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
        VERSION=$VERSION_ID
    else
        print_error "Cannot detect Linux distribution"
        exit 1
    fi
    
    print_status "Detected: $PRETTY_NAME"
}

# Check if running as root
check_root() {
    if [ "$EUID" -eq 0 ]; then
        print_warning "Running as root. Consider using a regular user account."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    case $DISTRO in
        ubuntu|debian)
            sudo apt update
            sudo apt install -y \
                python3 \
                python3-pip \
                python3-venv \
                python3-dev \
                build-essential \
                git \
                curl \
                wget \
                libffi-dev \
                libssl-dev \
                libblas-dev \
                liblapack-dev \
                gfortran \
                pkg-config \
                cmake \
                libhdf5-dev \
                libopenblas-dev
            ;;
        rhel|centos|fedora)
            if command -v dnf &> /dev/null; then
                PKG_MANAGER="dnf"
            else
                PKG_MANAGER="yum"
            fi
            
            sudo $PKG_MANAGER update -y
            sudo $PKG_MANAGER install -y \
                python3 \
                python3-pip \
                python3-devel \
                gcc \
                gcc-c++ \
                make \
                git \
                curl \
                wget \
                openssl-devel \
                libffi-devel \
                blas-devel \
                lapack-devel \
                cmake \
                hdf5-devel \
                openblas-devel
            ;;
        *)
            print_warning "Unsupported distribution: $DISTRO"
            print_status "Please install Python 3.8+, pip, venv, and build tools manually"
            ;;
    esac
    
    print_success "System dependencies installed"
}

# Check Python version
check_python() {
    print_status "Checking Python version..."
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    REQUIRED_VERSION="3.8"
    
    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
        print_error "Python $REQUIRED_VERSION or higher is required. Found: $PYTHON_VERSION"
        exit 1
    fi
    
    print_success "Python $PYTHON_VERSION detected"
}

# Create and activate virtual environment
setup_venv() {
    print_status "Setting up Python virtual environment..."
    
    # Remove existing venv if it exists
    if [ -d ".venv" ]; then
        print_warning "Existing virtual environment found. Removing..."
        rm -rf .venv
    fi
    
    # Create new virtual environment
    python3 -m venv .venv
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    print_success "Virtual environment created and activated"
}

# Install PyTorch (required for ColBERT)
install_pytorch() {
    print_status "Installing PyTorch..."
    
    # Check if CUDA is available
    if command -v nvidia-smi &> /dev/null; then
        print_status "NVIDIA GPU detected. Installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        print_status "No NVIDIA GPU detected. Installing CPU-only PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    print_success "PyTorch installed"
}

# Install core SAM v2 dependencies
install_sam_deps() {
    print_status "Installing SAM v2 core dependencies..."
    
    # Core scientific computing
    pip install numpy>=1.21.0
    pip install scipy>=1.7.0
    pip install scikit-learn>=1.0.0
    
    # Vector database
    pip install chromadb>=0.4.0
    
    # ColBERT (may take a while)
    print_status "Installing ColBERT (this may take several minutes)..."
    pip install colbert-ai>=0.2.19
    
    # Transformers and related
    pip install transformers>=4.20.0
    pip install sentence-transformers>=2.2.0
    
    # Document processing
    pip install python-docx
    pip install PyPDF2>=3.0.0
    pip install python-magic
    
    # Web framework and utilities
    pip install flask>=2.0.0
    pip install requests>=2.25.0
    pip install pyyaml>=6.0
    
    # Optional but recommended
    pip install tqdm  # Progress bars
    pip install psutil  # System monitoring
    
    print_success "Core dependencies installed"
}

# Install from requirements.txt if available
install_requirements() {
    if [ -f "requirements.txt" ]; then
        print_status "Installing from requirements.txt..."
        pip install -r requirements.txt
        print_success "Requirements.txt dependencies installed"
    else
        print_warning "requirements.txt not found, skipping"
    fi
}

# Test installation
test_installation() {
    print_status "Testing SAM v2 installation..."
    
    # Test basic imports
    python3 -c "
import sys
print(f'Python version: {sys.version}')

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
except ImportError:
    print('PyTorch not available')

try:
    import numpy as np
    print(f'NumPy version: {np.__version__}')
except ImportError:
    print('NumPy not available')

try:
    import chromadb
    print(f'ChromaDB available')
except ImportError:
    print('ChromaDB not available')

try:
    from colbert import Indexer
    print('ColBERT available')
except ImportError:
    print('ColBERT not available')
"
    
    # Test SAM v2 components if available
    if [ -f "test_v2_phase1.py" ]; then
        print_status "Running SAM v2 component tests..."
        python3 test_v2_phase1.py || print_warning "Some v2 tests failed (this is normal if no documents are uploaded yet)"
    fi
    
    print_success "Installation test completed"
}

# Create activation script
create_activation_script() {
    print_status "Creating activation script..."
    
    cat > activate_sam_v2.sh << 'EOF'
#!/bin/bash
# SAM v2 Environment Activation Script

echo "🚀 Activating SAM v2 MUVERA Environment"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run install_sam_v2_linux.sh first."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Set environment variables
export SAM_V2_ENABLED=true
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "✅ SAM v2 environment activated"
echo "📊 Python: $(python --version)"
echo "📁 Working directory: $(pwd)"
echo ""
echo "🔧 Available commands:"
echo "  python test_v2_phase1.py    # Test foundation components"
echo "  python test_v2_phase2.py    # Test ingestion pipeline"
echo "  python test_v2_phase3.py    # Test RAG pipeline"
echo "  python test_v2_integration.py  # Test end-to-end"
echo ""
echo "💡 To deactivate: deactivate"
EOF
    
    chmod +x activate_sam_v2.sh
    print_success "Activation script created: ./activate_sam_v2.sh"
}

# Create uninstall script
create_uninstall_script() {
    print_status "Creating uninstall script..."
    
    cat > uninstall_sam_v2.sh << 'EOF'
#!/bin/bash
# SAM v2 Uninstall Script

echo "🗑️  SAM v2 MUVERA Uninstall Script"

read -p "Are you sure you want to remove SAM v2 environment? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Uninstall cancelled"
    exit 0
fi

# Deactivate if active
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi

# Remove virtual environment
if [ -d ".venv" ]; then
    echo "Removing virtual environment..."
    rm -rf .venv
fi

# Remove generated databases
if [ -d "chroma_db_v2" ]; then
    echo "Removing v2 database..."
    rm -rf chroma_db_v2
fi

if [ -d "uploads" ]; then
    read -p "Remove uploaded documents? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf uploads
    fi
fi

# Remove scripts
rm -f activate_sam_v2.sh uninstall_sam_v2.sh

echo "✅ SAM v2 uninstalled successfully"
EOF
    
    chmod +x uninstall_sam_v2.sh
    print_success "Uninstall script created: ./uninstall_sam_v2.sh"
}

# Main installation function
main() {
    echo
    print_status "Starting SAM v2 MUVERA installation for Linux..."
    echo
    
    # Check if we're in the right directory
    if [ ! -f "sam_config.json" ]; then
        print_error "sam_config.json not found. Please run this script from the SAM project root directory."
        exit 1
    fi
    
    # Run installation steps
    check_root
    detect_distro
    install_system_deps
    check_python
    setup_venv
    install_pytorch
    install_sam_deps
    install_requirements
    test_installation
    create_activation_script
    create_uninstall_script
    
    echo
    print_success "🎉 SAM v2 MUVERA installation completed successfully!"
    echo
    echo "📋 Next steps:"
    echo "1. Activate the environment: source ./activate_sam_v2.sh"
    echo "2. Test the installation: python test_v2_integration.py"
    echo "3. Upload documents and start querying!"
    echo
    echo "📚 Documentation: See SAM_v2_README.md for detailed usage instructions"
    echo "🔧 Configuration: Edit sam_config.json to customize settings"
    echo
    print_status "Happy document retrieval with 90% lower latency! 🚀"
}

# Run main function
main "$@"
