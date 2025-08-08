#!/bin/bash
export STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
export STREAMLIT_SERVER_HEADLESS=true

# SAM Community Edition - Enhanced Launcher Script for Linux/macOS
# This script handles virtual environments and modern Linux PEP 668 restrictions

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[SAM]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SAM]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[SAM]${NC} $1"
}

print_error() {
    echo -e "${RED}[SAM]${NC} $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a port is in use
port_in_use() {
    lsof -i :$1 >/dev/null 2>&1
}

# Main function
main() {
    print_status "Starting SAM Community Edition..."
    echo "=================================="

    # Check if we're in the right directory
    if [ ! -f "start_sam.py" ]; then
        print_error "start_sam.py not found. Please run this script from the SAM directory."
        exit 1
    fi

    # Handle virtual environment for modern Linux systems
    if [ ! -d ".venv" ]; then
        print_warning "No virtual environment found. Creating one for PEP 668 compliance..."
        python3 -m venv .venv || {
            print_error "Failed to create virtual environment. Please ensure python3-venv is installed:"
            print_error "  sudo apt install python3-venv"
            exit 1
        }
        print_success "Virtual environment created"
    fi

    # Activate virtual environment
    print_status "Activating virtual environment..."
    source .venv/bin/activate || {
        print_error "Failed to activate virtual environment"
        exit 1
    }
    print_success "Virtual environment activated"
    
    # Check Python (use python from virtual environment)
    PYTHON_CMD="python"  # Virtual environment should have python available

    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_status "Using Python $PYTHON_VERSION (virtual environment)"

    # Install/upgrade core dependencies
    print_status "Installing/upgrading core dependencies..."
    pip install --upgrade pip >/dev/null 2>&1 || print_warning "Could not upgrade pip"

    # Install essential packages for SAM
    pip install streamlit==1.42.0 cryptography>=41.0.0,<43.0.0 numpy pandas requests "PyPDF2>=3.0.0,<4.0.0" >/dev/null 2>&1 || {
        print_warning "Some dependencies may need manual installation"
        print_warning "Run: pip install streamlit==1.42.0 cryptography numpy pandas requests PyPDF2"
    }

    # Install optional packages for enhanced PDF processing
    print_status "Installing optional packages for enhanced functionality..."
    pip install langchain faiss-cpu >/dev/null 2>&1 || {
        print_warning "Optional packages failed - SAM will use fallback PDF processing"
    }

    print_success "Dependencies ready"
    
    # Check if Ollama is installed
    if ! command_exists ollama; then
        print_warning "Ollama not found. SAM requires Ollama for AI functionality."
        print_warning "Please install Ollama from: https://ollama.ai/download"
        print_warning "Or run: brew install ollama (on macOS)"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        print_success "Ollama found"
        
        # Check if Ollama service is running
        if ! ollama list >/dev/null 2>&1; then
            print_warning "Ollama service doesn't seem to be running."
            print_status "Attempting to start Ollama service..."
            
            # Try to start Ollama in background
            if command_exists systemctl; then
                systemctl --user start ollama 2>/dev/null || true
            fi
            
            # Give it a moment to start
            sleep 2
            
            if ! ollama list >/dev/null 2>&1; then
                print_warning "Could not start Ollama automatically."
                print_warning "Please start Ollama manually in another terminal:"
                print_warning "  ollama serve"
                read -p "Press Enter when Ollama is running..."
            fi
        fi
        
        # Check if the required model is available
        if ! ollama list | grep -q "DeepSeek-R1"; then
            print_warning "Required AI model not found."
            print_status "Downloading model (this may take several minutes)..."
            ollama pull hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M || {
                print_error "Failed to download model. Please run manually:"
                print_error "  ollama pull hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M"
                exit 1
            }
        fi
    fi
    
    # Check if ports are available
    if port_in_use 5001; then
        print_warning "Port 5001 is already in use. SAM's chat interface may not start."
    fi
    
    if port_in_use 8501; then
        print_warning "Port 8501 is already in use. SAM's memory interface may not start."
    fi
    
    # Create logs directory if it doesn't exist
    mkdir -p logs
    
    # Check if configuration exists
    if [ ! -f "config/sam_config.json" ]; then
        print_warning "Configuration not found. Running installer first..."
        $PYTHON_CMD install.py || {
            print_error "Installation failed. Please check the error messages above."
            exit 1
        }
    fi
    
    # Start SAM
    print_success "Starting SAM..."
    print_status "Chat Interface will be available at: http://localhost:5001"
    print_status "Memory Control Center will be available at: http://localhost:8501"
    print_status "Press Ctrl+C to stop SAM"
    echo
    
    # Run SAM with error handling
    $PYTHON_CMD start_sam.py || {
        print_error "SAM failed to start. Check logs/sam.log for details."
        exit 1
    }
}

# Trap Ctrl+C
trap 'print_status "Shutting down SAM..."; exit 0' INT

# Run main function
main "$@"
