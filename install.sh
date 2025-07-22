#!/bin/bash
# SAM Secure AI Assistant - One-Line Installer
# Usage: curl -sSL https://raw.githubusercontent.com/your-repo/SAM/main/install.sh | bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                    🧠 SAM - SECURE AI ASSISTANT 🔒                          ║"
echo "║                         One-Line Installer Script                           ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if running on supported OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    CYGWIN*)    MACHINE=Cygwin;;
    MINGW*)     MACHINE=MinGw;;
    *)          MACHINE="UNKNOWN:${OS}"
esac

echo -e "${BLUE}🖥️  Detected OS: ${MACHINE}${NC}"

# Check Python installation
echo -e "${BLUE}🐍 Checking Python installation...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}❌ Python not found!${NC}"
    echo -e "${YELLOW}📥 Please install Python 3.8+ from https://python.org/downloads/${NC}"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo -e "${GREEN}✅ Found Python ${PYTHON_VERSION}${NC}"

# Check if version is compatible
if $PYTHON_CMD -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo -e "${GREEN}✅ Python version compatible${NC}"
else
    echo -e "${RED}❌ Python 3.8+ required, found ${PYTHON_VERSION}${NC}"
    exit 1
fi

# Check if git is available
echo -e "${BLUE}📥 Checking Git installation...${NC}"
if command -v git &> /dev/null; then
    echo -e "${GREEN}✅ Git found${NC}"
    USE_GIT=true
else
    echo -e "${YELLOW}⚠️  Git not found, will download ZIP instead${NC}"
    USE_GIT=false
fi

# Create installation directory
INSTALL_DIR="$HOME/SAM"
echo -e "${BLUE}📁 Installation directory: ${INSTALL_DIR}${NC}"

if [ -d "$INSTALL_DIR" ]; then
    echo -e "${YELLOW}⚠️  Directory exists. Backing up...${NC}"
    mv "$INSTALL_DIR" "$INSTALL_DIR.backup.$(date +%Y%m%d_%H%M%S)"
fi

# Download SAM
echo -e "${BLUE}📥 Downloading SAM...${NC}"
if [ "$USE_GIT" = true ]; then
    git clone https://github.com/your-repo/SAM.git "$INSTALL_DIR"
else
    # Download ZIP (fallback)
    if command -v curl &> /dev/null; then
        curl -L https://github.com/your-repo/SAM/archive/main.zip -o sam.zip
    elif command -v wget &> /dev/null; then
        wget https://github.com/your-repo/SAM/archive/main.zip -O sam.zip
    else
        echo -e "${RED}❌ Neither curl nor wget found. Please install git or download manually.${NC}"
        exit 1
    fi
    
    # Extract ZIP
    if command -v unzip &> /dev/null; then
        unzip -q sam.zip
        mv SAM-main "$INSTALL_DIR"
        rm sam.zip
    else
        echo -e "${RED}❌ unzip not found. Please install unzip or use git.${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✅ SAM downloaded successfully${NC}"

# Change to installation directory
cd "$INSTALL_DIR"

# Install dependencies
echo -e "${BLUE}📦 Installing dependencies...${NC}"
$PYTHON_CMD -m pip install --upgrade pip
$PYTHON_CMD -m pip install -r requirements.txt

echo -e "${GREEN}✅ Dependencies installed${NC}"

# Make scripts executable
chmod +x start_sam_secure.py install_sam.py

# Run setup
echo -e "${BLUE}⚙️  Running setup...${NC}"
$PYTHON_CMD install_sam.py

echo -e "${GREEN}"
echo "🎉 SAM installation completed successfully!"
echo ""
echo "📍 Installation location: $INSTALL_DIR"
echo "🚀 To launch SAM:"
echo "   cd $INSTALL_DIR"
echo "   python start_sam_secure.py --mode full"
echo ""
echo "📖 Documentation: README_SECURE_INSTALLATION.md"
echo "🔒 Security: Enterprise-grade encryption enabled"
echo "🏠 Privacy: 100% local processing"
echo -e "${NC}"
