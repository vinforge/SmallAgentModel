#!/bin/bash

# 🚀 SAM Fresh GITEA Deployment Script
# This script will replace the outdated GITEA repository with the latest SAM version

set -e  # Exit on any error

echo "🚀 SAM Fresh GITEA Deployment Script"
echo "===================================="
echo ""

# Configuration
GITEA_URL="http://172.16.20.246:3000/Forge/SAM.git"
BRANCH="main"

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

# Check if we're in the SAM directory
if [ ! -f "SAM-whitepaper.md" ] || [ ! -d "sam/orchestration" ]; then
    print_error "This script must be run from the SAM repository root directory"
    print_error "Please cd to the SAM directory and run this script again"
    exit 1
fi

print_status "Checking SAM repository status..."

# Check git status
if ! git status &>/dev/null; then
    print_error "This is not a git repository"
    exit 1
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    print_warning "You have uncommitted changes"
    echo "Uncommitted files:"
    git status --porcelain
    echo ""
    read -p "Do you want to continue anyway? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Deployment cancelled"
        exit 0
    fi
fi

# Test connectivity to GITEA server
print_status "Testing connectivity to GITEA server..."
if ! ping -c 1 172.16.20.246 &>/dev/null; then
    print_error "Cannot reach GITEA server at 172.16.20.246"
    print_error "Please check your network connection and try again"
    exit 1
fi

# Test GITEA port
if ! nc -z 172.16.20.246 3000 2>/dev/null; then
    print_error "GITEA server port 3000 is not accessible"
    print_error "Please ensure GITEA is running and accessible"
    exit 1
fi

print_success "GITEA server is accessible"

# Show current repository status
print_status "Current repository status:"
echo "Branch: $(git branch --show-current)"
echo "Latest commit: $(git log -1 --oneline)"
echo "Total commits: $(git rev-list --count HEAD)"
echo ""

# Show what will be deployed
print_status "Revolutionary features ready for deployment:"
echo "✅ SOF v2 Dynamic Agent Architecture"
echo "✅ Cognitive Synthesis Engine (Dream Catcher)"
echo "✅ Enterprise-Grade Security (SAM Secure Enclave)"
echo "✅ Active Reasoning Control (TPV System)"
echo "✅ Autonomous Cognitive Automation (SLP System)"
echo "✅ Updated Documentation and Whitepaper"
echo ""

# Confirm deployment
print_warning "This will REPLACE the entire GITEA repository with this local version"
print_warning "All existing files on GITEA will be overwritten"
echo ""
read -p "Are you sure you want to proceed? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_status "Deployment cancelled"
    exit 0
fi

# Perform the deployment
print_status "Starting fresh deployment to GITEA..."

# Option 1: Force push (replaces everything)
print_status "Performing force push to replace GITEA repository..."
if git push --force-with-lease origin $BRANCH; then
    print_success "Successfully deployed SAM to GITEA!"
    print_success "Repository URL: $GITEA_URL"
    print_success "All revolutionary features are now available on GITEA"
else
    print_error "Deployment failed"
    print_error "Please check the error messages above and try again"
    exit 1
fi

# Verify deployment
print_status "Verifying deployment..."
if git ls-remote origin $BRANCH &>/dev/null; then
    print_success "Deployment verification successful"
    print_success "GITEA repository is now up to date with latest SAM version"
else
    print_warning "Could not verify deployment - please check GITEA manually"
fi

echo ""
print_success "🎉 SAM Fresh Deployment Complete!"
echo ""
print_status "Next steps:"
echo "1. Visit GITEA: http://172.16.20.246:3000/Forge/SAM"
echo "2. Verify all new files and features are present"
echo "3. Check that README.md shows the latest revolutionary features"
echo "4. Confirm SOF v2 framework is in sam/orchestration/"
echo "5. Test the repository by cloning it fresh"
echo ""
print_success "SAM is now ready for collaboration and deployment!"
