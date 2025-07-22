#!/usr/bin/env python3
"""
SAM Setup Launcher

Main entry point for new users to set up SAM.
Provides clear options and guides users to the right setup method.

Usage:
    python setup.py

Author: SAM Development Team
Version: 1.0.0
"""

import os
import sys
import subprocess
from pathlib import Path

def print_banner():
    """Display the main SAM setup banner."""
    print("\n" + "=" * 80)
    print("🚀 SAM: Secure AI Memory - Setup Launcher")
    print("=" * 80)
    print("Welcome to SAM! Choose your preferred setup method below.")
    print("SAM is the FIRST AI system with human-like conceptual understanding")
    print("and enterprise-grade security.")
    print("=" * 80)

def show_setup_options():
    """Display all available setup options."""
    print("\n🎯 **Setup Options for New Users**")
    print("=" * 60)
    
    print("\n1. 🎯 **Interactive Script (Recommended)**")
    print("   • Guided setup wizard with step-by-step instructions")
    print("   • Automatic dependency detection and installation")
    print("   • Interactive encryption setup with password validation")
    print("   • System optimization and configuration")
    print("   • ⏱️  Time: ~10-15 minutes | 💡 Difficulty: Beginner-friendly")
    
    print("\n2. ⚡ **Quick Setup**")
    print("   • Fast installation with minimal prompts")
    print("   • Default configuration (can be customized later)")
    print("   • Basic encryption setup")
    print("   • ⏱️  Time: ~5 minutes | 💡 Difficulty: Easy")
    
    print("\n3. 🔧 **Manual Installation**")
    print("   • Complete control over all settings")
    print("   • Custom configuration options")
    print("   • Advanced security settings")
    print("   • ⏱️  Time: ~20-30 minutes | 💡 Difficulty: Advanced")
    
    print("\n4. 🔐 **Encryption Only Setup**")
    print("   • For existing SAM installations")
    print("   • Adds enterprise-grade encryption")
    print("   • Master password creation with validation")
    print("   • ⏱️  Time: ~5 minutes | 💡 Difficulty: Easy")
    
    print("\n5. 📖 **View Documentation**")
    print("   • Read setup guides and documentation")
    print("   • Troubleshooting information")
    print("   • Advanced configuration options")
    
    print("\n6. ❌ **Exit**")
    print("   • Exit without setting up")

def get_user_choice():
    """Get user's setup choice."""
    print("\n" + "─" * 60)
    
    while True:
        try:
            choice = input("Enter your choice (1-6) [1]: ").strip()
            if not choice:
                return 1  # Default to interactive
            
            choice_num = int(choice)
            if 1 <= choice_num <= 6:
                return choice_num
            else:
                print("❌ Please enter a number between 1 and 6")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n👋 Setup cancelled")
            return 6

def run_interactive_setup():
    """Run the interactive setup script."""
    print("\n🎯 Starting Interactive Setup...")
    print("This will guide you through the complete SAM installation process.")
    
    if not Path("setup_sam.py").exists():
        print("❌ Interactive setup script not found")
        print("Please ensure you're in the SAM directory")
        return False
    
    try:
        subprocess.run([sys.executable, "setup_sam.py"], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Interactive setup failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Setup cancelled by user")
        return False

def run_quick_setup():
    """Run the quick setup process."""
    print("\n⚡ Starting Quick Setup...")
    
    if not Path("install_sam.py").exists():
        print("❌ Quick setup script not found")
        print("Please ensure you're in the SAM directory")
        return False
    
    try:
        subprocess.run([sys.executable, "install_sam.py"], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Quick setup failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Setup cancelled by user")
        return False

def show_manual_instructions():
    """Display manual installation instructions."""
    print("\n🔧 Manual Installation Instructions")
    print("=" * 50)
    
    print("\n📋 **Step-by-Step Process:**")
    print("\n1. 📦 **Install Dependencies:**")
    print("   pip install streamlit chromadb sentence-transformers")
    print("   pip install argon2-cffi cryptography requests")
    print("   pip install beautifulsoup4 PyPDF2 python-docx psutil")
    
    print("\n2. 🤖 **Install Ollama (AI Model):**")
    print("   • Visit: https://ollama.ai/download")
    print("   • Download for your platform (Windows/macOS/Linux)")
    print("   • Install Ollama")
    print("   • Download SAM's model:")
    print("     ollama pull hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M")
    
    print("\n3. 📁 **Create Directories:**")
    print("   mkdir logs memory_store security config uploads quarantine approved archive")
    
    print("\n4. 🔐 **Setup Encryption:**")
    print("   python setup_encryption.py")
    
    print("\n5. 🚀 **Launch SAM:**")
    print("   python start_sam_secure.py --mode full")
    
    print("\n📚 **Documentation:**")
    print("   • docs/ENCRYPTION_SETUP_GUIDE.md - Complete encryption guide")
    print("   • docs/README_SECURE_INSTALLATION.md - Full installation guide")
    print("   • docs/README.md - Main documentation")
    
    input("\nPress Enter to continue...")

def run_encryption_setup():
    """Run encryption-only setup."""
    print("\n🔐 Starting Encryption Setup...")
    print("This will add enterprise-grade encryption to your SAM installation.")
    
    if not Path("setup_encryption.py").exists():
        print("❌ Encryption setup script not found")
        print("Please ensure you're in the SAM directory")
        return False
    
    try:
        subprocess.run([sys.executable, "setup_encryption.py"], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Encryption setup failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Setup cancelled by user")
        return False

def show_documentation():
    """Show available documentation."""
    print("\n📖 SAM Documentation")
    print("=" * 50)
    
    docs = [
        ("SETUP_OPTIONS.md", "Overview of all setup options"),
        ("docs/QUICK_ENCRYPTION_SETUP.md", "5-minute quick start guide"),
        ("docs/ENCRYPTION_SETUP_GUIDE.md", "Complete encryption setup guide"),
        ("docs/README_SECURE_INSTALLATION.md", "Full installation guide"),
        ("docs/README.md", "Main SAM documentation"),
    ]
    
    print("\n📚 **Available Documentation:**")
    for doc_file, description in docs:
        if Path(doc_file).exists():
            print(f"   ✅ {doc_file} - {description}")
        else:
            print(f"   ❌ {doc_file} - {description} (missing)")
    
    print("\n🌐 **Online Resources:**")
    print("   • Ollama Installation: https://ollama.ai/download")
    print("   • Python Downloads: https://python.org/downloads/")
    
    print("\n💡 **Quick Tips:**")
    print("   • Start with Option 1 (Interactive Script) if you're new")
    print("   • Use Option 2 (Quick Setup) for fast deployment")
    print("   • Choose Option 3 (Manual) for advanced customization")
    print("   • Option 4 adds encryption to existing installations")
    
    input("\nPress Enter to continue...")

def main():
    """Main setup launcher."""
    try:
        print_banner()
        
        while True:
            show_setup_options()
            choice = get_user_choice()
            
            if choice == 1:
                # Interactive Script
                if run_interactive_setup():
                    print("\n🎉 Interactive setup completed successfully!")
                    break
                else:
                    print("\n❌ Interactive setup failed. Try another option.")
            
            elif choice == 2:
                # Quick Setup
                if run_quick_setup():
                    print("\n🎉 Quick setup completed successfully!")
                    break
                else:
                    print("\n❌ Quick setup failed. Try another option.")
            
            elif choice == 3:
                # Manual Installation
                show_manual_instructions()
                # Don't break - let user choose another option or exit
            
            elif choice == 4:
                # Encryption Only
                if run_encryption_setup():
                    print("\n🎉 Encryption setup completed successfully!")
                    break
                else:
                    print("\n❌ Encryption setup failed. Try another option.")
            
            elif choice == 5:
                # Documentation
                show_documentation()
                # Don't break - let user choose another option or exit
            
            elif choice == 6:
                # Exit
                print("\n👋 Goodbye! Run this script again anytime to set up SAM.")
                break
        
        # Final message
        if choice != 6:
            print("\n🚀 **SAM is ready!**")
            print("\n📍 **Access Points:**")
            print("   • Secure Chat: http://localhost:8502")
            print("   • Memory Center: http://localhost:8501")
            print("   • Standard Chat: http://localhost:5001")

            print("\n🔑 **Start SAM:**")
            print("   python start_sam_secure.py --mode full")

            print("\n🎯 **Unlock SAM Pro Features:**")
            print("   python register_sam_pro.py")
            print("   • Dream Canvas Memory Visualization")
            print("   • TPV Active Reasoning Control")
            print("   • Cognitive Automation (SLP System)")
            print("   • Advanced Analytics and Insights")

            print("\n📖 **Documentation:**")
            print("   • SETUP_OPTIONS.md - All setup options")
            print("   • docs/ - Complete documentation")
    
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check the logs and try again")

if __name__ == "__main__":
    main()
