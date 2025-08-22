#!/usr/bin/env python3
"""
SAM Enhanced Startup Script
============================

Enhanced startup script with intelligent Docker management and environment adaptation.

Features:
- Automatic environment detection
- Intelligent Docker management (lazy provisioning)
- User-friendly setup guidance
- Graceful fallbacks for all scenarios

Usage:
    python start_sam_enhanced.py

Author: SAM Development Team
Version: 1.0.0
"""

import sys
import time
import logging
from pathlib import Path

# Add SAM to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def display_welcome_banner():
    """Display welcome banner."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🤖 SAM Enhanced - Smart Environment Adaptation            ║
║                                                              ║
║   Intelligent AI with Automatic Docker Management           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


def test_enhanced_startup():
    """Test the enhanced startup system."""
    print("🧪 Testing Enhanced Startup System")
    print("=" * 45)
    
    try:
        # Test deployment manager
        print("🔍 Testing deployment strategy...")
        from sam.deployment.deployment_strategy import get_deployment_manager
        
        manager = get_deployment_manager()
        optimal_mode = manager.determine_optimal_deployment_mode()
        config = manager.configure_deployment(optimal_mode)
        
        print(f"   Optimal Mode: {optimal_mode}")
        print(f"   Security Level: {config['security_level']}")
        print(f"   Features: {list(config['features'].keys())}")
        
        # Test startup manager
        print("\n🚀 Testing startup manager...")
        from sam.startup.startup_manager import initialize_sam
        
        results = initialize_sam()
        
        print(f"   Startup Success: {results['success']}")
        print(f"   Mode: {results.get('mode', 'unknown')}")
        print(f"   Docker Status: {results.get('docker_status', 'unknown')}")
        
        # Show user guidance
        if 'user_guidance' in results:
            guidance = results['user_guidance']
            print(f"\n💬 {guidance['welcome_message']}")
            
            print("\n🎯 Capabilities:")
            for capability in guidance['current_capabilities'][:3]:
                print(f"   {capability}")
        
        return results['success']
        
    except Exception as e:
        print(f"❌ Enhanced startup test failed: {e}")
        return False


def show_docker_strategy():
    """Show the Docker strategy for different user scenarios."""
    print("\n🐳 DOCKER STRATEGY FOR NEW USERS")
    print("=" * 45)
    print()
    print("🎯 **SAM's Intelligent Docker Management:**")
    print()
    print("1. 🔍 **Auto-Detection**")
    print("   • Automatically detects if Docker is installed")
    print("   • Checks if Docker daemon is running")
    print("   • Assesses system resources")
    print()
    print("2. 🚀 **Lazy Provisioning** (Recommended)")
    print("   • Docker starts ONLY when needed for data analysis")
    print("   • No manual Docker management required")
    print("   • Automatic fallback if Docker unavailable")
    print()
    print("3. 🔄 **Deployment Modes**")
    print("   • Full Docker: Maximum security (auto-managed)")
    print("   • Local Enhanced: Good performance (no Docker needed)")
    print("   • Basic: Minimal requirements (works everywhere)")
    print()
    print("4. 👤 **User Experience**")
    print("   • New users: Works immediately without Docker")
    print("   • Docker users: Enhanced security automatically")
    print("   • No manual configuration required")


def show_new_user_recommendations():
    """Show recommendations for new users."""
    print("\n📋 RECOMMENDATIONS FOR NEW USERS")
    print("=" * 45)
    print()
    print("🎯 **Option 1: Start Immediately (Recommended)**")
    print("   • Run SAM without Docker for instant setup")
    print("   • Full CSV analysis capabilities available")
    print("   • Good security with local execution")
    print("   • Perfect for trying SAM and learning")
    print()
    print("🐳 **Option 2: Install Docker Later (Optional)**")
    print("   • Install Docker Desktop when you want maximum security")
    print("   • SAM will automatically detect and use Docker")
    print("   • No configuration changes needed")
    print("   • Seamless upgrade path")
    print()
    print("⚡ **Option 3: Background Docker Setup**")
    print("   • SAM can attempt to start Docker automatically")
    print("   • Falls back gracefully if Docker unavailable")
    print("   • Best of both worlds approach")
    print()
    print("💡 **Recommendation**: Start with Option 1, upgrade to Docker later if desired")


def start_sam_with_enhanced_management():
    """Start SAM with enhanced environment management."""
    print("\n🚀 Starting SAM with Enhanced Management...")
    print("=" * 50)
    
    try:
        # Initialize enhanced startup
        from sam.startup.startup_manager import initialize_sam
        
        print("🔧 Initializing intelligent environment...")
        results = initialize_sam()
        
        if results['success']:
            print("✅ SAM initialized successfully!")
            print(f"🔧 Mode: {results.get('mode', 'unknown').replace('_', ' ').title()}")
            print(f"🔒 Security: {results.get('security_level', 'unknown').title()}")
            
            # Show Docker status
            docker_status = results.get('docker_status', 'unknown')
            if docker_status == 'lazy_provisioning':
                print("🐳 Docker: Will start automatically when needed")
            elif docker_status == 'not_needed':
                print("⚡ Docker: Not required for current mode")
            else:
                print(f"🐳 Docker: {docker_status}")
        
        # Start the secure chat interface
        print("\n🌐 Starting Secure Chat Interface...")
        print("🔗 Opening at: http://localhost:8502")
        print("💡 Use Ctrl+C to stop SAM")
        print("=" * 50)
        
        import subprocess
        subprocess.run([sys.executable, "secure_streamlit_app.py"], check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 SAM stopped by user")
    except Exception as e:
        print(f"\n❌ Failed to start SAM: {e}")
        print("💡 Fallback: Try running 'python secure_streamlit_app.py'")


def main():
    """Main function."""
    display_welcome_banner()
    
    # Test the enhanced system
    print("🧪 Testing Enhanced Startup System...")
    test_success = test_enhanced_startup()
    
    if test_success:
        print("\n✅ Enhanced startup system is working!")
        
        # Show strategies
        show_docker_strategy()
        show_new_user_recommendations()
        
        # Ask user preference
        print("\n🎯 STARTUP OPTIONS")
        print("=" * 20)
        print("1. Start SAM immediately (works without Docker)")
        print("2. Show Docker installation guide")
        print("3. Test Docker auto-provisioning")
        print("4. Exit")
        
        try:
            choice = input("\nChoose option (1-4): ").strip()
            
            if choice == "1":
                start_sam_with_enhanced_management()
            elif choice == "2":
                print("\n🐳 Docker Installation Guide:")
                print("• macOS: https://docs.docker.com/desktop/install/mac-install/")
                print("• Windows: https://docs.docker.com/desktop/install/windows-install/")
                print("• Linux: https://docs.docker.com/desktop/install/linux-install/")
                print("\n💡 After installing, restart this script for automatic Docker detection!")
            elif choice == "3":
                print("\n🧪 Testing Docker auto-provisioning...")
                from sam.code_interpreter.docker_auto_provisioner import get_docker_auto_provisioner
                provisioner = get_docker_auto_provisioner()
                result = provisioner.auto_provision_complete_environment()
                print(f"Result: {result['message']}")
            else:
                print("👋 Goodbye!")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
    else:
        print("\n⚠️ Enhanced startup system needs attention")
        print("💡 Falling back to basic startup...")
        
        # Fallback to basic startup
        import subprocess
        subprocess.run([sys.executable, "secure_streamlit_app.py"], check=False)


if __name__ == "__main__":
    main()
