#!/usr/bin/env python3
"""
SAM Secure Launcher

Launches SAM with integrated security features.
Provides options for migration, secure mode, and legacy compatibility.

Author: SAM Development Team
Version: 1.0.0
"""

import os
import sys
import subprocess
import argparse
import time
import socket

# Suppress PyTorch/Streamlit compatibility warnings
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

def check_port_available(port):
    """Check if a port is available."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            return True
    except OSError:
        return False

def find_available_port(start_port, max_attempts=10):
    """Find an available port starting from start_port."""
    for i in range(max_attempts):
        port = start_port + i
        if check_port_available(port):
            return port
    return None

def kill_process_on_port(port):
    """Kill process running on specified port."""
    try:
        if sys.platform == "darwin":  # macOS
            subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, check=True)
            result = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True)
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    subprocess.run(["kill", "-9", pid], capture_output=True)
                print(f"✅ Killed process on port {port}")
                time.sleep(2)  # Give time for port to be released
                return True
        else:  # Linux/Windows
            subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)
            print(f"✅ Killed process on port {port}")
            time.sleep(2)
            return True
    except Exception as e:
        print(f"⚠️ Could not kill process on port {port}: {e}")
        return False

def print_banner():
    """Print SAM Secure Enclave banner."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          🧠 SAM SECURE ENCLAVE 🔒                           ║
║                     Your AI Assistant with Enterprise Security               ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        ('streamlit', 'Streamlit web framework'),
        ('flask', 'Flask web server'),
        ('chromadb', 'ChromaDB vector database'),
        ('argon2', 'Argon2 password hashing'),
        ('cryptography', 'Cryptography library')
    ]

    missing_packages = []

    for module_name, description in required_packages:
        try:
            # Special handling for argon2 which is installed as argon2-cffi
            if module_name == 'argon2':
                import argon2
            else:
                __import__(module_name)
            print(f"  ✅ {description}")
        except ImportError:
            print(f"  ❌ {description}")
            # Map module names to pip package names
            pip_name = {
                'argon2': 'argon2-cffi'
            }.get(module_name, module_name)
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All dependencies satisfied")
    return True

def check_security_setup():
    """Check if security is already set up."""
    try:
        from security import SecureStateManager
        from pathlib import Path

        # Check if keystore exists (primary indicator)
        keystore_path = Path("security/keystore.json")
        if not keystore_path.exists():
            print("🔧 Security setup required (keystore not found)")
            return False

        # Check if keystore is valid and get current state
        security_manager = SecureStateManager()
        current_state = security_manager.get_state()

        # Only require setup if state is SETUP_REQUIRED
        # LOCKED state means setup is complete, just needs authentication
        if security_manager.is_setup_required():
            print("🔧 Security setup required (keystore invalid)")
            return False
        else:
            print(f"✅ Security already configured (state: {current_state.value})")
            if current_state.value == 'locked':
                print("💡 System is ready - will prompt for password authentication in web interface")
            return True

    except ImportError:
        print("❌ Security module not available")
        return False
    except Exception as e:
        print(f"⚠️  Security check failed: {e}")
        return False

def run_encryption_setup():
    """Run encryption setup for new users."""
    print("\n🔐 Setting up SAM encryption...")

    try:
        import subprocess
        import sys

        # Run the encryption setup script
        result = subprocess.run([sys.executable, "setup_encryption.py"],
                               capture_output=False, text=True)

        if result.returncode == 0:
            print("✅ Encryption setup completed successfully!")
            return True
        else:
            print("❌ Encryption setup failed!")
            return False

    except FileNotFoundError:
        print("❌ setup_encryption.py not found")
        return False
    except Exception as e:
        print(f"❌ Encryption setup failed: {e}")
        return False

def run_migration():
    """Run data migration to encrypted format."""
    print("\n🔄 Starting data migration to encrypted format...")
    
    try:
        # Run migration script
        result = subprocess.run([
            sys.executable, 
            "scripts/migrate_to_secure_enclave.py"
        ], capture_output=False, text=True)
        
        if result.returncode == 0:
            print("✅ Migration completed successfully!")
            return True
        else:
            print("❌ Migration failed!")
            return False
            
    except Exception as e:
        print(f"❌ Migration error: {e}")
        return False

def launch_secure_streamlit():
    """Launch secure Streamlit application."""
    print("\n🚀 Launching SAM Secure Streamlit Application...")

    port = 8502

    # Check if port is available
    if not check_port_available(port):
        print(f"⚠️ Port {port} is already in use")

        # Try to kill existing process
        print(f"🔄 Attempting to free port {port}...")
        if kill_process_on_port(port):
            if check_port_available(port):
                print(f"✅ Port {port} is now available")
            else:
                print(f"❌ Port {port} still in use, finding alternative...")
                alt_port = find_available_port(8503)
                if alt_port:
                    port = alt_port
                    print(f"🔄 Using alternative port {port}")
                else:
                    print("❌ No available ports found")
                    return
        else:
            # Find alternative port
            alt_port = find_available_port(8503)
            if alt_port:
                port = alt_port
                print(f"🔄 Using alternative port {port}")
            else:
                print("❌ No available ports found")
                return

    try:
        print(f"🌐 Starting SAM at: http://localhost:{port}")
        # Launch secure Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "secure_streamlit_app.py",
            f"--server.port={port}",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ])

    except KeyboardInterrupt:
        print("\n👋 SAM Secure Streamlit stopped by user")
    except Exception as e:
        print(f"❌ Failed to launch Secure Streamlit: {e}")

def launch_secure_web_ui():
    """Launch secure web UI."""
    print("\n🌐 Launching SAM Secure Web UI...")
    
    try:
        # Legacy web UI removed - redirect to Streamlit
        print("🌐 Legacy web UI has been removed. Please use the Streamlit interface:")
        print("   http://localhost:8502")
        return
        
    except KeyboardInterrupt:
        print("\n👋 SAM Secure Web UI stopped by user")
    except Exception as e:
        print(f"❌ Failed to launch Secure Web UI: {e}")

def launch_memory_ui():
    """Launch standalone Memory Control Center with Dream Canvas access."""
    print("\n🧠 Memory Control Center with Dream Canvas...")
    print("\n✨ **STANDALONE MEMORY CENTER:**")
    print("   🎨 Includes full Dream Canvas functionality")
    print("   🔗 Access it at: http://localhost:8503")
    print("   🧠 Advanced memory visualization and synthesis")
    print("   🌙 Cognitive landscape generation")
    print("   🔬 Research integration and insights")
    print("\n🚀 Launching Memory Control Center...")

    try:
        # Launch standalone memory control center with Dream Canvas
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "ui/memory_app.py",
            "--server.port=8503",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ])

    except KeyboardInterrupt:
        print("\n👋 Memory Control Center stopped by user")
    except Exception as e:
        print(f"❌ Failed to launch Memory Control Center: {e}")

def launch_both_interfaces():
    """Launch both Secure SAM Interface and Memory Control Center."""
    print("\n🚀 Launching Both SAM Interfaces...")
    print("This will start:")
    print("  🔒 Secure SAM Interface (port 8502) - Main SAM functionality")
    print("  🧠 Memory Control Center (port 8503) - Dream Canvas & Memory Management")
    print("\n✨ **DUAL INTERFACE MODE:**")
    print("   🔗 Main SAM: http://localhost:8502")
    print("   🎨 Memory Center: http://localhost:8503")
    print("   🌙 Dream Canvas available in Memory Center")

    processes = []

    try:
        # Launch Secure Streamlit Interface
        print("\n🔒 Starting Secure SAM Interface...")
        streamlit_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run",
            "secure_streamlit_app.py",
            "--server.port=8502",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ])
        processes.append(("Secure SAM", streamlit_process))
        time.sleep(3)  # Give it time to start

        # Launch Memory Control Center
        print("🧠 Starting Memory Control Center...")
        memory_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run",
            "ui/memory_app.py",
            "--server.port=8503",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ])
        processes.append(("Memory Center", memory_process))
        time.sleep(2)

        print("\n✅ Both interfaces started successfully!")
        print("\n🌐 Available interfaces:")
        print("  • 🔒 Secure SAM Interface: http://localhost:8502")
        print("  • 🧠 Memory Control Center: http://localhost:8503")
        print("  • 🎨 Dream Canvas: Available in Memory Center dropdown")
        print("\n💡 **Usage Tips:**")
        print("  • Use Secure SAM for main conversations and document uploads")
        print("  • Use Memory Center for advanced memory management and Dream Canvas")
        print("  • Both interfaces share the same memory store")
        print("\n⚠️  Press Ctrl+C to stop both services")

        # Wait for processes
        try:
            while True:
                time.sleep(1)
                # Check if any process has died
                for name, process in processes:
                    if process.poll() is not None:
                        print(f"\n⚠️ {name} process has stopped")
                        return
        except KeyboardInterrupt:
            print("\n👋 Stopping both interfaces...")
            for name, process in processes:
                print(f"   Stopping {name}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            print("✅ Both interfaces stopped")

    except Exception as e:
        print(f"❌ Failed to launch interfaces: {e}")
        # Clean up any started processes
        for name, process in processes:
            try:
                process.terminate()
            except:
                pass

def launch_full_suite():
    """Launch full SAM suite with security."""
    print("\n🚀 Launching Full SAM Secure Suite...")
    print("This will start:")
    print("  📱 Secure Streamlit App (port 8502) - Primary interface with integrated Memory Center")
    print("  🌐 Secure Web UI (port 5001)")
    print("  ✅ Memory Control Center is now integrated into the secure interface")

    processes = []

    try:
        # Launch Secure Streamlit FIRST (primary authentication interface with integrated memory center)
        print("\n📱 Starting Secure Streamlit App (Primary Interface with Memory Center)...")
        streamlit_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run",
            "secure_streamlit_app.py",
            "--server.port=8502",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false",
            "--server.enableCORS=false",
            "--server.enableXsrfProtection=false"
        ])
        processes.append(("Secure Streamlit", streamlit_process))
        time.sleep(3)  # Give more time for primary interface to start

        # Legacy Web UI removed - only using Streamlit interface
        print("🌐 Using Streamlit interface only (legacy web UI removed)")
        time.sleep(1)

        print("✅ Memory Control Center integrated into Secure Streamlit App")
        print("   Access via the 'Memory Center' tab at http://localhost:8502")

        print("\n✅ All services started successfully!")
        print("\n🔐 **IMPORTANT - Authentication Required:**")
        print("  1. SAM will open automatically at: http://localhost:8502")
        print("  2. Enter your master password to unlock SAM")
        print("  3. Use the navigation buttons in SAM to access other interfaces")
        print("\n🌐 Available interfaces:")
        print("  • 🔑 Secure SAM Interface: http://localhost:8502 (OPENS AUTOMATICALLY)")
        print("  • 🧠 Memory Control Center: Integrated into main interface")
        print("\n🎯 **SAM Pro Activation:**")
        print("  • Look for '🔑 SAM Pro Activation' in the sidebar after authentication")
        print("  • Enter your activation key to unlock premium features")
        print("  • Premium features include TPV Active Reasoning, Dream Canvas, and more")
        print("\n🔑 **Need an Activation Key?**")
        print("  • Register with: python register_sam_pro.py")
        print("  • This starts the registration interface at localhost:8503")
        print("  • Keys are delivered automatically via email")
        print("\n💡 **Tip**: After authentication, use the navigation buttons in the")
        print("    SAM sidebar to easily access the Memory Control Center and Web UI.")
        print("\n⚠️  Press Ctrl+C to stop all services")
        
        # Wait for processes
        while True:
            time.sleep(1)
            # Check if any process died
            for name, process in processes:
                if process.poll() is not None:
                    print(f"⚠️  {name} stopped unexpectedly")
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping all SAM services...")
        for name, process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"  ✅ {name} stopped")
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"  🔪 {name} force killed")
            except Exception as e:
                print(f"  ⚠️  Error stopping {name}: {e}")
        
        print("👋 SAM Secure Suite stopped")
    
    except Exception as e:
        print(f"❌ Failed to launch full suite: {e}")
        # Cleanup
        for name, process in processes:
            try:
                process.terminate()
            except Exception:
                pass

def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(description="SAM Secure Enclave Launcher")
    parser.add_argument("--mode", choices=["web", "streamlit", "memory", "both", "full", "migrate"],
                       default="full", help="Launch mode: web=Web UI only, streamlit=Secure SAM only, memory=Memory Center only, both=Both interfaces, full=Complete suite, migrate=Data migration")
    parser.add_argument("--skip-checks", action="store_true", help="Skip dependency checks")
    parser.add_argument("--force-migration", action="store_true", help="Force data migration")
    
    args = parser.parse_args()
    
    print_banner()
    
    # Check dependencies
    if not args.skip_checks:
        if not check_dependencies():
            sys.exit(1)
    
    # Handle migration mode
    if args.mode == "migrate" or args.force_migration:
        if run_migration():
            print("\n✅ Migration completed! You can now launch SAM securely.")
        else:
            print("\n❌ Migration failed! Please check the logs.")
        return
    
    # Check security setup
    security_ready = check_security_setup()

    if not security_ready:
        print("\n🔧 Security setup required.")
        print("This is your first time running SAM. Let's set up your master password.")

        # Ask user if they want to run setup now or if they already did it
        try:
            response = input("\n❓ Have you already run the setup process (setup.py)? (y/n) [n]: ").strip().lower()
            if response in ['y', 'yes']:
                print("💡 If you've already run setup.py, the keystore might not be properly created.")
                print("💡 You can try running: python setup_encryption.py")
                print("💡 Or continue with automatic setup below.")

                continue_setup = input("❓ Continue with automatic setup? (y/n) [y]: ").strip().lower()
                if continue_setup in ['n', 'no']:
                    print("⏭️ Skipping automatic setup. Please run setup_encryption.py manually.")
                    return
        except KeyboardInterrupt:
            print("\n👋 Setup cancelled by user")
            return

        # Automatically run encryption setup for new users
        print("\n🔐 Starting encryption setup...")
        success = run_encryption_setup()

        if not success:
            print("❌ Encryption setup failed. You can also try:")
            print("  1. Run migration: python start_sam_secure.py --mode migrate")
            print("  2. Manual setup: python setup_encryption.py")
            print("  3. Re-run setup: python setup.py")
            return

        print("✅ Encryption setup completed! Continuing with SAM launch...")
    
    # Launch based on mode
    if args.mode == "web":
        launch_secure_web_ui()
    elif args.mode == "streamlit":
        launch_secure_streamlit()
    elif args.mode == "memory":
        launch_memory_ui()
    elif args.mode == "both":
        launch_both_interfaces()
    elif args.mode == "full":
        launch_full_suite()
    else:
        print(f"❌ Unknown mode: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
