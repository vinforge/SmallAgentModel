#!/usr/bin/env python3
"""
SAM Setup Wizard UI
==================

Streamlit-based setup wizard for first-time SAM users.
Guides users through master password creation, SAM Pro activation,
and basic onboarding.
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def show_setup_wizard():
    """Display the setup wizard interface."""
    try:
        from utils.first_time_setup import get_first_time_setup_manager
        setup_manager = get_first_time_setup_manager()
        
        # Get setup progress
        progress = setup_manager.get_setup_progress()
        next_step = progress['next_step']
        
        # Header
        st.markdown("# 🚀 Welcome to SAM!")
        st.markdown("### Let's get you set up in just a few steps")
        
        # Progress bar
        progress_bar = st.progress(progress['progress_percent'] / 100)
        st.markdown(f"**Setup Progress:** {progress['completed_steps']}/{progress['total_steps']} steps complete")
        
        st.divider()
        
        # Show appropriate setup step
        if next_step == 'master_password':
            show_master_password_setup(setup_manager)
        elif next_step == 'sam_pro_activation':
            show_sam_pro_activation(setup_manager)
        elif next_step == 'onboarding':
            show_onboarding_tour(setup_manager)
        else:
            show_setup_complete()
            
    except Exception as e:
        st.error(f"Setup wizard error: {e}")
        st.markdown("### 🔧 Manual Setup")
        st.markdown("If you're seeing this error, you can set up SAM manually:")
        st.markdown("1. Create your master password in the Security section")
        st.markdown("2. Enter your SAM Pro activation key")
        st.markdown("3. Start using SAM!")

def show_master_password_setup(setup_manager):
    """Show master password creation step."""
    st.markdown("## 🔐 Step 1: Create Master Password")
    st.markdown("""
    Your master password protects all SAM data with enterprise-grade encryption.
    
    **Important:**
    - Choose a strong, memorable password
    - This password encrypts all your conversations and data
    - You'll need this password every time you start SAM
    """)
    
    with st.form("master_password_form"):
        password = st.text_input("Master Password", type="password", 
                                help="Choose a strong password (8+ characters)")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        if st.form_submit_button("Create Master Password", type="primary"):
            if not password:
                st.error("Please enter a password")
            elif len(password) < 8:
                st.error("Password must be at least 8 characters")
            elif password != confirm_password:
                st.error("Passwords don't match")
            else:
                # Create master password
                if create_master_password(password):
                    setup_manager.update_setup_status('master_password_created', True)
                    st.success("✅ Master password created successfully!")
                    st.rerun()
                else:
                    st.error("Failed to create master password. Please try again.")

def show_sam_pro_activation(setup_manager):
    """Show SAM Pro activation step."""
    st.markdown("## 🔑 Step 2: Activate SAM Pro")

    # Get the SAM Pro key
    sam_pro_key = setup_manager.get_sam_pro_key()

    if sam_pro_key:
        st.markdown("### Your SAM Pro Activation Key:")
        st.code(sam_pro_key, language=None)
        st.markdown("**💾 Important: Save this key!** You can use it to activate SAM Pro on other devices.")

        st.markdown("""
        ### 🎉 SAM Pro Features Included:
        - 🧠 **Cognitive Distillation Engine** - Learn from every interaction
        - 🧠 **TPV Active Reasoning Control** - Advanced reasoning capabilities
        - 📚 **MEMOIR Lifelong Learning** - Persistent memory across sessions
        - 🎨 **Dream Canvas Visualization** - Interactive memory landscapes
        - 🤖 **Cognitive Automation** - Autonomous task execution
        - 📊 **Advanced Analytics** - Deep insights and performance metrics
        """)

        if st.button("✅ Activate SAM Pro Features", type="primary"):
            # Activate SAM Pro
            if activate_sam_pro(sam_pro_key):
                setup_manager.update_setup_status('sam_pro_activated', True)
                st.success("🎉 SAM Pro activated successfully!")
                st.rerun()
            else:
                st.error("Failed to activate SAM Pro. Please try again.")
    else:
        # No key found - offer registration options
        st.warning("No SAM Pro key found.")

        st.markdown("### 🔑 Get Your SAM Pro Key")
        st.markdown("Choose how you'd like to get your free SAM Pro activation key:")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🌐 Web Registration", type="primary"):
                st.markdown("### 🌐 Opening Registration Interface...")
                st.markdown("We're starting the SAM Pro registration interface for you.")
                st.info("📱 **Registration will open at:** http://localhost:8503")

                # Start the registration interface
                if start_pro_registration():
                    st.success("✅ Registration interface started!")
                    st.markdown("**Next Steps:**")
                    st.markdown("1. Complete registration at http://localhost:8503")
                    st.markdown("2. Check your email for the activation key")
                    st.markdown("3. Return here and refresh to continue setup")
                else:
                    st.error("❌ Failed to start registration interface")
                    st.markdown("💡 Try the quick registration option instead")

        with col2:
            if st.button("⚡ Quick Registration"):
                st.markdown("### ⚡ Quick SAM Pro Registration")

                with st.form("quick_registration"):
                    email = st.text_input("Email Address", placeholder="your@email.com")
                    name = st.text_input("Name (Optional)", placeholder="Your Name")

                    if st.form_submit_button("🔑 Get My SAM Pro Key"):
                        if email:
                            # Generate key quickly
                            key = generate_quick_pro_key(email, name)
                            if key:
                                st.success("🎉 SAM Pro key generated!")
                                st.code(key, language=None)
                                st.markdown("**💾 Save this key!**")

                                # Update setup manager
                                setup_manager.update_setup_status('sam_pro_activated', True)
                                st.rerun()
                            else:
                                st.error("Failed to generate key. Please try again.")
                        else:
                            st.error("Please enter your email address")

        st.markdown("---")
        if st.button("⏭️ Skip Pro Activation (Use Basic SAM)"):
            setup_manager.update_setup_status('sam_pro_activated', True)
            st.info("Skipped Pro activation. You can activate later in Settings.")
            st.rerun()

def show_onboarding_tour(setup_manager):
    """Show onboarding tour step."""
    st.markdown("## 🎓 Step 3: Welcome Tour")
    st.markdown("### You're almost ready! Let's quickly show you around SAM.")
    
    # Quick feature overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 💬 Chat Interface
        - Natural conversation with SAM
        - Upload documents for analysis
        - Ask questions about your data
        - Get intelligent insights
        """)
        
        st.markdown("""
        ### 🧠 Memory System
        - SAM remembers your conversations
        - Learns your preferences
        - Builds knowledge over time
        - Connects related information
        """)
    
    with col2:
        st.markdown("""
        ### 🎨 Dream Canvas
        - Visualize your memory landscape
        - Explore knowledge connections
        - Discover insights and patterns
        - Interactive memory exploration
        """)
        
        st.markdown("""
        ### 🔧 Advanced Features
        - Cognitive Automation
        - Document processing
        - Web research capabilities
        - Custom workflows
        """)
    
    st.markdown("### 🚀 Ready to start using SAM?")
    
    if st.button("🎉 Complete Setup & Start Using SAM!", type="primary"):
        setup_manager.update_setup_status('onboarding_completed', True)
        st.success("✅ Setup complete! Welcome to SAM!")
        st.rerun()

def show_setup_complete():
    """Show setup completion screen."""
    st.markdown("# 🎉 Setup Complete!")
    st.markdown("### Welcome to SAM - You're all set!")
    
    st.success("✅ All setup steps completed successfully")
    
    st.markdown("""
    ### 🚀 What you can do now:
    - Start chatting with SAM using the interface below
    - Upload documents to build your knowledge base
    - Explore the Memory Control Center
    - Try the Dream Canvas visualization
    - Access all SAM Pro features
    """)
    
    st.markdown("### 💡 Quick Tips:")
    st.markdown("- Use the sidebar to access different features")
    st.markdown("- Upload PDFs, documents, or text files for analysis")
    st.markdown("- Ask SAM questions about your uploaded content")
    st.markdown("- Explore the Memory section to see how SAM learns")
    
    if st.button("🚀 Start Using SAM", type="primary"):
        # Redirect to main chat interface
        st.switch_page("secure_streamlit_app.py")

def create_master_password(password: str) -> bool:
    """Create master password for encryption."""
    try:
        from security import SecureStateManager
        security_manager = SecureStateManager()

        # Initialize security system with the password
        success = security_manager.initialize_security(password)
        return success

    except Exception as e:
        st.error(f"Error creating master password: {e}")
        return False

def start_pro_registration() -> bool:
    """Start the SAM Pro registration interface on localhost:8503."""
    try:
        import subprocess
        import sys

        # Start the registration interface
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run",
            "sam_pro_registration.py",
            "--server.port=8503",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false",
            "--server.headless=true"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Give it a moment to start
        import time
        time.sleep(2)

        # Check if it's still running
        if process.poll() is None:
            return True
        else:
            return False

    except Exception as e:
        st.error(f"Error starting registration: {e}")
        return False

def generate_quick_pro_key(email: str, name: str = "") -> str:
    """Generate a quick SAM Pro key."""
    try:
        import uuid
        import json
        import hashlib
        from datetime import datetime
        from pathlib import Path

        # Generate key
        activation_key = str(uuid.uuid4())

        # Add to keystore
        security_dir = Path("security")
        security_dir.mkdir(exist_ok=True)
        keystore_file = security_dir / "keystore.json"

        keystore = {}
        if keystore_file.exists():
            try:
                with open(keystore_file, 'r') as f:
                    keystore = json.load(f)
            except:
                keystore = {}

        keystore[activation_key] = {
            'email': email,
            'name': name,
            'created_date': datetime.now().isoformat(),
            'key_type': 'sam_pro_free',
            'status': 'active',
            'source': 'setup_wizard'
        }

        with open(keystore_file, 'w') as f:
            json.dump(keystore, f, indent=2)

        # Add key hash to entitlements for validation
        add_key_hash_to_entitlements_config(activation_key)

        # Save key for future reference
        try:
            from utils.first_time_setup import get_first_time_setup_manager
            setup_manager = get_first_time_setup_manager()
            setup_manager.update_setup_status('sam_pro_key', activation_key)
        except:
            pass  # Continue even if this fails

        return activation_key

    except Exception as e:
        st.error(f"Error generating key: {e}")
        return None

def add_key_hash_to_entitlements_config(activation_key: str):
    """Add key hash to entitlements configuration for validation."""
    try:
        import hashlib
        import json
        from datetime import datetime
        from pathlib import Path

        # Generate SHA-256 hash of the key
        key_hash = hashlib.sha256(activation_key.encode('utf-8')).hexdigest()

        # Load entitlements config from sam/config/entitlements.json
        entitlements_config_file = Path("sam/config/entitlements.json")
        if entitlements_config_file.exists():
            with open(entitlements_config_file, 'r') as f:
                config = json.load(f)
        else:
            # Create basic config if it doesn't exist
            config = {
                "version": "1.1",
                "features": {},
                "valid_key_hashes": [],
                "metadata": {
                    "generated": datetime.now().isoformat(),
                    "total_keys": 0,
                    "hash_algorithm": "SHA-256"
                }
            }

        # Add the hash to valid_key_hashes if not already present
        if "valid_key_hashes" not in config:
            config["valid_key_hashes"] = []

        if key_hash not in config["valid_key_hashes"]:
            config["valid_key_hashes"].append(key_hash)

            # Update metadata
            if "metadata" not in config:
                config["metadata"] = {}
            config["metadata"]["last_updated"] = datetime.now().isoformat()
            config["metadata"]["total_keys"] = len(config["valid_key_hashes"])

            # Ensure sam/config directory exists
            entitlements_config_file.parent.mkdir(parents=True, exist_ok=True)

            # Save updated config
            with open(entitlements_config_file, 'w') as f:
                json.dump(config, f, indent=2)

    except Exception as e:
        st.error(f"Failed to add key hash to entitlements: {e}")

def activate_sam_pro(activation_key: str) -> bool:
    """Activate SAM Pro features."""
    try:
        # The key should already be in the keystore from setup
        # Just need to mark it as activated in the entitlements
        from pathlib import Path
        import json

        entitlements_file = Path("security/entitlements.json")
        if entitlements_file.exists():
            with open(entitlements_file, 'r') as f:
                entitlements = json.load(f)

            # Mark SAM Pro as activated
            entitlements["sam_pro_keys"][activation_key] = {
                "activated": True,
                "activation_date": "2025-01-01T00:00:00",
                "features": [
                    "tpv_active_reasoning",
                    "enhanced_slp_learning",
                    "memoir_lifelong_learning",
                    "dream_canvas",
                    "cognitive_distillation",
                    "cognitive_automation"
                ]
            }

            with open(entitlements_file, 'w') as f:
                json.dump(entitlements, f, indent=2)

            return True

        return False

    except Exception as e:
        st.error(f"Error activating SAM Pro: {e}")
        return False

if __name__ == "__main__":
    show_setup_wizard()
