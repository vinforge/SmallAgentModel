#!/usr/bin/env python3
"""
SAM Model Setup Script
Downloads and initializes required models for SAM Community Edition.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_banner():
    """Print setup banner."""
    print("🤖 SAM MODEL SETUP")
    print("=" * 80)
    print("Setting up AI models for SAM Community Edition...")
    print("This will download required models for document processing and AI responses.")
    print("=" * 80)

def check_internet_connection():
    """Check if internet connection is available."""
    print("\n🌐 Checking internet connection...")
    try:
        import requests
        response = requests.get("https://huggingface.co", timeout=10)
        if response.status_code == 200:
            print("✅ Internet connection available")
            return True
        else:
            print("❌ Internet connection issues")
            return False
    except Exception as e:
        print(f"❌ No internet connection: {e}")
        return False

def setup_sentence_transformers():
    """Setup sentence transformers model for embeddings."""
    print("\n📚 Setting up Sentence Transformers (for document processing)...")
    print("This model enables SAM to understand and search through your documents.")
    
    try:
        # Import and initialize sentence transformers
        from sentence_transformers import SentenceTransformer
        
        print("⬇️  Downloading all-MiniLM-L6-v2 model (first time only)...")
        print("   Size: ~90MB - This may take a few minutes...")
        
        # Create models directory
        models_dir = Path("models/embeddings")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Download and cache the model
        model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=str(models_dir))
        
        # Test the model
        test_embedding = model.encode("This is a test sentence.")
        print(f"✅ Sentence Transformers model ready (dimension: {len(test_embedding)})")
        
        return True
        
    except ImportError:
        print("❌ sentence-transformers not installed")
        print("🔧 Installing sentence-transformers...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "sentence-transformers"], 
                         check=True, capture_output=True)
            print("✅ sentence-transformers installed")
            return setup_sentence_transformers()  # Retry
        except Exception as e:
            print(f"❌ Failed to install sentence-transformers: {e}")
            return False
    except Exception as e:
        print(f"❌ Error setting up sentence transformers: {e}")
        return False

def check_ollama_installation():
    """Check if Ollama is installed and running."""
    print("\n🦙 Checking Ollama installation...")
    
    try:
        # Check if ollama command exists
        result = subprocess.run(["ollama", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Ollama is installed")
            
            # Check if Ollama is running
            try:
                import requests
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code == 200:
                    print("✅ Ollama is running")
                    return True
                else:
                    print("⚠️  Ollama installed but not running")
                    print("🔧 Please start Ollama:")
                    print("   • Windows/Mac: Start Ollama app")
                    print("   • Linux: ollama serve")
                    return False
            except Exception:
                print("⚠️  Ollama installed but not running")
                print("🔧 Please start Ollama and run this script again")
                return False
        else:
            print("❌ Ollama not found")
            return False
            
    except FileNotFoundError:
        print("❌ Ollama not installed")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False

def setup_ollama_model():
    """Setup the main LLM model via Ollama."""
    print("\n🧠 Setting up Main AI Model (via Ollama)...")
    
    if not check_ollama_installation():
        print("\n📋 OLLAMA INSTALLATION REQUIRED:")
        print("SAM uses Ollama for AI responses. Please install it:")
        print("• Visit: https://ollama.ai")
        print("• Download and install Ollama for your platform")
        print("• Start Ollama, then run this script again")
        return False
    
    try:
        # Check what models are available
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            # Check if we have SAM's preferred model or suitable alternatives
            preferred_models = [
                "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M",  # SAM's default
                "qwen2.5:7b",
                "qwen2.5:3b",
                "llama3.2:3b",
                "llama3.2:1b",
                "phi3:mini"
            ]
            
            available_model = None
            for model in preferred_models:
                if any(model in name for name in model_names):
                    available_model = model
                    break
            
            if available_model:
                print(f"✅ Found suitable model: {available_model}")
                return True
            else:
                print("⚠️  No suitable model found")
                print("🔧 Downloading SAM's recommended model...")

                # Download SAM's default model (DeepSeek-R1 Qwen 8B Q4)
                model_to_download = "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M"
                print(f"⬇️  Downloading {model_to_download}")
                print("   This is SAM's default model (DeepSeek-R1 Qwen 8B, 4-bit quantized)")
                print("   Size: ~4.3GB - This may take 10-20 minutes depending on connection...")
                
                result = subprocess.run(
                    ["ollama", "pull", model_to_download],
                    capture_output=True, text=True, timeout=1800  # 30 minutes
                )
                
                if result.returncode == 0:
                    print(f"✅ SAM's default model downloaded successfully")
                    print("🎉 You now have SAM's full AI capabilities!")
                    return True
                else:
                    print(f"❌ Failed to download SAM's default model: {result.stderr}")
                    print("🔧 Trying fallback to smaller model...")

                    # Fallback to smaller model
                    fallback_model = "qwen2.5:3b"
                    print(f"⬇️  Downloading fallback model: {fallback_model}")

                    fallback_result = subprocess.run(
                        ["ollama", "pull", fallback_model],
                        capture_output=True, text=True, timeout=900  # 15 minutes
                    )

                    if fallback_result.returncode == 0:
                        print(f"✅ Fallback model {fallback_model} downloaded successfully")
                        return True
                    else:
                        print("❌ Fallback model download also failed")
                        print("🔧 You can manually download a model later:")
                        print("   ollama pull qwen2.5:3b")
                        print("   ollama pull hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M")
                        return False
        else:
            print("❌ Could not connect to Ollama")
            return False
            
    except Exception as e:
        print(f"❌ Error setting up Ollama model: {e}")
        return False

def create_model_config():
    """Create model configuration file."""
    print("\n⚙️  Creating model configuration...")
    
    config_content = """# SAM Model Configuration
# This file is created automatically by setup_models.py

[embedding_model]
name = "all-MiniLM-L6-v2"
provider = "sentence-transformers"
cache_dir = "models/embeddings"

[llm_model]
provider = "ollama"
api_url = "http://localhost:11434"
default_model = "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M"
fallback_models = ["qwen2.5:7b", "qwen2.5:3b", "llama3.2:3b", "phi3:mini"]

[model_settings]
max_context_length = 4096
temperature = 0.7
max_tokens = 1000
timeout_seconds = 60
"""
    
    try:
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)
        
        config_file = config_dir / "models.conf"
        with open(config_file, "w") as f:
            f.write(config_content)
        
        print(f"✅ Model configuration saved to {config_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating model config: {e}")
        return False

def show_setup_summary(embedding_success, ollama_success, config_success):
    """Show setup summary."""
    print(f"\n📊 MODEL SETUP SUMMARY")
    print("=" * 80)
    
    print(f"📚 Embedding Model (Document Processing): {'✅ SUCCESS' if embedding_success else '❌ FAILED'}")
    print(f"🧠 LLM Model (AI Responses): {'✅ SUCCESS' if ollama_success else '❌ FAILED'}")
    print(f"⚙️  Configuration: {'✅ SUCCESS' if config_success else '❌ FAILED'}")
    
    total_success = sum([embedding_success, ollama_success, config_success])
    
    print(f"\n📊 Overall: {total_success}/3 components ready")
    
    if total_success == 3:
        print(f"\n🎉 ALL MODELS READY!")
        print(f"✅ SAM is fully configured for AI functionality")
        print(f"✅ Document processing enabled")
        print(f"✅ AI responses enabled")
        print(f"🚀 You can now start SAM with: python start_sam.py")
        
    elif total_success >= 2:
        print(f"\n✅ MOSTLY READY!")
        print(f"✅ Core functionality available")
        if not ollama_success:
            print(f"⚠️  Install Ollama for full AI responses")
        
    else:
        print(f"\n❌ SETUP INCOMPLETE")
        print(f"❌ Some models failed to install")
        print(f"🔧 Check error messages above and try again")
    
    print(f"\n💡 WHAT THESE MODELS DO:")
    print(f"📚 Embedding Model (all-MiniLM-L6-v2): Document search and understanding")
    print(f"🧠 LLM Model (DeepSeek-R1 Qwen 8B Q4): Advanced AI responses and reasoning")
    print(f"⚙️  Configuration: Optimizes performance for your system")

if __name__ == "__main__":
    print_banner()
    
    # Check prerequisites
    if not check_internet_connection():
        print("\n❌ Internet connection required for model downloads")
        print("🔧 Please check your connection and try again")
        sys.exit(1)
    
    # Setup models
    embedding_success = setup_sentence_transformers()
    ollama_success = setup_ollama_model()
    config_success = create_model_config()
    
    # Show summary
    show_setup_summary(embedding_success, ollama_success, config_success)
    
    print(f"\n🏁 MODEL SETUP COMPLETE!")
    
    if embedding_success and ollama_success:
        print("🎉 SAM is ready with full AI capabilities!")
    elif embedding_success:
        print("📚 SAM ready for document processing (install Ollama for AI chat)")
    else:
        print("🔧 Please resolve issues above and run again")
    
    print(f"\n📋 NEXT STEPS:")
    print(f"1. ✅ Models configured")
    print(f"2. 🔐 Run: python setup_encryption.py (if not done)")
    print(f"3. 🚀 Run: python start_sam.py")
    
    print(f"\n🌟 Welcome to SAM Community Edition!")
