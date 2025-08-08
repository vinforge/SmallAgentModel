#!/usr/bin/env python3
"""
SAM 2.0 Startup Script with Config-Driven Model Selection
=========================================================

This script demonstrates the config-driven model switching capability
and can be used to start SAM with different model configurations.

Usage:
    python sam_startup.py --model transformer  # Use transformer model
    python sam_startup.py --model hybrid       # Use hybrid model
    python sam_startup.py --auto               # Auto-detect best model
    python sam_startup.py --config             # Show current configuration

Author: SAM Development Team
Version: 1.0.0
"""

import sys
import argparse
import logging
from pathlib import Path

# Add SAM core to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sam.config import (
    get_sam_config, set_model_backend, get_current_model_backend,
    ModelBackend, is_hybrid_model_enabled, validate_config
)
from sam.core.sam_model_client import get_sam_model_client, validate_sam_model_setup

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SAMStartup:
    """Handles SAM startup with configuration management."""
    
    def __init__(self):
        self.config = None
        self.model_client = None
        
    def load_configuration(self) -> bool:
        """Load and validate SAM configuration."""
        try:
            self.config = get_sam_config()
            validate_config(self.config)
            
            logger.info("✅ SAM configuration loaded and validated")
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration error: {e}")
            return False
    
    def show_configuration(self):
        """Display current SAM configuration."""
        if not self.config:
            self.load_configuration()
        
        print("\n" + "="*60)
        print("SAM 2.0 CONFIGURATION")
        print("="*60)
        
        # Model configuration
        print(f"Model Backend: {self.config.model.backend.value}")
        print(f"Model Size: {self.config.model.size.value}")
        print(f"Max Context Length: {self.config.model.max_context_length:,} tokens")
        print(f"Temperature: {self.config.model.temperature}")
        print(f"Max Tokens: {self.config.model.max_tokens}")
        
        # Application settings
        print(f"\nApplication Version: {self.config.version}")
        print(f"Debug Mode: {self.config.debug_mode}")
        print(f"Log Level: {self.config.log_level}")
        
        # Ports
        print(f"\nStreamlit Port: {self.config.streamlit_port}")
        print(f"Memory Center Port: {self.config.memory_center_port}")
        print(f"Welcome Page Port: {self.config.welcome_page_port}")
        
        # Feature flags
        print(f"\nFeatures Enabled:")
        print(f"  Memory Center: {self.config.enable_memory_center}")
        print(f"  Dream Canvas: {self.config.enable_dream_canvas}")
        print(f"  Cognitive Automation: {self.config.enable_cognitive_automation}")
        print(f"  Self-Reflect: {self.config.enable_self_reflect}")
        
        print("="*60)
    
    def set_model_backend(self, backend: str) -> bool:
        """
        Set the model backend.
        
        Args:
            backend: "transformer", "hybrid", or "auto"
            
        Returns:
            True if successful
        """
        try:
            if backend == "auto":
                # Auto-detect best available model
                backend = self.auto_detect_model()
            
            success = set_model_backend(backend)
            
            if success:
                logger.info(f"✅ Model backend set to: {backend}")
                return True
            else:
                logger.error(f"❌ Failed to set model backend to: {backend}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error setting model backend: {e}")
            return False
    
    def auto_detect_model(self) -> str:
        """
        Auto-detect the best available model backend.
        
        Returns:
            "hybrid" or "transformer"
        """
        try:
            # Try to initialize hybrid model
            from sam.models import create_sam_hybrid_model, HybridModelConfigs
            
            # Test with debug config to avoid memory issues
            debug_config = HybridModelConfigs.sam_debug_hybrid()
            test_model = create_sam_hybrid_model(debug_config)
            
            if test_model is not None:
                logger.info("🔍 Hybrid model available - selecting hybrid backend")
                return "hybrid"
            else:
                logger.info("🔍 Hybrid model not available - selecting transformer backend")
                return "transformer"
                
        except Exception as e:
            logger.warning(f"🔍 Hybrid model test failed: {e} - selecting transformer backend")
            return "transformer"
    
    def initialize_model_client(self) -> bool:
        """Initialize the SAM model client."""
        try:
            self.model_client = get_sam_model_client()
            
            # Validate setup
            validation = validate_sam_model_setup()
            
            if validation.get('status') == 'healthy':
                logger.info("✅ SAM model client initialized successfully")
                return True
            else:
                logger.warning(f"⚠️ SAM model client initialized with warnings: {validation.get('error', 'Unknown')}")
                return True  # Still functional
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize model client: {e}")
            return False
    
    def test_model_functionality(self) -> bool:
        """Test basic model functionality."""
        try:
            if not self.model_client:
                logger.error("Model client not initialized")
                return False
            
            # Test basic generation
            response = self.model_client.generate("Hello", max_tokens=5)
            
            if response and len(response) > 0:
                logger.info(f"✅ Model test successful: '{response.strip()}'")
                return True
            else:
                logger.error("❌ Model test failed: empty response")
                return False
                
        except Exception as e:
            logger.error(f"❌ Model test failed: {e}")
            return False
    
    def start_sam(self, model_backend: str = None) -> bool:
        """
        Start SAM with the specified configuration.
        
        Args:
            model_backend: Model backend to use (optional)
            
        Returns:
            True if successful
        """
        logger.info("🚀 Starting SAM 2.0...")
        
        # Load configuration
        if not self.load_configuration():
            return False
        
        # Set model backend if specified
        if model_backend:
            if not self.set_model_backend(model_backend):
                return False
        
        # Initialize model client
        if not self.initialize_model_client():
            return False
        
        # Test functionality
        if not self.test_model_functionality():
            logger.warning("⚠️ Model functionality test failed, but continuing...")
        
        # Show final configuration
        current_backend = get_current_model_backend()
        logger.info(f"🎯 SAM 2.0 started successfully with {current_backend.value} model")
        
        return True
    
    def get_status(self) -> dict:
        """Get SAM status information."""
        try:
            current_backend = get_current_model_backend()
            hybrid_enabled = is_hybrid_model_enabled()
            
            status = {
                "sam_version": "2.0.0",
                "model_backend": current_backend.value,
                "hybrid_enabled": hybrid_enabled,
                "configuration_loaded": self.config is not None,
                "model_client_initialized": self.model_client is not None
            }
            
            if self.model_client:
                model_status = self.model_client.get_model_status()
                status["model_status"] = model_status
            
            return status
            
        except Exception as e:
            return {"error": str(e)}

def main():
    """Main startup function."""
    parser = argparse.ArgumentParser(description="SAM 2.0 Startup Script")
    # Get available models dynamically
    try:
        from sam.config import ModelBackend
        available_models = ModelBackend.get_available_backends()
    except ImportError:
        available_models = ["transformer", "hybrid", "auto"]

    parser.add_argument("--model", choices=available_models,
                       help="Model backend to use")
    parser.add_argument("--config", action="store_true", 
                       help="Show current configuration")
    parser.add_argument("--status", action="store_true",
                       help="Show SAM status")
    parser.add_argument("--test", action="store_true",
                       help="Test model functionality")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        startup = SAMStartup()
        
        # Handle different commands
        if args.config:
            startup.show_configuration()
            return 0
        
        if args.status:
            status = startup.get_status()
            print("\nSAM Status:")
            for key, value in status.items():
                print(f"  {key}: {value}")
            return 0
        
        if args.test:
            if startup.start_sam():
                success = startup.test_model_functionality()
                return 0 if success else 1
            else:
                return 1
        
        # Normal startup
        model_backend = args.model
        success = startup.start_sam(model_backend)
        
        if success:
            print("\n🎉 SAM 2.0 is ready!")
            print("You can now use SAM with the configured model backend.")
            
            # Show quick status
            status = startup.get_status()
            print(f"\nActive Model: {status.get('model_backend', 'unknown')}")
            print(f"Hybrid Enabled: {status.get('hybrid_enabled', False)}")
            
            return 0
        else:
            print("\n❌ SAM startup failed. Check logs for details.")
            return 1
            
    except KeyboardInterrupt:
        print("\n🛑 Startup interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Fatal startup error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
