"""
SAM Model Client - Unified Model Access
======================================

This module provides a unified client interface for all SAM components
to access the language model through the ModelInterface compatibility layer.

This replaces direct Ollama API calls throughout the SAM codebase and
enables seamless switching between Transformer and Hybrid models.

Author: SAM Development Team
Version: 1.0.0
"""

import logging
from typing import Optional, Dict, Any
from sam.core.model_interface import (
    get_model_manager, initialize_sam_models,
    GenerationRequest, GenerationResponse,
    ModelManager
)

logger = logging.getLogger(__name__)

class SAMModelClient:
    """Unified client for accessing SAM's language model."""
    
    def __init__(self):
        """Initialize the SAM model client."""
        self._manager: Optional[ModelManager] = None
        self._initialized = False
        
    def _ensure_initialized(self):
        """Ensure the model manager is initialized."""
        if not self._initialized:
            try:
                self._manager = initialize_sam_models()
                self._initialized = True
                logger.info("✅ SAM Model Client initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize SAM Model Client: {e}")
                raise
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500, 
                 top_p: float = 0.9, stop_sequences: Optional[list] = None) -> str:
        """
        Generate text using SAM's active model.
        
        Args:
            prompt: Input text prompt
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            stop_sequences: List of sequences to stop generation
            
        Returns:
            Generated text string
            
        Raises:
            Exception: If generation fails
        """
        self._ensure_initialized()
        
        try:
            request = GenerationRequest(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop_sequences=stop_sequences
            )
            
            response = self._manager.generate(request)
            
            if response.success:
                logger.debug(f"Generated {len(response.text)} characters in {response.inference_time:.2f}s")
                return response.text
            else:
                error_msg = f"Generation failed: {response.error_message}"
                logger.error(error_msg)
                raise Exception(error_msg)
                
        except Exception as e:
            logger.error(f"SAM Model Client generation error: {e}")
            raise
    
    def generate_with_metadata(self, prompt: str, temperature: float = 0.7, 
                              max_tokens: int = 500, top_p: float = 0.9, 
                              stop_sequences: Optional[list] = None) -> GenerationResponse:
        """
        Generate text with full response metadata.
        
        Args:
            prompt: Input text prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            stop_sequences: List of sequences to stop generation
            
        Returns:
            Complete GenerationResponse with metadata
        """
        self._ensure_initialized()
        
        request = GenerationRequest(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop_sequences=stop_sequences
        )
        
        return self._manager.generate(request)
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status and performance metrics."""
        self._ensure_initialized()
        return self._manager.get_model_status()
    
    def switch_to_hybrid_model(self):
        """Switch to using the hybrid model as primary."""
        self._ensure_initialized()
        from sam.core.model_interface import switch_to_hybrid_model
        switch_to_hybrid_model()
        logger.info("🔄 Switched to hybrid model")
    
    def switch_to_transformer_model(self):
        """Switch to using the transformer model as primary."""
        self._ensure_initialized()
        from sam.core.model_interface import switch_to_transformer_model
        switch_to_transformer_model()
        logger.info("🔄 Switched to transformer model")
    
    def get_current_model_info(self) -> Dict[str, Any]:
        """Get information about the currently active model."""
        self._ensure_initialized()
        from sam.core.model_interface import get_current_model_info
        return get_current_model_info()
    
    def health_check(self) -> bool:
        """Check if the model client is healthy and responsive."""
        try:
            self._ensure_initialized()
            status = self.get_model_status()
            
            # Check if at least one model is healthy
            for model_name, model_status in status.items():
                if model_status.get('health', False):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

# Global SAM model client instance
_sam_client = None

def get_sam_model_client() -> SAMModelClient:
    """Get the global SAM model client instance."""
    global _sam_client
    if _sam_client is None:
        _sam_client = SAMModelClient()
    return _sam_client

# Convenience functions for backward compatibility
def generate_sam_response(prompt: str, temperature: float = 0.7, 
                         max_tokens: int = 500) -> str:
    """
    Generate SAM response (backward compatibility function).
    
    Args:
        prompt: Input prompt
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        
    Returns:
        Generated response text
    """
    client = get_sam_model_client()
    return client.generate(prompt, temperature=temperature, max_tokens=max_tokens)

def create_ollama_compatible_client():
    """
    Create an Ollama-compatible client wrapper for legacy code.
    
    Returns:
        Client object with generate() method compatible with existing code
    """
    class OllamaCompatibleClient:
        def __init__(self):
            self.client = get_sam_model_client()
            self.base_url = "http://localhost:11434"  # For compatibility
            self.model_name = "sam-unified-model"     # Abstracted name
        
        def generate(self, prompt: str, temperature: float = 0.7, 
                    max_tokens: int = 500, stop_sequences: Optional[list] = None) -> str:
            """Generate text using SAM's unified model interface."""
            return self.client.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stop_sequences=stop_sequences
            )
    
    return OllamaCompatibleClient()

# Legacy support functions
class LegacyOllamaInterface:
    """Legacy Ollama interface for backward compatibility."""
    
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.model_name = "sam-unified-model"
        self.client = get_sam_model_client()
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
        """Generate response using unified SAM model interface."""
        return self.client.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )

def create_legacy_ollama_client():
    """Create legacy Ollama client for existing code compatibility."""
    return create_ollama_compatible_client()

# Configuration and model switching utilities
def configure_sam_model(model_type: str = "transformer"):
    """
    Configure SAM to use a specific model type.
    
    Args:
        model_type: "transformer" or "hybrid"
    """
    client = get_sam_model_client()
    
    if model_type.lower() == "hybrid":
        client.switch_to_hybrid_model()
    elif model_type.lower() == "transformer":
        client.switch_to_transformer_model()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_sam_model_performance() -> Dict[str, Any]:
    """Get SAM model performance metrics."""
    client = get_sam_model_client()
    return client.get_model_status()

# Model testing utilities
def test_sam_model_connection() -> bool:
    """Test SAM model connection and basic functionality."""
    try:
        client = get_sam_model_client()
        
        # Test basic generation
        response = client.generate("Hello", max_tokens=10)
        
        if response and len(response) > 0:
            logger.info("✅ SAM model connection test passed")
            return True
        else:
            logger.error("❌ SAM model connection test failed: empty response")
            return False
            
    except Exception as e:
        logger.error(f"❌ SAM model connection test failed: {e}")
        return False

def validate_sam_model_setup() -> Dict[str, Any]:
    """Validate SAM model setup and return diagnostic information."""
    client = get_sam_model_client()
    
    try:
        # Get model status
        status = client.get_model_status()
        
        # Test generation
        test_response = client.generate("Test", max_tokens=5)
        
        # Get current model info
        model_info = client.get_current_model_info()
        
        return {
            "status": "healthy",
            "model_status": status,
            "test_generation": {
                "success": len(test_response) > 0,
                "response_length": len(test_response)
            },
            "current_model": model_info,
            "health_check": client.health_check()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "health_check": False
        }

# Export main interfaces
__all__ = [
    'SAMModelClient',
    'get_sam_model_client',
    'generate_sam_response',
    'create_ollama_compatible_client',
    'create_legacy_ollama_client',
    'configure_sam_model',
    'get_sam_model_performance',
    'test_sam_model_connection',
    'validate_sam_model_setup'
]
