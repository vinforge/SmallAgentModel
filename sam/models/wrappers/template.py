"""
SAM Model Foundry: ModelWrapper Template
=======================================

This template provides a standardized interface for integrating new models
into SAM's ModelInterface system. Copy this file and implement the abstract
methods to add support for any new model.

Usage:
1. Copy this file to a new wrapper (e.g., llama31_wrapper.py)
2. Implement all abstract methods
3. Add model to configuration system
4. Test with benchmark suite

Author: SAM Development Team
Version: 1.0.0
"""

import abc
import time
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from pathlib import Path

from sam.core.model_interface import ModelInterface, GenerationRequest, GenerationResponse

logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    """Standardized metadata for model comparison and leaderboard."""
    name: str
    version: str
    parameters: str  # e.g., "8B", "70B"
    context_window: int
    license: str
    provider: str  # e.g., "HuggingFace", "Anthropic", "OpenAI"
    model_family: str  # e.g., "Llama", "Mistral", "DeepSeek"
    architecture: str  # e.g., "Transformer", "Mamba", "Hybrid"
    quantization: Optional[str] = None  # e.g., "4bit", "8bit", "fp16"
    fine_tuned: bool = False
    release_date: Optional[str] = None
    paper_url: Optional[str] = None
    model_url: Optional[str] = None

@dataclass
class ModelCapabilities:
    """Model capability flags for benchmark selection."""
    supports_function_calling: bool = False
    supports_code_generation: bool = True
    supports_reasoning: bool = True
    supports_multimodal: bool = False
    supports_streaming: bool = True
    max_output_tokens: int = 4096
    languages_supported: List[str] = None
    
    def __post_init__(self):
        if self.languages_supported is None:
            self.languages_supported = ["en"]  # Default to English

@dataclass
class ModelCostEstimate:
    """Cost estimation for budget analysis."""
    cost_per_input_token: float = 0.0  # USD per token
    cost_per_output_token: float = 0.0  # USD per token
    setup_cost: float = 0.0  # One-time setup cost
    monthly_hosting_cost: float = 0.0  # Monthly hosting if applicable
    currency: str = "USD"
    
    def estimate_request_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a single request."""
        return (input_tokens * self.cost_per_input_token + 
                output_tokens * self.cost_per_output_token)

class ModelWrapperTemplate(ModelInterface):
    """
    Abstract base class for all SAM model wrappers.
    
    This template defines the standard interface that all model wrappers
    must implement to be compatible with SAM's evaluation framework.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model wrapper.

        Args:
            config: Model configuration dictionary
        """
        # Create a ModelConfig for the parent class
        from sam.core.model_interface import ModelConfig, ModelType

        model_config = ModelConfig(
            model_type=ModelType.TRANSFORMER,
            model_name=config.get("model_name", "unknown"),
            api_url=config.get("api_url", ""),
            max_context_length=config.get("max_context_length", 32000),
            timeout_seconds=config.get("timeout_seconds", 120),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 1000)
        )

        super().__init__(model_config)
        self.config = config
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
        self._load_time = None
        self._total_requests = 0
        self._total_tokens_generated = 0
        self._total_inference_time = 0.0
        
        # Initialize from config
        self.model_name = config.get("model_name", "unknown")
        self.device = config.get("device", "auto")
        self.max_memory = config.get("max_memory", None)
        
    @abc.abstractmethod
    def load_model(self) -> bool:
        """
        Load the model and tokenizer.
        
        Returns:
            True if successful, False otherwise
            
        Implementation should:
        1. Download/load model weights
        2. Initialize tokenizer
        3. Set up any required configurations
        4. Set self._is_loaded = True on success
        """
        pass
    
    @abc.abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate text using the model.
        
        Args:
            request: Generation request with prompt and parameters
            
        Returns:
            Generation response with text and metadata
            
        Implementation should:
        1. Validate request parameters
        2. Tokenize input
        3. Generate response
        4. Return formatted response
        """
        pass
    
    @abc.abstractmethod
    def get_model_metadata(self) -> ModelMetadata:
        """
        Return comprehensive model metadata.
        
        Returns:
            ModelMetadata object with all model information
        """
        pass
    
    @abc.abstractmethod
    def get_model_capabilities(self) -> ModelCapabilities:
        """
        Return model capability information.
        
        Returns:
            ModelCapabilities object describing what the model can do
        """
        pass
    
    @abc.abstractmethod
    def get_cost_estimate(self) -> ModelCostEstimate:
        """
        Return cost estimation information.
        
        Returns:
            ModelCostEstimate object for budget analysis
        """
        pass
    
    @abc.abstractmethod
    def health_check(self) -> bool:
        """
        Perform a health check on the model.
        
        Returns:
            True if model is healthy and ready
        """
        pass
    
    @abc.abstractmethod
    def unload_model(self) -> bool:
        """
        Unload the model to free memory.
        
        Returns:
            True if successful
        """
        pass
    
    # Standard implementation methods (can be overridden if needed)
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    def get_load_time(self) -> Optional[float]:
        """Get model load time in seconds."""
        return self._load_time
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_inference_time = (self._total_inference_time / self._total_requests 
                             if self._total_requests > 0 else 0.0)
        avg_tokens_per_request = (self._total_tokens_generated / self._total_requests 
                                 if self._total_requests > 0 else 0.0)
        
        return {
            "total_requests": self._total_requests,
            "total_tokens_generated": self._total_tokens_generated,
            "total_inference_time": self._total_inference_time,
            "average_inference_time": avg_inference_time,
            "average_tokens_per_request": avg_tokens_per_request,
            "tokens_per_second": (self._total_tokens_generated / self._total_inference_time 
                                 if self._total_inference_time > 0 else 0.0)
        }
    
    def _record_request(self, inference_time: float, tokens_generated: int):
        """Record performance metrics for a request."""
        self._total_requests += 1
        self._total_inference_time += inference_time
        self._total_tokens_generated += tokens_generated
    
    def validate_config(self) -> bool:
        """
        Validate the model configuration.
        
        Returns:
            True if configuration is valid
        """
        required_fields = ["model_name"]
        for field in required_fields:
            if field not in self.config:
                logger.error(f"Missing required config field: {field}")
                return False
        return True
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get current memory usage (override for specific implementations).
        
        Returns:
            Dictionary with memory usage information
        """
        return {
            "gpu_memory_used": "N/A",
            "cpu_memory_used": "N/A",
            "model_size_gb": "N/A"
        }
    
    def supports_feature(self, feature: str) -> bool:
        """
        Check if model supports a specific feature.
        
        Args:
            feature: Feature name to check
            
        Returns:
            True if feature is supported
        """
        capabilities = self.get_model_capabilities()
        feature_map = {
            "function_calling": capabilities.supports_function_calling,
            "code_generation": capabilities.supports_code_generation,
            "reasoning": capabilities.supports_reasoning,
            "multimodal": capabilities.supports_multimodal,
            "streaming": capabilities.supports_streaming
        }
        return feature_map.get(feature, False)
    
    def get_benchmark_compatibility(self) -> Dict[str, bool]:
        """
        Return which benchmark categories this model can handle.
        
        Returns:
            Dictionary mapping benchmark categories to compatibility
        """
        capabilities = self.get_model_capabilities()
        
        return {
            "short_form_qa": True,  # All models should handle this
            "summarization": True,  # All models should handle this
            "reasoning": capabilities.supports_reasoning,
            "code_generation": capabilities.supports_code_generation,
            "function_calling": capabilities.supports_function_calling,
            "long_context": True,  # Will test up to model's context window
            "safety_refusal": True,  # All models should be tested for safety
            "multimodal": capabilities.supports_multimodal
        }
    
    def __str__(self) -> str:
        """String representation of the model wrapper."""
        metadata = self.get_model_metadata()
        return f"{metadata.name} ({metadata.parameters} parameters)"
    
    def __repr__(self) -> str:
        """Detailed representation of the model wrapper."""
        return f"ModelWrapper(name='{self.model_name}', loaded={self._is_loaded})"

# Utility functions for wrapper development

def validate_model_wrapper(wrapper_class) -> bool:
    """
    Validate that a wrapper class properly implements the template.
    
    Args:
        wrapper_class: Class to validate
        
    Returns:
        True if valid implementation
    """
    required_methods = [
        'load_model', 'generate', 'get_model_metadata',
        'get_model_capabilities', 'get_cost_estimate',
        'health_check', 'unload_model'
    ]
    
    for method in required_methods:
        if not hasattr(wrapper_class, method):
            logger.error(f"Wrapper missing required method: {method}")
            return False
        
        if getattr(wrapper_class, method) is getattr(ModelWrapperTemplate, method):
            logger.error(f"Wrapper has not implemented abstract method: {method}")
            return False
    
    return True

def create_wrapper_config_template(model_name: str) -> Dict[str, Any]:
    """
    Create a configuration template for a new model wrapper.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Configuration dictionary template
    """
    return {
        "model_name": model_name,
        "device": "auto",
        "max_memory": None,
        "temperature": 0.7,
        "max_tokens": 1000,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.0,
        "do_sample": True,
        "pad_token_id": None,
        "eos_token_id": None,
        "use_cache": True,
        "torch_dtype": "auto",
        "trust_remote_code": False,
        "revision": "main"
    }

# Export main classes and functions
__all__ = [
    'ModelWrapperTemplate',
    'ModelMetadata',
    'ModelCapabilities', 
    'ModelCostEstimate',
    'validate_model_wrapper',
    'create_wrapper_config_template'
]
