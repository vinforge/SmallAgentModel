"""
SAM Model Foundry: Llama-3.1-8B Wrapper
=======================================

Concrete implementation of ModelWrapperTemplate for Meta's Llama-3.1-8B model.
This serves as the proof-of-concept for the SAM Model Foundry framework.

Model: meta-llama/Meta-Llama-3.1-8B-Instruct
Context Window: 128K tokens
Parameters: 8B
License: Llama 3.1 Community License

Author: SAM Development Team
Version: 1.0.0
"""

import time
import logging
from typing import Dict, Any, Optional, List
import torch

from .template import (
    ModelWrapperTemplate, ModelMetadata, ModelCapabilities, 
    ModelCostEstimate, create_wrapper_config_template
)
from sam.core.model_interface import GenerationRequest, GenerationResponse

logger = logging.getLogger(__name__)

class Llama31Wrapper(ModelWrapperTemplate):
    """
    Llama-3.1-8B model wrapper for SAM evaluation framework.
    
    This wrapper integrates Meta's Llama-3.1-8B-Instruct model using
    the transformers library with optimizations for SAM's use cases.
    """
    
    MODEL_NAME = "llama31-8b"
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Llama-3.1-8B wrapper."""
        super().__init__(config)
        
        # Llama-specific configuration
        self.model_id = config.get("model_id", "meta-llama/Meta-Llama-3.1-8B-Instruct")
        self.use_flash_attention = config.get("use_flash_attention", True)
        self.load_in_4bit = config.get("load_in_4bit", False)
        self.load_in_8bit = config.get("load_in_8bit", False)
        
        # Performance settings
        self.batch_size = config.get("batch_size", 1)
        self.max_memory_per_gpu = config.get("max_memory_per_gpu", "auto")
        
        # Chat template settings
        self.system_prompt = config.get("system_prompt", 
            "You are a helpful, harmless, and honest AI assistant.")
        
    def initialize(self) -> bool:
        """Initialize the model (alias for load_model)."""
        return self.load_model()

    def load_model(self) -> bool:
        """Load Llama-3.1-8B model and tokenizer."""
        try:
            start_time = time.time()
            logger.info(f"🔄 Loading Llama-3.1-8B model: {self.model_id}")
            
            # Import required libraries
            try:
                from transformers import (
                    AutoTokenizer, AutoModelForCausalLM, 
                    BitsAndBytesConfig, pipeline
                )
                import torch
            except ImportError as e:
                logger.error(f"❌ Required libraries not installed: {e}")
                logger.error("Install with: pip install transformers torch accelerate bitsandbytes")
                return False
            
            # Configure quantization if requested
            quantization_config = None
            if self.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                logger.info("🔧 Using 4-bit quantization")
            elif self.load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                logger.info("🔧 Using 8-bit quantization")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=self.config.get("trust_remote_code", False)
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            model_kwargs = {
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "device_map": self.device if self.device != "auto" else "auto",
                "trust_remote_code": self.config.get("trust_remote_code", False)
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            
            if self.max_memory:
                model_kwargs["max_memory"] = self.max_memory
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **model_kwargs
            )
            
            # Create text generation pipeline for easier use
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            
            self._load_time = time.time() - start_time
            self._is_loaded = True
            
            logger.info(f"✅ Llama-3.1-8B loaded successfully in {self._load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load Llama-3.1-8B: {e}")
            return False
    
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using Llama-3.1-8B."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        try:
            # Format prompt with chat template
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": request.prompt}
            ]
            
            # Apply chat template
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Generation parameters
            generation_kwargs = {
                "max_new_tokens": request.max_tokens or 1000,
                "temperature": request.temperature or 0.7,
                "top_p": request.top_p or 0.9,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "return_full_text": False
            }
            
            # Generate response
            outputs = self.pipeline(
                formatted_prompt,
                **generation_kwargs
            )
            
            # Extract generated text
            generated_text = outputs[0]["generated_text"]
            
            # Calculate metrics
            inference_time = time.time() - start_time
            input_tokens = len(self.tokenizer.encode(formatted_prompt))
            output_tokens = len(self.tokenizer.encode(generated_text))
            
            # Record performance
            self._record_request(inference_time, output_tokens)
            
            return GenerationResponse(
                text=generated_text,
                finish_reason="stop",
                usage={
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                },
                model=self.model_id,
                inference_time=inference_time
            )
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return GenerationResponse(
                text="",
                finish_reason="error",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                model=self.model_id,
                error=str(e)
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information (ModelInterface compatibility)."""
        metadata = self.get_model_metadata()
        capabilities = self.get_model_capabilities()

        return {
            "name": metadata.name,
            "version": metadata.version,
            "parameters": metadata.parameters,
            "context_window": metadata.context_window,
            "license": metadata.license,
            "provider": metadata.provider,
            "model_family": metadata.model_family,
            "architecture": metadata.architecture,
            "supports_function_calling": capabilities.supports_function_calling,
            "supports_code_generation": capabilities.supports_code_generation,
            "max_output_tokens": capabilities.max_output_tokens
        }

    def get_model_metadata(self) -> ModelMetadata:
        """Return Llama-3.1-8B metadata."""
        return ModelMetadata(
            name="Llama-3.1-8B-Instruct",
            version="3.1",
            parameters="8B",
            context_window=128000,
            license="Llama 3.1 Community License",
            provider="Meta",
            model_family="Llama",
            architecture="Transformer",
            quantization="4bit" if self.load_in_4bit else ("8bit" if self.load_in_8bit else "fp16"),
            fine_tuned=True,
            release_date="2024-07-23",
            paper_url="https://arxiv.org/abs/2407.21783",
            model_url="https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct"
        )
    
    def get_model_capabilities(self) -> ModelCapabilities:
        """Return Llama-3.1-8B capabilities."""
        return ModelCapabilities(
            supports_function_calling=True,  # Llama 3.1 supports tool use
            supports_code_generation=True,
            supports_reasoning=True,
            supports_multimodal=False,
            supports_streaming=True,
            max_output_tokens=4096,
            languages_supported=["en", "es", "fr", "de", "it", "pt", "hi", "th", "zh"]
        )
    
    def get_cost_estimate(self) -> ModelCostEstimate:
        """Return cost estimation for Llama-3.1-8B."""
        # Estimates based on typical cloud hosting costs
        return ModelCostEstimate(
            cost_per_input_token=0.0001,   # $0.10 per 1M input tokens
            cost_per_output_token=0.0002,  # $0.20 per 1M output tokens
            setup_cost=0.0,                # Open source model
            monthly_hosting_cost=200.0,    # Estimated GPU hosting cost
            currency="USD"
        )
    
    def health_check(self) -> bool:
        """Perform health check on Llama-3.1-8B."""
        if not self._is_loaded:
            return False
        
        try:
            # Simple generation test
            test_request = GenerationRequest(
                prompt="Hello",
                max_tokens=5,
                temperature=0.1
            )
            
            response = self.generate(test_request)
            return response.finish_reason != "error" and len(response.text) > 0
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def unload_model(self) -> bool:
        """Unload Llama-3.1-8B to free memory."""
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
            if hasattr(self, 'pipeline') and self.pipeline is not None:
                del self.pipeline
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self._is_loaded = False
            logger.info("✅ Llama-3.1-8B unloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to unload model: {e}")
            return False
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage."""
        memory_info = {
            "model_size_gb": "~16GB (fp16)" if not (self.load_in_4bit or self.load_in_8bit) else "~4-8GB (quantized)",
            "cpu_memory_used": "N/A",
            "gpu_memory_used": "N/A"
        }
        
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
                gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3
                memory_info.update({
                    "gpu_memory_used": f"{gpu_memory:.2f} GB",
                    "gpu_memory_cached": f"{gpu_memory_cached:.2f} GB"
                })
            except Exception:
                pass
        
        return memory_info

# Factory function for easy creation
def create_llama31_wrapper(config: Optional[Dict[str, Any]] = None) -> Llama31Wrapper:
    """
    Create a Llama-3.1-8B wrapper with default configuration.
    
    Args:
        config: Optional configuration overrides
        
    Returns:
        Initialized Llama31Wrapper
    """
    default_config = create_wrapper_config_template("llama31-8b")
    default_config.update({
        "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "use_flash_attention": True,
        "load_in_4bit": True,  # Enable 4-bit quantization by default
        "system_prompt": "You are a helpful, harmless, and honest AI assistant."
    })
    
    if config:
        default_config.update(config)
    
    return Llama31Wrapper(default_config)

# Export main class
__all__ = ['Llama31Wrapper', 'create_llama31_wrapper']
