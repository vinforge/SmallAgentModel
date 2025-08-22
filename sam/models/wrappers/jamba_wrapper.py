"""
SAM Model Foundry: Jamba Hybrid Model Wrapper
============================================

This wrapper provides integration for AI21 Labs' Jamba hybrid model,
which combines Transformer and Mamba (SSM) layers for efficient
long-context processing.

Jamba Features:
- Hybrid architecture: Transformer + State Space Model (Mamba)
- 256K context length capability
- Efficient inference for long sequences
- 7B parameters with competitive performance

Author: SAM Development Team
Version: 1.0.0
"""

import time
import logging
from typing import Dict, Any, Optional, List
import torch
from pathlib import Path

from sam.core.model_interface import ModelInterface, GenerationRequest, GenerationResponse
from sam.core.model_interface import ModelConfig, ModelType, ModelStatus

logger = logging.getLogger(__name__)


class JambaWrapper(ModelInterface):
    """
    Jamba hybrid model wrapper for SAM.
    
    Integrates AI21 Labs' Jamba model which combines Transformer and Mamba
    architectures for efficient long-context processing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Jamba wrapper.

        Args:
            config: Model configuration dictionary
        """
        # Create a ModelConfig for the parent class
        model_config = ModelConfig(
            model_type=ModelType.TRANSFORMER,  # Hybrid, but closest to transformer
            model_name=config.get("model_name", "ai21labs/Jamba-v0.1"),
            api_url=config.get("api_url", ""),
            max_context_length=config.get("max_context_length", 256000),
            timeout_seconds=config.get("timeout_seconds", 180),
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
        self.model_id = config.get("model_name", "ai21labs/Jamba-v0.1")
        self.device = config.get("device", "auto")
        self.max_memory = config.get("max_memory", None)
        self.trust_remote_code = config.get("trust_remote_code", True)
        
    def initialize(self) -> bool:
        """Initialize the model (alias for load_model)."""
        return self.load_model()

    def load_model(self) -> bool:
        """Load Jamba hybrid model and tokenizer."""
        try:
            start_time = time.time()
            logger.info(f"🔄 Loading Jamba hybrid model: {self.model_id}")
            
            # Import required libraries
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                import torch
            except ImportError as e:
                logger.error(f"❌ Required libraries not installed: {e}")
                logger.error("Install with: pip install transformers torch accelerate")
                return False

            # Check if CUDA is available
            if torch.cuda.is_available():
                logger.info(f"🚀 CUDA detected: {torch.cuda.get_device_name()}")
                torch_dtype = torch.float16
                attn_implementation = "flash_attention_2"
            else:
                logger.info("💻 Using CPU inference")
                torch_dtype = torch.float32
                attn_implementation = "eager"

            # Load tokenizer
            logger.info("📝 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=self.trust_remote_code
            )
            
            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("🔧 Added pad token")
            
            # Load model with appropriate settings for Jamba
            logger.info("🧠 Loading Jamba model...")
            model_kwargs = {
                "torch_dtype": torch_dtype,
                "device_map": self.device,
                "trust_remote_code": self.trust_remote_code,
                "attn_implementation": attn_implementation
            }
            
            if self.max_memory:
                model_kwargs["max_memory"] = self.max_memory
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **model_kwargs
            )
            
            self._load_time = time.time() - start_time
            self._is_loaded = True
            self.status = ModelStatus.READY
            
            logger.info(f"✅ Jamba model loaded successfully in {self._load_time:.2f}s")
            logger.info(f"📊 Model info: {self.model.config.num_parameters:,} parameters")
            logger.info(f"🎯 Context length: {self.config.max_context_length:,} tokens")
            
            return True
            
        except Exception as e:
            self.status = ModelStatus.ERROR
            logger.error(f"❌ Failed to load Jamba model: {e}")
            return False

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using the Jamba model."""
        if not self._is_loaded or self.model is None:
            return GenerationResponse(
                text="",
                model_type=ModelType.TRANSFORMER,
                inference_time=0.0,
                context_length=0,
                success=False,
                error_message="Jamba model not loaded"
            )
        
        try:
            start_time = time.time()
            self._total_requests += 1
            
            # Set generation parameters
            generation_kwargs = {
                "max_new_tokens": min(request.max_tokens or 1000, 2048),
                "temperature": request.temperature or 0.7,
                "do_sample": (request.temperature or 0.7) > 0,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            # Add top_p if specified
            if hasattr(request, 'top_p') and request.top_p:
                generation_kwargs["top_p"] = request.top_p
            
            # Tokenize input
            max_context = self.config.get("max_context_length", 256000)
            inputs = self.tokenizer(
                request.prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_context - generation_kwargs["max_new_tokens"]
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            input_length = inputs["input_ids"].shape[1]
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **generation_kwargs
                )
            
            # Decode only the new tokens
            generated_tokens = outputs[0][input_length:]
            response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            inference_time = time.time() - start_time
            self._total_inference_time += inference_time
            self._total_tokens_generated += len(generated_tokens)
            
            # Update metrics
            self.metrics.total_requests = self._total_requests
            self.metrics.successful_requests += 1
            self.metrics.average_inference_time = self._total_inference_time / self._total_requests
            self.metrics.last_request_time = time.time()
            
            return GenerationResponse(
                text=response_text.strip(),
                model_type=ModelType.TRANSFORMER,
                inference_time=inference_time,
                context_length=input_length,
                success=True
            )
            
        except Exception as e:
            logger.error(f"❌ Jamba generation error: {e}")
            return GenerationResponse(
                text="",
                model_type=ModelType.TRANSFORMER,
                inference_time=0.0,
                context_length=0,
                success=False,
                error_message=str(e)
            )

    def health_check(self) -> bool:
        """Check if the Jamba model is healthy and responsive."""
        if not self._is_loaded or self.model is None:
            return False
        
        try:
            # Simple test generation
            test_request = GenerationRequest(
                prompt="Hello",
                max_tokens=5,
                temperature=0.1
            )
            response = self.generate(test_request)
            return response.success
        except Exception:
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed Jamba model information."""
        info = {
            "model_id": self.model_id,
            "model_type": "Hybrid (Transformer+SSM)",
            "architecture": "Jamba",
            "context_length": self.config.get("max_context_length", 256000),
            "is_loaded": self._is_loaded,
            "load_time": self._load_time,
            "total_requests": self._total_requests,
            "total_tokens_generated": self._total_tokens_generated,
            "average_inference_time": self._total_inference_time / max(self._total_requests, 1),
            "device": str(self.device),
            "trust_remote_code": self.trust_remote_code
        }
        
        if self.model and hasattr(self.model, 'config'):
            info.update({
                "num_parameters": getattr(self.model.config, 'num_parameters', 'Unknown'),
                "vocab_size": getattr(self.model.config, 'vocab_size', 'Unknown'),
                "hidden_size": getattr(self.model.config, 'hidden_size', 'Unknown')
            })
        
        return info

    def unload_model(self) -> bool:
        """Unload the Jamba model from memory."""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self._is_loaded = False
            self.status = ModelStatus.LOADING
            logger.info(f"✅ Jamba model unloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to unload Jamba model: {e}")
            return False
