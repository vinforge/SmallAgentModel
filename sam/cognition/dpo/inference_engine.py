"""
Personalized Inference Engine

Handles runtime loading and switching of personalized LoRA adapters
for the SAM Personalized Tuner system.

Author: SAM Development Team
Version: 1.0.0
"""

import os
import gc
import time
import logging
import threading
from typing import Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path

# Import torch with fallback
try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)


@dataclass
class ModelCacheEntry:
    """Represents a cached model entry."""
    model: Any  # The loaded model
    tokenizer: Any  # The tokenizer
    last_used: datetime
    usage_count: int = 0
    memory_usage_mb: float = 0.0
    load_time_seconds: float = 0.0


@dataclass
class InferenceMetrics:
    """Metrics for inference performance tracking."""
    total_requests: int = 0
    personalized_requests: int = 0
    base_model_requests: int = 0
    fallback_requests: int = 0
    average_inference_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    model_switches: int = 0
    errors: int = 0


class PersonalizedInferenceEngine:
    """
    Core engine for loading and switching between LoRA adapters at runtime.
    
    Features:
    - Intelligent model caching with LRU eviction
    - Graceful fallback to base model
    - Performance monitoring and metrics
    - Thread-safe operations
    - Memory management
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the personalized inference engine.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(f"{__name__}.PersonalizedInferenceEngine")
        
        # Configuration
        self.config = config or {}
        self.max_cached_models = self.config.get('max_cached_models', 3)
        self.cache_ttl_hours = self.config.get('cache_ttl_hours', 24)
        self.enable_fallback = self.config.get('enable_fallback', True)
        self.fallback_timeout = self.config.get('fallback_timeout', 5.0)
        
        # Model cache and management
        self.model_cache: Dict[str, ModelCacheEntry] = {}
        self.base_model = None
        self.base_tokenizer = None
        self.cache_lock = threading.RLock()
        
        # Active model tracking
        self.active_models: Dict[str, str] = {}  # user_id -> model_id
        self.model_status: Dict[str, str] = {}   # model_id -> status
        
        # Performance metrics
        self.metrics = InferenceMetrics()
        
        # Model manager integration
        self.model_manager = None
        self._initialize_model_manager()
        
        self.logger.info("Personalized Inference Engine initialized")
    
    def _initialize_model_manager(self):
        """Initialize the DPO model manager."""
        try:
            from .model_manager import get_dpo_model_manager
            self.model_manager = get_dpo_model_manager()
        except ImportError as e:
            self.logger.warning(f"Could not initialize model manager: {e}")
    
    def load_base_model(self, model_name: Optional[str] = None) -> bool:
        """
        Load the base model for inference and fallback.
        
        Args:
            model_name: Base model name (uses config default if None)
            
        Returns:
            True if loaded successfully
        """
        try:
            # Import here to avoid dependency issues
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            if model_name is None:
                # Get from DPO config
                try:
                    from .dpo_config import get_dpo_config_manager
                    config_manager = get_dpo_config_manager()
                    model_name = config_manager.config.model.base_model_name
                except Exception:
                    model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Fallback
            
            self.logger.info(f"Loading base model: {model_name}")
            start_time = time.time()
            
            # Load tokenizer
            self.base_tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.base_tokenizer.pad_token is None:
                self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
            
            # Load model with optimizations
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                load_in_4bit=True  # Memory optimization
            )
            
            load_time = time.time() - start_time
            self.logger.info(f"Base model loaded successfully in {load_time:.2f}s")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load base model: {e}")
            return False
    
    def load_personalized_model(self, model_id: str, force_reload: bool = False) -> bool:
        """
        Load a personalized LoRA adapter.
        
        Args:
            model_id: Model identifier
            force_reload: Force reload even if cached
            
        Returns:
            True if loaded successfully
        """
        try:
            # Check cache first
            if not force_reload and model_id in self.model_cache:
                entry = self.model_cache[model_id]
                entry.last_used = datetime.now()
                entry.usage_count += 1
                self.metrics.cache_hits += 1
                self.logger.debug(f"Using cached model: {model_id}")
                return True
            
            # Get model info from model manager
            if not self.model_manager:
                self.logger.error("Model manager not available")
                return False
            
            model_info = self.model_manager.get_model_info(model_id)
            if not model_info:
                self.logger.error(f"Model not found: {model_id}")
                return False
            
            # Load the personalized model
            self.logger.info(f"Loading personalized model: {model_id}")
            start_time = time.time()
            
            personalized_model, tokenizer = self.model_manager.load_model_for_inference(model_id)
            
            load_time = time.time() - start_time
            
            # Cache the model
            with self.cache_lock:
                # Evict old models if cache is full
                self._evict_if_needed()
                
                # Add to cache
                self.model_cache[model_id] = ModelCacheEntry(
                    model=personalized_model,
                    tokenizer=tokenizer,
                    last_used=datetime.now(),
                    usage_count=1,
                    load_time_seconds=load_time
                )
            
            self.metrics.cache_misses += 1
            self.logger.info(f"Personalized model loaded and cached: {model_id} ({load_time:.2f}s)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load personalized model {model_id}: {e}")
            self.metrics.errors += 1
            return False
    
    def activate_personalized_model(self, user_id: str, model_id: str) -> bool:
        """
        Activate a personalized model for a user.
        
        Args:
            user_id: User identifier
            model_id: Model identifier to activate
            
        Returns:
            True if activated successfully
        """
        try:
            # Load the model if not already loaded
            if not self.load_personalized_model(model_id):
                return False
            
            # Update active model tracking
            old_model = self.active_models.get(user_id)
            self.active_models[user_id] = model_id
            self.model_status[model_id] = 'active'
            
            # Update model manager
            if self.model_manager:
                self.model_manager.activate_model(user_id, model_id)
            
            if old_model and old_model != model_id:
                self.metrics.model_switches += 1
                self.logger.info(f"Switched user {user_id} from {old_model} to {model_id}")
            else:
                self.logger.info(f"Activated personalized model for user {user_id}: {model_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to activate model {model_id} for user {user_id}: {e}")
            return False
    
    def deactivate_personalized_model(self, user_id: str) -> bool:
        """
        Deactivate personalized model for a user (fall back to base model).
        
        Args:
            user_id: User identifier
            
        Returns:
            True if deactivated successfully
        """
        try:
            if user_id in self.active_models:
                model_id = self.active_models[user_id]
                del self.active_models[user_id]
                
                # Update model status if no other users are using it
                if model_id not in self.active_models.values():
                    self.model_status[model_id] = 'inactive'
                
                # Update model manager
                if self.model_manager:
                    self.model_manager.deactivate_model(user_id)
                
                self.logger.info(f"Deactivated personalized model for user {user_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to deactivate model for user {user_id}: {e}")
            return False
    
    def generate_response(self, user_id: str, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Generate response using appropriate model (personalized or base).
        
        Args:
            user_id: User identifier
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Tuple of (response, metadata)
        """
        start_time = time.time()
        self.metrics.total_requests += 1
        
        metadata = {
            'user_id': user_id,
            'model_type': 'base',
            'model_id': None,
            'inference_time': 0.0,
            'fallback_used': False,
            'error': None
        }
        
        try:
            # Check if user has an active personalized model
            model_id = self.active_models.get(user_id)
            
            if model_id and model_id in self.model_cache:
                # Use personalized model
                response = self._generate_with_personalized_model(
                    model_id, prompt, metadata, **kwargs
                )
                self.metrics.personalized_requests += 1
            else:
                # Use base model
                response = self._generate_with_base_model(
                    prompt, metadata, **kwargs
                )
                self.metrics.base_model_requests += 1
            
            # Update metrics
            inference_time = time.time() - start_time
            metadata['inference_time'] = inference_time
            
            # Update average inference time
            total_requests = self.metrics.total_requests
            self.metrics.average_inference_time = (
                (self.metrics.average_inference_time * (total_requests - 1) + inference_time) / total_requests
            )
            
            return response, metadata
            
        except Exception as e:
            self.logger.error(f"Error in response generation: {e}")
            metadata['error'] = str(e)
            self.metrics.errors += 1
            
            # Try fallback
            if self.enable_fallback:
                try:
                    response = self._generate_fallback_response(prompt, metadata)
                    self.metrics.fallback_requests += 1
                    return response, metadata
                except Exception as fallback_error:
                    self.logger.error(f"Fallback generation failed: {fallback_error}")
            
            return "I apologize, but I'm experiencing technical difficulties. Please try again.", metadata
    
    def _generate_with_personalized_model(self, model_id: str, prompt: str,
                                        metadata: Dict[str, Any], **kwargs) -> str:
        """Generate response using personalized model."""
        try:
            if torch is None:
                raise RuntimeError("PyTorch not available for personalized model inference")

            cache_entry = self.model_cache[model_id]
            model = cache_entry.model
            tokenizer = cache_entry.tokenizer

            # Update cache entry
            cache_entry.last_used = datetime.now()
            cache_entry.usage_count += 1

            # Update metadata
            metadata['model_type'] = 'personalized'
            metadata['model_id'] = model_id

            # Generate response
            inputs = tokenizer(prompt, return_tensors="pt")

            generation_kwargs = {
                'max_new_tokens': kwargs.get('max_tokens', 500),
                'temperature': kwargs.get('temperature', 0.7),
                'do_sample': True,
                'pad_token_id': tokenizer.eos_token_id
            }

            with torch.no_grad():
                outputs = model.generate(**inputs, **generation_kwargs)
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from the response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating with personalized model {model_id}: {e}")
            raise
    
    def _generate_with_base_model(self, prompt: str, metadata: Dict[str, Any], **kwargs) -> str:
        """Generate response using base model."""
        try:
            if torch is None:
                raise RuntimeError("PyTorch not available for base model inference")

            if not self.base_model or not self.base_tokenizer:
                # Try to load base model
                if not self.load_base_model():
                    raise RuntimeError("Base model not available")

            # Generate response using base model
            inputs = self.base_tokenizer(prompt, return_tensors="pt")

            generation_kwargs = {
                'max_new_tokens': kwargs.get('max_tokens', 500),
                'temperature': kwargs.get('temperature', 0.7),
                'do_sample': True,
                'pad_token_id': self.base_tokenizer.eos_token_id
            }

            with torch.no_grad():
                outputs = self.base_model.generate(**inputs, **generation_kwargs)
            
            # Decode response
            response = self.base_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from the response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating with base model: {e}")
            raise
    
    def _generate_fallback_response(self, prompt: str, metadata: Dict[str, Any]) -> str:
        """Generate fallback response when models fail."""
        metadata['fallback_used'] = True
        
        # Try to use SAM's existing response generation as fallback
        try:
            from sam.core.sam_model_client import get_sam_model_client
            client = get_sam_model_client()
            return client.generate(prompt)
        except Exception:
            # Ultimate fallback
            return "I apologize, but I'm currently unable to process your request. Please try again later."
    
    def _evict_if_needed(self):
        """Evict old models from cache if needed."""
        if len(self.model_cache) < self.max_cached_models:
            return
        
        # Find least recently used model
        lru_model_id = min(
            self.model_cache.keys(),
            key=lambda mid: self.model_cache[mid].last_used
        )
        
        # Don't evict currently active models
        if lru_model_id in self.active_models.values():
            return
        
        # Evict the model
        self._evict_model(lru_model_id)
    
    def _evict_model(self, model_id: str):
        """Evict a specific model from cache."""
        try:
            if model_id in self.model_cache:
                entry = self.model_cache[model_id]
                
                # Clean up model memory
                del entry.model
                del entry.tokenizer
                del self.model_cache[model_id]
                
                # Force garbage collection
                gc.collect()
                
                self.logger.info(f"Evicted model from cache: {model_id}")
        except Exception as e:
            self.logger.error(f"Error evicting model {model_id}: {e}")
    
    def cleanup_cache(self):
        """Clean up expired cache entries."""
        try:
            current_time = datetime.now()
            ttl_delta = timedelta(hours=self.cache_ttl_hours)
            
            expired_models = [
                model_id for model_id, entry in self.model_cache.items()
                if current_time - entry.last_used > ttl_delta
                and model_id not in self.active_models.values()
            ]
            
            for model_id in expired_models:
                self._evict_model(model_id)
            
            if expired_models:
                self.logger.info(f"Cleaned up {len(expired_models)} expired models")
                
        except Exception as e:
            self.logger.error(f"Error during cache cleanup: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current engine status and metrics."""
        return {
            'active_models': dict(self.active_models),
            'cached_models': list(self.model_cache.keys()),
            'cache_size': len(self.model_cache),
            'max_cache_size': self.max_cached_models,
            'base_model_loaded': self.base_model is not None,
            'metrics': {
                'total_requests': self.metrics.total_requests,
                'personalized_requests': self.metrics.personalized_requests,
                'base_model_requests': self.metrics.base_model_requests,
                'fallback_requests': self.metrics.fallback_requests,
                'average_inference_time': self.metrics.average_inference_time,
                'cache_hits': self.metrics.cache_hits,
                'cache_misses': self.metrics.cache_misses,
                'model_switches': self.metrics.model_switches,
                'errors': self.metrics.errors,
                'cache_hit_rate': self.metrics.cache_hits / max(1, self.metrics.cache_hits + self.metrics.cache_misses)
            }
        }


# Global inference engine instance
_inference_engine = None

def get_personalized_inference_engine(config: Optional[Dict[str, Any]] = None) -> PersonalizedInferenceEngine:
    """Get or create a global personalized inference engine instance."""
    global _inference_engine
    
    if _inference_engine is None:
        _inference_engine = PersonalizedInferenceEngine(config)
    
    return _inference_engine
