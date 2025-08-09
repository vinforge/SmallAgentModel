"""
SAM Integration for Personalized Inference

Integrates the personalized inference engine with SAM's existing
response generation pipeline.

Author: SAM Development Team
Version: 1.0.0
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PersonalizedResponse:
    """Response with personalization metadata."""
    content: str
    metadata: Dict[str, Any]
    is_personalized: bool
    model_id: Optional[str] = None
    inference_time: float = 0.0
    fallback_used: bool = False


class PersonalizedSAMClient:
    """
    SAM client wrapper that integrates personalized inference.
    
    This class wraps SAM's existing model client and adds personalization
    capabilities while maintaining backward compatibility.
    """
    
    def __init__(self, enable_personalization: bool = True):
        """
        Initialize the personalized SAM client.
        
        Args:
            enable_personalization: Whether to enable personalization features
        """
        self.logger = logging.getLogger(f"{__name__}.PersonalizedSAMClient")
        self.enable_personalization = enable_personalization
        
        # Initialize components
        self.inference_engine = None
        self.base_client = None
        
        # Initialize inference engine if personalization is enabled
        if self.enable_personalization:
            self._initialize_inference_engine()
        
        # Initialize base SAM client
        self._initialize_base_client()
        
        self.logger.info(f"Personalized SAM client initialized (personalization: {enable_personalization})")
    
    def _initialize_inference_engine(self):
        """Initialize the personalized inference engine."""
        try:
            from .inference_engine import get_personalized_inference_engine
            self.inference_engine = get_personalized_inference_engine()
            self.logger.info("Personalized inference engine initialized")
        except Exception as e:
            self.logger.warning(f"Could not initialize inference engine: {e}")
            self.enable_personalization = False
    
    def _initialize_base_client(self):
        """Initialize the base SAM client."""
        try:
            from sam.core.sam_model_client import get_sam_model_client
            self.base_client = get_sam_model_client()
            self.logger.info("Base SAM client initialized")
        except Exception as e:
            self.logger.error(f"Could not initialize base SAM client: {e}")
    
    def generate_response(self, prompt: str, user_id: Optional[str] = None, 
                         context: Optional[Dict[str, Any]] = None, **kwargs) -> PersonalizedResponse:
        """
        Generate response with personalization support.
        
        Args:
            prompt: Input prompt
            user_id: User identifier for personalization
            context: Additional context information
            **kwargs: Additional generation parameters
            
        Returns:
            PersonalizedResponse with content and metadata
        """
        start_time = time.time()
        
        # Default response metadata
        metadata = {
            'user_id': user_id,
            'personalization_enabled': self.enable_personalization,
            'context': context or {},
            'generation_params': kwargs
        }
        
        try:
            # Try personalized generation if enabled and user_id provided
            if (self.enable_personalization and 
                user_id and 
                self.inference_engine and 
                self._should_use_personalization(user_id, context)):
                
                return self._generate_personalized_response(prompt, user_id, metadata, **kwargs)
            
            # Fall back to base SAM client
            return self._generate_base_response(prompt, metadata, **kwargs)
            
        except Exception as e:
            self.logger.error(f"Error in response generation: {e}")
            
            # Ultimate fallback
            return PersonalizedResponse(
                content="I apologize, but I'm experiencing technical difficulties. Please try again.",
                metadata={**metadata, 'error': str(e), 'fallback_used': True},
                is_personalized=False,
                inference_time=time.time() - start_time,
                fallback_used=True
            )
    
    def _should_use_personalization(self, user_id: str, context: Optional[Dict[str, Any]]) -> bool:
        """
        Determine if personalization should be used for this request.
        
        Args:
            user_id: User identifier
            context: Request context
            
        Returns:
            True if personalization should be used
        """
        try:
            # Check if user has an active personalized model
            if not self.inference_engine:
                return False
            
            status = self.inference_engine.get_status()
            active_models = status.get('active_models', {})
            
            return user_id in active_models
            
        except Exception as e:
            self.logger.debug(f"Error checking personalization eligibility: {e}")
            return False
    
    def _generate_personalized_response(self, prompt: str, user_id: str, 
                                      metadata: Dict[str, Any], **kwargs) -> PersonalizedResponse:
        """Generate response using personalized model."""
        try:
            response_text, inference_metadata = self.inference_engine.generate_response(
                user_id=user_id,
                prompt=prompt,
                **kwargs
            )
            
            # Merge metadata
            metadata.update(inference_metadata)
            
            return PersonalizedResponse(
                content=response_text,
                metadata=metadata,
                is_personalized=inference_metadata.get('model_type') == 'personalized',
                model_id=inference_metadata.get('model_id'),
                inference_time=inference_metadata.get('inference_time', 0.0),
                fallback_used=inference_metadata.get('fallback_used', False)
            )
            
        except Exception as e:
            self.logger.error(f"Personalized generation failed: {e}")
            
            # Fall back to base response
            metadata['personalization_error'] = str(e)
            return self._generate_base_response(prompt, metadata, **kwargs)
    
    def _generate_base_response(self, prompt: str, metadata: Dict[str, Any], **kwargs) -> PersonalizedResponse:
        """Generate response using base SAM client."""
        try:
            start_time = time.time()
            
            if self.base_client:
                # Use SAM's unified model client
                response_text = self.base_client.generate(
                    prompt=prompt,
                    temperature=kwargs.get('temperature', 0.7),
                    max_tokens=kwargs.get('max_tokens', 500),
                    stop_sequences=kwargs.get('stop_sequences')
                )
            else:
                # Ultimate fallback
                response_text = "I'm currently unable to process your request. Please try again later."
            
            inference_time = time.time() - start_time
            
            return PersonalizedResponse(
                content=response_text,
                metadata={**metadata, 'inference_time': inference_time},
                is_personalized=False,
                inference_time=inference_time
            )
            
        except Exception as e:
            self.logger.error(f"Base generation failed: {e}")
            
            return PersonalizedResponse(
                content="I apologize, but I'm experiencing technical difficulties. Please try again.",
                metadata={**metadata, 'base_generation_error': str(e), 'fallback_used': True},
                is_personalized=False,
                fallback_used=True
            )
    
    def activate_personalized_model(self, user_id: str, model_id: str) -> bool:
        """
        Activate a personalized model for a user.
        
        Args:
            user_id: User identifier
            model_id: Model identifier
            
        Returns:
            True if activated successfully
        """
        if not self.enable_personalization or not self.inference_engine:
            self.logger.warning("Personalization not enabled")
            return False
        
        try:
            return self.inference_engine.activate_personalized_model(user_id, model_id)
        except Exception as e:
            self.logger.error(f"Error activating personalized model: {e}")
            return False
    
    def deactivate_personalized_model(self, user_id: str) -> bool:
        """
        Deactivate personalized model for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if deactivated successfully
        """
        if not self.enable_personalization or not self.inference_engine:
            return True  # Already deactivated
        
        try:
            return self.inference_engine.deactivate_personalized_model(user_id)
        except Exception as e:
            self.logger.error(f"Error deactivating personalized model: {e}")
            return False
    
    def get_personalization_status(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get personalization status.
        
        Args:
            user_id: User identifier (optional)
            
        Returns:
            Status dictionary
        """
        if not self.enable_personalization or not self.inference_engine:
            return {
                'personalization_enabled': False,
                'reason': 'Personalization not available'
            }
        
        try:
            status = self.inference_engine.get_status()
            
            if user_id:
                # User-specific status
                active_models = status.get('active_models', {})
                user_model = active_models.get(user_id)
                
                return {
                    'personalization_enabled': True,
                    'user_id': user_id,
                    'has_active_model': user_model is not None,
                    'active_model_id': user_model,
                    'engine_status': status
                }
            else:
                # Global status
                return {
                    'personalization_enabled': True,
                    'engine_status': status
                }
                
        except Exception as e:
            self.logger.error(f"Error getting personalization status: {e}")
            return {
                'personalization_enabled': False,
                'error': str(e)
            }
    
    def cleanup_cache(self):
        """Clean up model cache."""
        if self.inference_engine:
            try:
                self.inference_engine.cleanup_cache()
            except Exception as e:
                self.logger.error(f"Error cleaning up cache: {e}")


# Global personalized SAM client instance
_personalized_client = None

def get_personalized_sam_client(enable_personalization: bool = True) -> PersonalizedSAMClient:
    """Get or create a global personalized SAM client instance."""
    global _personalized_client
    
    if _personalized_client is None:
        _personalized_client = PersonalizedSAMClient(enable_personalization)
    
    return _personalized_client


# Convenience functions for backward compatibility
def generate_personalized_response(prompt: str, user_id: Optional[str] = None, 
                                 context: Optional[Dict[str, Any]] = None, **kwargs) -> PersonalizedResponse:
    """
    Generate response with personalization support.
    
    Args:
        prompt: Input prompt
        user_id: User identifier for personalization
        context: Additional context information
        **kwargs: Additional generation parameters
        
    Returns:
        PersonalizedResponse with content and metadata
    """
    client = get_personalized_sam_client()
    return client.generate_response(prompt, user_id, context, **kwargs)


def activate_user_model(user_id: str, model_id: str) -> bool:
    """
    Activate a personalized model for a user.
    
    Args:
        user_id: User identifier
        model_id: Model identifier
        
    Returns:
        True if activated successfully
    """
    client = get_personalized_sam_client()
    return client.activate_personalized_model(user_id, model_id)


def deactivate_user_model(user_id: str) -> bool:
    """
    Deactivate personalized model for a user.
    
    Args:
        user_id: User identifier
        
    Returns:
        True if deactivated successfully
    """
    client = get_personalized_sam_client()
    return client.deactivate_personalized_model(user_id)


def get_user_personalization_status(user_id: str) -> Dict[str, Any]:
    """
    Get personalization status for a user.
    
    Args:
        user_id: User identifier
        
    Returns:
        Status dictionary
    """
    client = get_personalized_sam_client()
    return client.get_personalization_status(user_id)
