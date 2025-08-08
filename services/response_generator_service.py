#!/usr/bin/env python3
"""
Response Generator Service
Handles response generation using various backends (Ollama, OpenAI, etc.).
"""

import logging
import requests
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for response generation."""
    model: str = "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M"
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 120
    stream: bool = False

class ResponseGeneratorService:
    """
    Service for generating responses using various backends.
    Handles fallbacks, error recovery, and response formatting.
    """
    
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/generate"
        self.default_config = GenerationConfig()
    
    def generate_response(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """
        Generate a response using the configured backend.
        
        Args:
            prompt: The full prompt to send to the model
            config: Optional generation configuration
            
        Returns:
            Generated response string
        """
        if config is None:
            config = self.default_config
        
        logger.info(f"🤖 Generating response with {config.model}")
        logger.debug(f"🤖 Prompt length: {len(prompt)} characters")
        
        try:
            # Try Ollama first
            response = self._generate_with_ollama(prompt, config)
            if response:
                logger.info(f"✅ Response generated successfully ({len(response)} chars)")
                return response
            
            # Fallback to simple response
            logger.warning("⚠️ Ollama generation failed, using fallback")
            return self._generate_fallback_response(prompt)
            
        except Exception as e:
            logger.error(f"❌ Response generation failed: {e}")
            return self._generate_error_response(prompt, e)
    
    def _generate_with_ollama(self, prompt: str, config: GenerationConfig) -> Optional[str]:
        """Generate response using Ollama API."""
        try:
            payload = {
                "model": config.model,
                "prompt": prompt,
                "stream": config.stream,
                "options": {
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens
                }
            }
            
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=config.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                logger.warning(f"⚠️ Ollama returned status {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            logger.warning("⚠️ Ollama request timed out")
            return None
        except requests.exceptions.ConnectionError:
            logger.warning("⚠️ Could not connect to Ollama")
            return None
        except Exception as e:
            logger.warning(f"⚠️ Ollama generation error: {e}")
            return None
    
    def _generate_fallback_response(self, prompt: str) -> str:
        """Generate a fallback response when primary generation fails."""
        # Extract the query from the prompt
        lines = prompt.split('\n')
        query_line = None
        
        for line in lines:
            if line.startswith('Question:'):
                query_line = line.replace('Question:', '').strip()
                break
        
        if query_line:
            return f"I understand you're asking about: {query_line}"
        else:
            return "I understand your question and I'm processing it. Please try again in a moment."
    
    def _generate_error_response(self, prompt: str, error: Exception) -> str:
        """Generate an error response with helpful information."""
        # Extract the query from the prompt
        lines = prompt.split('\n')
        query_line = None
        
        for line in lines:
            if line.startswith('Question:'):
                query_line = line.replace('Question:', '').strip()
                break
        
        # Provide specific error messages based on error type
        if "timeout" in str(error).lower() or "read timed out" in str(error).lower():
            if query_line:
                return f"I apologize, but I'm experiencing slower than usual response times. Your question about '{query_line}' is being processed, but it may take longer than expected. Please try again in a moment, or consider asking a simpler question."
            else:
                return "I apologize for the delay. I'm experiencing slower response times. Please try again in a moment."
        
        elif "connection" in str(error).lower():
            return "I'm having trouble connecting to my response generation system. Please try again in a moment."
        
        else:
            if query_line:
                return f"I understand you're asking about: {query_line}"
            else:
                return "I understand your question. Please try rephrasing it or try again in a moment."
    
    def test_connection(self) -> bool:
        """Test if the response generation backend is available."""
        try:
            response = requests.get(
                "http://localhost:11434/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    def get_available_models(self) -> list:
        """Get list of available models from Ollama."""
        try:
            response = requests.get(
                "http://localhost:11434/api/tags",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            else:
                return []
        except:
            return []

# Global instance for easy access
_response_generator_service = None

def get_response_generator_service() -> ResponseGeneratorService:
    """Get or create the global response generator service instance."""
    global _response_generator_service
    if _response_generator_service is None:
        _response_generator_service = ResponseGeneratorService()
    return _response_generator_service
