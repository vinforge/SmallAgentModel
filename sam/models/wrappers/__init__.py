"""
SAM Model Foundry: Model Wrappers Package
=========================================

This package contains model wrappers for integrating different AI models
into SAM's evaluation and comparison framework.

Author: SAM Development Team
Version: 1.0.0
"""

from .template import (
    ModelWrapperTemplate,
    ModelMetadata,
    ModelCapabilities,
    ModelCostEstimate,
    validate_model_wrapper,
    create_wrapper_config_template
)

# Dynamic wrapper discovery
import importlib
import pkgutil
from pathlib import Path
from typing import Dict, List, Type
import logging

logger = logging.getLogger(__name__)

def discover_available_wrappers() -> Dict[str, Type[ModelWrapperTemplate]]:
    """
    Dynamically discover all available model wrappers.
    
    Returns:
        Dictionary mapping model names to wrapper classes
    """
    wrappers = {}
    
    # Get the current package path
    package_path = Path(__file__).parent
    
    # Iterate through all Python files in the wrappers directory
    for module_info in pkgutil.iter_modules([str(package_path)]):
        if module_info.name == 'template':
            continue  # Skip the template
        
        try:
            # Import the module
            module = importlib.import_module(f"sam.models.wrappers.{module_info.name}")
            
            # Look for classes that inherit from ModelWrapperTemplate
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                
                if (isinstance(attr, type) and 
                    issubclass(attr, ModelWrapperTemplate) and 
                    attr != ModelWrapperTemplate):
                    
                    # Validate the wrapper
                    if validate_model_wrapper(attr):
                        # Use the class name or a custom model name
                        model_name = getattr(attr, 'MODEL_NAME', attr_name.lower().replace('wrapper', ''))
                        wrappers[model_name] = attr
                        logger.info(f"✅ Discovered model wrapper: {model_name}")
                    else:
                        logger.warning(f"⚠️ Invalid wrapper implementation: {attr_name}")
                        
        except Exception as e:
            logger.warning(f"⚠️ Failed to load wrapper module {module_info.name}: {e}")
    
    return wrappers

def get_available_models() -> List[str]:
    """
    Get list of available model names.
    
    Returns:
        List of model names that can be used
    """
    return list(discover_available_wrappers().keys())

def create_model_wrapper(model_name: str, config: Dict) -> ModelWrapperTemplate:
    """
    Create a model wrapper instance by name.
    
    Args:
        model_name: Name of the model wrapper
        config: Configuration dictionary
        
    Returns:
        Initialized model wrapper instance
        
    Raises:
        ValueError: If model wrapper not found
    """
    wrappers = discover_available_wrappers()
    
    if model_name not in wrappers:
        available = ", ".join(wrappers.keys())
        raise ValueError(f"Model wrapper '{model_name}' not found. Available: {available}")
    
    wrapper_class = wrappers[model_name]
    return wrapper_class(config)

def list_wrapper_capabilities() -> Dict[str, Dict]:
    """
    Get capabilities of all available wrappers.
    
    Returns:
        Dictionary mapping model names to their capabilities
    """
    wrappers = discover_available_wrappers()
    capabilities = {}
    
    for model_name, wrapper_class in wrappers.items():
        try:
            # Create a temporary instance to get capabilities
            temp_config = create_wrapper_config_template(model_name)
            temp_wrapper = wrapper_class(temp_config)
            
            capabilities[model_name] = {
                "metadata": temp_wrapper.get_model_metadata(),
                "capabilities": temp_wrapper.get_model_capabilities(),
                "cost_estimate": temp_wrapper.get_cost_estimate()
            }
        except Exception as e:
            logger.warning(f"Failed to get capabilities for {model_name}: {e}")
            capabilities[model_name] = {"error": str(e)}
    
    return capabilities

__all__ = [
    # Template classes
    'ModelWrapperTemplate',
    'ModelMetadata',
    'ModelCapabilities',
    'ModelCostEstimate',
    
    # Utility functions
    'validate_model_wrapper',
    'create_wrapper_config_template',
    
    # Discovery functions
    'discover_available_wrappers',
    'get_available_models',
    'create_model_wrapper',
    'list_wrapper_capabilities'
]

__version__ = "1.0.0"
