"""
Base Prompt Template System

Provides the foundation for structured, maintainable prompts
inspired by LongBioBench's organization.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass

class PromptType(Enum):
    """Types of prompts in the system."""
    CITATION = "citation"
    REASONING = "reasoning"
    IDK_REFUSAL = "idk_refusal"
    SUMMARY = "summary"
    SELF_DISCOVER = "self_discover"
    CRITIC = "critic"

@dataclass
class PromptTemplate:
    """Template for a structured prompt."""
    system_prompt: str
    user_prompt_template: str
    prompt_type: PromptType
    description: str
    variables: Dict[str, str]  # Variable name -> description
    examples: Optional[Dict[str, Any]] = None

class BasePromptTemplate(ABC):
    """
    Base class for all prompt templates.
    
    This provides a structured approach to prompt management,
    making the system easier to debug, update, and extend.
    """
    
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self._initialize_templates()
    
    @abstractmethod
    def _initialize_templates(self):
        """Initialize the prompt templates for this category."""
        pass
    
    def get_template(self, template_name: str) -> Optional[PromptTemplate]:
        """Get a specific prompt template."""
        return self.templates.get(template_name)
    
    def format_prompt(self, template_name: str, **kwargs) -> tuple[str, str]:
        """
        Format a prompt template with provided variables.
        
        Returns:
            tuple: (system_prompt, user_prompt)
        """
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        # Validate required variables
        missing_vars = set(template.variables.keys()) - set(kwargs.keys())
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        # Format prompts
        try:
            system_prompt = template.system_prompt.format(**kwargs)
            user_prompt = template.user_prompt_template.format(**kwargs)
            return system_prompt, user_prompt
        except KeyError as e:
            raise ValueError(f"Error formatting template '{template_name}': {e}")
    
    def list_templates(self) -> Dict[str, str]:
        """List all available templates with descriptions."""
        return {name: template.description for name, template in self.templates.items()}
    
    def validate_template(self, template_name: str, **kwargs) -> bool:
        """Validate that all required variables are provided."""
        template = self.get_template(template_name)
        if not template:
            return False
        
        required_vars = set(template.variables.keys())
        provided_vars = set(kwargs.keys())
        
        return required_vars.issubset(provided_vars)

class PromptManager:
    """
    Central manager for all prompt templates.
    
    This provides a single interface to access all prompt categories
    while maintaining organization and type safety.
    """
    
    def __init__(self):
        self.prompt_categories: Dict[PromptType, BasePromptTemplate] = {}
    
    def register_category(self, prompt_type: PromptType, prompt_class: BasePromptTemplate):
        """Register a prompt category."""
        self.prompt_categories[prompt_type] = prompt_class
    
    def get_prompt(self, prompt_type: PromptType, template_name: str, **kwargs) -> tuple[str, str]:
        """Get a formatted prompt from any category."""
        if prompt_type not in self.prompt_categories:
            raise ValueError(f"Prompt type '{prompt_type}' not registered")
        
        category = self.prompt_categories[prompt_type]
        return category.format_prompt(template_name, **kwargs)
    
    def list_all_templates(self) -> Dict[str, Dict[str, str]]:
        """List all templates across all categories."""
        all_templates = {}
        for prompt_type, category in self.prompt_categories.items():
            all_templates[prompt_type.value] = category.list_templates()
        return all_templates

# Global prompt manager instance
_prompt_manager = None

def get_prompt_manager() -> PromptManager:
    """Get or create the global prompt manager."""
    global _prompt_manager
    
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
        
        # Register all prompt categories
        from .citation import CitationPrompts
        from .reasoning import ReasoningPrompts
        from .idk import IDKPrompts
        from .summary import SummaryPrompts
        
        _prompt_manager.register_category(PromptType.CITATION, CitationPrompts())
        _prompt_manager.register_category(PromptType.REASONING, ReasoningPrompts())
        _prompt_manager.register_category(PromptType.IDK_REFUSAL, IDKPrompts())
        _prompt_manager.register_category(PromptType.SUMMARY, SummaryPrompts())
    
    return _prompt_manager
