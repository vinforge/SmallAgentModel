"""
DNA Expert Modules
==================

Implementation of specialized expert modules for the DNA layer.
Each module represents a different computational pathway that tokens can be routed to.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import math


class BaseModule(nn.Module, ABC):
    """
    Base class for all DNA expert modules.
    
    All expert modules must inherit from this class and implement the forward method.
    This ensures consistent interface and enables proper routing and metrics tracking.
    """
    
    def __init__(self, hidden_size: int, config: Dict[str, Any]):
        super().__init__()
        self.hidden_size = hidden_size
        self.config = config
        self.module_type = self.__class__.__name__.lower().replace('module', '')
        
        # Metrics tracking
        self.token_count = 0
        self.compute_flops = 0
        
    @abstractmethod
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the expert module.
        
        Args:
            hidden_states: Input token embeddings [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask [batch_size, seq_len]
            
        Returns:
            Processed token embeddings [batch_size, seq_len, hidden_size]
        """
        pass
    
    def get_flops_per_token(self) -> int:
        """Return approximate FLOPs per token for this module."""
        return 0  # Override in subclasses
    
    def reset_metrics(self):
        """Reset metrics tracking."""
        self.token_count = 0
        self.compute_flops = 0


class IdentityModule(BaseModule):
    """
    Identity module that passes tokens through unchanged.
    
    This is the key efficiency module - tokens routed here skip computation entirely.
    High usage of this module indicates learned compute savings.
    """
    
    def __init__(self, hidden_size: int, config: Dict[str, Any]):
        super().__init__(hidden_size, config)
        # No parameters - this is a pure skip connection
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pass through unchanged - zero computation."""
        self.token_count += hidden_states.numel() // self.hidden_size
        # No FLOPs added - this is the efficiency gain!
        return hidden_states
    
    def get_flops_per_token(self) -> int:
        """Identity module uses zero FLOPs."""
        return 0


class AttentionModule(BaseModule):
    """
    Specialized multi-head self-attention module.
    
    Designed to handle tokens that require contextual understanding
    and complex inter-token relationships.
    """
    
    def __init__(self, hidden_size: int, config: Dict[str, Any]):
        super().__init__(hidden_size, config)
        
        self.num_heads = config.get('num_heads', 12)
        self.head_dim = hidden_size // self.num_heads
        self.dropout = config.get('dropout', 0.1)
        
        assert hidden_size % self.num_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({self.num_heads})"
        
        # Multi-head attention components
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        
        self.attention_dropout = nn.Dropout(self.dropout)
        self.output_dropout = nn.Dropout(self.dropout)
        
        # Layer normalization and residual connection
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Multi-head self-attention forward pass."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        self.token_count += batch_size * seq_len
        
        # Store residual for skip connection
        residual = hidden_states
        
        # Apply layer norm first (pre-norm architecture)
        hidden_states = self.layer_norm(hidden_states)
        
        # Compute Q, K, V
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, value)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        output = self.output(context)
        output = self.output_dropout(output)
        
        # Residual connection
        return residual + output
    
    def get_flops_per_token(self) -> int:
        """Approximate FLOPs for attention computation."""
        # Q, K, V projections: 3 * hidden_size^2
        # Attention computation: 2 * seq_len * hidden_size (approximation)
        # Output projection: hidden_size^2
        return 4 * self.hidden_size * self.hidden_size


class MLPModule(BaseModule):
    """
    Specialized MLP (feed-forward) module.
    
    Designed to handle tokens requiring abstract processing and
    non-linear transformations without inter-token dependencies.
    """
    
    def __init__(self, hidden_size: int, config: Dict[str, Any]):
        super().__init__(hidden_size, config)
        
        self.intermediate_size = config.get('intermediate_size', 4 * hidden_size)
        self.dropout = config.get('dropout', 0.1)
        self.activation = config.get('activation', 'gelu')
        
        # MLP layers
        self.dense_1 = nn.Linear(hidden_size, self.intermediate_size)
        self.dense_2 = nn.Linear(self.intermediate_size, hidden_size)
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Layer normalization and activation
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.activation_fn = self._get_activation_fn(self.activation)
        
    def _get_activation_fn(self, activation: str):
        """Get activation function by name."""
        if activation == 'gelu':
            return F.gelu
        elif activation == 'relu':
            return F.relu
        elif activation == 'swish':
            return F.silu
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """MLP forward pass with residual connection."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        self.token_count += batch_size * seq_len
        
        # Store residual for skip connection
        residual = hidden_states
        
        # Apply layer norm first (pre-norm architecture)
        hidden_states = self.layer_norm(hidden_states)
        
        # MLP transformation
        hidden_states = self.dense_1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.dropout_layer(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout_layer(hidden_states)
        
        # Residual connection
        return residual + hidden_states
    
    def get_flops_per_token(self) -> int:
        """Approximate FLOPs for MLP computation."""
        # Two linear transformations
        return 2 * self.hidden_size * self.intermediate_size


class NormalizationModule(BaseModule):
    """
    Lightweight normalization module.
    
    Handles tokens that need normalization and residual connections
    but not full computation. More efficient than full MLP/Attention.
    """
    
    def __init__(self, hidden_size: int, config: Dict[str, Any]):
        super().__init__(hidden_size, config)
        
        self.eps = config.get('eps', 1e-12)
        
        # Lightweight components
        self.layer_norm = nn.LayerNorm(hidden_size, eps=self.eps)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Lightweight normalization and linear transformation."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        self.token_count += batch_size * seq_len
        
        # Store residual
        residual = hidden_states
        
        # Normalize and apply lightweight transformation
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Residual connection
        return residual + hidden_states
    
    def get_flops_per_token(self) -> int:
        """Approximate FLOPs for normalization module."""
        # Layer norm + linear transformation
        return self.hidden_size * self.hidden_size


# Module factory for creating expert modules
def create_expert_module(module_type: str, hidden_size: int, config: Dict[str, Any]) -> BaseModule:
    """
    Factory function to create expert modules by type.
    
    Args:
        module_type: Type of module ('attention', 'mlp', 'identity', 'normalization')
        hidden_size: Model hidden dimension
        config: Module-specific configuration
        
    Returns:
        Initialized expert module
    """
    module_classes = {
        'attention': AttentionModule,
        'mlp': MLPModule,
        'identity': IdentityModule,
        'normalization': NormalizationModule
    }
    
    if module_type not in module_classes:
        raise ValueError(f"Unknown module type: {module_type}. Available: {list(module_classes.keys())}")
    
    return module_classes[module_type](hidden_size, config)
