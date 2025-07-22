"""
SAM 2.0 Hybrid Linear Attention Model
=====================================

Implementation of the HGRN-2 Hybrid model with 3:1 linear-to-full attention ratio.
This is the core model architecture for SAM 2.0's enhanced context capabilities.

Based on:
- Phase 0 experimental results
- HGRN-2 architecture from flash-linear-attention
- 3:1 ratio optimization for parameter efficiency

Author: SAM Development Team
Version: 1.0.0
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import logging

from .hybrid_config import HybridModelConfig

logger = logging.getLogger(__name__)

try:
    from flash_linear_attention import HGRN2Attention, GatedDeltaNetAttention
    FLASH_LINEAR_AVAILABLE = True
    logger.info("✅ Flash Linear Attention available")
except ImportError:
    FLASH_LINEAR_AVAILABLE = False
    logger.warning("⚠️ Flash Linear Attention not available, using fallback implementations")

class LinearAttentionLayer(nn.Module):
    """Linear attention layer with O(n) complexity."""
    
    def __init__(self, config: HybridModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        
        # Projection layers
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Feature map for linear attention
        if config.feature_map_type == "relu":
            self.feature_map = nn.ReLU()
        elif config.feature_map_type == "elu":
            self.feature_map = nn.ELU()
        else:
            self.feature_map = nn.Identity()  # For softmax or custom
        
        self.dropout = nn.Dropout(config.linear_attention_dropout)
        
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with linear attention.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply feature map for linear attention
        q = self.feature_map(q)
        k = self.feature_map(k)
        
        # Linear attention computation: O(n) instead of O(n²)
        # Compute K^T V first (d x d matrix)
        kv = torch.einsum('bshd,bshe->bhde', k, v)
        
        # Then Q (K^T V) (n x d matrix)
        out = torch.einsum('bqhd,bhde->bqhe', q, kv)
        
        # Reshape and project
        out = out.contiguous().view(batch_size, seq_len, self.hidden_size)
        out = self.o_proj(out)
        
        return self.dropout(out)

class FullAttentionLayer(nn.Module):
    """Standard full attention layer with O(n²) complexity."""
    
    def __init__(self, config: HybridModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5
        
        # Projection layers
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.dropout = nn.Dropout(config.attention_dropout)
        
    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with full attention.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch, heads, seq, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores: O(n²) complexity
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        out = self.o_proj(out)
        
        return out

class FeedForwardNetwork(nn.Module):
    """Feed-forward network with SwiGLU activation."""
    
    def __init__(self, config: HybridModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU activation: SiLU(gate) * up"""
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class HybridTransformerLayer(nn.Module):
    """Single transformer layer with either linear or full attention."""
    
    def __init__(self, config: HybridModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention_type = config.get_layer_type(layer_idx)
        
        # Choose attention mechanism
        if self.attention_type == "linear":
            self.attention = LinearAttentionLayer(config)
        else:
            self.attention = FullAttentionLayer(config)
        
        self.feed_forward = FeedForwardNetwork(config)
        
        # Layer normalization
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the hybrid transformer layer.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor
        """
        # Pre-norm attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        # Pre-norm feed-forward
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class SAMHybridModel(nn.Module):
    """
    SAM 2.0 Hybrid Linear Attention Model.
    
    Combines linear attention (O(n)) and full attention (O(n²)) layers
    in a 3:1 ratio for optimal efficiency and performance.
    """
    
    def __init__(self, config: HybridModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        
        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            HybridTransformerLayer(config, i) for i in range(config.num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"✅ SAM Hybrid Model initialized with {self.get_parameter_count():,} parameters")
        logger.info(f"   Architecture: {self._get_architecture_summary()}")
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def get_parameter_count(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def _get_architecture_summary(self) -> str:
        """Get a summary of the model architecture."""
        linear_count = sum(1 for i in range(self.num_layers) 
                          if self.config.get_layer_type(i) == "linear")
        full_count = sum(1 for i in range(self.num_layers) 
                        if self.config.get_layer_type(i) == "full")
        
        return f"HGRN-2 {linear_count}L+{full_count}F ({self.config.linear_ratio}:1 ratio)"
    
    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]
            
        Returns:
            Logits tensor [batch_size, seq_len, vocab_size]
        """
        # Embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Transform layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        return logits
    
    def generate_text(self, input_ids: torch.Tensor, max_length: int = 100,
                     temperature: float = 1.0, do_sample: bool = True) -> torch.Tensor:
        """
        Simple text generation (placeholder for full implementation).
        
        Args:
            input_ids: Input token IDs
            max_length: Maximum generation length
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            Generated token IDs
        """
        self.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                logits = self.forward(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated

# Factory function for easy model creation
def create_sam_hybrid_model(config: Optional[HybridModelConfig] = None) -> SAMHybridModel:
    """
    Create a SAM hybrid model with the specified configuration.
    
    Args:
        config: Model configuration (uses default if None)
        
    Returns:
        Initialized SAM hybrid model
    """
    if config is None:
        from .hybrid_config import get_default_config
        config = get_default_config()
    
    return SAMHybridModel(config)

# Export main classes
__all__ = [
    'SAMHybridModel',
    'HybridTransformerLayer', 
    'LinearAttentionLayer',
    'FullAttentionLayer',
    'create_sam_hybrid_model'
]
