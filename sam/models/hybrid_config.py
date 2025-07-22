"""
SAM 2.0 Hybrid Model Configuration
==================================

Configuration classes for the HGRN-2 Hybrid Linear Attention model.
Based on Phase 0 experimental results selecting 3:1 linear-to-full attention ratio.

Author: SAM Development Team
Version: 1.0.0
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class HybridModelConfig:
    """Configuration for SAM's HGRN-2 Hybrid model."""
    
    # Model Architecture
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_layers: int = 32
    num_attention_heads: int = 32
    intermediate_size: int = 11008
    
    # Hybrid Attention Configuration
    linear_ratio: int = 3  # 3 linear attention layers per 1 full attention layer
    model_type: str = "hgrn2"  # "hgrn2" or "gated_deltanet"
    
    # Context and Performance
    max_context_length: int = 100000  # Target context window
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    
    # Linear Attention Specific
    feature_map_type: str = "relu"  # "relu", "elu", "softmax"
    linear_attention_dropout: float = 0.0
    
    # Full Attention Specific  
    attention_dropout: float = 0.0
    use_flash_attention: bool = True  # Use flash attention when available
    
    # Model Behavior
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    use_cache: bool = True
    
    # Training Configuration (for future use)
    gradient_checkpointing: bool = False
    tie_word_embeddings: bool = False
    
    # SAM-Specific Features
    enable_thinking_tokens: bool = True  # Support for <think> tokens
    thinking_token_id: int = 32001  # Special token ID for thinking
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.linear_ratio < 1:
            raise ValueError("linear_ratio must be at least 1")
        
        if self.model_type not in ["hgrn2", "gated_deltanet"]:
            raise ValueError("model_type must be 'hgrn2' or 'gated_deltanet'")
        
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        
        # Adjust vocab_size for thinking tokens if enabled
        if self.enable_thinking_tokens and self.vocab_size <= self.thinking_token_id:
            self.vocab_size = self.thinking_token_id + 100  # Add buffer for special tokens
    
    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.hidden_size // self.num_attention_heads
    
    @property
    def total_blocks(self) -> int:
        """Total number of hybrid blocks (each containing linear_ratio + 1 layers)."""
        return self.num_layers // (self.linear_ratio + 1)
    
    @property
    def linear_layers_per_block(self) -> int:
        """Number of linear attention layers per hybrid block."""
        return self.linear_ratio
    
    @property
    def full_layers_per_block(self) -> int:
        """Number of full attention layers per hybrid block (always 1)."""
        return 1
    
    def get_layer_type(self, layer_idx: int) -> str:
        """
        Get the attention type for a given layer index.
        
        Args:
            layer_idx: Layer index (0-based)
            
        Returns:
            "linear" or "full"
        """
        block_size = self.linear_ratio + 1
        position_in_block = layer_idx % block_size
        
        # Last layer in each block is full attention
        if position_in_block == self.linear_ratio:
            return "full"
        else:
            return "linear"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "linear_ratio": self.linear_ratio,
            "model_type": self.model_type,
            "max_context_length": self.max_context_length,
            "rope_theta": self.rope_theta,
            "rope_scaling": self.rope_scaling,
            "feature_map_type": self.feature_map_type,
            "linear_attention_dropout": self.linear_attention_dropout,
            "attention_dropout": self.attention_dropout,
            "use_flash_attention": self.use_flash_attention,
            "layer_norm_eps": self.layer_norm_eps,
            "initializer_range": self.initializer_range,
            "use_cache": self.use_cache,
            "gradient_checkpointing": self.gradient_checkpointing,
            "tie_word_embeddings": self.tie_word_embeddings,
            "enable_thinking_tokens": self.enable_thinking_tokens,
            "thinking_token_id": self.thinking_token_id
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "HybridModelConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

# Predefined configurations for different model sizes
class HybridModelConfigs:
    """Predefined configurations for different SAM hybrid model sizes."""
    
    @staticmethod
    def sam_8b_hybrid() -> HybridModelConfig:
        """Configuration for SAM 8B hybrid model (production target)."""
        return HybridModelConfig(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_attention_heads=32,
            intermediate_size=11008,
            linear_ratio=3,
            max_context_length=100000
        )
    
    @staticmethod
    def sam_3b_hybrid() -> HybridModelConfig:
        """Configuration for SAM 3B hybrid model (efficient version)."""
        return HybridModelConfig(
            vocab_size=32000,
            hidden_size=2560,
            num_layers=24,
            num_attention_heads=20,
            intermediate_size=6912,
            linear_ratio=3,
            max_context_length=100000
        )
    
    @staticmethod
    def sam_1b_hybrid() -> HybridModelConfig:
        """Configuration for SAM 1B hybrid model (development/testing)."""
        return HybridModelConfig(
            vocab_size=32000,
            hidden_size=2048,
            num_layers=16,
            num_attention_heads=16,
            intermediate_size=5504,
            linear_ratio=3,
            max_context_length=50000
        )
    
    @staticmethod
    def sam_debug_hybrid() -> HybridModelConfig:
        """Configuration for debugging and rapid testing."""
        return HybridModelConfig(
            vocab_size=1000,
            hidden_size=256,
            num_layers=4,
            num_attention_heads=4,
            intermediate_size=1024,
            linear_ratio=3,
            max_context_length=2048,
            enable_thinking_tokens=False
        )

def get_default_config() -> HybridModelConfig:
    """Get the default configuration for SAM hybrid model."""
    return HybridModelConfigs.sam_8b_hybrid()

def validate_config(config: HybridModelConfig) -> bool:
    """
    Validate a hybrid model configuration.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    try:
        # Test layer type assignment for all layers
        for i in range(config.num_layers):
            layer_type = config.get_layer_type(i)
            if layer_type not in ["linear", "full"]:
                raise ValueError(f"Invalid layer type at index {i}: {layer_type}")
        
        # Validate that we have the right number of each layer type
        linear_count = sum(1 for i in range(config.num_layers) 
                          if config.get_layer_type(i) == "linear")
        full_count = sum(1 for i in range(config.num_layers) 
                        if config.get_layer_type(i) == "full")
        
        expected_ratio = linear_count / full_count if full_count > 0 else float('inf')
        
        if abs(expected_ratio - config.linear_ratio) > 0.1:
            raise ValueError(f"Actual ratio {expected_ratio:.1f} doesn't match configured ratio {config.linear_ratio}")
        
        return True
        
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}")

# Export main classes and functions
__all__ = [
    'HybridModelConfig',
    'HybridModelConfigs', 
    'get_default_config',
    'validate_config'
]
