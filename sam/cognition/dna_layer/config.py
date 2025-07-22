"""
DNA Layer Configuration
=======================

Configuration management for the DNA (Dynamic Neural Architecture) layer.
Defines hyperparameters, module specifications, and training settings.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import torch


@dataclass
class DNAConfig:
    """Configuration for DNA Layer implementation."""
    
    # Core Architecture
    hidden_size: int = 768  # Model hidden dimension
    num_experts: int = 4    # Number of expert modules
    top_k: int = 1         # Number of experts to route each token to
    
    # Expert Module Configuration
    expert_types: List[str] = field(default_factory=lambda: [
        'attention',
        'mlp', 
        'identity',
        'normalization'
    ])
    
    # Router Configuration
    router_hidden_size: int = 256  # Router internal dimension
    router_dropout: float = 0.1    # Router dropout rate
    
    # Attention Module Settings
    num_attention_heads: int = 12
    attention_dropout: float = 0.1
    
    # MLP Module Settings  
    mlp_intermediate_size: int = 3072  # 4x hidden_size typically
    mlp_dropout: float = 0.1
    mlp_activation: str = 'gelu'
    
    # Training Configuration
    load_balancing_weight: float = 0.01  # Weight for load balancing loss
    routing_temperature: float = 1.0     # Temperature for routing softmax
    use_gumbel_softmax: bool = True      # Use Gumbel-Softmax for differentiable routing
    
    # Efficiency Settings
    compute_efficiency_target: float = 0.3  # Target % of tokens using IdentityModule
    routing_entropy_threshold: float = 0.5  # Minimum routing entropy to prevent collapse
    
    # Metrics and Monitoring
    track_routing_stats: bool = True
    track_compute_savings: bool = True
    track_specialization: bool = True
    save_routing_visualizations: bool = True
    
    # Device and Performance
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    
    # Experimental Features
    adaptive_routing: bool = False      # Future: adaptive top-k based on complexity
    hierarchical_routing: bool = False  # Future: multi-level routing
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.num_experts == len(self.expert_types), \
            f"Number of experts ({self.num_experts}) must match expert types ({len(self.expert_types)})"
        
        assert self.top_k <= self.num_experts, \
            f"top_k ({self.top_k}) cannot exceed num_experts ({self.num_experts})"
        
        assert 0.0 <= self.load_balancing_weight <= 1.0, \
            f"load_balancing_weight must be between 0 and 1, got {self.load_balancing_weight}"
        
        assert self.routing_temperature > 0.0, \
            f"routing_temperature must be positive, got {self.routing_temperature}"
    
    def get_expert_config(self, expert_type: str) -> Dict[str, Any]:
        """Get configuration for a specific expert type."""
        base_config = {
            'hidden_size': self.hidden_size,
            'device': self.device
        }
        
        if expert_type == 'attention':
            return {
                **base_config,
                'num_heads': self.num_attention_heads,
                'dropout': self.attention_dropout
            }
        elif expert_type == 'mlp':
            return {
                **base_config,
                'intermediate_size': self.mlp_intermediate_size,
                'dropout': self.mlp_dropout,
                'activation': self.mlp_activation
            }
        elif expert_type == 'identity':
            return base_config
        elif expert_type == 'normalization':
            return {
                **base_config,
                'eps': 1e-12
            }
        else:
            raise ValueError(f"Unknown expert type: {expert_type}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DNAConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, path: str):
        """Save configuration to file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'DNAConfig':
        """Load configuration from file."""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Predefined configurations for different use cases
class DNAConfigs:
    """Predefined DNA configurations for different scenarios."""
    
    @staticmethod
    def proof_of_concept() -> DNAConfig:
        """Configuration for initial proof-of-concept."""
        return DNAConfig(
            hidden_size=768,
            num_experts=4,
            top_k=1,
            load_balancing_weight=0.01,
            compute_efficiency_target=0.3,
            track_routing_stats=True
        )
    
    @staticmethod
    def efficiency_focused() -> DNAConfig:
        """Configuration optimized for compute efficiency."""
        return DNAConfig(
            hidden_size=768,
            num_experts=4,
            top_k=1,
            load_balancing_weight=0.02,
            compute_efficiency_target=0.5,  # Higher efficiency target
            routing_temperature=0.8,        # Sharper routing decisions
            track_compute_savings=True
        )
    
    @staticmethod
    def performance_focused() -> DNAConfig:
        """Configuration optimized for model performance."""
        return DNAConfig(
            hidden_size=768,
            num_experts=6,  # More experts for specialization
            top_k=2,        # Allow tokens to visit multiple experts
            load_balancing_weight=0.005,  # Lower load balancing for performance
            routing_temperature=1.2,      # Softer routing for exploration
            track_specialization=True
        )
    
    @staticmethod
    def research() -> DNAConfig:
        """Configuration for research and analysis."""
        return DNAConfig(
            hidden_size=768,
            num_experts=4,
            top_k=1,
            track_routing_stats=True,
            track_compute_savings=True,
            track_specialization=True,
            save_routing_visualizations=True,
            use_gumbel_softmax=True
        )


# Default configuration
DEFAULT_DNA_CONFIG = DNAConfigs.proof_of_concept()
