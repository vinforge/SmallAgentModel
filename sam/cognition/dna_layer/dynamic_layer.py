"""
DNA Dynamic Layer
=================

Main implementation of the DNA (Dynamic Neural Architecture) layer.
This layer replaces standard transformer blocks with dynamic, data-dependent routing.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import time

from .modules import BaseModule, create_expert_module
from .router import TokenRouter
from .config import DNAConfig
from .metrics import DNAMetrics


class DNALayer(nn.Module):
    """
    Dynamic Neural Architecture Layer.
    
    This layer contains multiple expert modules and a learned router that
    dynamically decides which expert(s) each token should be processed by.
    This enables data-dependent compute allocation and learned efficiency.
    """
    
    def __init__(self, config: DNAConfig):
        super().__init__()
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        
        # Initialize expert modules
        self.experts = nn.ModuleList()
        for i, expert_type in enumerate(config.expert_types):
            expert_config = config.get_expert_config(expert_type)
            expert = create_expert_module(expert_type, config.hidden_size, expert_config)
            self.experts.append(expert)
        
        # Initialize token router
        router_config = {
            'top_k': config.top_k,
            'router_hidden_size': config.router_hidden_size,
            'router_dropout': config.router_dropout,
            'routing_temperature': config.routing_temperature,
            'use_gumbel_softmax': config.use_gumbel_softmax,
            'load_balancing_weight': config.load_balancing_weight
        }
        self.router = TokenRouter(config.hidden_size, config.num_experts, router_config)
        
        # Initialize metrics tracking
        if config.track_routing_stats:
            self.metrics = DNAMetrics(config.num_experts, config.expert_types)
        else:
            self.metrics = None
        
        # Performance tracking
        self.total_forward_time = 0.0
        self.total_forward_calls = 0
        
    def forward(self, 
                hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                token_types: Optional[List[str]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through the DNA layer.
        
        Args:
            hidden_states: Input token embeddings [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask [batch_size, seq_len]
            token_types: Optional token type labels for analysis
            
        Returns:
            Tuple of (output_hidden_states, routing_info)
        """
        start_time = time.time()
        
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device
        
        # Get routing decisions from the router
        routing_weights, routing_info = self.router(hidden_states, training=self.training)
        
        # Process tokens through experts based on routing decisions
        if self.top_k == 1:
            # Efficient processing for top-1 routing
            output_hidden_states = self._process_top1_routing(
                hidden_states, routing_weights, attention_mask
            )
        else:
            # General processing for top-k routing
            output_hidden_states = self._process_topk_routing(
                hidden_states, routing_weights, attention_mask
            )
        
        # Record metrics if enabled
        if self.metrics is not None:
            expert_indices = torch.argmax(routing_weights, dim=-1)
            self.metrics.record_routing_decision(
                expert_indices, routing_weights, routing_info, token_types
            )
        
        # Update performance metrics
        forward_time = time.time() - start_time
        self.total_forward_time += forward_time
        self.total_forward_calls += 1
        
        # Add performance info to routing_info
        routing_info.update({
            'forward_time': forward_time,
            'average_forward_time': self.total_forward_time / self.total_forward_calls,
            'tokens_processed': batch_size * seq_len
        })
        
        return output_hidden_states, routing_info
    
    def _process_top1_routing(self, 
                             hidden_states: torch.Tensor,
                             routing_weights: torch.Tensor,
                             attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Efficient processing for top-1 routing (each token goes to one expert).
        
        This is the most efficient implementation as it groups tokens by expert
        and processes them in batches.
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device
        
        # Get expert assignments for each token
        expert_indices = torch.argmax(routing_weights, dim=-1)  # [batch_size, seq_len]
        
        # Initialize output tensor
        output_hidden_states = torch.zeros_like(hidden_states)
        
        # Process tokens for each expert
        for expert_id in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_mask = (expert_indices == expert_id)
            
            if not expert_mask.any():
                continue  # No tokens for this expert
            
            # Extract tokens for this expert
            expert_tokens = hidden_states[expert_mask]  # [num_tokens, hidden_size]
            
            if expert_tokens.numel() == 0:
                continue
            
            # Create attention mask for expert tokens if provided
            expert_attention_mask = None
            if attention_mask is not None:
                expert_attention_mask = attention_mask[expert_mask]
            
            # Process through expert
            expert_output = self.experts[expert_id](
                expert_tokens.unsqueeze(0),  # Add batch dimension
                expert_attention_mask.unsqueeze(0) if expert_attention_mask is not None else None
            ).squeeze(0)  # Remove batch dimension
            
            # Place processed tokens back in output
            output_hidden_states[expert_mask] = expert_output
        
        return output_hidden_states

    def _process_topk_routing(self,
                             hidden_states: torch.Tensor,
                             routing_weights: torch.Tensor,
                             attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        General processing for top-k routing (tokens can go to multiple experts).

        This implementation handles the case where each token can be processed
        by multiple experts and the results are combined.
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device

        # Initialize output tensor
        output_hidden_states = torch.zeros_like(hidden_states)

        # For each position, process through top-k experts and combine
        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                token_hidden = hidden_states[batch_idx, seq_idx:seq_idx+1, :]  # [1, 1, hidden_size]
                token_weights = routing_weights[batch_idx, seq_idx, :]  # [num_experts]

                # Get top-k experts for this token
                top_k_weights, top_k_indices = torch.topk(token_weights, self.top_k)

                # Normalize weights
                top_k_weights = top_k_weights / (top_k_weights.sum() + 1e-8)

                # Process through each top-k expert and combine
                combined_output = torch.zeros_like(token_hidden)

                for i, expert_id in enumerate(top_k_indices):
                    expert_weight = top_k_weights[i]

                    # Create attention mask for this token if provided
                    token_attention_mask = None
                    if attention_mask is not None:
                        token_attention_mask = attention_mask[batch_idx, seq_idx:seq_idx+1].unsqueeze(0)

                    # Process through expert
                    expert_output = self.experts[expert_id](token_hidden, token_attention_mask)

                    # Add weighted contribution
                    combined_output += expert_weight * expert_output

                # Store combined output
                output_hidden_states[batch_idx, seq_idx, :] = combined_output.squeeze()

        return output_hidden_states

    def get_routing_analysis(self, hidden_states: torch.Tensor, token_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze routing behavior for the given input.

        Args:
            hidden_states: Input token embeddings
            token_types: Optional token type labels

        Returns:
            Dictionary with routing analysis
        """
        with torch.no_grad():
            routing_weights, routing_info = self.router(hidden_states, training=False)
            expert_indices = torch.argmax(routing_weights, dim=-1)

            analysis = {
                'routing_info': routing_info,
                'expert_usage': {},
                'routing_patterns': self.router.analyze_routing_patterns(hidden_states, token_types)
            }

            # Analyze expert usage
            for expert_id in range(self.num_experts):
                expert_usage = (expert_indices == expert_id).sum().item()
                total_tokens = expert_indices.numel()
                analysis['expert_usage'][f'expert_{expert_id}_{self.config.expert_types[expert_id]}'] = {
                    'count': expert_usage,
                    'percentage': expert_usage / total_tokens if total_tokens > 0 else 0.0
                }

            return analysis

    def get_efficiency_metrics(self) -> Dict[str, Any]:
        """Get compute efficiency metrics."""
        router_metrics = self.router.get_efficiency_metrics()

        # Add layer-specific metrics
        layer_metrics = {
            'average_forward_time': self.total_forward_time / max(self.total_forward_calls, 1),
            'total_forward_calls': self.total_forward_calls,
            'expert_configuration': {
                'num_experts': self.num_experts,
                'expert_types': self.config.expert_types,
                'top_k': self.top_k
            }
        }

        # Combine metrics
        efficiency_metrics = {**router_metrics, **layer_metrics}

        # Add metrics from DNAMetrics if available
        if self.metrics is not None:
            efficiency_metrics['detailed_metrics'] = self.metrics.get_efficiency_summary()

        return efficiency_metrics

    def reset_metrics(self):
        """Reset all metrics."""
        self.router.reset_metrics()
        if self.metrics is not None:
            self.metrics.reset_metrics()
        self.total_forward_time = 0.0
        self.total_forward_calls = 0

    def save_routing_analysis(self, filepath: str, hidden_states: torch.Tensor, token_types: Optional[List[str]] = None):
        """Save comprehensive routing analysis to file."""
        analysis = self.get_routing_analysis(hidden_states, token_types)
        efficiency_metrics = self.get_efficiency_metrics()

        report = {
            'timestamp': time.time(),
            'config': self.config.to_dict(),
            'routing_analysis': analysis,
            'efficiency_metrics': efficiency_metrics
        }

        # Add detailed metrics if available
        if self.metrics is not None:
            report['detailed_report'] = self.metrics.generate_report()

        import json
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

    def __repr__(self) -> str:
        """String representation of the DNA layer."""
        return (f"DNALayer(hidden_size={self.hidden_size}, "
                f"num_experts={self.num_experts}, "
                f"expert_types={self.config.expert_types}, "
                f"top_k={self.top_k})")


class DNALayerFactory:
    """Factory for creating DNA layers with different configurations."""

    @staticmethod
    def create_proof_of_concept_layer(hidden_size: int = 768) -> DNALayer:
        """Create a DNA layer for proof-of-concept testing."""
        from .config import DNAConfigs
        config = DNAConfigs.proof_of_concept()
        config.hidden_size = hidden_size
        return DNALayer(config)

    @staticmethod
    def create_efficiency_focused_layer(hidden_size: int = 768) -> DNALayer:
        """Create a DNA layer optimized for efficiency."""
        from .config import DNAConfigs
        config = DNAConfigs.efficiency_focused()
        config.hidden_size = hidden_size
        return DNALayer(config)

    @staticmethod
    def create_performance_focused_layer(hidden_size: int = 768) -> DNALayer:
        """Create a DNA layer optimized for performance."""
        from .config import DNAConfigs
        config = DNAConfigs.performance_focused()
        config.hidden_size = hidden_size
        return DNALayer(config)

    @staticmethod
    def create_research_layer(hidden_size: int = 768) -> DNALayer:
        """Create a DNA layer for research and analysis."""
        from .config import DNAConfigs
        config = DNAConfigs.research()
        config.hidden_size = hidden_size
        return DNALayer(config)
