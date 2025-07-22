"""
Token Router for DNA Layer
===========================

Implements the learned routing mechanism that decides which expert modules
each token should be processed by. This is the "brain" of the DNA layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
import math


class TokenRouter(nn.Module):
    """
    Learned token router for dynamic expert selection.
    
    The router analyzes each token's hidden state and decides which expert module(s)
    should process it. This enables data-dependent compute allocation.
    """
    
    def __init__(self, hidden_size: int, num_experts: int, config: Dict):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = config.get('top_k', 1)
        self.router_hidden_size = config.get('router_hidden_size', 256)
        self.dropout = config.get('router_dropout', 0.1)
        self.temperature = config.get('routing_temperature', 1.0)
        self.use_gumbel_softmax = config.get('use_gumbel_softmax', True)
        self.load_balancing_weight = config.get('load_balancing_weight', 0.01)
        
        # Router network - simple MLP
        self.router_network = nn.Sequential(
            nn.Linear(hidden_size, self.router_hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.router_hidden_size, self.router_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.router_hidden_size // 2, num_experts)
        )
        
        # Initialize router weights
        self._init_router_weights()
        
        # Metrics tracking
        self.routing_stats = {
            'expert_usage': torch.zeros(num_experts),
            'routing_entropy': 0.0,
            'load_balance_loss': 0.0,
            'total_tokens': 0
        }
        
    def _init_router_weights(self):
        """Initialize router weights for stable training."""
        for module in self.router_network:
            if isinstance(module, nn.Linear):
                # Xavier initialization for stable gradients
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
        
        # Initialize final layer with small weights to start with uniform routing
        final_layer = self.router_network[-1]
        nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)
        nn.init.constant_(final_layer.bias, 0.0)
    
    def forward(self, hidden_states: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, Dict]:
        """
        Route tokens to expert modules.
        
        Args:
            hidden_states: Input token embeddings [batch_size, seq_len, hidden_size]
            training: Whether in training mode (affects Gumbel-Softmax)
            
        Returns:
            Tuple of (routing_weights, routing_info)
            - routing_weights: [batch_size, seq_len, num_experts]
            - routing_info: Dictionary with routing statistics and losses
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Flatten for router processing
        flat_hidden = hidden_states.view(-1, hidden_size)  # [batch_size * seq_len, hidden_size]
        
        # Compute routing logits
        routing_logits = self.router_network(flat_hidden)  # [batch_size * seq_len, num_experts]
        
        # Apply temperature scaling
        routing_logits = routing_logits / self.temperature
        
        # Compute routing probabilities
        if self.use_gumbel_softmax and training:
            # Use Gumbel-Softmax for differentiable routing during training
            routing_weights = F.gumbel_softmax(routing_logits, tau=self.temperature, hard=False)
        else:
            # Use standard softmax during inference
            routing_weights = F.softmax(routing_logits, dim=-1)
        
        # Apply top-k gating
        if self.top_k < self.num_experts:
            routing_weights = self._apply_top_k_gating(routing_weights)
        
        # Reshape back to original dimensions
        routing_weights = routing_weights.view(batch_size, seq_len, self.num_experts)
        
        # Compute routing statistics and losses
        routing_info = self._compute_routing_info(routing_weights, routing_logits)
        
        # Update metrics
        self._update_metrics(routing_weights)
        
        return routing_weights, routing_info
    
    def _apply_top_k_gating(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """Apply top-k gating to routing weights."""
        # Get top-k experts for each token
        top_k_values, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Create mask for top-k experts
        mask = torch.zeros_like(routing_weights)
        mask.scatter_(-1, top_k_indices, 1.0)
        
        # Zero out non-top-k experts and renormalize
        routing_weights = routing_weights * mask
        routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        return routing_weights
    
    def _compute_routing_info(self, routing_weights: torch.Tensor, routing_logits: torch.Tensor) -> Dict:
        """Compute routing statistics and auxiliary losses."""
        batch_size, seq_len, num_experts = routing_weights.shape
        
        # Compute load balancing loss
        # Encourages even distribution of tokens across experts
        expert_usage = routing_weights.sum(dim=(0, 1))  # [num_experts]
        total_tokens = batch_size * seq_len
        
        # Target: each expert should get 1/num_experts of the tokens
        target_usage = total_tokens / num_experts
        load_balance_loss = torch.sum((expert_usage - target_usage) ** 2) / num_experts
        load_balance_loss = load_balance_loss * self.load_balancing_weight
        
        # Compute routing entropy (measure of routing diversity)
        # Higher entropy = more diverse routing, lower entropy = routing collapse
        routing_probs = routing_weights.mean(dim=(0, 1))  # Average probability per expert
        routing_entropy = -torch.sum(routing_probs * torch.log(routing_probs + 1e-8))
        
        # Compute expert utilization statistics
        expert_utilization = expert_usage / total_tokens
        
        # Compute routing confidence (how confident the router is in its decisions)
        max_routing_weights = routing_weights.max(dim=-1)[0]  # [batch_size, seq_len]
        routing_confidence = max_routing_weights.mean()
        
        return {
            'load_balance_loss': load_balance_loss,
            'routing_entropy': routing_entropy,
            'expert_utilization': expert_utilization,
            'routing_confidence': routing_confidence,
            'expert_usage': expert_usage,
            'total_tokens': total_tokens
        }
    
    def _update_metrics(self, routing_weights: torch.Tensor):
        """Update running metrics for monitoring."""
        batch_size, seq_len, num_experts = routing_weights.shape
        
        # Update expert usage statistics
        expert_usage = routing_weights.sum(dim=(0, 1))
        self.routing_stats['expert_usage'] += expert_usage.detach().cpu()
        self.routing_stats['total_tokens'] += batch_size * seq_len
        
        # Update routing entropy
        routing_probs = routing_weights.mean(dim=(0, 1))
        entropy = -torch.sum(routing_probs * torch.log(routing_probs + 1e-8))
        self.routing_stats['routing_entropy'] = entropy.detach().cpu().item()
    
    def get_routing_decisions(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get discrete routing decisions for inference.
        
        Returns:
            Tuple of (expert_indices, routing_weights)
            - expert_indices: [batch_size, seq_len] - which expert each token goes to
            - routing_weights: [batch_size, seq_len, num_experts] - routing probabilities
        """
        with torch.no_grad():
            routing_weights, _ = self.forward(hidden_states, training=False)
            expert_indices = torch.argmax(routing_weights, dim=-1)
            return expert_indices, routing_weights
    
    def analyze_routing_patterns(self, hidden_states: torch.Tensor, token_types: Optional[List[str]] = None) -> Dict:
        """
        Analyze routing patterns for different token types.
        
        Args:
            hidden_states: Input token embeddings
            token_types: Optional list of token type labels for analysis
            
        Returns:
            Dictionary with routing pattern analysis
        """
        expert_indices, routing_weights = self.get_routing_decisions(hidden_states)
        
        analysis = {
            'expert_distribution': {},
            'routing_confidence': routing_weights.max(dim=-1)[0].mean().item(),
            'routing_entropy': self.routing_stats['routing_entropy']
        }
        
        # Analyze expert usage distribution
        for expert_id in range(self.num_experts):
            expert_usage = (expert_indices == expert_id).sum().item()
            total_tokens = expert_indices.numel()
            analysis['expert_distribution'][f'expert_{expert_id}'] = expert_usage / total_tokens
        
        # Analyze by token type if provided
        if token_types is not None:
            analysis['token_type_routing'] = {}
            # Flatten expert_indices to match token_types length
            flat_expert_indices = expert_indices.flatten()

            for token_type in set(token_types):
                type_mask = torch.tensor([t == token_type for t in token_types])
                if type_mask.any():
                    type_routing = flat_expert_indices[type_mask]
                    type_distribution = {}
                    for expert_id in range(self.num_experts):
                        expert_usage = (type_routing == expert_id).sum().item()
                        type_distribution[f'expert_{expert_id}'] = expert_usage / len(type_routing) if len(type_routing) > 0 else 0.0
                    analysis['token_type_routing'][token_type] = type_distribution
        
        return analysis
    
    def reset_metrics(self):
        """Reset routing metrics."""
        self.routing_stats = {
            'expert_usage': torch.zeros(self.num_experts),
            'routing_entropy': 0.0,
            'load_balance_loss': 0.0,
            'total_tokens': 0
        }
    
    def get_efficiency_metrics(self) -> Dict:
        """Get compute efficiency metrics."""
        if self.routing_stats['total_tokens'] == 0:
            return {'identity_usage': 0.0, 'compute_savings': 0.0}
        
        # Assuming expert 2 is the IdentityModule (index 2)
        identity_expert_id = 2  # This should match the expert configuration
        identity_usage = self.routing_stats['expert_usage'][identity_expert_id] / self.routing_stats['total_tokens']
        
        # Compute savings = percentage of tokens that skip computation
        compute_savings = identity_usage.item()
        
        return {
            'identity_usage': identity_usage.item(),
            'compute_savings': compute_savings,
            'expert_usage_distribution': (self.routing_stats['expert_usage'] / self.routing_stats['total_tokens']).tolist()
        }
