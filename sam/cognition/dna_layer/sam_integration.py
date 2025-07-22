"""
SAM DNA Layer Integration
=========================

Integration of DNA (Dynamic Neural Architecture) layer with SAM's existing
transformer architecture. Provides hybrid models that combine MEMOIR capabilities
with DNA routing efficiency.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, List
import logging
import time

from sam.core.model_layers import MEMOIRTransformerBlock
from .dynamic_layer import DNALayer, DNALayerFactory
from .config import DNAConfig, DNAConfigs
from .metrics import DNAMetrics

logger = logging.getLogger(__name__)


class DNAEnhancedMEMOIRBlock(nn.Module):
    """
    Hybrid transformer block that combines MEMOIR capabilities with DNA routing.
    
    This block can operate in three modes:
    1. MEMOIR-only: Standard MEMOIR transformer block
    2. DNA-only: Pure DNA layer routing
    3. Hybrid: DNA routing with MEMOIR memory capabilities
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
        max_edits: int = 10000,
        enable_memoir: bool = True,
        enable_dna: bool = True,
        dna_config: Optional[DNAConfig] = None,
        operation_mode: str = 'hybrid'  # 'memoir', 'dna', 'hybrid'
    ):
        """
        Initialize the DNA-enhanced MEMOIR block.
        
        Args:
            hidden_size: Hidden dimension size
            num_attention_heads: Number of attention heads
            intermediate_size: FFN intermediate dimension
            dropout: Dropout probability
            activation: Activation function name
            max_edits: Maximum edits for MEMOIR layer
            enable_memoir: Whether to enable MEMOIR functionality
            enable_dna: Whether to enable DNA functionality
            dna_config: DNA layer configuration
            operation_mode: How to combine MEMOIR and DNA ('memoir', 'dna', 'hybrid')
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.enable_memoir = enable_memoir
        self.enable_dna = enable_dna
        self.operation_mode = operation_mode
        
        # Initialize MEMOIR transformer block
        if enable_memoir:
            self.memoir_block = MEMOIRTransformerBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
                activation=activation,
                max_edits=max_edits,
                enable_memoir=True
            )
            logger.info("MEMOIR block initialized in DNA-enhanced block")
        else:
            self.memoir_block = None
        
        # Initialize DNA layer
        if enable_dna:
            if dna_config is None:
                dna_config = DNAConfigs.proof_of_concept()
                dna_config.hidden_size = hidden_size
                dna_config.num_attention_heads = num_attention_heads
                dna_config.mlp_intermediate_size = intermediate_size
            
            self.dna_layer = DNALayer(dna_config)
            logger.info("DNA layer initialized in DNA-enhanced block")
        else:
            self.dna_layer = None
        
        # Hybrid mode components
        if operation_mode == 'hybrid' and enable_memoir and enable_dna:
            # Gating mechanism to decide between MEMOIR and DNA
            self.mode_gate = nn.Linear(hidden_size, 2)  # Binary choice: MEMOIR or DNA
            self.gate_temperature = 1.0
            logger.info("Hybrid mode gating mechanism initialized")
        else:
            self.mode_gate = None
        
        # Performance tracking
        self.forward_times = {'memoir': [], 'dna': [], 'hybrid': []}
        self.mode_usage = {'memoir': 0, 'dna': 0, 'hybrid': 0}
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        edit_mask: Optional[torch.Tensor] = None,
        edit_id: Optional[str] = None,
        token_types: Optional[List[str]] = None,
        return_attention_weights: bool = False,
        enable_memoir_retrieval: bool = True,
        force_mode: Optional[str] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through the DNA-enhanced MEMOIR block.
        
        Args:
            hidden_states: Input token embeddings [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            edit_mask: MEMOIR edit mask
            edit_id: MEMOIR edit identifier
            token_types: Token type labels for DNA analysis
            return_attention_weights: Whether to return attention weights
            enable_memoir_retrieval: Whether to enable MEMOIR retrieval
            force_mode: Force specific mode ('memoir', 'dna', 'hybrid')
            
        Returns:
            Tuple of (output_hidden_states, info_dict)
        """
        start_time = time.time()
        
        # Determine operation mode
        active_mode = force_mode or self.operation_mode
        
        if active_mode == 'memoir' and self.memoir_block is not None:
            return self._forward_memoir_only(
                hidden_states, attention_mask, edit_mask, edit_id,
                return_attention_weights, enable_memoir_retrieval, start_time
            )
        
        elif active_mode == 'dna' and self.dna_layer is not None:
            return self._forward_dna_only(
                hidden_states, attention_mask, token_types, start_time
            )
        
        elif active_mode == 'hybrid' and self.memoir_block is not None and self.dna_layer is not None:
            return self._forward_hybrid(
                hidden_states, attention_mask, edit_mask, edit_id,
                token_types, return_attention_weights, enable_memoir_retrieval, start_time
            )
        
        else:
            # Fallback to available mode
            if self.memoir_block is not None:
                logger.warning(f"Requested mode '{active_mode}' not available, falling back to MEMOIR")
                return self._forward_memoir_only(
                    hidden_states, attention_mask, edit_mask, edit_id,
                    return_attention_weights, enable_memoir_retrieval, start_time
                )
            elif self.dna_layer is not None:
                logger.warning(f"Requested mode '{active_mode}' not available, falling back to DNA")
                return self._forward_dna_only(
                    hidden_states, attention_mask, token_types, start_time
                )
            else:
                raise RuntimeError("No valid processing mode available")
    
    def _forward_memoir_only(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        edit_mask: Optional[torch.Tensor],
        edit_id: Optional[str],
        return_attention_weights: bool,
        enable_memoir_retrieval: bool,
        start_time: float
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass using only MEMOIR block."""
        output_states, attention_weights = self.memoir_block(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            edit_mask=edit_mask,
            edit_id=edit_id,
            return_attention_weights=return_attention_weights,
            enable_memoir_retrieval=enable_memoir_retrieval
        )
        
        forward_time = time.time() - start_time
        self.forward_times['memoir'].append(forward_time)
        self.mode_usage['memoir'] += 1
        
        info_dict = {
            'mode': 'memoir',
            'forward_time': forward_time,
            'attention_weights': attention_weights if return_attention_weights else None,
            'memoir_active': True,
            'dna_active': False
        }
        
        return output_states, info_dict
    
    def _forward_dna_only(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        token_types: Optional[List[str]],
        start_time: float
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass using only DNA layer."""
        output_states, routing_info = self.dna_layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            token_types=token_types
        )
        
        forward_time = time.time() - start_time
        self.forward_times['dna'].append(forward_time)
        self.mode_usage['dna'] += 1
        
        info_dict = {
            'mode': 'dna',
            'forward_time': forward_time,
            'routing_info': routing_info,
            'memoir_active': False,
            'dna_active': True
        }
        
        return output_states, info_dict
    
    def _forward_hybrid(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        edit_mask: Optional[torch.Tensor],
        edit_id: Optional[str],
        token_types: Optional[List[str]],
        return_attention_weights: bool,
        enable_memoir_retrieval: bool,
        start_time: float
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass using hybrid MEMOIR + DNA approach."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Compute gating decision
        gate_logits = self.mode_gate(hidden_states.mean(dim=1))  # [batch_size, 2]
        gate_probs = torch.softmax(gate_logits / self.gate_temperature, dim=-1)
        
        # For simplicity in Phase 1B, use hard gating based on probability
        use_dna = gate_probs[:, 1] > gate_probs[:, 0]  # [batch_size]
        
        # Process each sample in the batch
        output_states = torch.zeros_like(hidden_states)
        combined_info = {
            'mode': 'hybrid',
            'memoir_samples': 0,
            'dna_samples': 0,
            'gating_decisions': gate_probs.detach().cpu().numpy().tolist()
        }
        
        for batch_idx in range(batch_size):
            sample_hidden = hidden_states[batch_idx:batch_idx+1]  # [1, seq_len, hidden_size]
            sample_mask = attention_mask[batch_idx:batch_idx+1] if attention_mask is not None else None
            
            if use_dna[batch_idx]:
                # Use DNA layer for this sample
                sample_output, dna_info = self.dna_layer(sample_hidden, sample_mask, token_types)
                output_states[batch_idx] = sample_output.squeeze(0)
                combined_info['dna_samples'] += 1
                if 'dna_routing_info' not in combined_info:
                    combined_info['dna_routing_info'] = []
                combined_info['dna_routing_info'].append(dna_info)
            else:
                # Use MEMOIR block for this sample
                sample_output, memoir_attention = self.memoir_block(
                    sample_hidden, sample_mask, edit_mask, edit_id,
                    return_attention_weights, enable_memoir_retrieval
                )
                output_states[batch_idx] = sample_output.squeeze(0)
                combined_info['memoir_samples'] += 1
        
        forward_time = time.time() - start_time
        self.forward_times['hybrid'].append(forward_time)
        self.mode_usage['hybrid'] += 1
        
        combined_info.update({
            'forward_time': forward_time,
            'memoir_active': True,
            'dna_active': True
        })
        
        return output_states, combined_info
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the hybrid block."""
        stats = {
            'mode_usage': self.mode_usage.copy(),
            'average_forward_times': {},
            'total_forward_calls': sum(self.mode_usage.values())
        }
        
        for mode, times in self.forward_times.items():
            if times:
                stats['average_forward_times'][mode] = sum(times) / len(times)
            else:
                stats['average_forward_times'][mode] = 0.0
        
        return stats
    
    def get_efficiency_metrics(self) -> Dict[str, Any]:
        """Get efficiency metrics from DNA layer if available."""
        if self.dna_layer is not None:
            return self.dna_layer.get_efficiency_metrics()
        else:
            return {'dna_not_available': True}
    
    def reset_metrics(self):
        """Reset all performance metrics."""
        self.forward_times = {'memoir': [], 'dna': [], 'hybrid': []}
        self.mode_usage = {'memoir': 0, 'dna': 0, 'hybrid': 0}
        
        if self.dna_layer is not None:
            self.dna_layer.reset_metrics()


class DNAEnhancedSAMModel(nn.Module):
    """
    SAM model with DNA layer integration.
    
    This model replaces one or more transformer layers with DNA-enhanced blocks
    for proof-of-concept validation and performance comparison.
    """
    
    def __init__(
        self,
        num_layers: int = 12,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        dna_layer_positions: List[int] = [6],  # Which layers to replace with DNA
        dna_config: Optional[DNAConfig] = None,
        operation_mode: str = 'hybrid'
    ):
        """
        Initialize DNA-enhanced SAM model.
        
        Args:
            num_layers: Total number of transformer layers
            hidden_size: Hidden dimension size
            num_attention_heads: Number of attention heads
            intermediate_size: FFN intermediate dimension
            dna_layer_positions: List of layer indices to replace with DNA (0-indexed)
            dna_config: DNA layer configuration
            operation_mode: Operation mode for DNA layers
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dna_layer_positions = set(dna_layer_positions)
        self.operation_mode = operation_mode
        
        # Create layers
        self.layers = nn.ModuleList()
        
        for layer_idx in range(num_layers):
            if layer_idx in self.dna_layer_positions:
                # Create DNA-enhanced layer
                layer = DNAEnhancedMEMOIRBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    intermediate_size=intermediate_size,
                    enable_memoir=True,
                    enable_dna=True,
                    dna_config=dna_config,
                    operation_mode=operation_mode
                )
                logger.info(f"Layer {layer_idx}: DNA-enhanced MEMOIR block created")
            else:
                # Create standard MEMOIR layer
                layer = MEMOIRTransformerBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    intermediate_size=intermediate_size,
                    enable_memoir=True
                )
                logger.info(f"Layer {layer_idx}: Standard MEMOIR block created")
            
            self.layers.append(layer)
        
        logger.info(f"DNA-enhanced SAM model created with {len(self.dna_layer_positions)} DNA layers at positions {list(self.dna_layer_positions)}")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_types: Optional[List[str]] = None,
        return_layer_outputs: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through the DNA-enhanced SAM model.
        
        Args:
            hidden_states: Input embeddings [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            token_types: Token type labels for DNA analysis
            return_layer_outputs: Whether to return outputs from each layer
            
        Returns:
            Tuple of (final_hidden_states, model_info)
        """
        layer_outputs = []
        model_info = {
            'dna_layers_info': {},
            'total_layers': self.num_layers,
            'dna_layer_positions': list(self.dna_layer_positions)
        }
        
        current_hidden_states = hidden_states
        
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx in self.dna_layer_positions:
                # DNA-enhanced layer
                current_hidden_states, layer_info = layer(
                    current_hidden_states,
                    attention_mask=attention_mask,
                    token_types=token_types
                )
                model_info['dna_layers_info'][layer_idx] = layer_info
            else:
                # Standard MEMOIR layer
                current_hidden_states, _ = layer(
                    current_hidden_states,
                    attention_mask=attention_mask
                )
            
            if return_layer_outputs:
                layer_outputs.append(current_hidden_states.clone())
        
        if return_layer_outputs:
            model_info['layer_outputs'] = layer_outputs
        
        return current_hidden_states, model_info
    
    def get_dna_efficiency_summary(self) -> Dict[str, Any]:
        """Get efficiency summary from all DNA layers."""
        summary = {
            'total_dna_layers': len(self.dna_layer_positions),
            'layer_efficiencies': {},
            'average_efficiency': 0.0,
            'total_compute_savings': 0.0
        }
        
        total_efficiency = 0.0
        for layer_idx in self.dna_layer_positions:
            layer = self.layers[layer_idx]
            if hasattr(layer, 'get_efficiency_metrics'):
                efficiency_metrics = layer.get_efficiency_metrics()
                summary['layer_efficiencies'][layer_idx] = efficiency_metrics
                
                # Extract compute savings if available
                if 'compute_savings' in efficiency_metrics:
                    total_efficiency += efficiency_metrics['compute_savings']
        
        if len(self.dna_layer_positions) > 0:
            summary['average_efficiency'] = total_efficiency / len(self.dna_layer_positions)
            summary['total_compute_savings'] = total_efficiency
        
        return summary


# Factory functions for easy model creation
def create_dna_enhanced_sam_model(
    dna_layer_position: int = 6,
    operation_mode: str = 'hybrid',
    hidden_size: int = 768
) -> DNAEnhancedSAMModel:
    """Create a DNA-enhanced SAM model with a single DNA layer."""
    dna_config = DNAConfigs.proof_of_concept()
    dna_config.hidden_size = hidden_size
    
    return DNAEnhancedSAMModel(
        dna_layer_positions=[dna_layer_position],
        dna_config=dna_config,
        operation_mode=operation_mode,
        hidden_size=hidden_size
    )


def create_multi_dna_sam_model(
    dna_layer_positions: List[int] = [4, 6, 8],
    operation_mode: str = 'dna',
    hidden_size: int = 768
) -> DNAEnhancedSAMModel:
    """Create a SAM model with multiple DNA layers."""
    dna_config = DNAConfigs.efficiency_focused()
    dna_config.hidden_size = hidden_size
    
    return DNAEnhancedSAMModel(
        dna_layer_positions=dna_layer_positions,
        dna_config=dna_config,
        operation_mode=operation_mode,
        hidden_size=hidden_size
    )
