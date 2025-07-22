"""
SAM Core Model Layers

Implementation of MEMOIR ResidualMemoryLayer and enhanced transformer blocks
for lifelong learning and knowledge editing capabilities.

Based on the MEMOIR paper: "Localized Model Editing for Lifelong Learning"
Author: SAM Development Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Tuple, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class ResidualMemoryLayer(nn.Module):
    """
    MEMOIR Residual Memory Layer for localized model editing.
    
    This layer implements the core memory mechanism from the MEMOIR paper,
    allowing for non-destructive knowledge updates through sparse residual
    connections that are activated based on input similarity.
    
    Key Features:
    - Zero-weight initialization (no effect before edits)
    - Sparse activation based on edit masks
    - Support for multiple concurrent edits
    - Gradient isolation for targeted updates
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        max_edits: int = 10000,
        sparsity_ratio: float = 0.01,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the ResidualMemoryLayer.
        
        Args:
            hidden_size: Dimension of the hidden states
            max_edits: Maximum number of edits this layer can store
            sparsity_ratio: Ratio of neurons to activate per edit (default 1%)
            device: Device to place the layer on
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.max_edits = max_edits
        self.sparsity_ratio = sparsity_ratio
        self.device = device or torch.device('cpu')
        
        # Core memory parameters - initialized to zero for no initial effect
        self.edit_weights = nn.Parameter(
            torch.zeros(max_edits, hidden_size, device=self.device),
            requires_grad=True
        )
        
        # Track which edit slots are active
        self.register_buffer(
            'edit_active', 
            torch.zeros(max_edits, dtype=torch.bool, device=self.device)
        )
        
        # Metadata for each edit
        self.edit_metadata = {}
        
        # Statistics
        self.register_buffer(
            'total_edits_made',
            torch.tensor(0, dtype=torch.long, device=self.device)
        )
        
        logger.info(f"ResidualMemoryLayer initialized: {hidden_size}D, max_edits={max_edits}")
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        edit_mask: Optional[torch.Tensor] = None,
        edit_id: Optional[str] = None
    ) -> torch.Tensor:
        """
        Forward pass through the residual memory layer.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            edit_mask: Sparse binary mask indicating which neurons to activate
            edit_id: Identifier for the specific edit to apply
            
        Returns:
            Residual output to be added to the original FFN output
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # If no edit mask provided, return zeros (no memory activation)
        if edit_mask is None:
            return torch.zeros_like(hidden_states)
        
        # Apply the edit mask to the hidden states
        masked_states = hidden_states * edit_mask.unsqueeze(0).unsqueeze(0)
        
        # If specific edit_id provided, use only that edit
        if edit_id is not None and edit_id in self.edit_metadata:
            edit_slot = self.edit_metadata[edit_id]['slot']
            if self.edit_active[edit_slot]:
                edit_weight = self.edit_weights[edit_slot]
                # Apply the edit: element-wise multiplication then sum
                residual = masked_states * edit_weight.unsqueeze(0).unsqueeze(0)
                return residual
        
        # Otherwise, apply all active edits (weighted combination)
        residual_output = torch.zeros_like(hidden_states)
        
        active_slots = torch.where(self.edit_active)[0]
        for slot in active_slots:
            edit_weight = self.edit_weights[slot]
            # Apply this edit's contribution
            edit_contribution = masked_states * edit_weight.unsqueeze(0).unsqueeze(0)
            residual_output += edit_contribution
        
        return residual_output
    
    def add_edit(
        self, 
        edit_id: str, 
        edit_mask: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add a new edit to the memory layer.
        
        Args:
            edit_id: Unique identifier for this edit
            edit_mask: Sparse binary mask for this edit
            metadata: Optional metadata about the edit
            
        Returns:
            Slot number where the edit was stored
        """
        # Find an available slot
        available_slots = torch.where(~self.edit_active)[0]
        
        if len(available_slots) == 0:
            raise RuntimeError(f"No available edit slots (max: {self.max_edits})")
        
        slot = available_slots[0].item()
        
        # Store the edit metadata
        self.edit_metadata[edit_id] = {
            'slot': slot,
            'mask': edit_mask.clone(),
            'created_at': datetime.now(),
            'metadata': metadata or {}
        }
        
        # Mark slot as active
        self.edit_active[slot] = True
        self.total_edits_made += 1
        
        logger.info(f"Added edit '{edit_id}' to slot {slot}")
        return slot
    
    def remove_edit(self, edit_id: str) -> bool:
        """
        Remove an edit from the memory layer.
        
        Args:
            edit_id: Identifier of the edit to remove
            
        Returns:
            True if edit was found and removed, False otherwise
        """
        if edit_id not in self.edit_metadata:
            return False
        
        slot = self.edit_metadata[edit_id]['slot']
        
        # Clear the slot
        self.edit_active[slot] = False
        self.edit_weights[slot].data.zero_()
        
        # Remove metadata
        del self.edit_metadata[edit_id]
        
        logger.info(f"Removed edit '{edit_id}' from slot {slot}")
        return True
    
    def get_edit_info(self) -> Dict[str, Any]:
        """Get information about current edits."""
        active_count = self.edit_active.sum().item()
        return {
            'total_edits_made': self.total_edits_made.item(),
            'active_edits': active_count,
            'available_slots': self.max_edits - active_count,
            'edit_ids': list(self.edit_metadata.keys()),
            'memory_usage_mb': self.edit_weights.numel() * 4 / (1024 * 1024)  # Assuming float32
        }
    
    def consolidate_edits(self, similarity_threshold: float = 0.8) -> int:
        """
        Consolidate similar edits to free up memory slots.
        
        Args:
            similarity_threshold: Minimum similarity to merge edits
            
        Returns:
            Number of edits consolidated
        """
        # This is a placeholder for future implementation
        # Would involve comparing edit masks and merging similar ones
        logger.info("Edit consolidation not yet implemented")
        return 0


class MEMOIRTransformerBlock(nn.Module):
    """
    Enhanced transformer block with MEMOIR ResidualMemoryLayer integration.
    
    This block extends the standard transformer architecture to include
    the MEMOIR memory mechanism for lifelong learning capabilities.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
        max_edits: int = 10000,
        enable_memoir: bool = True
    ):
        """
        Initialize the MEMOIR-enhanced transformer block.
        
        Args:
            hidden_size: Hidden dimension size
            num_attention_heads: Number of attention heads
            intermediate_size: FFN intermediate dimension
            dropout: Dropout probability
            activation: Activation function name
            max_edits: Maximum edits for MEMOIR layer
            enable_memoir: Whether to enable MEMOIR functionality
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.enable_memoir = enable_memoir
        
        # Standard transformer components
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size)
        )
        
        # MEMOIR ResidualMemoryLayer
        if enable_memoir:
            self.residual_memory = ResidualMemoryLayer(
                hidden_size=hidden_size,
                max_edits=max_edits
            )
            logger.info("MEMOIR ResidualMemoryLayer enabled in transformer block")
        else:
            self.residual_memory = None
            logger.info("MEMOIR ResidualMemoryLayer disabled")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        edit_mask: Optional[torch.Tensor] = None,
        edit_id: Optional[str] = None,
        return_attention_weights: bool = False,
        enable_memoir_retrieval: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Enhanced forward pass with MEMOIR retrieval logic.

        This implements the "Read" operation from the MEMOIR paper:
        1. Generate query mask from input activations
        2. Search EditMaskDatabase for similar masks
        3. Apply retrieved edit if similarity above threshold
        4. Combine with standard transformer computation

        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask
            edit_mask: MEMOIR edit mask (optional override)
            edit_id: Specific edit to apply (optional override)
            return_attention_weights: Whether to return attention weights
            enable_memoir_retrieval: Whether to enable automatic retrieval

        Returns:
            Tuple of (output_states, attention_weights)
        """
        # Self-attention with residual connection
        # Convert attention mask format for PyTorch MultiheadAttention
        # attention_mask should be [batch_size, seq_len] with 1 for valid tokens, 0 for padding
        # key_padding_mask should be [batch_size, seq_len] with True for padding tokens
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)  # True for padding tokens

        attn_output, attn_weights = self.attention(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention_weights
        )
        hidden_states = self.norm1(hidden_states + attn_output)

        # Feed-forward with residual connection
        ffn_output = self.ffn(hidden_states)

        # MEMOIR retrieval and memory contribution
        memory_output = torch.zeros_like(ffn_output)
        retrieved_edit_info = None

        if self.residual_memory is not None and self.enable_memoir:
            # MEMOIR "Read" operation
            if enable_memoir_retrieval and edit_mask is None and edit_id is None:
                # Automatic retrieval: generate query mask and search database
                retrieved_edit_info = self._retrieve_relevant_edit(hidden_states)
                if retrieved_edit_info:
                    edit_mask = retrieved_edit_info['mask']
                    edit_id = retrieved_edit_info['edit_id']

            # Apply memory layer with retrieved or provided edit
            if edit_mask is not None or edit_id is not None:
                memory_output = self.residual_memory(
                    hidden_states,
                    edit_mask=edit_mask,
                    edit_id=edit_id
                )

        # Combine FFN output with memory output
        combined_output = ffn_output + memory_output

        # Final layer norm and residual
        output_states = self.norm2(hidden_states + combined_output)

        # Store retrieval info for debugging/monitoring
        if retrieved_edit_info:
            if not hasattr(self, '_last_retrieval_info'):
                self._last_retrieval_info = {}
            self._last_retrieval_info = retrieved_edit_info

        if return_attention_weights:
            return output_states, attn_weights
        else:
            return output_states, None

    def _retrieve_relevant_edit(self, hidden_states: torch.Tensor) -> Optional[Dict[str, Any]]:
        """
        Retrieve relevant edit from the database based on current activations.

        This implements the core MEMOIR retrieval logic:
        1. Generate query mask from current activations
        2. Search EditMaskDatabase for similar masks
        3. Return edit info if similarity above threshold

        Args:
            hidden_states: Current hidden states to generate query from

        Returns:
            Dictionary with edit info if found, None otherwise
        """
        try:
            # Import here to avoid circular imports
            from ..fingerprinter import TopHashFingerprinter
            from ...memory.edit_mask_db import EditMaskDatabase

            # Get or create fingerprinter
            if not hasattr(self, '_fingerprinter'):
                self._fingerprinter = TopHashFingerprinter(
                    hidden_size=self.hidden_size,
                    top_k=max(50, self.hidden_size // 100),
                    permutation_file="data/memoir_permutation.pkl"
                )

            # Get or create database
            if not hasattr(self, '_mask_database'):
                self._mask_database = EditMaskDatabase(
                    hidden_size=self.hidden_size,
                    storage_dir="data/memoir_masks"
                )

            # Generate query mask from current activations
            # Use the last token's activations as the query
            query_activations = hidden_states[:, -1, :].mean(dim=0)  # Average across batch
            query_mask = self._fingerprinter.generate_mask(query_activations)

            # Search for similar edit in database
            similarity_threshold = getattr(self, 'retrieval_threshold', 0.7)
            result = self._mask_database.find_closest(query_mask, threshold=similarity_threshold)

            if result is not None:
                edit_id, retrieved_mask, similarity = result

                logger.debug(f"Retrieved edit {edit_id} with similarity {similarity:.3f}")

                return {
                    'edit_id': edit_id,
                    'mask': retrieved_mask,
                    'similarity': similarity,
                    'query_mask': query_mask,
                    'retrieval_timestamp': datetime.now()
                }

            return None

        except Exception as e:
            logger.warning(f"Error during MEMOIR retrieval: {e}")
            return None
    
    def get_memory_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the MEMOIR memory layer."""
        if self.residual_memory is not None:
            return self.residual_memory.get_edit_info()
        return None
    
    def add_memory_edit(
        self, 
        edit_id: str, 
        edit_mask: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[int]:
        """Add an edit to the memory layer."""
        if self.residual_memory is not None:
            return self.residual_memory.add_edit(edit_id, edit_mask, metadata)
        return None
    
    def remove_memory_edit(self, edit_id: str) -> bool:
        """Remove an edit from the memory layer."""
        if self.residual_memory is not None:
            return self.residual_memory.remove_edit(edit_id)
        return False

    def set_retrieval_threshold(self, threshold: float):
        """Set the similarity threshold for automatic edit retrieval."""
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self.retrieval_threshold = threshold
        logger.info(f"MEMOIR retrieval threshold set to {threshold}")

    def get_last_retrieval_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the last edit retrieval."""
        return getattr(self, '_last_retrieval_info', None)

    def clear_retrieval_cache(self):
        """Clear the retrieval cache and force reinitialization."""
        if hasattr(self, '_fingerprinter'):
            self._fingerprinter.clear_cache()
        if hasattr(self, '_mask_database'):
            self._mask_database.search_cache.clear()
        logger.info("MEMOIR retrieval cache cleared")

    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """Get statistics about MEMOIR retrieval operations."""
        stats = {
            'retrieval_threshold': getattr(self, 'retrieval_threshold', 0.7),
            'last_retrieval': self.get_last_retrieval_info(),
            'memory_info': self.get_memory_info()
        }

        if hasattr(self, '_fingerprinter'):
            stats['fingerprinter_stats'] = self._fingerprinter.get_statistics()

        if hasattr(self, '_mask_database'):
            stats['database_stats'] = self._mask_database.get_statistics()

        return stats

    def force_edit_retrieval(self, query_text: str) -> Optional[Dict[str, Any]]:
        """
        Force retrieval of an edit based on query text.

        Args:
            query_text: Text to generate query activations from

        Returns:
            Retrieved edit info if found
        """
        try:
            # Generate synthetic activations from text
            # In real implementation, this would use actual model encoding
            torch.manual_seed(hash(query_text) % 2**32)
            synthetic_activations = torch.randn(1, 1, self.hidden_size)

            return self._retrieve_relevant_edit(synthetic_activations)

        except Exception as e:
            logger.error(f"Error in forced edit retrieval: {e}")
            return None
