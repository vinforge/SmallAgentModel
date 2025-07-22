"""
SAM Core Fingerprinter

Implementation of TopHashFingerprinter for MEMOIR framework.
Generates deterministic sparse masks based on activation patterns.

Based on the MEMOIR paper: "Localized Model Editing for Lifelong Learning"
Author: SAM Development Team
Version: 1.0.0
"""

import torch
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple
import hashlib
import pickle

logger = logging.getLogger(__name__)

class TopHashFingerprinter:
    """
    TopHash Fingerprinter for generating deterministic sparse masks.
    
    This class implements the fingerprinting mechanism from the MEMOIR paper,
    which creates sparse binary masks based on the top-k highest-magnitude
    activations, hashed through a fixed permutation matrix.
    
    Key Features:
    - Deterministic mask generation (same input → same mask)
    - Configurable sparsity (top-k selection)
    - Fixed permutation matrix for consistent hashing
    - Support for different activation patterns
    """
    
    def __init__(
        self,
        hidden_size: int,
        top_k: int = 512,
        permutation_file: Optional[str] = None,
        seed: int = 42
    ):
        """
        Initialize the TopHashFingerprinter.
        
        Args:
            hidden_size: Dimension of the activation vectors
            top_k: Number of top activations to select for the mask
            permutation_file: Path to saved permutation matrix (optional)
            seed: Random seed for permutation generation
        """
        self.hidden_size = hidden_size
        self.top_k = min(top_k, hidden_size)  # Ensure top_k doesn't exceed hidden_size
        self.seed = seed
        
        # Load or generate permutation matrix
        self.permutation_matrix = self._load_or_generate_permutation(permutation_file)
        
        # Statistics
        self.masks_generated = 0
        self.cache = {}  # Simple cache for recently generated masks
        self.cache_size = 1000
        
        logger.info(f"TopHashFingerprinter initialized: {hidden_size}D, top_k={top_k}")
    
    def _load_or_generate_permutation(self, permutation_file: Optional[str]) -> np.ndarray:
        """
        Load existing permutation matrix or generate a new one.
        
        Args:
            permutation_file: Path to permutation file
            
        Returns:
            Permutation matrix as numpy array
        """
        if permutation_file and Path(permutation_file).exists():
            try:
                with open(permutation_file, 'rb') as f:
                    permutation_data = pickle.load(f)
                
                # Validate the loaded permutation
                if (permutation_data['hidden_size'] == self.hidden_size and
                    permutation_data['seed'] == self.seed):
                    logger.info(f"Loaded permutation matrix from {permutation_file}")
                    return permutation_data['permutation']
                else:
                    logger.warning("Loaded permutation doesn't match current config, generating new one")
            except Exception as e:
                logger.warning(f"Failed to load permutation from {permutation_file}: {e}")
        
        # Generate new permutation matrix
        logger.info("Generating new permutation matrix")
        np.random.seed(self.seed)
        permutation = np.random.permutation(self.hidden_size)
        
        # Save the permutation if file path provided
        if permutation_file:
            try:
                Path(permutation_file).parent.mkdir(parents=True, exist_ok=True)
                permutation_data = {
                    'hidden_size': self.hidden_size,
                    'seed': self.seed,
                    'permutation': permutation,
                    'created_at': np.datetime64('now')
                }
                with open(permutation_file, 'wb') as f:
                    pickle.dump(permutation_data, f)
                logger.info(f"Saved permutation matrix to {permutation_file}")
            except Exception as e:
                logger.warning(f"Failed to save permutation to {permutation_file}: {e}")
        
        return permutation
    
    def generate_mask(
        self, 
        activations: torch.Tensor,
        use_cache: bool = True
    ) -> torch.Tensor:
        """
        Generate a sparse binary mask based on input activations.
        
        Args:
            activations: Input activation tensor [batch_size, seq_len, hidden_size] or [hidden_size]
            use_cache: Whether to use caching for performance
            
        Returns:
            Sparse binary mask tensor of same shape as activations
        """
        # Handle different input shapes
        original_shape = activations.shape
        if len(original_shape) == 3:
            # [batch_size, seq_len, hidden_size] -> use last token
            flat_activations = activations[:, -1, :].flatten()
        elif len(original_shape) == 2:
            # [seq_len, hidden_size] -> use last token
            flat_activations = activations[-1, :].flatten()
        elif len(original_shape) == 1:
            # [hidden_size] -> use as is
            flat_activations = activations.flatten()
        else:
            raise ValueError(f"Unsupported activation shape: {original_shape}")
        
        # Ensure we have the right dimension
        if flat_activations.shape[0] != self.hidden_size:
            raise ValueError(f"Activation size {flat_activations.shape[0]} doesn't match hidden_size {self.hidden_size}")
        
        # Check cache first
        if use_cache:
            activation_hash = self._hash_activations(flat_activations)
            if activation_hash in self.cache:
                cached_mask = self.cache[activation_hash]
                return self._reshape_mask(cached_mask, original_shape)
        
        # Convert to numpy for processing
        activations_np = flat_activations.detach().cpu().numpy()
        
        # Find top-k highest magnitude activations
        magnitude = np.abs(activations_np)
        top_k_indices = np.argpartition(magnitude, -self.top_k)[-self.top_k:]
        
        # Apply permutation (the "hashing" step)
        permuted_indices = self.permutation_matrix[top_k_indices]
        
        # Create sparse binary mask
        mask = np.zeros(self.hidden_size, dtype=np.float32)
        mask[permuted_indices] = 1.0
        
        # Convert back to torch tensor
        mask_tensor = torch.from_numpy(mask).to(activations.device)
        
        # Cache the result
        if use_cache:
            self._update_cache(activation_hash, mask_tensor.clone())
        
        self.masks_generated += 1
        
        # Reshape to match original input shape
        return self._reshape_mask(mask_tensor, original_shape)
    
    def _hash_activations(self, activations: torch.Tensor) -> str:
        """
        Generate a hash of the activations for caching.
        
        Args:
            activations: Flattened activation tensor
            
        Returns:
            Hash string
        """
        # Use a subset of activations for hashing (for performance)
        sample_size = min(100, len(activations))
        sample_indices = torch.linspace(0, len(activations)-1, sample_size, dtype=torch.long)
        sample_activations = activations[sample_indices]
        
        # Convert to bytes and hash
        activation_bytes = sample_activations.detach().cpu().numpy().tobytes()
        return hashlib.md5(activation_bytes).hexdigest()
    
    def _update_cache(self, activation_hash: str, mask: torch.Tensor):
        """Update the mask cache."""
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[activation_hash] = mask
    
    def _reshape_mask(self, mask: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Reshape the flat mask to match the target shape.
        
        Args:
            mask: Flat mask tensor [hidden_size]
            target_shape: Target shape to match
            
        Returns:
            Reshaped mask tensor
        """
        if len(target_shape) == 1:
            return mask
        elif len(target_shape) == 2:
            # [seq_len, hidden_size]
            seq_len = target_shape[0]
            return mask.unsqueeze(0).expand(seq_len, -1)
        elif len(target_shape) == 3:
            # [batch_size, seq_len, hidden_size]
            batch_size, seq_len = target_shape[0], target_shape[1]
            return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        else:
            raise ValueError(f"Cannot reshape mask to target shape: {target_shape}")
    
    def get_statistics(self) -> dict:
        """Get fingerprinter statistics."""
        return {
            'masks_generated': self.masks_generated,
            'cache_size': len(self.cache),
            'cache_hit_ratio': getattr(self, '_cache_hits', 0) / max(self.masks_generated, 1),
            'hidden_size': self.hidden_size,
            'top_k': self.top_k,
            'sparsity_ratio': self.top_k / self.hidden_size
        }
    
    def clear_cache(self):
        """Clear the mask cache."""
        self.cache.clear()
        logger.info("Fingerprinter cache cleared")
    
    def validate_determinism(self, test_activations: torch.Tensor, num_tests: int = 10) -> bool:
        """
        Validate that the fingerprinter produces deterministic results.
        
        Args:
            test_activations: Test activation tensor
            num_tests: Number of tests to run
            
        Returns:
            True if all tests produce identical masks
        """
        logger.info(f"Running determinism validation with {num_tests} tests")
        
        # Generate first mask
        first_mask = self.generate_mask(test_activations, use_cache=False)
        
        # Test multiple times
        for i in range(num_tests - 1):
            test_mask = self.generate_mask(test_activations, use_cache=False)
            if not torch.equal(first_mask, test_mask):
                logger.error(f"Determinism test failed at iteration {i+1}")
                return False
        
        logger.info("Determinism validation passed")
        return True
    
    def analyze_mask_properties(self, mask: torch.Tensor) -> dict:
        """
        Analyze properties of a generated mask.
        
        Args:
            mask: Generated mask tensor
            
        Returns:
            Dictionary of mask properties
        """
        flat_mask = mask.flatten()
        
        return {
            'sparsity': (flat_mask == 0).float().mean().item(),
            'density': (flat_mask != 0).float().mean().item(),
            'num_active': (flat_mask != 0).sum().item(),
            'num_total': flat_mask.numel(),
            'mask_sum': flat_mask.sum().item(),
            'mask_mean': flat_mask.mean().item(),
            'mask_std': flat_mask.std().item()
        }
