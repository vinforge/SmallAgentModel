"""
Dissonance Monitor Module for SAM
Phase 5B - Cognitive Dissonance Detection

This module provides real-time cognitive dissonance monitoring capabilities
by analyzing logit distributions to detect internal reasoning conflicts.
"""

import torch
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class DissonanceCalculationMode(Enum):
    """Different methods for calculating dissonance."""
    ENTROPY = "entropy"
    VARIANCE = "variance"
    KL_DIVERGENCE = "kl_divergence"
    COMPOSITE = "composite"

@dataclass
class DissonanceScore:
    """Container for dissonance calculation results."""
    score: float
    calculation_mode: DissonanceCalculationMode
    metadata: Dict[str, Any]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'score': self.score,
            'calculation_mode': self.calculation_mode.value,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }

class DissonanceMonitor:
    """
    Real-time cognitive dissonance monitoring system.
    
    Calculates dissonance scores from model logits to detect internal
    reasoning conflicts and uncertainty patterns.
    """
    
    def __init__(self, 
                 vocab_size: int,
                 calculation_mode: DissonanceCalculationMode = DissonanceCalculationMode.ENTROPY,
                 fallback_mode: bool = True,
                 enable_profiling: bool = False,
                 device: Optional[str] = None):
        """
        Initialize the DissonanceMonitor.
        
        Args:
            vocab_size: Size of the model vocabulary
            calculation_mode: Method for calculating dissonance
            fallback_mode: Whether to return 0.0 on errors instead of raising
            enable_profiling: Whether to track performance metrics
            device: Device to use for calculations (auto-detected if None)
        """
        self.vocab_size = vocab_size
        self.calculation_mode = calculation_mode
        self.fallback_mode = fallback_mode
        self.enable_profiling = enable_profiling
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Pre-calculate maximum entropy for normalization
        self.max_entropy = torch.log(torch.tensor(vocab_size, dtype=torch.float32, device=self.device))
        
        # Performance tracking
        self.calculation_times: List[float] = []
        self.total_calculations = 0
        self.failed_calculations = 0
        
        # Configuration
        self.config = {
            'entropy_epsilon': 1e-9,  # Small value to prevent log(0)
            'variance_threshold': 0.1,  # Threshold for variance-based detection
            'composite_weights': {
                'entropy': 0.6,
                'variance': 0.4
            }
        }
        
        logger.info(f"DissonanceMonitor initialized: vocab_size={vocab_size}, "
                   f"mode={calculation_mode.value}, device={self.device}")
    
    def calculate_dissonance(self, logits: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> DissonanceScore:
        """
        Calculate dissonance score from logits.
        
        Args:
            logits: Model logits tensor [vocab_size] or [batch_size, vocab_size]
            context: Optional context information for adaptive scoring
            
        Returns:
            DissonanceScore object with calculated score and metadata
        """
        start_time = time.time() if self.enable_profiling else None
        self.total_calculations += 1
        
        try:
            # Input validation and preprocessing
            processed_logits = self._preprocess_logits(logits)
            if processed_logits is None:
                return self._create_fallback_score("Invalid logits input")
            
            # Calculate dissonance based on mode
            if self.calculation_mode == DissonanceCalculationMode.ENTROPY:
                score, metadata = self._calculate_entropy_dissonance(processed_logits)
            elif self.calculation_mode == DissonanceCalculationMode.VARIANCE:
                score, metadata = self._calculate_variance_dissonance(processed_logits)
            elif self.calculation_mode == DissonanceCalculationMode.KL_DIVERGENCE:
                score, metadata = self._calculate_kl_dissonance(processed_logits, context)
            elif self.calculation_mode == DissonanceCalculationMode.COMPOSITE:
                score, metadata = self._calculate_composite_dissonance(processed_logits, context)
            else:
                raise ValueError(f"Unknown calculation mode: {self.calculation_mode}")
            
            # Apply context-aware adjustments if available
            if context:
                score = self._apply_contextual_adjustments(score, context)
                metadata['context_adjusted'] = True
            
            # Performance tracking
            if self.enable_profiling and start_time:
                calculation_time = time.time() - start_time
                self.calculation_times.append(calculation_time)
                metadata['calculation_time'] = calculation_time
            
            return DissonanceScore(
                score=score,
                calculation_mode=self.calculation_mode,
                metadata=metadata,
                timestamp=time.time()
            )
            
        except Exception as e:
            self.failed_calculations += 1
            logger.error(f"Dissonance calculation failed: {e}")
            
            if self.fallback_mode:
                return self._create_fallback_score(f"Calculation error: {e}")
            else:
                raise
    
    def _preprocess_logits(self, logits: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Preprocess and validate logits tensor.
        
        Args:
            logits: Raw logits tensor
            
        Returns:
            Processed logits tensor or None if invalid
        """
        try:
            # Type and dimension validation
            if not isinstance(logits, torch.Tensor):
                logger.warning("Logits must be a torch.Tensor")
                return None
            
            if logits.dim() == 0:
                logger.warning("Logits tensor has no dimensions")
                return None
            
            # Handle batch dimension
            if logits.dim() == 1:
                # Single sequence: [vocab_size]
                processed = logits.unsqueeze(0)  # [1, vocab_size]
            elif logits.dim() == 2:
                # Batch: [batch_size, vocab_size]
                processed = logits
            else:
                logger.warning(f"Unexpected logits dimensions: {logits.shape}")
                return None
            
            # Device compatibility
            if processed.device != self.device:
                processed = processed.to(self.device)
            
            # Ensure float type
            if not processed.dtype.is_floating_point:
                processed = processed.float()
            
            return processed
            
        except Exception as e:
            logger.error(f"Logits preprocessing failed: {e}")
            return None
    
    def _calculate_entropy_dissonance(self, logits: torch.Tensor) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate dissonance using entropy of probability distribution.

        High entropy (uniform distribution) = high dissonance (uncertainty)
        Low entropy (peaked distribution) = low dissonance (certainty)

        Args:
            logits: Preprocessed logits tensor [batch_size, vocab_size]

        Returns:
            Tuple of (normalized_score, metadata)
        """
        # Convert logits to probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        # Calculate entropy: H = -sum(p * log(p))
        log_probs = torch.log(probabilities + self.config['entropy_epsilon'])
        entropy = -torch.sum(probabilities * log_probs, dim=-1)

        # Take mean across batch if multiple sequences
        mean_entropy = torch.mean(entropy)

        # Normalize to 0-1 range using maximum possible entropy
        # High entropy (close to max) -> high dissonance (close to 1.0)
        # Low entropy (close to 0) -> low dissonance (close to 0.0)
        normalized_score = (mean_entropy / self.max_entropy).item()

        # Clamp to valid range
        normalized_score = max(0.0, min(1.0, normalized_score))

        metadata = {
            'raw_entropy': mean_entropy.item(),
            'max_entropy': self.max_entropy.item(),
            'batch_size': logits.shape[0],
            'vocab_size': logits.shape[1],
            'probability_max': torch.max(probabilities).item(),
            'probability_min': torch.min(probabilities).item()
        }

        return normalized_score, metadata
    
    def _calculate_variance_dissonance(self, logits: torch.Tensor) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate dissonance using variance of probability distribution.
        
        Args:
            logits: Preprocessed logits tensor [batch_size, vocab_size]
            
        Returns:
            Tuple of (normalized_score, metadata)
        """
        # Convert logits to probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Calculate variance across vocabulary dimension
        variance = torch.var(probabilities, dim=-1)
        
        # Take mean across batch
        mean_variance = torch.mean(variance)
        
        # Normalize variance (theoretical max variance for uniform distribution)
        # For uniform distribution: var = (1/n) * (1 - 1/n) ≈ 1/n for large n
        max_variance = 1.0 / self.vocab_size
        normalized_score = (mean_variance / max_variance).item()
        
        # Clamp to valid range
        normalized_score = max(0.0, min(1.0, normalized_score))
        
        metadata = {
            'raw_variance': mean_variance.item(),
            'max_variance': max_variance,
            'batch_size': logits.shape[0]
        }
        
        return normalized_score, metadata
    
    def _calculate_kl_dissonance(self, logits: torch.Tensor, context: Optional[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate dissonance using KL divergence from uniform distribution.
        
        Args:
            logits: Preprocessed logits tensor [batch_size, vocab_size]
            context: Context information (unused in basic implementation)
            
        Returns:
            Tuple of (normalized_score, metadata)
        """
        # Convert logits to probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Create uniform distribution as reference
        uniform_probs = torch.ones_like(probabilities) / self.vocab_size
        
        # Calculate KL divergence: KL(P||Q) = sum(P * log(P/Q))
        kl_div = torch.nn.functional.kl_div(
            torch.log(uniform_probs + self.config['entropy_epsilon']),
            probabilities,
            reduction='batchmean'
        )
        
        # Normalize by maximum possible KL divergence
        max_kl = torch.log(torch.tensor(self.vocab_size, dtype=torch.float32, device=self.device))
        normalized_score = (kl_div / max_kl).item()
        
        # Clamp to valid range
        normalized_score = max(0.0, min(1.0, normalized_score))
        
        metadata = {
            'raw_kl_divergence': kl_div.item(),
            'max_kl_divergence': max_kl.item(),
            'batch_size': logits.shape[0]
        }
        
        return normalized_score, metadata
    
    def _calculate_composite_dissonance(self, logits: torch.Tensor, context: Optional[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate composite dissonance using multiple metrics.
        
        Args:
            logits: Preprocessed logits tensor [batch_size, vocab_size]
            context: Context information for weighting
            
        Returns:
            Tuple of (composite_score, metadata)
        """
        # Calculate individual scores
        entropy_score, entropy_meta = self._calculate_entropy_dissonance(logits)
        variance_score, variance_meta = self._calculate_variance_dissonance(logits)
        
        # Get weights from config
        weights = self.config['composite_weights']
        
        # Calculate weighted composite score
        composite_score = (weights['entropy'] * entropy_score + 
                          weights['variance'] * variance_score)
        
        metadata = {
            'entropy_score': entropy_score,
            'variance_score': variance_score,
            'weights': weights,
            'entropy_metadata': entropy_meta,
            'variance_metadata': variance_meta
        }
        
        return composite_score, metadata
    
    def _apply_contextual_adjustments(self, score: float, context: Dict[str, Any]) -> float:
        """
        Apply context-aware adjustments to dissonance score.
        
        Args:
            score: Base dissonance score
            context: Context information
            
        Returns:
            Adjusted dissonance score
        """
        # Example contextual adjustments (can be extended)
        adjusted_score = score
        
        # Adjust based on query complexity
        if 'query_complexity' in context:
            complexity = context['query_complexity']
            if complexity == 'high':
                adjusted_score *= 0.9  # Slightly lower threshold for complex queries
            elif complexity == 'low':
                adjusted_score *= 1.1  # Slightly higher threshold for simple queries
        
        # Adjust based on domain
        if 'domain' in context:
            domain = context['domain']
            if domain in ['technical', 'scientific']:
                adjusted_score *= 0.95  # Lower threshold for technical domains
            elif domain in ['casual', 'general']:
                adjusted_score *= 1.05  # Higher threshold for casual domains
        
        return max(0.0, min(1.0, adjusted_score))
    
    def _create_fallback_score(self, reason: str) -> DissonanceScore:
        """
        Create a fallback dissonance score when calculation fails.
        
        Args:
            reason: Reason for fallback
            
        Returns:
            DissonanceScore with score 0.0 and error metadata
        """
        return DissonanceScore(
            score=0.0,
            calculation_mode=self.calculation_mode,
            metadata={
                'fallback': True,
                'reason': reason,
                'error': True
            },
            timestamp=time.time()
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the dissonance monitor.
        
        Returns:
            Dictionary of performance metrics
        """
        stats = {
            'total_calculations': self.total_calculations,
            'failed_calculations': self.failed_calculations,
            'success_rate': ((self.total_calculations - self.failed_calculations) / 
                           max(1, self.total_calculations)),
            'calculation_mode': self.calculation_mode.value,
            'device': str(self.device),
            'vocab_size': self.vocab_size
        }
        
        if self.enable_profiling and self.calculation_times:
            times = np.array(self.calculation_times)
            stats.update({
                'avg_calculation_time': float(np.mean(times)),
                'min_calculation_time': float(np.min(times)),
                'max_calculation_time': float(np.max(times)),
                'total_calculation_time': float(np.sum(times))
            })
        
        return stats
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.calculation_times.clear()
        self.total_calculations = 0
        self.failed_calculations = 0
        logger.info("DissonanceMonitor statistics reset")
    
    def update_config(self, new_config: Dict[str, Any]):
        """
        Update configuration parameters.
        
        Args:
            new_config: Dictionary of configuration updates
        """
        self.config.update(new_config)
        logger.info(f"DissonanceMonitor configuration updated: {new_config}")
