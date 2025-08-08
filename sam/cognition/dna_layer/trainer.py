"""
DNA Layer Training System
=========================

Training and fine-tuning system for DNA layer routing optimization.
Implements specialized training procedures for efficient routing learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import logging
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
import json
from dataclasses import dataclass

from .dynamic_layer import DNALayer
from .sam_integration import DNAEnhancedSAMModel
from .training_data import TrainingExample, SAMWorkloadGenerator
from .metrics import DNAMetrics
from .config import DNAConfig, DNAConfigs

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for DNA layer training."""
    learning_rate: float = 1e-4
    batch_size: int = 8
    num_epochs: int = 10
    warmup_steps: int = 100
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    
    # DNA-specific training parameters
    routing_loss_weight: float = 1.0
    efficiency_loss_weight: float = 0.5
    specialization_loss_weight: float = 0.3
    load_balance_weight: float = 0.1
    
    # Training schedule
    save_every_n_epochs: int = 2
    validate_every_n_steps: int = 50
    early_stopping_patience: int = 5
    
    # Target metrics
    target_efficiency: float = 0.25  # 25% identity usage target
    target_routing_entropy: float = 1.2  # Minimum routing diversity


class DNADataset(Dataset):
    """PyTorch dataset for DNA layer training."""
    
    def __init__(self, examples: List[TrainingExample]):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            'hidden_states': example.hidden_states.squeeze(0),  # Remove batch dim
            'attention_mask': example.attention_mask,
            'token_types': example.token_types,
            'complexity_score': torch.tensor(example.complexity_score),
            'scenario_type': example.scenario_type,
            'expected_routing': example.expected_routing
        }


class DNATrainer:
    """
    Trainer for DNA layer optimization.
    
    Implements specialized training procedures for routing efficiency,
    expert specialization, and load balancing.
    """
    
    def __init__(
        self,
        model: DNAEnhancedSAMModel,
        config: TrainingConfig,
        device: str = 'cpu'
    ):
        self.model = model
        self.config = config
        self.device = device
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs * 100,  # Approximate steps per epoch
            eta_min=config.learning_rate * 0.1
        )
        
        # Training metrics
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'efficiency_scores': [],
            'routing_entropy': [],
            'specialization_scores': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_efficiency = 0.0
        self.patience_counter = 0
        
        logger.info(f"DNA Trainer initialized on {device}")
    
    def train(
        self,
        train_dataset: List[TrainingExample],
        val_dataset: List[TrainingExample],
        save_dir: str = "checkpoints/dna_training"
    ) -> Dict[str, Any]:
        """
        Train the DNA layer with the given datasets.
        
        Args:
            train_dataset: Training examples
            val_dataset: Validation examples
            save_dir: Directory to save checkpoints
            
        Returns:
            Training results and metrics
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Create data loaders
        train_loader = DataLoader(
            DNADataset(train_dataset),
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        val_loader = DataLoader(
            DNADataset(val_dataset),
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )
        
        logger.info(f"Starting DNA layer training")
        logger.info(f"  - Training examples: {len(train_dataset)}")
        logger.info(f"  - Validation examples: {len(val_dataset)}")
        logger.info(f"  - Epochs: {self.config.num_epochs}")
        logger.info(f"  - Batch size: {self.config.batch_size}")
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            
            # Training phase
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # Validation phase
            val_metrics = self._validate_epoch(val_loader, epoch)
            
            # Update training history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['efficiency_scores'].append(val_metrics['efficiency'])
            self.training_history['routing_entropy'].append(val_metrics['routing_entropy'])
            self.training_history['specialization_scores'].append(val_metrics['specialization'])
            
            epoch_time = time.time() - epoch_start
            
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} ({epoch_time:.2f}s)")
            logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
            logger.info(f"  Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"  Efficiency: {val_metrics['efficiency']:.1%}")
            logger.info(f"  Routing Entropy: {val_metrics['routing_entropy']:.3f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                checkpoint_path = save_path / f"checkpoint_epoch_{epoch+1}.pt"
                self._save_checkpoint(checkpoint_path, epoch, val_metrics)
            
            # Early stopping check
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_efficiency = val_metrics['efficiency']
                self.patience_counter = 0
                
                # Save best model
                best_model_path = save_path / "best_model.pt"
                self._save_checkpoint(best_model_path, epoch, val_metrics)
            else:
                self.patience_counter += 1
                
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        total_time = time.time() - start_time
        
        # Final results
        results = {
            'total_training_time': total_time,
            'best_val_loss': self.best_val_loss,
            'best_efficiency': self.best_efficiency,
            'final_metrics': val_metrics,
            'training_history': self.training_history,
            'epochs_completed': epoch + 1
        }
        
        # Save training results
        results_path = save_path / "training_results.json"
        self._save_results(results, results_path)
        
        logger.info(f"Training completed in {total_time:.2f}s")
        logger.info(f"  Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"  Best efficiency: {self.best_efficiency:.1%}")
        
        return results
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            
            output_states, model_info = self.model(
                hidden_states=batch['hidden_states'],
                attention_mask=batch['attention_mask'],
                token_types=batch['token_types']
            )
            
            # Compute loss
            loss = self._compute_training_loss(batch, model_info)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.debug(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        return {'loss': total_loss / num_batches}
    
    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        efficiency_scores = []
        routing_entropies = []
        specialization_scores = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                output_states, model_info = self.model(
                    hidden_states=batch['hidden_states'],
                    attention_mask=batch['attention_mask'],
                    token_types=batch['token_types']
                )
                
                # Compute loss
                loss = self._compute_training_loss(batch, model_info)
                total_loss += loss.item()
                
                # Compute metrics
                metrics = self._compute_validation_metrics(model_info)
                efficiency_scores.append(metrics['efficiency'])
                routing_entropies.append(metrics['routing_entropy'])
                specialization_scores.append(metrics['specialization'])
        
        return {
            'loss': total_loss / len(val_loader),
            'efficiency': np.mean(efficiency_scores),
            'routing_entropy': np.mean(routing_entropies),
            'specialization': np.mean(specialization_scores)
        }
    
    def _compute_training_loss(self, batch: Dict, model_info: Dict) -> torch.Tensor:
        """Compute training loss with multiple components."""
        total_loss = torch.tensor(0.0, device=self.device)
        
        # Extract DNA layer info
        dna_layers_info = model_info.get('dna_layers_info', {})
        
        for layer_idx, layer_info in dna_layers_info.items():
            if 'routing_info' in layer_info:
                routing_info = layer_info['routing_info']
                
                # 1. Load balancing loss (already computed)
                load_balance_loss = routing_info.get('load_balance_loss', 0)
                if isinstance(load_balance_loss, torch.Tensor):
                    total_loss += self.config.load_balance_weight * load_balance_loss
                
                # 2. Efficiency loss (encourage identity module usage)
                expert_utilization = routing_info.get('expert_utilization', torch.zeros(4))
                if len(expert_utilization) > 2:  # Ensure identity module exists
                    identity_usage = expert_utilization[2]  # Identity is index 2
                    efficiency_loss = torch.abs(identity_usage - self.config.target_efficiency)
                    total_loss += self.config.efficiency_loss_weight * efficiency_loss
                
                # 3. Routing entropy loss (encourage diversity)
                routing_entropy = routing_info.get('routing_entropy', 0)
                if isinstance(routing_entropy, torch.Tensor):
                    entropy_loss = torch.relu(self.config.target_routing_entropy - routing_entropy)
                    total_loss += self.config.routing_loss_weight * entropy_loss
        
        return total_loss
    
    def _compute_validation_metrics(self, model_info: Dict) -> Dict[str, float]:
        """Compute validation metrics."""
        efficiency_scores = []
        routing_entropies = []
        
        dna_layers_info = model_info.get('dna_layers_info', {})
        
        for layer_info in dna_layers_info.values():
            if 'routing_info' in layer_info:
                routing_info = layer_info['routing_info']
                
                # Efficiency (identity module usage)
                expert_utilization = routing_info.get('expert_utilization', torch.zeros(4))
                if len(expert_utilization) > 2:
                    identity_usage = expert_utilization[2].item()
                    efficiency_scores.append(identity_usage)
                
                # Routing entropy
                routing_entropy = routing_info.get('routing_entropy', 0)
                if isinstance(routing_entropy, torch.Tensor):
                    routing_entropies.append(routing_entropy.item())
                else:
                    routing_entropies.append(routing_entropy)
        
        return {
            'efficiency': np.mean(efficiency_scores) if efficiency_scores else 0.0,
            'routing_entropy': np.mean(routing_entropies) if routing_entropies else 0.0,
            'specialization': 0.5  # Placeholder for now
        }
    
    def _collate_fn(self, batch):
        """Custom collate function for batching."""
        # Stack tensors
        hidden_states = torch.stack([item['hidden_states'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        complexity_scores = torch.stack([item['complexity_score'] for item in batch])
        
        # Collect lists
        token_types = [item['token_types'] for item in batch]
        scenario_types = [item['scenario_type'] for item in batch]
        expected_routing = [item['expected_routing'] for item in batch]
        
        return {
            'hidden_states': hidden_states,
            'attention_mask': attention_mask,
            'token_types': token_types,
            'complexity_scores': complexity_scores,
            'scenario_types': scenario_types,
            'expected_routing': expected_routing
        }
    
    def _save_checkpoint(self, path: Path, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config.__dict__,
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def _save_results(self, results: Dict, path: Path):
        """Save training results to JSON."""
        # Convert tensors to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                serializable_results[key] = value.tolist()
            elif isinstance(value, dict):
                serializable_results[key] = {
                    k: v.tolist() if isinstance(v, torch.Tensor) else v
                    for k, v in value.items()
                }
            else:
                serializable_results[key] = value
        
        with open(path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Training results saved to {path}")


def create_dna_trainer(
    dna_layer_position: int = 6,
    training_config: Optional[TrainingConfig] = None,
    device: str = 'cpu'
) -> DNATrainer:
    """Create a DNA trainer with default configuration."""
    if training_config is None:
        training_config = TrainingConfig()
    
    # Create DNA-enhanced SAM model
    from .sam_integration import create_dna_enhanced_sam_model
    model = create_dna_enhanced_sam_model(
        dna_layer_position=dna_layer_position,
        operation_mode='dna'  # Pure DNA mode for training
    )
    
    trainer = DNATrainer(model, training_config, device)
    return trainer
