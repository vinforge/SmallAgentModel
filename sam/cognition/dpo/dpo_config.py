"""
DPO Configuration Manager

Handles loading, validation, and management of DPO training configurations
for the SAM Personalized Tuner system.

Author: SAM Development Team
Version: 1.0.0
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration settings."""
    base_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    model_cache_dir: str = "./models/cache"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    load_in_8bit: bool = False
    load_in_4bit: bool = True
    use_flash_attention: bool = True
    trust_remote_code: bool = False


@dataclass
class LoRAConfig:
    """LoRA configuration settings."""
    r: int = 16
    alpha: int = 32
    dropout: float = 0.1
    bias: str = "none"
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ])
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """Training configuration settings."""
    beta: float = 0.1
    learning_rate: float = 5.0e-7
    max_length: int = 2048
    max_prompt_length: int = 1024
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    optim: str = "adamw_torch"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    save_steps: int = 500
    logging_steps: int = 10
    eval_steps: int = 250
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    group_by_length: bool = True


@dataclass
class DataConfig:
    """Data configuration settings."""
    min_confidence_threshold: float = 0.8
    min_quality_threshold: float = 0.6
    max_training_samples: int = 1000
    shuffle_data: bool = True
    train_test_split: float = 0.9
    min_response_length: int = 10
    max_response_length: int = 1024
    filter_duplicates: bool = True


@dataclass
class OutputConfig:
    """Output configuration settings."""
    output_dir: str = "./models/personalized"
    save_total_limit: int = 3
    save_strategy: str = "steps"
    logging_dir: str = "./logs/dpo_training"
    report_to: list = field(default_factory=lambda: ["tensorboard"])
    run_name: Optional[str] = None


@dataclass
class DPOConfig:
    """Complete DPO configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    user_overrides: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"


class DPOConfigManager:
    """
    Manager for DPO configuration loading, validation, and customization.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the DPO configuration manager.
        
        Args:
            config_path: Path to the configuration file
        """
        self.logger = logging.getLogger(f"{__name__}.DPOConfigManager")
        
        # Default config path
        if config_path is None:
            config_path = Path(__file__).parent / "dpo_config.yaml"
        
        self.config_path = Path(config_path)
        self.config: Optional[DPOConfig] = None
        
        # Load configuration
        self.load_config()
    
    def load_config(self) -> DPOConfig:
        """Load configuration from YAML file."""
        try:
            if not self.config_path.exists():
                self.logger.warning(f"Config file not found: {self.config_path}")
                self.logger.info("Using default configuration")
                self.config = DPOConfig()
                return self.config
            
            with open(self.config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Parse configuration sections
            model_config = ModelConfig(**config_dict.get('model', {}))
            lora_config = LoRAConfig(**config_dict.get('lora', {}))
            training_config = TrainingConfig(**config_dict.get('training', {}))
            data_config = DataConfig(**config_dict.get('data', {}))
            output_config = OutputConfig(**config_dict.get('output', {}))
            
            self.config = DPOConfig(
                model=model_config,
                lora=lora_config,
                training=training_config,
                data=data_config,
                output=output_config,
                user_overrides=config_dict.get('user_overrides', {}),
                version=config_dict.get('version', '1.0.0')
            )
            
            self.logger.info(f"Loaded DPO configuration from {self.config_path}")
            return self.config
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            self.logger.info("Using default configuration")
            self.config = DPOConfig()
            return self.config
    
    def save_config(self, config_path: Optional[str] = None) -> bool:
        """Save current configuration to YAML file."""
        try:
            if config_path is None:
                config_path = self.config_path
            
            config_dict = self.to_dict()
            
            # Ensure directory exists
            Path(config_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Saved DPO configuration to {config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            return False
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """Update configuration with new values."""
        try:
            if self.config is None:
                self.load_config()
            
            # Apply updates to user_overrides
            self.config.user_overrides.update(updates)
            
            # Apply updates to actual config sections
            for section, values in updates.items():
                if hasattr(self.config, section) and isinstance(values, dict):
                    section_config = getattr(self.config, section)
                    for key, value in values.items():
                        if hasattr(section_config, key):
                            setattr(section_config, key, value)
            
            self.logger.info("Configuration updated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
            return False
    
    def get_training_args(self) -> Dict[str, Any]:
        """Get training arguments in format expected by transformers TrainingArguments."""
        if self.config is None:
            self.load_config()
        
        training_args = {
            # Core training parameters
            'learning_rate': self.config.training.learning_rate,
            'num_train_epochs': self.config.training.num_train_epochs,
            'per_device_train_batch_size': self.config.training.per_device_train_batch_size,
            'gradient_accumulation_steps': self.config.training.gradient_accumulation_steps,
            'warmup_steps': self.config.training.warmup_steps,
            'weight_decay': self.config.training.weight_decay,
            'max_grad_norm': self.config.training.max_grad_norm,
            'lr_scheduler_type': self.config.training.lr_scheduler_type,
            
            # Logging and saving
            'logging_steps': self.config.training.logging_steps,
            'save_steps': self.config.training.save_steps,
            'eval_steps': self.config.training.eval_steps,
            'output_dir': self.config.output.output_dir,
            'logging_dir': self.config.output.logging_dir,
            'save_total_limit': self.config.output.save_total_limit,
            'save_strategy': self.config.output.save_strategy,
            'report_to': self.config.output.report_to,
            
            # Data loading
            'dataloader_num_workers': self.config.training.dataloader_num_workers,
            'remove_unused_columns': self.config.training.remove_unused_columns,
            'group_by_length': self.config.training.group_by_length,
            
            # Optimization
            'optim': self.config.training.optim,
            'bf16': True,  # Use bfloat16 for better performance
            'gradient_checkpointing': True,  # Save memory
        }
        
        # Add run name if specified
        if self.config.output.run_name:
            training_args['run_name'] = self.config.output.run_name
        
        return training_args
    
    def get_dpo_args(self) -> Dict[str, Any]:
        """Get DPO-specific arguments."""
        if self.config is None:
            self.load_config()
        
        return {
            'beta': self.config.training.beta,
            'max_length': self.config.training.max_length,
            'max_prompt_length': self.config.training.max_prompt_length,
        }
    
    def get_lora_config(self) -> Dict[str, Any]:
        """Get LoRA configuration for PEFT."""
        if self.config is None:
            self.load_config()
        
        return {
            'r': self.config.lora.r,
            'lora_alpha': self.config.lora.alpha,
            'lora_dropout': self.config.lora.dropout,
            'bias': self.config.lora.bias,
            'target_modules': self.config.lora.target_modules,
            'task_type': self.config.lora.task_type,
        }
    
    def create_user_config(self, user_id: str, overrides: Dict[str, Any]) -> 'DPOConfigManager':
        """Create a user-specific configuration with overrides."""
        user_config = DPOConfigManager(self.config_path)
        user_config.update_config(overrides)
        
        # Set user-specific output directory
        user_output_dir = Path(self.config.output.output_dir) / user_id
        user_config.config.output.output_dir = str(user_output_dir)
        
        # Set user-specific run name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        user_config.config.output.run_name = f"dpo_{user_id}_{timestamp}"
        
        return user_config
    
    def validate_config(self) -> tuple[bool, list[str]]:
        """Validate the current configuration."""
        if self.config is None:
            return False, ["Configuration not loaded"]
        
        issues = []
        
        # Validate model configuration
        if not self.config.model.base_model_name:
            issues.append("Base model name is required")
        
        # Validate LoRA configuration
        if self.config.lora.r <= 0:
            issues.append("LoRA rank must be positive")
        
        if self.config.lora.alpha <= 0:
            issues.append("LoRA alpha must be positive")
        
        # Validate training configuration
        if self.config.training.learning_rate <= 0:
            issues.append("Learning rate must be positive")
        
        if self.config.training.num_train_epochs <= 0:
            issues.append("Number of epochs must be positive")
        
        # Validate data configuration
        if not (0 < self.config.data.min_confidence_threshold <= 1):
            issues.append("Confidence threshold must be between 0 and 1")
        
        if not (0 < self.config.data.train_test_split <= 1):
            issues.append("Train/test split must be between 0 and 1")
        
        return len(issues) == 0, issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        if self.config is None:
            return {}
        
        return {
            'model': {
                'base_model_name': self.config.model.base_model_name,
                'model_cache_dir': self.config.model.model_cache_dir,
                'torch_dtype': self.config.model.torch_dtype,
                'device_map': self.config.model.device_map,
                'load_in_8bit': self.config.model.load_in_8bit,
                'load_in_4bit': self.config.model.load_in_4bit,
                'use_flash_attention': self.config.model.use_flash_attention,
                'trust_remote_code': self.config.model.trust_remote_code,
            },
            'lora': {
                'r': self.config.lora.r,
                'alpha': self.config.lora.alpha,
                'dropout': self.config.lora.dropout,
                'bias': self.config.lora.bias,
                'target_modules': self.config.lora.target_modules,
                'task_type': self.config.lora.task_type,
            },
            'training': {
                'beta': self.config.training.beta,
                'learning_rate': self.config.training.learning_rate,
                'max_length': self.config.training.max_length,
                'max_prompt_length': self.config.training.max_prompt_length,
                'num_train_epochs': self.config.training.num_train_epochs,
                'per_device_train_batch_size': self.config.training.per_device_train_batch_size,
                'gradient_accumulation_steps': self.config.training.gradient_accumulation_steps,
                'warmup_steps': self.config.training.warmup_steps,
                'optim': self.config.training.optim,
                'weight_decay': self.config.training.weight_decay,
                'max_grad_norm': self.config.training.max_grad_norm,
                'lr_scheduler_type': self.config.training.lr_scheduler_type,
                'save_steps': self.config.training.save_steps,
                'logging_steps': self.config.training.logging_steps,
                'eval_steps': self.config.training.eval_steps,
                'dataloader_num_workers': self.config.training.dataloader_num_workers,
                'remove_unused_columns': self.config.training.remove_unused_columns,
                'group_by_length': self.config.training.group_by_length,
            },
            'data': {
                'min_confidence_threshold': self.config.data.min_confidence_threshold,
                'min_quality_threshold': self.config.data.min_quality_threshold,
                'max_training_samples': self.config.data.max_training_samples,
                'shuffle_data': self.config.data.shuffle_data,
                'train_test_split': self.config.data.train_test_split,
                'min_response_length': self.config.data.min_response_length,
                'max_response_length': self.config.data.max_response_length,
                'filter_duplicates': self.config.data.filter_duplicates,
            },
            'output': {
                'output_dir': self.config.output.output_dir,
                'save_total_limit': self.config.output.save_total_limit,
                'save_strategy': self.config.output.save_strategy,
                'logging_dir': self.config.output.logging_dir,
                'report_to': self.config.output.report_to,
                'run_name': self.config.output.run_name,
            },
            'user_overrides': self.config.user_overrides,
            'version': self.config.version,
        }


# Global configuration manager instance
_config_manager = None

def get_dpo_config_manager(config_path: Optional[str] = None) -> DPOConfigManager:
    """Get or create a global DPO configuration manager instance."""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = DPOConfigManager(config_path)
    
    return _config_manager
