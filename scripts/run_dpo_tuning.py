#!/usr/bin/env python3
"""
DPO Training Script

Automated Direct Preference Optimization fine-tuning script for SAM personalization.
Reads preference data from the episodic memory database and trains a personalized LoRA adapter.

Usage:
    python scripts/run_dpo_tuning.py --user_id <user_id> [options]

Author: SAM Development Team
Version: 1.0.0
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Check for required dependencies
try:
    import torch
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM,
        TrainingArguments,
        BitsAndBytesConfig
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import DPOTrainer
    from datasets import Dataset
    import yaml
except ImportError as e:
    print(f"❌ Missing required dependency: {e}")
    print("Please install DPO dependencies: pip install -r requirements_dpo.txt")
    sys.exit(1)

# SAM imports
try:
    from sam.cognition.dpo import get_dpo_config_manager
    from sam.learning.dpo_data_manager import get_dpo_data_manager
    from memory.episodic_store import create_episodic_store
except ImportError as e:
    print(f"❌ SAM module import error: {e}")
    print("Please ensure you're running from the SAM project root directory")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DPOTrainingPipeline:
    """
    Complete DPO training pipeline for SAM personalization.
    """
    
    def __init__(self, user_id: str, config_overrides: Optional[Dict[str, Any]] = None):
        """
        Initialize the DPO training pipeline.
        
        Args:
            user_id: User identifier for personalization
            config_overrides: Configuration overrides
        """
        self.user_id = user_id
        self.config_manager = get_dpo_config_manager()
        
        # Apply user-specific configuration
        if config_overrides:
            self.config_manager.update_config(config_overrides)
        
        # Create user-specific config
        self.user_config = self.config_manager.create_user_config(user_id, config_overrides or {})
        
        # Initialize data manager
        self.episodic_store = create_episodic_store()
        self.dpo_manager = get_dpo_data_manager(self.episodic_store)
        
        # Training components (initialized later)
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        logger.info(f"Initialized DPO training pipeline for user: {user_id}")
    
    def validate_training_data(self) -> tuple[bool, str, int]:
        """
        Validate that sufficient training data is available.
        
        Returns:
            Tuple of (is_valid, message, data_count)
        """
        try:
            # Get user statistics
            stats = self.dpo_manager.get_user_stats(self.user_id)
            
            training_ready_pairs = stats.get('training_ready_pairs', 0)
            total_pairs = stats.get('total_pairs', 0)
            
            if total_pairs == 0:
                return False, "No preference data found for user", 0
            
            if training_ready_pairs < 10:
                return False, f"Insufficient training data: {training_ready_pairs} pairs (minimum: 10)", training_ready_pairs
            
            return True, f"Training data validated: {training_ready_pairs} high-quality pairs", training_ready_pairs
            
        except Exception as e:
            return False, f"Error validating training data: {e}", 0
    
    def prepare_dataset(self) -> Dataset:
        """
        Prepare the training dataset from DPO preference pairs.
        
        Returns:
            Hugging Face Dataset object
        """
        logger.info("Preparing training dataset...")
        
        # Get training data
        training_data = self.dpo_manager.get_training_dataset(
            user_id=self.user_id,
            min_confidence=self.user_config.config.data.min_confidence_threshold,
            min_quality=self.user_config.config.data.min_quality_threshold,
            limit=self.user_config.config.data.max_training_samples
        )
        
        if not training_data:
            raise ValueError("No training data available")
        
        logger.info(f"Loaded {len(training_data)} training examples")
        
        # Convert to Hugging Face dataset format
        dataset = Dataset.from_list(training_data)
        
        # Shuffle if configured
        if self.user_config.config.data.shuffle_data:
            dataset = dataset.shuffle(seed=42)
        
        # Split into train/validation
        split_ratio = self.user_config.config.data.train_test_split
        if split_ratio < 1.0:
            split_dataset = dataset.train_test_split(
                train_size=split_ratio,
                seed=42
            )
            train_dataset = split_dataset['train']
            eval_dataset = split_dataset['test']
            logger.info(f"Split dataset: {len(train_dataset)} train, {len(eval_dataset)} eval")
            return train_dataset, eval_dataset
        else:
            logger.info(f"Using full dataset for training: {len(dataset)} examples")
            return dataset, None
    
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer with quantization."""
        logger.info("Loading model and tokenizer...")
        
        model_config = self.user_config.config.model
        
        # Setup quantization config
        quantization_config = None
        if model_config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif model_config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config.base_model_name,
            cache_dir=model_config.model_cache_dir,
            trust_remote_code=model_config.trust_remote_code
        )
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config.base_model_name,
            cache_dir=model_config.model_cache_dir,
            quantization_config=quantization_config,
            device_map=model_config.device_map,
            torch_dtype=getattr(torch, model_config.torch_dtype),
            trust_remote_code=model_config.trust_remote_code,
            use_flash_attention_2=model_config.use_flash_attention
        )
        
        logger.info(f"Loaded model: {model_config.base_model_name}")
    
    def setup_lora(self):
        """Setup LoRA configuration and apply to model."""
        logger.info("Setting up LoRA configuration...")
        
        lora_config_dict = self.user_config.get_lora_config()
        
        # Create LoRA config
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            **lora_config_dict
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"LoRA setup complete:")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable ratio: {100 * trainable_params / total_params:.2f}%")
    
    def create_trainer(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """Create the DPO trainer."""
        logger.info("Creating DPO trainer...")
        
        # Get training arguments
        training_args_dict = self.user_config.get_training_args()
        training_args = TrainingArguments(**training_args_dict)
        
        # Get DPO-specific arguments
        dpo_args = self.user_config.get_dpo_args()
        
        # Create DPO trainer
        self.trainer = DPOTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            **dpo_args
        )
        
        logger.info("DPO trainer created successfully")
    
    def train(self) -> Dict[str, Any]:
        """Execute the training process."""
        logger.info("Starting DPO training...")
        
        # Create output directory
        output_dir = Path(self.user_config.config.output.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = output_dir / "training_config.yaml"
        self.user_config.save_config(config_path)
        
        # Start training
        start_time = datetime.now()
        
        try:
            # Train the model
            train_result = self.trainer.train()
            
            # Save the final model
            self.trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            # Calculate training time
            training_time = datetime.now() - start_time
            
            # Prepare results
            results = {
                'success': True,
                'user_id': self.user_id,
                'output_dir': str(output_dir),
                'training_time_seconds': training_time.total_seconds(),
                'final_loss': train_result.training_loss,
                'total_steps': train_result.global_step,
                'model_name': self.user_config.config.model.base_model_name,
                'lora_config': self.user_config.get_lora_config(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Save results
            results_path = output_dir / "training_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Training completed successfully in {training_time}")
            logger.info(f"Model saved to: {output_dir}")
            
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'user_id': self.user_id,
                'timestamp': datetime.now().isoformat()
            }
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete DPO training pipeline."""
        try:
            # Validate configuration
            config_valid, config_issues = self.user_config.validate_config()
            if not config_valid:
                return {
                    'success': False,
                    'error': f"Configuration validation failed: {config_issues}",
                    'user_id': self.user_id
                }
            
            # Validate training data
            data_valid, data_message, data_count = self.validate_training_data()
            if not data_valid:
                return {
                    'success': False,
                    'error': data_message,
                    'user_id': self.user_id,
                    'data_count': data_count
                }
            
            logger.info(data_message)
            
            # Prepare dataset
            train_dataset, eval_dataset = self.prepare_dataset()
            
            # Load model and tokenizer
            self.load_model_and_tokenizer()
            
            # Setup LoRA
            self.setup_lora()
            
            # Create trainer
            self.create_trainer(train_dataset, eval_dataset)
            
            # Train
            return self.train()
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'user_id': self.user_id,
                'timestamp': datetime.now().isoformat()
            }


def main():
    """Main entry point for the DPO training script."""
    parser = argparse.ArgumentParser(description="SAM DPO Training Script")
    
    # Required arguments
    parser.add_argument("--user_id", required=True, help="User ID for personalization")
    
    # Optional arguments
    parser.add_argument("--config", help="Path to custom configuration file")
    parser.add_argument("--output_dir", help="Output directory for trained model")
    parser.add_argument("--learning_rate", type=float, help="Learning rate override")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Training batch size")
    parser.add_argument("--lora_rank", type=int, help="LoRA rank override")
    parser.add_argument("--beta", type=float, help="DPO beta parameter")
    parser.add_argument("--dry_run", action="store_true", help="Validate setup without training")
    
    args = parser.parse_args()
    
    # Prepare configuration overrides
    config_overrides = {}
    
    if args.output_dir:
        config_overrides['output'] = {'output_dir': args.output_dir}
    
    if args.learning_rate:
        config_overrides.setdefault('training', {})['learning_rate'] = args.learning_rate
    
    if args.num_epochs:
        config_overrides.setdefault('training', {})['num_train_epochs'] = args.num_epochs
    
    if args.batch_size:
        config_overrides.setdefault('training', {})['per_device_train_batch_size'] = args.batch_size
    
    if args.lora_rank:
        config_overrides.setdefault('lora', {})['r'] = args.lora_rank
    
    if args.beta:
        config_overrides.setdefault('training', {})['beta'] = args.beta
    
    # Initialize pipeline
    pipeline = DPOTrainingPipeline(args.user_id, config_overrides)
    
    if args.dry_run:
        # Dry run - validate setup only
        logger.info("Performing dry run validation...")
        
        # Validate configuration
        config_valid, config_issues = pipeline.user_config.validate_config()
        if not config_valid:
            print(f"❌ Configuration validation failed: {config_issues}")
            return 1
        
        # Validate training data
        data_valid, data_message, data_count = pipeline.validate_training_data()
        if not data_valid:
            print(f"❌ {data_message}")
            return 1
        
        print(f"✅ Dry run successful: {data_message}")
        print(f"✅ Configuration validated")
        print(f"✅ Ready for training")
        return 0
    
    # Run full training pipeline
    logger.info(f"Starting DPO training for user: {args.user_id}")
    results = pipeline.run_full_pipeline()
    
    if results['success']:
        print(f"🎉 Training completed successfully!")
        print(f"📁 Model saved to: {results['output_dir']}")
        print(f"⏱️  Training time: {results['training_time_seconds']:.1f} seconds")
        print(f"📊 Final loss: {results['final_loss']:.4f}")
        return 0
    else:
        print(f"❌ Training failed: {results['error']}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
