#!/usr/bin/env python3
"""
SSRL Fine-Tuning Script
=======================

Reinforcement Learning fine-tuning script for Self-Supervised Reasoning and Learning (SSRL).
Adapts the existing DPO pipeline to train a specialized SSRL-LoRA adapter using GRPO
(Generalized Reward Policy Optimization) or similar RL techniques.

Features:
- Reuses existing model loading and LoRA infrastructure
- Implements SSRL-specific reward functions
- Supports QA dataset training (Natural Questions, HotpotQA, etc.)
- Comprehensive logging and checkpoint management
- Robust error handling and recovery mechanisms
- Performance monitoring and optimization

Author: SAM Development Team
Version: 1.0.0
"""

import os
import sys
import json
import logging
import argparse
import signal
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from contextlib import contextmanager

# Add SAM to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ssrl_training.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL_NAME = "microsoft/DialoGPT-medium"
DEFAULT_OUTPUT_DIR = "models/ssrl_lora"
DEFAULT_DATASET_PATH = "data/ssrl_training"
MAX_RETRIES = 3
CHECKPOINT_INTERVAL = 500

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
    DataCollatorForLanguageModeling, get_scheduler
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import wandb

# SAM imports
from sam.learning.ssrl_rewards import get_combined_reward, SSRLRewardResult

logger = logging.getLogger(__name__)


@dataclass
class SSRLTrainingArguments:
    """
    Arguments for SSRL training with validation.

    Attributes:
        model_name: HuggingFace model name or path
        dataset_path: Path to training dataset
        output_dir: Directory for saving trained models
        ... (other attributes)
    """

    # Model and data
    model_name: str = DEFAULT_MODEL_NAME
    dataset_path: str = DEFAULT_DATASET_PATH
    output_dir: str = DEFAULT_OUTPUT_DIR
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # Training parameters
    learning_rate: float = 1e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    max_length: int = 1024
    
    # PPO/RL specific
    ppo_epochs: int = 4
    mini_batch_size: int = 1
    init_kl_coef: float = 0.2
    target_kl: float = 6.0
    adap_kl_ctrl: bool = True
    
    # Reward function weights
    format_reward_weight: float = 0.3
    outcome_reward_weight: float = 0.7
    
    # Logging and checkpointing
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 250
    save_total_limit: int = 3
    
    # Experiment tracking
    run_name: str = "ssrl_training"
    use_wandb: bool = True
    wandb_project: str = "sam_ssrl"
    
    # Hardware
    fp16: bool = True
    dataloader_num_workers: int = 4
    
    # Safety
    max_grad_norm: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate training arguments after initialization."""
        self._validate_arguments()

    def _validate_arguments(self) -> None:
        """
        Validate training arguments and raise errors for invalid values.

        Raises:
            ValueError: If any argument is invalid
            FileNotFoundError: If required paths don't exist
        """
        # Validate numeric arguments
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")

        if self.num_train_epochs <= 0:
            raise ValueError(f"num_train_epochs must be positive, got {self.num_train_epochs}")

        if self.per_device_train_batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.per_device_train_batch_size}")

        if not 0.0 <= self.format_reward_weight <= 1.0:
            raise ValueError(f"format_reward_weight must be in [0,1], got {self.format_reward_weight}")

        if not 0.0 <= self.outcome_reward_weight <= 1.0:
            raise ValueError(f"outcome_reward_weight must be in [0,1], got {self.outcome_reward_weight}")

        # Validate weight sum
        weight_sum = self.format_reward_weight + self.outcome_reward_weight
        if abs(weight_sum - 1.0) > 0.01:
            raise ValueError(f"Reward weights must sum to 1.0, got {weight_sum:.3f}")

        # Validate paths
        if self.dataset_path and not Path(self.dataset_path).exists():
            logger.warning(f"Dataset path does not exist: {self.dataset_path}")

        # Create output directory if it doesn't exist
        try:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ValueError(f"Cannot create output directory {self.output_dir}: {e}")

        # Validate string arguments
        if not self.model_name.strip():
            raise ValueError("model_name cannot be empty")

        if not self.run_name.strip():
            raise ValueError("run_name cannot be empty")

        logger.info("Training arguments validation passed")

    def to_dict(self) -> Dict[str, Any]:
        """Convert arguments to dictionary for serialization."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SSRLTrainingArguments':
        """Create arguments from dictionary."""
        return cls(**data)

    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """Save arguments to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> 'SSRLTrainingArguments':
        """Load arguments from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class SSRLDataset(Dataset):
    """
    Dataset for SSRL training with QA pairs.
    
    Loads question-answer pairs and formats them for SSRL training.
    """
    
    def __init__(self, 
                 data_path: str,
                 tokenizer: AutoTokenizer,
                 max_length: int = 1024):
        """
        Initialize SSRL dataset.
        
        Args:
            data_path: Path to training data
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load training data
        self.data = self._load_training_data(data_path)
        
        logger.info(f"Loaded {len(self.data)} training examples from {data_path}")
    
    def _load_training_data(self, data_path: str) -> List[Dict[str, str]]:
        """Load training data from file."""
        data = []
        
        # Support multiple formats
        data_path = Path(data_path)
        
        if data_path.is_file():
            # Single file
            data.extend(self._load_single_file(data_path))
        elif data_path.is_dir():
            # Directory with multiple files
            for file_path in data_path.glob("*.json"):
                data.extend(self._load_single_file(file_path))
        else:
            # Create sample data if path doesn't exist
            logger.warning(f"Data path {data_path} not found, creating sample data")
            data = self._create_sample_data()
        
        return data
    
    def _load_single_file(self, file_path: Path) -> List[Dict[str, str]]:
        """Load data from a single JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Ensure data is in the correct format
            formatted_data = []
            for item in data:
                if isinstance(item, dict) and 'question' in item and 'answer' in item:
                    formatted_data.append({
                        'question': item['question'],
                        'answer': item['answer'],
                        'context': item.get('context', '')
                    })
            
            return formatted_data
            
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return []
    
    def _create_sample_data(self) -> List[Dict[str, str]]:
        """Create sample training data for testing."""
        return [
            {
                'question': 'What is the capital of France?',
                'answer': 'Paris',
                'context': ''
            },
            {
                'question': 'Explain quantum computing in simple terms.',
                'answer': 'Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information in ways that classical computers cannot.',
                'context': ''
            },
            {
                'question': 'What is machine learning?',
                'answer': 'Machine learning is a branch of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.',
                'context': ''
            }
        ]
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a training example."""
        item = self.data[idx]
        
        # Create SSRL prompt
        prompt = self._create_ssrl_prompt(item['question'], item.get('context', ''))
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'question': item['question'],
            'ground_truth': item['answer'],
            'context': item.get('context', ''),
            'prompt': prompt
        }
    
    def _create_ssrl_prompt(self, question: str, context: str = '') -> str:
        """Create SSRL-formatted prompt for training."""
        base_prompt = """You are an AI assistant using Self-Supervised Reasoning and Learning (SSRL). 
Your task is to answer the user's question through structured reasoning and self-assessment.

IMPORTANT: Structure your response using these exact tags:

<think>
[Your step-by-step reasoning process. Break down the problem, consider different angles, 
and work through the logic systematically.]
</think>

<search>
[Search your internal knowledge for relevant information. What do you know about this topic?
What facts, concepts, or examples are relevant?]
</search>

<information>
[Synthesize the information you found. What are the key points? How do they relate to the question?]
</information>

<confidence>
[Assess your confidence in this answer on a scale of 0.0 to 1.0. Consider:
- How certain are you about the facts?
- How complete is your knowledge on this topic?
- Are there any gaps or uncertainties?
Provide just the number, e.g., 0.8]
</confidence>

<answer>
[Your final, clear answer to the user's question based on your reasoning above.]
</answer>

"""
        
        if context:
            base_prompt += f"Context: {context}\n\n"
        
        base_prompt += f"User's question: {question}\n\n"
        base_prompt += "Remember: Be thorough in your reasoning, honest about uncertainties, and provide a realistic confidence assessment."
        
        return base_prompt


class SSRLDataCollator:
    """
    Data collator for SSRL training that handles batching and padding.
    """
    
    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 1024):
        """
        Initialize data collator.
        
        Args:
            tokenizer: Tokenizer for padding
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of examples.
        
        Args:
            batch: List of examples from dataset
            
        Returns:
            Batched and padded tensors
        """
        # Extract input_ids and attention_masks
        input_ids = [item['input_ids'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]
        
        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_masks = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )
        
        # Truncate if necessary
        if input_ids.size(1) > self.max_length:
            input_ids = input_ids[:, :self.max_length]
            attention_masks = attention_masks[:, :self.max_length]
        
        # Store metadata for reward calculation
        metadata = {
            'questions': [item['question'] for item in batch],
            'ground_truths': [item['ground_truth'] for item in batch],
            'contexts': [item['context'] for item in batch],
            'prompts': [item['prompt'] for item in batch]
        }
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'metadata': metadata
        }


class SSRLRewardFunction:
    """
    Reward function wrapper for SSRL training.
    
    Combines format and outcome rewards for RL training.
    """
    
    def __init__(self, 
                 format_weight: float = 0.3,
                 outcome_weight: float = 0.7):
        """
        Initialize reward function.
        
        Args:
            format_weight: Weight for format reward
            outcome_weight: Weight for outcome reward
        """
        self.reward_calculator = get_combined_reward()
        self.reward_calculator.format_weight = format_weight
        self.reward_calculator.outcome_weight = outcome_weight
        
        logger.info(f"SSRL reward function initialized (format: {format_weight}, outcome: {outcome_weight})")
    
    def __call__(self, 
                 generated_texts: List[str],
                 questions: List[str],
                 ground_truths: List[str]) -> List[float]:
        """
        Calculate rewards for a batch of generated texts.
        
        Args:
            generated_texts: List of generated SSRL responses
            questions: List of original questions
            ground_truths: List of correct answers
            
        Returns:
            List of reward scores (0.0 to 1.0)
        """
        rewards = []
        
        for generated, question, ground_truth in zip(generated_texts, questions, ground_truths):
            try:
                # Calculate combined reward
                reward_result = self.reward_calculator.calculate_reward(
                    generated_text=generated,
                    ground_truth=ground_truth,
                    question=question
                )
                
                rewards.append(reward_result.score)
                
            except Exception as e:
                logger.error(f"Reward calculation failed for question '{question}': {e}")
                rewards.append(0.0)  # Default to zero reward on error
        
        return rewards


class SSRLTrainingPipeline:
    """
    Main SSRL training pipeline using PPO/RL for fine-tuning.
    """

    def __init__(self, args: SSRLTrainingArguments):
        """
        Initialize SSRL training pipeline.

        Args:
            args: Training arguments
        """
        self.args = args

        # Set random seeds
        torch.manual_seed(args.seed)

        # Initialize components
        self.tokenizer = None
        self.model = None
        self.ref_model = None
        self.dataset = None
        self.reward_fn = None
        self.trainer = None

        # Setup logging
        self._setup_logging()

        logger.info(f"SSRL Training Pipeline initialized")
        logger.info(f"Output directory: {args.output_dir}")

    def _setup_logging(self):
        """Setup logging and experiment tracking."""
        # Create output directory
        os.makedirs(self.args.output_dir, exist_ok=True)

        # Setup file logging
        log_file = Path(self.args.output_dir) / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Setup wandb if enabled
        if self.args.use_wandb:
            try:
                wandb.init(
                    project=self.args.wandb_project,
                    name=self.args.run_name,
                    config=self.args.__dict__
                )
                logger.info("Weights & Biases logging enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
                self.args.use_wandb = False

    def load_model_and_tokenizer(self):
        """Load model and tokenizer with LoRA configuration."""
        logger.info(f"Loading model: {self.args.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            torch_dtype=torch.float16 if self.args.fp16 else torch.float32,
            device_map="auto"
        )

        # Setup LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            target_modules=self.args.lora_target_modules,
            bias="none"
        )

        # Apply LoRA to model
        self.model = get_peft_model(base_model, lora_config)

        # Create model with value head for PPO
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(self.model)

        # Create reference model (frozen copy)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            torch_dtype=torch.float16 if self.args.fp16 else torch.float32,
            device_map="auto"
        )

        # Print model info
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        logger.info(f"Model loaded successfully:")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable ratio: {100 * trainable_params / total_params:.2f}%")

    def prepare_dataset(self):
        """Prepare training dataset."""
        logger.info("Preparing training dataset...")

        self.dataset = SSRLDataset(
            data_path=self.args.dataset_path,
            tokenizer=self.tokenizer,
            max_length=self.args.max_length
        )

        logger.info(f"Dataset prepared with {len(self.dataset)} examples")

    def setup_trainer(self):
        """Setup PPO trainer for SSRL training."""
        logger.info("Setting up PPO trainer...")

        # PPO configuration
        ppo_config = PPOConfig(
            model_name=self.args.model_name,
            learning_rate=self.args.learning_rate,
            ppo_epochs=self.args.ppo_epochs,
            mini_batch_size=self.args.mini_batch_size,
            batch_size=self.args.per_device_train_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            init_kl_coef=self.args.init_kl_coef,
            target_kl=self.args.target_kl,
            adap_kl_ctrl=self.args.adap_kl_ctrl,
            max_grad_norm=self.args.max_grad_norm,
            seed=self.args.seed
        )

        # Create data collator
        data_collator = SSRLDataCollator(
            tokenizer=self.tokenizer,
            max_length=self.args.max_length
        )

        # Create reward function
        self.reward_fn = SSRLRewardFunction(
            format_weight=self.args.format_reward_weight,
            outcome_weight=self.args.outcome_reward_weight
        )

        # Create trainer
        self.trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
            dataset=self.dataset,
            data_collator=data_collator
        )

        logger.info("PPO trainer setup complete")

    def train(self):
        """Execute SSRL training."""
        logger.info("Starting SSRL training...")

        # Training loop
        for epoch in range(self.args.num_train_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.args.num_train_epochs}")

            # Create dataloader for this epoch
            dataloader = DataLoader(
                self.dataset,
                batch_size=self.args.per_device_train_batch_size,
                shuffle=True,
                collate_fn=SSRLDataCollator(self.tokenizer, self.args.max_length),
                num_workers=self.args.dataloader_num_workers
            )

            epoch_rewards = []

            for step, batch in enumerate(dataloader):
                try:
                    # Generate responses
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                    metadata = batch['metadata']

                    # Generate with the model
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=512,
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=self.tokenizer.pad_token_id
                        )

                    # Decode generated text
                    generated_texts = []
                    for i, gen_ids in enumerate(generated_ids):
                        # Remove input prompt from generated text
                        prompt_length = len(input_ids[i])
                        response_ids = gen_ids[prompt_length:]
                        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                        generated_texts.append(response_text)

                    # Calculate rewards
                    rewards = self.reward_fn(
                        generated_texts=generated_texts,
                        questions=metadata['questions'],
                        ground_truths=metadata['ground_truths']
                    )

                    # Convert rewards to tensors
                    reward_tensors = [torch.tensor(r, dtype=torch.float32) for r in rewards]

                    # PPO step
                    stats = self.trainer.step(
                        queries=input_ids,
                        responses=generated_ids[:, input_ids.size(1):],  # Only response part
                        scores=reward_tensors
                    )

                    # Track metrics
                    epoch_rewards.extend(rewards)

                    # Log progress
                    if step % self.args.logging_steps == 0:
                        avg_reward = sum(rewards) / len(rewards)
                        logger.info(f"Epoch {epoch + 1}, Step {step}: avg_reward={avg_reward:.4f}")

                        if self.args.use_wandb:
                            wandb.log({
                                'epoch': epoch + 1,
                                'step': step,
                                'avg_reward': avg_reward,
                                'rewards': rewards,
                                **stats
                            })

                    # Save checkpoint
                    if step % self.args.save_steps == 0 and step > 0:
                        self._save_checkpoint(epoch, step)

                except Exception as e:
                    logger.error(f"Training step failed: {e}")
                    continue

            # End of epoch logging
            epoch_avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0.0
            logger.info(f"Epoch {epoch + 1} completed. Average reward: {epoch_avg_reward:.4f}")

            if self.args.use_wandb:
                wandb.log({
                    'epoch_avg_reward': epoch_avg_reward,
                    'epoch': epoch + 1
                })

        # Save final model
        self._save_final_model()

        logger.info("SSRL training completed!")

    def _save_checkpoint(self, epoch: int, step: int):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.args.output_dir) / f"checkpoint-epoch-{epoch}-step-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save training state
        state = {
            'epoch': epoch,
            'step': step,
            'args': self.args.__dict__
        }

        with open(checkpoint_dir / "training_state.json", 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"Checkpoint saved: {checkpoint_dir}")

    def _save_final_model(self):
        """Save final trained model."""
        final_dir = Path(self.args.output_dir) / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)

        # Save training configuration
        with open(final_dir / "training_config.json", 'w') as f:
            json.dump(self.args.__dict__, f, indent=2)

        logger.info(f"Final model saved: {final_dir}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="SSRL Fine-tuning Script")

    # Add arguments
    parser.add_argument("--model_name", type=str, default="microsoft/DialoGPT-medium",
                       help="Base model name")
    parser.add_argument("--dataset_path", type=str, default="data/ssrl_training",
                       help="Path to training dataset")
    parser.add_argument("--output_dir", type=str, default="models/ssrl_lora",
                       help="Output directory for trained model")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--run_name", type=str, default="ssrl_training",
                       help="Run name for logging")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Enable Weights & Biases logging")

    args = parser.parse_args()

    # Create training arguments
    training_args = SSRLTrainingArguments(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        run_name=args.run_name,
        use_wandb=args.use_wandb
    )

    # Initialize and run training
    pipeline = SSRLTrainingPipeline(training_args)

    try:
        pipeline.load_model_and_tokenizer()
        pipeline.prepare_dataset()
        pipeline.setup_trainer()
        pipeline.train()

        logger.info("🎉 SSRL training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(f"❌ Training failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
