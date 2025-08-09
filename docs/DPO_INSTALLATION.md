# SAM Personalized Tuner - DPO Installation Guide

This guide covers the installation and setup of the Direct Preference Optimization (DPO) system for SAM personalization.

## Overview

The SAM Personalized Tuner enables fine-tuning of language models based on user feedback using Direct Preference Optimization (DPO). This system automatically collects preference pairs from user feedback and trains personalized LoRA adapters.

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- At least 8GB GPU memory for training small models
- 16GB+ system RAM recommended

## Installation Steps

### 1. Install Core SAM Dependencies

First, ensure SAM's core dependencies are installed:

```bash
pip install -r requirements.txt
```

### 2. Install DPO Dependencies

Install the additional dependencies required for DPO training:

```bash
pip install -r requirements_dpo.txt
```

This will install:
- `trl` - Transformers Reinforcement Learning (includes DPO)
- `peft` - Parameter-Efficient Fine-Tuning (LoRA)
- `datasets` - Dataset handling for training
- `wandb` - Weights & Biases for experiment tracking (optional)
- `tensorboard` - TensorBoard logging (optional)
- `evaluate` - Model evaluation metrics
- `rouge-score` - ROUGE metrics for text evaluation
- `deepspeed` - Memory-efficient training (optional)

### 3. Verify Installation

Run the installation verification script:

```bash
python test_dpo_phase2.py
```

You should see output indicating that all components are working correctly.

### 4. Test with Sample Data

Run the Phase 1 integration test to ensure data collection works:

```bash
python test_dpo_integration.py
```

## Configuration

### Default Configuration

The DPO system uses sensible defaults configured in `sam/cognition/dpo/dpo_config.yaml`. Key settings include:

- **Base Model**: `meta-llama/Llama-3.1-8B-Instruct`
- **LoRA Rank**: 16 (adjustable based on your needs)
- **Learning Rate**: 5e-7 (conservative for stability)
- **Training Epochs**: 3
- **Batch Size**: 1 (memory-efficient)

### Custom Configuration

You can customize the configuration by:

1. **UI Settings**: Use the Personalized Tuner interface in the Memory Control Center
2. **Configuration File**: Edit `sam/cognition/dpo/dpo_config.yaml`
3. **Runtime Overrides**: Pass parameters to the training script

## Usage

### 1. Collect Preference Data

Use SAM normally and provide feedback using the "✏️ Suggest Improvement" button. The system automatically:

- Classifies feedback types
- Validates feedback quality
- Creates DPO preference pairs
- Stores data for training

### 2. Monitor Data Collection

Access the Personalized Tuner in the Memory Control Center:

1. Open SAM's Memory Control Center
2. Select "🧠 Personalized Tuner"
3. View your preference data and statistics

### 3. Start Training

When you have sufficient data (10+ high-quality pairs):

1. Go to the "Training Controls" tab
2. Configure training parameters
3. Click "🚀 Start Fine-Tuning"
4. Monitor progress in real-time

### 4. Use Personalized Model

Once training completes:

1. The model is automatically registered
2. Activate it in the "Model Management" section
3. SAM will use your personalized model for responses

## Training Process

### Data Requirements

- **Minimum**: 10 high-quality preference pairs
- **Recommended**: 50+ pairs for better personalization
- **Quality Threshold**: Confidence score ≥ 0.8

### Training Time

Training time depends on:
- **Dataset size**: 10-100 pairs typically
- **Hardware**: GPU type and memory
- **Model size**: 8B parameters with LoRA is manageable
- **Typical duration**: 10-60 minutes

### Memory Requirements

- **8B Model + LoRA**: ~6-8GB GPU memory
- **4-bit Quantization**: Reduces memory by ~50%
- **Gradient Checkpointing**: Further memory savings

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Solutions**:
- Reduce batch size to 1
- Enable 4-bit quantization
- Use gradient checkpointing
- Reduce LoRA rank

#### 2. Training Fails to Start

**Check**:
- Sufficient training data (10+ pairs)
- GPU availability
- Dependencies installed correctly
- Configuration validation passes

#### 3. Poor Personalization Results

**Improve by**:
- Collecting more diverse feedback
- Ensuring high-quality corrections
- Adjusting learning rate
- Increasing training epochs

#### 4. Import Errors

**Solutions**:
```bash
# Reinstall DPO dependencies
pip install -r requirements_dpo.txt --force-reinstall

# Check CUDA compatibility
python -c "import torch; print(torch.cuda.is_available())"
```

### Performance Optimization

#### For Limited GPU Memory

```yaml
# In dpo_config.yaml
model:
  load_in_4bit: true
  
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  gradient_checkpointing: true

lora:
  r: 8  # Reduce from default 16
```

#### For Faster Training

```yaml
training:
  per_device_train_batch_size: 2  # If memory allows
  gradient_accumulation_steps: 2
  num_train_epochs: 2  # Reduce epochs
```

## Advanced Usage

### Command Line Training

For advanced users, train directly via command line:

```bash
python scripts/run_dpo_tuning.py \
  --user_id your_user_id \
  --learning_rate 0.0001 \
  --num_epochs 3 \
  --lora_rank 16 \
  --beta 0.1
```

### Custom Base Models

To use a different base model:

1. Update `base_model_name` in configuration
2. Ensure model is compatible with the training script
3. Adjust LoRA target modules if needed

### Batch Training

Train multiple users:

```bash
for user in user1 user2 user3; do
  python scripts/run_dpo_tuning.py --user_id $user
done
```

## Monitoring and Logging

### TensorBoard

View training progress:

```bash
tensorboard --logdir ./logs/dpo_training
```

### Weights & Biases

For advanced experiment tracking:

1. Install: `pip install wandb`
2. Login: `wandb login`
3. Enable in configuration: `report_to: ["wandb"]`

## File Structure

```
SmallAgentModel-main/
├── sam/cognition/dpo/           # DPO system core
│   ├── dpo_config.yaml          # Configuration
│   ├── dpo_config.py            # Config management
│   ├── training_manager.py      # Training orchestration
│   └── model_manager.py         # Model management
├── scripts/
│   └── run_dpo_tuning.py        # Training script
├── models/personalized/         # Trained models
├── logs/dpo_training/           # Training logs
└── requirements_dpo.txt         # DPO dependencies
```

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Run diagnostic tests: `python test_dpo_phase2.py`
3. Review training logs in `./logs/dpo_training/`
4. Ensure all dependencies are correctly installed

## Next Steps

After successful installation:

1. **Collect Data**: Use SAM and provide feedback regularly
2. **Monitor Quality**: Check preference data quality in the UI
3. **Train Models**: Start with small experiments
4. **Evaluate Results**: Compare personalized vs. base model responses
5. **Iterate**: Refine based on results and collect more data

The SAM Personalized Tuner represents a significant advancement in AI personalization, enabling models to learn and adapt to individual user preferences automatically.
