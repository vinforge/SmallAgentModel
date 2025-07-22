"""
DNA Layer Training Data Generator
=================================

Generates representative training datasets for DNA layer fine-tuning.
Creates diverse scenarios that SAM typically encounters in production.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingExample:
    """Single training example for DNA layer."""
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor
    token_types: List[str]
    complexity_score: float
    scenario_type: str
    expected_routing: Optional[Dict[str, float]] = None


class SAMWorkloadGenerator:
    """
    Generates realistic SAM workloads for DNA layer training.
    
    Simulates the types of content and processing patterns that SAM
    encounters in real-world usage scenarios.
    """
    
    def __init__(self, hidden_size: int = 768, device: str = 'cpu'):
        self.hidden_size = hidden_size
        self.device = device
        
        # Define scenario templates
        self.scenario_templates = {
            'simple_qa': {
                'description': 'Simple question-answering interactions',
                'complexity_range': (0.1, 0.3),
                'token_distribution': {
                    'question': 0.15,
                    'simple': 0.60,
                    'content': 0.20,
                    'padding': 0.05
                },
                'expected_identity_usage': 0.4  # High efficiency expected
            },
            'technical_analysis': {
                'description': 'Technical documentation and analysis',
                'complexity_range': (0.6, 0.9),
                'token_distribution': {
                    'technical': 0.40,
                    'content': 0.35,
                    'complex': 0.20,
                    'padding': 0.05
                },
                'expected_identity_usage': 0.15  # Low efficiency, needs processing
            },
            'code_review': {
                'description': 'Code analysis and review tasks',
                'complexity_range': (0.5, 0.8),
                'token_distribution': {
                    'technical': 0.30,
                    'content': 0.30,
                    'simple': 0.25,
                    'complex': 0.10,
                    'padding': 0.05
                },
                'expected_identity_usage': 0.25  # Medium efficiency
            },
            'document_summarization': {
                'description': 'Document processing and summarization',
                'complexity_range': (0.4, 0.7),
                'token_distribution': {
                    'content': 0.50,
                    'simple': 0.25,
                    'technical': 0.15,
                    'complex': 0.05,
                    'padding': 0.05
                },
                'expected_identity_usage': 0.30  # Good efficiency
            },
            'conversational': {
                'description': 'Natural conversation and chat',
                'complexity_range': (0.2, 0.5),
                'token_distribution': {
                    'simple': 0.50,
                    'content': 0.30,
                    'question': 0.15,
                    'padding': 0.05
                },
                'expected_identity_usage': 0.35  # High efficiency
            },
            'research_analysis': {
                'description': 'Research paper analysis and synthesis',
                'complexity_range': (0.7, 1.0),
                'token_distribution': {
                    'technical': 0.35,
                    'complex': 0.30,
                    'content': 0.25,
                    'simple': 0.05,
                    'padding': 0.05
                },
                'expected_identity_usage': 0.10  # Very low efficiency, needs heavy processing
            }
        }
        
        # Token type characteristics
        self.token_characteristics = {
            'simple': {'variance': 0.3, 'mean_shift': 0.0},
            'content': {'variance': 0.8, 'mean_shift': 0.1},
            'technical': {'variance': 1.2, 'mean_shift': 0.2},
            'complex': {'variance': 1.5, 'mean_shift': 0.3},
            'question': {'variance': 0.6, 'mean_shift': -0.1},
            'padding': {'variance': 0.1, 'mean_shift': -0.5}
        }
    
    def generate_training_batch(
        self, 
        batch_size: int = 8,
        seq_len: int = 128,
        scenario_mix: Optional[Dict[str, float]] = None
    ) -> List[TrainingExample]:
        """
        Generate a batch of training examples.
        
        Args:
            batch_size: Number of examples to generate
            seq_len: Sequence length for each example
            scenario_mix: Distribution of scenarios (if None, uses uniform)
            
        Returns:
            List of training examples
        """
        if scenario_mix is None:
            scenario_mix = {scenario: 1.0 for scenario in self.scenario_templates.keys()}
        
        # Normalize scenario mix
        total_weight = sum(scenario_mix.values())
        scenario_mix = {k: v/total_weight for k, v in scenario_mix.items()}
        
        examples = []
        
        for _ in range(batch_size):
            # Select scenario based on mix
            scenario_type = np.random.choice(
                list(scenario_mix.keys()),
                p=list(scenario_mix.values())
            )
            
            example = self._generate_single_example(scenario_type, seq_len)
            examples.append(example)
        
        return examples
    
    def _generate_single_example(self, scenario_type: str, seq_len: int) -> TrainingExample:
        """Generate a single training example for the given scenario."""
        template = self.scenario_templates[scenario_type]
        
        # Generate complexity score
        complexity_min, complexity_max = template['complexity_range']
        complexity_score = np.random.uniform(complexity_min, complexity_max)
        
        # Generate token types based on distribution
        token_types = self._generate_token_sequence(template['token_distribution'], seq_len)
        
        # Generate hidden states based on token types and complexity
        hidden_states = self._generate_hidden_states(token_types, complexity_score)
        
        # Generate attention mask (some sequences might be shorter)
        actual_length = np.random.randint(int(seq_len * 0.7), seq_len + 1)
        attention_mask = torch.ones(seq_len)
        attention_mask[actual_length:] = 0
        
        # Expected routing based on scenario
        expected_routing = self._compute_expected_routing(template, complexity_score)
        
        return TrainingExample(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            token_types=token_types,
            complexity_score=complexity_score,
            scenario_type=scenario_type,
            expected_routing=expected_routing
        )
    
    def _generate_token_sequence(self, distribution: Dict[str, float], seq_len: int) -> List[str]:
        """Generate a sequence of token types based on distribution."""
        token_types = []
        
        for token_type, prob in distribution.items():
            count = int(seq_len * prob)
            token_types.extend([token_type] * count)
        
        # Fill remaining slots with 'content' tokens
        while len(token_types) < seq_len:
            token_types.append('content')
        
        # Shuffle to avoid patterns
        random.shuffle(token_types)
        
        return token_types[:seq_len]
    
    def _generate_hidden_states(self, token_types: List[str], complexity_score: float) -> torch.Tensor:
        """Generate hidden states based on token types and complexity."""
        seq_len = len(token_types)
        hidden_states = torch.zeros(seq_len, self.hidden_size)
        
        for i, token_type in enumerate(token_types):
            char = self.token_characteristics.get(token_type, self.token_characteristics['content'])
            
            # Base variance and mean shift
            variance = char['variance'] * (1 + complexity_score * 0.5)
            mean_shift = char['mean_shift'] * complexity_score
            
            # Generate token embedding
            token_embedding = torch.randn(self.hidden_size) * variance + mean_shift
            hidden_states[i] = token_embedding
        
        return hidden_states.unsqueeze(0)  # Add batch dimension
    
    def _compute_expected_routing(self, template: Dict, complexity_score: float) -> Dict[str, float]:
        """Compute expected expert routing based on scenario and complexity."""
        base_identity_usage = template['expected_identity_usage']
        
        # Adjust based on complexity (higher complexity = less identity usage)
        adjusted_identity = base_identity_usage * (1 - complexity_score * 0.3)
        adjusted_identity = max(0.05, min(0.6, adjusted_identity))  # Clamp to reasonable range
        
        # Distribute remaining probability among other experts
        remaining = 1 - adjusted_identity
        
        if complexity_score > 0.7:
            # High complexity: favor attention and MLP
            attention_prob = remaining * 0.5
            mlp_prob = remaining * 0.35
            norm_prob = remaining * 0.15
        elif complexity_score > 0.4:
            # Medium complexity: balanced distribution
            attention_prob = remaining * 0.35
            mlp_prob = remaining * 0.35
            norm_prob = remaining * 0.30
        else:
            # Low complexity: favor normalization
            attention_prob = remaining * 0.25
            mlp_prob = remaining * 0.25
            norm_prob = remaining * 0.50
        
        return {
            'attention': attention_prob,
            'mlp': mlp_prob,
            'identity': adjusted_identity,
            'normalization': norm_prob
        }
    
    def generate_validation_dataset(
        self, 
        num_examples: int = 100,
        seq_len: int = 128
    ) -> List[TrainingExample]:
        """Generate a balanced validation dataset."""
        examples_per_scenario = num_examples // len(self.scenario_templates)
        validation_examples = []
        
        for scenario_type in self.scenario_templates.keys():
            for _ in range(examples_per_scenario):
                example = self._generate_single_example(scenario_type, seq_len)
                validation_examples.append(example)
        
        # Shuffle the validation set
        random.shuffle(validation_examples)
        
        return validation_examples
    
    def save_dataset(self, examples: List[TrainingExample], filepath: str):
        """Save dataset to disk."""
        dataset_data = {
            'examples': [],
            'metadata': {
                'num_examples': len(examples),
                'hidden_size': self.hidden_size,
                'scenarios': list(self.scenario_templates.keys())
            }
        }
        
        for example in examples:
            example_data = {
                'hidden_states': example.hidden_states.tolist(),
                'attention_mask': example.attention_mask.tolist(),
                'token_types': example.token_types,
                'complexity_score': example.complexity_score,
                'scenario_type': example.scenario_type,
                'expected_routing': example.expected_routing
            }
            dataset_data['examples'].append(example_data)
        
        with open(filepath, 'w') as f:
            json.dump(dataset_data, f, indent=2)
        
        print(f"Dataset saved to {filepath}")
        print(f"  - {len(examples)} examples")
        print(f"  - {len(set(ex.scenario_type for ex in examples))} scenario types")
    
    def load_dataset(self, filepath: str) -> List[TrainingExample]:
        """Load dataset from disk."""
        with open(filepath, 'r') as f:
            dataset_data = json.load(f)
        
        examples = []
        for example_data in dataset_data['examples']:
            example = TrainingExample(
                hidden_states=torch.tensor(example_data['hidden_states']),
                attention_mask=torch.tensor(example_data['attention_mask']),
                token_types=example_data['token_types'],
                complexity_score=example_data['complexity_score'],
                scenario_type=example_data['scenario_type'],
                expected_routing=example_data['expected_routing']
            )
            examples.append(example)
        
        print(f"Dataset loaded from {filepath}")
        print(f"  - {len(examples)} examples")
        
        return examples


def create_sam_training_datasets(output_dir: str = "data/dna_training"):
    """Create comprehensive training and validation datasets for DNA layer."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    generator = SAMWorkloadGenerator()
    
    print("🧬 Generating DNA Layer Training Datasets")
    print("=" * 50)
    
    # Generate training dataset
    print("📚 Generating training dataset...")
    training_examples = generator.generate_training_batch(
        batch_size=500,  # Large training set
        seq_len=128
    )
    
    training_path = output_path / "training_dataset.json"
    generator.save_dataset(training_examples, str(training_path))
    
    # Generate validation dataset
    print("\n📊 Generating validation dataset...")
    validation_examples = generator.generate_validation_dataset(
        num_examples=100,
        seq_len=128
    )
    
    validation_path = output_path / "validation_dataset.json"
    generator.save_dataset(validation_examples, str(validation_path))
    
    # Generate test scenarios
    print("\n🧪 Generating test scenarios...")
    test_scenarios = {}
    
    for scenario_type in generator.scenario_templates.keys():
        scenario_examples = []
        for _ in range(20):  # 20 examples per scenario
            example = generator._generate_single_example(scenario_type, 128)
            scenario_examples.append(example)
        test_scenarios[scenario_type] = scenario_examples
    
    # Save test scenarios
    for scenario_type, examples in test_scenarios.items():
        scenario_path = output_path / f"test_{scenario_type}.json"
        generator.save_dataset(examples, str(scenario_path))
    
    print(f"\n✅ Dataset generation complete!")
    print(f"   - Training: {len(training_examples)} examples")
    print(f"   - Validation: {len(validation_examples)} examples")
    print(f"   - Test scenarios: {len(test_scenarios)} types")
    print(f"   - Output directory: {output_dir}")
    
    return training_examples, validation_examples, test_scenarios


if __name__ == "__main__":
    # Generate datasets
    create_sam_training_datasets()
