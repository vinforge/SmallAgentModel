#!/usr/bin/env python3
"""
Phase 3: A/B Testing Framework
Comprehensive validation of Active Reasoning Control vs Baseline vs Monitoring-Only
"""

import sys
import json
import time
import logging
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# Add SAM to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestArm(Enum):
    """Test arm configurations for A/B testing."""
    BASELINE = "A"           # No TPV - Control group
    MONITORING = "B"         # Phase 1 - Passive monitoring only
    ACTIVE_CONTROL = "C"     # Phase 2 - Active reasoning control

class PromptCategory(Enum):
    """Categories of test prompts for comprehensive validation."""
    SIMPLE_FACTUAL = "simple_factual"
    COMPLEX_ANALYSIS = "complex_analysis"
    SUMMARIZATION = "summarization"
    OPEN_ENDED = "open_ended"

@dataclass
class TestPrompt:
    """Structured test prompt for A/B validation."""
    prompt_id: str
    category: PromptCategory
    text: str
    expected_complexity: str  # "low", "medium", "high"
    expected_tokens: int      # Rough estimate for baseline comparison
    description: str

@dataclass
class TestResult:
    """Results from a single test execution."""
    prompt_id: str
    test_arm: TestArm
    end_to_end_latency_ms: float
    total_tokens_generated: int
    tpv_halt_reason: Optional[str]
    response_text: str
    timestamp: float
    error: Optional[str] = None
    
    # TPV-specific metrics (only for Arms B and C)
    tpv_enabled: bool = False
    tpv_steps: int = 0
    final_score: float = 0.0
    trigger_type: Optional[str] = None
    control_decision: Optional[str] = None

class BenchmarkDataset:
    """Curated benchmark dataset for A/B testing."""
    
    def __init__(self):
        self.prompts: List[TestPrompt] = []
        self._create_benchmark_prompts()
    
    def _create_benchmark_prompts(self):
        """Create comprehensive benchmark dataset."""
        
        # Simple Factual Retrieval (should halt quickly)
        simple_prompts = [
            TestPrompt("SF001", PromptCategory.SIMPLE_FACTUAL, 
                      "What is the capital of France?", "low", 10,
                      "Basic factual query - should complete very quickly"),
            TestPrompt("SF002", PromptCategory.SIMPLE_FACTUAL,
                      "What year was the iPhone first released?", "low", 15,
                      "Simple date fact - minimal reasoning required"),
            TestPrompt("SF003", PromptCategory.SIMPLE_FACTUAL,
                      "What is 15 * 24?", "low", 8,
                      "Basic arithmetic - should halt immediately"),
            TestPrompt("SF004", PromptCategory.SIMPLE_FACTUAL,
                      "Who wrote 'Romeo and Juliet'?", "low", 12,
                      "Literature fact - straightforward answer"),
            TestPrompt("SF005", PromptCategory.SIMPLE_FACTUAL,
                      "What is the chemical symbol for gold?", "low", 8,
                      "Chemistry fact - single token answer"),
        ]
        
        # Complex Analysis (should allow deeper thought but halt before rambling)
        complex_prompts = [
            TestPrompt("CA001", PromptCategory.COMPLEX_ANALYSIS,
                      "Compare the security implications of cloud storage versus local storage for sensitive business data.", 
                      "high", 300, "Multi-faceted security analysis requiring structured reasoning"),
            TestPrompt("CA002", PromptCategory.COMPLEX_ANALYSIS,
                      "Analyze the potential economic impacts of widespread AI adoption on employment in the next decade.",
                      "high", 350, "Economic analysis with multiple variables and timeframes"),
            TestPrompt("CA003", PromptCategory.COMPLEX_ANALYSIS,
                      "Evaluate the pros and cons of renewable energy transition for developing countries.",
                      "high", 280, "Policy analysis with economic and environmental considerations"),
            TestPrompt("CA004", PromptCategory.COMPLEX_ANALYSIS,
                      "Compare machine learning approaches for natural language processing: transformers vs RNNs vs CNNs.",
                      "high", 320, "Technical comparison requiring deep understanding"),
            TestPrompt("CA005", PromptCategory.COMPLEX_ANALYSIS,
                      "Assess the ethical implications of genetic engineering in agriculture.",
                      "high", 290, "Ethical analysis with scientific and social dimensions"),
        ]
        
        # Summarization Tasks (should stop once key points covered)
        summarization_prompts = [
            TestPrompt("SU001", PromptCategory.SUMMARIZATION,
                      "Summarize the key benefits and challenges of remote work based on recent studies.",
                      "medium", 150, "Structured summarization of known topic"),
            TestPrompt("SU002", PromptCategory.SUMMARIZATION,
                      "Provide a concise summary of the main features of Python programming language.",
                      "medium", 120, "Technical summarization with clear scope"),
            TestPrompt("SU003", PromptCategory.SUMMARIZATION,
                      "Summarize the major causes and effects of climate change.",
                      "medium", 180, "Scientific summarization with cause-effect structure"),
            TestPrompt("SU004", PromptCategory.SUMMARIZATION,
                      "Give a brief overview of the key principles of agile software development.",
                      "medium", 140, "Methodology summarization with clear principles"),
            TestPrompt("SU005", PromptCategory.SUMMARIZATION,
                      "Summarize the main differences between SQL and NoSQL databases.",
                      "medium", 160, "Technical comparison summary"),
        ]
        
        # Open-Ended/Creative (stress-test plateau detection)
        open_ended_prompts = [
            TestPrompt("OE001", PromptCategory.OPEN_ENDED,
                      "Describe your ideal future city and explain why it would be better than current cities.",
                      "medium", 250, "Creative vision with reasoning - tests plateau detection"),
            TestPrompt("OE002", PromptCategory.OPEN_ENDED,
                      "If you could redesign the internet from scratch, what would you change and why?",
                      "medium", 280, "Hypothetical redesign - open-ended but should converge"),
            TestPrompt("OE003", PromptCategory.OPEN_ENDED,
                      "Explain how you would teach someone to think more creatively.",
                      "medium", 220, "Meta-cognitive advice - tests reasoning about reasoning"),
            TestPrompt("OE004", PromptCategory.OPEN_ENDED,
                      "What would be the most important considerations for establishing a colony on Mars?",
                      "medium", 300, "Speculative planning with multiple constraints"),
            TestPrompt("OE005", PromptCategory.OPEN_ENDED,
                      "How might education change in the next 50 years, and what should we prepare for?",
                      "medium", 260, "Future prediction with preparation advice"),
        ]
        
        # Combine all prompts
        self.prompts = simple_prompts + complex_prompts + summarization_prompts + open_ended_prompts
        
        # Shuffle for randomized testing
        random.shuffle(self.prompts)
        
        logger.info(f"Created benchmark dataset with {len(self.prompts)} prompts:")
        for category in PromptCategory:
            count = len([p for p in self.prompts if p.category == category])
            logger.info(f"  {category.value}: {count} prompts")
    
    def get_prompts(self) -> List[TestPrompt]:
        """Get all benchmark prompts."""
        return self.prompts
    
    def get_prompts_by_category(self, category: PromptCategory) -> List[TestPrompt]:
        """Get prompts filtered by category."""
        return [p for p in self.prompts if p.category == category]
    
    def save_to_file(self, filepath: Path):
        """Save benchmark dataset to JSON file."""
        data = {
            'metadata': {
                'total_prompts': len(self.prompts),
                'categories': {cat.value: len(self.get_prompts_by_category(cat)) for cat in PromptCategory},
                'created_timestamp': time.time()
            },
            'prompts': [asdict(prompt) for prompt in self.prompts]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Benchmark dataset saved to {filepath}")

class ABTestingFramework:
    """Main A/B testing framework for Phase 3 validation."""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("ab_testing_results")
        self.output_dir.mkdir(exist_ok=True)
        
        self.dataset = BenchmarkDataset()
        self.results: List[TestResult] = []
        
        logger.info(f"A/B Testing Framework initialized")
        logger.info(f"Output directory: {self.output_dir}")
    
    def setup_test_environment(self):
        """Setup and validate test environment."""
        logger.info("🔧 Setting up A/B test environment...")
        
        try:
            # Test Arm A (Baseline) - No TPV
            logger.info("  Testing Arm A (Baseline)...")
            # This will use standard Ollama without TPV
            
            # Test Arm B (Monitoring) - Phase 1
            logger.info("  Testing Arm B (Monitoring)...")
            from sam.cognition.tpv import SAMTPVIntegration, ReasoningController, ControlMode
            integration_b = SAMTPVIntegration()
            integration_b.reasoning_controller.mode = ControlMode.PASSIVE
            
            # Test Arm C (Active Control) - Phase 2
            logger.info("  Testing Arm C (Active Control)...")
            integration_c = SAMTPVIntegration()
            integration_c.reasoning_controller.mode = ControlMode.ACTIVE
            
            logger.info("✅ All test arms validated")
            return True
            
        except Exception as e:
            logger.error(f"❌ Test environment setup failed: {e}")
            return False
    
    def save_benchmark_dataset(self):
        """Save the benchmark dataset for reproducibility."""
        dataset_file = self.output_dir / "benchmark_dataset.json"
        self.dataset.save_to_file(dataset_file)
        return dataset_file
    
    def get_test_configuration(self) -> Dict[str, Any]:
        """Get comprehensive test configuration for documentation."""
        return {
            'test_arms': {
                'A_BASELINE': {
                    'description': 'No TPV - Control group',
                    'tpv_enabled': False,
                    'control_mode': None
                },
                'B_MONITORING': {
                    'description': 'Phase 1 - Passive monitoring only',
                    'tpv_enabled': True,
                    'control_mode': 'PASSIVE'
                },
                'C_ACTIVE_CONTROL': {
                    'description': 'Phase 2 - Active reasoning control',
                    'tpv_enabled': True,
                    'control_mode': 'ACTIVE'
                }
            },
            'benchmark_dataset': {
                'total_prompts': len(self.dataset.prompts),
                'categories': {cat.value: len(self.dataset.get_prompts_by_category(cat)) for cat in PromptCategory}
            },
            'metrics_collected': [
                'end_to_end_latency_ms',
                'total_tokens_generated', 
                'tpv_halt_reason',
                'response_text',
                'tpv_steps',
                'final_score',
                'control_decision'
            ],
            'hypotheses': [
                'Efficiency: Active Control reduces latency and tokens without quality loss',
                'Quality: Active Control improves answers by preventing stagnation',
                'User Experience: Faster, more concise answers with transparency improve UX'
            ]
        }

def main():
    """Initialize A/B testing framework and save benchmark dataset."""
    logger.info("🚀 Initializing Phase 3: A/B Testing Framework")
    
    # Create framework
    framework = ABTestingFramework()
    
    # Setup test environment
    if not framework.setup_test_environment():
        logger.error("❌ Failed to setup test environment")
        return 1
    
    # Save benchmark dataset
    dataset_file = framework.save_benchmark_dataset()
    
    # Save test configuration
    config_file = framework.output_dir / "test_configuration.json"
    with open(config_file, 'w') as f:
        json.dump(framework.get_test_configuration(), f, indent=2)
    
    logger.info("✅ A/B Testing Framework ready!")
    logger.info(f"📊 Benchmark dataset: {dataset_file}")
    logger.info(f"⚙️ Test configuration: {config_file}")
    logger.info(f"🎯 Ready to run comprehensive A/B validation")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
