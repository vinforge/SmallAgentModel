#!/usr/bin/env python3
"""
SSRL Phase 2 Test Suite
=======================

Comprehensive test suite for validating the Phase 2 SSRL implementation:
- SSRL reward functions (format, outcome, combined)
- SSRL training pipeline components
- Data management and collation
- Integration with existing infrastructure

Author: SAM Development Team
Version: 1.0.0
"""

import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, Any

# Add SAM to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SSRLPhase2TestSuite:
    """Comprehensive test suite for SSRL Phase 2 implementation."""
    
    def __init__(self):
        """Initialize test suite."""
        self.test_results = {
            'reward_functions': {},
            'training_components': {},
            'data_management': {},
            'integration': {}
        }
        
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
    
    def run_all_tests(self) -> bool:
        """
        Run all Phase 2 tests.
        
        Returns:
            True if all tests pass, False otherwise
        """
        print("🚀 SSRL Phase 2 Test Suite")
        print("=" * 50)
        
        # Test 1: Reward Functions
        print("\n📋 Testing SSRL Reward Functions...")
        reward_success = self.test_reward_functions()
        
        # Test 2: Training Components
        print("\n📋 Testing Training Components...")
        training_success = self.test_training_components()
        
        # Test 3: Data Management
        print("\n📋 Testing Data Management...")
        data_success = self.test_data_management()
        
        # Test 4: Integration
        print("\n📋 Testing Integration...")
        integration_success = self.test_integration()
        
        # Summary
        self.print_test_summary()
        
        return all([reward_success, training_success, data_success, integration_success])
    
    def test_reward_functions(self) -> bool:
        """Test SSRL reward functions."""
        try:
            from sam.learning.ssrl_rewards import (
                get_format_reward, get_outcome_reward, get_combined_reward,
                SSRLRewardType
            )
            
            # Test 1: Format Reward
            print("  🧪 Test 1: Format reward calculation")
            format_reward = get_format_reward()
            
            # Test with well-structured SSRL response
            good_response = """<think>
This is a factual question about geography. I need to recall the capital of France.
</think>

<search>
France is a country in Western Europe. Its capital is Paris.
</search>

<information>
Paris is the capital and largest city of France, located in the north-central part of the country.
</information>

<confidence>
0.95
</confidence>

<answer>
The capital of France is Paris.
</answer>"""
            
            format_result = format_reward.calculate_reward(good_response)
            format_success = (format_result.success and
                            format_result.reward_type == SSRLRewardType.FORMAT and
                            format_result.score > 0.7)
            
            self._record_test("reward_functions", "format_reward", format_success)
            
            if format_success:
                print(f"    ✅ Format reward: {format_result.score:.3f}")
            else:
                print(f"    ❌ Format reward failed: {format_result.explanation}")
            
            # Test 2: Outcome Reward
            print("  🧪 Test 2: Outcome reward calculation")
            outcome_reward = get_outcome_reward()
            
            generated_answer = "The capital of France is Paris."
            ground_truth = "Paris"
            question = "What is the capital of France?"
            
            outcome_result = outcome_reward.calculate_reward(
                generated_answer, ground_truth, question
            )
            
            outcome_success = (outcome_result.success and 
                             outcome_result.reward_type == SSRLRewardType.OUTCOME and
                             outcome_result.score > 0.7)
            
            self._record_test("reward_functions", "outcome_reward", outcome_success)
            
            if outcome_success:
                print(f"    ✅ Outcome reward: {outcome_result.score:.3f}")
            else:
                print(f"    ❌ Outcome reward failed: {outcome_result.explanation}")
            
            # Test 3: Combined Reward
            print("  🧪 Test 3: Combined reward calculation")
            combined_reward = get_combined_reward()
            
            combined_result = combined_reward.calculate_reward(
                generated_text=good_response,
                ground_truth=ground_truth,
                question=question
            )
            
            combined_success = (combined_result.success and 
                              combined_result.reward_type == SSRLRewardType.COMBINED and
                              combined_result.score > 0.7)
            
            self._record_test("reward_functions", "combined_reward", combined_success)
            
            if combined_success:
                print(f"    ✅ Combined reward: {combined_result.score:.3f}")
                print(f"    📊 Format: {combined_result.details['format_score']:.3f}, "
                      f"Outcome: {combined_result.details['outcome_score']:.3f}")
            else:
                print(f"    ❌ Combined reward failed: {combined_result.explanation}")
            
            # Test 4: Error Handling
            print("  🧪 Test 4: Error handling")
            error_result = format_reward.calculate_reward("Invalid response without tags")
            error_success = not error_result.success or error_result.score < 0.5
            
            self._record_test("reward_functions", "error_handling", error_success)
            
            if error_success:
                print("    ✅ Error handling works correctly")
            else:
                print("    ❌ Error handling failed")
            
            return format_success and outcome_success and combined_success and error_success
            
        except Exception as e:
            print(f"    ❌ Reward functions test failed: {e}")
            self._record_test("reward_functions", "exception", False, str(e))
            return False
    
    def test_training_components(self) -> bool:
        """Test SSRL training components."""
        try:
            from scripts.run_ssrl_tuning import (
                SSRLTrainingArguments, SSRLDataset, SSRLDataCollator, 
                SSRLRewardFunction
            )
            
            # Test 1: Training Arguments
            print("  🧪 Test 1: Training arguments")
            args = SSRLTrainingArguments()
            args_success = hasattr(args, 'model_name') and hasattr(args, 'learning_rate')
            
            self._record_test("training_components", "training_arguments", args_success)
            
            if args_success:
                print(f"    ✅ Training arguments initialized")
            else:
                print("    ❌ Training arguments failed")
            
            # Test 2: Dataset Creation
            print("  🧪 Test 2: Dataset creation")
            
            # Create temporary test data
            with tempfile.TemporaryDirectory() as temp_dir:
                test_data = [
                    {
                        "question": "What is 2+2?",
                        "answer": "4",
                        "context": ""
                    },
                    {
                        "question": "What is the capital of Japan?",
                        "answer": "Tokyo",
                        "context": ""
                    }
                ]
                
                data_file = Path(temp_dir) / "test_data.json"
                with open(data_file, 'w') as f:
                    json.dump(test_data, f)
                
                # Mock tokenizer for testing
                class MockTokenizer:
                    def __init__(self):
                        self.pad_token_id = 0
                        self.eos_token = "</s>"
                        self.pad_token = "<pad>"
                    
                    def __call__(self, text, **kwargs):
                        # Simple mock tokenization
                        tokens = text.split()[:kwargs.get('max_length', 100)]
                        input_ids = list(range(len(tokens)))
                        attention_mask = [1] * len(tokens)
                        
                        return {
                            'input_ids': torch.tensor([input_ids]),
                            'attention_mask': torch.tensor([attention_mask])
                        }
                
                mock_tokenizer = MockTokenizer()
                
                try:
                    dataset = SSRLDataset(
                        data_path=str(data_file),
                        tokenizer=mock_tokenizer,
                        max_length=512
                    )
                    
                    dataset_success = len(dataset) == 2
                    
                    # Test dataset item
                    if dataset_success:
                        item = dataset[0]
                        dataset_success = (
                            'input_ids' in item and 
                            'question' in item and 
                            'ground_truth' in item
                        )
                    
                except Exception as e:
                    print(f"    Dataset creation error: {e}")
                    dataset_success = False
            
            self._record_test("training_components", "dataset_creation", dataset_success)
            
            if dataset_success:
                print(f"    ✅ Dataset created with {len(dataset)} items")
            else:
                print("    ❌ Dataset creation failed")
            
            # Test 3: Data Collator
            print("  🧪 Test 3: Data collator")
            
            try:
                collator = SSRLDataCollator(mock_tokenizer, max_length=512)
                
                # Create mock batch
                mock_batch = [
                    {
                        'input_ids': torch.tensor([1, 2, 3]),
                        'attention_mask': torch.tensor([1, 1, 1]),
                        'question': 'Test question 1',
                        'ground_truth': 'Test answer 1',
                        'context': '',
                        'prompt': 'Test prompt 1'
                    },
                    {
                        'input_ids': torch.tensor([1, 2]),
                        'attention_mask': torch.tensor([1, 1]),
                        'question': 'Test question 2',
                        'ground_truth': 'Test answer 2',
                        'context': '',
                        'prompt': 'Test prompt 2'
                    }
                ]
                
                collated = collator(mock_batch)
                collator_success = (
                    'input_ids' in collated and 
                    'attention_mask' in collated and
                    'metadata' in collated and
                    collated['input_ids'].shape[0] == 2  # Batch size
                )
                
            except Exception as e:
                print(f"    Data collator error: {e}")
                collator_success = False
            
            self._record_test("training_components", "data_collator", collator_success)
            
            if collator_success:
                print("    ✅ Data collator working")
            else:
                print("    ❌ Data collator failed")
            
            # Test 4: Reward Function
            print("  🧪 Test 4: Reward function wrapper")
            
            try:
                reward_fn = SSRLRewardFunction()
                
                test_generated = ["<answer>Paris</answer>", "<answer>Tokyo</answer>"]
                test_questions = ["Capital of France?", "Capital of Japan?"]
                test_ground_truths = ["Paris", "Tokyo"]
                
                rewards = reward_fn(test_generated, test_questions, test_ground_truths)
                
                reward_fn_success = (
                    len(rewards) == 2 and
                    all(0.0 <= r <= 1.0 for r in rewards)
                )
                
            except Exception as e:
                print(f"    Reward function error: {e}")
                reward_fn_success = False
            
            self._record_test("training_components", "reward_function", reward_fn_success)
            
            if reward_fn_success:
                print(f"    ✅ Reward function: {rewards}")
            else:
                print("    ❌ Reward function failed")
            
            return args_success and dataset_success and collator_success and reward_fn_success
            
        except Exception as e:
            print(f"    ❌ Training components test failed: {e}")
            self._record_test("training_components", "exception", False, str(e))
            return False
    
    def test_data_management(self) -> bool:
        """Test data management capabilities."""
        try:
            # Test 1: Sample Data Generation
            print("  🧪 Test 1: Sample data generation")
            
            from scripts.run_ssrl_tuning import SSRLDataset
            
            class MockTokenizer:
                def __init__(self):
                    self.pad_token_id = 0
                    self.eos_token = "</s>"
                    self.pad_token = "<pad>"
                
                def __call__(self, text, **kwargs):
                    tokens = text.split()[:kwargs.get('max_length', 100)]
                    input_ids = list(range(len(tokens)))
                    attention_mask = [1] * len(tokens)
                    
                    return {
                        'input_ids': torch.tensor([input_ids]),
                        'attention_mask': torch.tensor([attention_mask])
                    }
            
            # Test with non-existent path (should create sample data)
            dataset = SSRLDataset(
                data_path="/non/existent/path",
                tokenizer=MockTokenizer(),
                max_length=512
            )
            
            sample_success = len(dataset) > 0
            
            self._record_test("data_management", "sample_data", sample_success)
            
            if sample_success:
                print(f"    ✅ Sample data generated: {len(dataset)} items")
            else:
                print("    ❌ Sample data generation failed")
            
            # Test 2: SSRL Prompt Creation
            print("  🧪 Test 2: SSRL prompt creation")
            
            if len(dataset) > 0:
                item = dataset[0]
                prompt = item.get('prompt', '')
                
                prompt_success = (
                    '<think>' in prompt and
                    '<search>' in prompt and
                    '<answer>' in prompt and
                    'User\'s question:' in prompt
                )
            else:
                prompt_success = False
            
            self._record_test("data_management", "prompt_creation", prompt_success)
            
            if prompt_success:
                print("    ✅ SSRL prompt structure correct")
            else:
                print("    ❌ SSRL prompt structure incorrect")
            
            return sample_success and prompt_success
            
        except Exception as e:
            print(f"    ❌ Data management test failed: {e}")
            self._record_test("data_management", "exception", False, str(e))
            return False
    
    def test_integration(self) -> bool:
        """Test integration with existing infrastructure."""
        try:
            # Test 1: Import Compatibility
            print("  🧪 Test 1: Import compatibility")
            
            try:
                from sam.learning.ssrl_rewards import get_combined_reward
                from scripts.run_ssrl_tuning import SSRLTrainingPipeline
                import_success = True
            except ImportError as e:
                print(f"    Import error: {e}")
                import_success = False
            
            self._record_test("integration", "imports", import_success)
            
            if import_success:
                print("    ✅ All imports successful")
            else:
                print("    ❌ Import failures detected")
            
            # Test 2: Configuration Compatibility
            print("  🧪 Test 2: Configuration compatibility")
            
            try:
                from scripts.run_ssrl_tuning import SSRLTrainingArguments
                args = SSRLTrainingArguments()
                
                config_success = (
                    hasattr(args, 'model_name') and
                    hasattr(args, 'learning_rate') and
                    hasattr(args, 'output_dir') and
                    hasattr(args, 'lora_r')
                )
            except Exception as e:
                print(f"    Configuration error: {e}")
                config_success = False
            
            self._record_test("integration", "configuration", config_success)
            
            if config_success:
                print("    ✅ Configuration compatibility verified")
            else:
                print("    ❌ Configuration compatibility issues")
            
            return import_success and config_success
            
        except Exception as e:
            print(f"    ❌ Integration test failed: {e}")
            self._record_test("integration", "exception", False, str(e))
            return False
    
    def _record_test(self, category: str, test_name: str, success: bool, error: str = None):
        """Record test result."""
        self.total_tests += 1
        if success:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
        
        self.test_results[category][test_name] = {
            'success': success,
            'error': error
        }
    
    def print_test_summary(self):
        """Print comprehensive test summary."""
        print("\n" + "=" * 50)
        print("🎯 SSRL PHASE 2 TEST SUMMARY")
        print("=" * 50)
        
        print(f"📊 Overall Results:")
        print(f"   Total Tests: {self.total_tests}")
        print(f"   Passed: {self.passed_tests} ✅")
        print(f"   Failed: {self.failed_tests} ❌")
        print(f"   Success Rate: {(self.passed_tests/self.total_tests)*100:.1f}%")
        
        print(f"\n📋 Detailed Results:")
        for category, tests in self.test_results.items():
            category_passed = sum(1 for t in tests.values() if t['success'])
            category_total = len(tests)
            print(f"   {category.replace('_', ' ').title()}: {category_passed}/{category_total}")
            
            for test_name, result in tests.items():
                status = "✅" if result['success'] else "❌"
                print(f"     {status} {test_name.replace('_', ' ').title()}")
                if not result['success'] and result['error']:
                    print(f"       Error: {result['error']}")
        
        if self.failed_tests == 0:
            print(f"\n🎉 ALL TESTS PASSED! SSRL Phase 2 is ready for training!")
        else:
            print(f"\n⚠️ {self.failed_tests} tests failed. Review and fix issues before training.")


def main():
    """Run the SSRL Phase 2 test suite."""
    test_suite = SSRLPhase2TestSuite()
    success = test_suite.run_all_tests()
    
    if success:
        print(f"\n🚀 SSRL Phase 2 implementation is READY for training!")
        return 0
    else:
        print(f"\n🔧 SSRL Phase 2 needs fixes before training.")
        return 1


if __name__ == "__main__":
    exit(main())
