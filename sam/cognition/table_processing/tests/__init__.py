"""
Test Suite for Table Processing Module
=====================================

Comprehensive test suite for all table processing components including
unit tests, integration tests, and performance benchmarks.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test discovery and execution utilities
def run_all_tests():
    """Run all tests in the test suite."""
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_specific_test(test_module: str):
    """Run tests from a specific module."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(test_module)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

# Test categories
TEST_CATEGORIES = {
    "unit": ["test_parser", "test_classifier", "test_validator", "test_utils"],
    "integration": ["test_integration"],
    "performance": ["test_performance"],
    "end_to_end": ["test_e2e"]
}

def run_test_category(category: str):
    """Run tests from a specific category."""
    if category not in TEST_CATEGORIES:
        raise ValueError(f"Unknown test category: {category}")
    
    success = True
    for test_module in TEST_CATEGORIES[category]:
        try:
            module_success = run_specific_test(test_module)
            success = success and module_success
        except Exception as e:
            print(f"Failed to run {test_module}: {e}")
            success = False
    
    return success

if __name__ == "__main__":
    # Run all tests when executed directly
    success = run_all_tests()
    sys.exit(0 if success else 1)
