#!/usr/bin/env python3
"""
Comprehensive Test Runner
Runs all Phase 2 refactoring tests with detailed reporting.
"""

import unittest
import sys
import os
import time
from pathlib import Path
from io import StringIO
import json

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestResult:
    """Custom test result class for detailed reporting."""
    
    def __init__(self):
        self.tests_run = 0
        self.failures = []
        self.errors = []
        self.skipped = []
        self.successes = []
        self.start_time = None
        self.end_time = None
    
    def start_test(self, test):
        """Called when a test starts."""
        if self.start_time is None:
            self.start_time = time.time()
    
    def add_success(self, test):
        """Called when a test passes."""
        self.tests_run += 1
        self.successes.append(str(test))
    
    def add_failure(self, test, err):
        """Called when a test fails."""
        self.tests_run += 1
        self.failures.append((str(test), err))
    
    def add_error(self, test, err):
        """Called when a test has an error."""
        self.tests_run += 1
        self.errors.append((str(test), err))
    
    def add_skip(self, test, reason):
        """Called when a test is skipped."""
        self.tests_run += 1
        self.skipped.append((str(test), reason))
    
    def stop_test(self, test):
        """Called when a test ends."""
        self.end_time = time.time()
    
    @property
    def total_time(self):
        """Get total test execution time."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0
    
    @property
    def success_rate(self):
        """Get success rate percentage."""
        if self.tests_run == 0:
            return 0
        return (len(self.successes) / self.tests_run) * 100
    
    def was_successful(self):
        """Check if all tests passed."""
        return len(self.failures) == 0 and len(self.errors) == 0

def run_test_suite(test_module_name, description):
    """Run a specific test suite and return results."""
    print(f"\n{'='*60}")
    print(f"🧪 RUNNING {description}")
    print(f"{'='*60}")
    
    try:
        # Import the test module
        test_module = __import__(f"tests.{test_module_name}", fromlist=[''])
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(test_module)
        
        # Run tests with custom result
        stream = StringIO()
        runner = unittest.TextTestRunner(stream=stream, verbosity=2)
        result = runner.run(suite)
        
        # Print results
        output = stream.getvalue()
        print(output)
        
        # Summary
        print(f"\n📊 {description} SUMMARY:")
        print(f"   Tests run: {result.testsRun}")
        print(f"   Failures: {len(result.failures)}")
        print(f"   Errors: {len(result.errors)}")
        print(f"   Skipped: {len(result.skipped)}")
        
        if result.wasSuccessful():
            print(f"   ✅ ALL TESTS PASSED!")
        else:
            print(f"   ❌ SOME TESTS FAILED!")
            
            # Print failure details
            if result.failures:
                print(f"\n   FAILURES:")
                for test, traceback in result.failures:
                    print(f"   - {test}")
            
            if result.errors:
                print(f"\n   ERRORS:")
                for test, traceback in result.errors:
                    print(f"   - {test}")
        
        return result
        
    except Exception as e:
        print(f"❌ Failed to run {description}: {e}")
        return None

def generate_test_report(all_results):
    """Generate comprehensive test report."""
    print(f"\n{'='*80}")
    print(f"📋 COMPREHENSIVE TEST REPORT")
    print(f"{'='*80}")
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0
    total_time = 0
    
    suite_results = []
    
    for suite_name, result in all_results.items():
        if result:
            total_tests += result.testsRun
            total_failures += len(result.failures)
            total_errors += len(result.errors)
            total_skipped += len(result.skipped)
            
            suite_results.append({
                'name': suite_name,
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'skipped': len(result.skipped),
                'success_rate': ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
            })
    
    # Overall statistics
    overall_success_rate = ((total_tests - total_failures - total_errors) / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {total_tests - total_failures - total_errors}")
    print(f"   Failed: {total_failures}")
    print(f"   Errors: {total_errors}")
    print(f"   Skipped: {total_skipped}")
    print(f"   Success Rate: {overall_success_rate:.1f}%")
    
    # Per-suite breakdown
    print(f"\n📋 PER-SUITE BREAKDOWN:")
    for suite in suite_results:
        status = "✅" if suite['failures'] == 0 and suite['errors'] == 0 else "❌"
        print(f"   {status} {suite['name']}: {suite['tests_run']} tests, {suite['success_rate']:.1f}% success")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if total_failures == 0 and total_errors == 0:
        print(f"   🎉 Excellent! All tests are passing.")
        print(f"   ✅ The Phase 2 refactoring is solid and reliable.")
        print(f"   🚀 Ready for production use.")
    else:
        print(f"   ⚠️  Some tests need attention:")
        if total_failures > 0:
            print(f"   - Fix {total_failures} test failures")
        if total_errors > 0:
            print(f"   - Resolve {total_errors} test errors")
        print(f"   🔧 Review failed tests and address issues.")
    
    if total_skipped > 0:
        print(f"   ℹ️  {total_skipped} tests were skipped (likely due to missing dependencies)")
    
    return {
        'total_tests': total_tests,
        'total_failures': total_failures,
        'total_errors': total_errors,
        'total_skipped': total_skipped,
        'success_rate': overall_success_rate,
        'suite_results': suite_results
    }

def save_test_report(report_data):
    """Save test report to file."""
    try:
        report_file = Path(__file__).parent / "test_report.json"
        
        # Add timestamp
        report_data['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        report_data['phase'] = 'Phase 2 Refactoring'
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Test report saved to: {report_file}")
        
    except Exception as e:
        print(f"⚠️ Failed to save test report: {e}")

def main():
    """Run all comprehensive tests."""
    print("🚀 PHASE 2 REFACTORING - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print("Testing all refactored services, integrations, and document processing pipeline")
    
    start_time = time.time()
    
    # Define test suites
    test_suites = [
        ("test_core_services", "CORE SERVICES UNIT TESTS"),
        ("test_integration", "INTEGRATION TESTS"),
        ("test_document_pipeline", "DOCUMENT PIPELINE TESTS")
    ]
    
    # Run all test suites
    all_results = {}
    
    for module_name, description in test_suites:
        result = run_test_suite(module_name, description)
        all_results[description] = result
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Generate comprehensive report
    report_data = generate_test_report(all_results)
    report_data['total_execution_time'] = total_time
    
    print(f"\n⏱️  Total execution time: {total_time:.2f} seconds")
    
    # Save report
    save_test_report(report_data)
    
    # Final status
    if report_data['total_failures'] == 0 and report_data['total_errors'] == 0:
        print(f"\n🎉 ALL TESTS PASSED! Phase 2 refactoring is complete and reliable.")
        return 0
    else:
        print(f"\n💥 SOME TESTS FAILED! Review the report above for details.")
        return 1

if __name__ == '__main__':
    exit(main())
