#!/usr/bin/env python3
"""
SSRL Phase 1 Test Suite
=======================

Comprehensive test suite for validating the Phase 1 SSRL implementation:
- SelfSearchTool functionality and safety mechanisms
- HybridQueryRouter 4-stage routing logic
- Integration layer with existing SAM systems

Author: SAM Development Team
Version: 1.0.0
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any

# Add SAM to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SSRLPhase1TestSuite:
    """Comprehensive test suite for SSRL Phase 1 implementation."""
    
    def __init__(self):
        """Initialize test suite."""
        self.test_results = {
            'self_search_tool': {},
            'hybrid_router': {},
            'integration_layer': {},
            'end_to_end': {}
        }
        
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
    
    def run_all_tests(self) -> bool:
        """
        Run all Phase 1 tests.
        
        Returns:
            True if all tests pass, False otherwise
        """
        print("🚀 SSRL Phase 1 Test Suite")
        print("=" * 50)
        
        # Test 1: SelfSearchTool
        print("\n📋 Testing SelfSearchTool...")
        self_search_success = self.test_self_search_tool()
        
        # Test 2: HybridQueryRouter
        print("\n📋 Testing HybridQueryRouter...")
        router_success = self.test_hybrid_router()
        
        # Test 3: Integration Layer
        print("\n📋 Testing Integration Layer...")
        integration_success = self.test_integration_layer()
        
        # Test 4: End-to-End
        print("\n📋 Testing End-to-End Workflow...")
        e2e_success = self.test_end_to_end()
        
        # Summary
        self.print_test_summary()
        
        return all([self_search_success, router_success, integration_success, e2e_success])
    
    def test_self_search_tool(self) -> bool:
        """Test SelfSearchTool functionality."""
        try:
            from sam.orchestration.skills.self_search_tool import get_self_search_tool, SSRLResult
            
            tool = get_self_search_tool()
            
            # Test 1: Basic functionality
            print("  🧪 Test 1: Basic self-search execution")
            result = tool.execute("What is the capital of France?")
            
            success = isinstance(result, SSRLResult) and result.success
            self._record_test("self_search_tool", "basic_execution", success)
            
            if success:
                print(f"    ✅ Success: {result.confidence_score:.2f} confidence")
                print(f"    📊 Reasoning steps: {len(result.reasoning_steps)}")
            else:
                print(f"    ❌ Failed: {result.error if result else 'No result'}")
            
            # Test 2: Safety mechanisms
            print("  🧪 Test 2: Infinite loop prevention")
            context_with_flag = {'is_self_search': True}
            safety_result = tool.execute("Test query", context_with_flag)
            
            safety_success = not safety_result.success and "nested" in safety_result.error.lower()
            self._record_test("self_search_tool", "safety_mechanisms", safety_success)
            
            if safety_success:
                print("    ✅ Safety mechanism working")
            else:
                print("    ❌ Safety mechanism failed")
            
            # Test 3: Confidence assessment
            print("  🧪 Test 3: Confidence assessment")
            confidence_test = result.confidence_score >= 0.0 and result.confidence_score <= 1.0
            self._record_test("self_search_tool", "confidence_assessment", confidence_test)
            
            if confidence_test:
                print(f"    ✅ Valid confidence score: {result.confidence_score:.2f}")
            else:
                print(f"    ❌ Invalid confidence score: {result.confidence_score}")
            
            return success and safety_success and confidence_test
            
        except Exception as e:
            print(f"    ❌ SelfSearchTool test failed: {e}")
            self._record_test("self_search_tool", "exception", False, str(e))
            return False
    
    def test_hybrid_router(self) -> bool:
        """Test HybridQueryRouter functionality."""
        try:
            from sam.orchestration.hybrid_query_router import get_hybrid_query_router, RoutingDecision
            
            router = get_hybrid_query_router()
            
            # Test 1: Fast-path routing (math)
            print("  🧪 Test 1: Fast-path math routing")
            math_result = router.route_query("2 + 2")
            
            math_success = (math_result.success and 
                          math_result.decision == RoutingDecision.CALCULATOR)
            self._record_test("hybrid_router", "fast_path_math", math_success)
            
            if math_success:
                print("    ✅ Math query routed to calculator")
            else:
                print(f"    ❌ Math routing failed: {math_result.decision}")
            
            # Test 2: CSV context routing
            print("  🧪 Test 2: CSV context routing")
            csv_context = {'uploaded_csv_files': {'test.csv': '/path/to/test.csv'}}
            csv_result = router.route_query("Calculate average salary", csv_context)
            
            csv_success = (csv_result.success and 
                          csv_result.decision == RoutingDecision.CODE_INTERPRETER)
            self._record_test("hybrid_router", "csv_routing", csv_success)
            
            if csv_success:
                print("    ✅ CSV query routed to code interpreter")
            else:
                print(f"    ❌ CSV routing failed: {csv_result.decision}")
            
            # Test 3: Self-search routing
            print("  🧪 Test 3: Self-search routing")
            general_result = router.route_query("Explain quantum computing")
            
            # Should either succeed with self-search or escalate to external
            self_search_success = general_result.success
            self._record_test("hybrid_router", "self_search_routing", self_search_success)
            
            if self_search_success:
                print(f"    ✅ General query handled: {general_result.decision}")
            else:
                print(f"    ❌ General query failed: {general_result.error}")
            
            # Test 4: Statistics tracking
            print("  🧪 Test 4: Statistics tracking")
            stats = router.get_routing_stats()
            stats_success = 'total_queries' in stats and stats['total_queries'] >= 3
            self._record_test("hybrid_router", "statistics", stats_success)
            
            if stats_success:
                print(f"    ✅ Statistics tracked: {stats['total_queries']} queries")
            else:
                print("    ❌ Statistics tracking failed")
            
            return math_success and csv_success and self_search_success and stats_success
            
        except Exception as e:
            print(f"    ❌ HybridQueryRouter test failed: {e}")
            self._record_test("hybrid_router", "exception", False, str(e))
            return False
    
    def test_integration_layer(self) -> bool:
        """Test SSRL integration layer."""
        try:
            from sam.orchestration.ssrl_integration import get_ssrl_integration
            
            integration = get_ssrl_integration()
            
            # Test 1: Basic integration
            print("  🧪 Test 1: Basic integration processing")
            response, metadata = integration.process_query_with_ssrl("What is AI?")
            
            basic_success = isinstance(response, str) and isinstance(metadata, dict)
            self._record_test("integration_layer", "basic_processing", basic_success)
            
            if basic_success:
                print(f"    ✅ Integration processing works")
                print(f"    📊 Metadata keys: {list(metadata.keys())}")
            else:
                print("    ❌ Integration processing failed")
            
            # Test 2: Fallback mechanism
            print("  🧪 Test 2: Fallback to existing system")
            fallback_response, fallback_metadata = integration.process_query_with_ssrl(
                "Test query", force_existing_system=True
            )
            
            fallback_success = (isinstance(fallback_response, str) and 
                              'EXISTING_SYSTEM' in fallback_response)
            self._record_test("integration_layer", "fallback_mechanism", fallback_success)
            
            if fallback_success:
                print("    ✅ Fallback mechanism works")
            else:
                print("    ❌ Fallback mechanism failed")
            
            # Test 3: Statistics tracking
            print("  🧪 Test 3: Integration statistics")
            stats = integration.get_integration_stats()
            stats_success = 'total_queries' in stats and stats['total_queries'] >= 2
            self._record_test("integration_layer", "statistics", stats_success)
            
            if stats_success:
                print(f"    ✅ Integration stats: {stats['total_queries']} queries")
            else:
                print("    ❌ Integration statistics failed")
            
            return basic_success and fallback_success and stats_success
            
        except Exception as e:
            print(f"    ❌ Integration layer test failed: {e}")
            self._record_test("integration_layer", "exception", False, str(e))
            return False
    
    def test_end_to_end(self) -> bool:
        """Test end-to-end SSRL workflow."""
        try:
            from sam.orchestration.ssrl_integration import process_query_with_ssrl_integration
            
            # Test different query types
            test_queries = [
                ("5 * 8", "calculator"),
                ("What is machine learning?", "general"),
                ("Explain the theory of relativity", "complex")
            ]
            
            all_success = True
            
            for i, (query, query_type) in enumerate(test_queries, 1):
                print(f"  🧪 Test {i}: End-to-end {query_type} query")
                
                start_time = time.time()
                response, metadata = process_query_with_ssrl_integration(query)
                execution_time = time.time() - start_time
                
                success = isinstance(response, str) and len(response) > 0
                self._record_test("end_to_end", f"{query_type}_query", success)
                
                if success:
                    print(f"    ✅ {query_type.title()} query processed ({execution_time:.2f}s)")
                    print(f"    🔧 Tool used: {metadata.get('tool_used', 'unknown')}")
                else:
                    print(f"    ❌ {query_type.title()} query failed")
                    all_success = False
            
            return all_success
            
        except Exception as e:
            print(f"    ❌ End-to-end test failed: {e}")
            self._record_test("end_to_end", "exception", False, str(e))
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
        print("🎯 SSRL PHASE 1 TEST SUMMARY")
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
            print(f"\n🎉 ALL TESTS PASSED! SSRL Phase 1 is ready for deployment!")
        else:
            print(f"\n⚠️ {self.failed_tests} tests failed. Review and fix issues before deployment.")


def main():
    """Run the SSRL Phase 1 test suite."""
    test_suite = SSRLPhase1TestSuite()
    success = test_suite.run_all_tests()
    
    if success:
        print(f"\n🚀 SSRL Phase 1 implementation is READY!")
        return 0
    else:
        print(f"\n🔧 SSRL Phase 1 needs fixes before deployment.")
        return 1


if __name__ == "__main__":
    exit(main())
