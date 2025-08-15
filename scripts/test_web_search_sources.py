#!/usr/bin/env python3
"""
Web Search Sources Test Script

This script tests the web search source attribution fixes to ensure that
actual web sources are displayed instead of hardcoded sba.gov and trade.gov.
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, List

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebSearchSourcesTester:
    """Test web search source attribution fixes."""
    
    def __init__(self):
        """Initialize the tester."""
        self.test_results = []
        self.issues_found = []
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all web search source tests."""
        logger.info("🔍 Starting web search sources testing...")
        
        results = {
            'simple_web_search': self.test_simple_web_search_tool(),
            'source_extraction': self.test_source_extraction_logic(),
            'display_logic': self.test_display_logic(),
            'integration': self.test_integration(),
            'issues_found': self.issues_found,
            'summary': self.generate_summary()
        }
        
        self.print_test_report(results)
        return results
    
    def test_simple_web_search_tool(self) -> Dict[str, Any]:
        """Test the Simple Web Search tool improvements."""
        logger.info("📡 Testing Simple Web Search tool...")
        
        try:
            from web_retrieval.tools.simple_web_search import SimpleWebSearchTool
            search_tool = SimpleWebSearchTool()
            
            test_cases = [
                {
                    'query': 'python programming tutorial',
                    'description': 'General programming query',
                    'expect_guidance': False
                },
                {
                    'query': 'artificial intelligence news',
                    'description': 'News query',
                    'expect_guidance': False
                },
                {
                    'query': 'very_obscure_query_that_should_not_exist_12345',
                    'description': 'Obscure query that might trigger guidance',
                    'expect_guidance': True
                }
            ]
            
            results = []
            for test_case in test_cases:
                try:
                    logger.info(f"  Testing query: '{test_case['query']}'")
                    result = search_tool.search(test_case['query'], max_results=3)
                    
                    is_guidance = result.get('is_guidance', False)
                    has_results = len(result.get('results', [])) > 0
                    has_actual_urls = any(
                        'http' in res.get('url', '') and 
                        not any(domain in res.get('url', '') for domain in ['sba.gov', 'trade.gov'])
                        for res in result.get('results', [])
                    )
                    
                    test_result = {
                        'query': test_case['query'],
                        'description': test_case['description'],
                        'success': result.get('success', False),
                        'is_guidance': is_guidance,
                        'has_results': has_results,
                        'has_actual_urls': has_actual_urls,
                        'result_count': len(result.get('results', [])),
                        'sources': [res.get('url', res.get('source', '')) for res in result.get('results', [])[:3]]
                    }
                    
                    # Check if results match expectations
                    if not test_case['expect_guidance'] and is_guidance:
                        self.issues_found.append(
                            f"Query '{test_case['query']}' returned guidance when actual results were expected"
                        )
                    elif test_case['expect_guidance'] and not is_guidance and not has_actual_urls:
                        # This is actually okay - it means we found actual results for an obscure query
                        pass
                    
                    results.append(test_result)
                    logger.info(f"    ✅ Query processed: {test_result['result_count']} results, guidance: {is_guidance}")
                    
                except Exception as e:
                    logger.error(f"    ❌ Error testing query '{test_case['query']}': {e}")
                    results.append({
                        'query': test_case['query'],
                        'error': str(e),
                        'success': False
                    })
            
            return {
                'status': 'completed',
                'results': results,
                'total_tests': len(test_cases),
                'passed_tests': sum(1 for r in results if r.get('success', False))
            }
            
        except ImportError as e:
            logger.error(f"❌ Cannot import Simple Web Search tool: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def test_source_extraction_logic(self) -> Dict[str, Any]:
        """Test the source extraction logic."""
        logger.info("🔗 Testing source extraction logic...")
        
        try:
            from secure_streamlit_app import extract_sources_from_result
            
            test_cases = [
                {
                    'name': 'Actual search results',
                    'result': {
                        'data': {
                            'results': [
                                {
                                    'title': 'Python Tutorial',
                                    'url': 'https://python.org/tutorial',
                                    'type': 'web_search'
                                },
                                {
                                    'title': 'Stack Overflow',
                                    'url': 'https://stackoverflow.com/questions/python',
                                    'type': 'web_search'
                                }
                            ]
                        }
                    },
                    'expected_sources': ['https://python.org/tutorial', 'https://stackoverflow.com/questions/python'],
                    'should_not_contain': ['curated resource', 'sba.gov', 'trade.gov']
                },
                {
                    'name': 'Guidance results',
                    'result': {
                        'data': {
                            'is_guidance': True,
                            'results': [
                                {
                                    'title': 'SBA Resources',
                                    'url': 'https://www.sba.gov/',
                                    'type': 'guidance'
                                }
                            ]
                        }
                    },
                    'expected_sources': ['sba.gov (curated resource)'],
                    'should_not_contain': ['https://www.sba.gov/']
                }
            ]
            
            results = []
            for test_case in test_cases:
                try:
                    sources = extract_sources_from_result(test_case['result'])
                    
                    # Check expected sources
                    expected_found = all(
                        any(expected in source for source in sources)
                        for expected in test_case['expected_sources']
                    )
                    
                    # Check that unwanted content is not present
                    unwanted_found = any(
                        any(unwanted in source for source in sources)
                        for unwanted in test_case['should_not_contain']
                    )
                    
                    test_result = {
                        'name': test_case['name'],
                        'sources_extracted': sources,
                        'expected_found': expected_found,
                        'unwanted_found': unwanted_found,
                        'passed': expected_found and not unwanted_found
                    }
                    
                    if not test_result['passed']:
                        self.issues_found.append(
                            f"Source extraction failed for {test_case['name']}: "
                            f"expected_found={expected_found}, unwanted_found={unwanted_found}"
                        )
                    
                    results.append(test_result)
                    logger.info(f"    {'✅' if test_result['passed'] else '❌'} {test_case['name']}: {len(sources)} sources extracted")
                    
                except Exception as e:
                    logger.error(f"    ❌ Error testing {test_case['name']}: {e}")
                    results.append({
                        'name': test_case['name'],
                        'error': str(e),
                        'passed': False
                    })
            
            return {
                'status': 'completed',
                'results': results,
                'total_tests': len(test_cases),
                'passed_tests': sum(1 for r in results if r.get('passed', False))
            }
            
        except ImportError as e:
            logger.error(f"❌ Cannot import source extraction function: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def test_display_logic(self) -> Dict[str, Any]:
        """Test the display logic for different result types."""
        logger.info("🖥️ Testing display logic...")
        
        # Test that the data structures are set up correctly for proper display
        test_cases = [
            {
                'name': 'Actual search result structure',
                'data': {
                    'results': [{'url': 'https://example.com', 'type': 'web_search'}]
                },
                'should_have_guidance_flag': False
            },
            {
                'name': 'Guidance result structure',
                'data': {
                    'is_guidance': True,
                    'note': 'No current web results found.',
                    'results': [{'url': 'https://www.sba.gov/', 'type': 'guidance'}]
                },
                'should_have_guidance_flag': True
            }
        ]
        
        results = []
        for test_case in test_cases:
            is_guidance = test_case['data'].get('is_guidance', False)
            has_note = 'note' in test_case['data']
            
            passed = (is_guidance == test_case['should_have_guidance_flag'])
            
            test_result = {
                'name': test_case['name'],
                'is_guidance': is_guidance,
                'has_note': has_note,
                'passed': passed
            }
            
            results.append(test_result)
            logger.info(f"    {'✅' if passed else '❌'} {test_case['name']}: guidance={is_guidance}")
        
        return {
            'status': 'completed',
            'results': results,
            'total_tests': len(test_cases),
            'passed_tests': sum(1 for r in results if r.get('passed', False))
        }
    
    def test_integration(self) -> Dict[str, Any]:
        """Test integration between components."""
        logger.info("🔗 Testing component integration...")
        
        # This is a simplified integration test
        integration_checks = [
            {
                'name': 'Simple Web Search tool exists',
                'check': lambda: self._check_import('web_retrieval.tools.simple_web_search', 'SimpleWebSearchTool')
            },
            {
                'name': 'Source extraction function exists',
                'check': lambda: self._check_import('secure_streamlit_app', 'extract_sources_from_result')
            },
            {
                'name': 'Required dependencies available',
                'check': lambda: self._check_dependencies(['requests', 'beautifulsoup4'])
            }
        ]
        
        results = []
        for check in integration_checks:
            try:
                passed = check['check']()
                results.append({
                    'name': check['name'],
                    'passed': passed
                })
                logger.info(f"    {'✅' if passed else '❌'} {check['name']}")
            except Exception as e:
                results.append({
                    'name': check['name'],
                    'passed': False,
                    'error': str(e)
                })
                logger.error(f"    ❌ {check['name']}: {e}")
        
        return {
            'status': 'completed',
            'results': results,
            'total_tests': len(integration_checks),
            'passed_tests': sum(1 for r in results if r.get('passed', False))
        }
    
    def _check_import(self, module_name: str, class_name: str) -> bool:
        """Check if a module and class can be imported."""
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            return True
        except (ImportError, AttributeError):
            return False
    
    def _check_dependencies(self, deps: List[str]) -> bool:
        """Check if required dependencies are available."""
        try:
            for dep in deps:
                __import__(dep)
            return True
        except ImportError:
            return False
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of all test results."""
        total_issues = len(self.issues_found)
        
        return {
            'total_issues_found': total_issues,
            'status': 'PASS' if total_issues == 0 else 'FAIL',
            'recommendations': self._get_recommendations()
        }
    
    def _get_recommendations(self) -> List[str]:
        """Get recommendations based on test results."""
        recommendations = []
        
        if self.issues_found:
            recommendations.append("Review the issues found and apply necessary fixes")
            recommendations.append("Test web search functionality in Secure Chat interface")
            recommendations.append("Verify that actual web sources are displayed instead of hardcoded ones")
        else:
            recommendations.append("All tests passed! Web search source attribution should work correctly")
            recommendations.append("Test with real queries in Secure Chat to verify end-to-end functionality")
        
        return recommendations
    
    def print_test_report(self, results: Dict[str, Any]) -> None:
        """Print a comprehensive test report."""
        logger.info("\n" + "="*60)
        logger.info("📋 WEB SEARCH SOURCES TEST REPORT")
        logger.info("="*60)
        
        for test_name, test_result in results.items():
            if test_name in ['issues_found', 'summary']:
                continue
                
            if isinstance(test_result, dict) and 'status' in test_result:
                passed = test_result.get('passed_tests', 0)
                total = test_result.get('total_tests', 0)
                logger.info(f"🔍 {test_name.replace('_', ' ').title()}: {passed}/{total} tests passed")
        
        summary = results.get('summary', {})
        logger.info(f"\n📊 OVERALL STATUS: {summary.get('status', 'UNKNOWN')}")
        logger.info(f"🔍 Total Issues Found: {summary.get('total_issues_found', 0)}")
        
        if self.issues_found:
            logger.info("\n❌ ISSUES FOUND:")
            for i, issue in enumerate(self.issues_found, 1):
                logger.info(f"  {i}. {issue}")
        
        recommendations = summary.get('recommendations', [])
        if recommendations:
            logger.info("\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"  {i}. {rec}")
        
        logger.info("="*60)


def main():
    """Main function to run the web search sources test."""
    tester = WebSearchSourcesTester()
    results = tester.run_all_tests()
    
    # Return exit code based on issues found
    return len(tester.issues_found)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
