#!/usr/bin/env python3
"""
Document Processing Diagnostic Script

This script helps diagnose and fix issues with document processing,
particularly the problems with relevance scoring and Deep Analysis functionality.
"""

import sys
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessingDiagnostic:
    """Diagnostic tool for document processing issues."""
    
    def __init__(self):
        """Initialize the diagnostic tool."""
        self.issues_found = []
        self.fixes_applied = []
    
    def run_full_diagnostic(self) -> Dict[str, Any]:
        """Run a comprehensive diagnostic of document processing."""
        logger.info("🔍 Starting document processing diagnostic...")
        
        results = {
            'relevance_scoring': self.test_relevance_scoring(),
            'query_detection': self.test_query_detection(),
            'pdf_integration': self.test_pdf_integration(),
            'deep_analysis': self.test_deep_analysis_functionality(),
            'document_search': self.test_document_search(),
            'issues_found': self.issues_found,
            'fixes_applied': self.fixes_applied
        }
        
        self.generate_diagnostic_report(results)
        return results
    
    def test_relevance_scoring(self) -> Dict[str, Any]:
        """Test the relevance scoring system."""
        logger.info("📊 Testing relevance scoring system...")
        
        try:
            from sam.document_rag.enhanced_response_generator import EnhancedResponseGenerator
            generator = EnhancedResponseGenerator()
            
            test_cases = [
                {
                    'query': '2305.18290v3.pdf',
                    'content': 'This document discusses the findings from 2305.18290v3.pdf which presents...',
                    'expected_min_score': 0.9,
                    'description': 'Exact filename match'
                },
                {
                    'query': '🔍 Deep Analysis: research paper',
                    'content': 'Abstract: This research study presents methodology and results for...',
                    'expected_min_score': 0.5,
                    'description': 'Deep analysis query'
                },
                {
                    'query': 'analyze the uploaded document',
                    'content': 'This research paper presents a comprehensive analysis of machine learning techniques...',
                    'expected_min_score': 0.1,
                    'description': 'Generic analysis query'
                }
            ]
            
            results = []
            for test_case in test_cases:
                try:
                    score = generator._calculate_relevance_to_query(
                        test_case['content'], 
                        test_case['query']
                    )
                    
                    passed = score >= test_case['expected_min_score']
                    results.append({
                        'test': test_case['description'],
                        'query': test_case['query'],
                        'score': score,
                        'expected_min': test_case['expected_min_score'],
                        'passed': passed
                    })
                    
                    if not passed:
                        self.issues_found.append(
                            f"Relevance scoring issue: {test_case['description']} "
                            f"scored {score:.3f}, expected >= {test_case['expected_min_score']}"
                        )
                    
                    logger.info(f"  ✅ {test_case['description']}: {score:.3f} {'✓' if passed else '✗'}")
                    
                except Exception as e:
                    logger.error(f"  ❌ Error testing {test_case['description']}: {e}")
                    results.append({
                        'test': test_case['description'],
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
            logger.error(f"❌ Cannot import enhanced response generator: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def test_query_detection(self) -> Dict[str, Any]:
        """Test the document query detection system."""
        logger.info("🔍 Testing query detection system...")
        
        try:
            sys.path.insert(0, str(project_root / "web_ui"))
            from app import is_document_query
            
            test_cases = [
                # Should be detected as document queries
                ('🔍 Deep Analysis: 2305.18290v3.pdf', True, 'Deep Analysis with filename'),
                ('analyze the uploaded document', True, 'Generic analysis request'),
                ('2305.18290v3.pdf', True, 'arXiv filename'),
                ('summarize the document', True, 'Document summarization'),
                
                # Should NOT be detected as document queries
                ('what is 5+4', False, 'Mathematical query'),
                ('calculate 100-50', False, 'Calculation query'),
                ('hello world', False, 'Generic greeting'),
            ]
            
            results = []
            for query, expected, description in test_cases:
                try:
                    result = is_document_query(query)
                    passed = result == expected
                    
                    results.append({
                        'test': description,
                        'query': query,
                        'result': result,
                        'expected': expected,
                        'passed': passed
                    })
                    
                    if not passed:
                        self.issues_found.append(
                            f"Query detection issue: '{query}' returned {result}, expected {expected}"
                        )
                    
                    logger.info(f"  {'✅' if passed else '❌'} {description}: {result} {'✓' if passed else '✗'}")
                    
                except Exception as e:
                    logger.error(f"  ❌ Error testing '{query}': {e}")
                    results.append({
                        'test': description,
                        'query': query,
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
            logger.error(f"❌ Cannot import query detection function: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def test_pdf_integration(self) -> Dict[str, Any]:
        """Test the PDF integration query detection."""
        logger.info("📄 Testing PDF integration...")
        
        try:
            from sam.document_processing.proven_pdf_integration import SAMPDFIntegration
            integration = SAMPDFIntegration()
            
            test_cases = [
                ('🔍 Deep Analysis: document.pdf', True, 'Deep Analysis PDF query'),
                ('2305.18290v3.pdf', True, 'arXiv PDF filename'),
                ('analyze the document', True, 'Generic analysis'),
                ('hello world', False, 'Non-document query'),
            ]
            
            results = []
            for query, expected, description in test_cases:
                try:
                    result = integration.is_pdf_query(query)
                    passed = result == expected
                    
                    results.append({
                        'test': description,
                        'query': query,
                        'result': result,
                        'expected': expected,
                        'passed': passed
                    })
                    
                    if not passed:
                        self.issues_found.append(
                            f"PDF query detection issue: '{query}' returned {result}, expected {expected}"
                        )
                    
                    logger.info(f"  {'✅' if passed else '❌'} {description}: {result} {'✓' if passed else '✗'}")
                    
                except Exception as e:
                    logger.error(f"  ❌ Error testing '{query}': {e}")
                    results.append({
                        'test': description,
                        'query': query,
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
            logger.error(f"❌ Cannot import PDF integration: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def test_deep_analysis_functionality(self) -> Dict[str, Any]:
        """Test the Deep Analysis functionality."""
        logger.info("🧠 Testing Deep Analysis functionality...")
        
        try:
            from secure_streamlit_app import generate_enhanced_analysis_prompt
            
            filename = "2305.18290v3.pdf"
            prompt = generate_enhanced_analysis_prompt(filename)
            
            checks = [
                (filename in prompt, "Filename included in prompt"),
                ("🔍 Deep Analysis:" in prompt, "Deep Analysis marker present"),
                ("knowledge base" in prompt, "Knowledge base reference"),
                ("uploaded document" in prompt, "Uploaded document reference"),
                ("troubleshooting" in prompt, "Troubleshooting guidance"),
            ]
            
            results = []
            for check, description in checks:
                results.append({
                    'test': description,
                    'passed': check
                })
                
                if not check:
                    self.issues_found.append(f"Deep Analysis prompt issue: {description}")
                
                logger.info(f"  {'✅' if check else '❌'} {description}: {'✓' if check else '✗'}")
            
            return {
                'status': 'completed',
                'results': results,
                'total_tests': len(checks),
                'passed_tests': sum(1 for r in results if r.get('passed', False)),
                'sample_prompt_length': len(prompt)
            }
            
        except ImportError as e:
            logger.error(f"❌ Cannot import Deep Analysis function: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def test_document_search(self) -> Dict[str, Any]:
        """Test document search functionality."""
        logger.info("🔎 Testing document search...")
        
        # This is a placeholder for document search testing
        # In practice, you would test the actual search functionality
        return {
            'status': 'placeholder',
            'message': 'Document search testing requires uploaded documents'
        }
    
    def generate_diagnostic_report(self, results: Dict[str, Any]) -> None:
        """Generate a comprehensive diagnostic report."""
        logger.info("\n" + "="*60)
        logger.info("📋 DOCUMENT PROCESSING DIAGNOSTIC REPORT")
        logger.info("="*60)
        
        total_issues = len(self.issues_found)
        total_fixes = len(self.fixes_applied)
        
        logger.info(f"🔍 Issues Found: {total_issues}")
        logger.info(f"🛠️  Fixes Applied: {total_fixes}")
        
        if self.issues_found:
            logger.info("\n❌ ISSUES FOUND:")
            for i, issue in enumerate(self.issues_found, 1):
                logger.info(f"  {i}. {issue}")
        
        if self.fixes_applied:
            logger.info("\n✅ FIXES APPLIED:")
            for i, fix in enumerate(self.fixes_applied, 1):
                logger.info(f"  {i}. {fix}")
        
        logger.info("\n📊 TEST RESULTS SUMMARY:")
        for test_name, test_result in results.items():
            if isinstance(test_result, dict) and 'status' in test_result:
                if test_result['status'] == 'completed':
                    passed = test_result.get('passed_tests', 0)
                    total = test_result.get('total_tests', 0)
                    logger.info(f"  {test_name}: {passed}/{total} tests passed")
                else:
                    logger.info(f"  {test_name}: {test_result['status']}")
        
        logger.info("="*60)


def main():
    """Main function to run the diagnostic."""
    diagnostic = DocumentProcessingDiagnostic()
    results = diagnostic.run_full_diagnostic()
    
    # Return exit code based on issues found
    return len(diagnostic.issues_found)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
