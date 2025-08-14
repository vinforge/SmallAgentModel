#!/usr/bin/env python3
"""
Test suite for document processing fixes.
Tests the improvements made to relevance scoring, query detection, and Deep Analysis functionality.
"""

import pytest
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestDocumentRelevanceScoring:
    """Test the enhanced relevance scoring system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        try:
            from sam.document_rag.enhanced_response_generator import EnhancedResponseGenerator
            self.generator = EnhancedResponseGenerator()
        except ImportError:
            pytest.skip("Enhanced response generator not available")
    
    def test_exact_match_scoring(self):
        """Test that exact matches get high relevance scores."""
        query = "2305.18290v3.pdf"
        content = "This document discusses the findings from 2305.18290v3.pdf which presents..."
        
        score = self.generator._calculate_relevance_to_query(content, query)
        assert score >= 0.9, f"Expected high score for exact match, got {score}"
    
    def test_filename_relevance_scoring(self):
        """Test that filename patterns are properly detected and scored."""
        query = "Deep Analysis: 2305.18290v3.pdf"
        content = "arXiv paper 2305.18290v3 discusses machine learning approaches..."
        
        score = self.generator._calculate_relevance_to_query(content, query)
        assert score >= 0.8, f"Expected high score for filename match, got {score}"
    
    def test_deep_analysis_relevance(self):
        """Test that deep analysis queries get appropriate relevance scores."""
        query = "🔍 Deep Analysis: research paper"
        content = "Abstract: This research study presents methodology and results for..."
        
        score = self.generator._calculate_relevance_to_query(content, query)
        assert score >= 0.5, f"Expected moderate score for deep analysis query, got {score}"
    
    def test_arxiv_pattern_detection(self):
        """Test that arXiv patterns are properly detected."""
        test_cases = [
            ("2305.18290v3.pdf", "Document 2305.18290v3 contains"),
            ("2305.18290v3", "arXiv:2305.18290v3 paper"),
            ("arxiv:2305.18290", "This is arxiv:2305.18290 research"),
        ]
        
        for query, content in test_cases:
            score = self.generator._calculate_relevance_to_query(content, query)
            assert score >= 0.8, f"Expected high score for arXiv pattern '{query}' in '{content}', got {score}"
    
    def test_zero_relevance_prevention(self):
        """Test that the fixes prevent inappropriate zero relevance scores."""
        # This was a common issue - documents showing 0.00 relevance
        query = "analyze the uploaded document"
        content = "This research paper presents a comprehensive analysis of machine learning techniques..."
        
        score = self.generator._calculate_relevance_to_query(content, query)
        assert score > 0.0, f"Expected non-zero score for analysis query, got {score}"


class TestDocumentQueryDetection:
    """Test the enhanced document query detection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Import the function we're testing
        sys.path.insert(0, str(project_root / "web_ui"))
        try:
            from app import is_document_query
            self.is_document_query = is_document_query
        except ImportError:
            pytest.skip("Web UI app not available")
    
    def test_deep_analysis_detection(self):
        """Test that Deep Analysis queries are properly detected."""
        test_queries = [
            "🔍 Deep Analysis: 2305.18290v3.pdf",
            "Deep Analysis: research paper",
            "analyze the uploaded document",
            "comprehensive analysis of the file",
            "detailed analysis needed"
        ]
        
        for query in test_queries:
            result = self.is_document_query(query)
            assert result, f"Expected '{query}' to be detected as document query"
    
    def test_arxiv_pattern_detection(self):
        """Test that arXiv patterns are detected as document queries."""
        test_queries = [
            "2305.18290v3.pdf",
            "2305.18290v3",
            "arxiv:2305.18290",
            "What is 2305.18290v3 about?",
            "Summarize 2305.18290v3.pdf"
        ]
        
        for query in test_queries:
            result = self.is_document_query(query)
            assert result, f"Expected '{query}' to be detected as document query"
    
    def test_mathematical_query_exclusion(self):
        """Test that mathematical queries are not treated as document queries."""
        test_queries = [
            "what is 5+4",
            "calculate 100-50",
            "15% of 100",
            "what is 2+2"
        ]
        
        for query in test_queries:
            result = self.is_document_query(query)
            assert not result, f"Expected '{query}' to NOT be detected as document query"
    
    def test_enhanced_document_indicators(self):
        """Test that enhanced document indicators are properly detected."""
        test_queries = [
            "analyze the uploaded file",
            "review the document",
            "examine the paper",
            "breakdown the content",
            "explain the uploaded document"
        ]
        
        for query in test_queries:
            result = self.is_document_query(query)
            assert result, f"Expected '{query}' to be detected as document query"


class TestPDFQueryDetection:
    """Test the enhanced PDF query detection in the proven integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        try:
            from sam.document_processing.proven_pdf_integration import SAMPDFIntegration
            self.integration = SAMPDFIntegration()
        except ImportError:
            pytest.skip("PDF integration not available")
    
    def test_deep_analysis_pdf_detection(self):
        """Test that Deep Analysis queries are detected as PDF queries."""
        test_queries = [
            "🔍 Deep Analysis: document.pdf",
            "deep analysis: research paper",
            "analyze the document",
            "comprehensive analysis",
            "examine the file"
        ]
        
        for query in test_queries:
            result = self.integration.is_pdf_query(query)
            assert result, f"Expected '{query}' to be detected as PDF query"
    
    def test_arxiv_pdf_detection(self):
        """Test that arXiv patterns are detected as PDF queries."""
        test_queries = [
            "2305.18290v3.pdf",
            "2305.18290v3",
            "arxiv:2305.18290",
            "2305.18290v2.pdf"
        ]
        
        for query in test_queries:
            result = self.integration.is_pdf_query(query)
            assert result, f"Expected '{query}' to be detected as PDF query"


class TestDeepAnalysisPromptGeneration:
    """Test the enhanced Deep Analysis prompt generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        try:
            from secure_streamlit_app import generate_enhanced_analysis_prompt
            self.generate_prompt = generate_enhanced_analysis_prompt
        except ImportError:
            pytest.skip("Secure streamlit app not available")
    
    def test_enhanced_prompt_includes_filename(self):
        """Test that the enhanced prompt includes explicit filename references."""
        filename = "2305.18290v3.pdf"
        prompt = self.generate_prompt(filename)
        
        # Check that the prompt includes the filename multiple times for better retrieval
        assert filename in prompt, "Prompt should include the filename"
        assert "🔍 Deep Analysis:" in prompt, "Prompt should include Deep Analysis marker"
        assert "knowledge base" in prompt, "Prompt should reference knowledge base"
        assert "uploaded document" in prompt, "Prompt should reference uploaded document"
    
    def test_prompt_includes_troubleshooting(self):
        """Test that the prompt includes troubleshooting guidance."""
        filename = "test_document.pdf"
        prompt = self.generate_prompt(filename)
        
        assert "cannot find this document" in prompt, "Prompt should include troubleshooting guidance"
        assert "troubleshooting steps" in prompt, "Prompt should suggest troubleshooting"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
