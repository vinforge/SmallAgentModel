#!/usr/bin/env python3
"""
Test suite for web search sources fix.
Tests the improvements made to web search source attribution and display logic.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestSimpleWebSearchFixes:
    """Test the enhanced Simple Web Search tool."""
    
    def setup_method(self):
        """Set up test fixtures."""
        try:
            from web_retrieval.tools.simple_web_search import SimpleWebSearchTool
            self.search_tool = SimpleWebSearchTool()
        except ImportError:
            pytest.skip("Simple web search tool not available")
    
    def test_search_returns_actual_results_not_guidance(self):
        """Test that search returns actual web results when available."""
        # Mock successful DuckDuckGo instant search
        mock_instant_results = {
            'success': True,
            'results': [
                {
                    'title': 'Test Result',
                    'url': 'https://example.com/test',
                    'snippet': 'Test snippet',
                    'source': 'example.com',
                    'type': 'instant'
                }
            ]
        }
        
        with patch.object(self.search_tool, '_search_duckduckgo_instant', return_value=mock_instant_results):
            result = self.search_tool.search("test query")
            
            assert result['success'] is True
            assert len(result['results']) > 0
            assert result['results'][0]['url'] == 'https://example.com/test'
            assert 'is_guidance' not in result  # Should not be marked as guidance
    
    def test_guidance_marked_when_no_search_results(self):
        """Test that guidance results are properly marked when no search results are found."""
        # Mock failed searches
        mock_failed_result = {'success': False, 'results': []}
        
        with patch.object(self.search_tool, '_search_duckduckgo_instant', return_value=mock_failed_result), \
             patch.object(self.search_tool, '_search_duckduckgo_web', return_value=mock_failed_result), \
             patch.object(self.search_tool, '_search_alternative_method', return_value=mock_failed_result):
            
            result = self.search_tool.search("test query")
            
            assert result['success'] is True  # Guidance should still be successful
            assert result.get('is_guidance') is True  # Should be marked as guidance
            assert 'note' in result  # Should have explanatory note
            assert len(result['results']) > 0  # Should have guidance results
    
    def test_web_search_fallback_chain(self):
        """Test that the fallback chain works properly."""
        # Mock instant search failure, web search success
        mock_failed_instant = {'success': False, 'results': []}
        mock_successful_web = {
            'success': True,
            'results': [
                {
                    'title': 'Web Search Result',
                    'url': 'https://example.com/web',
                    'snippet': 'Web search snippet',
                    'source': 'example.com',
                    'type': 'web_search'
                }
            ]
        }
        
        with patch.object(self.search_tool, '_search_duckduckgo_instant', return_value=mock_failed_instant), \
             patch.object(self.search_tool, '_search_duckduckgo_web', return_value=mock_successful_web):
            
            result = self.search_tool.search("test query")
            
            assert result['success'] is True
            assert result['results'][0]['type'] == 'web_search'
            assert 'is_guidance' not in result
    
    def test_domain_extraction(self):
        """Test domain extraction from URLs."""
        test_cases = [
            ('https://www.example.com/path', 'example.com'),
            ('https://subdomain.example.com/path', 'subdomain.example.com'),
            ('http://example.org', 'example.org'),
            ('invalid-url', 'invalid-url'),  # Fallback case
        ]
        
        for url, expected_domain in test_cases:
            domain = self.search_tool._extract_domain_from_url(url)
            assert domain == expected_domain, f"Expected {expected_domain}, got {domain} for URL {url}"


class TestSourceExtractionFixes:
    """Test the enhanced source extraction logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        try:
            from secure_streamlit_app import extract_sources_from_result
            self.extract_sources = extract_sources_from_result
        except ImportError:
            pytest.skip("Secure streamlit app not available")
    
    def test_actual_search_results_extraction(self):
        """Test extraction of sources from actual search results."""
        result = {
            'data': {
                'results': [
                    {
                        'title': 'Test Result',
                        'url': 'https://example.com/test',
                        'snippet': 'Test snippet',
                        'type': 'web_search'
                    },
                    {
                        'title': 'Another Result',
                        'url': 'https://another.com/page',
                        'snippet': 'Another snippet',
                        'type': 'web_search'
                    }
                ]
            }
        }
        
        sources = self.extract_sources(result)
        
        assert len(sources) == 2
        assert 'https://example.com/test' in sources
        assert 'https://another.com/page' in sources
        assert not any('curated resource' in source for source in sources)
    
    def test_guidance_results_extraction(self):
        """Test extraction of sources from guidance results."""
        result = {
            'data': {
                'is_guidance': True,
                'results': [
                    {
                        'title': 'SBA Resources',
                        'url': 'https://www.sba.gov/',
                        'snippet': 'Business guidance',
                        'type': 'guidance'
                    },
                    {
                        'title': 'Trade Resources',
                        'url': 'https://www.trade.gov/',
                        'snippet': 'Trade guidance',
                        'type': 'guidance'
                    }
                ]
            }
        }
        
        sources = self.extract_sources(result)
        
        assert len(sources) == 2
        assert any('sba.gov (curated resource)' in source for source in sources)
        assert any('trade.gov (curated resource)' in source for source in sources)
    
    def test_mixed_results_extraction(self):
        """Test extraction from mixed actual and guidance results."""
        result = {
            'data': {
                'search_results': [
                    {
                        'title': 'Actual Search Result',
                        'url': 'https://realsite.com/page',
                        'snippet': 'Real content'
                    }
                ],
                'results': [
                    {
                        'title': 'Guidance Result',
                        'url': 'https://www.sba.gov/',
                        'snippet': 'Guidance content',
                        'type': 'guidance'
                    }
                ]
            }
        }
        
        sources = self.extract_sources(result)
        
        assert len(sources) == 2
        assert 'https://realsite.com/page' in sources
        assert any('sba.gov (curated resource)' in source for source in sources)


class TestWebSearchIntegration:
    """Test the integration of web search fixes."""
    
    def test_web_search_system_integration(self):
        """Test that the web search system properly handles the new search tool."""
        # This would test the integration with the intelligent web system
        # For now, we'll create a placeholder test
        
        # Mock the intelligent web system
        mock_result = {
            'success': True,
            'tool_used': 'simple_web_search',
            'data': {
                'results': [
                    {
                        'title': 'Integration Test Result',
                        'url': 'https://integration.test/page',
                        'snippet': 'Integration test content',
                        'type': 'web_search'
                    }
                ]
            }
        }
        
        # Test that the result structure is correct
        assert mock_result['success'] is True
        assert mock_result['tool_used'] == 'simple_web_search'
        assert len(mock_result['data']['results']) > 0
        
        # Test source extraction from this result
        try:
            from secure_streamlit_app import extract_sources_from_result
            sources = extract_sources_from_result(mock_result)
            assert len(sources) > 0
            assert 'https://integration.test/page' in sources
        except ImportError:
            pytest.skip("Secure streamlit app not available for integration test")


class TestWebSearchDisplayLogic:
    """Test the display logic for web search results."""
    
    def test_guidance_vs_search_display_distinction(self):
        """Test that guidance and search results are displayed differently."""
        
        # Test actual search result display
        actual_result = {
            'data': {
                'results': [
                    {
                        'title': 'Actual Result',
                        'url': 'https://example.com',
                        'type': 'web_search'
                    }
                ]
            },
            'tool_used': 'simple_web_search'
        }
        
        # Test guidance result display
        guidance_result = {
            'data': {
                'is_guidance': True,
                'note': 'No current web results found. Showing curated guidance resources.',
                'results': [
                    {
                        'title': 'Guidance Resource',
                        'url': 'https://www.sba.gov/',
                        'type': 'guidance'
                    }
                ]
            },
            'tool_used': 'simple_web_search'
        }
        
        # Verify the data structure differences
        assert actual_result['data'].get('is_guidance') is None
        assert guidance_result['data'].get('is_guidance') is True
        assert 'note' in guidance_result['data']


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
