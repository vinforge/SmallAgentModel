"""
Web Retrieval Suggester - Phase 7.1 UI Integration

This module provides intelligent suggestions for web content retrieval
when SAM cannot answer questions from its local knowledge base.

Features:
- Automatic query-to-search-URL conversion
- Formatted CLI command generation
- Context-aware retrieval suggestions
- Integration with SAM's response logic

Security:
- Manual workflow prevents automatic web access
- Clear instructions for safe content review
- Quarantine-based content handling
"""

import urllib.parse
import re
from typing import Optional, List, Dict, Any
from datetime import datetime


class WebRetrievalSuggester:
    """
    Generates web retrieval suggestions when SAM lacks information.
    
    This class analyzes user queries and generates appropriate web search
    commands using SAM's manual web retrieval system.
    """
    
    def __init__(self):
        """Initialize the web retrieval suggester."""
        self.search_engines = {
            'google': 'https://www.google.com/search?q={}',
            'bing': 'https://www.bing.com/search?q={}',
            'duckduckgo': 'https://duckduckgo.com/?q={}',
            'scholar': 'https://scholar.google.com/scholar?q={}'
        }
        
        self.domain_suggestions = {
            'news': ['news.ycombinator.com', 'reuters.com', 'bbc.com'],
            'tech': ['stackoverflow.com', 'github.com', 'techcrunch.com'],
            'academic': ['arxiv.org', 'scholar.google.com', 'pubmed.ncbi.nlm.nih.gov'],
            'reference': ['wikipedia.org', 'britannica.com', 'merriam-webster.com']
        }
    
    def should_suggest_web_retrieval(self, query: str, context_results: List[Any]) -> bool:
        """
        Determine if web retrieval should be suggested.

        Args:
            query: User's query
            context_results: Results from local knowledge search

        Returns:
            True if web retrieval should be suggested
        """
        query_lower = query.lower()

        # CRITICAL FIX: Don't suggest web retrieval for basic queries that LLM can handle
        basic_query_indicators = [
            'tell me a joke', 'joke', 'funny', 'humor',
            'what is', 'calculate', 'compute', 'math', '+', '-', '*', '/',
            'hello', 'hi', 'how are you', 'good morning', 'good afternoon',
            'explain', 'define', 'meaning of', 'what does', 'how to',
            'write', 'create', 'generate', 'make', 'help me',
            'story', 'poem', 'essay', 'letter', 'email',
            'translate', 'language', 'grammar', 'spelling'
        ]

        # If it's a basic query the LLM can handle, don't suggest web retrieval
        if any(indicator in query_lower for indicator in basic_query_indicators):
            return False

        # Don't suggest for very short queries (likely basic questions)
        if len(query.split()) <= 3:
            return False

        # Only suggest web retrieval for specific types of queries that need current information
        web_retrieval_indicators = [
            'latest', 'recent', 'current', 'today', 'now', '2024', '2025',
            'news', 'breaking', 'update', 'announcement',
            'price', 'stock', 'weather', 'forecast',
            'who is', 'biography', 'profile of', 'information about',
            'research', 'study', 'paper', 'article',
            'company', 'organization', 'website', 'official'
        ]

        # Check if query explicitly needs web information
        needs_web_info = any(indicator in query_lower for indicator in web_retrieval_indicators)

        # Only suggest web retrieval if:
        # 1. Query explicitly needs web information AND
        # 2. No local results found OR very limited results
        if needs_web_info:
            if not context_results or len(context_results) == 0:
                return True
            if len(context_results) < 2:
                return True

        return False
    
    def generate_search_url(self, query: str, engine: str = 'google') -> str:
        """
        Generate search URL for the given query.
        
        Args:
            query: Search query
            engine: Search engine to use
            
        Returns:
            Formatted search URL
        """
        if engine not in self.search_engines:
            engine = 'google'
        
        # Clean and encode query
        clean_query = self._clean_query(query)
        encoded_query = urllib.parse.quote_plus(clean_query)
        
        return self.search_engines[engine].format(encoded_query)
    
    def generate_fetch_command(self, query: str, engine: str = 'google', 
                             timeout: int = 30) -> str:
        """
        Generate CLI command for web content retrieval.
        
        Args:
            query: Original user query
            engine: Search engine to use
            timeout: Timeout for fetch operation
            
        Returns:
            Complete CLI command string
        """
        search_url = self.generate_search_url(query, engine)
        
        command_parts = [
            'python scripts/fetch_web_content.py',
            f'"{search_url}"'
        ]
        
        if timeout != 30:
            command_parts.append(f'--timeout {timeout}')
        
        return ' '.join(command_parts)
    
    def suggest_specific_domains(self, query: str) -> List[str]:
        """
        Suggest specific domains based on query content.
        
        Args:
            query: User query
            
        Returns:
            List of suggested domain-specific URLs
        """
        suggestions = []
        query_lower = query.lower()
        
        # Check for academic/research queries
        academic_keywords = ['research', 'study', 'paper', 'academic', 'journal']
        if any(keyword in query_lower for keyword in academic_keywords):
            encoded_query = urllib.parse.quote_plus(query)
            suggestions.append(f'https://scholar.google.com/scholar?q={encoded_query}')
            suggestions.append(f'https://arxiv.org/search/?query={encoded_query}')
        
        # Check for technical queries
        tech_keywords = ['programming', 'code', 'software', 'api', 'library']
        if any(keyword in query_lower for keyword in tech_keywords):
            encoded_query = urllib.parse.quote_plus(query)
            suggestions.append(f'https://stackoverflow.com/search?q={encoded_query}')
            suggestions.append(f'https://github.com/search?q={encoded_query}')
        
        # Check for news queries
        news_keywords = ['news', 'breaking', 'latest', 'current events']
        if any(keyword in query_lower for keyword in news_keywords):
            encoded_query = urllib.parse.quote_plus(query)
            suggestions.append(f'https://news.ycombinator.com/search?q={encoded_query}')
        
        return suggestions
    
    def format_retrieval_suggestion(self, query: str, 
                                  include_alternatives: bool = True) -> str:
        """
        Format complete retrieval suggestion for user display.
        
        Args:
            query: Original user query
            include_alternatives: Whether to include alternative search options
            
        Returns:
            Formatted suggestion text with commands and instructions
        """
        primary_command = self.generate_fetch_command(query)
        
        suggestion_parts = [
            "🌐 **Web Retrieval Available**",
            "",
            "I don't have this information in my current knowledge base. "
            "To retrieve it from the web, you can use SAM's secure web fetching system:",
            "",
            "**Primary Search:**",
            f"```bash",
            f"{primary_command}",
            f"```"
        ]
        
        if include_alternatives:
            # Add alternative search engines
            alt_commands = []
            for engine in ['bing', 'duckduckgo']:
                alt_commands.append(self.generate_fetch_command(query, engine))
            
            if alt_commands:
                suggestion_parts.extend([
                    "",
                    "**Alternative Search Engines:**",
                    "```bash",
                    f"# Bing: {alt_commands[0]}",
                    f"# DuckDuckGo: {alt_commands[1]}",
                    "```"
                ])
            
            # Add domain-specific suggestions
            domain_suggestions = self.suggest_specific_domains(query)
            if domain_suggestions:
                suggestion_parts.extend([
                    "",
                    "**Specialized Sources:**"
                ])
                for i, url in enumerate(domain_suggestions[:3], 1):
                    suggestion_parts.append(f"```bash")
                    suggestion_parts.append(f"python scripts/fetch_web_content.py \"{url}\"")
                    suggestion_parts.append(f"```")
        
        # Add instructions
        suggestion_parts.extend([
            "",
            "**📋 Next Steps:**",
            "1. 🔄 Run one of the commands above in your terminal",
            "2. 📁 Check the `quarantine/` folder for downloaded content",
            "3. 🔍 Review the JSON file for safety and relevance",
            "4. 📤 If safe, upload through SAM's Documents interface",
            "5. 🔄 Ask your question again",
            "",
            "⚠️ **Security Note:** All web content is quarantined for manual review before use."
        ])
        
        return "\n".join(suggestion_parts)
    
    def _clean_query(self, query: str) -> str:
        """
        Clean query for web search.
        
        Args:
            query: Raw query string
            
        Returns:
            Cleaned query suitable for web search
        """
        # Remove common question words that don't help search
        stop_words = ['what', 'is', 'are', 'how', 'why', 'when', 'where', 'who']
        
        # Split into words
        words = query.lower().split()
        
        # Remove stop words from beginning
        while words and words[0] in stop_words:
            words.pop(0)
        
        # Remove question marks and other punctuation
        cleaned = ' '.join(words)
        cleaned = re.sub(r'[?!.,;:]', '', cleaned)
        
        return cleaned.strip()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get suggester statistics and configuration."""
        return {
            'available_search_engines': list(self.search_engines.keys()),
            'domain_categories': list(self.domain_suggestions.keys()),
            'total_domain_suggestions': sum(len(domains) for domains in self.domain_suggestions.values())
        }


# Global instance
_web_retrieval_suggester = None

def get_web_retrieval_suggester() -> WebRetrievalSuggester:
    """Get or create a global web retrieval suggester instance."""
    global _web_retrieval_suggester

    if _web_retrieval_suggester is None:
        _web_retrieval_suggester = WebRetrievalSuggester()

    return _web_retrieval_suggester
