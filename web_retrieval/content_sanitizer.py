"""
Content Sanitizer for Phase 7.2: Automated Vetting Engine

This module provides comprehensive content sanitization and security analysis
for web content retrieved through SAM's web fetching system.

Features:
- HTML tag removal and script sanitization
- Suspicious pattern detection (prompt injection, etc.)
- Content purity scoring
- Encoding and formatting anomaly detection
"""

from bs4 import BeautifulSoup
import re
import html
import unicodedata
from typing import Dict, Any, List, Tuple
import logging


class ContentSanitizer:
    """
    Advanced content sanitization for security analysis.
    
    This class handles the cleaning and security analysis of raw web content,
    detecting potential threats like prompt injection, malicious scripts,
    and other security risks.
    """
    
    def __init__(self):
        """Initialize the content sanitizer with security patterns."""
        self.logger = logging.getLogger(__name__)
        
        # Suspicious patterns that might indicate prompt injection or attacks
        self.suspicious_patterns = [
            # Prompt injection patterns
            r'ignore\s+previous\s+instructions',
            r'forget\s+everything\s+above',
            r'system\s*:\s*',
            r'assistant\s*:\s*',
            r'human\s*:\s*',
            r'user\s*:\s*',
            r'ai\s*:\s*',
            
            # Hidden instruction patterns
            r'<!--.*?-->',
            r'\[INST\].*?\[/INST\]',
            r'<\s*instruction\s*>.*?<\s*/\s*instruction\s*>',
            
            # Script and code injection
            r'<script[^>]*>.*?</script>',
            r'javascript\s*:',
            r'data\s*:\s*text/html',
            r'vbscript\s*:',
            r'onload\s*=',
            r'onerror\s*=',
            
            # Encoding attacks
            r'%[0-9a-fA-F]{2}',  # URL encoding
            r'&#[0-9]+;',        # HTML entities
            r'&[a-zA-Z]+;',      # Named HTML entities
            
            # Unusual formatting that might hide content
            r'\u200b|\u200c|\u200d|\ufeff',  # Zero-width characters
            r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]',  # Control characters
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [
            (pattern, re.compile(pattern, re.IGNORECASE | re.DOTALL))
            for pattern in self.suspicious_patterns
        ]
        
        # Dangerous HTML tags to remove
        self.dangerous_tags = [
            'script', 'style', 'iframe', 'frame', 'frameset',
            'object', 'embed', 'applet', 'form', 'input',
            'button', 'select', 'textarea', 'link', 'meta'
        ]
        
        # Dangerous attributes to remove
        self.dangerous_attrs = [
            'onclick', 'onload', 'onerror', 'onmouseover',
            'onmouseout', 'onfocus', 'onblur', 'onchange',
            'onsubmit', 'onreset', 'onselect', 'onkeydown',
            'onkeyup', 'onkeypress', 'href', 'src'
        ]
    
    def sanitize_content(self, raw_content: str) -> Dict[str, Any]:
        """
        Comprehensive content sanitization and security analysis.
        
        Args:
            raw_content: Raw HTML/text content from web fetch
            
        Returns:
            Dictionary containing:
            - clean_text: Sanitized text content
            - removed_elements: List of removed HTML elements
            - suspicious_patterns: List of detected suspicious patterns
            - purity_score: Security purity score (0.0-1.0)
            - encoding_issues: List of encoding anomalies
            - content_stats: Statistics about the content
        """
        self.logger.info(f"Starting content sanitization for {len(raw_content)} characters")
        
        try:
            # Step 1: Initial content analysis
            initial_stats = self._analyze_content_stats(raw_content)
            
            # Step 2: Detect encoding issues
            encoding_issues = self._detect_encoding_issues(raw_content)
            
            # Step 3: HTML sanitization
            soup = BeautifulSoup(raw_content, 'html.parser')
            removed_elements = self._remove_dangerous_elements(soup)
            
            # Step 4: Extract clean text
            clean_text = soup.get_text(separator=' ', strip=True)
            
            # Step 5: Text normalization
            normalized_text = self._normalize_text(clean_text)
            
            # Step 6: Suspicious pattern detection
            suspicious_patterns = self._detect_suspicious_patterns(normalized_text)
            
            # Step 7: Calculate purity score
            purity_score = self._calculate_purity_score(
                normalized_text, removed_elements, suspicious_patterns, encoding_issues
            )
            
            # Step 8: Final content statistics
            final_stats = self._analyze_content_stats(normalized_text)
            
            result = {
                'clean_text': normalized_text,
                'removed_elements': removed_elements,
                'suspicious_patterns': suspicious_patterns,
                'purity_score': purity_score,
                'encoding_issues': encoding_issues,
                'content_stats': {
                    'original_length': initial_stats['length'],
                    'clean_length': final_stats['length'],
                    'reduction_ratio': 1 - (final_stats['length'] / max(initial_stats['length'], 1)),
                    'word_count': final_stats['word_count'],
                    'line_count': final_stats['line_count'],
                    'non_ascii_ratio': final_stats['non_ascii_ratio']
                }
            }
            
            self.logger.info(f"Sanitization complete: {len(normalized_text)} chars, "
                           f"purity: {purity_score:.3f}, "
                           f"suspicious patterns: {len(suspicious_patterns)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error during content sanitization: {e}")
            return {
                'clean_text': '',
                'removed_elements': [],
                'suspicious_patterns': [f"Sanitization error: {str(e)}"],
                'purity_score': 0.0,
                'encoding_issues': ['sanitization_failed'],
                'content_stats': {
                    'original_length': len(raw_content),
                    'clean_length': 0,
                    'reduction_ratio': 1.0,
                    'word_count': 0,
                    'line_count': 0,
                    'non_ascii_ratio': 0.0
                }
            }
    
    def _remove_dangerous_elements(self, soup: BeautifulSoup) -> List[str]:
        """Remove dangerous HTML elements and attributes."""
        removed_elements = []
        
        # Remove dangerous tags
        for tag_name in self.dangerous_tags:
            for tag in soup.find_all(tag_name):
                removed_elements.append(f"{tag_name}_tag")
                tag.decompose()
        
        # Remove dangerous attributes from remaining tags
        for tag in soup.find_all():
            for attr in list(tag.attrs.keys()):
                if attr.lower() in self.dangerous_attrs:
                    removed_elements.append(f"{attr}_attribute")
                    del tag[attr]
        
        return list(set(removed_elements))
    
    def _detect_suspicious_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Detect suspicious patterns in the text."""
        suspicious_found = []
        
        for pattern_desc, compiled_pattern in self.compiled_patterns:
            matches = compiled_pattern.findall(text)
            if matches:
                suspicious_found.append({
                    'pattern': pattern_desc,
                    'matches': len(matches),
                    'examples': matches[:3]  # Store first 3 examples
                })
        
        return suspicious_found
    
    def _detect_encoding_issues(self, content: str) -> List[str]:
        """Detect encoding and character issues."""
        issues = []
        
        try:
            # Check for mixed encodings
            content.encode('ascii')
        except UnicodeEncodeError:
            issues.append('non_ascii_characters')
        
        # Check for control characters
        if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', content):
            issues.append('control_characters')
        
        # Check for zero-width characters
        if re.search(r'[\u200b\u200c\u200d\ufeff]', content):
            issues.append('zero_width_characters')
        
        # Check for unusual Unicode categories
        unusual_chars = 0
        for char in content[:1000]:  # Sample first 1000 chars
            category = unicodedata.category(char)
            if category.startswith('C'):  # Control characters
                unusual_chars += 1
        
        if unusual_chars > 10:
            issues.append('excessive_control_chars')
        
        return issues
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text content for analysis."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove zero-width characters
        text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Normalize Unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Decode HTML entities
        text = html.unescape(text)
        
        return text.strip()
    
    def _analyze_content_stats(self, content: str) -> Dict[str, Any]:
        """Analyze content statistics."""
        if not content:
            return {
                'length': 0,
                'word_count': 0,
                'line_count': 0,
                'non_ascii_ratio': 0.0
            }
        
        words = content.split()
        lines = content.split('\n')
        
        # Calculate non-ASCII ratio
        non_ascii_chars = sum(1 for char in content if ord(char) > 127)
        non_ascii_ratio = non_ascii_chars / len(content) if content else 0.0
        
        return {
            'length': len(content),
            'word_count': len(words),
            'line_count': len(lines),
            'non_ascii_ratio': non_ascii_ratio
        }
    
    def _calculate_purity_score(self, text: str, removed_elements: List[str],
                               suspicious_patterns: List[Dict], 
                               encoding_issues: List[str]) -> float:
        """
        Calculate content purity score (0.0 = suspicious, 1.0 = clean).
        
        Factors:
        - Number of removed dangerous elements
        - Suspicious patterns detected
        - Encoding anomalies
        - Text quality indicators
        """
        score = 1.0
        
        # Penalty for removed dangerous elements
        dangerous_penalty = min(len(removed_elements) * 0.1, 0.5)
        score -= dangerous_penalty
        
        # Penalty for suspicious patterns
        pattern_penalty = min(len(suspicious_patterns) * 0.15, 0.6)
        score -= pattern_penalty
        
        # Penalty for encoding issues
        encoding_penalty = min(len(encoding_issues) * 0.1, 0.3)
        score -= encoding_penalty
        
        # Bonus for clean, readable text
        if text and len(text) > 50:
            # Check for reasonable word/character ratio
            words = text.split()
            if words:
                avg_word_length = len(text) / len(words)
                if 3 <= avg_word_length <= 8:  # Reasonable word length
                    score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def get_sanitizer_stats(self) -> Dict[str, Any]:
        """Get sanitizer configuration and statistics."""
        return {
            'suspicious_patterns_count': len(self.suspicious_patterns),
            'dangerous_tags_count': len(self.dangerous_tags),
            'dangerous_attrs_count': len(self.dangerous_attrs),
            'version': '1.0'
        }
