#!/usr/bin/env python3
"""
Document Classifier
Analyzes uploaded documents to determine type, structure, and content patterns
for intelligent processing optimization.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

class DocumentType(Enum):
    """Types of documents for specialized processing."""
    FINANCIAL_REPORT = "financial_report"
    RESEARCH_PAPER = "research_paper"
    TECHNICAL_MANUAL = "technical_manual"
    LEGAL_DOCUMENT = "legal_document"
    MEETING_NOTES = "meeting_notes"
    PRESENTATION = "presentation"
    SPREADSHEET = "spreadsheet"
    CODE_DOCUMENTATION = "code_documentation"
    GENERAL_TEXT = "general_text"

class ContentPattern(Enum):
    """Content patterns found in documents."""
    TABLES = "tables"
    FORMULAS = "formulas"
    CODE_BLOCKS = "code_blocks"
    FINANCIAL_DATA = "financial_data"
    DEFINITIONS = "definitions"
    LISTS = "lists"
    CHARTS_REFERENCES = "charts_references"
    DATES_TIMELINE = "dates_timeline"

@dataclass
class DocumentStructure:
    """Maps the structural elements of a document."""
    headers: List[Tuple[str, int]]  # (header_text, level)
    sections: List[Dict[str, Any]]  # Section boundaries and metadata
    tables: List[Dict[str, Any]]    # Table locations and properties
    formulas: List[Dict[str, Any]]  # Formula locations and content
    lists: List[Dict[str, Any]]     # List locations and types
    page_count: int
    has_toc: bool
    
@dataclass
class DocumentClassification:
    """Complete classification result for a document."""
    document_type: DocumentType
    confidence: float
    content_patterns: List[ContentPattern]
    structure: DocumentStructure
    processing_hints: Dict[str, Any]
    metadata: Dict[str, Any]

class DocumentClassifier:
    """
    Analyzes documents to determine optimal processing strategy.
    """
    
    def __init__(self):
        self.type_indicators = {
            DocumentType.FINANCIAL_REPORT: [
                'income statement', 'balance sheet', 'cash flow', 'revenue', 'profit',
                'quarterly', 'annual report', 'financial', 'earnings', 'assets',
                'liabilities', 'equity', 'ebitda', 'roi', 'margin'
            ],
            DocumentType.RESEARCH_PAPER: [
                'abstract', 'introduction', 'methodology', 'results', 'conclusion',
                'references', 'bibliography', 'hypothesis', 'experiment', 'analysis',
                'literature review', 'discussion', 'findings'
            ],
            DocumentType.TECHNICAL_MANUAL: [
                'installation', 'configuration', 'troubleshooting', 'specifications',
                'requirements', 'procedure', 'manual', 'guide', 'instructions',
                'setup', 'maintenance', 'operation'
            ],
            DocumentType.LEGAL_DOCUMENT: [
                'agreement', 'contract', 'terms', 'conditions', 'clause', 'whereas',
                'party', 'jurisdiction', 'liability', 'indemnity', 'breach',
                'governing law', 'arbitration', 'confidentiality'
            ],
            DocumentType.MEETING_NOTES: [
                'agenda', 'attendees', 'action items', 'decisions', 'next steps',
                'meeting', 'discussion', 'follow up', 'assigned to', 'deadline'
            ],
            DocumentType.PRESENTATION: [
                'slide', 'presentation', 'overview', 'agenda', 'summary',
                'key points', 'takeaways', 'objectives', 'goals'
            ],
            DocumentType.CODE_DOCUMENTATION: [
                'api', 'function', 'class', 'method', 'parameter', 'return',
                'example', 'usage', 'documentation', 'readme', 'changelog'
            ]
        }
        
        self.content_pattern_indicators = {
            ContentPattern.TABLES: [
                r'\|.*\|.*\|',  # Markdown tables
                r'table \d+', r'exhibit \d+', r'figure \d+',
                r'\btable\b', r'\bexhibit\b', r'\bfigure\b'
            ],
            ContentPattern.FORMULAS: [
                r'[=]\s*[A-Za-z0-9\+\-\*\/\(\)\s]+',  # Basic formulas
                r'\$[A-Z]+\$?\d+',  # Excel references
                r'[A-Za-z]+\s*=\s*[0-9\+\-\*\/\(\)\s]+',  # Variable assignments
                r'∑|∏|∫|√|±|≤|≥|≠|≈'  # Mathematical symbols
            ],
            ContentPattern.FINANCIAL_DATA: [
                r'\$[\d,]+(?:\.\d{2})?[KMB]?',  # Currency amounts
                r'\d+(?:\.\d+)?%',  # Percentages
                r'\b(?:revenue|profit|loss|margin|ebitda|roi)\b',
                r'\b\d{4}\s*(?:q[1-4]|quarter|fiscal|fy)\b'  # Fiscal periods
            ],
            ContentPattern.CODE_BLOCKS: [
                r'```[\s\S]*?```',  # Markdown code blocks
                r'def\s+\w+\s*\(',  # Python functions
                r'function\s+\w+\s*\(',  # JavaScript functions
                r'class\s+\w+\s*[:\{]'  # Class definitions
            ],
            ContentPattern.DEFINITIONS: [
                r'\b\w+\s*:\s*[A-Z]',  # Term: Definition
                r'\b\w+\s*-\s*[A-Z]',  # Term - Definition
                r'definition\s*of\s*\w+',
                r'glossary', r'terminology'
            ]
        }
        
        logger.info("DocumentClassifier initialized")
    
    def classify_document(self, content: str, filename: str = "") -> DocumentClassification:
        """
        Perform complete document classification and analysis.
        
        Args:
            content: Full document text content
            filename: Original filename for additional hints
            
        Returns:
            Complete classification with processing recommendations
        """
        try:
            # Analyze document type
            doc_type, type_confidence = self._classify_document_type(content, filename)
            
            # Detect content patterns
            content_patterns = self._detect_content_patterns(content)
            
            # Map document structure
            structure = self._map_document_structure(content)
            
            # Generate processing hints
            processing_hints = self._generate_processing_hints(doc_type, content_patterns, structure)
            
            # Extract metadata
            metadata = self._extract_document_metadata(content, filename)
            
            classification = DocumentClassification(
                document_type=doc_type,
                confidence=type_confidence,
                content_patterns=content_patterns,
                structure=structure,
                processing_hints=processing_hints,
                metadata=metadata
            )
            
            logger.info(f"📄 Document classified as {doc_type.value} (confidence: {type_confidence:.2f})")
            logger.info(f"🔍 Content patterns detected: {[p.value for p in content_patterns]}")
            
            return classification
            
        except Exception as e:
            logger.error(f"Document classification failed: {e}")
            # Return safe fallback classification
            return DocumentClassification(
                document_type=DocumentType.GENERAL_TEXT,
                confidence=0.5,
                content_patterns=[],
                structure=DocumentStructure([], [], [], [], [], 1, False),
                processing_hints={'chunking_strategy': 'standard'},
                metadata={'error': str(e)}
            )
    
    def _classify_document_type(self, content: str, filename: str) -> Tuple[DocumentType, float]:
        """Classify the document type based on content and filename."""
        content_lower = content.lower()
        filename_lower = filename.lower()
        
        type_scores = {}
        
        # Score based on content indicators
        for doc_type, indicators in self.type_indicators.items():
            score = 0
            for indicator in indicators:
                # Count occurrences, with diminishing returns
                count = len(re.findall(r'\b' + re.escape(indicator) + r'\b', content_lower))
                score += min(count * 0.1, 0.5)  # Cap contribution per indicator
            
            # Filename bonus
            if any(indicator in filename_lower for indicator in indicators):
                score += 0.3
            
            type_scores[doc_type] = score
        
        # Find best match
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            confidence = min(type_scores[best_type], 1.0)
            
            # Require minimum confidence threshold
            if confidence < 0.2:
                return DocumentType.GENERAL_TEXT, 0.5
            
            return best_type, confidence
        
        return DocumentType.GENERAL_TEXT, 0.5
    
    def _detect_content_patterns(self, content: str) -> List[ContentPattern]:
        """Detect specific content patterns in the document."""
        detected_patterns = []
        
        for pattern_type, indicators in self.content_pattern_indicators.items():
            pattern_found = False
            
            for indicator in indicators:
                if re.search(indicator, content, re.IGNORECASE | re.MULTILINE):
                    pattern_found = True
                    break
            
            if pattern_found:
                detected_patterns.append(pattern_type)
        
        return detected_patterns
    
    def _map_document_structure(self, content: str) -> DocumentStructure:
        """Map the structural elements of the document."""
        lines = content.split('\n')
        
        # Detect headers (simple heuristic)
        headers = []
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if line_stripped:
                # Check for header patterns
                if (line_stripped.isupper() and len(line_stripped) < 100) or \
                   re.match(r'^#+\s+', line) or \
                   re.match(r'^\d+\.?\s+[A-Z]', line_stripped):
                    level = 1 if line_stripped.isupper() else 2
                    headers.append((line_stripped, level))
        
        # Detect tables (simple pattern matching)
        tables = []
        table_patterns = [r'\|.*\|.*\|', r'table \d+', r'exhibit \d+']
        for pattern in table_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                tables.append({
                    'start_pos': match.start(),
                    'content': match.group(),
                    'type': 'detected_table'
                })
        
        # Detect formulas
        formulas = []
        formula_patterns = self.content_pattern_indicators[ContentPattern.FORMULAS]
        for pattern in formula_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                formulas.append({
                    'start_pos': match.start(),
                    'content': match.group(),
                    'type': 'mathematical_expression'
                })
        
        # Detect lists
        lists = []
        list_patterns = [r'^\s*[-*+]\s+', r'^\s*\d+\.\s+']
        for pattern in list_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                lists.append({
                    'start_pos': match.start(),
                    'type': 'bullet_list' if pattern.startswith(r'^\s*[-*+]') else 'numbered_list'
                })
        
        return DocumentStructure(
            headers=headers,
            sections=[],  # Will be enhanced in future iterations
            tables=tables,
            formulas=formulas,
            lists=lists,
            page_count=max(1, content.count('\f') + 1),  # Form feed as page separator
            has_toc='table of contents' in content.lower()
        )
    
    def _generate_processing_hints(self, doc_type: DocumentType, patterns: List[ContentPattern], 
                                 structure: DocumentStructure) -> Dict[str, Any]:
        """Generate processing hints based on classification."""
        hints = {
            'chunking_strategy': 'semantic',
            'preserve_tables': ContentPattern.TABLES in patterns,
            'preserve_formulas': ContentPattern.FORMULAS in patterns,
            'extract_financial_data': ContentPattern.FINANCIAL_DATA in patterns,
            'priority_sections': [],
            'special_handling': []
        }
        
        # Document type specific hints
        if doc_type == DocumentType.FINANCIAL_REPORT:
            hints['priority_sections'] = ['executive summary', 'financial highlights', 'income statement']
            hints['extract_numerical_data'] = True
            hints['chunking_strategy'] = 'financial_aware'
        
        elif doc_type == DocumentType.RESEARCH_PAPER:
            hints['priority_sections'] = ['abstract', 'conclusion', 'results']
            hints['preserve_citations'] = True
            hints['chunking_strategy'] = 'academic'
        
        elif doc_type == DocumentType.TECHNICAL_MANUAL:
            hints['preserve_procedures'] = True
            hints['maintain_step_sequences'] = True
            hints['chunking_strategy'] = 'procedural'
        
        # Pattern-based hints
        if ContentPattern.CODE_BLOCKS in patterns:
            hints['special_handling'].append('preserve_code_blocks')
        
        if ContentPattern.DEFINITIONS in patterns:
            hints['special_handling'].append('extract_definitions')
        
        return hints
    
    def _extract_document_metadata(self, content: str, filename: str) -> Dict[str, Any]:
        """Extract document metadata for enhanced processing."""
        metadata = {
            'filename': filename,
            'content_length': len(content),
            'word_count': len(content.split()),
            'line_count': len(content.split('\n')),
            'has_numerical_data': bool(re.search(r'\d+', content)),
            'language': 'english',  # Could be enhanced with language detection
            'complexity_score': self._calculate_complexity_score(content)
        }
        
        # Extract dates
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'
        ]
        
        dates_found = []
        for pattern in date_patterns:
            dates_found.extend(re.findall(pattern, content, re.IGNORECASE))
        
        metadata['dates_found'] = dates_found[:10]  # Limit to first 10 dates
        
        return metadata
    
    def _calculate_complexity_score(self, content: str) -> float:
        """Calculate a complexity score for the document."""
        factors = {
            'length': min(len(content) / 10000, 1.0),  # Longer = more complex
            'technical_terms': len(re.findall(r'\b[A-Z]{2,}\b', content)) / 100,  # Acronyms
            'numbers': len(re.findall(r'\d+', content)) / 100,  # Numerical content
            'formulas': len(re.findall(r'[=]\s*[A-Za-z0-9\+\-\*\/\(\)]+', content)) / 10,
            'structure': len(re.findall(r'^#+\s+', content, re.MULTILINE)) / 20  # Headers
        }
        
        return min(sum(factors.values()), 1.0)

# Global instance
_document_classifier = None

def get_document_classifier() -> DocumentClassifier:
    """Get or create the global document classifier instance."""
    global _document_classifier
    if _document_classifier is None:
        _document_classifier = DocumentClassifier()
    return _document_classifier
