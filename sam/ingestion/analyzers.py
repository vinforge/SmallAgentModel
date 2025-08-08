#!/usr/bin/env python3
"""
Content Analyzers
Specialized analyzers for extracting rich metadata from document chunks.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class ContentSignature:
    """Rich metadata signature for a content chunk."""
    # Content type flags
    contains_formulas: bool = False
    contains_tables: bool = False
    contains_code: bool = False
    contains_financial_data: bool = False
    contains_definitions: bool = False
    contains_dates: bool = False
    contains_lists: bool = False
    
    # Extracted entities
    numerical_values: List[Dict[str, Any]] = field(default_factory=list)
    formulas: List[Dict[str, Any]] = field(default_factory=list)
    definitions: List[Dict[str, Any]] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    key_terms: List[str] = field(default_factory=list)
    
    # Content characteristics
    complexity_score: float = 0.0
    technical_level: str = "basic"  # basic, intermediate, advanced
    domain_indicators: List[str] = field(default_factory=list)
    
    # Query optimization hints
    query_keywords: Set[str] = field(default_factory=set)
    semantic_tags: List[str] = field(default_factory=list)

class ContentAnalyzer(ABC):
    """Base class for content analyzers."""
    
    @abstractmethod
    def analyze(self, content: str) -> Dict[str, Any]:
        """Analyze content and return metadata."""
        pass
    
    @abstractmethod
    def get_signature_updates(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get updates to apply to ContentSignature."""
        pass

class FormulaAnalyzer(ContentAnalyzer):
    """Analyzes mathematical formulas and expressions."""
    
    def __init__(self):
        self.formula_patterns = [
            # Variable assignments
            r'([A-Za-z_]\w*)\s*=\s*([^=\n]+)',
            # Excel formulas
            r'=\s*([A-Z]+\d+(?:[+\-*/]\s*[A-Z]+\d+)*)',
            # Mathematical expressions
            r'([A-Za-z_]\w*)\s*=\s*([\d\w\s+\-*/()]+)',
            # Financial ratios
            r'(ROI|ROE|ROA|P/E|EPS|EBITDA)\s*=\s*([\d\w\s+\-*/()%]+)',
            # Percentage calculations
            r'(\d+(?:\.\d+)?)\s*%\s*(?:of|×|x)\s*([\d\w\s+\-*/()]+)',
        ]
        
        self.mathematical_symbols = ['∑', '∏', '∫', '√', '±', '≤', '≥', '≠', '≈', '∞']
        
    def analyze(self, content: str) -> Dict[str, Any]:
        """Analyze mathematical content."""
        analysis = {
            'formulas_found': [],
            'mathematical_symbols': [],
            'variables': [],
            'calculations': []
        }
        
        # Find formulas
        for pattern in self.formula_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                formula_info = {
                    'full_expression': match.group(0),
                    'variable': match.group(1) if match.lastindex >= 1 else None,
                    'expression': match.group(2) if match.lastindex >= 2 else None,
                    'position': match.span(),
                    'type': self._classify_formula_type(match.group(0))
                }
                analysis['formulas_found'].append(formula_info)
        
        # Find mathematical symbols
        for symbol in self.mathematical_symbols:
            if symbol in content:
                analysis['mathematical_symbols'].append(symbol)
        
        # Extract numerical calculations
        calc_patterns = [
            r'(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)',
        ]
        
        for pattern in calc_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                analysis['calculations'].append({
                    'expression': match.group(0),
                    'components': match.groups(),
                    'position': match.span()
                })
        
        return analysis
    
    def get_signature_updates(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get ContentSignature updates for formula analysis."""
        updates = {}
        
        if analysis['formulas_found'] or analysis['mathematical_symbols']:
            updates['contains_formulas'] = True
            updates['formulas'] = analysis['formulas_found']
            updates['technical_level'] = 'intermediate'
            
            # Add formula-related keywords
            formula_keywords = set()
            for formula in analysis['formulas_found']:
                if formula['variable']:
                    formula_keywords.add(formula['variable'].lower())
                formula_keywords.add('formula')
                formula_keywords.add('calculation')
            
            updates['query_keywords'] = formula_keywords
            updates['semantic_tags'] = ['mathematical', 'quantitative']
        
        return updates
    
    def _classify_formula_type(self, formula: str) -> str:
        """Classify the type of formula."""
        formula_lower = formula.lower()
        
        if any(term in formula_lower for term in ['roi', 'roe', 'roa', 'p/e', 'eps']):
            return 'financial_ratio'
        elif '%' in formula:
            return 'percentage'
        elif any(op in formula for op in ['+', '-', '*', '/']):
            return 'arithmetic'
        elif '=' in formula:
            return 'equation'
        else:
            return 'expression'

class TableAnalyzer(ContentAnalyzer):
    """Analyzes tabular data and structures."""
    
    def __init__(self):
        self.table_indicators = [
            r'\|.*\|.*\|',  # Markdown tables
            r'^\s*\+[-=]+\+',  # ASCII tables
            r'Table\s+\d+',  # Table captions
            r'Exhibit\s+\d+',  # Exhibit captions
        ]
    
    def analyze(self, content: str) -> Dict[str, Any]:
        """Analyze tabular content."""
        analysis = {
            'tables_found': [],
            'table_structure': {},
            'data_types': [],
            'headers': []
        }
        
        # Detect markdown tables
        table_pattern = r'\|(.+)\|\n\|[-\s|:]+\|\n((?:\|.+\|\n?)+)'
        matches = re.finditer(table_pattern, content, re.MULTILINE)
        
        for match in matches:
            header_row = match.group(1)
            data_rows = match.group(2)
            
            headers = [h.strip() for h in header_row.split('|') if h.strip()]
            rows = []
            
            for row_line in data_rows.strip().split('\n'):
                if '|' in row_line:
                    row_data = [cell.strip() for cell in row_line.split('|') if cell.strip()]
                    if row_data:
                        rows.append(row_data)
            
            table_info = {
                'headers': headers,
                'rows': rows,
                'row_count': len(rows),
                'column_count': len(headers),
                'position': match.span(),
                'data_types': self._analyze_column_types(rows)
            }
            
            analysis['tables_found'].append(table_info)
            analysis['headers'].extend(headers)
        
        # Detect other table indicators
        for pattern in self.table_indicators:
            if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                analysis['table_structure']['has_table_indicators'] = True
                break
        
        return analysis
    
    def get_signature_updates(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get ContentSignature updates for table analysis."""
        updates = {}
        
        if analysis['tables_found'] or analysis['table_structure'].get('has_table_indicators'):
            updates['contains_tables'] = True
            updates['technical_level'] = 'intermediate'
            
            # Add table-related keywords
            table_keywords = set(['table', 'data', 'chart'])
            for header in analysis['headers']:
                table_keywords.add(header.lower())
            
            updates['query_keywords'] = table_keywords
            updates['semantic_tags'] = ['tabular', 'structured_data']
            
            # Check for numerical data in tables
            has_numerical = any(
                'numerical' in table.get('data_types', [])
                for table in analysis['tables_found']
            )
            if has_numerical:
                updates['contains_financial_data'] = True
                updates['semantic_tags'].append('quantitative')
        
        return updates
    
    def _analyze_column_types(self, rows: List[List[str]]) -> List[str]:
        """Analyze the data types in table columns."""
        if not rows:
            return []
        
        column_count = max(len(row) for row in rows) if rows else 0
        column_types = []
        
        for col_idx in range(column_count):
            column_values = []
            for row in rows:
                if col_idx < len(row):
                    column_values.append(row[col_idx])
            
            col_type = self._infer_column_type(column_values)
            column_types.append(col_type)
        
        return column_types
    
    def _infer_column_type(self, values: List[str]) -> str:
        """Infer the data type of a column."""
        if not values:
            return 'empty'
        
        numerical_count = 0
        date_count = 0
        
        for value in values:
            value_clean = value.strip()
            if not value_clean:
                continue
            
            # Check for numbers (including currency and percentages)
            if re.match(r'^[\$€£¥]?[\d,]+(?:\.\d+)?[%]?$', value_clean):
                numerical_count += 1
            
            # Check for dates
            elif re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', value_clean):
                date_count += 1
        
        total_values = len([v for v in values if v.strip()])
        
        if numerical_count / total_values > 0.7:
            return 'numerical'
        elif date_count / total_values > 0.7:
            return 'date'
        else:
            return 'text'

class FinancialAnalyzer(ContentAnalyzer):
    """Analyzes financial data and metrics."""
    
    def __init__(self):
        self.financial_terms = [
            'revenue', 'profit', 'loss', 'margin', 'ebitda', 'roi', 'roe', 'roa',
            'assets', 'liabilities', 'equity', 'cash flow', 'earnings', 'eps',
            'dividend', 'interest', 'debt', 'investment', 'return', 'yield'
        ]
        
        self.currency_patterns = [
            r'\$[\d,]+(?:\.\d{2})?[KMB]?',  # USD
            r'€[\d,]+(?:\.\d{2})?[KMB]?',   # EUR
            r'£[\d,]+(?:\.\d{2})?[KMB]?',   # GBP
            r'¥[\d,]+(?:\.\d{2})?[KMB]?',   # JPY/CNY
        ]
        
        self.percentage_pattern = r'\d+(?:\.\d+)?%'
        
    def analyze(self, content: str) -> Dict[str, Any]:
        """Analyze financial content."""
        analysis = {
            'financial_terms': [],
            'currency_amounts': [],
            'percentages': [],
            'financial_ratios': [],
            'fiscal_periods': []
        }
        
        content_lower = content.lower()
        
        # Find financial terms
        for term in self.financial_terms:
            if term in content_lower:
                analysis['financial_terms'].append(term)
        
        # Find currency amounts
        for pattern in self.currency_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                analysis['currency_amounts'].append({
                    'amount': match.group(0),
                    'position': match.span()
                })
        
        # Find percentages
        matches = re.finditer(self.percentage_pattern, content)
        for match in matches:
            analysis['percentages'].append({
                'percentage': match.group(0),
                'position': match.span()
            })
        
        # Find fiscal periods
        fiscal_patterns = [
            r'\b\d{4}\s*(?:q[1-4]|quarter)\b',
            r'\b(?:q[1-4]|quarter)\s*\d{4}\b',
            r'\bfy\s*\d{4}\b',
            r'\bfiscal\s*year\s*\d{4}\b'
        ]
        
        for pattern in fiscal_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                analysis['fiscal_periods'].append(match.group(0))
        
        return analysis
    
    def get_signature_updates(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get ContentSignature updates for financial analysis."""
        updates = {}
        
        if (analysis['financial_terms'] or analysis['currency_amounts'] or 
            len(analysis['percentages']) > 2):  # More than 2 percentages suggests financial data
            
            updates['contains_financial_data'] = True
            updates['domain_indicators'] = ['finance', 'business']
            
            # Add financial keywords
            financial_keywords = set(analysis['financial_terms'])
            financial_keywords.update(['financial', 'money', 'business', 'economic'])
            
            updates['query_keywords'] = financial_keywords
            updates['semantic_tags'] = ['financial', 'quantitative', 'business']
            
            # Extract numerical values
            numerical_values = []
            for amount in analysis['currency_amounts']:
                numerical_values.append({
                    'value': amount['amount'],
                    'type': 'currency',
                    'context': 'financial_amount'
                })
            
            for percentage in analysis['percentages']:
                numerical_values.append({
                    'value': percentage['percentage'],
                    'type': 'percentage',
                    'context': 'financial_metric'
                })
            
            updates['numerical_values'] = numerical_values
        
        return updates

class CodeAnalyzer(ContentAnalyzer):
    """Analyzes code blocks and technical content."""
    
    def __init__(self):
        self.code_patterns = [
            r'```[\s\S]*?```',  # Fenced code blocks
            r'`[^`\n]+`',       # Inline code
            r'def\s+\w+\s*\(',  # Python functions
            r'function\s+\w+\s*\(',  # JavaScript functions
            r'class\s+\w+\s*[:\{]',  # Class definitions
            r'import\s+\w+',    # Import statements
            r'#include\s*<\w+>', # C/C++ includes
        ]
        
        self.programming_languages = {
            'python': ['def ', 'import ', 'class ', 'if __name__'],
            'javascript': ['function ', 'var ', 'let ', 'const ', '=>'],
            'java': ['public class', 'private ', 'public static void'],
            'c++': ['#include', 'using namespace', 'int main'],
            'sql': ['SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE'],
        }
    
    def analyze(self, content: str) -> Dict[str, Any]:
        """Analyze code content."""
        analysis = {
            'code_blocks': [],
            'programming_language': None,
            'code_elements': [],
            'technical_terms': []
        }
        
        # Find code blocks
        for pattern in self.code_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                analysis['code_blocks'].append({
                    'code': match.group(0),
                    'position': match.span(),
                    'type': self._classify_code_type(match.group(0))
                })
        
        # Detect programming language
        for lang, indicators in self.programming_languages.items():
            if any(indicator in content for indicator in indicators):
                analysis['programming_language'] = lang
                break
        
        # Find technical terms
        technical_terms = [
            'api', 'function', 'method', 'class', 'variable', 'parameter',
            'return', 'algorithm', 'database', 'server', 'client', 'framework'
        ]
        
        content_lower = content.lower()
        for term in technical_terms:
            if term in content_lower:
                analysis['technical_terms'].append(term)
        
        return analysis
    
    def get_signature_updates(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get ContentSignature updates for code analysis."""
        updates = {}
        
        if analysis['code_blocks'] or analysis['programming_language']:
            updates['contains_code'] = True
            updates['technical_level'] = 'advanced'
            updates['domain_indicators'] = ['technology', 'programming']
            
            # Add code-related keywords
            code_keywords = set(['code', 'programming', 'software', 'development'])
            if analysis['programming_language']:
                code_keywords.add(analysis['programming_language'])
            
            code_keywords.update(analysis['technical_terms'])
            
            updates['query_keywords'] = code_keywords
            updates['semantic_tags'] = ['technical', 'programming', 'code']
        
        return updates
    
    def _classify_code_type(self, code: str) -> str:
        """Classify the type of code block."""
        if code.startswith('```'):
            return 'fenced_block'
        elif code.startswith('`') and code.endswith('`'):
            return 'inline_code'
        elif 'def ' in code or 'function ' in code:
            return 'function_definition'
        elif 'class ' in code:
            return 'class_definition'
        else:
            return 'code_snippet'

class DefinitionAnalyzer(ContentAnalyzer):
    """Analyzes definitions and glossary terms."""
    
    def __init__(self):
        self.definition_patterns = [
            r'([A-Z][a-z\s]+):\s*([A-Z][^.]+\.)',  # Term: Definition.
            r'([A-Z][a-z\s]+)\s*-\s*([A-Z][^.]+\.)',  # Term - Definition.
            r'([A-Z][a-z\s]+)\s*means\s*([^.]+\.)',  # Term means definition.
            r'([A-Z][a-z\s]+)\s*is\s*defined\s*as\s*([^.]+\.)',  # Term is defined as...
        ]
        
        self.glossary_indicators = ['glossary', 'definitions', 'terminology', 'acronyms']
    
    def analyze(self, content: str) -> Dict[str, Any]:
        """Analyze definitions and terminology."""
        analysis = {
            'definitions': [],
            'terms': [],
            'is_glossary_section': False
        }
        
        # Check if this is a glossary section
        content_lower = content.lower()
        analysis['is_glossary_section'] = any(
            indicator in content_lower for indicator in self.glossary_indicators
        )
        
        # Find definitions
        for pattern in self.definition_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                definition_info = {
                    'term': match.group(1).strip(),
                    'definition': match.group(2).strip(),
                    'full_text': match.group(0),
                    'position': match.span()
                }
                analysis['definitions'].append(definition_info)
                analysis['terms'].append(definition_info['term'])
        
        return analysis
    
    def get_signature_updates(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get ContentSignature updates for definition analysis."""
        updates = {}
        
        if analysis['definitions'] or analysis['is_glossary_section']:
            updates['contains_definitions'] = True
            updates['definitions'] = analysis['definitions']
            
            # Add definition-related keywords
            definition_keywords = set(['definition', 'term', 'meaning', 'glossary'])
            for term in analysis['terms']:
                definition_keywords.add(term.lower())
            
            updates['query_keywords'] = definition_keywords
            updates['semantic_tags'] = ['definitions', 'terminology', 'reference']
            
            if analysis['is_glossary_section']:
                updates['semantic_tags'].append('glossary')
        
        return updates

class ContentAnalyzerPipeline:
    """Pipeline that runs all content analyzers on a chunk."""
    
    def __init__(self):
        self.analyzers = [
            FormulaAnalyzer(),
            TableAnalyzer(),
            FinancialAnalyzer(),
            CodeAnalyzer(),
            DefinitionAnalyzer()
        ]
        
        logger.info(f"ContentAnalyzerPipeline initialized with {len(self.analyzers)} analyzers")
    
    def analyze_chunk(self, content: str) -> ContentSignature:
        """Run all analyzers on a chunk and create comprehensive signature."""
        signature = ContentSignature()
        
        try:
            for analyzer in self.analyzers:
                # Run analyzer
                analysis = analyzer.analyze(content)
                
                # Get signature updates
                updates = analyzer.get_signature_updates(analysis)
                
                # Apply updates to signature
                for key, value in updates.items():
                    if hasattr(signature, key):
                        if isinstance(getattr(signature, key), set):
                            getattr(signature, key).update(value)
                        elif isinstance(getattr(signature, key), list):
                            getattr(signature, key).extend(value)
                        else:
                            setattr(signature, key, value)
            
            # Calculate overall complexity score
            signature.complexity_score = self._calculate_complexity_score(signature)
            
            # Determine technical level
            signature.technical_level = self._determine_technical_level(signature)
            
            return signature
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            return ContentSignature()  # Return empty signature
    
    def _calculate_complexity_score(self, signature: ContentSignature) -> float:
        """Calculate overall complexity score for the content."""
        score = 0.0
        
        # Base complexity factors
        if signature.contains_formulas:
            score += 0.3
        if signature.contains_tables:
            score += 0.2
        if signature.contains_code:
            score += 0.4
        if signature.contains_financial_data:
            score += 0.2
        if signature.contains_definitions:
            score += 0.1
        
        # Additional complexity from entities
        score += min(len(signature.numerical_values) * 0.05, 0.2)
        score += min(len(signature.formulas) * 0.1, 0.3)
        score += min(len(signature.definitions) * 0.05, 0.2)
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _determine_technical_level(self, signature: ContentSignature) -> str:
        """Determine the technical level of the content."""
        if signature.contains_code or signature.complexity_score > 0.7:
            return 'advanced'
        elif (signature.contains_formulas or signature.contains_financial_data or 
              signature.complexity_score > 0.4):
            return 'intermediate'
        else:
            return 'basic'

# Global instance
_analyzer_pipeline = None

def get_analyzer_pipeline() -> ContentAnalyzerPipeline:
    """Get or create the global analyzer pipeline instance."""
    global _analyzer_pipeline
    if _analyzer_pipeline is None:
        _analyzer_pipeline = ContentAnalyzerPipeline()
    return _analyzer_pipeline
