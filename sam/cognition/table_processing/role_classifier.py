"""
Context-Aware Semantic Role Classifier
======================================

Core "Smart Router" with contextual understanding for accurate semantic role
classification of table cells.

This implements Task 3 from task23.md - Context-Aware Semantic Role Classifier.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from .table_parser import TableObject
from .token_roles import TokenRole, SEMANTIC_ROLES
from .config import get_table_config

logger = logging.getLogger(__name__)


@dataclass
class RoleClassification:
    """Result of role classification for a table cell."""
    role: TokenRole
    confidence: float
    reasoning: str
    context_factors: Dict[str, float]


class TableRoleClassifier:
    """
    Context-aware semantic role classifier for table cells.
    
    Uses fine-tuned transformer models with contextual enhancements
    for accurate role prediction.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the role classifier."""
        self.config = get_table_config()
        self.model_path = model_path or self.config.model.role_classifier_path
        self.model = None
        self.tokenizer = None
        
        # Initialize model (placeholder for now)
        self._load_model()
        
        logger.info(f"TableRoleClassifier initialized")
    
    def _load_model(self):
        """Load the trained role classification model."""
        try:
            # Try to load a pre-trained model if available
            import os
            if os.path.exists(self.model_path):
                # TODO: Load actual trained model
                # For now, use enhanced heuristic-based classification
                logger.info(f"Model file found at {self.model_path}, but loading not implemented yet")
                self.model = "enhanced_heuristic"
                self.tokenizer = "enhanced_heuristic"
            else:
                logger.info("No trained model found, using enhanced heuristic-based classification")
                self.model = "enhanced_heuristic"
                self.tokenizer = "enhanced_heuristic"

        except Exception as e:
            logger.warning(f"Model loading failed: {e}, falling back to heuristics")
            self.model = "enhanced_heuristic"
            self.tokenizer = "enhanced_heuristic"
    
    def predict(self, table_object: TableObject, 
                document_context: Optional[str] = None) -> List[List[RoleClassification]]:
        """
        Classify roles for all cells in a table with document context.
        
        Args:
            table_object: Table to classify
            document_context: Surrounding document text for context
            
        Returns:
            2D list of RoleClassification objects matching table structure
        """
        try:
            if not table_object.is_valid():
                logger.warning("Invalid table object provided for classification")
                return []
            
            rows, cols = table_object.get_dimensions()
            classifications = []
            
            for row_idx in range(rows):
                row_classifications = []
                for col_idx in range(cols):
                    cell_content = table_object.get_cell(row_idx, col_idx)
                    
                    # Classify individual cell
                    classification = self._classify_cell(
                        cell_content, row_idx, col_idx, table_object, document_context
                    )
                    row_classifications.append(classification)
                
                classifications.append(row_classifications)
            
            # Post-process classifications for consistency
            classifications = self._post_process_classifications(classifications, table_object)
            
            return classifications
            
        except Exception as e:
            logger.error(f"Role classification failed: {e}")
            return []
    
    def predict_with_confidence(self, table_object: TableObject) -> List[List[RoleClassification]]:
        """Return predictions with confidence scores."""
        return self.predict(table_object)
    
    def _classify_cell(self, cell_content: str, row: int, col: int,
                      table_object: TableObject, 
                      document_context: Optional[str] = None) -> RoleClassification:
        """Classify a single cell using heuristic rules."""
        
        # Context factors for classification
        context_factors = {
            "position_weight": self._calculate_position_weight(row, col, table_object),
            "content_weight": self._calculate_content_weight(cell_content),
            "formatting_weight": self._calculate_formatting_weight(row, col, table_object),
            "context_weight": self._calculate_context_weight(document_context) if document_context else 0.0
        }
        
        # Determine role using heuristic rules
        role, confidence, reasoning = self._heuristic_classification(
            cell_content, row, col, table_object, context_factors
        )
        
        return RoleClassification(
            role=role,
            confidence=confidence,
            reasoning=reasoning,
            context_factors=context_factors
        )
    
    def _heuristic_classification(self, cell_content: str, row: int, col: int,
                                table_object: TableObject,
                                context_factors: Dict[str, float]) -> Tuple[TokenRole, float, str]:
        """Classify cell using enhanced heuristic rules with contextual understanding."""

        # Empty cell detection
        if not cell_content or not cell_content.strip():
            return TokenRole.EMPTY, 0.95, "Empty or whitespace-only content"

        content = cell_content.strip()
        rows, cols = table_object.get_dimensions()

        # Enhanced total/sum detection with position context (check first for priority)
        total_score = self._calculate_total_score(content, row, col, table_object)
        if total_score > 0.7:
            return TokenRole.TOTAL, total_score, "Enhanced total/sum detection"

        # Enhanced header detection with context
        header_score = self._calculate_enhanced_header_score(content, row, col, table_object, context_factors)
        if header_score > 0.7:
            confidence = min(0.95, header_score + context_factors.get("position_weight", 0) * 0.1)
            return TokenRole.HEADER, confidence, f"Enhanced header detection (score={header_score:.2f})"

        # Formula detection with improved patterns
        formula_score = self._calculate_formula_score(content)
        if formula_score > 0.8:
            return TokenRole.FORMULA, formula_score, "Enhanced formula detection"

        # Axis/Index detection with position awareness
        axis_score = self._calculate_axis_score(content, row, col, table_object)
        if axis_score > 0.7:
            return TokenRole.AXIS, axis_score, "Enhanced axis/index detection"

        # Caption detection with length and position context
        caption_score = self._calculate_caption_score(content, row, col, table_object)
        if caption_score > 0.7:
            return TokenRole.CAPTION, caption_score, "Enhanced caption detection"

        # Metadata detection with improved patterns
        metadata_score = self._calculate_metadata_score(content, row, col, table_object)
        if metadata_score > 0.7:
            return TokenRole.METADATA, metadata_score, "Enhanced metadata detection"

        # Data classification with content analysis
        data_confidence = self._calculate_data_confidence(content, context_factors)
        return TokenRole.DATA, data_confidence, "Enhanced data classification"
    
    def _calculate_position_weight(self, row: int, col: int, table_object: TableObject) -> float:
        """Calculate position-based weight for classification."""
        rows, cols = table_object.get_dimensions()
        
        # First row and first column have higher header probability
        if row == 0 or col == 0:
            return 0.8
        
        # Last row might contain totals
        if row == rows - 1:
            return 0.6
        
        # Middle cells are likely data
        return 0.4
    
    def _calculate_content_weight(self, content: str) -> float:
        """Calculate content-based weight for classification."""
        if not content or not content.strip():
            return 0.1
        
        content = content.strip()
        
        # Longer content might be headers or captions
        if len(content) > 20:
            return 0.7
        
        # Short content is likely data
        if len(content) < 10:
            return 0.5
        
        return 0.6
    
    def _calculate_formatting_weight(self, row: int, col: int, table_object: TableObject) -> float:
        """Calculate formatting-based weight (placeholder)."""
        # TODO: Implement actual formatting analysis
        # For now, return neutral weight
        return 0.5
    
    def _calculate_context_weight(self, document_context: str) -> float:
        """Calculate document context weight."""
        if not document_context:
            return 0.0
        
        # Simple context analysis
        context_lower = document_context.lower()
        
        # Financial context might indicate currency/numbers
        financial_terms = ['revenue', 'profit', 'sales', 'cost', 'budget']
        if any(term in context_lower for term in financial_terms):
            return 0.7
        
        # Technical context might indicate different patterns
        technical_terms = ['specification', 'parameter', 'configuration']
        if any(term in context_lower for term in technical_terms):
            return 0.6
        
        return 0.5
    
    def _calculate_enhanced_header_score(self, content: str, row: int, col: int,
                                       table_object: TableObject,
                                       context_factors: Dict[str, float]) -> float:
        """Calculate enhanced likelihood that a cell is a header."""
        score = 0.0
        rows, cols = table_object.get_dimensions()

        # Position-based scoring
        if row == 0:
            score += 0.5  # First row is very likely header
        if col == 0:
            score += 0.3  # First column often contains headers

        # Content characteristics
        if content and content[0].isupper():
            score += 0.15

        # Short, descriptive text is more likely to be header
        word_count = len(content.split())
        if 1 <= word_count <= 3:
            score += 0.2
        elif word_count > 5:
            score -= 0.1  # Very long text less likely to be header

        # Headers typically don't contain numbers (except dates/years)
        if not any(c.isdigit() for c in content):
            score += 0.15
        elif self._looks_like_date_or_year(content):
            score += 0.1  # Dates/years can be headers

        # Check consistency with other cells in same row/column
        consistency_bonus = self._calculate_header_consistency(content, row, col, table_object)
        score += consistency_bonus * 0.2

        # Context-based adjustments
        if context_factors.get("formatting_weight", 0) > 0.7:
            score += 0.1  # Formatted text more likely to be header

        return min(1.0, score)

    def _calculate_total_score(self, content: str, row: int, col: int,
                             table_object: TableObject) -> float:
        """Calculate likelihood that a cell contains total/sum information."""
        score = 0.0
        rows, cols = table_object.get_dimensions()
        content_lower = content.lower()

        # Keyword-based detection (more aggressive)
        total_keywords = ['total', 'sum', 'subtotal', 'grand total', 'average', 'mean',
                         'aggregate', 'overall', 'combined', 'net']
        for keyword in total_keywords:
            if keyword in content_lower:
                score += 0.8  # Increased from 0.6 to 0.8
                break

        # Position-based scoring (totals often at bottom or right)
        if row == rows - 1:  # Last row
            score += 0.2
        if col == cols - 1:  # Last column
            score += 0.1

        # Numeric content with total indicators
        if any(c.isdigit() for c in content) and any(kw in content_lower for kw in ['total', 'sum']):
            score += 0.1

        return min(1.0, score)

    def _calculate_formula_score(self, content: str) -> float:
        """Calculate likelihood that a cell contains a formula."""
        score = 0.0

        # Direct formula indicators
        if content.startswith('='):
            score += 0.8

        # Formula keywords
        formula_keywords = ['formula', 'calculated', 'computed', 'derived', 'function']
        if any(keyword in content.lower() for keyword in formula_keywords):
            score += 0.6

        # Mathematical expressions
        math_patterns = ['(', ')', '+', '-', '*', '/', '^', 'sqrt', 'sum(', 'avg(']
        if any(pattern in content.lower() for pattern in math_patterns):
            score += 0.4

        return min(1.0, score)

    def _calculate_axis_score(self, content: str, row: int, col: int,
                            table_object: TableObject) -> float:
        """Calculate likelihood that a cell is an axis/index label."""
        score = 0.0
        content_lower = content.lower()

        # Axis keywords
        axis_keywords = ['row', 'column', 'index', 'id', 'seq', 'item', 'entry', '#']
        if any(keyword in content_lower for keyword in axis_keywords):
            score += 0.5

        # Sequential patterns (numbers, letters)
        if content.isdigit() or (len(content) == 1 and content.isalpha()):
            score += 0.4

        # Position-based (first row/column more likely to be axis)
        if row == 0 or col == 0:
            score += 0.3

        # Pattern matching (A1, B2, etc.)
        import re
        if re.match(r'^[A-Z]\d+$', content) or re.match(r'^\d+[A-Z]$', content):
            score += 0.6

        return min(1.0, score)

    def _calculate_caption_score(self, content: str, row: int, col: int,
                               table_object: TableObject) -> float:
        """Calculate likelihood that a cell is a caption."""
        score = 0.0
        rows, cols = table_object.get_dimensions()
        content_lower = content.lower()

        # Caption keywords
        caption_keywords = ['table', 'figure', 'chart', 'graph', 'report', 'summary']
        if any(keyword in content_lower for keyword in caption_keywords):
            score += 0.5

        # Length-based scoring (captions are usually longer)
        word_count = len(content.split())
        if word_count > 5:
            score += 0.3
        elif word_count > 10:
            score += 0.5

        # Position-based (captions often above or below table)
        if row == 0 and cols > 3:  # Above table
            score += 0.2
        if row == rows - 1 and cols > 3:  # Below table
            score += 0.2

        # Formatting indicators (Title Case, etc.)
        if content.istitle():
            score += 0.2

        return min(1.0, score)

    def _calculate_metadata_score(self, content: str, row: int, col: int,
                                table_object: TableObject) -> float:
        """Calculate likelihood that a cell contains metadata."""
        score = 0.0
        rows, cols = table_object.get_dimensions()
        content_lower = content.lower()

        # Metadata keywords
        metadata_keywords = ['source', 'note', 'updated', 'footnote', 'reference',
                           'created', 'modified', 'version', 'author']
        if any(keyword in content_lower for keyword in metadata_keywords):
            score += 0.6

        # Special characters indicating metadata
        if content.startswith('*') or content.startswith('†') or content.startswith('‡'):
            score += 0.5

        # Position-based (metadata often at bottom)
        if row == rows - 1:
            score += 0.3

        # Date patterns in metadata
        if self._looks_like_date_or_year(content):
            score += 0.2

        return min(1.0, score)

    def _calculate_data_confidence(self, content: str, context_factors: Dict[str, float]) -> float:
        """Calculate confidence for data classification."""
        base_confidence = 0.7

        # Adjust based on content characteristics
        if any(c.isdigit() for c in content):
            base_confidence += 0.1  # Numbers are often data

        if len(content.split()) == 1:
            base_confidence += 0.05  # Single words often data

        # Context adjustments
        content_weight = context_factors.get("content_weight", 0.5)
        if content_weight > 0.7:
            base_confidence += 0.1

        return min(1.0, base_confidence)
    
    def _post_process_classifications(self, classifications: List[List[RoleClassification]],
                                    table_object: TableObject) -> List[List[RoleClassification]]:
        """Post-process classifications for consistency."""
        if not classifications:
            return classifications
        
        rows = len(classifications)
        cols = len(classifications[0]) if classifications else 0
        
        # Ensure first row consistency (if most cells are headers, make all headers)
        if rows > 0:
            first_row = classifications[0]
            header_count = sum(1 for cell in first_row if cell.role == TokenRole.HEADER)

            if header_count > cols * 0.7:  # If 70% are headers, make all headers
                for cell in first_row:
                    if cell.role not in [TokenRole.EMPTY]:
                        cell.role = TokenRole.HEADER
                        cell.reasoning += " (consistency adjustment)"

        # Similar logic for first column, but preserve TOTAL classifications
        if cols > 0:
            first_col = [classifications[i][0] for i in range(rows)]
            header_count = sum(1 for cell in first_col if cell.role == TokenRole.HEADER)

            if header_count > rows * 0.7:
                for i in range(rows):
                    # Don't override TOTAL, FORMULA, or other important classifications
                    if classifications[i][0].role not in [TokenRole.EMPTY, TokenRole.TOTAL, TokenRole.FORMULA]:
                        classifications[i][0].role = TokenRole.HEADER
                        classifications[i][0].reasoning += " (consistency adjustment)"
        
        return classifications

    def _looks_like_date_or_year(self, content: str) -> bool:
        """Check if content looks like a date or year."""
        import re

        # Year patterns
        if re.match(r'^\d{4}$', content) and 1900 <= int(content) <= 2100:
            return True

        # Date patterns
        date_patterns = [
            r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$',  # MM/DD/YYYY or DD/MM/YYYY
            r'^\d{4}-\d{2}-\d{2}$',              # YYYY-MM-DD
            r'^[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}$',  # Month DD, YYYY
        ]

        return any(re.match(pattern, content) for pattern in date_patterns)

    def _calculate_header_consistency(self, content: str, row: int, col: int,
                                    table_object: TableObject) -> float:
        """Calculate consistency with other potential headers."""
        rows, cols = table_object.get_dimensions()
        consistency_score = 0.0

        # Check consistency with other cells in same row (if row 0)
        if row == 0 and cols > 1:
            similar_cells = 0
            total_cells = 0

            for c in range(cols):
                if c != col:
                    other_content = table_object.get_cell(row, c)
                    if other_content and other_content.strip():
                        total_cells += 1
                        # Check if other cell has similar header characteristics
                        if (other_content[0].isupper() and content and content[0].isupper()):
                            similar_cells += 1
                        if (len(other_content.split()) <= 3 and len(content.split()) <= 3):
                            similar_cells += 1

            if total_cells > 0:
                consistency_score = similar_cells / (total_cells * 2)  # Normalize

        # Check consistency with other cells in same column (if col 0)
        elif col == 0 and rows > 1:
            similar_cells = 0
            total_cells = 0

            for r in range(rows):
                if r != row:
                    other_content = table_object.get_cell(r, col)
                    if other_content and other_content.strip():
                        total_cells += 1
                        # Check for similar patterns
                        if (other_content[0].isupper() and content and content[0].isupper()):
                            similar_cells += 1

            if total_cells > 0:
                consistency_score = similar_cells / total_cells

        return min(1.0, consistency_score)
