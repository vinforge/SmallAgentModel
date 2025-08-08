"""
Utility Functions for Table Processing
======================================

Helper functions and utilities for table processing operations including
coordinate systems, data type detection, and common table operations.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CellDataType(Enum):
    """Data types for table cells."""
    TEXT = "text"
    NUMBER = "number"
    INTEGER = "integer"
    FLOAT = "float"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    DATE = "date"
    TIME = "time"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    EMAIL = "email"
    URL = "url"
    PHONE = "phone"
    EMPTY = "empty"
    UNKNOWN = "unknown"


@dataclass
class CoordinateSystem:
    """Coordinate system for table cells."""
    row: int
    column: int
    
    def __str__(self) -> str:
        return f"({self.row}, {self.column})"
    
    def to_excel_notation(self) -> str:
        """Convert to Excel-style notation (A1, B2, etc.)."""
        col_letter = self._number_to_column_letter(self.column)
        return f"{col_letter}{self.row + 1}"
    
    def _number_to_column_letter(self, n: int) -> str:
        """Convert column number to Excel column letter."""
        result = ""
        while n >= 0:
            result = chr(n % 26 + ord('A')) + result
            n = n // 26 - 1
            if n < 0:
                break
        return result


@dataclass
class CellInfo:
    """Information about a table cell."""
    content: str
    coordinates: CoordinateSystem
    data_type: CellDataType
    confidence: float
    formatting: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class TableUtils:
    """Utility class for table processing operations."""
    
    # Regular expressions for data type detection
    PATTERNS = {
        CellDataType.INTEGER: re.compile(r'^-?\d+$'),
        CellDataType.FLOAT: re.compile(r'^-?\d*\.\d+$'),
        CellDataType.CURRENCY: re.compile(r'^[\$€£¥]?\s*-?\d{1,3}(,\d{3})*(\.\d{2})?$'),
        CellDataType.PERCENTAGE: re.compile(r'^-?\d*\.?\d+%$'),
        CellDataType.DATE: re.compile(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$|^\d{4}-\d{2}-\d{2}$'),
        CellDataType.TIME: re.compile(r'^\d{1,2}:\d{2}(:\d{2})?(\s*[AaPp][Mm])?$'),
        CellDataType.DATETIME: re.compile(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}$'),
        CellDataType.BOOLEAN: re.compile(r'^(true|false|yes|no|y|n|1|0)$', re.IGNORECASE),
        CellDataType.EMAIL: re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
        CellDataType.URL: re.compile(r'^https?://[^\s]+$'),
        CellDataType.PHONE: re.compile(r'^[\+]?[1-9]?[\d\s\-\(\)]{7,15}$'),
        CellDataType.EMPTY: re.compile(r'^\s*$|^(n/?a|null|none|empty|-)$', re.IGNORECASE)
    }
    
    @staticmethod
    def detect_cell_data_type(content: str) -> Tuple[CellDataType, float]:
        """
        Detect the data type of cell content.
        
        Returns:
            Tuple of (data_type, confidence_score)
        """
        if not content or not content.strip():
            return CellDataType.EMPTY, 1.0
        
        content = content.strip()
        
        # Check each pattern in order of specificity
        for data_type, pattern in TableUtils.PATTERNS.items():
            if pattern.match(content):
                confidence = TableUtils._calculate_type_confidence(content, data_type)
                return data_type, confidence
        
        # Default to text if no pattern matches
        return CellDataType.TEXT, 0.8
    
    @staticmethod
    def _calculate_type_confidence(content: str, data_type: CellDataType) -> float:
        """Calculate confidence score for data type detection."""
        base_confidence = 0.9
        
        # Adjust confidence based on content characteristics
        if data_type == CellDataType.CURRENCY:
            # Higher confidence for properly formatted currency
            if '$' in content or '€' in content or '£' in content:
                base_confidence = 0.95
        elif data_type == CellDataType.DATE:
            # Higher confidence for ISO format dates
            if re.match(r'^\d{4}-\d{2}-\d{2}$', content):
                base_confidence = 0.98
        elif data_type == CellDataType.EMAIL:
            # Very high confidence for email patterns
            base_confidence = 0.99
        elif data_type == CellDataType.URL:
            # High confidence for URLs
            base_confidence = 0.95
        
        return base_confidence
    
    @staticmethod
    def normalize_table_data(table_data: List[List[str]]) -> List[List[str]]:
        """Normalize table data by cleaning and standardizing cells."""
        normalized = []
        
        for row in table_data:
            normalized_row = []
            for cell in row:
                # Clean whitespace
                cleaned_cell = cell.strip() if cell else ""
                
                # Normalize common empty indicators
                if cleaned_cell.lower() in ['n/a', 'null', 'none', 'empty', '-', '']:
                    cleaned_cell = ""
                
                normalized_row.append(cleaned_cell)
            normalized.append(normalized_row)
        
        return normalized
    
    @staticmethod
    def get_table_dimensions(table_data: List[List[str]]) -> Tuple[int, int]:
        """Get table dimensions (rows, columns)."""
        if not table_data:
            return 0, 0
        
        rows = len(table_data)
        cols = max(len(row) for row in table_data) if table_data else 0
        return rows, cols
    
    @staticmethod
    def is_valid_table_size(table_data: List[List[str]], 
                           min_size: Tuple[int, int] = (2, 2),
                           max_size: Tuple[int, int] = (100, 50)) -> bool:
        """Check if table size is within valid bounds."""
        rows, cols = TableUtils.get_table_dimensions(table_data)
        min_rows, min_cols = min_size
        max_rows, max_cols = max_size
        
        return (min_rows <= rows <= max_rows and 
                min_cols <= cols <= max_cols)
    
    @staticmethod
    def calculate_empty_cell_ratio(table_data: List[List[str]]) -> float:
        """Calculate the ratio of empty cells in the table."""
        if not table_data:
            return 1.0
        
        total_cells = sum(len(row) for row in table_data)
        if total_cells == 0:
            return 1.0
        
        empty_cells = 0
        for row in table_data:
            for cell in row:
                if not cell or not cell.strip():
                    empty_cells += 1
        
        return empty_cells / total_cells
    
    @staticmethod
    def extract_table_context(document_text: str, table_position: int, 
                             window_size: int = 3) -> Dict[str, str]:
        """Extract context around a table in a document."""
        sentences = re.split(r'[.!?]+', document_text)
        
        # Find sentences around table position
        before_sentences = sentences[max(0, table_position - window_size):table_position]
        after_sentences = sentences[table_position + 1:table_position + 1 + window_size]
        
        return {
            "before": " ".join(before_sentences).strip(),
            "after": " ".join(after_sentences).strip()
        }
    
    @staticmethod
    def identify_header_patterns(table_data: List[List[str]]) -> Dict[str, Any]:
        """Identify potential header patterns in table data."""
        if not table_data or len(table_data) < 2:
            return {"has_headers": False}
        
        first_row = table_data[0]
        second_row = table_data[1] if len(table_data) > 1 else []
        
        # Check for header indicators
        header_indicators = {
            "capitalized_words": sum(1 for cell in first_row if cell and cell[0].isupper()),
            "short_text": sum(1 for cell in first_row if cell and len(cell.split()) <= 3),
            "no_numbers": sum(1 for cell in first_row if cell and not any(c.isdigit() for c in cell)),
            "different_from_data": 0
        }
        
        # Compare first row with second row
        if second_row:
            for i, (header_cell, data_cell) in enumerate(zip(first_row, second_row)):
                if header_cell and data_cell:
                    header_type, _ = TableUtils.detect_cell_data_type(header_cell)
                    data_type, _ = TableUtils.detect_cell_data_type(data_cell)
                    if header_type != data_type:
                        header_indicators["different_from_data"] += 1
        
        # Calculate header probability
        total_cells = len(first_row)
        if total_cells == 0:
            return {"has_headers": False}
        
        header_score = (
            header_indicators["capitalized_words"] / total_cells * 0.3 +
            header_indicators["short_text"] / total_cells * 0.2 +
            header_indicators["no_numbers"] / total_cells * 0.3 +
            header_indicators["different_from_data"] / total_cells * 0.2
        )
        
        return {
            "has_headers": header_score > 0.6,
            "header_score": header_score,
            "indicators": header_indicators
        }
    
    @staticmethod
    def generate_table_summary(table_data: List[List[str]]) -> Dict[str, Any]:
        """Generate a comprehensive summary of table characteristics."""
        if not table_data:
            return {"error": "Empty table data"}
        
        rows, cols = TableUtils.get_table_dimensions(table_data)
        empty_ratio = TableUtils.calculate_empty_cell_ratio(table_data)
        header_info = TableUtils.identify_header_patterns(table_data)
        
        # Analyze data types
        data_types = {}
        for row in table_data:
            for cell in row:
                if cell and cell.strip():
                    cell_type, _ = TableUtils.detect_cell_data_type(cell)
                    data_types[cell_type.value] = data_types.get(cell_type.value, 0) + 1
        
        return {
            "dimensions": {"rows": rows, "columns": cols},
            "empty_cell_ratio": empty_ratio,
            "header_analysis": header_info,
            "data_type_distribution": data_types,
            "total_cells": rows * cols,
            "non_empty_cells": sum(data_types.values()),
            "quality_score": TableUtils._calculate_quality_score(rows, cols, empty_ratio, header_info)
        }
    
    @staticmethod
    def _calculate_quality_score(rows: int, cols: int, empty_ratio: float, 
                                header_info: Dict[str, Any]) -> float:
        """Calculate overall quality score for a table."""
        # Base score from size
        size_score = min(1.0, (rows * cols) / 100)  # Normalize to reasonable table size
        
        # Penalty for too many empty cells
        empty_penalty = empty_ratio * 0.5
        
        # Bonus for having headers
        header_bonus = 0.2 if header_info.get("has_headers", False) else 0
        
        # Calculate final score
        quality_score = size_score - empty_penalty + header_bonus
        return max(0.0, min(1.0, quality_score))
