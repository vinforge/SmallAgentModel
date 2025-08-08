"""
Multi-Modal Table Parser
========================

Robust table detection and extraction system with multiple detection strategies
for HTML, Markdown, PDF, Image, and CSV formats.

This implements Task 2 from task23.md - Multi-Modal Table Parser.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod

from .config import get_table_config
from .utils import TableUtils, CoordinateSystem

logger = logging.getLogger(__name__)


@dataclass
class TableObject:
    """
    Enriched table object with comprehensive metadata.
    
    Contains raw cell data, detection confidence, table metadata,
    and quality indicators as specified in task23.md.
    """
    raw_data: List[List[str]]
    coordinates: List[List[CoordinateSystem]]
    detection_confidence: float
    table_metadata: Dict[str, Any]
    quality_indicators: Dict[str, float]
    source_format: str
    table_id: Optional[str] = None
    title: Optional[str] = None
    caption: Optional[str] = None
    
    def get_dimensions(self) -> Tuple[int, int]:
        """Get table dimensions (rows, columns)."""
        return TableUtils.get_table_dimensions(self.raw_data)
    
    def get_cell(self, row: int, col: int) -> str:
        """Get cell content at specific coordinates."""
        if 0 <= row < len(self.raw_data) and 0 <= col < len(self.raw_data[row]):
            return self.raw_data[row][col]
        return ""
    
    def is_valid(self) -> bool:
        """Check if table meets basic validity criteria."""
        config = get_table_config()
        return (
            self.detection_confidence >= config.detection.confidence_threshold and
            TableUtils.is_valid_table_size(
                self.raw_data, 
                config.detection.min_table_size,
                config.detection.max_table_size
            )
        )


class TableDetectionStrategy(ABC):
    """Abstract base class for table detection strategies."""
    
    @abstractmethod
    def detect_tables(self, content: Any, **kwargs) -> List[TableObject]:
        """Detect tables in the given content."""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this detection strategy."""
        pass


class HTMLTableDetector(TableDetectionStrategy):
    """HTML table detection strategy."""
    
    def detect_tables(self, html_content: str, **kwargs) -> List[TableObject]:
        """Detect tables in HTML content."""
        try:
            # Try to import BeautifulSoup for HTML parsing
            try:
                from bs4 import BeautifulSoup
            except ImportError:
                logger.warning("BeautifulSoup not available. Using regex fallback for HTML tables.")
                return self._regex_html_detection(html_content)
            
            soup = BeautifulSoup(html_content, 'html.parser')
            tables = soup.find_all('table')
            
            detected_tables = []
            for i, table in enumerate(tables):
                table_data = self._extract_html_table_data(table)
                if table_data:
                    table_obj = self._create_table_object(
                        table_data, f"html_table_{i}", "html", table
                    )
                    detected_tables.append(table_obj)
            
            return detected_tables
            
        except Exception as e:
            logger.error(f"HTML table detection failed: {e}")
            return []
    
    def _extract_html_table_data(self, table_element) -> List[List[str]]:
        """Extract data from HTML table element."""
        rows = table_element.find_all('tr')
        table_data = []
        
        for row in rows:
            cells = row.find_all(['td', 'th'])
            row_data = [cell.get_text(strip=True) for cell in cells]
            if row_data:  # Only add non-empty rows
                table_data.append(row_data)
        
        return table_data
    
    def _regex_html_detection(self, html_content: str) -> List[TableObject]:
        """Fallback regex-based HTML table detection."""
        # Simple regex pattern for HTML tables
        table_pattern = r'<table[^>]*>(.*?)</table>'
        tables = re.findall(table_pattern, html_content, re.DOTALL | re.IGNORECASE)
        
        detected_tables = []
        for i, table_html in enumerate(tables):
            # Extract rows using regex
            row_pattern = r'<tr[^>]*>(.*?)</tr>'
            rows = re.findall(row_pattern, table_html, re.DOTALL | re.IGNORECASE)
            
            table_data = []
            for row_html in rows:
                # Extract cells using regex
                cell_pattern = r'<t[hd][^>]*>(.*?)</t[hd]>'
                cells = re.findall(cell_pattern, row_html, re.DOTALL | re.IGNORECASE)
                if cells:
                    # Clean HTML tags from cell content
                    clean_cells = [re.sub(r'<[^>]+>', '', cell).strip() for cell in cells]
                    table_data.append(clean_cells)
            
            if table_data:
                table_obj = self._create_table_object(
                    table_data, f"html_regex_table_{i}", "html"
                )
                detected_tables.append(table_obj)
        
        return detected_tables
    
    def _create_table_object(self, table_data: List[List[str]], 
                           table_id: str, source_format: str, 
                           original_element=None) -> TableObject:
        """Create TableObject from extracted data."""
        # Generate coordinates
        coordinates = []
        for row_idx, row in enumerate(table_data):
            coord_row = []
            for col_idx, _ in enumerate(row):
                coord_row.append(CoordinateSystem(row_idx, col_idx))
            coordinates.append(coord_row)
        
        # Calculate quality indicators
        quality_indicators = {
            "completeness": 1.0 - TableUtils.calculate_empty_cell_ratio(table_data),
            "structure_score": self._calculate_structure_score(table_data),
            "content_quality": self._calculate_content_quality(table_data)
        }
        
        # Extract metadata
        metadata = {
            "extraction_method": "html_parser",
            "table_index": table_id,
            "source_element": str(original_element) if original_element else None,
            "extraction_timestamp": None  # Will be set by parser
        }
        
        # Calculate detection confidence
        confidence = self._calculate_detection_confidence(table_data, quality_indicators)
        
        return TableObject(
            raw_data=table_data,
            coordinates=coordinates,
            detection_confidence=confidence,
            table_metadata=metadata,
            quality_indicators=quality_indicators,
            source_format=source_format,
            table_id=table_id
        )
    
    def _calculate_structure_score(self, table_data: List[List[str]]) -> float:
        """Calculate structural quality score."""
        if not table_data:
            return 0.0
        
        # Check for consistent row lengths
        row_lengths = [len(row) for row in table_data]
        if not row_lengths:
            return 0.0
        
        max_length = max(row_lengths)
        min_length = min(row_lengths)
        consistency_score = min_length / max_length if max_length > 0 else 0.0
        
        return consistency_score
    
    def _calculate_content_quality(self, table_data: List[List[str]]) -> float:
        """Calculate content quality score."""
        if not table_data:
            return 0.0
        
        total_cells = sum(len(row) for row in table_data)
        if total_cells == 0:
            return 0.0
        
        # Count cells with meaningful content
        meaningful_cells = 0
        for row in table_data:
            for cell in row:
                if cell and len(cell.strip()) > 0:
                    meaningful_cells += 1
        
        return meaningful_cells / total_cells
    
    def _calculate_detection_confidence(self, table_data: List[List[str]],
                                      quality_indicators: Dict[str, float]) -> float:
        """Calculate overall detection confidence."""
        if not table_data:
            return 0.0

        # Base confidence from table size
        rows, cols = TableUtils.get_table_dimensions(table_data)
        size_confidence = min(1.0, (rows * cols) / 15)  # More generous size scoring

        # Quality-based confidence
        quality_confidence = sum(quality_indicators.values()) / len(quality_indicators)

        # Bonus for well-structured tables
        structure_bonus = 0.0
        if rows >= 2 and cols >= 2:  # Minimum viable table
            structure_bonus = 0.2

        # Combined confidence with structure bonus
        base_confidence = (size_confidence * 0.3 + quality_confidence * 0.7)
        return min(1.0, base_confidence + structure_bonus)
    
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return "html_table_detector"


class MarkdownTableDetector(TableDetectionStrategy):
    """Markdown table detection strategy."""
    
    def detect_tables(self, markdown_content: str, **kwargs) -> List[TableObject]:
        """Detect tables in Markdown content."""
        try:
            # Split content into lines and look for table patterns
            lines = markdown_content.split('\n')

            detected_tables = []
            table_lines = []
            in_table = False
            table_count = 0

            for line in lines:
                line = line.strip()

                # Check if line looks like a table row
                if line.startswith('|') and line.endswith('|') and line.count('|') >= 3:
                    # This looks like a table row
                    if not in_table:
                        in_table = True
                        table_lines = []
                    table_lines.append(line)
                else:
                    # Not a table line
                    if in_table and table_lines:
                        # End of table, process it
                        table_data = self._parse_markdown_table_lines(table_lines)
                        if table_data and len(table_data) >= 2:  # At least header + 1 data row
                            table_obj = self._create_markdown_table_object(
                                table_data, f"md_table_{table_count}"
                            )
                            detected_tables.append(table_obj)
                            table_count += 1

                        table_lines = []
                        in_table = False

            # Handle table at end of content
            if in_table and table_lines:
                table_data = self._parse_markdown_table_lines(table_lines)
                if table_data and len(table_data) >= 2:
                    table_obj = self._create_markdown_table_object(
                        table_data, f"md_table_{table_count}"
                    )
                    detected_tables.append(table_obj)

            return detected_tables

        except Exception as e:
            logger.error(f"Markdown table detection failed: {e}")
            return []
    
    def _parse_markdown_table_lines(self, table_lines: List[str]) -> List[List[str]]:
        """Parse markdown table lines into structured data."""
        table_data = []

        for line in table_lines:
            line = line.strip()
            if not line or not line.startswith('|') or not line.endswith('|'):
                continue

            # Check if this is a separator line (contains only |, -, :, and spaces)
            # More robust pattern to catch separator lines
            separator_pattern = r'^\|\s*[-:]+\s*(\|\s*[-:]+\s*)*\|$'
            if re.match(separator_pattern, line):
                continue

            # Extract cells
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            if cells and any(cell for cell in cells):  # Only add rows with actual content
                table_data.append(cells)

        return table_data

    def _parse_markdown_table(self, table_text: str) -> List[List[str]]:
        """Parse markdown table text into structured data (legacy method)."""
        lines = table_text.strip().split('\n')
        return self._parse_markdown_table_lines(lines)
    
    def _create_markdown_table_object(self, table_data: List[List[str]], 
                                    table_id: str) -> TableObject:
        """Create TableObject for markdown table."""
        # Similar to HTML table object creation but with markdown-specific metadata
        coordinates = []
        for row_idx, row in enumerate(table_data):
            coord_row = []
            for col_idx, _ in enumerate(row):
                coord_row.append(CoordinateSystem(row_idx, col_idx))
            coordinates.append(coord_row)
        
        quality_indicators = {
            "completeness": 1.0 - TableUtils.calculate_empty_cell_ratio(table_data),
            "structure_score": self._calculate_markdown_structure_score(table_data),
            "content_quality": self._calculate_content_quality(table_data)
        }
        
        metadata = {
            "extraction_method": "markdown_parser",
            "table_index": table_id,
            "format_type": "markdown_table"
        }
        
        confidence = self._calculate_detection_confidence(table_data, quality_indicators)
        
        return TableObject(
            raw_data=table_data,
            coordinates=coordinates,
            detection_confidence=confidence,
            table_metadata=metadata,
            quality_indicators=quality_indicators,
            source_format="markdown",
            table_id=table_id
        )
    
    def _calculate_markdown_structure_score(self, table_data: List[List[str]]) -> float:
        """Calculate structure score for markdown tables."""
        if not table_data:
            return 0.0
        
        # Markdown tables should have consistent column counts
        row_lengths = [len(row) for row in table_data]
        if not row_lengths:
            return 0.0
        
        # All rows should have the same length in well-formed markdown tables
        expected_length = row_lengths[0]
        consistent_rows = sum(1 for length in row_lengths if length == expected_length)
        
        return consistent_rows / len(row_lengths)
    
    def _calculate_content_quality(self, table_data: List[List[str]]) -> float:
        """Calculate content quality score."""
        if not table_data:
            return 0.0
        
        total_cells = sum(len(row) for row in table_data)
        if total_cells == 0:
            return 0.0
        
        meaningful_cells = 0
        for row in table_data:
            for cell in row:
                if cell and len(cell.strip()) > 0:
                    meaningful_cells += 1
        
        return meaningful_cells / total_cells
    
    def _calculate_detection_confidence(self, table_data: List[List[str]], 
                                      quality_indicators: Dict[str, float]) -> float:
        """Calculate detection confidence."""
        if not table_data:
            return 0.0
        
        rows, cols = TableUtils.get_table_dimensions(table_data)
        size_confidence = min(1.0, (rows * cols) / 15)
        quality_confidence = sum(quality_indicators.values()) / len(quality_indicators)
        
        return (size_confidence * 0.3 + quality_confidence * 0.7)
    
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return "markdown_table_detector"


class PDFTableDetector(TableDetectionStrategy):
    """PDF table detection strategy using PyMuPDF."""

    def detect_tables(self, pdf_content: Any, **kwargs) -> List[TableObject]:
        """Detect tables in PDF content."""
        try:
            # Try to import PyMuPDF
            try:
                import fitz  # PyMuPDF
            except ImportError:
                logger.warning("PyMuPDF not available. Install with: pip install PyMuPDF")
                return []

            detected_tables = []

            # Handle different input types
            if isinstance(pdf_content, str):
                # Assume it's a file path
                doc = fitz.open(pdf_content)
            elif hasattr(pdf_content, 'read'):
                # File-like object
                doc = fitz.open(stream=pdf_content.read(), filetype="pdf")
            else:
                logger.error("Unsupported PDF content type")
                return []

            # Extract tables from each page
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)

                # Try to find tables using PyMuPDF's table detection
                try:
                    tables = page.find_tables()
                    for i, table in enumerate(tables):
                        table_data = table.extract()
                        if table_data:
                            table_obj = self._create_pdf_table_object(
                                table_data, f"pdf_page_{page_num}_table_{i}", page_num
                            )
                            detected_tables.append(table_obj)
                except Exception as e:
                    logger.debug(f"Table detection failed on page {page_num}: {e}")
                    continue

            doc.close()
            return detected_tables

        except Exception as e:
            logger.error(f"PDF table detection failed: {e}")
            return []

    def _create_pdf_table_object(self, table_data: List[List[str]],
                               table_id: str, page_num: int) -> TableObject:
        """Create TableObject from PDF table data."""
        # Clean and normalize table data
        cleaned_data = []
        for row in table_data:
            cleaned_row = []
            for cell in row:
                # Clean cell content
                cleaned_cell = str(cell).strip() if cell is not None else ""
                cleaned_row.append(cleaned_cell)
            if any(cell for cell in cleaned_row):  # Only add non-empty rows
                cleaned_data.append(cleaned_row)

        if not cleaned_data:
            # Return empty table object
            return self._create_empty_table_object(table_id)

        # Generate coordinates
        coordinates = []
        for row_idx, row in enumerate(cleaned_data):
            coord_row = []
            for col_idx, _ in enumerate(row):
                coord_row.append(CoordinateSystem(row_idx, col_idx))
            coordinates.append(coord_row)

        # Calculate quality indicators
        quality_indicators = {
            "completeness": 1.0 - TableUtils.calculate_empty_cell_ratio(cleaned_data),
            "structure_score": self._calculate_pdf_structure_score(cleaned_data),
            "extraction_confidence": 0.8  # PDF extraction is generally reliable
        }

        # Create metadata
        metadata = {
            "extraction_method": "pymupdf",
            "source_page": page_num,
            "table_index": table_id,
            "format_type": "pdf_table"
        }

        # Calculate detection confidence
        confidence = self._calculate_pdf_confidence(cleaned_data, quality_indicators)

        return TableObject(
            raw_data=cleaned_data,
            coordinates=coordinates,
            detection_confidence=confidence,
            table_metadata=metadata,
            quality_indicators=quality_indicators,
            source_format="pdf",
            table_id=table_id
        )

    def _create_empty_table_object(self, table_id: str) -> TableObject:
        """Create empty table object for failed extractions."""
        return TableObject(
            raw_data=[],
            coordinates=[],
            detection_confidence=0.0,
            table_metadata={"extraction_method": "pymupdf", "error": "empty_extraction"},
            quality_indicators={"completeness": 0.0, "structure_score": 0.0},
            source_format="pdf",
            table_id=table_id
        )

    def _calculate_pdf_structure_score(self, table_data: List[List[str]]) -> float:
        """Calculate structure score for PDF tables."""
        if not table_data:
            return 0.0

        # Check row length consistency
        row_lengths = [len(row) for row in table_data]
        if not row_lengths:
            return 0.0

        max_length = max(row_lengths)
        min_length = min(row_lengths)
        consistency = min_length / max_length if max_length > 0 else 0.0

        return consistency

    def _calculate_pdf_confidence(self, table_data: List[List[str]],
                                quality_indicators: Dict[str, float]) -> float:
        """Calculate confidence for PDF table detection."""
        if not table_data:
            return 0.0

        # Base confidence from extraction method
        base_confidence = 0.7  # PDF extraction is generally reliable

        # Adjust based on quality indicators
        quality_bonus = sum(quality_indicators.values()) / len(quality_indicators) * 0.3

        return min(1.0, base_confidence + quality_bonus)

    def get_strategy_name(self) -> str:
        return "pdf_table_detector"


class ImageTableDetector(TableDetectionStrategy):
    """Image table detection strategy using OCR and vision models."""

    def detect_tables(self, image_content: Any, **kwargs) -> List[TableObject]:
        """Detect tables in image content."""
        try:
            # Try to import required libraries
            try:
                import cv2
                import numpy as np
            except ImportError:
                logger.warning("OpenCV not available. Install with: pip install opencv-python")
                return []

            try:
                from PIL import Image
            except ImportError:
                logger.warning("PIL not available. Install with: pip install Pillow")
                return []

            # For now, implement basic table detection using contours
            # In a full implementation, this would use specialized table detection models

            detected_tables = []

            # Convert image to OpenCV format
            if isinstance(image_content, str):
                # File path
                image = cv2.imread(image_content)
            elif hasattr(image_content, 'read'):
                # File-like object
                image_data = np.frombuffer(image_content.read(), np.uint8)
                image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            else:
                logger.error("Unsupported image content type")
                return []

            if image is None:
                logger.error("Failed to load image")
                return []

            # Basic table detection using contours (placeholder implementation)
            table_regions = self._detect_table_regions(image)

            for i, region in enumerate(table_regions):
                # Extract table data from region (would use OCR in real implementation)
                table_data = self._extract_table_from_region(image, region)
                if table_data:
                    table_obj = self._create_image_table_object(
                        table_data, f"image_table_{i}", region
                    )
                    detected_tables.append(table_obj)

            return detected_tables

        except Exception as e:
            logger.error(f"Image table detection failed: {e}")
            return []

    def _detect_table_regions(self, image) -> List[Dict[str, Any]]:
        """Detect table regions in image (basic implementation)."""
        # This is a simplified implementation
        # A real implementation would use specialized table detection models

        import cv2

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply threshold
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours that might be tables (rectangular, large enough)
        table_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.5 < aspect_ratio < 5:  # Reasonable aspect ratio for tables
                    table_regions.append({
                        "bbox": (x, y, w, h),
                        "area": area,
                        "contour": contour
                    })

        return table_regions

    def _extract_table_from_region(self, image, region: Dict[str, Any]) -> List[List[str]]:
        """Extract table data from image region (placeholder)."""
        # This would use OCR (like Tesseract) to extract text from table cells
        # For now, return placeholder data
        x, y, w, h = region["bbox"]

        # Placeholder: return a simple 2x2 table
        return [
            ["Header 1", "Header 2"],
            ["Data 1", "Data 2"]
        ]

    def _create_image_table_object(self, table_data: List[List[str]],
                                 table_id: str, region: Dict[str, Any]) -> TableObject:
        """Create TableObject from image table data."""
        # Generate coordinates
        coordinates = []
        for row_idx, row in enumerate(table_data):
            coord_row = []
            for col_idx, _ in enumerate(row):
                coord_row.append(CoordinateSystem(row_idx, col_idx))
            coordinates.append(coord_row)

        # Quality indicators (lower confidence for image extraction)
        quality_indicators = {
            "completeness": 1.0 - TableUtils.calculate_empty_cell_ratio(table_data),
            "ocr_confidence": 0.6,  # Placeholder OCR confidence
            "detection_confidence": 0.5  # Lower confidence for basic detection
        }

        # Metadata
        metadata = {
            "extraction_method": "image_ocr",
            "table_index": table_id,
            "bbox": region["bbox"],
            "area": region["area"]
        }

        # Calculate confidence
        confidence = 0.5  # Lower confidence for image-based detection

        return TableObject(
            raw_data=table_data,
            coordinates=coordinates,
            detection_confidence=confidence,
            table_metadata=metadata,
            quality_indicators=quality_indicators,
            source_format="image",
            table_id=table_id
        )

    def get_strategy_name(self) -> str:
        return "image_table_detector"


class CSVDetector(TableDetectionStrategy):
    """CSV detection and parsing strategy."""

    def detect_tables(self, csv_content: str, **kwargs) -> List[TableObject]:
        """Detect tables in CSV content."""
        try:
            import csv
            import io

            detected_tables = []

            # Handle different input types
            if isinstance(csv_content, str):
                if '\n' in csv_content or ',' in csv_content:
                    # Content is CSV data
                    csv_data = csv_content
                else:
                    # Assume it's a file path
                    with open(csv_content, 'r', encoding='utf-8') as f:
                        csv_data = f.read()
            else:
                csv_data = str(csv_content)

            # Try different CSV dialects
            dialects = [csv.excel, csv.excel_tab]

            for dialect in dialects:
                try:
                    # Parse CSV data
                    reader = csv.reader(io.StringIO(csv_data), dialect=dialect)
                    table_data = list(reader)

                    if table_data and len(table_data) > 1:  # Must have at least 2 rows
                        table_obj = self._create_csv_table_object(
                            table_data, f"csv_table_{dialect.__name__}"
                        )
                        detected_tables.append(table_obj)
                        break  # Use first successful dialect

                except Exception as e:
                    logger.debug(f"CSV parsing failed with {dialect.__name__}: {e}")
                    continue

            return detected_tables

        except Exception as e:
            logger.error(f"CSV table detection failed: {e}")
            return []

    def _create_csv_table_object(self, table_data: List[List[str]],
                               table_id: str) -> TableObject:
        """Create TableObject from CSV data."""
        # Clean and normalize data
        cleaned_data = []
        for row in table_data:
            cleaned_row = [cell.strip() if cell else "" for cell in row]
            cleaned_data.append(cleaned_row)

        # Generate coordinates
        coordinates = []
        for row_idx, row in enumerate(cleaned_data):
            coord_row = []
            for col_idx, _ in enumerate(row):
                coord_row.append(CoordinateSystem(row_idx, col_idx))
            coordinates.append(coord_row)

        # Quality indicators
        quality_indicators = {
            "completeness": 1.0 - TableUtils.calculate_empty_cell_ratio(cleaned_data),
            "structure_score": self._calculate_csv_structure_score(cleaned_data),
            "parsing_confidence": 0.9  # CSV parsing is very reliable
        }

        # Metadata
        metadata = {
            "extraction_method": "csv_parser",
            "table_index": table_id,
            "format_type": "csv_table",
            "delimiter_detected": ","  # Could be enhanced to detect actual delimiter
        }

        # High confidence for CSV parsing
        confidence = 0.95

        return TableObject(
            raw_data=cleaned_data,
            coordinates=coordinates,
            detection_confidence=confidence,
            table_metadata=metadata,
            quality_indicators=quality_indicators,
            source_format="csv",
            table_id=table_id
        )

    def _calculate_csv_structure_score(self, table_data: List[List[str]]) -> float:
        """Calculate structure score for CSV tables."""
        if not table_data:
            return 0.0

        # CSV files should have very consistent structure
        row_lengths = [len(row) for row in table_data]
        if not row_lengths:
            return 0.0

        # Check if all rows have the same length
        expected_length = row_lengths[0]
        consistent_rows = sum(1 for length in row_lengths if length == expected_length)

        return consistent_rows / len(row_lengths)

    def get_strategy_name(self) -> str:
        return "csv_detector"


class TableParser:
    """
    Enhanced TableParser class with multiple detection strategies.
    
    Implements the multi-strategy approach from task23.md with confidence
    scoring and quality indicators.
    """
    
    def __init__(self):
        """Initialize the table parser with all detection strategies."""
        self.strategies = [
            HTMLTableDetector(),
            MarkdownTableDetector(),
            PDFTableDetector(),
            ImageTableDetector(),
            CSVDetector()
        ]
        self.config = get_table_config()
        logger.info(f"TableParser initialized with {len(self.strategies)} strategies")
    
    def extract_tables_from_document(self, doc_content: str,
                                   doc_type: str) -> List[TableObject]:
        """
        Extract tables from document using multiple strategies.

        Args:
            doc_content: Document content
            doc_type: Document type (html, markdown, pdf, etc.)

        Returns:
            List of detected TableObject instances
        """
        try:
            all_tables = []

            # Try strategies based on document type
            relevant_strategies = self._get_relevant_strategies(doc_type)

            # If no specific strategies found, try all strategies
            if not relevant_strategies:
                relevant_strategies = self.strategies

            for strategy in relevant_strategies:
                try:
                    tables = strategy.detect_tables(doc_content)
                    # Filter tables by confidence threshold
                    valid_tables = [
                        table for table in tables
                        if table.detection_confidence >= self.config.detection.confidence_threshold
                    ]
                    all_tables.extend(valid_tables)

                    logger.debug(f"{strategy.get_strategy_name()} found {len(valid_tables)} valid tables")

                except Exception as e:
                    logger.warning(f"Strategy {strategy.get_strategy_name()} failed: {e}")
                    continue

            # Remove duplicates and combine results
            unique_tables = self._deduplicate_tables(all_tables)

            logger.info(f"Extracted {len(unique_tables)} unique tables from document")
            return unique_tables

        except Exception as e:
            logger.error(f"Table extraction failed: {e}")
            return []
    
    def _get_relevant_strategies(self, doc_type: str) -> List[TableDetectionStrategy]:
        """Get relevant strategies for document type."""
        strategy_map = {
            "html": [HTMLTableDetector],
            "markdown": [MarkdownTableDetector],
            "pdf": [PDFTableDetector],
            "image": [ImageTableDetector],
            "csv": [CSVDetector]
        }

        relevant_classes = strategy_map.get(doc_type.lower(), [])
        return [s for s in self.strategies if type(s).__name__ in [cls.__name__ for cls in relevant_classes]]
    
    def _deduplicate_tables(self, tables: List[TableObject]) -> List[TableObject]:
        """Remove duplicate tables based on content similarity."""
        if not tables:
            return []
        
        unique_tables = []
        for table in tables:
            is_duplicate = False
            for existing_table in unique_tables:
                if self._tables_are_similar(table, existing_table):
                    # Keep the one with higher confidence
                    if table.detection_confidence > existing_table.detection_confidence:
                        unique_tables.remove(existing_table)
                        unique_tables.append(table)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_tables.append(table)
        
        return unique_tables
    
    def _tables_are_similar(self, table1: TableObject, table2: TableObject, 
                           similarity_threshold: float = 0.9) -> bool:
        """Check if two tables are similar enough to be considered duplicates."""
        # Simple similarity check based on dimensions and content
        dims1 = table1.get_dimensions()
        dims2 = table2.get_dimensions()
        
        if dims1 != dims2:
            return False
        
        # Check content similarity
        total_cells = dims1[0] * dims1[1]
        if total_cells == 0:
            return True
        
        matching_cells = 0
        for i in range(dims1[0]):
            for j in range(dims1[1]):
                if table1.get_cell(i, j) == table2.get_cell(i, j):
                    matching_cells += 1
        
        similarity = matching_cells / total_cells
        return similarity >= similarity_threshold
