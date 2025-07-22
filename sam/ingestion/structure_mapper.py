#!/usr/bin/env python3
"""
Document Structure Mapper
Advanced structural analysis for intelligent document processing.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class StructuralElement(Enum):
    """Types of structural elements in documents."""
    HEADER = "header"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST = "list"
    FORMULA = "formula"
    CODE_BLOCK = "code_block"
    QUOTE = "quote"
    FOOTNOTE = "footnote"
    SECTION_BREAK = "section_break"

@dataclass
class DocumentElement:
    """Represents a structural element in the document."""
    element_type: StructuralElement
    content: str
    start_pos: int
    end_pos: int
    level: int  # Hierarchy level (for headers, lists)
    metadata: Dict[str, Any]
    parent_id: Optional[str] = None
    element_id: Optional[str] = None

@dataclass
class DocumentMap:
    """Complete structural map of a document."""
    elements: List[DocumentElement]
    hierarchy: Dict[str, List[str]]  # parent_id -> [child_ids]
    sections: List[Dict[str, Any]]   # Section boundaries and metadata
    atomic_blocks: List[Dict[str, Any]]  # Blocks that should not be split
    processing_zones: Dict[str, List[int]]  # Special processing areas
    
class StructureMapper:
    """
    Maps the detailed structure of documents for intelligent chunking.
    """
    
    def __init__(self):
        self.header_patterns = [
            r'^#{1,6}\s+(.+)$',  # Markdown headers
            r'^([A-Z][A-Z\s]{2,50})$',  # ALL CAPS headers
            r'^\d+\.?\s+([A-Z].{5,100})$',  # Numbered sections
            r'^([A-Z][a-z\s]{5,50}):?\s*$',  # Title case headers
            r'^([IVX]+\.?\s+[A-Z].{5,100})$',  # Roman numeral headers
        ]
        
        self.table_patterns = [
            r'\|.*\|.*\|',  # Markdown tables
            r'^\s*\+[-=]+\+',  # ASCII tables
            r'Table\s+\d+[:\.]',  # Table captions
            r'Exhibit\s+\d+[:\.]',  # Exhibit captions
        ]
        
        self.list_patterns = [
            r'^\s*[-*+]\s+',  # Bullet lists
            r'^\s*\d+\.\s+',  # Numbered lists
            r'^\s*[a-z]\)\s+',  # Lettered lists
            r'^\s*[IVX]+\.\s+',  # Roman numeral lists
        ]
        
        self.formula_patterns = [
            r'[A-Za-z_]\w*\s*=\s*[^=\n]+',  # Variable assignments
            r'\$[A-Z]+\$?\d+',  # Excel cell references
            r'∑|∏|∫|√|±|≤|≥|≠|≈|∞',  # Mathematical symbols
            r'\b(?:sin|cos|tan|log|ln|exp|sqrt)\s*\(',  # Mathematical functions
        ]
        
        self.code_patterns = [
            r'```[\s\S]*?```',  # Markdown code blocks
            r'`[^`\n]+`',  # Inline code
            r'^\s{4,}[^\s].*$',  # Indented code blocks
            r'def\s+\w+\s*\(',  # Python functions
            r'function\s+\w+\s*\(',  # JavaScript functions
        ]
        
        logger.info("StructureMapper initialized")
    
    def map_document_structure(self, content: str) -> DocumentMap:
        """
        Create a comprehensive structural map of the document.
        
        Args:
            content: Full document text
            
        Returns:
            Complete document map with all structural elements
        """
        try:
            lines = content.split('\n')
            elements = []
            current_pos = 0
            
            # First pass: Identify all structural elements
            for line_num, line in enumerate(lines):
                line_start = current_pos
                line_end = current_pos + len(line)
                
                # Check for headers
                header_match = self._detect_header(line, line_num)
                if header_match:
                    elements.append(DocumentElement(
                        element_type=StructuralElement.HEADER,
                        content=header_match['content'],
                        start_pos=line_start,
                        end_pos=line_end,
                        level=header_match['level'],
                        metadata={'header_type': header_match['type']},
                        element_id=f"header_{len(elements)}"
                    ))
                
                # Check for lists
                list_match = self._detect_list_item(line)
                if list_match:
                    elements.append(DocumentElement(
                        element_type=StructuralElement.LIST,
                        content=list_match['content'],
                        start_pos=line_start,
                        end_pos=line_end,
                        level=list_match['level'],
                        metadata={'list_type': list_match['type']},
                        element_id=f"list_{len(elements)}"
                    ))
                
                # Check for formulas
                formula_matches = self._detect_formulas(line)
                for formula in formula_matches:
                    elements.append(DocumentElement(
                        element_type=StructuralElement.FORMULA,
                        content=formula['content'],
                        start_pos=line_start + formula['start'],
                        end_pos=line_start + formula['end'],
                        level=0,
                        metadata={'formula_type': formula['type']},
                        element_id=f"formula_{len(elements)}"
                    ))
                
                current_pos = line_end + 1  # +1 for newline
            
            # Second pass: Detect multi-line elements (tables, code blocks)
            multiline_elements = self._detect_multiline_elements(content)
            elements.extend(multiline_elements)
            
            # Third pass: Build hierarchy and relationships
            hierarchy = self._build_hierarchy(elements)
            
            # Fourth pass: Identify sections and atomic blocks
            sections = self._identify_sections(elements)
            atomic_blocks = self._identify_atomic_blocks(elements, content)
            
            # Fifth pass: Define processing zones
            processing_zones = self._define_processing_zones(elements, content)
            
            document_map = DocumentMap(
                elements=sorted(elements, key=lambda x: x.start_pos),
                hierarchy=hierarchy,
                sections=sections,
                atomic_blocks=atomic_blocks,
                processing_zones=processing_zones
            )
            
            logger.info(f"📋 Document mapped: {len(elements)} elements, {len(sections)} sections")
            return document_map
            
        except Exception as e:
            logger.error(f"Structure mapping failed: {e}")
            # Return minimal structure
            return DocumentMap([], {}, [], [], {})
    
    def _detect_header(self, line: str, line_num: int) -> Optional[Dict[str, Any]]:
        """Detect if a line is a header."""
        line_stripped = line.strip()
        if not line_stripped or len(line_stripped) > 200:
            return None
        
        # Markdown headers
        md_match = re.match(r'^(#{1,6})\s+(.+)$', line_stripped)
        if md_match:
            return {
                'content': md_match.group(2),
                'level': len(md_match.group(1)),
                'type': 'markdown'
            }
        
        # ALL CAPS headers (likely important)
        if line_stripped.isupper() and 5 <= len(line_stripped) <= 100:
            return {
                'content': line_stripped,
                'level': 1,
                'type': 'caps'
            }
        
        # Numbered sections
        num_match = re.match(r'^(\d+\.?\d*\.?)\s+([A-Z].+)$', line_stripped)
        if num_match:
            return {
                'content': num_match.group(2),
                'level': num_match.group(1).count('.') + 1,
                'type': 'numbered'
            }
        
        # Title case headers (heuristic)
        if (line_stripped[0].isupper() and 
            5 <= len(line_stripped) <= 100 and
            line_stripped.count(' ') <= 10 and
            not line_stripped.endswith('.')):
            return {
                'content': line_stripped,
                'level': 2,
                'type': 'title_case'
            }
        
        return None
    
    def _detect_list_item(self, line: str) -> Optional[Dict[str, Any]]:
        """Detect if a line is a list item."""
        # Bullet lists
        bullet_match = re.match(r'^(\s*)([-*+])\s+(.+)$', line)
        if bullet_match:
            return {
                'content': bullet_match.group(3),
                'level': len(bullet_match.group(1)) // 2,  # Indentation level
                'type': 'bullet'
            }
        
        # Numbered lists
        num_match = re.match(r'^(\s*)(\d+\.)\s+(.+)$', line)
        if num_match:
            return {
                'content': num_match.group(3),
                'level': len(num_match.group(1)) // 2,
                'type': 'numbered'
            }
        
        return None
    
    def _detect_formulas(self, line: str) -> List[Dict[str, Any]]:
        """Detect mathematical formulas in a line."""
        formulas = []
        
        # Variable assignments
        for match in re.finditer(r'([A-Za-z_]\w*)\s*=\s*([^=\n]+)', line):
            formulas.append({
                'content': match.group(0),
                'start': match.start(),
                'end': match.end(),
                'type': 'assignment',
                'variable': match.group(1),
                'expression': match.group(2).strip()
            })
        
        # Excel references
        for match in re.finditer(r'\$?[A-Z]+\$?\d+', line):
            formulas.append({
                'content': match.group(0),
                'start': match.start(),
                'end': match.end(),
                'type': 'excel_reference'
            })
        
        # Mathematical symbols
        for match in re.finditer(r'[∑∏∫√±≤≥≠≈∞]', line):
            formulas.append({
                'content': match.group(0),
                'start': match.start(),
                'end': match.end(),
                'type': 'mathematical_symbol'
            })
        
        return formulas
    
    def _detect_multiline_elements(self, content: str) -> List[DocumentElement]:
        """Detect elements that span multiple lines."""
        elements = []
        
        # Markdown code blocks
        for match in re.finditer(r'```(.*?)\n(.*?)```', content, re.DOTALL):
            elements.append(DocumentElement(
                element_type=StructuralElement.CODE_BLOCK,
                content=match.group(2),
                start_pos=match.start(),
                end_pos=match.end(),
                level=0,
                metadata={
                    'language': match.group(1).strip(),
                    'block_type': 'fenced'
                },
                element_id=f"code_{len(elements)}"
            ))
        
        # Tables (simple detection)
        table_lines = []
        current_table_start = None
        
        for line_num, line in enumerate(content.split('\n')):
            if '|' in line and line.count('|') >= 2:
                if current_table_start is None:
                    current_table_start = line_num
                table_lines.append(line)
            else:
                if table_lines:
                    # End of table
                    table_content = '\n'.join(table_lines)
                    start_pos = content.find(table_lines[0])
                    end_pos = start_pos + len(table_content)
                    
                    elements.append(DocumentElement(
                        element_type=StructuralElement.TABLE,
                        content=table_content,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        level=0,
                        metadata={
                            'rows': len(table_lines),
                            'columns': table_lines[0].count('|') - 1 if table_lines else 0
                        },
                        element_id=f"table_{len(elements)}"
                    ))
                    
                    table_lines = []
                    current_table_start = None
        
        return elements
    
    def _build_hierarchy(self, elements: List[DocumentElement]) -> Dict[str, List[str]]:
        """Build hierarchical relationships between elements."""
        hierarchy = {}
        header_stack = []  # Stack of (level, element_id)
        
        for element in elements:
            if element.element_type == StructuralElement.HEADER:
                # Pop headers of same or lower level
                while header_stack and header_stack[-1][0] >= element.level:
                    header_stack.pop()
                
                # Set parent relationship
                if header_stack:
                    parent_id = header_stack[-1][1]
                    element.parent_id = parent_id
                    if parent_id not in hierarchy:
                        hierarchy[parent_id] = []
                    hierarchy[parent_id].append(element.element_id)
                
                header_stack.append((element.level, element.element_id))
            
            else:
                # Non-header elements belong to the current header
                if header_stack:
                    parent_id = header_stack[-1][1]
                    element.parent_id = parent_id
                    if parent_id not in hierarchy:
                        hierarchy[parent_id] = []
                    hierarchy[parent_id].append(element.element_id)
        
        return hierarchy
    
    def _identify_sections(self, elements: List[DocumentElement]) -> List[Dict[str, Any]]:
        """Identify logical sections in the document."""
        sections = []
        current_section = None
        
        for element in elements:
            if element.element_type == StructuralElement.HEADER:
                # End previous section
                if current_section:
                    current_section['end_pos'] = element.start_pos
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    'title': element.content,
                    'level': element.level,
                    'start_pos': element.start_pos,
                    'end_pos': None,  # Will be set when section ends
                    'element_ids': [element.element_id]
                }
            
            elif current_section:
                current_section['element_ids'].append(element.element_id)
        
        # Close final section
        if current_section:
            current_section['end_pos'] = elements[-1].end_pos if elements else 0
            sections.append(current_section)
        
        return sections
    
    def _identify_atomic_blocks(self, elements: List[DocumentElement], content: str) -> List[Dict[str, Any]]:
        """Identify blocks that should not be split during chunking."""
        atomic_blocks = []
        
        for element in elements:
            # Tables should not be split
            if element.element_type == StructuralElement.TABLE:
                atomic_blocks.append({
                    'type': 'table',
                    'start_pos': element.start_pos,
                    'end_pos': element.end_pos,
                    'reason': 'preserve_table_structure',
                    'element_id': element.element_id
                })
            
            # Code blocks should not be split
            elif element.element_type == StructuralElement.CODE_BLOCK:
                atomic_blocks.append({
                    'type': 'code_block',
                    'start_pos': element.start_pos,
                    'end_pos': element.end_pos,
                    'reason': 'preserve_code_integrity',
                    'element_id': element.element_id
                })
            
            # Formulas should not be split
            elif element.element_type == StructuralElement.FORMULA:
                atomic_blocks.append({
                    'type': 'formula',
                    'start_pos': element.start_pos,
                    'end_pos': element.end_pos,
                    'reason': 'preserve_mathematical_expression',
                    'element_id': element.element_id
                })
        
        return atomic_blocks
    
    def _define_processing_zones(self, elements: List[DocumentElement], content: str) -> Dict[str, List[int]]:
        """Define special processing zones for enhanced analysis."""
        zones = {
            'high_value_content': [],  # Important sections
            'numerical_data': [],     # Areas with lots of numbers
            'technical_content': [],  # Code, formulas, technical terms
            'structural_markers': [] # Headers, section breaks
        }
        
        for element in elements:
            # High-value content (headers, important sections)
            if element.element_type == StructuralElement.HEADER:
                zones['high_value_content'].extend(range(element.start_pos, element.end_pos))
                zones['structural_markers'].extend(range(element.start_pos, element.end_pos))
            
            # Technical content
            elif element.element_type in [StructuralElement.CODE_BLOCK, StructuralElement.FORMULA]:
                zones['technical_content'].extend(range(element.start_pos, element.end_pos))
            
            # Numerical data (tables often contain numbers)
            elif element.element_type == StructuralElement.TABLE:
                zones['numerical_data'].extend(range(element.start_pos, element.end_pos))
        
        return zones

# Global instance
_structure_mapper = None

def get_structure_mapper() -> StructureMapper:
    """Get or create the global structure mapper instance."""
    global _structure_mapper
    if _structure_mapper is None:
        _structure_mapper = StructureMapper()
    return _structure_mapper
