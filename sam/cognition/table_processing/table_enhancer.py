"""
Table Enhancer
==============

Post-processing enhancements for improved table understanding including
cell type detection, relationship analysis, and semantic enrichment.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from .table_parser import TableObject
from .role_classifier import RoleClassification
from .utils import TableUtils, CellDataType
from .config import get_table_config

logger = logging.getLogger(__name__)


@dataclass
class EnhancementResult:
    """Result of table enhancement processing."""
    enhanced_table: TableObject
    cell_types: List[List[CellDataType]]
    relationships: Dict[str, Any]
    semantic_metadata: Dict[str, Any]
    enhancement_metrics: Dict[str, float]


class TableEnhancer:
    """
    Post-processing enhancement system for tables.
    
    Provides cell type detection, relationship analysis,
    and semantic enrichment capabilities.
    """
    
    def __init__(self):
        """Initialize the table enhancer."""
        self.config = get_table_config()
        logger.info("TableEnhancer initialized")
    
    def enhance_table(self, table_object: TableObject,
                     classifications: Optional[List[List[RoleClassification]]] = None) -> EnhancementResult:
        """
        Perform comprehensive table enhancement.
        
        Args:
            table_object: Table to enhance
            classifications: Optional role classifications for context
            
        Returns:
            EnhancementResult with enhanced table and metadata
        """
        try:
            # Create enhanced copy of table
            enhanced_table = self._create_enhanced_copy(table_object)
            
            # Detect cell data types
            cell_types = self._detect_cell_types(enhanced_table)
            
            # Analyze relationships
            relationships = self._analyze_relationships(enhanced_table, classifications)
            
            # Generate semantic metadata
            semantic_metadata = self._generate_semantic_metadata(
                enhanced_table, classifications, cell_types
            )
            
            # Calculate enhancement metrics
            enhancement_metrics = self._calculate_enhancement_metrics(
                enhanced_table, cell_types, relationships
            )
            
            # Update table metadata with enhancements
            enhanced_table.table_metadata.update({
                "enhanced": True,
                "enhancement_timestamp": None,  # Would be set in real implementation
                "cell_type_analysis": True,
                "relationship_analysis": True,
                "semantic_enrichment": True
            })
            
            return EnhancementResult(
                enhanced_table=enhanced_table,
                cell_types=cell_types,
                relationships=relationships,
                semantic_metadata=semantic_metadata,
                enhancement_metrics=enhancement_metrics
            )
            
        except Exception as e:
            logger.error(f"Table enhancement failed: {e}")
            # Return minimal enhancement result
            return EnhancementResult(
                enhanced_table=table_object,
                cell_types=[],
                relationships={},
                semantic_metadata={},
                enhancement_metrics={"error": 1.0}
            )
    
    def _create_enhanced_copy(self, table_object: TableObject) -> TableObject:
        """Create an enhanced copy of the table object."""
        # For now, return a copy with updated metadata
        enhanced_metadata = table_object.table_metadata.copy()
        enhanced_metadata["enhancement_applied"] = True
        
        return TableObject(
            raw_data=table_object.raw_data.copy(),
            coordinates=table_object.coordinates.copy(),
            detection_confidence=table_object.detection_confidence,
            table_metadata=enhanced_metadata,
            quality_indicators=table_object.quality_indicators.copy(),
            source_format=table_object.source_format,
            table_id=table_object.table_id,
            title=table_object.title,
            caption=table_object.caption
        )
    
    def _detect_cell_types(self, table_object: TableObject) -> List[List[CellDataType]]:
        """Detect data types for all cells in the table."""
        cell_types = []
        
        for row in table_object.raw_data:
            row_types = []
            for cell in row:
                cell_type, _ = TableUtils.detect_cell_data_type(cell)
                row_types.append(cell_type)
            cell_types.append(row_types)
        
        return cell_types
    
    def _analyze_relationships(self, table_object: TableObject,
                             classifications: Optional[List[List[RoleClassification]]]) -> Dict[str, Any]:
        """Analyze relationships within the table."""
        relationships = {
            "header_data_mapping": {},
            "calculated_fields": [],
            "data_patterns": {},
            "column_relationships": {}
        }
        
        rows, cols = table_object.get_dimensions()
        
        # Analyze header-data relationships
        if classifications and rows > 1:
            # Find header cells
            headers = {}
            for col_idx in range(cols):
                if col_idx < len(classifications[0]):
                    if classifications[0][col_idx].role.value == "HEADER":
                        headers[col_idx] = table_object.get_cell(0, col_idx)
            
            relationships["header_data_mapping"] = headers
        
        # Detect calculated fields (cells that might be formulas or totals)
        if classifications:
            calculated_fields = []
            for row_idx, row_classifications in enumerate(classifications):
                for col_idx, classification in enumerate(row_classifications):
                    if classification.role.value in ["TOTAL", "FORMULA"]:
                        calculated_fields.append({
                            "position": (row_idx, col_idx),
                            "type": classification.role.value,
                            "content": table_object.get_cell(row_idx, col_idx)
                        })
            relationships["calculated_fields"] = calculated_fields
        
        # Analyze data patterns by column
        column_patterns = {}
        for col_idx in range(cols):
            column_data = []
            for row_idx in range(rows):
                cell_content = table_object.get_cell(row_idx, col_idx)
                if cell_content and cell_content.strip():
                    column_data.append(cell_content)
            
            if column_data:
                # Analyze data types in column
                data_types = [TableUtils.detect_cell_data_type(cell)[0] for cell in column_data]
                type_counts = {}
                for dt in data_types:
                    type_counts[dt.value] = type_counts.get(dt.value, 0) + 1
                
                column_patterns[col_idx] = {
                    "dominant_type": max(type_counts, key=type_counts.get) if type_counts else "unknown",
                    "type_distribution": type_counts,
                    "sample_values": column_data[:3]  # First 3 values as samples
                }
        
        relationships["data_patterns"] = column_patterns
        
        return relationships
    
    def _generate_semantic_metadata(self, table_object: TableObject,
                                   classifications: Optional[List[List[RoleClassification]]],
                                   cell_types: List[List[CellDataType]]) -> Dict[str, Any]:
        """Generate semantic metadata for the table."""
        metadata = {
            "table_purpose": "unknown",
            "domain_indicators": [],
            "data_characteristics": {},
            "structural_features": {}
        }
        
        # Analyze table purpose based on content
        all_content = []
        for row in table_object.raw_data:
            for cell in row:
                if cell and cell.strip():
                    all_content.append(cell.lower())
        
        content_text = " ".join(all_content)
        
        # Domain detection
        domain_keywords = {
            "financial": ["revenue", "profit", "cost", "budget", "sales", "$", "€", "£"],
            "scientific": ["measurement", "experiment", "data", "result", "analysis"],
            "inventory": ["product", "item", "quantity", "stock", "inventory"],
            "personnel": ["name", "employee", "department", "salary", "position"],
            "temporal": ["date", "time", "year", "month", "day", "period"]
        }
        
        detected_domains = []
        for domain, keywords in domain_keywords.items():
            if any(keyword in content_text for keyword in keywords):
                detected_domains.append(domain)
        
        metadata["domain_indicators"] = detected_domains
        
        # Data characteristics
        if cell_types:
            all_types = []
            for row_types in cell_types:
                all_types.extend([ct.value for ct in row_types])
            
            type_counts = {}
            for ct in all_types:
                type_counts[ct] = type_counts.get(ct, 0) + 1
            
            metadata["data_characteristics"] = {
                "type_distribution": type_counts,
                "primary_data_type": max(type_counts, key=type_counts.get) if type_counts else "unknown",
                "data_diversity": len(set(all_types))
            }
        
        # Structural features
        rows, cols = table_object.get_dimensions()
        metadata["structural_features"] = {
            "dimensions": {"rows": rows, "columns": cols},
            "has_headers": self._has_headers(classifications),
            "has_totals": self._has_totals(classifications),
            "complexity_score": self._calculate_complexity_score(table_object, classifications)
        }
        
        return metadata
    
    def _has_headers(self, classifications: Optional[List[List[RoleClassification]]]) -> bool:
        """Check if table has header cells."""
        if not classifications:
            return False
        
        for row_classifications in classifications:
            for classification in row_classifications:
                if classification.role.value == "HEADER":
                    return True
        return False
    
    def _has_totals(self, classifications: Optional[List[List[RoleClassification]]]) -> bool:
        """Check if table has total/sum cells."""
        if not classifications:
            return False
        
        for row_classifications in classifications:
            for classification in row_classifications:
                if classification.role.value == "TOTAL":
                    return True
        return False
    
    def _calculate_complexity_score(self, table_object: TableObject,
                                   classifications: Optional[List[List[RoleClassification]]]) -> float:
        """Calculate table complexity score."""
        rows, cols = table_object.get_dimensions()
        
        # Base complexity from size
        size_complexity = min(1.0, (rows * cols) / 100)
        
        # Role diversity complexity
        role_complexity = 0.0
        if classifications:
            all_roles = []
            for row_classifications in classifications:
                for classification in row_classifications:
                    all_roles.append(classification.role.value)
            
            unique_roles = len(set(all_roles))
            role_complexity = min(1.0, unique_roles / 9)  # 9 total possible roles
        
        # Content complexity (data type diversity)
        content_complexity = 0.0
        all_types = set()
        for row in table_object.raw_data:
            for cell in row:
                if cell and cell.strip():
                    cell_type, _ = TableUtils.detect_cell_data_type(cell)
                    all_types.add(cell_type)
        
        content_complexity = min(1.0, len(all_types) / 10)  # Normalize by reasonable type count
        
        # Combined complexity score
        return (size_complexity * 0.4 + role_complexity * 0.3 + content_complexity * 0.3)
    
    def _calculate_enhancement_metrics(self, table_object: TableObject,
                                     cell_types: List[List[CellDataType]],
                                     relationships: Dict[str, Any]) -> Dict[str, float]:
        """Calculate metrics for enhancement quality."""
        metrics = {}
        
        # Type detection coverage
        total_cells = sum(len(row) for row in table_object.raw_data)
        typed_cells = sum(len(row) for row in cell_types)
        metrics["type_detection_coverage"] = typed_cells / total_cells if total_cells > 0 else 0
        
        # Relationship detection score
        relationship_score = 0.0
        if relationships.get("header_data_mapping"):
            relationship_score += 0.3
        if relationships.get("calculated_fields"):
            relationship_score += 0.3
        if relationships.get("data_patterns"):
            relationship_score += 0.4
        
        metrics["relationship_detection_score"] = relationship_score
        
        # Enhancement completeness
        enhancement_features = [
            bool(cell_types),
            bool(relationships),
            "enhanced" in table_object.table_metadata
        ]
        metrics["enhancement_completeness"] = sum(enhancement_features) / len(enhancement_features)
        
        return metrics
