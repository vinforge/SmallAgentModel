"""
SAM Integration for Table Processing
===================================

Integration layer that connects the table processing system with SAM's
existing memory, synthesis, and web retrieval systems.

This implements Task 4 from task23.md - Enhanced Integration with SAM's Ecosystem.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from .table_parser import TableParser, TableObject
from .role_classifier import TableRoleClassifier, RoleClassification
from .table_validator import TableValidator, ValidationResult
from .table_enhancer import TableEnhancer, EnhancementResult
from .utils import TableUtils, CellDataType

logger = logging.getLogger(__name__)


@dataclass
class TableProcessingResult:
    """Result of complete table processing pipeline."""
    tables: List[TableObject]
    classifications: List[List[List[RoleClassification]]]
    validations: List[ValidationResult]
    enhancements: List[EnhancementResult]
    enhanced_chunks: List[Dict[str, Any]]
    processing_metrics: Dict[str, Any]


class TableAwareChunker:
    """
    Table-aware chunking system that integrates with SAM's enhanced chunker.
    
    Processes tables and creates enriched chunks with table metadata.
    """
    
    def __init__(self):
        """Initialize the table-aware chunker."""
        self.parser = TableParser()
        self.classifier = TableRoleClassifier()
        self.validator = TableValidator()
        self.enhancer = TableEnhancer()
        
        logger.info("TableAwareChunker initialized")
    
    def process_document_with_tables(self, doc_content: str, doc_type: str,
                                   document_context: Optional[str] = None) -> TableProcessingResult:
        """
        Process a document and extract table intelligence.
        
        Args:
            doc_content: Document content
            doc_type: Document type (html, markdown, pdf, etc.)
            document_context: Additional context about the document
            
        Returns:
            TableProcessingResult with all processing outputs
        """
        try:
            # Step 1: Extract tables
            tables = self.parser.extract_tables_from_document(doc_content, doc_type)
            logger.info(f"Extracted {len(tables)} tables from document")
            
            # Step 2: Classify roles for each table
            classifications = []
            for table in tables:
                table_classifications = self.classifier.predict(table, document_context)
                classifications.append(table_classifications)
            
            # Step 3: Validate tables
            validations = self.validator.validate_table_batch(tables)
            
            # Step 4: Enhance tables
            enhancements = []
            for i, table in enumerate(tables):
                table_classifications = classifications[i] if i < len(classifications) else None
                enhancement = self.enhancer.enhance_table(table, table_classifications)
                enhancements.append(enhancement)
            
            # Step 5: Create enhanced chunks
            enhanced_chunks = self._create_enhanced_chunks(
                tables, classifications, enhancements, document_context
            )
            
            # Step 6: Calculate processing metrics
            processing_metrics = self._calculate_processing_metrics(
                tables, classifications, validations, enhancements
            )
            
            return TableProcessingResult(
                tables=tables,
                classifications=classifications,
                validations=validations,
                enhancements=enhancements,
                enhanced_chunks=enhanced_chunks,
                processing_metrics=processing_metrics
            )
            
        except Exception as e:
            logger.error(f"Table processing failed: {e}")
            return TableProcessingResult(
                tables=[],
                classifications=[],
                validations=[],
                enhancements=[],
                enhanced_chunks=[],
                processing_metrics={"error": str(e)}
            )
    
    def _create_enhanced_chunks(self, tables: List[TableObject],
                              classifications: List[List[List[RoleClassification]]],
                              enhancements: List[EnhancementResult],
                              document_context: Optional[str]) -> List[Dict[str, Any]]:
        """Create enhanced chunks with table metadata."""
        enhanced_chunks = []
        
        for table_idx, table in enumerate(tables):
            table_classifications = classifications[table_idx] if table_idx < len(classifications) else []
            enhancement = enhancements[table_idx] if table_idx < len(enhancements) else None
            
            # Create chunks for each cell
            rows, cols = table.get_dimensions()
            
            for row_idx in range(rows):
                for col_idx in range(cols):
                    cell_content = table.get_cell(row_idx, col_idx)
                    
                    if not cell_content or not cell_content.strip():
                        continue  # Skip empty cells
                    
                    # Get classification for this cell
                    cell_classification = None
                    if (row_idx < len(table_classifications) and 
                        col_idx < len(table_classifications[row_idx])):
                        cell_classification = table_classifications[row_idx][col_idx]
                    
                    # Get cell data type
                    cell_data_type, _ = TableUtils.detect_cell_data_type(cell_content)
                    
                    # Create enhanced chunk metadata
                    chunk_metadata = self._create_table_chunk_metadata(
                        table, table_idx, row_idx, col_idx, cell_classification,
                        cell_data_type, enhancement, document_context
                    )
                    
                    enhanced_chunks.append(chunk_metadata)
        
        return enhanced_chunks
    
    def _create_table_chunk_metadata(self, table: TableObject, table_idx: int,
                                   row_idx: int, col_idx: int,
                                   cell_classification: Optional[RoleClassification],
                                   cell_data_type: CellDataType,
                                   enhancement: Optional[EnhancementResult],
                                   document_context: Optional[str]) -> Dict[str, Any]:
        """Create metadata for a table cell chunk."""
        cell_content = table.get_cell(row_idx, col_idx)
        
        # Base chunk metadata
        chunk_metadata = {
            "content": cell_content,
            "chunk_type": "TABLE_CELL",
            "source_location": f"table_{table_idx}_cell_{row_idx}_{col_idx}",
            
            # Table-specific metadata
            "is_table_part": True,
            "table_id": table.table_id or f"table_{table_idx}",
            "table_title": table.title or table.caption,
            "cell_coordinates": (row_idx, col_idx),
            "cell_data_type": cell_data_type.value,
            "table_context": document_context,
            
            # Table structure metadata
            "table_structure": {
                "dimensions": table.get_dimensions(),
                "source_format": table.source_format,
                "detection_confidence": table.detection_confidence,
                "quality_indicators": table.quality_indicators,
                "table_metadata": table.table_metadata
            }
        }
        
        # Add classification metadata
        if cell_classification:
            chunk_metadata.update({
                "cell_role": cell_classification.role.value,
                "confidence_score": cell_classification.confidence,
                "classification_reasoning": cell_classification.reasoning,
                "context_factors": cell_classification.context_factors
            })
        
        # Add enhancement metadata
        if enhancement:
            chunk_metadata.update({
                "enhancement_metrics": enhancement.enhancement_metrics,
                "semantic_metadata": enhancement.semantic_metadata,
                "relationships": enhancement.relationships
            })
        
        return chunk_metadata
    
    def _calculate_processing_metrics(self, tables: List[TableObject],
                                    classifications: List[List[List[RoleClassification]]],
                                    validations: List[ValidationResult],
                                    enhancements: List[EnhancementResult]) -> Dict[str, Any]:
        """Calculate processing metrics for monitoring and optimization."""
        metrics = {
            "tables_processed": len(tables),
            "total_cells_processed": 0,
            "classification_accuracy": 0.0,
            "validation_success_rate": 0.0,
            "enhancement_completeness": 0.0,
            "average_confidence": 0.0,
            "role_distribution": {},
            "data_type_distribution": {},
            "quality_scores": []
        }
        
        if not tables:
            return metrics
        
        # Calculate cell and classification metrics
        total_cells = 0
        total_confidence = 0.0
        confidence_count = 0
        role_counts = {}
        
        for table_idx, table in enumerate(tables):
            rows, cols = table.get_dimensions()
            total_cells += rows * cols
            
            if table_idx < len(classifications):
                table_classifications = classifications[table_idx]
                for row_classifications in table_classifications:
                    for cell_classification in row_classifications:
                        total_confidence += cell_classification.confidence
                        confidence_count += 1
                        
                        role = cell_classification.role.value
                        role_counts[role] = role_counts.get(role, 0) + 1
        
        metrics["total_cells_processed"] = total_cells
        metrics["average_confidence"] = total_confidence / confidence_count if confidence_count > 0 else 0.0
        metrics["role_distribution"] = role_counts
        
        # Calculate validation metrics
        if validations:
            valid_count = sum(1 for v in validations if v.is_valid)
            metrics["validation_success_rate"] = valid_count / len(validations)
            metrics["quality_scores"] = [v.quality_score for v in validations]
        
        # Calculate enhancement metrics
        if enhancements:
            enhancement_scores = [
                e.enhancement_metrics.get("enhancement_completeness", 0.0)
                for e in enhancements
                if e.enhancement_metrics
            ]
            metrics["enhancement_completeness"] = (
                sum(enhancement_scores) / len(enhancement_scores)
                if enhancement_scores else 0.0
            )
        
        return metrics


@dataclass
class ReconstructedTable:
    """Reconstructed table from memory chunks."""
    table_id: str
    title: Optional[str]
    dimensions: Tuple[int, int]
    data: List[List[str]]
    headers: List[str]
    role_matrix: List[List[str]]
    data_types: List[List[str]]
    metadata: Dict[str, Any]
    confidence_scores: List[List[float]]
    source_document: str


class TableAwareRetrieval:
    """
    Advanced table-aware retrieval system for Phase 2.

    Reconstructs complete tables from Phase 1 metadata and enables
    sophisticated querying for the Table-to-Code Expert Tool.
    """

    def __init__(self, memory_store):
        """Initialize with memory store reference."""
        self.memory_store = memory_store
        self._table_cache = {}  # Cache for reconstructed tables
        logger.info("TableAwareRetrieval initialized for Phase 2")
    
    def search_table_content(self, query: str, role_filter: Optional[str] = None,
                           table_id_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search specifically within table content with role and table filtering.
        
        Args:
            query: Search query
            role_filter: Filter by cell role (HEADER, DATA, etc.)
            table_id_filter: Filter by specific table ID
            
        Returns:
            List of matching table chunks
        """
        try:
            # Search for table content using tags
            tags = ["table"]

            # Perform search with table tag
            search_results = self.memory_store.search_memories(
                query=query if query.strip() else "table",
                max_results=50,  # Get more results to filter
                tags=tags
            )

            # Filter results based on table metadata
            filtered_results = []
            for result in search_results:
                metadata = result.chunk.metadata

                # Check if it's a table part
                if not metadata.get("is_table_part", False):
                    continue

                # Apply role filter
                if role_filter and metadata.get("cell_role") != role_filter:
                    continue

                # Apply table ID filter
                if table_id_filter and metadata.get("table_id") != table_id_filter:
                    continue

                # Convert to dict format for compatibility
                filtered_results.append({
                    "content": result.chunk.content,
                    "metadata": metadata,
                    "similarity": result.similarity_score
                })

            logger.info(f"Table search returned {len(filtered_results)} results")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Table search failed: {e}")
            return []
    
    def get_table_summary(self, table_id: str) -> Dict[str, Any]:
        """Get comprehensive summary of a specific table."""
        try:
            # Get all chunks for this table
            table_chunks = self.search_table_content("", table_id_filter=table_id)
            
            if not table_chunks:
                return {"error": f"No table found with ID: {table_id}"}
            
            # Analyze table structure and content
            summary = {
                "table_id": table_id,
                "total_cells": len(table_chunks),
                "role_distribution": {},
                "data_type_distribution": {},
                "quality_metrics": {}
            }
            
            # Aggregate statistics
            for chunk in table_chunks:
                metadata = chunk.get("metadata", {})
                
                # Role distribution
                role = metadata.get("cell_role", "UNKNOWN")
                summary["role_distribution"][role] = summary["role_distribution"].get(role, 0) + 1
                
                # Data type distribution
                data_type = metadata.get("cell_data_type", "unknown")
                summary["data_type_distribution"][data_type] = summary["data_type_distribution"].get(data_type, 0) + 1
            
            return summary
            
        except Exception as e:
            logger.error(f"Table summary generation failed: {e}")
            return {"error": str(e)}

    def reconstruct_table(self, table_id: str) -> Optional[ReconstructedTable]:
        """
        Reconstruct a complete table from memory chunks.

        Args:
            table_id: ID of the table to reconstruct

        Returns:
            ReconstructedTable object or None if not found
        """
        try:
            # Check cache first
            if table_id in self._table_cache:
                return self._table_cache[table_id]

            # Get all chunks for this table
            table_chunks = self.search_table_content("", table_id_filter=table_id)

            if not table_chunks:
                logger.warning(f"No chunks found for table ID: {table_id}")
                return None

            # Organize chunks by coordinates
            chunk_matrix = {}
            max_row, max_col = 0, 0
            table_title = None
            source_document = None
            table_metadata = {}

            for chunk in table_chunks:
                metadata = chunk.get("metadata", {})
                coordinates = metadata.get("cell_coordinates")

                if coordinates:
                    row, col = coordinates
                    chunk_matrix[(row, col)] = {
                        "content": chunk.get("content", ""),
                        "role": metadata.get("cell_role", "DATA"),
                        "data_type": metadata.get("cell_data_type", "text"),
                        "confidence": metadata.get("confidence_score", 0.0)
                    }

                    max_row = max(max_row, row)
                    max_col = max(max_col, col)

                    # Extract table-level metadata
                    if table_title is None:
                        table_title = metadata.get("table_title")
                    if source_document is None:
                        source_document = metadata.get("source", "unknown")
                    if not table_metadata:
                        table_metadata = metadata.get("table_structure", {})

            # Build the reconstructed table
            dimensions = (max_row + 1, max_col + 1)
            data = [["" for _ in range(dimensions[1])] for _ in range(dimensions[0])]
            role_matrix = [["DATA" for _ in range(dimensions[1])] for _ in range(dimensions[0])]
            data_types = [["text" for _ in range(dimensions[1])] for _ in range(dimensions[0])]
            confidence_scores = [[0.0 for _ in range(dimensions[1])] for _ in range(dimensions[0])]

            # Fill the matrices
            for (row, col), cell_data in chunk_matrix.items():
                data[row][col] = cell_data["content"]
                role_matrix[row][col] = cell_data["role"]
                data_types[row][col] = cell_data["data_type"]
                confidence_scores[row][col] = cell_data["confidence"]

            # Extract headers (first row with HEADER role)
            headers = []
            for col in range(dimensions[1]):
                if role_matrix[0][col] == "HEADER":
                    headers.append(data[0][col])
                else:
                    headers.append(f"Column_{col}")

            reconstructed_table = ReconstructedTable(
                table_id=table_id,
                title=table_title,
                dimensions=dimensions,
                data=data,
                headers=headers,
                role_matrix=role_matrix,
                data_types=data_types,
                metadata=table_metadata,
                confidence_scores=confidence_scores,
                source_document=source_document
            )

            # Cache the result
            self._table_cache[table_id] = reconstructed_table

            logger.info(f"Successfully reconstructed table {table_id} with dimensions {dimensions}")
            return reconstructed_table

        except Exception as e:
            logger.error(f"Table reconstruction failed for {table_id}: {e}")
            return None

    def find_tables_by_content(self, query: str, max_results: int = 5) -> List[str]:
        """
        Find table IDs that contain content matching the query and have retrievable data.

        Args:
            query: Search query
            max_results: Maximum number of table IDs to return

        Returns:
            List of table IDs that actually have data
        """
        try:
            # Search for table content
            results = self.search_table_content(query)

            # Extract unique table IDs and validate they have data
            validated_table_ids = []
            candidate_table_ids = set()

            for result in results:
                metadata = result.get("metadata", {})
                table_id = metadata.get("table_id")
                if table_id and table_id not in candidate_table_ids:
                    candidate_table_ids.add(table_id)

                    # Validate that this table actually has retrievable chunks
                    table_chunks = self.search_table_content("", table_id_filter=table_id)
                    if table_chunks:
                        validated_table_ids.append(table_id)
                        logger.debug(f"Validated table {table_id} with {len(table_chunks)} chunks")

                        if len(validated_table_ids) >= max_results:
                            break
                    else:
                        logger.warning(f"Table {table_id} found in search but has no retrievable chunks")

            logger.info(f"Table search validated {len(validated_table_ids)} tables from {len(candidate_table_ids)} candidates")
            return validated_table_ids

        except Exception as e:
            logger.error(f"Table search by content failed: {e}")
            return []

    def get_table_data_for_analysis(self, table_id: str) -> Optional[Dict[str, Any]]:
        """
        Get table data formatted for data analysis and code generation.

        Args:
            table_id: ID of the table

        Returns:
            Dictionary with structured data for analysis
        """
        try:
            table = self.reconstruct_table(table_id)
            if not table:
                logger.warning(f"Failed to reconstruct table {table_id}")
                # Fallback: try to get data directly from chunks
                return self._get_table_data_from_chunks(table_id)

            # Extract data rows (skip header row)
            data_rows = []
            header_row_idx = None

            # Find the header row
            for row_idx, row_roles in enumerate(table.role_matrix):
                if any(role == "HEADER" for role in row_roles):
                    header_row_idx = row_idx
                    break

            # Extract data starting after header
            start_row = (header_row_idx + 1) if header_row_idx is not None else 0

            for row_idx in range(start_row, table.dimensions[0]):
                row_data = {}
                for col_idx, header in enumerate(table.headers):
                    if col_idx < len(table.data[row_idx]):
                        cell_value = table.data[row_idx][col_idx]
                        cell_type = table.data_types[row_idx][col_idx]

                        # Convert to appropriate Python type
                        converted_value = self._convert_cell_value(cell_value, cell_type)
                        row_data[header] = converted_value

                data_rows.append(row_data)

            return {
                "table_id": table_id,
                "title": table.title,
                "headers": table.headers,
                "data": data_rows,
                "dimensions": table.dimensions,
                "source": table.source_document,
                "metadata": table.metadata
            }

        except Exception as e:
            logger.error(f"Failed to get table data for analysis: {e}")
            return None

    def _convert_cell_value(self, value: str, data_type: str):
        """Convert cell value to appropriate Python type."""
        if not value or value.strip() == "":
            return None

        try:
            if data_type == "integer":
                return int(value.replace(",", ""))
            elif data_type == "float":
                return float(value.replace(",", ""))
            elif data_type == "currency":
                # Remove currency symbols and convert to float
                cleaned = value.replace("$", "").replace(",", "").strip()
                return float(cleaned)
            elif data_type == "percentage":
                # Remove % and convert to decimal
                cleaned = value.replace("%", "").strip()
                return float(cleaned) / 100.0
            else:
                return value.strip()
        except (ValueError, AttributeError):
            return value

    def _get_table_data_from_chunks(self, table_id: str) -> Optional[Dict[str, Any]]:
        """
        Fallback method to get table data directly from chunks when reconstruction fails.

        Args:
            table_id: ID of the table

        Returns:
            Dictionary with structured data for analysis
        """
        try:
            # Get all chunks for this table
            table_chunks = self.search_table_content("", table_id_filter=table_id)

            if not table_chunks:
                logger.warning(f"No chunks found for table ID: {table_id}")
                return None

            # Extract data from chunks
            data_rows = []
            headers = []

            # Group chunks by row
            rows_data = {}
            for chunk in table_chunks:
                metadata = chunk.get("metadata", {})
                coordinates = metadata.get("cell_coordinates")

                if coordinates:
                    row, col = coordinates
                    if row not in rows_data:
                        rows_data[row] = {}

                    rows_data[row][col] = {
                        "content": chunk.get("content", ""),
                        "role": metadata.get("cell_role", "DATA")
                    }

            # Sort rows and build data structure
            sorted_rows = sorted(rows_data.keys())

            if sorted_rows:
                # Extract headers from first row or use column indices
                first_row = rows_data[sorted_rows[0]]
                max_col = max(first_row.keys()) if first_row else 0

                for col in range(max_col + 1):
                    if col in first_row:
                        headers.append(first_row[col]["content"])
                    else:
                        headers.append(f"Column_{col}")

                # Extract data rows (skip header row if it exists)
                start_row = 1 if any(cell.get("role") == "HEADER" for cell in first_row.values()) else 0

                for row_idx in sorted_rows[start_row:]:
                    row_data = {}
                    row_cells = rows_data[row_idx]

                    for col_idx, header in enumerate(headers):
                        if col_idx in row_cells:
                            cell_value = row_cells[col_idx]["content"]
                            # Try to convert to appropriate type
                            converted_value = self._convert_cell_value(
                                cell_value,
                                "currency" if "sales" in header.lower() or "$" in cell_value else "text"
                            )
                            row_data[header] = converted_value
                        else:
                            row_data[header] = None

                    data_rows.append(row_data)

            if not data_rows:
                logger.warning(f"No data rows extracted for table {table_id}")
                return None

            return {
                "table_id": table_id,
                "title": f"Table {table_id}",
                "headers": headers,
                "data": data_rows,
                "dimensions": (len(data_rows), len(headers)),
                "source": "memory_chunks",
                "metadata": {"fallback_method": True}
            }

        except Exception as e:
            logger.error(f"Fallback table data extraction failed for {table_id}: {e}")
            return None


# Global instances for easy access
_table_aware_chunker: Optional[TableAwareChunker] = None
_table_aware_retrieval: Optional[TableAwareRetrieval] = None


def get_table_aware_chunker() -> TableAwareChunker:
    """Get global table-aware chunker instance."""
    global _table_aware_chunker
    if _table_aware_chunker is None:
        _table_aware_chunker = TableAwareChunker()
    return _table_aware_chunker


def get_table_aware_retrieval(memory_store) -> TableAwareRetrieval:
    """Get global table-aware retrieval instance."""
    global _table_aware_retrieval
    if _table_aware_retrieval is None:
        _table_aware_retrieval = TableAwareRetrieval(memory_store)
    return _table_aware_retrieval
