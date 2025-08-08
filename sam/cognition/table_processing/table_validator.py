"""
Table Validator
===============

Quality validation and structure verification for detected tables.
Ensures reliable table processing through comprehensive validation checks.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .table_parser import TableObject
from .role_classifier import RoleClassification
from .config import get_table_config
from .utils import TableUtils

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of table validation."""
    is_valid: bool
    quality_score: float
    validation_errors: List[str]
    validation_warnings: List[str]
    metrics: Dict[str, Any]


class TableValidator:
    """
    Quality validation and structure verification for tables.
    
    Performs comprehensive validation including structure, content,
    and consistency checks.
    """
    
    def __init__(self):
        """Initialize the table validator."""
        self.config = get_table_config()
        logger.info("TableValidator initialized")
    
    def validate_table(self, table_object: TableObject, 
                      classifications: Optional[List[List[RoleClassification]]] = None) -> ValidationResult:
        """
        Perform comprehensive table validation.
        
        Args:
            table_object: Table to validate
            classifications: Optional role classifications for enhanced validation
            
        Returns:
            ValidationResult with validation status and metrics
        """
        try:
            errors = []
            warnings = []
            metrics = {}
            
            # Structure validation
            structure_result = self._validate_structure(table_object)
            errors.extend(structure_result["errors"])
            warnings.extend(structure_result["warnings"])
            metrics.update(structure_result["metrics"])
            
            # Content validation
            content_result = self._validate_content(table_object)
            errors.extend(content_result["errors"])
            warnings.extend(content_result["warnings"])
            metrics.update(content_result["metrics"])
            
            # Consistency validation
            consistency_result = self._validate_consistency(table_object, classifications)
            errors.extend(consistency_result["errors"])
            warnings.extend(consistency_result["warnings"])
            metrics.update(consistency_result["metrics"])
            
            # Calculate overall quality score
            quality_score = self._calculate_quality_score(metrics)
            
            # Determine if table is valid
            is_valid = (
                len(errors) == 0 and 
                quality_score >= self.config.validation.quality_threshold
            )
            
            return ValidationResult(
                is_valid=is_valid,
                quality_score=quality_score,
                validation_errors=errors,
                validation_warnings=warnings,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Table validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                quality_score=0.0,
                validation_errors=[f"Validation error: {str(e)}"],
                validation_warnings=[],
                metrics={}
            )
    
    def _validate_structure(self, table_object: TableObject) -> Dict[str, Any]:
        """Validate table structure."""
        errors = []
        warnings = []
        metrics = {}
        
        # Check dimensions
        rows, cols = table_object.get_dimensions()
        metrics["rows"] = rows
        metrics["columns"] = cols
        
        # Minimum size check
        min_rows, min_cols = self.config.detection.min_table_size
        if rows < min_rows or cols < min_cols:
            errors.append(f"Table too small: {rows}x{cols}, minimum: {min_rows}x{min_cols}")
        
        # Maximum size check
        max_rows, max_cols = self.config.detection.max_table_size
        if rows > max_rows or cols > max_cols:
            errors.append(f"Table too large: {rows}x{cols}, maximum: {max_rows}x{max_cols}")
        
        # Row consistency check
        row_lengths = [len(row) for row in table_object.raw_data]
        if row_lengths:
            max_length = max(row_lengths)
            min_length = min(row_lengths)
            consistency_ratio = min_length / max_length if max_length > 0 else 0
            metrics["row_consistency"] = consistency_ratio
            
            if consistency_ratio < 0.8:
                warnings.append(f"Inconsistent row lengths: {min_length}-{max_length}")
        
        # Empty table check
        if rows == 0 or cols == 0:
            errors.append("Empty table detected")
        
        return {
            "errors": errors,
            "warnings": warnings,
            "metrics": metrics
        }
    
    def _validate_content(self, table_object: TableObject) -> Dict[str, Any]:
        """Validate table content quality."""
        errors = []
        warnings = []
        metrics = {}
        
        # Calculate empty cell ratio
        empty_ratio = TableUtils.calculate_empty_cell_ratio(table_object.raw_data)
        metrics["empty_cell_ratio"] = empty_ratio
        
        # Check empty cell threshold
        if empty_ratio > self.config.validation.max_empty_cell_ratio:
            warnings.append(f"High empty cell ratio: {empty_ratio:.2f}")
        
        # Calculate data cell ratio
        total_cells = sum(len(row) for row in table_object.raw_data)
        data_cells = 0
        
        for row in table_object.raw_data:
            for cell in row:
                if cell and cell.strip() and cell.strip().lower() not in ['', 'n/a', 'null', '-']:
                    data_cells += 1
        
        data_ratio = data_cells / total_cells if total_cells > 0 else 0
        metrics["data_cell_ratio"] = data_ratio
        
        # Check minimum data ratio
        if data_ratio < self.config.validation.min_data_cell_ratio:
            warnings.append(f"Low data cell ratio: {data_ratio:.2f}")
        
        # Content diversity check
        unique_values = set()
        for row in table_object.raw_data:
            for cell in row:
                if cell and cell.strip():
                    unique_values.add(cell.strip().lower())
        
        diversity_ratio = len(unique_values) / total_cells if total_cells > 0 else 0
        metrics["content_diversity"] = diversity_ratio
        
        if diversity_ratio < 0.1:
            warnings.append(f"Low content diversity: {diversity_ratio:.2f}")
        
        return {
            "errors": errors,
            "warnings": warnings,
            "metrics": metrics
        }
    
    def _validate_consistency(self, table_object: TableObject, 
                            classifications: Optional[List[List[RoleClassification]]]) -> Dict[str, Any]:
        """Validate table consistency and role assignments."""
        errors = []
        warnings = []
        metrics = {}
        
        if not classifications:
            return {"errors": errors, "warnings": warnings, "metrics": metrics}
        
        rows, cols = table_object.get_dimensions()
        
        # Check classification completeness
        if len(classifications) != rows:
            errors.append(f"Classification row count mismatch: {len(classifications)} vs {rows}")
            return {"errors": errors, "warnings": warnings, "metrics": metrics}
        
        # Role distribution analysis
        role_counts = {}
        confidence_scores = []
        
        for row_idx, row_classifications in enumerate(classifications):
            if len(row_classifications) != len(table_object.raw_data[row_idx]):
                errors.append(f"Classification column count mismatch in row {row_idx}")
                continue
            
            for classification in row_classifications:
                role = classification.role.value
                role_counts[role] = role_counts.get(role, 0) + 1
                confidence_scores.append(classification.confidence)
        
        metrics["role_distribution"] = role_counts
        metrics["average_confidence"] = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        metrics["min_confidence"] = min(confidence_scores) if confidence_scores else 0
        
        # Check for reasonable role distribution
        total_cells = sum(role_counts.values())
        if total_cells > 0:
            # Should have some headers
            header_ratio = role_counts.get("HEADER", 0) / total_cells
            if header_ratio < 0.05:
                warnings.append("Very few header cells detected")
            elif header_ratio > 0.5:
                warnings.append("Too many header cells detected")
            
            # Should have substantial data content
            data_ratio = role_counts.get("DATA", 0) / total_cells
            if data_ratio < 0.3:
                warnings.append("Low data content ratio")
            
            # Check empty cell ratio
            empty_ratio = role_counts.get("EMPTY", 0) / total_cells
            if empty_ratio > 0.5:
                warnings.append("High empty cell ratio in classifications")
        
        # Confidence validation
        low_confidence_count = sum(1 for score in confidence_scores if score < 0.6)
        if low_confidence_count > len(confidence_scores) * 0.3:
            warnings.append("Many low-confidence classifications")
        
        return {
            "errors": errors,
            "warnings": warnings,
            "metrics": metrics
        }
    
    def _calculate_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score from validation metrics."""
        score = 0.0
        weight_sum = 0.0
        
        # Structure quality (30% weight)
        if "row_consistency" in metrics:
            score += metrics["row_consistency"] * 0.3
            weight_sum += 0.3
        
        # Content quality (40% weight)
        if "data_cell_ratio" in metrics:
            score += metrics["data_cell_ratio"] * 0.2
            weight_sum += 0.2
        
        if "empty_cell_ratio" in metrics:
            # Invert empty ratio (less empty = better)
            score += (1.0 - metrics["empty_cell_ratio"]) * 0.1
            weight_sum += 0.1
        
        if "content_diversity" in metrics:
            # Cap diversity at 1.0 for scoring
            diversity_score = min(1.0, metrics["content_diversity"] * 2)
            score += diversity_score * 0.1
            weight_sum += 0.1
        
        # Classification quality (30% weight)
        if "average_confidence" in metrics:
            score += metrics["average_confidence"] * 0.3
            weight_sum += 0.3
        
        # Normalize by actual weights used
        if weight_sum > 0:
            score = score / weight_sum
        
        return max(0.0, min(1.0, score))
    
    def validate_table_batch(self, tables: List[TableObject]) -> List[ValidationResult]:
        """Validate multiple tables in batch."""
        results = []
        
        for i, table in enumerate(tables):
            try:
                result = self.validate_table(table)
                results.append(result)
            except Exception as e:
                logger.error(f"Validation failed for table {i}: {e}")
                results.append(ValidationResult(
                    is_valid=False,
                    quality_score=0.0,
                    validation_errors=[f"Validation error: {str(e)}"],
                    validation_warnings=[],
                    metrics={}
                ))
        
        return results
    
    def get_validation_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate summary statistics for validation results."""
        if not results:
            return {"error": "No validation results provided"}
        
        valid_count = sum(1 for r in results if r.is_valid)
        total_count = len(results)
        
        quality_scores = [r.quality_score for r in results]
        avg_quality = sum(quality_scores) / len(quality_scores)
        
        total_errors = sum(len(r.validation_errors) for r in results)
        total_warnings = sum(len(r.validation_warnings) for r in results)
        
        return {
            "total_tables": total_count,
            "valid_tables": valid_count,
            "validation_rate": valid_count / total_count,
            "average_quality_score": avg_quality,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "quality_distribution": {
                "high": sum(1 for s in quality_scores if s >= 0.8),
                "medium": sum(1 for s in quality_scores if 0.6 <= s < 0.8),
                "low": sum(1 for s in quality_scores if s < 0.6)
            }
        }
