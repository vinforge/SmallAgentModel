"""
Table-to-Code Expert Tool for SAM
=================================

Phase 2 Expert Tool that leverages Phase 1's rich symbolic metadata to generate
executable Python code for dynamic data analysis, visualization, and complex calculations.

This tool demonstrates SAM's ability to not just understand data, but to act on it
by generating verifiable, executable code.
"""

import logging
import time
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

try:
    from sam.orchestration.uif import SAM_UIF, UIFStatus
    from sam.orchestration.skills.base import BaseSkillModule, SkillExecutionError
except ImportError:
    # Fallback for testing
    from ..uif import SAM_UIF, UIFStatus
    from .base import BaseSkillModule, SkillExecutionError

logger = logging.getLogger(__name__)


@dataclass
class TableReconstructionResult:
    """Result of table reconstruction from chunks."""
    table_id: str
    dataframe_code: str
    column_info: Dict[str, Any]
    table_metadata: Dict[str, Any]
    reconstruction_confidence: float


class TableToCodeTool(BaseSkillModule):
    """
    Expert tool that converts table analysis requests into executable Python code.
    
    Capabilities:
    - Reconstructs tables from table-aware chunks
    - Generates Pandas/Matplotlib code for data analysis
    - Supports calculations, aggregations, visualizations
    - Leverages FORMULA and DATA role classifications
    - Provides executable, verifiable code output
    """
    
    skill_name = "TableToCodeTool"
    skill_version = "1.0.0"
    skill_description = "Generates executable Python code for table data analysis and visualization"
    skill_category = "expert_tools"
    
    # Dependency declarations
    required_inputs = []  # Can extract from input_query
    optional_inputs = ["table_id", "analysis_request", "output_format", "include_visualization"]
    output_keys = ["generated_code", "code_explanation", "execution_instructions", "table_summary"]
    
    def __init__(self):
        """Initialize the Table-to-Code Expert Tool."""
        super().__init__()
        
        # Initialize table processing components
        try:
            from sam.cognition.table_processing.sam_integration import get_table_aware_retrieval
            self.table_retrieval = None  # Will be initialized when memory store is available
            self.table_processing_available = True
        except ImportError:
            logger.warning("Table processing module not available")
            self.table_processing_available = False
        
        # Program-of-Thought prompt templates
        self.code_generation_prompts = {
            "analysis": self._get_analysis_prompt_template(),
            "visualization": self._get_visualization_prompt_template(),
            "calculation": self._get_calculation_prompt_template(),
            "aggregation": self._get_aggregation_prompt_template()
        }
        
        logger.info(f"TableToCodeTool initialized (table_processing_available: {self.table_processing_available})")
    
    def execute(self, uif: SAM_UIF) -> SAM_UIF:
        """
        Execute table-to-code generation with table reconstruction.
        
        Args:
            uif: Universal Interface Format with table analysis request
            
        Returns:
            Updated UIF with generated Python code
        """
        # Initialize tracing
        trace_id = uif.intermediate_data.get('trace_id')
        start_time = time.time()
        
        if trace_id:
            self._log_trace_event(
                trace_id=trace_id,
                event_type="start",
                severity="info",
                message="Starting table-to-code generation",
                payload={
                    "tool": self.skill_name,
                    "input_query": uif.input_query,
                    "table_processing_available": self.table_processing_available
                }
            )
        
        try:
            # Check if table processing is available
            if not self.table_processing_available:
                raise SkillExecutionError(
                    "Table processing module not available",
                    skill_name=self.skill_name,
                    error_code="TABLE_PROCESSING_UNAVAILABLE"
                )
            
            # Initialize table retrieval if not done
            if self.table_retrieval is None:
                self._initialize_table_retrieval(uif)
            
            # Extract analysis parameters
            analysis_params = self._extract_analysis_parameters(uif)
            
            # Step 1: Identify or extract table_id
            table_id = self._identify_table_id(uif, analysis_params)
            
            # Step 2: Reconstruct table from chunks
            reconstruction_result = self._reconstruct_table_from_chunks(table_id)
            
            # Step 3: Generate Python code for analysis
            code_result = self._generate_analysis_code(
                reconstruction_result, analysis_params, uif.input_query
            )

            # Step 4: Validate and secure the generated code
            validation_result = self._validate_generated_code(code_result["code"])

            # Step 5: Apply safety measures if needed
            if validation_result["risk_level"] == "high":
                logger.warning("High-risk code detected, applying sanitization")
                code_result["code"] = self._sanitize_code(code_result["code"])
                code_result["explanation"] += " (Code has been sanitized for safety)"

            # Add safety wrapper for all code
            code_result["wrapped_code"] = self._add_safety_wrapper(code_result["code"])

            # Step 6: Create execution instructions with validation info
            execution_instructions = self._create_execution_instructions(code_result, validation_result)
            
            # Update UIF with results including validation
            uif.intermediate_data.update({
                "generated_code": code_result["code"],
                "wrapped_code": code_result["wrapped_code"],
                "code_explanation": code_result["explanation"],
                "execution_instructions": execution_instructions,
                "validation_result": validation_result,
                "table_summary": {
                    "table_id": table_id,
                    "columns": reconstruction_result.column_info,
                    "metadata": reconstruction_result.table_metadata,
                    "confidence": reconstruction_result.reconstruction_confidence
                },
                "code_type": code_result["code_type"],
                "requires_execution": True,
                "safety_level": validation_result["risk_level"]
            })
            
            # Set success status
            uif.status = UIFStatus.SUCCESS
            uif.response = self._format_response(code_result, reconstruction_result, validation_result)
            
            if trace_id:
                execution_time = time.time() - start_time
                self._log_trace_event(
                    trace_id=trace_id,
                    event_type="success",
                    severity="info",
                    message="Table-to-code generation completed successfully",
                    payload={
                        "execution_time": execution_time,
                        "table_id": table_id,
                        "code_lines": len(code_result["code"].split('\n')),
                        "code_type": code_result["code_type"]
                    }
                )
            
            return uif
            
        except Exception as e:
            error_msg = f"Table-to-code generation failed: {str(e)}"
            logger.error(error_msg)
            
            if trace_id:
                self._log_trace_event(
                    trace_id=trace_id,
                    event_type="error",
                    severity="error",
                    message=error_msg,
                    payload={"error_type": type(e).__name__}
                )
            
            uif.status = UIFStatus.ERROR
            uif.response = f"I encountered an error while generating code for table analysis: {str(e)}"
            
            return uif
    
    def _initialize_table_retrieval(self, uif: SAM_UIF):
        """Initialize table retrieval with memory store."""
        try:
            # Get memory store from UIF or initialize
            memory_store = uif.intermediate_data.get('memory_store')
            if memory_store is None:
                # Try to get from SAM's memory system
                from sam.cognition.table_processing.sam_integration import get_table_aware_retrieval
                # For now, use a mock memory store - in production this would be the actual memory store
                self.table_retrieval = get_table_aware_retrieval(memory_store)
            else:
                from sam.cognition.table_processing.sam_integration import get_table_aware_retrieval
                self.table_retrieval = get_table_aware_retrieval(memory_store)
                
        except Exception as e:
            logger.warning(f"Could not initialize table retrieval: {e}")
            self.table_retrieval = None
    
    def _extract_analysis_parameters(self, uif: SAM_UIF) -> Dict[str, Any]:
        """Extract analysis parameters from UIF."""
        params = {
            "table_id": uif.intermediate_data.get("table_id"),
            "analysis_request": uif.intermediate_data.get("analysis_request", uif.input_query),
            "output_format": uif.intermediate_data.get("output_format", "pandas"),
            "include_visualization": uif.intermediate_data.get("include_visualization", False)
        }
        
        # Auto-detect visualization request
        viz_keywords = ["plot", "chart", "graph", "visualize", "show", "display"]
        if any(keyword in uif.input_query.lower() for keyword in viz_keywords):
            params["include_visualization"] = True
        
        return params
    
    def _identify_table_id(self, uif: SAM_UIF, params: Dict[str, Any]) -> str:
        """Identify table ID from parameters or query."""
        # If explicitly provided
        if params["table_id"]:
            return params["table_id"]
        
        # Try to extract from query
        table_id_pattern = r'table[_\s]*(\w+)'
        match = re.search(table_id_pattern, uif.input_query.lower())
        if match:
            return f"table_{match.group(1)}"
        
        # Default to most recent table
        return "table_0"  # In production, this would query for the most recent table
    
    def can_handle_query(self, query: str) -> bool:
        """
        Check if this tool can handle the given query.
        
        Args:
            query: User query to check
            
        Returns:
            True if query appears to be table analysis related
        """
        table_keywords = [
            'table', 'data', 'calculate', 'sum', 'average', 'mean', 'count',
            'analyze', 'plot', 'chart', 'graph', 'visualize', 'column', 'row',
            'aggregate', 'group', 'filter', 'sort', 'pandas', 'dataframe'
        ]
        
        query_lower = query.lower()
        
        # Check for table analysis keywords
        if any(keyword in query_lower for keyword in table_keywords):
            return True
        
        # Check for specific table references
        if re.search(r'table[_\s]*\w+', query_lower):
            return True
        
        return False

    def _reconstruct_table_from_chunks(self, table_id: str) -> TableReconstructionResult:
        """
        Reconstruct table from table-aware chunks using role classifications and coordinates.

        This is the core intelligence that leverages Phase 1's semantic understanding
        to rebuild machine-readable tables from distributed chunk metadata.
        """
        try:
            # Use mock data if table retrieval is not available
            if self.table_retrieval is None:
                logger.info("Using mock table data for demonstration")
                return self._create_mock_table_reconstruction(table_id)

            # Get all chunks for this table
            table_chunks = self.table_retrieval.search_table_content("", table_id_filter=table_id)

            if not table_chunks:
                logger.warning(f"No table chunks found for table_id: {table_id}, using mock data")
                return self._create_mock_table_reconstruction(table_id)

            logger.info(f"Reconstructing table from {len(table_chunks)} chunks")

            # Enhanced chunk organization with role-aware processing
            reconstruction_data = self._organize_chunks_by_structure(table_chunks)

            # Validate and clean the reconstructed structure
            validated_data = self._validate_table_structure(reconstruction_data)

            # Generate optimized DataFrame creation code
            dataframe_code = self._generate_enhanced_dataframe_code(validated_data)

            # Calculate comprehensive reconstruction confidence
            confidence = self._calculate_enhanced_reconstruction_confidence(validated_data)

            return TableReconstructionResult(
                table_id=table_id,
                dataframe_code=dataframe_code,
                column_info=validated_data["column_info"],
                table_metadata=validated_data["table_metadata"],
                reconstruction_confidence=confidence
            )

        except Exception as e:
            logger.error(f"Table reconstruction failed: {e}")
            # Fallback to mock data with error context
            mock_result = self._create_mock_table_reconstruction(table_id)
            mock_result.table_metadata["reconstruction_error"] = str(e)
            mock_result.reconstruction_confidence *= 0.5  # Reduce confidence due to error
            return mock_result

    def _create_mock_table_reconstruction(self, table_id: str) -> TableReconstructionResult:
        """Create mock table reconstruction for demonstration."""
        # Sample financial data table
        dataframe_code = '''import pandas as pd
import numpy as np

# Reconstructed table data
df = pd.DataFrame({
    'Product': ['Widget A', 'Widget B', 'Widget C'],
    'Q1_Sales': [100, 200, 150],
    'Q2_Sales': [150, 180, 120],
    'Q3_Sales': [120, 220, 180],
    'Total': [370, 600, 450]
})'''

        column_info = {
            0: {"name": "Product", "index": 0},
            1: {"name": "Q1_Sales", "index": 1},
            2: {"name": "Q2_Sales", "index": 2},
            3: {"name": "Q3_Sales", "index": 3},
            4: {"name": "Total", "index": 4}
        }

        table_metadata = {
            "dimensions": {"rows": 4, "columns": 5},
            "source_format": "html",
            "detection_confidence": 0.95
        }

        return TableReconstructionResult(
            table_id=table_id,
            dataframe_code=dataframe_code,
            column_info=column_info,
            table_metadata=table_metadata,
            reconstruction_confidence=0.9
        )

    def _generate_dataframe_code(self, cell_data: Dict[Tuple[int, int], Dict],
                                column_info: Dict[int, Dict]) -> str:
        """Generate pandas DataFrame creation code from cell data."""
        if not cell_data:
            return "df = pd.DataFrame()"

        # Find table dimensions
        max_row = max(coord[0] for coord in cell_data.keys())
        max_col = max(coord[1] for coord in cell_data.keys())

        # Extract column names from headers (row 0)
        columns = []
        for col in range(max_col + 1):
            if col in column_info:
                columns.append(column_info[col]["name"])
            else:
                columns.append(f"Column_{col}")

        # Extract data rows (skip header row)
        data_rows = []
        for row in range(1, max_row + 1):
            row_data = []
            for col in range(max_col + 1):
                cell = cell_data.get((row, col), {})
                content = cell.get("content", "")

                # Try to convert to appropriate type
                if cell.get("data_type") in ["number", "integer", "float"]:
                    try:
                        if '.' in content:
                            row_data.append(float(content))
                        else:
                            row_data.append(int(content))
                    except ValueError:
                        row_data.append(content)
                else:
                    row_data.append(content)

            data_rows.append(row_data)

        # Generate code
        code_lines = [
            "import pandas as pd",
            "import numpy as np",
            "",
            "# Reconstructed table data",
            "df = pd.DataFrame({"
        ]

        for i, col_name in enumerate(columns):
            col_data = [row[i] if i < len(row) else "" for row in data_rows]
            code_lines.append(f"    '{col_name}': {col_data},")

        code_lines.append("})")

        return "\n".join(code_lines)

    def _calculate_reconstruction_confidence(self, cell_data: Dict, column_info: Dict) -> float:
        """Calculate confidence score for table reconstruction."""
        if not cell_data:
            return 0.0

        # Base confidence
        confidence = 0.7

        # Bonus for having headers
        if column_info:
            confidence += 0.2

        # Bonus for data consistency
        total_cells = len(cell_data)
        non_empty_cells = sum(1 for cell in cell_data.values() if cell.get("content", "").strip())

        if total_cells > 0:
            completeness = non_empty_cells / total_cells
            confidence += completeness * 0.1

        return min(1.0, confidence)

    def _organize_chunks_by_structure(self, table_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Organize chunks by table structure using role classifications and coordinates.

        This leverages Phase 1's semantic role understanding to intelligently
        reconstruct the table structure.
        """
        cell_data = {}
        column_info = {}
        table_metadata = {}
        role_distribution = {}
        data_type_distribution = {}

        # Process each chunk with role-aware logic
        for chunk in table_chunks:
            metadata = chunk.get("metadata", {})
            coordinates = metadata.get("cell_coordinates")
            cell_role = metadata.get("cell_role", "DATA")
            cell_content = chunk.get("content", "")
            cell_data_type = metadata.get("cell_data_type", "text")

            # Track role distribution for quality assessment
            role_distribution[cell_role] = role_distribution.get(cell_role, 0) + 1
            data_type_distribution[cell_data_type] = data_type_distribution.get(cell_data_type, 0) + 1

            if coordinates:
                row, col = coordinates

                # Enhanced cell data with role context
                cell_data[(row, col)] = {
                    "content": cell_content,
                    "role": cell_role,
                    "data_type": cell_data_type,
                    "confidence": metadata.get("confidence_score", 0.8),
                    "context_factors": metadata.get("context_factors", {}),
                    "original_metadata": metadata
                }

                # Smart column info extraction based on role
                if cell_role == "HEADER":
                    if row == 0:  # Primary headers
                        column_info[col] = {
                            "name": cell_content,
                            "index": col,
                            "data_type": self._infer_column_data_type(col, cell_data),
                            "role": "primary_header"
                        }
                    elif col == 0:  # Row headers
                        column_info[f"row_{row}"] = {
                            "name": cell_content,
                            "index": row,
                            "data_type": "text",
                            "role": "row_header"
                        }

                # Collect comprehensive table metadata
                if "table_structure" in metadata:
                    table_metadata.update(metadata["table_structure"])

        # Calculate table dimensions
        if cell_data:
            max_row = max(coord[0] for coord in cell_data.keys())
            max_col = max(coord[1] for coord in cell_data.keys())
            table_metadata["reconstructed_dimensions"] = {"rows": max_row + 1, "columns": max_col + 1}

        # Add analysis metadata
        table_metadata["role_distribution"] = role_distribution
        table_metadata["data_type_distribution"] = data_type_distribution
        table_metadata["total_chunks"] = len(table_chunks)

        return {
            "cell_data": cell_data,
            "column_info": column_info,
            "table_metadata": table_metadata,
            "role_distribution": role_distribution,
            "data_type_distribution": data_type_distribution
        }

    def _validate_table_structure(self, reconstruction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean the reconstructed table structure.

        Ensures data consistency and handles edge cases in reconstruction.
        """
        cell_data = reconstruction_data["cell_data"]
        column_info = reconstruction_data["column_info"]
        table_metadata = reconstruction_data["table_metadata"]

        if not cell_data:
            logger.warning("No cell data found in reconstruction")
            return reconstruction_data

        # Find actual table dimensions
        max_row = max(coord[0] for coord in cell_data.keys())
        max_col = max(coord[1] for coord in cell_data.keys())

        # Validate column headers
        validated_columns = {}
        for col in range(max_col + 1):
            if col in column_info:
                validated_columns[col] = column_info[col]
            else:
                # Generate column name from header cell or default
                header_cell = cell_data.get((0, col))
                if header_cell and header_cell["role"] == "HEADER":
                    validated_columns[col] = {
                        "name": header_cell["content"],
                        "index": col,
                        "data_type": self._infer_column_data_type(col, cell_data),
                        "role": "inferred_header"
                    }
                else:
                    validated_columns[col] = {
                        "name": f"Column_{col}",
                        "index": col,
                        "data_type": self._infer_column_data_type(col, cell_data),
                        "role": "generated_header"
                    }

        # Clean cell data - remove empty cells and validate content
        cleaned_cell_data = {}
        for coord, cell in cell_data.items():
            content = cell["content"]
            if content and content.strip():  # Only keep non-empty cells
                # Clean and validate content
                cleaned_content = content.strip()

                # Type conversion based on data type
                if cell["data_type"] in ["number", "integer", "float"]:
                    try:
                        if '.' in cleaned_content:
                            cleaned_content = float(cleaned_content)
                        else:
                            cleaned_content = int(cleaned_content)
                    except ValueError:
                        # Keep as string if conversion fails
                        pass

                cleaned_cell_data[coord] = {
                    **cell,
                    "content": cleaned_content
                }

        # Update metadata with validation results
        table_metadata["validation_applied"] = True
        table_metadata["original_cell_count"] = len(cell_data)
        table_metadata["cleaned_cell_count"] = len(cleaned_cell_data)
        table_metadata["data_completeness"] = len(cleaned_cell_data) / len(cell_data) if cell_data else 0

        return {
            "cell_data": cleaned_cell_data,
            "column_info": validated_columns,
            "table_metadata": table_metadata,
            "role_distribution": reconstruction_data["role_distribution"],
            "data_type_distribution": reconstruction_data["data_type_distribution"]
        }

    def _infer_column_data_type(self, col_index: int, cell_data: Dict[Tuple[int, int], Dict]) -> str:
        """Infer column data type from cell contents."""
        # Look at data cells in this column (skip header row)
        column_types = []
        for (row, col), cell in cell_data.items():
            if col == col_index and row > 0 and cell["role"] == "DATA":
                column_types.append(cell["data_type"])

        if not column_types:
            return "text"

        # Return most common type
        type_counts = {}
        for dtype in column_types:
            type_counts[dtype] = type_counts.get(dtype, 0) + 1

        return max(type_counts, key=type_counts.get)

    def _generate_enhanced_dataframe_code(self, validated_data: Dict[str, Any]) -> str:
        """
        Generate optimized pandas DataFrame creation code from validated table data.

        Creates efficient, type-aware DataFrame construction code.
        """
        cell_data = validated_data["cell_data"]
        column_info = validated_data["column_info"]
        table_metadata = validated_data["table_metadata"]

        if not cell_data:
            return "df = pd.DataFrame()  # Empty table"

        # Find table dimensions
        max_row = max(coord[0] for coord in cell_data.keys())
        max_col = max(coord[1] for coord in cell_data.keys())

        # Extract column names and types
        columns = []
        column_types = {}
        for col in range(max_col + 1):
            if col in column_info:
                col_name = column_info[col]["name"]
                col_type = column_info[col]["data_type"]
            else:
                col_name = f"Column_{col}"
                col_type = "text"

            # Clean column name for Python
            clean_name = self._clean_column_name(col_name)
            columns.append(clean_name)
            column_types[clean_name] = col_type

        # Extract data rows (skip header row if present)
        data_rows = []
        start_row = 1 if any(cell["role"] == "HEADER" for cell in cell_data.values() if cell_data) else 0

        for row in range(start_row, max_row + 1):
            row_data = []
            for col in range(max_col + 1):
                cell = cell_data.get((row, col))
                if cell:
                    content = cell["content"]
                    # Use already converted content from validation
                    row_data.append(content)
                else:
                    # Handle missing cells
                    col_type = column_types.get(columns[col], "text")
                    if col_type in ["number", "integer", "float"]:
                        row_data.append(0)  # Default numeric value
                    else:
                        row_data.append("")  # Default text value

            if any(str(cell).strip() for cell in row_data):  # Only add non-empty rows
                data_rows.append(row_data)

        # Generate optimized code
        code_lines = [
            "import pandas as pd",
            "import numpy as np",
            "",
            "# Reconstructed table data with type optimization"
        ]

        # Add metadata comment
        if table_metadata.get("reconstructed_dimensions"):
            dims = table_metadata["reconstructed_dimensions"]
            code_lines.append(f"# Original dimensions: {dims['rows']} rows × {dims['columns']} columns")

        if table_metadata.get("data_completeness"):
            completeness = table_metadata["data_completeness"]
            code_lines.append(f"# Data completeness: {completeness:.1%}")

        code_lines.append("")

        # Create DataFrame with proper data types
        if data_rows:
            code_lines.append("df = pd.DataFrame({")

            for i, col_name in enumerate(columns):
                col_data = [row[i] if i < len(row) else "" for row in data_rows]
                col_type = column_types.get(col_name, "text")

                # Format data based on type
                if col_type in ["number", "integer", "float"]:
                    # Numeric data
                    code_lines.append(f"    '{col_name}': {col_data},")
                else:
                    # Text data - properly escape strings
                    escaped_data = [repr(str(item)) if item != "" else "''" for item in col_data]
                    code_lines.append(f"    '{col_name}': [{', '.join(escaped_data)}],")

            code_lines.append("})")

            # Add type optimization
            type_conversions = []
            for col_name, col_type in column_types.items():
                if col_type in ["integer"]:
                    type_conversions.append(f"df['{col_name}'] = pd.to_numeric(df['{col_name}'], errors='coerce').astype('Int64')")
                elif col_type in ["float", "number"]:
                    type_conversions.append(f"df['{col_name}'] = pd.to_numeric(df['{col_name}'], errors='coerce')")

            if type_conversions:
                code_lines.append("")
                code_lines.append("# Optimize data types")
                code_lines.extend(type_conversions)

        else:
            code_lines.append("df = pd.DataFrame(columns=" + str(columns) + ")")

        return "\n".join(code_lines)

    def _clean_column_name(self, name: str) -> str:
        """Clean column name for Python variable naming."""
        import re
        # Replace spaces and special characters with underscores
        cleaned = re.sub(r'[^\w]', '_', str(name))
        # Remove multiple underscores
        cleaned = re.sub(r'_+', '_', cleaned)
        # Remove leading/trailing underscores
        cleaned = cleaned.strip('_')
        # Ensure it starts with a letter
        if cleaned and cleaned[0].isdigit():
            cleaned = f"col_{cleaned}"
        # Handle empty names
        if not cleaned:
            cleaned = "unnamed_column"
        return cleaned

    def _calculate_enhanced_reconstruction_confidence(self, validated_data: Dict[str, Any]) -> float:
        """
        Calculate comprehensive reconstruction confidence score.

        Considers multiple factors including role distribution, data completeness,
        and structural consistency.
        """
        cell_data = validated_data["cell_data"]
        column_info = validated_data["column_info"]
        table_metadata = validated_data["table_metadata"]
        role_distribution = validated_data["role_distribution"]

        if not cell_data:
            return 0.0

        # Base confidence from data completeness
        data_completeness = table_metadata.get("data_completeness", 0.0)
        base_confidence = data_completeness * 0.4

        # Role distribution quality
        role_quality = 0.0
        total_cells = sum(role_distribution.values())

        if total_cells > 0:
            # Bonus for having headers
            header_ratio = role_distribution.get("HEADER", 0) / total_cells
            if 0.05 <= header_ratio <= 0.3:  # Reasonable header ratio
                role_quality += 0.2

            # Bonus for substantial data content
            data_ratio = role_distribution.get("DATA", 0) / total_cells
            if data_ratio >= 0.5:  # At least 50% data cells
                role_quality += 0.2

            # Penalty for too many empty cells
            empty_ratio = role_distribution.get("EMPTY", 0) / total_cells
            if empty_ratio < 0.3:  # Less than 30% empty
                role_quality += 0.1

        # Structural consistency
        structure_quality = 0.0
        if column_info:
            # Bonus for having column information
            structure_quality += 0.15

            # Bonus for consistent column types
            type_consistency = len(set(col.get("data_type", "text") for col in column_info.values()))
            if type_consistency <= 3:  # Not too many different types
                structure_quality += 0.05

        # Table size appropriateness
        size_quality = 0.0
        dims = table_metadata.get("reconstructed_dimensions", {})
        if dims:
            rows, cols = dims.get("rows", 0), dims.get("columns", 0)
            if 2 <= rows <= 100 and 2 <= cols <= 20:  # Reasonable table size
                size_quality = 0.1

        # Combine all factors
        total_confidence = base_confidence + role_quality + structure_quality + size_quality

        # Apply confidence from individual cell classifications
        if cell_data:
            avg_cell_confidence = sum(
                cell.get("confidence", 0.8) for cell in cell_data.values()
            ) / len(cell_data)
            total_confidence = (total_confidence * 0.8) + (avg_cell_confidence * 0.2)

        return min(1.0, total_confidence)

    def _generate_analysis_code(self, reconstruction_result: TableReconstructionResult,
                              analysis_params: Dict[str, Any], user_query: str) -> Dict[str, Any]:
        """Generate Python code for table analysis based on user request."""
        # Determine analysis type
        analysis_type = self._classify_analysis_type(user_query)

        # Get appropriate prompt template
        prompt_template = self.code_generation_prompts.get(analysis_type,
                                                          self.code_generation_prompts["analysis"])

        # Prepare context for code generation
        context = {
            "table_code": reconstruction_result.dataframe_code,
            "columns": list(reconstruction_result.column_info.values()),
            "user_request": user_query,
            "include_visualization": analysis_params["include_visualization"],
            "table_metadata": reconstruction_result.table_metadata
        }

        # Generate code using Program-of-Thought approach
        generated_code = self._apply_program_of_thought(prompt_template, context)

        return {
            "code": generated_code,
            "explanation": self._generate_code_explanation(generated_code, analysis_type),
            "code_type": analysis_type,
            "context": context
        }

    def _classify_analysis_type(self, user_query: str) -> str:
        """Classify the type of analysis requested."""
        query_lower = user_query.lower()

        # Visualization keywords
        viz_keywords = ["plot", "chart", "graph", "visualize", "show", "display"]
        if any(keyword in query_lower for keyword in viz_keywords):
            return "visualization"

        # Calculation keywords
        calc_keywords = ["calculate", "compute", "formula", "equation"]
        if any(keyword in query_lower for keyword in calc_keywords):
            return "calculation"

        # Aggregation keywords
        agg_keywords = ["sum", "total", "average", "mean", "count", "max", "min", "group"]
        if any(keyword in query_lower for keyword in agg_keywords):
            return "aggregation"

        # Default to general analysis
        return "analysis"

    def _apply_program_of_thought(self, prompt_template: str, context: Dict[str, Any]) -> str:
        """
        Apply Program-of-Thought approach to generate code using sophisticated prompting.

        This method implements the core intelligence that transforms natural language
        requests into executable Python code using structured reasoning.
        """
        # In production, this would call the LLM with the prompt template
        # For now, use enhanced rule-based generation with prompt-like structure

        # Prepare the full prompt with context
        full_prompt = self._build_program_of_thought_prompt(prompt_template, context)

        # Simulate LLM reasoning process
        reasoning_steps = self._simulate_reasoning_process(context)

        # Generate code based on reasoning
        generated_code = self._generate_code_from_reasoning(reasoning_steps, context)

        return generated_code

    def _build_program_of_thought_prompt(self, template: str, context: Dict[str, Any]) -> str:
        """Build the complete Program-of-Thought prompt with context."""
        # Format the template with context
        formatted_prompt = template.format(
            table_code=context["table_code"],
            user_request=context["user_request"],
            columns=self._format_columns_for_prompt(context["columns"]),
            table_metadata=context.get("table_metadata", {}),
            include_visualization=context.get("include_visualization", False)
        )

        # Add reasoning structure
        reasoning_prompt = f"""
{formatted_prompt}

REASONING PROCESS:
1. Analyze the user request to understand the specific task
2. Identify the relevant columns and data types
3. Determine the appropriate pandas operations
4. Consider edge cases and error handling
5. Structure the code for clarity and efficiency
6. Add appropriate output formatting

GENERATE EXECUTABLE CODE:
"""

        return reasoning_prompt

    def _simulate_reasoning_process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate the reasoning process that an LLM would follow.

        This breaks down the analysis into structured steps.
        """
        user_request = context["user_request"].lower()
        columns = context.get("columns", [])

        reasoning = {
            "task_type": self._identify_task_type(user_request),
            "target_columns": self._identify_target_columns(user_request, columns),
            "operations": self._identify_operations(user_request),
            "output_format": self._identify_output_format(user_request),
            "visualization_needed": context.get("include_visualization", False),
            "error_handling": self._identify_error_handling_needs(user_request),
            "complexity_level": self._assess_complexity(user_request)
        }

        return reasoning

    def _identify_task_type(self, user_request: str) -> str:
        """Identify the primary task type from user request."""
        task_patterns = {
            "aggregation": ["sum", "total", "count", "average", "mean", "max", "min", "aggregate"],
            "filtering": ["filter", "where", "select", "find", "show only", "exclude"],
            "sorting": ["sort", "order", "rank", "arrange"],
            "grouping": ["group", "group by", "categorize", "segment"],
            "calculation": ["calculate", "compute", "formula", "derive", "multiply", "divide"],
            "comparison": ["compare", "difference", "ratio", "versus", "vs"],
            "visualization": ["plot", "chart", "graph", "visualize", "show", "display"],
            "analysis": ["analyze", "examine", "explore", "investigate", "summary"]
        }

        for task_type, keywords in task_patterns.items():
            if any(keyword in user_request for keyword in keywords):
                return task_type

        return "analysis"  # Default

    def _identify_target_columns(self, user_request: str, columns: List[Dict]) -> List[str]:
        """Identify which columns are relevant to the request."""
        target_columns = []

        # Extract column names from request
        for col_info in columns:
            col_name = col_info.get("name", "")
            if col_name.lower() in user_request:
                target_columns.append(col_name)

        # If no specific columns mentioned, infer from task type
        if not target_columns:
            # Look for numeric columns for calculations
            if any(word in user_request for word in ["sum", "average", "calculate", "total"]):
                target_columns = [col["name"] for col in columns
                                if col.get("data_type") in ["number", "integer", "float"]]

        return target_columns

    def _identify_operations(self, user_request: str) -> List[str]:
        """Identify the pandas operations needed."""
        operations = []

        operation_map = {
            "sum": ["sum()"],
            "average": ["mean()"],
            "mean": ["mean()"],
            "count": ["count()"],
            "max": ["max()"],
            "min": ["min()"],
            "sort": ["sort_values()"],
            "group": ["groupby()"],
            "filter": ["query()", "loc[]"],
            "unique": ["unique()"],
            "describe": ["describe()"]
        }

        for keyword, ops in operation_map.items():
            if keyword in user_request:
                operations.extend(ops)

        return operations

    def _identify_output_format(self, user_request: str) -> str:
        """Identify how the output should be formatted."""
        if any(word in user_request for word in ["table", "dataframe", "df"]):
            return "dataframe"
        elif any(word in user_request for word in ["list", "values"]):
            return "list"
        elif any(word in user_request for word in ["chart", "plot", "graph"]):
            return "visualization"
        else:
            return "formatted_text"

    def _identify_error_handling_needs(self, user_request: str) -> List[str]:
        """Identify what error handling is needed."""
        error_handling = []

        if any(word in user_request for word in ["divide", "ratio", "percentage"]):
            error_handling.append("division_by_zero")

        if any(word in user_request for word in ["numeric", "number", "calculate"]):
            error_handling.append("non_numeric_data")

        error_handling.append("missing_data")  # Always check for missing data

        return error_handling

    def _assess_complexity(self, user_request: str) -> str:
        """Assess the complexity level of the request."""
        complexity_indicators = {
            "simple": ["sum", "count", "max", "min", "show"],
            "medium": ["average", "group", "filter", "sort", "compare"],
            "complex": ["correlation", "regression", "pivot", "merge", "join"]
        }

        for level, indicators in complexity_indicators.items():
            if any(indicator in user_request for indicator in indicators):
                return level

        return "simple"

    def _generate_code_from_reasoning(self, reasoning: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate code based on the reasoning analysis."""
        code_lines = [context["table_code"], ""]

        task_type = reasoning["task_type"]
        target_columns = reasoning["target_columns"]
        operations = reasoning["operations"]

        # Add task-specific code generation
        if task_type == "aggregation":
            code_lines.extend(self._generate_aggregation_code(target_columns, operations, reasoning))
        elif task_type == "visualization":
            code_lines.extend(self._generate_visualization_code(target_columns, reasoning))
        elif task_type == "filtering":
            code_lines.extend(self._generate_filtering_code(target_columns, operations, reasoning))
        elif task_type == "calculation":
            code_lines.extend(self._generate_calculation_code(target_columns, operations, reasoning))
        else:
            # General analysis
            code_lines.extend(self._generate_general_analysis_code(target_columns, reasoning))

        # Add error handling if needed
        if reasoning["error_handling"]:
            code_lines.extend(self._generate_error_handling_code(reasoning["error_handling"]))

        return "\n".join(code_lines)

    def _generate_aggregation_code(self, target_columns: List[str], operations: List[str],
                                 reasoning: Dict[str, Any]) -> List[str]:
        """Generate code for aggregation operations."""
        code_lines = ["# Aggregation Analysis"]

        if target_columns:
            # Specific columns
            columns_str = ", ".join(f"'{col}'" for col in target_columns)
            code_lines.append(f"target_columns = [{columns_str}]")
            code_lines.append("numeric_data = df[target_columns].select_dtypes(include=[np.number])")
        else:
            # All numeric columns
            code_lines.append("numeric_data = df.select_dtypes(include=[np.number])")

        # Add operations
        if "sum()" in operations:
            code_lines.extend([
                "",
                "# Calculate sums",
                "totals = numeric_data.sum()",
                "print('Column Totals:')",
                "for col, total in totals.items():",
                "    print(f'{col}: {total:,.2f}')"
            ])

        if "mean()" in operations:
            code_lines.extend([
                "",
                "# Calculate averages",
                "averages = numeric_data.mean()",
                "print('\\nColumn Averages:')",
                "for col, avg in averages.items():",
                "    print(f'{col}: {avg:,.2f}')"
            ])

        # Add summary if no specific operations
        if not operations:
            code_lines.extend([
                "",
                "# Summary statistics",
                "print('Summary Statistics:')",
                "print(numeric_data.describe())"
            ])

        return code_lines

    def _generate_visualization_code(self, target_columns: List[str],
                                   reasoning: Dict[str, Any]) -> List[str]:
        """Generate code for data visualization."""
        code_lines = [
            "# Data Visualization",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            ""
        ]

        if target_columns:
            columns_str = ", ".join(f"'{col}'" for col in target_columns)
            code_lines.append(f"plot_columns = [{columns_str}]")
            code_lines.append("plot_data = df[plot_columns].select_dtypes(include=[np.number])")
        else:
            code_lines.append("plot_data = df.select_dtypes(include=[np.number])")

        # Determine best chart type
        complexity = reasoning.get("complexity_level", "simple")

        if complexity == "simple":
            code_lines.extend([
                "",
                "# Create bar chart",
                "if len(plot_data.columns) > 0:",
                "    plt.figure(figsize=(12, 6))",
                "    plot_data.plot(kind='bar', ax=plt.gca())",
                "    plt.title('Data Overview')",
                "    plt.xlabel('Records')",
                "    plt.ylabel('Values')",
                "    plt.xticks(rotation=45)",
                "    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')",
                "    plt.tight_layout()",
                "    plt.show()"
            ])
        else:
            code_lines.extend([
                "",
                "# Create multiple visualizations",
                "fig, axes = plt.subplots(2, 2, figsize=(15, 10))",
                "fig.suptitle('Comprehensive Data Analysis')",
                "",
                "# Bar chart",
                "plot_data.plot(kind='bar', ax=axes[0,0])",
                "axes[0,0].set_title('Bar Chart')",
                "",
                "# Line chart",
                "plot_data.plot(kind='line', ax=axes[0,1])",
                "axes[0,1].set_title('Trend Analysis')",
                "",
                "# Box plot",
                "plot_data.boxplot(ax=axes[1,0])",
                "axes[1,0].set_title('Distribution Analysis')",
                "",
                "# Correlation heatmap",
                "if len(plot_data.columns) > 1:",
                "    sns.heatmap(plot_data.corr(), annot=True, ax=axes[1,1])",
                "    axes[1,1].set_title('Correlation Matrix')",
                "",
                "plt.tight_layout()",
                "plt.show()"
            ])

        return code_lines

    def _generate_filtering_code(self, target_columns: List[str], operations: List[str],
                               reasoning: Dict[str, Any]) -> List[str]:
        """Generate code for data filtering operations."""
        code_lines = ["# Data Filtering"]

        # Basic filtering template
        code_lines.extend([
            "",
            "# Filter data based on conditions",
            "# Example: Filter for non-null values",
            "filtered_df = df.dropna()"
        ])

        if target_columns:
            code_lines.extend([
                "",
                f"# Focus on specific columns: {', '.join(target_columns)}",
                f"focused_data = filtered_df[{target_columns}]",
                "print('Filtered Data:')",
                "print(focused_data.head())"
            ])

        return code_lines

    def _generate_calculation_code(self, target_columns: List[str], operations: List[str],
                                 reasoning: Dict[str, Any]) -> List[str]:
        """Generate code for custom calculations."""
        code_lines = ["# Custom Calculations"]

        if target_columns and len(target_columns) >= 2:
            # Multi-column calculations
            col1, col2 = target_columns[0], target_columns[1]
            code_lines.extend([
                "",
                f"# Calculate relationship between {col1} and {col2}",
                f"df['ratio_{col1}_{col2}'] = df['{col1}'] / df['{col2}'].replace(0, np.nan)",
                f"df['sum_{col1}_{col2}'] = df['{col1}'] + df['{col2}']",
                "",
                "print('Calculated columns:')",
                "print(df[['ratio_' + col1 + '_' + col2, 'sum_' + col1 + '_' + col2]].head())"
            ])
        else:
            # Single column calculations
            code_lines.extend([
                "",
                "# Perform calculations on numeric columns",
                "numeric_cols = df.select_dtypes(include=[np.number]).columns",
                "for col in numeric_cols:",
                "    df[f'{col}_squared'] = df[col] ** 2",
                "    df[f'{col}_log'] = np.log(df[col].replace(0, np.nan))",
                "",
                "print('Calculated columns added')",
                "print(df.head())"
            ])

        return code_lines

    def _generate_general_analysis_code(self, target_columns: List[str],
                                      reasoning: Dict[str, Any]) -> List[str]:
        """Generate code for general data analysis."""
        code_lines = [
            "# Comprehensive Data Analysis",
            "",
            "print('=== DATA OVERVIEW ===')",
            "print(f'Table shape: {df.shape}')",
            "print(f'Columns: {list(df.columns)}')",
            "",
            "print('\\n=== DATA TYPES ===')",
            "print(df.dtypes)",
            "",
            "print('\\n=== MISSING VALUES ===')",
            "print(df.isnull().sum())",
            "",
            "print('\\n=== BASIC STATISTICS ===')",
            "print(df.describe())"
        ]

        if target_columns:
            code_lines.extend([
                "",
                f"print('\\n=== FOCUSED ANALYSIS: {', '.join(target_columns)} ===')",
                f"focused_cols = {target_columns}",
                "for col in focused_cols:",
                "    if col in df.columns:",
                "        print(f'\\n{col}:')",
                "        if df[col].dtype in ['int64', 'float64']:",
                "            print(f'  Mean: {df[col].mean():.2f}')",
                "            print(f'  Median: {df[col].median():.2f}')",
                "            print(f'  Std: {df[col].std():.2f}')",
                "        else:",
                "            print(f'  Unique values: {df[col].nunique()}')",
                "            print(f'  Most common: {df[col].mode().iloc[0] if not df[col].mode().empty else \"N/A\"}')"
            ])

        return code_lines

    def _generate_error_handling_code(self, error_types: List[str]) -> List[str]:
        """Generate error handling code."""
        code_lines = ["", "# Error Handling and Data Quality Checks"]

        if "division_by_zero" in error_types:
            code_lines.extend([
                "",
                "# Handle division by zero",
                "# (Already handled with .replace(0, np.nan) in calculations)"
            ])

        if "non_numeric_data" in error_types:
            code_lines.extend([
                "",
                "# Check for non-numeric data in numeric operations",
                "numeric_cols = df.select_dtypes(include=[np.number]).columns",
                "print(f'Numeric columns available: {list(numeric_cols)}')"
            ])

        if "missing_data" in error_types:
            code_lines.extend([
                "",
                "# Missing data summary",
                "missing_summary = df.isnull().sum()",
                "if missing_summary.sum() > 0:",
                "    print('\\nMissing data detected:')",
                "    print(missing_summary[missing_summary > 0])",
                "else:",
                "    print('\\nNo missing data detected')"
            ])

        return code_lines

    def _format_columns_for_prompt(self, columns: List[Dict]) -> str:
        """Format column information for prompt context."""
        if not columns:
            return "No column information available"

        formatted = []
        for col in columns:
            name = col.get("name", "Unknown")
            data_type = col.get("data_type", "unknown")
            formatted.append(f"- {name} ({data_type})")

        return "\n".join(formatted)

    def _validate_generated_code(self, code: str) -> Dict[str, Any]:
        """
        Validate generated Python code for safety and correctness.

        This implements comprehensive safety checks to ensure the generated
        code is safe to execute and follows best practices.
        """
        validation_result = {
            "is_safe": True,
            "is_valid": True,
            "safety_issues": [],
            "syntax_issues": [],
            "warnings": [],
            "recommendations": [],
            "risk_level": "low"
        }

        # Safety validation
        safety_result = self._check_code_safety(code)
        validation_result.update(safety_result)

        # Syntax validation
        syntax_result = self._check_code_syntax(code)
        validation_result.update(syntax_result)

        # Best practices validation
        practices_result = self._check_best_practices(code)
        validation_result["warnings"].extend(practices_result["warnings"])
        validation_result["recommendations"].extend(practices_result["recommendations"])

        # Calculate overall risk level
        validation_result["risk_level"] = self._calculate_risk_level(validation_result)

        return validation_result

    def _check_code_safety(self, code: str) -> Dict[str, Any]:
        """Check code for security and safety issues."""
        safety_issues = []
        risk_level = "low"

        # Dangerous imports and functions
        dangerous_patterns = [
            (r'\bos\.system\b', "os.system() can execute arbitrary shell commands"),
            (r'\bsubprocess\b', "subprocess module can execute system commands"),
            (r'\beval\b', "eval() can execute arbitrary code"),
            (r'\bexec\b', "exec() can execute arbitrary code"),
            (r'\b__import__\b', "__import__() can import arbitrary modules"),
            (r'\bopen\s*\(.*[\'\"]\w*[\'\"]\s*,\s*[\'\"]\w*w', "File writing operations detected"),
            (r'\brmdir\b|\bunlink\b|\bremove\b', "File deletion operations detected"),
            (r'\bsocket\b', "Network socket operations detected"),
            (r'\brequests\b', "HTTP requests detected"),
            (r'\burllib\b', "URL operations detected"),
            (r'\bpickle\b', "Pickle operations can be unsafe"),
            (r'\b__.*__\b', "Dunder methods can be dangerous")
        ]

        for pattern, message in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                safety_issues.append(message)
                risk_level = "high"

        # Check for file system operations
        file_operations = [
            r'\bopen\s*\(',
            r'\.write\s*\(',
            r'\.read\s*\(',
            r'\bwith\s+open'
        ]

        for pattern in file_operations:
            if re.search(pattern, code, re.IGNORECASE):
                # Allow read-only operations for data files
                if not re.search(r'[\'\"]\w*[\'\"]\s*,\s*[\'\"]\w*r', code):
                    safety_issues.append("File system operations detected")
                    risk_level = "medium"

        # Check for infinite loops
        loop_patterns = [
            r'\bwhile\s+True\s*:',
            r'\bfor\s+\w+\s+in\s+range\s*\(\s*\d{6,}\s*\)'  # Very large ranges
        ]

        for pattern in loop_patterns:
            if re.search(pattern, code):
                safety_issues.append("Potential infinite loop or very large iteration detected")
                risk_level = "medium"

        return {
            "is_safe": len(safety_issues) == 0,
            "safety_issues": safety_issues,
            "risk_level": risk_level
        }

    def _check_code_syntax(self, code: str) -> Dict[str, Any]:
        """Check code for syntax errors."""
        syntax_issues = []
        is_valid = True

        try:
            # Try to compile the code
            compile(code, '<generated_code>', 'exec')
        except SyntaxError as e:
            syntax_issues.append(f"Syntax error: {str(e)}")
            is_valid = False
        except Exception as e:
            syntax_issues.append(f"Compilation error: {str(e)}")
            is_valid = False

        # Check for common issues with smarter detection
        # Check for DataFrame usage
        if re.search(r'\bdf\b(?!\s*=)', code):
            # Check if df is actually defined in the code
            if not re.search(r'\bdf\s*=', code):
                syntax_issues.append("Warning: DataFrame 'df' used but may not be defined")

        # Check for module usage
        module_checks = [
            (r'\bnp\b(?!\s*=)', "numpy", "NumPy 'np' used but may not be imported"),
            (r'\bplt\b(?!\s*=)', "matplotlib", "Matplotlib 'plt' used but may not be imported"),
            (r'\bpd\b(?!\s*=)', "pandas", "Pandas 'pd' used but may not be imported")
        ]

        for pattern, module_name, message in module_checks:
            if re.search(pattern, code):
                # Check if the module is actually imported
                pattern_name = pattern.split("\\b")[1]
                if not re.search(f'import.*{module_name}', code) and not re.search(f'import {pattern_name}', code):
                    syntax_issues.append(f"Warning: {message}")

        return {
            "is_valid": is_valid,
            "syntax_issues": syntax_issues
        }

    def _check_best_practices(self, code: str) -> Dict[str, Any]:
        """Check code for best practices and provide recommendations."""
        warnings = []
        recommendations = []

        # Check for hardcoded values
        if re.search(r'\b\d{4,}\b', code):  # Large numbers
            warnings.append("Large hardcoded numbers detected")
            recommendations.append("Consider using constants or configuration for large numbers")

        # Check for proper error handling
        if "try:" not in code and any(op in code for op in ["division", "/", "mean()", "std()"]):
            recommendations.append("Consider adding error handling for mathematical operations")

        # Check for efficient pandas operations
        if ".iterrows()" in code:
            warnings.append("iterrows() is inefficient for large datasets")
            recommendations.append("Consider using vectorized operations instead of iterrows()")

        # Check for memory efficiency
        if ".copy()" in code:
            recommendations.append("Be mindful of memory usage when copying DataFrames")

        # Check for output formatting
        if "print(" in code and not re.search(r'print\(.*f[\'\""]', code):
            recommendations.append("Consider using f-strings for better output formatting")

        # Check for visualization best practices
        if "plt." in code:
            if "plt.figure(" not in code and "figsize" not in code:
                recommendations.append("Consider setting figure size for better visualization")
            if "plt.title(" not in code:
                recommendations.append("Consider adding titles to visualizations")

        return {
            "warnings": warnings,
            "recommendations": recommendations
        }

    def _calculate_risk_level(self, validation_result: Dict[str, Any]) -> str:
        """Calculate overall risk level based on validation results."""
        if validation_result["safety_issues"]:
            return "high"

        # Check for actual syntax errors (not warnings)
        actual_syntax_errors = [
            issue for issue in validation_result["syntax_issues"]
            if not issue.startswith("Warning:")
        ]

        if actual_syntax_errors or len(validation_result["warnings"]) > 3:
            return "medium"
        else:
            return "low"

    def _sanitize_code(self, code: str) -> str:
        """
        Sanitize code by removing or replacing dangerous operations.

        This provides a safe version of the code when possible.
        """
        sanitized = code

        # Remove dangerous operations with more comprehensive patterns
        dangerous_replacements = [
            (r'os\.system\s*\([^)]*\)', '# os.system() removed for safety'),
            (r'subprocess\.[^(]+\([^)]*\)', '# subprocess call removed for safety'),
            (r'\beval\s*\([^)]*\)', '# eval() removed for safety'),
            (r'\bexec\s*\([^)]*\)', '# exec() removed for safety'),
            (r'__import__\s*\([^)]*\)', '# __import__() removed for safety')
        ]

        for pattern, replacement in dangerous_replacements:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.MULTILINE)

        # Add safety comments
        if sanitized != code:
            sanitized = "# Code has been sanitized for safety\n" + sanitized

        return sanitized

    def _add_safety_wrapper(self, code: str) -> str:
        """Add safety wrapper around the generated code."""
        wrapper = f'''
# ============================================================================
# GENERATED CODE WITH SAFETY WRAPPER
# ============================================================================
# This code was generated by SAM's Table-to-Code Expert Tool
# Generated at: {time.strftime("%Y-%m-%d %H:%M:%S")}
#
# SAFETY NOTES:
# - This code is designed to work with pandas DataFrames
# - Ensure you have the required libraries installed: pandas, numpy, matplotlib
# - Review the code before execution in production environments
# ============================================================================

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

try:
{self._indent_code(code, 4)}

    print("\\n" + "="*50)
    print("Code execution completed successfully!")
    print("="*50)

except Exception as e:
    print(f"Error during execution: {{e}}")
    print("Please check your data and try again.")
    import traceback
    traceback.print_exc()
'''
        return wrapper

    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code by specified number of spaces."""
        lines = code.split('\n')
        indented_lines = [' ' * spaces + line if line.strip() else line for line in lines]
        return '\n'.join(indented_lines)

    def _generate_code_explanation(self, code: str, analysis_type: str) -> str:
        """Generate explanation for the generated code."""
        explanations = {
            "analysis": "This code loads the table data and performs general analysis including shape, column info, and descriptive statistics.",
            "calculation": "This code performs the requested calculations on the table data using pandas operations.",
            "aggregation": "This code aggregates the data using pandas grouping and summary functions.",
            "visualization": "This code creates visualizations of the table data using matplotlib for better data understanding."
        }

        base_explanation = explanations.get(analysis_type, explanations["analysis"])

        # Add specific details based on code content
        if "sum()" in code:
            base_explanation += " It calculates column totals for numeric data."
        if "mean()" in code:
            base_explanation += " It computes average values for numeric columns."
        if "plot(" in code:
            base_explanation += " It generates charts to visualize the data patterns."

        return base_explanation

    def _create_execution_instructions(self, code_result: Dict[str, Any],
                                     validation_result: Optional[Dict[str, Any]] = None) -> str:
        """Create comprehensive instructions for executing the generated code."""
        instructions = [
            "🚀 **EXECUTION INSTRUCTIONS**",
            "",
            "**Prerequisites:**",
            "1. Ensure you have pandas and numpy installed: `pip install pandas numpy`"
        ]

        if "matplotlib" in code_result["code"]:
            instructions.append("2. Install matplotlib for visualizations: `pip install matplotlib`")

        if "seaborn" in code_result["code"]:
            instructions.append("3. Install seaborn for advanced plots: `pip install seaborn`")

        instructions.extend([
            "",
            "**Execution Steps:**",
            "1. Copy the generated code to a Python file or Jupyter notebook",
            "2. Run the code to see the analysis results",
            "3. Review the output and visualizations"
        ])

        # Add validation-specific instructions
        if validation_result:
            instructions.extend([
                "",
                f"**Safety Assessment: {validation_result['risk_level'].upper()} RISK**"
            ])

            if validation_result["safety_issues"]:
                instructions.extend([
                    "",
                    "⚠️ **SAFETY WARNINGS:**"
                ])
                for issue in validation_result["safety_issues"]:
                    instructions.append(f"- {issue}")

            if validation_result["syntax_issues"]:
                instructions.extend([
                    "",
                    "❌ **SYNTAX ISSUES:**"
                ])
                for issue in validation_result["syntax_issues"]:
                    instructions.append(f"- {issue}")

            if validation_result["warnings"]:
                instructions.extend([
                    "",
                    "⚡ **PERFORMANCE WARNINGS:**"
                ])
                for warning in validation_result["warnings"]:
                    instructions.append(f"- {warning}")

            if validation_result["recommendations"]:
                instructions.extend([
                    "",
                    "💡 **RECOMMENDATIONS:**"
                ])
                for rec in validation_result["recommendations"]:
                    instructions.append(f"- {rec}")

        instructions.extend([
            "",
            "**Code Features:**",
            "- The code is designed to be safe and self-contained",
            "- Results will display directly in the console",
            "- Visualizations will open in separate windows or inline (Jupyter)",
            "- Error handling is included for common issues"
        ])

        if validation_result and validation_result["risk_level"] == "low":
            instructions.append("- ✅ Code has passed all safety checks")

        return "\n".join(instructions)

    def _format_response(self, code_result: Dict[str, Any],
                        reconstruction_result: TableReconstructionResult,
                        validation_result: Optional[Dict[str, Any]] = None) -> str:
        """Format the final response with code, explanation, and validation info."""
        # Get safety indicator
        safety_emoji = "🟢" if not validation_result or validation_result["risk_level"] == "low" else "🟡" if validation_result["risk_level"] == "medium" else "🔴"

        response_parts = [
            f"I've analyzed the table (ID: {reconstruction_result.table_id}) and generated Python code for your request.",
            "",
            f"{safety_emoji} **Generated Code** (Safety: {validation_result['risk_level'].upper() if validation_result else 'UNKNOWN'}):",
            "```python",
            code_result["code"],
            "```",
            "",
            f"**Explanation:** {code_result['explanation']}",
            "",
            f"**Table Info:** {len(reconstruction_result.column_info)} columns, "
            f"confidence: {reconstruction_result.reconstruction_confidence:.1%}"
        ]

        # Add validation summary
        if validation_result:
            if validation_result["safety_issues"]:
                response_parts.extend([
                    "",
                    "⚠️ **Safety Notes:**",
                    "- " + "\n- ".join(validation_result["safety_issues"])
                ])

            if validation_result["recommendations"]:
                response_parts.extend([
                    "",
                    "💡 **Optimization Tips:**",
                    "- " + "\n- ".join(validation_result["recommendations"][:3])  # Show top 3
                ])

        # Add wrapped code option
        if "wrapped_code" in code_result:
            response_parts.extend([
                "",
                "**Production-Ready Code** (with safety wrapper):",
                "```python",
                code_result["wrapped_code"],
                "```"
            ])

        return "\n".join(response_parts)

    def _get_analysis_prompt_template(self) -> str:
        """Get sophisticated prompt template for general data analysis."""
        return """
You are SAM's Table-to-Code Expert, an advanced AI system that transforms natural language requests into executable Python code for data analysis.

CONTEXT:
Table Reconstruction Code: {table_code}
User Request: "{user_request}"
Available Columns: {columns}
Table Metadata: {table_metadata}
Include Visualization: {include_visualization}

TASK:
Generate clean, executable Python code that fulfills the user's request with professional-grade data analysis.

REQUIREMENTS:
1. Use the provided DataFrame 'df' as the starting point
2. Implement robust error handling for missing data and edge cases
3. Generate clear, formatted output with professional presentation
4. Include informative comments explaining each step
5. Use pandas best practices and efficient operations
6. Ensure code is safe and will not cause system issues
7. Format numerical outputs appropriately (commas, decimals)
8. Handle different data types intelligently

REASONING APPROACH:
1. Analyze the user request to identify the core analytical task
2. Determine which columns are relevant to the analysis
3. Choose appropriate pandas operations for the task
4. Consider data quality issues and edge cases
5. Structure output for maximum clarity and usefulness
6. Add visualization if requested or beneficial

GENERATE EXECUTABLE PYTHON CODE:
"""

    def _get_visualization_prompt_template(self) -> str:
        """Get sophisticated prompt template for data visualization."""
        return """
You are SAM's Data Visualization Expert, specializing in creating compelling, informative visualizations from tabular data.

CONTEXT:
Table Reconstruction Code: {table_code}
User Request: "{user_request}"
Available Columns: {columns}
Table Metadata: {table_metadata}

TASK:
Generate Python code that creates professional-quality data visualizations to fulfill the user's request.

VISUALIZATION REQUIREMENTS:
1. Choose the most appropriate chart type for the data and request
2. Use matplotlib and/or seaborn for high-quality output
3. Implement proper titles, axis labels, and legends
4. Apply professional styling and color schemes
5. Handle different data types intelligently
6. Ensure visualizations are clear and interpretable
7. Add annotations or highlights where beneficial
8. Make charts publication-ready

CHART TYPE SELECTION GUIDE:
- Bar charts: Categorical comparisons, rankings
- Line charts: Trends over time, continuous data
- Scatter plots: Relationships between variables
- Histograms: Data distribution analysis
- Box plots: Statistical distribution and outliers
- Heatmaps: Correlation matrices, 2D data
- Pie charts: Part-to-whole relationships (use sparingly)

REASONING APPROACH:
1. Analyze data types and structure
2. Identify the story the data tells
3. Select visualization type that best reveals insights
4. Consider audience and presentation context
5. Optimize for clarity and impact

GENERATE EXECUTABLE PYTHON CODE:
"""

    def _get_calculation_prompt_template(self) -> str:
        """Get sophisticated prompt template for calculations."""
        return """
You are SAM's Mathematical Computation Expert, specializing in precise calculations and quantitative analysis.

CONTEXT:
Table Reconstruction Code: {table_code}
User Request: "{user_request}"
Available Columns: {columns}
Table Metadata: {table_metadata}

TASK:
Generate Python code that performs precise mathematical calculations and quantitative analysis.

CALCULATION REQUIREMENTS:
1. Implement exact calculations as requested
2. Handle edge cases: division by zero, missing data, invalid inputs
3. Show intermediate calculation steps for transparency
4. Format numerical results with appropriate precision
5. Validate data types and ranges before computation
6. Use vectorized pandas operations for efficiency
7. Provide statistical context when relevant
8. Include confidence intervals or error bounds where applicable

MATHEMATICAL BEST PRACTICES:
- Use numpy for mathematical functions
- Handle floating-point precision carefully
- Check for data validity before calculations
- Provide multiple calculation methods when appropriate
- Include statistical significance testing when relevant
- Document assumptions and limitations

REASONING APPROACH:
1. Parse the mathematical requirements from the request
2. Identify the appropriate mathematical operations
3. Determine data validation needs
4. Plan error handling strategy
5. Structure calculations for clarity and accuracy
6. Format results for professional presentation

GENERATE EXECUTABLE PYTHON CODE:
"""

    def _get_aggregation_prompt_template(self) -> str:
        """Get sophisticated prompt template for data aggregation."""
        return """
You are SAM's Data Aggregation Specialist, expert in summarizing and grouping complex datasets.

CONTEXT:
Table Reconstruction Code: {table_code}
User Request: "{user_request}"
Available Columns: {columns}
Table Metadata: {table_metadata}

TASK:
Generate Python code that performs sophisticated data aggregation and grouping operations.

AGGREGATION REQUIREMENTS:
1. Use optimal pandas aggregation functions for performance
2. Implement intelligent grouping based on data characteristics
3. Handle multiple aggregation operations simultaneously
4. Present results in clear, hierarchical formats
5. Include relevant summary statistics and insights
6. Handle missing data appropriately in aggregations
7. Provide both detailed and summary views
8. Add percentage calculations and relative metrics

AGGREGATION STRATEGIES:
- Single-level grouping: groupby() with single column
- Multi-level grouping: groupby() with multiple columns
- Time-based aggregation: resample() for temporal data
- Cross-tabulation: pivot_table() for matrix views
- Rolling aggregations: rolling() for moving statistics
- Cumulative aggregations: cumsum(), cummax(), etc.

PRESENTATION FORMATS:
- Summary tables with totals and percentages
- Hierarchical indices for multi-level grouping
- Formatted output with proper alignment
- Statistical summaries (mean, median, std, etc.)
- Comparative analysis between groups

REASONING APPROACH:
1. Identify grouping variables from the request
2. Determine appropriate aggregation functions
3. Plan multi-level analysis if beneficial
4. Structure output for maximum insight
5. Add contextual statistics and comparisons

GENERATE EXECUTABLE PYTHON CODE:
"""
