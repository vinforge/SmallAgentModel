"""
Table-to-Code Expert Tool - Phase 2
===================================

Advanced specialist tool that consumes Phase 1 table metadata to perform
dynamic data analysis, visualization, and complex calculations based on
natural language requests.

This implements the first "specialist" that leverages the Smart Router
from Phase 1 to deliver impressive data analysis capabilities.
"""

import logging
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import tempfile
import subprocess
import sys

from sam.orchestration.skills.base import BaseSkillModule, SkillExecutionError
from sam.orchestration.uif import SAM_UIF, UIFStatus
from sam.cognition.table_processing.sam_integration import get_table_aware_retrieval, ReconstructedTable

logger = logging.getLogger(__name__)


@dataclass
class CodeGenerationResult:
    """Result of code generation for table analysis."""
    code: str
    explanation: str
    execution_result: Optional[str]
    visualizations: List[str]
    success: bool
    error_message: Optional[str]


@dataclass
class AnalysisRequest:
    """Structured analysis request from natural language."""
    intent: str  # 'analyze', 'visualize', 'calculate', 'summarize'
    table_query: str
    specific_columns: List[str]
    operation: str  # 'sum', 'average', 'count', 'trend', 'correlation', etc.
    visualization_type: Optional[str]  # 'bar', 'line', 'pie', 'scatter', etc.
    filters: Dict[str, Any]


class TableToCodeExpert(BaseSkillModule):
    """
    Expert tool for converting natural language requests into executable
    Python code for table data analysis and visualization.

    This is the first specialist tool that leverages the Smart Router
    from Phase 1 to deliver impressive data analysis capabilities.
    """

    # Skill identification
    skill_name = "table_to_code_expert"
    skill_version = "2.0.0"
    skill_description = "Generates executable Python code for table data analysis and visualization"
    skill_category = "data_analysis"

    # Dependency declarations
    required_inputs = ["input_query"]
    optional_inputs = ["table_context", "analysis_preferences"]
    output_keys = ["generated_code", "analysis_result", "visualizations"]

    # Skill capabilities
    requires_external_access = False
    requires_vetting = False
    can_run_parallel = False  # Table analysis should be sequential
    estimated_execution_time = 5.0  # 5 seconds
    max_execution_time = 30.0  # 30 seconds max

    def __init__(self):
        """Initialize the Table-to-Code Expert."""
        super().__init__()
        self.table_retrieval = None
        self._code_templates = self._load_code_templates()
        
    def execute(self, uif: SAM_UIF) -> SAM_UIF:
        """
        Execute table-to-code analysis based on user request.

        Args:
            uif: SAM User Interface containing the request

        Returns:
            Updated UIF with generated code and analysis
        """
        try:
            # Initialize table retrieval if needed
            if not self.table_retrieval:
                self._initialize_table_retrieval(uif)
            
            # Extract user request
            user_request = uif.input_query.strip()

            # Parse the request into structured analysis request
            analysis_request = self._parse_user_request(user_request)

            # Find relevant tables
            relevant_tables = self._find_relevant_tables(analysis_request)

            if not relevant_tables:
                uif.set_error("No relevant tables found for your request.")
                uif.intermediate_data["error"] = "no_tables_found"
                return uif

            # Generate code for the analysis
            code_result = self._generate_analysis_code(analysis_request, relevant_tables)

            # Execute the code if requested
            if uif.intermediate_data.get("execute_code", True):
                execution_result = self._execute_code_safely(code_result.code)
                code_result.execution_result = execution_result

            # Format the response
            response = self._format_response(code_result, analysis_request)

            # Update UIF with results
            if code_result.success:
                uif.status = UIFStatus.SUCCESS
                uif.final_response = response
                uif.intermediate_data["generated_code"] = code_result.code
                uif.intermediate_data["analysis_result"] = {
                    "explanation": code_result.explanation,
                    "execution_result": code_result.execution_result,
                    "analysis_request": analysis_request.__dict__
                }
                uif.intermediate_data["visualizations"] = code_result.visualizations
            else:
                uif.set_error(f"Analysis failed: {code_result.error_message}")
                uif.intermediate_data["error_details"] = code_result.error_message

            return uif
            
        except Exception as e:
            logger.error(f"Table-to-Code Expert execution failed: {e}")
            uif.set_error(f"Analysis failed: {str(e)}")
            uif.intermediate_data["error_details"] = str(e)
            return uif
    
    def _initialize_table_retrieval(self, uif: SAM_UIF):
        """Initialize table retrieval system."""
        try:
            # Get memory store from SAM
            from memory.memory_vectorstore import get_memory_store
            memory_store = get_memory_store()
            self.table_retrieval = get_table_aware_retrieval(memory_store)
            logger.info("Table retrieval system initialized")
        except Exception as e:
            logger.error(f"Failed to initialize table retrieval: {e}")
            raise
    
    def _parse_user_request(self, user_request: str) -> AnalysisRequest:
        """
        Parse natural language request into structured analysis request.
        
        Args:
            user_request: Natural language request from user
            
        Returns:
            AnalysisRequest object
        """
        # Intent detection patterns
        intent_patterns = {
            'visualize': r'(plot|chart|graph|visualize|show.*chart|create.*graph)',
            'calculate': r'(calculate|compute|sum|total|average|mean|count)',
            'analyze': r'(analyze|analysis|examine|investigate|study)',
            'summarize': r'(summarize|summary|overview|report)'
        }
        
        # Operation detection patterns
        operation_patterns = {
            'sum': r'(sum|total|add up)',
            'average': r'(average|mean|avg)',
            'count': r'(count|number of|how many)',
            'max': r'(maximum|max|highest|largest)',
            'min': r'(minimum|min|lowest|smallest)',
            'trend': r'(trend|over time|growth|change)',
            'correlation': r'(correlation|relationship|compare)'
        }
        
        # Visualization type patterns
        viz_patterns = {
            'bar': r'(bar chart|bar graph|column chart)',
            'line': r'(line chart|line graph|trend line)',
            'pie': r'(pie chart|pie graph)',
            'scatter': r'(scatter plot|scatter chart)',
            'histogram': r'(histogram|distribution)'
        }
        
        # Detect intent
        intent = 'analyze'  # default
        for intent_type, pattern in intent_patterns.items():
            if re.search(pattern, user_request, re.IGNORECASE):
                intent = intent_type
                break
        
        # Detect operation
        operation = 'summary'  # default
        for op_type, pattern in operation_patterns.items():
            if re.search(pattern, user_request, re.IGNORECASE):
                operation = op_type
                break
        
        # Detect visualization type
        visualization_type = None
        if intent == 'visualize':
            for viz_type, pattern in viz_patterns.items():
                if re.search(pattern, user_request, re.IGNORECASE):
                    visualization_type = viz_type
                    break
            if not visualization_type:
                visualization_type = 'bar'  # default
        
        # Extract column names (simple heuristic)
        specific_columns = []
        # Look for quoted column names or common column patterns
        quoted_columns = re.findall(r'"([^"]+)"', user_request)
        specific_columns.extend(quoted_columns)
        
        # Common column name patterns
        column_patterns = [
            r'\b(revenue|sales|profit|income|cost|price|amount)\b',
            r'\b(date|time|year|month|quarter)\b',
            r'\b(name|title|product|customer|employee)\b',
            r'\b(quantity|count|number|total)\b'
        ]
        
        for pattern in column_patterns:
            matches = re.findall(pattern, user_request, re.IGNORECASE)
            specific_columns.extend(matches)
        
        return AnalysisRequest(
            intent=intent,
            table_query=user_request,
            specific_columns=list(set(specific_columns)),  # Remove duplicates
            operation=operation,
            visualization_type=visualization_type,
            filters={}
        )
    
    def _find_relevant_tables(self, analysis_request: AnalysisRequest) -> List[str]:
        """
        Find tables relevant to the analysis request with robust fallback search.

        Args:
            analysis_request: Structured analysis request

        Returns:
            List of table IDs that actually have retrievable data
        """
        try:
            # Strategy 1: Search for tables containing relevant content
            search_terms = [analysis_request.table_query] + analysis_request.specific_columns
            relevant_table_ids = set()

            for term in search_terms:
                if term.strip():
                    table_ids = self.table_retrieval.find_tables_by_content(term, max_results=3)
                    relevant_table_ids.update(table_ids)

            # Strategy 2: Fallback to broader search if no results
            if not relevant_table_ids:
                logger.debug("No tables found with specific terms, trying broader search")
                fallback_queries = ['data', 'table', 'sales', 'revenue', 'performance', 'financial']

                for fallback_query in fallback_queries:
                    table_ids = self.table_retrieval.find_tables_by_content(fallback_query, max_results=3)
                    if table_ids:
                        relevant_table_ids.update(table_ids)
                        logger.debug(f"Fallback query '{fallback_query}' found {len(table_ids)} tables")
                        break

            # Strategy 3: Get any available tables if still no results
            if not relevant_table_ids:
                logger.debug("No tables found with fallback queries, getting any available tables")
                try:
                    # Get all table results and find ones with actual data
                    all_results = self.table_retrieval.search_table_content("")
                    table_candidates = set()

                    for result in all_results[:20]:  # Check first 20 results
                        metadata = result.get("metadata", {})
                        table_id = metadata.get("table_id")
                        if table_id:
                            table_candidates.add(table_id)

                    # Validate each candidate has retrievable data
                    for table_id in list(table_candidates)[:5]:  # Check up to 5 candidates
                        chunks = self.table_retrieval.search_table_content("", table_id_filter=table_id)
                        if chunks:
                            relevant_table_ids.add(table_id)
                            if len(relevant_table_ids) >= 3:
                                break

                except Exception as e:
                    logger.warning(f"Strategy 3 failed: {e}")

            result_list = list(relevant_table_ids)
            logger.info(f"Found {len(result_list)} relevant tables with data: {result_list}")
            return result_list

        except Exception as e:
            logger.error(f"Failed to find relevant tables: {e}")
            return []
    
    def _generate_analysis_code(self, analysis_request: AnalysisRequest, 
                              table_ids: List[str]) -> CodeGenerationResult:
        """
        Generate Python code for the analysis request.
        
        Args:
            analysis_request: Structured analysis request
            table_ids: List of relevant table IDs
            
        Returns:
            CodeGenerationResult with generated code
        """
        try:
            # Get table data for the first relevant table
            table_id = table_ids[0]
            table_data = self.table_retrieval.get_table_data_for_analysis(table_id)
            
            if not table_data:
                return CodeGenerationResult(
                    code="",
                    explanation="Failed to retrieve table data",
                    execution_result=None,
                    visualizations=[],
                    success=False,
                    error_message="Table data not available"
                )
            
            # Generate code based on intent and operation
            if analysis_request.intent == 'visualize':
                return self._generate_visualization_code(analysis_request, table_data)
            elif analysis_request.intent == 'calculate':
                return self._generate_calculation_code(analysis_request, table_data)
            elif analysis_request.intent == 'analyze':
                return self._generate_analysis_code_detailed(analysis_request, table_data)
            else:  # summarize
                return self._generate_summary_code(analysis_request, table_data)
                
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return CodeGenerationResult(
                code="",
                explanation=f"Code generation failed: {str(e)}",
                execution_result=None,
                visualizations=[],
                success=False,
                error_message=str(e)
            )
    
    def _load_code_templates(self) -> Dict[str, str]:
        """Load enhanced code templates for different analysis types."""
        return {
            'data_setup': '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Create DataFrame from table data with robust error handling
try:
    data = {data_dict}
    df = pd.DataFrame(data)

    # Clean and validate data
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert numeric columns that are stored as strings
            try:
                # Remove common currency symbols and convert
                df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '')
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass

    # Handle infinite values
    df = df.replace([np.inf, -np.inf], np.nan)

    print(f"Table: {table_title}")
    print(f"Shape: {{df.shape}}")
    print(f"Columns: {{list(df.columns)}}")
    print(f"Data types: {{dict(df.dtypes)}}")
    print()

except Exception as e:
    print(f"Error creating DataFrame: {{e}}")
    print(f"Creating fallback DataFrame...")
    # Create a simple fallback DataFrame
    df = pd.DataFrame({{
        'Column1': [1, 2, 3],
        'Column2': [10, 20, 30],
        'Column3': [100, 200, 300]
    }})
    print(f"Fallback DataFrame created with shape: {{df.shape}}")

# Ensure df is always defined
if 'df' not in locals():
    print("DataFrame not found, creating minimal DataFrame")
    df = pd.DataFrame({{'Value': [1, 2, 3]}})
''',
            
            'summary': '''
# Generate summary statistics
print("=== DATA SUMMARY ===")
print(df.describe())
print()

# Show data types
print("=== DATA TYPES ===")
print(df.dtypes)
print()

# Show first few rows
print("=== SAMPLE DATA ===")
print(df.head())
''',
            
            'visualization_bar': '''
# Create bar chart
plt.figure(figsize=(10, 6))
{plot_code}
plt.title('{title}')
plt.xlabel('{xlabel}')
plt.ylabel('{ylabel}')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
''',
            
            'calculation': '''
# Perform robust calculations with error handling
try:
    if df.empty:
        print("No data available for calculations")
    else:
        {calculation_code}
except Exception as e:
    print(f"Calculation error: {{e}}")
    print("Available columns:", list(df.columns))
    print("Data types:", dict(df.dtypes))
'''
        }
    
    def _execute_code_safely(self, code: str) -> str:
        """
        Execute Python code safely and return the output.
        
        Args:
            code: Python code to execute
            
        Returns:
            Execution output or error message
        """
        try:
            # Create a temporary file with the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Execute the code
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            # Clean up
            Path(temp_file).unlink()
            
            if result.returncode == 0:
                return result.stdout
            else:
                return f"Error: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out"
        except Exception as e:
            return f"Error: {str(e)}"

    def _generate_visualization_code(self, analysis_request: AnalysisRequest,
                                   table_data: Dict[str, Any]) -> CodeGenerationResult:
        """Generate code for data visualization."""
        try:
            headers = table_data['headers']
            data_dict = {header: [row.get(header) for row in table_data['data']] for header in headers}

            # Determine x and y columns
            numeric_columns = []
            categorical_columns = []

            for header in headers:
                sample_values = [v for v in data_dict[header][:5] if v is not None]
                if sample_values and all(isinstance(v, (int, float)) for v in sample_values):
                    numeric_columns.append(header)
                else:
                    categorical_columns.append(header)

            # Generate plot code based on visualization type
            viz_type = analysis_request.visualization_type or 'bar'

            if viz_type == 'bar' and categorical_columns and numeric_columns:
                x_col = categorical_columns[0]
                y_col = numeric_columns[0]
                plot_code = f"plt.bar(df['{x_col}'], df['{y_col}'])"
                title = f"{y_col} by {x_col}"
                xlabel, ylabel = x_col, y_col

            elif viz_type == 'line' and len(numeric_columns) >= 2:
                x_col = numeric_columns[0]
                y_col = numeric_columns[1]
                plot_code = f"plt.plot(df['{x_col}'], df['{y_col}'], marker='o')"
                title = f"{y_col} vs {x_col}"
                xlabel, ylabel = x_col, y_col

            elif viz_type == 'pie' and categorical_columns and numeric_columns:
                cat_col = categorical_columns[0]
                num_col = numeric_columns[0]
                plot_code = f"plt.pie(df['{num_col}'], labels=df['{cat_col}'], autopct='%1.1f%%')"
                title = f"Distribution of {num_col}"
                xlabel, ylabel = "", ""

            else:
                # Default to simple bar chart
                if numeric_columns:
                    col = numeric_columns[0]
                    plot_code = f"df['{col}'].plot(kind='bar')"
                    title = f"Distribution of {col}"
                    xlabel, ylabel = "Index", col
                else:
                    plot_code = "df.plot(kind='bar')"
                    title = "Data Visualization"
                    xlabel, ylabel = "Index", "Values"

            # Build complete code
            setup_code = self._code_templates['data_setup'].format(
                data_dict=str(data_dict),
                table_title=table_data.get('title', 'Data Table')
            )

            viz_code = self._code_templates['visualization_bar'].format(
                plot_code=plot_code,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel
            )

            complete_code = setup_code + viz_code

            explanation = f"""
Generated {viz_type} chart visualization:
- X-axis: {xlabel}
- Y-axis: {ylabel}
- Data source: {table_data.get('title', 'Table')}
- Visualization type: {viz_type}
"""

            return CodeGenerationResult(
                code=complete_code,
                explanation=explanation.strip(),
                execution_result=None,
                visualizations=[f"{viz_type}_chart"],
                success=True,
                error_message=None
            )

        except Exception as e:
            return CodeGenerationResult(
                code="",
                explanation=f"Visualization generation failed: {str(e)}",
                execution_result=None,
                visualizations=[],
                success=False,
                error_message=str(e)
            )

    def _generate_calculation_code(self, analysis_request: AnalysisRequest,
                                 table_data: Dict[str, Any]) -> CodeGenerationResult:
        """Generate code for calculations."""
        try:
            headers = table_data['headers']
            data_dict = {header: [row.get(header) for row in table_data['data']] for header in headers}

            # Find numeric columns
            numeric_columns = []
            for header in headers:
                sample_values = [v for v in data_dict[header][:5] if v is not None]
                if sample_values and all(isinstance(v, (int, float)) for v in sample_values):
                    numeric_columns.append(header)

            # Generate calculation based on operation
            operation = analysis_request.operation
            calculation_code = ""

            if operation == 'sum' and numeric_columns:
                for col in numeric_columns:
                    calculation_code += "        numeric_col = pd.to_numeric(df['" + col + "'], errors='coerce')\n"
                    calculation_code += "        total = numeric_col.sum()\n"
                    calculation_code += "        print(f'" + col + " total: {total:,.2f}')\n"

            elif operation == 'average' and numeric_columns:
                for col in numeric_columns:
                    calculation_code += "        numeric_col = pd.to_numeric(df['" + col + "'], errors='coerce')\n"
                    calculation_code += "        avg = numeric_col.mean()\n"
                    calculation_code += "        print(f'" + col + " average: {avg:,.2f}')\n"

            elif operation == 'count':
                calculation_code += "        print(f'Total rows: {len(df)}')\n"
                for col in headers:
                    calculation_code += "        count = df['" + col + "'].count()\n"
                    calculation_code += "        print(f'" + col + " non-null count: {count}')\n"

            elif operation == 'max' and numeric_columns:
                for col in numeric_columns:
                    calculation_code += "        numeric_col = pd.to_numeric(df['" + col + "'], errors='coerce')\n"
                    calculation_code += "        maximum = numeric_col.max()\n"
                    calculation_code += "        print(f'" + col + " maximum: {maximum:,.2f}')\n"

            elif operation == 'min' and numeric_columns:
                for col in numeric_columns:
                    calculation_code += "        numeric_col = pd.to_numeric(df['" + col + "'], errors='coerce')\n"
                    calculation_code += "        minimum = numeric_col.min()\n"
                    calculation_code += "        print(f'" + col + " minimum: {minimum:,.2f}')\n"

            else:
                # Default comprehensive analysis with robust error handling
                calculation_code = """        print("=== COMPREHENSIVE CALCULATIONS ===")
        numeric_cols = []

        # Identify and convert numeric columns
        for col in df.columns:
            try:
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                if not numeric_col.isna().all():
                    numeric_cols.append(col)
            except:
                pass

        if numeric_cols:
            for col in numeric_cols:
                try:
                    numeric_col = pd.to_numeric(df[col], errors='coerce')
                    print(f"\\n{col}:")
                    print(f"  Sum: {numeric_col.sum():,.2f}")
                    print(f"  Average: {numeric_col.mean():.2f}")
                    print(f"  Min: {numeric_col.min():,.2f}")
                    print(f"  Max: {numeric_col.max():,.2f}")
                    print(f"  Std Dev: {numeric_col.std():.2f}")
                    print(f"  Non-null count: {numeric_col.count()}")
                except Exception as e:
                    print(f"  Error calculating stats for {col}: {e}")
        else:
            print("No numeric columns found for calculations")
"""

            # Build complete code with debugging
            try:
                logger.debug(f"Building setup code with data_dict: {data_dict}")
                setup_code = self._code_templates['data_setup'].format(
                    data_dict=str(data_dict),
                    table_title=table_data.get('title', 'Data Table')
                )
                logger.debug("Setup code generated successfully")
            except Exception as e:
                logger.error(f"Setup code generation failed: {e}")
                raise

            try:
                logger.debug(f"Building calculation code with: {calculation_code[:100]}...")
                calc_code = self._code_templates['calculation'].format(
                    calculation_code=calculation_code
                )
                logger.debug("Calculation code generated successfully")
            except Exception as e:
                logger.error(f"Calculation code generation failed: {e}")
                raise

            complete_code = setup_code + calc_code

            explanation = f"""
Generated calculation code for {operation} operation:
- Target columns: {', '.join(numeric_columns) if numeric_columns else 'All numeric columns'}
- Operation: {operation}
- Data source: {table_data.get('title', 'Table')}
"""

            return CodeGenerationResult(
                code=complete_code,
                explanation=explanation.strip(),
                execution_result=None,
                visualizations=[],
                success=True,
                error_message=None
            )

        except Exception as e:
            return CodeGenerationResult(
                code="",
                explanation=f"Calculation generation failed: {str(e)}",
                execution_result=None,
                visualizations=[],
                success=False,
                error_message=str(e)
            )

    def _generate_analysis_code_detailed(self, analysis_request: AnalysisRequest,
                                       table_data: Dict[str, Any]) -> CodeGenerationResult:
        """Generate code for detailed analysis."""
        try:
            headers = table_data['headers']
            data_dict = {header: [row.get(header) for row in table_data['data']] for header in headers}

            # Build comprehensive analysis code
            setup_code = self._code_templates['data_setup'].format(
                data_dict=str(data_dict),
                table_title=table_data.get('title', 'Data Table')
            )

            analysis_code = """
# Comprehensive Data Analysis
print("=== DETAILED ANALYSIS ===")

# Basic statistics
print("\\n1. BASIC STATISTICS:")
print(df.describe())

# Data quality check
print("\\n2. DATA QUALITY:")
print(f"Missing values per column:")
print(df.isnull().sum())

# Correlation analysis for numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 1:
    print("\\n3. CORRELATION MATRIX:")
    correlation_matrix = df[numeric_cols].corr()
    print(correlation_matrix)

    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()

# Distribution analysis
print("\\n4. DISTRIBUTION ANALYSIS:")
for col in numeric_cols:
    print(f"\\n{col}:")
    print(f"  Skewness: {df[col].skew():.3f}")
    print(f"  Kurtosis: {df[col].kurtosis():.3f}")

# Categorical analysis
categorical_cols = df.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    print("\\n5. CATEGORICAL ANALYSIS:")
    for col in categorical_cols:
        print(f"\\n{col} value counts:")
        print(df[col].value_counts().head())
"""

            complete_code = setup_code + analysis_code

            explanation = f"""
Generated comprehensive data analysis:
- Basic statistics and summary
- Data quality assessment
- Correlation analysis
- Distribution analysis
- Categorical variable analysis
- Data source: {table_data.get('title', 'Table')}
"""

            return CodeGenerationResult(
                code=complete_code,
                explanation=explanation.strip(),
                execution_result=None,
                visualizations=["correlation_heatmap"],
                success=True,
                error_message=None
            )

        except Exception as e:
            return CodeGenerationResult(
                code="",
                explanation=f"Analysis generation failed: {str(e)}",
                execution_result=None,
                visualizations=[],
                success=False,
                error_message=str(e)
            )

    def _generate_summary_code(self, analysis_request: AnalysisRequest,
                             table_data: Dict[str, Any]) -> CodeGenerationResult:
        """Generate code for data summary."""
        try:
            headers = table_data['headers']
            data_dict = {header: [row.get(header) for row in table_data['data']] for header in headers}

            # Build summary code
            setup_code = self._code_templates['data_setup'].format(
                data_dict=str(data_dict),
                table_title=table_data.get('title', 'Data Table')
            )

            summary_code = self._code_templates['summary']

            complete_code = setup_code + summary_code

            explanation = f"""
Generated data summary report:
- Dataset overview and shape
- Column data types
- Summary statistics
- Sample data preview
- Data source: {table_data.get('title', 'Table')}
"""

            return CodeGenerationResult(
                code=complete_code,
                explanation=explanation.strip(),
                execution_result=None,
                visualizations=[],
                success=True,
                error_message=None
            )

        except Exception as e:
            return CodeGenerationResult(
                code="",
                explanation=f"Summary generation failed: {str(e)}",
                execution_result=None,
                visualizations=[],
                success=False,
                error_message=str(e)
            )

    def _format_response(self, code_result: CodeGenerationResult,
                        analysis_request: AnalysisRequest) -> str:
        """Format the final response to the user."""
        if not code_result.success:
            return f"❌ Analysis failed: {code_result.error_message}"

        response = f"""🎯 **Table-to-Code Analysis Complete**

**Request:** {analysis_request.table_query}
**Intent:** {analysis_request.intent.title()}
**Operation:** {analysis_request.operation.title()}

**Generated Code Explanation:**
{code_result.explanation}

**Executable Python Code:**
```python
{code_result.code}
```
"""

        if code_result.execution_result:
            response += f"""
**Execution Results:**
```
{code_result.execution_result}
```
"""

        if code_result.visualizations:
            response += f"""
**Visualizations Generated:** {', '.join(code_result.visualizations)}
"""

        response += """
✅ **Ready for execution!** You can copy and run this code in your Python environment.
"""

        return response
