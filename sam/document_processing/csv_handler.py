"""
CSV File Handler for SAM
========================

Specialized handler for CSV files that integrates with the Code Interpreter
to enable data science capabilities.

Author: SAM Development Team
Version: 1.0.0
"""

import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class CSVHandler:
    """Handler for CSV file processing and analysis."""
    
    def __init__(self):
        """Initialize the CSV handler."""
        self.logger = logging.getLogger(__name__)
    
    def process_csv_file(self, file_path: str, filename: str = None, 
                        session_id: str = None) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Process a CSV file for data science capabilities.
        
        Args:
            file_path: Path to the CSV file
            filename: Original filename
            session_id: Session identifier
            
        Returns:
            Tuple of (success, message, metadata)
        """
        try:
            self.logger.info(f"📊 Processing CSV file: {filename or file_path}")
            
            # Read and analyze the CSV
            df = pd.read_csv(file_path)
            
            # Generate basic analysis
            analysis = self._analyze_csv_data(df)
            
            # Create metadata
            metadata = {
                'file_type': 'csv',
                'filename': filename or Path(file_path).name,
                'session_id': session_id,
                'processing_time': datetime.now().isoformat(),
                'data_analysis': analysis,
                'code_interpreter_ready': True,
                'data_science_capabilities': {
                    'basic_analysis': True,
                    'statistical_analysis': True,
                    'visualization': True,
                    'correlation_analysis': True
                }
            }
            
            # Success message with data science hints
            message = self._generate_success_message(analysis, filename)
            
            self.logger.info(f"✅ CSV processing complete: {analysis['shape']}")
            return True, message, metadata
            
        except Exception as e:
            self.logger.error(f"❌ CSV processing failed: {e}")
            return False, f"Failed to process CSV file: {str(e)}", {}
    
    def _analyze_csv_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze CSV data and generate insights."""
        try:
            analysis = {
                'shape': f"{df.shape[0]} rows × {df.shape[1]} columns",
                'columns': list(df.columns),
                'data_types': df.dtypes.to_dict(),
                'numeric_columns': list(df.select_dtypes(include=['number']).columns),
                'categorical_columns': list(df.select_dtypes(include=['object']).columns),
                'missing_values': df.isnull().sum().to_dict(),
                'sample_data': df.head(3).to_dict('records')
            }
            
            # Add statistical summary for numeric columns
            if analysis['numeric_columns']:
                numeric_summary = df[analysis['numeric_columns']].describe().to_dict()
                analysis['statistical_summary'] = numeric_summary
            
            # Detect potential correlations
            if len(analysis['numeric_columns']) >= 2:
                correlations = df[analysis['numeric_columns']].corr()
                # Find strong correlations (> 0.7)
                strong_correlations = []
                for i, col1 in enumerate(analysis['numeric_columns']):
                    for j, col2 in enumerate(analysis['numeric_columns']):
                        if i < j:  # Avoid duplicates
                            corr_value = correlations.loc[col1, col2]
                            if abs(corr_value) > 0.7:
                                strong_correlations.append({
                                    'column1': col1,
                                    'column2': col2,
                                    'correlation': round(corr_value, 3)
                                })
                
                analysis['strong_correlations'] = strong_correlations
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"CSV analysis failed: {e}")
            return {'error': str(e)}
    
    def _generate_success_message(self, analysis: Dict[str, Any], filename: str) -> str:
        """Generate a user-friendly success message with data science hints."""
        try:
            filename = filename or "your CSV file"
            
            message_parts = [
                f"✅ **{filename} successfully uploaded and analyzed!**",
                "",
                f"📊 **Dataset Overview:**",
                f"   • {analysis['shape']}",
                f"   • {len(analysis['numeric_columns'])} numeric columns",
                f"   • {len(analysis['categorical_columns'])} categorical columns"
            ]
            
            # Add data science suggestions
            message_parts.extend([
                "",
                f"🧠 **Data Science Capabilities Unlocked:**",
                f"   • Ask for statistical analysis: *\"Calculate the average of [column]\"*",
                f"   • Request correlations: *\"What are the correlations in this data?\"*",
                f"   • Generate visualizations: *\"Create a plot showing [relationship]\"*",
                f"   • Perform grouped analysis: *\"Analyze by [category column]\"*"
            ])
            
            # Add specific suggestions based on data
            if analysis.get('strong_correlations'):
                message_parts.extend([
                    "",
                    f"🔍 **Interesting Patterns Detected:**"
                ])
                for corr in analysis['strong_correlations'][:3]:  # Show top 3
                    message_parts.append(
                        f"   • Strong correlation between {corr['column1']} and {corr['column2']} ({corr['correlation']})"
                    )
            
            # Add example prompts based on columns
            if analysis['numeric_columns']:
                example_col = analysis['numeric_columns'][0]
                message_parts.extend([
                    "",
                    f"💡 **Try asking:**",
                    f"   • *\"What's the average {example_col}?\"*",
                    f"   • *\"Show me a histogram of {example_col}\"*"
                ])
            
            if len(analysis['numeric_columns']) >= 2:
                col1, col2 = analysis['numeric_columns'][:2]
                message_parts.append(f"   • *\"Plot {col1} vs {col2}\"*")
            
            if analysis['categorical_columns']:
                cat_col = analysis['categorical_columns'][0]
                if analysis['numeric_columns']:
                    num_col = analysis['numeric_columns'][0]
                    message_parts.append(f"   • *\"Compare {num_col} by {cat_col}\"*")
            
            return "\n".join(message_parts)
            
        except Exception as e:
            self.logger.error(f"Message generation failed: {e}")
            return f"✅ CSV file {filename} processed successfully and ready for data science analysis!"


def handle_csv_upload(file_path: str, filename: str = None, 
                     session_id: str = None) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Handle CSV file upload with data science integration.
    
    Args:
        file_path: Path to the uploaded CSV file
        filename: Original filename
        session_id: Session identifier
        
    Returns:
        Tuple of (success, message, metadata)
    """
    handler = CSVHandler()
    return handler.process_csv_file(file_path, filename, session_id)


def is_csv_file(file_path: str) -> bool:
    """Check if a file is a CSV file."""
    try:
        return Path(file_path).suffix.lower() == '.csv'
    except:
        return False


def get_csv_analysis_suggestions(df: pd.DataFrame) -> Dict[str, Any]:
    """Get analysis suggestions for a CSV dataset."""
    try:
        suggestions = {
            'basic_stats': [],
            'visualizations': [],
            'correlations': [],
            'grouping': []
        }
        
        numeric_cols = list(df.select_dtypes(include=['number']).columns)
        categorical_cols = list(df.select_dtypes(include=['object']).columns)
        
        # Basic statistics suggestions
        for col in numeric_cols[:3]:  # Top 3 numeric columns
            suggestions['basic_stats'].append(f"Calculate statistics for {col}")
        
        # Visualization suggestions
        if len(numeric_cols) >= 2:
            suggestions['visualizations'].append(f"Scatter plot: {numeric_cols[0]} vs {numeric_cols[1]}")
        
        for col in numeric_cols[:2]:
            suggestions['visualizations'].append(f"Histogram of {col}")
        
        # Correlation suggestions
        if len(numeric_cols) >= 2:
            suggestions['correlations'].append("Generate correlation matrix")
            suggestions['correlations'].append("Find strongest correlations")
        
        # Grouping suggestions
        if categorical_cols and numeric_cols:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            suggestions['grouping'].append(f"Group {num_col} by {cat_col}")
        
        return suggestions
        
    except Exception as e:
        logger.error(f"Failed to generate suggestions: {e}")
        return {}
