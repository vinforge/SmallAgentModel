"""
Visualization Generator - Phase 2
=================================

Automatic chart and graph generation system based on table data and user intent.
Creates intelligent visualizations that best represent the data characteristics.

Supports:
- Automatic chart type selection based on data types
- Interactive visualizations with Plotly
- Statistical plots and business dashboards
- Export capabilities for various formats
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import base64
from io import BytesIO

logger = logging.getLogger(__name__)


@dataclass
class VisualizationResult:
    """Result of visualization generation."""
    chart_type: str
    title: str
    description: str
    code: str
    html_output: Optional[str]
    image_data: Optional[str]  # Base64 encoded image
    insights: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any]


class VisualizationGenerator:
    """
    Intelligent visualization generator that creates appropriate charts
    based on data characteristics and user intent.
    """
    
    def __init__(self):
        """Initialize the visualization generator."""
        self.chart_templates = self._load_chart_templates()
        self.color_palettes = {
            'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            'business': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83'],
            'professional': ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51'],
            'cool': ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087']
        }
        logger.info("VisualizationGenerator initialized")
    
    def generate_visualization(self, table_data: Dict[str, Any], 
                             chart_type: Optional[str] = None,
                             parameters: Optional[Dict[str, Any]] = None) -> VisualizationResult:
        """
        Generate appropriate visualization for table data.
        
        Args:
            table_data: Table data from TableAwareRetrieval
            chart_type: Specific chart type or 'auto' for automatic selection
            parameters: Additional parameters for customization
            
        Returns:
            VisualizationResult with generated chart
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame(table_data['data'])
            
            if df.empty:
                return self._create_error_result("No data available for visualization")
            
            # Auto-select chart type if not specified
            if not chart_type or chart_type == 'auto':
                chart_type = self._auto_select_chart_type(df)
            
            # Generate visualization based on type
            if chart_type == 'bar':
                return self._generate_bar_chart(df, table_data, parameters or {})
            elif chart_type == 'line':
                return self._generate_line_chart(df, table_data, parameters or {})
            elif chart_type == 'pie':
                return self._generate_pie_chart(df, table_data, parameters or {})
            elif chart_type == 'scatter':
                return self._generate_scatter_plot(df, table_data, parameters or {})
            elif chart_type == 'histogram':
                return self._generate_histogram(df, table_data, parameters or {})
            elif chart_type == 'heatmap':
                return self._generate_heatmap(df, table_data, parameters or {})
            elif chart_type == 'dashboard':
                return self._generate_dashboard(df, table_data, parameters or {})
            else:
                return self._create_error_result(f"Unsupported chart type: {chart_type}")
                
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            return self._create_error_result(f"Visualization failed: {str(e)}")
    
    def _auto_select_chart_type(self, df: pd.DataFrame) -> str:
        """Automatically select the best chart type based on data characteristics."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        num_numeric = len(numeric_cols)
        num_categorical = len(categorical_cols)
        num_rows = len(df)
        
        # Decision logic for chart type selection
        if num_categorical >= 1 and num_numeric >= 1:
            if num_rows <= 10:
                return 'bar'  # Good for small datasets
            elif df[categorical_cols[0]].nunique() <= 8:
                return 'bar'  # Good for limited categories
            else:
                return 'line'  # Better for many categories
                
        elif num_numeric >= 2:
            if num_numeric == 2:
                return 'scatter'  # Perfect for 2 numeric variables
            else:
                return 'heatmap'  # Good for correlation visualization
                
        elif num_categorical >= 1:
            unique_values = df[categorical_cols[0]].nunique()
            if unique_values <= 8:
                return 'pie'  # Good for limited categories
            else:
                return 'bar'  # Better for many categories
                
        elif num_numeric == 1:
            return 'histogram'  # Good for distribution analysis
            
        else:
            return 'bar'  # Default fallback
    
    def _generate_bar_chart(self, df: pd.DataFrame, table_data: Dict[str, Any], 
                           parameters: Dict[str, Any]) -> VisualizationResult:
        """Generate bar chart visualization."""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            # Determine x and y columns
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                x_col = categorical_cols[0]
                y_col = numeric_cols[0]
            elif len(numeric_cols) >= 2:
                x_col = df.columns[0]
                y_col = numeric_cols[0]
            else:
                x_col = df.columns[0]
                y_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            
            title = parameters.get('title', f"{y_col} by {x_col}")
            
            # Generate Plotly code
            code = f"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Data
data = {df.to_dict('records')}
df = pd.DataFrame(data)

# Create bar chart
fig = px.bar(df, x='{x_col}', y='{y_col}', 
             title='{title}',
             color_discrete_sequence=['#2E86AB'])

# Customize layout
fig.update_layout(
    xaxis_title='{x_col}',
    yaxis_title='{y_col}',
    showlegend=False,
    template='plotly_white',
    font=dict(size=12),
    title_font_size=16
)

# Show chart
fig.show()

# Save as HTML
fig.write_html('bar_chart.html')
"""
            
            # Generate insights
            insights = []
            if len(numeric_cols) > 0:
                max_value = df[y_col].max()
                max_category = df.loc[df[y_col].idxmax(), x_col]
                insights.append(f"Highest value: {max_value} in {max_category}")
                
                min_value = df[y_col].min()
                min_category = df.loc[df[y_col].idxmin(), x_col]
                insights.append(f"Lowest value: {min_value} in {min_category}")
                
                avg_value = df[y_col].mean()
                insights.append(f"Average value: {avg_value:.2f}")
            
            recommendations = [
                "Consider sorting bars by value for better readability",
                "Add data labels for precise values",
                "Use consistent color scheme for related charts"
            ]
            
            return VisualizationResult(
                chart_type='bar',
                title=title,
                description=f"Bar chart showing {y_col} across different {x_col} categories",
                code=code,
                html_output=None,
                image_data=None,
                insights=insights,
                recommendations=recommendations,
                metadata={
                    'x_column': x_col,
                    'y_column': y_col,
                    'data_points': len(df)
                }
            )
            
        except Exception as e:
            logger.error(f"Bar chart generation failed: {e}")
            raise
    
    def _generate_line_chart(self, df: pd.DataFrame, table_data: Dict[str, Any], 
                            parameters: Dict[str, Any]) -> VisualizationResult:
        """Generate line chart visualization."""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 1:
                return self._create_error_result("Need at least 1 numeric column for line chart")
            
            # Use index as x-axis if no clear time column
            x_col = df.columns[0]
            y_col = numeric_cols[0]
            
            title = parameters.get('title', f"{y_col} Trend")
            
            code = f"""
import plotly.graph_objects as go
import pandas as pd

# Data
data = {df.to_dict('records')}
df = pd.DataFrame(data)

# Create line chart
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df['{x_col}'],
    y=df['{y_col}'],
    mode='lines+markers',
    name='{y_col}',
    line=dict(color='#2E86AB', width=3),
    marker=dict(size=6)
))

# Customize layout
fig.update_layout(
    title='{title}',
    xaxis_title='{x_col}',
    yaxis_title='{y_col}',
    template='plotly_white',
    font=dict(size=12),
    title_font_size=16
)

# Show chart
fig.show()

# Save as HTML
fig.write_html('line_chart.html')
"""
            
            # Generate insights
            insights = []
            if len(df) > 1:
                trend_slope = np.polyfit(range(len(df)), df[y_col], 1)[0]
                if trend_slope > 0:
                    insights.append(f"{y_col} shows an increasing trend")
                elif trend_slope < 0:
                    insights.append(f"{y_col} shows a decreasing trend")
                else:
                    insights.append(f"{y_col} shows a stable trend")
                
                volatility = df[y_col].std()
                insights.append(f"Data volatility (std dev): {volatility:.2f}")
            
            recommendations = [
                "Consider adding trend lines for clearer patterns",
                "Use different colors for multiple series",
                "Add annotations for significant events"
            ]
            
            return VisualizationResult(
                chart_type='line',
                title=title,
                description=f"Line chart showing {y_col} trend over {x_col}",
                code=code,
                html_output=None,
                image_data=None,
                insights=insights,
                recommendations=recommendations,
                metadata={
                    'x_column': x_col,
                    'y_column': y_col,
                    'data_points': len(df)
                }
            )
            
        except Exception as e:
            logger.error(f"Line chart generation failed: {e}")
            raise
    
    def _generate_pie_chart(self, df: pd.DataFrame, table_data: Dict[str, Any], 
                           parameters: Dict[str, Any]) -> VisualizationResult:
        """Generate pie chart visualization."""
        try:
            categorical_cols = df.select_dtypes(include=['object']).columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(categorical_cols) == 0:
                return self._create_error_result("Need categorical column for pie chart")
            
            category_col = categorical_cols[0]
            
            # Use value counts or numeric column for values
            if len(numeric_cols) > 0:
                value_col = numeric_cols[0]
                # Group by category and sum values
                pie_data = df.groupby(category_col)[value_col].sum().reset_index()
                labels = pie_data[category_col].tolist()
                values = pie_data[value_col].tolist()
            else:
                # Use value counts
                value_counts = df[category_col].value_counts()
                labels = value_counts.index.tolist()
                values = value_counts.values.tolist()
            
            title = parameters.get('title', f"Distribution of {category_col}")
            
            code = f"""
import plotly.graph_objects as go
import pandas as pd

# Data
labels = {labels}
values = {values}

# Create pie chart
fig = go.Figure(data=[go.Pie(
    labels=labels,
    values=values,
    hole=0.3,  # Donut chart
    textinfo='label+percent',
    textposition='outside'
)])

# Customize layout
fig.update_layout(
    title='{title}',
    template='plotly_white',
    font=dict(size=12),
    title_font_size=16,
    showlegend=True
)

# Show chart
fig.show()

# Save as HTML
fig.write_html('pie_chart.html')
"""
            
            # Generate insights
            insights = []
            if values:
                max_idx = values.index(max(values))
                max_category = labels[max_idx]
                max_percentage = (max(values) / sum(values)) * 100
                insights.append(f"Largest segment: {max_category} ({max_percentage:.1f}%)")
                
                if len(values) > 1:
                    min_idx = values.index(min(values))
                    min_category = labels[min_idx]
                    min_percentage = (min(values) / sum(values)) * 100
                    insights.append(f"Smallest segment: {min_category} ({min_percentage:.1f}%)")
            
            recommendations = [
                "Consider combining small segments into 'Others' category",
                "Use consistent color scheme across related charts",
                "Add data labels for better readability"
            ]
            
            return VisualizationResult(
                chart_type='pie',
                title=title,
                description=f"Pie chart showing distribution of {category_col}",
                code=code,
                html_output=None,
                image_data=None,
                insights=insights,
                recommendations=recommendations,
                metadata={
                    'category_column': category_col,
                    'segments': len(labels)
                }
            )
            
        except Exception as e:
            logger.error(f"Pie chart generation failed: {e}")
            raise
    
    def _generate_scatter_plot(self, df: pd.DataFrame, table_data: Dict[str, Any], 
                              parameters: Dict[str, Any]) -> VisualizationResult:
        """Generate scatter plot visualization."""
        # Implementation for scatter plot
        pass
    
    def _generate_histogram(self, df: pd.DataFrame, table_data: Dict[str, Any], 
                           parameters: Dict[str, Any]) -> VisualizationResult:
        """Generate histogram visualization."""
        # Implementation for histogram
        pass
    
    def _generate_heatmap(self, df: pd.DataFrame, table_data: Dict[str, Any], 
                         parameters: Dict[str, Any]) -> VisualizationResult:
        """Generate heatmap visualization."""
        # Implementation for heatmap
        pass
    
    def _generate_dashboard(self, df: pd.DataFrame, table_data: Dict[str, Any], 
                           parameters: Dict[str, Any]) -> VisualizationResult:
        """Generate comprehensive dashboard."""
        # Implementation for dashboard
        pass
    
    def _create_error_result(self, error_message: str) -> VisualizationResult:
        """Create error result for failed visualizations."""
        return VisualizationResult(
            chart_type='error',
            title='Visualization Error',
            description=error_message,
            code='',
            html_output=None,
            image_data=None,
            insights=[error_message],
            recommendations=['Check data format and try again'],
            metadata={'error': True}
        )
    
    def _load_chart_templates(self) -> Dict[str, str]:
        """Load chart templates for different visualization types."""
        return {
            'base_plotly': '''
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Data setup
data = {data}
df = pd.DataFrame(data)
''',
            'styling': '''
# Apply consistent styling
fig.update_layout(
    template='plotly_white',
    font=dict(size=12),
    title_font_size=16,
    margin=dict(l=50, r=50, t=80, b=50)
)
'''
        }
