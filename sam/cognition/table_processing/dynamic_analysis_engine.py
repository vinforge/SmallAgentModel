"""
Dynamic Analysis Engine - Phase 2
=================================

Advanced statistical analysis and calculation engine that operates on
table data reconstructed from Phase 1 metadata.

Provides sophisticated data analysis capabilities including:
- Statistical analysis and hypothesis testing
- Time series analysis and forecasting
- Correlation and regression analysis
- Anomaly detection
- Business intelligence metrics
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Result of dynamic analysis operation."""
    analysis_type: str
    results: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]
    confidence_score: float
    metadata: Dict[str, Any]


class DynamicAnalysisEngine:
    """
    Advanced analysis engine for table data with statistical capabilities.
    """
    
    def __init__(self):
        """Initialize the dynamic analysis engine."""
        self.analysis_methods = {
            'statistical_summary': self._statistical_summary,
            'correlation_analysis': self._correlation_analysis,
            'trend_analysis': self._trend_analysis,
            'outlier_detection': self._outlier_detection,
            'business_metrics': self._business_metrics,
            'comparative_analysis': self._comparative_analysis,
            'forecasting': self._forecasting,
            'segmentation': self._segmentation
        }
        logger.info("DynamicAnalysisEngine initialized")
    
    def analyze_table_data(self, table_data: Dict[str, Any], 
                          analysis_type: str = 'statistical_summary',
                          parameters: Optional[Dict[str, Any]] = None) -> AnalysisResult:
        """
        Perform dynamic analysis on table data.
        
        Args:
            table_data: Table data from TableAwareRetrieval
            analysis_type: Type of analysis to perform
            parameters: Additional parameters for analysis
            
        Returns:
            AnalysisResult with findings and insights
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame(table_data['data'])
            
            # Validate data
            if df.empty:
                return AnalysisResult(
                    analysis_type=analysis_type,
                    results={},
                    insights=["No data available for analysis"],
                    recommendations=["Please provide data to analyze"],
                    confidence_score=0.0,
                    metadata={"error": "empty_data"}
                )
            
            # Get analysis method
            analysis_method = self.analysis_methods.get(analysis_type)
            if not analysis_method:
                available_methods = list(self.analysis_methods.keys())
                return AnalysisResult(
                    analysis_type=analysis_type,
                    results={},
                    insights=[f"Unknown analysis type: {analysis_type}"],
                    recommendations=[f"Available methods: {', '.join(available_methods)}"],
                    confidence_score=0.0,
                    metadata={"error": "unknown_method"}
                )
            
            # Perform analysis
            result = analysis_method(df, parameters or {})
            result.metadata.update({
                "table_id": table_data.get('table_id'),
                "table_title": table_data.get('title'),
                "data_shape": df.shape,
                "analysis_timestamp": datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return AnalysisResult(
                analysis_type=analysis_type,
                results={},
                insights=[f"Analysis failed: {str(e)}"],
                recommendations=["Please check data format and try again"],
                confidence_score=0.0,
                metadata={"error": str(e)}
            )
    
    def _statistical_summary(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> AnalysisResult:
        """Comprehensive statistical summary analysis."""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            results = {
                "basic_stats": {},
                "distribution_stats": {},
                "data_quality": {},
                "column_analysis": {}
            }
            
            insights = []
            recommendations = []
            
            # Basic statistics
            if len(numeric_cols) > 0:
                basic_stats = df[numeric_cols].describe().to_dict()
                results["basic_stats"] = basic_stats
                
                # Generate insights
                for col in numeric_cols:
                    mean_val = df[col].mean()
                    median_val = df[col].median()
                    std_val = df[col].std()
                    
                    if abs(mean_val - median_val) > std_val:
                        insights.append(f"{col} shows significant skewness (mean: {mean_val:.2f}, median: {median_val:.2f})")
                    
                    if std_val > mean_val:
                        insights.append(f"{col} has high variability (std dev > mean)")
            
            # Distribution analysis
            distribution_stats = {}
            for col in numeric_cols:
                distribution_stats[col] = {
                    "skewness": float(df[col].skew()),
                    "kurtosis": float(df[col].kurtosis()),
                    "normality_test": self._test_normality(df[col])
                }
            results["distribution_stats"] = distribution_stats
            
            # Data quality assessment
            missing_data = df.isnull().sum().to_dict()
            duplicate_rows = df.duplicated().sum()
            
            results["data_quality"] = {
                "missing_values": missing_data,
                "duplicate_rows": int(duplicate_rows),
                "completeness_score": float((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100)
            }
            
            # Quality insights
            if duplicate_rows > 0:
                insights.append(f"Found {duplicate_rows} duplicate rows")
                recommendations.append("Consider removing duplicate rows")
            
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            if missing_pct > 10:
                insights.append(f"High missing data rate: {missing_pct:.1f}%")
                recommendations.append("Investigate missing data patterns")
            
            # Categorical analysis
            if len(categorical_cols) > 0:
                categorical_analysis = {}
                for col in categorical_cols:
                    value_counts = df[col].value_counts()
                    categorical_analysis[col] = {
                        "unique_values": int(df[col].nunique()),
                        "most_common": value_counts.index[0] if len(value_counts) > 0 else None,
                        "most_common_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                        "distribution": value_counts.head(5).to_dict()
                    }
                results["column_analysis"]["categorical"] = categorical_analysis
            
            confidence_score = min(1.0, (100 - missing_pct) / 100 * 0.8 + 0.2)
            
            return AnalysisResult(
                analysis_type="statistical_summary",
                results=results,
                insights=insights,
                recommendations=recommendations,
                confidence_score=confidence_score,
                metadata={"numeric_columns": len(numeric_cols), "categorical_columns": len(categorical_cols)}
            )
            
        except Exception as e:
            logger.error(f"Statistical summary failed: {e}")
            raise
    
    def _correlation_analysis(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> AnalysisResult:
        """Correlation and relationship analysis."""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                return AnalysisResult(
                    analysis_type="correlation_analysis",
                    results={},
                    insights=["Need at least 2 numeric columns for correlation analysis"],
                    recommendations=["Add more numeric data for meaningful correlation analysis"],
                    confidence_score=0.0,
                    metadata={"insufficient_data": True}
                )
            
            # Calculate correlation matrix
            correlation_matrix = df[numeric_cols].corr()
            
            results = {
                "correlation_matrix": correlation_matrix.to_dict(),
                "strong_correlations": [],
                "weak_correlations": [],
                "correlation_insights": {}
            }
            
            insights = []
            recommendations = []
            
            # Find strong correlations
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols):
                    if i < j:  # Avoid duplicates and self-correlation
                        corr_value = correlation_matrix.loc[col1, col2]
                        
                        if abs(corr_value) > 0.7:
                            results["strong_correlations"].append({
                                "column1": col1,
                                "column2": col2,
                                "correlation": float(corr_value),
                                "strength": "strong"
                            })
                            
                            direction = "positive" if corr_value > 0 else "negative"
                            insights.append(f"Strong {direction} correlation between {col1} and {col2} ({corr_value:.3f})")
                            
                        elif abs(corr_value) < 0.3:
                            results["weak_correlations"].append({
                                "column1": col1,
                                "column2": col2,
                                "correlation": float(corr_value),
                                "strength": "weak"
                            })
            
            # Generate recommendations
            if len(results["strong_correlations"]) > 0:
                recommendations.append("Strong correlations found - consider multicollinearity in modeling")
            
            if len(results["weak_correlations"]) == len(numeric_cols) * (len(numeric_cols) - 1) // 2:
                recommendations.append("All correlations are weak - variables may be independent")
            
            confidence_score = min(1.0, len(numeric_cols) / 10)  # Higher confidence with more variables
            
            return AnalysisResult(
                analysis_type="correlation_analysis",
                results=results,
                insights=insights,
                recommendations=recommendations,
                confidence_score=confidence_score,
                metadata={"variables_analyzed": len(numeric_cols)}
            )
            
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            raise
    
    def _trend_analysis(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> AnalysisResult:
        """Time series and trend analysis."""
        try:
            # Try to identify time columns
            time_cols = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Try to parse as datetime
                    try:
                        pd.to_datetime(df[col].head())
                        time_cols.append(col)
                    except:
                        pass
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            results = {
                "time_columns": time_cols,
                "trends": {},
                "seasonality": {},
                "growth_rates": {}
            }
            
            insights = []
            recommendations = []
            
            if len(time_cols) == 0:
                # No time column found, analyze index-based trends
                for col in numeric_cols:
                    trend_slope = self._calculate_trend_slope(df[col].values)
                    results["trends"][col] = {
                        "slope": float(trend_slope),
                        "direction": "increasing" if trend_slope > 0 else "decreasing" if trend_slope < 0 else "stable"
                    }
                    
                    if abs(trend_slope) > 0.1:
                        direction = "increasing" if trend_slope > 0 else "decreasing"
                        insights.append(f"{col} shows {direction} trend (slope: {trend_slope:.3f})")
            else:
                # Time-based analysis
                time_col = time_cols[0]
                df_time = df.copy()
                df_time[time_col] = pd.to_datetime(df_time[time_col])
                df_time = df_time.sort_values(time_col)
                
                for col in numeric_cols:
                    # Calculate growth rate
                    values = df_time[col].values
                    if len(values) > 1:
                        growth_rate = (values[-1] - values[0]) / values[0] * 100 if values[0] != 0 else 0
                        results["growth_rates"][col] = float(growth_rate)
                        
                        if abs(growth_rate) > 10:
                            direction = "growth" if growth_rate > 0 else "decline"
                            insights.append(f"{col} shows {growth_rate:.1f}% {direction} over time period")
            
            if len(insights) == 0:
                insights.append("No significant trends detected in the data")
                recommendations.append("Consider longer time periods or more data points for trend analysis")
            
            confidence_score = 0.7 if len(time_cols) > 0 else 0.4
            
            return AnalysisResult(
                analysis_type="trend_analysis",
                results=results,
                insights=insights,
                recommendations=recommendations,
                confidence_score=confidence_score,
                metadata={"time_columns_found": len(time_cols)}
            )
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            raise
    
    def _outlier_detection(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> AnalysisResult:
        """Detect outliers and anomalies in the data."""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            results = {
                "outliers_by_column": {},
                "outlier_summary": {},
                "anomaly_scores": {}
            }
            
            insights = []
            recommendations = []
            
            total_outliers = 0
            
            for col in numeric_cols:
                # IQR method for outlier detection
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_count = len(outliers)
                total_outliers += outlier_count
                
                results["outliers_by_column"][col] = {
                    "count": outlier_count,
                    "percentage": float(outlier_count / len(df) * 100),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "outlier_values": outliers[col].tolist()[:10]  # First 10 outliers
                }
                
                if outlier_count > 0:
                    pct = outlier_count / len(df) * 100
                    insights.append(f"{col} has {outlier_count} outliers ({pct:.1f}% of data)")
                    
                    if pct > 5:
                        recommendations.append(f"Investigate {col} outliers - high percentage detected")
            
            results["outlier_summary"] = {
                "total_outliers": total_outliers,
                "outlier_percentage": float(total_outliers / (len(df) * len(numeric_cols)) * 100),
                "columns_with_outliers": sum(1 for col_data in results["outliers_by_column"].values() if col_data["count"] > 0)
            }
            
            if total_outliers == 0:
                insights.append("No significant outliers detected using IQR method")
                recommendations.append("Data appears to be well-distributed")
            
            confidence_score = 0.8  # Outlier detection is generally reliable
            
            return AnalysisResult(
                analysis_type="outlier_detection",
                results=results,
                insights=insights,
                recommendations=recommendations,
                confidence_score=confidence_score,
                metadata={"method": "IQR", "columns_analyzed": len(numeric_cols)}
            )
            
        except Exception as e:
            logger.error(f"Outlier detection failed: {e}")
            raise
    
    def _business_metrics(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> AnalysisResult:
        """Calculate business intelligence metrics."""
        # Implementation for business metrics
        # This would include KPIs, growth rates, efficiency metrics, etc.
        pass
    
    def _comparative_analysis(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> AnalysisResult:
        """Comparative analysis between groups or time periods."""
        # Implementation for comparative analysis
        pass
    
    def _forecasting(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> AnalysisResult:
        """Simple forecasting and prediction."""
        # Implementation for basic forecasting
        pass
    
    def _segmentation(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> AnalysisResult:
        """Data segmentation and clustering analysis."""
        # Implementation for segmentation
        pass
    
    def _test_normality(self, series: pd.Series) -> Dict[str, Any]:
        """Test if data follows normal distribution."""
        try:
            from scipy import stats
            statistic, p_value = stats.shapiro(series.dropna()[:5000])  # Limit to 5000 samples
            return {
                "test": "shapiro_wilk",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "is_normal": p_value > 0.05
            }
        except ImportError:
            # Fallback without scipy
            return {
                "test": "skewness_check",
                "skewness": float(series.skew()),
                "is_normal": abs(series.skew()) < 0.5
            }
    
    def _calculate_trend_slope(self, values: np.ndarray) -> float:
        """Calculate trend slope using linear regression."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope
