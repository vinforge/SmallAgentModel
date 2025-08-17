"""
Code Interpreter Tool for Agent Zero
====================================

SAM tool that enables Agent Zero to execute Python code securely
for data analysis, mathematical computation, and visualization.

Author: SAM Development Team
Version: 1.0.0
"""

import json
import logging
import requests
import base64
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Import SAM tool framework
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sam.agent_zero.planning.sam_tool_registry import SAMTool, ToolCategory

logger = logging.getLogger(__name__)


@dataclass
class CodeInterpreterResult:
    """Result from code interpreter execution."""
    success: bool
    output: str
    error: str = ""
    execution_time: float = 0.0
    generated_plots: List[str] = None  # List of plot filenames
    generated_data: Dict[str, Any] = None  # Structured data results
    
    def __post_init__(self):
        if self.generated_plots is None:
            self.generated_plots = []
        if self.generated_data is None:
            self.generated_data = {}


class CodeInterpreterTool(SAMTool):
    """
    Secure Code Interpreter tool for Agent Zero.
    
    Enables the agent to:
    - Perform mathematical calculations
    - Analyze data with pandas/numpy
    - Create visualizations with matplotlib/plotly
    - Process files and generate reports
    - Execute complex algorithms
    """
    
    def __init__(self, sandbox_service_url: str = "http://localhost:5000"):
        """
        Initialize the Code Interpreter tool.
        
        Args:
            sandbox_service_url: URL of the sandbox service
        """
        super().__init__(
            name="code_interpreter",
            category=ToolCategory.DATA_ANALYSIS,
            description="Execute Python code securely for data analysis, calculations, and visualizations",
            parameters=["code", "data_files", "timeout"],
            cost_estimate=3,
            context_requirements=[]
        )
        
        self.sandbox_url = sandbox_service_url
        self.logger = logging.getLogger(f"{__name__}.CodeInterpreterTool")
        
        # Verify sandbox service is available
        self.verify_sandbox_service()
    
    def verify_sandbox_service(self) -> bool:
        """Verify that the sandbox service is available."""
        try:
            response = requests.get(f"{self.sandbox_url}/health", timeout=5)
            if response.status_code == 200:
                self.logger.info("✅ Sandbox service is available")
                return True
            else:
                self.logger.warning(f"⚠️ Sandbox service returned status {response.status_code}")
                return False
        except requests.RequestException as e:
            self.logger.warning(f"⚠️ Sandbox service not available: {e}")
            return False
    
    def execute(self, code: str, data_files: Optional[Dict[str, str]] = None, 
                timeout: int = 30, **kwargs) -> CodeInterpreterResult:
        """
        Execute Python code in the secure sandbox.
        
        Args:
            code: Python code to execute
            data_files: Optional dictionary of filename -> file content
            timeout: Execution timeout in seconds
            **kwargs: Additional parameters
            
        Returns:
            CodeInterpreterResult with execution results
        """
        try:
            # Prepare request payload
            request_data = {
                "code": code,
                "timeout_seconds": timeout,
                "data_files": {}
            }
            
            # Encode data files if provided
            if data_files:
                for filename, content in data_files.items():
                    if isinstance(content, str):
                        # Assume text content, encode as UTF-8
                        encoded_content = base64.b64encode(content.encode('utf-8')).decode('utf-8')
                    else:
                        # Assume binary content, encode directly
                        encoded_content = base64.b64encode(content).decode('utf-8')
                    request_data["data_files"][filename] = encoded_content
            
            # Execute code via sandbox service
            response = requests.post(
                f"{self.sandbox_url}/execute",
                json=request_data,
                timeout=timeout + 10  # Add buffer for network overhead
            )
            
            if response.status_code == 200:
                result_data = response.json()
                
                # Parse generated files for plots and data
                generated_plots = []
                generated_data = {}
                
                for filename, encoded_content in result_data.get("generated_files", {}).items():
                    if filename.endswith(('.png', '.jpg', '.jpeg', '.svg', '.pdf')):
                        generated_plots.append(filename)
                    elif filename.endswith(('.json', '.csv', '.txt')):
                        # Decode and parse structured data
                        try:
                            content = base64.b64decode(encoded_content).decode('utf-8')
                            if filename.endswith('.json'):
                                generated_data[filename] = json.loads(content)
                            else:
                                generated_data[filename] = content
                        except Exception as e:
                            self.logger.warning(f"Failed to parse {filename}: {e}")
                
                return CodeInterpreterResult(
                    success=result_data.get("success", False),
                    output=result_data.get("stdout", ""),
                    error=result_data.get("stderr", ""),
                    execution_time=result_data.get("execution_time", 0.0),
                    generated_plots=generated_plots,
                    generated_data=generated_data
                )
            
            else:
                error_msg = f"Sandbox service error: {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg += f" - {error_data.get('error', 'Unknown error')}"
                except:
                    pass
                
                return CodeInterpreterResult(
                    success=False,
                    output="",
                    error=error_msg
                )
        
        except requests.Timeout:
            return CodeInterpreterResult(
                success=False,
                output="",
                error="Code execution timed out"
            )
        
        except Exception as e:
            self.logger.error(f"❌ Code execution failed: {e}")
            return CodeInterpreterResult(
                success=False,
                output="",
                error=f"Execution error: {str(e)}"
            )
    
    def format_result_for_agent(self, result: CodeInterpreterResult) -> str:
        """
        Format the execution result for Agent Zero consumption.
        
        Args:
            result: Code execution result
            
        Returns:
            Formatted string for the agent
        """
        if not result.success:
            return f"❌ Code execution failed: {result.error}"
        
        output_parts = []
        
        # Add stdout output
        if result.output.strip():
            output_parts.append(f"📋 **Output:**\n```\n{result.output.strip()}\n```")
        
        # Add generated plots
        if result.generated_plots:
            plots_list = ", ".join(result.generated_plots)
            output_parts.append(f"📊 **Generated Visualizations:** {plots_list}")
        
        # Add structured data
        if result.generated_data:
            data_summary = []
            for filename, data in result.generated_data.items():
                if isinstance(data, dict):
                    data_summary.append(f"{filename} (JSON with {len(data)} keys)")
                elif isinstance(data, str):
                    lines = len(data.split('\n'))
                    data_summary.append(f"{filename} ({lines} lines)")
                else:
                    data_summary.append(filename)
            
            output_parts.append(f"📁 **Generated Data Files:** {', '.join(data_summary)}")
        
        # Add execution time
        output_parts.append(f"⏱️ **Execution Time:** {result.execution_time:.2f} seconds")
        
        # Add any errors as warnings
        if result.error.strip():
            output_parts.append(f"⚠️ **Warnings:**\n```\n{result.error.strip()}\n```")
        
        return "\n\n".join(output_parts)
    
    def get_usage_examples(self) -> List[str]:
        """Get usage examples for the agent."""
        return [
            "Calculate the mean and standard deviation of a dataset",
            "Create a scatter plot showing correlation between two variables", 
            "Analyze CSV data and generate summary statistics",
            "Solve mathematical equations or optimization problems",
            "Process text data and extract insights",
            "Generate random data for testing or simulation"
        ]
    
    def get_code_templates(self) -> Dict[str, str]:
        """Get common code templates for the agent."""
        return {
            "data_analysis": """
import pandas as pd
import numpy as np

# Load and analyze data
data = pd.read_csv('data.csv')
print("Dataset shape:", data.shape)
print("\\nSummary statistics:")
print(data.describe())
""",
            
            "visualization": """
import matplotlib.pyplot as plt
import numpy as np

# Create sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sine Wave')
plt.legend()
plt.grid(True)
plt.savefig('plot.png', dpi=150, bbox_inches='tight')
plt.show()
""",
            
            "calculation": """
import numpy as np
from scipy import stats

# Perform calculations
data = np.random.normal(100, 15, 1000)
mean = np.mean(data)
std = np.std(data)
median = np.median(data)

print(f"Mean: {mean:.2f}")
print(f"Standard Deviation: {std:.2f}")
print(f"Median: {median:.2f}")

# Statistical test
t_stat, p_value = stats.ttest_1samp(data, 100)
print(f"\\nT-test against population mean 100:")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
"""
        }


def register_code_interpreter_tool(registry, sandbox_url: str = "http://localhost:5000"):
    """
    Register the Code Interpreter tool with SAM's tool registry.
    
    Args:
        registry: SAM tool registry instance
        sandbox_url: URL of the sandbox service
    """
    tool = CodeInterpreterTool(sandbox_url)
    registry.register_tool(tool)
    logger.info("✅ Code Interpreter tool registered with Agent Zero")


# Example usage for testing
if __name__ == "__main__":
    # Test the tool
    tool = CodeInterpreterTool()
    
    test_code = """
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

# Create plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Sine Wave Example')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('sine_wave.png', dpi=150, bbox_inches='tight')

# Print some statistics
print(f"Maximum value: {np.max(y):.3f}")
print(f"Minimum value: {np.min(y):.3f}")
print(f"Mean value: {np.mean(y):.3f}")
"""
    
    result = tool.execute(test_code)
    formatted_result = tool.format_result_for_agent(result)
    print("Code Interpreter Result:")
    print(formatted_result)
