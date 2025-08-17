"""
Code Interpreter Integration with Agent Zero
============================================

Integration script to register the Code Interpreter tool with Agent Zero
and enhance the agent's prompts for code execution capabilities.

Author: SAM Development Team
Version: 1.0.0
"""

import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def integrate_code_interpreter_with_agent_zero(sandbox_url: str = "http://localhost:5000") -> bool:
    """
    Integrate Code Interpreter tool with Agent Zero.
    
    Args:
        sandbox_url: URL of the sandbox service
        
    Returns:
        True if integration successful, False otherwise
    """
    try:
        # Import required modules
        from sam.agent_zero.planning.sam_tool_registry import get_sam_tool_registry
        from sam.code_interpreter.code_interpreter_tool import CodeInterpreterTool
        
        # Get the tool registry
        registry = get_sam_tool_registry()
        
        # Create and register the Code Interpreter tool
        code_tool = CodeInterpreterTool(sandbox_url)
        registry._register_tool(code_tool)
        
        logger.info("✅ Code Interpreter tool integrated with Agent Zero")
        
        # Update Agent Zero's system prompts
        update_agent_prompts()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to integrate Code Interpreter: {e}")
        return False


def update_agent_prompts():
    """Update Agent Zero's system prompts to include Code Interpreter usage."""
    try:
        # This would typically update the agent's system prompts
        # For now, we'll create a prompt enhancement file
        
        prompt_enhancement = """
# Code Interpreter Tool Usage for Agent Zero

## When to Use Code Interpreter
Use the Code Interpreter tool when you need to:
- Perform mathematical calculations or statistical analysis
- Analyze data from CSV files or datasets
- Create visualizations (plots, charts, graphs)
- Process or transform data
- Solve computational problems
- Generate reports with calculations

## How to Use Code Interpreter
1. Formulate your logic in Python code
2. Use the code_interpreter tool with your Python code
3. Include any necessary data files
4. Interpret the results and present them to the user

## Example Usage Patterns

### Data Analysis
```python
import pandas as pd
import numpy as np

# Load and analyze data
data = pd.read_csv('data.csv')
print("Dataset Summary:")
print(data.describe())

# Calculate correlations
correlations = data.corr()
print("\\nCorrelations:")
print(correlations)
```

### Visualization
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Create visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='column1', y='column2')
plt.title('Relationship between Column1 and Column2')
plt.savefig('analysis_plot.png', dpi=150, bbox_inches='tight')
```

### Mathematical Computation
```python
import numpy as np
from scipy import optimize

# Solve optimization problem
def objective(x):
    return x**2 + 10*np.sin(x)

result = optimize.minimize_scalar(objective)
print(f"Minimum found at x = {result.x:.4f}")
print(f"Minimum value = {result.fun:.4f}")
```

## Best Practices
1. Always include necessary imports
2. Use descriptive variable names
3. Add print statements to show results
4. Save plots with descriptive filenames
5. Handle potential errors gracefully
6. Keep code focused and modular

## Security Notes
- Code runs in isolated Docker containers
- No network access from code execution
- Limited to approved Python packages
- Execution time limits enforced
- Memory and CPU limits applied
"""
        
        # Save prompt enhancement
        prompt_file = Path(__file__).parent / "agent_prompt_enhancement.md"
        prompt_file.write_text(prompt_enhancement)
        
        logger.info("✅ Agent Zero prompts updated for Code Interpreter")
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to update agent prompts: {e}")


def start_sandbox_service(port: int = 5000, background: bool = True) -> Optional[int]:
    """
    Start the sandbox service.
    
    Args:
        port: Port to run the service on
        background: Whether to run in background
        
    Returns:
        Process ID if background=True, None otherwise
    """
    try:
        from sam.code_interpreter.sandbox_service import create_sandbox_service
        
        if background:
            import subprocess
            import sys
            
            # Start service in background
            script_path = Path(__file__).parent / "sandbox_service.py"
            process = subprocess.Popen([
                sys.executable, str(script_path)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            logger.info(f"🚀 Sandbox service started in background (PID: {process.pid})")
            return process.pid
        else:
            # Start service in foreground
            service = create_sandbox_service()
            service.run(port=port)
            return None
            
    except Exception as e:
        logger.error(f"❌ Failed to start sandbox service: {e}")
        return None


def verify_integration() -> bool:
    """
    Verify that the Code Interpreter integration is working.
    
    Returns:
        True if integration is working, False otherwise
    """
    try:
        # Test sandbox service
        import requests
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code != 200:
            logger.error("❌ Sandbox service not responding")
            return False
        
        # Test tool registration
        from sam.agent_zero.planning.sam_tool_registry import get_sam_tool_registry
        registry = get_sam_tool_registry()
        
        tool = registry.get_tool("code_interpreter")
        if tool is None:
            logger.error("❌ Code Interpreter tool not registered")
            return False
        
        # Test code execution
        from sam.code_interpreter.code_interpreter_tool import CodeInterpreterTool
        code_tool = CodeInterpreterTool()
        
        test_code = "print('Hello from Code Interpreter!')"
        result = code_tool.execute(test_code)
        
        if not result.success:
            logger.error(f"❌ Code execution test failed: {result.error}")
            return False
        
        logger.info("✅ Code Interpreter integration verified successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration verification failed: {e}")
        return False


def setup_code_interpreter_complete():
    """Complete setup of Code Interpreter for SAM."""
    logger.info("🚀 Setting up Code Interpreter for SAM...")
    
    # Step 1: Start sandbox service
    logger.info("📦 Starting sandbox service...")
    pid = start_sandbox_service(background=True)
    
    if pid:
        # Give service time to start
        import time
        time.sleep(3)
        
        # Step 2: Integrate with Agent Zero
        logger.info("🔧 Integrating with Agent Zero...")
        if integrate_code_interpreter_with_agent_zero():
            
            # Step 3: Verify integration
            logger.info("✅ Verifying integration...")
            if verify_integration():
                logger.info("🎉 Code Interpreter setup complete!")
                logger.info("Agent Zero can now execute Python code securely.")
                return True
    
    logger.error("❌ Code Interpreter setup failed")
    return False


if __name__ == "__main__":
    setup_code_interpreter_complete()
