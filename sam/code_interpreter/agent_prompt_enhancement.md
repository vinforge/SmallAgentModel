
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
print("\nCorrelations:")
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
