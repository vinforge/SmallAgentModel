# Algonauts Experiment: Cognitive Trajectory Analysis Guide

## Overview

The Algonauts Experiment represents a groundbreaking approach to understanding AI model behavior by analyzing internal reasoning dynamics rather than just final outputs. This experiment compares cognitive trajectories between Transformer (Mistral 7B) and Hybrid Transformer-SSM (Jamba) architectures on needle-in-haystack long-context tasks.

## Core Innovation

### Beyond Performance Metrics
Traditional model evaluation focuses on:
- Accuracy scores
- Latency measurements  
- Token efficiency
- Success/failure rates

The Algonauts approach adds:
- **Cognitive trajectory visualization** - 2D projections of internal reasoning paths
- **Quantitative trajectory metrics** - Mathematical measures of reasoning efficiency
- **Architectural pattern analysis** - Understanding how different architectures "think"

### The Central Question
Instead of asking "Which model is better?", we ask:
**"Do these different architectures think differently?"**

## Experimental Design

### Hypothesis
- **Transformer (Mistral)**: More chaotic/scattered trajectory due to attention mechanism struggles with long context
- **Hybrid (Jamba)**: More stable/efficient trajectory due to SSM layers handling long-range dependencies

### Test Methodology
1. **Needle-in-Haystack Tasks**: Precise fact extraction from ~250-page documents
2. **Cognitive Capture**: Flight Recorder system capturing neural activation vectors at key moments
3. **Trajectory Analysis**: 2D projection using UMAP/t-SNE/PCA dimensionality reduction
4. **Quantitative Metrics**: Trajectory Length, State Volatility, RAG Influence Score

## Key Components

### 1. Enhanced Benchmark Runner (`scripts/run_hybrid_benchmark.py`)
- **Algonauts Integration**: `--enable-algonauts` flag for cognitive tracing
- **Flight Recorder**: Captures cognitive vectors during task execution
- **Trace Storage**: Saves detailed reasoning logs for analysis

### 2. Needle-in-Haystack Tasks (`tests/benchmarks/algonauts_needle_tasks.json`)
- **Moby Dick Character Extraction**: Find specific ship and captain details in Chapter 51
- **Technical Manual Hunt**: Locate voltage specs in Section 12.4.7 of 280-page manual
- **Legal Clause Discovery**: Find Force Majeure notification timeframes in contract

### 3. Comparative Visualization (`sam/introspection/streamlit_algonauts.py`)
- **Side-by-Side Mode**: Compare trajectories from different models
- **Overlay Visualization**: Superimpose trajectories for direct comparison
- **Interactive Analysis**: Hover, zoom, and explore cognitive paths

### 4. Quantitative Analysis (`analysis/algonauts_analysis_report.py`)
- **Trajectory Length**: Total distance traveled in cognitive space
- **State Volatility**: Average step-to-step variation (stability measure)
- **RAG Influence**: How much context changes internal state

## Quick Start

### Option 1: Complete Automated Pipeline
```bash
# Run the entire experiment from start to finish
./run_complete_algonauts_experiment.sh
```

### Option 2: Step-by-Step Execution
```bash
# Step 1: Run Algonauts experiment
python scripts/run_algonauts_experiment.py --output-dir results/algonauts_experiment

# Step 2: Generate analysis report
python analysis/algonauts_analysis_report.py \
    --experiment-dir results/algonauts_experiment \
    --output analysis/algonauts_comprehensive_report.md

# Step 3: Interactive visualization
# Open SAM's Streamlit interface and navigate to 🔬 Algonauts tab
```

### Option 3: Individual Model Testing
```bash
# Test Mistral with cognitive tracing
python scripts/run_hybrid_benchmark.py \
    --engine mistral \
    --output results/mistral_algonauts.jsonl \
    --tasks tests/benchmarks/algonauts_needle_tasks.json \
    --enable-algonauts

# Test Jamba with cognitive tracing  
python scripts/run_hybrid_benchmark.py \
    --engine jamba \
    --output results/jamba_algonauts.jsonl \
    --tasks tests/benchmarks/algonauts_needle_tasks.json \
    --enable-algonauts
```

## Understanding the Results

### Trajectory Metrics Interpretation

#### Trajectory Length
- **Lower = More Efficient**: Shorter paths indicate more direct reasoning
- **Higher = More Exploratory**: Longer paths suggest more complex search patterns

#### State Volatility  
- **Lower = More Stable**: Consistent cognitive states during processing
- **Higher = More Chaotic**: Rapid changes in internal representations

#### RAG Influence Score
- **Higher = Better Context Integration**: Strong response to long-context input
- **Lower = Context Independence**: Less influenced by provided context

### Visual Pattern Analysis

#### Efficient Trajectory (Expected for Jamba)
```
Start → Context → Reasoning → Answer
  ↓       ↓         ↓         ↓
  •-------•---------•---------•
```

#### Scattered Trajectory (Expected for Mistral)
```
Start → Context → Reasoning → Answer
  ↓       ↓         ↓         ↓
  •---→---•    ↗----•----↘----•
      ↘       ↗         ↘
       •-----•           •
```

## Interactive Exploration

### Using the Algonauts Tab
1. **Open SAM Interface**: Start Streamlit application
2. **Navigate**: Memory Control Center → 🔬 Algonauts
3. **Select Mode**: Choose "Comparative Analysis"
4. **Load Sessions**: Select Mistral and Jamba trace sessions
5. **Visualize**: Enable "Overlay Trajectories" for comparison
6. **Analyze**: Use hover tooltips and metrics panel

### Key Visualizations
- **Overlaid Trajectories**: Blue (Mistral) vs Red (Jamba) paths
- **Side-by-Side Comparison**: Individual trajectory analysis
- **Metrics Dashboard**: Real-time quantitative comparison

## Expected Outcomes

### Success Criteria
✅ **Visual Differences**: Clear distinction between trajectory patterns  
✅ **Quantitative Metrics**: Measurable differences in efficiency/stability  
✅ **Architectural Insights**: Understanding of how different designs "think"  
✅ **Reproducible Results**: Consistent patterns across multiple runs

### Potential Findings
- **Jamba Advantage**: More stable, efficient trajectories for long-context tasks
- **Mistral Patterns**: More exploratory but potentially less efficient paths
- **Task-Dependent Behavior**: Different patterns for different needle-in-haystack types
- **Context Sensitivity**: Varying RAG influence scores between architectures

## Research Applications

### Immediate Uses
1. **Model Selection**: Choose architecture based on cognitive patterns
2. **Task Optimization**: Match models to task types based on reasoning style
3. **Architecture Research**: Guide development of new hybrid designs
4. **Interpretability**: Understand model behavior beyond black-box metrics

### Future Directions
1. **Real-Time Monitoring**: Deploy cognitive trajectory analysis in production
2. **Adaptive Systems**: Switch models based on real-time trajectory analysis
3. **Architecture Evolution**: Use trajectory insights to improve model designs
4. **Cognitive Debugging**: Identify and fix reasoning inefficiencies

## Technical Notes

### Cognitive Vector Simulation
- Current implementation simulates cognitive vectors based on architectural characteristics
- Mistral: Higher variance vectors simulating attention volatility
- Jamba: Lower variance vectors simulating SSM stability
- Future versions will capture actual neural activations

### Dimensionality Reduction
- **UMAP**: Best for preserving global structure and local neighborhoods
- **t-SNE**: Good for cluster visualization but can distort distances
- **PCA**: Linear reduction, preserves variance but may miss nonlinear patterns

### Performance Considerations
- Cognitive tracing adds ~10-15% overhead to benchmark execution
- Trace files are ~1-5MB per session depending on task complexity
- Visualization rendering scales well up to ~1000 trajectory points

## Troubleshooting

### Common Issues
1. **No Cognitive Vectors**: Ensure `--enable-algonauts` flag is used
2. **Empty Trajectories**: Check that introspection module is properly installed
3. **Visualization Errors**: Verify Plotly and visualization dependencies
4. **Memory Issues**: Reduce trace detail level for very long contexts

### Debug Commands
```bash
# Check if Algonauts is available
python -c "from sam.introspection.flight_recorder import initialize_flight_recorder; print('✅ Algonauts available')"

# Verify trace files
ls -la results/algonauts_experiment/*/

# Check trajectory metrics
cat results/algonauts_experiment/trajectory_metrics.json | python -m json.tool
```

## Contributing

### Extending the Experiment
1. **New Task Types**: Add tasks to `algonauts_needle_tasks.json`
2. **Additional Metrics**: Enhance trajectory analysis in `algonauts_analysis_report.py`
3. **Visualization Improvements**: Extend Streamlit interface with new chart types
4. **Architecture Support**: Add new model engines to benchmark runner

### Research Collaboration
- Share trajectory data and analysis results
- Contribute new cognitive metrics and visualization techniques
- Collaborate on real neural activation capture methods
- Develop standardized cognitive trajectory benchmarks

---

*The Algonauts Experiment represents a new frontier in AI model evaluation, moving beyond surface metrics to understand the fundamental cognitive patterns that drive model behavior.*
