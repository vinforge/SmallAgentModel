# SAM Model Foundry - Quick Reference Guide

**One-page reference for the SAM Model Foundry & Evaluation Suite**

## 🚀 **Quick Commands**

### **Complete Evaluation Pipeline**
```bash
# 1. Run evaluation
python scripts/run_model_evaluation.py --models transformer,llama31-8b

# 2. Score results  
python scripts/score_evaluation_results.py

# 3. Generate leaderboard
python scripts/generate_leaderboard_report.py
```

### **Common Evaluation Scenarios**
```bash
# Compare all available models
python scripts/run_model_evaluation.py --models all

# Test specific categories
python scripts/run_model_evaluation.py --models transformer --categories qa,reasoning

# Quick QA-only evaluation
python scripts/run_model_evaluation.py --models transformer --categories qa --output qa_test

# Custom output filename
python scripts/run_model_evaluation.py --models hybrid --output my_hybrid_test
```

### **Scoring Options**
```bash
# Score with GPT-4 (requires API key)
python scripts/score_evaluation_results.py --judge gpt-4

# Score with Claude (requires API key)  
python scripts/score_evaluation_results.py --judge claude-3-opus

# Use mock scoring (no API key needed)
python scripts/score_evaluation_results.py

# Score specific file
python scripts/score_evaluation_results.py evaluation_results/my_test.jsonl
```

### **Report Generation**
```bash
# Generate default leaderboard
python scripts/generate_leaderboard_report.py

# Custom output filename
python scripts/generate_leaderboard_report.py --output MY_LEADERBOARD.md

# Process specific scored file
python scripts/generate_leaderboard_report.py scored_results.jsonl
```

## 📊 **Available Models**

| Model | Type | Context | Status |
|-------|------|---------|--------|
| `transformer` | Built-in | 16K | ✅ Ready |
| `hybrid` | Built-in | 100K | ✅ Ready |
| `llama31-8b` | Dynamic | 128K | ⚠️ Requires setup |

## 📋 **Benchmark Categories**

| Category | Prompts | Focus Area |
|----------|---------|------------|
| `qa` | 5 | Question answering |
| `reasoning` | 4 | Logical problem solving |
| `code_gen` | 4 | Programming tasks |
| `summarization` | 2 | Content distillation |
| `tool_use` | 4 | Function calling |
| `long_context_recall` | 1 | Information retrieval |
| `safety_refusal` | 4 | Harmful request refusal |
| `creativity` | 2 | Original content |
| `analysis` | 2 | Analytical reasoning |
| `instruction_following` | 2 | Format compliance |

## 🎯 **Scoring Criteria**

### **Core Metrics (1-5 scale)**
- **Correctness (40%)**: Factual accuracy
- **Completeness (30%)**: Thoroughness  
- **Clarity (20%)**: Structure and understanding
- **Conciseness (10%)**: Efficiency

### **Category-Specific**
- **Code Quality**: Programming tasks
- **Tool Identification**: Function calling
- **Appropriate Refusal**: Safety scenarios
- **Format Compliance**: Instruction following

## 🔧 **File Locations**

### **Input Files**
- **Benchmarks**: `sam/benchmarks/core_benchmark_v1.jsonl`
- **Configuration**: `sam/benchmarks/benchmark_config.py`
- **Model Wrappers**: `sam/models/wrappers/`

### **Output Files**
- **Evaluation Results**: `evaluation_results/run_TIMESTAMP.jsonl`
- **Scored Results**: `evaluation_results/scored_run_TIMESTAMP.jsonl`
- **Leaderboard**: `MODEL_LEADERBOARD.md`

## 🛠️ **Troubleshooting**

### **Common Issues**

**Model Not Found**
```bash
# Check available models
python -c "from sam.models.wrappers import get_available_models; print(get_available_models())"
```

**Evaluation Timeout**
```bash
# Increase timeout
python scripts/run_model_evaluation.py --models transformer --timeout 600
```

**Missing Dependencies**
```bash
# Install Model Foundry dependencies
pip install transformers torch bitsandbytes sentencepiece protobuf
```

**API Key Issues**
```bash
# Set environment variables
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"

# Or use mock scoring
python scripts/score_evaluation_results.py  # Uses mock by default
```

### **Debug Mode**
Add `--debug` to any script for detailed logging:
```bash
python scripts/run_model_evaluation.py --models transformer --debug
python scripts/score_evaluation_results.py --debug
python scripts/generate_leaderboard_report.py --debug
```

## 📈 **Performance Tips**

### **Speed Optimization**
- Use `--categories qa` for quick tests
- Test with single model first: `--models transformer`
- Use CPU for small models: `device: "cpu"` in config

### **Memory Management**
- Enable quantization: `load_in_4bit: true`
- Unload models between evaluations
- Monitor with `get_memory_usage()`

### **Cost Optimization**
- Use mock scoring for development
- Test with subset of categories first
- Use local models when possible

## 🧪 **Testing Commands**

### **Validate Setup**
```bash
# Test Model Foundry framework
python scripts/test_model_foundry.py

# Test specific model wrapper
python -c "
from sam.models.wrappers import create_model_wrapper
wrapper = create_model_wrapper('transformer', {})
print('Health:', wrapper.health_check())
"
```

### **Quick Validation**
```bash
# Single prompt test
python scripts/run_model_evaluation.py --models transformer --categories qa --output quick_test

# Check results
head -5 evaluation_results/quick_test.jsonl
```

## 📊 **Sample Output**

### **Evaluation Results**
```json
{
  "prompt_id": "qa_001",
  "model_name": "transformer", 
  "response": "Paris is the capital of France.",
  "performance": {
    "inference_time": 2.1,
    "completion_tokens": 8,
    "tokens_per_second": 3.8
  },
  "success": true
}
```

### **Scored Results**
```json
{
  "scoring": {
    "weighted_score": 4.8,
    "scores": {
      "correctness": 5,
      "completeness": 5, 
      "clarity": 4,
      "conciseness": 5
    }
  }
}
```

### **Leaderboard Summary**
```markdown
| Rank | Model | Avg Score | Success Rate | Tokens/sec |
|------|-------|-----------|--------------|------------|
| 1 | 🥇 transformer | 4.77/5.0 | 77.8% | 21.1 |
```

## 🔄 **Workflow Examples**

### **New Model Integration**
```bash
# 1. Copy template
cp sam/models/wrappers/template.py sam/models/wrappers/my_model.py

# 2. Implement methods (edit my_model.py)

# 3. Test integration
python scripts/test_model_foundry.py

# 4. Run evaluation
python scripts/run_model_evaluation.py --models my-model

# 5. Generate report
python scripts/score_evaluation_results.py
python scripts/generate_leaderboard_report.py
```

### **Model Comparison**
```bash
# 1. Evaluate multiple models
python scripts/run_model_evaluation.py --models transformer,hybrid,llama31-8b

# 2. Score results
python scripts/score_evaluation_results.py --judge gpt-4

# 3. Generate comparison report
python scripts/generate_leaderboard_report.py --output COMPARISON_REPORT.md
```

### **Category-Specific Analysis**
```bash
# 1. Test reasoning capabilities
python scripts/run_model_evaluation.py --models all --categories reasoning

# 2. Score and analyze
python scripts/score_evaluation_results.py
python scripts/generate_leaderboard_report.py --output REASONING_ANALYSIS.md
```

## 📚 **Additional Resources**

- **Full Documentation**: `sam/models/README.md`
- **Template Guide**: `sam/models/wrappers/template.py`
- **Configuration Reference**: `sam/benchmarks/benchmark_config.py`
- **Integration Tests**: `scripts/test_model_foundry.py`

## 🆘 **Getting Help**

### **Check Status**
```bash
# Model Foundry status
python scripts/test_model_foundry.py

# Available models
python -c "from sam.models.wrappers import get_available_models; print(get_available_models())"

# Benchmark categories
python -c "from sam.benchmarks import BenchmarkLoader; loader = BenchmarkLoader(); loader.load_benchmarks(); print(loader.get_category_stats())"
```

### **Common Solutions**
1. **Import Errors**: Check Python path and dependencies
2. **Model Loading**: Verify model wrapper implementation
3. **Evaluation Failures**: Check timeout and memory settings
4. **Scoring Issues**: Verify API keys or use mock mode
5. **Report Generation**: Ensure scored results file exists

---

**🚀 Ready to evaluate? Start with:**
```bash
python scripts/run_model_evaluation.py --models transformer --categories qa
```
