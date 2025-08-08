# SAM Model Foundry & Evaluation Suite

**⚠️ ADVANCED FEATURE - For AI Researchers, Model Developers, and Enterprise Teams**

**Regular SAM users:** If you just want to chat with SAM offline, you don't need this! SAM works great out of the box. This system is for advanced users who need to evaluate and compare multiple AI models.

**The world's first comprehensive AI model evaluation platform built into an AI assistant.**

Transform SAM from a "fixed model" system into a **"model-agnostic platform"** capable of rapidly integrating, testing, and comparing any AI model with data-driven precision.

## 🎯 **Overview**

The SAM Model Foundry enables:
- **🔄 Rapid Model Integration**: Add any AI model in <1 day
- **📊 Objective Evaluation**: 30-prompt comprehensive benchmark suite
- **🎯 Data-Driven Decisions**: LLM-as-a-Judge scoring with structured rubrics
- **💰 Cost Optimization**: Performance-per-dollar analysis
- **🏆 Model Leaderboards**: Comprehensive comparison reports

## 🚀 **Quick Start**

### **1. Evaluate Existing Models**
```bash
# Compare transformer and hybrid models
python scripts/run_model_evaluation.py --models transformer,hybrid

# Score results with GPT-4 judge
python scripts/score_evaluation_results.py --judge gpt-4

# Generate leaderboard report
python scripts/generate_leaderboard_report.py
```

### **2. Add a New Model**
```bash
# Copy the template
cp sam/models/wrappers/template.py sam/models/wrappers/my_model_wrapper.py

# Implement the required methods (see template for details)
# Test the integration
python scripts/test_model_foundry.py

# Evaluate performance
python scripts/run_model_evaluation.py --models my-model
```

## 📁 **Architecture**

```
sam/models/
├── wrappers/                    # Model integration layer
│   ├── template.py             # Base template for new models
│   ├── llama31_wrapper.py      # Llama-3.1-8B implementation
│   └── __init__.py             # Dynamic model discovery
├── sam_hybrid_model.py         # HGRN-2 hybrid architecture
├── hybrid_config.py            # Hybrid model configuration
└── __init__.py                 # Model package exports

sam/benchmarks/
├── core_benchmark_v1.jsonl     # 30-prompt test suite
├── benchmark_config.py         # Scoring framework
└── __init__.py                 # Benchmark utilities

scripts/
├── run_model_evaluation.py     # Automated benchmark runner
├── score_evaluation_results.py # LLM-as-a-Judge scorer
├── generate_leaderboard_report.py # Report generator
└── test_model_foundry.py       # Integration tests
```

## 🧪 **Benchmark Categories**

| Category | Prompts | Description |
|----------|---------|-------------|
| **📝 QA** | 5 | Factual accuracy and comprehension |
| **🧠 Reasoning** | 4 | Multi-step logical problem solving |
| **💻 Code Generation** | 4 | Programming task completion |
| **📊 Summarization** | 2 | Content distillation and clarity |
| **🔧 Tool Use** | 4 | Function calling and parameter extraction |
| **📚 Long Context** | 1 | Needle-in-haystack information retrieval |
| **🛡️ Safety** | 4 | Appropriate refusal of harmful requests |
| **🎨 Creativity** | 2 | Original content generation |
| **📈 Analysis** | 2 | Comparative and analytical reasoning |
| **📋 Instructions** | 2 | Format and constraint adherence |

## 🎯 **Scoring System**

### **Core Criteria (1-5 scale)**
- **Correctness (40%)**: Factual accuracy and correctness
- **Completeness (30%)**: Thoroughness in addressing all aspects
- **Clarity (20%)**: Structure and understandability
- **Conciseness (10%)**: Efficiency without verbosity

### **Category-Specific Criteria**
- **Code Quality**: For programming tasks
- **Tool Identification**: For function calling
- **Appropriate Refusal**: For safety scenarios
- **Format Compliance**: For instruction following

### **Judge Models Supported**
- **GPT-4 / GPT-4-Turbo**: OpenAI's flagship models
- **Claude-3-Opus / Claude-3-Sonnet**: Anthropic's advanced models
- **Mock Scoring**: For demonstration without API keys

## 🔧 **Model Integration Guide**

### **1. Create Model Wrapper**

Copy the template and implement required methods:

```python
from sam.models.wrappers.template import ModelWrapperTemplate

class MyModelWrapper(ModelWrapperTemplate):
    MODEL_NAME = "my-model"
    
    def load_model(self) -> bool:
        # Load your model here
        pass
    
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        # Generate response using your model
        pass
    
    def get_model_metadata(self) -> ModelMetadata:
        # Return model information
        pass
    
    # ... implement other required methods
```

### **2. Model Metadata**

Provide comprehensive model information:

```python
def get_model_metadata(self) -> ModelMetadata:
    return ModelMetadata(
        name="My Amazing Model",
        version="1.0",
        parameters="7B",
        context_window=32000,
        license="Apache-2.0",
        provider="MyCompany",
        model_family="MyFamily",
        architecture="Transformer"
    )
```

### **3. Capabilities Declaration**

Specify what your model can do:

```python
def get_model_capabilities(self) -> ModelCapabilities:
    return ModelCapabilities(
        supports_function_calling=True,
        supports_code_generation=True,
        supports_reasoning=True,
        supports_multimodal=False,
        max_output_tokens=4096,
        languages_supported=["en", "es", "fr"]
    )
```

## 📊 **Evaluation Pipeline**

### **1. Run Evaluation**
```bash
# Evaluate specific models and categories
python scripts/run_model_evaluation.py \
    --models transformer,my-model \
    --categories qa,reasoning,code_gen \
    --output my_evaluation

# Results saved to: evaluation_results/my_evaluation.jsonl
```

### **2. Score Results**
```bash
# Score with GPT-4 judge
python scripts/score_evaluation_results.py \
    evaluation_results/my_evaluation.jsonl \
    --judge gpt-4 \
    --output scored_results.jsonl

# Results include weighted scores and detailed assessments
```

### **3. Generate Report**
```bash
# Create comprehensive leaderboard
python scripts/generate_leaderboard_report.py \
    scored_results.jsonl \
    --output MY_MODEL_LEADERBOARD.md

# Report includes rankings, analysis, and recommendations
```

## 🏆 **Sample Leaderboard Output**

```markdown
# SAM Model Leaderboard Report

## 🏆 Overall Model Leaderboard

| Rank | Model | Avg Score | Success Rate | Avg Time (s) | Tokens/sec |
|------|-------|-----------|--------------|--------------|------------|
| 1 | 🥇 my-model | 4.85/5.0 | 98.5% | 12.3 | 45.2 |
| 2 | 🥈 transformer | 4.77/5.0 | 77.8% | 21.1 | 21.1 |

## 💰 Cost-Benefit Analysis

| Model | Quality Score | Cost/1K Tokens | Quality/Dollar | Recommendation |
|-------|---------------|-----------------|----------------|----------------|
| my-model | 4.85/5.0 | $0.001 | 4850 | 🌟 Excellent Value |
| transformer | 4.77/5.0 | $0.002 | 2385 | ✅ Recommended |
```

## 🔍 **Advanced Features**

### **Dynamic Model Discovery**
Models are automatically discovered from the `wrappers/` directory:

```python
from sam.models.wrappers import get_available_models, create_model_wrapper

# Get all available models
models = get_available_models()
print(models)  # ['transformer', 'hybrid', 'llama31-8b', 'my-model']

# Create model instance
wrapper = create_model_wrapper('my-model', config)
```

### **Custom Benchmarks**
Extend the benchmark suite with custom prompts:

```json
{"id": "custom_001", "category": "custom", "prompt": "Your custom prompt", "expected_type": "custom", "max_tokens": 200, "scoring_criteria": ["correctness", "creativity"]}
```

### **Batch Evaluation**
Evaluate multiple model combinations efficiently:

```bash
# Evaluate all models across all categories
python scripts/run_model_evaluation.py --models all

# Evaluate specific combinations
python scripts/run_model_evaluation.py \
    --models "transformer,hybrid,llama31-8b" \
    --categories "qa,reasoning,code_gen"
```

## 🛠️ **Configuration**

### **Model Configuration**
```python
config = {
    "model_name": "my-model",
    "device": "cuda",
    "load_in_4bit": True,
    "temperature": 0.7,
    "max_tokens": 1000,
    "timeout_seconds": 120
}
```

### **Benchmark Configuration**
```python
from sam.benchmarks import BenchmarkConfig

config = BenchmarkConfig(
    timeout_seconds=300,
    max_retries=3,
    temperature=0.7
)
```

## 📈 **Performance Optimization**

### **Memory Management**
- Use quantization (`load_in_4bit=True`) for large models
- Implement proper model unloading in `unload_model()`
- Monitor memory usage with `get_memory_usage()`

### **Speed Optimization**
- Batch similar requests when possible
- Use appropriate timeout values
- Implement efficient tokenization

### **Cost Optimization**
- Provide accurate cost estimates in `get_cost_estimate()`
- Consider local vs. API-based models
- Optimize prompt lengths for API models

## 🧪 **Testing**

### **Integration Tests**
```bash
# Test model wrapper implementation
python scripts/test_model_foundry.py

# Test specific model
python -c "
from sam.models.wrappers import create_model_wrapper
wrapper = create_model_wrapper('my-model', {})
print(wrapper.health_check())
"
```

### **Validation**
```bash
# Validate wrapper implementation
python -c "
from sam.models.wrappers import validate_model_wrapper
from sam.models.wrappers.my_model_wrapper import MyModelWrapper
print(validate_model_wrapper(MyModelWrapper))
"
```

## 🚀 **Production Deployment**

### **Environment Setup**
```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys (if using external judges)
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

### **Automated Evaluation**
```bash
# Weekly model evaluation
crontab -e
0 0 * * 0 cd /path/to/sam && python scripts/run_model_evaluation.py --models all
```

## 📚 **API Reference**

### **ModelWrapperTemplate**
Base class for all model integrations.

**Required Methods:**
- `load_model() -> bool`
- `generate(request) -> GenerationResponse`
- `get_model_metadata() -> ModelMetadata`
- `get_model_capabilities() -> ModelCapabilities`
- `get_cost_estimate() -> ModelCostEstimate`
- `health_check() -> bool`
- `unload_model() -> bool`

### **BenchmarkLoader**
Manages benchmark prompts and categories.

**Key Methods:**
- `load_benchmarks() -> List[BenchmarkPrompt]`
- `get_prompts_by_category(category) -> List[BenchmarkPrompt]`
- `filter_prompts(**criteria) -> List[BenchmarkPrompt]`

### **LLMJudge**
Handles scoring with external judge models.

**Supported Judges:**
- `gpt-4`, `gpt-4-turbo`
- `claude-3-opus`, `claude-3-sonnet`

## 🤝 **Contributing**

### **Adding New Models**
1. Fork the repository
2. Create model wrapper using template
3. Add comprehensive tests
4. Submit pull request with benchmark results

### **Extending Benchmarks**
1. Add prompts to `core_benchmark_v1.jsonl`
2. Update scoring criteria if needed
3. Test with existing models
4. Document new categories

## 📄 **License**

The SAM Model Foundry is part of the SAM project and follows the same licensing terms.

---

**Built with ❤️ by the SAM Development Team**  
*Transforming AI model evaluation, one benchmark at a time.*
