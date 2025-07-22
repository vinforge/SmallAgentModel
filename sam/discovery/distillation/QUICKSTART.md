# SAM Cognitive Distillation Engine - Quick Start Guide

## Overview

The Cognitive Distillation Engine analyzes SAM's successful behaviors and distills them into human-readable "Principles of Reasoning" for improved performance and explainability.

## Quick Setup

### 1. Import the Engine
```python
from sam.discovery.distillation import DistillationEngine

# Initialize the engine
engine = DistillationEngine()
```

### 2. Discover a Principle
```python
# Discover principle from a strategy
principle = engine.discover_principle("your_strategy_id", interaction_limit=20)

if principle:
    print(f"Discovered: {principle.principle_text}")
    print(f"Confidence: {principle.confidence_score:.2f}")
    print(f"Domains: {principle.domain_tags}")
else:
    print("No principle discovered")
```

### 3. Manage Principles
```python
from sam.discovery.distillation import PrincipleRegistry

registry = PrincipleRegistry()

# Get all active principles
active_principles = registry.get_active_principles(limit=10)

# Search by domain
financial_principles = registry.search_principles_by_domain("financial")

# Update principle performance
registry.update_principle_performance(principle.principle_id, "success", 0.1)
```

## Example: Complete Workflow

```python
from sam.discovery.distillation import DistillationEngine, PrincipleRegistry
from sam.discovery.distillation.collector import InteractionCollector, SuccessfulInteraction
import uuid
from datetime import datetime

# 1. Create test interaction data
collector = InteractionCollector()

test_interaction = SuccessfulInteraction(
    interaction_id=str(uuid.uuid4()),
    strategy_id="financial_analysis_v1",
    query_text="What are the current stock market trends?",
    context_provided="Recent market data from financial sources",
    response_text="Based on recent analysis from Bloomberg and Reuters, the market shows upward momentum with technology stocks leading gains.",
    success_metrics={"quality_score": 0.9, "user_rating": 5},
    source_system="test_system"
)

# Store the interaction
collector.store_interaction(test_interaction)

# 2. Discover principle
engine = DistillationEngine()
principle = engine.discover_principle("financial_analysis_v1")

if principle:
    print(f"✅ Discovered principle: {principle.principle_text}")
    
    # 3. Use the principle
    registry = PrincipleRegistry()
    
    # Get principle details
    stored_principle = registry.get_principle(principle.principle_id)
    print(f"📊 Confidence: {stored_principle.confidence_score:.2f}")
    print(f"🏷️  Domains: {stored_principle.domain_tags}")
    
    # Update performance after using it
    registry.update_principle_performance(principle.principle_id, "success", 0.05)
    
    print("✅ Principle ready for use in reasoning!")
```

## LLM Integration

### Using Different LLM Backends

```python
from sam.discovery.distillation.llm_integration import LLMIntegration

llm = LLMIntegration()

# The system will automatically try:
# 1. SAM's existing LLM infrastructure
# 2. OpenAI API (if OPENAI_API_KEY is set)
# 3. Local models via Ollama
# 4. Fallback to heuristic methods

interaction_data = [
    {
        'query': 'What are financial market trends?',
        'context': 'Recent market data from multiple sources',
        'response': 'Based on authoritative financial sources...',
        'success_metrics': {'quality_score': 0.9}
    }
]

response = llm.discover_principle(interaction_data)
if response:
    print(f"Principle: {response.discovered_principle}")
    print(f"Confidence: {response.confidence}")
    print(f"Reasoning: {response.reasoning}")
```

## Validation

### Validate Principle Quality

```python
from sam.discovery.distillation import PrincipleValidator

validator = PrincipleValidator()

principle_text = "For financial queries, prioritize recent sources and cite specific data points"
result = validator.validate_principle(principle_text)

print(f"Valid: {result['is_valid']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Quality Score: {result['quality_score']:.2f}")

if result['issues']:
    print(f"Issues: {result['issues']}")
if result['suggestions']:
    print(f"Suggestions: {result['suggestions']}")
```

## Monitoring and Analytics

### Get System Statistics

```python
# Engine statistics
engine_stats = engine.get_distillation_stats()
print(f"Total runs: {engine_stats['total_runs']}")
print(f"Success rate: {engine_stats['success_rate']}%")

# Registry statistics
registry_stats = registry.get_principle_stats()
print(f"Active principles: {registry_stats['active_principles']}")
print(f"Average confidence: {registry_stats['avg_confidence']}")

# Interaction statistics
interaction_summary = collector.get_interaction_summary("your_strategy_id")
print(f"Total interactions: {interaction_summary['total_interactions']}")
print(f"Average quality: {interaction_summary['avg_quality_score']}")
```

## Configuration

### Environment Variables

```bash
# Optional: OpenAI API key for LLM integration
export OPENAI_API_KEY="your-api-key-here"

# Optional: Custom database path
export SAM_DISTILLATION_DB_PATH="/path/to/custom/db"
```

### Validation Criteria Customization

```python
from sam.discovery.distillation.validator import PrincipleValidator

validator = PrincipleValidator()

# Customize validation criteria
validator.min_confidence = 0.5  # Minimum confidence threshold
validator.criteria_weights = {
    'clarity': 0.3,
    'actionability': 0.3,
    'specificity': 0.2,
    'generalizability': 0.1,
    'novelty': 0.1
}
```

## Troubleshooting

### Common Issues

1. **No principles discovered**
   - Check if sufficient interaction data exists (minimum 3 interactions)
   - Verify interaction quality scores are > 0.7
   - Check LLM connectivity (falls back to heuristics if unavailable)

2. **Low principle quality**
   - Increase interaction data quality
   - Adjust validation criteria weights
   - Review and improve interaction collection filters

3. **LLM integration fails**
   - Verify API keys are set correctly
   - Check network connectivity
   - System gracefully falls back to heuristic methods

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('sam.discovery.distillation')

# Now run your distillation process
principle = engine.discover_principle("strategy_id")
```

## Integration with SAM

### Automatic Principle Discovery

```python
# This can be integrated into SAM's self-improvement cycle
def auto_discover_principles():
    engine = DistillationEngine()
    registry = PrincipleRegistry()
    
    # Get strategies that need principle discovery
    strategies = ["financial_analysis", "technical_support", "research_queries"]
    
    for strategy_id in strategies:
        principle = engine.discover_principle(strategy_id)
        if principle:
            print(f"Auto-discovered principle for {strategy_id}")
            
            # Activate for immediate use
            registry.update_principle_performance(principle.principle_id, "activated", 0.0)

# Run periodically or trigger on successful interactions
auto_discover_principles()
```

## Next Steps

1. **Phase 2 Integration**: Principle-augmented prompting in live reasoning
2. **UI Integration**: Display discovered principles in SAM's interface
3. **Performance Optimization**: Semantic search and caching
4. **Advanced Analytics**: Principle effectiveness tracking

For detailed documentation, see `README.md` in the distillation module.
