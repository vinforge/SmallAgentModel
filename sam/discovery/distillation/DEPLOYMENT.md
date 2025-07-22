# SAM Cognitive Distillation Engine - Production Deployment Guide

## Overview

This guide covers deploying the Cognitive Distillation Engine to SAM's production environment, integrating with existing systems, and enabling automated principle discovery.

## Prerequisites

- SAM core system running (localhost:8502)
- Database access for principle storage
- LLM integration (OpenAI API key or local models)
- Python 3.8+ with required dependencies

## Quick Deployment

### 1. Initialize the System

```python
from sam.discovery.distillation import SAMCognitiveDistillation

# Initialize with automation enabled
sam_cognitive = SAMCognitiveDistillation(enable_automation=True)

# The system is now ready for production use
print("✅ SAM Cognitive Distillation Engine deployed")
```

### 2. Integrate with SAM's Reasoning Loop

```python
# In SAM's main reasoning function
def enhanced_reasoning(user_query, context=None):
    # Step 1: Enhance prompt with principles
    enhanced_prompt, transparency_data = sam_cognitive.enhance_reasoning(
        user_query, context
    )
    
    # Step 2: Generate response using enhanced prompt
    response = sam_llm.generate_response(enhanced_prompt)
    
    # Step 3: Complete reasoning trace
    completed_transparency = sam_cognitive.complete_reasoning_trace(
        transparency_data, response
    )
    
    # Step 4: Return response with transparency data
    return response, completed_transparency
```

## Integration Points

### SAM Core Systems

#### 1. Reasoning Engine Integration
```python
# In reasoning/core.py or similar
from sam.discovery.distillation import SAMCognitiveDistillation

class EnhancedReasoningEngine:
    def __init__(self):
        self.cognitive_distillation = SAMCognitiveDistillation()
        
    def process_query(self, query, context=None):
        # Enhance with principles
        enhanced_prompt, transparency = self.cognitive_distillation.enhance_reasoning(
            query, context
        )
        
        # Continue with normal reasoning...
        return self.generate_response(enhanced_prompt), transparency
```

#### 2. Memory System Integration
```python
# In memory/episodic_memory.py or similar
def store_successful_interaction(query, response, quality_score, strategy_id):
    # Store in episodic memory as usual
    interaction_id = store_interaction(query, response, quality_score)
    
    # Also store for distillation analysis
    from sam.discovery.distillation.collector import interaction_collector
    
    successful_interaction = SuccessfulInteraction(
        interaction_id=interaction_id,
        strategy_id=strategy_id,
        query_text=query,
        response_text=response,
        success_metrics={'quality_score': quality_score}
    )
    
    interaction_collector.store_interaction(successful_interaction)
```

#### 3. Self-Improvement Engine Integration
```python
# In self_improvement/engine.py or similar
def trigger_improvement_cycle():
    # Existing improvement logic...
    
    # Trigger principle discovery
    from sam.discovery.distillation import automated_distillation
    
    # Manual trigger for high-priority strategies
    for strategy in high_priority_strategies:
        principle_id = automated_distillation.manual_trigger(strategy)
        if principle_id:
            logger.info(f"Discovered new principle for {strategy}: {principle_id}")
```

### UI Integration

#### 1. Secure Chat Interface (localhost:8502)
```python
# In secure_streamlit_app.py
import streamlit as st
from sam.discovery.distillation import SAMCognitiveDistillation

# Initialize cognitive distillation
if 'cognitive_distillation' not in st.session_state:
    st.session_state.cognitive_distillation = SAMCognitiveDistillation()

# In the chat interface
def process_user_message(user_input):
    # Enhance reasoning
    enhanced_prompt, transparency_data = st.session_state.cognitive_distillation.enhance_reasoning(
        user_input, {'user_session': st.session_state}
    )
    
    # Generate response
    response = generate_sam_response(enhanced_prompt)
    
    # Complete trace
    completed_transparency = st.session_state.cognitive_distillation.complete_reasoning_trace(
        transparency_data, response
    )
    
    # Display response with transparency
    display_response_with_principles(response, completed_transparency)
```

#### 2. Principle Transparency Display
```python
def display_response_with_principles(response, transparency_data):
    # Main response
    st.write(response)
    
    # Principle transparency (expandable)
    with st.expander("🧠 Reasoning Transparency"):
        active_principles = transparency_data.get('active_principles', [])
        
        if active_principles:
            st.write("**Applied Cognitive Principles:**")
            for principle in active_principles:
                st.write(f"• {principle['text']}")
                st.caption(f"Confidence: {principle['confidence']:.2f} | Domain: {', '.join(principle['domains'])}")
        
        # Meta-cognition insights
        meta_cognition = transparency_data.get('meta_cognition', {})
        if meta_cognition:
            st.write("**Reasoning Approach:**")
            st.write(meta_cognition.get('reasoning_approach', 'Standard reasoning'))
```

#### 3. System Status Dashboard
```python
def display_cognitive_status():
    """Display cognitive distillation system status in sidebar."""
    with st.sidebar:
        st.subheader("🧠 Cognitive Status")
        
        # Get system status
        status = st.session_state.cognitive_distillation.get_system_status()
        
        # Health indicator
        health = status.get('system_health', {})
        health_status = health.get('status', 'unknown')
        
        if health_status == 'healthy':
            st.success("System Healthy")
        elif health_status == 'warning':
            st.warning("System Warning")
        else:
            st.error("System Degraded")
        
        # Active principles count
        registry_stats = status.get('registry', {})
        active_principles = registry_stats.get('active_principles', 0)
        st.metric("Active Principles", active_principles)
        
        # Recent discoveries
        if st.button("🔍 Discover New Principles"):
            with st.spinner("Discovering principles..."):
                result = st.session_state.cognitive_distillation.manual_principle_discovery(
                    "general_reasoning"
                )
                if result.get('success'):
                    st.success("New principle discovered!")
                else:
                    st.info("No new principles found")
```

## Configuration

### Environment Variables
```bash
# Required for LLM integration
export OPENAI_API_KEY="your-openai-api-key"

# Optional: Custom database path
export SAM_DISTILLATION_DB_PATH="/path/to/distillation.db"

# Optional: Cache directory
export SAM_DISTILLATION_CACHE_DIR="/path/to/cache"

# Optional: Enable debug logging
export SAM_DISTILLATION_DEBUG=true
```

### System Configuration
```python
# In config/distillation_config.py
DISTILLATION_CONFIG = {
    'automation': {
        'enabled': True,
        'check_interval_minutes': 60,
        'min_interactions_for_discovery': 15,
        'auto_activate_principles': True
    },
    'prompt_augmentation': {
        'max_principles_per_prompt': 3,
        'min_principle_confidence': 0.5,
        'semantic_similarity_threshold': 0.6
    },
    'performance': {
        'cache_enabled': True,
        'cache_ttl_seconds': 3600,
        'max_cache_size': 1000
    }
}
```

## Monitoring and Analytics

### 1. Performance Monitoring
```python
# Regular performance checks
def monitor_cognitive_performance():
    status = sam_cognitive.get_system_status()
    
    # Log key metrics
    logger.info(f"Active principles: {status['registry']['active_principles']}")
    logger.info(f"System health: {status['system_health']['status']}")
    
    # Alert on issues
    if status['system_health']['status'] == 'degraded':
        send_alert("Cognitive distillation system degraded")
```

### 2. Principle Effectiveness Tracking
```python
# Track principle effectiveness over time
def track_principle_effectiveness():
    recent_traces = sam_cognitive.get_recent_reasoning_traces(limit=50)
    
    effectiveness_by_principle = {}
    for trace in recent_traces:
        for principle in trace.get('applied_principles', []):
            principle_id = principle['id']
            if principle_id not in effectiveness_by_principle:
                effectiveness_by_principle[principle_id] = []
            
            # Calculate effectiveness based on user feedback, response quality, etc.
            effectiveness_score = calculate_effectiveness(trace)
            effectiveness_by_principle[principle_id].append(effectiveness_score)
    
    # Update principle performance
    for principle_id, scores in effectiveness_by_principle.items():
        avg_effectiveness = sum(scores) / len(scores)
        sam_cognitive.update_principle_feedback(
            principle_id, 
            'success' if avg_effectiveness > 0.7 else 'neutral'
        )
```

## Troubleshooting

### Common Issues

1. **No principles being applied**
   - Check if principles exist: `sam_cognitive.get_active_principles()`
   - Verify domain matching in queries
   - Lower similarity threshold in configuration

2. **Poor principle quality**
   - Review interaction data quality
   - Adjust validation criteria
   - Manually review and curate principles

3. **Performance issues**
   - Enable caching in configuration
   - Monitor cache hit rates
   - Optimize principle selection algorithms

4. **Automation not working**
   - Check automation status: `automated_distillation.get_automation_stats()`
   - Verify trigger conditions
   - Review interaction data availability

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Get detailed system diagnostics
diagnostics = {
    'system_status': sam_cognitive.get_system_status(),
    'recent_traces': sam_cognitive.get_recent_reasoning_traces(limit=5),
    'active_principles': sam_cognitive.get_active_principles(limit=10)
}

print(json.dumps(diagnostics, indent=2, default=str))
```

## Security Considerations

1. **Principle Data Protection**: Principles may contain sensitive reasoning patterns
2. **LLM API Security**: Secure API keys and monitor usage
3. **Database Security**: Encrypt principle database if containing sensitive data
4. **Access Control**: Limit access to principle management functions

## Maintenance

### Regular Tasks
- Monitor system health weekly
- Review principle effectiveness monthly
- Clean up old cache entries
- Update validation criteria based on performance

### Updates
- Test new versions in staging environment
- Backup principle database before updates
- Monitor performance after deployments

## Support

For issues or questions:
1. Check system status dashboard
2. Review logs for error messages
3. Consult troubleshooting section
4. Contact SAM development team

---

**Note**: This deployment guide assumes integration with SAM's existing architecture. Adjust paths and integration points based on your specific SAM implementation.
