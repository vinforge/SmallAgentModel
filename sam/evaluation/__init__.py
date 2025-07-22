"""
SAM Evaluation Module - Task 30 Phase 3
=======================================

Provides A/B testing framework and LLM-based evaluation capabilities
for validating conversational coherence improvements.

Part of Task 30: Advanced Conversational Coherence Engine
"""

from .ab_testing import (
    ABTestingFramework,
    ABTestConfig,
    ABTestResult,
    get_ab_testing_framework
)

from .llm_judge import (
    LLMJudge,
    EvaluationCriteria,
    EvaluationResult,
    get_llm_judge
)

__all__ = [
    'ABTestingFramework',
    'ABTestConfig', 
    'ABTestResult',
    'get_ab_testing_framework',
    'LLMJudge',
    'EvaluationCriteria',
    'EvaluationResult',
    'get_llm_judge'
]
