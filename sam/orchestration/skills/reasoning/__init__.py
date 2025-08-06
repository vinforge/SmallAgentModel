"""
SAM Orchestration Framework - Reasoning Skills Module
====================================================

This module contains reasoning-related skills for the SAM Orchestration Framework.
These skills implement advanced cognitive capabilities like implicit knowledge
generation, conceptual understanding, and multi-hop reasoning.

Reasoning Skills:
- ImplicitKnowledgeSkill: Generates implicit connections between explicit knowledge chunks
- TestTimeAdaptationSkill: Test-Time Training for few-shot reasoning adaptation
"""

from .implicit_knowledge import ImplicitKnowledgeSkill

try:
    from .test_time_adaptation import (
        TestTimeAdaptation,
        AdaptationResult,
        get_test_time_adaptation,
        is_test_time_adaptation_available
    )
    # Create alias for compatibility
    TestTimeAdaptationSkill = TestTimeAdaptation

    __all__ = [
        'ImplicitKnowledgeSkill',
        'TestTimeAdaptationSkill',
        'TestTimeAdaptation',
        'AdaptationResult',
        'get_test_time_adaptation',
        'is_test_time_adaptation_available'
    ]
except ImportError:
    __all__ = [
        'ImplicitKnowledgeSkill',
    ]
