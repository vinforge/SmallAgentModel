#!/usr/bin/env python3
"""
SAM Dream Canvas Research
=========================

Research and insight generation for the Dream Canvas.

Author: SAM Development Team
Version: 1.0.0
"""

from .deep_research import (
    DeepResearchEngine,
    get_deep_research_engine,
    generate_cluster_insights,
    render_cluster_research_controls
)

__all__ = [
    'DeepResearchEngine',
    'get_deep_research_engine',
    'generate_cluster_insights',
    'render_cluster_research_controls'
]
