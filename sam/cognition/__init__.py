#!/usr/bin/env python3
"""
SAM Cognition Module
Advanced cognitive processing capabilities for SAM's v2 retrieval pipeline.
"""

from .muvera_fde import (
    MuveraFDE,
    FDEResult,
    get_muvera_fde,
    generate_fde
)

from .similarity_metrics import (
    ChamferSimilarity,
    MaxSimSimilarity,
    compute_chamfer_distance,
    compute_maxsim_score,
    get_similarity_calculator
)

__all__ = [
    'MuveraFDE',
    'FDEResult', 
    'get_muvera_fde',
    'generate_fde',
    'ChamferSimilarity',
    'MaxSimSimilarity',
    'compute_chamfer_distance',
    'compute_maxsim_score',
    'get_similarity_calculator'
]
