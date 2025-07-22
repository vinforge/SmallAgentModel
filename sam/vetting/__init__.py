"""
SAM Vetting System - Automated Content Analysis and Approval
===========================================================

Comprehensive vetting system for downloaded research papers and web content
with multi-dimensional scoring, automated analysis, and manual review workflows.

Components:
- Analyzer: Core analysis engine with security, relevance, and credibility scoring
- Security Scanner: File integrity and security risk assessment
- Relevance Scorer: Content relevance analysis using embeddings
- Credibility Assessor: Source credibility and quality evaluation
- Auto-approval Engine: Threshold-based automated approval system

Part of SAM's Task 27: Automated "Dream & Discover" Engine
Author: SAM Development Team
Version: 1.0.0
"""

from .analyzer import (
    VettingAnalyzer, AnalysisResult, SecurityRisk, RelevanceScore, CredibilityScore,
    get_vetting_analyzer
)

__all__ = [
    'VettingAnalyzer',
    'AnalysisResult',
    'SecurityRisk',
    'RelevanceScore', 
    'CredibilityScore',
    'get_vetting_analyzer',
]
