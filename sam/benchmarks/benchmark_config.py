"""
SAM Core Benchmark Configuration
===============================

Configuration and utilities for the SAM Core Benchmark Suite.
This module defines benchmark categories, scoring criteria, and evaluation parameters.

Author: SAM Development Team
Version: 1.0.0
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

class BenchmarkCategory(Enum):
    """Benchmark categories for model evaluation."""
    QA = "qa"
    SUMMARIZATION = "summarization"
    REASONING = "reasoning"
    CODE_GENERATION = "code_gen"
    TOOL_USE = "tool_use"
    LONG_CONTEXT_RECALL = "long_context_recall"
    SAFETY_REFUSAL = "safety_refusal"
    CREATIVITY = "creativity"
    ANALYSIS = "analysis"
    INSTRUCTION_FOLLOWING = "instruction_following"

@dataclass
class BenchmarkPrompt:
    """Individual benchmark prompt configuration."""
    id: str
    category: str
    prompt: str
    expected_type: str
    max_tokens: int
    scoring_criteria: List[str]
    expected_function: Optional[str] = None
    difficulty: str = "medium"
    context_length: int = 0
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkPrompt":
        """Create BenchmarkPrompt from dictionary."""
        return cls(
            id=data["id"],
            category=data["category"],
            prompt=data["prompt"],
            expected_type=data["expected_type"],
            max_tokens=data["max_tokens"],
            scoring_criteria=data["scoring_criteria"],
            expected_function=data.get("expected_function"),
            difficulty=data.get("difficulty", "medium"),
            context_length=len(data["prompt"].split())
        )

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    name: str = "SAM Core Benchmark v1.0"
    version: str = "1.0.0"
    total_prompts: int = 30
    categories: List[str] = None
    timeout_seconds: int = 300
    max_retries: int = 3
    temperature: float = 0.7
    
    def __post_init__(self):
        if self.categories is None:
            self.categories = [cat.value for cat in BenchmarkCategory]

class BenchmarkLoader:
    """Loads and manages benchmark prompts."""
    
    def __init__(self, benchmark_file: str = "core_benchmark_v1.jsonl"):
        self.benchmark_file = Path(__file__).parent / benchmark_file
        self.prompts: List[BenchmarkPrompt] = []
        self.categories: Dict[str, List[BenchmarkPrompt]] = {}
        
    def load_benchmarks(self) -> List[BenchmarkPrompt]:
        """Load benchmark prompts from JSONL file."""
        if not self.benchmark_file.exists():
            raise FileNotFoundError(f"Benchmark file not found: {self.benchmark_file}")
        
        self.prompts = []
        self.categories = {}
        
        with open(self.benchmark_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    prompt = BenchmarkPrompt.from_dict(data)
                    self.prompts.append(prompt)
                    
                    # Group by category
                    if prompt.category not in self.categories:
                        self.categories[prompt.category] = []
                    self.categories[prompt.category].append(prompt)
        
        return self.prompts
    
    def get_prompts_by_category(self, category: str) -> List[BenchmarkPrompt]:
        """Get all prompts for a specific category."""
        return self.categories.get(category, [])
    
    def get_prompt_by_id(self, prompt_id: str) -> Optional[BenchmarkPrompt]:
        """Get a specific prompt by ID."""
        for prompt in self.prompts:
            if prompt.id == prompt_id:
                return prompt
        return None
    
    def get_category_stats(self) -> Dict[str, int]:
        """Get statistics about prompts per category."""
        return {category: len(prompts) for category, prompts in self.categories.items()}
    
    def filter_prompts(self, 
                      categories: Optional[List[str]] = None,
                      difficulty: Optional[str] = None,
                      max_context_length: Optional[int] = None) -> List[BenchmarkPrompt]:
        """Filter prompts based on criteria."""
        filtered = self.prompts
        
        if categories:
            filtered = [p for p in filtered if p.category in categories]
        
        if difficulty:
            filtered = [p for p in filtered if p.difficulty == difficulty]
        
        if max_context_length:
            filtered = [p for p in filtered if p.context_length <= max_context_length]
        
        return filtered

# Scoring rubric for LLM-as-a-Judge
SCORING_RUBRIC = {
    "correctness": {
        "description": "How factually accurate and correct is the response?",
        "scale": "1-5 (1=Completely incorrect, 5=Completely correct)",
        "weight": 0.4
    },
    "completeness": {
        "description": "How thoroughly does the response address all aspects of the prompt?",
        "scale": "1-5 (1=Incomplete, 5=Comprehensive)",
        "weight": 0.3
    },
    "clarity": {
        "description": "How clear, well-structured, and easy to understand is the response?",
        "scale": "1-5 (1=Very unclear, 5=Very clear)",
        "weight": 0.2
    },
    "conciseness": {
        "description": "How efficiently does the response convey information without unnecessary verbosity?",
        "scale": "1-5 (1=Very verbose, 5=Perfectly concise)",
        "weight": 0.1
    }
}

# Category-specific scoring adjustments
CATEGORY_SCORING = {
    "qa": {
        "primary_criteria": ["correctness", "conciseness"],
        "weights": {"correctness": 0.7, "conciseness": 0.3}
    },
    "summarization": {
        "primary_criteria": ["completeness", "conciseness", "correctness"],
        "weights": {"completeness": 0.4, "conciseness": 0.3, "correctness": 0.3}
    },
    "reasoning": {
        "primary_criteria": ["correctness", "clarity"],
        "weights": {"correctness": 0.6, "clarity": 0.4}
    },
    "code_gen": {
        "primary_criteria": ["correctness", "completeness"],
        "weights": {"correctness": 0.7, "completeness": 0.3},
        "additional_criteria": ["code_quality", "efficiency"]
    },
    "tool_use": {
        "primary_criteria": ["correctness", "completeness"],
        "weights": {"correctness": 0.8, "completeness": 0.2},
        "additional_criteria": ["tool_identification", "parameter_extraction"]
    },
    "long_context_recall": {
        "primary_criteria": ["correctness", "completeness"],
        "weights": {"correctness": 0.8, "completeness": 0.2},
        "additional_criteria": ["information_retrieval"]
    },
    "safety_refusal": {
        "primary_criteria": ["correctness", "clarity"],
        "weights": {"correctness": 0.9, "clarity": 0.1},
        "additional_criteria": ["appropriate_refusal", "helpful_alternative"]
    },
    "creativity": {
        "primary_criteria": ["completeness", "clarity"],
        "weights": {"completeness": 0.5, "clarity": 0.5},
        "additional_criteria": ["creativity", "originality"]
    },
    "analysis": {
        "primary_criteria": ["completeness", "correctness", "clarity"],
        "weights": {"completeness": 0.4, "correctness": 0.4, "clarity": 0.2}
    },
    "instruction_following": {
        "primary_criteria": ["correctness", "completeness"],
        "weights": {"correctness": 0.6, "completeness": 0.4},
        "additional_criteria": ["format_compliance", "constraint_adherence"]
    }
}

def get_scoring_template(category: str) -> Dict[str, Any]:
    """Get scoring template for a specific category."""
    base_template = {
        "correctness": {"score": 0, "reasoning": ""},
        "completeness": {"score": 0, "reasoning": ""},
        "clarity": {"score": 0, "reasoning": ""},
        "conciseness": {"score": 0, "reasoning": ""}
    }
    
    # Add category-specific criteria
    if category in CATEGORY_SCORING:
        category_config = CATEGORY_SCORING[category]
        if "additional_criteria" in category_config:
            for criterion in category_config["additional_criteria"]:
                base_template[criterion] = {"score": 0, "reasoning": ""}
    
    return base_template

def calculate_weighted_score(scores: Dict[str, int], category: str) -> float:
    """Calculate weighted score for a category."""
    if category not in CATEGORY_SCORING:
        # Use default weights
        weights = SCORING_RUBRIC
        total_score = sum(scores[criterion] * weights[criterion]["weight"] 
                         for criterion in scores if criterion in weights)
    else:
        # Use category-specific weights
        category_weights = CATEGORY_SCORING[category]["weights"]
        total_score = sum(scores[criterion] * category_weights.get(criterion, 0.1) 
                         for criterion in scores if criterion in category_weights)
    
    return round(total_score, 2)

# Export main classes and functions
__all__ = [
    'BenchmarkCategory',
    'BenchmarkPrompt',
    'BenchmarkConfig',
    'BenchmarkLoader',
    'SCORING_RUBRIC',
    'CATEGORY_SCORING',
    'get_scoring_template',
    'calculate_weighted_score'
]
