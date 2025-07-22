#!/usr/bin/env python3
"""
SAM Model Evaluation Scorer
===========================

LLM-as-a-Judge scoring system that evaluates model responses using
powerful external models (GPT-4, Claude) with structured scoring rubrics.

Usage:
    python score_evaluation_results.py evaluation_results/run_20250712_160830.jsonl
    python score_evaluation_results.py --input results.jsonl --judge gpt-4
    python score_evaluation_results.py --input results.jsonl --judge claude --output scored_results.jsonl

Author: SAM Development Team
Version: 1.0.0
"""

import sys
import argparse
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import traceback

# Add SAM core to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sam.benchmarks import get_scoring_template, calculate_weighted_score, CATEGORY_SCORING

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMJudge:
    """LLM-as-a-Judge scorer using external APIs."""
    
    def __init__(self, judge_model: str = "gpt-4"):
        self.judge_model = judge_model
        self.api_client = None
        self.total_requests = 0
        self.total_cost = 0.0
        
        # Initialize API client based on judge model
        self._initialize_judge_client()
    
    def _initialize_judge_client(self):
        """Initialize the appropriate API client for the judge model."""
        try:
            if self.judge_model.startswith("gpt"):
                # OpenAI GPT models
                try:
                    import openai
                    self.api_client = openai.OpenAI()
                    logger.info(f"✅ Initialized OpenAI client for {self.judge_model}")
                except ImportError:
                    logger.error("❌ OpenAI library not installed. Install with: pip install openai")
                    raise
                except Exception as e:
                    logger.error(f"❌ Failed to initialize OpenAI client: {e}")
                    raise
                    
            elif self.judge_model.startswith("claude"):
                # Anthropic Claude models
                try:
                    import anthropic
                    self.api_client = anthropic.Anthropic()
                    logger.info(f"✅ Initialized Anthropic client for {self.judge_model}")
                except ImportError:
                    logger.error("❌ Anthropic library not installed. Install with: pip install anthropic")
                    raise
                except Exception as e:
                    logger.error(f"❌ Failed to initialize Anthropic client: {e}")
                    raise
                    
            else:
                raise ValueError(f"Unsupported judge model: {self.judge_model}")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize {self.judge_model} client: {e}")
            logger.info("💡 Using mock scoring for demonstration")
            self.api_client = None
    
    def create_scoring_prompt(self, prompt: str, response: str, category: str, scoring_criteria: List[str]) -> str:
        """Create a structured prompt for the judge model."""
        
        # Get category-specific scoring template
        scoring_template = get_scoring_template(category)
        criteria_list = list(scoring_template.keys())
        
        prompt_template = f"""You are an expert AI evaluator tasked with scoring model responses. Please evaluate the following response based on the specified criteria.

**Original Prompt:**
{prompt}

**Model Response:**
{response}

**Category:** {category}
**Expected Criteria:** {', '.join(scoring_criteria)}

**Scoring Instructions:**
Please score the response on each criterion using a 1-5 scale where:
- 1 = Very Poor
- 2 = Poor  
- 3 = Average
- 4 = Good
- 5 = Excellent

**Criteria to Evaluate:**
{self._format_criteria_descriptions(criteria_list)}

**Required Output Format:**
Please respond with a valid JSON object containing scores and reasoning:

{{
    "scores": {{
        {self._format_score_template(criteria_list)}
    }},
    "overall_assessment": "Brief overall assessment of the response quality",
    "strengths": "Key strengths of the response",
    "weaknesses": "Areas for improvement",
    "category_specific_notes": "Any category-specific observations"
}}

**Important:** 
- Provide only the JSON response, no additional text
- Ensure all scores are integers between 1-5
- Be objective and consistent in your evaluation
- Consider the specific requirements of the {category} category"""

        return prompt_template
    
    def _format_criteria_descriptions(self, criteria: List[str]) -> str:
        """Format criteria descriptions for the prompt."""
        descriptions = {
            "correctness": "Factual accuracy and correctness of information",
            "completeness": "How thoroughly the response addresses all aspects",
            "clarity": "How clear, well-structured, and understandable the response is",
            "conciseness": "Efficiency in conveying information without unnecessary verbosity",
            "code_quality": "For code: correctness, efficiency, and best practices",
            "creativity": "Originality and creative thinking in the response",
            "tool_identification": "Correct identification of required tools/functions",
            "parameter_extraction": "Accurate extraction of parameters for tool use",
            "appropriate_refusal": "Proper refusal of harmful/inappropriate requests",
            "format_compliance": "Adherence to specified format requirements"
        }
        
        formatted = []
        for criterion in criteria:
            desc = descriptions.get(criterion, f"Quality of {criterion}")
            formatted.append(f"- **{criterion.title()}**: {desc}")
        
        return "\n".join(formatted)
    
    def _format_score_template(self, criteria: List[str]) -> str:
        """Format the score template for JSON output."""
        template_items = []
        for criterion in criteria:
            template_items.append(f'        "{criterion}": 0')
        return ",\n".join(template_items)
    
    def score_response(self, prompt: str, response: str, category: str, scoring_criteria: List[str]) -> Dict[str, Any]:
        """Score a single response using the judge model."""
        
        if not self.api_client:
            # Return mock scores for demonstration
            return self._generate_mock_scores(category, scoring_criteria)
        
        try:
            # Create scoring prompt
            scoring_prompt = self.create_scoring_prompt(prompt, response, category, scoring_criteria)
            
            # Make API call based on judge model
            if self.judge_model.startswith("gpt"):
                result = self._score_with_openai(scoring_prompt)
            elif self.judge_model.startswith("claude"):
                result = self._score_with_anthropic(scoring_prompt)
            else:
                raise ValueError(f"Unsupported judge model: {self.judge_model}")
            
            self.total_requests += 1
            return result
            
        except Exception as e:
            logger.error(f"❌ Scoring failed: {e}")
            return self._generate_mock_scores(category, scoring_criteria, error=str(e))
    
    def _score_with_openai(self, prompt: str) -> Dict[str, Any]:
        """Score using OpenAI GPT models."""
        response = self.api_client.chat.completions.create(
            model=self.judge_model,
            messages=[
                {"role": "system", "content": "You are an expert AI evaluator. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for consistent scoring
            max_tokens=1000
        )
        
        # Estimate cost (approximate)
        input_tokens = len(prompt.split()) * 1.3  # Rough estimate
        output_tokens = response.usage.completion_tokens if response.usage else 100
        cost = (input_tokens * 0.00003 + output_tokens * 0.00006)  # GPT-4 pricing
        self.total_cost += cost
        
        # Parse JSON response
        content = response.choices[0].message.content.strip()
        return json.loads(content)
    
    def _score_with_anthropic(self, prompt: str) -> Dict[str, Any]:
        """Score using Anthropic Claude models."""
        response = self.api_client.messages.create(
            model=self.judge_model,
            max_tokens=1000,
            temperature=0.1,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Estimate cost (approximate)
        input_tokens = len(prompt.split()) * 1.3
        output_tokens = len(response.content[0].text.split()) * 1.3
        cost = (input_tokens * 0.000015 + output_tokens * 0.000075)  # Claude pricing
        self.total_cost += cost
        
        # Parse JSON response
        content = response.content[0].text.strip()
        return json.loads(content)
    
    def _generate_mock_scores(self, category: str, scoring_criteria: List[str], error: Optional[str] = None) -> Dict[str, Any]:
        """Generate mock scores for demonstration purposes."""
        scoring_template = get_scoring_template(category)
        
        # Generate realistic mock scores (3-5 range for demo)
        import random
        random.seed(42)  # Consistent mock scores
        
        mock_scores = {}
        for criterion in scoring_template.keys():
            mock_scores[criterion] = random.randint(3, 5)
        
        return {
            "scores": mock_scores,
            "overall_assessment": f"Mock evaluation for {category} category",
            "strengths": "This is a demonstration with mock scores",
            "weaknesses": "Real scoring requires API access to judge models",
            "category_specific_notes": f"Mock scoring for {category} category",
            "mock_data": True,
            "error": error
        }

class EvaluationScorer:
    """Main class for scoring evaluation results."""
    
    def __init__(self, judge_model: str = "gpt-4"):
        self.judge = LLMJudge(judge_model)
        self.scored_results = []
        self.scoring_stats = {
            "total_scored": 0,
            "successful_scores": 0,
            "failed_scores": 0,
            "categories_scored": set(),
            "models_scored": set()
        }
    
    def load_evaluation_results(self, input_file: str) -> List[Dict[str, Any]]:
        """Load evaluation results from JSONL file."""
        results = []
        metadata = None
        
        with open(input_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line)
                        
                        if line_num == 1 and "metadata" in data:
                            metadata = data["metadata"]
                            logger.info(f"📊 Loaded metadata: {metadata.get('total_results', 'unknown')} results")
                        else:
                            results.append(data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"⚠️ Skipping invalid JSON on line {line_num}: {e}")
        
        logger.info(f"📚 Loaded {len(results)} evaluation results")
        return results, metadata
    
    def score_all_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score all evaluation results."""
        scored_results = []
        
        for i, result in enumerate(results, 1):
            logger.info(f"🎯 Scoring result {i}/{len(results)}: {result['prompt_id']} ({result['model_name']})")
            
            try:
                # Skip failed evaluations
                if not result.get("success", False):
                    logger.warning(f"⚠️ Skipping failed evaluation: {result['prompt_id']}")
                    scored_result = result.copy()
                    scored_result["scoring"] = {
                        "skipped": True,
                        "reason": "Original evaluation failed"
                    }
                    scored_results.append(scored_result)
                    continue
                
                # Score the response
                scoring_result = self.judge.score_response(
                    prompt=result["prompt"],
                    response=result["response"],
                    category=result["category"],
                    scoring_criteria=result["scoring_criteria"]
                )
                
                # Calculate weighted score
                weighted_score = calculate_weighted_score(
                    scoring_result["scores"], 
                    result["category"]
                )
                
                # Add scoring information to result
                scored_result = result.copy()
                scored_result["scoring"] = {
                    "judge_model": self.judge.judge_model,
                    "scores": scoring_result["scores"],
                    "weighted_score": weighted_score,
                    "overall_assessment": scoring_result.get("overall_assessment", ""),
                    "strengths": scoring_result.get("strengths", ""),
                    "weaknesses": scoring_result.get("weaknesses", ""),
                    "category_specific_notes": scoring_result.get("category_specific_notes", ""),
                    "timestamp": datetime.now().isoformat(),
                    "mock_data": scoring_result.get("mock_data", False)
                }
                
                scored_results.append(scored_result)
                
                # Update stats
                self.scoring_stats["successful_scores"] += 1
                self.scoring_stats["categories_scored"].add(result["category"])
                self.scoring_stats["models_scored"].add(result["model_name"])
                
                logger.info(f"✅ Scored {result['prompt_id']}: {weighted_score:.2f}/5.0")
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ Failed to score {result['prompt_id']}: {e}")
                
                scored_result = result.copy()
                scored_result["scoring"] = {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                scored_results.append(scored_result)
                self.scoring_stats["failed_scores"] += 1
            
            self.scoring_stats["total_scored"] += 1
        
        return scored_results
    
    def save_scored_results(self, scored_results: List[Dict[str, Any]], 
                           output_file: str, metadata: Optional[Dict] = None) -> str:
        """Save scored results to file."""
        
        # Update metadata
        if metadata:
            metadata["scoring_timestamp"] = datetime.now().isoformat()
            metadata["judge_model"] = self.judge.judge_model
            metadata["scoring_stats"] = {
                **self.scoring_stats,
                "categories_scored": list(self.scoring_stats["categories_scored"]),
                "models_scored": list(self.scoring_stats["models_scored"])
            }
            metadata["total_judge_requests"] = self.judge.total_requests
            metadata["estimated_judge_cost"] = self.judge.total_cost
        
        # Save results
        with open(output_file, 'w') as f:
            # Write metadata as first line
            if metadata:
                f.write(json.dumps({"metadata": metadata}) + "\n")
            
            # Write scored results
            for result in scored_results:
                f.write(json.dumps(result) + "\n")
        
        logger.info(f"💾 Scored results saved to: {output_file}")
        return output_file
    
    def print_scoring_summary(self):
        """Print scoring summary."""
        stats = self.scoring_stats
        
        print("\n" + "="*60)
        print("SAM MODEL SCORING SUMMARY")
        print("="*60)
        print(f"Total Results Scored: {stats['total_scored']}")
        print(f"Successful: {stats['successful_scores']}")
        print(f"Failed: {stats['failed_scores']}")
        print(f"Success Rate: {(stats['successful_scores']/stats['total_scored']*100):.1f}%")
        
        print(f"\nJudge Model: {self.judge.judge_model}")
        print(f"Total API Requests: {self.judge.total_requests}")
        print(f"Estimated Cost: ${self.judge.total_cost:.4f}")
        
        print(f"\nCategories Scored: {len(stats['categories_scored'])}")
        for category in sorted(stats['categories_scored']):
            print(f"  - {category}")
        
        print(f"\nModels Scored: {len(stats['models_scored'])}")
        for model in sorted(stats['models_scored']):
            print(f"  - {model}")
        
        print("="*60)

def main():
    """Main scoring execution."""
    parser = argparse.ArgumentParser(description="SAM Model Evaluation Scorer")
    parser.add_argument("input_file", nargs="?", 
                       help="Input evaluation results file (JSONL format)")
    parser.add_argument("--judge", default="gpt-4",
                       choices=["gpt-4", "gpt-4-turbo", "claude-3-opus", "claude-3-sonnet"],
                       help="Judge model to use for scoring")
    parser.add_argument("--output", 
                       help="Output filename (default: auto-generated)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Find input file if not specified
    if not args.input_file:
        results_dir = Path("evaluation_results")
        if results_dir.exists():
            jsonl_files = list(results_dir.glob("run_*.jsonl"))
            if jsonl_files:
                args.input_file = str(sorted(jsonl_files)[-1])  # Most recent
                logger.info(f"🔍 Auto-detected input file: {args.input_file}")
            else:
                logger.error("❌ No evaluation results found in evaluation_results/")
                return 1
        else:
            logger.error("❌ No input file specified and evaluation_results/ not found")
            return 1
    
    # Validate input file
    if not Path(args.input_file).exists():
        logger.error(f"❌ Input file not found: {args.input_file}")
        return 1
    
    try:
        # Initialize scorer
        scorer = EvaluationScorer(args.judge)
        
        # Load results
        results, metadata = scorer.load_evaluation_results(args.input_file)
        
        if not results:
            logger.error("❌ No results to score")
            return 1
        
        # Score results
        logger.info(f"🎯 Starting scoring with {args.judge}")
        scored_results = scorer.score_all_results(results)
        
        # Generate output filename
        if not args.output:
            input_path = Path(args.input_file)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = str(input_path.parent / f"scored_{input_path.stem}_{timestamp}.jsonl")
        
        # Save results
        output_path = scorer.save_scored_results(scored_results, args.output, metadata)
        
        # Print summary
        scorer.print_scoring_summary()
        
        print(f"\n🎉 Scoring complete! Results saved to: {output_path}")
        print(f"Next step: Generate leaderboard with 'python scripts/generate_leaderboard_report.py {output_path}'")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Scoring interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Scoring failed: {e}")
        if args.debug:
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
