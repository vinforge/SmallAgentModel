#!/usr/bin/env python3
"""
Phase 3: LLM-as-a-Judge Response Evaluation
Qualitative analysis of A/B test responses using external LLM evaluation
"""

import sys
import json
import time
import logging
import random
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add SAM to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class QualityScore:
    """Quality evaluation scores from LLM judge."""
    correctness: int        # 1-5: Factual accuracy and correctness
    conciseness: int        # 1-5: Relevance without rambling
    completeness: int       # 1-5: Covers key information
    overall: int           # 1-5: Overall quality
    reasoning: str         # Judge's reasoning for scores

class LLMJudge:
    """LLM-as-a-Judge evaluator for response quality."""
    
    def __init__(self, judge_model: str = "local"):
        self.judge_model = judge_model
        self.evaluation_prompt_template = self._create_evaluation_prompt()
    
    def _create_evaluation_prompt(self) -> str:
        """Create structured evaluation prompt for the judge."""
        return """You are an expert AI evaluator tasked with scoring response quality. You will evaluate three responses to the same query.

EVALUATION CRITERIA:
1. Correctness & Factual Accuracy (1-5): Is the information accurate and correct?
2. Conciseness & Relevance (1-5): Does it answer directly without unnecessary rambling?
3. Completeness (1-5): Does it cover the key information needed to answer the question?
4. Overall Quality (1-5): General assessment of response usefulness

SCORING SCALE:
5 = Excellent, 4 = Good, 3 = Average, 2 = Below Average, 1 = Poor

ORIGINAL QUERY: {query}

RESPONSE A: {response_a}

RESPONSE B: {response_b}

RESPONSE C: {response_c}

Please evaluate each response and provide scores in this exact JSON format:
{{
  "response_a": {{
    "correctness": <score>,
    "conciseness": <score>,
    "completeness": <score>,
    "overall": <score>,
    "reasoning": "<brief explanation>"
  }},
  "response_b": {{
    "correctness": <score>,
    "conciseness": <score>,
    "completeness": <score>,
    "overall": <score>,
    "reasoning": "<brief explanation>"
  }},
  "response_c": {{
    "correctness": <score>,
    "conciseness": <score>,
    "completeness": <score>,
    "overall": <score>,
    "reasoning": "<brief explanation>"
  }}
}}"""
    
    def evaluate_responses(self, query: str, responses: Dict[str, str]) -> Dict[str, QualityScore]:
        """Evaluate three responses using LLM judge."""
        
        # Randomize response order to prevent bias
        response_keys = list(responses.keys())
        random.shuffle(response_keys)
        
        # Map randomized responses
        response_map = {
            'response_a': responses[response_keys[0]],
            'response_b': responses[response_keys[1]], 
            'response_c': responses[response_keys[2]]
        }
        
        # Create evaluation prompt
        evaluation_prompt = self.evaluation_prompt_template.format(
            query=query,
            response_a=response_map['response_a'],
            response_b=response_map['response_b'],
            response_c=response_map['response_c']
        )
        
        try:
            # Get evaluation from judge
            judge_response = self._call_judge_llm(evaluation_prompt)
            
            # Parse JSON response
            evaluation_data = json.loads(judge_response)
            
            # Map back to original response keys
            quality_scores = {}
            for i, original_key in enumerate(response_keys):
                response_key = f'response_{chr(97+i)}'  # a, b, c
                eval_data = evaluation_data[response_key]
                
                quality_scores[original_key] = QualityScore(
                    correctness=eval_data['correctness'],
                    conciseness=eval_data['conciseness'],
                    completeness=eval_data['completeness'],
                    overall=eval_data['overall'],
                    reasoning=eval_data['reasoning']
                )
            
            return quality_scores
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            # Return default scores on failure
            return {key: QualityScore(3, 3, 3, 3, f"Evaluation failed: {e}") for key in responses.keys()}
    
    def _call_judge_llm(self, prompt: str) -> str:
        """Call the judge LLM with evaluation prompt."""
        if self.judge_model == "local":
            return self._call_local_ollama(prompt)
        else:
            raise NotImplementedError(f"Judge model {self.judge_model} not implemented")
    
    def _call_local_ollama(self, prompt: str) -> str:
        """Call local Ollama for evaluation."""
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for consistent evaluation
                        "top_p": 0.9,
                        "max_tokens": 1000
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                response_data = response.json()
                return response_data.get('response', '').strip()
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Local Ollama evaluation failed: {e}")
            raise

class ResponseEvaluator:
    """Main response evaluator for A/B test results."""
    
    def __init__(self, results_file: Path, output_dir: Path):
        self.results_file = results_file
        self.output_dir = output_dir
        self.judge = LLMJudge()
        
        # Load test results
        with open(results_file, 'r') as f:
            self.results_data = json.load(f)
        
        # Load benchmark dataset for prompt details
        dataset_file = output_dir / "benchmark_dataset.json"
        with open(dataset_file, 'r') as f:
            self.dataset = json.load(f)
        
        logger.info(f"Loaded {len(self.results_data)} test results for evaluation")
    
    def group_results_by_prompt(self) -> Dict[str, Dict[str, Any]]:
        """Group results by prompt ID for comparison."""
        grouped = {}
        
        for result in self.results_data:
            prompt_id = result['prompt_id']
            test_arm = result['test_arm']
            
            if prompt_id not in grouped:
                grouped[prompt_id] = {}
            
            grouped[prompt_id][test_arm] = result
        
        return grouped
    
    def evaluate_all_responses(self) -> List[Dict[str, Any]]:
        """Evaluate all responses using LLM judge."""
        logger.info("🔍 Starting LLM-as-a-Judge Evaluation")
        logger.info("=" * 60)
        
        grouped_results = self.group_results_by_prompt()
        evaluation_results = []
        
        total_prompts = len(grouped_results)
        
        for i, (prompt_id, prompt_results) in enumerate(grouped_results.items()):
            logger.info(f"\n📋 Evaluating Prompt {i+1}/{total_prompts}: {prompt_id}")
            
            # Get prompt text
            prompt_text = None
            for prompt_data in self.dataset['prompts']:
                if prompt_data['prompt_id'] == prompt_id:
                    prompt_text = prompt_data['text']
                    break
            
            if not prompt_text:
                logger.warning(f"  ⚠️ Prompt text not found for {prompt_id}")
                continue
            
            # Check if we have all three arms
            if len(prompt_results) != 3:
                logger.warning(f"  ⚠️ Incomplete results for {prompt_id}: {len(prompt_results)} arms")
                continue
            
            # Extract responses for evaluation
            responses = {}
            for arm, result in prompt_results.items():
                if result.get('error'):
                    logger.warning(f"  ⚠️ Error in arm {arm}: {result['error']}")
                    responses[arm] = f"[ERROR: {result['error']}]"
                else:
                    responses[arm] = result['response_text']
            
            logger.info(f"  🔍 Evaluating responses from {len(responses)} arms...")
            
            try:
                # Evaluate responses
                quality_scores = self.judge.evaluate_responses(prompt_text, responses)
                
                # Create evaluation result
                eval_result = {
                    'prompt_id': prompt_id,
                    'prompt_text': prompt_text,
                    'evaluation_timestamp': time.time(),
                    'arm_scores': {}
                }
                
                for arm, score in quality_scores.items():
                    eval_result['arm_scores'][arm] = {
                        'correctness': score.correctness,
                        'conciseness': score.conciseness,
                        'completeness': score.completeness,
                        'overall': score.overall,
                        'reasoning': score.reasoning
                    }
                    
                    logger.info(f"    {arm}: Overall={score.overall}/5, Conciseness={score.conciseness}/5")
                
                evaluation_results.append(eval_result)
                
            except Exception as e:
                logger.error(f"  ❌ Evaluation failed for {prompt_id}: {e}")
        
        logger.info(f"\n✅ Evaluation Complete!")
        logger.info(f"📊 Successfully evaluated {len(evaluation_results)} prompts")
        
        return evaluation_results
    
    def save_evaluation_results(self, evaluation_results: List[Dict[str, Any]]):
        """Save evaluation results to file."""
        output_file = self.output_dir / f"evaluation_results_{int(time.time())}.json"
        
        with open(output_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        logger.info(f"📁 Evaluation results saved to: {output_file}")
        return output_file
    
    def merge_results_with_evaluations(self, evaluation_results: List[Dict[str, Any]]):
        """Merge quantitative results with qualitative evaluations."""
        # Create evaluation lookup
        eval_lookup = {result['prompt_id']: result for result in evaluation_results}
        
        # Add evaluation scores to original results
        enhanced_results = []
        for result in self.results_data:
            enhanced_result = result.copy()
            
            prompt_id = result['prompt_id']
            test_arm = result['test_arm']
            
            if prompt_id in eval_lookup:
                eval_data = eval_lookup[prompt_id]
                if test_arm in eval_data['arm_scores']:
                    scores = eval_data['arm_scores'][test_arm]
                    enhanced_result.update({
                        'quality_correctness': scores['correctness'],
                        'quality_conciseness': scores['conciseness'],
                        'quality_completeness': scores['completeness'],
                        'quality_overall': scores['overall'],
                        'quality_reasoning': scores['reasoning']
                    })
            
            enhanced_results.append(enhanced_result)
        
        # Save enhanced results
        enhanced_file = self.output_dir / f"enhanced_results_{int(time.time())}.json"
        with open(enhanced_file, 'w') as f:
            json.dump(enhanced_results, f, indent=2)
        
        logger.info(f"📁 Enhanced results saved to: {enhanced_file}")
        return enhanced_file

def main():
    """Main evaluation function."""
    if len(sys.argv) != 2:
        logger.error("Usage: python evaluate_responses.py <results_file.json>")
        return 1
    
    results_file = Path(sys.argv[1])
    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        return 1
    
    output_dir = results_file.parent
    
    logger.info("🚀 Starting Phase 3: LLM-as-a-Judge Evaluation")
    
    # Create evaluator
    evaluator = ResponseEvaluator(results_file, output_dir)
    
    # Run evaluation
    evaluation_results = evaluator.evaluate_all_responses()
    
    # Save evaluation results
    eval_file = evaluator.save_evaluation_results(evaluation_results)
    
    # Merge with original results
    enhanced_file = evaluator.merge_results_with_evaluations(evaluation_results)
    
    logger.info("🎉 Response Evaluation Complete!")
    logger.info(f"📊 Ready for final analysis and reporting")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
