#!/usr/bin/env python3
"""
SAM Model Evaluation Runner
===========================

Automated benchmark runner that executes the SAM Core Benchmark Suite
across multiple models and records comprehensive performance results.

Usage:
    python run_model_evaluation.py --models transformer,llama31-8b
    python run_model_evaluation.py --models all --categories qa,reasoning
    python run_model_evaluation.py --models transformer --output custom_run

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

from sam.benchmarks import BenchmarkLoader, BenchmarkConfig, BenchmarkPrompt
from sam.models.wrappers import get_available_models, create_model_wrapper, create_wrapper_config_template
from sam.config import get_sam_config, ModelBackend
from sam.core.sam_model_client import get_sam_model_client
from sam.core.model_interface import GenerationRequest

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluationRunner:
    """Runs benchmark evaluation across multiple models."""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.benchmark_loader = BenchmarkLoader()
        self.config = BenchmarkConfig()
        
        # Results storage
        self.results = []
        self.run_metadata = {
            "timestamp": datetime.now().isoformat(),
            "benchmark_version": "1.0.0",
            "total_prompts": 0,
            "models_tested": [],
            "categories_tested": [],
            "config": self.config.__dict__
        }
        
    def load_benchmarks(self, categories: Optional[List[str]] = None) -> List[BenchmarkPrompt]:
        """Load benchmark prompts, optionally filtered by categories."""
        logger.info("📚 Loading benchmark suite...")
        
        prompts = self.benchmark_loader.load_benchmarks()
        
        if categories:
            prompts = [p for p in prompts if p.category in categories]
            logger.info(f"Filtered to {len(prompts)} prompts in categories: {categories}")
        
        self.run_metadata["total_prompts"] = len(prompts)
        self.run_metadata["categories_tested"] = list(set(p.category for p in prompts))
        
        # Show category breakdown
        category_stats = {}
        for prompt in prompts:
            category_stats[prompt.category] = category_stats.get(prompt.category, 0) + 1
        
        logger.info("📊 Benchmark breakdown:")
        for category, count in category_stats.items():
            logger.info(f"  {category}: {count} prompts")
        
        return prompts
    
    def get_available_models_for_testing(self) -> List[str]:
        """Get list of available models for testing."""
        models = ["transformer", "hybrid"]  # Built-in models
        
        try:
            dynamic_models = get_available_models()
            models.extend(dynamic_models)
        except Exception as e:
            logger.warning(f"Could not load dynamic models: {e}")
        
        return list(set(models))  # Remove duplicates
    
    def initialize_model(self, model_name: str) -> Any:
        """Initialize a model for testing."""
        logger.info(f"🔧 Initializing model: {model_name}")
        
        try:
            if model_name in ["transformer", "hybrid"]:
                # Use SAM's built-in model client
                client = get_sam_model_client()
                
                # Switch to the appropriate model
                if model_name == "hybrid":
                    client.switch_to_hybrid_model()
                else:
                    client.switch_to_transformer_model()
                
                return {"type": "sam_client", "client": client, "name": model_name}
            
            else:
                # Use dynamic model wrapper
                config = create_wrapper_config_template(model_name)
                config.update({
                    "device": "cpu",  # Use CPU for evaluation to avoid memory issues
                    "load_in_4bit": True,  # Use quantization
                    "temperature": self.config.temperature
                })
                
                wrapper = create_model_wrapper(model_name, config)
                
                # Load the model
                if not wrapper.load_model():
                    raise RuntimeError(f"Failed to load model: {model_name}")
                
                return {"type": "wrapper", "wrapper": wrapper, "name": model_name}
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize {model_name}: {e}")
            return None
    
    def run_single_evaluation(self, model_info: Dict, prompt: BenchmarkPrompt) -> Dict[str, Any]:
        """Run a single prompt evaluation on a model."""
        start_time = time.time()
        
        try:
            # Prepare generation request
            generation_request = GenerationRequest(
                prompt=prompt.prompt,
                max_tokens=prompt.max_tokens,
                temperature=self.config.temperature
            )
            
            # Generate response based on model type
            if model_info["type"] == "sam_client":
                response_text = model_info["client"].generate(
                    prompt.prompt, 
                    max_tokens=prompt.max_tokens
                )
                
                # Create mock response object for consistency
                response = {
                    "text": response_text,
                    "finish_reason": "stop",
                    "usage": {
                        "prompt_tokens": len(prompt.prompt.split()),
                        "completion_tokens": len(response_text.split()),
                        "total_tokens": len(prompt.prompt.split()) + len(response_text.split())
                    }
                }
                
            else:  # wrapper
                response_obj = model_info["wrapper"].generate(generation_request)
                response = {
                    "text": response_obj.text,
                    "finish_reason": response_obj.finish_reason,
                    "usage": response_obj.usage or {}
                }
            
            inference_time = time.time() - start_time
            
            # Create result record
            result = {
                "prompt_id": prompt.id,
                "category": prompt.category,
                "model_name": model_info["name"],
                "prompt": prompt.prompt,
                "response": response["text"],
                "expected_type": prompt.expected_type,
                "scoring_criteria": prompt.scoring_criteria,
                "performance": {
                    "inference_time": inference_time,
                    "prompt_tokens": response["usage"].get("prompt_tokens", 0),
                    "completion_tokens": response["usage"].get("completion_tokens", 0),
                    "total_tokens": response["usage"].get("total_tokens", 0),
                    "tokens_per_second": response["usage"].get("completion_tokens", 0) / inference_time if inference_time > 0 else 0,
                    "finish_reason": response["finish_reason"]
                },
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "error": None
            }
            
            # Add expected function for tool use prompts
            if hasattr(prompt, 'expected_function') and prompt.expected_function:
                result["expected_function"] = prompt.expected_function
            
            logger.info(f"✅ {prompt.id} ({model_info['name']}): {inference_time:.2f}s, {response['usage'].get('completion_tokens', 0)} tokens")
            
            return result
            
        except Exception as e:
            inference_time = time.time() - start_time
            error_msg = str(e)
            
            logger.error(f"❌ {prompt.id} ({model_info['name']}): {error_msg}")
            
            return {
                "prompt_id": prompt.id,
                "category": prompt.category,
                "model_name": model_info["name"],
                "prompt": prompt.prompt,
                "response": "",
                "expected_type": prompt.expected_type,
                "scoring_criteria": prompt.scoring_criteria,
                "performance": {
                    "inference_time": inference_time,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "tokens_per_second": 0,
                    "finish_reason": "error"
                },
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": error_msg
            }
    
    def run_model_evaluation(self, model_names: List[str], prompts: List[BenchmarkPrompt]) -> List[Dict[str, Any]]:
        """Run evaluation across multiple models."""
        results = []
        
        for model_name in model_names:
            logger.info(f"🚀 Starting evaluation for model: {model_name}")
            
            # Initialize model
            model_info = self.initialize_model(model_name)
            if model_info is None:
                logger.error(f"❌ Skipping {model_name} due to initialization failure")
                continue
            
            model_results = []
            
            try:
                # Run all prompts for this model
                for i, prompt in enumerate(prompts, 1):
                    logger.info(f"📝 Running prompt {i}/{len(prompts)}: {prompt.id}")
                    
                    result = self.run_single_evaluation(model_info, prompt)
                    model_results.append(result)
                    results.append(result)
                    
                    # Small delay to prevent overwhelming the model
                    time.sleep(0.1)
                
                # Calculate model summary
                successful_runs = [r for r in model_results if r["success"]]
                total_time = sum(r["performance"]["inference_time"] for r in model_results)
                total_tokens = sum(r["performance"]["completion_tokens"] for r in successful_runs)
                
                logger.info(f"📊 {model_name} Summary:")
                logger.info(f"  Successful: {len(successful_runs)}/{len(prompts)}")
                logger.info(f"  Total time: {total_time:.2f}s")
                logger.info(f"  Total tokens: {total_tokens}")
                logger.info(f"  Avg tokens/sec: {total_tokens/total_time:.2f}" if total_time > 0 else "  Avg tokens/sec: N/A")
                
            finally:
                # Clean up model
                if model_info["type"] == "wrapper":
                    try:
                        model_info["wrapper"].unload_model()
                        logger.info(f"🧹 Unloaded {model_name}")
                    except Exception as e:
                        logger.warning(f"Failed to unload {model_name}: {e}")
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_filename: Optional[str] = None) -> str:
        """Save evaluation results to file."""
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"run_{timestamp}.jsonl"
        
        output_path = self.output_dir / output_filename
        
        # Update metadata
        self.run_metadata["models_tested"] = list(set(r["model_name"] for r in results))
        self.run_metadata["total_results"] = len(results)
        self.run_metadata["output_file"] = str(output_path)
        
        # Save results
        with open(output_path, 'w') as f:
            # Write metadata as first line
            f.write(json.dumps({"metadata": self.run_metadata}) + "\n")
            
            # Write results
            for result in results:
                f.write(json.dumps(result) + "\n")
        
        logger.info(f"💾 Results saved to: {output_path}")
        return str(output_path)
    
    def run_evaluation(self, 
                      models: List[str], 
                      categories: Optional[List[str]] = None,
                      output_filename: Optional[str] = None) -> str:
        """Run complete evaluation pipeline."""
        logger.info("🎯 Starting SAM Model Evaluation")
        logger.info(f"Models: {models}")
        logger.info(f"Categories: {categories or 'all'}")
        
        # Load benchmarks
        prompts = self.load_benchmarks(categories)
        
        if not prompts:
            raise ValueError("No prompts loaded for evaluation")
        
        # Run evaluation
        results = self.run_model_evaluation(models, prompts)
        
        # Save results
        output_path = self.save_results(results, output_filename)
        
        # Print summary
        self.print_evaluation_summary(results)
        
        return output_path
    
    def print_evaluation_summary(self, results: List[Dict[str, Any]]):
        """Print evaluation summary."""
        print("\n" + "="*60)
        print("SAM MODEL EVALUATION SUMMARY")
        print("="*60)
        
        # Overall stats
        total_results = len(results)
        successful_results = len([r for r in results if r["success"]])
        
        print(f"Total Evaluations: {total_results}")
        print(f"Successful: {successful_results}")
        print(f"Failed: {total_results - successful_results}")
        print(f"Success Rate: {(successful_results/total_results*100):.1f}%")
        
        # Per-model breakdown
        models = list(set(r["model_name"] for r in results))
        print(f"\nModels Tested: {len(models)}")
        
        for model in models:
            model_results = [r for r in results if r["model_name"] == model]
            model_successful = len([r for r in model_results if r["success"]])
            avg_time = sum(r["performance"]["inference_time"] for r in model_results) / len(model_results)
            avg_tokens = sum(r["performance"]["completion_tokens"] for r in model_results if r["success"]) / max(model_successful, 1)
            
            print(f"  {model}:")
            print(f"    Success: {model_successful}/{len(model_results)} ({model_successful/len(model_results)*100:.1f}%)")
            print(f"    Avg Time: {avg_time:.2f}s")
            print(f"    Avg Tokens: {avg_tokens:.1f}")
        
        print("="*60)

def main():
    """Main evaluation execution."""
    parser = argparse.ArgumentParser(description="SAM Model Evaluation Runner")
    
    # Get available models
    try:
        available_models = ["transformer", "hybrid"] + get_available_models()
    except:
        available_models = ["transformer", "hybrid"]
    
    parser.add_argument("--models", required=True,
                       help=f"Comma-separated list of models to test. Available: {', '.join(available_models)}, or 'all'")
    parser.add_argument("--categories", 
                       help="Comma-separated list of categories to test (default: all)")
    parser.add_argument("--output", 
                       help="Output filename (default: auto-generated)")
    parser.add_argument("--output-dir", default="evaluation_results",
                       help="Output directory (default: evaluation_results)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Parse models
        if args.models.lower() == "all":
            models = available_models
        else:
            models = [m.strip() for m in args.models.split(",")]
        
        # Validate models
        invalid_models = [m for m in models if m not in available_models]
        if invalid_models:
            logger.error(f"Invalid models: {invalid_models}")
            logger.error(f"Available models: {available_models}")
            return 1
        
        # Parse categories
        categories = None
        if args.categories:
            categories = [c.strip() for c in args.categories.split(",")]
        
        # Run evaluation
        runner = ModelEvaluationRunner(args.output_dir)
        output_path = runner.run_evaluation(models, categories, args.output)
        
        print(f"\n🎉 Evaluation complete! Results saved to: {output_path}")
        print("Next step: Run scoring with 'python scripts/score_evaluation_results.py'")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Evaluation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        if args.debug:
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
