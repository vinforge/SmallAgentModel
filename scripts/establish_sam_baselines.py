#!/usr/bin/env python3
"""
SAM 2.0 Phase 0: Baseline Establishment Script
==============================================

This script establishes comprehensive baselines for the current SAM model
to quantify the problems we're solving with the Hybrid Linear Attention upgrade.

Key Metrics:
- Maximum context window before OOM
- Performance degradation curve as context increases
- Memory usage patterns
- Inference speed benchmarks
- Current model architecture details

Author: SAM Development Team
Version: 1.0.0
"""

import os
import sys
import json
import time
import psutil
import logging
import requests
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SAMBaselineEstablisher:
    """Establishes comprehensive baselines for current SAM model."""
    
    def __init__(self):
        """Initialize the baseline establishment system."""
        self.ollama_url = "http://localhost:11434"
        self.model_name = "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M"
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'model_info': {},
            'context_limits': {},
            'performance_metrics': {},
            'memory_usage': {},
            'architecture_details': {}
        }
        
        logger.info("SAM Baseline Establisher initialized")
    
    def check_ollama_availability(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Ollama is accessible")
                return True
            else:
                logger.error(f"❌ Ollama returned status code: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Cannot connect to Ollama: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the current model."""
        try:
            # Get model list
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                current_model = None
                
                for model in models:
                    if self.model_name in model.get('name', ''):
                        current_model = model
                        break
                
                if current_model:
                    model_info = {
                        'name': current_model.get('name'),
                        'size': current_model.get('size', 0),
                        'digest': current_model.get('digest'),
                        'modified_at': current_model.get('modified_at'),
                        'details': current_model.get('details', {})
                    }
                    
                    # Try to get more detailed model info
                    try:
                        show_response = requests.post(
                            f"{self.ollama_url}/api/show",
                            json={"name": self.model_name}
                        )
                        if show_response.status_code == 200:
                            show_data = show_response.json()
                            model_info.update({
                                'parameters': show_data.get('parameters', {}),
                                'template': show_data.get('template', ''),
                                'system': show_data.get('system', ''),
                                'modelfile': show_data.get('modelfile', '')
                            })
                    except Exception as e:
                        logger.warning(f"Could not get detailed model info: {e}")
                    
                    logger.info(f"✅ Model info retrieved: {model_info['name']}")
                    return model_info
                else:
                    logger.error(f"❌ Model {self.model_name} not found")
                    return {}
            else:
                logger.error(f"❌ Failed to get model list: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Error getting model info: {e}")
            return {}
    
    def test_context_window_limits(self) -> Dict[str, Any]:
        """Test the maximum context window before OOM or failure."""
        logger.info("🔍 Testing context window limits...")
        
        context_results = {
            'max_successful_tokens': 0,
            'failure_point': None,
            'performance_curve': [],
            'memory_curve': []
        }
        
        # Test with increasing context sizes
        test_sizes = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]
        
        for size in test_sizes:
            logger.info(f"Testing context size: {size} tokens")
            
            # Create test prompt with approximately the target token count
            # Rough estimate: 1 token ≈ 4 characters
            test_text = "This is a test sentence for context window evaluation. " * (size // 10)
            prompt = f"Please summarize the following text in one sentence:\n\n{test_text}\n\nSummary:"
            
            # Measure memory before test
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            start_time = time.time()
            success = False
            error_msg = None
            
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "top_p": 0.9,
                            "max_tokens": 100
                        }
                    },
                    timeout=120  # 2 minute timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('response'):
                        success = True
                        context_results['max_successful_tokens'] = size
                        
                        # Measure performance
                        end_time = time.time()
                        inference_time = end_time - start_time
                        
                        # Measure memory after test
                        memory_after = process.memory_info().rss / 1024 / 1024  # MB
                        memory_used = memory_after - memory_before
                        
                        context_results['performance_curve'].append({
                            'tokens': size,
                            'inference_time': inference_time,
                            'success': True
                        })
                        
                        context_results['memory_curve'].append({
                            'tokens': size,
                            'memory_mb': memory_used,
                            'total_memory_mb': memory_after
                        })
                        
                        logger.info(f"✅ Success at {size} tokens - Time: {inference_time:.2f}s, Memory: {memory_used:.1f}MB")
                    else:
                        error_msg = "Empty response"
                else:
                    error_msg = f"HTTP {response.status_code}"
                    
            except requests.exceptions.Timeout:
                error_msg = "Request timeout"
            except Exception as e:
                error_msg = str(e)
            
            if not success:
                logger.warning(f"❌ Failed at {size} tokens: {error_msg}")
                context_results['failure_point'] = {
                    'tokens': size,
                    'error': error_msg
                }
                context_results['performance_curve'].append({
                    'tokens': size,
                    'inference_time': None,
                    'success': False,
                    'error': error_msg
                })
                break
            
            # Small delay between tests
            time.sleep(2)
        
        return context_results
    
    def benchmark_inference_speed(self) -> Dict[str, Any]:
        """Benchmark inference speed with various prompt sizes."""
        logger.info("⚡ Benchmarking inference speed...")
        
        speed_results = {
            'short_prompt': {},
            'medium_prompt': {},
            'long_prompt': {}
        }
        
        test_prompts = {
            'short_prompt': "What is artificial intelligence?",
            'medium_prompt': "Explain the concept of machine learning and its applications in modern technology. " * 5,
            'long_prompt': "Provide a comprehensive analysis of the evolution of artificial intelligence from its inception to current state, including key milestones, breakthrough technologies, and future prospects. " * 10
        }
        
        for prompt_type, prompt in test_prompts.items():
            logger.info(f"Testing {prompt_type}...")
            
            times = []
            for i in range(3):  # Run 3 times for average
                start_time = time.time()
                
                try:
                    response = requests.post(
                        f"{self.ollama_url}/api/generate",
                        json={
                            "model": self.model_name,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0.7,
                                "max_tokens": 200
                            }
                        },
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        end_time = time.time()
                        inference_time = end_time - start_time
                        times.append(inference_time)
                        logger.info(f"  Run {i+1}: {inference_time:.2f}s")
                    else:
                        logger.warning(f"  Run {i+1}: Failed with status {response.status_code}")
                        
                except Exception as e:
                    logger.warning(f"  Run {i+1}: Failed with error {e}")
            
            if times:
                speed_results[prompt_type] = {
                    'times': times,
                    'average_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times)
                }
            else:
                speed_results[prompt_type] = {'error': 'All runs failed'}
        
        return speed_results
    
    def analyze_memory_usage(self) -> Dict[str, Any]:
        """Analyze memory usage patterns during inference."""
        logger.info("💾 Analyzing memory usage patterns...")
        
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_results = {
            'baseline_memory_mb': baseline_memory,
            'peak_memory_mb': baseline_memory,
            'memory_efficiency': {}
        }
        
        # Test memory usage with different context sizes
        test_sizes = [1000, 4000, 8000, 16000]
        
        for size in test_sizes:
            test_text = "Memory usage test content. " * (size // 5)
            prompt = f"Analyze this text: {test_text}"
            
            memory_before = process.memory_info().rss / 1024 / 1024
            
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"max_tokens": 50}
                    },
                    timeout=60
                )
                
                memory_after = process.memory_info().rss / 1024 / 1024
                memory_used = memory_after - memory_before
                
                memory_results['memory_efficiency'][f'{size}_tokens'] = {
                    'memory_used_mb': memory_used,
                    'memory_per_token': memory_used / size if size > 0 else 0
                }
                
                if memory_after > memory_results['peak_memory_mb']:
                    memory_results['peak_memory_mb'] = memory_after
                    
            except Exception as e:
                logger.warning(f"Memory test failed for {size} tokens: {e}")
        
        return memory_results
    
    def run_comprehensive_baseline(self) -> Dict[str, Any]:
        """Run comprehensive baseline establishment."""
        logger.info("🚀 Starting comprehensive SAM baseline establishment...")
        
        if not self.check_ollama_availability():
            logger.error("❌ Cannot proceed without Ollama access")
            return {}
        
        # Get model information
        logger.info("📊 Gathering model information...")
        self.results['model_info'] = self.get_model_info()
        
        # Test context window limits
        logger.info("🔍 Testing context window limits...")
        self.results['context_limits'] = self.test_context_window_limits()
        
        # Benchmark inference speed
        logger.info("⚡ Benchmarking inference speed...")
        self.results['performance_metrics'] = self.benchmark_inference_speed()
        
        # Analyze memory usage
        logger.info("💾 Analyzing memory usage...")
        self.results['memory_usage'] = self.analyze_memory_usage()
        
        # Add system information
        self.results['system_info'] = {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'python_version': sys.version,
            'platform': sys.platform
        }
        
        logger.info("✅ Baseline establishment complete!")
        return self.results
    
    def save_results(self, filename: str = None) -> str:
        """Save baseline results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sam_baseline_results_{timestamp}.json"
        
        filepath = Path("logs") / filename
        filepath.parent.mkdir(exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"📁 Results saved to: {filepath}")
        return str(filepath)

def main():
    """Main execution function."""
    try:
        establisher = SAMBaselineEstablisher()
        results = establisher.run_comprehensive_baseline()
        
        if results:
            filepath = establisher.save_results()
            
            # Print summary
            print("\n" + "="*60)
            print("SAM BASELINE ESTABLISHMENT SUMMARY")
            print("="*60)
            
            if 'context_limits' in results:
                max_tokens = results['context_limits'].get('max_successful_tokens', 0)
                print(f"Maximum Context Window: {max_tokens:,} tokens")
                
                if results['context_limits'].get('failure_point'):
                    failure = results['context_limits']['failure_point']
                    print(f"Failure Point: {failure['tokens']:,} tokens ({failure['error']})")
            
            if 'performance_metrics' in results:
                perf = results['performance_metrics']
                if 'short_prompt' in perf and 'average_time' in perf['short_prompt']:
                    print(f"Short Prompt Speed: {perf['short_prompt']['average_time']:.2f}s avg")
                if 'long_prompt' in perf and 'average_time' in perf['long_prompt']:
                    print(f"Long Prompt Speed: {perf['long_prompt']['average_time']:.2f}s avg")
            
            if 'memory_usage' in results:
                mem = results['memory_usage']
                print(f"Baseline Memory: {mem.get('baseline_memory_mb', 0):.1f} MB")
                print(f"Peak Memory: {mem.get('peak_memory_mb', 0):.1f} MB")
            
            print(f"\nDetailed results saved to: {filepath}")
            print("="*60)
        else:
            print("❌ Baseline establishment failed")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
