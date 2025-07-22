#!/usr/bin/env python3
"""
TPV Performance Profiling Script
Phase 2 - Task 1a: GPU Acceleration & Performance Optimization

Profiles the TPV pipeline to identify bottlenecks and optimize performance.
"""

import sys
import time
import logging
import cProfile
import pstats
import io
import torch
from pathlib import Path
from typing import Dict, Any, List

# Add SAM to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TPVPerformanceProfiler:
    """Comprehensive TPV performance profiler."""
    
    def __init__(self):
        self.profile_results = {}
        self.device_info = self._get_device_info()
        
    def _get_device_info(self) -> Dict[str, Any]:
        """Get device and GPU information."""
        device_info = {
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'current_device': 'cpu'
        }
        
        if torch.cuda.is_available():
            device_info.update({
                'current_device': f'cuda:{torch.cuda.current_device()}',
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory / 1e9,
                'gpu_memory_allocated': torch.cuda.memory_allocated(0) / 1e9,
                'gpu_memory_cached': torch.cuda.memory_reserved(0) / 1e9
            })
        
        return device_info
    
    def profile_tpv_initialization(self) -> Dict[str, float]:
        """Profile TPV component initialization times."""
        logger.info("🔍 Profiling TPV initialization...")
        
        results = {}
        
        # Profile TPV Monitor initialization
        start_time = time.time()
        from sam.cognition.tpv import TPVMonitor
        monitor = TPVMonitor()
        results['monitor_creation'] = time.time() - start_time
        
        start_time = time.time()
        monitor.initialize()
        results['monitor_initialization'] = time.time() - start_time
        
        # Profile TPV Core initialization
        start_time = time.time()
        from sam.cognition.tpv import TPVCore
        core = TPVCore()
        results['core_creation'] = time.time() - start_time
        
        start_time = time.time()
        core.initialize()
        results['core_initialization'] = time.time() - start_time
        
        # Profile Integration initialization
        start_time = time.time()
        from sam.cognition.tpv import SAMTPVIntegration
        integration = SAMTPVIntegration()
        results['integration_creation'] = time.time() - start_time
        
        start_time = time.time()
        integration.initialize()
        results['integration_initialization'] = time.time() - start_time
        
        return results
    
    def profile_tpv_inference(self, num_iterations: int = 10) -> Dict[str, Any]:
        """Profile TPV inference performance."""
        logger.info(f"🔍 Profiling TPV inference ({num_iterations} iterations)...")
        
        from sam.cognition.tpv import TPVMonitor
        
        monitor = TPVMonitor()
        monitor.initialize()
        
        # Test data
        test_responses = [
            "This is a test response for performance profiling.",
            "This is a test response for performance profiling. It contains more content to analyze.",
            "This is a test response for performance profiling. It contains more content to analyze. The reasoning process involves multiple steps and complex analysis.",
            "This is a test response for performance profiling. It contains more content to analyze. The reasoning process involves multiple steps and complex analysis. We need to measure the performance impact of TPV monitoring on response generation.",
            "This is a test response for performance profiling. It contains more content to analyze. The reasoning process involves multiple steps and complex analysis. We need to measure the performance impact of TPV monitoring on response generation. The goal is to optimize the system for production use."
        ]
        
        # Profile individual components
        results = {
            'total_times': [],
            'predict_progress_times': [],
            'tpv_core_times': [],
            'synthetic_states_times': [],
            'text_analysis_times': []
        }
        
        for iteration in range(num_iterations):
            query_id = monitor.start_monitoring(f"test_query_{iteration}")
            
            for i, response in enumerate(test_responses):
                # Time total predict_progress call
                start_time = time.time()
                score = monitor.predict_progress(response, query_id, token_count=(i+1)*20)
                total_time = time.time() - start_time
                results['total_times'].append(total_time)
                
                # Profile individual components (approximate)
                # Note: This is a simplified profiling - detailed profiling would require
                # instrumenting the actual TPV methods
                
            monitor.stop_monitoring(query_id)
        
        # Calculate statistics
        for key in ['total_times']:
            times = results[key]
            results[f'{key}_stats'] = {
                'mean': sum(times) / len(times),
                'min': min(times),
                'max': max(times),
                'total': sum(times)
            }
        
        return results
    
    def profile_device_performance(self) -> Dict[str, Any]:
        """Profile performance on different devices (CPU vs GPU)."""
        logger.info("🔍 Profiling device performance...")
        
        from sam.cognition.tpv import TPVCore
        
        results = {}
        
        # Test on CPU
        logger.info("Testing CPU performance...")
        cpu_core = TPVCore()
        cpu_core.initialize()
        
        # Create test tensor on CPU
        hidden_dim = cpu_core.config.get_hidden_dimension()
        test_tensor_cpu = torch.randn(1, 10, hidden_dim)
        
        cpu_times = []
        for _ in range(5):
            start_time = time.time()
            cpu_result = cpu_core.process_thinking(test_tensor_cpu)
            cpu_times.append(time.time() - start_time)
        
        results['cpu'] = {
            'mean_time': sum(cpu_times) / len(cpu_times),
            'device': 'cpu',
            'tensor_device': str(test_tensor_cpu.device)
        }
        
        # Test on GPU if available
        if torch.cuda.is_available():
            logger.info("Testing GPU performance...")
            
            # Move to GPU
            test_tensor_gpu = test_tensor_cpu.cuda()
            
            # Note: The current TPV core doesn't automatically move to GPU
            # This is one of the optimization opportunities
            
            gpu_times = []
            for _ in range(5):
                start_time = time.time()
                # Process on CPU (current implementation)
                gpu_result = cpu_core.process_thinking(test_tensor_gpu.cpu())
                gpu_times.append(time.time() - start_time)
            
            results['gpu'] = {
                'mean_time': sum(gpu_times) / len(gpu_times),
                'device': 'cuda',
                'tensor_device': str(test_tensor_gpu.device),
                'note': 'TPV core still processing on CPU - optimization needed'
            }
            
            # Calculate potential speedup
            if results['cpu']['mean_time'] > 0:
                results['potential_speedup'] = results['cpu']['mean_time'] / results['gpu']['mean_time']
        else:
            results['gpu'] = {'available': False}
        
        return results
    
    def profile_memory_usage(self) -> Dict[str, Any]:
        """Profile memory usage during TPV operations."""
        logger.info("🔍 Profiling memory usage...")
        
        import psutil
        import gc
        
        process = psutil.Process()
        
        # Baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1e6  # MB
        
        results = {
            'baseline_memory_mb': baseline_memory,
            'peak_memory_mb': baseline_memory,
            'memory_growth_mb': 0
        }
        
        if torch.cuda.is_available():
            baseline_gpu_memory = torch.cuda.memory_allocated(0) / 1e6  # MB
            results['baseline_gpu_memory_mb'] = baseline_gpu_memory
        
        # Initialize TPV system
        from sam.cognition.tpv import SAMTPVIntegration
        integration = SAMTPVIntegration()
        integration.initialize()
        
        # Memory after initialization
        current_memory = process.memory_info().rss / 1e6
        results['post_init_memory_mb'] = current_memory
        results['init_memory_growth_mb'] = current_memory - baseline_memory
        
        if torch.cuda.is_available():
            current_gpu_memory = torch.cuda.memory_allocated(0) / 1e6
            results['post_init_gpu_memory_mb'] = current_gpu_memory
            results['init_gpu_memory_growth_mb'] = current_gpu_memory - baseline_gpu_memory
        
        # Memory during operation
        peak_memory = current_memory
        for i in range(10):
            # Simulate TPV operation
            trigger_result = integration.tpv_trigger.should_activate_tpv(
                f"Test query {i} for memory profiling",
                initial_confidence=0.5
            )
            
            current_memory = process.memory_info().rss / 1e6
            peak_memory = max(peak_memory, current_memory)
        
        results['peak_memory_mb'] = peak_memory
        results['memory_growth_mb'] = peak_memory - baseline_memory
        
        return results
    
    def run_comprehensive_profile(self) -> Dict[str, Any]:
        """Run comprehensive performance profiling."""
        logger.info("🚀 Starting Comprehensive TPV Performance Profiling")
        logger.info("=" * 60)
        
        comprehensive_results = {
            'device_info': self.device_info,
            'timestamp': time.time()
        }
        
        # Profile initialization
        logger.info("\n📋 Profiling Initialization Performance")
        comprehensive_results['initialization'] = self.profile_tpv_initialization()
        
        # Profile inference
        logger.info("\n📋 Profiling Inference Performance")
        comprehensive_results['inference'] = self.profile_tpv_inference()
        
        # Profile device performance
        logger.info("\n📋 Profiling Device Performance")
        comprehensive_results['device_performance'] = self.profile_device_performance()
        
        # Profile memory usage
        logger.info("\n📋 Profiling Memory Usage")
        comprehensive_results['memory'] = self.profile_memory_usage()
        
        return comprehensive_results
    
    def generate_optimization_report(self, results: Dict[str, Any]) -> str:
        """Generate optimization recommendations based on profiling results."""
        report = []
        report.append("=" * 60)
        report.append("TPV PERFORMANCE OPTIMIZATION REPORT")
        report.append("=" * 60)
        
        # Device Information
        report.append("\n📊 DEVICE INFORMATION:")
        device_info = results['device_info']
        report.append(f"  CUDA Available: {device_info['cuda_available']}")
        if device_info['cuda_available']:
            report.append(f"  GPU: {device_info.get('gpu_name', 'Unknown')}")
            report.append(f"  GPU Memory: {device_info.get('gpu_memory_total', 0):.1f} GB")
        
        # Initialization Performance
        report.append("\n📊 INITIALIZATION PERFORMANCE:")
        init_results = results['initialization']
        total_init_time = sum(init_results.values())
        report.append(f"  Total Initialization Time: {total_init_time:.3f}s")
        for component, time_taken in init_results.items():
            report.append(f"  {component}: {time_taken:.3f}s ({time_taken/total_init_time*100:.1f}%)")
        
        # Inference Performance
        report.append("\n📊 INFERENCE PERFORMANCE:")
        inference_results = results['inference']
        if 'total_times_stats' in inference_results:
            stats = inference_results['total_times_stats']
            report.append(f"  Mean Inference Time: {stats['mean']:.4f}s")
            report.append(f"  Min/Max: {stats['min']:.4f}s / {stats['max']:.4f}s")
        
        # Memory Usage
        report.append("\n📊 MEMORY USAGE:")
        memory_results = results['memory']
        report.append(f"  Baseline Memory: {memory_results['baseline_memory_mb']:.1f} MB")
        report.append(f"  Peak Memory: {memory_results['peak_memory_mb']:.1f} MB")
        report.append(f"  Memory Growth: {memory_results['memory_growth_mb']:.1f} MB")
        
        # Optimization Recommendations
        report.append("\n🚀 OPTIMIZATION RECOMMENDATIONS:")
        
        # GPU Acceleration
        if device_info['cuda_available']:
            device_perf = results.get('device_performance', {})
            if 'gpu' in device_perf and device_perf['gpu'].get('note'):
                report.append("  🔥 HIGH PRIORITY: GPU Acceleration")
                report.append("    - TPV core is currently processing on CPU")
                report.append("    - Move TPV processor to GPU for 10-50x speedup")
                report.append("    - Ensure tensor operations stay on GPU")
        
        # Memory Optimization
        if memory_results['memory_growth_mb'] > 100:
            report.append("  📈 MEDIUM PRIORITY: Memory Optimization")
            report.append(f"    - Memory growth: {memory_results['memory_growth_mb']:.1f} MB")
            report.append("    - Consider tensor caching and reuse")
            report.append("    - Implement garbage collection between operations")
        
        # Initialization Optimization
        if total_init_time > 2.0:
            report.append("  ⚡ LOW PRIORITY: Initialization Optimization")
            report.append(f"    - Total init time: {total_init_time:.3f}s")
            report.append("    - Consider lazy loading of TPV components")
            report.append("    - Cache initialized models")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)

def main():
    """Main profiling function."""
    profiler = TPVPerformanceProfiler()
    
    # Run comprehensive profiling
    results = profiler.run_comprehensive_profile()
    
    # Generate and display report
    report = profiler.generate_optimization_report(results)
    print(report)
    
    # Save results
    import json
    results_file = Path("performance_profiles") / f"tpv_profile_{int(time.time())}.json"
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"📁 Detailed results saved to: {results_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
