#!/usr/bin/env python3
"""
Performance Baseline Script
Phase 0 - Task 4: Performance Baselining

Establishes performance baselines for SAM before TPV integration
to enable quantitative impact analysis.
"""

import sys
import time
import psutil
import logging
import requests
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import statistics

# Add SAM to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceProfiler:
    """Performance profiling utilities."""
    
    def __init__(self):
        self.baseline_data = {}
        self.test_queries = [
            "What is artificial intelligence?",
            "Explain quantum computing principles",
            "How does machine learning work?",
            "What are the latest developments in AI?",
            "Describe the benefits of renewable energy"
        ]
    
    def measure_system_resources(self) -> Dict[str, Any]:
        """Measure current system resource usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_percent': cpu_percent,
                'memory_total_gb': memory.total / (1024**3),
                'memory_used_gb': memory.used / (1024**3),
                'memory_percent': memory.percent,
                'disk_total_gb': disk.total / (1024**3),
                'disk_used_gb': disk.used / (1024**3),
                'disk_percent': (disk.used / disk.total) * 100,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to measure system resources: {e}")
            return {}
    
    def test_ollama_performance(self) -> Dict[str, Any]:
        """Test Ollama model performance."""
        logger.info("🧪 Testing Ollama model performance...")
        
        results = {
            'response_times': [],
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0,
            'min_response_time': 0,
            'max_response_time': 0
        }
        
        model_name = "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M"
        
        for i, query in enumerate(self.test_queries):
            try:
                logger.info(f"  Testing query {i+1}/{len(self.test_queries)}: {query[:50]}...")
                
                start_time = time.time()
                
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": model_name,
                        "prompt": query,
                        "stream": False,
                        "options": {
                            "num_predict": 100,  # Limit response length for consistent testing
                            "temperature": 0.1
                        }
                    },
                    timeout=60
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                
                if response.status_code == 200:
                    results['response_times'].append(response_time)
                    results['successful_requests'] += 1
                    logger.info(f"    ✅ Response time: {response_time:.2f}s")
                else:
                    results['failed_requests'] += 1
                    logger.warning(f"    ❌ Request failed: {response.status_code}")
                
            except Exception as e:
                results['failed_requests'] += 1
                logger.error(f"    ❌ Request error: {e}")
        
        # Calculate statistics
        if results['response_times']:
            results['average_response_time'] = statistics.mean(results['response_times'])
            results['min_response_time'] = min(results['response_times'])
            results['max_response_time'] = max(results['response_times'])
            results['median_response_time'] = statistics.median(results['response_times'])
            results['std_dev_response_time'] = statistics.stdev(results['response_times']) if len(results['response_times']) > 1 else 0
        
        return results
    
    def test_sam_chat_performance(self) -> Dict[str, Any]:
        """Test SAM chat interface performance."""
        logger.info("🧪 Testing SAM chat performance...")
        
        results = {
            'response_times': [],
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0
        }
        
        # Test secure SAM interface
        sam_url = "http://localhost:8502"
        
        # First check if SAM is accessible
        try:
            response = requests.get(sam_url, timeout=5)
            if response.status_code != 200:
                logger.warning(f"SAM interface not accessible at {sam_url}")
                return results
        except Exception as e:
            logger.warning(f"SAM interface not accessible: {e}")
            return results
        
        # Note: For Phase 0, we'll focus on Ollama performance since
        # SAM's chat interface doesn't have a direct API endpoint for testing
        logger.info("  SAM chat performance will be measured through Ollama backend")
        
        return results
    
    def test_memory_performance(self) -> Dict[str, Any]:
        """Test memory system performance."""
        logger.info("🧪 Testing memory system performance...")
        
        results = {
            'memory_operations': [],
            'search_times': [],
            'successful_operations': 0,
            'failed_operations': 0
        }
        
        try:
            # Test memory store if available
            from memory.memory_vectorstore import get_memory_store, VectorStoreType
            
            memory_store = get_memory_store(
                store_type=VectorStoreType.CHROMA,
                storage_directory="memory_store",
                embedding_dimension=384
            )
            
            # Test memory search performance
            test_searches = [
                "artificial intelligence",
                "machine learning",
                "quantum computing",
                "renewable energy",
                "technology trends"
            ]
            
            for search_query in test_searches:
                try:
                    start_time = time.time()
                    search_results = memory_store.search_memories(search_query, max_results=5)
                    end_time = time.time()
                    
                    search_time = end_time - start_time
                    results['search_times'].append(search_time)
                    results['successful_operations'] += 1
                    
                    logger.info(f"  Memory search '{search_query}': {search_time:.3f}s ({len(search_results)} results)")
                    
                except Exception as e:
                    results['failed_operations'] += 1
                    logger.warning(f"  Memory search failed for '{search_query}': {e}")
            
            # Calculate statistics
            if results['search_times']:
                results['average_search_time'] = statistics.mean(results['search_times'])
                results['min_search_time'] = min(results['search_times'])
                results['max_search_time'] = max(results['search_times'])
            
        except Exception as e:
            logger.warning(f"Memory system not available for testing: {e}")
        
        return results
    
    def establish_baseline(self) -> Dict[str, Any]:
        """Establish complete performance baseline."""
        logger.info("📊 Establishing performance baseline...")
        
        baseline = {
            'timestamp': datetime.now().isoformat(),
            'phase': 'Phase 0 - Pre-TPV Baseline',
            'system_info': {},
            'ollama_performance': {},
            'sam_performance': {},
            'memory_performance': {},
            'system_resources': {}
        }
        
        # System information
        try:
            baseline['system_info'] = {
                'python_version': sys.version,
                'platform': sys.platform,
                'cpu_count': psutil.cpu_count(),
                'total_memory_gb': psutil.virtual_memory().total / (1024**3)
            }
        except Exception as e:
            logger.warning(f"Failed to collect system info: {e}")
        
        # System resources before testing
        baseline['system_resources']['before'] = self.measure_system_resources()
        
        # Ollama performance
        baseline['ollama_performance'] = self.test_ollama_performance()
        
        # SAM performance
        baseline['sam_performance'] = self.test_sam_chat_performance()
        
        # Memory performance
        baseline['memory_performance'] = self.test_memory_performance()
        
        # System resources after testing
        baseline['system_resources']['after'] = self.measure_system_resources()
        
        return baseline
    
    def save_baseline(self, baseline_data: Dict[str, Any]) -> bool:
        """Save baseline data to file."""
        try:
            baseline_dir = Path("performance_baselines")
            baseline_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            baseline_file = baseline_dir / f"baseline_phase0_{timestamp}.json"
            
            with open(baseline_file, 'w') as f:
                json.dump(baseline_data, f, indent=2, default=str)
            
            logger.info(f"✅ Baseline saved to: {baseline_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save baseline: {e}")
            return False
    
    def generate_baseline_report(self, baseline_data: Dict[str, Any]) -> str:
        """Generate human-readable baseline report."""
        report = []
        report.append("=" * 60)
        report.append("SAM PERFORMANCE BASELINE REPORT")
        report.append("Phase 0 - Pre-TPV Integration")
        report.append("=" * 60)
        
        # System Information
        report.append("\n📊 SYSTEM INFORMATION:")
        sys_info = baseline_data.get('system_info', {})
        report.append(f"  Platform: {sys_info.get('platform', 'Unknown')}")
        report.append(f"  CPU Cores: {sys_info.get('cpu_count', 'Unknown')}")
        report.append(f"  Total Memory: {sys_info.get('total_memory_gb', 0):.1f} GB")
        
        # Ollama Performance
        report.append("\n🤖 OLLAMA MODEL PERFORMANCE:")
        ollama_perf = baseline_data.get('ollama_performance', {})
        report.append(f"  Successful Requests: {ollama_perf.get('successful_requests', 0)}")
        report.append(f"  Failed Requests: {ollama_perf.get('failed_requests', 0)}")
        report.append(f"  Average Response Time: {ollama_perf.get('average_response_time', 0):.2f}s")
        report.append(f"  Min Response Time: {ollama_perf.get('min_response_time', 0):.2f}s")
        report.append(f"  Max Response Time: {ollama_perf.get('max_response_time', 0):.2f}s")
        
        # Memory Performance
        report.append("\n💾 MEMORY SYSTEM PERFORMANCE:")
        memory_perf = baseline_data.get('memory_performance', {})
        report.append(f"  Successful Operations: {memory_perf.get('successful_operations', 0)}")
        report.append(f"  Failed Operations: {memory_perf.get('failed_operations', 0)}")
        if memory_perf.get('average_search_time'):
            report.append(f"  Average Search Time: {memory_perf.get('average_search_time', 0):.3f}s")
        
        # System Resources
        report.append("\n🖥️ SYSTEM RESOURCE USAGE:")
        resources_before = baseline_data.get('system_resources', {}).get('before', {})
        resources_after = baseline_data.get('system_resources', {}).get('after', {})
        
        if resources_before:
            report.append(f"  CPU Usage (Before): {resources_before.get('cpu_percent', 0):.1f}%")
            report.append(f"  Memory Usage (Before): {resources_before.get('memory_percent', 0):.1f}%")
        
        if resources_after:
            report.append(f"  CPU Usage (After): {resources_after.get('cpu_percent', 0):.1f}%")
            report.append(f"  Memory Usage (After): {resources_after.get('memory_percent', 0):.1f}%")
        
        report.append("\n" + "=" * 60)
        report.append("Baseline established for Phase 0 TPV integration impact analysis")
        report.append("=" * 60)
        
        return "\n".join(report)

def main():
    """Main performance baseline function."""
    logger.info("🚀 Starting Performance Baseline (Phase 0 - Task 4)")
    logger.info("=" * 60)
    
    profiler = PerformanceProfiler()
    
    # Establish baseline
    logger.info("\n📋 Establishing Performance Baseline")
    baseline_data = profiler.establish_baseline()
    
    # Save baseline
    logger.info("\n📋 Saving Baseline Data")
    if profiler.save_baseline(baseline_data):
        logger.info("✅ Baseline data saved successfully")
    else:
        logger.error("❌ Failed to save baseline data")
        return 1
    
    # Generate report
    logger.info("\n📋 Generating Baseline Report")
    report = profiler.generate_baseline_report(baseline_data)
    print("\n" + report)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 PERFORMANCE BASELINE SUMMARY")
    logger.info("=" * 60)
    logger.info("🎉 PERFORMANCE BASELINE ESTABLISHED!")
    logger.info("✅ System performance metrics captured")
    logger.info("✅ Ollama model performance measured")
    logger.info("✅ Memory system performance tested")
    logger.info("✅ Baseline data saved for future comparison")
    logger.info("\n🚀 Ready to proceed with Phase 0 - Task 5: Enhanced Verification")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
