#!/usr/bin/env python3
"""
SAM Engine Upgrade Framework - Performance Benchmarking
=======================================================

Comprehensive benchmarking script to quantify performance characteristics
of the migration process including timing, resource usage, and model loading.

Author: SAM Development Team
Version: 1.0.0
"""

import os
import sys
import json
import time
import psutil
import logging
import argparse
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import threading

# Add SAM to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """Monitor system resource usage during operations."""
    
    def __init__(self, interval: float = 1.0):
        """
        Initialize resource monitor.
        
        Args:
            interval: Monitoring interval in seconds
        """
        self.interval = interval
        self.monitoring = False
        self.monitor_thread = None
        self.metrics = {
            'cpu_percent': [],
            'memory_mb': [],
            'disk_io_read_mb': [],
            'disk_io_write_mb': [],
            'network_sent_mb': [],
            'network_recv_mb': [],
            'timestamps': []
        }
        
        # Baseline measurements
        self.baseline_disk_io = psutil.disk_io_counters()
        self.baseline_network = psutil.net_io_counters()
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=None)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_mb = memory.used / (1024 * 1024)
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                if disk_io and self.baseline_disk_io:
                    disk_read_mb = (disk_io.read_bytes - self.baseline_disk_io.read_bytes) / (1024 * 1024)
                    disk_write_mb = (disk_io.write_bytes - self.baseline_disk_io.write_bytes) / (1024 * 1024)
                else:
                    disk_read_mb = disk_write_mb = 0
                
                # Network I/O
                network = psutil.net_io_counters()
                if network and self.baseline_network:
                    network_sent_mb = (network.bytes_sent - self.baseline_network.bytes_sent) / (1024 * 1024)
                    network_recv_mb = (network.bytes_recv - self.baseline_network.bytes_recv) / (1024 * 1024)
                else:
                    network_sent_mb = network_recv_mb = 0
                
                # Store metrics
                self.metrics['cpu_percent'].append(cpu_percent)
                self.metrics['memory_mb'].append(memory_mb)
                self.metrics['disk_io_read_mb'].append(disk_read_mb)
                self.metrics['disk_io_write_mb'].append(disk_write_mb)
                self.metrics['network_sent_mb'].append(network_sent_mb)
                self.metrics['network_recv_mb'].append(network_recv_mb)
                self.metrics['timestamps'].append(datetime.now().isoformat())
                
                time.sleep(self.interval)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(self.interval)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of resource usage."""
        if not self.metrics['cpu_percent']:
            return {}
        
        def safe_stats(values):
            if not values:
                return {'min': 0, 'max': 0, 'avg': 0}
            return {
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values)
            }
        
        return {
            'duration_seconds': len(self.metrics['cpu_percent']) * self.interval,
            'cpu_percent': safe_stats(self.metrics['cpu_percent']),
            'memory_mb': safe_stats(self.metrics['memory_mb']),
            'disk_io_read_mb': safe_stats(self.metrics['disk_io_read_mb']),
            'disk_io_write_mb': safe_stats(self.metrics['disk_io_write_mb']),
            'network_sent_mb': safe_stats(self.metrics['network_sent_mb']),
            'network_recv_mb': safe_stats(self.metrics['network_recv_mb']),
            'sample_count': len(self.metrics['cpu_percent'])
        }


class EngineUpgradeBenchmark:
    """Comprehensive benchmarking for the Engine Upgrade Framework."""
    
    def __init__(self, output_dir: str = "./benchmarks"):
        """
        Initialize benchmark suite.
        
        Args:
            output_dir: Directory to store benchmark results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {
            'benchmark_id': f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'system_info': self._get_system_info(),
            'tests': {}
        }
        
        # Setup logging
        log_file = self.output_dir / f"benchmark_{self.results['benchmark_id']}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(f"{__name__}.EngineUpgradeBenchmark")
        self.logger.info(f"Benchmark suite initialized: {self.results['benchmark_id']}")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmark context."""
        try:
            return {
                'platform': sys.platform,
                'python_version': sys.version,
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'disk_total_gb': psutil.disk_usage('/').total / (1024**3),
                'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown'
            }
        except Exception as e:
            self.logger.warning(f"Could not gather system info: {e}")
            return {}
    
    def benchmark_migration_controller_performance(self) -> Dict[str, Any]:
        """Benchmark migration controller operations."""
        self.logger.info("🔄 Benchmarking Migration Controller Performance")
        
        test_results = {
            'test_name': 'migration_controller_performance',
            'start_time': datetime.now().isoformat(),
            'operations': {}
        }
        
        try:
            from sam.core.migration_controller import get_migration_controller
            
            migration_controller = get_migration_controller()
            
            # Benchmark: Create migration plan
            start_time = time.time()
            migration_id = migration_controller.create_migration_plan(
                from_engine="benchmark_engine_1",
                to_engine="benchmark_engine_2",
                user_id="benchmark_user"
            )
            create_plan_time = time.time() - start_time
            
            test_results['operations']['create_migration_plan'] = {
                'duration_seconds': create_plan_time,
                'migration_id': migration_id
            }
            
            # Benchmark: Get migration status
            start_time = time.time()
            status = migration_controller.get_migration_status(migration_id)
            get_status_time = time.time() - start_time
            
            test_results['operations']['get_migration_status'] = {
                'duration_seconds': get_status_time,
                'status_found': status is not None
            }
            
            # Benchmark: Cancel migration
            start_time = time.time()
            cancelled = migration_controller.cancel_migration(migration_id)
            cancel_time = time.time() - start_time
            
            test_results['operations']['cancel_migration'] = {
                'duration_seconds': cancel_time,
                'success': cancelled
            }
            
            test_results['status'] = 'completed'
            self.logger.info("✅ Migration Controller benchmarks completed")
            
        except Exception as e:
            test_results['status'] = 'failed'
            test_results['error'] = str(e)
            self.logger.error(f"❌ Migration Controller benchmarks failed: {e}")
        
        test_results['end_time'] = datetime.now().isoformat()
        return test_results
    
    def benchmark_model_library_operations(self) -> Dict[str, Any]:
        """Benchmark model library manager operations."""
        self.logger.info("📚 Benchmarking Model Library Operations")
        
        test_results = {
            'test_name': 'model_library_operations',
            'start_time': datetime.now().isoformat(),
            'operations': {}
        }
        
        try:
            from sam.core.model_library_manager import get_model_library_manager
            
            library_manager = get_model_library_manager()
            
            # Benchmark: Get available models
            start_time = time.time()
            available_models = library_manager.get_available_models()
            get_available_time = time.time() - start_time
            
            test_results['operations']['get_available_models'] = {
                'duration_seconds': get_available_time,
                'model_count': len(available_models)
            }
            
            # Benchmark: Get downloaded models
            start_time = time.time()
            downloaded_models = library_manager.get_downloaded_models()
            get_downloaded_time = time.time() - start_time
            
            test_results['operations']['get_downloaded_models'] = {
                'duration_seconds': get_downloaded_time,
                'model_count': len(downloaded_models)
            }
            
            # Benchmark: Model status checks
            status_check_times = []
            for model in available_models[:5]:  # Test first 5 models
                start_time = time.time()
                status = library_manager.get_model_status(model.model_id)
                status_time = time.time() - start_time
                status_check_times.append(status_time)
            
            test_results['operations']['model_status_checks'] = {
                'average_duration_seconds': sum(status_check_times) / len(status_check_times) if status_check_times else 0,
                'checks_performed': len(status_check_times)
            }
            
            test_results['status'] = 'completed'
            self.logger.info("✅ Model Library benchmarks completed")
            
        except Exception as e:
            test_results['status'] = 'failed'
            test_results['error'] = str(e)
            self.logger.error(f"❌ Model Library benchmarks failed: {e}")
        
        test_results['end_time'] = datetime.now().isoformat()
        return test_results
    
    def benchmark_ui_response_times(self) -> Dict[str, Any]:
        """Benchmark UI component response times."""
        self.logger.info("🖥️ Benchmarking UI Response Times")
        
        test_results = {
            'test_name': 'ui_response_times',
            'start_time': datetime.now().isoformat(),
            'operations': {}
        }
        
        try:
            # Simulate UI component loading times
            
            # Benchmark: Core Engines UI data loading
            start_time = time.time()
            from sam.core.model_library_manager import get_model_library_manager
            library_manager = get_model_library_manager()
            available_models = library_manager.get_available_models()
            downloaded_models = library_manager.get_downloaded_models()
            ui_data_load_time = time.time() - start_time
            
            test_results['operations']['core_engines_data_load'] = {
                'duration_seconds': ui_data_load_time,
                'available_models': len(available_models),
                'downloaded_models': len(downloaded_models)
            }
            
            # Benchmark: Migration wizard initialization
            start_time = time.time()
            from sam.core.migration_controller import get_migration_controller
            migration_controller = get_migration_controller()
            wizard_init_time = time.time() - start_time
            
            test_results['operations']['migration_wizard_init'] = {
                'duration_seconds': wizard_init_time
            }
            
            # Benchmark: Engine status indicator
            start_time = time.time()
            try:
                from sam.core.model_interface import get_current_model_info
                current_info = get_current_model_info()
            except:
                current_info = {'primary_model': 'test'}
            engine_status_time = time.time() - start_time
            
            test_results['operations']['engine_status_indicator'] = {
                'duration_seconds': engine_status_time
            }
            
            test_results['status'] = 'completed'
            self.logger.info("✅ UI Response Time benchmarks completed")
            
        except Exception as e:
            test_results['status'] = 'failed'
            test_results['error'] = str(e)
            self.logger.error(f"❌ UI Response Time benchmarks failed: {e}")
        
        test_results['end_time'] = datetime.now().isoformat()
        return test_results
    
    def run_full_benchmark_suite(self) -> Dict[str, Any]:
        """Run the complete benchmark suite."""
        self.logger.info("🚀 Starting Full Engine Upgrade Benchmark Suite")
        
        # Resource monitor for overall benchmarking
        monitor = ResourceMonitor(interval=0.5)
        monitor.start_monitoring()
        
        try:
            # Run individual benchmarks
            self.results['tests']['migration_controller'] = self.benchmark_migration_controller_performance()
            self.results['tests']['model_library'] = self.benchmark_model_library_operations()
            self.results['tests']['ui_response'] = self.benchmark_ui_response_times()
            
            # Overall results
            self.results['status'] = 'completed'
            self.results['end_time'] = datetime.now().isoformat()
            
        except Exception as e:
            self.results['status'] = 'failed'
            self.results['error'] = str(e)
            self.logger.error(f"❌ Benchmark suite failed: {e}")
        
        finally:
            monitor.stop_monitoring()
            self.results['resource_usage'] = monitor.get_summary()
        
        # Save results
        self._save_results()
        
        # Generate summary
        self._generate_summary()
        
        return self.results
    
    def _save_results(self):
        """Save benchmark results to file."""
        results_file = self.output_dir / f"results_{self.results['benchmark_id']}.json"
        
        try:
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            self.logger.info(f"✅ Benchmark results saved: {results_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save results: {e}")
    
    def _generate_summary(self):
        """Generate human-readable benchmark summary."""
        summary_file = self.output_dir / f"summary_{self.results['benchmark_id']}.md"
        
        try:
            with open(summary_file, 'w') as f:
                f.write(f"# Engine Upgrade Framework Benchmark Report\n\n")
                f.write(f"**Benchmark ID:** {self.results['benchmark_id']}\n")
                f.write(f"**Timestamp:** {self.results['timestamp']}\n")
                f.write(f"**Status:** {self.results.get('status', 'unknown')}\n\n")
                
                # System info
                f.write("## System Information\n\n")
                for key, value in self.results['system_info'].items():
                    f.write(f"- **{key.replace('_', ' ').title()}:** {value}\n")
                
                # Test results
                f.write("\n## Test Results\n\n")
                for test_name, test_data in self.results['tests'].items():
                    f.write(f"### {test_name.replace('_', ' ').title()}\n\n")
                    f.write(f"- **Status:** {test_data.get('status', 'unknown')}\n")
                    
                    if 'operations' in test_data:
                        f.write("- **Operations:**\n")
                        for op_name, op_data in test_data['operations'].items():
                            duration = op_data.get('duration_seconds', 0)
                            f.write(f"  - {op_name.replace('_', ' ').title()}: {duration:.3f}s\n")
                
                # Resource usage
                if 'resource_usage' in self.results:
                    f.write("\n## Resource Usage Summary\n\n")
                    usage = self.results['resource_usage']
                    f.write(f"- **Duration:** {usage.get('duration_seconds', 0):.1f} seconds\n")
                    f.write(f"- **Peak CPU:** {usage.get('cpu_percent', {}).get('max', 0):.1f}%\n")
                    f.write(f"- **Peak Memory:** {usage.get('memory_mb', {}).get('max', 0):.1f} MB\n")
                    f.write(f"- **Disk I/O Read:** {usage.get('disk_io_read_mb', {}).get('max', 0):.1f} MB\n")
                    f.write(f"- **Disk I/O Write:** {usage.get('disk_io_write_mb', {}).get('max', 0):.1f} MB\n")
            
            self.logger.info(f"✅ Benchmark summary saved: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate summary: {e}")


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description="Benchmark SAM Engine Upgrade Framework")
    parser.add_argument("--output-dir", default="./benchmarks", help="Output directory for results")
    parser.add_argument("--test", choices=['migration', 'library', 'ui', 'all'], default='all',
                       help="Specific test to run")
    
    args = parser.parse_args()
    
    # Create benchmark suite
    benchmark = EngineUpgradeBenchmark(args.output_dir)
    
    # Run specified tests
    if args.test == 'all':
        results = benchmark.run_full_benchmark_suite()
    elif args.test == 'migration':
        results = {'tests': {'migration_controller': benchmark.benchmark_migration_controller_performance()}}
    elif args.test == 'library':
        results = {'tests': {'model_library': benchmark.benchmark_model_library_operations()}}
    elif args.test == 'ui':
        results = {'tests': {'ui_response': benchmark.benchmark_ui_response_times()}}
    
    # Print summary
    print("\n" + "="*60)
    print("🏁 BENCHMARK COMPLETE")
    print("="*60)
    
    if 'tests' in results:
        for test_name, test_data in results['tests'].items():
            status = test_data.get('status', 'unknown')
            status_icon = "✅" if status == 'completed' else "❌"
            print(f"{status_icon} {test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nResults saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
