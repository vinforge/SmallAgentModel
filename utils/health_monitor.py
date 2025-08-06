"""
System Health Monitor for SAM
Status API, health checks, and performance monitoring.

Sprint 13 Task 5: System Health Dashboard
"""

import time
import json
import psutil
import logging
import requests
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class HealthStatus:
    """Health status for a component."""
    name: str
    status: str  # healthy, warning, critical, unknown
    last_check: str
    response_time_ms: float
    error_message: str = ""
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}

@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_percent: float
    disk_used_gb: float
    disk_free_gb: float
    network_sent_mb: float
    network_recv_mb: float
    process_count: int
    uptime_seconds: float

class HealthMonitor:
    """
    Monitors system health and provides status API.
    """
    
    def __init__(self, chat_port: int = 5001, memory_ui_port: int = 8501, streamlit_chat_port: int = 8502):
        """Initialize the health monitor."""
        self.chat_port = chat_port
        self.memory_ui_port = memory_ui_port
        self.streamlit_chat_port = streamlit_chat_port
        
        self.health_status = {}
        self.metrics_history = []
        self.start_time = time.time()
        self.monitoring = False
        self.monitor_thread = None
        
        # Configuration
        self.check_interval = 30  # seconds
        self.metrics_retention = 1440  # 24 hours of 1-minute intervals
        self.timeout = 10  # seconds
        
        logger.info("Health monitor initialized")
    
    def check_web_interface(self) -> HealthStatus:
        """Check web chat interface health."""
        try:
            start_time = time.time()
            url = f"http://localhost:{self.chat_port}/health"
            
            response = requests.get(url, timeout=self.timeout)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                status = "healthy"
                error_message = ""
            else:
                status = "warning"
                error_message = f"HTTP {response.status_code}"
            
            return HealthStatus(
                name="web_interface",
                status=status,
                last_check=datetime.now().isoformat(),
                response_time_ms=response_time,
                error_message=error_message,
                details={
                    "url": url,
                    "status_code": response.status_code,
                    "port": self.chat_port
                }
            )
            
        except requests.exceptions.ConnectionError:
            return HealthStatus(
                name="web_interface",
                status="critical",
                last_check=datetime.now().isoformat(),
                response_time_ms=0,
                error_message="Connection refused",
                details={"port": self.chat_port}
            )
        except requests.exceptions.Timeout:
            return HealthStatus(
                name="web_interface",
                status="warning",
                last_check=datetime.now().isoformat(),
                response_time_ms=self.timeout * 1000,
                error_message="Request timeout",
                details={"port": self.chat_port}
            )
        except Exception as e:
            return HealthStatus(
                name="web_interface",
                status="critical",
                last_check=datetime.now().isoformat(),
                response_time_ms=0,
                error_message=str(e),
                details={"port": self.chat_port}
            )
    
    def check_memory_ui(self) -> HealthStatus:
        """Check Memory Control Center health."""
        try:
            start_time = time.time()
            url = f"http://localhost:{self.memory_ui_port}/_stcore/health"
            
            response = requests.get(url, timeout=self.timeout)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                status = "healthy"
                error_message = ""
            else:
                status = "warning"
                error_message = f"HTTP {response.status_code}"
            
            return HealthStatus(
                name="memory_ui",
                status=status,
                last_check=datetime.now().isoformat(),
                response_time_ms=response_time,
                error_message=error_message,
                details={
                    "url": url,
                    "status_code": response.status_code,
                    "port": self.memory_ui_port
                }
            )
            
        except requests.exceptions.ConnectionError:
            return HealthStatus(
                name="memory_ui",
                status="critical",
                last_check=datetime.now().isoformat(),
                response_time_ms=0,
                error_message="Connection refused",
                details={"port": self.memory_ui_port}
            )
        except requests.exceptions.Timeout:
            return HealthStatus(
                name="memory_ui",
                status="warning",
                last_check=datetime.now().isoformat(),
                response_time_ms=self.timeout * 1000,
                error_message="Request timeout",
                details={"port": self.memory_ui_port}
            )
        except Exception as e:
            return HealthStatus(
                name="memory_ui",
                status="critical",
                last_check=datetime.now().isoformat(),
                response_time_ms=0,
                error_message=str(e),
                details={"port": self.memory_ui_port}
            )

    def check_streamlit_chat(self) -> HealthStatus:
        """Check Streamlit Chat Interface health."""
        try:
            start_time = time.time()
            url = f"http://localhost:{self.streamlit_chat_port}/_stcore/health"

            response = requests.get(url, timeout=self.timeout)
            response_time = (time.time() - start_time) * 1000

            if response.status_code == 200:
                status = "healthy"
                error_message = ""
            else:
                status = "warning"
                error_message = f"HTTP {response.status_code}"

            return HealthStatus(
                name="streamlit_chat",
                status=status,
                last_check=datetime.now().isoformat(),
                response_time_ms=response_time,
                error_message=error_message,
                details={
                    "url": url,
                    "status_code": response.status_code,
                    "port": self.streamlit_chat_port
                }
            )

        except requests.exceptions.ConnectionError:
            return HealthStatus(
                name="streamlit_chat",
                status="critical",
                last_check=datetime.now().isoformat(),
                response_time_ms=0,
                error_message="Connection refused",
                details={"port": self.streamlit_chat_port}
            )
        except requests.exceptions.Timeout:
            return HealthStatus(
                name="streamlit_chat",
                status="warning",
                last_check=datetime.now().isoformat(),
                response_time_ms=self.timeout * 1000,
                error_message="Request timeout",
                details={"port": self.streamlit_chat_port}
            )
        except Exception as e:
            return HealthStatus(
                name="streamlit_chat",
                status="critical",
                last_check=datetime.now().isoformat(),
                response_time_ms=0,
                error_message=str(e),
                details={"port": self.streamlit_chat_port}
            )

    def check_memory_store(self) -> HealthStatus:
        """Check memory store health."""
        try:
            from memory.memory_vectorstore import get_memory_store
            
            start_time = time.time()
            memory_store = get_memory_store()
            stats = memory_store.get_memory_stats()
            response_time = (time.time() - start_time) * 1000
            
            if stats and not stats.get('error'):
                status = "healthy"
                error_message = ""
                
                # Check for warnings with fallback values
                total_memories = stats.get('total_memories', 0)
                total_size_mb = stats.get('total_size_mb', 0.0)

                if total_memories > 10000:
                    status = "warning"
                    error_message = "High memory count"
                elif total_size_mb > 500:
                    status = "warning"
                    error_message = "High memory usage"
            else:
                status = "critical"
                error_message = stats.get('error', 'Unknown error')
            
            return HealthStatus(
                name="memory_store",
                status=status,
                last_check=datetime.now().isoformat(),
                response_time_ms=response_time,
                error_message=error_message,
                details=stats if stats else {}
            )
            
        except Exception as e:
            return HealthStatus(
                name="memory_store",
                status="critical",
                last_check=datetime.now().isoformat(),
                response_time_ms=0,
                error_message=str(e),
                details={}
            )
    
    def check_agent_mode(self) -> HealthStatus:
        """Check agent mode controller health."""
        try:
            from config.agent_mode import get_mode_controller
            
            start_time = time.time()
            mode_controller = get_mode_controller()
            mode_status = mode_controller.get_mode_status()
            response_time = (time.time() - start_time) * 1000
            
            status = "healthy"
            error_message = ""
            
            # Check for issues
            if mode_status.key_status.value == "expired":
                status = "warning"
                error_message = "Collaboration key expired"
            elif mode_status.key_status.value == "invalid":
                status = "warning"
                error_message = "Invalid collaboration key"
            
            return HealthStatus(
                name="agent_mode",
                status=status,
                last_check=datetime.now().isoformat(),
                response_time_ms=response_time,
                error_message=error_message,
                details={
                    "current_mode": mode_status.current_mode.value,
                    "key_status": mode_status.key_status.value,
                    "uptime_seconds": mode_status.uptime_seconds,
                    "enabled_capabilities": len(mode_status.enabled_capabilities)
                }
            )
            
        except Exception as e:
            return HealthStatus(
                name="agent_mode",
                status="critical",
                last_check=datetime.now().isoformat(),
                response_time_ms=0,
                error_message=str(e),
                details={}
            )
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Network stats
            network = psutil.net_io_counters()
            
            # Process count
            process_count = len(psutil.pids())
            
            # Uptime
            uptime_seconds = time.time() - self.start_time
            
            return SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_percent=disk.percent,
                disk_used_gb=disk.used / (1024 * 1024 * 1024),
                disk_free_gb=disk.free / (1024 * 1024 * 1024),
                network_sent_mb=network.bytes_sent / (1024 * 1024),
                network_recv_mb=network.bytes_recv / (1024 * 1024),
                process_count=process_count,
                uptime_seconds=uptime_seconds
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=0,
                memory_percent=0,
                memory_used_mb=0,
                memory_available_mb=0,
                disk_percent=0,
                disk_used_gb=0,
                disk_free_gb=0,
                network_sent_mb=0,
                network_recv_mb=0,
                process_count=0,
                uptime_seconds=time.time() - self.start_time
            )
    
    def run_health_checks(self):
        """Run all health checks."""
        try:
            # Check all components
            self.health_status['web_interface'] = self.check_web_interface()
            self.health_status['memory_ui'] = self.check_memory_ui()
            self.health_status['streamlit_chat'] = self.check_streamlit_chat()
            self.health_status['memory_store'] = self.check_memory_store()
            self.health_status['agent_mode'] = self.check_agent_mode()
            
            # Collect system metrics
            metrics = self.collect_system_metrics()
            self.metrics_history.append(metrics)
            
            # Trim metrics history
            if len(self.metrics_history) > self.metrics_retention:
                self.metrics_history = self.metrics_history[-self.metrics_retention:]
            
            logger.debug("Health checks completed")
            
        except Exception as e:
            logger.error(f"Error running health checks: {e}")
    
    def get_overall_status(self) -> str:
        """Get overall system status."""
        if not self.health_status:
            return "unknown"
        
        statuses = [status.status for status in self.health_status.values()]
        
        if "critical" in statuses:
            return "critical"
        elif "warning" in statuses:
            return "warning"
        elif all(status == "healthy" for status in statuses):
            return "healthy"
        else:
            return "unknown"
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        overall_status = self.get_overall_status()
        
        # Get latest metrics
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        # Calculate uptime
        uptime_seconds = time.time() - self.start_time
        uptime_str = str(timedelta(seconds=int(uptime_seconds)))
        
        return {
            "overall_status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "uptime": uptime_str,
            "uptime_seconds": uptime_seconds,
            "components": {name: asdict(status) for name, status in self.health_status.items()},
            "system_metrics": asdict(latest_metrics) if latest_metrics else None,
            "metrics_history_count": len(self.metrics_history)
        }
    
    def get_metrics_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get metrics history for the specified number of hours."""
        if not self.metrics_history:
            return []
        
        # Calculate cutoff time
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter metrics
        filtered_metrics = []
        for metrics in self.metrics_history:
            metrics_time = datetime.fromisoformat(metrics.timestamp)
            if metrics_time >= cutoff_time:
                filtered_metrics.append(asdict(metrics))
        
        return filtered_metrics
    
    def start_monitoring(self):
        """Start health monitoring."""
        self.monitoring = True
        logger.info("Health monitoring started")
        
        while self.monitoring:
            try:
                self.run_health_checks()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring = False
        logger.info("Health monitoring stopped")

# Global health monitor instance
_health_monitor = None

def get_health_monitor() -> HealthMonitor:
    """Get or create a global health monitor instance."""
    global _health_monitor
    
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    
    return _health_monitor
