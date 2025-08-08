#!/usr/bin/env python3
"""
SAM System Health Monitor

Real-time monitoring service for SAM's critical thinking components,
including SOF v2, mathematical query routing, and reasoning systems.

Author: SAM Development Team
Version: 1.0.0
"""

import time
import json
import logging
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path

@dataclass
class ComponentStatus:
    """Status of a system component."""
    name: str
    status: str  # "healthy", "warning", "error", "unknown"
    last_check: str
    response_time_ms: float
    details: Dict[str, Any]
    error_message: Optional[str] = None

@dataclass
class SystemHealthReport:
    """Complete system health report."""
    overall_status: str
    overall_score: float
    last_updated: str
    components: List[ComponentStatus]
    recommendations: List[str]
    uptime_seconds: float

class SystemHealthMonitor:
    """
    Monitors SAM's critical thinking and decision-making components.
    
    Features:
    - Real-time health monitoring
    - Component status tracking
    - Performance metrics
    - Automatic issue detection
    - Health score calculation
    - Recommendations generation
    """
    
    def __init__(self):
        """Initialize the health monitor."""
        self.logger = logging.getLogger(f"{__name__}.SystemHealthMonitor")
        self.start_time = datetime.now()
        self.last_report = None
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Component check intervals (seconds)
        self.check_intervals = {
            'sof_v2': 30,
            'mathematical_routing': 60,
            'reasoning_systems': 45,
            'memory_systems': 30,
            'web_retrieval': 60
        }
        
        # Last check times
        self.last_checks = {}
    
    def start_monitoring(self):
        """Start continuous monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("System health monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("System health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Check if any components need checking
                current_time = datetime.now()
                
                for component, interval in self.check_intervals.items():
                    last_check = self.last_checks.get(component)
                    
                    if not last_check or (current_time - last_check).seconds >= interval:
                        self._check_component(component)
                        self.last_checks[component] = current_time
                
                # Sleep for a short interval
                time.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(10)
    
    def _check_component(self, component_name: str):
        """Check a specific component."""
        try:
            if component_name == 'sof_v2':
                self._check_sof_v2()
            elif component_name == 'mathematical_routing':
                self._check_mathematical_routing()
            elif component_name == 'reasoning_systems':
                self._check_reasoning_systems()
            elif component_name == 'memory_systems':
                self._check_memory_systems()
            elif component_name == 'web_retrieval':
                self._check_web_retrieval()
        except Exception as e:
            self.logger.error(f"Component check failed for {component_name}: {e}")
    
    def get_health_report(self) -> SystemHealthReport:
        """Get current system health report."""
        start_time = time.time()
        
        components = []
        
        # Check all components
        components.append(self._check_sof_v2())
        components.append(self._check_mathematical_routing())
        components.append(self._check_reasoning_systems())
        components.append(self._check_memory_systems())
        components.append(self._check_web_retrieval())
        
        # Calculate overall health
        healthy_count = sum(1 for c in components if c.status == "healthy")
        warning_count = sum(1 for c in components if c.status == "warning")
        error_count = sum(1 for c in components if c.status == "error")
        
        total_components = len(components)
        overall_score = (healthy_count + warning_count * 0.5) / total_components * 100
        
        if error_count > 0:
            overall_status = "error"
        elif warning_count > 0:
            overall_status = "warning"
        else:
            overall_status = "healthy"
        
        # Generate recommendations
        recommendations = self._generate_recommendations(components)
        
        # Calculate uptime
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        report = SystemHealthReport(
            overall_status=overall_status,
            overall_score=overall_score,
            last_updated=datetime.now().isoformat(),
            components=components,
            recommendations=recommendations,
            uptime_seconds=uptime
        )
        
        self.last_report = report
        return report
    
    def _check_sof_v2(self) -> ComponentStatus:
        """Check SOF v2 Dynamic Agent Architecture."""
        start_time = time.time()
        
        try:
            # Test SOF imports and configuration
            from sam.orchestration import is_sof_enabled, get_sof_integration
            
            enabled = is_sof_enabled()
            details = {"enabled": enabled}
            
            if enabled:
                try:
                    sof_integration = get_sof_integration()
                    details["integration_available"] = sof_integration is not None
                    details["initialized"] = getattr(sof_integration, '_initialized', False)
                    
                    if sof_integration and sof_integration._initialized:
                        # Test plan generation
                        from sam.orchestration.uif import SAM_UIF
                        test_uif = SAM_UIF(input_query="test query")
                        
                        if hasattr(sof_integration, '_coordinator') and sof_integration._coordinator:
                            coordinator = sof_integration._coordinator
                            if hasattr(coordinator, '_dynamic_planner') and coordinator._dynamic_planner:
                                plan_result = coordinator._dynamic_planner.create_plan(test_uif)
                                details["plan_generation"] = plan_result is not None
                            else:
                                details["plan_generation"] = False
                        
                        status = "healthy"
                        error_message = None
                    else:
                        status = "warning"
                        error_message = "SOF integration not initialized"
                except Exception as e:
                    status = "error"
                    error_message = f"SOF integration failed: {e}"
                    details["integration_error"] = str(e)
            else:
                status = "warning"
                error_message = "SOF v2 is disabled"
            
        except Exception as e:
            status = "error"
            error_message = f"SOF v2 check failed: {e}"
            details = {"error": str(e)}
        
        response_time = (time.time() - start_time) * 1000
        
        return ComponentStatus(
            name="SOF v2 Dynamic Agent Architecture",
            status=status,
            last_check=datetime.now().isoformat(),
            response_time_ms=response_time,
            details=details,
            error_message=error_message
        )
    
    def _check_mathematical_routing(self) -> ComponentStatus:
        """Check mathematical query routing."""
        start_time = time.time()
        
        try:
            import re
            
            # Test mathematical pattern detection
            test_query = "What is 10+5?"
            math_pattern = r'\d+\s*[\+\-\*\/]\s*\d+'
            
            details = {}
            
            # Test pattern detection
            pattern_detected = bool(re.search(math_pattern, test_query))
            details["pattern_detection"] = pattern_detected
            
            if pattern_detected:
                # Test expression extraction
                expression_matches = re.findall(r'[\d\+\-\*\/\(\)\.\s]+', test_query)
                best_expression = None
                
                for match in expression_matches:
                    cleaned = match.strip()
                    if re.search(r'\d+\s*[\+\-\*\/]\s*\d+', cleaned) and re.match(r'^[\d\+\-\*\/\(\)\.\s]+$', cleaned):
                        best_expression = cleaned
                        break
                
                details["expression_extraction"] = best_expression is not None
                
                if best_expression:
                    # Test calculation
                    try:
                        result = eval(best_expression)
                        details["calculation_works"] = True
                        details["test_result"] = f"{best_expression} = {result}"
                        status = "healthy"
                        error_message = None
                    except Exception as calc_error:
                        details["calculation_works"] = False
                        details["calculation_error"] = str(calc_error)
                        status = "error"
                        error_message = f"Calculation failed: {calc_error}"
                else:
                    status = "warning"
                    error_message = "Expression extraction failed"
            else:
                status = "error"
                error_message = "Mathematical pattern not detected"
            
            # Test Calculator Tool if available
            try:
                from sam.orchestration.skills.calculator_tool import CalculatorTool
                calculator = CalculatorTool()
                details["calculator_tool_available"] = True
            except Exception:
                details["calculator_tool_available"] = False
            
        except Exception as e:
            status = "error"
            error_message = f"Mathematical routing check failed: {e}"
            details = {"error": str(e)}
        
        response_time = (time.time() - start_time) * 1000
        
        return ComponentStatus(
            name="Mathematical Query Routing",
            status=status,
            last_check=datetime.now().isoformat(),
            response_time_ms=response_time,
            details=details,
            error_message=error_message
        )
    
    def _check_reasoning_systems(self) -> ComponentStatus:
        """Check reasoning and decision-making systems."""
        start_time = time.time()
        
        details = {}
        issues = []
        
        try:
            # Check TPV (Active Reasoning Control)
            try:
                from reasoning.tpv_integration import get_tpv_integration
                tpv = get_tpv_integration()
                details["tpv_available"] = tpv is not None
                details["tpv_enabled"] = getattr(tpv, 'enabled', False) if tpv else False
            except Exception:
                details["tpv_available"] = False
                issues.append("TPV not available")
            
            # Check SLP (Autonomous Cognitive Automation)
            try:
                from reasoning.slp_integration import get_slp_integration
                slp = get_slp_integration()
                details["slp_available"] = slp is not None
                details["slp_enabled"] = getattr(slp, 'enabled', False) if slp else False
            except Exception:
                details["slp_available"] = False
                issues.append("SLP not available")
            
            # Check Confidence Assessor
            try:
                from reasoning.confidence_assessor import get_confidence_assessor
                assessor = get_confidence_assessor()
                details["confidence_assessor_available"] = assessor is not None
            except Exception:
                details["confidence_assessor_available"] = False
                issues.append("Confidence assessor not available")
            
            # Determine status
            available_systems = sum([
                details.get("tpv_available", False),
                details.get("slp_available", False),
                details.get("confidence_assessor_available", False)
            ])
            
            if available_systems == 3:
                status = "healthy"
                error_message = None
            elif available_systems >= 2:
                status = "warning"
                error_message = f"Some reasoning systems unavailable: {', '.join(issues)}"
            else:
                status = "error"
                error_message = f"Critical reasoning systems unavailable: {', '.join(issues)}"
            
        except Exception as e:
            status = "error"
            error_message = f"Reasoning systems check failed: {e}"
            details = {"error": str(e)}
        
        response_time = (time.time() - start_time) * 1000
        
        return ComponentStatus(
            name="Reasoning Systems",
            status=status,
            last_check=datetime.now().isoformat(),
            response_time_ms=response_time,
            details=details,
            error_message=error_message
        )
    
    def _check_memory_systems(self) -> ComponentStatus:
        """Check memory and knowledge systems."""
        start_time = time.time()
        
        details = {}
        issues = []
        
        try:
            # Check Secure Memory Store
            try:
                from memory.secure_memory_vectorstore import get_secure_memory_store, VectorStoreType
                store = get_secure_memory_store(
                    store_type=VectorStoreType.CHROMA,
                    storage_directory="memory_store",
                    embedding_dimension=384,
                    enable_encryption=False
                )
                # Test search
                results = store.search_memories("test", max_results=1)
                details["secure_memory_available"] = True
                details["secure_memory_search_works"] = True
            except Exception as e:
                details["secure_memory_available"] = False
                issues.append(f"Secure memory: {e}")
            
            # Check Regular Memory Store
            try:
                from memory.memory_vectorstore import get_memory_store
                store = get_memory_store()
                results = store.search_memories("test", max_results=1)
                details["regular_memory_available"] = True
                details["regular_memory_search_works"] = True
            except Exception as e:
                details["regular_memory_available"] = False
                issues.append(f"Regular memory: {e}")
            
            # Check Unified Search
            try:
                import secure_streamlit_app
                details["unified_search_available"] = hasattr(secure_streamlit_app, 'search_unified_memory')
            except Exception:
                details["unified_search_available"] = False
                issues.append("Unified search not available")
            
            # Determine status
            if not issues:
                status = "healthy"
                error_message = None
            elif len(issues) <= 1:
                status = "warning"
                error_message = f"Minor memory issues: {', '.join(issues)}"
            else:
                status = "error"
                error_message = f"Memory system issues: {', '.join(issues)}"
            
        except Exception as e:
            status = "error"
            error_message = f"Memory systems check failed: {e}"
            details = {"error": str(e)}
        
        response_time = (time.time() - start_time) * 1000
        
        return ComponentStatus(
            name="Memory Systems",
            status=status,
            last_check=datetime.now().isoformat(),
            response_time_ms=response_time,
            details=details,
            error_message=error_message
        )
    
    def _check_web_retrieval(self) -> ComponentStatus:
        """Check web retrieval and search systems."""
        start_time = time.time()
        
        details = {}
        issues = []
        
        try:
            # Check Web Query Router
            try:
                from web_retrieval.query_router import WebQueryRouter
                router = WebQueryRouter()
                details["web_router_available"] = True
            except Exception:
                details["web_router_available"] = False
                issues.append("Web router not available")
            
            # Check Search Tools
            try:
                from web_retrieval.tools.search_api_tool import SearchAPITool
                tool = SearchAPITool()
                details["search_tools_available"] = True
            except Exception:
                details["search_tools_available"] = False
                issues.append("Search tools not available")
            
            # Check Knowledge Consolidation
            try:
                from knowledge_consolidation.knowledge_integrator import KnowledgeIntegrator
                integrator = KnowledgeIntegrator()
                details["knowledge_consolidation_available"] = True
            except Exception:
                details["knowledge_consolidation_available"] = False
                issues.append("Knowledge consolidation not available")
            
            # Determine status
            if not issues:
                status = "healthy"
                error_message = None
            elif len(issues) <= 1:
                status = "warning"
                error_message = f"Minor web retrieval issues: {', '.join(issues)}"
            else:
                status = "error"
                error_message = f"Web retrieval issues: {', '.join(issues)}"
            
        except Exception as e:
            status = "error"
            error_message = f"Web retrieval check failed: {e}"
            details = {"error": str(e)}
        
        response_time = (time.time() - start_time) * 1000
        
        return ComponentStatus(
            name="Web Retrieval Systems",
            status=status,
            last_check=datetime.now().isoformat(),
            response_time_ms=response_time,
            details=details,
            error_message=error_message
        )
    
    def _generate_recommendations(self, components: List[ComponentStatus]) -> List[str]:
        """Generate recommendations based on component status."""
        recommendations = []
        
        for component in components:
            if component.status == "error":
                if "SOF v2" in component.name:
                    recommendations.append("🔧 Enable SOF v2: Run python fix_sof_v2.py")
                elif "Mathematical" in component.name:
                    recommendations.append("🧮 Fix math routing: Check mathematical pattern detection")
                elif "Reasoning" in component.name:
                    recommendations.append("💭 Check reasoning systems: Verify TPV/SLP integration")
                elif "Memory" in component.name:
                    recommendations.append("💾 Fix memory systems: Check vector store configuration")
                elif "Web" in component.name:
                    recommendations.append("🌐 Fix web retrieval: Check search tool configuration")
            
            elif component.status == "warning":
                if "SOF v2" in component.name:
                    recommendations.append("⚠️ SOF v2 needs attention: Check initialization")
                elif "Mathematical" in component.name:
                    recommendations.append("⚠️ Math routing has issues: Verify calculator tool")
        
        if not recommendations:
            recommendations.append("🎉 All systems are healthy!")
        
        return recommendations

# Global health monitor instance
_health_monitor = None

def get_health_monitor() -> SystemHealthMonitor:
    """Get the global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = SystemHealthMonitor()
        # Auto-start monitoring
        _health_monitor.start_monitoring()
    return _health_monitor

def start_health_monitoring():
    """Start the health monitoring service."""
    monitor = get_health_monitor()
    monitor.start_monitoring()
    return monitor

def stop_health_monitoring():
    """Stop the health monitoring service."""
    global _health_monitor
    if _health_monitor:
        _health_monitor.stop_monitoring()
