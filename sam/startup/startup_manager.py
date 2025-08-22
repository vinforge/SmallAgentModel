#!/usr/bin/env python3
"""
SAM Startup Manager
===================

Intelligent startup manager that configures SAM based on the user's environment
and provides optimal experience without requiring manual Docker management.

Features:
- Environment detection and adaptation
- Lazy Docker provisioning (start only when needed)
- Graceful fallbacks for all scenarios
- User-friendly setup guidance
- Background optimization

Author: SAM Development Team
Version: 1.0.0
"""

import os
import sys
import time
import logging
import threading
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class SAMStartupManager:
    """
    Manages SAM startup process with intelligent environment adaptation.
    """
    
    def __init__(self):
        """Initialize startup manager."""
        self.logger = logging.getLogger(f"{__name__}.SAMStartupManager")
        self.startup_config = {}
        self.docker_status = 'unknown'
        self.background_tasks = []
        
        self.logger.info("🚀 SAM Startup Manager initialized")
    
    def perform_startup_sequence(self) -> Dict[str, Any]:
        """
        Perform the complete SAM startup sequence.
        
        Returns:
            Startup results and configuration
        """
        self.logger.info("🎯 Starting SAM initialization sequence...")
        
        startup_results = {
            'success': True,
            'mode': 'unknown',
            'docker_status': 'unknown',
            'capabilities': {},
            'user_message': '',
            'warnings': [],
            'next_steps': []
        }
        
        try:
            # Step 1: Environment Detection
            self.logger.info("🔍 Step 1: Detecting environment capabilities...")
            env_result = self._detect_environment()
            startup_results.update(env_result)
            
            # Step 2: Configure Deployment Mode
            self.logger.info("⚙️ Step 2: Configuring deployment mode...")
            config_result = self._configure_deployment_mode()
            startup_results.update(config_result)
            
            # Step 3: Initialize Execution Environment
            self.logger.info("🔧 Step 3: Initializing execution environment...")
            exec_result = self._initialize_execution_environment()
            startup_results.update(exec_result)
            
            # Step 4: Background Optimization (Optional)
            self.logger.info("⚡ Step 4: Starting background optimization...")
            self._start_background_optimization()
            
            # Step 5: Generate User Guidance
            self.logger.info("📋 Step 5: Generating user guidance...")
            guidance = self._generate_user_guidance(startup_results)
            startup_results['user_guidance'] = guidance
            
            self.logger.info("✅ SAM startup sequence completed successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Startup sequence failed: {e}")
            startup_results['success'] = False
            startup_results['error'] = str(e)
            startup_results['fallback_mode'] = 'basic'
        
        return startup_results
    
    def _detect_environment(self) -> Dict[str, Any]:
        """Detect environment capabilities."""
        try:
            from sam.deployment.deployment_strategy import get_deployment_manager
            
            manager = get_deployment_manager()
            system_info = manager.system_info
            
            return {
                'system_info': system_info,
                'docker_available': system_info['docker_available'],
                'docker_running': system_info['docker_running'],
                'memory_gb': system_info['memory_gb'],
                'cpu_cores': system_info['cpu_cores']
            }
            
        except Exception as e:
            self.logger.warning(f"Environment detection failed: {e}")
            return {
                'system_info': {'platform': 'unknown'},
                'docker_available': False,
                'docker_running': False,
                'memory_gb': 4.0,
                'cpu_cores': 2
            }
    
    def _configure_deployment_mode(self) -> Dict[str, Any]:
        """Configure the optimal deployment mode."""
        try:
            from sam.deployment.deployment_strategy import configure_sam_for_environment
            
            config = configure_sam_for_environment()
            self.startup_config = config
            
            return {
                'mode': config['mode'],
                'security_level': config['security_level'],
                'features': config['features'],
                'user_message': config['user_message']
            }
            
        except Exception as e:
            self.logger.error(f"Deployment configuration failed: {e}")
            return {
                'mode': 'basic',
                'security_level': 'low',
                'features': {'csv_analysis': True},
                'user_message': 'SAM configured in basic mode due to configuration error'
            }
    
    def _initialize_execution_environment(self) -> Dict[str, Any]:
        """Initialize the execution environment."""
        try:
            from sam.code_interpreter.smart_sandbox_manager import get_smart_sandbox_manager
            
            # Initialize sandbox manager
            sandbox_manager = get_smart_sandbox_manager()
            
            # Get execution status
            status = sandbox_manager.get_execution_status()
            
            # Determine if we should attempt Docker provisioning
            should_provision_docker = (
                self.startup_config.get('auto_provisioning', False) and
                status.get('docker_available', False)
            )
            
            if should_provision_docker:
                self.logger.info("🐳 Attempting Docker provisioning...")
                # Docker provisioning will happen lazily when first needed
                docker_status = 'lazy_provisioning'
            else:
                docker_status = 'not_needed'
            
            return {
                'execution_environment': 'ready',
                'docker_status': docker_status,
                'best_mode': status.get('best_mode', 'local_restricted'),
                'fallback_available': status.get('fallback_available', True)
            }
            
        except Exception as e:
            self.logger.error(f"Execution environment initialization failed: {e}")
            return {
                'execution_environment': 'fallback',
                'docker_status': 'failed',
                'best_mode': 'local_safe',
                'fallback_available': True
            }
    
    def _start_background_optimization(self):
        """Start background optimization tasks."""
        try:
            # Only start background tasks if we have sufficient resources
            if self.startup_config.get('mode') in ['full_docker', 'local_enhanced']:
                
                # Background Docker preparation (if applicable)
                if (self.startup_config.get('auto_provisioning', False) and
                    not self.startup_config.get('docker_running', False)):
                    
                    def background_docker_prep():
                        try:
                            self.logger.info("🔄 Background: Preparing Docker environment...")
                            from sam.code_interpreter.docker_auto_provisioner import get_docker_auto_provisioner
                            
                            provisioner = get_docker_auto_provisioner()
                            # Check if Docker can be started
                            daemon_status = provisioner.check_docker_daemon_status()
                            if not daemon_status['running']:
                                # Attempt to start Docker in background
                                start_result = provisioner.attempt_docker_start()
                                if start_result['success']:
                                    self.logger.info("✅ Background: Docker started successfully")
                                else:
                                    self.logger.info("ℹ️ Background: Docker start deferred to first use")
                        except Exception as e:
                            self.logger.debug(f"Background Docker prep failed: {e}")
                    
                    # Start background thread
                    thread = threading.Thread(target=background_docker_prep, daemon=True)
                    thread.start()
                    self.background_tasks.append(thread)
            
        except Exception as e:
            self.logger.debug(f"Background optimization failed: {e}")
    
    def _generate_user_guidance(self, startup_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate user guidance based on startup results."""
        mode = startup_results.get('mode', 'basic')
        docker_available = startup_results.get('docker_available', False)
        
        guidance = {
            'welcome_message': '',
            'current_capabilities': [],
            'setup_complete': True,
            'optional_improvements': [],
            'troubleshooting': []
        }
        
        if mode == 'full_docker':
            guidance['welcome_message'] = "🚀 Welcome to SAM! You're running with full Docker support for maximum security and capabilities."
            guidance['current_capabilities'] = [
                "✅ Secure CSV data analysis with Docker isolation",
                "✅ Advanced code interpretation and execution",
                "✅ Professional data visualizations",
                "✅ Large dataset processing",
                "✅ Automatic Docker container management"
            ]
            
        elif mode == 'local_enhanced':
            guidance['welcome_message'] = "⚡ Welcome to SAM! You're running in enhanced local mode with excellent performance."
            guidance['current_capabilities'] = [
                "✅ Fast CSV data analysis with pandas",
                "✅ Secure local code execution",
                "✅ Data visualizations and statistics",
                "✅ Medium security with good performance"
            ]
            
            if not docker_available:
                guidance['optional_improvements'] = [
                    "💡 Install Docker Desktop for highest security mode",
                    "💡 Docker provides isolated execution environment",
                    "💡 Download: https://www.docker.com/products/docker-desktop"
                ]
            
        else:  # basic mode
            guidance['welcome_message'] = "📱 Welcome to SAM! You're running in basic mode - perfect for getting started."
            guidance['current_capabilities'] = [
                "✅ Basic CSV data analysis",
                "✅ Simple calculations and statistics",
                "✅ Minimal system requirements",
                "✅ Easy to use interface"
            ]
            
            guidance['optional_improvements'] = [
                "💡 Upgrade system memory for enhanced features",
                "💡 Install Docker Desktop for full capabilities",
                "💡 Enhanced mode provides better CSV analysis"
            ]
        
        return guidance
    
    def get_startup_summary(self) -> str:
        """Get a formatted startup summary for display."""
        if not self.startup_config:
            return "SAM startup not completed"
        
        mode = self.startup_config.get('mode', 'unknown')
        security = self.startup_config.get('security_level', 'unknown')
        features = self.startup_config.get('features', {})
        
        summary = f"""
🎯 SAM Startup Summary
=====================
Mode: {mode.replace('_', ' ').title()}
Security Level: {security.title()}
CSV Analysis: {'✅' if features.get('csv_analysis') else '❌'}
Code Interpreter: {'✅' if features.get('code_interpreter') else '❌'}
Visualizations: {'✅' if features.get('visualizations') else '❌'}

{self.startup_config.get('user_message', '')}
"""
        return summary


# Global startup manager
_startup_manager = None

def get_startup_manager() -> SAMStartupManager:
    """Get the global startup manager instance."""
    global _startup_manager
    if _startup_manager is None:
        _startup_manager = SAMStartupManager()
    return _startup_manager


def initialize_sam() -> Dict[str, Any]:
    """
    Initialize SAM with intelligent environment adaptation.
    
    Returns:
        Initialization results
    """
    manager = get_startup_manager()
    return manager.perform_startup_sequence()
