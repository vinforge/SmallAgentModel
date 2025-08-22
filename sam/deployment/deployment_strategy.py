#!/usr/bin/env python3
"""
SAM Deployment Strategy
=======================

Intelligent deployment strategy that adapts to different user environments
and provides optimal experience regardless of Docker availability.

Deployment Modes:
1. Full Docker Mode (highest security)
2. Local Enhanced Mode (good security + performance)
3. Basic Mode (minimal requirements)

Author: SAM Development Team
Version: 1.0.0
"""

import os
import sys
import logging
import platform
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class SAMDeploymentManager:
    """
    Manages SAM deployment strategy based on user environment.
    
    Automatically detects capabilities and configures optimal execution mode.
    """
    
    def __init__(self):
        """Initialize deployment manager."""
        self.logger = logging.getLogger(f"{__name__}.SAMDeploymentManager")
        self.system_info = self._detect_system_capabilities()
        self.deployment_mode = None
        self.capabilities = {}
        
        self.logger.info("🚀 SAM Deployment Manager initialized")
    
    def _detect_system_capabilities(self) -> Dict[str, Any]:
        """Detect system capabilities and constraints."""
        system_info = {
            'platform': platform.system().lower(),
            'python_version': sys.version_info,
            'docker_available': False,
            'docker_running': False,
            'memory_gb': self._get_available_memory(),
            'cpu_cores': os.cpu_count(),
            'user_type': 'unknown'  # developer, researcher, business_user
        }
        
        # Check Docker availability
        try:
            import subprocess
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, timeout=5)
            if result.returncode == 0:
                system_info['docker_available'] = True
                
                # Check if Docker daemon is running
                daemon_result = subprocess.run(['docker', 'info'], 
                                             capture_output=True, timeout=5)
                system_info['docker_running'] = (daemon_result.returncode == 0)
                
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        return system_info
    
    def _get_available_memory(self) -> float:
        """Get available system memory in GB."""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        except ImportError:
            # Fallback estimation
            return 8.0  # Assume 8GB default
    
    def determine_optimal_deployment_mode(self) -> str:
        """
        Determine the optimal deployment mode for this environment.
        
        Returns:
            Deployment mode: 'full_docker', 'local_enhanced', or 'basic'
        """
        system = self.system_info
        
        # Full Docker Mode (Best Experience)
        if (system['docker_available'] and 
            system['memory_gb'] >= 4 and 
            system['cpu_cores'] >= 2):
            return 'full_docker'
        
        # Local Enhanced Mode (Good Experience)
        elif (system['memory_gb'] >= 2 and 
              system['cpu_cores'] >= 2):
            return 'local_enhanced'
        
        # Basic Mode (Minimal Requirements)
        else:
            return 'basic'
    
    def configure_deployment(self, mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Configure SAM for the specified deployment mode.
        
        Args:
            mode: Deployment mode or None for auto-detection
            
        Returns:
            Configuration details
        """
        if mode is None:
            mode = self.determine_optimal_deployment_mode()
        
        self.deployment_mode = mode
        
        if mode == 'full_docker':
            return self._configure_full_docker_mode()
        elif mode == 'local_enhanced':
            return self._configure_local_enhanced_mode()
        else:
            return self._configure_basic_mode()
    
    def _configure_full_docker_mode(self) -> Dict[str, Any]:
        """Configure Full Docker Mode."""
        self.logger.info("🐳 Configuring Full Docker Mode")
        
        config = {
            'mode': 'full_docker',
            'security_level': 'high',
            'features': {
                'csv_analysis': True,
                'code_interpreter': True,
                'visualizations': True,
                'large_datasets': True,
                'parallel_processing': True
            },
            'execution_strategy': 'docker_sandbox',
            'auto_provisioning': True,
            'fallback_enabled': True,
            'user_message': "🚀 SAM configured with full Docker support for maximum security and capabilities!"
        }
        
        # Configure Docker auto-provisioning
        self.capabilities = {
            'max_memory_mb': 2048,
            'max_execution_time': 300,
            'network_isolation': True,
            'file_system_isolation': True
        }
        
        return config
    
    def _configure_local_enhanced_mode(self) -> Dict[str, Any]:
        """Configure Local Enhanced Mode."""
        self.logger.info("🔒 Configuring Local Enhanced Mode")
        
        config = {
            'mode': 'local_enhanced',
            'security_level': 'medium',
            'features': {
                'csv_analysis': True,
                'code_interpreter': True,
                'visualizations': True,
                'large_datasets': False,  # Limited by local memory
                'parallel_processing': False
            },
            'execution_strategy': 'local_restricted',
            'auto_provisioning': False,
            'fallback_enabled': True,
            'user_message': "⚡ SAM configured for local execution with enhanced security and good performance!"
        }
        
        # Configure local restrictions
        self.capabilities = {
            'max_memory_mb': min(1024, int(self.system_info['memory_gb'] * 256)),
            'max_execution_time': 60,
            'network_isolation': False,
            'file_system_isolation': False
        }
        
        return config
    
    def _configure_basic_mode(self) -> Dict[str, Any]:
        """Configure Basic Mode."""
        self.logger.info("📱 Configuring Basic Mode")
        
        config = {
            'mode': 'basic',
            'security_level': 'low',
            'features': {
                'csv_analysis': True,
                'code_interpreter': False,  # Limited to basic calculations
                'visualizations': False,
                'large_datasets': False,
                'parallel_processing': False
            },
            'execution_strategy': 'local_safe',
            'auto_provisioning': False,
            'fallback_enabled': True,
            'user_message': "📱 SAM configured for basic operation - perfect for getting started!"
        }
        
        # Configure basic restrictions
        self.capabilities = {
            'max_memory_mb': 256,
            'max_execution_time': 30,
            'network_isolation': False,
            'file_system_isolation': False
        }
        
        return config
    
    def get_user_setup_instructions(self) -> Dict[str, Any]:
        """
        Get setup instructions for the user based on their environment.
        
        Returns:
            Setup instructions and recommendations
        """
        mode = self.deployment_mode or self.determine_optimal_deployment_mode()
        
        instructions = {
            'current_mode': mode,
            'system_info': self.system_info,
            'setup_steps': [],
            'optional_improvements': [],
            'troubleshooting': []
        }
        
        if mode == 'full_docker':
            instructions['setup_steps'] = [
                "✅ Docker detected and ready",
                "✅ System meets requirements for full Docker mode",
                "🚀 SAM will automatically manage Docker containers"
            ]
            
            if not self.system_info['docker_running']:
                instructions['setup_steps'].append(
                    "🔄 Docker will be started automatically when needed"
                )
        
        elif mode == 'local_enhanced':
            instructions['setup_steps'] = [
                "✅ System configured for local enhanced mode",
                "⚡ Good performance with medium security",
                "📊 Full CSV analysis capabilities available"
            ]
            
            instructions['optional_improvements'] = [
                "💡 Install Docker Desktop for highest security mode",
                "💡 Docker provides isolated execution environment",
                "💡 Download from: https://www.docker.com/products/docker-desktop"
            ]
        
        else:  # basic mode
            instructions['setup_steps'] = [
                "✅ Basic mode configured",
                "📱 Minimal system requirements met",
                "🎯 Perfect for getting started with SAM"
            ]
            
            instructions['optional_improvements'] = [
                "💡 Consider upgrading system memory for enhanced mode",
                "💡 Install Docker Desktop for full capabilities",
                "💡 Enhanced mode provides better CSV analysis features"
            ]
        
        return instructions
    
    def should_auto_start_docker(self) -> bool:
        """
        Determine if Docker should be auto-started on SAM startup.
        
        Returns:
            True if Docker should be auto-started
        """
        # Only auto-start Docker if:
        # 1. Docker is available
        # 2. System has sufficient resources
        # 3. User is in full_docker mode
        return (
            self.system_info['docker_available'] and
            self.system_info['memory_gb'] >= 4 and
            self.deployment_mode == 'full_docker'
        )
    
    def get_startup_strategy(self) -> Dict[str, Any]:
        """
        Get the startup strategy for SAM.
        
        Returns:
            Startup configuration
        """
        return {
            'auto_start_docker': self.should_auto_start_docker(),
            'lazy_docker_provisioning': True,  # Start Docker only when needed
            'fallback_always_available': True,
            'user_notification': True,
            'background_optimization': True
        }


# Global deployment manager
_deployment_manager = None

def get_deployment_manager() -> SAMDeploymentManager:
    """Get the global deployment manager instance."""
    global _deployment_manager
    if _deployment_manager is None:
        _deployment_manager = SAMDeploymentManager()
    return _deployment_manager


def configure_sam_for_environment() -> Dict[str, Any]:
    """
    Configure SAM for the current environment.
    
    Returns:
        Configuration details
    """
    manager = get_deployment_manager()
    config = manager.configure_deployment()
    
    # Log configuration
    logger.info(f"🎯 SAM configured for {config['mode']} mode")
    logger.info(f"🔒 Security level: {config['security_level']}")
    logger.info(f"⚡ Features: {list(config['features'].keys())}")
    
    return config
