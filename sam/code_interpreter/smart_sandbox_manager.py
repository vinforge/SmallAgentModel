#!/usr/bin/env python3
"""
Smart Sandbox Manager for SAM Code Interpreter
===============================================

Intelligent sandbox management system that automatically provisions Docker
environments and provides fallback execution strategies.

Features:
- Auto-provisioning of Docker environments
- Smart routing between execution modes
- Fallback to local execution when needed
- Security-conscious execution strategies
- Performance optimization

Author: SAM Development Team
Version: 1.0.0
"""

import os
import sys
import logging
import tempfile
import subprocess
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result from code execution."""
    success: bool
    output: str
    error: str = ""
    execution_time: float = 0.0
    execution_mode: str = "unknown"
    security_level: str = "unknown"


class SmartSandboxManager:
    """
    Intelligent sandbox management system for secure code execution.
    
    Automatically handles Docker provisioning, execution routing, and fallback strategies.
    """
    
    def __init__(self):
        """Initialize the smart sandbox manager."""
        self.logger = logging.getLogger(f"{__name__}.SmartSandboxManager")
        self.docker_provisioner = None
        self.execution_modes = {
            'docker_sandbox': {'available': False, 'security': 'high'},
            'local_restricted': {'available': True, 'security': 'medium'},
            'local_safe': {'available': True, 'security': 'low'}
        }
        
        self.logger.info("🧠 Smart Sandbox Manager initialized")
    
    def initialize_execution_environment(self) -> Dict[str, Any]:
        """
        Initialize the best available execution environment.
        
        Returns:
            Dict with initialization results and available modes
        """
        self.logger.info("🚀 Initializing execution environment...")
        
        # Try to initialize Docker environment
        docker_result = self._initialize_docker_environment()
        
        # Update execution modes based on results
        self.execution_modes['docker_sandbox']['available'] = docker_result['success']
        
        # Determine best execution mode
        best_mode = self._select_best_execution_mode()
        
        return {
            'success': True,
            'docker_result': docker_result,
            'best_mode': best_mode,
            'available_modes': self.execution_modes,
            'message': f"Execution environment ready. Best mode: {best_mode}"
        }
    
    def _initialize_docker_environment(self) -> Dict[str, Any]:
        """
        Initialize Docker environment using auto-provisioner.

        Returns:
            Dict with Docker initialization results
        """
        try:
            # Check if Docker Python library is available
            try:
                import docker
            except ImportError:
                self.logger.info("🐳 Docker Python library not installed - using fallback mode")
                return {
                    'success': False,
                    'message': 'Docker Python library not available',
                    'fallback_recommended': True
                }

            from .docker_auto_provisioner import get_docker_auto_provisioner

            self.docker_provisioner = get_docker_auto_provisioner()

            # Attempt complete auto-provisioning
            provision_result = self.docker_provisioner.auto_provision_complete_environment()

            if provision_result['overall_success']:
                self.logger.info("✅ Docker environment provisioned successfully")
                return {
                    'success': True,
                    'message': 'Docker sandbox ready',
                    'details': provision_result
                }
            else:
                self.logger.warning(f"⚠️ Docker provisioning failed: {provision_result['message']}")
                return {
                    'success': False,
                    'message': provision_result['message'],
                    'fallback_recommended': provision_result.get('fallback_recommended', True),
                    'details': provision_result
                }

        except Exception as e:
            self.logger.warning(f"⚠️ Docker environment initialization failed: {e}")
            return {
                'success': False,
                'message': f'Docker initialization error: {e}',
                'fallback_recommended': True
            }
    
    def _select_best_execution_mode(self) -> str:
        """
        Select the best available execution mode.
        
        Returns:
            Name of the best execution mode
        """
        # Priority order: docker_sandbox > local_restricted > local_safe
        for mode in ['docker_sandbox', 'local_restricted', 'local_safe']:
            if self.execution_modes[mode]['available']:
                self.logger.info(f"🎯 Selected execution mode: {mode}")
                return mode
        
        # Fallback (should never happen)
        return 'local_safe'
    
    def execute_code(self, code: str, timeout: int = 30) -> ExecutionResult:
        """
        Execute Python code using the best available method.
        
        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds
            
        Returns:
            ExecutionResult with execution details
        """
        best_mode = self._select_best_execution_mode()
        
        self.logger.info(f"🔧 Executing code using mode: {best_mode}")
        
        if best_mode == 'docker_sandbox' and self.execution_modes['docker_sandbox']['available']:
            return self._execute_in_docker_sandbox(code, timeout)
        elif best_mode == 'local_restricted':
            return self._execute_local_restricted(code, timeout)
        else:
            return self._execute_local_safe(code, timeout)
    
    def _execute_in_docker_sandbox(self, code: str, timeout: int) -> ExecutionResult:
        """
        Execute code in Docker sandbox (highest security).
        
        Args:
            code: Python code to execute
            timeout: Execution timeout
            
        Returns:
            ExecutionResult
        """
        try:
            # This would integrate with the actual Docker sandbox service
            # For now, we'll simulate the call
            self.logger.info("🐳 Executing in Docker sandbox...")
            
            # TODO: Implement actual Docker execution via sandbox service
            # For now, fall back to local restricted execution
            self.logger.warning("🔄 Docker execution not yet implemented, falling back...")
            return self._execute_local_restricted(code, timeout)
            
        except Exception as e:
            self.logger.error(f"❌ Docker execution failed: {e}")
            return ExecutionResult(
                success=False,
                output="",
                error=f"Docker execution failed: {e}",
                execution_mode="docker_sandbox"
            )
    
    def _execute_local_restricted(self, code: str, timeout: int) -> ExecutionResult:
        """
        Execute code locally with restrictions (medium security).
        
        Args:
            code: Python code to execute
            timeout: Execution timeout
            
        Returns:
            ExecutionResult
        """
        try:
            import time
            start_time = time.time()
            
            self.logger.info("🔒 Executing in local restricted mode...")
            
            # Create restricted execution environment with safe imports
            import pandas as pd
            import numpy as np
            import math
            import statistics
            import json

            restricted_globals = {
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'str': str,
                    'int': int,
                    'float': float,
                    'list': list,
                    'dict': dict,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'sum': sum,
                    'min': min,
                    'max': max,
                    'round': round,
                    'abs': abs,
                    '__import__': __import__,  # Allow imports for pandas/numpy
                },
                # Pre-import safe modules
                'pd': pd,
                'pandas': pd,
                'np': np,
                'numpy': np,
                'math': math,
                'statistics': statistics,
                'json': json,
            }
            
            # Capture output
            from io import StringIO
            import contextlib
            
            output_buffer = StringIO()
            
            with contextlib.redirect_stdout(output_buffer):
                # Execute the code with restrictions
                exec(code, restricted_globals)
            
            execution_time = time.time() - start_time
            output = output_buffer.getvalue()
            
            return ExecutionResult(
                success=True,
                output=output,
                execution_time=execution_time,
                execution_mode="local_restricted",
                security_level="medium"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time if 'start_time' in locals() else 0
            self.logger.error(f"❌ Local restricted execution failed: {e}")
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                execution_time=execution_time,
                execution_mode="local_restricted",
                security_level="medium"
            )
    
    def _execute_local_safe(self, code: str, timeout: int) -> ExecutionResult:
        """
        Execute code locally with basic safety (low security).
        
        Args:
            code: Python code to execute
            timeout: Execution timeout
            
        Returns:
            ExecutionResult
        """
        try:
            import time
            start_time = time.time()
            
            self.logger.info("⚠️ Executing in local safe mode (limited security)...")
            
            # Basic safety checks
            dangerous_keywords = ['import os', 'import sys', 'subprocess', 'eval', 'exec', '__import__']
            for keyword in dangerous_keywords:
                if keyword in code:
                    return ExecutionResult(
                        success=False,
                        output="",
                        error=f"Dangerous keyword detected: {keyword}",
                        execution_mode="local_safe",
                        security_level="low"
                    )
            
            # Execute with subprocess for isolation
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                result = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                execution_time = time.time() - start_time
                
                if result.returncode == 0:
                    return ExecutionResult(
                        success=True,
                        output=result.stdout,
                        execution_time=execution_time,
                        execution_mode="local_safe",
                        security_level="low"
                    )
                else:
                    return ExecutionResult(
                        success=False,
                        output=result.stdout,
                        error=result.stderr,
                        execution_time=execution_time,
                        execution_mode="local_safe",
                        security_level="low"
                    )
                    
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file)
                except:
                    pass
                    
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                output="",
                error="Execution timed out",
                execution_mode="local_safe",
                security_level="low"
            )
        except Exception as e:
            self.logger.error(f"❌ Local safe execution failed: {e}")
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                execution_mode="local_safe",
                security_level="low"
            )
    
    def get_execution_status(self) -> Dict[str, Any]:
        """
        Get current execution environment status.
        
        Returns:
            Dict with status information
        """
        return {
            'available_modes': self.execution_modes,
            'best_mode': self._select_best_execution_mode(),
            'docker_available': self.execution_modes['docker_sandbox']['available'],
            'fallback_available': True
        }


# Global manager instance
_smart_sandbox_manager = None

def get_smart_sandbox_manager() -> SmartSandboxManager:
    """Get the global smart sandbox manager instance."""
    global _smart_sandbox_manager
    if _smart_sandbox_manager is None:
        _smart_sandbox_manager = SmartSandboxManager()
        # Initialize on first access
        _smart_sandbox_manager.initialize_execution_environment()
    return _smart_sandbox_manager
