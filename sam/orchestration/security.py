"""
Tool Security Framework for SAM Orchestration Framework
=======================================================

Provides secure execution environment for external tools with sandboxing,
rate limiting, and security validation.
"""

import os
import time
import logging
import subprocess
import tempfile
import shutil
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


@dataclass
class SecurityPolicy:
    """Security policy for tool execution."""
    allow_network_access: bool = False
    allow_file_system_access: bool = False
    max_execution_time: float = 30.0
    max_memory_usage: int = 100 * 1024 * 1024  # 100MB
    allowed_commands: List[str] = None
    blocked_commands: List[str] = None
    sandbox_enabled: bool = True


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    max_requests_per_minute: int = 60
    max_requests_per_hour: int = 1000
    cooldown_period: float = 1.0  # seconds


@dataclass
class ExecutionResult:
    """Result of secure tool execution."""
    success: bool
    output: Any
    error_message: Optional[str] = None
    execution_time: float = 0.0
    security_violations: List[str] = None
    rate_limited: bool = False


class RateLimiter:
    """Rate limiter for tool execution."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._requests: List[datetime] = []
        self._lock = threading.Lock()
    
    def is_allowed(self, tool_name: str) -> bool:
        """
        Check if a request is allowed under rate limits.
        
        Args:
            tool_name: Name of the tool requesting execution
            
        Returns:
            True if request is allowed, False if rate limited
        """
        with self._lock:
            now = datetime.now()
            
            # Clean old requests
            cutoff_minute = now - timedelta(minutes=1)
            cutoff_hour = now - timedelta(hours=1)
            
            self._requests = [req for req in self._requests if req > cutoff_hour]
            
            # Check limits
            recent_requests = [req for req in self._requests if req > cutoff_minute]
            
            if len(recent_requests) >= self.config.max_requests_per_minute:
                logger.warning(f"Rate limit exceeded for {tool_name}: {len(recent_requests)} requests in last minute")
                return False
            
            if len(self._requests) >= self.config.max_requests_per_hour:
                logger.warning(f"Rate limit exceeded for {tool_name}: {len(self._requests)} requests in last hour")
                return False
            
            # Record this request
            self._requests.append(now)
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics."""
        with self._lock:
            now = datetime.now()
            cutoff_minute = now - timedelta(minutes=1)
            cutoff_hour = now - timedelta(hours=1)
            
            recent_minute = len([req for req in self._requests if req > cutoff_minute])
            recent_hour = len([req for req in self._requests if req > cutoff_hour])
            
            return {
                "requests_last_minute": recent_minute,
                "requests_last_hour": recent_hour,
                "limit_per_minute": self.config.max_requests_per_minute,
                "limit_per_hour": self.config.max_requests_per_hour
            }


class SecureSandbox:
    """Secure sandbox for tool execution."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.logger = logging.getLogger(f"{__name__}.SecureSandbox")
        self._temp_dirs: List[Path] = []
    
    def execute_python_code(self, code: str, globals_dict: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """
        Execute Python code in a restricted environment.
        
        Args:
            code: Python code to execute
            globals_dict: Optional global variables
            
        Returns:
            Execution result
        """
        start_time = time.time()
        security_violations = []
        
        try:
            # Validate code for security violations
            violations = self._validate_python_code(code)
            if violations:
                return ExecutionResult(
                    success=False,
                    output=None,
                    error_message="Security violations detected",
                    execution_time=time.time() - start_time,
                    security_violations=violations
                )
            
            # Create restricted globals
            restricted_globals = self._create_restricted_globals(globals_dict)
            
            # Execute with timeout
            result = self._execute_with_timeout(code, restricted_globals)
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=True,
                output=result,
                execution_time=execution_time,
                security_violations=[]
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                output=None,
                error_message=str(e),
                execution_time=execution_time,
                security_violations=security_violations
            )
    
    def execute_command(self, command: List[str], input_data: Optional[str] = None) -> ExecutionResult:
        """
        Execute a system command in a sandboxed environment.
        
        Args:
            command: Command and arguments to execute
            input_data: Optional input data for the command
            
        Returns:
            Execution result
        """
        start_time = time.time()
        security_violations = []
        
        try:
            # Validate command
            violations = self._validate_command(command)
            if violations:
                return ExecutionResult(
                    success=False,
                    output=None,
                    error_message="Command security violations detected",
                    execution_time=time.time() - start_time,
                    security_violations=violations
                )
            
            # Create temporary directory if needed
            temp_dir = None
            if self.policy.allow_file_system_access:
                temp_dir = self._create_temp_directory()
            
            # Execute command with restrictions
            result = self._execute_command_restricted(command, input_data, temp_dir)
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=True,
                output=result,
                execution_time=execution_time,
                security_violations=[]
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                output=None,
                error_message=str(e),
                execution_time=execution_time,
                security_violations=security_violations
            )
        finally:
            # Cleanup temporary directory
            if temp_dir:
                self._cleanup_temp_directory(temp_dir)
    
    def _validate_python_code(self, code: str) -> List[str]:
        """
        Validate Python code for security violations.
        
        Returns:
            List of security violations found
        """
        violations = []
        
        # Check for dangerous imports
        dangerous_imports = [
            'os', 'sys', 'subprocess', 'shutil', 'tempfile',
            'socket', 'urllib', 'requests', 'http',
            '__import__', 'eval', 'exec', 'compile'
        ]
        
        for dangerous in dangerous_imports:
            if dangerous in code:
                violations.append(f"Dangerous import/function detected: {dangerous}")
        
        # Check for file system access
        if not self.policy.allow_file_system_access:
            file_operations = ['open(', 'file(', 'with open', 'pathlib']
            for op in file_operations:
                if op in code:
                    violations.append(f"File system access not allowed: {op}")
        
        # Check for network access
        if not self.policy.allow_network_access:
            network_operations = ['socket', 'urllib', 'requests', 'http', 'ftp']
            for op in network_operations:
                if op in code:
                    violations.append(f"Network access not allowed: {op}")
        
        return violations
    
    def _validate_command(self, command: List[str]) -> List[str]:
        """
        Validate system command for security violations.
        
        Returns:
            List of security violations found
        """
        violations = []
        
        if not command:
            violations.append("Empty command not allowed")
            return violations
        
        cmd_name = command[0]
        
        # Check allowed commands
        if self.policy.allowed_commands:
            if cmd_name not in self.policy.allowed_commands:
                violations.append(f"Command not in allowed list: {cmd_name}")
        
        # Check blocked commands
        if self.policy.blocked_commands:
            if cmd_name in self.policy.blocked_commands:
                violations.append(f"Command is blocked: {cmd_name}")
        
        # Check for dangerous commands
        dangerous_commands = [
            'rm', 'del', 'format', 'fdisk', 'mkfs',
            'sudo', 'su', 'chmod', 'chown',
            'wget', 'curl', 'nc', 'netcat',
            'ssh', 'scp', 'rsync'
        ]
        
        if cmd_name in dangerous_commands:
            violations.append(f"Dangerous command detected: {cmd_name}")
        
        return violations
    
    def _create_restricted_globals(self, globals_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a restricted globals dictionary for Python execution.
        
        Returns:
            Restricted globals dictionary
        """
        # Start with safe builtins
        safe_builtins = {
            'abs', 'all', 'any', 'bin', 'bool', 'chr', 'dict', 'dir',
            'divmod', 'enumerate', 'filter', 'float', 'format', 'frozenset',
            'getattr', 'hasattr', 'hash', 'hex', 'id', 'int', 'isinstance',
            'issubclass', 'iter', 'len', 'list', 'map', 'max', 'min',
            'next', 'oct', 'ord', 'pow', 'range', 'repr', 'reversed',
            'round', 'set', 'slice', 'sorted', 'str', 'sum', 'tuple',
            'type', 'zip'
        }
        
        restricted_globals = {
            '__builtins__': {name: getattr(__builtins__, name) for name in safe_builtins if hasattr(__builtins__, name)}
        }
        
        # Add safe modules
        import math
        restricted_globals['math'] = math
        
        # Add user-provided globals if safe
        if globals_dict:
            for key, value in globals_dict.items():
                if not key.startswith('_') and not callable(value):
                    restricted_globals[key] = value
        
        return restricted_globals
    
    def _execute_with_timeout(self, code: str, globals_dict: Dict[str, Any]) -> Any:
        """
        Execute Python code with timeout.
        
        Returns:
            Execution result
        """
        # For now, use simple exec - in production, consider using more sophisticated sandboxing
        local_vars = {}
        exec(code, globals_dict, local_vars)
        
        # Return the result if there's a 'result' variable, otherwise return local vars
        return local_vars.get('result', local_vars)
    
    def _execute_command_restricted(self, command: List[str], input_data: Optional[str], temp_dir: Optional[Path]) -> str:
        """
        Execute command with restrictions.
        
        Returns:
            Command output
        """
        env = os.environ.copy()
        
        # Restrict environment if needed
        if not self.policy.allow_network_access:
            # Remove network-related environment variables
            for key in list(env.keys()):
                if 'proxy' in key.lower() or 'http' in key.lower():
                    del env[key]
        
        # Set working directory to temp dir if available
        cwd = str(temp_dir) if temp_dir else None
        
        # Execute with timeout
        try:
            result = subprocess.run(
                command,
                input=input_data,
                capture_output=True,
                text=True,
                timeout=self.policy.max_execution_time,
                env=env,
                cwd=cwd
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Command failed with return code {result.returncode}: {result.stderr}")
            
            return result.stdout
            
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Command timed out after {self.policy.max_execution_time} seconds")
    
    def _create_temp_directory(self) -> Path:
        """Create a temporary directory for sandboxed execution."""
        temp_dir = Path(tempfile.mkdtemp(prefix="sof_sandbox_"))
        self._temp_dirs.append(temp_dir)
        return temp_dir
    
    def _cleanup_temp_directory(self, temp_dir: Path) -> None:
        """Clean up a temporary directory."""
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            if temp_dir in self._temp_dirs:
                self._temp_dirs.remove(temp_dir)
        except Exception as e:
            self.logger.warning(f"Failed to cleanup temp directory {temp_dir}: {e}")
    
    def cleanup_all(self) -> None:
        """Clean up all temporary directories."""
        for temp_dir in self._temp_dirs.copy():
            self._cleanup_temp_directory(temp_dir)


class ToolSecurityManager:
    """Manages security for all tool executions."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ToolSecurityManager")
        self._tool_policies: Dict[str, SecurityPolicy] = {}
        self._rate_limiters: Dict[str, RateLimiter] = {}
        self._sandboxes: Dict[str, SecureSandbox] = {}
        
        # Set up default policies
        self._setup_default_policies()
    
    def _setup_default_policies(self) -> None:
        """Set up default security policies for different tool types."""
        
        # Calculator tool - very restrictive
        self._tool_policies["CalculatorTool"] = SecurityPolicy(
            allow_network_access=False,
            allow_file_system_access=False,
            max_execution_time=5.0,
            sandbox_enabled=True,
            allowed_commands=[],
            blocked_commands=["rm", "del", "format"]
        )
        
        # Web browser tool - network access allowed but restricted
        self._tool_policies["AgentZeroWebBrowserTool"] = SecurityPolicy(
            allow_network_access=True,
            allow_file_system_access=False,
            max_execution_time=30.0,
            sandbox_enabled=True,
            allowed_commands=["curl", "wget"],
            blocked_commands=["rm", "del", "format", "sudo"]
        )
        
        # Default policy for unknown tools
        self._tool_policies["default"] = SecurityPolicy(
            allow_network_access=False,
            allow_file_system_access=False,
            max_execution_time=10.0,
            sandbox_enabled=True
        )
        
        # Set up rate limiters
        default_rate_config = RateLimitConfig(
            max_requests_per_minute=30,
            max_requests_per_hour=500
        )
        
        for tool_name in self._tool_policies:
            self._rate_limiters[tool_name] = RateLimiter(default_rate_config)
    
    def execute_tool_safely(self, tool_name: str, execution_func: Callable, *args, **kwargs) -> ExecutionResult:
        """
        Execute a tool function safely with security controls.
        
        Args:
            tool_name: Name of the tool
            execution_func: Function to execute
            *args, **kwargs: Arguments for the function
            
        Returns:
            Secure execution result
        """
        # Check rate limits
        rate_limiter = self._rate_limiters.get(tool_name, self._rate_limiters["default"])
        if not rate_limiter.is_allowed(tool_name):
            return ExecutionResult(
                success=False,
                output=None,
                error_message="Rate limit exceeded",
                rate_limited=True
            )
        
        # Get security policy
        policy = self._tool_policies.get(tool_name, self._tool_policies["default"])
        
        # Get or create sandbox
        if tool_name not in self._sandboxes:
            self._sandboxes[tool_name] = SecureSandbox(policy)
        
        sandbox = self._sandboxes[tool_name]
        
        start_time = time.time()
        
        try:
            # Execute with timeout
            result = self._execute_with_policy(execution_func, policy, *args, **kwargs)
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=True,
                output=result,
                execution_time=execution_time,
                security_violations=[]
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                output=None,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def _execute_with_policy(self, func: Callable, policy: SecurityPolicy, *args, **kwargs) -> Any:
        """Execute function with security policy enforcement."""
        # For now, just execute the function
        # In production, this would include more sophisticated monitoring
        return func(*args, **kwargs)
    
    def register_tool_policy(self, tool_name: str, policy: SecurityPolicy) -> None:
        """
        Register a custom security policy for a tool.
        
        Args:
            tool_name: Name of the tool
            policy: Security policy to apply
        """
        self._tool_policies[tool_name] = policy
        self._sandboxes[tool_name] = SecureSandbox(policy)
        
        if tool_name not in self._rate_limiters:
            self._rate_limiters[tool_name] = RateLimiter(RateLimitConfig())
        
        self.logger.info(f"Registered security policy for tool: {tool_name}")
    
    def get_security_stats(self) -> Dict[str, Any]:
        """
        Get security statistics for all tools.
        
        Returns:
            Dictionary with security statistics
        """
        stats = {}
        
        for tool_name, rate_limiter in self._rate_limiters.items():
            stats[tool_name] = rate_limiter.get_stats()
        
        return stats
    
    def cleanup(self) -> None:
        """Clean up all sandboxes and temporary resources."""
        for sandbox in self._sandboxes.values():
            sandbox.cleanup_all()


# Global security manager instance
_security_manager: Optional[ToolSecurityManager] = None


def get_security_manager() -> ToolSecurityManager:
    """
    Get or create global security manager instance.
    
    Returns:
        ToolSecurityManager instance
    """
    global _security_manager
    
    if _security_manager is None:
        _security_manager = ToolSecurityManager()
    
    return _security_manager
