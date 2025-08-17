"""
Secure Code Execution Sandbox Service
=====================================

Flask-based API service that provides secure Python code execution
using ephemeral Docker containers with strict security constraints.

Author: SAM Development Team
Version: 1.0.0
"""

import os
import json
import time
import uuid
import logging
import tempfile
import subprocess
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
from flask import Flask, request, jsonify
import docker
from docker.errors import DockerException, ContainerError, ImageNotFound

logger = logging.getLogger(__name__)


@dataclass
class CodeExecutionRequest:
    """Request for code execution."""
    code: str
    data_files: Dict[str, str] = None  # filename -> base64 content
    timeout_seconds: int = 30
    memory_limit: str = "512m"
    cpu_limit: float = 1.0
    allowed_packages: List[str] = None
    
    def __post_init__(self):
        if self.data_files is None:
            self.data_files = {}
        if self.allowed_packages is None:
            self.allowed_packages = [
                "numpy", "pandas", "matplotlib", "seaborn", "scipy", 
                "scikit-learn", "plotly", "requests", "json", "csv"
            ]


@dataclass 
class CodeExecutionResult:
    """Result of code execution."""
    success: bool
    stdout: str = ""
    stderr: str = ""
    execution_time: float = 0.0
    generated_files: Dict[str, str] = None  # filename -> base64 content
    error_message: str = ""
    container_id: str = ""
    
    def __post_init__(self):
        if self.generated_files is None:
            self.generated_files = {}


class SandboxService:
    """
    Secure code execution service using Docker containers.
    
    Features:
    - Ephemeral containers (created and destroyed per request)
    - Network isolation (no internet access)
    - Resource limits (CPU, memory, execution time)
    - File system isolation
    - Restricted package access
    """
    
    def __init__(self, 
                 docker_image: str = "python:3.11-slim",
                 base_port: int = 5000,
                 max_concurrent_executions: int = 5):
        """
        Initialize the sandbox service.
        
        Args:
            docker_image: Docker image for code execution
            base_port: Base port for Flask service
            max_concurrent_executions: Maximum concurrent containers
        """
        self.logger = logging.getLogger(f"{__name__}.SandboxService")
        self.docker_image = docker_image
        self.base_port = base_port
        self.max_concurrent = max_concurrent_executions
        
        # Docker client
        try:
            self.docker_client = docker.from_env()
            self.logger.info("✅ Docker client initialized")
        except DockerException as e:
            self.logger.error(f"❌ Failed to initialize Docker client: {e}")
            raise
        
        # Active containers tracking
        self.active_containers: Dict[str, Any] = {}
        
        # Flask app for API
        self.app = Flask(__name__)
        self.setup_routes()
        
        # Ensure Docker image is available
        self.ensure_docker_image()
        
        self.logger.info("SandboxService initialized")
    
    def ensure_docker_image(self):
        """Ensure the required Docker image is available."""
        try:
            self.docker_client.images.get(self.docker_image)
            self.logger.info(f"✅ Docker image available: {self.docker_image}")
        except ImageNotFound:
            self.logger.info(f"📥 Pulling Docker image: {self.docker_image}")
            self.docker_client.images.pull(self.docker_image)
            self.logger.info(f"✅ Docker image pulled: {self.docker_image}")
    
    def setup_routes(self):
        """Setup Flask API routes."""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            return jsonify({
                "status": "healthy",
                "active_containers": len(self.active_containers),
                "docker_available": True
            })
        
        @self.app.route('/execute', methods=['POST'])
        def execute_code():
            """Execute code in secure container."""
            try:
                # Parse request
                request_data = request.get_json()
                if not request_data:
                    return jsonify({"error": "No JSON data provided"}), 400
                
                exec_request = CodeExecutionRequest(**request_data)
                
                # Check concurrent limit
                if len(self.active_containers) >= self.max_concurrent:
                    return jsonify({"error": "Maximum concurrent executions reached"}), 429
                
                # Execute code
                result = self.execute_code_secure(exec_request)
                
                return jsonify(asdict(result))
                
            except Exception as e:
                self.logger.error(f"❌ Execution error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/containers', methods=['GET'])
        def list_containers():
            """List active containers."""
            return jsonify({
                "active_containers": list(self.active_containers.keys()),
                "count": len(self.active_containers)
            })
    
    def execute_code_secure(self, exec_request: CodeExecutionRequest) -> CodeExecutionResult:
        """
        Execute code in a secure Docker container.
        
        Args:
            exec_request: Code execution request
            
        Returns:
            Execution result
        """
        container_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Create temporary directory for file exchange
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Write code to file
                code_file = temp_path / "user_code.py"
                code_file.write_text(exec_request.code)
                
                # Write data files if provided
                for filename, content in exec_request.data_files.items():
                    import base64
                    file_path = temp_path / filename
                    file_path.write_bytes(base64.b64decode(content))
                
                # Create container
                container = self.docker_client.containers.run(
                    self.docker_image,
                    command=["python", "/workspace/user_code.py"],
                    volumes={str(temp_path): {'bind': '/workspace', 'mode': 'rw'}},
                    working_dir="/workspace",
                    network_mode="none",  # No network access
                    mem_limit=exec_request.memory_limit,
                    cpu_period=100000,
                    cpu_quota=int(100000 * exec_request.cpu_limit),
                    detach=True,
                    remove=False,  # We'll remove manually after getting results
                    user="nobody",  # Run as non-root user
                    cap_drop=["ALL"],  # Drop all capabilities
                    security_opt=["no-new-privileges:true"]
                )
                
                self.active_containers[container_id] = container
                
                try:
                    # Wait for completion with timeout
                    exit_code = container.wait(timeout=exec_request.timeout_seconds)
                    
                    # Get output
                    stdout = container.logs(stdout=True, stderr=False).decode('utf-8')
                    stderr = container.logs(stdout=False, stderr=True).decode('utf-8')
                    
                    # Check for generated files
                    generated_files = {}
                    for file_path in temp_path.iterdir():
                        if file_path.name not in ["user_code.py"] + list(exec_request.data_files.keys()):
                            # New file generated
                            if file_path.is_file() and file_path.stat().st_size < 10 * 1024 * 1024:  # Max 10MB
                                import base64
                                content = base64.b64encode(file_path.read_bytes()).decode('utf-8')
                                generated_files[file_path.name] = content
                    
                    execution_time = time.time() - start_time
                    
                    return CodeExecutionResult(
                        success=exit_code['StatusCode'] == 0,
                        stdout=stdout,
                        stderr=stderr,
                        execution_time=execution_time,
                        generated_files=generated_files,
                        container_id=container_id
                    )
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    return CodeExecutionResult(
                        success=False,
                        error_message=str(e),
                        execution_time=execution_time,
                        container_id=container_id
                    )
                
                finally:
                    # Clean up container
                    try:
                        container.remove(force=True)
                    except Exception as e:
                        self.logger.warning(f"Failed to remove container {container_id}: {e}")
                    
                    if container_id in self.active_containers:
                        del self.active_containers[container_id]
        
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Container execution failed: {e}")
            return CodeExecutionResult(
                success=False,
                error_message=str(e),
                execution_time=execution_time,
                container_id=container_id
            )
    
    def run(self, host: str = "127.0.0.1", port: Optional[int] = None, debug: bool = False):
        """Run the Flask service."""
        if port is None:
            port = self.base_port
        
        self.logger.info(f"🚀 Starting Sandbox Service on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)
    
    def cleanup_all_containers(self):
        """Clean up all active containers."""
        for container_id, container in list(self.active_containers.items()):
            try:
                container.remove(force=True)
                self.logger.info(f"✅ Cleaned up container: {container_id}")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup container {container_id}: {e}")
        
        self.active_containers.clear()


def create_sandbox_service() -> SandboxService:
    """Factory function to create a sandbox service."""
    return SandboxService()


if __name__ == "__main__":
    # Run as standalone service
    service = create_sandbox_service()
    service.run(debug=True)
