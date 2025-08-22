#!/usr/bin/env python3
"""
Docker Auto-Provisioner for SAM Code Interpreter
=================================================

Intelligent system to automatically detect, start, and manage Docker containers
for the SAM Code Interpreter sandbox service.

Features:
- Auto-detect Docker availability
- Auto-start Docker daemon if possible
- Auto-provision sandbox containers
- Smart fallback strategies
- Container lifecycle management

Author: SAM Development Team
Version: 1.0.0
"""

import os
import sys
import time
import logging
import subprocess
import platform
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class DockerAutoProvisioner:
    """
    Intelligent Docker auto-provisioning system for SAM Code Interpreter.
    
    Handles automatic detection, startup, and management of Docker containers
    for secure code execution.
    """
    
    def __init__(self):
        """Initialize the Docker auto-provisioner."""
        self.logger = logging.getLogger(f"{__name__}.DockerAutoProvisioner")
        self.system_platform = platform.system().lower()
        self.docker_available = False
        self.docker_daemon_running = False
        self.sandbox_container_ready = False
        
        # Container configuration
        self.sandbox_image = "python:3.11-slim"
        self.container_name = "sam-code-interpreter-sandbox"
        self.container_port = 6821
        
        self.logger.info("🐳 Docker Auto-Provisioner initialized")
    
    def check_docker_availability(self) -> Dict[str, Any]:
        """
        Check if Docker is available on the system.
        
        Returns:
            Dict with availability status and details
        """
        try:
            # Check if docker command exists
            result = subprocess.run(
                ["docker", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode == 0:
                docker_version = result.stdout.strip()
                self.docker_available = True
                self.logger.info(f"✅ Docker found: {docker_version}")
                
                return {
                    'available': True,
                    'version': docker_version,
                    'status': 'Docker is installed'
                }
            else:
                self.logger.warning("❌ Docker command failed")
                return {
                    'available': False,
                    'version': None,
                    'status': 'Docker command failed'
                }
                
        except FileNotFoundError:
            self.logger.warning("❌ Docker not found in PATH")
            return {
                'available': False,
                'version': None,
                'status': 'Docker not installed or not in PATH'
            }
        except subprocess.TimeoutExpired:
            self.logger.warning("❌ Docker command timed out")
            return {
                'available': False,
                'version': None,
                'status': 'Docker command timed out'
            }
        except Exception as e:
            self.logger.error(f"❌ Error checking Docker: {e}")
            return {
                'available': False,
                'version': None,
                'status': f'Error: {e}'
            }
    
    def check_docker_daemon_status(self) -> Dict[str, Any]:
        """
        Check if Docker daemon is running.
        
        Returns:
            Dict with daemon status and details
        """
        if not self.docker_available:
            return {
                'running': False,
                'status': 'Docker not available'
            }
        
        try:
            # Try to connect to Docker daemon
            result = subprocess.run(
                ["docker", "info"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode == 0:
                self.docker_daemon_running = True
                self.logger.info("✅ Docker daemon is running")
                return {
                    'running': True,
                    'status': 'Docker daemon is running'
                }
            else:
                self.logger.warning("❌ Docker daemon not running")
                return {
                    'running': False,
                    'status': 'Docker daemon not running'
                }
                
        except Exception as e:
            self.logger.error(f"❌ Error checking Docker daemon: {e}")
            return {
                'running': False,
                'status': f'Error: {e}'
            }
    
    def attempt_docker_start(self) -> Dict[str, Any]:
        """
        Attempt to start Docker daemon automatically.
        
        Returns:
            Dict with start attempt results
        """
        if not self.docker_available:
            return {
                'success': False,
                'message': 'Docker not available'
            }
        
        self.logger.info("🚀 Attempting to start Docker daemon...")
        
        try:
            if self.system_platform == "darwin":  # macOS
                # Try to start Docker Desktop
                subprocess.run(["open", "-a", "Docker"], check=False)
                self.logger.info("📱 Attempted to start Docker Desktop on macOS")
                
                # Wait for Docker to start
                for i in range(30):  # Wait up to 30 seconds
                    time.sleep(1)
                    status = self.check_docker_daemon_status()
                    if status['running']:
                        return {
                            'success': True,
                            'message': 'Docker Desktop started successfully'
                        }
                
                return {
                    'success': False,
                    'message': 'Docker Desktop did not start within 30 seconds'
                }
                
            elif self.system_platform == "linux":
                # Try to start Docker service
                try:
                    subprocess.run(["sudo", "systemctl", "start", "docker"], check=True)
                    self.logger.info("🐧 Started Docker service on Linux")
                    
                    # Wait for Docker to start
                    for i in range(10):  # Wait up to 10 seconds
                        time.sleep(1)
                        status = self.check_docker_daemon_status()
                        if status['running']:
                            return {
                                'success': True,
                                'message': 'Docker service started successfully'
                            }
                    
                    return {
                        'success': False,
                        'message': 'Docker service did not start within 10 seconds'
                    }
                    
                except subprocess.CalledProcessError:
                    return {
                        'success': False,
                        'message': 'Failed to start Docker service (may need sudo privileges)'
                    }
                    
            elif self.system_platform == "windows":
                # Try to start Docker Desktop on Windows
                subprocess.run(["start", "Docker Desktop"], shell=True, check=False)
                self.logger.info("🪟 Attempted to start Docker Desktop on Windows")
                
                # Wait for Docker to start
                for i in range(30):  # Wait up to 30 seconds
                    time.sleep(1)
                    status = self.check_docker_daemon_status()
                    if status['running']:
                        return {
                            'success': True,
                            'message': 'Docker Desktop started successfully'
                        }
                
                return {
                    'success': False,
                    'message': 'Docker Desktop did not start within 30 seconds'
                }
            
            else:
                return {
                    'success': False,
                    'message': f'Unsupported platform: {self.system_platform}'
                }
                
        except Exception as e:
            self.logger.error(f"❌ Error starting Docker: {e}")
            return {
                'success': False,
                'message': f'Error starting Docker: {e}'
            }
    
    def provision_sandbox_container(self) -> Dict[str, Any]:
        """
        Provision the sandbox container for code execution.
        
        Returns:
            Dict with provisioning results
        """
        if not self.docker_daemon_running:
            return {
                'success': False,
                'message': 'Docker daemon not running'
            }
        
        try:
            self.logger.info(f"🏗️ Provisioning sandbox container: {self.container_name}")
            
            # Check if container already exists
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if self.container_name in result.stdout:
                self.logger.info("📦 Container already exists, starting it...")
                
                # Start existing container
                start_result = subprocess.run(
                    ["docker", "start", self.container_name],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if start_result.returncode == 0:
                    self.sandbox_container_ready = True
                    return {
                        'success': True,
                        'message': 'Existing container started successfully'
                    }
                else:
                    self.logger.warning("Failed to start existing container, creating new one...")
            
            # Pull the image if needed
            self.logger.info(f"📥 Pulling Docker image: {self.sandbox_image}")
            pull_result = subprocess.run(
                ["docker", "pull", self.sandbox_image],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes for image pull
            )
            
            if pull_result.returncode != 0:
                self.logger.warning(f"Failed to pull image, using local copy if available")
            
            # Create and start the container
            self.logger.info("🚀 Creating sandbox container...")
            create_result = subprocess.run([
                "docker", "run", "-d",
                "--name", self.container_name,
                "-p", f"{self.container_port}:{self.container_port}",
                "--memory", "512m",
                "--cpus", "1.0",
                "--network", "none",  # No network access for security
                self.sandbox_image,
                "python", "-c", "import time; time.sleep(3600)"  # Keep container alive
            ], capture_output=True, text=True, timeout=60)
            
            if create_result.returncode == 0:
                self.sandbox_container_ready = True
                self.logger.info("✅ Sandbox container created and started successfully")
                return {
                    'success': True,
                    'message': 'Sandbox container provisioned successfully'
                }
            else:
                error_msg = create_result.stderr.strip()
                self.logger.error(f"❌ Failed to create container: {error_msg}")
                return {
                    'success': False,
                    'message': f'Failed to create container: {error_msg}'
                }
                
        except Exception as e:
            self.logger.error(f"❌ Error provisioning container: {e}")
            return {
                'success': False,
                'message': f'Error provisioning container: {e}'
            }
    
    def auto_provision_complete_environment(self) -> Dict[str, Any]:
        """
        Perform complete auto-provisioning of the Docker environment.
        
        Returns:
            Dict with complete provisioning results
        """
        self.logger.info("🎯 Starting complete Docker environment auto-provisioning...")
        
        results = {
            'docker_check': None,
            'daemon_check': None,
            'docker_start': None,
            'container_provision': None,
            'overall_success': False,
            'message': '',
            'fallback_recommended': False
        }
        
        # Step 1: Check Docker availability
        results['docker_check'] = self.check_docker_availability()
        if not results['docker_check']['available']:
            results['message'] = 'Docker not available on system'
            results['fallback_recommended'] = True
            return results
        
        # Step 2: Check Docker daemon status
        results['daemon_check'] = self.check_docker_daemon_status()
        if not results['daemon_check']['running']:
            # Step 3: Attempt to start Docker
            results['docker_start'] = self.attempt_docker_start()
            if not results['docker_start']['success']:
                results['message'] = 'Failed to start Docker daemon'
                results['fallback_recommended'] = True
                return results
        
        # Step 4: Provision sandbox container
        results['container_provision'] = self.provision_sandbox_container()
        if results['container_provision']['success']:
            results['overall_success'] = True
            results['message'] = 'Complete Docker environment provisioned successfully'
        else:
            results['message'] = 'Failed to provision sandbox container'
            results['fallback_recommended'] = True
        
        return results
    
    def cleanup_containers(self) -> Dict[str, Any]:
        """
        Clean up sandbox containers.
        
        Returns:
            Dict with cleanup results
        """
        try:
            self.logger.info("🧹 Cleaning up sandbox containers...")
            
            # Stop and remove the container
            subprocess.run(["docker", "stop", self.container_name], 
                         capture_output=True, check=False)
            subprocess.run(["docker", "rm", self.container_name], 
                         capture_output=True, check=False)
            
            self.sandbox_container_ready = False
            return {
                'success': True,
                'message': 'Containers cleaned up successfully'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error cleaning up containers: {e}")
            return {
                'success': False,
                'message': f'Error cleaning up: {e}'
            }


# Global provisioner instance
_docker_provisioner = None

def get_docker_auto_provisioner() -> DockerAutoProvisioner:
    """Get the global Docker auto-provisioner instance."""
    global _docker_provisioner
    if _docker_provisioner is None:
        _docker_provisioner = DockerAutoProvisioner()
    return _docker_provisioner
