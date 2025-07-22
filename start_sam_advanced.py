#!/usr/bin/env python3
"""
SAM Main Launcher
================

Streamlined launcher for SAM with first-time setup detection.
Automatically guides new users through setup and launches SAM.

Usage: python start_sam.py
"""

import os
import sys
import time
import signal
import logging
import threading
import subprocess
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import SAM components
from config.config_manager import get_config_manager
from config.onboarding import get_onboarding_manager
from utils.health_monitor import HealthMonitor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/sam_launcher.log')
    ]
)
logger = logging.getLogger(__name__)

class SAMLauncher:
    """
    Unified launcher for SAM with health monitoring and graceful shutdown.
    """
    
    def __init__(self):
        """Initialize the SAM launcher."""
        self.config_manager = get_config_manager()
        self.onboarding_manager = get_onboarding_manager()
        self.config = self.config_manager.get_config()
        
        self.processes = {}
        self.health_monitor = None
        self.shutdown_event = threading.Event()
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("SAM Launcher initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()
    
    def _check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        try:
            # Check Python packages
            required_packages = [
                'streamlit', 'flask', 'requests', 'numpy',
                'sentence_transformers'
            ]

            # Optional packages (warn but don't fail)
            optional_packages = ['faiss-cpu']
            
            missing_packages = []
            for package in required_packages:
                try:
                    __import__(package.replace('-', '_'))
                except ImportError:
                    missing_packages.append(package)

            if missing_packages:
                logger.error(f"Missing required packages: {', '.join(missing_packages)}")
                logger.info("Install them with: pip install " + " ".join(missing_packages))
                return False

            # Check optional packages
            missing_optional = []
            for package in optional_packages:
                try:
                    __import__(package.replace('-', '_'))
                except ImportError:
                    missing_optional.append(package)

            if missing_optional:
                logger.warning(f"Optional packages not installed: {', '.join(missing_optional)}")
                logger.info("These are optional but may improve performance: pip install " + " ".join(missing_optional))
            
            # Check directories
            required_dirs = [
                self.config.memory_storage_dir,
                "logs",
                "config",
                "backups"
            ]
            
            for dir_path in required_dirs:
                Path(dir_path).mkdir(exist_ok=True)
            
            logger.info("All dependencies and directories are available")
            return True
            
        except Exception as e:
            logger.error(f"Error checking dependencies: {e}")
            return False
    
    def _start_web_interface(self) -> bool:
        """Start the web chat interface."""
        try:
            logger.info(f"Starting web chat interface on port {self.config.chat_port}...")
            
            cmd = [
                sys.executable, "launch_web_ui.py",
                "--port", str(self.config.chat_port),
                "--host", self.config.host
            ]
            
            if self.config.debug_mode:
                cmd.append("--debug")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(Path(__file__).parent)
            )
            
            self.processes['web_interface'] = process
            
            # Wait a moment and check if it started successfully
            time.sleep(3)
            if process.poll() is None:
                logger.info(f"Web interface started successfully (PID: {process.pid})")
                return True
            else:
                logger.error("Web interface failed to start")
                return False
                
        except Exception as e:
            logger.error(f"Error starting web interface: {e}")
            return False
    
    def _start_memory_ui(self) -> bool:
        """Start the memory control center UI."""
        try:
            logger.info(f"Starting Memory Control Center on port {self.config.memory_ui_port}...")

            cmd = [
                sys.executable, "-m", "streamlit", "run",
                "ui/memory_app.py",
                "--server.port", str(self.config.memory_ui_port),
                "--server.address", self.config.host,
                "--browser.gatherUsageStats", "false",
                "--server.headless", "true"
            ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(Path(__file__).parent)
            )

            self.processes['memory_ui'] = process

            # Wait a moment and check if it started successfully
            time.sleep(5)
            if process.poll() is None:
                logger.info(f"Memory Control Center started successfully (PID: {process.pid})")
                return True
            else:
                logger.error("Memory Control Center failed to start")
                return False

        except Exception as e:
            logger.error(f"Error starting Memory Control Center: {e}")
            return False

    def _start_streamlit_chat(self) -> bool:
        """Start the Streamlit chat interface."""
        try:
            logger.info(f"Starting Streamlit Chat Interface on port {self.config.streamlit_chat_port}...")

            cmd = [
                sys.executable, "-m", "streamlit", "run",
                "secure_streamlit_app.py",
                "--server.port", str(self.config.streamlit_chat_port),
                "--server.address", self.config.host,
                "--browser.gatherUsageStats", "false",
                "--server.headless", "true"
            ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(Path(__file__).parent)
            )

            self.processes['streamlit_chat'] = process

            # Wait a moment and check if it started successfully
            time.sleep(5)
            if process.poll() is None:
                logger.info(f"Streamlit Chat Interface started successfully (PID: {process.pid})")
                return True
            else:
                logger.error("Streamlit Chat Interface failed to start")
                return False

        except Exception as e:
            logger.error(f"Error starting Streamlit Chat Interface: {e}")
            return False
    
    def _start_health_monitor(self):
        """Start the health monitoring system."""
        try:
            self.health_monitor = HealthMonitor(
                chat_port=self.config.chat_port,
                memory_ui_port=self.config.memory_ui_port,
                streamlit_chat_port=self.config.streamlit_chat_port
            )
            
            # Start health monitoring in a separate thread
            health_thread = threading.Thread(
                target=self.health_monitor.start_monitoring,
                daemon=True
            )
            health_thread.start()
            
            logger.info("Health monitoring started")
            
        except Exception as e:
            logger.error(f"Error starting health monitor: {e}")
    
    def _open_browser(self):
        """Open browser to SAM interfaces."""
        if not self.config.auto_open_browser:
            return
        
        try:
            # Wait a moment for services to be ready
            time.sleep(2)

            # Open default Streamlit chat interface
            streamlit_chat_url = f"http://{self.config.host}:{self.config.streamlit_chat_port}"
            logger.info(f"Opening browser to default chat interface: {streamlit_chat_url}")
            webbrowser.open(streamlit_chat_url)
            
            # If first launch, also show onboarding info
            if self.onboarding_manager.should_show_welcome():
                logger.info("First launch detected - onboarding will be shown")
                
        except Exception as e:
            logger.error(f"Error opening browser: {e}")
    
    def _show_startup_info(self):
        """Display startup information."""
        print("\n" + "="*70)
        print("🤖 SAM (Small Agent Model) - Starting Up")
        print("="*70)
        print(f"Version: {self.config.version}")
        print(f"Agent Mode: {self.config.agent_mode}")
        print(f"Memory Backend: {self.config.memory_backend}")
        print(f"Model Provider: {self.config.model_provider}")
        print()
        print("🌐 Web Interfaces:")
        print(f"  Chat Interface:      http://{self.config.host}:{self.config.chat_port}")
        print(f"  Streamlit Chat:      http://{self.config.host}:{self.config.streamlit_chat_port} (Default)")
        print(f"  Memory Control:      http://{self.config.host}:{self.config.memory_ui_port}")
        print()
        print("📁 Storage Locations:")
        print(f"  Memory Store:        {self.config.memory_storage_dir}")
        print(f"  Configuration:       config/")
        print(f"  Logs:               logs/")
        print(f"  Backups:            backups/")
        print()
        print("🔧 Controls:")
        print("  Press Ctrl+C to stop SAM")
        print("  Check logs/sam_launcher.log for detailed logs")
        print("="*70)
        print()
    
    def _monitor_processes(self):
        """Monitor running processes and restart if needed."""
        while not self.shutdown_event.is_set():
            try:
                # Check each process
                for name, process in list(self.processes.items()):
                    if process.poll() is not None:
                        logger.warning(f"Process {name} has stopped unexpectedly")
                        
                        # Attempt restart
                        if name == 'web_interface':
                            logger.info(f"Restarting {name}...")
                            if self._start_web_interface():
                                logger.info(f"Successfully restarted {name}")
                            else:
                                logger.error(f"Failed to restart {name}")
                        elif name == 'memory_ui':
                            logger.info(f"Restarting {name}...")
                            if self._start_memory_ui():
                                logger.info(f"Successfully restarted {name}")
                            else:
                                logger.error(f"Failed to restart {name}")
                        elif name == 'streamlit_chat':
                            logger.info(f"Restarting {name}...")
                            if self._start_streamlit_chat():
                                logger.info(f"Successfully restarted {name}")
                            else:
                                logger.error(f"Failed to restart {name}")
                
                # Wait before next check
                self.shutdown_event.wait(30)
                
            except Exception as e:
                logger.error(f"Error in process monitoring: {e}")
                self.shutdown_event.wait(10)
    
    def _graceful_shutdown(self):
        """Perform graceful shutdown of all services."""
        logger.info("Starting graceful shutdown...")
        
        # Stop health monitor
        if self.health_monitor:
            self.health_monitor.stop_monitoring()
        
        # Stop all processes
        for name, process in self.processes.items():
            try:
                logger.info(f"Stopping {name}...")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=10)
                    logger.info(f"{name} stopped gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning(f"{name} did not stop gracefully, forcing...")
                    process.kill()
                    process.wait()
                    
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")
        
        logger.info("SAM shutdown complete")
    
    def start(self):
        """Start SAM with all components."""
        try:
            logger.info("Starting SAM...")
            
            # Show startup information
            self._show_startup_info()
            
            # Check dependencies
            if not self._check_dependencies():
                logger.error("Dependency check failed")
                return False
            
            # Validate configuration
            validation = self.config_manager.validate_config()
            if not validation['valid']:
                logger.error("Configuration validation failed:")
                for error in validation['errors']:
                    logger.error(f"  - {error}")
                return False
            
            if validation['warnings']:
                for warning in validation['warnings']:
                    logger.warning(f"Config warning: {warning}")
            
            # Start web interface
            if not self._start_web_interface():
                logger.error("Failed to start web interface")
                return False
            
            # Start memory UI
            if not self._start_memory_ui():
                logger.error("Failed to start Memory Control Center")
                return False

            # Start Streamlit chat interface
            if not self._start_streamlit_chat():
                logger.error("Failed to start Streamlit Chat Interface")
                return False

            # Start health monitoring
            self._start_health_monitor()
            
            # Open browser
            if self.config.auto_open_browser:
                browser_thread = threading.Thread(target=self._open_browser, daemon=True)
                browser_thread.start()
            
            # Start process monitoring
            monitor_thread = threading.Thread(target=self._monitor_processes, daemon=True)
            monitor_thread.start()
            
            logger.info("SAM started successfully!")
            print("✅ SAM is now running!")
            print(f"🌐 Chat Interface: http://{self.config.host}:{self.config.chat_port}")
            print(f"📱 Streamlit Chat: http://{self.config.host}:{self.config.streamlit_chat_port} (Default)")
            print(f"🧠 Memory Control: http://{self.config.host}:{self.config.memory_ui_port}")
            print("\nPress Ctrl+C to stop SAM")
            
            # Wait for shutdown signal
            self.shutdown_event.wait()
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting SAM: {e}")
            return False
        finally:
            self._graceful_shutdown()

def main():
    """Main entry point."""
    try:
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
        
        # Initialize and start SAM
        launcher = SAMLauncher()
        success = launcher.start()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n👋 SAM stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
