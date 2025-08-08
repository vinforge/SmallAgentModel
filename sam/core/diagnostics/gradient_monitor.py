"""
Gradient Health Monitoring System

PINN-inspired gradient diagnostics for SAM's lifelong learning processes.
Monitors gradient health during MEMOIR edits to detect and prevent learning failures.

Key Features:
- Real-time gradient norm monitoring
- Detection of vanishing/exploding gradients
- Noisy convergence analysis
- Learning stability diagnostics
- Integration with SOF error reporting

Author: SAM Development Team
Version: 1.0.0
"""

import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class GradientPathology(Enum):
    """Types of gradient pathologies that can be detected."""
    HEALTHY = "healthy"
    VANISHING = "vanishing_gradients"
    EXPLODING = "exploding_gradients"
    NOISY = "noisy_convergence"
    STALLED = "learning_stalled"
    OSCILLATING = "oscillating_gradients"

@dataclass
class GradientSnapshot:
    """Snapshot of gradient information at a training step."""
    step: int
    gradient_norm: float
    layer_norms: Dict[str, float]
    timestamp: datetime
    loss_value: Optional[float] = None
    learning_rate: Optional[float] = None

@dataclass
class GradientHealthReport:
    """Comprehensive report on gradient health status."""
    pathology: GradientPathology
    severity: float  # 0.0 (mild) to 1.0 (severe)
    description: str
    recommendations: List[str]
    snapshots: List[GradientSnapshot]
    analysis_window: int
    confidence: float

class GradientLogger:
    """
    Context manager for logging gradients during training.
    
    Captures gradient norms and provides them to the GradientHealthMonitor
    for analysis and pathology detection.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: Optional[List[str]] = None,
        log_frequency: int = 1
    ):
        """
        Initialize gradient logger.
        
        Args:
            model: PyTorch model to monitor
            target_layers: Specific layers to monitor (None = all layers)
            log_frequency: Log every N steps
        """
        self.model = model
        self.target_layers = target_layers or []
        self.log_frequency = log_frequency
        self.snapshots: List[GradientSnapshot] = []
        self.step_count = 0
        self.hooks = []
        
        self.logger = logging.getLogger(f"{__name__}.GradientLogger")
    
    def __enter__(self):
        """Enter context manager and start gradient logging."""
        self._register_hooks()
        self.logger.info("Gradient logging started")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and clean up hooks."""
        self._remove_hooks()
        self.logger.info(f"Gradient logging completed. Captured {len(self.snapshots)} snapshots")
    
    def _register_hooks(self):
        """Register backward hooks to capture gradients."""
        for name, module in self.model.named_modules():
            if self._should_monitor_layer(name):
                hook = module.register_backward_hook(
                    lambda module, grad_input, grad_output, layer_name=name: 
                    self._gradient_hook(layer_name, grad_output)
                )
                self.hooks.append(hook)
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def _should_monitor_layer(self, layer_name: str) -> bool:
        """Determine if a layer should be monitored."""
        if not self.target_layers:
            # Monitor ResidualMemoryLayer and key transformer components
            monitor_patterns = [
                'residual_memory',
                'edit_weights',
                'attention',
                'ffn',
                'norm'
            ]
            return any(pattern in layer_name.lower() for pattern in monitor_patterns)
        
        return any(target in layer_name for target in self.target_layers)
    
    def _gradient_hook(self, layer_name: str, grad_output):
        """Hook function called during backward pass."""
        if grad_output is None or len(grad_output) == 0:
            return
        
        # Only log at specified frequency
        if self.step_count % self.log_frequency != 0:
            return
        
        # Calculate gradient norm for this layer
        grad_tensor = grad_output[0] if isinstance(grad_output, tuple) else grad_output
        if grad_tensor is not None:
            grad_norm = torch.norm(grad_tensor).item()
            
            # Store for snapshot creation
            if not hasattr(self, '_current_layer_norms'):
                self._current_layer_norms = {}
            self._current_layer_norms[layer_name] = grad_norm
    
    def log_step(
        self,
        loss_value: Optional[float] = None,
        learning_rate: Optional[float] = None
    ):
        """
        Log a training step with current gradient information.
        
        Args:
            loss_value: Current loss value
            learning_rate: Current learning rate
        """
        if self.step_count % self.log_frequency == 0:
            # Calculate overall gradient norm
            total_norm = 0.0
            layer_norms = getattr(self, '_current_layer_norms', {})
            
            for norm in layer_norms.values():
                total_norm += norm ** 2
            total_norm = total_norm ** 0.5
            
            # Create snapshot
            snapshot = GradientSnapshot(
                step=self.step_count,
                gradient_norm=total_norm,
                layer_norms=layer_norms.copy(),
                timestamp=datetime.now(),
                loss_value=loss_value,
                learning_rate=learning_rate
            )
            
            self.snapshots.append(snapshot)
            
            # Clear layer norms for next step
            self._current_layer_norms = {}
        
        self.step_count += 1
    
    def get_snapshots(self) -> List[GradientSnapshot]:
        """Get all captured gradient snapshots."""
        return self.snapshots.copy()

class GradientHealthMonitor:
    """
    Monitors gradient health and detects pathologies during training.
    
    Analyzes gradient snapshots to identify common training problems
    and provides actionable recommendations for resolution.
    """
    
    def __init__(
        self,
        vanishing_threshold: float = 1e-7,
        exploding_threshold: float = 100.0,
        noise_window: int = 10,
        stall_window: int = 5,
        oscillation_threshold: float = 0.5
    ):
        """
        Initialize gradient health monitor.
        
        Args:
            vanishing_threshold: Threshold below which gradients are considered vanishing
            exploding_threshold: Threshold above which gradients are considered exploding
            noise_window: Window size for noise analysis
            stall_window: Window size for stall detection
            oscillation_threshold: Threshold for detecting oscillations
        """
        self.vanishing_threshold = vanishing_threshold
        self.exploding_threshold = exploding_threshold
        self.noise_window = noise_window
        self.stall_window = stall_window
        self.oscillation_threshold = oscillation_threshold
        
        self.logger = logging.getLogger(f"{__name__}.GradientHealthMonitor")
    
    def analyze_gradient_health(
        self,
        snapshots: List[GradientSnapshot],
        analysis_window: Optional[int] = None
    ) -> GradientHealthReport:
        """
        Analyze gradient snapshots for pathologies.
        
        Args:
            snapshots: List of gradient snapshots to analyze
            analysis_window: Number of recent snapshots to analyze (None = all)
            
        Returns:
            Comprehensive health report with detected pathologies
        """
        if not snapshots:
            return GradientHealthReport(
                pathology=GradientPathology.HEALTHY,
                severity=0.0,
                description="No gradient data available for analysis",
                recommendations=["Ensure gradient logging is enabled"],
                snapshots=[],
                analysis_window=0,
                confidence=0.0
            )
        
        # Use specified window or all snapshots
        window_size = analysis_window or len(snapshots)
        recent_snapshots = snapshots[-window_size:]
        
        # Extract gradient norms for analysis
        gradient_norms = [s.gradient_norm for s in recent_snapshots]
        
        # Run pathology detection
        pathologies = []
        
        # Check for vanishing gradients
        vanishing_result = self._detect_vanishing_gradients(gradient_norms)
        if vanishing_result[0] != GradientPathology.HEALTHY:
            pathologies.append(vanishing_result)
        
        # Check for exploding gradients
        exploding_result = self._detect_exploding_gradients(gradient_norms)
        if exploding_result[0] != GradientPathology.HEALTHY:
            pathologies.append(exploding_result)
        
        # Check for noisy convergence
        noise_result = self._detect_noisy_convergence(gradient_norms)
        if noise_result[0] != GradientPathology.HEALTHY:
            pathologies.append(noise_result)
        
        # Check for learning stall
        stall_result = self._detect_learning_stall(gradient_norms)
        if stall_result[0] != GradientPathology.HEALTHY:
            pathologies.append(stall_result)
        
        # Check for oscillations
        oscillation_result = self._detect_oscillations(gradient_norms)
        if oscillation_result[0] != GradientPathology.HEALTHY:
            pathologies.append(oscillation_result)
        
        # Determine most severe pathology
        if pathologies:
            # Sort by severity (highest first)
            pathologies.sort(key=lambda x: x[1], reverse=True)
            primary_pathology, severity, description, recommendations = pathologies[0]
        else:
            primary_pathology = GradientPathology.HEALTHY
            severity = 0.0
            description = "Gradient health appears normal"
            recommendations = ["Continue monitoring gradient health"]
        
        # Calculate confidence based on sample size
        confidence = min(1.0, len(recent_snapshots) / 20.0)  # Full confidence at 20+ samples
        
        return GradientHealthReport(
            pathology=primary_pathology,
            severity=severity,
            description=description,
            recommendations=recommendations,
            snapshots=recent_snapshots,
            analysis_window=window_size,
            confidence=confidence
        )
    
    def _detect_vanishing_gradients(
        self,
        gradient_norms: List[float]
    ) -> Tuple[GradientPathology, float, str, List[str]]:
        """Detect vanishing gradient pathology."""
        if len(gradient_norms) < self.stall_window:
            return GradientPathology.HEALTHY, 0.0, "", []
        
        recent_norms = gradient_norms[-self.stall_window:]
        vanishing_count = sum(1 for norm in recent_norms if norm < self.vanishing_threshold)
        
        if vanishing_count >= self.stall_window * 0.8:  # 80% of recent steps
            severity = min(1.0, vanishing_count / self.stall_window)
            return (
                GradientPathology.VANISHING,
                severity,
                f"Vanishing gradients detected: {vanishing_count}/{self.stall_window} recent steps below threshold {self.vanishing_threshold}",
                [
                    "Increase learning rate",
                    "Use gradient clipping",
                    "Check for dead neurons",
                    "Consider different activation functions",
                    "Reduce model depth if possible"
                ]
            )
        
        return GradientPathology.HEALTHY, 0.0, "", []
    
    def _detect_exploding_gradients(
        self,
        gradient_norms: List[float]
    ) -> Tuple[GradientPathology, float, str, List[str]]:
        """Detect exploding gradient pathology."""
        max_norm = max(gradient_norms) if gradient_norms else 0.0
        exploding_count = sum(1 for norm in gradient_norms if norm > self.exploding_threshold)
        
        if exploding_count > 0:
            severity = min(1.0, max_norm / self.exploding_threshold)
            return (
                GradientPathology.EXPLODING,
                severity,
                f"Exploding gradients detected: {exploding_count} steps above threshold {self.exploding_threshold}, max norm: {max_norm:.2e}",
                [
                    "Apply gradient clipping",
                    "Reduce learning rate",
                    "Check for numerical instabilities",
                    "Use gradient normalization",
                    "Inspect loss function for discontinuities"
                ]
            )
        
        return GradientPathology.HEALTHY, 0.0, "", []
    
    def _detect_noisy_convergence(
        self,
        gradient_norms: List[float]
    ) -> Tuple[GradientPathology, float, str, List[str]]:
        """Detect noisy convergence pathology."""
        if len(gradient_norms) < self.noise_window:
            return GradientPathology.HEALTHY, 0.0, "", []
        
        recent_norms = gradient_norms[-self.noise_window:]
        
        # Calculate coefficient of variation (std/mean)
        mean_norm = np.mean(recent_norms)
        std_norm = np.std(recent_norms)
        
        if mean_norm > 0:
            cv = std_norm / mean_norm
            
            # High coefficient of variation indicates noise
            if cv > 1.0:  # Very noisy
                severity = min(1.0, cv / 2.0)
                return (
                    GradientPathology.NOISY,
                    severity,
                    f"Noisy convergence detected: coefficient of variation {cv:.2f} indicates unstable gradients",
                    [
                        "Reduce learning rate",
                        "Increase batch size",
                        "Use learning rate scheduling",
                        "Apply gradient smoothing",
                        "Check data quality and preprocessing"
                    ]
                )
        
        return GradientPathology.HEALTHY, 0.0, "", []
    
    def _detect_learning_stall(
        self,
        gradient_norms: List[float]
    ) -> Tuple[GradientPathology, float, str, List[str]]:
        """Detect learning stall pathology."""
        if len(gradient_norms) < self.stall_window:
            return GradientPathology.HEALTHY, 0.0, "", []
        
        recent_norms = gradient_norms[-self.stall_window:]
        
        # Check if gradients are consistently very small (but not vanishing)
        small_threshold = self.vanishing_threshold * 10  # 10x vanishing threshold
        small_count = sum(1 for norm in recent_norms if norm < small_threshold)
        
        # Also check for lack of progress (very similar values)
        norm_range = max(recent_norms) - min(recent_norms)
        mean_norm = np.mean(recent_norms)
        relative_range = norm_range / mean_norm if mean_norm > 0 else 0
        
        if small_count >= self.stall_window * 0.7 and relative_range < 0.1:
            severity = 1.0 - relative_range  # Higher severity for less variation
            return (
                GradientPathology.STALLED,
                severity,
                f"Learning stall detected: {small_count}/{self.stall_window} steps with small gradients and low variation",
                [
                    "Increase learning rate",
                    "Use learning rate warmup",
                    "Check for local minima",
                    "Try different optimization algorithm",
                    "Verify training data quality"
                ]
            )
        
        return GradientPathology.HEALTHY, 0.0, "", []
    
    def _detect_oscillations(
        self,
        gradient_norms: List[float]
    ) -> Tuple[GradientPathology, float, str, List[str]]:
        """Detect oscillating gradient pathology."""
        if len(gradient_norms) < 6:  # Need at least 6 points to detect oscillations
            return GradientPathology.HEALTHY, 0.0, "", []
        
        # Look for alternating increases and decreases
        changes = []
        for i in range(1, len(gradient_norms)):
            if gradient_norms[i] > gradient_norms[i-1]:
                changes.append(1)  # Increase
            elif gradient_norms[i] < gradient_norms[i-1]:
                changes.append(-1)  # Decrease
            else:
                changes.append(0)  # No change
        
        # Count sign changes
        sign_changes = 0
        for i in range(1, len(changes)):
            if changes[i] != 0 and changes[i-1] != 0 and changes[i] != changes[i-1]:
                sign_changes += 1
        
        # High frequency of sign changes indicates oscillation
        oscillation_ratio = sign_changes / len(changes) if changes else 0
        
        if oscillation_ratio > self.oscillation_threshold:
            severity = min(1.0, oscillation_ratio)
            return (
                GradientPathology.OSCILLATING,
                severity,
                f"Oscillating gradients detected: {oscillation_ratio:.2f} sign change ratio indicates unstable training",
                [
                    "Reduce learning rate significantly",
                    "Use momentum or adaptive optimization",
                    "Apply gradient smoothing",
                    "Check for learning rate scheduling issues",
                    "Verify loss function smoothness"
                ]
            )
        
        return GradientPathology.HEALTHY, 0.0, "", []
    
    def get_health_summary(self, report: GradientHealthReport) -> str:
        """Get a human-readable summary of gradient health."""
        if report.pathology == GradientPathology.HEALTHY:
            return f"✅ Gradient health: HEALTHY (confidence: {report.confidence:.2f})"
        
        severity_emoji = "🟡" if report.severity < 0.5 else "🔴"
        return f"{severity_emoji} Gradient health: {report.pathology.value.upper()} (severity: {report.severity:.2f}, confidence: {report.confidence:.2f})"
