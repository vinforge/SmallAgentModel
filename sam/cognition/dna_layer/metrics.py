"""
DNA Layer Metrics and Analysis
===============================

Comprehensive metrics tracking and analysis for the DNA layer.
Monitors routing behavior, compute efficiency, and specialization patterns.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, Counter
import json
import time
from dataclasses import dataclass, asdict


@dataclass
class RoutingSnapshot:
    """Snapshot of routing decisions at a specific time."""
    timestamp: float
    expert_distribution: Dict[str, float]
    routing_entropy: float
    routing_confidence: float
    compute_savings: float
    load_balance_loss: float
    total_tokens: int


class DNAMetrics:
    """
    Comprehensive metrics tracking for DNA layer performance.
    
    Tracks routing behavior, compute efficiency, specialization patterns,
    and provides analysis tools for understanding DNA layer behavior.
    """
    
    def __init__(self, num_experts: int, expert_types: List[str]):
        self.num_experts = num_experts
        self.expert_types = expert_types
        
        # Core metrics
        self.routing_history: List[RoutingSnapshot] = []
        self.token_type_routing: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.expert_specialization: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Performance metrics
        self.compute_flops_saved = 0
        self.total_compute_flops = 0
        self.inference_times: List[float] = []
        
        # Routing analysis
        self.routing_patterns: Dict[str, Any] = {}
        self.efficiency_trends: List[float] = []
        
        # Token analysis
        self.token_complexity_routing: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        
    def record_routing_decision(self, 
                              expert_indices: torch.Tensor,
                              routing_weights: torch.Tensor,
                              routing_info: Dict,
                              token_types: Optional[List[str]] = None,
                              token_complexities: Optional[List[str]] = None):
        """Record a routing decision for analysis."""
        
        # Create routing snapshot
        expert_distribution = {}
        for expert_id in range(self.num_experts):
            expert_usage = (expert_indices == expert_id).sum().item()
            total_tokens = expert_indices.numel()
            expert_distribution[f'expert_{expert_id}_{self.expert_types[expert_id]}'] = expert_usage / total_tokens
        
        # Ensure scalar values for snapshot
        routing_entropy = routing_info.get('routing_entropy', 0.0)
        if hasattr(routing_entropy, 'item'):
            routing_entropy = routing_entropy.item()

        routing_confidence = routing_info.get('routing_confidence', 0.0)
        if hasattr(routing_confidence, 'item'):
            routing_confidence = routing_confidence.item()

        load_balance_loss = routing_info.get('load_balance_loss', 0.0)
        if hasattr(load_balance_loss, 'item'):
            load_balance_loss = load_balance_loss.item()

        snapshot = RoutingSnapshot(
            timestamp=time.time(),
            expert_distribution=expert_distribution,
            routing_entropy=routing_entropy,
            routing_confidence=routing_confidence,
            compute_savings=expert_distribution.get('expert_2_identity', 0.0),  # Assuming identity is expert 2
            load_balance_loss=load_balance_loss,
            total_tokens=expert_indices.numel()
        )
        
        self.routing_history.append(snapshot)
        
        # Record token type routing if provided
        if token_types is not None:
            # Handle both single list and batch of lists
            if isinstance(token_types[0], list):
                # Batch of token type lists
                flat_expert_indices = expert_indices.flatten()
                token_idx = 0
                for batch_token_types in token_types:
                    for token_type in batch_token_types:
                        if token_idx < len(flat_expert_indices):
                            expert_id = flat_expert_indices[token_idx].item()
                            self.token_type_routing[token_type][expert_id] += 1
                            self.expert_specialization[expert_id][token_type] += 1
                            token_idx += 1
            else:
                # Single list of token types
                for i, token_type in enumerate(token_types):
                    if i < expert_indices.numel():
                        expert_id = expert_indices.flatten()[i].item()
                        self.token_type_routing[token_type][expert_id] += 1
                        self.expert_specialization[expert_id][token_type] += 1
        
        # Record token complexity routing if provided
        if token_complexities is not None:
            for i, complexity in enumerate(token_complexities):
                if i < expert_indices.numel():
                    expert_id = expert_indices.flatten()[i].item()
                    self.token_complexity_routing[complexity][expert_id] += 1
    
    def record_compute_metrics(self, flops_saved: int, total_flops: int, inference_time: float):
        """Record compute efficiency metrics."""
        self.compute_flops_saved += flops_saved
        self.total_compute_flops += total_flops
        self.inference_times.append(inference_time)
        
        # Track efficiency trend
        if self.total_compute_flops > 0:
            efficiency = self.compute_flops_saved / self.total_compute_flops
            self.efficiency_trends.append(efficiency)
    
    def analyze_specialization(self) -> Dict[str, Any]:
        """Analyze expert specialization patterns."""
        specialization_analysis = {}
        
        for expert_id in range(self.num_experts):
            expert_name = self.expert_types[expert_id]
            expert_tokens = self.expert_specialization[expert_id]
            
            if not expert_tokens:
                continue
            
            total_tokens = sum(expert_tokens.values())
            token_distribution = {
                token_type: count / total_tokens 
                for token_type, count in expert_tokens.items()
            }
            
            # Find most common token types for this expert
            top_tokens = sorted(token_distribution.items(), key=lambda x: x[1], reverse=True)[:5]
            
            specialization_analysis[f'expert_{expert_id}_{expert_name}'] = {
                'total_tokens_processed': total_tokens,
                'token_distribution': token_distribution,
                'top_token_types': top_tokens,
                'specialization_score': self._compute_specialization_score(token_distribution)
            }
        
        return specialization_analysis
    
    def _compute_specialization_score(self, token_distribution: Dict[str, float]) -> float:
        """Compute specialization score (higher = more specialized)."""
        if not token_distribution:
            return 0.0
        
        # Use entropy to measure specialization (lower entropy = higher specialization)
        entropy = -sum(p * np.log(p + 1e-8) for p in token_distribution.values())
        max_entropy = np.log(len(token_distribution))
        
        # Normalize to 0-1 scale (1 = highly specialized, 0 = uniform)
        specialization_score = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
        return specialization_score
    
    def analyze_efficiency_trends(self) -> Dict[str, Any]:
        """Analyze compute efficiency trends over time."""
        if not self.routing_history:
            return {}
        
        # Extract efficiency metrics over time
        timestamps = [snapshot.timestamp for snapshot in self.routing_history]
        compute_savings = [snapshot.compute_savings for snapshot in self.routing_history]
        routing_entropies = [snapshot.routing_entropy for snapshot in self.routing_history]
        
        # Compute trends
        efficiency_trend = np.polyfit(range(len(compute_savings)), compute_savings, 1)[0] if len(compute_savings) > 1 else 0.0
        entropy_trend = np.polyfit(range(len(routing_entropies)), routing_entropies, 1)[0] if len(routing_entropies) > 1 else 0.0
        
        return {
            'current_efficiency': compute_savings[-1] if compute_savings else 0.0,
            'efficiency_trend': efficiency_trend,
            'average_efficiency': np.mean(compute_savings) if compute_savings else 0.0,
            'efficiency_stability': np.std(compute_savings) if compute_savings else 0.0,
            'routing_entropy_trend': entropy_trend,
            'average_routing_entropy': np.mean(routing_entropies) if routing_entropies else 0.0,
            'total_snapshots': len(self.routing_history)
        }
    
    def analyze_token_type_preferences(self) -> Dict[str, Any]:
        """Analyze which experts prefer which token types."""
        preferences = {}
        
        for token_type, expert_counts in self.token_type_routing.items():
            total_tokens = sum(expert_counts.values())
            if total_tokens == 0:
                continue
            
            expert_preferences = {}
            for expert_id in range(self.num_experts):
                count = expert_counts.get(expert_id, 0)
                preference = count / total_tokens
                expert_preferences[f'expert_{expert_id}_{self.expert_types[expert_id]}'] = preference
            
            # Find preferred expert for this token type
            preferred_expert = max(expert_preferences.items(), key=lambda x: x[1])
            
            preferences[token_type] = {
                'total_occurrences': total_tokens,
                'expert_preferences': expert_preferences,
                'preferred_expert': preferred_expert[0],
                'preference_strength': preferred_expert[1]
            }
        
        return preferences
    
    def get_efficiency_summary(self) -> Dict[str, Any]:
        """Get comprehensive efficiency summary."""
        if not self.routing_history:
            return {'status': 'no_data'}
        
        latest_snapshot = self.routing_history[-1]
        
        # Compute overall statistics
        all_compute_savings = [s.compute_savings for s in self.routing_history]
        all_entropies = [s.routing_entropy.item() if hasattr(s.routing_entropy, 'item') else s.routing_entropy for s in self.routing_history]
        
        return {
            'current_compute_savings': latest_snapshot.compute_savings,
            'average_compute_savings': np.mean(all_compute_savings),
            'max_compute_savings': np.max(all_compute_savings),
            'compute_savings_trend': np.polyfit(range(len(all_compute_savings)), all_compute_savings, 1)[0] if len(all_compute_savings) > 1 else 0.0,
            'routing_entropy': latest_snapshot.routing_entropy,
            'average_routing_entropy': np.mean(all_entropies),
            'routing_stability': 1.0 - np.std(all_compute_savings),  # Higher = more stable
            'total_tokens_processed': sum(s.total_tokens for s in self.routing_history),
            'efficiency_target_met': latest_snapshot.compute_savings >= 0.3,  # 30% target
            'routing_health': 'healthy' if latest_snapshot.routing_entropy > 0.5 else 'concerning'
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        return {
            'timestamp': time.time(),
            'summary': self.get_efficiency_summary(),
            'specialization_analysis': self.analyze_specialization(),
            'efficiency_trends': self.analyze_efficiency_trends(),
            'token_type_preferences': self.analyze_token_type_preferences(),
            'routing_snapshots': len(self.routing_history),
            'expert_configuration': {
                'num_experts': self.num_experts,
                'expert_types': self.expert_types
            }
        }
    
    def save_report(self, filepath: str):
        """Save analysis report to file."""
        report = self.generate_report()
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        report = convert_numpy(report)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.routing_history.clear()
        self.token_type_routing.clear()
        self.expert_specialization.clear()
        self.compute_flops_saved = 0
        self.total_compute_flops = 0
        self.inference_times.clear()
        self.routing_patterns.clear()
        self.efficiency_trends.clear()
        self.token_complexity_routing.clear()
    
    def get_real_time_stats(self) -> Dict[str, Any]:
        """Get real-time statistics for monitoring."""
        if not self.routing_history:
            return {'status': 'no_data'}
        
        latest = self.routing_history[-1]
        
        return {
            'current_efficiency': latest.compute_savings,
            'routing_entropy': latest.routing_entropy,
            'routing_confidence': latest.routing_confidence,
            'load_balance_loss': latest.load_balance_loss,
            'expert_distribution': latest.expert_distribution,
            'tokens_processed': latest.total_tokens,
            'timestamp': latest.timestamp
        }
