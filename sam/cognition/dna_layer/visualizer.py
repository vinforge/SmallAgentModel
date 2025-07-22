"""
DNA Layer Routing Visualizer
=============================

Visualization tools for analyzing and understanding DNA layer routing behavior.
Provides real-time and post-hoc analysis of routing decisions and patterns.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class RoutingVisualizer:
    """
    Comprehensive visualization tools for DNA layer routing analysis.
    
    Provides various visualization methods to understand routing behavior,
    expert specialization, and compute efficiency patterns.
    """
    
    def __init__(self, expert_types: List[str]):
        self.expert_types = expert_types
        self.num_experts = len(expert_types)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_expert_usage_distribution(self, 
                                     expert_indices: torch.Tensor,
                                     title: str = "Expert Usage Distribution",
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the distribution of tokens across experts.
        
        Args:
            expert_indices: Tensor of expert assignments [batch_size, seq_len]
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Count expert usage
        expert_counts = []
        for expert_id in range(self.num_experts):
            count = (expert_indices == expert_id).sum().item()
            expert_counts.append(count)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        expert_labels = [f"Expert {i}\n({self.expert_types[i]})" for i in range(self.num_experts)]
        bars = ax.bar(expert_labels, expert_counts, alpha=0.8)
        
        # Add percentage labels on bars
        total_tokens = sum(expert_counts)
        for bar, count in zip(bars, expert_counts):
            height = bar.get_height()
            percentage = (count / total_tokens) * 100 if total_tokens > 0 else 0
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(expert_counts),
                   f'{percentage:.1f}%', ha='center', va='bottom')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Tokens', fontsize=12)
        ax.set_xlabel('Expert Modules', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_routing_heatmap(self,
                           routing_weights: torch.Tensor,
                           token_labels: Optional[List[str]] = None,
                           title: str = "Routing Weights Heatmap",
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot a heatmap of routing weights for tokens.
        
        Args:
            routing_weights: Routing weights [batch_size, seq_len, num_experts]
            token_labels: Optional token labels for y-axis
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Flatten and take a sample if too large
        batch_size, seq_len, num_experts = routing_weights.shape
        flat_weights = routing_weights.view(-1, num_experts).detach().cpu().numpy()
        
        # Sample if too many tokens
        max_tokens = 100
        if flat_weights.shape[0] > max_tokens:
            indices = np.random.choice(flat_weights.shape[0], max_tokens, replace=False)
            flat_weights = flat_weights[indices]
            if token_labels:
                token_labels = [token_labels[i] for i in indices]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        expert_labels = [f"Expert {i}\n({self.expert_types[i]})" for i in range(self.num_experts)]
        
        im = ax.imshow(flat_weights, cmap='viridis', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(num_experts))
        ax.set_xticklabels(expert_labels, rotation=45, ha='right')
        
        if token_labels and len(token_labels) == flat_weights.shape[0]:
            ax.set_yticks(range(len(token_labels)))
            ax.set_yticklabels(token_labels)
        else:
            ax.set_ylabel('Token Index')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Routing Weight', rotation=270, labelpad=20)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_efficiency_trends(self,
                             efficiency_history: List[float],
                             title: str = "Compute Efficiency Over Time",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot compute efficiency trends over time.
        
        Args:
            efficiency_history: List of efficiency values over time
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        steps = range(len(efficiency_history))
        ax.plot(steps, efficiency_history, linewidth=2, marker='o', markersize=4)
        
        # Add target line (30% efficiency)
        target_efficiency = 0.3
        ax.axhline(y=target_efficiency, color='red', linestyle='--', alpha=0.7, 
                  label=f'Target Efficiency ({target_efficiency:.1%})')
        
        # Add trend line
        if len(efficiency_history) > 1:
            z = np.polyfit(steps, efficiency_history, 1)
            p = np.poly1d(z)
            ax.plot(steps, p(steps), "r--", alpha=0.8, label='Trend')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Compute Efficiency (Identity Module Usage)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_token_type_specialization(self,
                                     token_type_routing: Dict[str, Dict[int, int]],
                                     title: str = "Token Type Specialization",
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot specialization patterns for different token types.
        
        Args:
            token_type_routing: Dictionary mapping token types to expert usage
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Prepare data for plotting
        token_types = list(token_type_routing.keys())
        expert_data = []
        
        for token_type in token_types:
            expert_counts = token_type_routing[token_type]
            total_count = sum(expert_counts.values())
            
            for expert_id in range(self.num_experts):
                count = expert_counts.get(expert_id, 0)
                percentage = (count / total_count) * 100 if total_count > 0 else 0
                expert_data.append({
                    'token_type': token_type,
                    'expert': f"Expert {expert_id}\n({self.expert_types[expert_id]})",
                    'percentage': percentage
                })
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame(expert_data)
        
        # Create grouped bar plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Pivot for grouped bar plot
        pivot_df = df.pivot(index='token_type', columns='expert', values='percentage')
        pivot_df.plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Token Types', fontsize=12)
        ax.set_ylabel('Percentage of Tokens', fontsize=12)
        ax.legend(title='Expert Modules', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_routing_dashboard(self,
                                           routing_data: Dict[str, Any],
                                           save_path: Optional[str] = None) -> go.Figure:
        """
        Create an interactive dashboard for routing analysis using Plotly.
        
        Args:
            routing_data: Dictionary containing routing analysis data
            save_path: Optional path to save the HTML file
            
        Returns:
            Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Expert Usage Distribution', 'Efficiency Trends', 
                          'Routing Entropy', 'Load Balance'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Expert usage distribution
        expert_usage = routing_data.get('expert_usage', {})
        expert_names = list(expert_usage.keys())
        expert_counts = list(expert_usage.values())
        
        fig.add_trace(
            go.Bar(x=expert_names, y=expert_counts, name="Expert Usage"),
            row=1, col=1
        )
        
        # Efficiency trends
        efficiency_history = routing_data.get('efficiency_history', [])
        if efficiency_history:
            fig.add_trace(
                go.Scatter(x=list(range(len(efficiency_history))), y=efficiency_history,
                          mode='lines+markers', name="Efficiency"),
                row=1, col=2
            )
        
        # Routing entropy
        entropy_history = routing_data.get('entropy_history', [])
        if entropy_history:
            fig.add_trace(
                go.Scatter(x=list(range(len(entropy_history))), y=entropy_history,
                          mode='lines+markers', name="Entropy"),
                row=2, col=1
            )
        
        # Load balance
        load_balance_history = routing_data.get('load_balance_history', [])
        if load_balance_history:
            fig.add_trace(
                go.Bar(x=list(range(len(load_balance_history))), y=load_balance_history,
                      name="Load Balance Loss"),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="DNA Layer Routing Analysis Dashboard",
            showlegend=False,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def generate_routing_report(self,
                              routing_analysis: Dict[str, Any],
                              save_path: str):
        """
        Generate a comprehensive routing analysis report with multiple visualizations.
        
        Args:
            routing_analysis: Complete routing analysis data
            save_path: Base path for saving visualizations (without extension)
        """
        # Create individual plots
        if 'expert_indices' in routing_analysis:
            expert_fig = self.plot_expert_usage_distribution(
                routing_analysis['expert_indices'],
                save_path=f"{save_path}_expert_usage.png"
            )
            plt.close(expert_fig)
        
        if 'routing_weights' in routing_analysis:
            heatmap_fig = self.plot_routing_heatmap(
                routing_analysis['routing_weights'],
                save_path=f"{save_path}_routing_heatmap.png"
            )
            plt.close(heatmap_fig)
        
        if 'efficiency_history' in routing_analysis:
            efficiency_fig = self.plot_efficiency_trends(
                routing_analysis['efficiency_history'],
                save_path=f"{save_path}_efficiency_trends.png"
            )
            plt.close(efficiency_fig)
        
        if 'token_type_routing' in routing_analysis:
            specialization_fig = self.plot_token_type_specialization(
                routing_analysis['token_type_routing'],
                save_path=f"{save_path}_specialization.png"
            )
            plt.close(specialization_fig)
        
        # Create interactive dashboard
        dashboard = self.create_interactive_routing_dashboard(
            routing_analysis,
            save_path=f"{save_path}_dashboard.html"
        )
        
        print(f"Routing analysis report generated:")
        print(f"  - Expert usage: {save_path}_expert_usage.png")
        print(f"  - Routing heatmap: {save_path}_routing_heatmap.png")
        print(f"  - Efficiency trends: {save_path}_efficiency_trends.png")
        print(f"  - Specialization: {save_path}_specialization.png")
        print(f"  - Interactive dashboard: {save_path}_dashboard.html")
