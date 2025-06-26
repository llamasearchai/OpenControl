"""
Visualization Tools for OpenControl Evaluation Results.

This module provides visualization capabilities for evaluation metrics,
benchmarks, and performance analysis.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class ResultsVisualizer:
    """
    Comprehensive visualization tools for evaluation results.
    
    This class provides various plotting and visualization capabilities
    for analyzing world model and control system performance.
    """
    
    def __init__(self, style: str = 'seaborn'):
        """Initialize visualizer with plotting style."""
        self.style = style
        if style == 'seaborn':
            sns.set_style("whitegrid")
        
    def plot_prediction_metrics(
        self, 
        metrics: Dict[str, Any], 
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot prediction accuracy metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('World Model Prediction Metrics', fontsize=16)
        
        # Extract metrics by modality
        modalities = ['video', 'audio', 'actions', 'proprioception']
        
        # Plot MSE by horizon
        ax = axes[0, 0]
        for modality in modalities:
            mse_values = []
            horizons = []
            for key, value in metrics.items():
                if f'{modality}_mse_h' in key:
                    horizon = int(key.split('_h')[1])
                    horizons.append(horizon)
                    mse_values.append(value)
            
            if horizons:
                sorted_data = sorted(zip(horizons, mse_values))
                horizons, mse_values = zip(*sorted_data)
                ax.plot(horizons, mse_values, marker='o', label=modality)
        
        ax.set_xlabel('Prediction Horizon')
        ax.set_ylabel('MSE')
        ax.set_title('MSE vs Prediction Horizon')
        ax.legend()
        ax.grid(True)
        
        # Plot modality comparison
        ax = axes[0, 1]
        modality_scores = []
        modality_names = []
        
        for modality in modalities:
            mse_values = [v for k, v in metrics.items() if f'{modality}_mse' in k]
            if mse_values:
                avg_mse = np.mean(mse_values)
                modality_scores.append(1.0 / (1.0 + avg_mse))  # Convert to score
                modality_names.append(modality)
        
        if modality_scores:
            bars = ax.bar(modality_names, modality_scores)
            ax.set_ylabel('Prediction Score')
            ax.set_title('Prediction Accuracy by Modality')
            ax.grid(True, axis='y')
            
            # Add value labels on bars
            for bar, score in zip(bars, modality_scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{score:.3f}', ha='center', va='bottom')
        
        # Plot temporal consistency
        ax = axes[1, 0]
        consistency_scores = []
        consistency_names = []
        
        for key, value in metrics.items():
            if 'consistency_score' in key:
                modality = key.replace('_consistency_score', '')
                consistency_scores.append(value)
                consistency_names.append(modality)
        
        if consistency_scores:
            ax.bar(consistency_names, consistency_scores, color='orange')
            ax.set_ylabel('Consistency Score')
            ax.set_title('Temporal Consistency')
            ax.grid(True, axis='y')
        
        # Plot efficiency metrics
        ax = axes[1, 1]
        if 'efficiency' in metrics:
            efficiency = metrics['efficiency']
            labels = []
            values = []
            
            if 'throughput_fps' in efficiency:
                labels.append('Throughput\n(FPS)')
                values.append(efficiency['throughput_fps'])
            
            if 'avg_inference_time' in efficiency:
                labels.append('Avg Time\n(ms)')
                values.append(efficiency['avg_inference_time'] * 1000)
            
            if labels:
                ax.bar(labels, values, color='green')
                ax.set_title('Efficiency Metrics')
                ax.grid(True, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_control_performance(
        self, 
        metrics: Dict[str, Any], 
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot control system performance metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Control System Performance', fontsize=16)
        
        # Plot tracking performance
        ax = axes[0, 0]
        if 'tracking_performance' in metrics:
            tracking = metrics['tracking_performance']
            trajectories = []
            errors = []
            
            for key, value in tracking.items():
                if '_avg_error' in key:
                    traj_name = key.replace('_avg_error', '')
                    trajectories.append(traj_name)
                    errors.append(value)
            
            if trajectories:
                bars = ax.bar(trajectories, errors, color='red', alpha=0.7)
                ax.set_ylabel('Average Error')
                ax.set_title('Tracking Performance')
                ax.grid(True, axis='y')
                
                # Add value labels
                for bar, error in zip(bars, errors):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{error:.3f}', ha='center', va='bottom')
        
        # Plot solve times
        ax = axes[0, 1]
        if 'control_efficiency' in metrics:
            efficiency = metrics['control_efficiency']
            
            if 'avg_solve_time' in efficiency:
                solve_times = [
                    efficiency.get('avg_solve_time', 0) * 1000,
                    efficiency.get('max_solve_time', 0) * 1000
                ]
                labels = ['Average', 'Maximum']
                
                bars = ax.bar(labels, solve_times, color='blue', alpha=0.7)
                ax.set_ylabel('Solve Time (ms)')
                ax.set_title('Control Solve Times')
                ax.grid(True, axis='y')
                
                # Add real-time threshold line
                real_time_limit = 1000 / 10  # 10 Hz control frequency
                ax.axhline(y=real_time_limit, color='red', linestyle='--', 
                          label=f'Real-time limit ({real_time_limit:.0f}ms)')
                ax.legend()
        
        # Plot safety metrics
        ax = axes[1, 0]
        if 'safety_compliance' in metrics:
            safety = metrics['safety_compliance']
            
            safety_metrics = []
            safety_values = []
            
            if 'action_bound_violation_rate' in safety:
                safety_metrics.append('Bound\nViolations')
                safety_values.append(safety['action_bound_violation_rate'] * 100)
            
            if 'emergency_response_time' in safety:
                safety_metrics.append('Emergency\nResponse (ms)')
                safety_values.append(safety['emergency_response_time'] * 1000)
            
            if safety_metrics:
                ax.bar(safety_metrics, safety_values, color='orange', alpha=0.7)
                ax.set_title('Safety Metrics')
                ax.grid(True, axis='y')
        
        # Plot robustness
        ax = axes[1, 1]
        if 'control_robustness' in metrics:
            robustness = metrics['control_robustness']
            
            noise_levels = []
            degradations = []
            
            for key, value in robustness.items():
                if 'noise_robustness_' in key:
                    noise_level = float(key.split('_')[-1])
                    noise_levels.append(noise_level)
                    degradations.append(value)
            
            if noise_levels:
                ax.plot(noise_levels, degradations, marker='o', color='purple')
                ax.set_xlabel('Noise Level')
                ax.set_ylabel('Performance Degradation')
                ax.set_title('Robustness to Noise')
                ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_dashboard(
        self, 
        world_model_metrics: Dict[str, Any],
        control_metrics: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Create a comprehensive dashboard."""
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('OpenControl System Dashboard', fontsize=20, y=0.95)
        
        # World model metrics (top row)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_metric_summary(ax1, world_model_metrics, 'World Model Performance')
        
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_metric_summary(ax2, control_metrics, 'Control System Performance')
        
        # Detailed metrics (middle and bottom rows)
        if 'prediction_accuracy' in world_model_metrics:
            ax3 = fig.add_subplot(gs[1, :2])
            self._plot_prediction_accuracy(ax3, world_model_metrics['prediction_accuracy'])
        
        if 'tracking_performance' in control_metrics:
            ax4 = fig.add_subplot(gs[1, 2:])
            self._plot_tracking_performance(ax4, control_metrics['tracking_performance'])
        
        if 'efficiency' in world_model_metrics:
            ax5 = fig.add_subplot(gs[2, :2])
            self._plot_efficiency_metrics(ax5, world_model_metrics['efficiency'])
        
        if 'control_efficiency' in control_metrics:
            ax6 = fig.add_subplot(gs[2, 2:])
            self._plot_control_efficiency(ax6, control_metrics['control_efficiency'])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_metric_summary(self, ax, metrics: Dict[str, Any], title: str):
        """Plot summary metrics."""
        # Extract key metrics for summary
        key_metrics = []
        values = []
        
        if 'overall_score' in metrics:
            key_metrics.append('Overall Score')
            values.append(metrics['overall_score'])
        
        # Add other relevant metrics based on type
        if 'prediction_accuracy' in title.lower():
            if 'efficiency' in metrics and 'throughput_fps' in metrics['efficiency']:
                key_metrics.append('Throughput (FPS)')
                values.append(metrics['efficiency']['throughput_fps'] / 100)  # Normalize
        
        if 'control' in title.lower():
            if 'control_efficiency' in metrics and 'real_time_factor' in metrics['control_efficiency']:
                rtf = metrics['control_efficiency']['real_time_factor']
                key_metrics.append('Real-time Factor')
                values.append(max(0, 1 - rtf))  # Convert to score
        
        if key_metrics:
            bars = ax.bar(key_metrics, values, alpha=0.7)
            ax.set_ylim(0, 1)
            ax.set_ylabel('Score')
            ax.set_title(title)
            ax.grid(True, axis='y')
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{value:.3f}', ha='center', va='bottom')
    
    def _plot_prediction_accuracy(self, ax, metrics: Dict[str, Any]):
        """Plot prediction accuracy details."""
        modalities = ['video', 'audio', 'actions', 'proprioception']
        scores = []
        
        for modality in modalities:
            mse_values = [v for k, v in metrics.items() if f'{modality}_mse' in k]
            if mse_values:
                avg_mse = np.mean(mse_values)
                scores.append(1.0 / (1.0 + avg_mse))
            else:
                scores.append(0)
        
        bars = ax.bar(modalities, scores, color='skyblue', alpha=0.7)
        ax.set_ylabel('Prediction Score')
        ax.set_title('Prediction Accuracy by Modality')
        ax.grid(True, axis='y')
        
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.2f}', ha='center', va='bottom')
    
    def _plot_tracking_performance(self, ax, metrics: Dict[str, Any]):
        """Plot tracking performance details."""
        trajectories = []
        errors = []
        
        for key, value in metrics.items():
            if '_avg_error' in key:
                traj_name = key.replace('_avg_error', '')
                trajectories.append(traj_name)
                errors.append(value)
        
        if trajectories:
            bars = ax.bar(trajectories, errors, color='lightcoral', alpha=0.7)
            ax.set_ylabel('Average Error')
            ax.set_title('Tracking Performance')
            ax.grid(True, axis='y')
    
    def _plot_efficiency_metrics(self, ax, metrics: Dict[str, Any]):
        """Plot efficiency metrics."""
        labels = []
        values = []
        
        if 'throughput_fps' in metrics:
            labels.append('Throughput\n(FPS)')
            values.append(metrics['throughput_fps'])
        
        if 'avg_inference_time' in metrics:
            labels.append('Inference\n(ms)')
            values.append(metrics['avg_inference_time'] * 1000)
        
        if labels:
            ax.bar(labels, values, color='lightgreen', alpha=0.7)
            ax.set_title('Efficiency Metrics')
            ax.grid(True, axis='y')
    
    def _plot_control_efficiency(self, ax, metrics: Dict[str, Any]):
        """Plot control efficiency metrics."""
        labels = []
        values = []
        
        if 'avg_solve_time' in metrics:
            labels.append('Avg Solve\n(ms)')
            values.append(metrics['avg_solve_time'] * 1000)
        
        if 'real_time_factor' in metrics:
            labels.append('Real-time\nFactor')
            values.append(metrics['real_time_factor'])
        
        if labels:
            bars = ax.bar(labels, values, color='lightyellow', alpha=0.7)
            ax.set_title('Control Efficiency')
            ax.grid(True, axis='y')
            
            # Add real-time threshold for solve time
            if 'avg_solve_time' in metrics:
                real_time_limit = 100  # 100ms for 10Hz
                ax.axhline(y=real_time_limit, color='red', linestyle='--', alpha=0.5)
    
    def save_interactive_dashboard(
        self, 
        metrics: Dict[str, Any], 
        save_path: str
    ):
        """Create and save interactive dashboard using Plotly."""
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Install with: pip install plotly")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Prediction Metrics', 'Control Performance', 
                          'Efficiency', 'Safety & Robustness'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add traces (simplified for brevity)
        # This would include interactive versions of the plots above
        
        fig.update_layout(
            title_text="OpenControl Interactive Dashboard",
            showlegend=True,
            height=800
        )
        
        fig.write_html(save_path)
        print(f"Interactive dashboard saved to {save_path}")


# Convenience functions
def plot_evaluation_results(results: Dict[str, Any], output_dir: str = "plots"):
    """Plot all evaluation results."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    visualizer = ResultsVisualizer()
    
    # Plot world model metrics
    if 'world_model' in results:
        fig = visualizer.plot_prediction_metrics(
            results['world_model'],
            save_path=str(output_path / "world_model_metrics.png")
        )
        plt.close(fig)
    
    # Plot control metrics
    if 'control' in results:
        fig = visualizer.plot_control_performance(
            results['control'],
            save_path=str(output_path / "control_metrics.png")
        )
        plt.close(fig)
    
    # Create dashboard
    if 'world_model' in results and 'control' in results:
        fig = visualizer.create_dashboard(
            results['world_model'],
            results['control'],
            save_path=str(output_path / "dashboard.png")
        )
        plt.close(fig)
    
    print(f"Plots saved to {output_dir}/")


def create_report(results: Dict[str, Any], output_file: str = "evaluation_report.html"):
    """Create HTML evaluation report."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>OpenControl Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ text-align: center; color: #333; }}
            .section {{ margin: 20px 0; }}
            .metric {{ background: #f5f5f5; padding: 10px; margin: 5px 0; }}
            .score {{ font-weight: bold; color: #007acc; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>OpenControl Evaluation Report</h1>
            <p>Generated on {results.get('timestamp', 'Unknown')}</p>
        </div>
        
        <div class="section">
            <h2>Summary</h2>
            <div class="metric">
                <strong>Overall Score:</strong> 
                <span class="score">{results.get('overall_score', 'N/A')}</span>
            </div>
        </div>
        
        <!-- Add more sections as needed -->
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Report saved to {output_file}") 