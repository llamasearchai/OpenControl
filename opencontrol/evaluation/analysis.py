"""
Performance Analysis Tools for OpenControl.

This module provides advanced analysis capabilities for understanding
system performance, identifying bottlenecks, and optimization opportunities.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import logging
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


class PerformanceAnalyzer:
    """
    Advanced performance analysis for OpenControl systems.
    
    This class provides statistical analysis, trend detection,
    performance profiling, and optimization recommendations.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def analyze_prediction_performance(
        self, 
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze world model prediction performance."""
        analysis = {
            'summary': {},
            'modality_analysis': {},
            'temporal_analysis': {},
            'recommendations': []
        }
        
        # Overall performance summary
        if 'prediction_accuracy' in metrics:
            pred_metrics = metrics['prediction_accuracy']
            
            # Collect MSE values by modality
            modality_mse = {}
            for key, value in pred_metrics.items():
                if '_mse_h1' in key:  # Focus on 1-step prediction
                    modality = key.replace('_mse_h1', '')
                    modality_mse[modality] = value
            
            if modality_mse:
                analysis['summary'] = {
                    'best_modality': min(modality_mse.items(), key=lambda x: x[1]),
                    'worst_modality': max(modality_mse.items(), key=lambda x: x[1]),
                    'avg_mse': np.mean(list(modality_mse.values())),
                    'mse_std': np.std(list(modality_mse.values()))
                }
                
                # Modality-specific analysis
                for modality, mse in modality_mse.items():
                    score = 1.0 / (1.0 + mse)
                    performance_level = self._classify_performance(score)
                    
                    analysis['modality_analysis'][modality] = {
                        'mse': mse,
                        'score': score,
                        'performance_level': performance_level,
                        'relative_performance': mse / analysis['summary']['avg_mse']
                    }
        
        # Temporal analysis (multi-horizon)
        horizon_performance = {}
        for key, value in metrics.get('prediction_accuracy', {}).items():
            if '_mse_h' in key:
                parts = key.split('_mse_h')
                if len(parts) == 2:
                    modality = parts[0]
                    horizon = int(parts[1])
                    
                    if modality not in horizon_performance:
                        horizon_performance[modality] = {}
                    horizon_performance[modality][horizon] = value
        
        # Analyze degradation over time
        for modality, horizons in horizon_performance.items():
            if len(horizons) > 1:
                sorted_horizons = sorted(horizons.items())
                horizon_vals, mse_vals = zip(*sorted_horizons)
                
                # Compute degradation rate
                if len(mse_vals) > 1:
                    degradation_rate = (mse_vals[-1] - mse_vals[0]) / (horizon_vals[-1] - horizon_vals[0])
                    
                    analysis['temporal_analysis'][modality] = {
                        'degradation_rate': degradation_rate,
                        'stability': 1.0 / (1.0 + abs(degradation_rate)),
                        'long_term_viable': degradation_rate < 0.1
                    }
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_prediction_recommendations(analysis)
        
        return analysis
    
    def analyze_control_performance(
        self, 
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze control system performance."""
        analysis = {
            'efficiency_analysis': {},
            'safety_analysis': {},
            'robustness_analysis': {},
            'recommendations': []
        }
        
        # Efficiency analysis
        if 'control_efficiency' in metrics:
            efficiency = metrics['control_efficiency']
            
            real_time_factor = efficiency.get('real_time_factor', 1.0)
            avg_solve_time = efficiency.get('avg_solve_time', 0.1)
            
            analysis['efficiency_analysis'] = {
                'real_time_performance': real_time_factor < 1.0,
                'real_time_margin': max(0, 1.0 - real_time_factor),
                'solve_time_category': self._classify_solve_time(avg_solve_time),
                'efficiency_score': max(0, 1.0 - real_time_factor)
            }
        
        # Safety analysis
        if 'safety_compliance' in metrics:
            safety = metrics['safety_compliance']
            
            violation_rate = safety.get('action_bound_violation_rate', 0.0)
            response_time = safety.get('emergency_response_time', 0.1)
            
            analysis['safety_analysis'] = {
                'safety_level': self._classify_safety_level(violation_rate),
                'violation_rate': violation_rate,
                'response_time_acceptable': response_time < 0.05,  # 50ms threshold
                'safety_score': max(0, 1.0 - violation_rate)
            }
        
        # Robustness analysis
        if 'control_robustness' in metrics:
            robustness = metrics['control_robustness']
            
            noise_degradations = []
            for key, value in robustness.items():
                if 'noise_robustness_' in key:
                    noise_degradations.append(value)
            
            if noise_degradations:
                avg_degradation = np.mean(noise_degradations)
                max_degradation = np.max(noise_degradations)
                
                analysis['robustness_analysis'] = {
                    'avg_degradation': avg_degradation,
                    'max_degradation': max_degradation,
                    'robustness_level': self._classify_robustness(avg_degradation),
                    'robustness_score': max(0, 1.0 - avg_degradation)
                }
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_control_recommendations(analysis)
        
        return analysis
    
    def compare_configurations(
        self, 
        results_list: List[Dict[str, Any]], 
        config_names: List[str]
    ) -> Dict[str, Any]:
        """Compare performance across different configurations."""
        comparison = {
            'summary': {},
            'detailed_comparison': {},
            'statistical_analysis': {},
            'recommendations': []
        }
        
        if len(results_list) != len(config_names):
            raise ValueError("Number of results must match number of config names")
        
        # Extract key metrics for comparison
        metrics_to_compare = [
            'overall_score',
            'prediction_accuracy.video_mse_h1',
            'control_efficiency.avg_solve_time',
            'safety_compliance.action_bound_violation_rate'
        ]
        
        comparison_data = {}
        for metric in metrics_to_compare:
            comparison_data[metric] = []
            
            for results in results_list:
                value = self._extract_nested_metric(results, metric)
                comparison_data[metric].append(value)
        
        # Statistical analysis
        for metric, values in comparison_data.items():
            valid_values = [v for v in values if v is not None]
            
            if len(valid_values) > 1:
                comparison['statistical_analysis'][metric] = {
                    'mean': np.mean(valid_values),
                    'std': np.std(valid_values),
                    'min': np.min(valid_values),
                    'max': np.max(valid_values),
                    'range': np.max(valid_values) - np.min(valid_values),
                    'coefficient_of_variation': np.std(valid_values) / np.mean(valid_values)
                }
        
        # Ranking
        overall_scores = comparison_data.get('overall_score', [])
        if overall_scores and all(s is not None for s in overall_scores):
            ranked_indices = np.argsort(overall_scores)[::-1]  # Descending order
            
            comparison['summary']['ranking'] = [
                {
                    'rank': i + 1,
                    'config': config_names[idx],
                    'score': overall_scores[idx]
                }
                for i, idx in enumerate(ranked_indices)
            ]
        
        # Best configuration analysis
        if 'ranking' in comparison['summary'] and comparison['summary']['ranking']:
            best_config = comparison['summary']['ranking'][0]['config']
            comparison['summary']['best_configuration'] = best_config
            comparison['recommendations'].append(
                f"Use configuration '{best_config}' for best overall performance"
            )
        
        return comparison
    
    def identify_bottlenecks(
        self, 
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Identify performance bottlenecks in the system."""
        bottlenecks = {
            'computational': [],
            'algorithmic': [],
            'system': [],
            'recommendations': []
        }
        
        # Computational bottlenecks
        if 'efficiency' in metrics:
            efficiency = metrics['efficiency']
            
            if efficiency.get('avg_inference_time', 0) > 0.1:  # > 100ms
                bottlenecks['computational'].append({
                    'type': 'slow_inference',
                    'severity': 'high',
                    'description': 'Model inference time exceeds real-time requirements',
                    'metric_value': efficiency.get('avg_inference_time', 0)
                })
            
            if efficiency.get('throughput_fps', 0) < 10:  # < 10 FPS
                bottlenecks['computational'].append({
                    'type': 'low_throughput',
                    'severity': 'medium',
                    'description': 'System throughput is below optimal levels',
                    'metric_value': efficiency.get('throughput_fps', 0)
                })
        
        # Control system bottlenecks
        if 'control_efficiency' in metrics:
            control_eff = metrics['control_efficiency']
            
            if control_eff.get('real_time_factor', 1.0) > 0.8:
                bottlenecks['computational'].append({
                    'type': 'control_latency',
                    'severity': 'high',
                    'description': 'Control system approaching real-time limits',
                    'metric_value': control_eff.get('real_time_factor', 1.0)
                })
        
        # Algorithmic bottlenecks
        if 'prediction_accuracy' in metrics:
            pred_acc = metrics['prediction_accuracy']
            
            # Check for poor prediction quality
            poor_modalities = []
            for key, value in pred_acc.items():
                if '_mse_h1' in key and value > 0.1:  # High MSE
                    modality = key.replace('_mse_h1', '')
                    poor_modalities.append((modality, value))
            
            if poor_modalities:
                bottlenecks['algorithmic'].append({
                    'type': 'poor_prediction_quality',
                    'severity': 'medium',
                    'description': f'Poor prediction quality for: {[m[0] for m in poor_modalities]}',
                    'affected_modalities': poor_modalities
                })
        
        # Generate specific recommendations
        bottlenecks['recommendations'] = self._generate_bottleneck_recommendations(bottlenecks)
        
        return bottlenecks
    
    def _classify_performance(self, score: float) -> str:
        """Classify performance level based on score."""
        if score >= 0.9:
            return 'excellent'
        elif score >= 0.7:
            return 'good'
        elif score >= 0.5:
            return 'fair'
        else:
            return 'poor'
    
    def _classify_solve_time(self, solve_time: float) -> str:
        """Classify control solve time."""
        if solve_time < 0.01:  # < 10ms
            return 'very_fast'
        elif solve_time < 0.05:  # < 50ms
            return 'fast'
        elif solve_time < 0.1:  # < 100ms
            return 'acceptable'
        else:
            return 'slow'
    
    def _classify_safety_level(self, violation_rate: float) -> str:
        """Classify safety level based on violation rate."""
        if violation_rate == 0.0:
            return 'excellent'
        elif violation_rate < 0.01:  # < 1%
            return 'good'
        elif violation_rate < 0.05:  # < 5%
            return 'acceptable'
        else:
            return 'poor'
    
    def _classify_robustness(self, degradation: float) -> str:
        """Classify robustness level."""
        if degradation < 0.1:
            return 'excellent'
        elif degradation < 0.3:
            return 'good'
        elif degradation < 0.5:
            return 'fair'
        else:
            return 'poor'
    
    def _extract_nested_metric(self, data: Dict[str, Any], metric_path: str) -> Optional[float]:
        """Extract nested metric from results dictionary."""
        keys = metric_path.split('.')
        current = data
        
        try:
            for key in keys:
                current = current[key]
            return float(current) if current is not None else None
        except (KeyError, TypeError, ValueError):
            return None
    
    def _generate_prediction_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving prediction performance."""
        recommendations = []
        
        # Check modality performance
        if 'modality_analysis' in analysis:
            poor_modalities = [
                modality for modality, data in analysis['modality_analysis'].items()
                if data.get('performance_level') == 'poor'
            ]
            
            if poor_modalities:
                recommendations.append(
                    f"Improve {', '.join(poor_modalities)} prediction quality through "
                    f"architecture changes or additional training data"
                )
        
        # Check temporal stability
        if 'temporal_analysis' in analysis:
            unstable_modalities = [
                modality for modality, data in analysis['temporal_analysis'].items()
                if not data.get('long_term_viable', True)
            ]
            
            if unstable_modalities:
                recommendations.append(
                    f"Address temporal instability in {', '.join(unstable_modalities)} "
                    f"through regularization or sequence modeling improvements"
                )
        
        return recommendations
    
    def _generate_control_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving control performance."""
        recommendations = []
        
        # Efficiency recommendations
        if 'efficiency_analysis' in analysis:
            eff = analysis['efficiency_analysis']
            
            if not eff.get('real_time_performance', True):
                recommendations.append(
                    "Optimize control system for real-time performance: "
                    "reduce planning horizon, use faster algorithms, or improve hardware"
                )
            
            if eff.get('solve_time_category') == 'slow':
                recommendations.append(
                    "Reduce control solve time through algorithm optimization "
                    "or hardware acceleration"
                )
        
        # Safety recommendations
        if 'safety_analysis' in analysis:
            safety = analysis['safety_analysis']
            
            if safety.get('safety_level') in ['poor', 'acceptable']:
                recommendations.append(
                    "Improve safety system: tighten constraints, add safety margins, "
                    "or implement additional safety checks"
                )
        
        # Robustness recommendations
        if 'robustness_analysis' in analysis:
            robustness = analysis['robustness_analysis']
            
            if robustness.get('robustness_level') in ['poor', 'fair']:
                recommendations.append(
                    "Enhance system robustness through noise injection during training, "
                    "adaptive control, or sensor fusion improvements"
                )
        
        return recommendations
    
    def _generate_bottleneck_recommendations(self, bottlenecks: Dict[str, Any]) -> List[str]:
        """Generate recommendations for addressing bottlenecks."""
        recommendations = []
        
        # Computational bottlenecks
        for bottleneck in bottlenecks.get('computational', []):
            if bottleneck['type'] == 'slow_inference':
                recommendations.append(
                    "Optimize model inference: use model quantization, pruning, "
                    "or specialized hardware acceleration"
                )
            elif bottleneck['type'] == 'low_throughput':
                recommendations.append(
                    "Increase system throughput: implement batching, parallel processing, "
                    "or pipeline optimization"
                )
            elif bottleneck['type'] == 'control_latency':
                recommendations.append(
                    "Reduce control latency: optimize planning algorithms, "
                    "reduce horizon, or use approximate methods"
                )
        
        # Algorithmic bottlenecks
        for bottleneck in bottlenecks.get('algorithmic', []):
            if bottleneck['type'] == 'poor_prediction_quality':
                recommendations.append(
                    "Improve prediction quality: increase model capacity, "
                    "add training data, or adjust loss functions"
                )
        
        return recommendations
    
    def generate_performance_report(
        self, 
        metrics: Dict[str, Any]
    ) -> str:
        """Generate a comprehensive performance analysis report."""
        report_lines = []
        
        report_lines.append("=" * 60)
        report_lines.append("OPENCONTROL PERFORMANCE ANALYSIS REPORT")
        report_lines.append("=" * 60)
        
        # World model analysis
        if any('prediction' in key for key in metrics.keys()):
            report_lines.append("\nWORLD MODEL ANALYSIS")
            report_lines.append("-" * 30)
            
            pred_analysis = self.analyze_prediction_performance(metrics)
            
            if 'summary' in pred_analysis:
                summary = pred_analysis['summary']
                report_lines.append(f"Average MSE: {summary.get('avg_mse', 'N/A'):.4f}")
                
                if 'best_modality' in summary:
                    best_mod, best_mse = summary['best_modality']
                    report_lines.append(f"Best modality: {best_mod} (MSE: {best_mse:.4f})")
                
                if 'worst_modality' in summary:
                    worst_mod, worst_mse = summary['worst_modality']
                    report_lines.append(f"Worst modality: {worst_mod} (MSE: {worst_mse:.4f})")
            
            if pred_analysis.get('recommendations'):
                report_lines.append("\nRecommendations:")
                for rec in pred_analysis['recommendations']:
                    report_lines.append(f"  • {rec}")
        
        # Control system analysis
        if any('control' in key for key in metrics.keys()):
            report_lines.append("\nCONTROL SYSTEM ANALYSIS")
            report_lines.append("-" * 30)
            
            control_analysis = self.analyze_control_performance(metrics)
            
            if 'efficiency_analysis' in control_analysis:
                eff = control_analysis['efficiency_analysis']
                report_lines.append(f"Real-time performance: {eff.get('real_time_performance', 'N/A')}")
                report_lines.append(f"Efficiency score: {eff.get('efficiency_score', 'N/A'):.3f}")
            
            if 'safety_analysis' in control_analysis:
                safety = control_analysis['safety_analysis']
                report_lines.append(f"Safety level: {safety.get('safety_level', 'N/A')}")
                report_lines.append(f"Safety score: {safety.get('safety_score', 'N/A'):.3f}")
            
            if control_analysis.get('recommendations'):
                report_lines.append("\nRecommendations:")
                for rec in control_analysis['recommendations']:
                    report_lines.append(f"  • {rec}")
        
        # Bottleneck analysis
        report_lines.append("\nBOTTLENECK ANALYSIS")
        report_lines.append("-" * 30)
        
        bottlenecks = self.identify_bottlenecks(metrics)
        
        total_bottlenecks = (
            len(bottlenecks.get('computational', [])) +
            len(bottlenecks.get('algorithmic', [])) +
            len(bottlenecks.get('system', []))
        )
        
        if total_bottlenecks == 0:
            report_lines.append("No significant bottlenecks detected")
        else:
            report_lines.append(f"{total_bottlenecks} bottlenecks identified")
            
            for category, issues in bottlenecks.items():
                if category != 'recommendations' and issues:
                    report_lines.append(f"\n{category.title()} Issues:")
                    for issue in issues:
                        severity_indicator = "[HIGH]" if issue['severity'] == 'high' else "[MEDIUM]"
                        report_lines.append(f"  {severity_indicator} {issue['description']}")
        
        if bottlenecks.get('recommendations'):
            report_lines.append("\nBottleneck Recommendations:")
            for rec in bottlenecks['recommendations']:
                report_lines.append(f"  • {rec}")
        
        report_lines.append("\n" + "=" * 60)
        
        return "\n".join(report_lines) 