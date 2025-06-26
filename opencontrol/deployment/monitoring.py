"""
Production Monitoring System for OpenControl.

This module provides comprehensive monitoring, logging, and alerting
capabilities for production OpenControl deployments.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import time
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
from collections import deque, defaultdict
import numpy as np
import torch

from opencontrol.cli.commands import OpenControlConfig


class ProductionMonitor:
    """
    Comprehensive production monitoring system.
    
    This class provides:
    - Performance metrics collection
    - Resource usage monitoring
    - Error tracking and alerting
    - Health status reporting
    - Metrics persistence and retrieval
    """
    
    def __init__(self, config: OpenControlConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Metrics storage
        self.metrics = {
            'prediction_times': deque(maxlen=1000),
            'control_times': deque(maxlen=1000),
            'prediction_counts': deque(maxlen=1000),
            'control_counts': deque(maxlen=1000),
            'error_counts': defaultdict(int),
            'resource_usage': deque(maxlen=100)
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'max_prediction_time': 1.0,  # seconds
            'max_control_time': 0.1,     # seconds
            'max_error_rate': 0.05,      # 5%
            'max_memory_usage': 0.9,     # 90%
            'max_gpu_memory': 0.9        # 90%
        }
        
        # System state
        self.start_time = time.time()
        self.last_health_check = time.time()
        self.alerts = deque(maxlen=100)
        
        # Background monitoring task
        self.monitoring_task = None
        self.is_monitoring = False
        
    async def start_monitoring(self):
        """Start background monitoring."""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.logger.info("Production monitoring started")
    
    async def stop_monitoring(self):
        """Stop background monitoring."""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Production monitoring stopped")
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.is_monitoring:
            try:
                await self._collect_system_metrics()
                await self._check_alerts()
                await asyncio.sleep(10)  # Check every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(30)  # Wait longer on error
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            import psutil
            
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            # GPU metrics if available
            gpu_metrics = {}
            if torch.cuda.is_available():
                gpu_metrics = {
                    'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**3,
                    'gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**3,
                    'gpu_utilization': self._get_gpu_utilization()
                }
            
            resource_usage = {
                'timestamp': time.time(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / 1024**3,
                **gpu_metrics
            }
            
            self.metrics['resource_usage'].append(resource_usage)
            
        except ImportError:
            # psutil not available
            pass
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return utilization.gpu
        except:
            return 0.0
    
    async def _check_alerts(self):
        """Check for alert conditions."""
        current_time = time.time()
        
        # Check prediction time alerts
        if self.metrics['prediction_times']:
            avg_pred_time = np.mean(list(self.metrics['prediction_times']))
            if avg_pred_time > self.alert_thresholds['max_prediction_time']:
                await self._trigger_alert(
                    'high_prediction_time',
                    f"Average prediction time {avg_pred_time:.3f}s exceeds threshold"
                )
        
        # Check control time alerts
        if self.metrics['control_times']:
            avg_control_time = np.mean(list(self.metrics['control_times']))
            if avg_control_time > self.alert_thresholds['max_control_time']:
                await self._trigger_alert(
                    'high_control_time',
                    f"Average control time {avg_control_time:.3f}s exceeds threshold"
                )
        
        # Check error rate alerts
        total_requests = len(self.metrics['prediction_times']) + len(self.metrics['control_times'])
        total_errors = sum(self.metrics['error_counts'].values())
        if total_requests > 0:
            error_rate = total_errors / total_requests
            if error_rate > self.alert_thresholds['max_error_rate']:
                await self._trigger_alert(
                    'high_error_rate',
                    f"Error rate {error_rate:.3f} exceeds threshold"
                )
        
        # Check resource usage alerts
        if self.metrics['resource_usage']:
            latest_usage = self.metrics['resource_usage'][-1]
            
            if latest_usage['memory_percent'] > self.alert_thresholds['max_memory_usage'] * 100:
                await self._trigger_alert(
                    'high_memory_usage',
                    f"Memory usage {latest_usage['memory_percent']:.1f}% exceeds threshold"
                )
            
            if 'gpu_memory_allocated' in latest_usage:
                gpu_usage = latest_usage['gpu_memory_allocated'] / latest_usage.get('gpu_memory_reserved', 1.0)
                if gpu_usage > self.alert_thresholds['max_gpu_memory']:
                    await self._trigger_alert(
                        'high_gpu_memory',
                        f"GPU memory usage {gpu_usage:.1f} exceeds threshold"
                    )
    
    async def _trigger_alert(self, alert_type: str, message: str):
        """Trigger an alert."""
        alert = {
            'timestamp': time.time(),
            'type': alert_type,
            'message': message,
            'severity': self._get_alert_severity(alert_type)
        }
        
        self.alerts.append(alert)
        self.logger.warning(f"ALERT [{alert_type}]: {message}")
        
        # Here you could integrate with external alerting systems
        # e.g., send to Slack, PagerDuty, email, etc.
    
    def _get_alert_severity(self, alert_type: str) -> str:
        """Get severity level for alert type."""
        high_severity = ['high_error_rate', 'high_gpu_memory']
        medium_severity = ['high_prediction_time', 'high_control_time']
        
        if alert_type in high_severity:
            return 'high'
        elif alert_type in medium_severity:
            return 'medium'
        else:
            return 'low'
    
    def log_prediction_metrics(self, processing_time: float, num_predictions: int):
        """Log prediction performance metrics."""
        self.metrics['prediction_times'].append(processing_time)
        self.metrics['prediction_counts'].append(num_predictions)
    
    def log_control_metrics(self, processing_time: float, cost: float, safety_status: str):
        """Log control performance metrics."""
        self.metrics['control_times'].append(processing_time)
        
        # Track safety status
        if safety_status != 'safe':
            self.metrics['error_counts'][f'unsafe_control_{safety_status}'] += 1
    
    def log_error(self, error_type: str, error_message: str):
        """Log an error occurrence."""
        self.metrics['error_counts'][error_type] += 1
        self.logger.error(f"Error [{error_type}]: {error_message}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Compute statistics
        prediction_stats = self._compute_time_stats(self.metrics['prediction_times'])
        control_stats = self._compute_time_stats(self.metrics['control_times'])
        
        # Resource usage stats
        resource_stats = {}
        if self.metrics['resource_usage']:
            latest_resources = self.metrics['resource_usage'][-1]
            resource_stats = {
                'cpu_percent': latest_resources.get('cpu_percent', 0),
                'memory_percent': latest_resources.get('memory_percent', 0),
                'gpu_memory_gb': latest_resources.get('gpu_memory_allocated', 0),
                'gpu_utilization': latest_resources.get('gpu_utilization', 0)
            }
        
        # Throughput calculations
        prediction_throughput = len(self.metrics['prediction_times']) / (uptime / 3600)  # per hour
        control_throughput = len(self.metrics['control_times']) / (uptime / 3600)  # per hour
        
        return {
            'uptime_seconds': uptime,
            'prediction_metrics': {
                'count': len(self.metrics['prediction_times']),
                'throughput_per_hour': prediction_throughput,
                **prediction_stats
            },
            'control_metrics': {
                'count': len(self.metrics['control_times']),
                'throughput_per_hour': control_throughput,
                **control_stats
            },
            'error_counts': dict(self.metrics['error_counts']),
            'resource_usage': resource_stats,
            'alerts': {
                'total_count': len(self.alerts),
                'recent_alerts': list(self.alerts)[-5:] if self.alerts else []
            },
            'health_status': self.get_health_status()
        }
    
    def _compute_time_stats(self, times: deque) -> Dict[str, float]:
        """Compute timing statistics."""
        if not times:
            return {'count': 0}
        
        times_array = np.array(list(times))
        return {
            'count': len(times),
            'mean': float(np.mean(times_array)),
            'std': float(np.std(times_array)),
            'min': float(np.min(times_array)),
            'max': float(np.max(times_array)),
            'p50': float(np.percentile(times_array, 50)),
            'p95': float(np.percentile(times_array, 95)),
            'p99': float(np.percentile(times_array, 99))
        }
    
    def get_health_status(self) -> str:
        """Get overall system health status."""
        # Check recent alerts
        recent_alerts = [a for a in self.alerts if time.time() - a['timestamp'] < 300]  # Last 5 minutes
        
        critical_alerts = [a for a in recent_alerts if a['severity'] == 'high']
        if critical_alerts:
            return 'critical'
        
        warning_alerts = [a for a in recent_alerts if a['severity'] == 'medium']
        if warning_alerts:
            return 'warning'
        
        # Check if we have recent activity
        if not self.metrics['prediction_times'] and not self.metrics['control_times']:
            return 'idle'
        
        return 'healthy'
    
    def reset_metrics(self):
        """Reset all metrics."""
        for key in self.metrics:
            if isinstance(self.metrics[key], deque):
                self.metrics[key].clear()
            elif isinstance(self.metrics[key], defaultdict):
                self.metrics[key].clear()
        
        self.alerts.clear()
        self.start_time = time.time()
        self.logger.info("Metrics reset")
    
    async def save_metrics(self, filepath: str):
        """Save metrics to file."""
        try:
            metrics_data = {
                'timestamp': time.time(),
                'metrics': self.get_metrics(),
                'raw_data': {
                    'prediction_times': list(self.metrics['prediction_times']),
                    'control_times': list(self.metrics['control_times']),
                    'error_counts': dict(self.metrics['error_counts']),
                    'alerts': list(self.alerts)
                }
            }
            
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
            
            self.logger.info(f"Metrics saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")
    
    async def load_metrics(self, filepath: str):
        """Load metrics from file."""
        try:
            with open(filepath, 'r') as f:
                metrics_data = json.load(f)
            
            raw_data = metrics_data.get('raw_data', {})
            
            # Restore metrics
            if 'prediction_times' in raw_data:
                self.metrics['prediction_times'].extend(raw_data['prediction_times'])
            
            if 'control_times' in raw_data:
                self.metrics['control_times'].extend(raw_data['control_times'])
            
            if 'error_counts' in raw_data:
                self.metrics['error_counts'].update(raw_data['error_counts'])
            
            if 'alerts' in raw_data:
                self.alerts.extend(raw_data['alerts'])
            
            self.logger.info(f"Metrics loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load metrics: {e}")
    
    def get_performance_report(self) -> str:
        """Generate a human-readable performance report."""
        metrics = self.get_metrics()
        
        report = []
        report.append("=== OpenControl Performance Report ===")
        report.append(f"Uptime: {metrics['uptime_seconds']:.1f} seconds")
        report.append(f"Health Status: {metrics['health_status']}")
        report.append("")
        
        # Prediction metrics
        pred_metrics = metrics['prediction_metrics']
        report.append("Prediction Performance:")
        report.append(f"  Total Requests: {pred_metrics['count']}")
        report.append(f"  Throughput: {pred_metrics['throughput_per_hour']:.1f} req/hour")
        if pred_metrics['count'] > 0:
            report.append(f"  Average Time: {pred_metrics['mean']:.3f}s")
            report.append(f"  P95 Time: {pred_metrics['p95']:.3f}s")
        report.append("")
        
        # Control metrics
        control_metrics = metrics['control_metrics']
        report.append("Control Performance:")
        report.append(f"  Total Requests: {control_metrics['count']}")
        report.append(f"  Throughput: {control_metrics['throughput_per_hour']:.1f} req/hour")
        if control_metrics['count'] > 0:
            report.append(f"  Average Time: {control_metrics['mean']:.3f}s")
            report.append(f"  P95 Time: {control_metrics['p95']:.3f}s")
        report.append("")
        
        # Resource usage
        resource_usage = metrics['resource_usage']
        if resource_usage:
            report.append("Resource Usage:")
            report.append(f"  CPU: {resource_usage['cpu_percent']:.1f}%")
            report.append(f"  Memory: {resource_usage['memory_percent']:.1f}%")
            if 'gpu_memory_gb' in resource_usage:
                report.append(f"  GPU Memory: {resource_usage['gpu_memory_gb']:.1f} GB")
                report.append(f"  GPU Utilization: {resource_usage['gpu_utilization']:.1f}%")
        report.append("")
        
        # Errors and alerts
        error_counts = metrics['error_counts']
        if error_counts:
            report.append("Error Summary:")
            for error_type, count in error_counts.items():
                report.append(f"  {error_type}: {count}")
        report.append("")
        
        alerts = metrics['alerts']
        if alerts['recent_alerts']:
            report.append("Recent Alerts:")
            for alert in alerts['recent_alerts']:
                report.append(f"  [{alert['severity'].upper()}] {alert['type']}: {alert['message']}")
        
        return "\n".join(report)


class MetricsCollector:
    """Lightweight metrics collector for embedding in other components."""
    
    def __init__(self, monitor: ProductionMonitor):
        self.monitor = monitor
    
    def time_prediction(self, func):
        """Decorator to time prediction functions."""
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                processing_time = time.time() - start_time
                self.monitor.log_prediction_metrics(processing_time, 1)
                return result
            except Exception as e:
                self.monitor.log_error('prediction_error', str(e))
                raise
        return wrapper
    
    def time_control(self, func):
        """Decorator to time control functions."""
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                processing_time = time.time() - start_time
                # Extract safety status from result if available
                safety_status = 'safe'
                if isinstance(result, tuple) and len(result) > 1:
                    info = result[1]
                    if isinstance(info, dict) and 'safety_info' in info:
                        safety_status = 'safe' if info['safety_info'].get('is_safe', True) else 'unsafe'
                
                self.monitor.log_control_metrics(processing_time, 0.0, safety_status)
                return result
            except Exception as e:
                self.monitor.log_error('control_error', str(e))
                raise
        return wrapper 