"""
Training Callbacks for OpenControl World Model Training.

This module provides various callback classes for monitoring training progress,
logging metrics, saving checkpoints, and integrating with external services.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

import torch
import numpy as np


class TrainingCallback(ABC):
    """Base class for training callbacks."""
    
    def on_train_begin(self):
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self):
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, epoch: int):
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        """Called at the end of each epoch."""
        pass
    
    def on_step_end(self, step: int, metrics: Dict[str, Any]):
        """Called at the end of each training step."""
        pass


class LoggingCallback(TrainingCallback):
    """Callback for logging training metrics to file and console."""
    
    def __init__(self, log_dir: str, log_interval: int = 100):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        
        # Set up logging
        self.logger = logging.getLogger('TrainingLogger')
        handler = logging.FileHandler(self.log_dir / 'training.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Metrics storage
        self.metrics_history = []
        self.start_time = None
        
    def on_train_begin(self):
        self.start_time = time.time()
        self.logger.info("Training started")
        
    def on_train_end(self):
        total_time = time.time() - self.start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Save metrics history
        metrics_file = self.log_dir / 'metrics_history.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
            
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        metrics_str = ', '.join([f'{k}: {v:.6f}' if isinstance(v, float) else f'{k}: {v}' 
                                for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch}: {metrics_str}")
        
        # Store metrics
        epoch_metrics = {'epoch': epoch, 'timestamp': time.time(), **metrics}
        self.metrics_history.append(epoch_metrics)
        
    def on_step_end(self, step: int, metrics: Dict[str, Any]):
        if step % self.log_interval == 0:
            metrics_str = ', '.join([f'{k}: {v:.6f}' if isinstance(v, float) else f'{k}: {v}' 
                                    for k, v in metrics.items()])
            self.logger.info(f"Step {step}: {metrics_str}")


class WandbCallback(TrainingCallback):
    """Callback for logging to Weights & Biases."""
    
    def __init__(self, project: str, entity: Optional[str] = None, config: Optional[Dict] = None):
        try:
            import wandb
            self.wandb = wandb
            self.project = project
            self.entity = entity
            self.config = config
            self.run = None
        except ImportError:
            raise ImportError("wandb is required for WandbCallback. Install with: pip install wandb")
    
    def on_train_begin(self):
        self.run = self.wandb.init(
            project=self.project,
            entity=self.entity,
            config=self.config
        )
        
    def on_train_end(self):
        if self.run:
            self.run.finish()
            
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        if self.run:
            log_metrics = {'epoch': epoch}
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    log_metrics[key] = value
            self.wandb.log(log_metrics)
            
    def on_step_end(self, step: int, metrics: Dict[str, Any]):
        if self.run:
            log_metrics = {'step': step}
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    log_metrics[key] = value
            self.wandb.log(log_metrics)


class TensorBoardCallback(TrainingCallback):
    """Callback for logging to TensorBoard."""
    
    def __init__(self, log_dir: str):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.log_dir = log_dir
            self.writer = SummaryWriter(log_dir)
        except ImportError:
            raise ImportError("tensorboard is required for TensorBoardCallback. Install with: pip install tensorboard")
    
    def on_train_end(self):
        self.writer.close()
        
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'epoch/{key}', value, epoch)
                
    def on_step_end(self, step: int, metrics: Dict[str, Any]):
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'step/{key}', value, step)


class CheckpointCallback(TrainingCallback):
    """Callback for saving model checkpoints."""
    
    def __init__(
        self, 
        checkpoint_dir: str,
        save_interval: int = 1000,
        save_best: bool = True,
        monitor: str = 'val_loss',
        mode: str = 'min'
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = save_interval
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.last_checkpoint_step = 0
        
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        if self.save_best and self.monitor in metrics:
            current_value = metrics[self.monitor]
            is_better = (
                (self.mode == 'min' and current_value < self.best_value) or
                (self.mode == 'max' and current_value > self.best_value)
            )
            
            if is_better:
                self.best_value = current_value
                # This would typically save the model, but we need access to the trainer
                # In practice, this would be handled by the trainer itself
                pass
                
    def on_step_end(self, step: int, metrics: Dict[str, Any]):
        if step - self.last_checkpoint_step >= self.save_interval:
            self.last_checkpoint_step = step
            # Checkpoint saving would be handled by the trainer


class EarlyStopping(TrainingCallback):
    """Callback for early stopping based on validation metrics."""
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'min'
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
        self.should_stop = False
        
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        if self.monitor not in metrics:
            return
            
        current_value = metrics[self.monitor]
        
        if self.mode == 'min':
            is_better = current_value < self.best_value - self.min_delta
        else:
            is_better = current_value > self.best_value + self.min_delta
            
        if is_better:
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.should_stop = True
            print(f"Early stopping triggered after {epoch} epochs")


class LearningRateSchedulerCallback(TrainingCallback):
    """Callback for learning rate scheduling based on metrics."""
    
    def __init__(self, scheduler, monitor: str = 'val_loss'):
        self.scheduler = scheduler
        self.monitor = monitor
        
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        if hasattr(self.scheduler, 'step'):
            if self.monitor in metrics and hasattr(self.scheduler, 'step') and 'ReduceLROnPlateau' in str(type(self.scheduler)):
                self.scheduler.step(metrics[self.monitor])
            else:
                self.scheduler.step()


class ModelVisualizationCallback(TrainingCallback):
    """Callback for visualizing model predictions during training."""
    
    def __init__(self, log_dir: str, visualize_interval: int = 1000):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.visualize_interval = visualize_interval
        self.step_count = 0
        
    def on_step_end(self, step: int, metrics: Dict[str, Any]):
        self.step_count += 1
        if self.step_count % self.visualize_interval == 0:
            # This would typically receive model outputs and create visualizations
            # Implementation would depend on having access to the model and data
            pass


class MetricsTracker(TrainingCallback):
    """Callback for tracking and computing running statistics of metrics."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_buffer = {}
        
    def on_step_end(self, step: int, metrics: Dict[str, Any]):
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if key not in self.metrics_buffer:
                    self.metrics_buffer[key] = []
                
                self.metrics_buffer[key].append(value)
                
                # Keep only the last window_size values
                if len(self.metrics_buffer[key]) > self.window_size:
                    self.metrics_buffer[key] = self.metrics_buffer[key][-self.window_size:]
    
    def get_running_average(self, metric: str) -> Optional[float]:
        """Get the running average of a metric."""
        if metric in self.metrics_buffer and self.metrics_buffer[metric]:
            return sum(self.metrics_buffer[metric]) / len(self.metrics_buffer[metric])
        return None
    
    def get_running_std(self, metric: str) -> Optional[float]:
        """Get the running standard deviation of a metric."""
        if metric in self.metrics_buffer and len(self.metrics_buffer[metric]) > 1:
            values = self.metrics_buffer[metric]
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            return variance ** 0.5
        return None


class GradientMonitorCallback(TrainingCallback):
    """Callback for monitoring gradient statistics."""
    
    def __init__(self, model: torch.nn.Module, log_interval: int = 100):
        self.model = model
        self.log_interval = log_interval
        self.step_count = 0
        
    def on_step_end(self, step: int, metrics: Dict[str, Any]):
        self.step_count += 1
        if self.step_count % self.log_interval == 0:
            grad_stats = self._compute_gradient_stats()
            
            # Add gradient statistics to metrics
            for key, value in grad_stats.items():
                metrics[f'grad_{key}'] = value
                
    def _compute_gradient_stats(self) -> Dict[str, float]:
        """Compute gradient statistics."""
        total_norm = 0.0
        param_count = 0
        grad_norms = []
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                grad_norms.append(param_norm.item())
                param_count += 1
        
        total_norm = total_norm ** (1.0 / 2)
        
        return {
            'total_norm': total_norm,
            'mean_norm': np.mean(grad_norms) if grad_norms else 0.0,
            'std_norm': np.std(grad_norms) if grad_norms else 0.0,
            'max_norm': np.max(grad_norms) if grad_norms else 0.0,
            'min_norm': np.min(grad_norms) if grad_norms else 0.0
        }


class CallbackManager:
    """Manager for handling multiple callbacks."""
    
    def __init__(self, callbacks: Optional[List[TrainingCallback]] = None):
        self.callbacks = callbacks or []
        
    def add_callback(self, callback: TrainingCallback):
        """Add a callback to the manager."""
        self.callbacks.append(callback)
        
    def remove_callback(self, callback: TrainingCallback):
        """Remove a callback from the manager."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            
    def on_train_begin(self):
        """Call on_train_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin()
            
    def on_train_end(self):
        """Call on_train_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_end()
            
    def on_epoch_begin(self, epoch: int):
        """Call on_epoch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)
            
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        """Call on_epoch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, metrics)
            
    def on_step_end(self, step: int, metrics: Dict[str, Any]):
        """Call on_step_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_step_end(step, metrics) 