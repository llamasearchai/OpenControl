"""
Advanced Distributed Trainer for OpenControl World Models.

This module encapsulates the entire training loop, including optimization,
mixed-precision, gradient accumulation, checkpointing, and logging.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
import time
import os
from pathlib import Path
from typing import Callable, Optional, Dict, Any
import json

from opencontrol.core.world_model import OpenControlWorldModel
from opencontrol.data.dataset_manager import MultiModalDatasetManager
from opencontrol.cli.commands import OpenControlConfig
from .optimizers import create_optimizer, create_scheduler
from .losses import WorldModelLoss
from .callbacks import TrainingCallback


class DistributedWorldModelTrainer:
    """Manages the distributed training of the world model."""
    
    def __init__(
        self,
        model: OpenControlWorldModel,
        dataset_manager: MultiModalDatasetManager,
        config: OpenControlConfig,
        logger: logging.Logger
    ):
        self.model = model
        self.dataset_manager = dataset_manager
        self.config = config
        self.logger = logger
        self.device = next(model.parameters()).device
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        # Wrap model in DDP if distributed
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.rank])
            
        self.optimizer = create_optimizer(self.model, config)
        self.scheduler = create_scheduler(self.optimizer, config)
        self.scaler = GradScaler(enabled=self.config.training.mixed_precision)
        self.loss_fn = WorldModelLoss(config)
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_loss = float('inf')
        
        # Callbacks
        self.callbacks = []
        
        # Create checkpoint directory
        if self.rank == 0:
            self.checkpoint_dir = Path(config.infrastructure.checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def add_callback(self, callback: TrainingCallback):
        """Add a training callback."""
        self.callbacks.append(callback)

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        if self.rank != 0:
            return
            
        # Get the actual model (unwrap DDP if needed)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        checkpoint = {
            'epoch': self.current_epoch,
            'step': self.current_step,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config.__dict__
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            
        # Keep only last 3 checkpoints
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        if len(checkpoints) > 3:
            for old_checkpoint in checkpoints[:-3]:
                old_checkpoint.unlink()
                
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Get the actual model (unwrap DDP if needed)
        model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.current_step = checkpoint['step']
        self.best_loss = checkpoint['best_loss']
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")

    def train_epoch(self, progress_callback: Optional[Callable] = None) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        train_loader = self.dataset_manager.get_train_loader()
        
        # Set epoch for distributed sampler
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(self.current_epoch)
            
        for batch_idx, batch in enumerate(train_loader):
            batch_start_time = time.time()
            
            # Move batch to device
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            with autocast(enabled=self.config.training.mixed_precision):
                outputs = self.model(batch, prediction_horizon=1)
                loss = self.loss_fn(outputs, batch)
                loss = loss / self.config.training.gradient_accumulation_steps
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Optimizer step with gradient accumulation
            if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training.gradient_clipping
                )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                
                self.current_step += 1
                
                # Logging and callbacks
                if self.rank == 0:
                    batch_time = time.time() - batch_start_time
                    lr = self.optimizer.param_groups[0]['lr']
                    
                    metrics = {
                        'loss': loss.item() * self.config.training.gradient_accumulation_steps,
                        'lr': lr,
                        'batch_time': batch_time,
                        'epoch': self.current_epoch,
                        'step': self.current_step
                    }
                    
                    if self.current_step % 100 == 0:
                        self.logger.info(
                            f"Epoch {self.current_epoch}, Step {self.current_step}, "
                            f"Loss: {metrics['loss']:.4f}, LR: {lr:.6f}, "
                            f"Time: {batch_time:.3f}s"
                        )
                    
                    # Execute callbacks
                    for callback in self.callbacks:
                        callback.on_step_end(self.current_step, metrics)
                    
                    if progress_callback:
                        progress_callback(self.current_epoch, self.current_step, metrics)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Early stopping for steps per epoch
            if batch_idx >= self.config.training.steps_per_epoch:
                break
                
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'avg_loss': avg_loss}

    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        val_loader = self.dataset_manager.get_val_loader()
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                
                with autocast(enabled=self.config.training.mixed_precision):
                    outputs = self.model(batch, prediction_horizon=1)
                    loss = self.loss_fn(outputs, batch)
                
                total_loss += loss.item()
                num_batches += 1
                
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Synchronize validation loss across all ranks
        if self.world_size > 1:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / self.world_size
            
        return {'val_loss': avg_loss}

    async def train(self, progress_callback: Optional[Callable] = None):
        """Main training loop."""
        self.logger.info(f"Starting training for {self.config.training.num_epochs} epochs")
        
        # Execute callbacks
        for callback in self.callbacks:
            callback.on_train_begin()
            
        try:
            for epoch in range(self.current_epoch, self.config.training.num_epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()
                
                # Execute callbacks
                for callback in self.callbacks:
                    callback.on_epoch_begin(epoch)
                
                # Training
                train_metrics = self.train_epoch(progress_callback)
                
                # Validation
                val_metrics = self.validate()
                
                epoch_time = time.time() - epoch_start_time
                
                # Combine metrics
                metrics = {**train_metrics, **val_metrics, 'epoch_time': epoch_time}
                
                if self.rank == 0:
                    self.logger.info(
                        f"Epoch {epoch + 1}/{self.config.training.num_epochs} completed - "
                        f"Train Loss: {train_metrics['avg_loss']:.4f}, "
                        f"Val Loss: {val_metrics['val_loss']:.4f}, "
                        f"Time: {epoch_time:.2f}s"
                    )
                    
                    # Save checkpoint
                    is_best = val_metrics['val_loss'] < self.best_loss
                    if is_best:
                        self.best_loss = val_metrics['val_loss']
                        
                    if (epoch + 1) % self.config.infrastructure.checkpoint_interval_steps == 0:
                        self.save_checkpoint(is_best)
                
                # Execute callbacks
                for callback in self.callbacks:
                    callback.on_epoch_end(epoch, metrics)
                    
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}", exc_info=True)
            raise
        finally:
            # Execute callbacks
            for callback in self.callbacks:
                callback.on_train_end()
                
            if self.rank == 0:
                self.save_checkpoint()
                
        self.logger.info("Training completed successfully") 