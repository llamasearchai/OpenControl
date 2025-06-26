"""
Advanced Optimizer and Scheduler Creation for OpenControl.

This module provides factory functions for creating optimizers and learning rate
schedulers with proper weight decay handling and advanced scheduling strategies.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import Dict, List, Union
from opencontrol.cli.commands import OpenControlConfig


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    Cosine annealing with warm restarts and linear warmup.
    
    This scheduler combines the benefits of:
    - Linear warmup to prevent early training instability
    - Cosine annealing for smooth learning rate decay
    - Warm restarts to escape local minima
    """
    
    def __init__(
        self, 
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1
    ):
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # Initialize learning rates
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr
                    for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) * 
                    (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) / 
                                  (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
        
        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def create_optimizer(model: torch.nn.Module, config: OpenControlConfig) -> torch.optim.Optimizer:
    """
    Create an optimizer with proper weight decay handling.
    
    This function separates parameters into two groups:
    - Parameters that should have weight decay (weights)
    - Parameters that should not have weight decay (biases, layer norms)
    """
    # Separate parameters for weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Don't apply weight decay to biases and normalization layers
        if len(param.shape) == 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    # Create parameter groups
    param_groups = [
        {
            'params': decay_params,
            'weight_decay': config.training.weight_decay,
            'lr': config.training.learning_rate
        },
        {
            'params': no_decay_params,
            'weight_decay': 0.0,
            'lr': config.training.learning_rate
        }
    ]
    
    optimizer_name = config.training.optimizer.lower()
    
    if optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            param_groups,
            lr=config.training.learning_rate,
            betas=(config.training.adam_beta1, config.training.adam_beta2),
            eps=config.training.adam_eps,
            weight_decay=config.training.weight_decay
        )
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(
            param_groups,
            lr=config.training.learning_rate,
            betas=(config.training.adam_beta1, config.training.adam_beta2),
            eps=config.training.adam_eps
        )
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(
            param_groups,
            lr=config.training.learning_rate,
            momentum=0.9,
            weight_decay=config.training.weight_decay
        )
    elif optimizer_name == 'lion':
        try:
            from lion_pytorch import Lion
            optimizer = Lion(
                param_groups,
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay
            )
        except ImportError:
            raise ImportError("Lion optimizer requires 'lion-pytorch' package. Install with: pip install lion-pytorch")
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, config: OpenControlConfig) -> _LRScheduler:
    """
    Create a learning rate scheduler based on the configuration.
    """
    scheduler_name = config.training.lr_schedule.lower()
    
    if scheduler_name == 'cosine_with_warmup':
        total_steps = config.training.num_epochs * config.training.steps_per_epoch
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=total_steps,
            max_lr=config.training.learning_rate,
            min_lr=config.training.learning_rate * 0.01,
            warmup_steps=config.training.warmup_steps
        )
    elif scheduler_name == 'cosine':
        total_steps = config.training.num_epochs * config.training.steps_per_epoch
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=config.training.learning_rate * 0.01
        )
    elif scheduler_name == 'linear_warmup':
        def lr_lambda(step):
            if step < config.training.warmup_steps:
                return step / config.training.warmup_steps
            else:
                return max(0.01, (config.training.num_epochs * config.training.steps_per_epoch - step) / 
                          (config.training.num_epochs * config.training.steps_per_epoch - config.training.warmup_steps))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_name == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif scheduler_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config.training.steps_per_epoch * 10,  # Decay every 10 epochs
            gamma=0.5
        )
    elif scheduler_name == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=config.training.learning_rate * 0.001
        )
    elif scheduler_name == 'none':
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
    return scheduler


def get_parameter_count(model: torch.nn.Module) -> Dict[str, int]:
    """Get detailed parameter count information."""
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    
    param_details = {}
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        if param.requires_grad:
            trainable_params += param_count
        else:
            frozen_params += param_count
            
        # Group by module type
        module_type = name.split('.')[0] if '.' in name else 'root'
        if module_type not in param_details:
            param_details[module_type] = 0
        param_details[module_type] += param_count
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params,
        'by_module': param_details
    }


def apply_weight_init(model: torch.nn.Module, init_type: str = 'xavier_uniform'):
    """Apply weight initialization to model parameters."""
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            if init_type == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(module.weight)
            elif init_type == 'xavier_normal':
                torch.nn.init.xavier_normal_(module.weight)
            elif init_type == 'kaiming_uniform':
                torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            elif init_type == 'kaiming_normal':
                torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            elif init_type == 'normal':
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (torch.nn.LayerNorm, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) 