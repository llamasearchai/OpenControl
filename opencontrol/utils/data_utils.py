"""
Data Processing Utilities for OpenControl.

This module provides utility functions for data preprocessing, normalization,
and tensor operations commonly used throughout the OpenControl system.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import torch
import numpy as np
from typing import Tuple, Dict, Any, Optional


def normalize_tensor(
    tensor: torch.Tensor, 
    mean: Optional[torch.Tensor] = None, 
    std: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Normalize tensor using mean and standard deviation."""
    if mean is None:
        mean = tensor.mean()
    if std is None:
        std = tensor.std()
    
    return (tensor - mean) / (std + 1e-8)


def denormalize_tensor(
    tensor: torch.Tensor, 
    mean: torch.Tensor, 
    std: torch.Tensor
) -> torch.Tensor:
    """Denormalize tensor using mean and standard deviation."""
    return tensor * std + mean


def tensor_stats(tensor: torch.Tensor) -> Dict[str, float]:
    """Compute comprehensive statistics for a tensor."""
    return {
        'mean': tensor.mean().item(),
        'std': tensor.std().item(),
        'min': tensor.min().item(),
        'max': tensor.max().item(),
        'shape': list(tensor.shape),
        'numel': tensor.numel()
    } 