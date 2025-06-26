"""
Mathematical Utilities for OpenControl.

This module provides mathematical utility functions for robotics computations
including rotations, transformations, and geometric operations.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import torch
import numpy as np
from typing import Tuple


def rotation_matrix(angle: float, axis: str = 'z') -> torch.Tensor:
    """Create a rotation matrix for the given angle and axis."""
    cos_a = torch.cos(torch.tensor(angle))
    sin_a = torch.sin(torch.tensor(angle))
    
    if axis.lower() == 'x':
        return torch.tensor([
            [1, 0, 0],
            [0, cos_a, -sin_a],
            [0, sin_a, cos_a]
        ], dtype=torch.float32)
    elif axis.lower() == 'y':
        return torch.tensor([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ], dtype=torch.float32)
    elif axis.lower() == 'z':
        return torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=torch.float32)
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'")


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to rotation matrix."""
    # Ensure quaternion is normalized
    q = safe_normalize(quaternion)
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    # Compute rotation matrix
    return torch.tensor([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ], dtype=torch.float32)


def safe_normalize(tensor: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Safely normalize tensor to unit length."""
    norm = torch.norm(tensor, dim=dim, keepdim=True)
    return tensor / (norm + eps) 