"""
OpenControl: Advanced Multi-Modal World Model Platform

A complete, production-ready platform for building, training, and deploying
large-scale multi-modal world models for embodied AI.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

__version__ = "1.0.0"
__author__ = "Nik Jois"
__email__ = "nikjois@llamasearch.ai"
__description__ = "Advanced Multi-Modal World Model Platform for Embodied AI"

# Core imports for convenience
from .core.world_model import OpenControlWorldModel
from .cli.commands import OpenControlConfig

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "__description__",
    "OpenControlWorldModel",
    "OpenControlConfig"
] 