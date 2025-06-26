"""
OpenControl CLI Package

Command-line interface components for the OpenControl platform.
"""

from .main import cli
from .commands import OpenControlConfig

__all__ = ["cli", "OpenControlConfig"] 