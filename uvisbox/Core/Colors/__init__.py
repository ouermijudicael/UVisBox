"""
Core Colors Module

This module contains core color functionality including color interpolation and tree structures.
These are fundamental color operations used across the UVisBox package.
"""

from .color_interpolator import interpolate_lab
from .colortree import ColorTree

__all__ = [
    "interpolate_lab",
    "ColorTree",
]