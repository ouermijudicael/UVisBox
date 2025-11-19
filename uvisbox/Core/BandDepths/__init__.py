"""
Core BandDepths Module

This module contains core statistical methods for band depth calculations.
These are fundamental algorithms used across the UVisBox package.
"""


from .contour_banddepth import contour_banddepth
from .curve_banddepth import curve_banddepths
from .functional_banddepth import functional_banddepth, modified_functional_banddepth
from .vector_depths import compute_vector_depths_2D, compute_vector_depths_3D

__all__ = [
    "contour_banddepth",
    "curve_banddepths",
    "functional_banddepth",
    "modified_functional_banddepth",
    "compute_vector_depths_2D",
    "compute_vector_depths_3D",
]