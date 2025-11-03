"""
ProbabilisticMarchingTriangles Module

This module provides functionality for performing probabilistic marching triangles 
on 2D triangular meshes with uncertainty. It includes methods for calculating 
triangle crossing probabilities and visualizing the results using matplotlib.
"""

from .probabilistic_marching_triangles_stats import (
    probabilistic_marching_triangle_summary_statistics,
)
from .probabilistic_marching_triangles_mesh import (
    probabilistic_marching_triangle_mesh,
)
from .probabilistic_marching_triangles_vis import (
    visualize_probabilistic_marching_triangle,
)
from .probabilistic_marching_triangles import (
    probabilistic_marching_triangle,
)

__all__ = [
    'probabilistic_marching_triangle_summary_statistics',
    'probabilistic_marching_triangle_mesh',
    'visualize_probabilistic_marching_triangle',
    'probabilistic_marching_triangle',
]
