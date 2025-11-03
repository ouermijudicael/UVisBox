"""
ProbabilisticMarchingSquares Module

This module provides functionality for performing probabilistic marching squares 
on 2D datasets with uncertainty. It includes methods for calculating cell crossing 
probabilities and visualizing the results using matplotlib.
"""

from .probabilistic_marching_squares_stats import (
    probabilistic_marching_squares_summary_statistics,
)
from .probabilistic_marching_squares_mesh import (
    probabilistic_marching_squares_mesh,
)
from .probabilistic_marching_squares_vis import (
    visualize_probabilistic_marching_squares,
)
from .probabilistic_marching_squares import (
    probabilistic_marching_squares,
)

__all__ = [
    'probabilistic_marching_squares_summary_statistics',
    'probabilistic_marching_squares_mesh',
    'visualize_probabilistic_marching_squares',
    'probabilistic_marching_squares',
]
