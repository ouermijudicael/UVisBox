"""
ProbabilisticMarchingSquares Module

This module provides functionality for performing probabilistic marching squares 
on 2D datasets with uncertainty. It includes methods for calculating cell crossing 
probabilities and visualizing the results using matplotlib.
"""

from .probabilistic_marching_squares_stats import (
    probabilistic_marching_square_summary_statistics,
)
from .probabilistic_marching_squares_mesh import (
    probabilistic_marching_square_mesh,
)
from .probabilistic_marching_squares_vis import (
    visualize_probabilistic_marching_square,
)
from .probabilistic_marching_squares import (
    probabilistic_marching_square,
)

__all__ = [
    'probabilistic_marching_square_summary_statistics',
    'probabilistic_marching_square_mesh',
    'visualize_probabilistic_marching_square',
    'probabilistic_marching_square',
]
