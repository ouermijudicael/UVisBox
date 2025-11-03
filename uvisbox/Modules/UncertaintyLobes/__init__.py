"""
UncertaintyLobes Module

This module provides functionality for visualizing uncertainty lobes in 2D vector field ensembles.
It follows a three-stage visualization pipeline:
    1. Compute statistics (vector depths, spreads, angular ranges)
    2. Build mesh geometry (wedge vertices, triangles)
    3. Render visualization (matplotlib)

Main functions:
    - uncertainty_lobes(): High-level API for complete visualization
    - compute_uncertainty_lobes_stats_2d(): Stage 1 - Statistics computation
    - build_uncertainty_lobes_mesh_2d(): Stage 2 - Mesh generation
    - render_uncertainty_lobes_2d(): Stage 3 - Visualization rendering
""" 

from .uncertainty_lobes import uncertainty_lobes
from .uncertainty_lobes_stats import compute_uncertainty_lobes_stats_2d
from .uncertainty_lobes_mesh import build_uncertainty_lobes_mesh_2d
from .uncertainty_lobes_vis import render_uncertainty_lobes_2d

__all__ = [
    'uncertainty_lobes',
    'compute_uncertainty_lobes_stats_2d',
    'build_uncertainty_lobes_mesh_2d',
    'render_uncertainty_lobes_2d'
]
