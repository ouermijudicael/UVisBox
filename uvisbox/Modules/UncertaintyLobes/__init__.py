"""
UncertaintyLobes Module

This module provides functionality for visualizing uncertainty lobes in vector field ensembles.
It follows a three-stage visualization pipeline:
    1. Compute statistics (vector depths, spreads, angular ranges)
    2. Build mesh geometry (wedge vertices, triangles)
    3. Render visualization (matplotlib)

Main functions:
    - uncertainty_lobes(): High-level API for complete visualization
    - uncertainty_lobes_summary_statistics(): Stage 1 - Statistics computation
    - uncertainty_lobes_mesh(): Stage 2 - Mesh generation
    - visualize_uncertainty_lobes(): Stage 3 - Visualization rendering
""" 

from .uncertainty_lobes import uncertainty_lobes
from .uncertainty_lobes_stats import uncertainty_lobes_summary_statistics
from .uncertainty_lobes_mesh import uncertainty_lobes_mesh
from .uncertainty_lobes_vis import visualize_uncertainty_lobes

__all__ = [
    'uncertainty_lobes',
    'uncertainty_lobes_summary_statistics',
    'uncertainty_lobes_mesh',
    'visualize_uncertainty_lobes'
]
