"""
UncertaintyLobes Module

This module provides functionality for visualizing uncertainty lobes in vector field ensembles.
It follows a three-stage visualization pipeline:

1. Compute statistics (vector depths, spreads, angular ranges).
2. Build mesh geometry (wedge vertices, triangles).
3. Render the visualization with Matplotlib.

Main functions:

- ``uncertainty_lobes``: high-level API for complete visualization.
- ``uncertainty_lobes_summary_statistics``: statistics computation.
- ``uncertainty_lobes_mesh``: mesh generation.
- ``visualize_uncertainty_lobes``: rendering.
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
