"""
Top-level API for uncertainty lobe visualization.

This module provides high-level functions that orchestrate the three-stage pipeline:
    1. Compute statistics (vector depths, spreads, angular ranges)
    2. Build mesh geometry (wedge vertices, triangles)
    3. Render visualization (matplotlib)

For fine-grained control, use the individual stage functions directly:
    - uncertainty_lobes_summary_statistics()
    - uncertainty_lobes_mesh()
    - visualize_uncertainty_lobes()

Based on: M. Jarema, I. Demir, J. Kehrer and R. Westermann, "Comparative visual 
analysis of vector field ensembles," 2015 IEEE VAST, doi: 10.1109/VAST.2015.7347634.

This implementation uses vector depth and doesn't fit ensemble to a Gaussian Mixture 
Model as in the original paper. In addition, this implementation doesn't perform 
clustering of the vectors, instead it draws lobes for all vectors at each position.
"""

from .uncertainty_lobes_stats import uncertainty_lobes_summary_statistics
from .uncertainty_lobes_mesh import uncertainty_lobes_mesh
from .uncertainty_lobes_vis import visualize_uncertainty_lobes
import matplotlib.pyplot as plt


def uncertainty_lobes(positions, ensemble_vectors, percentile1=90, percentile2=50, 
                      scale=0.2, ax=None, show_median=True, workers=None):
    """
    Visualize 2D uncertainty lobe glyphs.
    
    This function orchestrates the three-stage pipeline:
        1. Convert Cartesian → Polar, compute vector depths and spreads
        2. Build wedge-shaped mesh (vertices, triangles)
        3. Render with matplotlib
    
    Draws uncertainty lobe glyphs representing directional uncertainty with dual-lobe 
    visualization. Each glyph shows:
    - Outer lobe (percentile1, semi-transparent): larger angular and magnitude spread
    - Inner lobe (percentile2, opaque): smaller spread showing most probable region
    - Median arrow: most probable direction based on vector depth
    
    Parameters:
    -----------
    positions : numpy.ndarray
        Array of shape (n, 2) representing the positions of the lobe glyphs.
    ensemble_vectors : numpy.ndarray
        Array of shape (n, m, 2) representing the ensemble vectors for each position.
    percentile1 : float (0-100)
        The first percentile for depth filtering (outer lobe). Range 0-100. 
        Higher values include more vectors (larger lobes).
        Default: 90
    percentile2 : float or None (0-100)
        The second percentile for depth filtering (inner lobe). Range 0-100.
        If None, only outer lobe is drawn.
        Default: 50
    scale : float
        The scale factor for the glyphs (default: 0.2).
    ax : matplotlib.Axes, optional
        The axis to draw on. If None, a new figure and axis will be created.
    show_median : bool
        Whether to show the median vector as an arrow (default: True).
    workers : int, optional
        Number of worker processes for parallel vector depth computation.
        If None or <= 1, uses sequential computation. For ensemble size >= 30,
        parallelization can provide significant speedup.
        Default: None

    Returns:
    --------
    ax : matplotlib.Axes
        The axis with the drawn lobe glyphs.
    """
    # Stage 1: Compute statistics
    stats = uncertainty_lobes_summary_statistics(
        ensemble_vectors, 
        percentile1=percentile1, 
        percentile2=percentile2,
        workers=workers
    )
    
    # Stage 2: Build mesh
    mesh = uncertainty_lobes_mesh(positions, stats, scale=scale)
    
    # Stage 3: Render
    ax = visualize_uncertainty_lobes(mesh, show_median=show_median, ax=ax)
    
    return ax