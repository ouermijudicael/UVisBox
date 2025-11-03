"""
Top-level API for squid glyph visualization.

This module provides high-level functions that orchestrate the three-stage pipeline:
    1. Compute statistics (vector depths, spreads, variations)
    2. Build mesh geometry (vertices, triangles)
    3. Render visualization (matplotlib or pyvista)

For fine-grained control, use the individual stage functions directly:
    - squid_glyphs_2d_summary_statistics() / squid_glyphs_3d_summary_statistics()
    - squid_glyphs_2d_mesh() / squid_glyphs_3d_mesh()
    - visualize_squid_glyphs_2d() / visualize_squid_glyphs_3d()
"""

from .squid_glyphs_stats import squid_glyphs_2d_summary_statistics, squid_glyphs_3d_summary_statistics
from .squid_glyphs_mesh import squid_glyphs_2d_mesh, squid_glyphs_3d_mesh
from .squid_glyphs_vis import visualize_squid_glyphs_2d, visualize_squid_glyphs_3d
import numpy as np


def squid_glyph_3D(positions, ensemble_vectors, point_values=None, percentile=95, scale=0.5,
                   show_edges=True, glyph_color='lightblue', ax=None):
    """
    Visualize 3D uncertainty squid glyphs.
    
    This function orchestrates the three-stage pipeline:
        1. Convert Cartesian → Spherical, compute vector depths, spreads, and PCA
        2. Build superelliptical glyph mesh (vertices, triangles)
        3. Render with pyvista
    
    Based on: T. A. J. Ouermi et al., "Glyph-Based Uncertainty Visualization
    and Analysis of Time-Varying Vector Fields," IEEE UncertaintyVis 2024.

    Parameters:
    -----------
    positions : numpy.ndarray
        Array of shape (n, 3) The positions of the squid glyphs.
    ensemble_vectors : numpy.ndarray
        Array of shape (n, m, 3) The ensemble vectors for each position in Cartesian coordinates.
    point_values : numpy.ndarray, optional
        Array of shape (n,) The values associated with each position for coloring.
    percentile : float (0-100)
        Percentile of ensemble members to include based on depth ranking.
        Higher values include more vectors (larger glyphs showing more variation).
        - percentile=50: Include top 50% deepest vectors
        - percentile=95: Include top 95% deepest vectors (typical, default)
        - percentile=100: Include ALL vectors (maximum variation)
    scale : float, optional
        The scale factor for the glyphs. Default is 0.5.
    show_edges : bool, optional
        Whether to show edges of the glyphs. Default is True.
    glyph_color : str, optional
        The color of the glyphs. Default is 'lightblue'.
    ax : pyvista.Plotter, optional
        The pyvista plotter to use. If None, a new plotter will be created.

    Returns:
    --------
    plotter : pyvista.Plotter
        The pyvista plotter with the drawn squid glyphs.
    points : numpy.ndarray
        The points of the squid glyphs.
    polygons : numpy.ndarray
        The polygon connectivity for the glyphs.

    """
    # Stage 1: Compute statistics
    stats = squid_glyphs_3d_summary_statistics(ensemble_vectors, percentile)
    
    # Stage 2: Build mesh
    mesh = squid_glyphs_3d_mesh(positions, stats, point_values, scale, resolution=10)
    
    # Stage 3: Render
    plotter = visualize_squid_glyphs_3d(mesh, point_values, show_edges, glyph_color, ax=ax)
    
    return plotter, mesh['points'], mesh['polygons']


def squid_glyph_2D(positions, ensemble_vectors, percentile=95, scale=0.2, ax=None, workers=None):
    """
    Visualize 2D uncertainty squid glyphs.
    
    This function orchestrates the three-stage pipeline:
        1. Convert Cartesian → Polar, compute vector depths and spreads
        2. Build glyph mesh (vertices, triangles)
        3. Render with matplotlib
    
    Based on: T. A. J. Ouermi et al., "Glyph-Based Uncertainty Visualization
    and Analysis of Time-Varying Vector Fields," IEEE UncertaintyVis 2024.

    Parameters:
    -----------
    positions : numpy.ndarray
        Array of shape (n, 2) representing the positions of the squid glyphs.
    ensemble_vectors : numpy.ndarray
        Array of shape (n, m, 2) representing the ensemble vectors for each position.
    percentile : float (0-100)
        Percentile of ensemble members to include based on depth ranking.
        Higher values include more vectors (larger glyphs showing more variation).
        - percentile=50: Include top 50% deepest vectors
        - percentile=95: Include top 95% deepest vectors (typical, default)
        - percentile=100: Include ALL vectors (maximum variation)
    scale : float
        The scale factor for the glyphs.
    ax : matplotlib axis
        The axis to draw on. If None, a new figure and axis will be created.
    workers : int, optional
        Number of parallel workers for computation. Default is None (sequential).

    Returns:
    --------
    ax : matplotlib axis
        The axis with the drawn squid glyphs.

    """
        # 1. Compute vector depth statistics
    stats = squid_glyphs_2d_summary_statistics(ensemble_vectors, percentile, workers=workers)
    
    # Stage 2: Build mesh
    mesh = squid_glyphs_2d_mesh(positions, stats, scale)
    
    # Stage 3: Render
    ax = visualize_squid_glyphs_2d(mesh, ax)
    
    return ax
