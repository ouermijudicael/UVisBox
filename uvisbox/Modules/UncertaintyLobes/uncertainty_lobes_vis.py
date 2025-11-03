"""
Visualization rendering for uncertainty lobes.

This module renders uncertainty lobe meshes using matplotlib:
    - Draw outer lobes (wedges)
    - Draw inner lobes (wedges)
    - Draw median direction arrows
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def render_uncertainty_lobes_2d(mesh_2d, show_median=True, ax=None):
    """
    Render 2D uncertainty lobe mesh with matplotlib.
    
    Draws wedge-shaped glyphs representing directional uncertainty with:
    - Outer wedges (light blue, semi-transparent) - larger angular spread
    - Inner wedges (light blue, opaque) - smaller angular spread
    - Median arrows (blue) - showing the most probable direction
    
    Parameters:
    -----------
    mesh_2d : dict
        From build_uncertainty_lobes_mesh_2d() containing:
        - 'wedges': outer lobe wedge data
        - 'inner_wedges': inner lobe wedge data (optional)
        - 'arrows': median direction arrow data
    show_median : bool
        Whether to show median direction arrows (default: True)
    ax : matplotlib.Axes, optional
        Existing axis to draw on. If None, creates new figure and axis.
    
    Returns:
    --------
    ax : matplotlib.Axes
        The axis with drawn lobe glyphs
    """
    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Render outer wedges (percentile1)
    for wedge_data in mesh_2d['wedges']:
        vertices = wedge_data['vertices']
        
        # Create polygon patch for the wedge
        wedge_polygon = Polygon(
            vertices, 
            closed=True, 
            facecolor='skyblue', 
            edgecolor='skyblue', 
            alpha=0.3
        )
        ax.add_patch(wedge_polygon)
    
    # Render inner wedges (percentile2) if present
    if mesh_2d['inner_wedges'] is not None:
        for wedge_data in mesh_2d['inner_wedges']:
            vertices = wedge_data['vertices']
            
            wedge_polygon = Polygon(
                vertices, 
                closed=True, 
                facecolor='skyblue', 
                edgecolor='skyblue', 
                alpha=1.0
            )
            ax.add_patch(wedge_polygon)
    
    # Render median arrows if requested
    if show_median:
        arrows = mesh_2d['arrows']
        positions = arrows['positions']
        directions = arrows['directions']
        lengths = arrows['lengths']
        
        for i in range(positions.shape[0]):
            center = positions[i]
            direction = directions[i]
            length = lengths[i]
            
            # Arrow endpoint
            arrow_end = center + direction * length
            
            ax.annotate(
                '',
                xy=arrow_end,
                xytext=center,
                arrowprops=dict(
                    facecolor='blue', 
                    edgecolor='blue', 
                    arrowstyle='->', 
                    lw=3, 
                    mutation_scale=20, 
                    alpha=1.0
                )
            )
    
    return ax