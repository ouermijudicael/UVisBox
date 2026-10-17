"""
Visualization rendering for uncertainty lobes.

This module renders uncertainty lobe meshes using matplotlib:
    - Draw outer lobes (wedges)
    - Draw inner lobes (wedges)
    - Draw median direction arrows
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Polygon


def _validate_alpha(name, value):
    """Validate an opacity value and return it as a float."""
    if not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a number between 0 and 1.")
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1.")
    return float(value)


def visualize_uncertainty_lobes(
    mesh,
    show_median=True,
    ax=None,
    outer_lobe_edge_color='skyblue',
    outer_lobe_face_color='skyblue',
    inner_lobe_edge_color='skyblue',
    inner_lobe_face_color='skyblue',
    median_arrow_color='blue',
    outer_lobe_alpha=0.3,
    inner_lobe_alpha=1.0,
    median_arrow_alpha=1.0,
):
    """
    Render uncertainty lobe mesh with matplotlib.
    
    Draws wedge-shaped glyphs representing directional uncertainty with:
    - Outer wedges (light blue, semi-transparent) - larger angular spread
    - Inner wedges (light blue, opaque) - smaller angular spread
    - Median arrows (blue) - showing the most probable direction
    
    Parameters
    -----------
    mesh : dict
        From uncertainty_lobes_mesh() containing:
        - 'wedges': outer lobe wedge data
        - 'inner_wedges': inner lobe wedge data (optional)
        - 'arrows': median direction arrow data
    show_median : bool
        Whether to show median direction arrows (default: True)
    ax : matplotlib.Axes, optional
        Existing axis to draw on. If None, creates new figure and axis.
    outer_lobe_edge_color : color, optional
        Matplotlib color for outer lobe edges. Default is ``'skyblue'``.
    outer_lobe_face_color : color, optional
        Matplotlib color for outer lobe faces. Default is ``'skyblue'``.
    inner_lobe_edge_color : color, optional
        Matplotlib color for inner lobe edges. Default is ``'skyblue'``.
    inner_lobe_face_color : color, optional
        Matplotlib color for inner lobe faces. Default is ``'skyblue'``.
    median_arrow_color : color, optional
        Matplotlib color for median direction arrows. Default is ``'blue'``.
    outer_lobe_alpha : float, optional
        Opacity for outer lobe wedges, including their edges. Must be in
        ``[0, 1]``. Default is ``0.3``.
    inner_lobe_alpha : float, optional
        Opacity for inner lobe wedges, including their edges. Must be in
        ``[0, 1]``. Default is ``1.0``.
    median_arrow_alpha : float, optional
        Opacity for median direction arrows. Must be in ``[0, 1]``. Default
        is ``1.0``.
    
    Returns
    --------
    ax : matplotlib.Axes
        The axis with drawn lobe glyphs
    """
    outer_lobe_alpha = _validate_alpha('outer_lobe_alpha', outer_lobe_alpha)
    inner_lobe_alpha = _validate_alpha('inner_lobe_alpha', inner_lobe_alpha)
    median_arrow_alpha = _validate_alpha('median_arrow_alpha', median_arrow_alpha)

    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Render outer wedges (percentile1)
    for wedge_data in mesh['wedges']:
        vertices = wedge_data['vertices']
        
        # Create polygon patch for the wedge
        wedge_polygon = Polygon(
            vertices, 
            closed=True, 
            facecolor=mcolors.to_rgba(outer_lobe_face_color, outer_lobe_alpha),
            edgecolor=mcolors.to_rgba(outer_lobe_edge_color, outer_lobe_alpha),
        )
        ax.add_patch(wedge_polygon)
    
    # Render inner wedges (percentile2) if present
    if mesh['inner_wedges'] is not None:
        for wedge_data in mesh['inner_wedges']:
            vertices = wedge_data['vertices']
            
            wedge_polygon = Polygon(
                vertices, 
                closed=True, 
                facecolor=mcolors.to_rgba(inner_lobe_face_color, inner_lobe_alpha),
                edgecolor=mcolors.to_rgba(inner_lobe_edge_color, inner_lobe_alpha),
            )
            ax.add_patch(wedge_polygon)
    
    # Render median arrows if requested
    if show_median:
        arrows = mesh['arrows']
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
                    facecolor=median_arrow_color,
                    edgecolor=median_arrow_color,
                    arrowstyle='->', 
                    lw=2, 
                    mutation_scale=10, 
                    alpha=median_arrow_alpha
                )
            )
    
    return ax
