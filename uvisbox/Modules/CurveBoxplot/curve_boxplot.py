from .curve_boxplot_stats import curve_banddepths
from .curve_boxplot_mesh import curves_band_mesh
from .curve_boxplot_vis import matplotlib_curve_boxplot_vis, matplotlib_plot_band
import numpy as np
import matplotlib.pyplot as plt

def curve_boxplot(curves, curve_depths=None, percentiles=[25, 50, 90, 100], ax=None, 
                 colors=None, median_color='red', alpha=0.7):
    """
    Create a curve band depth plot with multiple percentile bands.

    Parameters:
    -----------
    curves : numpy.ndarray
        3D array of shape (n_curves, n_steps, n_dims) containing curve data.
        Input data is not modified (computation happens on a copy).
    curve_depths : numpy.ndarray, optional
        1D array of shape (n_curves,) containing the depth of each curve.
        If None, depths will be computed automatically.
    percentiles : list of float, optional
        List of percentiles for the bands to be plotted (default is [25, 50, 90, 100]).
        Bands are plotted in descending order so smaller bands appear on top.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, creates a new figure.
        For 3D curves, must be a 3D axes if provided.
    colors : list of str or tuples, optional
        List of colors for each percentile band. If None, uses a default color scheme.
        Must have the same length as percentiles if provided.
    median_color : str, optional
        The color to use for the median curve (default is 'red').
    alpha : float, optional
        The transparency level for the mesh bands (default is 0.7).

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axes with the plot.

    Notes:
    ------
    - The function does not modify the input curves array
    - Bands are plotted from largest to smallest percentile for proper layering
    - The median curve is the curve with the highest depth value
    
    Examples:
    ---------
    >>> # Basic usage with default percentiles
    >>> ax = curve_boxplot(curves)
    
    >>> # Custom percentiles and colors
    >>> ax = curve_boxplot(curves, percentiles=[10, 50, 90], 
    ...                    colors=['lightblue', 'blue', 'darkblue'])
    
    >>> # Plot on existing axes
    >>> fig, ax = plt.subplots()
    >>> curve_boxplot(curves, ax=ax, median_color='black')
    """
    
    # Work on a copy to avoid modifying input data
    curves_copy = curves.copy()
    
    # Compute depths if not provided
    if curve_depths is None:
        curve_depths = curve_banddepths(curves_copy)
    else:
        # Also copy the depths to avoid modifying input
        curve_depths = curve_depths.copy()

    # Sort the curves by depth from deepest to shallowest
    sorted_indices = np.argsort(curve_depths)[::-1]
    sorted_curves = curves_copy[sorted_indices]
    
    # Determine curve dimensionality
    curve_dim = curves.shape[2]
    
    # Create figure/axes if not provided
    if ax is None:
        if curve_dim == 2:
            fig, ax = plt.subplots(figsize=(10, 8))
        elif curve_dim == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        else:
            raise ValueError(f"Unsupported curve dimension: {curve_dim}. Must be 2 or 3.")
    
    # Set default colors if not provided
    if colors is None:
        # Default color scheme: light to dark blue
        colors = ['#e0e0e0', '#a0c4e8', '#5a8dc4', '#2e5f8a']
        # Extend or truncate to match number of percentiles
        while len(colors) < len(percentiles):
            colors.append('#1a3a52')  # Add darker blue if needed
        colors = colors[:len(percentiles)]
    elif len(colors) != len(percentiles):
        raise ValueError(f"Length of colors ({len(colors)}) must match length of percentiles ({len(percentiles)})")
    
    # Sort percentiles in descending order for proper plotting (largest first)
    sorted_percentile_indices = np.argsort(percentiles)[::-1]
    sorted_percentiles = [percentiles[i] for i in sorted_percentile_indices]
    sorted_colors = [colors[i] for i in sorted_percentile_indices]
    
    # Plot each percentile band from largest to smallest
    for percentile, color in zip(sorted_percentiles, sorted_colors):
        points, triangles = curves_band_mesh(sorted_curves, percentile=percentile)
        ax = matplotlib_plot_band(points, triangles, ax=ax, color=color, alpha=alpha)
    
    # Plot the median curve (curve with maximum depth)
    median_curve = sorted_curves[0]
    if curve_dim == 2:
        ax.plot(median_curve[:, 0], median_curve[:, 1], 
               color=median_color, linewidth=2.5, label='Median Curve', zorder=10)
    elif curve_dim == 3:
        ax.plot(median_curve[:, 0], median_curve[:, 1], median_curve[:, 2],
               color=median_color, linewidth=2.5, label='Median Curve', zorder=10)
    
    return ax