from .curve_boxplot_stats import curve_banddepths
from .curve_boxplot_mesh import curves_band_mesh
from .curve_boxplot_vis import matplotlib_curve_boxplot_vis, matplotlib_plot_band
from uvisbox.Core.CommonInterface import BoxplotStyleConfig
import numpy as np
import matplotlib.pyplot as plt

def curve_boxplot(curves, boxplot_style=None, ax=None, workers=12):
    """
    Create a curve band depth plot with multiple percentile bands.

    Parameters:
    -----------
    curves : numpy.ndarray
        3D array of shape (n_curves, n_steps, n_dims) containing curve data.
        Input data is not modified (computation happens on a copy).
    boxplot_style : BoxplotStyleConfig, optional
        Configuration for the boxplot visualization including percentiles, colors,
        and median/outlier styling. If None, uses default configuration.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, creates a new figure.
        For 3D curves, must be a 3D axes if provided.
    workers : int, optional
        Number of worker processes for parallel computation of band depths. Default is 12.
        Set to 1 or None to use sequential processing (useful for debugging).

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axes with the plot.

    Notes:
    ------
    - The function does not modify the input curves array
    - Bands are plotted from largest to smallest percentile for proper layering
    - The median curve is the curve with the highest depth value
    - Outliers are curves beyond the largest percentile
    - Curve depths are always computed internally
    
    Examples:
    ---------
    >>> # Basic usage with defaults
    >>> ax = curve_boxplot(curves)
    
    >>> # Custom styling
    >>> from uvisbox.Core.CommonInterface import BoxplotStyleConfig
    >>> style = BoxplotStyleConfig(
    ...     percentiles=[10, 50, 90],
    ...     percentile_colors=['lightblue', 'blue', 'darkblue'],
    ...     show_median=True,
    ...     median_color='red',
    ...     show_outliers=True
    ... )
    >>> ax = curve_boxplot(curves, boxplot_style=style)
    
    >>> # Hide median and outliers
    >>> style = BoxplotStyleConfig(show_median=False, show_outliers=False)
    >>> ax = curve_boxplot(curves, boxplot_style=style)
    """
    
    # Use default config if none provided
    if boxplot_style is None:
        boxplot_style = BoxplotStyleConfig()
    
    # Work on a copy to avoid modifying input data
    curves_copy = curves.copy()
    
    # Always compute depths internally
    curve_depths = curve_banddepths(curves_copy, workers=workers)

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
    
    # Get colors from colormap
    colors = boxplot_style.get_percentile_colors()
    percentiles = boxplot_style.percentiles
    
    # Sort percentiles in descending order for proper plotting (largest first)
    sorted_percentile_indices = np.argsort(percentiles)[::-1]
    sorted_percentiles = [percentiles[i] for i in sorted_percentile_indices]
    sorted_colors = [colors[i] for i in sorted_percentile_indices]
    
    # Plot each percentile band from largest to smallest
    for percentile, color in zip(sorted_percentiles, sorted_colors):
        points, triangles = curves_band_mesh(sorted_curves, percentile=percentile)
        ax = matplotlib_plot_band(points, triangles, ax=ax, color=color, alpha=1.0)
    
    # Plot outliers (curves beyond the largest percentile)
    if boxplot_style.show_outliers:
        largest_percentile = max(percentiles)
        outlier_start_idx = int(np.ceil(len(sorted_curves) * largest_percentile / 100))
        
        for idx in range(outlier_start_idx, len(sorted_curves)):
            outlier_curve = sorted_curves[idx]
            # Add label only for the first outlier to avoid duplicate legend entries
            label = 'Outliers' if idx == outlier_start_idx else None
            if curve_dim == 2:
                ax.plot(outlier_curve[:, 0], outlier_curve[:, 1], 
                       color=boxplot_style.outliers_color, 
                       linewidth=boxplot_style.outliers_width, 
                       alpha=boxplot_style.outliers_alpha,
                       label=label,
                       zorder=5)
            elif curve_dim == 3:
                ax.plot(outlier_curve[:, 0], outlier_curve[:, 1], outlier_curve[:, 2],
                       color=boxplot_style.outliers_color, 
                       linewidth=boxplot_style.outliers_width, 
                       alpha=boxplot_style.outliers_alpha,
                       label=label,
                       zorder=5)
    
    # Plot the median curve (curve with maximum depth)
    if boxplot_style.show_median:
        median_curve = sorted_curves[0]
        if curve_dim == 2:
            ax.plot(median_curve[:, 0], median_curve[:, 1], 
                   color=boxplot_style.median_color, 
                   linewidth=boxplot_style.median_width, 
                   alpha=boxplot_style.median_alpha, 
                   label='Median Curve', zorder=10)
        elif curve_dim == 3:
            ax.plot(median_curve[:, 0], median_curve[:, 1], median_curve[:, 2],
                   color=boxplot_style.median_color, 
                   linewidth=boxplot_style.median_width, 
                   alpha=boxplot_style.median_alpha, 
                   label='Median Curve', zorder=10)
    
    return ax