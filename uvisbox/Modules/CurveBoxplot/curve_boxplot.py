from .curve_boxplot_stats import curve_boxplot_summary_statistics
from .curve_boxplot_mesh import curve_boxplot_mesh
from .curve_boxplot_vis import visualize_curve_boxplot
from uvisbox.Core.CommonInterface import BoxplotStyleConfig
import numpy as np
import matplotlib.pyplot as plt

def curve_boxplot(curves, boxplot_style=None, ax=None, workers=12):
    """
    Create a curve band depth plot with multiple percentile bands.
    
    This function follows a 3-stage pipeline: statistics → mesh → visualization.
    It computes curve band depths, generates triangular meshes for percentile bands,
    and creates a matplotlib visualization with median and outlier curves.

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
    
    # Validate input data
    if not isinstance(curves, np.ndarray):
        curves = np.array(curves)
    if curves.ndim != 3:
        raise ValueError(f"Input curves must be a 3D array of shape (n_curves, n_steps, n_dims). Got {curves.ndim}D array.")
    
    # Pipeline: Stats -> Mesh -> Vis
    # Step 1: Compute summary statistics
    stats = curve_boxplot_summary_statistics(curves, boxplot_style=boxplot_style, workers=workers)
    
    # Step 2: Build triangular mesh
    mesh_data = curve_boxplot_mesh(stats)
    
    # Step 3: Visualize
    ax = visualize_curve_boxplot(mesh_data, boxplot_style=boxplot_style, ax=ax)
    
    return ax
