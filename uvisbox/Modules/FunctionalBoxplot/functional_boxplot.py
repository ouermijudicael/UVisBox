import numpy as np
import matplotlib.pyplot as plt
from .functional_boxplot_stats import band_depths
from .functional_boxplot_mesh import get_band
from .functional_boxplot_vis import plot_band
from uvisbox.Core.CommonInterface import BoxplotStyleConfig


def functional_boxplot(data, method='fdb', boxplot_style=None, ax=None):
    """
    Create a functional band depth boxplot with multiple percentile bands.
    
    This function computes functional band depths, plots bands in descending percentile 
    order (largest to smallest for proper layering), and highlights the median curve.
    
    Parameters:
    -----------
    data : np.ndarray
        2D array of shape (N, D) where N is the number of curves and D is the number 
        of points per curve.
    method : str, optional
        Method for computing band depth. Options are:
        - 'fdb': functional band depth (default)
        - 'mfdb': modified functional band depth
    boxplot_style : BoxplotStyleConfig, optional
        Configuration for the boxplot visualization including percentiles, colormap,
        and median/outlier styling. If None, uses default configuration.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes to plot on. If None, creates a new figure.
    
    Returns:
    --------
    ax : matplotlib.axes.Axes
        The matplotlib axes object with the plot.
    
    Raises:
    -------
    ValueError
        If data is not 2D or if method is invalid.
    
    Notes:
    ------
    - Input data is not modified (computation happens on a copy)
    - Bands are plotted from largest to smallest percentile for proper visual layering
    - The median curve is the curve with the highest band depth value
    - Outliers are curves beyond the largest percentile
    - Curve depths are always computed internally
    
    Examples:
    ---------
    >>> import numpy as np
    >>> from uvisbox.Modules.FunctionalBoxplot import functional_boxplot
    >>> from uvisbox.Core.CommonInterface import BoxplotStyleConfig
    >>> 
    >>> # Generate synthetic functional data
    >>> t = np.linspace(0, 1, 100)
    >>> data = np.array([np.sin(2*np.pi*t) + 0.2*np.random.randn(100) for _ in range(50)])
    >>> 
    >>> # Basic usage with default settings
    >>> ax = functional_boxplot(data)
    >>> 
    >>> # Custom styling
    >>> style = BoxplotStyleConfig(
    ...     percentiles=[10, 50, 90],
    ...     percentile_colormap='plasma',
    ...     show_median=True,
    ...     show_outliers=True
    ... )
    >>> ax = functional_boxplot(data, boxplot_style=style)
    >>> 
    >>> # Plot on existing axes
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(figsize=(12, 6))
    >>> functional_boxplot(data, ax=ax)
    >>> plt.show()
    """
    # Use default config if none provided
    if boxplot_style is None:
        boxplot_style = BoxplotStyleConfig()
    
    # Validate and copy input data
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if data.ndim != 2:
        raise ValueError(f"Input data must be a 2D array of shape (N, D). Got {data.ndim}D array.")
    
    # Work on a copy to avoid modifying input data
    data_copy = data.copy()
    
    # Compute band depths
    if method == 'fdb' or method == 'mfbd':
        depths = band_depths(data_copy, method=method)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'fdb' or 'mfbd'.")

    # Sort curves by depth (descending order - highest depth first)
    sorted_indices = np.argsort(depths)[::-1]
    sorted_data = data_copy[sorted_indices]
    
    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get colors from colormap
    colors = boxplot_style.get_percentile_colors()
    percentiles = boxplot_style.percentiles
    
    # Sort percentiles in descending order for proper plotting (largest first)
    sorted_percentile_indices = np.argsort(percentiles)[::-1]
    sorted_percentiles = [percentiles[i] for i in sorted_percentile_indices]
    sorted_colors = [colors[i] for i in sorted_percentile_indices]
    
    # Plot each percentile band from largest to smallest
    for percentile, color in zip(sorted_percentiles, sorted_colors):
        bottom, top = get_band(data_copy, percentile, method=method)
        plot_band(bottom, top, ax=ax, color=color, alpha=1.0)
    
    # Setup x-axis for curve plotting
    n_points = data_copy.shape[1]
    x = np.linspace(0, 1, n_points)
    
    # Plot outliers (curves beyond the largest percentile)
    if boxplot_style.show_outliers:
        largest_percentile = max(percentiles)
        outlier_start_idx = int(np.ceil(len(sorted_data) * largest_percentile / 100))
        
        for idx in range(outlier_start_idx, len(sorted_data)):
            outlier_curve = sorted_data[idx]
            # Add label only for the first outlier to avoid duplicate legend entries
            label = 'Outliers' if idx == outlier_start_idx else None
            ax.plot(x, outlier_curve, 
                   color=boxplot_style.outliers_color, 
                   linewidth=boxplot_style.outliers_width, 
                   alpha=boxplot_style.outliers_alpha,
                   label=label,
                   zorder=5)
    
    # Plot the median curve (curve with maximum depth) if requested
    if boxplot_style.show_median:
        median_curve = sorted_data[0]
        ax.plot(x, median_curve, 
               color=boxplot_style.median_color, 
               linewidth=boxplot_style.median_width,
               alpha=boxplot_style.median_alpha, 
               label='Median Curve', zorder=10)
    
    # Add labels and legend
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax