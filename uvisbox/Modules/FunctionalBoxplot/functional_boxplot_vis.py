import numpy as np
import matplotlib.pyplot as plt
from uvisbox.Core.CommonInterface import BoxplotStyleConfig


def plot_band(bottom_curve, top_curve, vmin=0, vmax=1, ax=None, color='red', alpha=1.0, scale=1.0):
    """
    Plot a functional band envelope between bottom and top curves.

    This function creates a filled area between the curves using matplotlib's fill_between.

    Parameters:
    -----------
    bottom_curve : np.ndarray
        1D array representing the bottom boundary of the band.
    top_curve : np.ndarray
        1D array representing the top boundary of the band.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes to plot on. If None, creates new figure with axes.
    color : str or tuple, optional
        Color for the band. Default is 'red'.
        Can be any matplotlib color specification (named color, hex, RGB tuple, etc.).
    alpha : float, optional
        Transparency of the band (0=transparent, 1=opaque). Default is 1.0.
    scale : float, optional
        Scale factor for the curves. Default is 1.0.

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The matplotlib axes object used for plotting.

    Raises:
    -------
    ValueError
        If bottom_curve and top_curve have different shapes.

    Examples:
    ---------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from uvisbox.Modules.FunctionalBoxplot import plot_band
    >>> from uvisbox.Modules.FunctionalBoxplot.functional_boxplot_stats import functional_boxplot_get_band
    >>> 
    >>> # Generate synthetic data
    >>> data = np.random.randn(100, 50).cumsum(axis=1)
    >>> 
    >>> # Get 50th percentile band
    >>> bottom, top = functional_boxplot_get_band(data, 50, method='fbd')
    >>> 
    >>> # Plot the band
    >>> fig, ax = plt.subplots()
    >>> plot_band(bottom, top, ax=ax, color='blue', alpha=0.5)
    >>> plt.show()
    """
    # Validate inputs
    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = 1
    if not isinstance(bottom_curve, np.ndarray):
        bottom_curve = np.array(bottom_curve)
    if not isinstance(top_curve, np.ndarray):
        top_curve = np.array(top_curve)

    if bottom_curve.shape != top_curve.shape:
        raise ValueError(f"bottom_curve and top_curve must have the same shape. "
                         f"Got {bottom_curve.shape} and {top_curve.shape}")

    if bottom_curve.ndim != 1:
        raise ValueError(f"Curves must be 1D arrays. Got {bottom_curve.ndim}D")

    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots()

    # Apply scale
    bottom_scaled = bottom_curve * scale
    top_scaled = top_curve * scale

    # Create x-coordinates (normalized to [0, 1])
    n_points = len(bottom_curve)
    x = np.linspace(vmin, vmax, n_points)

    # Use fill_between to create the band
    ax.fill_between(x, bottom_scaled, top_scaled,
                    color=color, alpha=alpha, edgecolor='none')

    return ax


def visualize_functional_boxplot(mesh_data, vmin=0, vmax=1, boxplot_style=None, ax=None):
    """
    Visualize functional boxplot from mesh data (summary statistics).

    This function creates a matplotlib visualization of the functional boxplot using
    the summary statistics output from the stats/mesh pipeline.

    Parameters:
    -----------
    mesh_data : dict
        Dictionary containing summary statistics with the following keys:
        - 'percentile_bands': dict of percentile bands
        - 'median': median curve
        - 'outliers': outlier curves
        - 'sorted_curves': all curves sorted by depth
    boxplot_style : BoxplotStyleConfig, optional
        Configuration for the boxplot visualization including percentiles, colormap,
        and median/outlier styling. If None, uses default configuration.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes to plot on. If None, creates a new figure.

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The matplotlib axes object with the plot.

    Examples:
    ---------
    >>> import numpy as np
    >>> from uvisbox.Modules.FunctionalBoxplot.functional_boxplot_stats import functional_boxplot_summary_statistics
    >>> from uvisbox.Modules.FunctionalBoxplot.functional_boxplot_mesh import functional_boxplot_mesh
    >>> from uvisbox.Modules.FunctionalBoxplot.functional_boxplot_vis import visualize_functional_boxplot
    >>> from uvisbox.Core.CommonInterface import BoxplotStyleConfig
    >>> 
    >>> # Generate synthetic functional data
    >>> t = np.linspace(0, 1, 100)
    >>> data = np.array([np.sin(2*np.pi*t) + 0.2*np.random.randn(100) for _ in range(50)])
    >>> 
    >>> # Process through pipeline
    >>> stats = functional_boxplot_summary_statistics(data)
    >>> mesh_data = functional_boxplot_mesh(stats)
    >>> 
    >>> # Visualize
    >>> ax = visualize_functional_boxplot(mesh_data)
    >>> 
    >>> # Custom styling
    >>> style = BoxplotStyleConfig(percentiles=[25, 50, 75, 90], show_outliers=True)
    >>> ax = visualize_functional_boxplot(mesh_data, boxplot_style=style)
    """
    # Use default config if none provided
    if boxplot_style is None:
        boxplot_style = BoxplotStyleConfig()

    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    try:
        vmin = float(vmin)
    except (ValueError, TypeError):
        vmin = float(0)
    try:
        vmax = float(vmax)
    except (ValueError, TypeError):
        vmax = float(1)

    # Get colors from colormap
    colors = boxplot_style.get_percentile_colors()
    percentiles = boxplot_style.percentiles

    # Sort percentiles in descending order for proper plotting (largest first)
    sorted_percentile_indices = np.argsort(percentiles)[::-1]
    sorted_percentiles = [percentiles[i] for i in sorted_percentile_indices]
    sorted_colors = [colors[i] for i in sorted_percentile_indices]

    # Plot each percentile band from largest to smallest
    for percentile, color in zip(sorted_percentiles, sorted_colors):
        band_key = f'{int(percentile)}_percentile_band'
        if band_key in mesh_data['percentile_bands']:
            bottom, top = mesh_data['percentile_bands'][band_key]
            plot_band(bottom, top, vmin=vmin, vmax=vmax, ax=ax, color=color, alpha=1.0)

    # Setup x-axis for curve plotting
    n_points = mesh_data['median'].shape[0]
    x = np.linspace(vmin, vmax, n_points)

    # Plot outliers if requested and available
    if boxplot_style.show_outliers and mesh_data['outliers'].shape[0] > 0:
        outliers = mesh_data['outliers']
        for idx in range(len(outliers)):
            outlier_curve = outliers[idx]
            # Add label only for the first outlier to avoid duplicate legend entries
            label = 'Outliers' if idx == 0 else None
            ax.plot(x, outlier_curve,
                    color=boxplot_style.outliers_color,
                    linewidth=boxplot_style.outliers_width,
                    alpha=boxplot_style.outliers_alpha,
                    label=label,
                    zorder=5)

    # Plot the median curve if requested
    if boxplot_style.show_median:
        median_curve = mesh_data['median']
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
