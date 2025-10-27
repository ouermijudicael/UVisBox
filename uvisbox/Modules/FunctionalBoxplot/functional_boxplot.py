import numpy as np
import matplotlib.pyplot as plt
from .functional_boxplot_stats import band_depths
from .functional_boxplot_mesh import get_band
from .functional_boxplot_vis import plot_band


def functional_boxplot(data, method='fdb', percentiles=[25, 50, 90, 100], ax=None, 
                      colors=None, median_color='red', alpha=1, plot_all_curves=False):
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
    percentiles : list of float, optional
        List of percentiles for the bands to be plotted (default is [25, 50, 90, 100]).
        Bands are plotted in descending order so smaller bands appear on top.
        Values should be between 0 and 100.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes to plot on. If None, creates a new figure.
    colors : list of str or tuples, optional
        List of colors for each percentile band. If None, uses a default color scheme.
        Must have the same length as percentiles if provided.
    median_color : str or tuple, optional
        Color for the median curve (default is 'red').
    alpha : float, optional
        Transparency level for the bands (0=transparent, 1=opaque). Default is 0.7.
    
    Returns:
    --------
    ax : matplotlib.axes.Axes
        The matplotlib axes object with the plot.
    
    Raises:
    -------
    ValueError
        If data is not 2D, if method is invalid, or if colors length doesn't match percentiles.
    
    Notes:
    ------
    - Input data is not modified (computation happens on a copy)
    - Bands are plotted from largest to smallest percentile for proper visual layering
    - The median curve is the curve with the highest band depth value
    - All curves are plotted with light gray transparency for context
    
    Examples:
    ---------
    >>> import numpy as np
    >>> from uvisbox.Modules.FunctionalBoxplot import functional_boxplot
    >>> 
    >>> # Generate synthetic functional data
    >>> t = np.linspace(0, 1, 100)
    >>> data = np.array([np.sin(2*np.pi*t) + 0.2*np.random.randn(100) for _ in range(50)])
    >>> 
    >>> # Basic usage with default settings
    >>> ax = functional_boxplot(data)
    >>> 
    >>> # Custom percentiles and colors
    >>> ax = functional_boxplot(data, percentiles=[10, 50, 90], 
    ...                         colors=['lightblue', 'blue', 'darkblue'],
    ...                         method='fdb')
    >>> 
    >>> # Plot on existing axes
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(figsize=(12, 6))
    >>> functional_boxplot(data, ax=ax, median_color='black', alpha=0.5)
    >>> plt.show()
    """
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
        bottom, top = get_band(data_copy, percentile, method=method)
        plot_band(bottom, top, ax=ax, color=color, alpha=alpha)
    
    # Plot all curves in light gray for context
    n_points = data_copy.shape[1]
    x = np.linspace(0, 1, n_points)
    
    if plot_all_curves:
        for curve in data_copy:
            ax.plot(x, curve, color='gray', alpha=0.1, linewidth=0.5, zorder=1)
    
    # Plot the median curve (curve with maximum depth)
    median_curve = sorted_data[0]
    ax.plot(x, median_curve, color=median_color, linewidth=2.5, 
           label='Median Curve', zorder=10)
    
    # Add labels and legend
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax