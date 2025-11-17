from uvisbox.Core.BandDepths.curve_banddepth import curve_banddepths
import numpy as np
from uvisbox.Core.CommonInterface import BoxplotStyleConfig


def curve_boxplot_summary_statistics(curves, boxplot_style=None, workers=12):
    """
    Compute curve band depth summary statistics without visualization.
    
    This function computes curve band depths, sorts curves, and identifies percentile bands
    and outliers based on the configuration.
    
    Parameters:
    -----------
    curves : numpy.ndarray
        3D array of shape (n_curves, n_steps, n_dims) containing curve data.
        Input data is not modified (computation happens on a copy).
    boxplot_style : BoxplotStyleConfig, optional
        Configuration for the boxplot including percentiles and outlier settings.
        If None, uses default configuration.
    workers : int, optional
        Number of worker processes for parallel computation of band depths. Default is 12.
        Set to 1 or None to use sequential processing (useful for debugging).
    
    Returns:
    --------
    stats : dict
        Dictionary containing the following keys:
        - 'depths': np.ndarray of shape (n_curves,) - band depths for each curve
        - 'sorted_indices': np.ndarray of shape (n_curves,) - original indices sorted by depth (descending)
        - 'sorted_curves': np.ndarray of shape (n_curves, n_steps, n_dims) - curves sorted by depth (descending)
        - 'median_curve': np.ndarray of shape (n_steps, n_dims) - curve with highest depth
        - 'percentiles': list of floats - percentile values from boxplot_style
        - 'outliers': np.ndarray of shape (n_outliers, n_steps, n_dims) - outlier curves beyond largest percentile
        - 'n_dims': int - dimensionality of curves (2 or 3)
    
    Raises:
    -------
    ValueError
        If curves is not a 3D array or if n_dims is not 2 or 3.
    
    Examples:
    ---------
    >>> import numpy as np
    >>> from uvisbox.Modules.CurveBoxplot.curve_boxplot_stats import curve_boxplot_summary_statistics
    >>> from uvisbox.Core.CommonInterface import BoxplotStyleConfig
    >>> 
    >>> # Generate synthetic curve data (50 curves, 100 time steps, 2D)
    >>> curves = np.random.randn(50, 100, 2).cumsum(axis=1)
    >>> 
    >>> # Basic usage with default settings
    >>> stats = curve_boxplot_summary_statistics(curves)
    >>> print(f"Median curve shape: {stats['median_curve'].shape}")
    >>> print(f"Number of outliers: {stats['outliers'].shape[0]}")
    >>> 
    >>> # Custom percentiles
    >>> style = BoxplotStyleConfig(percentiles=[10, 50, 90], show_outliers=True)
    >>> stats = curve_boxplot_summary_statistics(curves, boxplot_style=style)
    """
    # Use default config if none provided
    if boxplot_style is None:
        boxplot_style = BoxplotStyleConfig()
    
    # Validate input
    if not isinstance(curves, np.ndarray):
        curves = np.array(curves)
    if curves.ndim != 3:
        raise ValueError(f"Input curves must be a 3D array of shape (n_curves, n_steps, n_dims). Got {curves.ndim}D array.")
    
    n_dims = curves.shape[2]
    if n_dims not in [2, 3]:
        raise ValueError(f"Curves must be 2D or 3D. Got {n_dims}D curves.")
    
    # Work on a copy to avoid modifying input data
    curves_copy = curves.copy()
    
    # Compute curve band depths
    depths = curve_banddepths(curves_copy, workers=workers)
    
    # Sort curves by depth (descending order - highest depth first)
    sorted_indices = np.argsort(depths)[::-1]
    sorted_curves = curves_copy[sorted_indices]
    sorted_depths = depths[sorted_indices]
    
    # Initialize results dictionary
    stats = {
        'depths': depths,
        'sorted_indices': sorted_indices,
        'sorted_curves': sorted_curves,
        'median_curve': sorted_curves[0],  # Curve with maximum depth
        'percentiles': boxplot_style.percentiles,
        'n_dims': n_dims
    }
    
    # Compute outliers if requested
    if boxplot_style.show_outliers and len(boxplot_style.percentiles) > 0:
        largest_percentile = max(boxplot_style.percentiles)
        outlier_start_idx = int(np.ceil(len(sorted_curves) * largest_percentile / 100))
        stats['outliers'] = sorted_curves[outlier_start_idx:]
    else:
        stats['outliers'] = np.array([]).reshape(0, curves_copy.shape[1], curves_copy.shape[2])
    
    return stats

