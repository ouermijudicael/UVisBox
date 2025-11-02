from uvisbox.Core.BandDepths.functional_banddepth import *
import numpy as np
from uvisbox.Core.CommonInterface import BoxplotStyleConfig

# available functions:
#   - calculate_band(sorted_curves, percentile)
#   - functional_banddepth(data, dtype=np.float64):
#   - modified_functional_banddepth(data, dtype=np.float64):
#   - fbd = functional_banddepth
#   - mfbd = modified_functional_banddepth
#

def band_depths(data, method='fbd'):
    """
    Compute band depths for a set of functional curves.

    Parameters:
    -----------
    curves : np.ndarray
        2D array of shape (N, D) where N is the number of curves and D is the number of points per curve.
    method : str, optional
        Method for computing band depth. Options are:
        - 'fbd': Functional band depth (default)
        - 'mfbd': Modified functional band depth
    """
    if method == 'fbd':
        return functional_banddepth(data)
    elif method == 'mfbd':
        return modified_functional_banddepth(data)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'fbd' or 'mfbd'.")


def get_band(data, percentile, method='fbd'):
    """
    Compute the band envelope for a given percentile using functional band depth.
    
    Parameters:
    -----------
    data : np.ndarray
        2D array of shape (N, D) where N is the number of curves and D is the number of points per curve.
    percentile : float
        Percentile value (0-100) for the band envelope.
    method : str, optional
        Method for computing band depth. Options are:
        - 'fbd': functional band depth (default)
        - 'mfbd': modified functional band depth
    
    Returns:
    --------
    bottom_curve : np.ndarray
        1D array representing the bottom envelope of the band.
    top_curve : np.ndarray
        1D array representing the top envelope of the band.
    
    Examples:
    ---------
    >>> data = np.random.randn(100, 50)  # 100 curves, 50 points each
    >>> bottom, top = get_band(data, 50, method='fbd')  # 50th percentile band
    """
    # Validate input
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array of shape (N, D).")
    if not 0 <= percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100.")
    
    # Compute band depths based on method
    if method == 'fbd':
        depths = functional_banddepth(data)
    elif method == 'mfbd':
        depths = modified_functional_banddepth(data)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'fbd' or 'mfbd'.")

    # Sort curves by depth (descending order - highest depth first)
    sorted_indices = np.argsort(depths)[::-1]
    sorted_curves = data[sorted_indices]
    
    # Calculate the number of curves to include
    n_curves = sorted_curves.shape[0]
    index = int(np.ceil(n_curves * (percentile / 100)))
    
    # Get the top `index` curves with highest depth
    selected_curves = sorted_curves[:index]
    
    # Compute envelope
    top_curve = np.max(selected_curves, axis=0)
    bottom_curve = np.min(selected_curves, axis=0)
    
    return bottom_curve, top_curve


def summary_statistics(data, method='fbd', boxplot_style=None):
    """
    Compute functional boxplot summary statistics without visualization.
    
    This function computes the same statistics as functional_boxplot but returns
    them as numpy arrays in a dictionary instead of creating a plot.
    
    Parameters:
    -----------
    data : np.ndarray
        2D array of shape (N, D) where N is the number of curves and D is the number 
        of points per curve.
    method : str, optional
        Method for computing band depth. Options are:
        - 'fbd': functional band depth (default)
        - 'mfbd': modified functional band depth
    boxplot_style : BoxplotStyleConfig, optional
        Configuration for the boxplot including percentiles and outlier settings.
        If None, uses default configuration.
    
    Returns:
    --------
    stats : dict
        Dictionary containing the following keys:
        - 'depths': np.ndarray of shape (N,) - band depths for each curve
        - 'median': np.ndarray of shape (D,) - median curve (highest depth)
        - 'percentile_bands': dict - percentile bands with keys like '25_percentile_band'
          containing tuples (bottom_curve, top_curve) of shape (D,) each
        - 'outliers': np.ndarray of shape (n_outliers, D) - outlier curves beyond largest percentile
        - 'sorted_curves': np.ndarray of shape (N, D) - curves sorted by depth (descending)
        - 'sorted_indices': np.ndarray of shape (N,) - original indices sorted by depth
    
    Raises:
    -------
    ValueError
        If data is not 2D or if method is invalid.
    
    Examples:
    ---------
    >>> import numpy as np
    >>> from uvisbox.Modules.FunctionalBoxplot.functional_boxplot_stats import summary_statistics
    >>> from uvisbox.Core.CommonInterface import BoxplotStyleConfig
    >>> 
    >>> # Generate synthetic functional data
    >>> t = np.linspace(0, 1, 100)
    >>> data = np.array([np.sin(2*np.pi*t) + 0.2*np.random.randn(100) for _ in range(50)])
    >>> 
    >>> # Basic usage with default settings
    >>> stats = summary_statistics(data)
    >>> print(f"Median curve shape: {stats['median'].shape}")
    >>> print(f"Number of outliers: {stats['outliers'].shape[0]}")
    >>> print(f"Available percentile bands: {list(stats['percentile_bands'].keys())}")
    >>> 
    >>> # Access specific percentile band
    >>> bottom_25, top_25 = stats['percentile_bands']['25_percentile_band']
    >>> print(f"25th percentile band shapes: {bottom_25.shape}, {top_25.shape}")
    >>> 
    >>> # Custom percentiles
    >>> style = BoxplotStyleConfig(percentiles=[10, 50, 90], show_outliers=True)
    >>> stats = summary_statistics(data, boxplot_style=style)
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
    if method == 'fbd' or method == 'mfbd':
        depths = band_depths(data_copy, method=method)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'fbd' or 'mfbd'.")

    # Sort curves by depth (descending order - highest depth first)
    sorted_indices = np.argsort(depths)[::-1]
    sorted_data = data_copy[sorted_indices]
    sorted_depths = depths[sorted_indices]
    
    # Initialize results dictionary
    stats = {
        'depths': depths,
        'sorted_indices': sorted_indices,
        'sorted_curves': sorted_data,
        'percentile_bands': {}
    }
    
    # Compute median curve (curve with maximum depth)
    stats['median'] = sorted_data[0]
    
    # Compute percentile bands
    for percentile in boxplot_style.percentiles:
        bottom, top = get_band(data_copy, percentile, method=method)
        band_key = f'{int(percentile)}_percentile_band'
        stats['percentile_bands'][band_key] = (bottom, top)
    
    # Compute outliers if requested
    if boxplot_style.show_outliers and len(boxplot_style.percentiles) > 0:
        largest_percentile = max(boxplot_style.percentiles)
        outlier_start_idx = int(np.ceil(len(sorted_data) * largest_percentile / 100))
        stats['outliers'] = sorted_data[outlier_start_idx:]
    else:
        stats['outliers'] = np.array([]).reshape(0, data_copy.shape[1])
    
    return stats