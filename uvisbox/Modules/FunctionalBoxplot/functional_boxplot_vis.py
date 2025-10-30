import numpy as np
import matplotlib.pyplot as plt


def plot_band(bottom_curve, top_curve, ax=None, color='red', alpha=1.0, scale=1.0):
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
    >>> from uvisbox.Modules.FunctionalBoxplot import plot_band, get_band
    >>> 
    >>> # Generate synthetic data
    >>> data = np.random.randn(100, 50).cumsum(axis=1)
    >>> 
    >>> # Get 50th percentile band
    >>> bottom, top = get_band(data, 50, method='fbd')
    >>> 
    >>> # Plot the band
    >>> fig, ax = plt.subplots()
    >>> plot_band(bottom, top, ax=ax, color='blue', alpha=0.5)
    >>> plt.show()
    """
    # Validate inputs
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
    x = np.linspace(0, 1, n_points)
    
    # Use fill_between to create the band
    ax.fill_between(x, bottom_scaled, top_scaled, 
                    color=color, alpha=alpha, edgecolor='none')
    
    return ax


