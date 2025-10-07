import numpy as np
import matplotlib.pyplot as plt
from .functional_boxplot_stats import functional_banddepth, modified_functional_banddepth, calculate_band
from .functional_boxplot_vis import vis


def functional_boxplot(curves, method='functional_bd', curves_depths=None, percentil=100, scale=1.0, ax=None, 
         show_median=True, band_alpha=0.5):
    """
    Plot the functional band depth area between the top and bottom curves along with the median curve.

    Parameters:
    -----------
    curves : np.ndarray
        2D array of shape (N, D) where N is the number of samples and D is the number of features
    method : str, optional
        The method to use for plotting. Options are 'functional_bd' and 'modified_bd'. Default is 'functional_bd'.
    curves_depths : np.ndarray, optional
        1D array of band depths of shape (N,). If None, it will be computed.
    percentil : float, optional
        Percentile for the band depth calculation. Default is 100.
    scale : float, optional
        Scale factor for the depth area. Default is 1.0.
    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes object to plot on. If None, a new figure and axes will be created.
    show_median : bool, optional
        Whether to plot the median curve. Default is True.
    band_alpha : float, optional
        Alpha value for the band depth area. Default is 0.5.

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The Axes object with the plot.

    Usage Example:
    --------------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from uvisbox.Modules.FunctionalBoxplot.plot import plot
    >>> # Generate synthetic functional data
    >>> np.random.seed(0)
    >>> x = np.linspace(0, 10, 100)
    >>> curves = np.array([np.sin(x) + np.random.normal(0, 0.1, size=x.shape) for _ in range(50)])
    >>> # Create a figure and plot the functional boxplot
    >>> fig, ax = plt.subplots(figsize=(10, 6))
    >>> plot(curves, method='functional_bd', percentil=50, scale=1.0, ax=ax)
    >>> plt.title("Functional Boxplot using Functional Band Depth")
    >>> plt.xlabel("X-axis")
    >>> plt.ylabel("Y-axis")
    >>> plt.show()
    
    """
    # Compute band depths if not provided
    if curves_depths is None and method == 'functional_bd':
        curves_depths = functional_banddepth(curves)
    elif curves_depths is None and method == 'modified_bd':
        curves_depths = modified_functional_banddepth(curves)
    elif curves_depths is None and method not in ['functional_bd', 'modified_bd']:
        raise ValueError("Invalid method. Choose 'functional_bd' or 'modified_bd'.")

    # Determine top and bottom curves for the specified percentile
    bottom_curve, top_curve = calculate_band(curves[np.argsort(-curves_depths)], percentil)

    # Plot the functional band depth
    ax = vis(top_curve, bottom_curve, curves, ax=ax, show_median=show_median, band_alpha=band_alpha)

    return ax