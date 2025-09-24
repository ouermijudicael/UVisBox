import numpy as np
import matplotlib.pyplot as plt
from ..Stat.functional_banddepth import functional_banddepth, calculate_band, modified_functional_banddepth

def functional_banddepth_plot(curves, curves_depths=None, percentil=100, scale=1.0, ax=None, 
                              show_median=True, band_alpha=0.5):
    """
    Plot the functional band depth area between the top and bottom curves along with the median curve.
    
    Parameters:
    -----------
    curves : np.ndarray
        2D array of shape (N, D) where N is the number of samples and D is the number of features
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
    """

    if ax is None:
        fig, ax = plt.subplots()

    if curves_depths is None:
        curves_depths = functional_banddepth(curves)

    bottom_curve, top_curve = calculate_band(curves[np.argsort(-curves_depths)], percentil)
    # Plot the functional band depth
    ax.fill_between(np.arange(bottom_curve.shape[0]), bottom_curve * scale, top_curve * scale, color='skyblue', alpha=band_alpha, label=f'Band Depth {percentil}%')

    if show_median:
        # Plot the median curve
        median_curve = np.median(curves, axis=0)
        ax.plot(median_curve, color='red', label='Median Curve', linewidth=2)

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Functional Band Depth Plot')
    ax.legend()

    return ax
        


def modified_functional_banddepth_plot(curves, curves_depths=None, percentil=100, scale=1.0, ax=None, 
                              show_median=True, band_alpha=0.5):
    """
    Plot the functional band depth area between the top and bottom curves along with the median curve.
    Parameters:
    -----------
    curves : np.ndarray
        2D array of shape (N, D) where N is the number of samples and D is the number of features
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
    """

    if ax is None:
        fig, ax = plt.subplots()

    if curves_depths is None:
        curves_depths = functional_banddepth(curves)

    bottom_curve, top_curve = calculate_band(curves[np.argsort(-curves_depths)], percentil)
    # Plot the functional band depth
    ax.fill_between(np.arange(bottom_curve.shape[0]), bottom_curve * scale, top_curve * scale, color='skyblue', alpha=band_alpha, label=f'Band Depth {percentil}%')

    if show_median:
        # Plot the median curve
        median_curve = np.median(curves, axis=0)
        ax.plot(median_curve, color='red', label='Median Curve', linewidth=2)

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Functional Band Depth Plot')
    ax.legend()

    return ax
        
fbd_plot = functional_banddepth_plot
mfbd_plot = modified_functional_banddepth_plot