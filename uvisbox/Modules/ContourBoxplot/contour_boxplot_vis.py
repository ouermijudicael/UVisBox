import matplotlib.pyplot as plt
import numpy as np
from .contour_boxplot_mesh import get_band

def matplotlib_contour_boxplot(ordered_binary_images, percentiles=[25,50,75,90], colormap='viridis', show_median=True, show_outliers=False, ax=None):
    """
    Plot the contour boxplot bands by summing band envelopes for given percentiles.
    
    Parameters:
    -----------
    ordered_binary_images : np.ndarray
        3D array of shape (N, H, W) where N is the number of binary images and H, W are the height and width of each image.
        The images should be ordered by their contour band depths in descending order (highest depth first).
    percentiles : list of float, optional
        List of percentile values (0-100) for the band envelopes. Default is [25, 50, 75, 90].
        Percentiles do not need to be sorted.
    colormap : str, optional
        Matplotlib colormap name. Default is 'viridis'.
    outliers : bool, optional
        Whether to display outlier contours. Default is False.
    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes object to plot on. If None, a new figure and axes will be created.
    
    Returns:
    --------
    ax : matplotlib.axes.Axes
        Matplotlib Axes object with the plotted contour boxplot.
    """

    if ax is None:
        fig, ax = plt.subplots()

    # Sort percentiles in decreasing order
    sorted_percentiles = sorted(percentiles, reverse=True)
    
    # Initialize sum image
    h, w = ordered_binary_images.shape[1], ordered_binary_images.shape[2]
    band_sum = np.zeros((h, w), dtype=np.int32)
    
    # Get bands for each percentile and sum them
    for percentile in sorted_percentiles:
        band = get_band(ordered_binary_images, percentile)
        band_sum += band.astype(np.int32)
    
    # Display with origin='lower'
    im = ax.imshow(band_sum, cmap=colormap, origin='lower', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Band sum')

    # Track handles and labels for legend
    legend_handles = []
    legend_labels = []

    if show_outliers:
        start_idx = np.ceil(sorted_percentiles[0] / 100 * ordered_binary_images.shape[0]).astype(int)
        for i, idx in enumerate(range(start_idx, ordered_binary_images.shape[0])):
            contour_set = ax.contour(ordered_binary_images[idx], levels=[0.5], colors='gray', linewidths=1, alpha=0.5)
            # Only add to legend once (first outlier)
            if i == 0:
                handles, _ = contour_set.legend_elements()
                legend_handles.append(handles[0])
                legend_labels.append('Outliers')
    
    if show_median:
        median_idx = 0
        contour_set = ax.contour(ordered_binary_images[median_idx], levels=[0.5], colors='red', linewidths=3)
        handles, _ = contour_set.legend_elements()
        legend_handles.append(handles[0])
        legend_labels.append('Median')

    # Add legend only if there are items to show
    if legend_handles:
        ax.legend(legend_handles, legend_labels)

    return ax





