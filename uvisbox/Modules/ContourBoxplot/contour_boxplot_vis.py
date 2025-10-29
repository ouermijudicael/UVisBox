import matplotlib.pyplot as plt
import numpy as np
from .contour_boxplot_mesh import get_band
from uvisbox.Core.CommonInterface import BoxplotStyleConfig

def matplotlib_contour_boxplot(ordered_binary_images, boxplot_style=None, ax=None):
    """
    Plot the contour boxplot bands by summing band envelopes for given percentiles.
    
    Parameters:
    -----------
    ordered_binary_images : np.ndarray
        3D array of shape (N, H, W) where N is the number of binary images and H, W are the height and width of each image.
        The images should be ordered by their contour band depths in descending order (highest depth first).
    boxplot_style : BoxplotStyleConfig, optional
        Configuration for the boxplot visualization including percentiles,
        and median/outlier styling. If None, uses default configuration.
        The percentile_colormap is used for the band sum visualization.
    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes object to plot on. If None, a new figure and axes will be created.
    
    Returns:
    --------
    ax : matplotlib.axes.Axes
        Matplotlib Axes object with the plotted contour boxplot.
    """

    # Use default config if none provided
    if boxplot_style is None:
        boxplot_style = BoxplotStyleConfig()

    if ax is None:
        fig, ax = plt.subplots()

    # Sort percentiles in descending order (highest to lowest)
    sorted_percentiles = sorted(boxplot_style.percentiles, reverse=True)
    
    # Initialize color value image with zeros
    h, w = ordered_binary_images.shape[1], ordered_binary_images.shape[2]
    color_array = np.zeros((h, w), dtype=np.float32)
    
    # Apply bands in descending order of percentile
    # Higher percentiles overwrite lower ones at non-zero pixels
    for percentile in sorted_percentiles:
        band = get_band(ordered_binary_images, percentile)
        # Where band is non-zero, set color value to percentile/100
        color_value = percentile / 100.0
        mask = band > 0
        color_array[mask] = color_value
    
    # Display with origin='lower'
    im = ax.imshow(color_array, cmap=boxplot_style.percentile_colormap, origin='lower', 
                   interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Percentile')

    # Track handles and labels for legend
    legend_handles = []
    legend_labels = []

    if boxplot_style.show_outliers:
        start_idx = np.ceil(sorted_percentiles[0] / 100 * ordered_binary_images.shape[0]).astype(int)
        for i, idx in enumerate(range(start_idx, ordered_binary_images.shape[0])):
            contour_set = ax.contour(ordered_binary_images[idx], levels=[0.5], 
                                   colors=boxplot_style.outliers_color, 
                                   linewidths=boxplot_style.outliers_width, 
                                   alpha=boxplot_style.outliers_alpha)
            # Only add to legend once (first outlier)
            if i == 0:
                handles, _ = contour_set.legend_elements()
                legend_handles.append(handles[0])
                legend_labels.append('Outliers')
    
    if boxplot_style.show_median:
        median_idx = 0
        contour_set = ax.contour(ordered_binary_images[median_idx], levels=[0.5], 
                               colors=boxplot_style.median_color, 
                               linewidths=boxplot_style.median_width,
                               alpha=boxplot_style.median_alpha)
        handles, _ = contour_set.legend_elements()
        legend_handles.append(handles[0])
        legend_labels.append('Median')

    # Add legend only if there are items to show
    if legend_handles:
        ax.legend(legend_handles, legend_labels)

    return ax





