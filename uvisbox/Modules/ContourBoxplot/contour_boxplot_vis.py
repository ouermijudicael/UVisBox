import matplotlib.pyplot as plt
import numpy as np
from uvisbox.Core.CommonInterface import BoxplotStyleConfig


def visualize_contour_boxplot(mesh_data, boxplot_style=None, ax=None):
    """
    Visualize contour boxplot using imshow for percentile bands and contour for median/outliers.
    
    Parameters:
    -----------
    mesh_data : dict
        Dictionary from contour_boxplot_mesh containing:
        - 'percentile_bands_image': 2D array with aggregated percentile band values
        - 'median': Binary image for median contour
        - 'outliers': List of binary images for outlier contours
    boxplot_style : BoxplotStyleConfig, optional
        Configuration for visualization including colormap, median/outlier styling.
        If None, uses default configuration.
    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes object to plot on. If None, creates new figure and axes.
    
    Returns:
    --------
    ax : matplotlib.axes.Axes
        The Axes object with the contour boxplot visualization.
    
    Notes:
    ------
    - Uses ax.imshow() to display the aggregated percentile bands with colormap
    - Uses ax.contour() at isovalue 0.5 to plot median and outlier contour lines
    - Colorbar shows percentile values from 0-100
    - Legend indicates median and outliers
    
    Examples:
    ---------
    >>> stats = contour_boxplot_summary_statistics(ensemble, isovalue=0.5)
    >>> mesh_data = contour_boxplot_mesh(stats)
    >>> ax = visualize_contour_boxplot(mesh_data)
    """
    # Use default config if none provided
    if boxplot_style is None:
        boxplot_style = BoxplotStyleConfig()
    
    if ax is None:
        fig, ax = plt.subplots()
    
    # Get data
    percentile_bands_image = mesh_data['percentile_bands_image']
    median = mesh_data['median']
    outliers = mesh_data['outliers']
    
    # Plot percentile bands using imshow
    # vmin=0, vmax=1 since values are percentile/100
    im = ax.imshow(
        percentile_bands_image,
        cmap=boxplot_style.percentile_colormap,
        origin='lower',
        interpolation='nearest',
        vmin=0,
        vmax=1
    )
    
    # Add colorbar with percentile scale (0-100)
    cbar = plt.colorbar(im, ax=ax, label='Percentile')
    cbar.ax.set_ylabel('Percentile', rotation=270, labelpad=15)
    # Set colorbar to show 0-100 scale
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['0', '25', '50', '75', '100'])
    
    # Track legend handles and labels
    legend_handles = []
    legend_labels = []
    
    # Plot median contour at isovalue 0.5
    if boxplot_style.show_median:
        contour_set = ax.contour(
            median,
            levels=[0.5],
            colors=boxplot_style.median_color,
            linewidths=boxplot_style.median_width,
            alpha=boxplot_style.median_alpha
        )
        # contour() returns a QuadContourSet, check if it has any contours
        if len(contour_set.allsegs[0]) > 0:
            legend_handles.append(plt.Line2D([0], [0], color=boxplot_style.median_color, 
                                            linewidth=boxplot_style.median_width,
                                            alpha=boxplot_style.median_alpha))
            legend_labels.append('Median')
    
    # Plot outlier contours at isovalue 0.5
    if boxplot_style.show_outliers and len(outliers) > 0:
        outlier_added = False
        for outlier in outliers:
            contour_set = ax.contour(
                outlier,
                levels=[0.5],
                colors=boxplot_style.outliers_color,
                linewidths=boxplot_style.outliers_width,
                alpha=boxplot_style.outliers_alpha
            )
            # Add to legend only once (first outlier with actual contours)
            if not outlier_added and len(contour_set.allsegs[0]) > 0:
                legend_handles.append(plt.Line2D([0], [0], color=boxplot_style.outliers_color,
                                                 linewidth=boxplot_style.outliers_width,
                                                 alpha=boxplot_style.outliers_alpha))
                legend_labels.append('Outliers')
                outlier_added = True
    
    # Add legend if there are items to show
    if legend_handles:
        ax.legend(legend_handles, legend_labels)
    
    return ax





