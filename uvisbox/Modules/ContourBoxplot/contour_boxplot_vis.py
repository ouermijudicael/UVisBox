
def matplotlib_contour_vis(result_image, median= None, outliers=None, ax=None):
    """
    Plot the contour boxplot with median and outliers.
    
    Parameters:
    -----------
    result_image : np.ndarray
        2D array representing the contour boxplot image.
    median : np.ndarray, optional
        2D binary array representing the median contour. Default is None.   
    outliers : list of np.ndarray, optional
        List of 2D binary arrays representing outlier contours. Default is None.
    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes object to plot on. If None, a new figure and axes will be created.
    
    Returns:
    --------
    ax : matplotlib.axes.Axes
        Matplotlib Axes object with the plotted contour boxplot.
    """

    ax.imshow(result_image, origin='lower', cmap='gray')

    if outliers is not None:
        for outlier in outliers:
            ax.contour(outlier, levels=[0.5], colors='blue',linewidths=1)

    if median is not None:
        ax.contour(median, levels=[0.5], colors='red', linewidths=2)
    
    return ax





