import numpy as np
import matplotlib.pyplot as plt
from ..Stat.contour_banddepth import contour_banddepth

def _find_percentile(sorted_images, percentile):
    n_images = sorted_images.shape[0]
    index = int(np.ceil(n_images * (percentile / 100)))

    before = sorted_images[:index]

    # Find union and intersection
    union = np.any(before, axis=0)
    intersection = np.all(before, axis=0)
    # Pixels in union but not in intersection
    union_minus_intersection = union & (~intersection)
    return union_minus_intersection


def contour_boxplot(binary_images, depths=None, ax=None, eps=0, allow_portion=False, show_median=True, show_iqr=True, show_non_outliers=False, show_outliers=False, show_firstquartile=False, outlier_percentile=95):
    """
    Create a contour boxplot for binary images based on their band depths.
    Parameters:
    ----------
    binary_images : np.ndarray
        3D array of shape (n_images, height, width) containing binary images (0s and 1s)
    depths : np.ndarray, optional
        1D array of precomputed depth scores for each image. If None, depths will be computed.
    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes object to plot on. If None, a new figure and axes will be created.
    eps : float, optional
        Tolerance for numerical precision when computing band depths. Default is 0.
    allow_portion : bool, optional
        If True, allows partial inclusion of contours in depth calculation. Default is False.
    show_median : bool, optional
        If True, highlights the median contour in red. Default is True.
    show_iqr : bool, optional
        If True, highlights the interquartile range (IQR) in gray. Default is True.
    show_non_outliers : bool, optional
        If True, highlights non-outlier regions in light gray. Default is False.
    show_outliers : bool, optional
        If True, outlines outlier contours in blue. Default is False.
    show_firstquartile : bool, optional
        If True, highlights the first quartile region in a different shade of gray. Default is False.
    outlier_percentile : float, optional
        Percentile threshold to define outliers. Default is 95.
    Returns:
    -------
    ax: matplotlib.axes.Axes
        The Axes object with the contour boxplot.
    """
    if depths is None:
        depths = contour_banddepth(binary_images, eps=eps, allow_portion=allow_portion)
    # sort the contours by the depth. order them from deepest to shallowest
    sorted_indices = np.argsort(depths)[::-1]
    sorted_images = binary_images[sorted_indices]

    # create figure if no ax is assigned
    if ax is None:
        fig, ax = plt.subplots()

    # background image
    # assuming the image is in y,x format, binary image is either [n,y,x,1] or [n,y,x]
    # create a background image
    result_image = np.zeros_like(sorted_images[0],dtype=np.float32) + 100
    n_images = sorted_images.shape[0]

    ### build the image from bottom up
    non_outlier_cutoff_index = int(n_images * (outlier_percentile / 100))
    outliers = sorted_images[non_outlier_cutoff_index:]

    if show_non_outliers:
        non_outliers_indices = _find_percentile(sorted_images, outlier_percentile)
        result_image[non_outliers_indices] = outlier_percentile

    if show_iqr:
        iqr = _find_percentile(sorted_images, 50)
        result_image[iqr] = 50

    if show_firstquartile:
        first_quartile = _find_percentile(sorted_images, 25)
        result_image[first_quartile] = 25
    
    ax.imshow(result_image, origin='lower', cmap='gray')

    if show_outliers:
        for outlier in outliers:
            ax.contour(outlier, levels=[0.5], colors='blue',linewidths=1)
    
    if show_median:
        median = sorted_images[0]
        ax.contour(median, levels=[0.5], colors='red', linewidths=2)
    
    return ax





