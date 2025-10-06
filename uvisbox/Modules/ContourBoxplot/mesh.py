from .stats import find_percentile as _find_percentile
import numpy as np

def mesh(binary_images, depths, outlier_percentile=95,
         show_non_outliers=True, show_iqr=True, show_firstquartile=True):
    """
    Create a contour boxplot mesh from binary images and their depths.

    Parameters:
    -----------
    binary_images : np.ndarray
        3D array of shape (n_images, height, width) containing binary images (0s and 1s)
    depths : np.ndarray
        1D array of precomputed depth scores for each image. If None, depths will be computed.
    outlier_percentile : float, optional
        Percentile threshold to define outliers. Default is 95.
    show_non_outliers : bool, optional
        If True, highlights non-outlier regions in light gray. Default is True.
    show_iqr : bool, optional
        If True, highlights the interquartile range (IQR) in gray. Default is True.
    show_firstquartile : bool, optional
        If True, highlights the first quartile region in a different shade of gray. Default is True.

    Returns:
    --------
    result_image : np.ndarray
        2D array representing the contour boxplot mesh.
    top_contour : np.ndarray
        2D array representing the top contour of the mesh.
    outliers : np.ndarray
        3D array containing the outlier images.

    """

    # sort the contours by the depth. order them from deepest to shallowest
    sorted_indices = np.argsort(depths)[::-1]
    sorted_images = binary_images[sorted_indices]


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
    
    median = sorted_images[0] if n_images > 0 else None

    return result_image, median, outliers
