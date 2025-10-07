import matplotlib.pyplot as plt
import numpy as np
from .contour_boxplot_stats import contour_banddepth
from .contour_boxplot_vis import matplotlib_contour_vis
from .contour_boxplot_mesh import countour_binary_image 

def contour_boxplot(binary_images, method='contour_bd', binary_images_depths=None, percentil=95, ax=None,
            show_median=True, show_outliers=True):
        """
        Plot the contour boxplot including the band depth area between the top and bottom contours along with the median contour.
    
        Parameters:
        -----------
        binary_images : np.ndarray
            3D array of shape (N, H, W) where N is the number of binary images and H, W are the height and width of each image.
        method : str, optional
            The method to use for plotting. Options are 'contour_bd'. Default is 'contour_bd'.
        binary_images_depths : np.ndarray, optional
            1D array of band depths of shape (N,). If None, it will be computed.
        percentil : float, optional
            Percentile for the band depth calculation. Default is 100.
        ax : matplotlib.axes.Axes, optional
            Matplotlib Axes object to plot on. If None, a new figure and axes will be created.
        show_median : bool, optional
            Whether to plot the median contour. Default is True.
        show_outliers : bool, optional
            Whether to plot the outlier contours. Default is True.

        Returns:
        --------
        ax : matplotlib.axes.Axes
            The Axes object with the plot.
        
        Usage Example:
        --------------

        
        """
        # Compute contour band depths if not provided
        if binary_images_depths is None:
            binary_images_depths = contour_banddepth(binary_images)

        # construct resulting image using contour function
        result_image, median_image, outliers = countour_binary_image(binary_images, binary_images_depths, outlier_percentile=percentil)

        # visualize the resulting image
        ax = matplotlib_contour_vis(result_image, median=median_image, outliers=outliers, ax=ax)

        return ax