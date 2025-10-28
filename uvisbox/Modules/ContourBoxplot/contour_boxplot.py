import matplotlib.pyplot as plt
import numpy as np
from .contour_boxplot_stats import contour_banddepth
from .contour_boxplot_vis import matplotlib_contour_boxplot

def contour_boxplot(ensemble_images, isovalue, percentiles=[25, 50, 75, 90], ax=None, colormap='viridis',
            show_median=True, show_outliers=True, workers=12):
        """
        Create a contour boxplot visualization from an ensemble of scalar fields.
        
        This function processes ensemble images by extracting binary contours at a given isovalue,
        computing their band depths, and visualizing the uncertainty using band envelopes.
        
        Parameters:
        -----------
        ensemble_images : np.ndarray
            3D or 4D array containing the ensemble scalar fields. 
            Can be shape (n_ensemble, y_dim, x_dim) or will be rearranged to this format.
        isovalue : float
            Threshold value for creating binary images. Pixels with values < isovalue are set to 1.
        percentiles : list of float, optional
            List of percentile values (0-100) for band envelope visualization. Default is [25, 50, 75, 90].
            Percentiles do not need to be sorted.
        ax : matplotlib.axes.Axes, optional
            Matplotlib Axes object to plot on. If None, a new figure and axes will be created.
        colormap : str, optional
            Matplotlib colormap name for the band visualization. Default is 'viridis'.
        show_median : bool, optional
            Whether to overlay the median contour in red. Default is True.
        show_outliers : bool, optional
            Whether to overlay outlier contours in gray. Default is True.
        workers : int, optional
            Number of parallel workers for band depth computation. Default is 12.
        
        Returns:
        --------
        ax : matplotlib.axes.Axes
            The Axes object with the contour boxplot visualization.
        
        Examples:
        ---------
        >>> ensemble = np.random.randn(50, 100, 100)  # 50 ensemble members
        >>> ax = contour_boxplot(ensemble, isovalue=0.5, percentiles=[25, 50, 75])
        """
        
        # Make a copy and ensure correct shape (n_ensemble, y_dim, x_dim)
        ensemble_copy = np.array(ensemble_images, copy=True)
        
        # Rearrange to (n_ensemble, y_dim, x_dim) if needed
        if ensemble_copy.ndim == 3:
            # Assume input is already (n_ensemble, y, x) or needs rearranging
            # Check if first dimension is likely the ensemble dimension
            if ensemble_copy.shape[0] > ensemble_copy.shape[2]:
                # Likely (y, x, n_ensemble) -> transpose to (n_ensemble, y, x)
                ensemble_copy = np.transpose(ensemble_copy, (2, 0, 1))
        elif ensemble_copy.ndim == 4:
            # Handle 4D case if needed (e.g., time, ensemble, y, x)
            raise ValueError("4D arrays not yet supported. Please provide 3D array of shape (n_ensemble, y, x)")
        
        # Get binary images: value < isovalue = 1, otherwise 0
        binary_images = (ensemble_copy < isovalue).astype(np.bool_)
        
        # Compute band depths using contour_banddepth
        depths = contour_banddepth(binary_images, workers=workers)
        
        # Sort images in descending depth order
        sorted_indices = np.argsort(depths)[::-1]
        ordered_binary_images = binary_images[sorted_indices]
        
        # Call matplotlib_contour_boxplot for visualization
        ax = matplotlib_contour_boxplot(
            ordered_binary_images, 
            percentiles=percentiles,
            colormap=colormap,
            show_median=show_median,
            show_outliers=show_outliers,
            ax=ax
        )
        
        return ax