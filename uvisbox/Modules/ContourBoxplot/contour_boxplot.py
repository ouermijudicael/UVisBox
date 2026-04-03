import matplotlib.pyplot as plt
import numpy as np
from .contour_boxplot_stats import contour_boxplot_summary_statistics
from .contour_boxplot_mesh import contour_boxplot_mesh
from .contour_boxplot_vis import visualize_contour_boxplot
from uvisbox.Core.CommonInterface import BoxplotStyleConfig


def contour_boxplot(ensemble_images, isovalue, boxplot_style=None, ax=None, eps=0, workers=11,
                    x_coords=None, y_coords=None):
    """
    Create a contour boxplot visualization from an ensemble of scalar fields.
    
    This function processes ensemble images by extracting binary contours at a given isovalue,
    computing their band depths, and visualizing the uncertainty using band envelopes.
    
    Parameters:
    -----------
    ensemble_images : np.ndarray
        3D array of shape (n_ensemble, y_dim, x_dim) containing the ensemble scalar fields.
    isovalue : float
        Threshold value for creating binary images. Pixels with values < isovalue are set to 1.
    boxplot_style : BoxplotStyleConfig, optional
        Configuration for the boxplot visualization including percentiles,
        and median/outlier styling. If None, uses default configuration.
        The percentile_colormap is used for the band sum visualization.
    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes object to plot on. If None, a new figure and axes will be created.
    eps : float, optional
        Epsilon tolerance for subset checks in band depth computation. With eps=0 (default),
        exact pixel-level containment is required, which can be too strict for continuously-varying
        rasterized contours. Values like 0.01-0.05 allow approximate containment.
    workers : int, optional
        Number of parallel workers for band depth computation. Default is 11.
    x_coords : np.ndarray, optional
        1D array of x-axis coordinates defining the spatial domain.
        Length must match the image width. If None, pixel indices are used.
    y_coords : np.ndarray, optional
        1D array of y-axis coordinates defining the spatial domain.
        Length must match the image height. If None, pixel indices are used.

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The Axes object with the contour boxplot visualization.
    
    Examples:
    ---------
    >>> # Basic usage with defaults
    >>> ensemble = np.random.randn(50, 100, 100)  # 50 ensemble members
    >>> ax = contour_boxplot(ensemble, isovalue=0.5)
    
    >>> # Custom styling
    >>> from uvisbox.Core.CommonInterface import BoxplotStyleConfig
    >>> style = BoxplotStyleConfig(
    ...     percentiles=[25, 50, 75],
    ...     percentile_colormap='hot',
    ...     show_median=True,
    ...     show_outliers=True
    ... )
    >>> ax = contour_boxplot(ensemble, isovalue=0.5, boxplot_style=style)
    """
    # Use default config if none provided
    if boxplot_style is None:
        boxplot_style = BoxplotStyleConfig()
    
    # Make a copy and ensure correct shape (n_ensemble, y_dim, x_dim)
    ensemble_copy = np.array(ensemble_images, copy=True)

    if ensemble_copy.ndim != 3:
        raise ValueError(f"ensemble_images must be a 3D array of shape (n_ensemble, y, x). Got {ensemble_copy.ndim}D")
    
    # Pipeline: Stats -> Mesh -> Vis
    # Step 1: Compute summary statistics
    stats = contour_boxplot_summary_statistics(
        ensemble_copy,
        isovalue,
        boxplot_style=boxplot_style,
        eps=eps,
        workers=workers
    )
    
    # Step 2: Process through mesh (aggregate bands)
    mesh_data = contour_boxplot_mesh(stats, x_coords=x_coords, y_coords=y_coords)
    
    # Step 3: Visualize
    ax = visualize_contour_boxplot(
        mesh_data,
        boxplot_style=boxplot_style,
        ax=ax
    )
    
    return ax