import matplotlib.pyplot as plt
import numpy as np
from .contour_boxplot_stats import contour_boxplot_summary_statistics
from .contour_boxplot_mesh import contour_boxplot_mesh
from .contour_boxplot_vis import visualize_contour_boxplot
from uvisbox.Core.CommonInterface import BoxplotStyleConfig


def contour_boxplot(ensemble_images, isovalue, boxplot_style=None, ax=None, workers=11):
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
    boxplot_style : BoxplotStyleConfig, optional
        Configuration for the boxplot visualization including percentiles,
        and median/outlier styling. If None, uses default configuration.
        The percentile_colormap is used for the band sum visualization.
    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes object to plot on. If None, a new figure and axes will be created.
    workers : int, optional
        Number of parallel workers for band depth computation. Default is 12.
    
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
    
    # Pipeline: Stats -> Mesh -> Vis
    # Step 1: Compute summary statistics
    stats = contour_boxplot_summary_statistics(
        ensemble_copy,
        isovalue,
        boxplot_style=boxplot_style,
        workers=workers
    )
    
    # Step 2: Process through mesh (aggregate bands)
    mesh_data = contour_boxplot_mesh(stats)
    
    # Step 3: Visualize
    ax = visualize_contour_boxplot(
        mesh_data,
        boxplot_style=boxplot_style,
        ax=ax
    )
    
    return ax