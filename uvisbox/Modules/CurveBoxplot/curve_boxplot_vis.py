import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from uvisbox.Core.CommonInterface import BoxplotStyleConfig


def visualize_curve_boxplot(mesh_data, boxplot_style=None, ax=None):
    """
    Visualize curve boxplot from mesh data.
    
    This function creates a matplotlib visualization of the curve boxplot using
    the mesh data output from the mesh pipeline. It handles both 2D and 3D curves
    with appropriate projection.
    
    Parameters:
    -----------
    mesh_data : dict
        Dictionary containing mesh data with the following keys:
        - 'percentile_meshes': dict of percentile meshes
        - 'median_curve': median curve
        - 'outliers': outlier curves
        - 'n_dims': dimensionality (2 or 3)
    boxplot_style : BoxplotStyleConfig, optional
        Configuration for the boxplot visualization including percentiles, colors,
        and median/outlier styling. If None, uses default configuration.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes to plot on. If None, creates a new figure.
        For 3D curves, must be a 3D axes if provided.
    
    Returns:
    --------
    ax : matplotlib.axes.Axes
        The matplotlib axes object with the plot.
    
    Examples:
    ---------
    >>> import numpy as np
    >>> from uvisbox.Modules.CurveBoxplot.curve_boxplot_stats import curve_boxplot_summary_statistics
    >>> from uvisbox.Modules.CurveBoxplot.curve_boxplot_mesh import curve_boxplot_mesh
    >>> from uvisbox.Modules.CurveBoxplot.curve_boxplot_vis import visualize_curve_boxplot
    >>> from uvisbox.Core.CommonInterface import BoxplotStyleConfig
    >>> 
    >>> # Generate synthetic curve data
    >>> curves = np.random.randn(50, 100, 2).cumsum(axis=1)
    >>> 
    >>> # Process through pipeline
    >>> stats = curve_boxplot_summary_statistics(curves)
    >>> mesh_data = curve_boxplot_mesh(stats)
    >>> 
    >>> # Visualize
    >>> ax = visualize_curve_boxplot(mesh_data)
    >>> 
    >>> # Custom styling
    >>> style = BoxplotStyleConfig(percentiles=[25, 50, 75], show_outliers=True)
    >>> ax = visualize_curve_boxplot(mesh_data, boxplot_style=style)
    """
    # Use default config if none provided
    if boxplot_style is None:
        boxplot_style = BoxplotStyleConfig()
    
    n_dims = mesh_data['n_dims']
    
    # Create figure/axes if not provided
    if ax is None:
        if n_dims == 2:
            fig, ax = plt.subplots(figsize=(10, 8))
        elif n_dims == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        else:
            raise ValueError(f"Unsupported curve dimension: {n_dims}. Must be 2 or 3.")
    
    # Get colors from colormap
    colors = boxplot_style.get_percentile_colors()
    percentiles = boxplot_style.percentiles
    
    # Sort percentiles in descending order for proper plotting (largest first)
    sorted_percentile_indices = np.argsort(percentiles)[::-1]
    sorted_percentiles = [percentiles[i] for i in sorted_percentile_indices]
    sorted_colors = [colors[i] for i in sorted_percentile_indices]
    
    # Plot each percentile band from largest to smallest
    for percentile, color in zip(sorted_percentiles, sorted_colors):
        mesh_key = f'{int(percentile)}_percentile_mesh'
        if mesh_key in mesh_data['percentile_meshes']:
            points, triangles = mesh_data['percentile_meshes'][mesh_key]
            _plot_band_mesh(points, triangles, ax=ax, color=color, alpha=1.0, n_dims=n_dims)
    
    # Plot outliers (curves beyond the largest percentile)
    if boxplot_style.show_outliers and mesh_data['outliers'].shape[0] > 0:
        outliers = mesh_data['outliers']
        for idx in range(len(outliers)):
            outlier_curve = outliers[idx]
            # Add label only for the first outlier to avoid duplicate legend entries
            label = 'Outliers' if idx == 0 else None
            if n_dims == 2:
                ax.plot(outlier_curve[:, 0], outlier_curve[:, 1], 
                       color=boxplot_style.outliers_color, 
                       linewidth=boxplot_style.outliers_width, 
                       alpha=boxplot_style.outliers_alpha,
                       label=label,
                       zorder=5)
            elif n_dims == 3:
                ax.plot(outlier_curve[:, 0], outlier_curve[:, 1], outlier_curve[:, 2],
                       color=boxplot_style.outliers_color, 
                       linewidth=boxplot_style.outliers_width, 
                       alpha=boxplot_style.outliers_alpha,
                       label=label,
                       zorder=5)
    
    # Plot the median curve (curve with maximum depth)
    if boxplot_style.show_median:
        median_curve = mesh_data['median_curve']
        if n_dims == 2:
            ax.plot(median_curve[:, 0], median_curve[:, 1], 
                   color=boxplot_style.median_color, 
                   linewidth=boxplot_style.median_width, 
                   alpha=boxplot_style.median_alpha, 
                   label='Median Curve', zorder=10)
        elif n_dims == 3:
            ax.plot(median_curve[:, 0], median_curve[:, 1], median_curve[:, 2],
                   color=boxplot_style.median_color, 
                   linewidth=boxplot_style.median_width, 
                   alpha=boxplot_style.median_alpha, 
                   label='Median Curve', zorder=10)
    
    return ax


def _plot_band_mesh(points, triangles, ax, color, alpha, n_dims):
    """
    Plot a triangulated mesh band in either 2D or 3D.
    
    Internal helper function for rendering triangular meshes.
    
    Parameters:
    -----------
    points : np.ndarray
        Vertex coordinates of the mesh.
        Shape: (n_points, 2) for 2D or (n_points, 3) for 3D
    triangles : np.ndarray
        Triangle faces defined by point indices.
        Shape: (n_triangles, 3)
    ax : matplotlib.axes.Axes
        Matplotlib axes to plot on.
    color : str or tuple
        Color for the mesh.
    alpha : float
        Transparency of the mesh (0=transparent, 1=opaque).
    n_dims : int
        Dimensionality (2 or 3)
    
    Returns:
    --------
    ax : matplotlib.axes.Axes
        The matplotlib axes object used for plotting.
    """
    if n_dims == 2:
        # 2D plotting with Polygon patches
        for tri in triangles:
            poly = Polygon(points[tri], facecolor=color, edgecolor='none', alpha=alpha)
            ax.add_patch(poly)
    elif n_dims == 3:
        # 3D plotting with plot_trisurf
        ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], 
                       triangles=triangles, color=color, alpha=alpha)
    else:
        raise ValueError(f"n_dims must be 2 or 3, got {n_dims}")
    
    return ax


def matplotlib_plot_band(points, triangles, ax=None, color='red', alpha=1.0):
    """
    Plot a triangulated mesh band in either 2D or 3D.
    
    DEPRECATED: Use visualize_curve_boxplot() instead.
    This function is kept for backward compatibility.
    
    Automatically determines dimensionality from the points array and uses
    appropriate matplotlib functions for visualization.
    
    Parameters:
    ----------
    points : numpy.ndarray
        Vertex coordinates of the mesh.
        Shape: (n_points, 2) for 2D or (n_points, 3) for 3D
    triangles : numpy.ndarray
        Triangle faces defined by point indices.
        Shape: (n_triangles, 3)
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes to plot on. If None, creates new figure.
        For 3D plots, must be a 3D axes if provided.
    color : str or tuple, optional
        Color for the mesh. Default is 'red'.
    alpha : float, optional
        Transparency of the mesh (0=transparent, 1=opaque). Default is 1.0.
    
    Returns:
    -------
    ax : matplotlib.axes.Axes
        The matplotlib axes object used for plotting.
    
    Raises:
    ------
    ValueError
        If points array is not 2D or 3D.
    
    Examples:
    --------
    >>> # 2D band
    >>> points_2d = np.array([[0, 0], [1, 0], [0.5, 1]])
    >>> triangles = np.array([[0, 1, 2]])
    >>> ax = matplotlib_plot_band(points_2d, triangles, color='blue', alpha=0.5)
    
    >>> # 3D band
    >>> points_3d = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0.5]])
    >>> ax = matplotlib_plot_band(points_3d, triangles, color='green', alpha=0.7)
    """
    # Determine dimensionality from points shape
    if points.ndim != 2:
        raise ValueError(f"points must be a 2D array, got shape {points.shape}")
    
    n_dims = points.shape[1]
    
    if n_dims == 2:
        # 2D plotting
        if ax is None:
            fig, ax = plt.subplots()
        
        # Use Polygon patches to fill triangles
        for tri in triangles:
            poly = Polygon(points[tri], facecolor=color, edgecolor='none', alpha=alpha)
            ax.add_patch(poly)
            
    elif n_dims == 3:
        # 3D plotting
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        
        # Use plot_trisurf for 3D surface
        ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], 
                       triangles=triangles, color=color, alpha=alpha)
    else:
        raise ValueError(f"points must have 2 or 3 columns (dimensions), got {n_dims}")
    
    return ax


def matplotlib_curve_boxplot_vis(points, triangles, median_curve, curve_dim, ax=None, color_map='viridis', median_color='red', alpha=1.0):
    """
    DEPRECATED: Use visualize_curve_boxplot() instead.
    This function is kept for backward compatibility.
    """
    if curve_dim == 2:
        if ax is None:  
            fig, ax = plt.subplots()
        # ax.triplot(points[:, 0], points[:, 1], triangles, color='gray', alpha=0.5)
        ax.tripcolor(points[:, 0], points[:, 1], triangles, facecolors=np.ones(triangles.shape[0]), cmap=color_map, alpha=alpha)
        ax.plot(median_curve[:, 0], median_curve[:, 1], color=median_color, linewidth=2)
    elif curve_dim == 3:
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=triangles, cmap=color_map, alpha=alpha)
            ax.plot(median_curve[:, 0], median_curve[:, 1], median_curve[:, 2], color=median_color, linewidth=2)
        else:
            ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=triangles, cmap=color_map, alpha=alpha)
            ax.plot(median_curve[:, 0], median_curve[:, 1], median_curve[:, 2], color=median_color, linewidth=2)
    else:
        raise ValueError("curve_dim must be 2 or 3 for plotting.")

    return ax

