import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from uvisbox.Core.CommonInterface import BoxplotStyleConfig

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False


def visualize_curve_boxplot(mesh_data, boxplot_style=None, ax=None):
    """
    Visualize curve boxplot from mesh data.
    
    This function creates a visualization of the curve boxplot using the mesh data 
    output from the mesh pipeline. For 2D curves, it uses matplotlib. For 3D curves, 
    it uses PyVista if available, otherwise falls back to matplotlib.
    
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
    ax : matplotlib.axes.Axes or pyvista.Plotter, optional
        Axes or plotter to use for visualization. Can be:
        - matplotlib.axes.Axes for 2D or 3D matplotlib rendering
        - pyvista.Plotter for 3D PyVista rendering
        - None: creates appropriate visualization object automatically
    
    Returns:
    --------
    ax : matplotlib.axes.Axes or pyvista.Plotter
        The visualization object (matplotlib axes for 2D, PyVista plotter for 3D).
    
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
    
    # Route to appropriate backend
    if n_dims == 2:
        return _visualize_curve_boxplot_2d_matplotlib(mesh_data, boxplot_style, ax)
    elif n_dims == 3:
        # Check if ax is a PyVista plotter
        if PYVISTA_AVAILABLE and ax is not None and hasattr(ax, 'add_mesh'):
            # It's a PyVista plotter
            return _visualize_curve_boxplot_3d_pyvista(mesh_data, boxplot_style, ax)
        elif PYVISTA_AVAILABLE and ax is None:
            # No ax provided, create PyVista plotter for 3D
            ax = pv.Plotter()
            return _visualize_curve_boxplot_3d_pyvista(mesh_data, boxplot_style, ax)
        else:
            # Fall back to matplotlib 3D (either ax is matplotlib axes or PyVista not available)
            return _visualize_curve_boxplot_3d_matplotlib(mesh_data, boxplot_style, ax)
    else:
        raise ValueError(f"Unsupported curve dimension: {n_dims}. Must be 2 or 3.")


def _visualize_curve_boxplot_2d_matplotlib(mesh_data, boxplot_style, ax=None):
    """
    Visualize 2D curve boxplot using matplotlib.
    
    Internal helper function for 2D matplotlib rendering.
    
    Parameters:
    -----------
    mesh_data : dict
        Mesh data from curve_boxplot_mesh
    boxplot_style : BoxplotStyleConfig
        Style configuration
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes to plot on
    
    Returns:
    --------
    ax : matplotlib.axes.Axes
        The matplotlib axes object with the plot
    """
    # Create figure/axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get colors from colormap
    colors = boxplot_style.get_percentile_colors()
    percentiles = boxplot_style.percentiles
    
    # Sort percentiles in descending order for proper plotting (largest first)
    sorted_percentile_indices = np.argsort(percentiles)[::-1]
    sorted_percentiles = [percentiles[i] for i in sorted_percentile_indices]
    sorted_colors = [colors[i] for i in sorted_percentile_indices]

    # Plot each percentile band from largest to smallest
    if boxplot_style.show_bands:
        for percentile, color in zip(sorted_percentiles, sorted_colors):
            mesh_key = f'{int(percentile)}_percentile_mesh'
            if mesh_key in mesh_data['percentile_meshes']:
                points, triangles = mesh_data['percentile_meshes'][mesh_key]
                _plot_band_mesh_2d(points, triangles, ax=ax, color=color, alpha=1.0)

    # Plot outliers
    if boxplot_style.show_outliers and mesh_data['outliers'].shape[0] > 0:
        outliers = mesh_data['outliers']
        for idx in range(len(outliers)):
            outlier_curve = outliers[idx]
            label = 'Outliers' if idx == 0 else None
            ax.plot(outlier_curve[:, 0], outlier_curve[:, 1], 
                   color=boxplot_style.outliers_color, 
                   linewidth=boxplot_style.outliers_width, 
                   alpha=boxplot_style.outliers_alpha,
                   label=label, zorder=5)
    
    # Plot median curve
    if boxplot_style.show_median:
        median_curve = mesh_data['median_curve']
        ax.plot(median_curve[:, 0], median_curve[:, 1], 
               color=boxplot_style.median_color, 
               linewidth=boxplot_style.median_width, 
               alpha=boxplot_style.median_alpha, 
               label='Median Curve', zorder=10)
    
    return ax


def _visualize_curve_boxplot_3d_matplotlib(mesh_data, boxplot_style, ax=None):
    """
    Visualize 3D curve boxplot using matplotlib 3D.
    
    Internal helper function for 3D matplotlib rendering.
    
    Parameters:
    -----------
    mesh_data : dict
        Mesh data from curve_boxplot_mesh
    boxplot_style : BoxplotStyleConfig
        Style configuration
    ax : matplotlib.axes.Axes, optional
        Matplotlib 3D axes to plot on
    
    Returns:
    --------
    ax : matplotlib.axes.Axes
        The matplotlib 3D axes object with the plot
    """
    # Create figure/axes if not provided
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Get colors from colormap
    colors = boxplot_style.get_percentile_colors()
    percentiles = boxplot_style.percentiles
    
    # Sort percentiles in descending order
    sorted_percentile_indices = np.argsort(percentiles)[::-1]
    sorted_percentiles = [percentiles[i] for i in sorted_percentile_indices]
    sorted_colors = [colors[i] for i in sorted_percentile_indices]
    
    # Plot each percentile band
    if boxplot_style.show_bands:
        for percentile, color in zip(sorted_percentiles, sorted_colors):
            mesh_key = f'{int(percentile)}_percentile_mesh'
            if mesh_key in mesh_data['percentile_meshes']:
                points, triangles = mesh_data['percentile_meshes'][mesh_key]
                _plot_band_mesh_3d(points, triangles, ax=ax, color=color, alpha=1.0)

    # Plot outliers
    if boxplot_style.show_outliers and mesh_data['outliers'].shape[0] > 0:
        outliers = mesh_data['outliers']
        for idx in range(len(outliers)):
            outlier_curve = outliers[idx]
            label = 'Outliers' if idx == 0 else None
            ax.plot(outlier_curve[:, 0], outlier_curve[:, 1], outlier_curve[:, 2],
                   color=boxplot_style.outliers_color, 
                   linewidth=boxplot_style.outliers_width, 
                   alpha=boxplot_style.outliers_alpha,
                   label=label, zorder=5)
    
    # Plot median curve
    if boxplot_style.show_median:
        median_curve = mesh_data['median_curve']
        ax.plot(median_curve[:, 0], median_curve[:, 1], median_curve[:, 2],
               color=boxplot_style.median_color, 
               linewidth=boxplot_style.median_width, 
               alpha=boxplot_style.median_alpha, 
               label='Median Curve', zorder=10)
    
    return ax


def _visualize_curve_boxplot_3d_pyvista(mesh_data, boxplot_style, ax):
    """
    Visualize 3D curve boxplot using PyVista.
    
    Internal helper function for 3D PyVista rendering with opacity based on percentile.
    
    Parameters:
    -----------
    mesh_data : dict
        Mesh data from curve_boxplot_mesh
    boxplot_style : BoxplotStyleConfig
        Style configuration
    ax : pyvista.Plotter
        PyVista plotter to use
    
    Returns:
    --------
    ax : pyvista.Plotter
        The PyVista plotter object with the visualization
    """
    if not PYVISTA_AVAILABLE:
        raise ImportError("PyVista is required for 3D visualization but is not installed. "
                         "Install it with: pip install pyvista")
    
    # Get colors from colormap
    colors = boxplot_style.get_percentile_colors()
    percentiles = boxplot_style.percentiles
    
    # Sort percentiles in descending order
    sorted_percentile_indices = np.argsort(percentiles)[::-1]
    sorted_percentiles = [percentiles[i] for i in sorted_percentile_indices]
    sorted_colors = [colors[i] for i in sorted_percentile_indices]
    
    # Plot each percentile band with opacity based on percentile
    if boxplot_style.show_bands:
        for percentile, color in zip(sorted_percentiles, sorted_colors):
            mesh_key = f'{int(percentile)}_percentile_mesh'
            if mesh_key in mesh_data['percentile_meshes']:
                points, triangles = mesh_data['percentile_meshes'][mesh_key]

                # Calculate opacity: (1 - (percentile / 100))^2
                opacity = (1 - (percentile / 100)) ** 2

                # Create PyVista mesh
                faces = np.hstack([np.full((triangles.shape[0], 1), 3), triangles]).astype(int)
                poly_mesh = pv.PolyData(points, faces)

                # Add mesh to plotter with label
                ax.add_mesh(poly_mesh, color=color, opacity=opacity,
                               smooth_shading=True, show_edges=False,
                               label=f'{int(percentile)}th percentile')

    # Plot outliers
    if boxplot_style.show_outliers and mesh_data['outliers'].shape[0] > 0:
        outliers = mesh_data['outliers']
        for idx, outlier_curve in enumerate(outliers):
            # Use Spline for smooth curve rendering
            # Add label only to first outlier to avoid duplicate legend entries
            label = 'Outliers' if idx == 0 else None
            curve_line = pv.Spline(outlier_curve[:, 0:3], n_points=outlier_curve.shape[0])
            ax.add_mesh(curve_line, 
                           color=boxplot_style.outliers_color,
                           line_width=boxplot_style.outliers_width,
                           opacity=boxplot_style.outliers_alpha,
                           label=label)
    
    # Plot median curve
    if boxplot_style.show_median:
        median_curve = mesh_data['median_curve']
        median_line = pv.Spline(median_curve[:, 0:3], n_points=median_curve.shape[0])
        ax.add_mesh(median_line,
                       color=boxplot_style.median_color,
                       line_width=boxplot_style.median_width,
                       opacity=boxplot_style.median_alpha,
                       label='Median Curve')
    
    return ax


def _plot_band_mesh_2d(points, triangles, ax, color, alpha):
    """
    Plot a triangulated mesh band in 2D using matplotlib.
    
    Internal helper function for rendering 2D triangular meshes.
    
    Parameters:
    -----------
    points : np.ndarray
        Vertex coordinates of the mesh. Shape: (n_points, 2)
    triangles : np.ndarray
        Triangle faces defined by point indices. Shape: (n_triangles, 3)
    ax : matplotlib.axes.Axes
        Matplotlib axes to plot on
    color : str or tuple
        Color for the mesh
    alpha : float
        Transparency of the mesh (0=transparent, 1=opaque)
    
    Returns:
    --------
    ax : matplotlib.axes.Axes
        The matplotlib axes object used for plotting
    """
    for tri in triangles:
        poly = Polygon(points[tri], facecolor=color, edgecolor='none', alpha=alpha)
        ax.add_patch(poly)
    return ax


def _plot_band_mesh_3d(points, triangles, ax, color, alpha):
    """
    Plot a triangulated mesh band in 3D using matplotlib.
    
    Internal helper function for rendering 3D triangular meshes.
    
    Parameters:
    -----------
    points : np.ndarray
        Vertex coordinates of the mesh. Shape: (n_points, 3)
    triangles : np.ndarray
        Triangle faces defined by point indices. Shape: (n_triangles, 3)
    ax : matplotlib.axes.Axes
        Matplotlib 3D axes to plot on
    color : str or tuple
        Color for the mesh
    alpha : float
        Transparency of the mesh (0=transparent, 1=opaque)
    
    Returns:
    --------
    ax : matplotlib.axes.Axes
        The matplotlib 3D axes object used for plotting
    """
    ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], 
                   triangles=triangles, color=color, alpha=alpha)
    return ax


