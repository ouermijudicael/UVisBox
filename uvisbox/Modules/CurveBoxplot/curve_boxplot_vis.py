import matplotlib.pyplot as plt
import numpy as np

def matplotlib_plot_band(points, triangles, ax=None, color='red', alpha=1.0):
    """
    Plot a triangulated mesh band in either 2D or 3D.
    
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
        from matplotlib.patches import Polygon
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

