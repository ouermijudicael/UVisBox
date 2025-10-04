import .Stats.stats.functional_banddepth as functional_banddepth
import .Mesh.mesh as mesh
import .Vis.vis as vis
import numpy as np

def plot(curves, curve_depths=None, percentile=50, ax=None, color_map='viridis', median_color='red', alpha=1.0):
    """
    Create a curve band depth plot using the provided curves and their depths.

    Parameters:
    -----------
    curves : numpy.ndarray
        3D array of shape (n_curves, n_steps, n_dims) containing curve data
    curve_depths : numpy.ndarray
        1D array of shape (n_curves,) containing the depth of each curve
    percentile : float
        The percentile for the band to be highlighted (default is 50)
    ax : matplotlib.axes.Axes, optional
        The axes to plot on (default is None, which creates a new figure)
    color_map : str
        The colormap to use for the mesh (default is 'viridis')
    median_color : str
        The color to use for the median curve (default is 'red')
    alpha : float
        The transparency level for the mesh (default is 1.0)

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axes with the plot.
    """

    if curve_depths is None:
        curve_depths = functional_banddepth(curves)

    # sort the curves by the depth. order them from deepest to shallowest
    sorted_indices = np.argsort(curve_depths)[::-1]
    sorted_curves = curves[sorted_indices]
    curve_dim = curves.shape[2]
    
    # build the band mesh for the specified percentile
    points, triangles = mesh(sorted_curves, percentile=percentile)
    
    # highlight the median curve in red
    median_curve = sorted_curves[0]

    # plot the band mesh using trisurf or tripcolor
    ax = vis(points, triangles, median_curve, curve_dim, ax=ax, 
             color_map=color_map, median_color=median_color, alpha=alpha)
    
    return ax