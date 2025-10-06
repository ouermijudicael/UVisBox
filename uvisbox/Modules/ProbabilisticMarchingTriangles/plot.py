import numpy as np
from .Stats.crossing_prob_triangles_mc import crossing_prob_triangles_mc
from .Vis.probabilistic_marching_triangles_plot import matplotlib_probabilistic_marching_triangles_vis

def plot(F, points, triangles, isovalue, prob_contour=None, cmap='viridis', ax=None):
    """
    Visualize the probabilistic marching triangles result using matplotlib.

    Parameters:
    -----------
        F : np.ndarray
            2D array of shape (n_points, n_ens) representing the scalar field with ensemble members.
        points : np.ndarray
            2D array of shape (n_points, 2) with point coordinates.
        triangles : np.ndarray
            2D array of shape (n_triangles, 3) with triangle indices.
        isovalue : float
            The isovalue for which to compute the isocontour.
        prob_contour : np.ndarray, optional
            1D array with probabilities of contour presence in each triangle.
            If None, it will be computed using probabilistic_marching_triangles function.
        cmap : str, optional
            Colormap for the probability map. Default is 'viridis'.
        ax : matplotlib axis, optional
            The axis to draw on. If None, a new figure and axis will be created.
    Returns:
    --------
        ax : matplotlib axis
            The axis with the visualized probabilistic isocontour.
    """
    if prob_contour is None:
        prob_contour = crossing_prob_triangles_mc(F, triangles, isovalue)
    ax = matplotlib_probabilistic_marching_triangles_vis(points,triangles, prob_contour, cmap, ax)
    return ax
