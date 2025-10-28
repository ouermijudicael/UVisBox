from .probabilistic_marching_squares_stats import crossing_prob_squares_mc
from .probabilistic_marching_squares_vis import matplotlib_probabilistic_marching_squares_vis

def probabilistic_marching_squares(ensemble_images, isovalue, prob_contour=None, cmap='viridis', ax=None):
    """
    Visualize the probabilistic marching squares result using matplotlib.

    Parameters:
    -----------
        ensemble_images : np.ndarray
            3D array of shape [y, x, n_ens] representing the scalar field with ensemble members.
        isovalue : float
            The isovalue for which to compute the isocontour.
        prob_contour : np.ndarray, optional
            2D array of shape (y-1, x-1) with probabilities of contour presence in each cell.
            If None, it will be computed using probabilistic_marching_squares function.
        cmap : str, optional
            Colormap for the visualization. Default is 'viridis'.
        ax : matplotlib axis, optional
            The axis to draw on. If None, a new figure and axis will be created.
    
    Returns:
    --------
        ax : matplotlib axis
            The axis with the visualized probabilistic isocontour.
    """
    if prob_contour is None:
        prob_contour = crossing_prob_squares_mc(ensemble_images, isovalue)

    ax = matplotlib_probabilistic_marching_squares_vis(prob_contour, cmap, ax)

    return ax