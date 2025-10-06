from .crossing_prob_squares_mc import crossing_prob_squares_mc
from .probabilistic_marching_squares_plot import matplotlib_probabilistic_marching_squares_vis
def plot(F, isovalue, prob_contour=None, cmap='viridis', ax=None):
    """
    Visualize the probabilistic marching squares result using matplotlib.

    Parameters:
    -----------
        F : np.ndarray
            3D array of shape (n, m, n_ens) representing the scalar field with ensemble members.
        isovalue : float
            The isovalue for which to compute the isocontour.
        prob_contour : np.ndarray, optional
            2D array of shape (n-1, m-1) with probabilities of contour presence in each cell.
            If None, it will be computed using probabilistic_marching_squares function.
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
        prob_contour = crossing_prob_squares_mc(F, isovalue)
    ax = matplotlib_probabilistic_marching_squares_vis(prob_contour, cmap, ax)
    return ax