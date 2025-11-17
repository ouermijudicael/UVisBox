from .probabilistic_marching_squares_stats import probabilistic_marching_squares_summary_statistics
from .probabilistic_marching_squares_mesh import probabilistic_marching_squares_mesh
from .probabilistic_marching_squares_vis import visualize_probabilistic_marching_squares


def probabilistic_marching_squares(ensemble_images, isovalue, ax=None, colormap='viridis'):
    """
    Compute and visualize probabilistic marching squares.
    
    This function implements the complete stats->mesh->vis pipeline for probabilistic
    marching squares visualization. It calculates the probability of isocontour 
    presence in each cell and creates a matplotlib visualization.

    Parameters:
    -----------
        ensemble_images : np.ndarray
            3D array of shape (y_dim, x_dim, n_ensemble) representing the scalar 
            field with ensemble members.
        isovalue : float
            The isovalue for which to compute the isocontour.
        ax : matplotlib.axes.Axes, optional
            The axis to draw on. If None, a new figure and axis will be created.
        colormap : str, optional
            Colormap for the visualization. Default is 'viridis'.
    
    Returns:
    --------
        ax : matplotlib.axes.Axes
            The axis with the visualized probabilistic isocontour.
    """
    # Stats: compute level crossing probabilities
    summary_statistics = probabilistic_marching_squares_summary_statistics(ensemble_images, isovalue)
    
    # Mesh: identity function (no transformation needed)
    mesh_data = probabilistic_marching_squares_mesh(summary_statistics)
    
    # Vis: create visualization
    ax = visualize_probabilistic_marching_squares(mesh_data, ax=ax, colormap=colormap)
    
    return ax
