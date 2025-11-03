from .probabilistic_marching_triangles_stats import probabilistic_marching_triangles_summary_statistics
from .probabilistic_marching_triangles_mesh import probabilistic_marching_triangles_mesh
from .probabilistic_marching_triangles_vis import visualize_probabilistic_marching_triangles


def probabilistic_marching_triangles(ensemble_data, triangle_mesh, points, isovalue, ax=None, colormap='viridis'):
    """
    Compute and visualize probabilistic marching triangles.
    
    This function implements the complete stats->mesh->vis pipeline for probabilistic
    marching triangles visualization. It calculates the probability of isocontour 
    presence in each triangle and creates a matplotlib visualization.

    Parameters:
    -----------
        ensemble_data : np.ndarray
            2D array of shape (n_points, n_ensemble) where each column is a realization 
            and each row corresponds to a vertex in the triangular mesh.
        triangle_mesh : np.ndarray
            2D array of shape (n_triangles, 3) with triangle indices.
        points : np.ndarray
            2D array of shape (n_points, 2) with point coordinates.
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
    summary_statistics = probabilistic_marching_triangles_summary_statistics(ensemble_data, triangle_mesh, isovalue)
    
    # Mesh: identity function (no transformation needed)
    mesh_data = probabilistic_marching_triangles_mesh(summary_statistics)
    
    # Vis: create visualization
    ax = visualize_probabilistic_marching_triangles(mesh_data, points, triangle_mesh, ax=ax, colormap=colormap)
    
    return ax
