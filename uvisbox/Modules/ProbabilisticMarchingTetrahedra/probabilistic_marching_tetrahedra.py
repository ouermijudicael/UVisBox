from .probabilistic_marching_tetrahedra_stats import probabilistic_marching_tetrahedra_summary_statistics
from .probabilistic_marching_tetrahedra_mesh import probabilistic_marching_tetrahedra_mesh
from .probabilistic_marching_tetrahedra_vis import visualize_probabilistic_marching_tetrahedra


def probabilistic_marching_tetrahedra(ensemble_data, tetrahedral_mesh, points, isovalue, plotter=None, opacity='linear', colormap='viridis'):
    """
    Compute and visualize probabilistic marching tetrahedra.
    
    This function implements the complete stats->mesh->vis pipeline for probabilistic
    marching tetrahedra visualization. It calculates the probability of isosurface 
    presence in each tetrahedron and creates a PyVista volume rendering visualization.

    Parameters:
    -----------
        ensemble_data : np.ndarray
            2D array of shape (n_points, n_ensemble) where each column is a realization 
            and each row corresponds to a vertex in the tetrahedral mesh.
        tetrahedral_mesh : np.ndarray
            2D array of shape (n_tetrahedra, 4) representing the tetrahedralization of the points.
        points : np.ndarray
            2D array of shape (n_points, 3) with point coordinates.
        isovalue : float
            The isovalue for which to compute the isosurface.
        plotter : pyvista.Plotter, optional
            An existing PyVista plotter to add the volume rendering to. If None, 
            a new plotter is created.
        opacity : str or list, optional
            Opacity mapping for the volume rendering. Default is 'linear'.
        colormap : str, optional
            Colormap for the volume rendering. Default is 'viridis'.
            
    Returns:
    --------
        plotter : pyvista.Plotter
            The pyvista plotter with the visualized probabilistic isosurface.
    """
    # Stats: compute level crossing probabilities
    summary_statistics = probabilistic_marching_tetrahedra_summary_statistics(ensemble_data, tetrahedral_mesh, isovalue)
    
    # Mesh: identity function (no transformation needed)
    mesh_data = probabilistic_marching_tetrahedra_mesh(summary_statistics)
    
    # Vis: create visualization
    plotter = visualize_probabilistic_marching_tetrahedra(mesh_data, points, tetrahedral_mesh, plotter=plotter, opacity=opacity, colormap=colormap)
    
    return plotter
