from .probabilistic_marching_cubes_stats import probabilistic_marching_cube_summary_statistics
from .probabilistic_marching_cubes_mesh import probabilistic_marching_cube_mesh
from .probabilistic_marching_cubes_vis import visualize_probabilistic_marching_cube


def probabilistic_marching_cube(ensemble_images, isovalue, plotter=None, opacity='linear', colormap='viridis'):
    """
    Compute and visualize probabilistic marching cubes.
    
    This function implements the complete stats->mesh->vis pipeline for probabilistic
    marching cubes visualization. It calculates the probability of isosurface 
    presence in each cell and creates a PyVista volume rendering visualization.

    Parameters:
    -----------
        ensemble_images : np.ndarray
            4D array of shape (n_z, n_y, n_x, n_ensemble) representing the scalar 
            field with ensemble members.
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
    summary_statistics = probabilistic_marching_cube_summary_statistics(ensemble_images, isovalue)
    
    # Mesh: identity function (no transformation needed)
    mesh_data = probabilistic_marching_cube_mesh(summary_statistics)
    
    # Vis: create visualization
    plotter = visualize_probabilistic_marching_cube(mesh_data, plotter=plotter, opacity=opacity, colormap=colormap)
    
    return plotter
