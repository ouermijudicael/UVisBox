from .probabilistic_marching_cubes_stats import crossing_probability_cubes_monte_carlo
from .probabilistic_marching_cubes_vis import pyvista_probabilistic_marching_cubes_vis

def probabilistic_marching_cubes(F, isovalue, cross_prob=None, opacity='linear', cmap='viridis',   
                                      plotter=None):
    """
    Visualize the probabilistic marching cubes result using PyVista.

    Parameters:
    -----------
        F : np.ndarray
            4D array of shape (n_x, n_y, n_z, n_ens) representing the scalar field with ensemble members.
        isovalue : float
            The isovalue for which to compute the isosurface.
        cross_prob : np.ndarray, optional
            3D array of shape (n_x-1, n_y-1, n_z-1) with probabilities of isosurface presence in each cell.
            If None, it will be computed using probabilistic_marching_cubes function.
        opacity : str or list, optional
            Opacity mapping for the volume rendering. Default is 'linear'.  
        cmap : str, optional
            Colormap for the volume rendering. Default is 'viridis'.
        plotter : pyvista.Plotter, optional
            An existing PyVista plotter to add the volume rendering to. If None, a new plotter is created.
            
    Returns:
    --------
        plotter : pyvista.Plotter
            The pyvista plotter with the visualized probabilistic isosurface.
    """

    if cross_prob is None:
        cross_prob = crossing_probability_cubes_monte_carlo(F, isovalue)

    plotter = pyvista_probabilistic_marching_cubes_vis(cross_prob, opacity, cmap, plotter)
   

    return plotter