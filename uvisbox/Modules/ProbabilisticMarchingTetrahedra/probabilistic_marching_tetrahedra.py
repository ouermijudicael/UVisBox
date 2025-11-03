import numpy as np
from .probabilistic_marching_tetrahedra_stats import crossing_probability_tetrahedra_monte_carlo
from .probabilistic_marching_tetrahedra_vis import pyvista_probabilistic_marching_tetrahedra_vis


def probabilistic_marching_tetrahedra(points, F, tetrahedra, isovalue, cross_prob=None, 
                                           opacity='linear', cmap='viridis', plotter=None):
    """
    Visualize the probabilistic marching tetrahedra result using PyVista.

    Parameters:
    -----------
        F : np.ndarray
            2D array of shape (n_points, n_ens) representing the scalar field with ensemble members.
        tetrahedra : np.ndarray
            2D array of shape (n_tetrahedra, 4) representing the tetrahedralization of the points.  
        isovalue : float
            The isovalue for which to compute the isosurface.
        cross_prob : np.ndarray, optional
            3D array of shape (n_x-1, n_y-1, n_z-1) with probabilities of isosurface presence in each cell.
            If None, it will be computed using probabilistic_marching_tetrahedra function.
        opacity : str or list, optional
            Opacity mapping for the volume rendering. Default is 'linear'.
        cmap : str, optional
            Colormap for the volume rendering. Default is 'viridis'.
            
    Returns:
    --------
        plotter : pyvista.Plotter
            The pyvista plotter with the visualized probabilistic isosurface.
    """

    if cross_prob is None:
        cross_prob = crossing_probability_tetrahedra_monte_carlo(F, tetrahedra, isovalue)

    plotter = pyvista_probabilistic_marching_tetrahedra_vis(points, tetrahedra, isovalue, cross_prob, 
                                           opacity=opacity, cmap=cmap, plotter=plotter) 
    
    return plotter