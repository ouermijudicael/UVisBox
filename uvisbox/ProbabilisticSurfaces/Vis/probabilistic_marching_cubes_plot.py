import pyvista as pv
import numpy as np
from ..Stat.probabilistic_marching_cubes import probabilistic_marching_cubes



def probabilistic_marching_cubes_plot(F, isovalue, cross_prob=None, opacity='linear', cmap='viridis'):
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
    Returns:
    --------
        plotter : pyvista.Plotter
            The pyvista plotter with the visualized probabilistic isosurface.
    """

    if cross_prob is None:
        cross_prob = probabilistic_marching_cubes(F, isovalue)

    grid_dimensions = cross_prob.shape
    origin = (0, 0, 0)
    spacing = (1, 1, 1)
    grid = pv.ImageData(dimensions=grid_dimensions, origin=origin, spacing=spacing)
    # Add the cross probability to the cell data
    grid.point_data["cross_prob"] = cross_prob.flatten(order='F')

    # Volume rendering of the grid
    plotter = pv.Plotter()
    plotter.add_volume(grid, scalars="cross_prob", opacity=opacity, cmap=cmap)
    plotter.add_axes()

    return plotter