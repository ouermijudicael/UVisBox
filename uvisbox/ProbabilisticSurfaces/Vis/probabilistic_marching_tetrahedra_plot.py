import pyvista as pv
from pyvista import CellType
import numpy as np
from ..Stat.probabilistic_marching_tetrahedra import probabilistic_marching_tetrahedra



def probabilistic_marching_tetrahedra_plot(points, F, tetrahedra, isovalue, cross_prob=None, 
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
        cross_prob = probabilistic_marching_tetrahedra(F, tetrahedra, isovalue)

    n_cells = cross_prob.shape[0]
    n_points = F.shape[0]
    celltypes = np.full(n_cells, fill_value=CellType.TETRA, dtype=np.uint32)  

    if plotter is None:
        plotter = pv.Plotter()
            
    grid = pv.UnstructuredGrid(tetrahedra, celltypes, points)
    grid.cell_data["crossing_probability"] = cross_prob.flatten(order='F')

    plotter.add_volume(grid, scalars="crossing_probability", opacity=opacity, cmap=cmap)

    return plotter

