import pyvista as pv
import numpy as np
from pyvista import CellType


def visualize_probabilistic_marching_tetrahedra(mesh_data, points, tetrahedral_mesh, plotter=None, opacity='linear', colormap='viridis'):
    """
    Visualize probabilistic marching tetrahedra using PyVista volume rendering.
    
    This function creates a 3D visualization of the crossing probabilities using
    PyVista's volume rendering capabilities for tetrahedral meshes.

    Parameters:
    -----------
        mesh_data : np.ndarray
            1D array of shape (n_tetrahedra,) with probabilities of isosurface presence 
            in each tetrahedron.
        points : np.ndarray
            2D array of shape (n_points, 3) representing the coordinates of the points.
        tetrahedral_mesh : np.ndarray
            2D array of shape (n_tetrahedra, 4) representing the tetrahedralization of the points.
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
    n_cells = mesh_data.shape[0]
    celltypes = np.full(n_cells, fill_value=CellType.TETRA, dtype=np.uint32)  

    # append a column of 4s to the tetrahedral mesh to indicate that each cell is a tetrahedron
    tetrahedra = np.hstack((np.full((tetrahedral_mesh.shape[0], 1), 4), tetrahedral_mesh))


    if plotter is None:
        plotter = pv.Plotter()
            
    grid = pv.UnstructuredGrid(tetrahedra, celltypes, points)
    grid.cell_data["crossing_probability"] = mesh_data.flatten(order='F')

    plotter.add_volume(grid, scalars="crossing_probability", opacity=opacity, cmap=colormap)

    return plotter
