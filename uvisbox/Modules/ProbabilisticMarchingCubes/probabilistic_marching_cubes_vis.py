import pyvista as pv


def visualize_probabilistic_marching_cubes(mesh_data, plotter=None, opacity='linear', colormap='viridis'):
    """
    Visualize probabilistic marching cubes using PyVista volume rendering.
    
    This function creates a 3D visualization of the crossing probabilities using
    PyVista's volume rendering capabilities.
    
    Parameters:
    -----------
        mesh_data : np.ndarray
            3D array of shape (n_z-1, n_y-1, n_x-1) with probabilities of isosurface 
            presence in each cell.
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
    if plotter is None:
        plotter = pv.Plotter()

    grid_dimensions = mesh_data.shape
    origin = (0, 0, 0)
    spacing = (1, 1, 1)
    grid = pv.ImageData(dimensions=grid_dimensions, origin=origin, spacing=spacing)
    # Add the crossing probability to the cell data
    grid.point_data["crossing_probability"] = mesh_data.flatten(order='F')

    # Volume rendering of the grid
    plotter.add_volume(grid, scalars="crossing_probability", opacity=opacity, cmap=colormap)
    plotter.add_axes()

    return plotter
