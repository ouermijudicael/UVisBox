
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

def visualize_cone_glyphs(mesh_3d, point_values=None, show_edges=True, 
                          glyph_color='lightblue', cmap='RdBu_r', ax=None):
    """
    Render 3D cone glyph mesh with pyvista.
    
    Parameters:
    -----------
    mesh_3d : dict
        From cone_glyphs_mesh()
    point_values : numpy.ndarray, optional
        Override mesh point_values for coloring
    show_edges : bool
        Show glyph edges (default: True)
    glyph_color : str
        Solid color when no point_values (default: 'lightblue')
    cmap : str
        Colormap for scalar coloring (default: 'RdBu_r')
    ax : pyvista.Plotter, optional
        Existing plotter to use
    
    Returns:
    --------
    plotter : pyvista.Plotter
        The plotter with drawn glyphs
    """
    points = mesh_3d['points']
    triangles = mesh_3d['polygons']
    points_values = point_values if point_values is not None else mesh_3d['point_values']
    
    triangles = np.hstack([np.full((triangles.shape[0], 1), 3), triangles])
    triangles_flat = triangles.reshape(-1)
    mesh = pv.PolyData(points, triangles_flat)
    
    if ax is None:
        ax = pv.Plotter()
    
    if points_values is not None:
        mesh.point_data['Values'] = points_values
        ax.add_mesh(mesh, scalars='Values', cmap=cmap, show_edges=show_edges)
    else:
        ax.add_mesh(mesh, color=glyph_color, show_edges=show_edges)
    
    ax.add_axes()
    ax.set_background('white')

    return ax

