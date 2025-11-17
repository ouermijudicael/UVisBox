
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv


def visualize_squid_glyphs_2d(mesh_2d, ax=None):
    """
    Render 2D squid glyph mesh with matplotlib.
    
    Parameters:
    -----------
    mesh_2d : dict
        From squid_glyphs_2d_mesh()
    ax : matplotlib.Axes, optional
        Existing axis to draw on
    
    Returns:
    --------
    ax : matplotlib.Axes
        The axis with drawn glyphs
    """
    points = mesh_2d['points']
    polygons = mesh_2d['polygons']
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    tri_colors = np.ones((polygons.shape[0])) * 0.8
    ax.tripcolor(points[:, 0], points[:, 1], polygons, facecolors=tri_colors, cmap='RdBu_r')
   
    return ax


def visualize_squid_glyphs_3d(mesh_3d, point_values=None, show_edges=True, 
                          glyph_color='lightblue', cmap='RdBu_r', ax=None):
    """
    Render 3D squid glyph mesh with pyvista.
    
    Parameters:
    -----------
    mesh_3d : dict
        From squid_glyphs_3d_mesh()
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
    points_values = point_values if point_values is not None else mesh_3d.get('point_values')
    
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

