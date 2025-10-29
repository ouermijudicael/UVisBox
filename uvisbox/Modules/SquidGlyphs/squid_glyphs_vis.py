
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv


def render_squid_glyph_2d(mesh_2d, ax=None):
    """
    Render 2D squid glyph mesh with matplotlib.
    
    Parameters:
    -----------
    mesh_2d : dict
        From build_squid_glyph_mesh_2d()
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


def render_squid_glyph_3d(mesh_3d, point_values=None, show_edges=True, 
                          glyph_color='lightblue', cmap='RdBu_r', ax=None):
    """
    Render 3D squid glyph mesh with pyvista.
    
    Parameters:
    -----------
    mesh_3d : dict
        From build_squid_glyph_mesh_3d()
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


def matplotlib_uncertainty_squid_glyphs_2D_vis(glyphs_points, glyphs_polygons, ax=None):
    """
    Plots the squid glyphs in 2D using matplotlib.
    
    Parameters:
    ----------
    glyphs_points : numpy.ndarray
        Array of shape (k, 2) The points of the squid glyphs.
    glyphs_polygons : numpy.ndarray
        Array of shape (m, 3) The polygons of the squid glyphs.
    ax : matplotlib axis
        The axis to draw on. If None, a new figure and axis will be created.
    
    Returns:
    -------
    ax : matplotlib axis
        The axis with the drawn squid glyphs.
    """
   
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    tri_colors = np.ones((glyphs_polygons.shape[0]))*0.8
    ax.tripcolor(glyphs_points[:, 0], glyphs_points[:, 1], glyphs_polygons, facecolors=tri_colors, cmap='RdBu_r',)
   
    return ax

    
def pyvista_uncertainty_squid_glyphs_3D_vis(points, triangles, points_values=None, ax=None, show_edges=True, 
                                            glyph_color='lightblue', cmap='RdBu_r'):
    """
    Plots the 3D squid glyphs using pyvista.
    
    Parameters:
    -----------
    points : numpy.ndarray
        Array of shape (m, 3) The points of the squid glyphs.
    triangles : numpy.ndarray
        Array of shape (n, 3) The triangle connectivity of the squid glyphs.
    points_values : numpy.ndarray, optional
        Array of shape (m,) The values associated with each point for coloring.
    ax : pyvista.Plotter, optional
        The pyvista plotter to use. If None, a new plotter will be created.
    show_edges : bool, optional
        Whether to show edges of the glyphs. Default is True.
    glyph_color : str, optional
        The color of the glyphs. Default is 'lightblue'.

    Returns:
    --------
    ax : pyvista.Plotter
        The pyvista plotter with the drawn squid glyphs.
    """
    triangles = np.hstack([np.full((triangles.shape[0], 1), 3), triangles])
    triangles_flat = triangles.reshape(-1)
    mesh = pv.PolyData(points, triangles_flat) # 
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


