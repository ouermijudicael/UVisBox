
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

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


