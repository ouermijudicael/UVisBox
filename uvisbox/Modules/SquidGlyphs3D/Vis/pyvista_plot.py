
import numpy as np
import pyvista as pv

def uncertainty_squid_glyphs_3D_plot(points, triangles, ax=None, show_edges=True, glyph_color='lightblue'):
    """
    Plots the 3D squid glyphs using pyvista.
    
    Parameters:
    -----------
    points : numpy.ndarray
        Array of shape (m, 3) The points of the squid glyphs.
    triangles : numpy.ndarray
        Array of shape (n, 3) The triangle connectivity of the squid glyphs.
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
    ax.add_mesh(mesh, color=glyph_color, show_edges=show_edges)
    ax.add_axes()
    ax.set_background('white')

    return ax
