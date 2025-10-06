
import numpy as np
import matplotlib.pyplot as plt

def uncertainty_squid_glyphs_2D_plot(glyphs_points, glyphs_polygons, ax=None):
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
