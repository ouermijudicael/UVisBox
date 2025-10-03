from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
from ..Mesh.curve_banddepth_meshing import curve_banddepth_meshing
from ....Core.BandDepths.curve_banddepth import curve_banddepths

def curve_banddepth_plot(curves, depths=None, percentile=50, ax=None):
    """
    Create a curve band depth plot.
    
    Parameters:
    ----------
    curves : numpy.ndarray
        3D array of shape (n_curves, n_steps, n_dims) containing curve data
    depths : numpy.ndarray, optional
        1D array of precomputed band depths of shape (n_curves,). If None, band depths will be computed.
    percentile : float, optional
        Percentile for the band to be highlighted. Default is 50 (median band).
    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes object to plot on. If None, a new figure and axes will be created.
    
    Returns:
    -------
    ax: matplotlib.axes.Axes
        The Axes object with the curve band depth plot.
    """
    print(f"cuver_banddepth_plot called with curves shape: {curves.shape}, percentile: {percentile}")
    if depths is None:
        
        n_curves = curves.shape[0]
        indices = list(combinations(range(n_curves), 2))
        print("Calculating curve band depths...")
        depths = curve_banddepths(curves, indices)
        print("Curve band depths calculated.")

    curve_dim = curves.shape[2]
    # sort the curves by the depth. order them from deepest to shallowest
    sorted_indices = np.argsort(depths)[::-1]
    sorted_curves = curves[sorted_indices]
    # sorted_curves = sorted_curves[:,:200,:]

    # create figure if no ax is assigned
    if ax is None:
        fig, ax = plt.subplots()

    # build the band mesh for the specified percentile
    points, triangles = curve_banddepth_meshing(sorted_curves, percentile=percentile)
     # highlight the median curve in red
    median_curve = sorted_curves[0]
    # plot the band mesh using trisurf or tripcolor
    if curve_dim == 2:
        if ax is None:  
            fig, ax = plt.subplots()
        # ax.triplot(points[:, 0], points[:, 1], triangles, color='gray', alpha=0.5)
        ax.tripcolor(points[:, 0], points[:, 1], triangles, facecolors=np.ones(triangles.shape[0]), cmap='viridis', alpha=1.0)
        ax.plot(median_curve[:, 0], median_curve[:, 1], color='red', linewidth=2)
        # plot 50% curves in light gray
        # for curve in sorted_curves[:len(sorted_curves)//2]:
            # ax.plot(curve[:, 0], curve[:, 1], color='lightgray', linewidth=1)
    elif curve_dim == 3:
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=triangles, cmap='viridis', alpha=0.5)
            ax.plot(median_curve[:, 0], median_curve[:, 1], median_curve[:, 2], color='red', linewidth=2)
        else:
            ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=triangles, cmap='viridis', alpha=1.0)
            ax.plot(median_curve[:, 0], median_curve[:, 1], median_curve[:, 2], color='red', linewidth=2)
    else:
        raise ValueError("curve_dim must be 2 or 3 for plotting.")    
    
   
    return ax

