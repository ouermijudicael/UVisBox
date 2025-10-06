import matplotlib.pyplot as plt

def matplotlib_probabilistic_marching_triangles_vis(points, triangles, prob_contour, cmap='viridis', ax=None):
    """
    Visualize the probability map of isocontour presence using matplotlib.

    Parameters:
    -----------
        points : np.ndarray
            2D array of shape (n_points, 2) representing the coordinates of the points.
        triangles : np.ndarray
            2D array of shape (n_triangles, 3) representing the triangulation of the points.
        prob_contour : np.ndarray
            1D array of shape (n_triangles,) with probabilities of contour presence in each triangle.
        cmap : str, optional
            Colormap for the probability map. Default is 'viridis'.
        ax : matplotlib axis, optional
            The axis to draw on. If None, a new figure and axis will be created.

    Returns:    
    --------
        ax : matplotlib axis
            The axis with the visualized probabilistic isocontour.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    tpc = ax.tripcolor(points[:, 0], points[:, 1], triangles, prob_contour, shading='flat', cmap=cmap)
    plt.colorbar(tpc, ax=ax, label='Probability of Isocontour')
    ax.set_title('Probabilistic Marching Triangles')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    return ax