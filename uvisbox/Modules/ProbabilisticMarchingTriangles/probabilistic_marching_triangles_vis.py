import matplotlib.pyplot as plt


def visualize_probabilistic_marching_triangle(mesh_data, points, triangle_mesh, ax=None, colormap='viridis'):
    """
    Visualize probabilistic marching triangles using matplotlib.
    
    This function creates a 2D visualization of the crossing probabilities using
    matplotlib's tripcolor for triangular meshes.

    Parameters:
    -----------
        mesh_data : np.ndarray
            1D array of shape (n_triangles,) with probabilities of contour presence 
            in each triangle.
        points : np.ndarray
            2D array of shape (n_points, 2) representing the coordinates of the points.
        triangle_mesh : np.ndarray
            2D array of shape (n_triangles, 3) representing the triangulation of the points.
        ax : matplotlib.axes.Axes, optional
            The axis to draw on. If None, a new figure and axis will be created.
        colormap : str, optional
            Colormap for the probability map. Default is 'viridis'.

    Returns:    
    --------
        ax : matplotlib.axes.Axes
            The axis with the visualized probabilistic isocontour.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    tpc = ax.tripcolor(points[:, 0], points[:, 1], triangle_mesh, mesh_data, shading='flat', cmap=colormap)
    plt.colorbar(tpc, ax=ax, label='Probability of Isocontour')
    ax.set_title('Probabilistic Marching Triangles')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    return ax
