import matplotlib.pyplot as plt


def visualize_probabilistic_marching_squares(mesh_data, ax=None, colormap='viridis'):
    """
    Visualize the probability map of isocontour presence using matplotlib.

    Parameters:
    -----------
        mesh_data : dict
            Dictionary from probabilistic_marching_squares_mesh containing:
            - 'level_crossing_probability': np.ndarray
                2D array of shape (y_dim-1, x_dim-1) with probabilities of contour
                presence in each cell.
            - 'extent': tuple (x_min, x_max, y_min, y_max) or None
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

    prob = mesh_data['level_crossing_probability']
    extent = mesh_data.get('extent')

    imshow_kwargs = dict(origin='lower', cmap=colormap, vmin=0, vmax=1)
    if extent is not None:
        imshow_kwargs['extent'] = extent

    im = ax.imshow(prob, **imshow_kwargs)
    plt.colorbar(im, ax=ax, label='probability of contour')
    ax.set_title('Probabilistic Marching Squares')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    return ax
