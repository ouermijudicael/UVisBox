import matplotlib.pyplot as plt

def matplotlib_probabilistic_marching_squares_vis(prob_contour, cmap='viridis', ax=None):
    """
    Visualize the probability map of isocontour presence using matplotlib.

    Parameters:
    -----------
        prob_contour : np.ndarray
            2D array of shape (n-1, m-1) with probabilities of contour presence in each cell.
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
    im = ax.imshow(prob_contour, origin='lower', cmap=cmap)
    plt.colorbar(im, ax=ax, label='Probability of Contour')
    ax.set_title('Probabilistic Marching Squares')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return ax
