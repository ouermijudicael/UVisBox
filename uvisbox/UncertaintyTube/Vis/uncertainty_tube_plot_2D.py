import numpy as np
import matplotlib.pyplot as plt

def plot_uncertainty_tube_2D(points, tube_mesh, mean_trajectories, axis=None):
    """
    Plot 2D uncertainty tubes from pre-generated mesh data.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (n_points, 2) representing the tube mesh vertices.
    tube_mesh : np.ndarray
        Array of shape (n_faces, 3) representing the tube mesh faces.
    mean_trajectories : np.ndarray
        Array of shape (n_trajectories, n_time_steps, 2) representing the mean trajectory.
    axis : matplotlib.axes.Axes, optional
        Axis to plot on. If None, creates a new figure and axis.

    Returns
    -------
    None
    """

    if axis is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        ax = axis

    # Plot mean trajectory lines
    n_trajectories, n_time_steps = mean_trajectories.shape[:2]
    for i in range(n_trajectories):
        ax.plot(mean_trajectories[i, :, 0], mean_trajectories[i, :, 1],
                color='black', alpha=1.0, linewidth=2)

    # Plot uncertainty tubes using triangular faces
    tri_colors = np.ones((tube_mesh.shape[0]))*0.8
    ax.tripcolor(points[:, 0], points[:, 1], tube_mesh, facecolors=tri_colors, edgecolors='gray', alpha=0.5)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('2D Trajectories with Uncertainty')
    plt.grid()
    if axis is None:
        return fig, ax
    else:
        return ax

