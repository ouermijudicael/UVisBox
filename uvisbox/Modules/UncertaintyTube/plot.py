
from .uncertainty_tube_2D import generate_cross_sections_2D
from .uncertainty_tube_meshing_2D import generate_uncertainty_tube_mesh_2D
from .uncertainty_tube_plot_2D import plot_uncertainty_tube_2D


def plot(trajectories, axis=None):
    """
    Generate and plot 2D uncertainty tubes from trajectories.

    Parameters
    ----------
    trajectories : np.ndarray
        Array of shape (n_trajectories, n_time_steps, n_ensemble_members, 2) representing the 2D trajectories.
    axis : matplotlib.axes.Axes, optional
        Axis to plot on. If None, creates a new figure and axis.

    Returns
    -------
    axis : matplotlib.axes.Axes
        Axis with the plotted uncertainty tubes.
    """
    mean_trajectories, cross_sections = generate_cross_sections_2D(trajectories)
    points, tube_mesh = generate_uncertainty_tube_mesh_2D(mean_trajectories, cross_sections)
    axis = plot_uncertainty_tube_2D(points, tube_mesh, mean_trajectories, axis=axis)

    return axis