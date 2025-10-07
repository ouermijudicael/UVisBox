
from .uncertainty_tubes_stats import generate_cross_sections_2D
from .uncertainty_tubes_mesh import generate_uncertainty_tube_mesh_2D
from .uncertainty_tubes_vis import matplotlib_uncertainty_tube_2D_vis
from .uncertainty_tubes_stats import generate_cross_sections
from .uncertainty_tubes_mesh import generate_tube_mesh
from .uncertainty_tubes_vis import matplotlib_uncertainty_tube_vis


def uncertainty_tubes_2D(trajectories, axis=None):
    """
    Generate and plot 2D uncertainty tubes from trajectories.

    Parameters:
    -----------
    trajectories : np.ndarray
        Array of shape (n_trajectories, n_time_steps, n_ensemble_members, 2) representing the 2D trajectories.
    axis : matplotlib.axes.Axes, optional
        Axis to plot on. If None, creates a new figure and axis.

    Returns:
    --------
    axis : matplotlib.axes.Axes
        Axis with the plotted uncertainty tubes.
    """
    mean_trajectories, cross_sections = generate_cross_sections_2D(trajectories)
    points, tube_mesh = generate_uncertainty_tube_mesh_2D(mean_trajectories, cross_sections)
    axis = matplotlib_uncertainty_tube_2D_vis(points, tube_mesh, mean_trajectories, axis=axis)

    return axis

def uncertainty_tubes_3D(trajectories, axis=None):
    """
    Generate and plot 3D uncertainty tubes from trajectories.

    Parameters
    ----------
    trajectories : np.ndarray
        Array of shape (n_trajectories, n_time_steps, n_ensemble_members, 3) representing the 3D trajectories.
    axis : matplotlib.axes.Axes, optional
        Axis to plot on. If None, creates a new figure and axis.

    Returns
    -------
    axis : matplotlib.axes.Axes
        Axis with the plotted uncertainty tubes.
    """
    mean_trajectories, cross_sections = generate_cross_sections(trajectories)
    vertices, faces, uv_coords = generate_tube_mesh(mean_trajectories, cross_sections)
    axis = matplotlib_uncertainty_tube_vis(vertices, faces, mean_trajectories, uv_coords, axis=axis)

    return axis

