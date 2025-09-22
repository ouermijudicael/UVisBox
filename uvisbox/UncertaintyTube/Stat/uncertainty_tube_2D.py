import numpy as np
from ..Meshing.uncertainty_tube_meshing_2D import generate_uncertainty_tube_mesh_2D
from ..Vis.uncertainty_tube_plot_2D import plot_uncertainty_tube_2D

def project_points_onto_line(point0, point1, points):
    """
    Project points onto a line defined by two points.

    Parameters
    ----------
    point0 : np.ndarray
        Array of shape (2,) representing the first point on the line.
    point1 : np.ndarray
        Array of shape (2,) representing the second point on the line.
    points : np.ndarray
        Array of shape (n_points, 2) representing the points to be projected.

    Returns
    -------
    projections : np.ndarray
        Array of shape (n_points, 2) representing the projected points on the line.
    """
    line_dir = point1 - point0
    line_dir = line_dir / np.linalg.norm(line_dir)  # Normalize direction
    perp_line_dir = np.array([-line_dir[1], line_dir[0]])  # Perpendicular direction
    # project points onto the line passing through point1 and perpendicular to line_dir
    projections = point1 + np.dot(points - point1, perp_line_dir)[:, np.newaxis] * perp_line_dir

    return projections



def generate_cross_sections_2D(trajectories):
    """
    Compute cross-sections of 2D trajectories.

    Parameters
    ----------
    trajectories : np.ndarray
        Array of shape (n_trajectories, n_time_steps, n_ensemble_members, 2) representing the 2D trajectories.

    Returns
    -------
    cross_sections : list of np.ndarray
        mean_trajectory : np.ndarray
            Array of shape (n_trajectories, n_time_steps, 2) representing the mean trajectory.
        cross_sections : np.ndarray
            Array of shape (n_trajectories, n_time_steps)
    """
    n_trajectories, n_time_steps, n_ensemble_members, _ = trajectories.shape
    cross_sections = np.zeros((n_trajectories, n_time_steps, 2, 2))  # Each cross-section is represented by a line segment (x1, y1, x2, y2)
    mean_trajectories = np.mean(trajectories, axis=2)


    # Compute cross-sections at each time step

    for i_t in range(1, n_time_steps):

        for i_traj in range(n_trajectories):
            
            # project points onto the onto the passing through mean_trajectories[:, i_t] and
            # perpendicular to direction_unit
            projected_points = project_points_onto_line(mean_trajectories[i_traj, i_t - 1], mean_trajectories[i_traj, i_t], trajectories[i_traj, i_t])
            # Compute covariance matrix of the projected points
            cov_matrix = np.cov(projected_points.T)
            # Eigen decomposition
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            # Sort eigenvalues and eigenvectors
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]

            cross_sections[i_traj, i_t] = eigenvalues[0]

    return mean_trajectories, cross_sections



def uncertainty_tube_2D(trajectories, axis=None):
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