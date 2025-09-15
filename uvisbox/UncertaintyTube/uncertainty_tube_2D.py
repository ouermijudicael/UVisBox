import numpy as np
import matplotlib.pyplot as plt

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
    print("point1.shape:", point1.shape, "points.shape:", points.shape, "perp_line_dir.shape:", perp_line_dir.shape)
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


def generate_uncertainty_tube_mesh_2D(mean_trajectories, cross_sections):
    """
    Generate uncertainty tube mesh from mean trajectories and cross-sections.

    Parameters
    ----------
    mean_trajectories : np.ndarray
        Array of shape (n_trajectories, n_time_steps, 2) representing the mean trajectory.
    cross_sections : np.ndarray
        Array of shape (n_trajectories, n_time_steps, 2, 2) representing the cross-sections.

    Returns
    -------
    points: np.ndarray
        Array of shape (n_trajectories*n_time_steps*2, 2) representing the tube mesh vertices.
    tube_mesh : np.ndarray
        Array of shape (n_trajectories*n_time_steps*2, 3) representing the tube mesh faces.
    """
    n_trajectories, n_time_steps, _ = mean_trajectories.shape
    points = np.zeros((n_trajectories * n_time_steps * 2, 2))
    tube_mesh = np.zeros((n_trajectories * n_time_steps * 2, 3), dtype=int)
    i_point = 0
    i_face = 0
    for i_traj in range(n_trajectories):    
        for i_t in range(n_time_steps):
            if i_t == 0:
                points[i_point] = mean_trajectories[i_traj, i_t] + cross_sections[i_traj, i_t, 0]
                points[i_point + 1] = mean_trajectories[i_traj, i_t] - cross_sections[i_traj, i_t, 0]
                i_point += 2
            else:
                line_dir = mean_trajectories[i_traj, i_t] - mean_trajectories[i_traj, i_t - 1]
                line_dir = line_dir / np.linalg.norm(line_dir)  # Normalize direction
                perp_line_dir = np.array([-line_dir[1], line_dir[0]])  # Perpendicular direction
                # add point onto perp_line_dir direction passing through mean_trajectories[i_traj, i_t] 
                # with distance cross_sections[i_traj, i_t, 0] from mean_trajectories[i_traj, i_t]
                points[i_point] = mean_trajectories[i_traj, i_t] + cross_sections[i_traj, i_t, 0] * perp_line_dir
                points[i_point + 1] = mean_trajectories[i_traj, i_t] - cross_sections[i_traj, i_t, 0] * perp_line_dir
                # create faces
                tube_mesh[i_face] = [i_point - 2, i_point - 1, i_point]
                tube_mesh[i_face + 1] = [i_point - 1, i_point + 1, i_point]
                i_point += 2
                i_face += 2

    return points, tube_mesh


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