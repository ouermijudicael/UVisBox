import numpy as np



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

