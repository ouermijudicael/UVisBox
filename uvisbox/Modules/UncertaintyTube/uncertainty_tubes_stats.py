import numpy as np

def expcos(x, e): return np.sign(np.cos(x)) * np.abs(np.cos(x))**e

def expsin(x, e): return np.sign(np.sin(x)) * np.abs(np.sin(x))**e

def project_points_to_plane(point0, point1, points):
    """
    Projects points onto a plane perpendicular to the line segment from point0 to point1.
    
    Args:
        point0, point1 (np.ndarray): Define the line segment direction
        points (np.ndarray): Points to project (shape: (n_points, num_dims))
        
    Returns:
        np.ndarray: Projected points on the orthogonal plane
    """
    # Get normalized direction vector (with zero-vector check)
    direction = point1 - point0
    norm = np.linalg.norm(direction)
    if norm < 1e-10: return points
    direction = direction / norm
    
    # Project: subtract the component along the direction
    components = np.dot(points - point1, direction)
    projected_points = points - np.outer(components, direction)
    return projected_points

def compute_eigen_2d(points):
    # given the points projected to the 2D plan, compute the eigenvalues and eigenvectors
    cov = np.cov(points, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort eigenvalues in descending order
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 1e-8, np.inf)  # Prevent numerical issues
    eigvecs = eigvecs[:, order]
    return eigvals, eigvecs

def _compute_eigen_info(point0, point1, points):
    """
    Helper function to compute eigenvalues and eigenvectors for a single cross-section.
    """
    # Project onto orthogonal plane
    projected_points = project_points_to_plane(point0, point1, points)

    # Calculate covariance and eigendecomposition
    eigvals_proj, eigvecs_proj = compute_eigen_2d(projected_points)

    return eigvals_proj[:2], eigvecs_proj[:, :2] # Return only the top 2 eigvals and their vectors

def uncertainty_tube_summary_statistics(trajectories, n_jobs=1):
    """
    Compute 3D uncertainty summary statistics along trajectory paths.

    Args:
        trajectories (np.ndarray): Ensemble trajectories with shape 
            (n_steps, n_starting_locations, n_samples, num_dims).
        n_jobs (int, optional): Number of parallel jobs to use. If n_jobs=1, uses sequential processing.
            Defaults to 1.

    Returns:
        dict:
            - "mean_trajectory" (np.ndarray): The mean trajectory with shape 
              (n_steps, n_starting_locations, num_dims).
            - "eigen_values" (np.ndarray): Eigenvalues for each cross-section with shape 
              (n_steps, n_starting_locations, 2).
            - "eigen_vectors" (np.ndarray): Eigenvectors for each cross-section with shape
              (n_steps, n_starting_locations, num_dims, 2).
    """
    major_trajectory = np.mean(trajectories, axis=2)
    n_steps, n_starting_locations, n_samples, num_dims = np.shape(trajectories)

    eigen_values = np.zeros((n_steps, n_starting_locations, 2))
    eigen_vectors = np.zeros((n_steps, n_starting_locations, num_dims, 2))

    use_parallel = n_jobs != 1
    if use_parallel:
        try:
            import multiprocessing as mp
            # Use 'fork' context if available, otherwise fallback
            ctx = mp.get_context('fork')
            Pool = ctx.Pool
        except (ImportError, ValueError, NotImplementedError):
            use_parallel = False
            print("Warning: multiprocessing with 'fork' context not available. Using sequential processing instead.")

    # Calculate uncertainty cross-sections for each time step and starting location
    for step_index in range(1, n_steps):
        if use_parallel:
            with Pool(processes=n_jobs) as pool:
                results = pool.starmap(
                    _compute_eigen_info,
                    [
                        (
                            major_trajectory[step_index-1, location_index, :],
                            major_trajectory[step_index, location_index, :],
                            trajectories[step_index, location_index, :, :],
                        ) for location_index in range(n_starting_locations)
                    ]
                )
            for location_index, (ev, evecs) in enumerate(results):
                eigen_values[step_index, location_index, :] = ev
                eigen_vectors[step_index, location_index, :, :] = evecs
        else:
            for location_index in range(n_starting_locations):
                ev, evecs = _compute_eigen_info(
                    major_trajectory[step_index-1, location_index, :],
                    major_trajectory[step_index, location_index, :],
                    trajectories[step_index, location_index, :, :],
                )
                eigen_values[step_index, location_index, :] = ev
                eigen_vectors[step_index, location_index, :, :] = evecs

    return {
        "mean_trajectory": major_trajectory,
        "eigen_values": eigen_values,
        "eigen_vectors": eigen_vectors
    }
