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

def build_2d_superellipse(eigvals, resolution=16, e_proj=1, sym=False):
    """
    Build a 2D superellipse given the eigenvalues.

    Args:
        eigvals (np.ndarray): Eigenvalues for the 2D projection.
        resolution (int, optional): Number of points to sample on the superellipse boundary.
            Defaults to 16.
        e_proj (float, optional): Exponent controlling the superellipse shape.
            e_proj=1 creates a standard ellipse, higher values approach a rectangle.
            Defaults to 1.
        sym (bool, optional): If True, forces the superellipse to be symmetric by
            setting both axes to the same length. Defaults to False.

    Returns:
        np.ndarray: A 2D array (shape: resolution × 2) representing the boundary points
        of the superellipse.
    """
    # Calculate the radii for the superellipse axes
    radii = 2.0 * np.sqrt(eigvals[:2])  # 2-sigma ellipse
    if sym:
        radii[1] = radii[0]  # Force symmetric shape if requested

    # Calculate the boundary points of the superellipse
    theta = np.linspace(0, 2 * np.pi, resolution+1)
    x = radii[0] * expcos(theta, e_proj)
    y = radii[1] * expsin(theta, e_proj)
    superellipse_2d = np.stack([x[:-1], y[:-1]], axis=1)  # shape (resolution, 2)
    return superellipse_2d

def uncertainty_cross_section(point0, point1, points, resolution=16, e_proj=1, sym=False):
    """
    Calculate a cross-sectional uncertainty profile using start, end, and point cloud.
    
    This function generates a 2D superellipse embedded in 3D space that represents the 
    uncertainty distribution around a trajectory segment. It projects points onto a plane 
    perpendicular to the segment direction, performs principal component analysis to 
    determine the main uncertainty directions, and creates a superellipse boundary.
    
    Args:
        point0 (np.ndarray): A 3D point representing the start of the segment.
        point1 (np.ndarray): A 3D point representing the end of the segment.
        points (np.ndarray): A 2D array of points (shape: n_samples × num_dims) 
            representing the point cloud.
        resolution (int, optional): Number of points to sample on the superellipse boundary.
            Defaults to 16.
        e_proj (float, optional): Exponent controlling the superellipse shape.
            e_proj=1 creates a standard ellipse, higher values approach a rectangle.
            Defaults to 1.
        sym (bool, optional): If True, forces the superellipse to be symmetric by 
            setting both axes to the same length. Defaults to False.
    
    Returns:
        tuple:
            - cross_section_3d (np.ndarray): A 2D array (shape: resolution × num_dims) 
              representing the boundary points of the 3D cross-sectional profile.
            - eigen_values (np.ndarray): A 2D array (shape: resolution × 2) containing
              the two principal eigenvalues for each boundary point, representing
              the magnitudes of uncertainty in the primary directions.
    
    Notes:
        - The function creates a 2D superellipse in 3D space (not a true 3D ellipsoid)
        - The cross-section is oriented perpendicular to the trajectory direction
        - The function uses a 2-sigma confidence region (covering ~95% of variance)
        - The superellipse shape can be controlled via the e_proj parameter
    """
    # Project onto orthogonal plane
    projected_points = project_points_to_plane(point0, point1, points)
    mean_projected = np.mean(projected_points, axis=0)
    
    # Calculate covariance and eigendecomposition
    eigvals_proj, eigvecs_proj = compute_eigen_2d(projected_points)
    cross_section_2d = build_2d_superellipse(eigvals_proj, resolution=resolution, e_proj=e_proj, sym=sym)

    # Primary and secondary axes
    primary_axis = eigvecs_proj[:, 0]
    secondary_axis = eigvecs_proj[:, 1] 
        
    # Transform 2D cross-section to 3D space
    cross_section_3d = mean_projected + cross_section_2d[:, 0, np.newaxis] * primary_axis + \
                      cross_section_2d[:, 1, np.newaxis] * secondary_axis

    # Return eigenvalues repeated for each point in the cross-section
    eigen_values = np.tile(eigvals_proj[:2], (cross_section_3d.shape[0], 1))
    return cross_section_3d, eigen_values

def generate_uncertainty_tube(trajectories, major_trajectory=None, resolution=20, e_proj=1, sym=False):
    """
    Compute 3D uncertainty cross-sections along trajectory paths.

    For each time step and starting location, this function calculates a cross-sectional 
    uncertainty profile by projecting the ensemble trajectories onto a plane perpendicular 
    to the major trajectory direction, performing principal component analysis, and 
    constructing a superellipse boundary representing the uncertainty distribution.

    Args:
        trajectories (np.ndarray): Ensemble trajectories with shape 
            (n_steps, n_starting_locations, n_samples, num_dims).
        major_trajectory (np.ndarray, optional): The main trajectory to use for cross-section orientation,
            with shape (n_steps, n_starting_locations, num_dims). If None, uses the mean of trajectories.
        resolution (int, optional): Number of points to sample on each cross-section boundary. 
            Defaults to 20.
        e_proj (float, optional): Exponent controlling the superellipse shape. 
            e_proj=1 creates a standard ellipse, higher values approach a rectangle.
            Defaults to 1.
        sym (bool, optional): If True, forces the superellipse to be symmetric by
            setting both axes to the same length. Defaults to False.

    Returns:
        tuple:
            - uncertainty_tube (np.ndarray): Cross-section boundary points with shape 
              (n_steps, n_starting_locations, resolution, num_dims).
            - eigen_values (np.ndarray): Eigenvalues for each cross-section boundary point with shape 
              (n_steps, n_starting_locations, resolution, 2).

    Notes:
        - The function computes a 2D superellipse embedded in 3D space for each trajectory segment.
        - The cross-section is oriented perpendicular to the trajectory direction at each time step.
        - The initial positions (t=0) are set to the starting points of the major trajectory.
    """
    # Compute major trajectory if not provided (mean over samples)
    if major_trajectory is None:
        major_trajectory = np.mean(trajectories, axis=2)
    n_steps, n_starting_locations, n_samples, num_dims = np.shape(trajectories)
    
    # Initialize output arrays for cross-section boundary points and eigenvalues
    uncertainty_tube = np.zeros((n_steps, n_starting_locations, resolution, num_dims))
    eigen_values = np.zeros((n_steps, n_starting_locations, resolution, 2))
    
    # Set initial positions (t=0) to major trajectory starting points
    for location_index in range(n_starting_locations):
        for res_index in range(resolution):
            uncertainty_tube[0, location_index, res_index, :] = major_trajectory[0, location_index, :]

    # Calculate uncertainty cross-sections for each time step and starting location
    for step_index in range(1, n_steps):
        for location_index in range(n_starting_locations):
            # Calculate cross-section boundary and eigenvalues
            uncertainty_tube[step_index, location_index, :, :], eigen_values[step_index, location_index, :, :] = uncertainty_cross_section(
                major_trajectory[step_index-1, location_index, :],  # previous major trajectory point
                major_trajectory[step_index, location_index, :],    # current major trajectory point
                trajectories[step_index, location_index, :, :],     # ensemble samples at current time
                resolution=resolution,
                e_proj=e_proj, 
                sym=sym
            )

    return uncertainty_tube, eigen_values
