import numpy as np
from scipy.spatial import ConvexHull, Delaunay

def point_in_hull(point, hull_or_vertices, eps=1e-6):
    """
    Check if a point is inside a convex hull.
    
    Parameters:
    ----------
    point : array-like
        The coordinates of the point to check. Should be a 1D array or list of length equal to the dimension of the hull.
    hull_or_vertices : scipy.spatial.ConvexHull or array-like
        Either a SciPy ConvexHull object, or an array/list of vertices that define the convex hull.
        If vertices are provided, a ConvexHull object will be constructed internally.
    eps : float, optional
        Tolerance for numerical precision when checking if the point is inside the hull. Default is 1e-6.
    
    Returns:
    -------
    bool
        True if the point is inside the convex hull (within the specified tolerance), False otherwise.
    
    Raises
    ------
    ValueError
        If `hull_or_vertices` is not a ConvexHull object, ndarray, or list of vertices.
    
    Notes
    -----
    - The function uses the hull's half-space equations to determine if the point is inside.
    - Points on the boundary (within `eps` tolerance) are considered inside.
    """

    if isinstance(hull_or_vertices, ConvexHull):
        hull = hull_or_vertices
    else:
        ### hull_or_vertices needs to be an ndarray or list of vertices
        if isinstance(hull_or_vertices, np.ndarray):
            vertices = hull_or_vertices
        elif isinstance(hull_or_vertices, list):
            vertices = np.array(hull_or_vertices)
        else:
            raise ValueError("Invalid input: hull_or_vertices must be a SciPy ConvexHull object, an ndarray or list of vertices.")
        hull = ConvexHull(vertices)
    return all((np.dot(eq[:-1], point) + eq[-1] < eps) for eq in hull.equations)

def curve_band_depths(curves, indices):
    """
    Calculate band depth for curves based on how often each curve's points lie within convex hulls formed by bands of other curves.
    
    Parameters:
    ----------
    curves : numpy.ndarray
        3D array of shape (n_curves, n_steps, n_dims) containing curve data
    indices : list of lists
        Each inner list contains indices of curves that form a band
        indices can be created by itertools.combinations(range(n_curves), k) for some k
        recommended caching for indices
    
    Returns:
    -------
    numpy.ndarray
        1D array of normalized depth scores for each curve
    """
    # Extract dimensions: number of curves, time steps, and coordinate dimensions
    n_curves, n_steps, n_dims = curves.shape

    # Initialize depth scores for each curve (starts at zero)
    depths = np.zeros(n_curves)
    
    # Iterate through each time step (starting from step 1, not 0)
    # This is because we need at least 2 points to form a meaningful convex hull
    for step_idx in range(1, n_steps):
        # For each curve, check its current point against all band hulls
        for curve_point_idx in range(n_curves):
            # Get the current point of the curve at this time step
            point = curves[curve_point_idx, step_idx, :]
            
            # Check this point against each band's convex hull
            for band in indices:
                # Extract all points from curves in this band up to current time step
                # Shape: (len(band), step_idx+1, n_dims) -> (total_points, n_dims)
                band_curves = curves[band, :step_idx+1, :]
                
                try:
                    # Create convex hull from all band points flattened to 2D
                    # Reshape combines all time steps and curves in band into single point cloud
                    hull = ConvexHull(band_curves.reshape(-1, n_dims))
                except:
                    # Skip if hull construction fails (e.g., insufficient points, colinear points)
                    continue
                
                # Check if current curve's point lies inside this band's convex hull
                if point_in_hull(point, hull):
                    # Increment depth score if point is contained within the hull
                    depths[curve_point_idx] += 1
    
    # Normalize depths by total number of comparisons made
    # (n_steps-1) time steps * len(indices) bands per time step
    depths /= (n_steps-1) * len(indices)
    return depths
