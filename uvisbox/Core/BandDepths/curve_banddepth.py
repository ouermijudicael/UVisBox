import numpy as np
from scipy.spatial import ConvexHull
from itertools import combinations
import multiprocessing as mp



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


def _process_time_step(args):
    """
    Worker function to process a single time step for parallel computation.
    
    Parameters:
    ----------
    args : tuple
        Tuple containing (step_idx, curves, n_curves, indices, n_dims)
    
    Returns:
    -------
    numpy.ndarray
        Depth increments for each curve at this time step
    """
    step_idx, curves, n_curves, indices, n_dims = args
    
    # Initialize depth increments for this time step
    step_depths = np.zeros(n_curves)
    
    # OPTIMIZATION: Build each hull once, then test all points against it
    # This reduces hull construction from O(n_curves * len(indices)) to O(len(indices))
    for band in indices:
        # Extract all points from curves in this band up to current time step
        # Shape: (len(band), step_idx+1, n_dims) -> (total_points, n_dims)
        band_curves = curves[band, :step_idx+1, :]
        
        try:
            # Create convex hull from all band points flattened to 2D
            # Reshape combines all time steps and curves in band into single point cloud
            # BUILD HULL ONCE per band (instead of once per curve per band)
            hull = ConvexHull(band_curves.reshape(-1, n_dims))
        except:
            # Skip if hull construction fails (e.g., insufficient points, colinear points)
            continue
        
        # Now test all curve points against this hull
        for curve_point_idx in range(n_curves):
            # Get the current point of the curve at this time step
            point = curves[curve_point_idx, step_idx, :]
            
            # Check if current curve's point lies inside this band's convex hull
            if point_in_hull(point, hull):
                # Increment depth score if point is contained within the hull
                step_depths[curve_point_idx] += 1
    
    return step_depths


def curve_banddepths(curves, indices=None, workers=12):
    """
    Calculate band depth for curves based on how often each curve's points lie within convex hulls 
    formed by bands of other curves. M. Mirzargar, R. T. Whitaker and R. M. Kirby, "Curve Boxplot: 
    Generalization of Boxplot for Ensembles of Curves," in IEEE Transactions on Visualization and 
    Computer Graphics, vol. 20, no. 12, pp. 2654-2663, 31 Dec. 2014, doi: 10.1109/TVCG.2014.2346455.
    
    Parameters:
    ----------
    curves : numpy.ndarray
        3D array of shape (n_curves, n_steps, n_dims) containing curve data
    indices : list of lists, optional
        Each inner list contains indices of curves that form a band
        indices can be created by itertools.combinations(range(n_curves), k) for some k
        recommended caching for indices
    workers : int, optional
        Number of worker processes for parallel computation. Default is 12.
        Set to 1 or None to use sequential processing (useful for debugging).
        Uses fork context for macOS compatibility.
    
    Returns:
    -------
    numpy.ndarray
        1D array of normalized depth scores for each curve
    """
    

    # Extract dimensions: number of curves, time steps, and coordinate dimensions
    n_curves, n_steps, n_dims = curves.shape

    if indices is None:
        # Default to using all combinations of 2 curves if no indices provided
        indices = list(combinations(range(n_curves), 2))

    # Initialize depth scores for each curve (starts at zero)
    depths = np.zeros(n_curves)
    
    # Decide whether to use parallel or sequential processing
    use_parallel = workers is not None and workers > 1
    
    if use_parallel:
        # Parallel processing using multiprocessing with fork context
        # Prepare arguments for each time step
        step_args = [
            (step_idx, curves, n_curves, indices, n_dims)
            for step_idx in range(1, n_steps)
        ]
        
        # Create a multiprocessing pool using fork context for macOS compatibility
        ctx = mp.get_context('fork')
        pool = ctx.Pool(processes=workers)
        
        try:
            # Process all time steps in parallel
            step_depths_list = pool.map(_process_time_step, step_args)
            
            # Sum up depth increments from all time steps
            for step_depths in step_depths_list:
                depths += step_depths
        finally:
            # Ensure pool is properly closed and joined
            pool.close()
            pool.join()
    else:
        # Sequential processing (optimized implementation)
        # Iterate through each time step (starting from step 1, not 0)
        # This is because we need at least 2 points to form a meaningful convex hull
        for step_idx in range(1, n_steps):
            # OPTIMIZATION: Build each hull once, then test all points against it
            # This reduces hull construction from O(n_curves * len(indices)) to O(len(indices))
            for band in indices:
                # Extract all points from curves in this band up to current time step
                # Shape: (len(band), step_idx+1, n_dims) -> (total_points, n_dims)
                band_curves = curves[band, :step_idx+1, :]
                
                try:
                    # Create convex hull from all band points flattened to 2D
                    # Reshape combines all time steps and curves in band into single point cloud
                    # BUILD HULL ONCE per band (instead of once per curve per band)
                    hull = ConvexHull(band_curves.reshape(-1, n_dims))
                except:
                    # Skip if hull construction fails (e.g., insufficient points, colinear points)
                    continue
                
                # Now test all curve points against this hull
                for curve_point_idx in range(n_curves):
                    # Get the current point of the curve at this time step
                    point = curves[curve_point_idx, step_idx, :]
                    
                    # Check if current curve's point lies inside this band's convex hull
                    if point_in_hull(point, hull):
                        # Increment depth score if point is contained within the hull
                        depths[curve_point_idx] += 1
    
    # Normalize depths by total number of comparisons made
    # (n_steps-1) time steps * len(indices) bands per time step
    depths /= (n_steps-1) * len(indices)
    return depths



