import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from itertools import combinations

def build_curve_band_mesh(sorted_curves, percentile=50):

    num_curves = sorted_curves.shape[0]
    index = int(np.ceil(num_curves * (percentile / 100)))
    before = sorted_curves[:index]

    final_points = []
    final_triangles = []
    num_time_steps = before.shape[1]
    i_point = 0
    i_triangle = 0    
    if num_time_steps < 20:
        stride = 1
    else:
        stride = num_time_steps // 20
    for i_t in range(1, num_time_steps, stride):
        print(f"Processing time step {i_t}/{num_time_steps} with stride {stride}")
        i_t_start = np.maximum(i_t - stride, 0)
        i_t_end = np.minimum( i_t, num_time_steps - 1)
        points = before[:,i_t_start:i_t_end+1,:].reshape(-1, before.shape[2]) 
        
        try:
            hull = ConvexHull(points)
            delaunay = Delaunay(points[hull.vertices])
        except:
            continue    
        final_points.append(points)
        final_triangles.append(delaunay.simplices + i_point)
        i_point += points.shape[0]
        i_triangle += delaunay.simplices.shape[0]
    final_points = np.vstack(final_points)
    final_triangles = np.vstack(final_triangles)
    return final_points, final_triangles

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



def curve_banddepth_plot(curves, depths=None, percentile=50, ax=None):
    """
    Create a curve band depth plot.
    Parameters:
    ----------
    curves : numpy.ndarray
        3D array of shape (n_curves, n_steps, n_dims) containing curve data
    depths : numpy.ndarray, optional
        1D array of precomputed band depths of shape (n_curves,). If None, band depths will be computed.
    percentile : float, optional
        Percentile for the band to be highlighted. Default is 50 (median band).
    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes object to plot on. If None, a new figure and axes will be created.
    Returns:
    -------
    ax: matplotlib.axes.Axes
        The Axes object with the curve band depth plot.
    """
    if depths is None:
        
        n_curves = curves.shape[0]
        indices = list(combinations(range(n_curves), 2))
        print("Calculating curve band depths...")
        depths = curve_band_depths(curves, indices)
        print("Curve band depths calculated.")

    # sort the curves by the depth. order them from deepest to shallowest
    sorted_indices = np.argsort(depths)[::-1]
    sorted_curves = curves[sorted_indices]

    # create figure if no ax is assigned
    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

    # build the band mesh for the specified percentile
    print("Building curve band mesh...")
    points, triangles = build_curve_band_mesh(sorted_curves, percentile=percentile)
    print("Curve band mesh built.")
    
    # plot the band mesh using trisurf or tripcolor
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    ax = plt.axes(projection='3d') if ax.name != '3d' else ax

    ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=triangles, cmap='viridis', alpha=0.5)
    
    
    # highlight the median curve in red
    median_curve = sorted_curves[0]
    ax.plot(median_curve[:, 0], median_curve[:, 1], median_curve[:, 2], color='red', linewidth=2, label='Median Curve')

    return ax

