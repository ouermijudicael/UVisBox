import numpy as np
from scipy.spatial import ConvexHull, Delaunay


def curve_boxplot_mesh(summary_stats):
    """
    Build triangular mesh for curve band depth visualization.
    
    This function generates triangular meshes for each percentile band using
    the sorted curves from the summary statistics. For 2D curves, it uses
    ConvexHull and Delaunay triangulation with a center point. For 3D curves,
    it uses the ConvexHull faces directly.
    
    Parameters:
    -----------
    summary_stats : dict
        Dictionary from curve_boxplot_summary_statistics() containing:
        - 'sorted_curves': curves sorted by depth (descending)
        - 'percentiles': list of percentile values
        - 'n_dims': dimensionality (2 or 3)
    
    Returns:
    --------
    mesh_data : dict
        Dictionary containing the following keys:
        - 'percentile_meshes': dict with keys like '50_percentile_mesh' containing
          tuples (points, triangles) where:
          * points: np.ndarray of shape (n_points, n_dims)
          * triangles: np.ndarray of shape (n_triangles, 3)
        - 'median_curve': median curve from summary_stats
        - 'outliers': outlier curves from summary_stats
        - 'n_dims': dimensionality
    
    Examples:
    ---------
    >>> import numpy as np
    >>> from uvisbox.Modules.CurveBoxplot.curve_boxplot_stats import curve_boxplot_summary_statistics
    >>> from uvisbox.Modules.CurveBoxplot.curve_boxplot_mesh import curve_boxplot_mesh
    >>> 
    >>> # Generate synthetic curve data
    >>> curves = np.random.randn(50, 100, 2).cumsum(axis=1)
    >>> 
    >>> # Compute statistics
    >>> stats = curve_boxplot_summary_statistics(curves)
    >>> 
    >>> # Build mesh
    >>> mesh_data = curve_boxplot_mesh(stats)
    >>> 
    >>> # Access mesh for 50th percentile
    >>> points, triangles = mesh_data['percentile_meshes']['50_percentile_mesh']
    >>> print(f"Mesh has {points.shape[0]} points and {triangles.shape[0]} triangles")
    """
    sorted_curves = summary_stats['sorted_curves']
    percentiles = summary_stats['percentiles']
    n_dims = summary_stats['n_dims']
    
    # Initialize mesh data dictionary
    mesh_data = {
        'percentile_meshes': {},
        'median_curve': summary_stats['median_curve'],
        'outliers': summary_stats['outliers'],
        'n_dims': n_dims
    }
    
    # Build mesh for each percentile
    for percentile in percentiles:
        points, triangles = _build_percentile_mesh(sorted_curves, percentile, n_dims)
        mesh_key = f'{int(percentile)}_percentile_mesh'
        mesh_data['percentile_meshes'][mesh_key] = (points, triangles)
    
    return mesh_data


def _build_percentile_mesh(sorted_curves, percentile, n_dims):
    """
    Build triangular mesh for a single percentile band.
    
    This function takes curves up to the specified percentile and builds a
    triangular mesh representing the band envelope. It samples the curves
    at regular intervals to build the mesh efficiently.
    
    Parameters:
    -----------
    sorted_curves : np.ndarray
        3D array of shape (n_curves, n_steps, n_dims) sorted by depth (descending)
    percentile : float
        Percentile value (0-100) for the band
    n_dims : int
        Dimensionality (2 or 3)
    
    Returns:
    --------
    points : np.ndarray
        2D array of shape (n_points, n_dims) containing mesh vertices
    triangles : np.ndarray
        2D array of shape (n_triangles, 3) containing triangle indices
    """
    num_curves = sorted_curves.shape[0]
    index = int(np.ceil(num_curves * (percentile / 100)))
    selected_curves = sorted_curves[:index]
    
    num_time_steps = selected_curves.shape[1]
    
    # Determine sampling stride for efficiency
    if num_time_steps < 100:
        stride = 1
    else:
        stride = num_time_steps // 100
    
    final_points = []
    final_triangles = []
    point_offset = 0
    
    # Sample at regular intervals and build mesh at each time slice
    for i_t in range(1, num_time_steps, stride):
        i_t_start = np.maximum(i_t - stride, 0)
        i_t_end = np.minimum(i_t, num_time_steps - 1)
        
        # Get points from all curves in this time window
        points = selected_curves[:, i_t_start:i_t_end+1, :].reshape(-1, n_dims)
        
        # Build convex hull
        hull = ConvexHull(points)
        
        if n_dims == 2:
            # 2D case: Triangulate hull interior with center point
            # hull.simplices are edges in 2D
            
            # Get hull vertices and triangulate
            hull_points = Delaunay(points[hull.vertices])
            
            # Add center point
            center = np.mean(points[hull.vertices], axis=0)
            points_with_center = np.vstack([points, center])
            center_idx = points.shape[0]
            
            # Create triangles from center to hull edges
            new_triangles = []
            for simplex in hull.simplices:
                new_triangles.append([simplex[0], simplex[1], center_idx])
            
            # Remap Delaunay indices from hull vertex indices to original point indices
            remapped_delaunay = hull.vertices[hull_points.simplices]
            
            # Combine all triangles
            all_triangles = np.vstack([remapped_delaunay, new_triangles])
            
        else:  # n_dims == 3
            # 3D case: hull.simplices are already triangular faces
            points_with_center = points
            all_triangles = hull.simplices
        
        # Accumulate points and triangles with proper indexing
        final_points.append(points_with_center)
        final_triangles.append(all_triangles + point_offset)
        point_offset += points_with_center.shape[0]
    
    # Concatenate all points and triangles
    if len(final_points) > 0:
        final_points = np.vstack(final_points)
        final_triangles = np.vstack(final_triangles)
    else:
        final_points = np.array([]).reshape(0, n_dims)
        final_triangles = np.array([]).reshape(0, 3)
    
    return final_points, final_triangles

    