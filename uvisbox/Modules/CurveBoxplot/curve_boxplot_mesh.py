import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from mpl_toolkits.mplot3d import Axes3D

def curves_band_mesh(sorted_curves, percentile=50):
    """
    Build a mesh for the curve band depth plot.

    Parameters:
    -----------
    sorted_curves : numpy.ndarray
        3D array of shape (n_curves, n_steps, n_dims) containing curve data sorted by depth (deepest first)
    percentile : float, optional
        Percentile for the band to be highlighted. Default is 50 (median band).
    
    Returns:
    --------
    points : numpy.ndarray
        2D array of shape (n_points, n_dims) containing the points of the mesh.
    triangles : numpy.ndarray
        2D array of shape (n_triangles, 3) containing the indices of the points forming the triangles.
    """

    num_curves = sorted_curves.shape[0]
    index = int(np.ceil(num_curves * (percentile / 100)))
    before = sorted_curves[:index]

    final_points = []
    final_triangles = []
    num_time_steps = before.shape[1]
    i_point = 0
    i_triangle = 0    
    if num_time_steps < 100:
        stride = 1
    else:
        stride = num_time_steps // 100
    for i_t in range(1, num_time_steps, stride):
        i_t_start = np.maximum(i_t - stride, 0)
        i_t_end = np.minimum( i_t, num_time_steps - 1)
        points = before[:,i_t_start:i_t_end+1,:].reshape(-1, before.shape[2]) 
        
        n_dims = points.shape[1]
        hull = ConvexHull(points)

        if n_dims == 2:
            # 2D case: hull.simplices are edges, need to triangulate interior
            # and create triangles from center to hull edges
            
            # Triangulate the convex hull interior
            hull_points = Delaunay(points[hull.vertices])
            
            # Add center point
            center = np.mean(points[hull.vertices], axis=0)
            points_with_center = np.vstack([points, center])
            center_idx = points.shape[0]
            
            # Create triangles from center to hull edges
            # hull.simplices contains direct point indices (edges)
            new_triangles = []
            for simplex in hull.simplices:
                new_triangles.append([simplex[0], simplex[1], center_idx])
            
            # Remap Delaunay indices from 0..len(hull.vertices)-1 to original point indices
            remapped_delaunay = hull.vertices[hull_points.simplices]
            
            # Combine all triangles
            all_triangles = np.vstack([remapped_delaunay, new_triangles])
            
        else:  # 3D case
            # In 3D, hull.simplices already contains triangular faces of the convex hull
            # hull.simplices contains direct point indices
            points_with_center = points
            all_triangles = hull.simplices
        
        final_points.append(points_with_center)
        final_triangles.append(all_triangles + i_point)
        i_point += points_with_center.shape[0]
        i_triangle += len(all_triangles)
    final_points = np.vstack(final_points)
    final_triangles = np.vstack(final_triangles)

    return final_points, final_triangles
    