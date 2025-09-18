import numpy as np
from scipy.spatial import ConvexHull, Delaunay

def curve_banddepth_meshing(sorted_curves, percentile=50):
    """
    Build a mesh for the curve band depth plot.
    Parameters:
    ----------
    sorted_curves : numpy.ndarray
        3D array of shape (n_curves, n_steps, n_dims) containing curve data sorted by depth (deepest first)
    percentile : float, optional
        Percentile for the band to be highlighted. Default is 50 (median band).
    Returns:
    -------
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
