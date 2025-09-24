import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from mpl_toolkits.mplot3d import Axes3D

def curve_banddepth_meshing(sorted_curves, percentile=50):
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
        
        hull = ConvexHull(points)


        # get convex hull vertices
        hull_points = Delaunay(points[hull.vertices])
        # add a point at the center of the convex hull
        center = np.mean(points[hull.vertices], axis=0)
        points = np.vstack([points, center])
        # create new triangles connecting the center to the hull edges
        new_triangles = []
        for simplex in hull.simplices:
            new_triangles.append([simplex[0], simplex[1], points.shape[0]-1])
        # new_triangles = np.array(new_triangles)

        final_points.append(points)
        final_triangles.append(np.vstack([hull_points.simplices, new_triangles]) + i_point)
        i_point += points.shape[0]
        i_triangle +=  len(new_triangles)
    final_points = np.vstack(final_points)
    final_triangles = np.vstack(final_triangles)

    return final_points, final_triangles
    