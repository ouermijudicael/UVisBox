import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from mpl_toolkits.mplot3d import Axes3D

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
    #     # delaunay = Delaunay(points[hull.vertices])
       
    #     # import matplotlib.pyplot as plt
    #     # if points.shape[1] == 2:
    #     #     plt.figure()
    #     #     plt.plot(points[:, 0], points[:, 1], 'o')
    #     #     # Plot convex hull
    #     #     simplex = hull.simplices[0]
    #     #     plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
    #     #     # Plot Delaunay triangulation
    #     #     for tri in delaunay.simplices:
    #     #         print(tri)
    #     #         index = np.append(tri, tri[0])  # close the triangle
    #     #         plt.plot(points[index, 0], points[index, 1], 'r--', alpha=0.5)
    #     #     plt.title(f'Convex Hull & Delaunay at time step {i_t}')
    #     #     plt.savefig(f"curve_banddepth_meshing_t{str(i_t).zfill(3)}.png")
    #     #     plt.show()
    #     #     print(f"number of hull simplices: {len(hull.simplices)}, number of delaunay simplices: {len(delaunay.simplices)}")
    #     # elif points.shape[1] == 3:
    #     #     fig = plt.figure()
    #     #     ax = fig.add_subplot(111, projection='3d')
    #     #     ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    #     #     # Plot convex hull
    #     #     for simplex in hull.simplices:
    #     #         simplex = np.append(simplex, simplex[0])  # cycle back to the first point
    #     #         ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'k-')
    #     #         # Plot Delaunay triangulation
    #     #         for tri in delaunay.simplices:
    #     #             tri_points = np.append(tri, tri[0])  # close the triangle
    #     #     ax.plot(points[tri_points, 0], points[tri_points, 1], points[tri_points, 2], 'r--', alpha=0.5)
    #     #     plt.title(f'Convex Hull & Delaunay at time step {i_t}')
    #     #     plt.show()

    #     final_points.append(points)
    #     final_triangles.append(delaunay.simplices + i_point)
    #     i_point += points.shape[0]
    #     i_triangle += delaunay.simplices.shape[0]
    # final_points = np.vstack(final_points)
    # final_triangles = np.vstack(final_triangles)
    
    # return final_points, final_triangles
