import numpy as np

def functional_depth_mesh(top_curve, bottom_curve, scale=1.0):
    """
    Create a 2D triangular mesh representing the functional depth area between the top and bottom curves.
    Parameters:
    -----------
    top_curve : np.ndarray
        1D array representing the top curve.
    bottom_curve : np.ndarray
        1D array representing the bottom curve.
    scale : float, optional
        Scale factor for the depth area. Default is 1.0.
    Returns:
    -----------
    vertices : np.ndarray
        2D array of shape (n_vertices, 2) representing the vertices of the mesh.
    """
    if top_curve.shape != bottom_curve.shape or top_curve.shape != median_curve.shape:
        raise ValueError("All input curves must have the same shape.")
    
    # Create a grid of points between the curves
    x = np.linspace(0, 1, top_curve.shape[0])
    y_top = top_curve * scale
    y_bottom = bottom_curve * scale

    points = np.zeros((top_curve.shape[0] * 2, 2))
    triangles = np.zeros(( (top_curve.shape[0]-1) * 2, 3), dtype=int)
    i_pt = 0
    i_tr = 0
    for i in range(top_curve.shape[0]):
        points[i_pt] = [x[i], y_top[i]]
        points[i_pt + 1] = [x[i], y_bottom[i]]
        i_pt += 2
        if i > 0:
            triangles[i_tr] = [i_pt - 4, i_pt - 3, i_pt - 2]
            triangles[i_tr + 1] = [i_pt - 3, i_pt - 1, i_pt - 2]
            i_tr += 2
            
    return points, triangles