import numpy as np
from scipy import ConvexHull, Delaunay

def point_in_hull(point, hull_or_vertices, eps=1e-6):
    """
    Check if a point is inside a convex hull.
    Parameters
    ----------
    point : array-like
        The coordinates of the point to check. Should be a 1D array or list of length equal to the dimension of the hull.
    hull_or_vertices : scipy.spatial.ConvexHull or array-like
        Either a SciPy ConvexHull object, or an array/list of vertices that define the convex hull.
        If vertices are provided, a ConvexHull object will be constructed internally.
    eps : float, optional
        Tolerance for numerical precision when checking if the point is inside the hull. Default is 1e-6.
    Returns
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