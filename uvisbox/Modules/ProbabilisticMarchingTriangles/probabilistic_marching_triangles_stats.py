from uvisbox.Core.CellsCrossingProb.crossing_probability_mc import crossing_probability_triangles_monte_carlo


def probabilistic_marching_triangles_summary_statistics(ensemble_data, triangle_mesh, isovalue):
    """
    Compute level crossing probability for probabilistic marching triangles.
    
    This function calculates the probability of an isocontour crossing through each
    triangle in a 2D triangular mesh based on an ensemble of scalar fields.
    
    Parameters
    -----------
    ensemble_data : np.ndarray
        2D array of shape ``(n_points, n_ensemble)`` where each column is a
        realization and each row corresponds to a mesh vertex.
    triangle_mesh : np.ndarray
        2D array of shape ``(n_triangles, 3)`` containing vertex indices.
    isovalue : float
        The isovalue for which to compute the contour crossing probability.
    
    Returns
    --------
    dict
        Contains ``level_crossing_probability``, a 1D array of shape
        ``(n_triangles,)`` with values from 0 to 1.
    """
    probability_contour = crossing_probability_triangles_monte_carlo(ensemble_data, triangle_mesh, isovalue)
    return {'level_crossing_probability': probability_contour}
