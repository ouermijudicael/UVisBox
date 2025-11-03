from uvisbox.Core.CellsCrossingProb.crossing_probability_mc import crossing_probability_triangles_monte_carlo


def probabilistic_marching_triangles_summary_statistics(ensemble_data, triangle_mesh, isovalue):
    """
    Compute level crossing probability for probabilistic marching triangles.
    
    This function calculates the probability of an isocontour crossing through each
    triangle in a 2D triangular mesh based on an ensemble of scalar fields.
    
    Parameters:
    -----------
        ensemble_data : np.ndarray
            2D array of shape (n_points, n_ensemble) where each column is a realization 
            and each row corresponds to a vertex in the triangular mesh.
        triangle_mesh : np.ndarray
            2D array of shape (n_triangles, 3) representing the triangles. Each row 
            contains the indices of the three vertices forming a triangle.
        isovalue : float
            The isovalue for which to compute the contour crossing probability.
    
    Returns:
    --------
        probability_contour : np.ndarray
            1D array of shape (n_triangles,) with probabilities of contour presence 
            in each triangle. Values range from 0 to 1.
    """
    return crossing_probability_triangles_monte_carlo(ensemble_data, triangle_mesh, isovalue)
