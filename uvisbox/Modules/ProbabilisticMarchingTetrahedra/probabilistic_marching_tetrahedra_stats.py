from uvisbox.Core.CellsCrossingProb.crossing_probability_mc import crossing_probability_tetrahedra_monte_carlo


def probabilistic_marching_tetrahedron_summary_statistics(ensemble_data, tetrahedral_mesh, isovalue):
    """
    Compute level crossing probability for probabilistic marching tetrahedra.
    
    This function calculates the probability of an isosurface crossing through each
    tetrahedron in a 3D tetrahedral mesh based on an ensemble of scalar fields.
    
    Parameters:
    -----------
        ensemble_data : np.ndarray
            2D array of shape (n_points, n_ensemble) where each column is a realization 
            and each row corresponds to a vertex in the tetrahedral mesh.
        tetrahedral_mesh : np.ndarray
            2D array of shape (n_tetrahedra, 4) representing the tetrahedra. Each row 
            contains the indices of the four vertices forming a tetrahedron.
        isovalue : float
            The isovalue for which to compute the isosurface crossing probability.
    
    Returns:
    --------
        probability_contour : np.ndarray
            1D array of shape (n_tetrahedra,) with probabilities of isosurface presence 
            in each tetrahedron. Values range from 0 to 1.
    """
    return crossing_probability_tetrahedra_monte_carlo(ensemble_data, tetrahedral_mesh, isovalue)
