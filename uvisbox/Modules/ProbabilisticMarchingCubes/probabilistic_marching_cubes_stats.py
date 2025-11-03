from uvisbox.Core.CellsCrossingProb.crossing_probability_mc import crossing_probability_cubes_monte_carlo


def probabilistic_marching_cube_summary_statistics(ensemble_images, isovalue):
    """
    Compute level crossing probability for probabilistic marching cubes.
    
    This function calculates the probability of an isosurface crossing through each
    cell in a 3D grid based on an ensemble of scalar fields.
    
    Parameters:
    -----------
        ensemble_images : np.ndarray
            4D array of shape (n_z, n_y, n_x, n_ensemble) representing the scalar 
            field with ensemble members.
        isovalue : float
            The isovalue for which to compute the isosurface crossing probability.
    
    Returns:
    --------
        probability_contour : np.ndarray
            3D array of shape (n_z-1, n_y-1, n_x-1) with probabilities of isosurface 
            presence in each cell. Values range from 0 to 1.
    """
    return crossing_probability_cubes_monte_carlo(ensemble_images, isovalue)
