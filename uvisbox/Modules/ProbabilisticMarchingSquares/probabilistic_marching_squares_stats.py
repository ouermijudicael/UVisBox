from uvisbox.Core.CellsCrossingProb.crossing_probability_mc import crossing_probability_squares_monte_carlo


def probabilistic_marching_squares_summary_statistics(ensemble_images, isovalue):
    """
    Compute level crossing probability for probabilistic marching squares.
    
    This function calculates the probability of an isocontour crossing through each
    cell in a 2D grid based on an ensemble of scalar fields.
    
    Parameters:
    -----------
        ensemble_images : np.ndarray
            3D array of shape (y_dim, x_dim, n_ensemble) representing the scalar 
            field with ensemble members.
        isovalue : float
            The isovalue for which to compute the contour crossing probability.
    
    Returns:
    --------
        dict
            Dictionary containing:
            - 'level_crossing_probability': np.ndarray
                2D array of shape (y_dim-1, x_dim-1) with probabilities of contour 
                presence in each cell. Values range from 0 to 1.
    """
    prob_contour = crossing_probability_squares_monte_carlo(ensemble_images, isovalue)
    return {'level_crossing_probability': prob_contour}
