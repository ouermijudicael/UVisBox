from uvisbox.Core.CellsCrossingProb.crossing_probability_mc import crossing_prob_squares_mc


def probabilistic_marching_square_summary_statistics(ensemble_images, isovalue):
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
        prob_contour : np.ndarray
            2D array of shape (y_dim-1, x_dim-1) with probabilities of contour 
            presence in each cell. Values range from 0 to 1.
    """
    return crossing_prob_squares_mc(ensemble_images, isovalue)
