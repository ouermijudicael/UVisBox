import numpy as np
import matplotlib.pyplot as plt
from uvisbox.Core.BandDepths.vector_depths import cartesian_to_polar, compute_vector_depths_2D
from .uncertainty_lobes_stats import calculate_spread_2D
from .uncertainty_lobes_vis import matplotlib_uncertainty_lobes_vis


def uncertainty_lobes(positions, ensemble_vectors, percentile1=90, percentile2=50, 
                               scale=0.2, ax=None, show_median=True):
    """
    Draws uncertainty lobe glyphs for the given positions and ensemble vectors. This implemantation is inspired by
    M. Jarema, I. Demir, J. Kehrer and R. Westermann, "Comparative visual analysis of vector field ensembles," 2015 
    IEEE Conference on Visual Analytics Science and Technology (VAST), Chicago, IL, USA, 2015, pp. 81-88, 
    doi: 10.1109/VAST.2015.7347634. This implementation uses vector depth and doesn't fit ensemble to a 
    Gaussian Mixture Model as in the original paper. In addition, this implementation doesn't perform clustering
    of the vectors, instead it draws lobes for all vectors at each position.

    Parameters:
    -----------
    positions : numpy.ndarray
        Array of shape (n, 2) representing the positions of the lobe glyphs.
    ensemble_vectors : numpy.ndarray
        Array of shape (n, m, 2) representing the ensemble vectors for each position.
    percentile1 : float, optional
        The first percentile for depth filtering, range 0-100 (default: 90).
    percentile2 : float, optional
        The second percentile for depth filtering, range 0-100 (default: 50).
        If None, only one lobe is drawn.
    scale : float, optional
        The scale factor for the glyphs (default: 0.2).
    ax : matplotlib axis, optional
        The axis to draw on. If None, a new figure and axis will be created (default: None).
    show_median : bool, optional
        Whether to show the median vector as an arrow (default: True).

    Returns:
    --------
    ax : matplotlib axis
        The axis with the drawn lobe glyphs.
    """
    num_positions, num_ens_members = ensemble_vectors.shape[0], ensemble_vectors.shape[1]
    
    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert ensemble_vectors to spherical coordinates
    ensemble_spherical_vectors = np.zeros_like(ensemble_vectors)
    for i in range(num_positions):
        ensemble_spherical_vectors[i] = cartesian_to_polar(ensemble_vectors[i])

    # Ccalculate vector depths in spherical coordinates
    depths = np.zeros((num_positions, num_ens_members))
    for i in range(num_positions):
        depths[i] = compute_vector_depths_2D(ensemble_spherical_vectors[i])

    theta1 = np.zeros((num_positions, 2))
    theta2 = np.zeros((num_positions, 2)) if percentile2 is not None else None
    mid_angle = np.zeros(num_positions)
    r1 = np.zeros(num_positions)
    r2 = np.zeros(num_positions) 
    r_arrow = np.zeros(num_positions)
    for i_pos in range(num_positions):
        median_idx, min_mag, max_mag, min_angle, max_angle = calculate_spread_2D(ensemble_spherical_vectors[i_pos], depths[i_pos], percentile1)
        median_vector = ensemble_spherical_vectors[i_pos][median_idx] if median_idx is not None else np.array([0,0])
        theta1[i_pos] = np.degrees([min_angle, max_angle]) 
       
        mid_angle[i_pos] = np.degrees(median_vector[1]) if median_idx is not None else 0
        r_arrow[i_pos] = median_vector[0] * scale if median_idx is not None else 0
        r1[i_pos] = min_mag * scale

        if percentile2 is not None:
            r2[i_pos] = max_mag * scale
            _, _, _, min_angle2, max_angle2 = calculate_spread_2D(ensemble_spherical_vectors[i_pos], depths[i_pos], percentile2)
            theta2[i_pos] = np.degrees([min_angle2, max_angle2])
    ax =matplotlib_uncertainty_lobes_vis(ax, positions, theta1, theta2, mid_angle, r1, r2, r_arrow, show_median)

    return ax

  