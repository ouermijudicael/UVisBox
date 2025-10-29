import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from uvisbox.Core.BandDepths.vector_depths import cartesian_to_polar, compute_vector_depths_2D
from .uncertainty_lobes_stats import calculate_spread_2D
from .uncertainty_lobes_vis import matplotlib_uncertainty_lobes_vis


def _process_position(args):
    """
    Worker function to process depths and spread for a single position (for parallelization).
    
    Parameters:
    -----------
    args : tuple
        (position_idx, ensemble_spherical, percentile1, percentile2)
    
    Returns:
    --------
    tuple : (position_idx, depths, theta1, theta2, mid_angle, r1, r2, r_arrow)
    """
    i_pos, ensemble_spherical, percentile1, percentile2 = args
    
    # Compute depths for this position
    depths = compute_vector_depths_2D(ensemble_spherical)
    
    # Calculate spread
    median_idx, min_mag, max_mag, min_angle, max_angle = calculate_spread_2D(
        ensemble_spherical, depths, percentile1
    )
    
    median_vector = ensemble_spherical[median_idx] if median_idx is not None else np.array([0, 0])
    theta1 = np.degrees([min_angle, max_angle])
    mid_angle = np.degrees(median_vector[1]) if median_idx is not None else 0
    r_arrow = median_vector[0] if median_idx is not None else 0
    r1 = min_mag
    
    if percentile2 is not None:
        _, _, max_mag2, min_angle2, max_angle2 = calculate_spread_2D(
            ensemble_spherical, depths, percentile2
        )
        theta2 = np.degrees([min_angle2, max_angle2])
        r2 = max_mag2
    else:
        theta2 = None
        r2 = 0.0
    
    return i_pos, depths, theta1, theta2, mid_angle, r1, r2, r_arrow


def uncertainty_lobes(positions, ensemble_vectors, percentile1=90, percentile2=50, 
                               scale=0.2, ax=None, show_median=True, workers=None):
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
    workers : int, optional
        Number of worker processes for parallel vector depth computation (default: None).
        If None or <= 1, uses sequential computation. For ensemble size >= 30, 
        parallelization can provide significant speedup.

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

    # Calculate vector depths and spread statistics
    # Use parallelization for larger problems (more positions or larger ensembles)
    use_parallel = workers is not None and workers > 1 and \
                   (num_positions >= 10 or num_ens_members >= 30)
    
    if use_parallel:
        # Parallel processing across positions
        ctx = mp.get_context('fork')
        pool = ctx.Pool(processes=workers)
        
        try:
            # Prepare arguments for each position
            args_list = [
                (i, ensemble_spherical_vectors[i], percentile1, percentile2)
                for i in range(num_positions)
            ]
            
            # Process all positions in parallel
            results = pool.map(_process_position, args_list)
            
            # Unpack results
            theta1 = np.zeros((num_positions, 2))
            theta2 = np.zeros((num_positions, 2)) if percentile2 is not None else None
            mid_angle = np.zeros(num_positions)
            r1 = np.zeros(num_positions)
            r2 = np.zeros(num_positions)
            r_arrow = np.zeros(num_positions)
            
            for i_pos, _, t1, t2, m_ang, r_1, r_2, r_arr in results:
                theta1[i_pos] = t1
                if theta2 is not None:
                    theta2[i_pos] = t2
                mid_angle[i_pos] = m_ang
                r1[i_pos] = r_1
                r2[i_pos] = r_2
                r_arrow[i_pos] = r_arr
        finally:
            pool.close()
            pool.join()
    else:
        # Sequential processing (optimized with vectorized depth computation)
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
            median_idx, min_mag, max_mag, min_angle, max_angle = calculate_spread_2D(
                ensemble_spherical_vectors[i_pos], depths[i_pos], percentile1
            )
            median_vector = ensemble_spherical_vectors[i_pos][median_idx] if median_idx is not None else np.array([0,0])
            theta1[i_pos] = np.degrees([min_angle, max_angle]) 
           
            mid_angle[i_pos] = np.degrees(median_vector[1]) if median_idx is not None else 0
            r_arrow[i_pos] = median_vector[0] if median_idx is not None else 0
            r1[i_pos] = min_mag

            if percentile2 is not None:
                r2[i_pos] = max_mag
                _, _, _, min_angle2, max_angle2 = calculate_spread_2D(
                    ensemble_spherical_vectors[i_pos], depths[i_pos], percentile2
                )
                theta2[i_pos] = np.degrees([min_angle2, max_angle2])
    
    # Apply scale to radii
    r1 = r1 * scale
    r2 = r2 * scale
    r_arrow = r_arrow * scale
    
    ax = matplotlib_uncertainty_lobes_vis(ax, positions, theta1, theta2, mid_angle, r1, r2, r_arrow, show_median)

    return ax

  