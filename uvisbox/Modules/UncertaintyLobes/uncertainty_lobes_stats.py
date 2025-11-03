"""
Statistics computation for uncertainty lobes.

This module computes vector depth statistics for uncertainty lobe visualization:
    - Convert Cartesian → Polar coordinates
    - Compute vector depths
    - Calculate spreads (min/max magnitude and angle)
"""

import numpy as np
import multiprocessing as mp
from uvisbox.Core.BandDepths.vector_depths import (
    cartesian_to_polar, 
    compute_vector_depths_2D,
    calculate_spread_2D
)


def _process_position_stats(args):
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
    
    # Calculate spread for first percentile
    median_idx, min_mag, max_mag, min_angle, max_angle = calculate_spread_2D(
        ensemble_spherical, depths, percentile1
    )
    
    median_vector = ensemble_spherical[median_idx] if median_idx is not None else np.array([0, 0])
    theta1 = np.array([min_angle, max_angle])
    mid_angle = median_vector[1] if median_idx is not None else 0
    r_arrow = median_vector[0] if median_idx is not None else 0
    r1 = min_mag
    
    # Calculate spread for second percentile (if provided)
    if percentile2 is not None:
        _, _, max_mag2, min_angle2, max_angle2 = calculate_spread_2D(
            ensemble_spherical, depths, percentile2
        )
        theta2 = np.array([min_angle2, max_angle2])
        r2 = max_mag2
    else:
        theta2 = None
        r2 = 0.0
    
    return i_pos, depths, theta1, theta2, mid_angle, r1, r2, r_arrow


def uncertainty_lobes_summary_statistics(ensemble_vectors, percentile1=90, percentile2=50, workers=None):
    """
    Compute vector depth statistics for 2D uncertainty lobes.
    
    Parameters:
    -----------
    ensemble_vectors : numpy.ndarray
        Shape (n, m, 2) - Cartesian ensemble vectors at n positions with m members
    percentile1 : float (0-100)
        First percentile for depth filtering (outer lobe). Higher values include more vectors.
        Default: 90
    percentile2 : float or None (0-100)
        Second percentile for depth filtering (inner lobe). If None, only one lobe is drawn.
        Default: 50
    workers : int, optional
        Number of parallel workers for computation. Default is None (sequential).
        For ensemble size >= 30, parallelization can provide speedup.
    
    Returns:
    --------
    stats_2d : dict
        {
            'ensemble_polar_vectors': (n, m, 2) - polar coordinates [r, theta],
            'depths': (n, m) - vector depths,
            'median_vectors': (n, 2) - median vectors per position [r, theta],
            'theta1': (n, 2) - angular spread for percentile1 [min_angle, max_angle],
            'theta2': (n, 2) or None - angular spread for percentile2 (if provided),
            'mid_angle': (n,) - median angles (radians),
            'r1': (n,) - minimum magnitude for percentile1,
            'r2': (n,) - maximum magnitude for percentile2 (or 0 if None),
            'r_arrow': (n,) - median vector magnitudes
        }
    """
    num_positions, num_ens_members = ensemble_vectors.shape[0], ensemble_vectors.shape[1]
    
    # Convert ensemble_vectors to polar coordinates
    ensemble_polar_vectors = np.zeros_like(ensemble_vectors)
    for i in range(num_positions):
        ensemble_polar_vectors[i] = cartesian_to_polar(ensemble_vectors[i])

    # Calculate vector depths and spread statistics
    # Use parallelization for larger problems
    use_parallel = workers is not None and workers > 1 and \
                   (num_positions >= 10 or num_ens_members >= 30)
    
    if use_parallel:
        # Parallel processing across positions
        ctx = mp.get_context('fork')
        pool = ctx.Pool(processes=workers)
        
        try:
            # Prepare arguments for each position
            args_list = [
                (i, ensemble_polar_vectors[i], percentile1, percentile2)
                for i in range(num_positions)
            ]
            
            # Process all positions in parallel
            results = pool.map(_process_position_stats, args_list)
            
            # Unpack results
            depths = np.zeros((num_positions, num_ens_members))
            theta1 = np.zeros((num_positions, 2))
            theta2 = np.zeros((num_positions, 2)) if percentile2 is not None else None
            mid_angle = np.zeros(num_positions)
            r1 = np.zeros(num_positions)
            r2 = np.zeros(num_positions)
            r_arrow = np.zeros(num_positions)
            
            for i_pos, d, t1, t2, m_ang, r_1, r_2, r_arr in results:
                depths[i_pos] = d
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
        # Sequential processing
        depths = np.zeros((num_positions, num_ens_members))
        for i in range(num_positions):
            depths[i] = compute_vector_depths_2D(ensemble_polar_vectors[i])

        theta1 = np.zeros((num_positions, 2))
        theta2 = np.zeros((num_positions, 2)) if percentile2 is not None else None
        mid_angle = np.zeros(num_positions)
        r1 = np.zeros(num_positions)
        r2 = np.zeros(num_positions) 
        r_arrow = np.zeros(num_positions)
        
        for i_pos in range(num_positions):
            median_idx, min_mag, max_mag, min_angle, max_angle = calculate_spread_2D(
                ensemble_polar_vectors[i_pos], depths[i_pos], percentile1
            )
            median_vector = ensemble_polar_vectors[i_pos][median_idx] if median_idx is not None else np.array([0, 0])
            theta1[i_pos] = np.array([min_angle, max_angle])
            mid_angle[i_pos] = median_vector[1] if median_idx is not None else 0
            r_arrow[i_pos] = median_vector[0] if median_idx is not None else 0
            r1[i_pos] = min_mag

            if percentile2 is not None:
                r2[i_pos] = max_mag
                _, _, _, min_angle2, max_angle2 = calculate_spread_2D(
                    ensemble_polar_vectors[i_pos], depths[i_pos], percentile2
                )
                theta2[i_pos] = np.array([min_angle2, max_angle2])
    
    # Extract median vectors
    median_vectors = np.zeros((num_positions, 2))
    for i in range(num_positions):
        median_idx = np.argmax(depths[i])
        median_vectors[i] = ensemble_polar_vectors[i][median_idx]
    
    return {
        'ensemble_polar_vectors': ensemble_polar_vectors,
        'depths': depths,
        'median_vectors': median_vectors,
        'theta1': theta1,
        'theta2': theta2,
        'mid_angle': mid_angle,
        'r1': r1,
        'r2': r2,
        'r_arrow': r_arrow
    }