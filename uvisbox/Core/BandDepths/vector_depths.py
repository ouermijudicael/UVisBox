import numpy as np
from itertools import combinations
import multiprocessing as mp


def _compute_depth_for_vector(args):
    """
    Worker function to compute depth for a single vector (for parallelization).
    
    Parameters:
    -----------
    args : tuple
        (vector_idx, angles, magnitudes, pairs)
    
    Returns:
    --------
    int : depth count for this vector
    """
    i, angles, magnitudes, pairs = args
    angle_i = angles[i]
    mag_i = magnitudes[i]
    
    depth = 0
    for j, k in pairs:
        angle_j, angle_k = angles[j], angles[k]
        mag_j, mag_k = magnitudes[j], magnitudes[k]
        
        # Check if vector i is between vectors j and k
        if (angle_j <= angle_i <= angle_k or angle_k <= angle_i <= angle_j) and \
           (min(mag_j, mag_k) <= mag_i <= max(mag_j, mag_k)):
            depth += 1
    
    return depth


def calculate_spread_2D(vectors, depths, percentile):
    """
    Calculate the spread in 2D polar coordinates.
    Parameters:
    -----------
        vectors : numpy.ndarray
            Array of shape (n, 2) in polar coordinates (magnitude, angle)
        depths : numpy.ndarray
            Array of shape (n,) representing the depth of each vector
        percentile : float
            Percentile of vectors to include based on depth ranking (0-100).
            Higher values include more vectors.
            - percentile=50: Include top 50% deepest vectors
            - percentile=95: Include top 95% deepest vectors (typical)
            - percentile=100: Include ALL vectors
    Returns:
    --------
        tuple
            Indices of vectors with min/max magnitude, angle among those passing depth threshold
    """
    # FIXED: Use proper percentile semantics
    # percentile=95 means keep top 95% by depth = depth >= 5th percentile
    depth_threshold = np.percentile(depths, 100.0 - percentile)
    
    first_quadrant = False # Set to True if you want to restrict to first quadrant 0 to pi/2
    second_quadrant = False # Set to True if you want to restrict to second quadrant pi/2 to pi
    third_quadrant = False # Set to True if you want to restrict to third quadrant -pi to -pi/2
    fourth_quadrant = False # Set to True if you want to restrict to fourth quadrant -pi/2 to 0
    indices = np.where(depths >= depth_threshold)[0]
    median_idx = np.argmax(depths)
    if indices.size > 0:
        filtered_vectors = vectors[indices]
        # Check if any vector is in the first quadrant
        first_quadrant = np.any((filtered_vectors[:, 1] >= 0) & (filtered_vectors[:, 1] <= np.pi/2)) 
        second_quadrant = np.any((filtered_vectors[:, 1] > np.pi/2) & (filtered_vectors[:, 1] <= np.pi))
        third_quadrant = np.any((filtered_vectors[:, 1] < -np.pi/2) & (filtered_vectors[:, 1] >= -np.pi))
        fourth_quadrant = np.any((filtered_vectors[:, 1] < 0) & (filtered_vectors[:, 1] >= -np.pi/2))
        if ((not first_quadrant and second_quadrant and third_quadrant and not fourth_quadrant ) or  
           (not first_quadrant and second_quadrant and third_quadrant and fourth_quadrant ) or  
              (first_quadrant and second_quadrant and third_quadrant and not fourth_quadrant )):
            # find smallest positive angle and smallest negative angle
            pos_angles = filtered_vectors[filtered_vectors[:, 1] >= 0]
            neg_angles = filtered_vectors[filtered_vectors[:, 1] < 0]
            min_angle = np.min(pos_angles[:, 1])
            max_angle = np.max(neg_angles[:, 1])
            min_mag = np.min(filtered_vectors[:, 0])
            max_mag = np.max(filtered_vectors[:, 0])
        else:           
            min_angle = np.min(filtered_vectors[:, 1])
            max_angle = np.max(filtered_vectors[:, 1])
            min_mag = np.min(filtered_vectors[:, 0])
            max_mag = np.max(filtered_vectors[:, 0])
    else:
        min_angle, max_angle = 0.0, 0.0
        min_mag, max_mag = 0.0, 0.0
    return median_idx, min_mag, max_mag, min_angle, max_angle

def calculate_spread_3D(vectors, depths, percentile):
    """
    Calculate the spread in 3D spherical coordinates.

    Parameters:
    -----------
        vectors : numpy.ndarray
            Array of shape (n, 3) in spherical coordinates (magnitude, theta, phi)
        depths : numpy.ndarray
            Array of shape (n,) representing the depth of each vector
        percentile : float
            Percentile of vectors to include based on depth ranking (0-100).
            Higher values include more vectors.
            - percentile=50: Include top 50% deepest vectors
            - percentile=95: Include top 95% deepest vectors (typical)
            - percentile=100: Include ALL vectors

    Returns:
    --------
        tuple
            Indices of vectors with min/max magnitude, theta, phi among those passing depth threshold
    """
    # FIXED: Use proper percentile semantics
    # percentile=95 means keep top 95% by depth = depth >= 5th percentile
    depth_threshold = np.percentile(depths, 100.0 - percentile)
    
    indices = np.where(depths >= depth_threshold)[0]
    median_idx = np.argmax(depths)
    if indices.size > 0:
        filtered_vectors = vectors[indices]
        min_phi_idx = np.argmin(filtered_vectors[:, 2])
        max_phi_idx = np.argmax(filtered_vectors[:, 2])
        min_theta_idx = np.argmin(filtered_vectors[:, 1])
        max_theta_idx = np.argmax(filtered_vectors[:, 1])
        min_mag_idx = np.argmin(filtered_vectors[:, 0])
        max_mag_idx = np.argmax(filtered_vectors[:, 0])
    else:
        min_phi_idx, max_phi_idx = None, None
        min_theta_idx, max_theta_idx = None, None
        min_mag_idx, max_mag_idx = None, None
    return median_idx, min_mag_idx, max_mag_idx, min_theta_idx, max_theta_idx, min_phi_idx, max_phi_idx


def cartesian_to_polar(vectors):
    """
    Convert 2D Cartesian vectors to polar coordinates (magnitude, angle).

    Parameters:
    -----------
    vectors : numpy.ndarray
        Array of shape (n, 2) representing 2D Cartesian vectors.

    Returns:
    --------
    polar_coords : numpy.ndarray
        Array of shape (n, 2) with columns [magnitude, angle in radians].
    """
    magnitudes = np.linalg.norm(vectors, axis=1)
    valid_indices = magnitudes > 1.0e-8  # Filter vectors with norm greater than 1.0e-8
    angles = np.zeros_like(magnitudes)
    angles[valid_indices] = np.arctan2(vectors[valid_indices, 1], vectors[valid_indices, 0])
    angles = np.unwrap(angles)  # Unwrap to ensure continuity in angles
    angles = (angles + np.pi) % (2 * np.pi) - np.pi  # Ensure angles are in the range [-pi, pi]
    polar_coords = np.column_stack((magnitudes, angles))
    return polar_coords


def cartesian_to_spherical(vectors):
    """
    Convert 3D Cartesian vectors to spherical coordinates (magnitude, theta, phi).

    Parameters:
    -----------
        vectors : numpy.ndarray
            Array of shape (n, 3) representing 3D Cartesian vectors.
    
    Returns:
    --------
        spherical_coords : numpy.ndarray
            Array of shape (n, 3) with columns [magnitude, theta, phi].
    """
    magnitudes = np.linalg.norm(vectors, axis=1)
    valid_indices = magnitudes > 1.0e-8  # Filter vectors with norm greater than 1.0e-8
    theta = np.zeros_like(magnitudes)
    phi = np.zeros_like(magnitudes)
    theta[valid_indices] = np.arccos(np.clip(vectors[valid_indices, 2] / magnitudes[valid_indices], -1.0, 1.0))  # Clip to handle numerical precision issues
    phi[valid_indices] = np.arctan2(vectors[valid_indices, 1], vectors[valid_indices, 0])
    phi = np.unwrap(phi)  # Unwrap to ensure continuity in angles
    spherical_coords = np.column_stack((magnitudes, theta, phi))
    # if np.any(np.isnan(spherical_coords)):
    #     print("Invalid vectors detected:", vectors)
    #     print("Spherical coordinates:", spherical_coords)
    #     raise ValueError("NaN values found in spherical coordinates conversion.")
    return spherical_coords


def angular_spread(vectors):
    """
    Calculate the angular spread and magnitude spread of a set of vectors.

    Parameters
    ----------
        vectors : numpy.ndarray
            Array of shape (n, 2) representing 2D Cartesian vectors.

    Returns
    -------
        angular_spread_deg : float
            The angular spread in degrees.
        magnitude_spread : float
            The magnitude spread.
        min_idx : int
            Index of the vector with the minimum angle.
        max_idx : int
            Index of the vector with the maximum angle.
    """
    # Calculate the angular spread (max-min angle from origin) and return indices
    angles_from_origin = np.arctan2(vectors[:, 1], vectors[:, 0])
    angles_unwrapped = np.unwrap(angles_from_origin)
    min_idx = np.argmin(angles_unwrapped)
    max_idx = np.argmax(angles_unwrapped)
    angular_spread = angles_unwrapped[max_idx] - angles_unwrapped[min_idx]
    angular_spread_deg = np.degrees(angular_spread)
    return angular_spread_deg, min_idx, max_idx


def magnitude_spread(vectors):
    """
    Calculate the magnitude spread of a set of vectors.
    Parameters
    ----------
        vectors : numpy.ndarray
            Array of shape (n, 2) representing 2D Cartesian vectors.
    Returns
    -------
        magnitude_spread : float
            The magnitude spread.
        min_mag_idx : int
            Index of the vector with the minimum magnitude.
        max_mag_idx : int
            Index of the vector with the maximum magnitude.
    """
    # Calculate the magnitude spread (max-min magnitude) and return indices
    magnitudes = np.linalg.norm(vectors, axis=1)
    min_mag_idx = np.argmin(magnitudes)
    max_mag_idx = np.argmax(magnitudes)
    magnitude_spread = magnitudes[max_mag_idx] - magnitudes[min_mag_idx]
    return magnitude_spread, min_mag_idx, max_mag_idx


def compute_vector_depths_2D(vectors, workers=None):
    """
    Compute the depth of each vector in the ensemble using optimized vectorized operations.
    The vector depth calculation is based on T. A. J. Ouermi, J. Li, Z. Morrow, 
    B. Van Bloemen Waanders and C. R. Johnson, "Glyph-Based Uncertainty Visualization 
    and Analysis of Time-Varying Vector Fields," 2024 IEEE Workshop on Uncertainty 
    Visualization: Applications, Techniques, Software, and Decision Frameworks, 
    St Pete Beach, FL, USA, 2024, pp. 73-77, 
    doi: 10.1109/UncertaintyVisualization63963.2024.00014.

    Parameters:
    -----------
        vectors : numpy.ndarray
            Array of shape (n, 2) representing 2D Cartesian vectors.
        workers : int, optional
            Number of worker processes for parallel computation. Default is None (sequential).
            Set to a positive integer to enable multiprocessing.

    Returns:
    --------
        depths : numpy.ndarray
            Array of shape (n,) with the depth of each vector.
    
    Notes:
    ------
        Optimized version using:
        1. Vectorized angle/magnitude computation (precomputed once)
        2. Algorithm optimization (avoid redundant computations)
        3. Optional parallelization for large ensembles
    """
    n = vectors.shape[0]
    
    # Fast path for small ensembles
    if n < 3:
        return np.zeros(n)
    
    # OPTIMIZATION 1: Precompute all angles and magnitudes once (vectorized)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    magnitudes = np.linalg.norm(vectors, axis=1)
    
    # Generate pairs once
    pairs = list(combinations(np.arange(n), 2))
    n_pairs = len(pairs)
    
    # OPTIMIZATION 2: Vectorized depth computation
    # For small ensembles, vectorized version is faster than parallelization overhead
    if workers is None or workers <= 1 or n < 30:
        depths = np.zeros(n, dtype=np.int32)
        
        # Convert pairs to arrays for vectorized operations
        pairs_arr = np.array(pairs)
        j_indices = pairs_arr[:, 0]
        k_indices = pairs_arr[:, 1]
        
        # Precompute pair properties (vectorized)
        angles_j = angles[j_indices]
        angles_k = angles[k_indices]
        mags_j = magnitudes[j_indices]
        mags_k = magnitudes[k_indices]
        
        # Precompute min/max for magnitudes
        min_mags = np.minimum(mags_j, mags_k)
        max_mags = np.maximum(mags_j, mags_k)
        
        # Compute depth for each vector
        for i in range(n):
            angle_i = angles[i]
            mag_i = magnitudes[i]
            
            # Vectorized angle check: i is between j and k
            angle_between = ((angles_j <= angle_i) & (angle_i <= angles_k)) | \
                           ((angles_k <= angle_i) & (angle_i <= angles_j))
            
            # Vectorized magnitude check: i is between j and k
            mag_between = (min_mags <= mag_i) & (mag_i <= max_mags)
            
            # Count how many pairs satisfy both conditions
            depths[i] = np.sum(angle_between & mag_between)
    
    else:
        # OPTIMIZATION 3: Parallel computation for large ensembles
        ctx = mp.get_context('fork')
        pool = ctx.Pool(processes=workers)
        
        try:
            # Prepare arguments for each vector
            args_list = [(i, angles, magnitudes, pairs) for i in range(n)]
            
            # Compute depths in parallel
            depths = np.array(pool.map(_compute_depth_for_vector, args_list), dtype=np.int32)
        finally:
            pool.close()
            pool.join()
    
    # Scale depths between 0 and 1
    depths = depths.astype(np.float64)
    if depths.max() > 0.0:
        depths = (depths - depths.min()) / (depths.max() - depths.min())
    
    return depths


def _compute_depth_for_vector_3D(args):
    """
    Worker function to compute depth for a single 3D vector (for parallelization).
    
    Parameters:
    -----------
    args : tuple
        (vector_idx, magnitudes, thetas, phis, pairs)
    
    Returns:
    --------
    int : depth count for this vector
    """
    i, magnitudes, thetas, phis, pairs = args
    mag_i = magnitudes[i]
    theta_i = thetas[i]
    phi_i = phis[i]
    
    depth = 0
    for j, k in pairs:
        mag_j, mag_k = magnitudes[j], magnitudes[k]
        theta_j, theta_k = thetas[j], thetas[k]
        phi_j, phi_k = phis[j], phis[k]
        
        # Check if vector i is between vectors j and k in spherical coordinates
        if (theta_j < theta_i < theta_k or theta_k < theta_i < theta_j) and \
           (phi_j < phi_i < phi_k or phi_k < phi_i < phi_j) and \
           (min(mag_j, mag_k) < mag_i < max(mag_j, mag_k)):
            depth += 1
    
    return depth


def compute_vector_depths_3D(vectors, workers=None):
    """
    Compute the depth of each vector in the ensemble in 3D using optimized vectorized operations.
    Assumes vectors are in spherical coordinates (magnitude, theta, phi). The vector depth 
    calculation is based on T. A. J. Ouermi, J. Li, Z. Morrow, B. Van Bloemen Waanders and 
    C. R. Johnson, "Glyph-Based Uncertainty Visualization and Analysis of Time-Varying Vector 
    Fields," 2024 IEEE Workshop on Uncertainty Visualization: Applications, Techniques, Software, 
    and Decision Frameworks, St Pete Beach, FL, USA, 2024, pp. 73-77, 
    doi: 10.1109/UncertaintyVisualization63963.2024.00014.

    Parameters:
    -----------
        vectors : numpy.ndarray
            Array of shape (n, 3) representing 3D spherical coordinates (magnitude, theta, phi).
        workers : int, optional
            Number of worker processes for parallel computation. Default is None (sequential).
            Set to a positive integer to enable multiprocessing.

    Returns:
    --------
        depths : numpy.ndarray
            Array of shape (n,) with the depth of each vector.
    
    Notes:
    ------
        Optimized version using:
        1. Vectorized magnitude/angle computation (precomputed once)
        2. Algorithm optimization (avoid redundant computations)
        3. Optional parallelization for large ensembles
    """
    n = vectors.shape[0]
    
    # Fast path for small ensembles
    if n < 3:
        return np.zeros(n)
    
    # OPTIMIZATION 1: Precompute all components once (vectorized)
    magnitudes = vectors[:, 0]
    thetas = vectors[:, 1]
    phis = vectors[:, 2]
    
    # Generate pairs once
    pairs = list(combinations(np.arange(n), 2))
    
    # OPTIMIZATION 2: Vectorized depth computation
    # For small ensembles, vectorized version is faster than parallelization overhead
    if workers is None or workers <= 1 or n < 30:
        depths = np.zeros(n, dtype=np.int32)
        
        # Convert pairs to arrays for vectorized operations
        pairs_arr = np.array(pairs)
        j_indices = pairs_arr[:, 0]
        k_indices = pairs_arr[:, 1]
        
        # Precompute pair properties (vectorized)
        mags_j = magnitudes[j_indices]
        mags_k = magnitudes[k_indices]
        thetas_j = thetas[j_indices]
        thetas_k = thetas[k_indices]
        phis_j = phis[j_indices]
        phis_k = phis[k_indices]
        
        # Precompute min/max for magnitudes
        min_mags = np.minimum(mags_j, mags_k)
        max_mags = np.maximum(mags_j, mags_k)
        
        # Compute depth for each vector
        for i in range(n):
            mag_i = magnitudes[i]
            theta_i = thetas[i]
            phi_i = phis[i]
            
            # Vectorized theta check: i is between j and k
            theta_between = ((thetas_j < theta_i) & (theta_i < thetas_k)) | \
                           ((thetas_k < theta_i) & (theta_i < thetas_j))
            
            # Vectorized phi check: i is between j and k
            phi_between = ((phis_j < phi_i) & (phi_i < phis_k)) | \
                         ((phis_k < phi_i) & (phi_i < phis_j))
            
            # Vectorized magnitude check: i is between j and k (strict inequality)
            mag_between = (min_mags < mag_i) & (mag_i < max_mags)
            
            # Count how many pairs satisfy all three conditions
            depths[i] = np.sum(theta_between & phi_between & mag_between)
    
    else:
        # OPTIMIZATION 3: Parallel computation for large ensembles
        ctx = mp.get_context('fork')
        pool = ctx.Pool(processes=workers)
        
        try:
            # Prepare arguments for each vector
            args_list = [(i, magnitudes, thetas, phis, pairs) for i in range(n)]
            
            # Compute depths in parallel
            depths = np.array(pool.map(_compute_depth_for_vector_3D, args_list), dtype=np.int32)
        finally:
            pool.close()
            pool.join()
    
    # Scale depths between 0 and 1
    depths = depths.astype(np.float64)
    if depths.max() > 0.0:
        depths = (depths - depths.min()) / (depths.max() - depths.min())
    
    return depths

