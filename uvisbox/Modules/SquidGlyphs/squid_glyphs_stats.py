 
import numpy as np
from sklearn.decomposition import PCA
from uvisbox.Core.BandDepths.vector_depths import calculate_spread_2D, compute_vector_depths_2D
from uvisbox.Core.BandDepths.vector_depths import calculate_spread_3D, compute_vector_depths_3D
from uvisbox.Core.BandDepths.vector_depths import cartesian_to_polar, cartesian_to_spherical


def squid_glyphs_2d_summary_statistics(ensemble_vectors, percentile, workers=None):
    """
    Compute vector depth statistics for 2D squid glyphs.
    
    Parameters:
    -----------
    ensemble_vectors : numpy.ndarray
        Shape (n, m, 2) - Cartesian ensemble vectors
    percentile : float
        Percentile of ensemble members to include based on depth ranking (0-100).
        Higher values include more vectors (larger glyphs showing more variation).
        - percentile=50: Include top 50% deepest vectors
        - percentile=95: Include top 95% deepest vectors (typical)
        - percentile=100: Include ALL vectors (maximum variation)
    workers : int, optional
        Number of parallel workers for depth computation. Default is None (sequential)
    
    Returns:
    --------
    stats_2d : dict
        {
            'ensemble_polar_vectors': (n, m, 2) - polar coordinates,
            'depths': (n, m) - vector depths,
            'median_vectors': (n, 2) - median vectors per position,
            'min_mag': (n,) - min magnitude per position,
            'max_mag': (n,) - max magnitude per position,
            'min_angle': (n,) - min angle per position,
            'max_angle': (n,) - max angle per position
        }
    """
    num_positions, num_ensemble = ensemble_vectors.shape[0], ensemble_vectors.shape[1]
    
    # Convert to polar coordinates
    ensemble_polar_vectors = np.zeros_like(ensemble_vectors)
    for i in range(num_positions):
        ensemble_polar_vectors[i] = cartesian_to_polar(ensemble_vectors[i])
    
    # Compute depths
    depths = np.zeros((num_positions, num_ensemble))
    for i in range(num_positions):
        depths[i] = compute_vector_depths_2D(ensemble_polar_vectors[i], workers=workers)
    
    # Calculate spreads using CORRECTED percentile semantics
    # percentile=95 means "keep top 95% by depth" = depth >= 5th percentile
    median_vectors = np.zeros((num_positions, 2))
    min_mags = np.zeros(num_positions)
    max_mags = np.zeros(num_positions)
    min_angles = np.zeros(num_positions)
    max_angles = np.zeros(num_positions)
    
    for i in range(num_positions):
        median_idx = np.argmax(depths[i])
        median_vectors[i] = ensemble_polar_vectors[i][median_idx]
        
        # FIXED: Use np.percentile to compute depth threshold
        # percentile=95 → keep vectors with depth >= 5th percentile of depths
        depth_threshold = np.percentile(depths[i], 100.0 - percentile)
        indices = np.where(depths[i] >= depth_threshold)[0]
        
        if indices.size > 0:
            min_mags[i] = np.min(ensemble_polar_vectors[i][indices][:, 0])
            max_mags[i] = np.max(ensemble_polar_vectors[i][indices][:, 0])
            min_angles[i] = np.min(ensemble_polar_vectors[i][indices][:, 1])
            max_angles[i] = np.max(ensemble_polar_vectors[i][indices][:, 1])
    
    return {
        'ensemble_polar_vectors': ensemble_polar_vectors,
        'depths': depths,
        'median_vectors': median_vectors,
        'min_mag': min_mags,
        'max_mag': max_mags,
        'min_angle': min_angles,
        'max_angle': max_angles
    }


def squid_glyphs_3d_summary_statistics(ensemble_vectors, percentile):
    """
    Compute vector depth statistics for 3D squid glyphs.
    
    Parameters:
    -----------
    ensemble_vectors : numpy.ndarray
        Shape (n, m, 3) - Cartesian ensemble vectors
    percentile : float
        Percentile of ensemble members to include based on depth ranking (0-100).
        Higher values include more vectors (larger glyphs showing more variation).
        - percentile=50: Include top 50% deepest vectors
        - percentile=95: Include top 95% deepest vectors (typical)
        - percentile=100: Include ALL vectors (maximum variation)
    
    Returns:
    --------
    stats_3d : dict
        {
            'ensemble_spherical_vectors': (n, m, 3) - spherical coordinates,
            'depths': (n, m) - vector depths,
            'median_vectors': (n, 3) - median vectors,
            'min_vectors': (n, 3) - min spread vectors,
            'max_vectors': (n, 3) - max spread vectors,
            'glyph_markers': (n,) - glyph type markers,
            'directional_variations': (n, 3, 2) - PCA components,
            'num_glyphs': int - count of full glyphs
        }
    """
    num_positions, num_ensemble = ensemble_vectors.shape[0], ensemble_vectors.shape[1]
    
    # Convert to spherical coordinates
    ensemble_spherical_vectors = np.zeros_like(ensemble_vectors)
    for i in range(num_positions):
        ensemble_spherical_vectors[i] = cartesian_to_spherical(ensemble_vectors[i])
    
    # Compute depths
    depths = np.zeros((num_positions, num_ensemble))
    for i in range(num_positions):
        depths[i] = compute_vector_depths_3D(ensemble_spherical_vectors[i])
    
    # Calculate spreads and build min/max/median vectors
    min_vectors = np.zeros((num_positions, 3))
    max_vectors = np.zeros((num_positions, 3))
    median_vectors = np.zeros((num_positions, 3))
    glyph_markers = np.zeros(num_positions, dtype=int)
    num_glyphs = 0
    
    for i in range(num_positions):
        median_idx, min_mag_idx, max_mag_idx, min_theta_idx, max_theta_idx, min_phi_idx, max_phi_idx = \
            calculate_spread_3D(ensemble_spherical_vectors[i], depths[i], percentile)
        
        # Median vector
        median_vectors[i] = ensemble_spherical_vectors[i][median_idx] if median_idx is not None else [0, 0, 0]
        
        # Min/max vectors
        min_mag_vec = ensemble_spherical_vectors[i][min_mag_idx] if min_mag_idx is not None else [0, 0, 0]
        max_mag_vec = ensemble_spherical_vectors[i][max_mag_idx] if max_mag_idx is not None else [0, 0, 0]
        min_phi = ensemble_spherical_vectors[i][min_phi_idx][2] if min_phi_idx is not None else 0
        max_phi = ensemble_spherical_vectors[i][max_phi_idx][2] if max_phi_idx is not None else 0
        min_theta = ensemble_spherical_vectors[i][min_theta_idx][1] if min_theta_idx is not None else 0
        max_theta = ensemble_spherical_vectors[i][max_theta_idx][1] if max_theta_idx is not None else 0
        
        min_vectors[i] = [min_mag_vec[0], min_theta, min_phi]
        max_vectors[i] = [max_mag_vec[0], max_theta, max_phi]
        
        # Classify glyph type
        if (np.abs(max_vectors[i][0] - min_vectors[i][0]) > 1e-5) and \
           (np.abs(max_vectors[i][1] - min_vectors[i][1]) > 1e-5) and \
           (np.abs(max_vectors[i][2] - min_vectors[i][2]) > 1e-5):
            glyph_markers[i] = 1  # Full glyph
            num_glyphs += 1
        elif max_vectors[i][0] > 1e-3:
            glyph_markers[i] = 2  # Arrow only
    
    # Compute directional variations (PCA)
    # FIXED: Pass percentile directly - getDirectionalVariations will handle threshold
    directional_variations = getDirectionalVariations(
        ensemble_spherical_vectors, depths, percentile, 
        min_vectors, median_vectors, max_vectors
    )
    
    return {
        'ensemble_spherical_vectors': ensemble_spherical_vectors,
        'depths': depths,
        'median_vectors': median_vectors,
        'min_vectors': min_vectors,
        'max_vectors': max_vectors,
        'glyph_markers': glyph_markers,
        'directional_variations': directional_variations,
        'num_glyphs': num_glyphs
    }



def getDirectionalVariations(vectors, depths, percentile, min_vectors, median_vectors, max_vectors):
    """
    Compute the directional variation of the vectors.
    
    Parameters:
    ----------
    vectors : numpy.ndarray
        Array of shape (num_points, num_ensemble_members, 3) where the last dimension contains the vector components.
    depths : numpy.ndarray
        Array of shape (num_points, num_ensemble_members) containing depth values for each vector.
    percentile : float
        Percentile of ensemble members to include (0-100). Higher values include more vectors.
    min_vectors : numpy.ndarray
        Array of shape (num_points, 3) where the last dimension contains the min vector components.
    median_vectors : numpy.ndarray
        Array of shape (num_points, 3) where the last dimension contains the median vector components.
    max_vectors : numpy.ndarray
        Array of shape (num_points, 3) where the last dimension contains the max vector components.
    
    Returns:
    -------
    directional_variation : numpy.ndarray
        Array of shape (num_points, 3, 2) where the 3 represents the
        (pca first component, pca second component, pca mean) and the 2 represents
        the x and y components of the pca components.
    """

    num_points = vectors.shape[0]
    num_ensemble_members = vectors.shape[1]
    local_median_vector = np.zeros(3)
    local_XX = np.zeros([num_ensemble_members, 2])
    directional_variation = np.zeros([num_points, 3, 2])

    for i_p in range(num_points):
        # Compute depth threshold for this position
        # percentile=95 means keep top 95% by depth = depth >= 5th percentile
        depth_threshold = np.percentile(depths[i_p], 100.0 - percentile)
        
        r = median_vectors[i_p][0]
        theta = median_vectors[i_p][1]
        phi = median_vectors[i_p][2]
        local_median_vector[0] = r*np.sin(theta)*np.cos(phi)
        local_median_vector[1] = r*np.sin(theta)*np.sin(phi)
        local_median_vector[2] = r*np.cos(theta)
        phi = -phi
        theta = -theta

        Ry = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        Rz = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
        ii_m = 0
        for i_m in range(num_ensemble_members):
            if(depths[i_p][i_m] >= depth_threshold):
                tmp = vectors[i_p][i_m]
                vec = np.zeros(3)
                vec[0] = tmp[0]*np.sin(tmp[1])*np.cos(tmp[2])
                vec[1] = tmp[0]*np.sin(tmp[1])*np.sin(tmp[2])
                vec[2] = tmp[0]*np.cos(tmp[1])
                vec = np.dot(Ry, np.dot(Rz, vec))
                vec =  vec * (max_vectors[i_p][0]/(np.absolute(vec[2]) + 1.0e-10))
                local_XX[ii_m][0] = vec[0]
                local_XX[ii_m][1] = vec[1]
                ii_m = ii_m + 1
        local_X = local_XX[0:ii_m]
        n_local_points, n_local_features = local_X.shape

        # compute local std
        if n_local_points < 2:
            local_std = np.array([0.0, 0.0])
        else:
            local_std = np.std(local_X, axis=0)

        pca = PCA(n_components=2)

        # fit PCA if there are enough points and non-zero std
        if n_local_points > 2 and n_local_features > 1 and np.all(local_std > 1.0e-20): 
            pca.fit(local_X)
            pca_components = pca.components_
            # pca_mean = pca.mean_
            pca_variance = pca.explained_variance_
            v0_scale = pca_variance[0]
            v1_scale = pca_variance[1]
        else:
            pca_components = np.zeros((2, 2))
            v0_scale = 1.0
            v1_scale = 1.0

        if(np.absolute(v0_scale) < 1.0e-20): # ensure non-zero scale
            v0_scale = 1.0
            v1_scale = 1.0

        directional_variation[i_p][0][0] = v0_scale
        directional_variation[i_p][0][1] = np.maximum(v1_scale, 0.1*v0_scale)
        directional_variation[i_p][1][0] = pca_components[0][0]
        directional_variation[i_p][1][1] = pca_components[0][1]
        directional_variation[i_p][2][0] = pca_components[1][0]
        directional_variation[i_p][2][1] = pca_components[1][1]


    return directional_variation

