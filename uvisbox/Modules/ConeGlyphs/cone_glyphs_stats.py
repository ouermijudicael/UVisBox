 
import numpy as np
# from sklearn.decomposition import PCA
# from uvisbox.Core.BandDepths.vector_depths import calculate_spread_2D, compute_vector_depths_2D
from uvisbox.Core.BandDepths.vector_depths import calculate_spread_3D, compute_vector_depths_3D
from uvisbox.Core.BandDepths.vector_depths import cartesian_to_polar, cartesian_to_spherical



def cone_glyphs_summary_statistics(ensemble_vectors, percentile):
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
            'spread_min_vectors': (n, 3) - min spread vectors,
            'spread_max_vectors': (n, 3) - max spread vectors,
            'glyph_types': (n,) - glyph type markers,
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
    spread_min_vectors = np.zeros((num_positions, 3))
    spread_max_vectors = np.zeros((num_positions, 3))
    median_vectors = np.zeros((num_positions, 3))
    glyph_types = np.zeros(num_positions, dtype=int)
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
        
        spread_min_vectors[i] = [min_mag_vec[0], min_theta, min_phi]
        spread_max_vectors[i] = [max_mag_vec[0], max_theta, max_phi]
        
        # Classify glyph type
        if (np.abs(spread_max_vectors[i][0] - spread_min_vectors[i][0]) > 1e-5) and \
           (np.abs(spread_max_vectors[i][1] - spread_min_vectors[i][1]) > 1e-5) and \
           (np.abs(spread_max_vectors[i][2] - spread_min_vectors[i][2]) > 1e-5):
            glyph_types[i] = 1  # Full glyph
            num_glyphs += 1
        elif spread_max_vectors[i][0] > 1e-3:
            glyph_types[i] = 2  # Arrow only
    
    # # Compute directional variations (PCA)
    # # FIXED: Pass percentile directly - getDirectionalVariations will handle threshold
    # pca_components = getDirectionalVariations(
    #     ensemble_spherical_vectors, depths, percentile, 
    #     spread_min_vectors, median_vectors, spread_max_vectors
    # )
    
    return {
        'ensemble_spherical_vectors': ensemble_spherical_vectors,
        'depths': depths,
        'median_vectors': median_vectors,
        'spread_min_vectors': spread_min_vectors,
        'spread_max_vectors': spread_max_vectors,
        'glyph_types': glyph_types,
        # 'pca_components': pca_components,
        'num_glyphs': num_glyphs
    }

