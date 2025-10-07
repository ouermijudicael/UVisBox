
from .squid_glyphs_stats import compute_vector_depths_3D, getDirectionalVariations, calculate_spread_3D
from .squid_glyphs_mesh import squid_glyphs_meshing_3D
from .squid_glyphs_vis import pyvista_uncertainty_squid_glyphs_3D_vis
from uvisbox.Core.BandDepths.vector_depths import cartesian_to_spherical
from uvisbox.Core.BandDepths.vector_depths import cartesian_to_polar
import numpy as np
from .squid_glyphs_stats import compute_vector_depths_2D
from .squid_glyphs_vis import matplotlib_uncertainty_squid_glyphs_2D_vis
from .squid_glyphs_mesh import squid_glyphs_meshing_2D

def squid_glyph_3D(positions, ensemble_vectors, percentil, scale=0.5, 
                                show_edges=True, glyph_color='lightblue', ax=None):
    """
    Draws uncertainty squid glyphs for the given positions and ensemble vectors in 3D. this implementation is based on
    T. A. J. Ouermi, J. Li, Z. Morrow, B. Van Bloemen Waanders and C. R. Johnson, "Glyph-Based Uncertainty Visualization
    and Analysis of Time-Varying Vector Fields," 2024 IEEE Workshop on Uncertainty Visualization: Applications,
    Techniques, Software, and Decision Frameworks, St Pete Beach, FL, USA, 2024, pp. 73-77, doi: 10.1109/UncertaintyVisualization63963.2024.00014.

    Parameters:
    -----------
    positions : numpy.ndarray
        Array of shape (n, 3) The positions of the squid glyphs.
    ensemble_vectors : numpy.ndarray
        Array of shape (n, m, 3) The ensemble vectors in spherical coordinates.
        The ensemble vectors for each position in Cartesian coordinates.
    percentil : float
        The first percentile for depth filtering.
    scale : float, optional
        The scale factor for the glyphs. Default is 0.5.
    show_edges : bool, optional
        Whether to show edges of the glyphs. Default is True.
    glyph_color : str, optional
        The color of the glyphs. Default is 'lightblue'.
    ax : pyvista.Plotter, optional
        The pyvista plotter to use. If None, a new plotter will be created.

    Returns:
    --------
    plotter : pyvista.Plotter
        The pyvista plotter with the drawn squid glyphs.
    points : numpy.ndarray
        The points of the squid glyphs.
    polygons : numpy.ndarray
        The polygon connectivity for the glyphs.

    """

    num_positions, num_ens_members =ensemble_vectors.shape[0], ensemble_vectors.shape[1]
    
    # Convert ensemble_vectors to spherical coordinates
    ensemble_spherical_vectors = np.zeros_like(ensemble_vectors)
    for i in range(num_positions):
        ensemble_spherical_vectors[i] = cartesian_to_spherical(ensemble_vectors[i])

    # Ccalculate vector depths in spherical coordinates
    depths = np.zeros((num_positions, num_ens_members))
    for i in range(num_positions):
        depths[i] = compute_vector_depths_3D(ensemble_spherical_vectors[i])

    min_vectors = np.zeros((num_positions, 3))
    max_vectors = np.zeros((num_positions, 3))
    median_vectors = np.zeros((num_positions, 3))
    glyph_markers = np.zeros((num_positions,), dtype=int)
    numb_of_glyphs = 0
    num_arrows = 0
    for i_pos in range(num_positions):
        # find indices where depth > 1.0-percentil
        median_idx, min_mag_idx, max_mag_idx, min_theta_idx, max_theta_idx, min_phi_idx, max_phi_idx = calculate_spread_3D(ensemble_spherical_vectors[i_pos], depths[i_pos], percentil)
        median_vectors[i_pos] = ensemble_spherical_vectors[i_pos][median_idx] if median_idx is not None else np.array([0,0,0])
        min_mag_temp = ensemble_spherical_vectors[i_pos][min_mag_idx] if min_mag_idx is not None else np.array([0,0,0])
        max_mag_temp = ensemble_spherical_vectors[i_pos][max_mag_idx] if max_mag_idx is not None else np.array([0,0,0])
        min_phi = ensemble_spherical_vectors[i_pos][min_phi_idx][2] if min_phi_idx is not None else 0
        max_phi = ensemble_spherical_vectors[i_pos][max_phi_idx][2] if max_phi_idx is not None else 0
        min_theta = ensemble_spherical_vectors[i_pos][min_theta_idx][1] if min_theta_idx is not None else 0
        max_theta = ensemble_spherical_vectors[i_pos][max_theta_idx][1] if max_theta_idx is not None else 0
        min_vectors[i_pos] = np.array([min_mag_temp[0], min_theta, min_phi])
        max_vectors[i_pos] = np.array([max_mag_temp[0], max_theta, max_phi])
        if (np.absolute(max_vectors[i_pos][0]-min_vectors[i_pos][0]) > 1e-5) and \
           (np.absolute(max_vectors[i_pos][1]-min_vectors[i_pos][1]) > 1e-5) and \
           (np.absolute(max_vectors[i_pos][2]-min_vectors[i_pos][2]) > 1e-5):
            glyph_markers[i_pos] = 1
            numb_of_glyphs += 1
        elif(max_vectors[i_pos][0] > 1e-3):
            glyph_markers[i_pos] = 2
            num_arrows += 1


    # compute directional variations
    depth_threshold = 1.0 - percentil
    directional_variations = getDirectionalVariations(ensemble_vectors, depths, depth_threshold, min_vectors, median_vectors, max_vectors)
    
    # build squid glyphs
    points, polygons = squid_glyphs_meshing_3D(directional_variations, positions, ensemble_spherical_vectors, min_vectors, median_vectors, max_vectors, glyph_markers, scale, resolution=10, num_of_glyphs=numb_of_glyphs)
    
    # plot squid glyphs
    plotter = pyvista_uncertainty_squid_glyphs_3D_vis(points, polygons, ax, show_edges, glyph_color)

    return plotter, points, polygons



def squid_glyph_2D(positions, ensemble_vectors, percentil1, scale=0.2, ax=None):
    """
    Draws uncertainty squid glyphs for the given positions and ensemble vectors in 2D. This a 2D version
    of the 3D uncertainty squid glyphs in T. A. J. Ouermi, J. Li, Z. Morrow, B. Van Bloemen Waanders and 
    C. R. Johnson, "Glyph-Based Uncertainty Visualization and Analysis of Time-Varying Vector Fields," 
    2024 IEEE Workshop on Uncertainty Visualization: Applications, Techniques, Software, and Decision Frameworks, 
    St Pete Beach, FL, USA, 2024, pp. 73-77, doi: 10.1109/UncertaintyVisualization63963.2024.00014.

    Parameters:
    -----------
    positions : numpy.ndarray
        Array of shape (n, 2) representing the positions of the squid glyphs.
    ensemble_vectors : numpy.ndarray
        Array of shape (n, m, 2) representing the ensemble vectors for each position.
    percentil1 : float
        The first percentile for depth filtering.
    scale : float
        The scale factor for the glyphs.
    ax : matplotlib axis
        The axis to draw on. If None, a new figure and axis will be created.

    Returns:
    --------
    ax : matplotlib axis
        The axis with the drawn squid glyphs.

    """
    num_positions, num_ens_members =ensemble_vectors.shape[0], ensemble_vectors.shape[1]
    
    # Convert ensemble_vectors to spherical coordinates
    ensemble_polar_vectors = np.zeros_like(ensemble_vectors)
    for i in range(num_positions):
        ensemble_polar_vectors[i] = cartesian_to_polar(ensemble_vectors[i])

    # Ccalculate vector depths in spherical coordinates
    depths = np.zeros((num_positions, num_ens_members))
    for i in range(num_positions):
        depths[i] = compute_vector_depths_2D(ensemble_polar_vectors[i])
        if i == 0:
            print(f"max depth at position {i}: {depths[i].max()} with index {np.argmax(depths[i])}")
            print(f" and vector {ensemble_polar_vectors[i][np.argmax(depths[i])]}")

    # build squid glyphs
    glyphs_points, glyphs_polygons = squid_glyphs_meshing_2D(positions, ensemble_polar_vectors, depths, percentil1, scale)
    
    # plot squid glyphs
    ax = matplotlib_uncertainty_squid_glyphs_2D_vis(glyphs_points, glyphs_polygons, ax)

    return ax

