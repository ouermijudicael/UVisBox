
from .cone_glyphs_stats import compute_vector_depths_3D, calculate_spread_3D, cone_glyphs_summary_statistics
from .cone_glyphs_mesh import cone_glyphs_mesh
from .cone_glyphs_vis import visualize_cone_glyphs
from uvisbox.Core.BandDepths.vector_depths import cartesian_to_spherical
import numpy as np




def cone_glyph(positions, ensemble_vectors, point_values=None, percentile=95, scale=0.5,
                   show_edges=True, glyph_color='lightblue', ax=None):
    """
    Visualize 3D uncertainty cone glyphs.
    
    This function orchestrates the three-stage pipeline:
        1. Convert Cartesian → Spherical, compute vector depths, spreads, and PCA
        2. Build superelliptical glyph mesh (vertices, triangles)
        3. Render with pyvista


    Parameters:
    -----------
    positions : numpy.ndarray
        Array of shape (n, 3) The positions of the squid glyphs.
    ensemble_vectors : numpy.ndarray
        Array of shape (n, m, 3) The ensemble vectors for each position in Cartesian coordinates.
    point_values : numpy.ndarray, optional
        Array of shape (n,) The values associated with each position for coloring.
    percentile : float (0-100)
        Percentile of ensemble members to include based on depth ranking.
        Higher values include more vectors (larger glyphs showing more variation).
        - percentile=50: Include top 50% deepest vectors
        - percentile=95: Include top 95% deepest vectors (typical, default)
        - percentile=100: Include ALL vectors (maximum variation)
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
    # Stage 1: Compute statistics
    stats = cone_glyphs_summary_statistics(ensemble_vectors, percentile)
    
    # Stage 2: Build mesh
    mesh = cone_glyphs_mesh(positions, stats, point_values, scale, resolution=10)
    
    # Stage 3: Render
    plotter = visualize_cone_glyphs(mesh, show_edges=show_edges, glyph_color=glyph_color, ax=ax)
    
    return plotter, mesh['points'], mesh['polygons']



def cone_glyph_old(positions, ensemble_vectors, point_values=None, percentil=0.95, scale=0.5, 
                                show_edges=True, glyph_color='lightblue', ax=None):
    """
    Draws uncertainty squid glyphs for the given positions and ensemble vectors in 3D. 
    Parameters:
    -----------
    positions : numpy.ndarray
        Array of shape (n, 3) The positions of the squid glyphs.
    ensemble_vectors : numpy.ndarray
        Array of shape (n, m, 3) The ensemble vectors in spherical coordinates.
        The ensemble vectors for each position in Cartesian coordinates.
    point_values : numpy.ndarray, optional
        Array of shape (n,) The values associated with each position for coloring.
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
    # depth_threshold = 1.0 - percentil
    # directional_variations = getDirectionalVariations(ensemble_vectors, depths, depth_threshold, min_vectors, median_vectors, max_vectors)
    
    if point_values is None:
        point_values = np.zeros((num_positions,))
    # build squid glyphs
    points, polygons, scalar_values = cone_glyphs_meshing( positions, min_vectors, median_vectors, 
                                               max_vectors, point_values, glyph_markers, scale, resolution=10, num_of_glyphs=numb_of_glyphs)
    
    # plot squid glyphs
    plotter = pyvista_uncertainty_cone_glyphs_vis(points, polygons, points_values=scalar_values, ax=ax, show_edges=show_edges, glyph_color=glyph_color)
    return plotter, points, polygons
