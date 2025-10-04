from uvisbox.Core.CoordinateSystems import cartesian_to_polar
import numpy as np
from .Stats.vector_stats import compute_vector_depths_2D
from .Vis.matplotlib_vis import uncertainty_squid_glyphs_2D_plot
from .Mesh.squid_glyphs_meshing_2D import squid_glyphs_meshing_2D

def plot(positions, ensemble_vectors, percentil1, scale=0.2, ax=None):
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

    # build squid glyphs
    glyphs_points, glyphs_polygons = squid_glyphs_meshing_2D(positions, ensemble_polar_vectors, depths, percentil1, scale)
    
    # plot squid glyphs
    ax = uncertainty_squid_glyphs_2D_plot(glyphs_points, glyphs_polygons, ax)

    return ax

