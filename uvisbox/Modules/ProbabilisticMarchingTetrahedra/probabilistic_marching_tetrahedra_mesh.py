def probabilistic_marching_tetrahedra_mesh(summary_statistics):
    """
    Identity function that passes through summary statistics.
    
    This function exists to maintain consistency with the stats->mesh->vis pipeline
    architecture used in other modules, even though no mesh transformation is needed
    for probabilistic marching tetrahedra.
    
    Parameters
    -----------
    summary_statistics : dict
        Contains ``level_crossing_probability``, a 1D array of shape
        ``(n_tetrahedra,)`` representing isosurface probability per tetrahedron.
    
    Returns
    --------
    level_crossing_probability : np.ndarray
        1D array of probabilities extracted from the input dictionary.
    """
    return summary_statistics['level_crossing_probability']
