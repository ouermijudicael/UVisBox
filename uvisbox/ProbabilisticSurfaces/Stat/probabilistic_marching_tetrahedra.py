import numpy as np

def probabilistic_marching_tetrahedra(F, tetrahedra, isovalue, num_samples=200):
    """
    Perform probabilistic marching squares on a 2D scalar field with uncertainty.
    Parameters:
    -----------
        F : np.ndarray
            2D array of shape (n_points, n_ens) representing the scalar field with ensemble members.
        triangles : np.ndarray
            2D array of shape (n_tetrahedra, 4) representing the triangulation of the points.
        isovalue : float
            The isovalue for which to compute the contour.  
        num_samples : int, optional
            Number of samples to draw for estimating the probability of contour presence in each cell.
    Returns:
    --------
        prob_contour : np.ndarray
            1D array of shape (n_triangles,) with probabilities of contour presence in each triangle.
    """
    
    n_tetrahedra, n_ens = tetrahedra.shape
    crossing_porb = np.zeros(n_tetrahedra)

    for t in range(n_tetrahedra):
        vertex_indices = tetrahedra[t]  # Indices of the triangle's vertices
        F_cell = F[vertex_indices, :].reshape(-1, n_ens)  # Shape (4, n_ens)
        cov_mat = np.cov(F_cell)
        mean_vec = np.mean(F_cell, axis=1)
        samples = np.random.multivariate_normal(mean_vec, cov_mat, num_samples)  # Shape (num_samples, 3)
        count = 0
        for sample in samples:
            min_sample = np.min(sample)
            max_sample = np.max(sample)
            if min_sample <= isovalue <= max_sample:
                count += 1
        crossing_porb[t] = count / num_samples

    return crossing_porb


    