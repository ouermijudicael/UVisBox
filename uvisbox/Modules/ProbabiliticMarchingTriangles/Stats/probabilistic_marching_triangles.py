import numpy as np

def probabilistic_marching_triangles(F, triangles, isovalue, num_samples=200):
    """
    Perform probabilistic marching triangles on a 2D scalar field with uncertainty. 
    This function calculatesthe probability of the isocontour passing through each 
    triangle in the triangulated mesh based on an ensemble of scalar fields.
    
    The method is based on the paper: Pöthkow, K., Petz, C. and Hege, H.C., 2013. 
    Approximate level-crossing probabilities for interactive visualization of uncertain 
    isocontours. International Journal for Uncertainty Quantification, 3(2).
    doi: 10.1615/Int.J.UncertaintyQuantification.2012003958

    Parameters:
    -----------
        F : np.ndarray
            2D array of shape (n_points, n_ens) representing the scalar field with ensemble members.
        triangles : np.ndarray
            2D array of shape (n_triangles, 3) representing the triangulation of the points.
        isovalue : float
            The isovalue for which to compute the contour.  
        num_samples : int, optional
            Number of samples to draw for estimating the probability of contour presence in each cell.
    
    Returns:
    --------
        prob_contour : np.ndarray
            1D array of shape (n_triangles,) with probabilities of contour presence in each triangle.
    """
    
    n_triangles = triangles.shape[0]
    prob_contour = np.zeros(n_triangles)

    for t in range(n_triangles):
        vertex_indices = triangles[t]  # Indices of the triangle's vertices
        F_cell = F[vertex_indices, :]  # Shape (3, n_ens)
        cov_mat = np.cov(F_cell)
        mean_vec = np.mean(F_cell, axis=1)
        samples = np.random.multivariate_normal(mean_vec, cov_mat, num_samples)  # Shape (num_samples, 3)
        count = 0
        for sample in samples:
            min_sample = np.min(sample)
            max_sample = np.max(sample)
            if min_sample <= isovalue <= max_sample:
                count += 1
        prob_contour[t] = count / num_samples

    return prob_contour


    