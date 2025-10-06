import numpy as np


def crossing_prob_squares_mc(F, isovalue, num_samples=200):
    """
    Perform probabilistic marching squares on a 2D scalar field with uncertainty. This function calculates
    the probability of the isocontour passing through each cell in the grid based on an ensemble of scalar fields.
    The method is based on the paper: K. Pothkow and H. -C. Hege, "Positional Uncertainty of Isocontours: 
    Condition Analysis and Probabilistic Measures," in IEEE Transactions on Visualization and Computer Graphics, 
    vol. 17, no. 10, pp. 1393-1406, Oct. 2011, doi: 10.1109/TVCG.2010.247

    Parameters:
    -----------
        F : np.ndarray
            3D array of shape (n, m, n_ens) representing the scalar field with ensemble members.
        isovalue : float
            The isovalue for which to compute the contour.
        num_samples : int, optional
            Number of samples to draw for estimating the probability of contour presence in each cell.
            
    Returns:
    --------
        prob_contour : np.ndarray
            2D array of shape (n-1, m-1) with probabilities of contour presence in each cell.
    """
    
    n, m, n_ens = F.shape
    prob_contour = np.zeros((n - 1, m - 1))

    for i in range(n - 1):
        for j in range(m - 1):
            F_cell = F[i:i + 2, j:j + 2, :].reshape(-1, n_ens)  # Shape (4, n_ens)
            cov_mat = np.cov(F_cell)
            mean_vec = np.mean(F_cell, axis=1)
            samples = np.random.multivariate_normal(mean_vec, cov_mat, num_samples)  # Shape (num_samples, 4)
            count = 0
            for sample in samples:
                min_sample = np.min(sample)
                max_sample = np.max(sample)
                if min_sample <= isovalue <= max_sample:
                    count += 1
            prob_contour[i, j] = count / num_samples

    return prob_contour


def crossing_prob_triangles_mc(F, triangles, isovalue, num_samples=200):
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


    

def crossing_prob_cubes_mc(F, isovalue, num_samples=200):
    """
    Perform probabilistic marching squares on a 2D scalar field with uncertainty. This function calculates
    the probability of the isocontour passing through each cell in the grid based on an ensemble of scalar fields.
    The method is based on the paper: Pöthkow, K., Weber, B. and Hege, H.-C. (2011), Probabilistic Marching Cubes. Computer 
    Graphics Forum, 30: 931-940. https://doi.org/10.1111/j.1467-8659.2011.01942.x
    
    Parameters:
    -----------
        F : np.ndarray
            4D array of shape (n_x, n_y, n_z, n_ens) representing the scalar field with ensemble members.
        isovalue : float
            The isovalue for which to compute the isosurface.
        num_samples : int, optional
            Number of samples to draw for estimating the probability of contour presence in each cell.
    
    Returns:
    --------
        prob_contour : np.ndarray
            3D array of shape (n_x-1, n_y-1, n_z-1) with probabilities of contour presence in each cell.
    """
    n_x, n_y, n_z, n_ens = F.shape
    cross_prob = np.zeros((n_x - 1, n_y - 1, n_z-1))

    for i in range(n_x - 1):
        for j in range(n_y - 1):
            for k in range(n_z-1):
                F_cell = F[i:i+2, j:j+2, k:k+2, :].reshape(-1, n_ens)  # Shape (6, n_ens)
                cov_mat = np.cov(F_cell)
                mean_vec = np.mean(F_cell, axis=1)
                samples = np.random.multivariate_normal(mean_vec, cov_mat, num_samples)  # Shape (num_samples, 6)
                count = 0
                for sample in samples:
                    min_sample = np.min(sample)
                    max_sample = np.max(sample)
                    if min_sample <= isovalue <= max_sample:
                        count += 1
                cross_prob[i, j, k] = count / num_samples

    return cross_prob


def crossing_prob_tetrahedra_mc(F, tetrahedra, isovalue, num_samples=200):
    """
    Perform probabilistic marching squares on a 2D scalar field with uncertainty. This function calculates
    the probability of the isocontour passing through each triangle in the triangulated mesh based on an ensemble of scalar fields.

    Parameters:
    -----------
        F : np.ndarray
            2D array of shape (n_points, n_ens) representing the scalar field with ensemble members.
    tetrahedra : np.ndarray
            2D array of shape (n_tetrahedra, 4) representing the triangulation of the points.
        isovalue : float
            The isovalue for which to compute the contour.  
        num_samples : int, optional
            Number of samples to draw for estimating the probability of contour presence in each cell.

    Returns:
    --------    
        prob_contour : np.ndarray
            1D array of shape (n_tetrahedra,) with probabilities of contour presence in each tetrahedron.
    """
    
    n_tetrahedra, n_ens = tetrahedra.shape
    crossing_prob = np.zeros(n_tetrahedra)

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
        crossing_prob[t] = count / num_samples

    return crossing_prob
