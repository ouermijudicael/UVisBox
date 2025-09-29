import numpy as np

def probabilistic_marching_squares(F, isovalue, num_samples=200):
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

