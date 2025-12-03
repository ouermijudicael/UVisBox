import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial


def _process_square_cell(cell_coords, ensemble_images, isovalue, num_samples):
    """
    Process a single square cell for Monte Carlo crossing probability calculation.
    
    Parameters:
    -----------
    cell_coords : tuple
        Coordinates (i, j) of the cell to process
    ensemble_images : np.ndarray
        3D array of shape (n_y, n_x, n_ensemble)
    isovalue : float
        The isovalue for which to compute the isocontour
    num_samples : int
        Number of Monte Carlo samples
        
    Returns:
    --------
    tuple : (cell_coords, probability)
        Cell coordinates and computed crossing probability
    """
    i, j = cell_coords
    cell_data = ensemble_images[i:i + 2, j:j + 2, :].reshape(-1, ensemble_images.shape[2])  # Shape (4, n_ensemble)
    covariance_matrix = np.cov(cell_data)
    mean_vector = np.mean(cell_data, axis=1)
    
    # Handle potential singular covariance matrices
    try:
        samples = np.random.multivariate_normal(mean_vector, covariance_matrix, num_samples)  # Shape (num_samples, 4)
    except np.linalg.LinAlgError:
        # If covariance is singular, add small regularization
        regularized_cov = covariance_matrix + np.eye(covariance_matrix.shape[0]) * 1e-10
        samples = np.random.multivariate_normal(mean_vector, regularized_cov, num_samples)
    
    # Vectorized crossing check
    min_values = samples.min(axis=1)
    max_values = samples.max(axis=1)
    crossing = (min_values <= isovalue) & (isovalue <= max_values)
    probability = crossing.sum() / num_samples
    
    return (cell_coords, probability)


def crossing_probability_squares_monte_carlo(ensemble_images, isovalue, num_samples=200, n_jobs=None):
    """
    Perform probabilistic marching squares on a 2D scalar field with uncertainty. This function calculates
    the probability of the isocontour passing through each cell in the grid based on an ensemble of scalar fields.
    The method is based on the paper: K. Pothkow and H. -C. Hege, "Positional Uncertainty of Isocontours: 
    Condition Analysis and Probabilistic Measures," in IEEE Transactions on Visualization and Computer Graphics, 
    vol. 17, no. 10, pp. 1393-1406, Oct. 2011, doi: 10.1109/TVCG.2010.247
    
    This parallelized version processes cells concurrently using multiprocessing for improved performance.

    Parameters:
    -----------
        ensemble_images : np.ndarray
            3D array of shape [y, x, n_ensemble] representing the scalar field with ensemble members.
        isovalue : float
            The isovalue for which to compute the contour.
        num_samples : int, optional
            Number of Monte Carlo samples to draw for estimating the probability of contour presence in each cell.
        n_jobs : int, optional
            Number of parallel processes to use. If None, uses all available CPU cores.
            Set to 1 for serial execution (useful for debugging).
            
    Returns:
    --------
        probability_contour : np.ndarray
            2D array of shape (y-1, x-1) with probabilities of contour presence in each cell.
    """
    
    n, m, n_ensemble = ensemble_images.shape
    probability_contour = np.zeros((n - 1, m - 1))
    
    # Determine number of processes
    if n_jobs is None:
        n_jobs = cpu_count()
    elif n_jobs == 1:
        # Serial execution for debugging or small datasets
        for i in range(n - 1):
            for j in range(m - 1):
                cell_data = ensemble_images[i:i + 2, j:j + 2, :].reshape(-1, n_ensemble)  # Shape (4, n_ensemble)
                covariance_matrix = np.cov(cell_data)
                mean_vector = np.mean(cell_data, axis=1)
                
                # Handle potential singular covariance matrices
                try:
                    samples = np.random.multivariate_normal(mean_vector, covariance_matrix, num_samples)  # Shape (num_samples, 4)
                except np.linalg.LinAlgError:
                    regularized_cov = covariance_matrix + np.eye(covariance_matrix.shape[0]) * 1e-10
                    samples = np.random.multivariate_normal(mean_vector, regularized_cov, num_samples)
                
                # Vectorized crossing check
                min_values = samples.min(axis=1)
                max_values = samples.max(axis=1)
                crossing = (min_values <= isovalue) & (isovalue <= max_values)
                probability_contour[i, j] = crossing.sum() / num_samples
        
        return probability_contour
    
    # Generate all cell coordinates
    cell_coords = [(i, j) for i in range(n - 1) for j in range(m - 1)]
    
    # Create partial function with fixed parameters
    process_func = partial(_process_square_cell, 
                          ensemble_images=ensemble_images, 
                          isovalue=isovalue, 
                          num_samples=num_samples)
    
    # Parallel processing
    with Pool(processes=n_jobs) as pool:
        results = pool.map(process_func, cell_coords)
    
    # Fill the result array
    for (i, j), probability in results:
        probability_contour[i, j] = probability

    return probability_contour


def _process_triangle_cell(triangle_idx, ensemble_data, triangle_mesh, isovalue, num_samples):
    """
    Process a single triangle for Monte Carlo crossing probability calculation.
    
    Parameters:
    -----------
    triangle_idx : int
        Index of the triangle to process
    ensemble_data : np.ndarray
        2D array of shape (n_points, n_ensemble)
    triangle_mesh : np.ndarray
        2D array of shape (n_triangles, 3) representing the triangulation
    isovalue : float
        The isovalue for which to compute the isocontour
    num_samples : int
        Number of Monte Carlo samples
        
    Returns:
    --------
    tuple : (triangle_idx, probability)
        Triangle index and computed crossing probability
    """
    vertex_indices = triangle_mesh[triangle_idx]  # Indices of the triangle's vertices
    cell_data = ensemble_data[vertex_indices, :]  # Shape (3, n_ensemble)
    covariance_matrix = np.cov(cell_data)
    mean_vector = np.mean(cell_data, axis=1)
    
    # Handle potential singular covariance matrices
    try:
        samples = np.random.multivariate_normal(mean_vector, covariance_matrix, num_samples)  # Shape (num_samples, 3)
    except np.linalg.LinAlgError:
        # If covariance is singular, add small regularization
        regularized_cov = covariance_matrix + np.eye(covariance_matrix.shape[0]) * 1e-10
        samples = np.random.multivariate_normal(mean_vector, regularized_cov, num_samples)
    
    # Vectorized crossing check
    min_values = samples.min(axis=1)
    max_values = samples.max(axis=1)
    crossing = (min_values <= isovalue) & (isovalue <= max_values)
    probability = crossing.sum() / num_samples
    
    return (triangle_idx, probability)


def crossing_probability_triangles_monte_carlo(ensemble_data, triangle_mesh, isovalue, num_samples=200, n_jobs=None):
    """
    Perform probabilistic marching triangles on a 2D scalar field with uncertainty. 
    This function calculates the probability of the isocontour passing through each 
    triangle in the triangulated mesh based on an ensemble of scalar fields.
    
    The method is based on the paper: Pöthkow, K., Petz, C. and Hege, H.C., 2013. 
    Approximate level-crossing probabilities for interactive visualization of uncertain 
    isocontours. International Journal for Uncertainty Quantification, 3(2).
    doi: 10.1615/Int.J.UncertaintyQuantification.2012003958
    
    This parallelized version processes triangles concurrently using multiprocessing for improved performance.

    Parameters:
    -----------
        ensemble_data : np.ndarray
            2D array of shape (n_points, n_ensemble) representing the scalar field with ensemble members.
        triangle_mesh : np.ndarray
            2D array of shape (n_triangles, 3) representing the triangulation connectivity.
        isovalue : float
            The isovalue for which to compute the contour.  
        num_samples : int, optional
            Number of Monte Carlo samples to draw for estimating the probability of contour presence in each triangle.
        n_jobs : int, optional
            Number of parallel processes to use. If None, uses all available CPU cores.
            Set to 1 for serial execution (useful for debugging).
    
    Returns:
    --------
        probability_contour : np.ndarray
            1D array of shape (n_triangles,) with probabilities of contour presence in each triangle.
    """
    
    n_triangles = triangle_mesh.shape[0]
    probability_contour = np.zeros(n_triangles)
    
    # Determine number of processes
    if n_jobs is None:
        n_jobs = cpu_count()
    elif n_jobs == 1:
        # Serial execution for debugging or small datasets
        for t in range(n_triangles):
            vertex_indices = triangle_mesh[t]  # Indices of the triangle's vertices
            cell_data = ensemble_data[vertex_indices, :]  # Shape (3, n_ensemble)
            covariance_matrix = np.cov(cell_data)
            mean_vector = np.mean(cell_data, axis=1)
            
            # Handle potential singular covariance matrices
            try:
                samples = np.random.multivariate_normal(mean_vector, covariance_matrix, num_samples)  # Shape (num_samples, 3)
            except np.linalg.LinAlgError:
                regularized_cov = covariance_matrix + np.eye(covariance_matrix.shape[0]) * 1e-10
                samples = np.random.multivariate_normal(mean_vector, regularized_cov, num_samples)
            
            # Vectorized crossing check
            min_values = samples.min(axis=1)
            max_values = samples.max(axis=1)
            crossing = (min_values <= isovalue) & (isovalue <= max_values)
            probability_contour[t] = crossing.sum() / num_samples
        
        return probability_contour
    
    # Generate all triangle indices
    triangle_indices = list(range(n_triangles))
    
    # Create partial function with fixed parameters
    process_func = partial(_process_triangle_cell, 
                          ensemble_data=ensemble_data,
                          triangle_mesh=triangle_mesh,
                          isovalue=isovalue, 
                          num_samples=num_samples)
    
    # Parallel processing
    with Pool(processes=n_jobs) as pool:
        results = pool.map(process_func, triangle_indices)
    
    # Fill the result array
    for triangle_idx, probability in results:
        probability_contour[triangle_idx] = probability

    return probability_contour


    

def _process_cube_cell(cell_coords, ensemble_images, isovalue, num_samples):
    """
    Process a single cube cell for Monte Carlo crossing probability calculation.
    
    Parameters:
    -----------
    cell_coords : tuple
        Coordinates (k, j, i) of the cell to process
    ensemble_images : np.ndarray
        4D array of shape (n_z, n_y, n_x, n_ensemble)
    isovalue : float
        The isovalue for which to compute the isosurface
    num_samples : int
        Number of Monte Carlo samples
        
    Returns:
    --------
    tuple : (cell_coords, probability)
        Cell coordinates and computed crossing probability
    """
    k, j, i = cell_coords
    cell_data = ensemble_images[k:k+2, j:j+2, i:i+2, :].reshape(-1, ensemble_images.shape[3])  # Shape (8, n_ensemble)
    covariance_matrix = np.cov(cell_data)
    mean_vector = np.mean(cell_data, axis=1)
    
    # Handle potential singular covariance matrices
    try:
        samples = np.random.multivariate_normal(mean_vector, covariance_matrix, num_samples)  # Shape (num_samples, 8)
    except np.linalg.LinAlgError:
        # If covariance is singular, add small regularization
        regularized_cov = covariance_matrix + np.eye(covariance_matrix.shape[0]) * 1e-10
        samples = np.random.multivariate_normal(mean_vector, regularized_cov, num_samples)
    
    # Vectorized crossing check
    min_values = samples.min(axis=1)
    max_values = samples.max(axis=1)
    crossing = (min_values <= isovalue) & (isovalue <= max_values)
    probability = crossing.sum() / num_samples
    
    return (cell_coords, probability)


def crossing_probability_cubes_monte_carlo(ensemble_images, isovalue, num_samples=200, n_jobs=None):
    """
    Perform probabilistic marching cubes on a 3D scalar field with uncertainty. This function calculates
    the probability of the isosurface passing through each cell in the grid based on an ensemble of scalar fields.
    The method is based on the paper: Pöthkow, K., Weber, B. and Hege, H.-C. (2011), Probabilistic Marching Cubes. Computer 
    Graphics Forum, 30: 931-940. https://doi.org/10.1111/j.1467-8659.2011.01942.x
    
    This parallelized version processes cells concurrently using multiprocessing for improved performance.
    
    Parameters:
    -----------
        ensemble_images : np.ndarray
            4D array of shape (n_z, n_y, n_x, n_ensemble) representing the scalar field with ensemble members.
        isovalue : float
            The isovalue for which to compute the isosurface.
        num_samples : int, optional
            Number of Monte Carlo samples to draw for estimating the probability of contour presence in each cell.
        n_jobs : int, optional
            Number of parallel processes to use. If None, uses all available CPU cores.
            Set to 1 for serial execution (useful for debugging).
    
    Returns:
    --------
        probability_contour : np.ndarray
            3D array of shape (n_z-1, n_y-1, n_x-1) with probabilities of contour presence in each cell.
    """
    n_z, n_y, n_x, n_ensemble = ensemble_images.shape
    probability_contour = np.zeros((n_z - 1, n_y - 1, n_x - 1))
    
    # Determine number of processes
    if n_jobs is None:
        n_jobs = cpu_count()
    elif n_jobs == 1:
        # Serial execution for debugging or small datasets
        for k in range(n_z - 1):
            for j in range(n_y - 1):
                for i in range(n_x - 1):
                    cell_data = ensemble_images[k:k+2, j:j+2, i:i+2, :].reshape(-1, n_ensemble)  # Shape (8, n_ensemble)
                    covariance_matrix = np.cov(cell_data)
                    mean_vector = np.mean(cell_data, axis=1)
                    
                    # Handle potential singular covariance matrices
                    try:
                        samples = np.random.multivariate_normal(mean_vector, covariance_matrix, num_samples)  # Shape (num_samples, 8)
                    except np.linalg.LinAlgError:
                        regularized_cov = covariance_matrix + np.eye(covariance_matrix.shape[0]) * 1e-10
                        samples = np.random.multivariate_normal(mean_vector, regularized_cov, num_samples)
                    
                    # Vectorized crossing check
                    min_values = samples.min(axis=1)
                    max_values = samples.max(axis=1)
                    crossing = (min_values <= isovalue) & (isovalue <= max_values)
                    probability_contour[k, j, i] = crossing.sum() / num_samples
        
        return probability_contour
    
    # Generate all cell coordinates
    cell_coords = [(k, j, i) for k in range(n_z - 1) 
                   for j in range(n_y - 1) 
                   for i in range(n_x - 1)]
    
    # Create partial function with fixed parameters
    process_func = partial(_process_cube_cell, 
                          ensemble_images=ensemble_images, 
                          isovalue=isovalue, 
                          num_samples=num_samples)
    
    # Parallel processing
    with Pool(processes=n_jobs) as pool:
        results = pool.map(process_func, cell_coords)
    
    # Fill the result array
    for (k, j, i), probability in results:
        probability_contour[k, j, i] = probability

    return probability_contour


def _process_tetrahedron_cell(tet_idx, ensemble_data, tetrahedral_mesh, isovalue, num_samples):
    """
    Process a single tetrahedron for Monte Carlo crossing probability calculation.
    
    Parameters:
    -----------
    tet_idx : int
        Index of the tetrahedron to process
    ensemble_data : np.ndarray
        2D array of shape (num_points, n_ensemble)
    tetrahedral_mesh : np.ndarray
        2D array of shape (num_tetrahedra, 4) representing the tetrahedral mesh
    isovalue : float
        The isovalue for which to compute the isosurface
    num_samples : int
        Number of Monte Carlo samples
        
    Returns:
    --------
    tuple : (tet_idx, probability)
        Tetrahedron index and computed crossing probability
    """
    vertex_indices = tetrahedral_mesh[tet_idx]
    cell_data = ensemble_data[vertex_indices, :]  # Shape (4, n_ensemble)
    covariance_matrix = np.cov(cell_data)
    mean_vector = np.mean(cell_data, axis=1)
    
    # Handle potential singular covariance matrices
    try:
        samples = np.random.multivariate_normal(mean_vector, covariance_matrix, num_samples)  # Shape (num_samples, 4)
    except np.linalg.LinAlgError:
        # If covariance is singular, add small regularization
        regularized_cov = covariance_matrix + np.eye(covariance_matrix.shape[0]) * 1e-10
        samples = np.random.multivariate_normal(mean_vector, regularized_cov, num_samples)
    
    # Vectorized crossing check
    min_values = samples.min(axis=1)
    max_values = samples.max(axis=1)
    crossing = (min_values <= isovalue) & (isovalue <= max_values)
    probability = crossing.sum() / num_samples
    
    return (tet_idx, probability)


def crossing_probability_tetrahedra_monte_carlo(ensemble_data, tetrahedral_mesh, isovalue, num_samples=200, n_jobs=None):
    """
    Perform probabilistic marching tetrahedra on an ensemble of scalar fields defined on a tetrahedral mesh.
    The function calculates the probability of the isosurface passing through each tetrahedron based on Monte Carlo sampling.
    
    This parallelized version processes tetrahedra concurrently using multiprocessing for improved performance.
    
    Parameters:
    -----------
        ensemble_data : np.ndarray
            2D array of shape (num_points, n_ensemble) where each column is a realization and each row corresponds 
            to a vertex in the tetrahedral mesh.
        tetrahedral_mesh : np.ndarray
            2D array of shape (num_tetrahedra, 4) representing the tetrahedra. Each row contains the indices 
            of the four vertices forming a tetrahedron.
        isovalue : float
            The isovalue for which to compute the isosurface.
        num_samples : int, optional
            Number of Monte Carlo samples to draw for estimating the probability of contour presence in each tetrahedron.
        n_jobs : int, optional
            Number of parallel processes to use. If None, uses all available CPU cores.
            Set to 1 for serial execution (useful for debugging).
    
    Returns:
    --------
        probability_contour : np.ndarray
            1D array of shape (num_tetrahedra,) with probabilities of contour presence in each tetrahedron.
    """
    num_tetrahedra = tetrahedral_mesh.shape[0]
    probability_contour = np.zeros(num_tetrahedra)
    
    # Determine number of processes
    if n_jobs is None:
        n_jobs = cpu_count()
    elif n_jobs == 1:
        # Serial execution for debugging or small datasets
        for tet_idx in range(num_tetrahedra):
            vertex_indices = tetrahedral_mesh[tet_idx]
            cell_data = ensemble_data[vertex_indices, :]  # Shape (4, n_ensemble)
            covariance_matrix = np.cov(cell_data)
            mean_vector = np.mean(cell_data, axis=1)
            
            # Handle potential singular covariance matrices
            try:
                samples = np.random.multivariate_normal(mean_vector, covariance_matrix, num_samples)  # Shape (num_samples, 4)
            except np.linalg.LinAlgError:
                regularized_cov = covariance_matrix + np.eye(covariance_matrix.shape[0]) * 1e-10
                samples = np.random.multivariate_normal(mean_vector, regularized_cov, num_samples)
            
            # Vectorized crossing check
            min_values = samples.min(axis=1)
            max_values = samples.max(axis=1)
            crossing = (min_values <= isovalue) & (isovalue <= max_values)
            probability_contour[tet_idx] = crossing.sum() / num_samples
        
        return probability_contour
    
    # Generate all tetrahedron indices
    tet_indices = list(range(num_tetrahedra))
    
    # Create partial function with fixed parameters
    process_func = partial(_process_tetrahedron_cell, 
                          ensemble_data=ensemble_data,
                          tetrahedral_mesh=tetrahedral_mesh,
                          isovalue=isovalue, 
                          num_samples=num_samples)
    
    # Parallel processing
    with Pool(processes=n_jobs) as pool:
        results = pool.map(process_func, tet_indices)
    
    # Fill the result array
    for tet_idx, probability in results:
        probability_contour[tet_idx] = probability

    return probability_contour
