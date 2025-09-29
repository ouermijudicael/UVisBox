import numpy as np
import pyvista as pv

def tear_drop(x, y, z):
    return 0.5*x**5 + 0.5*x**4 - y**2 - z**2

def probabilistic_marching_cubes(F, isovalue, num_samples=200):
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

# Example usage
if __name__ == "__main__":
    # Generate synthetic 4D data (n_x, n_y, n_z, n_ens)
    n_x, n_y, n_z, n_ens = 32, 32, 32, 10
    x = np.linspace(-1, 1, n_x)
    y = np.linspace(-1, 1, n_y)
    z = np.linspace(-1, 1, n_z)
    X, Y, Z = np.meshgrid(x, y, z)
    # Create an ensemble of scalar fields with some noise
    noise_less_F = tear_drop(X, Y, Z)
    origin = (0, 0, 0)
    spacing = (1, 1, 1)
    grid_dimensions = (n_x, n_y, n_z)
    grid = pv.ImageData(dimensions=grid_dimensions, origin=origin, spacing=spacing)
    # Add some data to the cell data (e.g., a 4D NumPy array)
    grid.point_data["values"] = noise_less_F.flatten(order='F')

    isovalue = -0.001
    iso_surface = grid.contour([isovalue], scalars="values")

    plotter = pv.Plotter()
    plotter.add_mesh(iso_surface, color='lightblue', opacity=0.5)
    plotter.show()

    F = np.zeros((n_x, n_y, n_z, n_ens))
    for e in range(n_ens):
        noise = np.random.normal(0, 0.01, (n_x, n_y, n_z))
        F[:, :, :, e] = noise_less_F + noise
    
    # Compute probabilistic marching cubes
    prob_contour = probabilistic_marching_cubes(F, isovalue)

    # Create a PyVista grid for visualization
    grid_dimensions = np.array(prob_contour.shape) 
    grid_spacing = (1, 1, 1)
    gird_origin = (0, 0, 0)
    grid = pv.ImageData(dimensions=grid_dimensions, spacing=grid_spacing, origin=gird_origin)
    grid.point_data["probability"] = prob_contour.flatten(order='F')

    # Render the volume
    plotter = pv.Plotter()
    plotter.add_volume(grid, scalars="probability", opacity="linear", cmap="viridis")
    plotter.show()