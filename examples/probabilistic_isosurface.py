import numpy as np
import pyvista as pv
from uvisbox.ProbabilisticSurfaces import probabilistic_marching_cubes_plot as pmc

def tear_drop(x, y, z):
    return 0.5*x**5 + 0.5*x**4 - y**2 - z**2

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
plotter = pmc(F, isovalue)


plotter.show()