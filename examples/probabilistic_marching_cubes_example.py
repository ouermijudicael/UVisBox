"""
This example demonstrates how to compute and visualize probabilistic isosurfaces
using the marching cubes algorithm on a synthetic 4D dataset.
It compares a deterministic isosurface with a probabilistic isosurface derived from an ensemble of scalar fields.

Import necessary libraries

.. code-block:: python

    import numpy as np
    import pyvista as pv
    from uvisbox.ProbabilisticSurfaces import probabilistic_marching_cubes_plot as pmc

Generate a regular grid over a 3D domain [-1, 1] and use the tear drop function
to create an ensemble of scalar fields with some noise

.. code-block:: python

    # tear drop function
    def tear_drop(x, y, z):
        return 0.5*x**5 + 0.5*x**4 - y**2 - z**2

    # Generate synthetic 4D data (n_x, n_y, n_z, n_ens)
    n_x, n_y, n_z, n_ens = 16, 16, 16, 10
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

Set isovalue and compute deterministic isosurface

.. code-block:: python

    isovalue = -0.001
    iso_surface = grid.contour([isovalue], scalars="values")
    # Set up the plotter with two subplots
    plotter = pv.Plotter(shape=(1, 2), off_screen=True)
    # Plot deterministic isosurface
    plotter.subplot(0, 0)
    plotter.add_text("Deterministic Isosurface", font_size=12)
    plotter.add_mesh(iso_surface, color='lightblue', opacity=0.5)

Generate ensemble data with noise and calculate probabilistic isosurface 
using probabilistic marching cubes

.. code-block:: python

    # Generate ensemble data with noise
    F = np.zeros((n_x, n_y, n_z, n_ens))
    for e in range(n_ens):
        noise = np.random.normal(0, 0.01, (n_x, n_y, n_z))
        F[:, :, :, e] = noise_less_F + noise

    # Compute probabilistic marching cubes
    plotter.subplot(0, 1)
    plotter = pmc(F, isovalue, plotter=plotter)
    plotter.add_text("Probabilistic Isosurface", font_size=12)
    # plotter.show()
    plotter.screenshot("probabilistic_marching_cubes_example.png")

.. image:: _static/probabilistic_marching_cubes_example.png
   :alt: Probabilistic Marching Cubes Example
   :align: center

"""

# Import necessary libraries
import numpy as np
import pyvista as pv
from uvisbox.Modules.ProbabilisticMarchingCubes.plot import plot as pmc

# Generate a regular grid over a 3D domain [-1, 1] and use the tear drop function
# to create an ensemble of scalar fields with some noise

# tear drop function
def tear_drop(x, y, z):
    return 0.5*x**5 + 0.5*x**4 - y**2 - z**2

# Generate synthetic 4D data (n_x, n_y, n_z, n_ens)
n_x, n_y, n_z, n_ens = 16, 16, 16, 10
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

# Set isovalue and compute deterministic isosurface
isovalue = -0.001
iso_surface = grid.contour([isovalue], scalars="values")
# Set up the plotter with two subplots
plotter = pv.Plotter(shape=(1, 2), off_screen=True)
# Plot deterministic isosurface
plotter.subplot(0, 0)
plotter.add_text("Deterministic Isosurface", font_size=12)
plotter.add_mesh(iso_surface, color='lightblue', opacity=0.5)

# Generate ensemble data with noise and calculate probabilistic isosurface 
# using probabilistic marching cubes

# Generate ensemble data with noise
F = np.zeros((n_x, n_y, n_z, n_ens))
for e in range(n_ens):
    noise = np.random.normal(0, 0.01, (n_x, n_y, n_z))
    F[:, :, :, e] = noise_less_F + noise

# Compute probabilistic marching cubes
plotter.subplot(0, 1)
plotter = pmc(F, isovalue, plotter=plotter)
plotter.add_text("Probabilistic Isosurface", font_size=12)
plotter.show()
plotter.screenshot("probabilistic_marching_cubes_example.png")
