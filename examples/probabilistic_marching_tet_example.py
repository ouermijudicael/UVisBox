"""
This example demonstrates the use of probabilistic marching tetrahedra to compute and visualize
probabilistic isosurfaces from an ensemble of scalar fields defined on a tetrahedral mesh.
It compares a deterministic isosurface with a probabilistic isosurface derived from the ensemble.


Import necessary libraries

.. code-block:: python

    import numpy as np
    import pyvista as pv
    from uvisbox.Modules.ProbabilisticMarchingTetrahedra import probabilistic_marching_tetrahedron

Generate tetrahedral mesh over a 3D domain [-1, 1] and use the tear drop function
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
    F = np.zeros((n_x, n_y, n_z, n_ens))
    noise_less_F = tear_drop(X, Y, Z)
    # Create an ensemble of scalar fields with some noise
    for e in range(n_ens):
        noise = np.random.normal(0, 0.01, (n_x, n_y, n_z))
        F[:, :, :, e] = noise_less_F + noise
    points = np.c_[X.ravel(), Y.ravel(), Z.ravel()]

    # create teterahedral mesh
    pv_points= pv.PolyData(points)
    grid = pv_points.delaunay_3d(offset=0.1)
    # extract the tetrahedra
    tetrahedra = grid.cells.reshape(-1, 5)[:, 1:]   
    edges = grid.extract_all_edges()
    # edges.plot(line_width=1, color='k')
    cell_types = np.full(tetrahedra.shape[0], pv.CellType.TETRA)
    tetrahedra = np.hstack((np.full((tetrahedra.shape[0], 1), 4), tetrahedra))

    new_grid = pv.UnstructuredGrid(tetrahedra, cell_types, points)  
    new_grid.point_data["values"] = noise_less_F.flatten(order='F')
    isovalue = -0.001
    # Compute deterministic isosurface
    iso_surface = new_grid.contour([isovalue], scalars="values")

    # Set up the plotter with two subplots
    plotter = pv.Plotter(shape=(1, 2))
    # Plot deterministic isosurface
    plotter.subplot(0, 0)
    plotter.add_text("Deterministic Isosurface", font_size=12)
    plotter.add_mesh(iso_surface, color='lightblue', opacity=0.5)


    F_reshaped = F.reshape(-1, n_ens)   
    # Compute probabilistic marching tetrahedra
    plotter.subplot(0, 1)
    plotter = probabilistic_marching_tetrahedron(F_reshaped, tetrahedra, points, isovalue, plotter=plotter)
    plotter.add_text("Probabilistic Isosurface", font_size=12)
    plotter.show()

.. image:: _static/probabilistic_marching_tetrahedra_example.png
   :alt: Probabilistic Marching Tetrahedra Example
   :align: center
    
"""

# Import necessary libraries
import numpy as np
import pyvista as pv
from uvisbox.Modules.ProbabilisticMarchingTetrahedra import probabilistic_marching_tetrahedron

# Generate tetrahedral mesh over a 3D domain [-1, 1] and use the tear drop function
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
F = np.zeros((n_x, n_y, n_z, n_ens))
noise_less_F = tear_drop(X, Y, Z)
# Create an ensemble of scalar fields with some noise
for e in range(n_ens):
    noise = np.random.normal(0, 0.01, (n_x, n_y, n_z))
    F[:, :, :, e] = noise_less_F + noise
points = np.c_[X.ravel(), Y.ravel(), Z.ravel()]

# create teterahedral mesh
pv_points= pv.PolyData(points)
grid = pv_points.delaunay_3d(offset=0.1)
# extract the tetrahedra
tetrahedra = grid.cells.reshape(-1, 5)[:, 1:]   
edges = grid.extract_all_edges()
# edges.plot(line_width=1, color='k')
cell_types = np.full(tetrahedra.shape[0], pv.CellType.TETRA)
tetrahedra = np.hstack((np.full((tetrahedra.shape[0], 1), 4), tetrahedra))

new_grid = pv.UnstructuredGrid(tetrahedra, cell_types, points)  
new_grid.point_data["values"] = noise_less_F.flatten(order='F')
isovalue = -0.001
# Compute deterministic isosurface
iso_surface = new_grid.contour([isovalue], scalars="values")

# Set up the plotter with two subplots
plotter = pv.Plotter(shape=(1, 2))
# Plot deterministic isosurface
plotter.subplot(0, 0)
plotter.add_text("Deterministic Isosurface", font_size=12)
plotter.add_mesh(iso_surface, color='lightblue', opacity=0.5)


F_reshaped = F.reshape(-1, n_ens)   
# Compute probabilistic marching tetrahedra
plotter.subplot(0, 1)
plotter = probabilistic_marching_tetrahedron(F_reshaped, tetrahedra, points, isovalue, plotter=plotter)
plotter.add_text("Probabilistic Isosurface", font_size=12)
plotter.show()
# plotter.screenshot("probabilistic_marching_tetrahedra_example.png")