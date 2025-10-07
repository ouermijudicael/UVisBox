"""
This example demonstrates the use of the probabilistic marching triangles algorithm
to compute and visualize probabilistic isocontours from an ensemble of scalar fields defined on a
triangular mesh.
It visualizes the probability of the isocontour passing through each triangle in the mesh.

Import necessary libraries

.. code-block:: python

    from matplotlib.tri import Triangulation
    import matplotlib.pyplot as plt
    import numpy as np  
    from uvisbox.ProbabilisticContours import probabilistic_marching_triangles

Generate a triangular mesh over a 2D domain [0, 2π], use function f(x, y) = sin(x) * cos(y)
and create an ensemble of scalar fields with some noise

.. code-block:: python

    # Synthetic function: f(x, y) = sin(x) * cos(y)
    def synthetic_func(x, y):
        return np.sin(x) * np.cos(y)

    # Domain setup
    x = np.linspace(0, 2 * np.pi, 30)
    y = np.linspace(0, 2 * np.pi, 30)
    xv, yv = np.meshgrid(x, y)
    points = np.column_stack([xv.ravel(), yv.ravel()])

    # Triangulate the domain
    tri = Triangulation(points[:, 0], points[:, 1])
    triangles = tri.triangles

    # Generate ensemble samples with noise
    n_ens = 100
    F = np.array([
        synthetic_func(points[:, 0], points[:, 1]) + np.random.normal(0, 0.2, points.shape[0])
        for _ in range(n_ens)
    ]).T  # Shape (n_points, n_ens)

Set isovalue, run probabilistic marching triangles, and visualize result 

.. code-block:: python

    # Set isovalue
    isovalue = 0.5

    # Run probabilistic marching triangles
    prob_contour = probabilistic_marching_triangles(F, triangles, isovalue, num_samples=200)

    # Visualize probability map over triangles
    plt.figure(figsize=(8, 6))
    tpc = plt.tripcolor(points[:, 0], points[:, 1], triangles, prob_contour, shading='flat', cmap='viridis')
    plt.colorbar(tpc, label='Probability of Isocontour')
    plt.title('Probabilistic Marching Triangles Example')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig("probabilistic_marching_triangles_example.png")
    plt.show()

.. image:: _static/probabilistic_marching_triangles_example.png
   :alt: Probabilistic Marching Triangles Example
   :align: center

"""

# Import necessary libraries
from matplotlib.tri import Triangulation
import matplotlib.pyplot as plt
import numpy as np  
from uvisbox.Modules.ProbabilisticMarchingTriangles.probabilistic_marching_triangles import probabilistic_marching_triangles

# Generate a triangular mesh over a 2D domain [0, 2π], use function f(x, y) = sin(x) * cos(y)
# and create an ensemble of scalar fields with some noise

# Synthetic function: f(x, y) = sin(x) * cos(y)
def synthetic_func(x, y):
    return np.sin(x) * np.cos(y)

# Domain setup
x = np.linspace(0, 2 * np.pi, 30)
y = np.linspace(0, 2 * np.pi, 30)
xv, yv = np.meshgrid(x, y)
points = np.column_stack([xv.ravel(), yv.ravel()])

# Triangulate the domain
tri = Triangulation(points[:, 0], points[:, 1])
triangles = tri.triangles

# Generate ensemble samples with noise
n_ens = 100
F = np.array([
    synthetic_func(points[:, 0], points[:, 1]) + np.random.normal(0, 0.2, points.shape[0])
    for _ in range(n_ens)
]).T  # Shape (n_points, n_ens)

# Set isovalue, run probabilistic marching triangles, and visualize result 

# Set isovalue
isovalue = 0.5

# # Run probabilistic marching triangles
# prob_contour = probabilistic_marching_triangles(F, triangles, isovalue, num_samples=200)

# # Visualize probability map over triangles
# plt.figure(figsize=(8, 6))
# tpc = plt.tripcolor(points[:, 0], points[:, 1], triangles, prob_contour, shading='flat', cmap='viridis')
# plt.colorbar(tpc, label='Probability of Isocontour')
# plt.title('Probabilistic Marching Triangles Example')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.tight_layout()
fig, ax = plt.subplots(figsize=(8, 6))
ax = probabilistic_marching_triangles(F, points, triangles, isovalue, ax=ax)
plt.savefig("probabilistic_marching_triangles_example.png")
plt.show()