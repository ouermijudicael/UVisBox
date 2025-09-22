"""
This example demonstrates the use of the probabilistic marching triangles algorithm
to compute and visualize probabilistic isocontours from an ensemble of scalar fields defined on a
triangular mesh.
It visualizes the probability of the isocontour passing through each triangle in the mesh.

Import necessary libraries and modules.

.. code-block:: python

    from matplotlib.tri import Triangulation
    import matplotlib.pyplot as plt
    import numpy as np  
    from uvisbox.ProbabilisticContours.Stat.probabilistic_marching_triangles
    import probabilistic_marching_squares

Define a synthetic scalar field function (e.g., f(x, y) = sin(x) * cos(y)).
.. code-block:: python
    def synthetic_func(x, y):
        return np.sin(x) * np.cos(y)

Generate a triangular mesh over a 2D domain and create an ensemble of scalar fields with some noise.

.. code-block:: python

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
    # Set isovalue
    isovalue = 0.5  

Run probabilistic marching triangles

.. code-block:: python

    prob_contour = probabilistic_marching_squares(F, triangles, isovalue, num_samples=200)

Visualize the probability map over the triangles

.. code-block:: python

    plt.figure(figsize=(8, 6))
    tpc = plt.tripcolor(points[:, 0], points[:, 1], triangles, prob_contour, shading='flat', cmap='viridis')
    plt.colorbar(tpc, label='Probability of Isocontour')
    plt.title('Probabilistic Marching Triangles Example')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
"""

from matplotlib.tri import Triangulation
import matplotlib.pyplot as plt
import numpy as np  
from uvisbox.ProbabilisticContours.Stat.probabilistic_marching_triangles 
import probabilistic_marching_squares

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

# Set isovalue
isovalue = 0.5

# Run probabilistic marching triangles
prob_contour = probabilistic_marching_squares(F, triangles, isovalue, num_samples=200)

# Visualize probability map over triangles
plt.figure(figsize=(8, 6))
tpc = plt.tripcolor(points[:, 0], points[:, 1], triangles, prob_contour, shading='flat', cmap='viridis')
plt.colorbar(tpc, label='Probability of Isocontour')
plt.title('Probabilistic Marching Triangles Example')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig("probabilistic_marching_triangles.png")
plt.show()