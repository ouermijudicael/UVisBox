"""
This example demonstrates the use of the `probabilistic_marching_squares` function from the `uvisbox` library to compute
and visualize probabilistic isocontours from an ensemble of scalar fields defined on a regular grid
using the marching squares algorithm.
It visualizes the probability of the isocontour passing through each cell in the grid.

Import necessary libraries

.. code-block:: python

    import numpy as np
    from uvisbox.ProbabilisticContours.Stat.probabilistic_marching_squares import probabilistic_marching_squares
    import matplotlib.pyplot as plt

Generate a regular grid over a 2D domain [0, 4π], use function f(x, y) = sin(x) * cos(y)
and create an ensemble of scalar fields with some noise

.. code-block:: python

    n, m, n_ens = 50, 50, 100
    x = np.linspace(0, 4 * np.pi, n)
    y = np.linspace(0, 4 * np.pi, m)
    X, Y = np.meshgrid(x, y, indexing='ij')
    # Create ensemble with noise
    F = np.array([
        np.sin(X) * np.cos(Y) + 0.2 * np.random.randn(n, m)
        for _ in range(n_ens)
    ])
    F = np.transpose(F, (1, 2, 0))  # Shape (n, m, n_ens)

Set isovalue, run probabilistic marching squares, and visualize result

.. code-block:: python

    # Set isovalue
    isovalue = 0.5

    # Run probabilistic marching squares
    prob_contour = probabilistic_marching_squares(F, isovalue)

    # Visualize result
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(prob_contour, origin='lower', extent=(x.min(), x.max(), y.min(), y.max()), cmap='viridis')
    plt.colorbar(label='Probability of Contour')
    plt.title('Probabilistic Marching Squares')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("probabilistic_marching_squares_example.png")
    plt.show()

.. image:: _static/probabilistic_marching_squares_example.png
   :alt: Probabilistic Marching Squares Example
   :align: center

"""

# Import necessary libraries
import numpy as np
from uvisbox.Modules.ProbabilisticMarchingSquares.probabilistic_marching_squares import probabilistic_marching_squares
import matplotlib.pyplot as plt

# Generate a regular grid over a 2D domain [0, 4π], use function f(x, y) = sin(x) * cos(y)
# and create an ensemble of scalar fields with some noise

n, m, n_ens = 50, 50, 100
x = np.linspace(0, 4 * np.pi, n)
y = np.linspace(0, 4 * np.pi, m)
X, Y = np.meshgrid(x, y, indexing='ij')
# Create ensemble with noise
F = np.array([
    np.sin(X) * np.cos(Y) + 0.2 * np.random.randn(n, m)
    for _ in range(n_ens)
])
F = np.transpose(F, (1, 2, 0))  # Shape (n, m, n_ens)

# Set isovalue, run probabilistic marching squares, and visualize result

# Set isovalue
isovalue = 0.5

# # Run probabilistic marching squares
# prob_contour = probabilistic_marching_squares(F, isovalue)

# # Visualize result
# fig = plt.figure(figsize=(8, 6))
# plt.imshow(prob_contour, origin='lower', extent=(x.min(), x.max(), y.min(), y.max()), cmap='viridis')
# plt.colorbar(label='Probability of Contour')
# plt.title('Probabilistic Marching Squares')
# plt.xlabel('x')
# plt.ylabel('y')
fig, ax = plt.subplots(figsize=(8, 6))
ax = probabilistic_marching_squares(F, isovalue, ax=ax)
plt.savefig("probabilistic_marching_squares_example.png")
plt.show()
