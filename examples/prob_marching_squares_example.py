"""
This example demonstrates the use of the `probabilistic_marching_squares` function from the `uvisbox` library to compute
and visualize probabilistic isocontours from an ensemble of scalar fields defined on a regular grid
using the marching squares algorithm.
It visualizes the probability of the isocontour passing through each cell in the grid.

Import necessary libraries and modules.

.. code-block:: python

    import numpy as np
    from uvisbox.ProbabilisticContours.Stat.probabilistic_marching_squares
    import matplotlib.pyplot as plt

    
Define a synthetic scalar field function (e.g., f(x, y) = sin(x) * cos(y)).

.. code-block:: python
    def synthetic_func(x, y):
        return np.sin(x) * np.cos(y)
        
Generate a regular grid over a 2D domain and create an ensemble of scalar fields with some noise.

.. code-block:: python

    # Domain setup
    n, m, n_ens = 50, 50, 100
    x = np.linspace(0, 4 * np.pi, n)
    y = np.linspace(0, 4 * np.pi, m)
    X, Y = np.meshgrid(x, y, indexing='ij')
    # Create ensemble with noise
    F = np.array([
        synthetic_func(X, Y) + 0.2 * np.random.randn(n, m)
        for _ in range(n_ens)
    ])
    F = np.transpose(F, (1, 2, 0))  # Shape (n, m, n_ens)
    # Set isovalue
    isovalue = 0.5

Run probabilistic marching squares

.. code-block:: python

    prob_contour = probabilistic_marching_squares(F, isovalue)

Visualize the probability map over the grid

.. code-block:: python

    plt.imshow(prob_contour, origin='lower', extent=(x.min(), x.max(), y.min(), y.max()), cmap='viridis')
    plt.colorbar(label='Probability of Contour')
    plt.title('Probabilistic Marching Squares')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

"""

# Import necessary libraries
import numpy as np
from uvisbox.ProbabilisticContours.Stat.probabilistic_marching_squares import probabilistic_marching_squares
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

# Set isovalue
isovalue = 0.5

# Run probabilistic marching squares
prob_contour = probabilistic_marching_squares(F, isovalue)

# Visualize result
plt.imshow(prob_contour, origin='lower', extent=(x.min(), x.max(), y.min(), y.max()), cmap='viridis')
plt.colorbar(label='Probability of Contour')
plt.title('Probabilistic Marching Squares')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("probabilistic_marching_squares_example.png")
plt.show()
