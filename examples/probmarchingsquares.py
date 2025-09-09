import numpy as np
from uvisbox.ProbabilisticMarchingSquares.probabilistic_marching_squares import probabilistic_marching_squares
import matplotlib.pyplot as plt

# Example usage
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
isovalue = 0.5
prob_contour = probabilistic_marching_squares(F, isovalue)

# Visualize result
plt.imshow(prob_contour, origin='lower', extent=(x.min(), x.max(), y.min(), y.max()), cmap='viridis')
plt.colorbar(label='Probability of Contour')
plt.title('Probabilistic Marching Squares')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("probabilistic_marching_squares.png")
plt.show()
