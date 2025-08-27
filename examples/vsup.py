import numpy as np
import matplotlib.pyplot as plt

from uvisbox.Colors.colortree import ColorTree

fig, ax = plt.subplots(1,2, figsize=(12,6))
# Create a sample image where x (columns) is value (0 to 1), y (rows) is uncertainty (0 to 1)
height, width = 100, 100
value_grid = np.linspace(0, 1, width)[None, :]  # Shape (1, 100)
uncertainty_grid = np.linspace(0, 1, height)[:, None]  # Shape (100, 1)

# Create image array with shape (100, 100, 2) where last dim is [uncertainty, value]
image = np.stack([uncertainty_grid * np.ones((height, width)), value_grid * np.ones((height, width))], axis=-1)

# Initialize ColorTree with depth=4
colormap = ColorTree(depth=4, cmap="viridis")

# Generate colors
colors = colormap.get_colors(image, discrete=True)

# Plot the image
ax[0].imshow(colors, origin='lower', extent=(0, 1, 0, 1))
ax[0].set_title("Discrete Color Map")
ax[0].set_xlabel("Value")
ax[0].set_ylabel("Uncertainty")

continuous_color = colormap.get_colors(image, discrete=False)

# Plot the continuous color map
ax[1].imshow(continuous_color, origin='lower', extent=(0, 1, 0, 1))
ax[1].set_title("Continuous Color Map")
ax[1].set_xlabel("Value")
ax[1].set_ylabel("Uncertainty")

# Show the plot
plt.show()