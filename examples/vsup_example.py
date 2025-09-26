"""
Example script demonstrating the ColorTree colormap for visualizing value and uncertainty.
A ColorTree is inspired by VSUP (Value-Suppressing Uncertainty Palettes) and creates a tree-based colormap.
Value is suppressed based on uncertainty, allowing for more effective visualization of uncertain data.

VSUP:https://medium.com/@uwdata/value-suppressing-uncertainty-palettes-426130122ce9

This script creates a sample 2D image where the x-axis represents value (0 to 1) and the y-axis represents uncertainty (0 to 1).
It generates three visualizations:
- Discrete: Uses tree nodes for discrete levels.
- Continuous: Interpolates colormap colors with uncertainty.
- Continuous Leaves: Uses colormap for leaf levels in discrete mode.

Run this script to see the differences between the modes.

Important necessary libraries

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from uvisbox.Colors.colortree import ColorTree

Set up the figure with three subplots

.. code-block:: python

    fig, ax = plt.subplots(1, 3, figsize=(12, 6))

Create a sample image where x (columns) is value (0 to 1), y (rows) is uncertainty (0 to 1)

.. code-block:: python

    height, width = 100, 100
    value_grid = np.linspace(0, 1, width)[None, :]  # Shape (1, 100), broadcasted to (100, 100)
    uncertainty_grid = np.linspace(0, 1, height)[:, None]  # Shape (100, 1), broadcasted to (100, 100)

    # Create image array with shape (100, 100, 2) where last dim is [uncertainty, value]
    image = np.stack([uncertainty_grid * np.ones((height, width)), value_grid * np.ones((height, width))], axis=-1)
    # Initialize ColorTree with depth=4 and default settings
    colormap = ColorTree(depth=4, cmap="viridis")   
    # Generate colors for discrete mode (uses tree nodes)
    colors = colormap.get_colors(image, discrete=True)
    
    # Plot the discrete color map
    ax[0].imshow(colors, origin='lower', extent=(0, 1, 0, 1))
    ax[0].set_title("Discrete Color Map")
    ax[0].set_xlabel("Value")
    ax[0].set_ylabel("Uncertainty")
    # Generate colors for continuous mode (interpolates colormap with uncertainty)
    continuous_color = colormap.get_colors(image, discrete=False)
    
    # Plot the continuous color map
    ax[1].imshow(continuous_color, origin='lower', extent=(0, 1, 0, 1))
    ax[1].set_title("Continuous Color Map")
    ax[1].set_xlabel("Value")
    ax[1].set_ylabel("Uncertainty")
    
    # Generate colors for continuous leaves mode (discrete with colormap at leaves)
    continuous_leaves_color = colormap.get_colors(image, discrete=True, continuous_leaves=True)
    # Plot the continuous leaves color map
    ax[2].imshow(continuous_leaves_color, origin='lower', extent=(0, 1, 0, 1))
    ax[2].set_title("Continuous Leaves Color Map")
    ax[2].set_xlabel("Value")
    ax[2].set_ylabel("Uncertainty") 
    # Display the plot
    plt.show()

.. image:: _static/colortree_example.png
    :alt: ColorTree Example
    :align: center

"""
# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

from uvisbox.Colors.colortree import ColorTree

# Set up the figure with three subplots
fig, ax = plt.subplots(1, 3, figsize=(12, 6))

# Create a sample image where x (columns) is value (0 to 1), y (rows) is uncertainty (0 to 1)
height, width = 100, 100
value_grid = np.linspace(0, 1, width)[None, :]  # Shape (1, 100), broadcasted to (100, 100)
uncertainty_grid = np.linspace(0, 1, height)[:, None]  # Shape (100, 1), broadcasted to (100, 100)

# Create image array with shape (100, 100, 2) where last dim is [uncertainty, value]
image = np.stack([uncertainty_grid * np.ones((height, width)), value_grid * np.ones((height, width))], axis=-1)

# Initialize ColorTree with depth=4 and default settings
colormap = ColorTree(depth=4, cmap="viridis")

# Generate colors for discrete mode (uses tree nodes)
colors = colormap.get_colors(image, discrete=True)

# Plot the discrete color map
ax[0].imshow(colors, origin='lower', extent=(0, 1, 0, 1))
ax[0].set_title("Discrete Color Map")
ax[0].set_xlabel("Value")
ax[0].set_ylabel("Uncertainty")

# Generate colors for continuous mode (interpolates colormap with uncertainty)
continuous_color = colormap.get_colors(image, discrete=False)

# Plot the continuous color map
ax[1].imshow(continuous_color, origin='lower', extent=(0, 1, 0, 1))
ax[1].set_title("Continuous Color Map")
ax[1].set_xlabel("Value")
ax[1].set_ylabel("Uncertainty")

# Generate colors for continuous leaves mode (discrete with colormap at leaves)
continuous_leaves_color = colormap.get_colors(image, discrete=True, continuous_leaves=True)

# Plot the continuous leaves color map
ax[2].imshow(continuous_leaves_color, origin='lower', extent=(0, 1, 0, 1))
ax[2].set_title("Continuous Leaves Color Map")
ax[2].set_xlabel("Value")
ax[2].set_ylabel("Uncertainty")
plt.savefig("colortree_example.png")

# Display the plot
plt.show()