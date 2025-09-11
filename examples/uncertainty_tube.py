import dis
import numpy as np
import matplotlib.pyplot as plt

from uvisbox.Datasets import flowmap_3d
from uvisbox.Interpolations import linear_interpolate
from uvisbox.UncertaintyTube import (generate_cross_sections,generate_tube_mesh)
from uvisbox.UncertaintyTube import plot_uncertainty_tube_from_mesh
from uvisbox.Colors.colortree import ColorTree

t0 = 0
t1 = 5
n_steps = 30
number_of_seeds = 10  # Increased for better parallel demonstration

scale = np.arange(number_of_seeds)
scale = linear_interpolate(scale, 0, number_of_seeds-1, 1.5, 2.0)
xy_scale = np.ones(number_of_seeds) 
# change odd place of xy_scale to 0.1
xy_scale[1::2] = 0.1  # Set every second element to 0.1

# Generate random seed points in 3D in [-1,1]^3
seed_3d = np.random.uniform(-1, 1, (number_of_seeds, 3))
trajectories_3d = flowmap_3d(seed_3d, t0, t1, n_steps, scale=scale, xy_scale=xy_scale)

seeds = np.random.uniform(-1,1, (number_of_seeds, 3))

trajectories = flowmap_3d(seeds, t0, t1, n_steps, scale=scale, xy_scale=xy_scale)
# trajectories are generated in [n_steps, n_seeds, n_samples, n_dims]


###
### Key functions to generate uncertainty tubes
###
cross_sections, eigen_values = generate_cross_sections(trajectories, None, 16, e_proj=0.5, n_jobs=2)
vertices, faces, mean_trajectories = generate_tube_mesh(trajectories, cross_sections, n_jobs=12)

###
### use eigen values to create texture coordinates
###
eigen_values = np.transpose(eigen_values, (1,0,2,3))
eigen_values = eigen_values.reshape((-1,2)).astype(np.float32)

max_eigen_values = eigen_values.max(axis=1) # prepare to rescale eigen values to 0,1 to create texture coordinates
rescaled_max_eigen_values = linear_interpolate(max_eigen_values, max_eigen_values.min(), max_eigen_values.max(), 0.0, 1.0)
eigen_values_ratio = np.nan_to_num(eigen_values[:,1] / eigen_values[:,0] , nan=0.0, posinf=1.0, neginf=0.0) # the second axis

uv_coords = np.stack([rescaled_max_eigen_values, eigen_values_ratio], axis=1)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

# for the colormap, rescaled_max_eigen_values are used for level of uncertainty, 
# vivid color highlights high uncertainty (larger size),
# gray color represents low uncertainty (smaller size).
# eigen_values are used for the interpolating the colormap ('viridis'), 
# blue means high asymmetry, yellow means symmetry.
plot_uncertainty_tube_from_mesh(vertices, faces, mean_trajectories, uv_coords, axis=ax)

# ax = fig.add_subplot(122)
# height, width = 100, 100
# value_grid = np.linspace(0, 1, width)[None, :]  # Shape (1, 100), broadcasted to (100, 100)
# uncertainty_grid = np.linspace(0, 1, height)[:, None]  # Shape (100, 1), broadcasted to (100, 100)
# # Create image array with shape (100, 100, 2) where last dim is [uncertainty, value]
# image = np.stack([uncertainty_grid * np.ones((height, width)), value_grid * np.ones((height, width))], axis=-1)
# # Initialize ColorTree with depth=4 and default settings
# colormap = ColorTree(depth=4, cmap="viridis", invert_u=True)
# colors = colormap(image, discrete=True)
# ax.imshow(colors, origin='lower', extent=(0, 1, 0, 1))
# ax.margins(50)
# ax.set_title("Color Map")
# ax.set_ylabel("Uncertainty")
# ax.set_xlabel("Symmetry")
plt.tight_layout()
plt.savefig("uncertainty_tube.png", dpi=300)
plt.show()