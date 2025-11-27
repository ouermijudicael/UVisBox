import numpy as np
import matplotlib.pyplot as plt

from uvisbox.Datasets import flowmap_3d
from uvisbox.Core.Interpolations import linear_interpolate
from uvisbox.Modules.UncertaintyTube.uncertainty_tubes import uncertainty_tubes # New import

# Generate random seed points and compute their trajectories in a 3D flow field
t0 = 4
t1 = 5
n_steps = 10
number_of_seeds = 2

scale = np.arange(number_of_seeds)
scale = linear_interpolate(scale, 0, number_of_seeds - 1, 1.5, 2.0)
xy_scale = np.ones(number_of_seeds)
xy_scale[1::2] = 0.1

seeds = np.random.uniform(0.5, 1, (number_of_seeds, 3))
trajectories = flowmap_3d(seeds, t0, t1, n_steps, scale=scale, xy_scale=xy_scale)

# Create matplotlib figure with 3D subplots
fig = plt.figure(figsize=(14, 6))

# Subplot 1: Spaghetti plot of all trajectories (retaining for comparison)
ax1 = fig.add_subplot(121, projection='3d')
for i in range(trajectories.shape[1]):
    for i_sample in range(trajectories.shape[2]):
        points = trajectories[:, i, i_sample, :]
        ax1.plot(points[:, 0], points[:, 1], points[:, 2], 
                 color='black', alpha=0.5, linewidth=1)
ax1.set_title("Spaghetti Plot of Trajectories")
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.grid(False)

# Subplot 2: Uncertainty tubes with ColorTree coloring using the new function
ax2 = fig.add_subplot(122, projection='3d')

uncertainty_tubes(trajectories, colormap="viridis", plotter=ax2, e_proj=0.5, n_jobs=2)

ax2.set_title("Uncertainty Tubes (viridis colormap)")
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.grid(False)

plt.tight_layout()
plt.show()