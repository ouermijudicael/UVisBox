import numpy as np

from uvisbox.Datasets import flowmap_3d
from uvisbox.Interpolations import linear_interpolate
from uvisbox.UncertaintyTube import (generate_uncertainty_tube,generate_tube_mesh)
from uvisbox.UncertaintyTube import plot_uncertainty_path_3d_from_mesh

t0 = 0
t1 = 3
n_steps = 30
number_of_seeds = 6  # Increased for better parallel demonstration

scale = np.arange(number_of_seeds)
scale = linear_interpolate(scale, 0, number_of_seeds-1, 1.0, 2.0)
xy_scale = np.ones(number_of_seeds) 
# change odd place of xy_scale to 0.1
xy_scale[1::2] = 0.1  # Set every second element to 0.1

# Generate random seed points in 3D in [-1,1]^3
seed_3d = np.random.uniform(-1, 1, (number_of_seeds, 3))
trajectories_3d = flowmap_3d(seed_3d, t0, t1, n_steps, scale=scale, xy_scale=xy_scale)

seeds = np.random.uniform(-1,1, (number_of_seeds, 3))

trajectories = flowmap_3d(seeds, t0, t1, n_steps, scale=scale, xy_scale=xy_scale)

cross_sections, eigen_values = generate_uncertainty_tube(trajectories, None, 16, e_proj=0.5, n_jobs=4)
eigen_values = np.transpose(eigen_values, (1,0,2,3))
eigen_values = eigen_values.reshape((-1,2)).astype(np.float32)

max_eigen_values = eigen_values.max(axis=1)
rescaled_max_eigen_values = linear_interpolate(max_eigen_values, max_eigen_values.min(), max_eigen_values.max(), 0.0, 1.0)

eigen_values_ratio = np.nan_to_num(eigen_values[:,1] / eigen_values[:,0] , nan=0.0, posinf=1.0, neginf=0.0)

uv_coords = np.stack([rescaled_max_eigen_values, eigen_values_ratio], axis=1)

vertices, faces, mean_trajectories = generate_tube_mesh(trajectories, cross_sections)

plot_uncertainty_path_3d_from_mesh(vertices, faces, mean_trajectories, uv_coords)
