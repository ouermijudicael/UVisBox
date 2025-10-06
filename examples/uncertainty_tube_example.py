

"""
This example demonstrates how to visualize uncertainty tubes in 3D using the ``uvisbox`` library.
It generates random seed points, computes their trajectories in a 3D flow field, and visualizes the uncertainty tubes along these trajectories.

Import necessary libraries

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    from uvisbox.Datasets import flowmap_3d
    from uvisbox.Interpolations import linear_interpolate
    from uvisbox.UncertaintyTube import (generate_cross_sections,generate_tube_mesh)
    from uvisbox.UncertaintyTube import plot_uncertainty_tube_from_mesh
    from uvisbox.Colors.colortree import ColorTree

Generate random seed points and compute their trajectories in a 3D flow field

.. code-block:: python

    t0 = 0 # start time
    t1 = 5 # end time
    n_steps = 30 # number of time steps
    number_of_seeds = 10  # Increased for better parallel demonstration

    scale = np.arange(number_of_seeds)
    scale = linear_interpolate(scale, 0, number_of_seeds-1, 1.5, 2.0)
    xy_scale = np.ones(number_of_seeds) 
    # change odd place of xy_scale to 0.1
    xy_scale[1::2] = 0.1  # Set every second element to 0.1

    seeds = np.random.uniform(-1,1, (number_of_seeds, 3))

    # trajectories are generated in [n_steps, n_seeds, n_samples, n_dims]
    trajectories = flowmap_3d(seeds, t0, t1, n_steps, scale=scale, xy_scale=xy_scale)


.. code-block:: python

    # Key functions to generate uncertainty tubes
    cross_sections, eigen_values = generate_cross_sections(trajectories, None, 16, e_proj=0.5, n_jobs=2)
    vertices, faces, mean_trajectories = generate_tube_mesh(trajectories, cross_sections, n_jobs=12)

    # use eigen values to create texture coordinates

    eigen_values = np.transpose(eigen_values, (1,0,2,3))
    eigen_values = eigen_values.reshape((-1,2)).astype(np.float32)

    max_eigen_values = eigen_values.max(axis=1) # prepare to rescale eigen values to 0,1 to create texture coordinates
    rescaled_max_eigen_values = linear_interpolate(max_eigen_values, max_eigen_values.min(), max_eigen_values.max(), 0.0, 1.0)
    eigen_values_ratio = np.nan_to_num(eigen_values[:,1] / eigen_values[:,0] , nan=0.0, posinf=1.0, neginf=0.0) # the second axis

    uv_coords = np.stack([rescaled_max_eigen_values, eigen_values_ratio], axis=1)

.. code-block:: python

    #  create figure and axis with two subplots
    fig , (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), subplot_kw={'projection': '3d'})

    # spaghetti plot of all trajectories
    for i in range(trajectories.shape[1]):
        for i_sample in range(trajectories.shape[2]):
            ax1.plot(trajectories[:, i, i_sample, 0], trajectories[:, i, i_sample, 1], 
                    trajectories[:, i, i_sample, 2], color='black', alpha=0.50)
    ax1.set_title("Spaghetti Plot of Trajectories")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # for the colormap, rescaled_max_eigen_values are used for level of uncertainty, 
    # vivid color highlights high uncertainty (larger size),
    # gray color represents low uncertainty (smaller size).
    # eigen_values are used for the interpolating the colormap ('viridis'), 
    # blue means high asymmetry, yellow means symmetry.
    plot_uncertainty_tube_from_mesh(vertices, faces, mean_trajectories, uv_coords, axis=ax2)
    plt.savefig("uncertainty_tube.png")
    plt.show()
    
.. image:: _static/uncertainty_tube.png
    :alt: Uncertainty Tube Example
    :align: center
    
"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

from uvisbox.Datasets import flowmap_3d
from uvisbox.Core.Interpolations import linear_interpolate
from uvisbox.Modules.UncertaintyTube.uncertainty_tube import generate_cross_sections
from uvisbox.Modules.UncertaintyTube.uncertainty_tube_meshing import generate_tube_mesh
from uvisbox.Modules.UncertaintyTube.uncertainty_tube_plot import plot_uncertainty_tube_from_mesh
from uvisbox.Core.Colors.colortree import ColorTree

# Generate random seed points and compute their trajectories in a 3D flow field

t0 = 0 # start time
t1 = 5 # end time
n_steps = 30 # number of time steps
number_of_seeds = 10  # Increased for better parallel demonstration

scale = np.arange(number_of_seeds)
scale = linear_interpolate(scale, 0, number_of_seeds-1, 1.5, 2.0)
xy_scale = np.ones(number_of_seeds) 
# change odd place of xy_scale to 0.1
xy_scale[1::2] = 0.1  # Set every second element to 0.1

seeds = np.random.uniform(-1,1, (number_of_seeds, 3))

# trajectories are generated in [n_steps, n_seeds, n_samples, n_dims]
trajectories = flowmap_3d(seeds, t0, t1, n_steps, scale=scale, xy_scale=xy_scale)


# Key functions to generate uncertainty tubes
cross_sections, eigen_values = generate_cross_sections(trajectories, None, 16, e_proj=0.5, n_jobs=2)
vertices, faces, mean_trajectories = generate_tube_mesh(trajectories, cross_sections, n_jobs=12)

# use eigen values to create texture coordinates

eigen_values = np.transpose(eigen_values, (1,0,2,3))
eigen_values = eigen_values.reshape((-1,2)).astype(np.float32)

max_eigen_values = eigen_values.max(axis=1) # prepare to rescale eigen values to 0,1 to create texture coordinates
rescaled_max_eigen_values = linear_interpolate(max_eigen_values, max_eigen_values.min(), max_eigen_values.max(), 0.0, 1.0)
eigen_values_ratio = np.nan_to_num(eigen_values[:,1] / eigen_values[:,0] , nan=0.0, posinf=1.0, neginf=0.0) # the second axis

uv_coords = np.stack([rescaled_max_eigen_values, eigen_values_ratio], axis=1)

#  create figure and axis with two subplots
fig , (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), subplot_kw={'projection': '3d'})

# spaghetti plot of all trajectories
for i in range(trajectories.shape[1]):
    for i_sample in range(trajectories.shape[2]):
        ax1.plot(trajectories[:, i, i_sample, 0], trajectories[:, i, i_sample, 1], 
                 trajectories[:, i, i_sample, 2], color='black', alpha=0.50)
ax1.set_title("Spaghetti Plot of Trajectories")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")

# for the colormap, rescaled_max_eigen_values are used for level of uncertainty, 
# vivid color highlights high uncertainty (larger size),
# gray color represents low uncertainty (smaller size).
# eigen_values are used for the interpolating the colormap ('viridis'), 
# blue means high asymmetry, yellow means symmetry.
plot_uncertainty_tube_from_mesh(vertices, faces, mean_trajectories, uv_coords, axis=ax2)
plt.savefig("uncertainty_tube.png")
plt.show()