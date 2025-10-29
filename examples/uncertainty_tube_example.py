

"""
This example demonstrates how to visualize uncertainty tubes in 3D using the ``uvisbox`` library with PyVista.
It generates random seed points, computes their trajectories in a 3D flow field, and visualizes the uncertainty tubes along these trajectories.

Import necessary libraries

.. code-block:: python

    import numpy as np
    import pyvista as pv

    from uvisbox.Datasets import flowmap_3d
    from uvisbox.Core.Interpolations import linear_interpolate
    from uvisbox.Modules.UncertaintyTube.uncertainty_tubes_stats import generate_cross_sections
    from uvisbox.Modules.UncertaintyTube.uncertainty_tubes_mesh import generate_tube_mesh
    from uvisbox.Core.Colors.colortree import ColorTree

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


    # Key functions to generate uncertainty tubes
    cross_sections, eigen_values = generate_cross_sections(trajectories, None, 16, e_proj=0.5, n_jobs=2)
    vertices, faces, mean_trajectories = generate_tube_mesh(trajectories, cross_sections, n_jobs=12)

use eigen values to create texture coordinates

.. code-block:: python

    eigen_values = np.transpose(eigen_values, (1,0,2,3))
    eigen_values = eigen_values.reshape((-1,2)).astype(np.float32)

    max_eigen_values = eigen_values.max(axis=1) # prepare to rescale eigen values to 0,1 to create texture coordinates
    rescaled_max_eigen_values = linear_interpolate(max_eigen_values, max_eigen_values.min(), max_eigen_values.max(), 0.0, 1.0)
    eigen_values_ratio = np.nan_to_num(eigen_values[:,1] / eigen_values[:,0] , nan=0.0, posinf=1.0, neginf=0.0) # the second axis

    uv_coords = np.stack([rescaled_max_eigen_values, eigen_values_ratio], axis=1)

    # Create PyVista plotter with two subplots
    plotter = pv.Plotter(shape=(1, 2))

    # Subplot 0: Spaghetti plot of all trajectories
    plotter.subplot(0, 0)
    for i in range(trajectories.shape[1]):
        for i_sample in range(trajectories.shape[2]):
            points = trajectories[:, i, i_sample, :]
            line = pv.PolyData(points)
            line.lines = np.hstack([[2] + [j, j+1] for j in range(len(points)-1)])
            plotter.add_mesh(line, color='black', opacity=0.5, line_width=2)
    plotter.add_text("Spaghetti Plot of Trajectories", font_size=12)

    # Subplot 1: Uncertainty tubes with eigen value-based coloring
    # Blue = high asymmetry, Yellow = symmetry (viridis colormap)
    plotter.subplot(0, 1)
    tube_mesh = pv.PolyData(vertices, faces)
    tube_mesh['asymmetry'] = eigen_values_ratio
    plotter.add_mesh(tube_mesh, scalars='asymmetry', cmap='viridis', opacity=0.8)
    plotter.add_text("Uncertainty Tubes", font_size=12)
    
    plotter.link_views()
    plotter.show()
    
.. image:: _static/uncertainty_tube.png
    :alt: Uncertainty Tube Example
    :align: center
    
"""

# Import necessary libraries
import numpy as np
import pyvista as pv

from uvisbox.Datasets import flowmap_3d
from uvisbox.Core.Interpolations import linear_interpolate
from uvisbox.Modules.UncertaintyTube.uncertainty_tubes_stats import generate_cross_sections
from uvisbox.Modules.UncertaintyTube.uncertainty_tubes_mesh import generate_tube_mesh
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

# Create PyVista plotter with two subplots
plotter = pv.Plotter(shape=(1, 2))

# Subplot 0: Spaghetti plot of all trajectories
plotter.subplot(0, 0)
for i in range(trajectories.shape[1]):
    for i_sample in range(trajectories.shape[2]):
        points = trajectories[:, i, i_sample, :]
        # Create line connectivity: [n_points, point_0, point_1, n_points, point_1, point_2, ...]
        n_points = len(points)
        lines = np.full((n_points - 1, 3), 2, dtype=np.int_)
        lines[:, 1] = np.arange(n_points - 1)
        lines[:, 2] = np.arange(1, n_points)
        line = pv.PolyData(points, lines=lines.ravel())
        plotter.add_mesh(line, color='black', opacity=0.5, line_width=2)
plotter.add_axes()
plotter.add_text("Spaghetti Plot of Trajectories", font_size=12)

# Subplot 1: Uncertainty tubes
plotter.subplot(0, 1)

# Create separate mesh for each tube to avoid connecting different tubes
n_trajectories = faces.shape[0]
vertices_per_trajectory = vertices.shape[0] // n_trajectories

for i_traj in range(n_trajectories):
    # Extract vertices and faces for this trajectory
    vertex_start = i_traj * vertices_per_trajectory
    vertex_end = (i_traj + 1) * vertices_per_trajectory
    traj_vertices = vertices[vertex_start:vertex_end]
    
    # Get faces for this trajectory and adjust indices to local vertex indices
    traj_faces = faces[i_traj]
    traj_faces_local = traj_faces - vertex_start
    
    # Create mesh for this tube
    n_verts_per_face = 3  # triangles
    faces_with_count = np.column_stack([np.full(len(traj_faces_local), n_verts_per_face), traj_faces_local])
    tube_mesh = pv.PolyData(traj_vertices, faces_with_count.ravel())
    
    # Add scalars for coloring based on eigen values for this trajectory
    eigen_start = i_traj * (len(eigen_values_ratio) // n_trajectories)
    eigen_end = (i_traj + 1) * (len(eigen_values_ratio) // n_trajectories)
    tube_mesh['asymmetry'] = eigen_values_ratio[eigen_start:eigen_end]
    
    # Plot with colormap (Blue = high asymmetry, Yellow = symmetry)
    plotter.add_mesh(tube_mesh, scalars='asymmetry', cmap='viridis', 
                     opacity=0.8, show_edges=False, smooth_shading=True)

# Add mean trajectories as reference lines
# mean_trajectories shape is (n_steps, n_trajectories, 3), need to transpose
mean_trajectories_transposed = np.transpose(mean_trajectories, (1, 0, 2))  # -> (n_trajectories, n_steps, 3)
for i in range(mean_trajectories_transposed.shape[0]):
    points = mean_trajectories_transposed[i]
    n_points = len(points)
    lines = np.full((n_points - 1, 3), 2, dtype=np.int_)
    lines[:, 1] = np.arange(n_points - 1)
    lines[:, 2] = np.arange(1, n_points)
    mean_line = pv.PolyData(points, lines=lines.ravel())
    plotter.add_mesh(mean_line, color='red', line_width=3, opacity=0.7)

plotter.add_axes()
plotter.add_text("Uncertainty Tubes", font_size=12)

# Link camera views
plotter.link_views()

plotter.show()
# plotter.screenshot("uncertainty_tube.png")