

"""
This example demonstrates how to visualize uncertainty tubes in 3D using the ``uvisbox`` library with Matplotlib.
It generates random seed points, computes their trajectories in a 3D flow field, and visualizes the uncertainty tubes along these trajectories.

Import necessary libraries

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
    number_of_seeds = 2  # Two curves only

    scale = np.arange(number_of_seeds)
    scale = linear_interpolate(scale, 0, number_of_seeds-1, 1.5, 2.0)
    xy_scale = np.ones(number_of_seeds) 
    xy_scale[1::2] = 0.1  # Set every second element to 0.1

    seeds = np.random.uniform(-1,1, (number_of_seeds, 3))

    # trajectories are generated in [n_steps, n_seeds, n_samples, n_dims]
    trajectories = flowmap_3d(seeds, t0, t1, n_steps, scale=scale, xy_scale=xy_scale)


    # Key functions to generate uncertainty tubes
    cross_sections, eigen_values = generate_cross_sections(trajectories, None, 16, e_proj=0.5, n_jobs=2)
    vertices, faces, mean_trajectories = generate_tube_mesh(trajectories, cross_sections, n_jobs=2)

use eigen values to create texture coordinates

.. code-block:: python

    eigen_values = np.transpose(eigen_values, (1,0,2,3))
    eigen_values = eigen_values.reshape((-1,2)).astype(np.float32)

    max_eigen_values = eigen_values.max(axis=1)
    rescaled_max_eigen_values = linear_interpolate(max_eigen_values, max_eigen_values.min(), max_eigen_values.max(), 0.0, 1.0)
    eigen_values_ratio = np.nan_to_num(eigen_values[:,1] / eigen_values[:,0] , nan=0.0, posinf=1.0, neginf=0.0)

    # Initialize ColorTree for value-uncertainty visualization
    colormap = ColorTree(depth=4, cmap="viridis")

    # Create matplotlib figure with 3D subplots
    fig = plt.figure(figsize=(14, 6))
    
    # Subplot 1: Spaghetti plot of all trajectories
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
    
    # Subplot 2: Uncertainty tubes with ColorTree coloring
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Create mesh visualization
    n_trajectories = faces.shape[0]
    vertices_per_trajectory = vertices.shape[0] // n_trajectories
    
    for i_traj in range(n_trajectories):
        # Extract vertices and faces for this trajectory
        vertex_start = i_traj * vertices_per_trajectory
        vertex_end = (i_traj + 1) * vertices_per_trajectory
        traj_vertices = vertices[vertex_start:vertex_end]
        
        # Get faces for this trajectory and adjust indices
        traj_faces = faces[i_traj]
        traj_faces_local = traj_faces - vertex_start
        
        # Get values for ColorTree: uncertainty (asymmetry) and value (magnitude)
        eigen_start = i_traj * (len(eigen_values_ratio) // n_trajectories)
        eigen_end = (i_traj + 1) * (len(eigen_values_ratio) // n_trajectories)
        asymmetry_values = eigen_values_ratio[eigen_start:eigen_end]
        
        rescaled_start = i_traj * (len(rescaled_max_eigen_values) // n_trajectories)
        rescaled_end = (i_traj + 1) * (len(rescaled_max_eigen_values) // n_trajectories)
        value_magnitudes = rescaled_max_eigen_values[rescaled_start:rescaled_end]
        
        # Create triangles for Poly3DCollection with ColorTree colors
        triangles = []
        face_colors = []
        for face in traj_faces_local:
            triangle = traj_vertices[face]
            triangles.append(triangle)
            
            # Create [uncertainty, value] pair for ColorTree
            face_uncertainty = np.mean(asymmetry_values[face])
            face_value = np.mean(value_magnitudes[face])
            uv_input = np.array([[face_uncertainty, face_value]])
            
            # Get color from ColorTree
            color = colormap.get_colors(uv_input, discrete=True)
            face_colors.append(color[0])
        
        # Create opaque collection with ColorTree colors
        poly = Poly3DCollection(triangles, alpha=1.0, edgecolor='none')
        poly.set_facecolors(face_colors)
        ax2.add_collection3d(poly)
    
    # Set axis limits based on vertices
    ax2.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
    ax2.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
    ax2.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
    ax2.set_title("Uncertainty Tubes (ColorTree)")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    plt.tight_layout()
    plt.show()
    
.. image:: _static/uncertainty_tube.png
    :alt: Uncertainty Tube Example
    :align: center
    
"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from uvisbox.Datasets import flowmap_3d
from uvisbox.Core.Interpolations import linear_interpolate
from uvisbox.Modules.UncertaintyTube.uncertainty_tubes_stats import generate_cross_sections
from uvisbox.Modules.UncertaintyTube.uncertainty_tubes_mesh import generate_tube_mesh
from uvisbox.Core.Colors.colortree import ColorTree

# Generate random seed points and compute their trajectories in a 3D flow field

t0 = 4 # start time
t1 = 5 # end time
n_steps = 10 # number of time steps
number_of_seeds = 2  # Two curves only

scale = np.arange(number_of_seeds)
scale = linear_interpolate(scale, 0, number_of_seeds-1, 1.5, 2.0)
xy_scale = np.ones(number_of_seeds) 
xy_scale[1::2] = 0.1  # Set every second element to 0.1

seeds = np.random.uniform(0.5,1, (number_of_seeds, 3))

# trajectories are generated in [n_steps, n_seeds, n_samples, n_dims]
trajectories = flowmap_3d(seeds, t0, t1, n_steps, scale=scale, xy_scale=xy_scale)


# Key functions to generate uncertainty tubes
cross_sections, eigen_values = generate_cross_sections(trajectories, None, 16, e_proj=0.5, n_jobs=2)
vertices, faces, mean_trajectories = generate_tube_mesh(trajectories, cross_sections, n_jobs=2)

# use eigen values to create texture coordinates

eigen_values = np.transpose(eigen_values, (1,0,2,3))
eigen_values = eigen_values.reshape((-1,2)).astype(np.float32)

max_eigen_values = eigen_values.max(axis=1)
rescaled_max_eigen_values = linear_interpolate(max_eigen_values, max_eigen_values.min(), max_eigen_values.max(), 0.0, 1.0)
eigen_values_ratio = np.nan_to_num(eigen_values[:,1] / eigen_values[:,0] , nan=0.0, posinf=1.0, neginf=0.0)

# save vertices, faces, eigen_values_ratio, rescaled_max_eigen_values as numpy arrays
print(f"shapes: vertices {vertices.shape}, faces {faces.shape}, eigen_values_ratio {eigen_values_ratio.shape}, rescaled_max_eigen_values {rescaled_max_eigen_values.shape}")
np.save("vertices.npy", vertices)
np.save("faces.npy", faces)
np.save("eigen_values_ratio.npy", eigen_values_ratio)
np.save("rescaled_max_eigen_values.npy", rescaled_max_eigen_values)

# Initialize ColorTree for value-uncertainty visualization
colormap = ColorTree(depth=4, cmap="viridis")

# Create matplotlib figure with 3D subplots
fig = plt.figure(figsize=(14, 6))

# Subplot 1: Spaghetti plot of all trajectories
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

# Subplot 2: Uncertainty tubes with eigen value-based coloring
ax2 = fig.add_subplot(122, projection='3d')

# Create mesh visualization
n_trajectories = faces.shape[0]
vertices_per_trajectory = vertices.shape[0] // n_trajectories

for i_traj in range(n_trajectories):
    # Extract vertices and faces for this trajectory
    vertex_start = i_traj * vertices_per_trajectory
    vertex_end = (i_traj + 1) * vertices_per_trajectory
    traj_vertices = vertices[vertex_start:vertex_end]
    
    # Get faces for this trajectory and adjust indices
    traj_faces = faces[i_traj]
    traj_faces_local = traj_faces - vertex_start
    
    # Get asymmetry values for coloring
    eigen_start = i_traj * (len(eigen_values_ratio) // n_trajectories)
    eigen_end = (i_traj + 1) * (len(eigen_values_ratio) // n_trajectories)
    asymmetry_values = eigen_values_ratio[eigen_start:eigen_end]
    
    rescaled_start = i_traj * (len(rescaled_max_eigen_values) // n_trajectories)
    rescaled_end = (i_traj + 1) * (len(rescaled_max_eigen_values) // n_trajectories)
    uncertainty_magnitude = rescaled_max_eigen_values[rescaled_start:rescaled_end]
    
    # Create triangles for Poly3DCollection
    triangles = []
    face_colors = []
    for face_idx, face in enumerate(traj_faces_local):
        triangle = traj_vertices[face]
        triangles.append(triangle)
        
        # Create [uncertainty, value] pair for ColorTree
        face_asymmetry = np.mean(asymmetry_values[face])
        face_uncertainty = np.mean(uncertainty_magnitude[face])
        uv_input = np.array([[face_uncertainty,face_asymmetry ]])
        
        # Get color from ColorTree
        color = colormap.get_colors(uv_input, discrete=True)
        face_colors.append(color[0])
    
    # Create collection with ColorTree colors
    poly = Poly3DCollection(triangles, alpha=1.0, edgecolor='none')
    poly.set_facecolors(face_colors)
    ax2.add_collection3d(poly)

# Set axis limits based on vertices
ax2.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
ax2.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
ax2.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
ax2.set_title("Uncertainty Tubes (ColorTree)")
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.grid(False)

plt.tight_layout()
plt.show()