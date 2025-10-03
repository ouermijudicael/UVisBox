from operator import inv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from uvisbox.Colors.colortree import ColorTree

def plot_uncertainty_tube_from_mesh(vertices, faces, mean_trajectories, uv_coords, axis=None):
    """
    Plot 3D uncertainty tubes from pre-generated mesh data.

    Parameters:
    -----------
    vertices (np.ndarray): 
        Global vertex array with shape (total_vertices, 3)
    faces (np.ndarray): 
        Triangle face indices with shape (n_seeds, triangles_per_seed, 3)
    mean_trajectories (np.ndarray): 
        Mean trajectory positions with shape (n_steps+1, n_seeds, 3)
    uv_coords (np.ndarray): 
        UV coordinates for coloring, shape matching vertices.
    axis (matplotlib.axes.Axes, optional): 
        3D axis to plot on. If None, creates a new figure and axis.

    Returns:
    --------
    None
    """
    if axis is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        show_plot = True
    else:
        ax = axis
        if not hasattr(ax, 'get_proj') or ax.get_proj().shape != (4, 4):
            raise ValueError("The provided axis must be a 3D axis (projection='3d').")
        show_plot = False

    color_tree = ColorTree(invert_u=True, depth=4, cmap="viridis")
    n_steps, n_seeds = mean_trajectories.shape[:2]

    # Plot mean trajectory lines
    for i in range(n_seeds):
        for j in range(1, n_steps):
            segment = mean_trajectories[j-1:j+1, i, :]
            ax.plot(segment[:, 0], segment[:, 1], segment[:, 2],
                    color='black', alpha=1.0, linewidth=2)

    # Plot uncertainty tubes using triangular faces
    all_tube_faces = []
    all_face_color = []
    # Process faces for each seed
    for i in range(n_seeds):
        seed_faces = faces[i]  # Shape: (triangles_per_seed, 3)
        # Convert face indices to vertex coordinates
        triangle_vertices = vertices[seed_faces]  # Shape: (triangles_per_seed, 3, 3)
        color_values = uv_coords[seed_faces]
        face_colors = color_tree(color_values.mean(axis=1), discrete=True)
        all_tube_faces.extend(triangle_vertices)
        all_face_color.extend(face_colors)

    all_tube_faces = np.array(all_tube_faces)
    all_face_color = np.array(all_face_color)
    tube_collection = Poly3DCollection(all_tube_faces,
                                        facecolors=all_face_color)
    ax.add_collection3d(tube_collection)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('3D Trajectories with Uncertainty')
    plt.grid()

    if show_plot:
        plt.show()
